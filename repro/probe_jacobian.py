"""用去噪器的分数与 Hessian 直接预测 Lambda(alpha)：局部高斯模型是否成立？

恒等式（EDM / VE 参数化，p_t = p (*) N(0, t^2 I)）：
      grad log p_t(x)  = (D(x,t) - x) / t^2
      Hess log p_t(x)  = (dD/dx - I) / t^2

  （符号推导：p_t(x) = ∫ N(x; x', t^2) p(x') dx'，∇_x p_t = -(x - E[x'|x])/t^2 * p_t，
   而 E[x'|x] = D(x,t)，故 ∇log p_t = (D - x)/t^2。**这个符号我第一版写反了。**）

沿线段 y_alpha = x* + alpha * delta,  delta = (h - I) x*，在 x* 处展开：
      Lambda(alpha) = alpha*s1 + (alpha^2/2)*s2 + O(alpha^3)
      s1 = <grad log p_t(x*), delta>
      s2 = delta^T Hess log p_t(x*) delta = <delta, (dD/dx - I) delta> / t^2

三个模型（同一套局部量，只是展开点不同）：
  M0 线性      ：只用 s1
  M1 单点二次  ：s1 + s2/2，全在 x* 处取        <- 判决性：先验是否沿模糊方向局部高斯
  M2 分段局部  ：每段 [alpha_i, alpha_{i+1}] 用左端点 y_i 的 score/Hessian 预测增量
                 （检验局部二次性；段数->无穷时收敛到精确值，也是求积精度检验）

另外报 conditioning gap：
      (s2/2) / Lambda_Gauss
  即"局部先验比边际谱锐多少倍"。若它 ~ 20，就解释了 Lambda/Lambda_Gauss ~ 20。

数值注意：dD/dx - I 在低噪声下接近 0（好去噪器 J ~ I），所以
      <delta, (J - I) delta>
是相减。本脚本记录 ||w|| = ||J*delta - delta|| 与 ||delta|| 之比来判断有效精度，
并用有限差分验证 VJP。

用法：
  python probe_jacobian.py --smoke
  python probe_jacobian.py --nimg 20 --kernels gaussian:1 defocus:1 motion:3
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from putils.convolution import GaussianBlur, DefocusBlur, AiryDisc, MotionBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
H = W = 64
KS = 15
POOL = "/tmp/pool_ffhq64_256_s0.pt"

KERNELS = [
    ("gaussian", 1.0), ("gaussian", 2.0),
    ("defocus", 1.0), ("defocus", 2.0),
    ("airy", 4.0),
    ("motion", 3.0), ("motion", 7.0),
    ("lowpass", 0.25),
]
ALPHAS = [0.15, 0.25, 0.4, 0.6, 0.8, 1.0]


def blur(x, k, gate):
    if gate is not None:
        return torch.fft.irfft2(torch.fft.rfft2(x.to(torch.float64), dim=(-2, -1))
                                * gate, s=(H, W), dim=(-2, -1)).to(DT)
    if k.ndim == 2:
        k = k[None, None]
    kh, kw = k.shape[-2:]
    pad = torch.zeros((1, x.shape[1], H, W), device=x.device, dtype=x.dtype)
    top, left = (H - kh) // 2, (W - kw) // 2
    pad[:, :, top:top + kh, left:left + kw] = k.to(x.dtype)
    return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))
                            * torch.fft.rfft2(pad, dim=(-2, -1)),
                            s=(H, W), dim=(-2, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=20)
    ap.add_argument("--kernels", nargs="*", default=None)
    ap.add_argument("--alphas", nargs="*", type=float, default=ALPHAS)
    ap.add_argument("--t", type=float, default=0.002)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    if args.smoke:
        nimg, kers, alphas = 1, [("gaussian", 1.0)], [0.4]
    else:
        nimg, alphas = args.nimg, args.alphas
        kers = [(s.split(":")[0], float(s.split(":")[1]))
                for s in args.kernels] if args.kernels else KERNELS

    print(f"[cfg] nimg={nimg} kernels={kers} alphas={alphas} t={args.t}",
          flush=True)

    model = load_local_model(CKPT).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ---- 可微性冒烟 ----
    xt = torch.randn(1, 3, H, W, device=DEV, dtype=DT) * 0.3
    xt.requires_grad_(True)
    with torch.enable_grad():
        Dt = model(xt, args.t)
        loss = Dt.sum()
        g = torch.autograd.grad(loss, xt)[0]
    print(f"[diff-check] D finite={bool(torch.isfinite(Dt).all())} "
          f"grad finite={bool(torch.isfinite(g).all())} "
          f"|grad|={float(g.abs().mean()):.3e}", flush=True)
    del xt, Dt, g
    torch.cuda.empty_cache()

    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:nimg].clone()
    print(f"[pool] cached {tuple(pool.shape)}, using first {nimg}", flush=True)

    def make(fam, th):
        if fam == "lowpass":
            fx = np.fft.fftfreq(H) * H
            fy = np.fft.fftfreq(W) * W
            FX, FY = np.meshgrid(fx, fy, indexing="ij")
            R = np.sqrt(FX ** 2 + FY ** 2)
            g = torch.from_numpy((R <= th * R.max()).astype(np.float64))
            return None, g[:, :W // 2 + 1].to(DEV)[None, None]
        cls = {"gaussian": GaussianBlur, "defocus": DefocusBlur,
               "airy": AiryDisc, "motion": MotionBlur}[fam]
        return cls(kernel_size=KS, device=DEV, dtype=DT).kernel(float(th)), None

    t2 = args.t ** 2

    def local_terms(x0, delta):
        """在 x0 处算 s1 = <grad log p, delta> 与 s2 = delta^T Hess log p delta。
        返回 (s1, s2, ||w||, ||delta||)，w = (dD/dx) delta - delta。"""
        x0 = x0.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            D0 = model(x0, args.t)
            # grad log p = (D - x)/t^2  =>  s1 = <D - x, delta>/t^2
            s1 = float((D0 - x0).flatten().dot(delta.flatten())) / t2
            # VJP：grad(D0, x0, grad_outputs=delta) = (dD/dx)^T delta
            Jd = torch.autograd.grad(D0, x0, grad_outputs=delta,
                                     retain_graph=False)[0]
        # Hess log p = (dD/dx - I)/t^2  =>  s2 = <delta, (J - I) delta>/t^2
        w = (Jd - delta).detach()
        s2 = float(delta.flatten().dot(w.flatten())) / t2
        return s1, s2, float(w.norm()), float(delta.norm())

    @torch.no_grad()
    def fd_check(x0, delta, eps=1e-3):
        """有限差分验证 VJP：<delta, J delta> ?= <delta, [D(x+e d)-D(x-e d)]/(2e)>。"""
        Dp = model(x0 + eps * delta, args.t)
        Dm = model(x0 - eps * delta, args.t)
        Jd_fd = (Dp - Dm) / (2 * eps)
        return float(delta.flatten().dot(Jd_fd.flatten()))

    # ---------------- 扫描 ----------------
    records = []
    for fam, th in kers:
        k, gate = make(fam, th)
        y1 = blur(xs, k, gate)
        d1 = y1 - xs                                       # delta（alpha=1）
        print(f"\n[{fam}:{th:g}]", flush=True)

        # 每个 alpha 的点。j=0 是 x* 本身，但 delta 仍取满位移 d1
        # （这样 S1[0]/S2[0] 就是在 x* 处的 Taylor 系数）
        pts = [(xs, d1)]
        for a in alphas:
            pts.append(((1.0 - a) * xs + a * y1, d1))

        # 逐图算局部量
        per = []           # per[i] = list over pts of (s1, s2, wnorm, dnorm)
        for i in range(nimg):
            row = []
            for ya, da in pts:
                s1, s2, wn, dn = local_terms(ya[i:i + 1], da[i:i + 1])
                row.append(dict(s1=s1, s2=s2, wnorm=wn, dnorm=dn))
                torch.cuda.empty_cache()
            per.append(row)
            if (i + 1) % 5 == 0:
                print(f"  img {i+1}/{nimg}  "
                      f"s1(x*)={per[i][0]['s1']:+.3e}  "
                      f"s2(x*)={per[i][0]['s2']:+.3e}  "
                      f"||w||/||d||={per[i][0]['wnorm']/max(per[i][0]['dnorm'],1e-9):.2e}",
                      flush=True)

        # ---- VJP 有限差分校验（第 0 张图，x* 处）----
        if args.smoke:
            x0 = xs[0:1].detach().clone().requires_grad_(True)
            with torch.enable_grad():
                D0 = model(x0, args.t)
                Jd = torch.autograd.grad(D0, x0, grad_outputs=d1[0:1],
                                         retain_graph=False)[0]
            vjp = float(d1[0:1].flatten().dot(Jd.flatten()))
            fd = fd_check(xs[0:1], d1[0:1])
            print(f"  [fd-check] <d,Jd> autograd={vjp:.6e}  finite-diff={fd:.6e}  "
                  f"rel.err={abs(vjp-fd)/max(abs(fd),1e-12):.2e}", flush=True)

        # 汇总：对图求均值
        npts = len(pts)
        alist = [0.0] + list(alphas)                     # [0, a1, a2, ...]
        S1 = [float(np.mean([per[i][j]["s1"] for i in range(nimg)]))
              for j in range(npts)]
        S2 = [float(np.mean([per[i][j]["s2"] for i in range(nimg)]))
              for j in range(npts)]
        WN = [float(np.mean([per[i][j]["wnorm"] for i in range(nimg)]))
              for j in range(npts)]
        DN = [float(np.mean([per[i][j]["dnorm"] for i in range(nimg)]))
              for j in range(npts)]

        # M2：分段局部。第 j 段从 alpha_{j} 到 alpha_{j+1}，位移 = (a_{j+1}-a_j)*delta1
        #   dLambda_j = (a_{j+1}-a_j)*s1_j + 0.5*(a_{j+1}-a_j)^2 * s2_j
        # 其中 s1_j/s2_j 是左端点 y_{alpha_j} 处对 delta1 的收缩（已含 1/t^2）
        m2_cum = [0.0]
        for j in range(len(alphas)):
            da = alist[j + 1] - alist[j]
            m2_cum.append(m2_cum[-1] + da * S1[j] + 0.5 * da * da * S2[j])

        for j, a in enumerate(alphas):
            m0 = a * S1[0]
            m1 = m0 + 0.5 * a * a * S2[0]
            records.append(dict(
                family=fam, theta=float(th), alpha=float(a), t=args.t,
                dnorm=DN[j + 1], wnorm=WN[j + 1],
                s1_at_xstar=S1[0], s2_at_xstar=S2[0],
                s1_at_point=S1[j + 1], s2_at_point=S2[j + 1],
                pred_linear=m0, pred_quad=m1, pred_piecewise=m2_cum[j + 1],
                alpha_list=alist, s1_list=S1, s2_list=S2,
                wnorm_list=WN, dnorm_list=DN))
            print(f"  a={a:<5g} ||d||={DN[j+1]:7.2f} "
                  f"||w||/||d||={WN[j+1]/max(DN[j+1],1e-9):.1e} "
                  f"s1={S1[j+1]:+.4e} s2={S2[j+1]:+.4e} "
                  f"| M0={m0:+10.1f} M1={m1:+10.1f} M2={m2_cum[j+1]:+10.1f}",
                  flush=True)

        del y1, d1
        torch.cuda.empty_cache()

    outfn = os.path.join(OUT, f"probe_jacobian_t{args.t:g}.json")
    with open(outfn, "w") as f:
        json.dump(dict(cfg=dict(nimg=nimg, kernels=kers, alphas=alphas,
                                t=args.t, ckpt=CKPT,
                                identity="grad log p = (x-D)/t^2; "
                                         "Hess = (I - dD/dx)/t^2"),
                       records=records), f, indent=1)
    print(f"\n[saved] {outfn}", flush=True)


if __name__ == "__main__":
    main()
