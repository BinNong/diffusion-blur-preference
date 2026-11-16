"""验证分数恒等式：沿直线的 score 线积分能否复现精确的 Lambda？

恒等式（固定 t，对任意光滑路径 gamma 从 x 到 y）：
      log p_t(y) - log p_t(x) = ∫_0^1 <grad log p_t(gamma(tau)), gamma'(tau)> dtau
这是微积分基本定理，**只要 score 正确就精确成立**。所以它是最干净的判据：

  * 线积分能复现 ODE 精确值  => 分数恒等式 (D-x)/t^2 与符号都对，
                              那么"局部二次模型失败"就是真实的物理结论
  * 不能复现                  => 恒等式或符号有错，先修工具再下结论

同时对同一个量给三种预测，逐层加结构：
  M0 线性（只用 x* 处的 score 投影）
  M1 单点二次（x* 处 score + Hessian）
  M2 分段局部（每段用左端点的 score/Hessian）

用法： python probe_score_integral.py --smoke
      python probe_score_integral.py --nimg 20 --kernels gaussian:1
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
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn
from putils.convolution import GaussianBlur, DefocusBlur, AiryDisc, MotionBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
H = W = 64
KS = 15
POOL = "/tmp/pool_ffhq64_256_s0.pt"


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
    ap.add_argument("--kernels", nargs="*", default=["gaussian:1"])
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.15, 0.25, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--t", type=float, default=0.002)
    ap.add_argument("--nquad", type=int, default=21,
                    help="线积分求积点数（奇数，含端点）")
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    kers = [(s.split(":")[0], float(s.split(":")[1])) for s in args.kernels]
    nimg = 2 if args.smoke else args.nimg
    alphas = [0.4] if args.smoke else args.alphas

    print(f"[cfg] nimg={nimg} kernels={kers} alphas={alphas} t={args.t} "
          f"nquad={args.nquad}", flush=True)

    model = load_local_model(CKPT).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x.to(DT), t, *a, **k))
    ode_flow, t_spans = get_ode_flow(prior)

    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:nimg].clone()

    t2 = args.t ** 2

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

    @torch.no_grad()
    def score(z):
        """grad log p_t(z) = (D(z,t) - z)/t^2"""
        return (model(z, args.t) - z) / t2

    @torch.no_grad()
    def logp(z):
        out = []
        for i in range(0, z.shape[0], args.batch):
            nll = likelihood_fn(z[i:i + args.batch].to(DT), ode_flow,
                                t_spans=[80.0, args.t])
            out.extend([-float(v) for v in nll.reshape(-1)])
            torch.cuda.empty_cache()
        return np.array(out)

    # Simpson 权重（nquad 为奇数）
    nq = args.nquad if args.nquad % 2 == 1 else args.nquad + 1
    taus = np.linspace(0.0, 1.0, nq)
    w = np.ones(nq)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w = w / 3.0 / (nq - 1)

    lp_sharp = logp(xs)
    print(f"[sharp] t={args.t} mean log p = {lp_sharp.mean():.1f}", flush=True)

    rows = []
    for fam, th in kers:
        k, gate = make(fam, th)
        y1 = blur(xs, k, gate)
        d1 = y1 - xs
        print(f"\n[{fam}:{th:g}]", flush=True)
        for a in alphas:
            ya = (1.0 - a) * xs + a * y1
            lp_y = logp(ya)
            lam_exact = lp_y - lp_sharp

            # 线积分：逐图算（每图一个 batch，简单可靠）
            integ = np.zeros(nimg)
            for i in range(nimg):
                acc = 0.0
                for tau, wt in zip(taus, w):
                    z = xs[i:i + 1] + tau * a * d1[i:i + 1]
                    g = score(z)
                    acc += wt * float(g.flatten().dot(d1[i:i + 1].flatten()))
                integ[i] = acc * a          # gamma' = a*d1
                torch.cuda.empty_cache()

            # 局部模型（在 x* 处）
            s1 = np.zeros(nimg)
            s2 = np.zeros(nimg)
            for i in range(nimg):
                x0 = xs[i:i + 1].detach().clone().requires_grad_(True)
                with torch.enable_grad():
                    D0 = model(x0, args.t)
                    Jd = torch.autograd.grad(D0, x0, grad_outputs=d1[i:i + 1],
                                             retain_graph=False)[0]
                s1[i] = float((D0 - x0).flatten().dot(d1[i:i + 1].flatten())) / t2
                wv = (Jd - d1[i:i + 1]).detach()
                s2[i] = float(d1[i:i + 1].flatten().dot(wv.flatten())) / t2
                torch.cuda.empty_cache()

            m0 = a * s1
            m1 = m0 + 0.5 * a * a * s2
            print(f"  a={a:<5g}  精确 Lambda = {lam_exact.mean():+10.1f} "
                  f"+/- {lam_exact.std():.0f}")
            print(f"          线积分(score) = {integ.mean():+10.1f} "
                  f"  => 相对差 {100*(integ.mean()-lam_exact.mean())/abs(lam_exact.mean()):+7.2f}%")
            print(f"          M0 线性       = {m0.mean():+10.1f} "
                  f"  => 相对差 {100*(m0.mean()-lam_exact.mean())/abs(lam_exact.mean()):+7.2f}%")
            print(f"          M1 单点二次   = {m1.mean():+10.1f} "
                  f"  => 相对差 {100*(m1.mean()-lam_exact.mean())/abs(lam_exact.mean()):+7.2f}%",
                  flush=True)
            for i in range(nimg):
                rows.append(dict(family=fam, theta=float(th), alpha=float(a),
                                 t=args.t, img=i,
                                 lam_exact=float(lam_exact[i]),
                                 lam_integral=float(integ[i]),
                                 pred_linear=float(m0[i]),
                                 pred_quad=float(m1[i]),
                                 s1=float(s1[i]), s2=float(s2[i])))
        del y1, d1
        torch.cuda.empty_cache()

    outfn = os.path.join(OUT, f"probe_score_integral_t{args.t:g}.json")
    with open(outfn, "w") as f:
        json.dump(dict(cfg=dict(nimg=nimg, kernels=kers, alphas=alphas,
                                t=args.t, nquad=nq, ckpt=CKPT),
                       rows=rows), f, indent=1)
    print(f"\n[saved] {outfn}", flush=True)


if __name__ == "__main__":
    main()
