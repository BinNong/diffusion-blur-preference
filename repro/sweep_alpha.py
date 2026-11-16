"""Alpha-插值实验：把弱模糊区打开，测先验沿模糊方向的局部对数密度剖面

动机（RESULTS_2026-10-11.md §3.9）：
  旧的 theta 网格让 kunif 饱和在 0.92-0.99，幂律指数是杠杆假象。
  要测真实的小 kunif 行为，不需要构造更弱的核。

关键恒等式：像素空间线性插值
      y_alpha = (1-alpha) x + alpha (h * x) = x + alpha (h - I) x
  就是先验沿"从清晰图到模糊图"这条线段的参数化。冒烟模式数值验证：
  y_alpha 的谱 精确等于 x 的谱乘 ((1-alpha) + alpha*h_hat)（误差 ~1e-7，浮点精度）。

注意（一个我一开始推错的点）：有效算子是 h_alpha = (1-alpha) I + alpha h，
但 1-|h_alpha|^2 并不等于 (1-|h|^2)*alpha*(2-alpha)。因为核放在 64x64 的
(32,32)（= 半图），其 DFT 带符号 (-1)^{fx+fy}，于是
      |h_alpha|^2 = (1-alpha)^2 + 2 alpha (1-alpha) s |h| + alpha^2 |h|^2,  s = ±1
即有效滤波在半个频域上符号翻转。所以：
  * 主变量用 alpha（线段参数，物理意义干净：Lambda(alpha) 就是对数密度剖面）
  * kunif(alpha) 只作为"有效算子 h_alpha 的平均衰减"这个描述量，从实际 DFT 算，不假设闭式
  * 高斯对照 Lambda_Gauss(alpha) 同样从实际谱算（脚本里就是逐频率算的）

于是这个实验一次给出四件事：
  1. alpha 连续扫过 (0,1]，把旧 theta 网格完全够不着的弱模糊端打开
  2. 对同一个核形状测响应，消除"改 theta 时形状一起变"的混淆
  3. Lambda(alpha) 就是先验沿模糊方向的 对数密度剖面（Morse-Bott 局部几何）
  4. 两个尖锐的理论极限：
       高斯先验        => Lambda_Gauss 由谱精确决定，可逐频率算出来做基准
       分数项主导(s1≠0) => Lambda ~ alpha^1
       曲率项主导(s1=0) => Lambda ~ alpha^2
     实测局部指数落在 1 和 2 之间，其位置本身就是"先验在 x* 处分数是否为零"的判据

噪声底：Hutchinson 差分 std ~300-500 nats，20 图均值 SE ~70-110 nats。
  实测 alpha=0.15 处 Lambda~950-1300，SNR 8-12，可用。

用法：
  python sweep_alpha.py --smoke
  python sweep_alpha.py --nimg 20 --alphas 0.15 0.25 0.4 0.6 0.8 1.0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, ode_sampler, likelihood_fn
from putils.convolution import GaussianBlur, DefocusBlur, AiryDisc, MotionBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
H = W = 64
NB = 48
KS = 15
POOL = "/tmp/pool_ffhq64_256_s0.pt"

# 要测的核：覆盖 4 个观察族 + 理想低通作参照。每个核单独扫 alpha。
KERNELS = [
    ("gaussian", 1.0),
    ("gaussian", 2.0),
    ("defocus", 1.0),
    ("defocus", 2.0),
    ("airy", 2.0),
    ("airy", 4.0),
    ("motion", 3.0),
    ("motion", 7.0),
    ("lowpass", 0.25),
]


def mode_grid():
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    return np.sqrt(FX ** 2 + FY ** 2)


def radial_bins(nbin=NB):
    R = mode_grid()
    Rr = R[:, :W // 2 + 1]
    edges = np.linspace(0, Rr.max() + 1e-6, nbin + 1)
    idx = np.clip(np.digitize(Rr.ravel(), edges) - 1, 0, nbin - 1)
    return idx.reshape(Rr.shape), nbin, Rr


def cross_first(prof, rho, level):
    """prof 随 rho 递减，返回首次穿越 level 的半径（相邻点线性插值）。"""
    for i in range(len(prof) - 1):
        a, b = prof[i] - level, prof[i + 1] - level
        if a * b <= 0 and a != b:
            return rho[i] + (level - prof[i]) * (rho[i + 1] - rho[i]) / \
                (prof[i + 1] - prof[i])
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=20)
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.15, 0.25, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--ts", nargs="*", type=float, default=[0.002])
    ap.add_argument("--kernels", nargs="*", default=None,
                    help="'family:theta' 列表，默认用内置 KERNELS")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default="sweep_alpha.json",
                    help="输出文件名（分开跑不同核时用，避免互相覆盖）")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    if args.smoke:
        nimg, kers, alphas, ts = 2, [("gaussian", 1.5)], [0.5, 1.0], [0.002]
    else:
        nimg, alphas, ts = args.nimg, args.alphas, args.ts
        kers = [(s.split(":")[0], float(s.split(":")[1]))
                for s in args.kernels] if args.kernels else KERNELS

    print(f"[cfg] nimg={nimg} kernels={kers} alphas={alphas} ts={ts}", flush=True)

    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    idx, _, Rr = radial_bins()
    idx_t = torch.from_numpy(idx).to(DEV)
    rho_b = np.array([Rr.ravel()[idx.ravel() == b].mean() if (idx == b).any() else 0.0
                      for b in range(NB)])

    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:nimg].clone()
    print(f"[pool] cached {tuple(pool.shape)}, using first {nimg}", flush=True)

    # ---------------- 径向谱 S（用全部 256 张估）----------------
    with torch.no_grad():
        P = (torch.fft.rfft2(pool, dim=(-2, -1)).abs() ** 2) / (H * W)
        Pc = P.mean(1)
        Srad = torch.zeros(NB, device=DEV, dtype=torch.float64)
        for b in range(NB):
            m = (idx_t == b).reshape(-1)
            if m.any():
                Srad[b] = Pc.reshape(P.shape[0], -1)[:, m].mean()
        Sfull = Srad[idx_t][None, None]                      # (1,1,H,W//2+1)
        Zx = (torch.fft.rfft2(xs, dim=(-2, -1)).abs() ** 2).mean(1, keepdim=True) / (H * W)

    B = args.batch

    @torch.no_grad()
    def logp_chunked(z, t):
        out = []
        for i in range(0, z.shape[0], B):
            nll = likelihood_fn(z[i:i + B].to(DT), ode_flow,
                                t_spans=[80.0, float(t)])
            out.extend([-float(v) for v in nll.reshape(-1)])
            torch.cuda.empty_cache()
        return np.array(out)

    lp_sharp = {t: logp_chunked(xs, t) for t in ts}
    for t in ts:
        print(f"[sharp] t={t} mean log p = {lp_sharp[t].mean():.1f}", flush=True)

    # ---------------- 核工厂 ----------------
    def make_kernel(fam, th):
        if fam == "lowpass":
            fx = np.fft.fftfreq(H) * H
            fy = np.fft.fftfreq(W) * W
            FX, FY = np.meshgrid(fx, fy, indexing="ij")
            R = np.sqrt(FX ** 2 + FY ** 2)
            # 注意：谱门必须切到 rfft 半平面 (H, W//2+1)，不能给全 (H, W)
            gate = torch.from_numpy(
                (R <= th * R.max()).astype(np.float64))[:, :W // 2 + 1]
            return None, gate.to(DEV)[None, None]            # 谱门，无像素核
        if fam == "specgauss":
            # 纯谱高斯：没有核截断、没有循环回绕伪影（对照实验用）。
            # 注意 gate 必须传"振幅" |h| 而不是功率 |h|^2：
            #   blur() 用它乘 rfft2(x)，h2_of_alpha() 用它算 ((1-a)+a|h|)^2
            # 两者都要求 |h|。low-pass 的门是 {0,1}，振幅=功率，不受影响。
            fx = np.fft.fftfreq(H) * H
            fy = np.fft.fftfreq(W) * W
            FX, FY = np.meshgrid(fx, fy, indexing="ij")
            f2 = (FX ** 2 + FY ** 2)[:, :W // 2 + 1] / (H * W)
            g = torch.from_numpy(np.exp(-2 * np.pi ** 2 * th ** 2 * f2))
            return None, g.sqrt().to(DEV)[None, None].to(torch.float64)
        cls = {"gaussian": GaussianBlur, "defocus": DefocusBlur,
               "airy": AiryDisc, "motion": MotionBlur}[fam]
        k = cls(kernel_size=KS, device=DEV, dtype=DT).kernel(float(th))
        return k, None

    def blur(x, k, gate):
        if gate is not None:
            return torch.fft.irfft2(torch.fft.rfft2(x.to(torch.float64),
                                                    dim=(-2, -1)) * gate,
                                    s=(H, W), dim=(-2, -1)).to(DT)
        if k.ndim == 2:
            k = k[None, None]
        kh, kw = k.shape[-2:]
        pad = torch.zeros((1, x.shape[1], H, W), device=x.device, dtype=x.dtype)
        top, left = (H - kh) // 2, (W - kw) // 2
        pad[:, :, top:top + kh, left:left + kw] = k.to(x.dtype)
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))
                                * torch.fft.rfft2(pad, dim=(-2, -1)),
                                s=(H, W), dim=(-2, -1))

    def h2_of_alpha(alpha, k, gate):
        """(1,1,H,W//2+1) 的 |h_alpha|^2，h_alpha = (1-alpha) I + alpha h。"""
        if gate is not None:
            h2a = ((1.0 - alpha) + alpha * gate.to(torch.float64)) ** 2
            return h2a
        if k.ndim == 2:
            k = k[None, None]
        kh, kw = k.shape[-2:]
        pad = torch.zeros((1, 1, H, W), device=DEV, dtype=torch.float64)
        top, left = (H - kh) // 2, (W - kw) // 2
        pad[:, :, top:top + kh, left:left + kw] = k.to(torch.float64)
        delta = torch.zeros_like(pad)
        delta[:, :, 0, 0] = 1.0
        ha = (1.0 - alpha) * delta + alpha * pad
        return torch.fft.rfft2(ha, dim=(-2, -1)).abs() ** 2

    def desc(h2a):
        """kunif / kappa / 过渡带宽度 / 阻带尾。"""
        with torch.no_grad():
            kunif = float((1.0 - h2a).mean())
            kappa = float(((1.0 - h2a) * Sfull).sum() / Sfull.sum())
            h2n = h2a[0, 0].cpu().numpy()
            hb = np.array([h2n[idx == b].mean() if (idx == b).any() else 0.0
                           for b in range(NB)])
            r9 = cross_first(hb, rho_b, 0.1)
            r1 = cross_first(hb, rho_b, 0.9)
            width = float(r9 - r1) if (np.isfinite(r9) and np.isfinite(r1)) \
                else float("nan")
            tail = float(h2n[idx == NB - 1].mean()) if (idx == NB - 1).any() \
                else float("nan")
        return kunif, kappa, width, tail

    # ---------------- 冒烟：验证插值 == alpha-模糊 ----------------
    if args.smoke:
        print("\n[smoke] 验证 y_alpha 的谱 == x 的谱 * ((1-a)+a*h_hat)")
        k, gate = make_kernel(*kers[0])
        y1 = blur(xs, k, gate)
        for a in alphas:
            ya = (1.0 - a) * xs + a * y1
            if gate is not None:
                pred = torch.fft.irfft2(torch.fft.rfft2(xs.to(torch.float64),
                                                        dim=(-2, -1))
                                        * ((1 - a) + a * gate.to(torch.float64)),
                                        s=(H, W), dim=(-2, -1))
            else:
                kk = k[None, None] if k.ndim == 2 else k
                kh, kw = kk.shape[-2:]
                pad = torch.zeros((1, 1, H, W), device=DEV, dtype=torch.float64)
                pad[:, :, (H - kh) // 2:(H - kh) // 2 + kh,
                    (W - kw) // 2:(W - kw) // 2 + kw] = kk.to(torch.float64)
                pred = torch.fft.irfft2(torch.fft.rfft2(xs.to(torch.float64),
                                                        dim=(-2, -1))
                                        * ((1 - a) + a *
                                           torch.fft.rfft2(pad, dim=(-2, -1))),
                                        s=(H, W), dim=(-2, -1))
            err = float((ya.to(torch.float64) - pred).abs().max())
            print(f"  alpha={a}: max|插值 - alpha模糊| = {err:.3e}")
        print("[smoke] OK（误差应在浮点精度量级）")
        return

    # ---------------- 扫描 ----------------
    records = []
    for fam, th in kers:
        k, gate = make_kernel(fam, th)
        y1 = blur(xs, k, gate)
        print(f"\n[{fam}:{th:g}]", flush=True)
        for a in alphas:
            ya = (1.0 - a) * xs + a * y1
            h2a = h2_of_alpha(a, k, gate)
            kunif, kappa, width, tail = desc(h2a)
            with torch.no_grad():
                Zy = (torch.fft.rfft2(ya, dim=(-2, -1)).abs() ** 2).mean(
                    1, keepdim=True) / (H * W)
            for t in ts:
                lp = logp_chunked(ya, t)
                lam = lp - lp_sharp[t]
                Sf = Sfull + t ** 2
                lam_g = float((((Zx - Zy) / (2 * Sf)).sum(dim=(1, 2, 3))).mean())
                records.append(dict(
                    family=fam, theta=float(th), alpha=float(a), t=float(t),
                    kappa=kappa, kunif=kunif, width=width, tail=tail,
                    lam_mean=float(lam.mean()), lam_std=float(lam.std()),
                    lam_min=float(lam.min()), lam_max=float(lam.max()),
                    lam_gauss_mean=lam_g,
                    logp_blur_mean=float(lp.mean()),
                    logp_sharp_mean=float(lp_sharp[t].mean()),
                    # 二次性指标：局部二次先验应使 Lambda/alpha^2 恒定
                    lam_over_a2=float(lam.mean() / (a * a)) if a > 0 else float("nan"),
                    lam_per_img=[float(v) for v in lam]))
                print(f"  a={a:<5g} t={t:<6} kunif={kunif:.4f} kappa={kappa:5.3f} "
                      f"width={width:8.3f} tail={tail:.1e} "
                      f"Lambda={lam.mean():+10.1f} +/-{lam.std():6.1f} "
                      f"Gauss={lam_g:+8.1f} x{lam.mean()/lam_g:6.2f} "
                      f"L/a^2={lam.mean()/(a*a):9.0f}", flush=True)
                del lp
                torch.cuda.empty_cache()
        del y1, k, gate
        torch.cuda.empty_cache()

    outfn = os.path.join(OUT, args.out)
    with open(outfn, "w") as f:
        json.dump(dict(cfg=dict(nimg=nimg, kernels=kers, alphas=alphas, ts=ts,
                                ckpt=CKPT, pool="first nimg of 256-pool seed0",
                                identity="y_a=(1-a)x+a(h*x); "
                                         "1-|h_a|^2=(1-|h|^2)a(2-a)"),
                       records=records), f, indent=1)
    print(f"\n[saved] {outfn}", flush=True)


if __name__ == "__main__":
    main()
