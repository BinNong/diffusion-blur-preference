"""模糊偏好的噪声水平依赖性 + Hutchinson 噪声底。

两个目的：
  A. Delta(t) = log p_t(模糊图) - log p_t(清晰原图) 随 t 的变化
     - 若随 t 剧烈/无规律漂移 -> 小 t 外推伪影
     - 若平滑衰减 -> 稳健的先验性质
     附带高斯对照 Delta_Gauss(t) = sum_f (1-|h|^2)|X_f|^2 / (2(S_f+t^2))
     以及"非高斯放大倍数"随 t 的变化
  B. Hutchinson 噪声底：同一点重复 K 次（每次内部重抽 eps），
     看 log p_t 的重复性。这决定了任何单点差值的不确定度。

注意：likelihood_fn 内部用自适应 RK45（rtol=atol=1e-5），
所以 t_spans 的梯度不影响积分精度；真正的随机源是单次 Hutchinson 向量。

用法： python check_tdependence.py
"""

import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn, ode_sampler

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
H = W = 64
TSTAR = 1.5
NB = 48
TS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
S_N = 128
K_REP = 8


class BlurOp:
    def __init__(self, ks=15):
        lin = torch.linspace(-(ks // 2) + 1, ks // 2, ks, device=DEV, dtype=DT)
        self.grid = lin[None] ** 2 + lin[:, None] ** 2

    def kernel(self, s):
        if s == 0:
            return None
        k = torch.exp(-self.grid / (2.0 * s ** 2))
        return (k / k.sum())[None, None]

    def A(self, x, s):
        k = self.kernel(s)
        if k is None:
            return x
        B, C, Hh, Ww = x.shape
        pad = torch.zeros((1, C, Hh, Ww), device=DEV, dtype=DT)
        kh, kw = k.shape[-2:]
        t, l = (Hh - kh) // 2, (Ww - kw) // 2
        pad[:, :, t:t + kh, l:l + kw] = k
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))
                                * torch.fft.rfft2(pad, dim=(-2, -1)),
                                s=(Hh, Ww), dim=(-2, -1))


def radial_bins(nbin=NB):
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)
    Rr = R[:, :W // 2 + 1]
    edges = np.linspace(0, Rr.max() + 1e-6, nbin + 1)
    idx = np.clip(np.digitize(Rr.ravel(), edges) - 1, 0, nbin - 1)
    return idx.reshape(Rr.shape), nbin


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    blur = BlurOp()
    idx, _ = radial_bins()
    idx_t = torch.from_numpy(idx).to(DEV)

    CACHE = "/tmp/pool_ffhq64_256_s0.pt"
    if os.path.exists(CACHE):
        pool = torch.load(CACHE, map_location=DEV).to(DT)
        print(f"[pool] cached {tuple(pool.shape)}", flush=True)
    else:
        pool, _ = ode_sampler(ode_flow, shape=(S_N + 1, 3, H, W), t_spans=t_spans,
                              device=DEV)
        pool = pool.to(DT)
        torch.save(pool.cpu(), CACHE)
        pool = pool.to(DEV)
    x_star = pool[255:256]          # 与 check_baseline 用同一张 x*
    xs = pool[:S_N]
    y_clean = blur.A(x_star, TSTAR)
    print(f"[pool] {S_N}+1", flush=True)

    with torch.no_grad():
        P = (torch.fft.rfft2(xs, dim=(-2, -1)).abs() ** 2) / (H * W)
        Pc = P.mean(1)
        Srad = torch.zeros(NB, device=DEV, dtype=DT)
        for b in range(NB):
            m = (idx_t == b).reshape(-1)
            if m.any():
                Srad[b] = Pc.reshape(S_N, -1)[:, m].mean()
        Zx = (torch.fft.rfft2(x_star, dim=(-2, -1)).abs() ** 2).mean(1, keepdim=True) / (H * W)
        Zy = (torch.fft.rfft2(y_clean, dim=(-2, -1)).abs() ** 2).mean(1, keepdim=True) / (H * W)
    print(f"[spectrum] S in [{float(Srad.min()):.2e}, {float(Srad.max()):.2e}]", flush=True)

    @torch.no_grad()
    def exact_logp(z, t):
        return -float(likelihood_fn(z.to(DT), ode_flow,
                                    t_spans=[80.0, float(t)]).item())

    @torch.no_grad()
    def exact_logp_avg(z, t, k=K_REP):
        vals = [exact_logp(z, t) for _ in range(k)]
        return float(np.mean(vals)), float(np.std(vals))

    def gauss_logp(z_pow, t):
        Sf = (Srad + t ** 2)[idx_t][None, None]
        return float(-(z_pow / (2 * Sf)).sum()
                     - 0.5 * torch.log(2 * np.pi * Sf).sum())

    # ---- B. Hutchinson 噪声底（先做，因为它决定 A 的可信度）----
    print("\n" + "=" * 100)
    print(f"B. Hutchinson 噪声底：t=0.002，同一点重复 K={K_REP} 次")
    print("=" * 100)
    my, sy = exact_logp_avg(y_clean, 0.002)
    mx, sx = exact_logp_avg(x_star, 0.002)
    print(f"  log p(y_blur) = {my:.1f} +/- {sy:.1f}")
    print(f"  log p(x*)     = {mx:.1f} +/- {sx:.1f}")
    print(f"  Delta         = {my-mx:+.1f}  (Hutchinson 引入的不确定度约 "
          f"{np.sqrt(sy**2+sx**2):.1f})", flush=True)

    # ---- A. t 依赖性 ----
    print("\n" + "=" * 100)
    print("A. Delta(t) = log p_t(模糊) - log p_t(清晰)，及高斯对照")
    print("=" * 100)
    print(f"{'t':>7} | {'log p_t(y)':>12} | {'log p_t(x*)':>12} | {'Delta':>10} | "
          f"{'高斯 Delta':>11} | {'放大倍数':>9}")
    print("-" * 100)
    for t in TS:
        ey = exact_logp(y_clean, t)
        ex = exact_logp(x_star, t)
        dg = gauss_logp(Zy, t) - gauss_logp(Zx, t)
        amp = (ey - ex) / dg if abs(dg) > 1e-6 else float("nan")
        print(f"{t:>7} | {ey:>12.1f} | {ex:>12.1f} | {ey-ex:>+10.1f} | "
              f"{dg:>+11.1f} | {amp:>9.2f}", flush=True)

    print("\n[判据]")
    print("  * B 的重复标准差若 << |Delta| -> 单次估计足够，差值可信")
    print("  * A 的 Delta(t) 若平滑衰减 -> 模糊偏好是稳健性质，非小 t 外推伪影")
    print("  * 放大倍数若大致恒定 -> 可表述为'扩散先验把高斯偏好放大 ~常数 倍'")


if __name__ == "__main__":
    main()
