"""高斯近似误差的 theta 依赖性：方向判定。

已知：
  * likelihood_fn 已标定正确（高斯先验上相对误差 2e-4）
  * theta=0 处高斯近似误差 = -25933（远大于证据比 -2235）

问题：这个误差在 theta=theta* 处是更大还是更小？
若更负（清晰图的非高斯增益更大），则高斯近似系统性低估 theta* 的证据，
真实证据比 theta=0 的优势【大于】近似值 —— 定性结论被加强。
若更小，则近似高估 theta*，定性结论可能反转。

做法：对同一个 y，比较两处的"非高斯增益"
    Delta(z) = 精确 log p_t(z) - 高斯近似 log p_t(z)
  * z = y            （模糊观测，theta=0 模型下的有效点）
  * z = x*           （真实清晰图，theta=theta* 模型下的有效点）
用同一套估计出的平均谱 S，保证可比。

用法： python diag_gauss_bias.py
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


def radial_bins(nbin=48):
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
    S_N = 64
    TSTAR = 1.5
    TMIN = 0.002

    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    blur = BlurOp()
    idx, NB = radial_bins()
    idx_t = torch.from_numpy(idx).to(DEV)

    pool, _ = ode_sampler(ode_flow, shape=(S_N + 1, 3, 64, 64), t_spans=t_spans,
                          device=DEV)
    pool = pool.to(DT)
    x_star = pool[S_N:S_N + 1]
    xs = pool[:S_N]

    with torch.no_grad():
        P = (torch.fft.rfft2(xs, dim=(-2, -1)).abs() ** 2) / (H * W)
        Pc = P.mean(1)
        Srad = torch.zeros(NB, device=DEV, dtype=DT)
        for b in range(NB):
            m = (idx_t == b).reshape(-1)
            if m.any():
                Srad[b] = Pc.reshape(S_N, -1)[:, m].mean()
        # 目标 p_t 的谱：S + t^2
        Sfull = (Srad + TMIN ** 2)[idx_t][None, None]

    @torch.no_grad()
    def exact_logp(z):
        return -float(likelihood_fn(z.to(DT), ode_flow,
                                    t_spans=[80.0, TMIN]).item())

    @torch.no_grad()
    def gauss_logp(z):
        Z = torch.fft.rfft2(z, dim=(-2, -1))
        Zp = (Z.abs() ** 2).mean(1, keepdim=True) / (H * W)
        return float(-(Zp / (2 * Sfull)).sum()
                     - 0.5 * torch.log(2 * np.pi * Sfull).sum())

    y_clean = blur.A(x_star, TSTAR)
    print("=" * 92)
    print("高斯近似的非高斯增益 Delta(z) = 精确 log p_t(z) - 高斯 log p_t(z)")
    print("=" * 92)
    print(f"{'z':>22} | {'精确 log p_t':>14} | {'高斯 log p_t':>14} | {'Delta':>12}")
    print("-" * 92)

    deltas = {}
    for name, z in [("y (blurred obs)", y_clean),
                    ("x* (true sharp)", x_star),
                    ("pool mean-ish (sharp)", xs[:1])]:
        e, g = exact_logp(z), gauss_logp(z)
        deltas[name] = e - g
        print(f"{name:>22} | {e:>14.1f} | {g:>14.1f} | {e-g:>+12.1f}")

    print("\n[判据]")
    d_sharp = deltas["x* (true sharp)"]
    d_blur = deltas["y (blurred obs)"]
    print(f"  Delta(清晰图) - Delta(模糊图) = {d_sharp - d_blur:+.1f}")
    if d_sharp < d_blur - 500:
        print("  => 清晰图的非高斯增益显著更大：高斯近似系统性【低估】theta* 的证据，")
        print("     真实证据比 theta=0 的优势大于近似值，定性结论被【加强】。")
    elif d_sharp > d_blur + 500:
        print("  => 高斯近似【高估】theta* 的证据，定性结论可能反转，需谨慎。")
    else:
        print("  => 两者接近：近似误差近似与 theta 无关，证据比可能大致可靠。")


if __name__ == "__main__":
    main()
