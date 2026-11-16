"""Phase C 诊断：HQS 收敛性 + 多初值的必要性。

固定 (img, noise)，对若干 theta 做两件事：
  A. n_inner 扫描：L 是否随内循环步数稳定？
  B. 多初值：不同起点是否落到不同的 L？（若是，说明单初值找不到 min_x）

用法： python diag_phaseC.py
"""

import json
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
IMG_LO, IMG_HI = -1.0, 1.0


class BlurOp:
    def __init__(self, ks=15):
        lin = torch.linspace(-(ks // 2) + 1, ks // 2, ks, device=DEV, dtype=DT)
        self.grid = lin[None] ** 2 + lin[:, None] ** 2

    def kernel(self, s):
        if s == 0:
            return None
        k = torch.exp(-self.grid / (2.0 * s ** 2))
        return (k / k.sum())[None, None]

    def _K(self, k, C, H, W):
        pad = torch.zeros((1, C, H, W), device=DEV, dtype=DT)
        kh, kw = k.shape[-2:]
        t, l = (H - kh) // 2, (W - kw) // 2
        pad[:, :, t:t + kh, l:l + kw] = k
        return torch.fft.rfft2(pad, dim=(-2, -1))

    def A(self, x, s):
        k = self.kernel(s)
        if k is None:
            return x
        B, C, H, W = x.shape
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1)) * self._K(k, C, H, W),
                                s=(H, W), dim=(-2, -1))

    def eig(self, s, C, H, W):
        k = self.kernel(s)
        if k is None:
            return torch.ones((1, C, H, W // 2 + 1), device=DEV, dtype=DT)
        return self._K(k, C, H, W).abs() ** 2


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    IMG, NOISE, TSTAR = 0, 0.05, 1.5
    THETAS = [0.0, 0.75, 1.5, 2.0]
    INNERS = [60, 240, 600]
    N_INIT = 3

    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    blur = BlurOp()
    samples, _ = ode_sampler(ode_flow, shape=(1, 3, 64, 64), t_spans=t_spans, device=DEV)
    x_star = samples[0:1].to(DT)
    y_clean = blur.A(x_star, TSTAR)
    g = torch.Generator(device=DEV).manual_seed(0)
    y = (y_clean + NOISE * torch.randn(y_clean.shape, generator=g, device=DEV,
                                       dtype=DT)).to(DT)
    B, C, H, W = y.shape

    @torch.no_grad()
    def hqs(theta, n_inner, z0):
        ATy = blur.A(y, theta)  # 占位，下面重算
        ATy = _at(y, theta, blur)
        denom = blur.eig(theta, C, H, W) + 1.0
        sp = NOISE
        z = z0.clone()
        for _ in range(n_inner):
            xh = denoiser(z, sp).to(DT)
            v = 2 * xh - z
            xn = torch.fft.irfft2(torch.fft.rfft2(ATy + v, dim=(-2, -1)) / denom,
                                  s=(H, W), dim=(-2, -1))
            z = torch.clamp(z + xn - xh, IMG_LO, IMG_HI)
        return denoiser(z, sp).to(DT)

    def _at(x, s, bl):
        k = bl.kernel(s)
        if k is None:
            return x
        Bc, Cc, Hc, Wc = x.shape
        return torch.fft.irfft2(
            torch.conj(bl._K(k, Cc, Hc, Wc)) * torch.fft.rfft2(x, dim=(-2, -1)),
            s=(Hc, Wc), dim=(-2, -1))

    @torch.no_grad()
    def Lval(x, theta):
        resid = blur.A(x, theta) - y
        data = float((resid ** 2).sum().item() / (2 * NOISE ** 2))
        q = float(likelihood_fn(x.to(DT), ode_flow, t_spans=t_spans).item())
        return data + q, data, q

    INITS = {
        "y": lambda: y.clone(),
        "blur(y,theta)": None,     # 每个 theta 不同，运行时构造
        "randn*0.5": lambda: torch.randn_like(y) * 0.5,
    }

    print(f"[diag] img={IMG} noise={NOISE} theta*={TSTAR}\n", flush=True)
    print("=== A. n_inner 收敛性（初值 = y）===", flush=True)
    print(f"{'theta':>6} | " + " | ".join(f"n={n:<5}" for n in INNERS))
    for th in THETAS:
        row = []
        for ni in INNERS:
            x = hqs(th, ni, y.clone())
            L, d, q = Lval(x, th)
            row.append(f"{L:9.0f}")
            del x
            torch.cuda.empty_cache()
        print(f"{th:>6} | " + " | ".join(row), flush=True)

    print("\n=== B. 多初值（n_inner=240）===", flush=True)
    print(f"{'theta':>6} | " + " | ".join(f"{k:>12}" for k in
                                          ["y", "blur(y)", "randn*0.5"]) + " |   spread")
    for th in THETAS:
        vals = []
        x1 = hqs(th, 240, y.clone())
        vals.append(Lval(x1, th)[0]); del x1
        torch.cuda.empty_cache()
        x2 = hqs(th, 240, blur.A(y, th).clone())
        vals.append(Lval(x2, th)[0]); del x2
        torch.cuda.empty_cache()
        x3 = hqs(th, 240, (torch.randn_like(y) * 0.5))
        vals.append(Lval(x3, th)[0]); del x3
        torch.cuda.empty_cache()
        spread = max(vals) - min(vals)
        print(f"{th:>6} | " + " | ".join(f"{v:12.0f}" for v in vals)
              + f" | {spread:8.0f}", flush=True)

    print("\n[结论判据]", flush=True)
    print("  A: 若 L 随 n_inner 明显变化 -> 未收敛，profile 不可信", flush=True)
    print("  B: 若不同初值 L 差异大 -> 单初值找不到 min_x，必须多初值取最优", flush=True)


if __name__ == "__main__":
    main()
