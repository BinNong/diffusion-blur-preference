"""判决性检验：Lemma 12 的预测 vs 精确量。

Lemma 12（Morse-Bott 论文）：在模糊偏好条件 q(h*x) <= q(x) 下，
    min_x L(x, 0, y)  <=  L(x, theta, y)   对所有 (x, theta) 成立
特别地，取 (x,theta) = (x*, theta*)（真值）：
    min_x L(x, 0, y)  <=  L(x*, theta*, y)
右边可以【精确】计算，完全不依赖任何优化器：
    L(x*, theta*, y) = ||h_{theta*}*x* - y||^2 / (2 sigma^2) + q(x*)
                     = ||eta||^2/(2 sigma^2) + q(x*)
其中 eta 是已知的加性噪声，q(x*) 由作者的 likelihood_fn 给出。

若测得的 min_x L(x,0,y) > 这个精确值，则说明：
  (a) 我的 HQS 在 theta=0 处没有找到真极小（优化问题），或
  (b) Lemma 12 的假设在我的设定不成立（科学问题）。
这两者必须分开——(a) 是我的 bug，(b) 是可发表的发现。

用法： python test_lemma12.py
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
OUT = os.path.join(_PROJ, "results")


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
    TSTAR = 1.5
    N_INNER = 240
    NOISES = [0.01, 0.02, 0.06, 0.17, 0.5]

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    blur = BlurOp()

    @torch.no_grad()
    def q_value(x):
        return float(likelihood_fn(x.to(DT), ode_flow, t_spans=t_spans).item())

    @torch.no_grad()
    def hqs(theta, y, sn, z0, n_inner=N_INNER):
        B, C, H, W = y.shape
        ATy = blur.A(y, theta)  # A = AT for symmetric kernel；下面显式重算更稳
        k = blur.kernel(theta)
        if k is None:
            ATy = y
        else:
            K = blur._K(k, C, H, W)
            ATy = torch.fft.irfft2(torch.conj(K) * torch.fft.rfft2(y, dim=(-2, -1)),
                                   s=(H, W), dim=(-2, -1))
        denom = blur.eig(theta, C, H, W) + 1.0
        z = z0.clone()
        for _ in range(n_inner):
            xh = denoiser(z, sn).to(DT)
            v = 2 * xh - z
            xn = torch.fft.irfft2(torch.fft.rfft2(ATy + v, dim=(-2, -1)) / denom,
                                  s=(H, W), dim=(-2, -1))
            z = torch.clamp(z + xn - xh, IMG_LO, IMG_HI)
        return denoiser(z, sn).to(DT)

    samples, _ = ode_sampler(ode_flow, shape=(1, 3, 64, 64), t_spans=t_spans, device=DEV)
    x_star = samples[0:1].to(DT)
    y_clean = blur.A(x_star, TSTAR)

    print("Lemma 12 判决性检验", flush=True)
    print("Lemma 12 预测:  min_x L(x,0,y)  <=  L(x*,theta*,y)  (右边可精确计算)\n", flush=True)
    print(f"{'sigma':>7} | {'L(x*,t*) 精确':>15} | {'min_x L(x,0,y) 测得':>21} | "
          f"{'差':>9} | 判定", flush=True)
    print("-" * 80, flush=True)

    for sn in NOISES:
        g = torch.Generator(device=DEV).manual_seed(777)
        eta = sn * torch.randn(y_clean.shape, generator=g, device=DEV, dtype=DT)
        y = (y_clean + eta).to(DT)

        # --- 精确量：L(x*, theta*, y) = ||eta||^2/(2 sigma^2) + q(x*) ---
        with torch.no_grad():
            data_star = float((eta ** 2).sum().item() / (2 * sn ** 2))
        q_star = q_value(x_star)
        L_star = data_star + q_star

        # --- 测量：theta=0 处 HQS 多初值取最优 ---
        best = None
        for z0 in (y, blur.A(y, 0.0), torch.randn_like(y) * 0.5):
            x = hqs(0.0, y, sn, z0)
            with torch.no_grad():
                d = float(((x - y) ** 2).sum().item() / (2 * sn ** 2))
            q = q_value(x)
            if best is None or d + q < best:
                best = d + q
            del x
            torch.cuda.empty_cache()

        verdict = ("Lemma 12 成立" if best <= L_star + 1e-6
                   else "违反 Lemma 12 <-- 需查 HQS 是否找到真极小")
        print(f"{sn:>7} | {L_star:>15.1f} | {best:>21.1f} | {best-L_star:>+9.1f} | {verdict}",
              flush=True)

    print("\n[说明]", flush=True)
    print("  若 '违反' 且差距大：优先怀疑 HQS 在 theta=0 未收敛到真极小。", flush=True)
    print("  补救：theta=0 时 L 是纯去噪问题，可用更强的前台算法（如更多内循环、", flush=True)
    print("        或直接在 x* 邻域初始化）确认能否降到 L_star 以下。", flush=True)


if __name__ == "__main__":
    main()
