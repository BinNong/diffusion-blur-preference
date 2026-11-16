"""诊断：HQS 在 theta=0 是否系统性高估了 L(0)？

推理：Lemma 12 说 min_x L(x,0,y) 是全局最小。HQS 返回值是真极小的上界。
若 sweep 中实测 L(盆地) < 实测 L(0)，则 min_x L(x,0,y) < 实测 L(0)，
即 HQS 在 theta=0 高估了。

假设的机制：HQS 第一步就做 x_hat = D(y, sigma)，立刻把 x 推离 y。
但 theta=0 时数据项是 ||x-y||^2/(2 sigma^2)，取 x=y 时为 0，
而 q(y) 对模糊观测极低（Phase B 已验证模糊偏好）。
所以 L(y,0,y) = q(y) 很可能就是 theta=0 的全局极小，
而 HQS 从起点就走反了方向，再也回不来。

本脚本直接算 L(y,0,y) = q(y)，与 sweep 中实测的 L(0) 比较。

用法： python diag_theta0.py
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
TSTAR = 1.5


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
        B, C, H, W = x.shape
        pad = torch.zeros((1, C, H, W), device=DEV, dtype=DT)
        kh, kw = k.shape[-2:]
        t, l = (H - kh) // 2, (W - kw) // 2
        pad[:, :, t:t + kh, l:l + kw] = k
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))
                                * torch.fft.rfft2(pad, dim=(-2, -1)), s=(H, W), dim=(-2, -1))


def main():
    sweep_path = os.path.join(OUT, "phaseC_sweep.json")
    if not os.path.exists(sweep_path):
        print("缺少 phaseC_sweep.json"); return
    sweep = json.load(open(sweep_path))

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

    samples, _ = ode_sampler(ode_flow, shape=(2, 3, 64, 64), t_spans=t_spans, device=DEV)

    print("诊断：theta=0 处 HQS 是否高估 L(0)？\n", flush=True)
    print("L(y,0,y) = q(y) 是 theta=0 的一个【可行点】的值，"
          "故真极小 <= q(y)。\n", flush=True)
    print(f"{'img':>3} {'sigma':>7} | {'q(y)=L(y,0,y)':>14} | {'sweep L(0)':>12} | "
          f"{'差':>9} | 判定", flush=True)
    print("-" * 78, flush=True)

    for img in range(2):
        x_star = samples[img:img + 1].to(DT)
        y_clean = blur.A(x_star, TSTAR)
        for r in sweep["records"]:
            if r["img"] != img:
                continue
            sn = r["noise"]
            g = torch.Generator(device=DEV).manual_seed(1000 * img + int(sn * 100000))
            y = (y_clean + sn * torch.randn(y_clean.shape, generator=g, device=DEV,
                                            dtype=DT)).to(DT)
            qy = q_value(y)
            L0 = next(p["L"] for p in r["profile"] if p["theta"] == 0.0)
            diff = L0 - qy
            verdict = ("HQS 高估（真极小更低）" if diff > 1.0
                       else "HQS 已接近或低于 q(y)")
            print(f"{img:>3} {sn:>7} | {qy:>14.1f} | {L0:>12.1f} | {diff:>+9.1f} | {verdict}",
                  flush=True)
            del y
            torch.cuda.empty_cache()

    print("\n[结论]", flush=True)
    print("  若 '差' 为正 => sweep 的 L(0) 高于可行点 q(y) 的值，", flush=True)
    print("  说明 HQS 漏掉了更低的解，profile 的 theta=0 一侧被系统性高估，", flush=True)
    print("  盆地'获胜'是假象。修法：theta=0 必须把 x=y 作为一个初值。", flush=True)


if __name__ == "__main__":
    main()
