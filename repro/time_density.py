"""测量先验密度 log p_t(x) 的计算成本，决定 Phase D 的预算。

likelihood_fn 的 t_spans 控制积分终点：传 [80, t] 得到 -log p_t(x)。
作者默认 t_spans=[80, 0.002]（即 t=0.002）。容差 atol=rtol 决定耗时与精度。

用法： python time_density.py
"""

import os
import sys
import time

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn, ode_sampler

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    samples, _ = ode_sampler(ode_flow, shape=(2, 3, 64, 64), t_spans=t_spans, device=DEV)
    x = samples[0:1].to(DT)

    print("先验密度计算成本（64x64, RTX 4060 争用中）\n", flush=True)
    print(f"{'t':>7} | {'atol=rtol':>10} | {'耗时(s)':>8} | {'-log p_t':>12}", flush=True)
    print("-" * 50, flush=True)
    for t in (0.002, 0.2, 1.0):
        for tol in (1e-5, 1e-4, 1e-3):
            ts = [80.0, t]
            t0 = time.time()
            with torch.no_grad():
                v = float(likelihood_fn(x, ode_flow, t_spans=ts,
                                        atol=tol, rtol=tol).item())
            dt = time.time() - t0
            print(f"{t:>7} | {tol:>10.0e} | {dt:>8.2f} | {v:>12.1f}", flush=True)

    print("\n[用途] Phase D 的 MALA 接受率需要 log p_t(x)。", flush=True)
    print("  选 (t, tol) 的标准：t 要小到先验仍有区分力，tol 松到能负担", flush=True)
    print("  N 步 x 2 次求值/步 的成本。", flush=True)


if __name__ == "__main__":
    main()
