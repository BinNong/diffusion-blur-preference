"""标定检验：likelihood_fn 在解析已知的高斯先验上是否正确？

构造一个"精确去噪器"实现高斯先验 p_0 = N(0, s^2 I) 的 MMSE 去噪：
    D(x, sigma) = s^2/(s^2+sigma^2) * x
验证 Tweedie：grad log p_sigma(x) = (D-x)/sigma^2 = -x/(s^2+sigma^2)  ✓

该先验下概率流 ODE 有解析解（作者 edm/prior.py 的 edm 离散化，s(t)=1, sigma(t)=t）：
    dx/dt = -t * score(x,t) = t*x/(s^2+t^2)
    => x(t) 是纯缩放：x(t1) = x(t0) * sqrt((s^2+t1^2)/(s^2+t0^2))
    => div v = N * t/(s^2+t^2)
    => ∫div v dt = (N/2) log((s^2+t_max^2)/(s^2+t_min^2))

而 likelihood_fn 的 t_spans=[80, t] 积分从 t 到 80，故它算的是 -log p_t(x)：
    -log p_t(x) = ||x||^2/(2(s^2+t^2)) + (N/2) log(2 pi (s^2+t^2))

把三者对照，即可判断：
  (1) ODE/散度积分实现是否正确
  (2) likelihood_fn 返回的是否为正规化对数密度
  (3) 我之前对 EDM 数值量级的怀疑是否成立

用法： python calibrate_likelihood.py
"""

import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm.prior import ScorePrior, get_ode_flow, likelihood_fn

DEV = torch.device("cuda")
DT = torch.float64        # 用双精度，排除数值误差干扰


class GaussianDenoiser:
    """N(0, s^2 I) 先验的精确 MMSE 去噪器。"""

    def __init__(self, s):
        self.s2 = s ** 2

    def __call__(self, x, sigma, *a, **k):
        return (self.s2 / (self.s2 + sigma ** 2)) * x


def main():
    torch.manual_seed(0)
    N = 3 * 64 * 64
    s = 0.5
    t_min = 0.002
    t_max = 80.0

    print("=" * 88)
    print("标定检验：likelihood_fn vs 高斯先验解析解")
    print("=" * 88)
    print(f"维度 N={N}, 先验 N(0, {s}^2 I), t_spans=[{t_max}, {t_min}]")
    print(f"likelihood_fn 积分从 t={t_min} 到 {t_max}，故应返回 -log p_{{t_min}}(x)\n")

    den = GaussianDenoiser(s)
    prior = ScorePrior(denoiser=den)
    ode_flow, t_spans = get_ode_flow(prior)
    print(f"get_ode_flow 给出的 t_spans = {t_spans}\n")

    # ---- 1. Tweedie 恒等式检查 ----
    x = torch.randn(1, 3, 64, 64, device=DEV, dtype=DT)
    sg = 1.7
    with torch.no_grad():
        tweedie = prior.score(x, sg)                      # (D-x)/sigma^2
        analytic = -x / (s ** 2 + sg ** 2)
    err = float((tweedie - analytic).abs().max())
    print(f"[1] Tweedie 恒等式   max|score - 解析| = {err:.3e}   "
          f"{'OK' if err < 1e-9 else 'FAIL'}")

    # ---- 2. ODE 漂移项检查 ----
    with torch.no_grad():
        v = ode_flow(x, sg)
        v_analytic = sg * x / (s ** 2 + sg ** 2)
    err = float((v - v_analytic).abs().max())
    print(f"[2] ODE 漂移项      max|v - 解析|    = {err:.3e}   "
          f"{'OK' if err < 1e-9 else 'FAIL'}")

    # ---- 3. 主检验：-log p_t 的解析值 vs likelihood_fn ----
    # 取几个不同范数的 x（典型图 / 小范数 / 大范数）
    cases = {
        "typical (||x||^2=N*s^2)": torch.randn(1, 3, 64, 64, device=DEV, dtype=DT) * s,
        "small  (||x||^2=N*s^2/4)": torch.randn(1, 3, 64, 64, device=DEV,
                                                dtype=DT) * (s / 2),
        "large  (||x||^2=N*s^2*4)": torch.randn(1, 3, 64, 64, device=DEV, dtype=DT) * (2 * s),
    }
    print(f"\n[3] 对数密度主检验（解析 vs likelihood_fn）")
    print(f"{'case':>26} | {'||x||^2':>10} | {'解析 -log p_t':>15} | "
          f"{'likelihood_fn':>15} | {'差':>12}")
    print("-" * 90)
    for name, xx in cases.items():
        with torch.no_grad():
            n2 = float((xx ** 2).sum())
            analytic_nll = n2 / (2 * (s ** 2 + t_min ** 2)) \
                + (N / 2) * np.log(2 * np.pi * (s ** 2 + t_min ** 2))
            got = float(likelihood_fn(xx, ode_flow, t_spans=[t_max, t_min]).item())
        print(f"{name:>26} | {n2:>10.1f} | {analytic_nll:>15.2f} | {got:>15.2f} | "
              f"{got - analytic_nll:>+12.2f}")

    # ---- 4. 终端高斯项检查（sigma_max=80 的归一化）----
    print(f"\n[4] 终端分布归一化项：likelihood_fn 用 sigma_max=t_spans[0]={t_spans[0]}")
    print(f"    解析终端对数密度（在缩放后的 x_T 处）应为 N(0, s^2+{t_max}^2) 的对数密度")

    print("\n[判据]")
    print("  [3] 的差若接近 0 => likelihood_fn 实现正确，返回正规化对数密度；")
    print("      那么 EDM 模型上测得的 -log p ≈ -33570 是真实的（先验集中在低维流形附近）")
    print("  [3] 的差若很大 => 实现或用法有误，Phase B/C/D 的绝对数值全部不可信")


if __name__ == "__main__":
    main()
