"""在解析高斯先验上验证 score 恒等式与整条 log p 管线。

目的：probe_score_integral 发现"沿直线的 score 线积分"复现不出精确 Lambda。
要区分两种情况：
  (a) 我的用法/符号有错
  (b) EDM 模型去噪器不准，1/t^2 把误差放大

做法：用一个解析的、傅里叶可对角化的高斯先验
      p = N(0, Sigma),  Sigma_f = S_f  （逐频率独立）
它的精确去噪器、精确 score、精确 log p 全有闭式：
      D(x,t)_f   = S_f/(S_f + t^2) * X_f
      grad log p_t(x)_f = -(X_f)/(S_f + t^2)
      log p_t(x) = -1/2 sum_f [ |X_f|^2/(S_f+t^2) + log(2 pi (S_f+t^2)) ]
把这个 D 插进作者的 likelihood_fn / get_ode_flow，检查：
  1. likelihood_fn 返回的 log p 是否等于闭式
  2. 沿直线的 score 线积分是否等于闭式的 log p 差
  3. 单点二次模型是否等于闭式的 Taylor 展开
若 2 成立 => 管线无误，真实模型上的失败是模型误差（情形 b）。

用法： python validate_score_identity_gaussian.py
"""

import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm.prior import ScorePrior, get_ode_flow, likelihood_fn

H = W = 16
N = H * W
SIGMA_DATA = 0.5


def make_S(seed=0):
    """各向异性功率谱 S_f（逐频率），保证协方差正定。"""
    rng = np.random.default_rng(seed)
    fx = np.fft.fftfreq(H)
    fy = np.fft.fftfreq(W)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)
    S = SIGMA_DATA ** 2 * np.exp(-6.0 * R ** 2) + 0.02
    S = S * (1.0 + 0.3 * rng.standard_normal(S.shape))
    return np.maximum(S, 1e-3)                            # (H, W)


def main():
    S_np = make_S()
    S = torch.from_numpy(S_np).to(torch.float64)          # (H, W)
    Sf = S[None, None]                                    # (1,1,H,W) 广播到通道

    def denoiser(x, sigma, *a, **k):
        """精确高斯去噪器（VE: p_t = p (*) N(0,t^2)）。x: (B,C,H,W) float32"""
        t2 = float(sigma if not torch.is_tensor(sigma) else sigma.reshape(-1)[0]) ** 2
        X = torch.fft.fft2(x.to(torch.float64), dim=(-2, -1))
        D = torch.fft.ifft2(X * (Sf / (Sf + t2)), dim=(-2, -1))
        return D.real.to(torch.float32)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, _ = get_ode_flow(prior)

    rng = np.random.default_rng(1)
    torch.manual_seed(0)

    def sample_x(n):
        """从 N(0, S) 采 n 个样本（傅里叶域生成，逐通道独立）。"""
        z = torch.from_numpy(rng.standard_normal((n, 1, H, W))).to(torch.float64)
        Z = torch.fft.fft2(z, dim=(-2, -1))
        X = Z * Sf.sqrt()
        return torch.fft.ifft2(X, dim=(-2, -1)).real.repeat(1, 3, 1, 1).to(torch.float32)

    def analytic_logp(x, t):
        """闭式 log p_t。注意 torch.fft 未归一化：X_torch = sqrt(H*W) * X_unitary，
        所以二次项要除 N=H*W，对数行列式项不除。"""
        X = torch.fft.fft2(x.to(torch.float64), dim=(-2, -1))
        v = Sf + t ** 2
        N = H * W
        quad = (X.abs() ** 2 / v).sum(dim=(1, 2, 3)) / N
        logdet = torch.log(2 * np.pi * v).sum(dim=(1, 2, 3)) * 3
        return (-0.5 * (quad + logdet)).numpy()

    n = 4
    xs = sample_x(n)
    print(f"样本 {tuple(xs.shape)}，t 扫描")
    print(f"{'t':>8} | {'likelihood_fn':>14} | {'闭式':>14} | {'相对差':>10}")
    print("-" * 56)
    for t in [0.002, 0.01, 0.05, 0.2, 1.0]:
        nll = likelihood_fn(xs, ode_flow, t_spans=[80.0, t])
        lp_ode = -nll.numpy()
        lp_ex = analytic_logp(xs, t)
        rel = np.abs(lp_ode - lp_ex) / np.abs(lp_ex)
        print(f"{t:>8g} | {lp_ode.mean():>14.2f} | {lp_ex.mean():>14.2f} | "
              f"{rel.mean():>10.2e}", flush=True)

    # ---- 沿直线的 score 线积分 ----
    print("\n沿直线的 score 线积分 vs 闭式 log p 差（高斯先验）")
    t = 0.01
    # 位移方向：随机
    v = torch.from_numpy(rng.standard_normal((1, 3, H, W))).to(torch.float32)
    v = v / v.flatten().norm()
    print(f"{'eps':>8} | {'线积分':>14} | {'闭式差':>14} | {'相对差':>10}")
    print("-" * 56)
    for eps in [0.05, 0.1, 0.2, 0.5, 1.0]:
        # gamma(tau) = x + tau*eps*v,  gamma' = eps*v
        nq = 21
        taus = np.linspace(0, 1, nq)
        w = np.ones(nq)
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        w = w / 3.0 / (nq - 1)
        acc = np.zeros(n)
        with torch.no_grad():
            for tau, wt in zip(taus, w):
                z = xs + tau * eps * v
                g = (denoiser(z, t) - z) / (t * t)
                acc += wt * (g.flatten(1) * v.flatten(1)).sum(1).numpy()
        integ = eps * acc
        lp0 = analytic_logp(xs, t)
        lp1 = analytic_logp(xs + eps * v, t)
        ex = lp1 - lp0
        rel = np.abs(integ - ex) / np.abs(ex)
        print(f"{eps:>8g} | {integ.mean():>14.4f} | {ex.mean():>14.4f} | "
              f"{rel.mean():>10.2e}", flush=True)

    print("\n[判据]")
    print("  * 若线积分与闭式差一致（相对差 ~1e-6）=> 恒等式与用法都对，")
    print("    真实模型上的失败只能归因于去噪器误差被 1/t^2 放大")
    print("  * 若不一致 => 我的符号/用法有错，先修再解释真实模型")


if __name__ == "__main__":
    main()
