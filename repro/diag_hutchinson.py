"""关键检验：作者的 likelihood_fn 用【单个固定】Hutchinson 向量，
q(x) 的估计是否带依赖 x 的偏差？

edm/prior.py 的 likelihood_fn：
    epsilon = randint_like(data, 0, 2)*2-1     # 只采样一次
    def div_fn(x, t, noise): ...                # 整个 ODE 积分复用同一 epsilon
单样本 Hutchinson 是无偏的【期望】，但每次实现是一个有方差的具体值。
若这个偏差依赖 x，则跨 x（即跨 theta）比较 L 是无效的——
而 Lemma 12 的证明恰恰要求 L 在 (x,theta) 上可比较。

本脚本对同一批 x，分别用
  (a) 单个固定 epsilon（作者的做法）
  (b) K 个 epsilon 平均
计算 q(x)，比较两者的差与离散度。

用法： python diag_hutchinson.py
"""

import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, ode_sampler

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
K = 8


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    samples, _ = ode_sampler(ode_flow, shape=(3, 3, 64, 64), t_spans=t_spans, device=DEV)

    # 作者原版：epsilon 只采样一次，复用整个积分
    import edm.prior as P

    def q_single(x, eps):
        # 作者代码在 randint_like 之后还会做 .type(dtype)*2-1，
        # 所以 patch 必须返回 {0,1}，让后续变换产出 {-1,+1}
        orig_randint = torch.randint_like
        torch.randint_like = lambda t, low=0, high=2: ((eps + 1.0) / 2.0).to(t.dtype)
        try:
            with torch.no_grad():
                v = float(P.likelihood_fn(x.to(DT), ode_flow, t_spans=t_spans).item())
        finally:
            torch.randint_like = orig_randint
        return v

    def q_avg(x, k=K):
        vals = []
        for _ in range(k):
            eps = (torch.randint_like(x, 0, 2).float() * 2 - 1).to(DT)
            vals.append(q_single(x, eps))
        return float(np.mean(vals)), float(np.std(vals))

    print("Hutchinson 单向量偏差检验\n", flush=True)
    print("对每张图：报告 8 个独立 epsilon 下 q(x) 的均值与标准差，", flush=True)
    print("以及'单个实现'相对均值的最大偏离。\n", flush=True)
    print(f"{'img':>3} | {'mean q':>11} | {'std q':>8} | {'min':>11} | {'max':>11} | "
          f"{'max|dev|':>9}", flush=True)
    print("-" * 74, flush=True)

    all_dev = []
    for i in range(3):
        x = samples[i:i + 1].to(DT)
        m, s = q_avg(x)
        # 再取若干单个实现看偏离
        singles = []
        for _ in range(6):
            eps = (torch.randint_like(x, 0, 2).float() * 2 - 1).to(DT)
            singles.append(q_single(x, eps))
        dev = max(abs(v - m) for v in singles)
        all_dev.append(dev)
        print(f"{i:>3} | {m:>11.1f} | {s:>8.1f} | {min(singles):>11.1f} | "
              f"{max(singles):>11.1f} | {dev:>9.1f}", flush=True)
        del x
        torch.cuda.empty_cache()

    print(f"\n单个 epsilon 实现相对 K={K} 均值的最大偏离: "
          f"{max(all_dev):.1f}", flush=True)
    print("\n[判据]", flush=True)
    print("  若该偏离达数十甚至上千 -> 作者的单向量 q 估计不可用于跨 x 比较，", flush=True)
    print("  Phase C 的 profile（以及任何基于它的结论）必须改用多样本平均。", flush=True)
    print("  若偏离只有个位数 -> 不是偏差问题，需另找 L(盆地)<L(0) 的原因。", flush=True)


if __name__ == "__main__":
    main()
