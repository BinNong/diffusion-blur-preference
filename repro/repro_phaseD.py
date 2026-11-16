"""
Phase D v2 —— "MAP 已死，后验还活着吗？"（重要性采样版）

核心恒等式：
    p(theta|y) ∝ p(theta) * ∫ p_t(x) exp(-||h_theta*x - y||^2/(2 sigma_n^2)) dx
              = p(theta) * E_{x ~ p_t}[ exp(-data(x,theta)) ]

所以用【先验本身】做重要性采样的提议分布：
    1. 一次性生成 S 个 p_0 样本（EDM ODE 采样器，~5s/个）
    2. 目标若是 p_t，只需给每个样本加 t*xi —— 精确得到 p_t 样本
    3. 对每个 theta，Z(theta) = (1/S) * sum_s exp(-data(x_s, theta))
       p(theta|y) ∝ Z(theta)（theta 取均匀先验）

成本：生成样本 S*5s（一次性，可跨 theta / 跨噪声复用），
      每个 theta 只有 S 次 FFT —— 比精确 MALA（每步一次 3-7s ODE）便宜两个数量级。

为什么不用 v1 的 MCMC：
  v1 的 MALA 接受率需要 log p_t(x)，每次 3-7 秒，400 步就 40 分钟/噪声档。
  重要性采样完全绕开密度求值。

已知局限（必须报告）：
  * 重要性采样方差随噪声减小而增大（似然变尖）。用 ESS 诊断量化。
  * 目标是对【平滑先验 p_t】的后验，t 是可调参数；t=0 即原问题但方差最大。

用法： python repro_phaseD.py --t 0.1 --nsample 64
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
from edm.prior import ScorePrior, get_ode_flow, ode_sampler

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=0.1, help="先验平滑水平；0 = 不平滑")
    ap.add_argument("--nsample", type=int, default=64)
    ap.add_argument("--nimg", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-star", type=float, default=1.5)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    noises = [0.02, 0.06, 0.12, 0.25]
    thetas = [round(0.125 * i, 4) for i in range(21)]      # 0 .. 2.5
    T = args.t
    S = args.nsample

    print(f"[cfg] t={T} S={S} nimg={args.nimg} noises={noises} theta*={args.sigma_star}",
          flush=True)

    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    blur = BlurOp()

    # ---- 1. 一次性生成 p_0 样本 ----
    t0 = time.time()
    pool, _ = ode_sampler(ode_flow, shape=(S + args.nimg, 3, 64, 64), t_spans=t_spans,
                          device=DEV)
    pool = pool.to(DT)
    x_stars = pool[S:S + args.nimg]                    # 留几张当"真值"清晰图
    prior_pool = pool[:S]                              # 提议分布样本
    print(f"[pool] {S} 个先验样本 + {args.nimg} 张真值图，{time.time()-t0:.0f}s",
          flush=True)

    # 目标 p_t：给 p_0 样本加高斯噪声（精确）
    g = torch.Generator(device=DEV).manual_seed(12345)
    if T > 0:
        prop = prior_pool + T * torch.randn(prior_pool.shape, generator=g,
                                            device=DEV, dtype=DT)
    else:
        prop = prior_pool.clone()
    prop = prop.clamp(-1.0, 1.0)

    records = []
    for i in range(args.nimg):
        x_star = x_stars[i:i + 1].to(DT)
        y_clean = blur.A(x_star, args.sigma_star)
        for sn in noises:
            gg = torch.Generator(device=DEV).manual_seed(1000 * i + int(sn * 100000))
            y = (y_clean + sn * torch.randn(y_clean.shape, generator=gg, device=DEV,
                                            dtype=DT)).to(DT)
            # ---- 2. 对每个 theta 算重要性采样估计的 Z(theta) ----
            # 必须在对数空间做：低噪声时 data 巨大，exp(-data) 会整体下溢为 0
            Zs, ESSs, logZs = [], [], []
            for th in thetas:
                with torch.no_grad():
                    Aprop = blur.A(prop, th)
                    # log w_s = -data(x_s, theta)，逐样本
                    logw = -((Aprop - y) ** 2).flatten(1).sum(1) / (2 * sn ** 2)
                    # log Z = logsumexp(logw) - log S
                    lz = float(torch.logsumexp(logw, 0).item() - np.log(S))
                    logZs.append(lz)
                    Zs.append(float(np.exp(lz)))
                    # ESS（自归一化）：exp(2*lse(p) - lse(2p))，p 为归一化 log 权重
                    p = logw - torch.logsumexp(logw, 0)
                    lse2 = float(torch.logsumexp(2 * p, 0).item())
                    ESSs.append(float(np.exp(2 * float(torch.logsumexp(p, 0).item())
                                             - lse2)))
            Zs = np.array(Zs)
            ESSs = np.array(ESSs)
            # 后验（theta 均匀先验）：log p(theta|y) ∝ log Z(theta)
            logp = np.array(logZs)
            logp -= logp.max()
            post = np.exp(logp)
            post /= post.sum()
            cdf = np.cumsum(post)
            q = [float(thetas[np.searchsorted(cdf, p)]) for p in (0.05, 0.5, 0.95)]
            mode_th = thetas[int(np.argmax(post))]
            # 后验质量落在 theta* ± 0.3 的比例
            near = np.abs(np.array(thetas) - args.sigma_star) <= 0.3
            mass_near = float(post[near].sum())
            # 对照：theta=0 处的相对权重
            w0 = float(post[0] / post.max())

            rec = dict(img=i, noise=sn, t=T, S=S, theta_star=args.sigma_star,
                       thetas=thetas, Z=Zs.tolist(), post=post.tolist(),
                       ess=ESSs.tolist(), mode=mode_th, q05=q[0], q50=q[1],
                       q95=q[2], mass_near=mass_near, w0_rel=w0)
            records.append(rec)
            print(f"  img{i} sn={sn:<6} 后验: mode={mode_th:.3f} "
                  f"90%CI=[{q[0]:.2f},{q[2]:.2f}] | theta*±0.3 质量={mass_near:.2f} | "
                  f"ESS中位数={np.median(ESSs):.1f}/{S} | w(0)/w(max)={w0:.3f}",
                  flush=True)
            del y
            torch.cuda.empty_cache()

    path = os.path.join(OUT, f"phaseD_t{T}.json")
    with open(path, "w") as f:
        json.dump(dict(cfg=dict(t=T, S=S, nimg=args.nimg, noises=noises,
                                thetas=thetas, sigma_star=args.sigma_star,
                                seed=args.seed), records=records), f)
    print(f"\n[saved] {path}", flush=True)
    free, total = torch.cuda.mem_get_info()
    print(f"[gpu] free {free/2**30:.2f}/{total/2**30:.2f} GiB, "
          f"peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
