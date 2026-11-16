"""先验重要性采样的 ESS 由什么定标？——统一图景的第三个算法（CPU，零 GPU）。

预官（2026-10-22，跑前写定）
================================================================================
背景
----
§五点五 的统一图景说「任何算法的失败幅度都由 Λ 定标」。现已成立的两半：
  * 机理半（§3.9(y)+(ad)）：二阶参照捕获 Λ 的份额随 t 从 15% 涨到反超 2 倍
  * 幅度半（§3.9(z)+(ab)+(ac)）：全局 MAP 退化 = 三项恒等式，跨先验无关
第三块——采样器——原表只有「先验 IS 的 ESS = 1/64」，而且**不在同一量纲**，一直没检验。

推导（写预官前已想清楚）
----------------------
先验 IS：提议 q = p_t，权重 w(x) = exp(-data(x))，data(x)=‖g₂^{1/2}x̂-ŷ‖²/(2σ_n²)。
ESS = (Σw)²/Σw² ≈ n·E[w]²/E[w²]。对近对数正态的 w（E[e^{-β·data}]=e^{-βμ+β²s²/2}）：
    E[w] = e^{-μ+s²/2},  E[w²] = e^{-2μ+2s²}  ⟹  E[w²]/E[w]² = e^{s²}
    **log(n/ESS) ≈ Var(data)**（不是 2 倍——我第一版推成 2 倍，是 log-normal MGF 用错）
而 Var(data) = Σ_f g₂(f)²·V_t(f)²/(2σ_n⁴) 是**二阶（功率谱）量**；
Λ = log p_t(y) − log p_t(x*) 度量的是先验的**非高斯**结构。
所以采样器的失效可能和 MAP 的退化**不同源**——这正是本实验要判的。

设计
----
池：/tmp/pool_ffhq64_256_s0.pt（256 张，t=0.002，ffhq）。
算子：W2/W8/TK50 × 17 个 kunif 目标（u=1-kunif ∈ [0.14,0.85]），与第八/十组逐字对齐。
σ_n 固定，逐算子变 ⟹ Λ 变（859–21856，25 倍）。
x* = pool[0]，y = g₂^{1/2}x̂* + η̂（循环卷积，与 sweep 一致）。
对每个算子：算 256 个池图的 data，得 ESS。

预官判据（在看到任何 ESS 之前写定）
----------------------------------
Y1 守门（恒等式）：log(n/ESS) 与 **Var(data)** 的比值逐算子报告。近对数正态 + CLT 下应 ≈1；
   不要求通过，只要求报出——若系统性偏离，说明权重分布远非对数正态，ESS 的读法要改。

Y2 主判据（**可能失败，是本实验的核心**）：把 log(n/ESS) 只对 Λ 回归（51 个算子、
   固定 σ_n），报 R²。**预言 R²(Λ) ≤ 0.90**——若更高，说明采样器失效确实由 Λ 定标，
   统一图景扩到全部三个算法（这是个好消息，但不是我的预期）。

Y3 判别式（**最尖的一条**）：同数据上再对二阶预测量 Var(data) 回归，比 R²。
   **预言 R²(Var data) − R²(Λ) ≥ 0.05**，即二阶量至少不差于 Λ。
   **失败方式**：若 R²(Λ) 明显更高，则采样器的失效**是** Λ 驱动的，统一图景成立；
   若 R²(Var data) 更高，则采样器为二阶原因失效、MAP 为非高斯原因退化——**两者不同源**，
   统一图景的「一个数字」表述要改成「两类数字」。

Y4 诊断（无阈值）：log(n/ESS) 随 σ_n 的走向（固定算子）；ESS 的分布（有多少算子
   ESS<1，即权重塌到单样本）。

本实验**不**证明什么（诚实声明）
------------------------------
只在 t=0.002、ffhq、高斯族算子上做。ESS 是**先验 IS** 的，不是 MCMC/Laplace 的。

用法： venv2/bin/python ess_scale.py
"""

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_path = os.path.join(_PROJ, "code", "blind_deblur_diffusion")
import sys
sys.path.insert(0, sys_path)

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, ode_sampler

POOL = "/tmp/pool_ffhq64_256_s0.pt"
OUT = os.path.join(_PROJ, "results")
DENSE = ["W2", "W8", "TK50"]
T0 = 0.002
NOISES = [0.02, 0.05, 0.10]
TARGETS = [0.15, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.52,
           0.58, 0.65, 0.70, 0.74, 0.78, 0.82, 0.86]
H = W = 64
DEV = torch.device("cuda")


def halfplane():
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)[:, :W // 2 + 1]
    return R / R.max()


def make_profile(kind, par, sc, r):
    if kind == "weibull":
        return np.exp(-(r / sc) ** par)
    if kind == "tukey":
        r1 = sc * (1.0 - par)
        g = np.zeros_like(r)
        g[r <= r1] = 1.0
        m = (r > r1) & (r < sc)
        t = (r[m] - r1) / max(sc - r1, 1e-12)
        g[m] = 0.5 * (1.0 + np.cos(np.pi * t))
        return g
    raise ValueError(kind)


def kunif_of(kind, par, sc, r):
    return float(1.0 - make_profile(kind, par, sc, r).mean())


def tune_rc(kind, par, r, target, lo=1e-3, hi=3.0, it=60):
    """二分 sc 使 kunif 命中 target。

    **方向**：weibull / tukey 的 sc 是尺度（或归零半径），sc 越大衰减越弱，
    所以 kunif 随 sc **下降**。于是 kunif(mid) < target（衰减太弱）时要调**小** sc。
    第一版把 lo/hi 写反，三个 target 全部收敛到同一个 g2（mean=0.964），
    整批算子完全相同、测量退化——这是本项目「扫参数前先确认名义含义」那条坑。
    """
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if kunif_of(kind, par, mid, r) < target:
            hi = mid          # 衰减太弱 => 要更小 sc
        else:
            lo = mid
    sc = 0.5 * (lo + hi)
    assert abs(kunif_of(kind, par, sc, r) - target) < 5e-3, "tune 未收敛"
    return sc


def r2(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    c = np.corrcoef(x, y)[0, 1]
    return float(c * c)


def main():
    r = halfplane()
    # 重建算子：与 sweep_law_test.py 的 NEW_FAMILIES / tune 逻辑一致
    fams = {"W2": ("weibull", 2.0), "W8": ("weibull", 8.0), "TK50": ("tukey", 0.5)}
    ops = []
    for fid, (kind, par) in fams.items():
        sc = tune_rc(kind, par, r, 0.92)
        for tg in TARGETS:
            # 参考强度 0.92 的剖面乘上衰减量：g2 = ref2*(1-beta) + beta*target_profile
            # 简化：直接调 sc 命中 target（与 sweep 的 A 组等价——那里 ref 也调）
            sc2 = tune_rc(kind, par, r, tg)
            g2 = make_profile(kind, par, sc2, r)
            ops.append(dict(id=f"{fid}@{tg}", fid=fid, g2=g2, kunif=tg))

    # 载入池
    pool = torch.load(POOL, map_location="cpu").to(torch.float64)   # (256,3,64,64)
    n_pool = pool.shape[0]
    XH = torch.fft.rfft2(pool, dim=(-2, -1))                        # (n,3,H,W//2+1)
    print(f"池 {n_pool} 张；算子 {len(ops)} 个；σ_n ∈ {NOISES}", flush=True)

    # Λ(t0, u)：从已有 sweep 文件读，避免重算 ODE
    lam = {}
    for fn in ("sweep_bdense.json", "sweep_blow.json"):
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            continue
        for rec in json.load(open(p))["rows"]:
            if rec["meta"]["fid"] in DENSE:
                lam[(rec["meta"]["fid"], round(1 - rec["kunif"], 4))] = rec["lam_mean"]
    got = sum(1 for o in ops if (o["fid"], round(1 - o["kunif"], 4)) in lam)
    print(f"Λ(t0) 命中 {got}/{len(ops)} 个算子", flush=True)

    rows = []
    for sn in NOISES:
        g = torch.Generator().manual_seed(4242)
        eta = sn * torch.randn(pool[0:1].shape, generator=g).to(torch.float64)
        xstar = pool[0:1]
        for o in ops:
            G = torch.from_numpy(o["g2"]).to(torch.float64)[None, None]
            Gs = G.sqrt()
            yh = Gs * torch.fft.rfft2(xstar, dim=(-2, -1)) + torch.fft.rfft2(eta, dim=(-2, -1))
            # data(x_i) = ‖Gs*x̂_i − ŷ‖²/(2σ²)，对全部池图
            D = (Gs * XH - yh).abs() ** 2
            # Parseval：rfft2 域求和 = N·像素域求和，必须除 H*W 才是 data 的
            # 常规定义（与 test_lemma12 / evidence_scale 的像素域算法逐字对齐）。
            data = D.sum(dim=(1, 2, 3)) / (H * W) / (2 * sn ** 2)   # (n,)
            # log-sum-exp 版 ESS：直接 exp 会全部下溢（data 跨 1e6），NaN 不是物理。
            # log(n/ESS) = log n + 2·LSE(log w) - LSE(2 log w)，数值稳定。
            lw = -(data - data.min())                    # max(lw)=0
            m2 = float(torch.logsumexp(2 * lw, 0))
            m1 = float(torch.logsumexp(lw, 0))
            log_nrat = math.log(n_pool) + 2 * m1 - m2
            vd = float(data_var(data.numpy()))
            key = (o["fid"], round(1 - o["kunif"], 4))
            rows.append(dict(sn=sn, id=o["id"], fid=o["fid"], kunif=o["kunif"],
                             ess=float(math.exp(-log_nrat)) * n_pool,
                             log_nrat=log_nrat, var_data=vd, lam=lam.get(key)))
            del D, yh
        print(f"  σ_n={sn:g} 完成 {len(rows)} 行累计", flush=True)

    with open(os.path.join(OUT, "ess_scale.json"), "w") as fh:
        json.dump(dict(cfg=dict(pool=POOL, n_pool=n_pool, noises=NOISES,
                                targets=TARGETS, t=T0, eta_seed=4242), rows=rows),
                  fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'ess_scale.json')}")


def data_var(data):
    """data 的样本方差（对池图）。"""
    a = np.asarray(data, dtype=float)
    return float(((a - a.mean()) ** 2).sum() / (len(a) - 1))


if __name__ == "__main__":
    main()
