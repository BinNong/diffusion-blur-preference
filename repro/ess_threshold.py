"""先验重要性采样的塌缩点在哪里？——(ae) 的延伸：ESS 在哪个 sigma_n 上离开 1（CPU，零 GPU）

预官（2026-10-22，跑前写定）
================================================================================
背景（RESULTS §3.9(ae)）
------------------------
(ae) 在 51 算子 × sigma_n in {0.02, 0.05, 0.10} 上测到 ESS **精确为 1.0000**，
权重散度 Var(data) 与 Lambda 无恒定比例（之比散 15.2 倍），Var(data) 严格
propto sigma_n^{-4}。结论是：先验 IS 为**纯二阶/集中度**原因失效。

但那个实验留下一个审稿人一定会问的洞：**ESS=1 是不是因为 sigma_n 取得太小？**
sigma_n -> inf 时似然变平、后验 -> 先验，ESS 必然 -> n = 256。所以一定存在一个
crossover，而 (ae) 只覆盖了它左侧的三个点。本实验把 sigma_n 扫到 300（15 倍跨度），
把 crossover 找出来，并把「ESS=1 的范围」变成一个可引用的定量陈述。

结构（两条恒等式，不是模型）
--------------------------
(i) 精确恒等式（与 sigma、与分布无关）：
      ESS = (sum w)^2 / sum w^2 = n / (1 + CV^2(w)),   CV^2 = Var(w)/E[w]^2
    证明：sum w = n*wbar, sum w^2 = n*<w^2>，故 ESS = n*wbar^2/<w^2>。
    这条用作**裁判自身的数值守门**（Z1），不是物理判据。

(ii) 对数正态近似（(ae) 已推过，当时失败）：w = exp(-data) 近对数正态时
      CV^2 = exp(Var(data)) - 1   =>   ESS = n * exp(-Var(data))
    而 Var(data) = sum_f g2(f)^2 V_t(f)^2 / (2 sigma_n^4) 是**纯二阶谱量**。
    于是「先验 IS 可用」等价于一条**闭式的二阶谱不等式**
          sum_f g2^2 V_t^2 / (2 sigma^4)  <~  ln n
    这是本实验想要的公式级产出：把采样器的失效变成一个可计算的条件。

**写判据前已经看到的东西（披露，防事后合理化）**：
  * results/ess_scale.json 全部 153 行（Var(data)、ESS、Lambda），即 (ae) 的原始数据。
  * 由它算出的 C = Var(data)*sigma^4 跨算子极差只有 1.053（min 9.634e5, max 1.015e6），
    且逐算子在三个 sigma 上恒定到 0.4%。
  * 由 C 与 ln(256)=5.545 算出的 (C/ln n)^{1/4} ~ 20.7；(C)^{1/4} ~ 31.7。
  * §3.9(z) 的 Lambda_clean(sigma_n) 表（sigma=0.2 处 317/69/-137 nats）。
**没有看到过的东西**：sigma_n > 0.1 的任何 ESS、crossover 的位置、ESS( sigma ) 曲线形状、
Var(data) 的 sigma 标度是否在高 sigma 处从 sigma^-4 转成 sigma^-2。

本实验**不**改 (ae) 的任何阈值，也不重测 (ae) 已有的三个 sigma 的物理结论——
sigma=0.02/0.05/0.10 三点重测只是为了做跨轮次复现。

预官判据
--------
Z1 守门（精确恒等式，与 sigma 无关）：逐行 ESS 两种算法一致到 1e-10：
   (a) 数值稳定版 (sum W)^2/sum W^2，W_i = exp(-(data_i - min data)) in (0,1]
   (b) n / (1 + CV^2(W))，CV^2 直接在 W 上算（W 有界，不会溢出）
   不过则后面全部作废。

Z2 主判据（定量预言，**零自由参数**）：在过渡区 ESS in [2, n/2] 内，
   实测 ESS 与 n*exp(-Var(data)) 之比落在 [0.5, 2] 内。
   **同时预言该近似在 Var(data) >= 100 处失效**：(ae) 已实测 Var(data) ~ 1e12 而
   ESS 钉在 1（近似给 exp(-1e12)=0）。所以正确读法是：对数正态近似只在它自己的
   适用域（Var(data) 不超过 ~10）内成立。这条既是预言，也把 (ae) 的失败方式定量化。
   **失败方式**：若过渡区比值落在 [0.5,2] 之外，说明权重的对数正态假设连在
   Var(data)=O(1) 附近都不成立，ESS 的读法要整个换掉。

Z3 最尖（论文级陈述）：ESS 首次 > 1 的 sigma_n 下界 sigma_hi 满足 sigma_hi/0.2 >= 50。
   0.2 是「问题尚有意义的最大噪声」：sigma_n=0.2 处 Lambda_clean = 317/69/-137 nats
   (§3.9(z)），模糊已近乎不可辨。由 C 与 ln n 预言的 crossover 是 (C/ln 256)^{1/4} ~ 20.7，
   即 sigma_hi/0.2 ~ 103。**这条预言 103 倍，阈值只要 50 倍，留了一倍余量。**
   **失败方式**：若 sigma_hi/0.2 < 50，则「ESS=1 只是因为噪声取小了」这个辩解成立，
   (ae) 的结论必须降级为「在低噪声端成立」。

Z4 诊断（无阈值，但其中 (b) 有可判决的结构性预言）：
   (a) ESS(sigma_n) 完整曲线，报 sigma_n -> 300 处的 ESS（预言 -> n = 256）；
   (b) Var(data) 的 sigma 标度在某个 sigma 处从 sigma^-4 转成 sigma^-2。**这是可从
       结构推出来的、不是记录**：data_i = A_i + B_i，其中
         A_i = ||g2^{1/2}(x_i - x*)||^2/(2 sigma^2)   跨池图的**确定性**项，逐点 ~ sigma^-2
                                                       => Var_i(A) ~ sigma^-4
         B_i = ||eps||^2/2 - <g2^{1/2}(x_i-x*), eps>/sigma  同一个 eps 的**交叉**项，
                                               跨 i 的部分 ~ sigma^-1 => Var_i(B) ~ sigma^-2
       于是预言：var_det * sigma^4 与 var_x * sigma^2 **各自**与 sigma 无关（两条标度律），
       且两者在 sigma* = sqrt(var_det*sigma^2/var_x) 处相交、var_det(sigma*)=var_x(sigma*)。
       **三个可判决的推论**：(i) 两条标度律各自恒定；(ii) 交点处两者相等；
       (iii) 由 sigma^-4 外推的 sigma_hi = (C/ln n)^{1/4} 是**下界**（大 sigma 处
       Var(data) 掉得慢，所以真实 sigma_hi >= 外推值）。逐行另存 var_det、var_x 供此用。
   (c) sigma_hi 与 Var(data)(sigma_hi) 的对照：预言 Var(data)(sigma_hi) ~ ln n = 5.545。

诚实声明
--------
* 只测 t=0.002 的 ffhq 先验、W2/W8/TK50 三族、高斯型零相位算子、256 张池。
* ESS 是**先验 IS** 的（提议 q = p_t 就是先验本身），不是 MCMC/Laplace/最优点近似的。
  它对「用一个固定先验样本池去表示后验」这件事失效，不直接否证其它采样器。
* 噪声实现固定：eta = sigma_n * eps，eps 是用 seed 4242 抽的一次单位高斯，
  对所有 sigma_n 复用（与 ess_scale.py 的实现一致：那里每个 sigma 都重新 seed 4242）。
  于是 sigma_n 扫描是一个干净的**单参数族**。代价是 crossover 的精确位置
  会依赖这一个 eps 实现——所以 Z2/Z3 的容差必须宽，且必须报跨算子的散度而不是单点。

用法： venv2/bin/python ess_threshold.py
"""

import glob
import json
import math
import os

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_path = os.path.join(_PROJ, "code", "blind_deblur_diffusion")
import sys
sys.path.insert(0, sys_path)

POOL = "/tmp/pool_ffhq64_256_s0.pt"
OUT = os.path.join(_PROJ, "results")
DENSE = ["W2", "W8", "TK50"]
T0 = 0.002
# 跨 sigma_n 的栅格。0.02/0.05/0.10 与 (ae) 逐字相同（跨轮次复现）。
# **跑后加密（2026-10-22 11:3x，披露）**：首版栅格最粗只到 [.. 5, 10, 20 ..]，把 Z3 的
# sigma_hi 夹在 (5, 10] 里 —— 若真值是 6，sigma_hi/0.2 = 30 而 Z3 会失败。所以「Z3 在 50.0
# 恰好通过」在粗栅格上**不可判定**。加密 6/7/8/9 把 sigma_hi 定位到 ~12%，另加 12/15 描出
# 陡升段。加密只影响分辨率，不改任何判据阈值；ess_a（主列）不受影响。
SN = [0.02, 0.05, 0.1, 0.3, 1.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0,
      20.0, 30.0, 50.0, 100.0, 300.0]
OLD = [0.02, 0.05, 0.1]           # 与 ess_scale.py 的 NOISES 对齐，用于跨轮次复现
TARGETS = [0.15, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.52,
           0.58, 0.65, 0.70, 0.74, 0.78, 0.82, 0.86]
H = W = 64
DEV = torch.device("cuda")
ETA_SEED = 4242


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
    """二分 sc 使 kunif 命中 target。**方向**与 ess_scale.py 逐字相同：
    weibull/tukey 的 sc 越大衰减越弱、kunif 越小，所以 kunif(mid)<target 时要调小 sc。
    """
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if kunif_of(kind, par, mid, r) < target:
            hi = mid
        else:
            lo = mid
    sc = 0.5 * (lo + hi)
    assert abs(kunif_of(kind, par, sc, r) - target) < 5e-3, "tune 未收敛"
    return sc


def main():
    r = halfplane()
    fams = {"W2": ("weibull", 2.0), "W8": ("weibull", 8.0), "TK50": ("tukey", 0.5)}
    ops = []
    for fid, (kind, par) in fams.items():
        for tg in TARGETS:
            sc2 = tune_rc(kind, par, r, tg)
            ops.append(dict(id=f"{fid}@{tg}", fid=fid,
                            g2=make_profile(kind, par, sc2, r), kunif=tg))

    pool = torch.load(POOL, map_location="cpu").to(torch.float64)   # (256,3,64,64)
    n_pool = pool.shape[0]
    XH = torch.fft.rfft2(pool, dim=(-2, -1))
    XS = torch.fft.rfft2(pool[0:1], dim=(-2, -1))
    print(f"池 {n_pool} 张；算子 {len(ops)} 个；sigma_n 栅格 {len(SN)} 点 "
          f"({SN[0]:g}..{SN[-1]:g})", flush=True)

    lam = {}
    for fn in ("sweep_bdense.json", "sweep_blow.json"):
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            continue
        for rec in json.load(open(p))["rows"]:
            if rec["meta"]["fid"] in DENSE:
                lam[(rec["meta"]["fid"], round(1 - rec["kunif"], 4))] = rec["lam_mean"]

    # 一个固定的单位高斯实现，对所有 sigma_n 复用 => sigma_n 是单参数族
    g = torch.Generator().manual_seed(ETA_SEED)
    eps = torch.randn(pool[0:1].shape, generator=g).to(torch.float64)
    EPS_F = torch.fft.rfft2(eps, dim=(-2, -1))

    rows = []
    for sn in SN:
        Gs = None
        yh = None
        for o in ops:
            G = torch.from_numpy(o["g2"]).to(torch.float64)[None, None]
            Gs = G.sqrt()
            yh = Gs * XS + sn * EPS_F
            D = (Gs * XH - yh).abs() ** 2
            # Parseval：rfft2 域求和 = N*像素域求和，必须除 H*W 才是 data 的常规定义
            # （与 ess_scale.py / test_lemma12 的像素域算法逐字对齐）。
            data = D.sum(dim=(1, 2, 3)) / (H * W) / (2 * sn ** 2)   # (n,)

            # --- Z1：两种 ESS 算法必须一致 ---
            m = float(data.min())
            Wt = torch.exp(-(data - m))                    # in (0,1]，max=1
            ess_a = float(Wt.sum() ** 2 / (Wt ** 2).sum())
            wm = float(Wt.mean())
            # **总体**方差（除 n，不是 n-1）：恒等式 ESS = n/(1+CV^2) 里的 CV^2 是
            # Var(w)/E[w]^2 的**总体**版。用无偏版会差一个因子 n/(n-1)，即 1/256=3.9e-3
            # ——首版就是这么错的，被 Z1 抓到。
            cv2 = float(((Wt - wm) ** 2).sum() / len(Wt)) / (wm * wm)
            ess_b = n_pool / (1.0 + cv2)

            # log-sum-exp 版（跨轮次复现用）。
            # **符号修正（2026-10-22，披露）**：首版（以及 ess_scale.py / §3.9(ae)）写的是
            #   log_nrat = log n + 2*m1 - m2 = log n + log ESS
            # 而它自称是 log(n/ESS) = log n - log ESS = log n - 2*m1 + m2。
            # 两项在 ESS=1 处**完全相同**，所以 (ae) 里 ESS 恒为 1 时这个 bug 完全不可见；
            # 一旦 ESS>1 就现形（ess_lse 变成 1/ESS）。主列 ess_a 不受影响。
            lw = -(data - m)
            m2 = float(torch.logsumexp(2 * lw, 0))
            m1 = float(torch.logsumexp(lw, 0))
            log_nrat = math.log(n_pool) - 2 * m1 + m2
            ess_lse = float(math.exp(-log_nrat)) * n_pool

            # --- Z4(b)：把 Var(data) 拆成确定性项与噪声交叉项 ---
            # A_i = ||Gs*(x_i - x*)||^2/(2 s^2)（跨池图的确定性项，~ s^-2）
            # B_i = ||eps_F||^2/2 - <Gs*(x_i-x*), eps_F>/s（同一 eps 的交叉项，~ s^-1）
            dX = (Gs * (XH - XS)).abs() ** 2
            A = dX.sum(dim=(1, 2, 3)) / (H * W) / (2 * sn ** 2)
            B = data - A
            a_np = A.numpy()
            b_np = B.numpy()
            vd = float(data.var(unbiased=True).item())
            vdet = float(((a_np - a_np.mean()) ** 2).sum() / (len(a_np) - 1))
            vx = float(((b_np - b_np.mean()) ** 2).sum() / (len(b_np) - 1))

            key = (o["fid"], round(1 - o["kunif"], 4))
            rows.append(dict(sn=sn, id=o["id"], fid=o["fid"], kunif=o["kunif"],
                             ess=ess_a, ess_b=ess_b, ess_lse=ess_lse,
                             log_nrat=log_nrat, cv2=cv2,
                             var_data=vd, var_det=vdet, var_x=vx,
                             lam=lam.get(key)))
            del D, dX, A, B, data, Wt
        del Gs, yh
        print(f"  sigma_n={sn:<7g} 完成 {len(rows)} 行累计", flush=True)

    with open(os.path.join(OUT, "ess_threshold.json"), "w") as fh:
        json.dump(dict(cfg=dict(pool=POOL, n_pool=n_pool, noises=SN, old=OLD,
                                targets=TARGETS, t=T0, eta_seed=ETA_SEED), rows=rows),
                  fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'ess_threshold.json')}")


if __name__ == "__main__":
    main()
