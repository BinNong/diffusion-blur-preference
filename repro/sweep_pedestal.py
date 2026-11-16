"""第十一组：「g₂ 能到 0 但到不了 1」这一类是独立 regime 还是 tw_od 的分支？（GPU）

预官（2026-10-23，跑前写定）
================================================================================
背景（RESULTS §3.9(r5) 第三行 / 开放问题 10）
-------------------------------------------
§3.9(r5) 把已测剖面分成四类，第三类「$g_2$ 能到 0 但到不了 1」（衰减有 DC 底座、
但在高频能到完全衰减）当时只有 kref=0.60 的 4 个底座剖面，$C$ 散在 0.900–1.028，
「未找到判据」。本组判决它。

**跑前已从已有数据算出的事实（全部列在这里，防事后合理化）**：
1. 按 (amin>0, amax≈1) 精确定义，这一类只有 **3 个**成员（描述量在 ⟨a⟩=0.60 归一化下算）：
       RP35  φ=0.823  tw_od=6.282  C=1.028
       RP75  φ=0.595  tw_od=1.934  C=0.937
       EF35  φ=0.112  tw_od=0.599  C=0.900
   （LG35 的 amax=0.88、TS2 的 amin=0，都不属于本类。）
2. tw_od 定律（§3.9(q2)，28 个训练剖面）$C=0.704+0.361/(1+(tw_{\rm od}/0.421)^{-1.23})$
   在这三点上的预言是 1.057 / 1.013 / 0.846，**残差 −0.012 / −0.034 / +0.027 dex**。
   该定律自身的 LOO 是 0.0144 dex、最大逐点残差 0.035 dex。**所以它描述本类大致不差于
   描述它自己的训练数据**——「未找到判据」这个说法已经过期，问题变成「n=3 够不够」。
3. **本类在 tw_od < 0.599 处完全没有成员**，而这正是定律预言最强的地方：
       tw_od=0.214 → C=0.814 ；0.30 → 0.848 ；0.394 → 0.877
   若这一类有 $C$ 的地板（≈0.90，不随 tw_od 下降），残差会是 +0.043 / +0.022 / +0.011 dex。

**一个跑前发现的几何约束（本组设计的出发点）**：
固定 ⟨a⟩=0.60 时，$a=\phi_0+(1-\phi_0)\cdot S(r)$ 且 $S(0)=0,S(1)=1$ 的**单体单调边**
（logistic / erf / 升余弦）**不可能**同时满足 amin>0 与 amax=1——因为径向模密度 ∝ r²
把均值拉向高衰减端，线性边的最低均值是 $0.75+0.25\phi_0>0.60$。可用的是
**「底座 + 幂律过渡台阶」**：
$$a(r)=\phi_0+(1-\phi_0)\cdot\Big(\frac{\max(r-r_1,\,0)}{1-r_1}\Big)^{m}$$
$r_1$ 由 ⟨a⟩=0.60 二分解出，$m$ 是过渡形状。数值映射（跑前完成）给出本类的可达区：
    $\phi_0\in[0.10,0.30]$，$tw_{\rm od}\in[0.214,\,0.394]$，$\phi\in[0.25,0.75]$
且 **类内 corr(φ, tw_od) = −0.896**——在固定 ⟨a⟩ 下二者被几何锁死，**无法解耦**。
（$\phi_0\ge0.40$ 时解塌到 $r_1=1$，即 $a\equiv0.40$ 的均匀算子，属「窄带」类，已排除。）
所以「是 φ 还是 tw_od」这个问题在固定 ⟨a⟩ 下**不可答**，本组改问可答的那个：
**把本类延伸到从未采样的低 tw_od 区，看 C 是否跟随定律下降。**

设计
----
* 图池/图数/噪声级/提取方式与第八、十组逐字对齐：ffhq 池、20 张、$t$=0.002、
  $\Lambda=\log p_t(y)-\log p_t(x)$ 逐图配对。`/tmp/pool_ffhq64_256_s0.pt`。
* 全部算子 ⟨a⟩=0.60（kunif=0.60 ⇒ u=0.40），与 §3.9(q) 的共同 u 网格点 0.4 对齐，
  因此 $C(u{=}0.40)=\Lambda\cdot 0.40^{-b(0.40)}$ 可与全部 28 个基线剖面**逐点**比较。
* 13 个算子 = **10 个本类** + **3 个 $\phi_0=0$ 的有通带对照**。
* **定稿设计的描述量与 C 预言（跑前离线算出，全部列在这里）**：

  | 算子 | $\phi_0$ | $m$ | $r_1$ | amin | amax | $tw_{\rm od}$ | $\phi$ | $C$ 预言 | 类 |
  |---|---|---|---|---|---|---|---|---|---|
  | PD50m15 | 0.50 | 0.15 | 0.693 | 0.500 | 1.000 | **0.268** | 0.833 | **0.835** | 本类 |
  | PD45m15 | 0.45 | 0.15 | 0.650 | 0.450 | 1.000 | 0.292 | 0.750 | 0.845 | 本类 |
  | PD40m15 | 0.40 | 0.15 | 0.613 | 0.400 | 1.000 | 0.314 | 0.667 | 0.852 | 本类 |
  | PD40m20 | 0.40 | 0.20 | 0.598 | 0.400 | 1.000 | 0.389 | 0.667 | 0.876 | 本类 |
  | PD35m30 | 0.35 | 0.30 | 0.530 | 0.350 | 1.000 | 0.530 | 0.583 | 0.910 | 本类 |
  | PD30m50 | 0.30 | 0.50 | 0.417 | 0.300 | 1.000 | 0.711 | 0.500 | 0.941 | 本类 |
  | PD30m10 | 0.30 | 1.00 | 0.208 | 0.300 | 1.000 | 0.899 | 0.500 | 0.963 | 本类 |
  | PD20m50 | 0.20 | 0.50 | 0.341 | 0.200 | 1.000 | 0.782 | 0.333 | 0.950 | 本类 |
  | PD25m70 | 0.25 | 0.70 | 0.287 | 0.250 | 1.000 | 0.850 | 0.417 | 0.958 | 本类 |
  | PD20m10 | 0.20 | 1.00 | 0.086 | 0.200 | 1.000 | **1.011** | 0.333 | **0.973** | 本类 |
  | PB00m20 | 0.00 | 0.20 | 0.398 | 0.000 | 1.000 | 0.516 | 0.000 | 0.907 | 有通带 |
  | PB00m35 | 0.00 | 0.35 | 0.310 | 0.000 | 1.000 | 0.754 | 0.000 | 0.947 | 有通带 |
  | PB00m70 | 0.00 | 0.70 | 0.083 | 0.000 | 1.000 | 1.069 | 0.000 | 0.978 | 有通带 |

  **本类新点把 tw_od 从旧最低 0.599 向下延伸 2.24 倍**（到 0.268），$C$ 预言跨度 1.165×。
  **若本类有 $C=0.900$ 的地板，最低点残差 = +0.065 dex ≈ 6.5σ**（单点 $C$ 噪声 ~0.01 dex）。
  有通带对照给出 **5 个同类 tw_od 配对**（相差 ≤15%），定律预言二者只差 0.4–1.2%。
  固定 $\phi$ 的类内配对有 3 组（$\phi$ = 0.333 / 0.500 / 0.667），用于 G3。
* 每个算子同时落盘 `lam_gauss`、`kunif`、逐图 `lam_per_img`（判据要用）。

预官判据
--------
G0 守门（**必须全过，否则后面作废**）：
   (a) 每算子 kunif 命中 0.60，相对误差 ≤ 5e-3（$r_1$ 的二分收敛性）；
   (b) 逐算子报 amin / amax，并**按预官定义分类**：本类要求 amin ≥ 0.05 且 amax ≥ 0.99
       （$g_2$ 顶棚 ≤0.95、且能到 0）。不达标的算子**计数剔除**，不静默丢弃；
   (c) 逐格 SNR = Λ̄√n/s ≥ 3，否则计数剔除（§3.9(aa) S5 的同一规矩）；
   (d) **桥接**：$\phi_0=0$ 的 3 个对照算子退化为普通有通带剖面，它们在同 tw_od 区间的
       tw_od 定律残差须与有通带训练组同量级（|残差| ≤ 0.038 dex = 该定律的最大逐点残差）。
       若不过，说明新装置与旧数据不一致。

G1 最尖（**向下外推到本类从未采样的 tw_od<0.599**）：本类新算子上 tw_od 定律的残差，
   **预言 max|残差| ≤ 0.038 dex**（容差取该定律在 28 个训练剖面上的最大逐点残差，
   与 §3.9(q3) 的取法逐字一致，不硬编码）。**这条的预官间距已经量化**：最低 tw_od=0.268
   处定律给 0.835，而本类已测成员的 C 下限是 0.900——若存在地板，残差是 +0.065 dex
   （6.5σ），远大于容差；若不存在地板，残差应在 ±0.038 内。
   **失败方式（本条唯一有价值的失败方向）**：若残差**系统性为正**且均值 > +0.025 dex
   （= 2.5× 单点 C 噪声 0.01 dex），说明本类有 $C$ 的**地板**、不随 tw_od 降到 0.81，
   即它是**独立 regime** 而不是 tw_od 的分支。此时要报的是「地板在哪、多高」。

G2 判别（同类 tw_od 的跨类配对）：在 tw_od 相差 ≤15% 的 (本类, 有通带对照) 配对上，
   报 $C$ 之比与配对显著性。**预言二者之差 ≤ 0.02 dex**（定律给 1.4%）。
   **失败方式**：若本类系统高出 > +0.025 dex，则同 tw_od 下两类不同 ⇒ φ 是一等描述量、
   tw_od 不完备。注意这一条的预官间距只有 0.017 dex ≈ 1.7× 单点噪声，所以**同时报可探测
   下限**，若实际间距 <2σ 按项目规矩降级为「只记录」。

G3 判别（类内的 tw_od 效应）：本类新点的 $C$ 对 tw_od 的 Spearman ≥ 0.9，
   且 $C(\max tw_{\rm od})/C(\min tw_{\rm od}) \ge 1.06\times$（定律预言 1.078×）。
   **失败方式**：若 $C$ 在类内**不随** tw_od 变（比值 ≤1.03×），则类内 $C$ 是平台，
   与 §3.9(r) 的窄带平台同类。

G4 诊断（无阈值）：(φ, tw_od) 的实际覆盖与类内 corr；(a) 预官的「单体单调边不可能」
   这一几何约束的数值确认（报线性边在 ⟨a⟩=0.60 下的最低可达均值）。

诚实声明
--------
* 只测 $t$=0.002、ffhq 先验、20 张图、单一分辨率、$u=0.40$ 一个点。
* 本类只有 3 个旧成员，本组加 10 个；「独立 regime vs 分支」的结论强度受 n=13 限制。
* 固定 ⟨a⟩ 下 φ 与 tw_od 锁死（corr −0.896），所以本组**不能**回答「C 由 φ 还是由
  tw_od 决定」，只能回答「C 是否随 tw_od 下降」。要解耦需要换 ⟨a⟩（更多 GPU）。

用法： venv2/bin/python sweep_pedestal.py [--smoke]
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
H = W = 64
POOL = "/tmp/pool_ffhq64_256_s0.pt"
KUNIF_TARGET = 0.60

# (id, phi0, m)：phi0=DC 底座；m=过渡形状。phi0=0 的是有通带对照。
# **定稿设计（2026-10-23 跑前，离线映射后选定；描述量与 C 预言全部已算出，见 docstring）**
SPEC = [
    ("PD50m15", 0.50, 0.15), ("PD45m15", 0.45, 0.15), ("PD40m20", 0.40, 0.20),
    ("PD40m15", 0.40, 0.15), ("PD35m30", 0.35, 0.30), ("PD30m50", 0.30, 0.50),
    ("PD30m10", 0.30, 1.00), ("PD25m70", 0.25, 0.70), ("PD20m50", 0.20, 0.50),
    ("PD20m10", 0.20, 1.00),
    ("PB00m20", 0.00, 0.20), ("PB00m35", 0.00, 0.35), ("PB00m70", 0.00, 0.70),
]


def halfplane():
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)[:, :W // 2 + 1]
    return R / R.max()


def atten(phi0, r1, m, r):
    """a(r) = phi0 + (1-phi0)*S(r)，S = ((r-r1)/(1-r1))^m 截断到 [0,1]。"""
    s = np.zeros_like(r)
    msk = r >= r1
    s[msk] = ((r[msk] - r1) / max(1.0 - r1, 1e-12)) ** m
    return phi0 + (1.0 - phi0) * np.clip(s, 0.0, 1.0)


def kunif_of(phi0, r1, m, r):
    """kunif = <a> = mean(1-g2)。**注意**：atten() 返回的是衰减 a，不是传输 g2，
    所以这里**不能**再写 1 - mean（那给出的是 u）。第一版就是这么错的——
    「--targets 传 kunif 不是 u」那条坑的新变体，已由冒烟抓出。"""
    return float(atten(phi0, r1, m, r).mean())


def solve_r1(phi0, m, r, target=KUNIF_TARGET):
    """二分 r1 使 knif 命中 target。knif 随 r1 单调增（r1 越大 S 越早起来）。"""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        # **方向**：r1 越大 => S 越晚起来 => <a> 越**小**。所以 kunif(mid)<target
        # （均值太小）时要调**小** r1。第一版按递增写，冒烟时全部算子 kunif 读 0.40。
        if kunif_of(phi0, mid, m, r) < target:
            hi = mid
        else:
            lo = mid
    r1 = 0.5 * (lo + hi)
    # 解塌到边界意味着 phi0 太高（a 退化为常数），必须报错而不是静默给出均匀算子
    assert abs(kunif_of(phi0, r1, m, r) - target) < 2e-3, \
        f"r1 解不收敛：phi0={phi0} m={m} knif={kunif_of(phi0, r1, m, r):.4f}"
    assert 1e-3 < r1 < 0.999, f"r1={r1:.4f} 贴边（phi0={phi0} m={m}）"
    # r1 很小意味着过渡几乎从 r=0 就开始（退化为光滑底座+幂律），仍是本类合法成员。
    return r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=20)
    ap.add_argument("--t", type=float, default=0.002)
    ap.add_argument("--out", default="sweep_pedestal.json")
    args = ap.parse_args()
    if args.smoke and args.out == "sweep_pedestal.json":
        args.out = "smoke_pedestal.json"
    nimg = 2 if args.smoke else args.nimg

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    r = halfplane()
    ops = []
    for oid, phi0, m in SPEC:
        r1 = solve_r1(phi0, m, r)
        a = atten(phi0, r1, m, r)
        g2 = 1.0 - a
        ops.append((oid, g2, dict(fid=oid, kind="pedestal", par=float(m),
                                   phi0=float(phi0), r1=float(r1),
                                   K=float(KUNIF_TARGET), beta=1.0,
                                   kunif_ref=float(KUNIF_TARGET))))
        print(f"[build] {oid}: phi0={phi0:.2f} m={m:.2f} r1={r1:.4f} "
              f"kunif={1.0-g2.mean():.4f} amin={a.min():.4f} amax={a.max():.4f}",
              flush=True)
    print(f"[cfg] nimg={nimg} t={args.t:g} kunif={KUNIF_TARGET} "
          f"{len(ops)} 个算子", flush=True)

    model = load_local_model(CKPT).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x.to(DT), t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)

    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:nimg].clone()
    with torch.no_grad():
        P = (torch.fft.rfft2(pool.to(torch.float64), dim=(-2, -1)).abs() ** 2) / (H * W)
        Sfull = P.mean(1).mean(0)
        Zx = (torch.fft.rfft2(xs.to(torch.float64), dim=(-2, -1)).abs() ** 2) \
            .mean(1, keepdim=True).mean(0)[0] / (H * W)
    Sf = Sfull + args.t ** 2
    wg = Zx / (2.0 * Sf)
    cap = float((Zx / (2.0 * Sf)).sum())
    B = 4

    @torch.no_grad()
    def logp(z):
        out = []
        for i in range(0, z.shape[0], B):
            nll = likelihood_fn(z[i:i + B].to(DT), ode_flow, t_spans=[80.0, args.t])
            out.extend([-float(v) for v in nll.reshape(-1)])
            torch.cuda.empty_cache()
        return np.array(out)

    t0 = time.time()
    lp_sharp = logp(xs)
    print(f"[sharp] t={args.t:g} mean log p = {lp_sharp.mean():.1f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for i, (oid, g2, meta) in enumerate(ops):
        g = torch.from_numpy(g2).to(DEV)[None, None].to(torch.float64)
        with torch.no_grad():
            y = torch.fft.irfft2(torch.fft.rfft2(xs.to(torch.float64),
                                                 dim=(-2, -1)) * g.sqrt(),
                                 s=(H, W), dim=(-2, -1)).to(DT)
        tt = time.time()
        lp = logp(y)
        lam = lp - lp_sharp
        lam_g = float(((1.0 - g2) * wg.cpu().numpy()).sum())
        kunif = float(1.0 - g2.mean())
        rows.append(dict(id=oid, meta=meta, kunif=kunif,
                         lam_mean=float(lam.mean()), lam_std=float(lam.std()),
                         lam_per_img=[float(v) for v in lam],
                         lam_gauss=lam_g,
                         logp_blur_mean=float(lp.mean()),
                         logp_sharp_mean=float(lp_sharp.mean())))
        snr = lam.mean() * math.sqrt(nimg) / max(lam.std(), 1e-9)
        print(f"  [{i+1}/{len(ops)}] {oid:<10} knif={kunif:.4f} "
              f"Lambda={lam.mean():>9.0f}+-{lam.std()/math.sqrt(nimg):>6.0f} "
              f"SNR={snr:>5.1f}  L_G={lam_g:>7.1f}  ({time.time()-tt:.0f}s)", flush=True)
        del y, g
        torch.cuda.empty_cache()

    outfn = os.path.join(OUT, args.out)
    json.dump(dict(cfg=dict(nimg=nimg, t=args.t, ckpt=CKPT, pool=POOL,
                            kunif_target=KUNIF_TARGET, cap=cap,
                            spec=[dict(id=s[0], phi0=s[1], m=s[2]) for s in SPEC]),
                    rows=rows), open(outfn, "w"), indent=1)
    print(f"\n[saved] {outfn}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
