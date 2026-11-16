"""第十二组：把 (ag) 的 φ–tw_od 混淆解开——第二个 ⟨a⟩ 上的同类 tw_od 配对（GPU）

预官（2026-10-23，跑前写定）
================================================================================
背景（RESULTS §3.9(ag)）
------------------------
(ag) 判决了「$g_2$ 能到 0 但到不了 1」这一类：补 10 个新成员后，$tw_{\\rm od}$ 定律的残差从
$tw_{\\rm od}$=0.292 的 −0.119 dex 单调升到 1.011 的 +0.029 dex——**趋势不是散点**，定律在这一类
斜率太浅（类内 $C$ 实测跨 1.377× vs 定律预言 1.165×）。同类 $tw_{\\rm od}$ 的跨类配对显示
$\\phi$=0.583 比 $\\phi$=0 低 0.025 dex，$\\phi\\le0.417$ 时无差别。

**但 (ag) 留下一个没有闭合的口子**：类内 corr$(\\phi,tw_{\\rm od})=-0.925$。原因是结构性的——
固定 ⟨a⟩ 时这一类本质是**一参数族**：$\\phi=\\phi_0/\\langle a\\rangle$、$tw_{\\rm od}=tw_{\\rm lev}\\langle a\\rangle/(1-\\phi_0)$，
而 $tw_{\\rm lev}$ 又被 ⟨a⟩ 通过 $\\langle S\\rangle=(\\langle a\\rangle-\\phi_0)/(1-\\phi_0)$ 钉住。所以
「$C$ 依赖 $\\phi$」这个论断在 (ag) 里**没有干净的支撑**，只有 3 组固定 $\\phi$ 配对和 5 组跨类配对。

本组用**第二个 ⟨a⟩** 解开它
----------------------
跑前离线映射（零 GPU）给出两个 ⟨a⟩ 的可达区：
    ⟨a⟩=0.60（u=0.40，已在 (ag) 测过）：本类 tw_od ∈ [0.268, 1.011]，φ ∈ [0.167, 0.833]
    ⟨a⟩=0.35（u=0.65，本组新测）    ：本类 tw_od ∈ [0.061, 0.310]，φ ∈ [0.143, 0.857]
两者**只在 tw_od ≈ 0.26–0.31 重叠**，而正是在这个窗口里：
    u=0.65 侧可以取到 φ = 0.143 / 0.286 / 0.429 / 0.571 / 0.714 / 0.857
    u=0.40 侧在同一个 tw_od 上只有 φ = 0.750 / 0.833（(ag) 的 PD45m15 / PD50m15）
**即：同类 tw_od 下 φ 有 0.14 → 0.83 的 6 倍反差，而 (ag) 内部只有 0.75 vs 0.83。**
这就是解耦点——它只能靠第二个 ⟨a⟩ 得到，在单一 ⟨a⟩ 内物理上不存在。

设计
----
* 与 (ag) 逐字对齐：ffhq 池、20 张图、$t$=0.002、配对 Λ、底座+幂律过渡台阶族、
  $r_1$ 由 ⟨a⟩ 二分、$b(u)$ 与 mid(u) 用同一套普适曲线。唯一改的是 **kunif 目标 0.35**。
* 12 个算子 = **9 个本类**（φ 从 0.143 到 0.857，tw_od 从 0.061 到 0.308）
  + **3 个 φ₀=0 的有通带对照**（其中 2 个的 tw_od 与本类匹配，用于**同 u 内**的 φ 反差）。
* **主比较全部在 u=0.65 内部做**（同类 tw_od、同 u、同 b(u)、同 mid），
  因此 $C_{\\rm rel}$ 的 $u$ 漂移（§3.9(v1) 报过 ~1.10×）**不进入主判据**。
  跨 u（与 (ag) 的 u=0.40 点比）只作次级诊断，并明记 $u$ 漂移的混淆。

**跑前已算出的描述量与预言（全部列在这里）**（tw_od 定律用 (ag) 的预官版
$C=0.7032+0.3699/(1+(tw_{\\rm od}/0.4210)^{-1.231})$；$b(0.65)=-0.7569$、mid(0.65)=1655.36）：

  | 算子 | φ₀ | m | r₁ | amin | amax | tw_od | φ | C 预言 | 类 |
  |---|---|---|---|---|---|---|---|---|---|
  | PX30m015 | 0.30 | 0.15 | 0.804 | 0.300 | 1.000 | **0.061** | 0.857 | 0.735 | 本类 |
  | PX30m020 | 0.30 | 0.20 | 0.798 | 0.300 | 1.000 | 0.074 | 0.857 | 0.742 | 本类 |
  | PX25m015 | 0.25 | 0.15 | 0.739 | 0.250 | 1.000 | 0.082 | 0.714 | 0.747 | 本类 |
  | PX20m015 | 0.20 | 0.15 | 0.702 | 0.200 | 1.000 | 0.093 | 0.571 | 0.753 | 本类 |
  | PX25m030 | 0.25 | 0.30 | 0.716 | 0.250 | 1.000 | 0.126 | 0.714 | 0.771 | 本类 |
  | PX20m030 | 0.20 | 0.30 | 0.675 | 0.200 | 1.000 | 0.142 | 0.571 | 0.780 | 本类 |
  | PX15m020 | 0.15 | 0.20 | 0.662 | 0.150 | 1.000 | 0.124 | 0.429 | 0.770 | 本类 |
  | PX15m100 | 0.15 | 1.00 | 0.477 | 0.150 | 1.000 | 0.233 | 0.429 | 0.824 | 本类 |
  | PX10m200 | 0.10 | 2.00 | 0.135 | 0.100 | 1.000 | **0.285** | 0.286 | 0.845 | 本类 |
  | PX15m200 | 0.15 | 2.00 | 0.227 | 0.150 | 1.000 | **0.260** | 0.429 | 0.835 | 本类 |
  | PX00m100 | 0.00 | 1.00 | 0.324 | 0.000 | 1.000 | **0.286** | 0.000 | 0.845 | 有通带 |
  | PX00m070 | 0.00 | 0.70 | 0.427 | 0.000 | 1.000 | **0.258** | 0.000 | 0.834 | 有通带 |

  **解耦窗口（本组的目的地）**：tw_od ∈ [0.25, 0.30] 内有 5 个点，其中
  **两对 tw_od 几乎精确匹配**：
      配对A: PX00m100 (tw_od=0.286, φ=0.000)  vs  PX10m200 (tw_od=0.285, φ=0.286)   Δtw_od = 0.4%
      配对B: PX00m070 (tw_od=0.258, φ=0.000)  vs  PX15m200 (tw_od=0.260, φ=0.429)   Δtw_od = 0.8%
  另有一对宽松匹配：PX00m070 (0.258, φ=0) vs PX15m100 (0.233, φ=0.429)，Δtw_od = 9.7%。
  **即在同一个 u=0.65 上、tw_od 匹配到 1% 以内，φ 有 0 → 0.429 的反差。**
  第一版设计把类成员放在 tw_od ≤0.24，与对照不匹配（窗口内只有对照），已修正为 m=2.0
  ——这是跑前离线映射时发现的，不是跑后调整。

  **跨 u 的配对**：(ag) 的 PD50m15（u=0.40, tw_od=0.268, φ=0.833, C=0.760）与
  PD45m15（0.292, 0.750, 0.645）对 PX00m10/PX00m07（u=0.65, tw_od=0.286/0.258, φ=0）：
  若 φ 是因，则对照应接近定律（0.834–0.845）而 (ag) 那两点低 0.04–0.12 dex。

预官判据
--------
H0 守门（**必须全过**）：
   (a) 每算子 kunif 命中 0.35，重建对账 ≤5e−3（剖面由 (phi0,r1,m) 精确重建）；
   (b) 逐算子报 amin/amax 并按预官定义分类（本类：amin≥0.05 且 amax≥0.99）；
   (c) 逐格 SNR ≥3，否则**计数剔除**，不静默丢弃；
   (d) **同 u 桥接**：φ₀=0 的 3 个对照在 u=0.65 上的 tw_od 定律残差，须与有通带训练组
       在 u=0.65 上的最大逐点残差同量级。不过则新装置与旧数据不同尺，后面作废。

H1 主判据（**解耦检验，本组唯一的目的**）：在 u=0.65 内部、tw_od ∈ [0.25, 0.30] 的窗口里，
   用两对**tw_od 匹配到 1% 以内**的配对报 $\Delta\log_{10}C = \log_{10}C(\phi)-\log_{10}C(0)$：
       配对A（Δtw_od=0.4%）：φ=0.286 vs 0
       配对B（Δtw_od=0.8%）：φ=0.429 vs 0
   **预言两对的 Δlog₁₀C 都为负，且配对B ≤ −0.030 dex**（≥3% 的下降）。
   斜率估计来自 (ag)：在 tw_od≈0.29 处 class(φ=0.75) 比 baseline(φ=0) 低 0.107 dex，
   折合 **0.143 dex / 单位 φ**；线性外推给配对A −0.041、配对B −0.061 dex。
   阈值取 −0.030 dex（比预言松 50%），因为该斜率本身是单点估计且跨 u。
   **失败方式**：若两对的下降都 <0.015 dex，则 (ag) 的「$C$ 依赖 φ」被否掉，
   偏离必须归给 tw_od 本身或别的共变量——那 (ag) 的结论要改写成「本类整体低于定律」而
   不是「底座惩罚」。

H2 次级（跨 u 配对，**明记 u 漂移混淆**）：(ag) 的两个低 tw_od 类成员（u=0.40, φ=0.833/0.750）
   与本组的两个对照（u=0.65, φ=0）在 tw_od 匹配处的 $C_{\\rm rel}$ 差。
   **预言：对照不低于 (ag) 那两点**（差 ≥ +0.03 dex）。只作记录，不作判据——
   $C_{\\rm rel}$ 的 u 漂移 ~1.10× 与该效应同量级，本组无法分离。

H3 判别（向下外推）：本组 tw_od ≤0.15 的点上 tw_od 定律的残差。
   **预言 max|残差| ≤ 0.041 dex**（(ag) 的同一容差）。这些点的 tw_od 低到 0.061，
   而**全部 36 个基线剖面 + (ag) 的 13 个新点在这个区间一个都没有**——是真外推。
   **失败方式**：若残差系统性为负且均值 < −0.05 dex，则本类在 tw_od→0 端持续低于定律，
   (ag) 的「斜率太浅」被进一步加强。

H4 诊断（无阈值）：(a) u=0.65 上基线的 tw_od 覆盖与本组新点的关系；
   (b) $C_{\\rm rel}$ 的 u 漂移：对每个有 u=0.65 与 u=0.40 两个点的基线剖面，报
       $C_{\\rm rel}(0.65)/C_{\\rm rel}(0.40)$ 的分布——这给 H2 的混淆定量；
   (c) 本组新点与 (ag) 那 10 点合起来后的 corr(φ, tw_od)（预官：应显著低于 (ag) 的 −0.925）。

诚实声明
--------
* 只测 $t$=0.002、ffhq 先验、20 张图、单一分辨率、$u=0.65$ 一个点。
* 本组**不能**在单一 ⟨a⟩ 内解耦 φ 与 tw_od（几何约束，跑前已确认）；解耦靠第二个 ⟨a⟩，
  代价是引入 u 的变化，已用「主比较全在 u=0.65 内部」来规避。
* H1 的功效受限于 u=0.65 侧本类的 tw_od 上限 0.31：φ≥0.43 的点只能放到 tw_od 0.12–0.23，
  与对照（0.258–0.286）不匹配。所以 H1 实际只检验 φ ∈ [0, 0.286]，**不检验 φ>0.43**。

用法： venv2/bin/python sweep_pedestal2.py [--smoke]
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
KUNIF_TARGET = 0.35          # **本组唯一改的参数**：u = 0.65

# (id, phi0, m)：定稿设计见 docstring 的表。命名 PX{phi0*100}m{m*100}。
# **窗口内有两对 tw_od 几乎精确匹配的配对**（跑前离线算出）：
#   配对A: PX00m100(tw_od=0.286, phi=0)  vs PX10m200(tw_od=0.285, phi=0.286)  Δtw_od=0.4%
#   配对B: PX00m070(tw_od=0.258, phi=0)  vs PX15m200(tw_od=0.260, phi=0.429)  Δtw_od=0.8%
SPEC = [
    ("PX30m015", 0.30, 0.15), ("PX25m015", 0.25, 0.15), ("PX20m015", 0.20, 0.15),
    ("PX30m020", 0.30, 0.20), ("PX25m030", 0.25, 0.30), ("PX20m030", 0.20, 0.30),
    ("PX15m020", 0.15, 0.20), ("PX15m100", 0.15, 1.00), ("PX10m200", 0.10, 2.00),
    ("PX15m200", 0.15, 2.00),
    ("PX00m100", 0.00, 1.00), ("PX00m070", 0.00, 0.70),
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
    """kunif = <a> = mean(1-g2)。atten() 返回衰减 a，**不能**再写 1-mean（那是 u）。
    「--targets 传 kunif 不是 u」那条坑的第二次变体，冒烟的 kunif 列直接可查。"""
    return float(atten(phi0, r1, m, r).mean())


def solve_r1(phi0, m, r, target=KUNIF_TARGET):
    """二分 r1 使 kunif 命中 target。**方向**：r1 越大 S 越晚起来、<a> 越**小**，
    所以 kunif(mid)<target 时要调**小** r1。"""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if kunif_of(phi0, mid, m, r) < target:
            hi = mid
        else:
            lo = mid
    r1 = 0.5 * (lo + hi)
    assert abs(kunif_of(phi0, r1, m, r) - target) < 2e-3, \
        f"r1 解不收敛：phi0={phi0} m={m} knif={kunif_of(phi0, r1, m, r):.4f}"
    assert 1e-3 < r1 < 0.999, f"r1={r1:.4f} 贴边（phi0={phi0} m={m}）"
    return r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=20)
    ap.add_argument("--t", type=float, default=0.002)
    ap.add_argument("--out", default="sweep_pedestal2.json")
    args = ap.parse_args()
    if args.smoke and args.out == "sweep_pedestal2.json":
        args.out = "smoke_pedestal2.json"
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
