"""第十一组：定律的样本量升级 20 -> 120 张图（预官，跑前写定）
================================================================

动机（论文弱点，按投 TIP 的自我审查排的第一条）
----------------------------------------------
论文的定量核心全部建立在 **20 张图** 上：
  * 图 1（匹配 u 的跨族一致性 1.42-1.77x）<- sweep_profile_shape.json
  * 图 2（段指数 + 闭式曲线 + 5 个新族的样本外检验）<- sweep_profile_shape.json
    + sweep_law_test.json
  * 摘要里的闭式 b(u) = -0.70 - 6.4 u^3.1 <- sweep_bdense.json + sweep_blow.json
审稿人第一刀必然是"N=20 不是定律，是噪声实现"。本项目在 §3.9(ai) 已经做过一次
20->120 的升级，效应量从 3.2x 修正到 2.47x（**结论方向不变、数值变了 23%**）——
所以"更多图会改变数字"是已证实的风险，必须预官判据而不是跑完再解释。

实验设计（只改 nimg，其余逐项对齐）
----------------------------------
  同一个池文件 /tmp/pool_ffhq64_256_s0.pt，xs = pool[:nimg]，所以
  pool[:20] 严格是 pool[:120] 的前缀——**两轮的图像完全相同，只是后 100 张是新加的**。
  Hutchinson 探针的 RNG 流依赖 nimg（lp_sharp 先消费 nimg 份），所以逐图 Λ 值
  **不是**逐位相同：这是独立噪声下的复现，不是同 seed 重跑。
  算子表、kunif_ref、targets、t 全部逐字相同（下面逐个文件给出原始命令行）。

数据对（每对只差 nimg）
----------------------
  D1 图 1 / 段指数 / 带：
        N=20  sweep_profile_shape.json          (--nimg 20, parts A B C V)
        N=120 sweep_pshape_n120_A.json + sweep_pshape_n120_BCV.json
              (--nimg 120 --parts A ; --nimg 120 --parts B C V)
  D2 图 2 的样本外检验（5 个新族 W1/W2/W4/B1/M2 x 6 算子）：
        N=20  sweep_law_test.json               (--nimg 20，默认族/默认 targets)
        N=120 sweep_law_n120.json               (--nimg 120，同上)
  D3 闭式 b(u)=b_inf+A*u^q（3 族 W2/W8/TK50 x 20 个 u 点）：
        N=20  sweep_bdense.json + sweep_blow.json
        N=120 sweep_bdense_n120.json + sweep_blow_n120.json

判据（跑前写定；R0 是复现门，不过则任何结论都不可报告）
-----------------------------------------------------
R0 复现门：本裁判用**已发布脚本逐字复制**的提取逻辑，作用在 N=20 文件上，必须
   复现已发布数值（来源 results/fig_law.json，由已发布的 plot_law_test.py 产出）：
     (a) 三个段指数与 8 族范围：[0.65,0.40] mean -1.5721 range [-1.6802,-1.4985]
         [0.40,0.30] mean -0.8770 range [-0.9059,-0.8232]
         [0.30,0.15] mean -0.7569 range [-0.8159,-0.6764]
     (b) 5 个新族对 7 模糊带的最大偏离比：W1 1.0141 / W2 0.9957 / W4 1.0282 /
         B1 1.0189 / M2 1.0193（最大 1.0282）
     (c) 理想低通 C 范围 [0.6362, 0.9241]
     (d) D3 闭式参数 b_inf=-0.705 / A=-6.4 / q=3.1（§3.9(t) 基线，t=0.002）
   容差 1e-3（段指数/偏离比）与 0.02（闭式参数）。R0 任一不过 => 裁判坏，停止。

R1 逐点复现（统计门）：D3 的每个 (族, 算子) 共有点上
       z = (Λ̄_120 - Λ̄_20) / SE_20,   SE_20 = std_20/sqrt(20)（N=20 文件自带）
   若 20 图均值无偏，z 的 SD ≈ 0.91（推导：m120-m20 = (100/120)(m100-m20)，
   SD = (100/120)*sqrt(1/100+1/20)*σ = 0.204σ = 0.91*SE_20）。
   判据：median|z| <= 0.8 且 max|z| <= 3.5。同时报 |z|>2 的占比（二项期望 4.6%）。
   这条不过 => 20 图分析有系统性问题，已发布数字不可信。

R2 闭式参数不动（定律门）：用**同一套提取**（logΛ 对 log u 的 5 次多项式在
   **全部**共有 u 上拟合、闭式 b=b_inf+A*u^q 在 [lo,hi] 子集上拟合——与
   analyze_bt.py 逐字一致）在 N=120 上重拟，对 W2/W8/TK50 逐个要求
       |Δb_inf| <= 2*SE_20(b_inf)、|ΔA| <= 2*SE_20(A)、|Δq| <= 2*SE_20(q)
   SE_20 由对 N=20 文件的同一 2000 次有放回重采样给出。
   任一项不过 => 已发布参数不被支持，论文数字必须改，并在文档里记录改了多少。

R3 精度增益：SE(b_inf) 在 N=120 必须 <= SE_20/1.6（理想增益 sqrt(6)=2.45，
   1.6 = 理想的 65%）。不过 => 新数据比预期噪声大，升级无意义。

R4 样本外检验存活：5 个新族在 **N=120 自己的 7 模糊带** 上的最大偏离比
   <= 1.10（已发布值 1.0282；1.10 容忍新的噪声实现，但抓住真爆炸）。
   逐个报告 W1/W2/W4/B1/M2。

R5 符号稳定性：mean Λ <= 0 的 (族, 算子) 点数占比，N=120 必须 <= N=20。
   （§3.9(ai) 的实测：某一类 20 图时 25%、120 图时 9.2%。）

R6 诊断（无阈值，只报数）：每族每参数的 SE 增益、z 分布、N=120 的理想低通 C 范围、
   N=120 的匹配 u 跨族 max/min、以及 D2 的 C(u) 曲线是否仍塌缩到带内。

跑法
----
  venv2/bin/python judge_law_n120.py                 # 全跑
  venv2/bin/python judge_law_n120.py --nboot 500     # 快速版（只影响 R2/R3 的 SE）
  venv2/bin/python judge_law_n120.py --dump results/fig_law_n120.json

依赖：analyze_bt.py 的 bcurve/closed_form/grid/matrices/Dvec（逐字 import，不复制），
      plot_law_test.py 的 load/interp/段指数/带 逻辑（逐字复制到本文件）。
"""

import argparse
import glob
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
sys.path.insert(0, os.path.join(PROJ, "code", "blind_deblur_diffusion"))

# ---- 逐字复用 analyze_bt.py 的提取原语（D3 用）----
_spec = importlib.util.spec_from_file_location(
    "abt", os.path.join(PROJ, "repro", "analyze_bt.py"))
abt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abt)

DENSE = ["W2", "W8", "TK50"]          # 与 analyze_bt.py 一致
DEG = 5
NBOOT = 2000

# ---- 逐字复制自 plot_law_test.py（D1/D2 用）----
SMOOTH_OLD = [("gaussian", 1.0), ("gaussian", 2.0), ("defocus", 1.0),
              ("defocus", 2.0), ("airy", 4.0), ("motion", 3.0), ("motion", 7.0)]
IDEAL = ("lowpass", 0.25)
SEGS = [(0.65, 0.40), (0.40, 0.30), (0.30, 0.15)]
NEWF = ["W1", "W2", "W4", "B1", "M2"]

# R0 的已发布目标值（来源 results/fig_law.json）
PUB_SEG = [(-1.5721, -1.6802, -1.4985), (-0.8770, -0.9059, -0.8232),
           (-0.7569, -0.8159, -0.6764)]
PUB_NEWDEV = {"W1": 1.0141, "W2": 0.9957, "W4": 1.0282, "B1": 1.0189,
              "M2": 1.0193}
PUB_NEWREL = {"W1": 1.0535, "W2": 1.1270, "W4": 1.0790, "B1": 1.0651,
              "M2": 1.0444}
PUB_IDEAL = (0.6362, 0.9241)

# R2 的原始命令行（跑前记录，保证算子表逐字相同）
CMDS = {
    "D1a": "sweep_profile_shape.py --nimg 20 --parts A",
    "D1b": "sweep_profile_shape.py --nimg 20 --parts B C V",
    "D1a120": "sweep_profile_shape.py --nimg 120 --parts A --out sweep_pshape_n120_A.json",
    "D1b120": "sweep_profile_shape.py --nimg 120 --parts B C V --out sweep_pshape_n120_BCV.json",
    "D2": "sweep_law_test.py --nimg 20",
    "D2_120": "sweep_law_test.py --nimg 120 --out sweep_law_n120.json",
    "D3a": "sweep_law_test.py --nimg 20 --targets 0.15 0.18 0.22 0.26 0.30 0.34 0.38 0.42 0.46 0.52 0.58 0.65 0.70 --families W2 W8 TK50 --out sweep_bdense.json",
    "D3b": "sweep_law_test.py --nimg 20 --targets 0.74 0.78 0.82 0.86 --families W2 W8 TK50 --out sweep_blow.json",
    "D3a120": "sweep_law_test.py --nimg 120 --targets 0.15 0.18 0.22 0.26 0.30 0.34 0.38 0.42 0.46 0.52 0.58 0.65 0.70 --families W2 W8 TK50 --out sweep_bdense_n120.json",
    "D3b120": "sweep_law_test.py --nimg 120 --targets 0.74 0.78 0.82 0.86 --families W2 W8 TK50 --out sweep_blow_n120.json",
}


# =====================================================================
#  载入：D1/D2 用 plot_law_test.py 的约定；D3 用 analyze_bt.py 的约定
# =====================================================================
def load_ps(fn):
    """sweep_profile_shape.json -> {(family,theta): [(u, lam_mean), ...]}"""
    d = json.load(open(os.path.join(OUT, fn)))
    old = defaultdict(list)
    for r in d["rows"]:
        m = r["meta"]
        if "family" in m:
            old[(m["family"], m["theta"])].append((1.0 - r["kunif"], r["lam_mean"]))
    for k in old:
        old[k].sort(key=lambda c: -c[0])
    return d, dict(old)


def load_law(fn):
    """sweep_law_test.json -> {fid: [(u, lam_mean), ...]}"""
    d = json.load(open(os.path.join(OUT, fn)))
    new = defaultdict(list)
    for r in d["rows"]:
        new[r["meta"]["fid"]].append((1.0 - r["kunif"], r["lam_mean"]))
    for k in new:
        new[k].sort(key=lambda c: -c[0])
    return d, dict(new)


def load_bt(fns):
    """sweep_bdense/blow -> {fid: [(u, lam_mean, lam_per_img), ...]}（analyze_bt 约定）"""
    prof = defaultdict(list)
    gauss = defaultdict(dict)
    for fn in fns:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            return None, None
        d = json.load(open(p))
        for rec in d["rows"]:
            prof[rec["meta"]["fid"]].append(
                (1.0 - rec["kunif"], rec["lam_mean"], rec.get("lam_per_img")))
            gauss[rec["meta"]["fid"]][round(1.0 - rec["kunif"], 4)] = rec.get("lam_gauss")
    for k in prof:
        prof[k].sort(key=lambda c: -c[0])
    return dict(prof), {k: dict(v) for k, v in gauss.items()}


def interp(curve, u):
    """逐字复制自 plot_law_test.py：log-log 线性插值，超出支撑返回 None。"""
    ks = [c[0] for c in curve]
    if u < min(ks) or u > max(ks):
        return None
    for i in range(len(ks) - 1):
        a, b = ks[i], ks[i + 1]
        if u <= a and u >= b:
            w = (math.log(u) - math.log(a)) / (math.log(b) - math.log(a))
            return math.exp(math.log(curve[i][1]) * (1 - w)
                            + math.log(curve[i + 1][1]) * w)
    return None


# =====================================================================
#  D1/D2 的三个已发布量：段指数、7 模糊带、新族偏离比
# =====================================================================
def segments(old):
    seg_u, seg_b, seg_lo, seg_hi = [], [], [], []
    for a, b in SEGS:
        bs = []
        for k in SMOOTH_OLD:
            va, vb = interp(old.get(k, []), a), interp(old.get(k, []), b)
            if va and vb and va > 0 and vb > 0:
                bs.append(math.log(vb / va) / math.log(b / a))
        if len(bs) >= 5:
            seg_u.append(math.sqrt(a * b)); seg_b.append(float(np.mean(bs)))
            seg_lo.append(min(bs)); seg_hi.append(max(bs))
    return (np.array(seg_u), np.array(seg_b), np.array(seg_lo), np.array(seg_hi))


def b_of_u(seg_u, seg_b, u):
    if u <= seg_u[0]:
        return seg_b[0]
    if u >= seg_u[-1]:
        return seg_b[-1]
    lx = math.log(u)
    for i in range(len(seg_u) - 1):
        if seg_u[i] <= u <= seg_u[i + 1]:
            w = (lx - math.log(seg_u[i])) / \
                (math.log(seg_u[i + 1]) - math.log(seg_u[i]))
            return seg_b[i] * (1 - w) + seg_b[i + 1] * w
    return seg_b[-1]


def band(old, seg_u, seg_b, U):
    """7 模糊族的 min-max 带与几何均值（逐字复制 plot_law_test.py 的约定）。"""
    lo, hi, mid, us = [], [], [], []
    for u in U:
        if u < 0.12:
            continue
        vs = []
        for k in SMOOTH_OLD:
            v = interp(old.get(k, []), u)
            if v:
                vs.append(v * u ** (-b_of_u(seg_u, seg_b, u)))
        if len(vs) >= 5:
            lo.append(min(vs)); hi.append(max(vs))
            mid.append(float(np.exp(np.mean([math.log(v) for v in vs]))))
            us.append(u)
    return np.array(lo), np.array(hi), np.array(mid), np.array(us)


def new_dev(new, seg_u, seg_b, lo, hi, mid, us):
    """5 个新族的 C(u) 归一化后对带的最大偏离比。

    逐字对齐 plot_law_test.py：
        ys = [norm(u, interp(new[fid],u) * u**(-b_of(u))) for u in xs]
        norm(u,v) = v / np.interp(log u, log us_ok, band_mid)
    **u**(-b_of(u)) 这个因子不能漏**——漏了就从"比 C"变成"比 Λ"，而出带比会从 ~1 变成 ~25。
    （第一版就漏了，R0 复现门抓到：段指数和闭式逐位复现，只有这一块炸。）

    报两个口径，论文里两个都在用，缺一个就会对不上：
      maxdev = max(lo/c, c/hi)  —— 相对**带边缘**（出带多少倍；已发布最大 1.0282）
      maxrel = max|C/mid|       —— 相对**带均值**（论文图 2 题注的"1.13x of its mean"）
    """
    out = {}
    for f in NEWF:
        cu, cc = [], []
        for u in us:
            v = interp(new.get(f, []), u)
            if v:
                cu.append(u)
                cc.append((v * u ** (-b_of_u(seg_u, seg_b, u))) / float(
                    np.interp(math.log(u), np.log(us), mid)))
        if len(cu) < 5:
            out[f] = None
            continue
        c = np.array(cc)
        l = np.interp(np.log(np.array(cu)), np.log(us), lo / mid)
        h = np.interp(np.log(np.array(cu)), np.log(us), hi / mid)
        out[f] = dict(u=np.array(cu), C=c, lo=l, hi=h,
                      maxdev=float(np.maximum(l / c, c / h).max()),
                      maxrel=float(np.abs(c).max()))
    return out


# =====================================================================
#  D3：闭式参数的 bootstrap SE（与 analyze_bt.py 的配对 bootstrap 同源）
# =====================================================================
def fit_once(P, us_all, us):
    mu = P.mean(0)
    if not (mu > 0).all():
        return None
    try:
        c = np.polyfit(np.log(np.array(us_all)), np.log(mu), DEG)
    except Exception:
        return None
    d = np.polyder(c)
    return abt.closed_form(us, lambda u, d=d: float(np.polyval(d, math.log(u))))


def boot_se(P, us_all, us, nboot, rng):
    ok = []
    n = P.shape[0]
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        cf = fit_once(P[idx], us_all, us)
        if cf is not None:
            ok.append(cf["p"])
    if len(ok) < 50:
        return None, len(ok)
    a = np.array(ok)
    return dict(b_inf=float(a[:, 0].std(ddof=1)),
                A=float(a[:, 1].std(ddof=1)),
                q=float(a[:, 2].std(ddof=1))), len(ok)


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    U = np.logspace(math.log10(0.06), math.log10(0.72), 40)
    rep = {}

    # ---------------- R0：复现门 ----------------
    print("=" * 100)
    print(" R0 复现门：本裁判作用在 N=20 文件上，必须复现已发布数值")
    print("=" * 100)
    _, old20 = load_ps("sweep_profile_shape.json")
    _, new20 = load_law("sweep_law_test.json")
    su, sb, slo, shi = segments(old20)
    lo, hi, mid, us = band(old20, su, sb, U)
    dev20 = new_dev(new20, su, sb, lo, hi, mid, us)
    r0 = []
    for i, (pm, pl, ph) in enumerate(PUB_SEG):
        got = (float(sb[i]), float(slo[i]), float(shi[i]))
        ok = (abs(got[0] - pm) < 1e-3 and abs(got[1] - pl) < 1e-3
              and abs(got[2] - ph) < 1e-3)
        r0.append(("seg%d" % i, ok, got, (pm, pl, ph)))
        print(f" 段 {SEGS[i]}: mean={got[0]:+.4f} range=[{got[1]:+.4f},{got[2]:+.4f}]"
              f"  已发布 mean={pm:+.4f} range=[{pl:+.4f},{ph:+.4f}]  {'OK' if ok else 'FAIL'}")
    for f in NEWF:
        g = dev20[f]["maxdev"]
        ok = abs(g - PUB_NEWDEV[f]) < 1e-3
        r0.append(("dev_" + f, ok, g, PUB_NEWDEV[f]))
        print(f" {f} 出带比 = {g:.4f}（已发布 {PUB_NEWDEV[f]:.4f}）  "
              f"对均值 = {dev20[f]['maxrel']:.4f}  {'OK' if ok else 'FAIL'}")
    idu = [u for u in us if interp(old20.get(IDEAL, []), u)]
    idl = np.array([interp(old20.get(IDEAL, []), u) * u ** (-b_of_u(su, sb, u))
                    for u in idu]) / np.interp(np.log(np.array(idu)),
                                               np.log(us), mid)
    ok = abs(idl.min() - PUB_IDEAL[0]) < 1e-3 and abs(idl.max() - PUB_IDEAL[1]) < 1e-3
    r0.append(("ideal_C", ok, (float(idl.min()), float(idl.max())), PUB_IDEAL))
    print(f" 理想低通 C 范围 = [{idl.min():.4f}, {idl.max():.4f}]  "
          f"已发布 [{PUB_IDEAL[0]:.4f}, {PUB_IDEAL[1]:.4f}]  {'OK' if ok else 'FAIL'}")
    prof20, _ = load_bt(["sweep_bdense.json", "sweep_blow.json"])
    ug20_all, ug20 = abt.grid(prof20, DENSE)
    P20 = abt.matrices(prof20, DENSE, ug20_all)
    cf20 = {}
    for fid in DENSE:
        bf = abt.bcurve(P20[fid], ug20_all)
        cf = abt.closed_form(ug20, bf)
        cf20[fid] = cf
        print(f" {fid} 闭式(20图) b_inf={cf['p'][0]:+.3f} A={cf['p'][1]:+.3f} "
              f"q={cf['p'][2]:.3f} rms={cf['rms']:.3f}")
    if not all(r[1] for r in r0):
        print("\n[R0 FAIL] 裁判未能复现已发布基线 —— 停止，不评估任何判据。")
        return
    print("\n[R0 PASS] 复现门通过。\n")
    rep["R0"] = dict(pass_=True, seg=[list(map(float, x)) for x in
                                      zip(sb, slo, shi)],
                     dev={f: dev20[f]["maxdev"] for f in NEWF},
                     rel={f: dev20[f]["maxrel"] for f in NEWF},
                     ideal=[float(idl.min()), float(idl.max())],
                     cf20={f: list(map(float, cf20[f]["p"])) for f in DENSE})

    # ---------------- N=120 文件是否齐 ----------------
    need = ["sweep_pshape_n120_A.json", "sweep_pshape_n120_BCV.json",
            "sweep_law_n120.json", "sweep_bdense_n120.json",
            "sweep_blow_n120.json"]
    miss = [f for f in need if not os.path.exists(os.path.join(OUT, f))]
    if miss:
        print("[缺少 N=120 数据] " + ", ".join(miss))
        print("（R0 已通过；补齐后重跑即可，判据不受影响）")
        if args.dump:
            json.dump(rep, open(args.dump, "w"), indent=1)
        return

    # ---------------- R1：逐点复现 ----------------
    print("=" * 100)
    print(" R1 逐点复现：z = (Λ̄_120 - Λ̄_20)/SE_20，判据 median|z|<=0.8 且 max|z|<=3.5")
    print("=" * 100)
    zs = []
    d20 = {}
    for fn in ["sweep_bdense.json", "sweep_blow.json"]:
        for r in json.load(open(os.path.join(OUT, fn)))["rows"]:
            d20[(r["meta"]["fid"], r["id"])] = r
    d120 = {}
    for fn in ["sweep_bdense_n120.json", "sweep_blow_n120.json"]:
        for r in json.load(open(os.path.join(OUT, fn)))["rows"]:
            d120[(r["meta"]["fid"], r["id"])] = r
    print(f" {'算子':<14}{'u':>7}{'Λ̄20':>10}{'SE20':>8}{'Λ̄120':>10}{'SE120':>8}{'z':>7}")
    for k in sorted(d20, key=lambda k: (k[0], -d20[k]["kunif"])):
        if k not in d120:
            continue
        a, b = d20[k], d120[k]
        pi20 = np.array(a.get("lam_per_img") or [], dtype=float)
        pi120 = np.array(b.get("lam_per_img") or [], dtype=float)
        if len(pi20) < 2 or len(pi120) < 2:
            continue
        se20 = float(pi20.std(ddof=1) / math.sqrt(len(pi20)))
        z = (b["lam_mean"] - a["lam_mean"]) / se20
        zs.append(z)
        print(f" {k[1]:<14}{1-a['kunif']:>7.3f}{a['lam_mean']:>10.0f}{se20:>8.0f}"
              f"{b['lam_mean']:>10.0f}{float(pi120.std(ddof=1)/math.sqrt(len(pi120))):>8.0f}"
              f"{z:>7.2f}")
    zs = np.array(zs)
    med, mx = float(np.median(np.abs(zs))), float(np.abs(zs).max())
    frac2 = float((np.abs(zs) > 2).mean())
    r1 = (med <= 0.8 and mx <= 3.5)
    print(f"\n n={len(zs)}  median|z|={med:.3f}  max|z|={mx:.3f}  "
          f"P(|z|>2)={frac2:.3f}（二项期望 0.046）  => R1 {'PASS' if r1 else 'FAIL'}")
    rep["R1"] = dict(pass_=bool(r1), n=len(zs), med=med, max=mx, frac2=frac2)

    # ---------------- R2/R3：闭式参数与精度增益 ----------------
    print("\n" + "=" * 100)
    print(" R2 闭式参数不动（|Δ|<=2*SE_20）与 R3 精度增益（SE<=SE_20/1.6）")
    print("=" * 100)
    prof120, _ = load_bt(["sweep_bdense_n120.json", "sweep_blow_n120.json"])
    ug120_all, ug120 = abt.grid(prof120, DENSE)
    same = (len(ug120) == len(ug20)) and all(
        abs(a - b) < 1e-9 for a, b in zip(ug120, ug20))
    print(f" N=120 共有 u {len(ug120_all)} 点、闭式子网格 {len(ug120)} 点；"
          f"与 N=20 {'一致' if same else '不一致!'}")
    assert same, "u 子网格不一致，闭式参数不可比 —— 停止"
    P120 = abt.matrices(prof120, DENSE, ug120_all)
    rng = np.random.default_rng(0)
    r2ok, r3ok = True, True
    rep["R2"], rep["R3"] = {}, {}
    for fid in DENSE:
        bf = abt.bcurve(P120[fid], ug120_all)
        cf120 = abt.closed_form(ug120, bf)
        se20, n20 = boot_se(P20[fid], ug20_all, ug20, args.nboot, rng)
        se120, n120 = boot_se(P120[fid], ug120_all, ug120, args.nboot, rng)
        p20, p120 = cf20[fid]["p"], cf120["p"]
        d = np.abs(np.array(p120) - np.array(p20))
        lim = 2.0 * np.array([se20["b_inf"], se20["A"], se20["q"]])
        oks = d <= lim
        gains = {k: se20[k] / max(se120[k], 1e-12) for k in se20}
        r2ok = r2ok and bool(oks.all())
        r3ok = r3ok and all(g >= 1.6 for g in
                            [gains["b_inf"], gains["A"], gains["q"]])
        rep["R2"][fid] = dict(p20=list(map(float, p20)), p120=list(map(float, p120)),
                              se20=se20, se120=se120,
                              delta=list(map(float, d)), lim=list(map(float, lim)),
                              pass_=bool(oks.all()))
        rep["R3"][fid] = dict(gains={k: float(v) for k, v in gains.items()},
                              pass_=bool(all(g >= 1.6 for g in
                                             [gains["b_inf"], gains["A"], gains["q"]])))
        print(f" {fid:<5} 20图 b_inf={p20[0]:+.3f} A={p20[1]:+.3f} q={p20[2]:.3f} | "
              f"120图 b_inf={p120[0]:+.3f} A={p120[1]:+.3f} q={p120[2]:.3f}")
        print(f"       Δ=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})  2SE_20=("
              f"{lim[0]:.3f},{lim[1]:.3f},{lim[2]:.3f})  "
              f"{'OK' if oks.all() else 'FAIL'}   rms20={cf20[fid]['rms']:.3f}"
              f" rms120={cf120['rms']:.3f}")
        print(f"       SE 增益 b_inf {gains['b_inf']:.2f}x  A {gains['A']:.2f}x  "
              f"q {gains['q']:.2f}x（理想 2.45x，判据 >=1.6x）  "
              f"{'OK' if rep['R3'][fid]['pass_'] else 'FAIL'}")
    rep["R2"]["pass_"] = bool(r2ok)
    rep["R3"]["pass_"] = bool(r3ok)
    print(f"\n R2 {'PASS' if r2ok else 'FAIL'}   R3 {'PASS' if r3ok else 'FAIL'}")
    if not r2ok:
        print(" => 已发布的闭式参数不被 120 图支持，论文数字必须按 120 图更新。")

    # ---------------- R4：样本外检验 ----------------
    print("\n" + "=" * 100)
    print(" R4 样本外检验：5 个新族在 N=120 自己的 7 模糊带上的最大偏离比（<=1.10）")
    print("=" * 100)
    ps = {}
    for fn in ["sweep_pshape_n120_A.json", "sweep_pshape_n120_BCV.json"]:
        _, o = load_ps(fn)
        for k, v in o.items():
            ps.setdefault(k, []).extend(v)
    for k in ps:
        ps[k].sort(key=lambda c: -c[0])
    su1, sb1, slo1, shi1 = segments(ps)
    lo1, hi1, mid1, us1 = band(ps, su1, sb1, U)
    _, new120 = load_law("sweep_law_n120.json")
    dev120 = new_dev(new120, su1, sb1, lo1, hi1, mid1, us1)
    r4ok = True
    rep["R4"] = {}
    for f in NEWF:
        g = dev120[f]["maxdev"] if dev120[f] else float("nan")
        r = dev120[f]["maxrel"] if dev120[f] else float("nan")
        ok = (g == g) and g <= 1.10
        r4ok = r4ok and ok
        rep["R4"][f] = dict(maxdev=float(g), maxrel=float(r), pass_=bool(ok))
        print(f" {f:<4} 出带比 = {g:.4f}（20图 {PUB_NEWDEV[f]:.4f}）  "
              f"对均值 = {r:.4f}（20图 {PUB_NEWREL[f]:.4f}）  {'OK' if ok else 'FAIL'}")
    rep["R4"]["pass_"] = bool(r4ok)
    print(f" R4 {'PASS' if r4ok else 'FAIL'}")

    # ---------------- R5：符号稳定性 ----------------
    print("\n" + "=" * 100)
    print(" R5 符号稳定性：mean Λ<=0 的 (族,算子) 占比，120 图必须 <= 20 图")
    print("=" * 100)
    def negfrac(files):
        n = k = 0
        for fn in files:
            for r in json.load(open(os.path.join(OUT, fn)))["rows"]:
                n += 1; k += (r["lam_mean"] <= 0)
        return k, n
    k20, n20 = negfrac(["sweep_bdense.json", "sweep_blow.json"])
    k120, n120 = negfrac(["sweep_bdense_n120.json", "sweep_blow_n120.json"])
    f20, f120 = k20 / n20, k120 / n120
    r5 = f120 <= f20
    rep["R5"] = dict(pass_=bool(r5), f20=f20, f120=f120, k20=k20, n20=n20,
                     k120=k120, n120=n120)
    print(f" 20图 {k20}/{n20} = {f20:.3f}   120图 {k120}/{n120} = {f120:.3f}  "
          f"=> R5 {'PASS' if r5 else 'FAIL'}")

    # ---------------- R6：诊断 ----------------
    print("\n" + "=" * 100)
    print(" R6 诊断（无阈值）")
    print("=" * 100)
    idl120 = [interp(ps.get(IDEAL, []), u) for u in us1]
    idl120 = [v * u ** (-b_of_u(su1, sb1, u)) for u, v in zip(us1, idl120) if v]
    idl120 = np.array(idl120) / float(np.interp(np.log(0.3), np.log(us1), mid1))
    print(f" N=120 理想低通 C 范围 [{idl120.min():.4f}, {idl120.max():.4f}]"
          f"（20图 [{PUB_IDEAL[0]:.4f}, {PUB_IDEAL[1]:.4f}]）")
    sp = []
    # 匹配 u 的跨族 max/min（复现图 1 的口径：同一 u 桶内跨族取 max/min）
    buckets = []
    for fam, th in SMOOTH_OLD:
        for u, lm in ps.get((fam, th), []):
            for b in buckets:
                if abs(b[0] - u) < 0.03:
                    b[1][(fam, th)] = lm
                    break
            else:
                buckets.append((u, {(fam, th): lm}))
    for uc, m in sorted(buckets):
        if len(m) >= 4:
            vs = list(m.values())
            sp.append((uc, max(vs) / min(vs), len(m)))
    print(f" 匹配 u 的跨族 max/min（N=120，{len(sp)} 个 u 桶）：")
    for uc, r, n in sp:
        print(f"   u={uc:.3f}  n={n}  max/min={r:.3f}")
    rep["R6"] = dict(ideal120=[float(idl120.min()), float(idl120.max())],
                     spread=[[float(a), float(b), int(c)] for a, b, c in sp])

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 100)
    print(" 汇总")
    print("=" * 100)
    print(f" R1 逐点复现        {'PASS' if r1 else 'FAIL'}")
    print(f" R2 闭式参数不动    {'PASS' if r2ok else 'FAIL'}")
    print(f" R3 精度增益        {'PASS' if r3ok else 'FAIL'}")
    print(f" R4 样本外检验存活  {'PASS' if r4ok else 'FAIL'}")
    print(f" R5 符号稳定性      {'PASS' if r5 else 'FAIL'}")
    rep["summary"] = dict(R1=bool(r1), R2=bool(r2ok), R3=bool(r3ok),
                          R4=bool(r4ok), R5=bool(r5))
    rep["cfg"] = dict(nboot=args.nboot, cmds=CMDS,
                      ug20=[float(x) for x in ug20],
                      ug120=[float(x) for x in ug120])
    if args.dump:
        json.dump(rep, open(args.dump, "w"), indent=1)
        print(f"\n[saved] {args.dump}")


if __name__ == "__main__":
    main()
