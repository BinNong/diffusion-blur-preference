"""图 2（论文 Figure 2）：定律 Λ = C(shape)·u^{b(u)} 的样本外检验

只读 results/sweep_profile_shape.json（8 个参考剖面）+ results/sweep_law_test.json
（5 个没测过的新族）。**零 GPU**。计算逻辑与已发布的 `plot_law_test.py` 逐字相同
（同样的 interp / 分段指数 / C(u) 归一化），只改样式、加闭式曲线与定量标注。

面板：
  (a) 前缀 C(u) = Λ(u)·u^{−b(u)}（归一到 7 个平滑核的几何均值 = 1）：
      13 个剖面几乎全塌进一条带，只有理想低通（硬截止）掉出带外。
  (b) 局部指数 b(u) = d log Λ / d log u：7 个平滑核的分段均值（min–max 带）
      + 5 个新族的实测点 + 闭式 b(u) = −0.70 − 6.4 u^{3.1}
      + **高斯参照指数 b_G(u) = −u/(1−u)**（论文式 (11)，Reference 0）。
      后两者之差在实测 u 范围上近似常数（见 gauss_gap），即高斯抓住了
      u 依赖的**形状**、错在**电平**。

产出：
  results/fig_law.json  —— 图上每个数（论文表格可追溯）
  results/fig_law.png / .pdf

用法： venv2/bin/python repro/plot_fig2.py
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_PROJ, "results")

H = W = 64
KS = 15
SWEEP_OLD = "sweep_profile_shape.json"
SWEEP_NEW = "sweep_law_test.json"

SMOOTH_OLD = [("gaussian", 1.0), ("gaussian", 2.0), ("defocus", 1.0),
              ("defocus", 2.0), ("airy", 4.0), ("motion", 3.0), ("motion", 7.0)]
IDEAL = ("lowpass", 0.25)
NEW_FIDS = ["W1", "W2", "W4", "B1", "M2"]
NEW_LABEL = {"W1": "W1  $e^{-(r/r_c)^1}$", "W2": "W2  $e^{-(r/r_c)^2}$",
             "W4": "W4  $e^{-(r/r_c)^4}$", "B1": "B1  Butterworth $n$=1",
             "M2": "M2  two-scale"}
SEGS = [(0.65, 0.40), (0.40, 0.30), (0.30, 0.15)]

# 闭式候选（§3.9(s)，19 点密网格、三族极差 ≤0.345）
B_INF, A_QF, Q_QF = -0.70, 6.4, 3.1


def b_closed(u):
    """实测局部指数的闭式（论文式 (5) 的右半）。"""
    return B_INF - A_QF * u ** Q_QF


def b_gauss(u):
    """高斯先验的闭式指数（论文式 (11)，Reference 0）：与 t 无关。"""
    return -u / (1.0 - u)

# Okabe–Ito 固定序分配：5 个新族 + 理想低通（离群）
COLOR = {"W1": "#0072B2", "W2": "#D55E00", "W4": "#009E73",
         "B1": "#CC79A7", "M2": "#56B4E9"}
COLOR_IDEAL = "#E69F00"
MARK = {"W1": "o", "W2": "s", "W4": "^", "B1": "D", "M2": "v"}
BAND = "#8C959D"


def load():
    """与 plot_law_test.py 的 load() 逐字相同。"""
    d = json.load(open(os.path.join(OUT, SWEEP_OLD)))
    old = defaultdict(list)
    for r in d["rows"]:
        m = r["meta"]
        if "family" in m:
            old[(m["family"], m["theta"])].append((1.0 - r["kunif"], r["lam_mean"]))
    d2 = json.load(open(os.path.join(OUT, SWEEP_NEW)))
    new = defaultdict(list)
    for r in d2["rows"]:
        new[r["meta"]["fid"]].append((1.0 - r["kunif"], r["lam_mean"]))
    for k in list(old) + list(new):
        (old if k in old else new)[k].sort(key=lambda c: -c[0])
    return old, new


def interp(curve, u):
    """与 plot_law_test.py 的 interp() 逐字相同（对数线性内插）。"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=SWEEP_OLD)
    ap.add_argument("--new", default=SWEEP_NEW)
    # 闭式参数：默认是 N=20 三族均值的取整值；N=120 用 --binf -0.73 --Aq 7.2 --q 3.4
    ap.add_argument("--binf", type=float, default=B_INF)
    ap.add_argument("--Aq", type=float, default=A_QF)
    ap.add_argument("--q", type=float, default=Q_QF)
    args = ap.parse_args()
    globals()["B_INF"], globals()["A_QF"], globals()["Q_QF"] = args.binf, args.Aq, args.q
    globals()["SWEEP_OLD"], globals()["SWEEP_NEW"] = args.old, args.new
    old, new = load()
    U = np.logspace(math.log10(0.06), math.log10(0.72), 40)

    # ---- 旧 7 族的逐段指数 => b(u) 的分段线性表示（与已发布分析相同）----
    seg_u, seg_b, seg_lo, seg_hi = [], [], [], []
    for a, b in SEGS:
        bs = []
        for k in SMOOTH_OLD:
            va, vb = interp(old[k], a), interp(old[k], b)
            if va and vb and va > 0 and vb > 0:
                bs.append(math.log(vb / va) / math.log(b / a))
        if len(bs) >= 5:
            seg_u.append(math.sqrt(a * b))
            seg_b.append(float(np.mean(bs)))
            seg_lo.append(min(bs))
            seg_hi.append(max(bs))
    seg_u = np.array(seg_u)
    seg_b = np.array(seg_b)
    seg_lo = np.array(seg_lo)
    seg_hi = np.array(seg_hi)

    def b_of(u):
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

    # ---- 新族实测的逐段指数 ----
    new_seg = {}
    for fid in NEW_FIDS:
        pts = []
        for a, b in SEGS:
            va, vb = interp(new[fid], a), interp(new[fid], b)
            if va and vb and va > 0 and vb > 0:
                pts.append((math.sqrt(a * b), math.log(vb / va) / math.log(b / a)))
        new_seg[fid] = pts

    print("===== 逐段指数 b(u) =====")
    print(f"  {'段':>14} {'7族均值':>9} {'min':>7} {'max':>7} | "
          + " ".join(f"{f:>7}" for f in NEW_FIDS))
    for i, (a, b) in enumerate(SEGS):
        row = " ".join(
            f"{dict(new_seg[f]).get(seg_u[i], float('nan')):>7.3f}" for f in NEW_FIDS)
        print(f"  [{a:.2f},{b:.2f}] {seg_b[i]:>9.3f} {seg_lo[i]:>7.3f} "
              f"{seg_hi[i]:>7.3f} | {row}")
    print("  闭式对照 b(u) = %.2f %+.3f u^%.2f：" % (B_INF, -A_QF, Q_QF))
    for i in range(len(seg_u)):
        cf = B_INF - A_QF * seg_u[i] ** Q_QF
        print(f"    u={seg_u[i]:.4f}：闭式 {cf:>7.3f}  实测 {seg_b[i]:>7.3f}  "
              f"差 {cf - seg_b[i]:+.3f}")

    # ---- C(u) 塌缩 ----
    band_lo, band_hi, band_mid, us_ok = [], [], [], []
    for u in U:
        if u < 0.12:
            continue
        vs = []
        for k in SMOOTH_OLD:
            v = interp(old[k], u)
            if v:
                vs.append(v * u ** (-b_of(u)))
        if len(vs) >= 5:
            band_lo.append(min(vs))
            band_hi.append(max(vs))
            band_mid.append(float(np.exp(np.mean([math.log(v) for v in vs]))))
            us_ok.append(u)
    band_mid = np.array(band_mid)

    def norm(u, v):
        return v / np.interp(math.log(u), np.log(us_ok), band_mid)

    dev = {}
    for fid in NEW_FIDS:
        xs = [u for u in us_ok if interp(new[fid], u)]
        ys = [norm(u, interp(new[fid], u) * u ** (-b_of(u))) for u in xs]
        dev[fid] = (min(ys), max(ys))
    xs_i = [u for u in us_ok if interp(old[IDEAL], u)]
    ys_i = [norm(u, interp(old[IDEAL], u) * u ** (-b_of(u))) for u in xs_i]
    print("\n===== 前缀 C(u)（归一到 7 个平滑核几何均值 = 1）====")
    for fid in NEW_FIDS:
        print(f"  {fid}: C ∈ [{dev[fid][0]:.3f}, {dev[fid][1]:.3f}]，"
              f"最大偏离均值 {max(abs(dev[fid][0] - 1), abs(dev[fid][1] - 1)):.3f}")
    print(f"  理想低通: C ∈ [{min(ys_i):.3f}, {max(ys_i):.3f}]（唯一掉出带外的）")
    print(f"  7 个平滑核的带: [{min(np.array(band_lo) / band_mid):.3f}, "
          f"{max(np.array(band_hi) / band_mid):.3f}]")

    # 高斯参照指数 vs 闭式指数：论文 §6 Reference 0 的定量陈述
    # （b_meas - b_Gauss 在实测 u 范围上近似常数 => 高斯抓住形状、错在电平）
    ug = np.logspace(math.log10(0.135), math.log10(0.72), 40)
    gap = [b_closed(u) - b_gauss(u) for u in ug]
    gseg = [float(seg_b[i] - b_gauss(seg_u[i])) for i in range(len(SEGS))]
    print("\n===== 闭式指数 vs 高斯参照指数 b_G(u)=-u/(1-u) =====")
    print(f"  b_meas 本身跨度: {b_closed(0.135):.3f} -> {b_closed(0.72):.3f}"
          f"  (跨度 {b_closed(0.72) - b_closed(0.135):.3f})")
    print(f"  b_meas - b_Gauss: [{min(gap):.3f}, {max(gap):.3f}]，"
          f"跨度 {max(gap) - min(gap):.3f}，均值 {sum(gap) / len(gap):.3f}")
    print("  实测段均值减高斯: " + "  ".join(f"{g:+.3f}" for g in gseg))

    # ------------------------------------------------------------------ 落盘
    fig = {
        "cfg": {"old": args.old, "new": args.new, "segs": SEGS,
                "closed_form_args": {"b_inf": args.binf, "A": args.Aq, "q": args.q},
                "closed_form": {"b_inf": B_INF, "A": A_QF, "q": Q_QF},
                "gauss_exponent": "-u/(1-u)",
                "note": "计算逻辑与已发布的 plot_law_test.py 相同，仅样式与标注不同"},
        "gauss_gap": {"u": [float(x) for x in ug],
                      "b_closed": [float(b_closed(x)) for x in ug],
                      "b_gauss": [float(b_gauss(x)) for x in ug],
                      "gap": [float(g) for g in gap],
                      "gap_min": float(min(gap)), "gap_max": float(max(gap)),
                      "gap_mean": float(sum(gap) / len(gap)),
                      "b_closed_span": float(b_closed(0.72) - b_closed(0.135)),
                      "seg_gap": gseg},
        "segments": [{"u_lo": a, "u_hi": b, "u_mid": float(seg_u[i]),
                      "b_mean": float(seg_b[i]), "b_min": float(seg_lo[i]),
                      "b_max": float(seg_hi[i]),
                      "b_closed_form": float(b_closed(seg_u[i])),
                      "b_gauss": float(b_gauss(seg_u[i])),
                      "b_new": {f: dict(new_seg[f]).get(float(seg_u[i]))
                                for f in NEW_FIDS}}
                     for i, (a, b) in enumerate(SEGS)],
        "collapse": {
            "u": [float(x) for x in us_ok],
            "band_lo": [float(x) for x in np.array(band_lo) / band_mid],
            "band_hi": [float(x) for x in np.array(band_hi) / band_mid],
            "band_mid": [1.0] * len(us_ok),
            "new": {fid: {"u": [float(u) for u in us_ok if interp(new[fid], u)],
                          "C": [float(norm(u, interp(new[fid], u) * u ** (-b_of(u))))
                                for u in us_ok if interp(new[fid], u)]}
                    for fid in NEW_FIDS},
            "ideal": {"u": [float(x) for x in xs_i],
                      "C": [float(y) for y in ys_i]},
        },
    }
    with open(os.path.join(OUT, "fig_law.json"), "w") as fh:
        json.dump(fig, fh, indent=1)
    print("\n[写出] results/fig_law.json")

    # ------------------------------------------------------------------ 画图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.0,
        "axes.linewidth": 0.7, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0, "lines.linewidth": 1.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    })
    figp, (axa, axb) = plt.subplots(1, 2, figsize=(7.16, 3.35),
                                    gridspec_kw={"wspace": 0.30})

    def kill_minor_labels(ax):
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # ---- (a) C(u) 塌缩 ----
    axa.fill_between(us_ok, np.array(band_lo) / band_mid,
                     np.array(band_hi) / band_mid,
                     color=BAND, alpha=0.28, lw=0, zorder=2,
                     label="7 smooth blurs (min–max)")
    axa.plot(us_ok, np.ones(len(us_ok)), color="#4A5057", lw=1.8, zorder=3,
             label="mean of 7 smooth blurs")
    axa.plot(xs_i, ys_i, "-o", color=COLOR_IDEAL, lw=1.7, ms=4.2, mec="white",
             mew=0.7, zorder=6, label="Ideal low-pass (outlier)")
    for fid in NEW_FIDS:
        xs = [u for u in us_ok if interp(new[fid], u)]
        ys = [norm(u, interp(new[fid], u) * u ** (-b_of(u))) for u in xs]
        axa.plot(xs, ys, "-", color=COLOR[fid], lw=1.3, zorder=4)
        axa.plot(xs, ys, MARK[fid], color=COLOR[fid], ms=4.2, mec="white",
                 mew=0.7, zorder=5,
                 label=NEW_LABEL[fid] + (" (control)" if fid == "W2" else " (new)"))
    axa.axhline(1.0, color="#8A8F94", lw=0.7, ls=(0, (4, 2.5)), zorder=1)
    axa.set_xscale("log")
    axa.set_xlim(0.115, 0.78)
    axa.set_ylim(0.54, 1.30)
    axa.set_xticks([0.12, 0.2, 0.35, 0.5, 0.7])
    axa.set_xticklabels(["0.12", "0.2", "0.35", "0.5", "0.7"])
    axa.set_yticks([0.6, 0.8, 1.0, 1.2])
    axa.set_yticklabels(["0.6", "0.8", "1.0", "1.2"])
    axa.set_xlabel("retained spectral power  $u=\\langle|\\hat h|^2\\rangle$")
    axa.set_ylabel("$C(u)=\\Lambda(u)\\,u^{-b(u)}$  (7 smooth blurs $=1$)")
    axa.set_title("(a)  the prefix collapses; only the hard cut escapes", pad=3.5)
    axa.grid(True, which="major", axis="y", color="#E6E8EA", lw=0.6, zorder=0)
    axa.set_axisbelow(True)
    axa.text(0.03, 0.96, "4 new shape classes + 1 internal control\n"
             "all lie inside the 7-blur band",
             transform=axa.transAxes, fontsize=6.8, color="#3A3F44",
             ha="left", va="top", linespacing=1.3)

    # ---- (b) 指数 b(u) ----
    ucf = np.logspace(math.log10(0.13), math.log10(0.72), 60)
    axb.plot(ucf, b_gauss(ucf), "-", color="#7B5EA7", lw=1.4, zorder=4,
             label="Gaussian reference $b_{\\rm G}(u)=-u/(1-u)$")
    axb.plot(ucf, b_closed(ucf), "--", color="#B03A2E", lw=1.5,
             zorder=5, label="closed form $b(u)=b_\\infty+A\\,u^q$")
    for i, (a, b) in enumerate(SEGS):
        axb.fill_between([a, b], seg_lo[i], seg_hi[i], color=BAND, alpha=0.30,
                         lw=0, zorder=2)
    axb.plot([SEGS[0][1], SEGS[0][0]], [seg_b[0], seg_b[0]], color="#4A5057",
             lw=2.0, zorder=4, label="7 smooth blurs (mean)")
    for i in (1, 2):
        axb.plot([SEGS[i][1], SEGS[i][0]], [seg_b[i], seg_b[i]],
                 color="#4A5057", lw=2.0, zorder=4)
    for fid in NEW_FIDS:
        xs = [p[0] for p in new_seg[fid]]
        ys = [p[1] for p in new_seg[fid]]
        axb.plot(xs, ys, "-", color=COLOR[fid], lw=1.2, zorder=6)
        axb.plot(xs, ys, MARK[fid], color=COLOR[fid], ms=4.6, mec="white",
                 mew=0.7, zorder=7, label=NEW_LABEL[fid] + " (new)")
    axb.set_xscale("log")
    axb.set_xlim(0.135, 0.72)
    axb.set_ylim(-3.45, 0.18)
    axb.set_xticks([0.15, 0.2, 0.35, 0.5, 0.65])
    axb.set_xticklabels(["0.15", "0.2", "0.35", "0.5", "0.65"])
    axb.set_yticks([-3.0, -2.0, -1.0, 0.0])
    axb.set_yticklabels(["$-3$", "$-2$", "$-1$", "$0$"])
    axb.set_xlabel("$u=\\langle|\\hat h|^2\\rangle$  (segment midpoint)")
    axb.set_ylabel("$b(u)=\\mathrm{d}\\log\\Lambda/\\mathrm{d}\\log u$")
    axb.set_title("(b)  the exponent is shape-independent, and not Gaussian",
                  pad=3.5)
    axb.grid(True, which="major", axis="y", color="#E6E8EA", lw=0.6, zorder=0)
    axb.set_axisbelow(True)
    axb.text(0.975, 0.055, "4 new shape classes + 1 control: 4/4 predictions passed\n"
             "measured $-$ Gaussian exponent $=-0.47\\pm0.08$ over this range",
             transform=axb.transAxes, fontsize=6.8, color="#3A3F44",
             ha="right", va="bottom", linespacing=1.3)
    kill_minor_labels(axa)
    kill_minor_labels(axb)

    # 图例放到整张图下方横排：两个面板的内容区都要让位，8 项竖排会溢出
    h1, l1 = axb.get_legend_handles_labels()
    h2, l2 = axa.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in list(zip(h1, l1)) + list(zip(h2, l2)):
        if l not in seen:
            seen.add(l)
            H.append(h)
            L.append(l)
    figp.subplots_adjust(bottom=0.34)
    figp.legend(H, L, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
                frameon=False, fontsize=6.9, handlelength=2.0,
                columnspacing=1.5, labelspacing=0.4)
    figp.align_ylabels([axa, axb])
    for ext in ("png", "pdf"):
        figp.savefig(os.path.join(OUT, f"fig_law.{ext}"), dpi=300,
                     bbox_inches="tight", pad_inches=0.02)
    print("[写出] results/fig_law.png / .pdf")


if __name__ == "__main__":
    main()
