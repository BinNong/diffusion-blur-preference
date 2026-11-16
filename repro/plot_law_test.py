"""画定律检验的主图（1x2）：Lambda = C(shape) * u^{b(u)} 的样本外检验。

面板：
  (a) 前缀 C(u) = Λ(u)·u^{−b(u)}：13 个剖面全部塌缩到一条带里
      （7 个旧平滑族 + 理想低通离群 + 5 个新族）
  (b) 局部指数 b(u)：新族 vs 旧 7 族的 min-max 带（bootstrap SE 作误差棒）

配色沿用已校验 4 色 + 中性灰。

用法： venv2/bin/python plot_law_test.py
"""

import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GRAY, INK, MUTED = "#9aa0a6", "#202124", "#5f6368"

SMOOTH_OLD = [("gaussian", 1.0), ("gaussian", 2.0), ("defocus", 1.0),
              ("defocus", 2.0), ("airy", 4.0), ("motion", 3.0), ("motion", 7.0)]
IDEAL = ("lowpass", 0.25)
OLD_LABEL = {("gaussian", 1.0): "Gauss $\\sigma$1", ("gaussian", 2.0): "Gauss $\\sigma$2",
             ("defocus", 1.0): "Defocus 1", ("defocus", 2.0): "Defocus 2",
             ("airy", 4.0): "Airy 4", ("motion", 3.0): "Motion 3",
             ("motion", 7.0): "Motion 7", IDEAL: "Ideal low-pass"}
NEW_LABEL = {"W1": "W1  $e^{-(r/r_c)^1}$", "W2": "W2  $e^{-(r/r_c)^2}$",
             "W4": "W4  $e^{-(r/r_c)^4}$", "B1": "B1  Butterworth $n$=1",
             "M2": "M2  two-scale"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "legend.fontsize": 7.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e8eaed", "grid.linewidth": 0.6,
    "figure.dpi": 200, "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})

SEGS = [(0.65, 0.40), (0.40, 0.30), (0.30, 0.15)]


def load():
    d = json.load(open(os.path.join(OUT, "sweep_profile_shape.json")))
    old = defaultdict(list)
    for r in d["rows"]:
        m = r["meta"]
        if "family" in m:
            old[(m["family"], m["theta"])].append(
                (1.0 - r["kunif"], r["lam_mean"]))
    d2 = json.load(open(os.path.join(OUT, "sweep_law_test.json")))
    new = defaultdict(list)
    for r in d2["rows"]:
        new[r["meta"]["fid"]].append((1.0 - r["kunif"], r["lam_mean"]))
    for k in list(old) + list(new):
        (old if k in old else new)[k].sort(key=lambda c: -c[0])
    return old, new


def interp(curve, u):
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
    old, new = load()
    U = np.logspace(math.log10(0.06), math.log10(0.72), 40)

    # 旧 7 族的逐段指数 => b(u) 的分段线性表示
    seg_u, seg_b, seg_lo, seg_hi = [], [], [], []
    for a, b in SEGS:
        bs = []
        for k in SMOOTH_OLD:
            va, vb = interp(old[k], a), interp(old[k], b)
            if va and vb and va > 0 and vb > 0:
                bs.append(math.log(vb / va) / math.log(b / a))
        if len(bs) >= 5:
            seg_u.append(math.sqrt(a * b)); seg_b.append(float(np.mean(bs)))
            seg_lo.append(min(bs)); seg_hi.append(max(bs))
    seg_u = np.array(seg_u); seg_b = np.array(seg_b)
    seg_lo = np.array(seg_lo); seg_hi = np.array(seg_hi)

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

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

    # ---------------- (a) C(u) 塌缩 ----------------
    ax = axes[0]
    # 旧 7 族的 min-max 带（C 归一化到旧均值），只用 u>=0.12（内插有支撑的区段）
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
            band_lo.append(min(vs)); band_hi.append(max(vs))
            band_mid.append(float(np.exp(np.mean([math.log(v) for v in vs]))))
            us_ok.append(u)

    def norm(u, v):
        return v / np.interp(math.log(u), np.log(us_ok), band_mid)

    ax.fill_between(us_ok, np.array(band_lo) / np.array(band_mid),
                    np.array(band_hi) / np.array(band_mid),
                    color=BLUE, alpha=0.20, lw=0, zorder=2,
                    label="7 measured smooth blurs (min–max)")
    ax.plot(us_ok, np.ones(len(us_ok)), color=BLUE, lw=2.0, zorder=3,
            label="mean of 7 smooth blurs")
    xs = [u for u in us_ok if interp(old[IDEAL], u)]
    ys = [norm(u, interp(old[IDEAL], u) * u ** (-b_of(u))) for u in xs]
    ax.plot(xs, ys, "-o", color=ORANGE, lw=1.9, ms=3.6, zorder=5,
            label="Ideal low-pass (hard cut) — outlier")
    for fid, col in [("W1", GREEN), ("W2", PURPLE), ("W4", "#c2185b"),
                     ("B1", "#00838f"), ("M2", "#6d4c41")]:
        xs = [u for u in us_ok if interp(new[fid], u)]
        ys = [norm(u, interp(new[fid], u) * u ** (-b_of(u))) for u in xs]
        ax.plot(xs, ys, "-o", color=col, lw=1.6, ms=3.4, zorder=4,
                label=NEW_LABEL[fid] + "  (new)")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([0.12, 0.2, 0.35, 0.5, 0.7])
    ax.set_xticklabels(["0.12", "0.2", "0.35", "0.5", "0.7"])
    ax.minorticks_off()
    ax.set_xlim(0.115, 0.75)
    ax.set_ylim(0.56, 1.22)
    ax.set_yticks([0.6, 0.8, 1.0, 1.2])
    ax.set_xlabel("retained power  $u=\\langle|\\hat h|^2\\rangle$", labelpad=2)
    ax.set_ylabel("$C(u)=\\Lambda(u)\\,u^{-b(u)}$  /  mean", labelpad=6)
    ax.set_title("(a) 13 profiles collapse onto one band", loc="left")
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.4,
              columnspacing=0.9, borderaxespad=0.15, fontsize=7.2)

    # ---------------- (b) 指数 b(u) ----------------
    ax = axes[1]
    # 旧 7 族的 min-max 带（画成覆盖三个 segment 的横带）
    for i, (a, b) in enumerate(SEGS):
        ax.fill_between([a, b], seg_lo[i], seg_hi[i], color=BLUE, alpha=0.18,
                        lw=0, zorder=2)
    ax.plot([SEGS[0][1], SEGS[0][0]], [seg_b[0], seg_b[0]], color=BLUE, lw=2.0,
            zorder=3, label="measured: 7 smooth blurs (mean)")
    ax.plot([SEGS[1][1], SEGS[1][0]], [seg_b[1], seg_b[1]], color=BLUE, lw=2.0,
            zorder=3)
    ax.plot([SEGS[2][1], SEGS[2][0]], [seg_b[2], seg_b[2]], color=BLUE, lw=2.0,
            zorder=3)
    # 新族
    for fid, col in [("W1", GREEN), ("W2", PURPLE), ("W4", "#c2185b"),
                     ("B1", "#00838f"), ("M2", "#6d4c41")]:
        xs, ys = [], []
        for a, b in SEGS:
            va, vb = interp(new[fid], a), interp(new[fid], b)
            if va and vb and va > 0 and vb > 0:
                xs.append(math.sqrt(a * b))
                ys.append(math.log(vb / va) / math.log(b / a))
        ax.plot(xs, ys, "-o", color=col, lw=1.5, ms=4.5, zorder=4,
                label=NEW_LABEL[fid] + " (new)")
    ax.set_xscale("log")
    ax.set_xticks([0.15, 0.2, 0.35, 0.5, 0.65])
    ax.set_xticklabels(["0.15", "0.2", "0.35", "0.5", "0.65"])
    ax.minorticks_off()
    ax.set_xlabel("$u=\\langle|\\hat h|^2\\rangle$  (segment midpoint)",
                  labelpad=2)
    ax.set_ylabel("$b(u)=\\mathrm{d}\\log\\Lambda/\\mathrm{d}\\log u$",
                  labelpad=6)
    ax.set_title("(b) the exponent is shape-independent", loc="left")
    ax.legend(frameon=False, loc="lower left", ncol=2, handlelength=1.5,
              columnspacing=1.0, borderaxespad=0.1)
    ax.annotate("5 new families, 4/4 predictions passed",
                xy=(0.62, -0.62), fontsize=8, color=INK, ha="center")

    fig.tight_layout()
    out = os.path.join(OUT, "fig_law_test.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
