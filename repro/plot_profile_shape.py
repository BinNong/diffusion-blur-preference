"""画受控剖面形状实验的主图（1x3，论文用）。

配色：沿用上一轮已通过 dataviz 校验的 4 色（#2a78d6 / #eb6834 / #1baf7a / #4a3aa7，
  node scripts/validate_palette.js --pairs all 全项通过），另加中性灰给"单个剖面"这类
  非重点系列。不引入新色。

三个面板各回答一个问题：
  (a) 固定形状的 Λ(u) 曲线族：8 条曲线塌不塌缩？      -> 回答"形状进不进前缀"
  (b) 局部指数 b(u)：跨形状普适吗？比旧的 α 路径估计好在哪？ -> 回答"定律的形状"
  (c) 衰减放在哪个半径（B 组）：同 K、同 Σa 下 lp/bp/hp 差多少？ -> 回答"位置效应"

用法： venv2/bin/python plot_profile_shape.py
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

# 已校验的 4 色 + 中性灰
BLUE, ORANGE, GREEN, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GRAY, INK, MUTED = "#9aa0a6", "#202124", "#5f6368"

SMOOTH = {("gaussian", 1.0), ("gaussian", 2.0), ("defocus", 1.0),
          ("defocus", 2.0), ("airy", 4.0), ("motion", 3.0), ("motion", 7.0)}
LABEL = {("gaussian", 1.0): "Gaussian $\\sigma$=1",
         ("gaussian", 2.0): "Gaussian $\\sigma$=2",
         ("defocus", 1.0): "Defocus $r$=1",
         ("defocus", 2.0): "Defocus $r$=2",
         ("airy", 4.0): "Airy $r$=4",
         ("motion", 3.0): "Motion $L$=3",
         ("motion", 7.0): "Motion $L$=7",
         ("lowpass", 0.25): "Ideal low-pass (hard cut)"}

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#e8eaed", "grid.linewidth": 0.6,
    "figure.dpi": 200, "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})


def key_of(r):
    m = r["meta"]
    return (m["family"], m["theta"]) if "family" in m else \
        ("ideal-" + m.get("kind", "?"), 0.0)


def loglog_interp(curve, u):
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
    d = json.load(open(os.path.join(OUT, "sweep_profile_shape.json")))
    rows = d["rows"]
    nimg = d["cfg"]["nimg"]
    by_part = defaultdict(list)
    for r in rows:
        by_part[r["part"]].append(r)

    curves = defaultdict(list)
    for r in by_part["A"] + by_part["C"] + by_part["V"]:
        curves[key_of(r)].append((1.0 - r["kunif"], r["lam_mean"],
                                  r["lam_std"] / math.sqrt(nimg)))
    for k in curves:
        curves[k].sort(key=lambda c: -c[0])

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5))

    # ---------------- (a) 固定形状的 Λ(u) ----------------
    ax = axes[0]
    for k in sorted(curves, key=lambda k: LABEL.get(k, k[0])):
        c = curves[k]
        col = ORANGE if k == ("lowpass", 0.25) else GRAY
        lw = 1.5 if k == ("lowpass", 0.25) else 0.9
        ax.plot([p[0] for p in c], [p[1] for p in c], "-o", color=col,
                lw=lw, ms=2.6, zorder=2 if k == ("lowpass", 0.25) else 1)
    # 7 个平滑核的均值带
    us = np.logspace(math.log10(0.045), math.log10(0.735), 60)
    lo, hi, mid = [], [], []
    for u in us:
        vs = [loglog_interp([(p[0], p[1]) for p in curves[k]], u)
              for k in SMOOTH if k in curves]
        vs = [v for v in vs if v]
        if len(vs) >= 5:
            mid.append(np.mean(vs)); lo.append(min(vs)); hi.append(max(vs))
        else:
            mid.append(np.nan); lo.append(np.nan); hi.append(np.nan)
    ax.fill_between(us, lo, hi, color=BLUE, alpha=0.16, lw=0, zorder=3,
                    label="7 smooth-blur profiles (min–max)")
    ax.plot(us, mid, color=BLUE, lw=2.2, zorder=4,
            label="mean of 7 smooth blurs")
    ax.plot([p[0] for p in curves[("lowpass", 0.25)]],
            [p[1] for p in curves[("lowpass", 0.25)]], "-o", color=ORANGE,
            lw=1.6, ms=3.2, zorder=5, label=LABEL[("lowpass", 0.25)])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.4])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.4"])
    ax.set_yticks([2e3, 4e3, 1e4, 2e4, 4e4])
    ax.set_yticklabels(["2k", "4k", "10k", "20k", "40k"])
    ax.minorticks_off()
    ax.set_xlabel("retained power  $u=\\langle|\\hat h|^2\\rangle$",
                  labelpad=2)
    ax.set_ylabel("$\\Lambda$  (nats)", labelpad=6)
    ax.set_title("(a) $\\Lambda(u)$ at fixed profile shape", loc="left")
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)
    ax.annotate("individual profiles\n(7, nearly coincident)",
                xy=(0.30, 3.0e3), fontsize=7.5, color=MUTED, ha="center")

    # ---------------- (b) 局部指数 b(u) ----------------
    ax = axes[1]
    segs = [(0.73, 0.65), (0.65, 0.40), (0.40, 0.30), (0.30, 0.15),
            (0.15, 0.10)]
    ctr, bm, bmin, bmax = [], [], [], []
    for a, b in segs:
        vs = []
        for k, c in curves.items():
            va = loglog_interp([(p[0], p[1]) for p in c], a)
            vb = loglog_interp([(p[0], p[1]) for p in c], b)
            if va and vb and va > 0 and vb > 0:
                vs.append(math.log(vb / va) / math.log(b / a))
        if len(vs) >= 5:
            ctr.append(math.sqrt(a * b)); bm.append(np.mean(vs))
            bmin.append(min(vs)); bmax.append(max(vs))
    ax.fill_between(ctr, bmin, bmax, color=BLUE, alpha=0.16, lw=0)
    ax.plot(ctr, bm, "-o", color=BLUE, lw=2.2, ms=5, zorder=4,
            label="this work: fixed shape, 8 profiles")
    # 旧的 α 路径分段斜率（§3.9(h9)/(j)）作对照
    old_u = [0.55, 0.28, 0.135, 0.087]
    old_b = [-1.70, -1.43, -0.93, -0.76]
    old_e = [0.31, 0.31, 0.42, 0.42]
    ax.errorbar(old_u, old_b, yerr=old_e, fmt="s", color=ORANGE, ms=4.5,
                lw=1.0, capsize=2.0, alpha=0.95, zorder=3,
                label="$\\alpha$-path segment fits ($\\pm$1 s.e.)")
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.4, 0.6])
    ax.set_xticklabels(["0.1", "0.2", "0.4", "0.6"])
    ax.set_yticks([0.0, -0.5, -1.0, -1.5, -2.0])
    ax.minorticks_off()
    ax.set_xlabel("$u=\\langle|\\hat h|^2\\rangle$  (segment midpoint)",
                  labelpad=2)
    ax.set_ylabel("$b(u)=\\mathrm{d}\\log\\Lambda/\\mathrm{d}\\log u$",
                  labelpad=6)
    ax.set_title("(b) the exponent is universal across shapes", loc="left")
    ax.legend(frameon=False, loc="upper right", handlelength=1.6,
              borderaxespad=0.2)
    ax.annotate("flattening is real\n(measured at fixed shape)",
                xy=(0.055, -1.28), fontsize=7.5, color=MUTED, ha="left")

    # ---------------- (c) 位置效应 ----------------
    ax = axes[2]
    Ks = sorted({round(r["meta"]["K"], 4) for r in by_part["B"]})
    w = 0.26
    for i, (kind, col, lab) in enumerate([
            ("lp", BLUE, "low-pass: attenuate $r>r_c$ (outer)"),
            ("bp", GREEN, "band-stop: annulus"),
            ("hp", ORANGE, "high-pass: attenuate $r<r_c$ (inner, incl. DC)")]):
        vals, errs = [], []
        for K in Ks:
            sub = [r for r in by_part["B"]
                   if r["meta"]["kind"] == kind
                   and abs(r["meta"]["K"] - K) < 1e-9]
            if sub:
                vals.append(sub[0]["lam_mean"])
                errs.append(sub[0]["lam_std"] / math.sqrt(nimg))
            else:
                vals.append(np.nan); errs.append(np.nan)
        x = np.arange(len(Ks)) + (i - 1) * w
        ax.bar(x, vals, width=w * 0.9, color=col, alpha=0.9, label=lab,
               zorder=3, edgecolor="white", linewidth=0.8)
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=INK,
                    elinewidth=0.9, capsize=2, zorder=4)
    ax.axhline(0, color=INK, lw=1.0, zorder=5)
    ax.set_xticks(np.arange(len(Ks)))
    ax.set_xticklabels([f"{K:g}" for K in Ks])
    ax.set_yticks([-10000, -5000, 0, 5000, 10000])
    ax.set_yticklabels(["$-10$k", "$-5$k", "0", "5k", "10k"])
    ax.set_xlabel("target $\\mathrm{knif}$ (matched $\\sum_f a$)",
                  labelpad=2)
    ax.set_ylabel("$\\Lambda$  (nats)")
    ax.set_title("(c) where the attenuation sits", loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)
    ax.annotate("sign flip:\nsame $\\mathrm{knif}$, same $\\Lambda_{Gauss}$",
                xy=(1.42, -6200), fontsize=7.5, color=MUTED, ha="center")

    fig.tight_layout()
    out = os.path.join(OUT, "fig_profile_shape.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
