"""b(u) 是否随先验（模型）变？图：2x3。读 analyze_model.py --dump 的 JSON。

(a) b(u)：三格（A=ffq先验+ffq图 / B=afq先验+afq图 / C=ffq先验+afq图），
    三族均值 + 极差带，附高斯极限 b_Gauss=-u/(1-u)。
(b) b_inf：三格 + 2SE 误差棒。
(c) 每个 u 上的**先验效应** Δb(u)=b_B−b_C（固定图像，配对）+ 2SE 带；
    H0（先验无关）要求整条贴在 0 上。
(d) M5 对照：三种效应（先验 B−C / 图像 C−A / 总 B−A）在 b_inf 上的量级对比。
(e) 两个图池的径向功率谱（内→外 8 环）——证明两个先验的谱确实不同。
(f) 文字面板：预官判据与实测。

用法： venv2/bin/python analyze_model.py --dump fig_model.json
      venv2/bin/python plot_model.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

BLUE, ORANGE, GREEN, PURPLE, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#8a8f98"
INK = "#1c1f24"
CA, CB, CC = "#8a8f98", "#2a78d6", "#eb6834"      # A 灰 / B 蓝 / C 橙
NAMES = ["A(ffq+ffq)", "B(afq+afq)", "C(ffq+afq)"]
LBL = {"A(ffq+ffq)": "A: ffhq prior + ffhq images",
       "B(afq+afq)": "B: afhqv2 prior + afhqv2 images",
       "C(ffq+afq)": "C: ffhq prior + afhqv2 images"}
COL = {"A(ffq+ffq)": CA, "B(afq+afq)": CB, "C(ffq+afq)": CC}


def fam(cell, fid, key):
    return np.array(cell["fam"][fid][key], dtype=float)


def main():
    p = os.path.join(OUT, "fig_model.json")
    if not os.path.exists(p):
        print("[缺少]", p, "——先跑 analyze_model.py --dump fig_model.json")
        return
    d = json.load(open(p))
    ug = np.array(d["ug"])
    fams = d["dense"]
    cells = d["cells"]

    def band(nm, key):
        M = np.array([fam(cells[nm], f, key) for f in fams], dtype=float)
        return M.mean(0), M.min(0), M.max(0)

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.4))
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---------------- (a) b(u) 三格 ----------------
    ax = axes[0, 0]
    uu = np.linspace(ug.min(), ug.max(), 300)
    ax.plot(uu, -uu / (1.0 - uu), color=GRAY, lw=1.2, ls=":", zorder=1,
            label=r"Gaussian limit $-u/(1-u)$")
    for nm in NAMES:
        m, lo, hi = band(nm, "b")
        c = COL[nm]
        ax.fill_between(ug, lo, hi, color=c, alpha=0.12, zorder=2)
        ax.plot(ug, m, "o-", color=c, ms=4.0, lw=1.4, zorder=3, label=LBL[nm])
    ax.set_xlabel(r"retained power $u$")
    ax.set_ylabel(r"$b(u)$")
    ax.set_title("(a) $b(u)$ for three cells\n"
                 "(band = spread over 3 shape families)",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.0, loc="lower right", framealpha=0.93)

    # ---------------- (b) b_inf ----------------
    ax = axes[0, 1]
    xs = np.arange(len(NAMES))
    FCOL = [BLUE, ORANGE, GREEN]
    for k, fid in enumerate(fams):
        v = [cells[nm]["fam"][fid]["p"][0] for nm in NAMES]
        ax.plot(xs, v, "o-", color=FCOL[k], ms=6, lw=1.2, zorder=3,
                label=f"{fid} (family)")
    mv = np.array([[cells[nm]["fam"][f]["p"][0] for f in fams]
                   for nm in NAMES])
    ax.errorbar(xs, mv.mean(1), yerr=[mv.mean(1) - mv.min(1),
                                      mv.max(1) - mv.mean(1)],
                fmt="s", ms=9, color=INK, lw=1.8, capsize=6, zorder=4,
                label="mean $\\pm$ family spread")
    for i in range(len(NAMES)):
        ax.text(xs[i], mv.mean(1)[i] + 0.04, f"{mv.mean(1)[i]:.3f}",
                ha="center", fontsize=8.2, color=INK)
    ax.axhline(0.0, color=GRAY, lw=1.1, ls=":", zorder=1,
               label=r"Gaussian limit $b_\infty=0$")
    ax.set_xticks(xs)
    ax.set_xticklabels(["A\nffhq+ffhq", "B\nafhq+afhq", "C\nffhq+afhq"],
                       fontsize=8.4)
    ax.set_ylabel(r"$b_\infty$")
    ax.set_title(r"(b) $b_\infty$ per cell"
                 "\nB vs C isolates the prior (same images)",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.0, loc="best", framealpha=0.93)

    # ---------------- (c) 每个 u 的先验效应 ----------------
    ax = axes[0, 2]
    for k, fid in enumerate(fams):
        v = d["verdict"][fid]
        c = [BLUE, ORANGE, GREEN][k]
        est = np.array(v["curve"])
        se = np.array(v["curve_2se"])
        ax.fill_between(ug, est - se, est + se, color=c, alpha=0.13, zorder=2)
        ax.plot(ug, est, "o-", color=c, ms=3.8, lw=1.3, zorder=3, label=fid)
    ax.axhline(0.0, color=INK, lw=1.2, ls="--", zorder=1)
    ax.set_xlabel(r"retained power $u$")
    ax.set_ylabel(r"$\Delta b(u)=b_B(u)-b_C(u)$")
    ax.set_title("(c) prior effect at each $u$ (paired)\n"
                 r"$H_0$: prior-independent $\Rightarrow$ flat at 0",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.4, loc="best", framealpha=0.93)

    # ---------------- (d) M5 三种效应 ----------------
    ax = axes[1, 0]
    labs = ["prior\nB−C", "images\nC−A", "total\nB−A"]
    cols = [BLUE, ORANGE, PURPLE]
    xs = np.arange(3)
    for k, fid in enumerate(fams):
        v = d["verdict"][fid]
        c = [BLUE, ORANGE, GREEN][k]
        ax.plot(xs, [v["d_bc"], v["d_ca"], v["d_ba"]], "o-", color=c,
                ms=6, lw=1.3, zorder=3, label=fid)
    ax.axhline(0.0, color=INK, lw=1.2, ls="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labs, fontsize=8.4)
    ax.set_ylabel(r"$\Delta b_\infty$")
    ax.set_title("(d) M5 control: is the prior effect bigger\n"
                 "than \"any perturbation moves $b_\\infty$\"?",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.4, loc="best", framealpha=0.93)

    # ---------------- (e) 两个图池的径向谱 ----------------
    ax = axes[1, 1]
    rings = np.arange(1, 9)
    ffq = [12.49, 0.3693, 0.07892, 0.0298, 0.01248, 0.005556, 0.001469,
           0.0006247]
    afq = [10.14, 0.3103, 0.07491, 0.03115, 0.01326, 0.005913, 0.001896,
           0.0008082]
    ax.semilogy(rings, ffq, "o-", color=CA, ms=5, lw=1.4,
                label="ffhq pool (256 imgs)")
    ax.semilogy(rings, afq, "s-", color=CB, ms=5, lw=1.4,
                label="afhqv2 pool (256 imgs)")
    ax.semilogy(rings, np.array(afq) / np.array(ffq), "^:", color=PURPLE,
                ms=4.5, lw=1.2, label="ratio afhqv2/ffhq")
    ax.axhline(1.0, color=GRAY, lw=0.9, ls=":")
    ax.set_xticks(rings)
    ax.set_xlabel("radial ring (1 = innermost)")
    ax.set_ylabel("mean power")
    ax.set_title("(e) the two priors really differ\n"
                 "afhqv2: weaker low band, +29% outermost",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.2, loc="lower left", framealpha=0.93)

    # ---------------- (f) 文字面板 ----------------
    ax = axes[1, 2]
    ax.axis("off")
    lines = ["Pre-registered criteria (group 9, 2026-10-19)",
             "M1  closed-form rms <= 2x baseline (both cells)",
             "M2  |db_inf| prior effect > 2 SE  (expected)",
             "M3  |dq|    prior effect > 2 SE",
             "M4  slope of D_prior(u) > 2 SE",
             "M5  report image effect C-A alongside",
             "M6  sign flip of db(u) low vs high u",
             "M7  report 2 SE detection floor", "",
             "Measured (family-averaged):"]
    for nm in NAMES:
        mv = np.mean([cells[nm]["fam"][f]["p"][0] for f in fams])
        rms = np.mean([cells[nm]["fam"][f]["rms"] for f in fams])
        short = nm.split("(")[0]
        lines.append(f"cell {short}: b_inf={mv:+.3f}  rms={rms:.3f}")
    for fid in fams:
        v = d["verdict"][fid]
        lines.append(
            f"{fid}: dB={v['d_bc']:+.3f} 2SE={2*v['se_bc']:.3f} "
            f"dC-A={v['d_ca']:+.3f}")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.0,
            family="monospace", color=INK, transform=ax.transAxes)

    fig.suptitle(r"Does the exponent $b(u)$ in $\Lambda=C\,u^{\,b(u)}$ "
                 "depend on the prior (model)?", fontsize=11.5, y=1.01)
    fig.tight_layout()
    o = os.path.join(OUT, "fig_model.png")
    fig.savefig(o, dpi=170, bbox_inches="tight")
    print("[saved]", o)


if __name__ == "__main__":
    main()
