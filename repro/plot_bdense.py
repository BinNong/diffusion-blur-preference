"""b(u) 密网格的图：1x2。(a) b(u) vs u + 闭式曲线 + 三族误差棒；
(b) logΛ vs log u（三条光滑曲线 + 旧分段值对照）。

用法： venv2/bin/python plot_bdense.py
"""

import importlib.util
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
sys.path.insert(0, os.path.join(PROJ, "code", "blind_deblur_diffusion"))

_spec = importlib.util.spec_from_file_location(
    "hd", os.path.join(PROJ, "repro", "hunt_descriptor.py"))
hd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hd)

BLUE, ORANGE, GREEN, PURPLE, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#8a8f98"
DENSE = ["W2", "W8", "TK50"]
COL = {"W2": BLUE, "W8": ORANGE, "TK50": GREEN}
LBL = {"W2": "W2 (Gaussian)", "W8": "W8 ($p$=8)",
       "TK50": "TK50 (Tukey)"}
DEG = 5


def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    prof = defaultdict(list)
    for rec in d["rows"]:
        prof[rec["meta"]["fid"]].append((1.0 - rec["kunif"], rec["lam_mean"],
                                         rec.get("lam_per_img")))
    for k in prof:
        prof[k].sort(key=lambda c: -c[0])
    return dict(prof)


def main():
    pd = load("sweep_bdense.json")
    for fid, rows in load("sweep_blow.json").items():
        pd.setdefault(fid, []).extend(rows)
    for k in pd:
        pd[k].sort(key=lambda c: -c[0])
    lam = {fid: {round(u, 4): v for u, v, _ in pd[fid]} for fid in DENSE}

    fits = {}
    for fid in DENSE:
        us = np.array(sorted(lam[fid]))
        ll = np.array([math.log(lam[fid][u]) for u in us])
        c = np.polyfit(np.log(us), ll, DEG)
        per = {round(u, 4): np.array(pi) for u, _, pi in pd[fid] if pi}
        rng = np.random.default_rng(0)
        boots = []
        P = np.array([per[round(u, 4)] for u in us])
        for _ in range(2000):
            idx = rng.integers(0, P.shape[1], P.shape[1])
            try:
                cb = np.polyfit(np.log(us), np.log(P[:, idx].mean(1)), DEG)
                boots.append(np.polyval(np.polyder(cb), np.log(us)))
            except Exception:
                pass
        fits[fid] = dict(us=us, c=c,
                         se=np.array(boots).std(0) if boots else None)

    ug = np.array([u for u in sorted(lam["W2"]) if 0.13 <= u <= 0.86])

    # 闭式拟合（形式 (i)，预官顺序里的第一个也是胜出者）
    from scipy.optimize import curve_fit
    f1 = lambda u, *p: p[0] + p[1] * u ** p[2]
    closed = {}
    for fid in DENSE:
        bv = np.array([np.polyval(np.polyder(fits[fid]["c"]), math.log(u))
                       for u in ug])
        popt, _ = curve_fit(f1, ug, bv, p0=(-0.7, -6.0, 3.0),
                            bounds=([-1e3, -50, 0.1], [1e3, 50, 12]),
                            maxfev=200000)
        closed[fid] = popt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for ax in axes:
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---------------- (a) b(u) ----------------
    ax = axes[0]
    uu = np.linspace(0.13, 0.87, 300)
    for fid in DENSE:
        b = np.array([np.polyval(np.polyder(fits[fid]["c"]), math.log(u))
                      for u in ug])
        se = None
        if fits[fid]["se"] is not None:
            se = np.array([fits[fid]["se"][int(np.argmin(np.abs(
                fits[fid]["us"] - u)))] for u in ug])
        ax.errorbar(ug, b, yerr=se, fmt="o", ms=4.5, color=COL[fid], lw=1.2,
                    capsize=2.2, zorder=3, label=LBL[fid])
    for fid, ls in [("W2", "-"), ("W8", "--"), ("TK50", ":")]:
        p = closed[fid]
        ax.plot(uu, f1(uu, *p), color=COL[fid], lw=1.3, ls=ls, alpha=0.75,
                zorder=2)
    ax.plot([], [], color=GRAY, lw=1.3,
            label=r"$b=b_\infty+A u^q$ (fit)")
    # 旧分段值（用密网格直接算的）
    for (u1, u2, bseg) in [(0.65, 0.40, -1.68), (0.40, 0.30, -1.02),
                           (0.30, 0.15, -0.761)]:
        ax.plot([u1, u2], [bseg, bseg], color=PURPLE, lw=2.4, alpha=0.5,
                zorder=1, solid_capstyle="butt")
    ax.plot([], [], color=PURPLE, lw=2.4, alpha=0.5,
            label="old 3-segment values\n(re-measured on dense grid)")
    ax.axhline(-2.0, color=GRAY, lw=0.9, ls=":", zorder=1)
    ax.text(0.60, -2.10, r"$b=-2$ (the refuted prediction)", fontsize=7.2,
            color=GRAY)
    ax.set_xlabel(r"retained power $u$")
    ax.set_ylabel(r"$b(u)=\mathrm{d}\log\Lambda/\mathrm{d}\log u$")
    ax.set_title("(a) $b(u)$ spans 6.3$\\times$; closed form "
                 r"$b\approx-0.70-6.4\,u^{3.1}$", fontsize=10.5, loc="left")
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.93)

    # ---------------- (b) logΛ vs log u ----------------
    ax = axes[1]
    for fid in DENSE:
        us = np.array(sorted(lam[fid]))
        ax.plot(us, [math.log(lam[fid][u]) for u in us], "o-", ms=3.6, lw=1.4,
                color=COL[fid], label=LBL[fid], zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"retained power $u$")
    ax.set_ylabel(r"$\log\Lambda$")
    ax.set_title("(b) smooth and monotone across 1 decade in $u$",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.4, loc="upper right", framealpha=0.93)
    ax.text(0.16, 7.15, "one operator per point;\n20 images, paired",
            fontsize=7.2, color=GRAY)

    fig.suptitle("The exponent $b(u)$ in $\\Lambda=C(\\text{shape})\\,u^{\\,b(u)}$",
                 fontsize=11.5, y=1.01)
    fig.tight_layout()
    o = os.path.join(OUT, "fig_bdense.png")
    fig.savefig(o, dpi=170, bbox_inches="tight")
    print("[saved]", o)
    for fid in DENSE:
        print(fid, "closed:", np.round(closed[fid], 4))


if __name__ == "__main__":
    main()
