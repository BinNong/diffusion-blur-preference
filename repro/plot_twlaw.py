"""tw_lev 定律的图：1x3。
(a) C vs tw_lev——定律 + 6 个新族的预官检验（2 个底座剖面失败）；
(b) C vs B2——被推翻的对照；
(c) 径向剖面——失败的结构性原因（底座剖面 vs 有通带的剖面）。

用法： venv2/bin/python plot_twlaw.py
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
INK = "#2f333a"
NEW = ["TK30", "TK50", "EF60", "W3", "LG60", "TS3"]
FAIL = ["EF60", "LG60"]          # 预官检验失败的两个（底座剖面）


def main():
    tb = hd.build_tables()
    names, desc = tb["names"], tb["desc"]
    crel, r = tb["crel"], tb["r"]

    y = np.array([crel(n) for n in names])
    x = np.array([desc[n]["tw_lev"] for n in names])
    xb = np.array([desc[n]["B2"] for n in names])

    best = None
    for t0 in np.linspace(0.05, 3.0, 120):
        for p in np.linspace(0.3, 4.0, 76):
            X = np.column_stack([np.ones(len(x)),
                                 1.0 / (1.0 + (x / t0) ** (-p))])
            f = hd.ols(X, y)
            if f and (best is None or f["rmse"] < best[0]):
                best = (f["rmse"], f, t0, p)
    _, f, t0, pexp = best
    c0, c1 = f["coef"]
    fb = hd.ols(np.column_stack([np.ones(len(xb)), xb]), y)

    # ---- 新族 ----
    p = os.path.join(OUT, "sweep_twlaw.json")
    new = {}
    if os.path.exists(p):
        d = json.load(open(p))
        prof = defaultdict(list)
        meta = {}
        for rec in d["rows"]:
            prof[rec["meta"]["fid"]].append((1.0 - rec["kunif"], rec["lam_mean"]))
            meta[rec["meta"]["fid"]] = rec["meta"]
        for k in prof:
            prof[k].sort(key=lambda c: -c[0])
        b_of, mid = tb["b_of"], tb["mid"]
        for fid in NEW:
            if fid not in prof:
                continue
            m = meta[fid]
            a = 1.0 - hd.make_profile(m["kind"], m["par"], m["scale"], r)
            a = a * (0.60 / float(a.mean()))
            dd = hd.descriptors(a, r)
            vs = []
            for u in tb["Ucom"]:
                v = hd.interp(prof[fid], u)
                if v and v > 0 and u in mid:
                    vs.append(v * u ** (-b_of(u)) / mid[u])
            new[fid] = dict(tw=dd["tw_lev"], b2=dd["B2"], a=a,
                            c=float(np.median(vs)) if vs else float("nan"))

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4))
    for ax in axes:
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---------------- (a) C vs tw_lev ----------------
    ax = axes[0]
    tt = np.logspace(math.log10(0.012), math.log10(6.0), 300)
    cc = c0 + c1 / (1.0 + (tt / t0) ** (-pexp))
    ax.fill_between(tt, cc * 10 ** -0.045, cc * 10 ** 0.045, color=GRAY,
                    alpha=0.16, lw=0, label=r"law $\pm$0.045 dex (LOO tol.)")
    ax.plot(tt, cc, color=GRAY, lw=2.0, zorder=2,
            label=r"$C=%.3f+%.3f/[1+(tw/%.2f)^{-%.2f}]$, $R^2$=%.3f"
                  % (c0, c1, t0, pexp, f["r2"]))
    ax.scatter(x, y, s=44, color=BLUE, zorder=3, edgecolor="white", lw=0.8,
               label="22 measured profiles")
    if new:
        for fid in NEW:
            if fid not in new:
                continue
            ax.scatter([new[fid]["tw"]], [new[fid]["c"]], s=76, color=ORANGE,
                       zorder=4, marker="D", edgecolor="white", lw=0.9)
            dx, dy = (9, 3)
            if fid == "TK30":
                dx, dy = (8, -11)
            if fid == "TK50":
                dx, dy = (-40, -12)
            ax.annotate(fid, (new[fid]["tw"], new[fid]["c"]),
                        textcoords="offset points", xytext=(dx, dy), fontsize=7.6,
                        color=INK)
        ax.scatter([], [], s=76, color=ORANGE, marker="D", edgecolor="white",
                   lw=0.9, label="6 new families (pre-reg.)")
        # 两个失败点：一个共同的说明 + 两根箭头（避免标签互相压）
        for fid in FAIL:
            if fid in new:
                ax.scatter([new[fid]["tw"]], [new[fid]["c"]], s=210, zorder=5,
                           facecolor="none", edgecolor=PURPLE, lw=1.5)
        ax.annotate("pedestal profiles\n(no passband)\n$tw$ under-measures spread",
                    xy=(0.95, 1.075), xytext=(0.135, 1.115),
                    fontsize=7.6, color=PURPLE, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.1,
                                    connectionstyle="arc3,rad=-0.15"))
        ax.annotate("", xy=(1.16, 1.072), xytext=(0.62, 1.105),
                    arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.1,
                                    connectionstyle="arc3,rad=0.2"))
    ax.set_xscale("log")
    ax.set_xlabel(r"relative transition width  $tw_{\rm lev}$")
    ax.set_ylabel(r"$C$(shape) / smooth-family mean")
    ax.set_title("(a) transition width predicts $C$ — 5/6 pass", fontsize=10.5,
                 loc="left")
    ax.legend(fontsize=7.2, loc="lower right", framealpha=0.93)

    # ---------------- (b) C vs B2 ----------------
    ax = axes[1]
    ax.plot(xb, fb["coef"][0] + fb["coef"][1] * xb, color=GREEN, lw=2.0, ls="--",
            zorder=2, label=r"$B_2$ fit, $R^2$=%.3f (refuted)" % fb["r2"])
    ax.scatter(xb, y, s=44, color=BLUE, zorder=3, edgecolor="white", lw=0.8,
               label="22 measured profiles")
    if new:
        for fid in NEW:
            if fid not in new:
                continue
            ax.scatter([new[fid]["b2"]], [new[fid]["c"]], s=76, color=ORANGE,
                       zorder=4, marker="D", edgecolor="white", lw=0.9)
            ax.annotate(fid, (new[fid]["b2"], new[fid]["c"]),
                        textcoords="offset points", xytext=(7, 4), fontsize=7.6,
                        color=INK)
        ax.scatter([], [], s=76, color=ORANGE, marker="D", edgecolor="white",
                   lw=0.9, label="6 new families")
    hi = [(b, c) for b, c in zip(xb, y) if 0.66 <= b <= 0.71]
    if len(hi) >= 2:
        bs = [t[0] for t in hi]; cs = [t[1] for t in hi]
        ax.annotate("", xy=(np.mean(bs), max(cs)), xytext=(np.mean(bs), min(cs)),
                    arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=1.4))
        ax.text(np.mean(bs) + 0.004, np.mean(cs),
                "matched $B_2$:\n$C$ spans %.2fx" % (max(cs) / min(cs)),
                fontsize=7.6, color=PURPLE, va="center")
    ax.set_xlabel(r"binarization  $B_2=\langle a^2\rangle/\langle a\rangle$")
    ax.set_ylabel(r"$C$(shape) / smooth-family mean")
    ax.set_title("(b) $B_2$ is not sufficient", fontsize=10.5, loc="left")
    ax.legend(fontsize=7.2, loc="lower left", framealpha=0.93)

    # ---------------- (c) 径向剖面：失败的结构性原因 ----------------
    ax = axes[2]
    show = [("EF60", ORANGE, "-"), ("LG60", ORANGE, "-"),
            ("W4", BLUE, "-"), ("TK50", BLUE, "-"), ("W1", GRAY, "--")]
    for fid, col, ls in show:
        a = new[fid]["a"] if fid in new else tb["araw"][fid]
        rr, pr = hd.radial(a, r, nb=80)
        lbl = fid + (" (pedestal, fails)" if fid in FAIL else "")
        ax.plot(rr, pr, color=col, lw=2.0 if fid in FAIL else 1.6, ls=ls,
                label=lbl, zorder=3 if fid in FAIL else 2)
    ax.axhline(0.60, color=GRAY, lw=0.9, ls=":", zorder=1)
    ax.text(0.03, 0.635, r"$\langle a\rangle$ = 0.60 (common)", fontsize=7.2,
            color=GRAY)
    ax.set_xlabel(r"normalised radius $r$")
    ax.set_ylabel(r"radial attenuation $\bar a(r)$")
    ax.set_title("(c) why it fails: pedestal vs passband", fontsize=10.5,
                 loc="left")
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.93, borderpad=0.4)
    ax.set_ylim(-0.03, 1.06)

    fig.suptitle("Blur preference: what sets the prefactor $C$ in "
                 r"$\Lambda = C\,u^{\,b(u)}$", fontsize=11.5, y=1.01)
    fig.tight_layout()
    o = os.path.join(OUT, "fig_twlaw.png")
    fig.savefig(o, dpi=170, bbox_inches="tight")
    print("[saved]", o)


if __name__ == "__main__":
    main()
