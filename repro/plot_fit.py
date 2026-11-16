"""b(u,t) 四个参数的 t 依赖性图：2x2。读 fit_bt.py --dump 的 JSON。

每格：3 个族的实测点（带 2SE 误差棒）+ 预官的四个候选形式里最优的那条。
灰色点线是高斯极限（b_inf->0, A->-1, q->1；C 无闭式极限）。

用法： venv2/bin/python fit_bt.py --dump fig_fit.json
      venv2/bin/python plot_fit.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

BLUE, ORANGE, GREEN, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8f98"
INK = "#1c1f24"
FCOL = {"W2": BLUE, "W8": ORANGE, "TK50": GREEN}
KEYS = [("b_inf", r"$b_\infty$", r"$b_\infty=b_0+k\,t^\alpha$", 0.0),
        ("A", r"$A$", r"$A=A_0+k\,t$", -1.0),
        ("q", r"$q$", r"$q=q_0+k\log_{10}t$", 1.0),
        ("C", r"$C$ (geometric mean over $u$)", r"$C=C_0+k\log_{10}t$", None)]


def main():
    p = os.path.join(OUT, "fig_fit.json")
    if not os.path.exists(p):
        print("[缺少]", p, "——先跑 fit_bt.py --dump fig_fit.json")
        return
    d = json.load(open(p))
    ts = np.array(d["cells"], dtype=float)
    fams = d["dense"]

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0))
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xscale("log")

    for ax, (key, ylab, formlab, glim) in zip(axes.ravel(), KEYS):
        idx = {"b_inf": 0, "A": 1, "q": 2}[key] if key != "C" else None
        for fid in fams:
            ys, es = [], []
            for t in ts:
                e = d["est"][f"{t}|{fid}"]
                ys.append(e["p"][idx] if idx is not None else e["C"])
                es.append(d["se"].get(f"{key}|{fid}|{t}", np.nan))
            ys = np.array(ys, dtype=float)
            es = 2 * np.array(es, dtype=float)
            c = FCOL[fid]
            ax.errorbar(ts, ys, yerr=[es, es], fmt="o", ms=5.5, color=c,
                        lw=1.2, capsize=3, zorder=3, label=fid)
            # 该族的最优 2 参数形式
            fits = d["fit"][key][fid]
            two = {k: v for k, v in fits.items()
                   if k.startswith("(i)") or k.startswith("(iii)")}
            if two:
                best = min(two.items(), key=lambda kv: kv[1]["rms"])
                lab, v = best
                pa = v["p"]
                if lab.startswith("(i)"):
                    yy = pa[0] + pa[1] * np.log10(ts)
                else:
                    yy = pa[0] + pa[1] * ts
                ax.plot(ts, yy, "--", color=c, lw=1.2, alpha=0.8, zorder=2)
        if glim is not None:
            ax.axhline(glim, color=GRAY, lw=1.1, ls=":", zorder=1,
                       label=f"Gaussian limit {glim:g}")
        ax.set_xticks(ts)
        ax.set_xticklabels([f"{v:g}" for v in ts], fontsize=8.0, rotation=35,
                           ha="right")
        ax.minorticks_off()
        ax.set_xlabel(r"noise level $t$")
        ax.set_ylabel(ylab)
        # 判决
        thr = 0.5 * min(2 * d["se"].get(f"{key}|{f}|{t}", 9)
                        for f in fams for t in ts)
        best_rms = min(min(d["fit"][key][f][k]["rms"] for k in d["fit"][key][f]
                           if k.startswith("(i)") or k.startswith("(iii)"))
                      for f in fams)
        ok = best_rms <= thr
        ax.set_title(f"{formlab}\nbest 2-param rms={best_rms:.4g} vs "
                     f"threshold={thr:.4g}  P2 {'PASS' if ok else 'FAIL'}",
                     fontsize=9.6, loc="left")
        ax.legend(fontsize=7.2, loc="best", framealpha=0.93)

    fig.suptitle(r"The four parameters of $b(u,t)=b_\infty+A\,u^{\,q}$ "
                 "and $C$: are they simple functions of $t$?",
                 fontsize=11.5, y=1.01)
    fig.tight_layout()
    o = os.path.join(OUT, "fig_fit.png")
    fig.savefig(o, dpi=170, bbox_inches="tight")
    print("[saved]", o)


if __name__ == "__main__":
    main()
