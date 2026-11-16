"""b(u) 的 t 依赖性图：2x3。读 analyze_bt.py --dump 出来的 JSON（不重算 bootstrap）。

(a) b(u) vs u：基线 t=0.002 与各新 t 的三族均值（带三族极差带）+ 高斯极限
    b_Gauss(u) = -u/(1-u)（与 t 无关的闭式参照）。
(b) b_inf(t) 与 2SE 误差棒；虚线 = 基线值，点线 = 高斯极限 0。
(c) q(t) 与 2SE 误差棒；虚线 = 基线值，点线 = 高斯极限 1。
(d) D(u) = logΛ_t(u) - logΛ_0.002(u)（三族均值）：H0（b 与 t 无关）要求它与 u 无关。
(e) T5b：实测 b(t,u) 减两分量混合模型的预言 b_pred(t,u)；带 ±0.10 容差带。
(f) 文字面板：预官判据与实测判定。

用法： venv2/bin/python analyze_bt.py --dump fig_bt.json
      venv2/bin/python plot_bt.py
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
COL = {0.005: BLUE, 0.01: ORANGE, 0.02: GREEN, 0.05: PURPLE}


def main():
    p = os.path.join(OUT, "fig_bt.json")
    if not os.path.exists(p):
        print("[缺少]", p, "——先跑 analyze_bt.py --dump fig_bt.json")
        return
    d = json.load(open(p))
    ug = np.array(d["ug"])
    fams = d["dense"]
    news = sorted(d["new"], key=lambda e: e["t"])

    def band(ent, key):
        """ent 可以是 {'fam': {...}}（新 t）或直接 {fid: {...}}（基线）。"""
        src = ent["fam"] if "fam" in ent else ent
        M = np.array([src[f][key] for f in fams], dtype=float)
        return M.mean(0), M.min(0), M.max(0)

    b0m, b0lo, b0hi = band(d["base"], "b")

    fig, axes = plt.subplots(2, 3, figsize=(15.2, 8.4))
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---------------- (a) b(u) ----------------
    ax = axes[0, 0]
    uu = np.linspace(ug.min(), ug.max(), 300)
    ax.plot(uu, -uu / (1.0 - uu), color=GRAY, lw=1.2, ls=":", zorder=1,
            label=r"Gaussian limit $-u/(1-u)$")
    ax.fill_between(ug, b0lo, b0hi, color=INK, alpha=0.10, zorder=2)
    ax.plot(ug, b0m, "o-", color=INK, ms=4.2, lw=1.5, zorder=4,
            label=r"$t=0.002$ (baseline)")
    for e in news:
        m, lo, hi = band(e, "b")
        c = COL.get(round(e["t"], 4), GRAY)
        ax.fill_between(ug, lo, hi, color=c, alpha=0.12, zorder=2)
        ax.plot(ug, m, "s--", color=c, ms=3.8, lw=1.3, zorder=3,
                label=rf"$t={e['t']:g}$")
    ax.set_xlabel(r"retained power $u$")
    ax.set_ylabel(r"$b(u)=\mathrm{d}\log\Lambda/\mathrm{d}\log u$")
    ax.set_title("(a) $b(u)$ at several noise levels\n"
                 "(band = spread over 3 shape families)",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.2, loc="lower right", framealpha=0.93)

    # ---------------- (b) b_inf(t) ----------------
    ax = axes[0, 1]
    b0 = float(np.mean([d["base"][f]["p"][0] for f in fams]))
    ts = [e["t"] for e in news]
    for e in news:
        v = [e["fam"][f]["p"][0] for f in fams]
        se = [e["fam"][f].get("se_binf", np.nan) for f in fams]
        c = COL.get(round(e["t"], 4), GRAY)
        ax.errorbar([e["t"]] * len(fams), v,
                    yerr=[[2 * s for s in se], [2 * s for s in se]],
                    fmt="o", ms=5, color=c, lw=1.2, capsize=3, zorder=3)
    ax.axhline(b0, color=INK, lw=1.2, ls="--", zorder=2,
               label=rf"baseline $t=0.002$: ${b0:.3f}$")
    ax.axhline(0.0, color=GRAY, lw=1.1, ls=":", zorder=1,
               label=r"Gaussian limit $b_\infty=0$")
    ax.set_xscale("log")
    ax.set_xticks(ts)
    ax.set_xticklabels([f"{v:g}" for v in ts])
    ax.set_xlabel(r"noise level $t$")
    ax.set_ylabel(r"$b_\infty$")
    ax.set_title(r"(b) $b_\infty(t)$, $2\,{\rm SE}$ bars"
                 "\nflat $\\Rightarrow$ $b(u)$ is $t$-invariant",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.4, loc="lower left", framealpha=0.93)

    # ---------------- (c) q(t) ----------------
    ax = axes[1, 0]
    q0 = float(np.mean([d["base"][f]["p"][2] for f in fams]))
    for e in news:
        v = [e["fam"][f]["p"][2] for f in fams]
        se = [e["fam"][f].get("se_q", np.nan) for f in fams]
        c = COL.get(round(e["t"], 4), GRAY)
        ax.errorbar([e["t"]] * len(fams), v,
                    yerr=[[2 * s for s in se], [2 * s for s in se]],
                    fmt="o", ms=5, color=c, lw=1.2, capsize=3, zorder=3)
    ax.axhline(q0, color=INK, lw=1.2, ls="--", zorder=2,
               label=rf"baseline $t=0.002$: ${q0:.3f}$")
    ax.axhline(1.0, color=GRAY, lw=1.1, ls=":", zorder=1,
               label=r"Gaussian limit $q=1$")
    ax.set_xscale("log")
    ax.set_xticks(ts)
    ax.set_xticklabels([f"{v:g}" for v in ts])
    ax.set_xlabel(r"noise level $t$")
    ax.set_ylabel(r"$q$")
    ax.set_title(r"(c) exponent $q$ in $b=b_\infty+A\,u^q$"
                 "\nGaussian convergence predicts $q\\to1$",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.4, loc="upper right", framealpha=0.93)

    # ---------------- (d) D(u) ----------------
    ax = axes[0, 2]
    for e in news:
        M = np.array([e["fam"][f]["D"] for f in fams], dtype=float)
        c = COL.get(round(e["t"], 4), GRAY)
        ax.fill_between(ug, M.min(0), M.max(0), color=c, alpha=0.12, zorder=2)
        ax.plot(ug, M.mean(0), "s-", color=c, ms=3.8, lw=1.3, zorder=3,
                label=rf"$t={e['t']:g}$")
    ax.axhline(0.0, color=INK, lw=1.2, ls="--", zorder=1)
    ax.set_xlabel(r"retained power $u$")
    ax.set_ylabel(r"$D(u)=\log\Lambda_t(u)-\log\Lambda_{0.002}(u)$")
    ax.set_title("(d) model-free factorisation test\n"
                 r"$b(u)$ $t$-invariant $\Rightarrow$ $D(u)$ flat",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=7.4, loc="best", framealpha=0.93)

    # ---------------- (e) T5b ----------------
    ax = axes[1, 1]
    has = False
    for e in news:
        if "dbpred" not in e["fam"][fams[0]]:
            continue
        M = np.array([e["fam"][f]["dbpred"] for f in fams], dtype=float)
        c = COL.get(round(e["t"], 4), GRAY)
        ax.fill_between(ug, M.min(0), M.max(0), color=c, alpha=0.12, zorder=2)
        ax.plot(ug, M.mean(0), "s-", color=c, ms=3.8, lw=1.3, zorder=3,
                label=rf"$t={e['t']:g}$")
        has = True
    ax.axhspan(-0.10, 0.10, color=GRAY, alpha=0.13, zorder=1)
    ax.axhline(0.0, color=INK, lw=1.2, ls="--", zorder=2)
    ax.set_xlabel(r"retained power $u$")
    ax.set_ylabel(r"$b_{\rm meas}-b_{\rm pred}$")
    ax.set_title("(e) two-component prediction\n"
                 r"$\Lambda=\Lambda_{\rm Gauss}+\Lambda_{\rm ng}$, "
                 r"$b_{\rm ng}$ $t$-invariant", fontsize=10.5, loc="left")
    if has:
        ax.legend(fontsize=7.4, loc="best", framealpha=0.93)
        ax.text(0.03, 0.05, "shaded = pre-registered $\\pm0.10$ tolerance",
                transform=ax.transAxes, fontsize=7.2, color=GRAY)
    else:
        ax.text(0.5, 0.5, "T5b not evaluated\n(no dump with bpred)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=GRAY)

    # ---------------- (f) verdict text ----------------
    ax = axes[1, 2]
    ax.axis("off")
    lines = ["Pre-registered criteria (group 8, 2026-10-18)",
             "T1  closed-form rms $\\leq 2\\times$ baseline",
             "T2  $|\\Delta b_\\infty| \\leq 2\\,$SE  (paired bootstrap)",
             "T3  $|\\Delta q| \\leq 2\\,$SE",
             "T4  slope of $D(u)$ vs $\\log u$ $\\leq 2\\,$SE",
             "T5b $|b-b_{\\rm pred}|$ rms $\\leq0.10$",
             "T6  report $2\\,$SE detection floor",
             "", "Measured (family-averaged):"]
    for e in news:
        f0 = e["fam"][fams[0]]
        sb = np.mean([e["fam"][f].get("se_binf", np.nan) for f in fams])
        sq = np.mean([e["fam"][f].get("se_q", np.nan) for f in fams])
        bi = np.mean([e["fam"][f]["p"][0] for f in fams])
        qq = np.mean([e["fam"][f]["p"][2] for f in fams])
        r5 = np.mean([e["fam"][f].get("rms", np.nan) for f in fams])
        lines.append(rf"$t={e['t']:g}$:  $b_\infty={bi:+.3f}$  $q={qq:.2f}$  "
                     rf"rms$={r5:.3f}$")
        lines.append(rf"      $2$SE$(b_\infty)={2*sb:.3f}$   "
                     rf"$2$SE$(q)={2*sq:.3f}$")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8.2,
            family="monospace", color=INK, transform=ax.transAxes)

    fig.suptitle(r"Is the exponent $b(u)$ in $\Lambda=C\,u^{\,b(u)}$ "
                 "independent of the noise level $t$?", fontsize=11.5, y=1.01)
    fig.tight_layout()
    o = os.path.join(OUT, "fig_bt.png")
    fig.savefig(o, dpi=170, bbox_inches="tight")
    print("[saved]", o)


if __name__ == "__main__":
    main()
