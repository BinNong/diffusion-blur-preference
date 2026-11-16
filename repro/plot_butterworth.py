"""Butterworth 受控实验的可视化。

fig_butterworth.png  1x2
  左：Lambda vs kunif，每个阶数 n 一条曲线（t 最小），灰色虚线是高斯大对照
      —— 展示同一族内、只改滚降锐度时的幂律
  右：指数 b_n vs 过渡带宽度（对数横轴），误差棒是 bootstrap SE
      —— 判据图：若单调下降，机理成立（指数由滚降锐度决定）

同时把三模型分析里那 5 个观察核的 (width, b_f) 作为参照点叠在右图上
（不同标记），说明受控实验与观察性数据是否一致。

用法： python plot_butterworth.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

C = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
})


def style(ax):
    ax.grid(True, color=MUTED, alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fit_slope(ks, ys):
    A = np.vstack([np.ones(len(ks)), np.log(ks)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(ys), rcond=None)
    pred = A @ coef
    r2 = 1 - ((np.log(ys) - pred) ** 2).sum() / \
        ((np.log(ys) - np.log(ys).mean()) ** 2).sum()
    return coef[1], r2


def main():
    d = json.load(open(os.path.join(OUT, "sweep_butterworth.json")))
    recs, cfg = d["records"], d["cfg"]
    t0 = min(r["t"] for r in recs)
    orders = sorted({r["order"] for r in recs})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    # ---- 左：Lambda vs kunif，每阶一条 ----
    for n, col in zip(orders, C):
        rr = sorted([r for r in recs if r["order"] == n and r["t"] == t0],
                    key=lambda r: r["kunif"])
        if not rr:
            continue
        k = np.array([r["kunif"] for r in rr])
        l = np.array([r["lam_mean"] for r in rr])
        e = np.array([r["lam_std"] for r in rr]) / np.sqrt(cfg["nimg"])
        a1.errorbar(k, l, yerr=e, color=col, lw=1.9, marker="o", ms=5.5,
                    mec=SURFACE, mew=1.1, capsize=2.5, elinewidth=1.0,
                    ecolor=MUTED, zorder=3, label=f"$n$={n}")
        b, _ = fit_slope(k, l)
        a1.annotate(f"$n$={n}\n$b$={b:.2f}", xy=(k[-1], l[-1]), xytext=(7, 0),
                    textcoords="offset points", fontsize=8.5, color=col,
                    fontweight="bold", va="center")
    g = [r["lam_gauss_mean"] for r in recs if r["t"] == t0]
    a1.axhline(float(np.mean(g)), color=MUTED, ls="--", lw=1.3, zorder=1)
    a1.text(a1.get_xlim()[0], float(np.mean(g)) * 1.05,
            r"Gaussian control $\Lambda_{Gauss}$", color=MUTED, fontsize=8.5,
            va="bottom")
    a1.set_xlabel(r"kunif $=\langle 1-|\hat h|^2\rangle_f$", fontsize=10)
    a1.set_ylabel(r"$\Lambda$  (nats)", fontsize=10)
    a1.set_title("Same family, only the roll-off sharpness changes", color=INK,
                 fontsize=11.5, fontweight="bold", loc="left", pad=8)
    style(a1)

    # ---- 右：b_n vs 过渡带宽度 + 观察核参照 ----
    rows = []
    for n in orders:
        rr = sorted([r for r in recs if r["order"] == n and r["t"] == t0],
                    key=lambda r: r["kunif"])
        if len(rr) < 3:
            continue
        k = np.array([r["kunif"] for r in rr])
        l = np.array([r["lam_mean"] for r in rr])
        b, _ = fit_slope(k, l)
        per = np.array([r["lam_per_img"] for r in rr])
        nimg = per.shape[1]
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(200):
            m = per[:, rng.integers(0, nimg, nimg)].mean(axis=1)
            ok = m > 0
            if ok.sum() >= 3:
                bs.append(fit_slope(k[ok], m[ok])[0])
        se = float(np.std(bs)) if len(bs) > 10 else float("nan")
        rows.append((n, float(np.mean([r["width"] for r in rr])), b, se))

    for n, w, b, se in rows:
        a2.errorbar([w], [b], yerr=[[se], [se]], color="#2a78d6", lw=0,
                    ecolor="#2a78d6", elinewidth=1.4, capsize=3, zorder=3)
        a2.plot([w], [b], "o", color="#2a78d6", ms=8, mec=SURFACE, mew=1.3,
                zorder=4)
        a2.annotate(f"$n$={n}", xy=(w, b), xytext=(0, 11),
                    textcoords="offset points", fontsize=9, color="#2a78d6",
                    fontweight="bold", ha="center")

    # 观察核（来自 dump_spectral 的三模型分析）
    obs = [("low-pass", 0.156, 5.68), ("motion", 5.664, 3.91),
           ("defocus", 9.304, 5.62), ("gaussian", 11.993, 4.21),
           ("airy", 22.587, 3.12)]
    for name, w, b in obs:
        a2.plot([w], [b], "s", color=INK, ms=6.5, mec=SURFACE, mew=1.1, zorder=3)
        a2.annotate(name, xy=(w, b), xytext=(0, -14), textcoords="offset points",
                    fontsize=8, color=INK2, ha="center")

    a2.set_xscale("log")
    a2.set_xlabel(r"spectral transition bandwidth  $\rho(0.9)-\rho(0.1)$",
                  fontsize=10)
    a2.set_ylabel(r"power-law exponent  $b$", fontsize=10)
    a2.set_title("Does the exponent track the transition bandwidth?", color=INK,
                 fontsize=11.5, fontweight="bold", loc="left", pad=8)
    style(a2)
    from matplotlib.lines import Line2D
    a2.legend(handles=[
        Line2D([], [], color="#2a78d6", marker="o", ls="none", ms=7,
               mec=SURFACE, mew=1.1, label="Butterworth (controlled)"),
        Line2D([], [], color=INK, marker="s", ls="none", ms=6.5, mec=SURFACE,
               mew=1.1, label="observed kernels (3 models)")],
        frameon=False, fontsize=9, loc="lower right", labelcolor=INK2)

    if len(rows) >= 3:
        w = np.array([r[1] for r in rows])
        b = np.array([r[2] for r in rows])
        c = float(np.corrcoef(w, b)[0, 1])
        fig.text(0.006, 0.012,
                 f"Left: $t$={t0:g}, {cfg['nimg']} prior samples, error bars SEM. "
                 f"Right: controlled Butterworth family (circles, error bars = "
                 f"bootstrap SE over images) vs the five observed kernels "
                 f"(squares). corr(width, exponent) = {c:+.2f} over the "
                 f"controlled points.", fontsize=8.3, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    p = os.path.join(OUT, "fig_butterworth.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print("[saved]", p)


if __name__ == "__main__":
    main()
