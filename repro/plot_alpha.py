"""Alpha-插值实验的可视化（修订版）：先验沿模糊方向的局部对数密度剖面

fig_alpha.png  1x3
  左：Lambda(alpha) vs alpha（双对数），每核一条曲线，灰色虚线是高斯大对照
      —— 曲线本身是弯的，所以"幂律指数"不是常数
  中：局部指数 d log Lambda / d log alpha vs alpha
      —— 判据图：指数从 ~1.2 升到 2-4.6，而高斯大对照的指数从 ~0.9 掉到 ~0
  右：放大倍数 Lambda / Lambda_Gauss vs alpha
      —— 放大倍数本身随模糊强度增长（3 -> 22-37），不是常数幂律

配色：4 个观察族用 dataviz 已验证的 4 色分类板（validate_palette.js --pairs all 全过）；
      low-pass 是理想化构造而非观察核，用中性灰，避免凑第 5 个色相。
      绿色对比度 2.74:1 有 WARN，用图例 + 表格兜底（本图不依赖颜色单独传达信息）。

SNR < 3 的点（Hutchinson 底噪下不可用）不画。

用法： python plot_alpha.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

COL = {"gaussian": "#2a78d6", "defocus": "#eb6834",
       "airy": "#1baf7a", "motion": "#4a3aa7", "lowpass": "#8a8880"}
MARK = {"gaussian": "o", "defocus": "s", "airy": "^", "motion": "D",
        "lowpass": "v"}
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


def local_exp(a, y):
    a = np.asarray(a, float)
    y = np.asarray(y, float)
    out = []
    for i in range(len(a) - 1):
        if a[i] <= 0 or a[i + 1] <= 0 or y[i] <= 0 or y[i + 1] <= 0:
            continue
        da = np.log(a[i + 1] / a[i])
        if abs(da) < 1e-9:
            continue
        out.append((np.sqrt(a[i] * a[i + 1]), np.log(y[i + 1] / y[i]) / da))
    if not out:
        return np.array([]), np.array([])
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def main():
    p = os.path.join(OUT, "sweep_alpha.json")
    if not os.path.exists(p):
        print("[缺少]", p)
        return
    d = json.load(open(p))
    recs, cfg = d["records"], d["cfg"]
    t0 = min(r["t"] for r in recs)
    nimg = cfg["nimg"]

    def snr(r):
        return r["lam_mean"] / r["lam_std"] * np.sqrt(nimg) \
            if r["lam_std"] > 0 else np.inf

    rr = [r for r in recs if r["t"] == t0 and r["lam_mean"] > 0 and snr(r) >= 3.0]
    kers = sorted({(r["family"], r["theta"]) for r in rr})
    strong = {}
    for fam in {k[0] for k in kers}:
        strong[fam] = max(k[1] for k in kers if k[0] == fam)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.6, 5.0))

    # ================= 左：Lambda(alpha) =================
    gmean = float(np.mean([r["lam_gauss_mean"] for r in rr]))
    for (fam, th) in kers:
        pts = sorted([r for r in rr if (r["family"], r["theta"]) == (fam, th)],
                     key=lambda r: r["alpha"])
        a = np.array([r["alpha"] for r in pts])
        y = np.array([r["lam_mean"] for r in pts])
        e = np.array([r["lam_std"] for r in pts]) / np.sqrt(nimg)
        a1.errorbar(a, y, yerr=e, color=COL[fam], lw=1.9,
                    ls="-" if th == strong[fam] else "--", marker=MARK[fam],
                    ms=5.5, mec=SURFACE, mew=1.1, capsize=2.5, elinewidth=1.0,
                    ecolor=MUTED, zorder=3,
                    label=f"{fam} $\\theta$={th:g}")

    a1.axhline(gmean, color=MUTED, ls=":", lw=1.4, zorder=1)
    a1.text(0.017, gmean * 1.13, r"Gaussian control $\Lambda_{Gauss}$",
            color=MUTED, fontsize=8.5, va="bottom")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlim(0.012, 1.35)
    a1.set_ylim(3e2, 1.2e5)
    a1.set_xlabel(r"$\alpha$  (fraction of the way to the blurred image)",
                  fontsize=10)
    a1.set_ylabel(r"$\Lambda(\alpha)$  (nats)", fontsize=10)
    a1.set_title(r"$\Lambda$ along the blur segment", color=INK, fontsize=11.5,
                 fontweight="bold", loc="left", pad=8)
    a1.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2,
              handlelength=2.4, labelspacing=0.35, borderpad=0.2)
    style(a1)

    # ================= 中：局部指数 =================
    gexp = []
    for (fam, th) in kers:
        pts = sorted([r for r in rr if (r["family"], r["theta"]) == (fam, th)],
                     key=lambda r: r["alpha"])
        a = np.array([r["alpha"] for r in pts])
        gm, ge = local_exp(a, [r["lam_mean"] for r in pts])
        if len(gm):
            a2.plot(gm, ge, color=COL[fam], lw=1.9,
                    ls="-" if th == strong[fam] else "--", marker=MARK[fam],
                    ms=5.5, mec=SURFACE, mew=1.1, zorder=3)
        gm2, ge2 = local_exp(a, [r["lam_gauss_mean"] for r in pts])
        for x, y in zip(gm2, ge2):
            gexp.append((x, y))

    if gexp:
        gx = np.array([e[0] for e in gexp])
        gy = np.array([e[1] for e in gexp])
        o = np.argsort(gx)
        a2.plot(gx[o], gy[o], color=MUTED, lw=1.2, ls=(0, (1, 1.4)),
                marker="x", ms=4.5, mew=1.1, zorder=2, alpha=0.9)

    for lvl, txt, side in [(1.0, r"score-dominated:  $\Lambda\propto\alpha$",
                            "right"),
                           (2.0, r"curvature-dominated:  "
                                 r"$\Lambda\propto\alpha^2$", "left")]:
        a2.axhline(lvl, color=INK2, ls="--", lw=0.9, zorder=1)
        a2.text(1.40 if side == "right" else 0.198, lvl + 0.10, txt,
                color=INK2, fontsize=8.2, ha=side)
    a2.set_xscale("log")
    a2.set_xlim(0.19, 1.45)
    a2.set_ylim(-1.4, 5.4)
    a2.set_xlabel(r"$\alpha$", fontsize=10)
    a2.set_ylabel(r"local exponent  $d\log\Lambda/d\log\alpha$", fontsize=10)
    a2.set_title("The exponent is not a constant", color=INK, fontsize=11.5,
                 fontweight="bold", loc="left", pad=8)
    a2.legend(handles=[
        Line2D([], [], color=INK2, lw=1.9, marker="o", ms=5, mec=SURFACE,
               mew=1, label=r"$\Lambda$  (solid = strong $\theta$)"),
        Line2D([], [], color=MUTED, lw=1.2, ls=(0, (1, 1.4)), marker="x",
               ms=4.5, label=r"$\Lambda_{Gauss}$  (concave, saturates)")],
        frameon=False, fontsize=8.2, loc="lower left", labelcolor=INK2,
        handlelength=2.4, labelspacing=0.35, borderpad=0.2)
    style(a2)

    # ================= 右：放大倍数 =================
    for (fam, th) in kers:
        pts = sorted([r for r in rr if (r["family"], r["theta"]) == (fam, th)],
                     key=lambda r: r["alpha"])
        a = np.array([r["alpha"] for r in pts])
        ratio = np.array([r["lam_mean"] / r["lam_gauss_mean"] for r in pts])
        a3.plot(a, ratio, color=COL[fam], lw=1.9,
                ls="-" if th == strong[fam] else "--", marker=MARK[fam],
                ms=5.5, mec=SURFACE, mew=1.1, zorder=3)
    a3.axhline(1.0, color=MUTED, ls=":", lw=1.4, zorder=1)
    a3.text(1.32, 1.10, "Gaussian prediction", color=MUTED, fontsize=8.5,
            ha="right")
    a3.set_xscale("log")
    a3.set_yscale("log")
    a3.set_xlim(0.012, 1.35)
    a3.set_ylim(0.7, 60)
    a3.set_xlabel(r"$\alpha$", fontsize=10)
    a3.set_ylabel(r"amplification  $\Lambda/\Lambda_{Gauss}$", fontsize=10)
    a3.set_title("The excess over a matched Gaussian grows with blur",
                 color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)
    style(a3)

    fig.text(0.006, 0.012,
             f"$t$={t0:g}, {nimg} prior samples, error bars SEM; points with "
             f"SNR$<3$ dropped. Middle panel: the local exponent of $\\Lambda$ "
             f"rises from $\\sim$1.2 (score-dominated) toward 2-4.6, while the "
             f"Gaussian control's exponent falls from $\\sim$0.9 to $\\sim$0 "
             f"(it saturates). Opposite curvature, so the amplification is "
             f"not a constant power law.",
             fontsize=8.3, color=MUTED, ha="left")
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    p = os.path.join(OUT, "fig_alpha.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    print("[saved]", p)


if __name__ == "__main__":
    main()
