"""Phase C 可视化：marginalized profile ell_y^opt(theta) 跨噪声水平。

形式：3 个图像各一个面板（小倍数），每面板 5 条噪声曲线。
      单面板多序列 -> 需要图例 + 直接标签。
      theta 量纲四族一致（都是 Gaussian sigma），可共用一根 x 轴。
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
d = json.load(open(os.path.join(OUT, "phaseC_full.json")))

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
MARK = ["o", "s", "^", "D", "v"]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
                     "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.8})

TH = np.array(d["cfg"]["thetas"])
NOISES = d["cfg"]["noises"]
TSTAR = d["cfg"]["sigma_star"]
NIMG = d["cfg"]["nimg"]

fig, axes = plt.subplots(1, NIMG, figsize=(4.4 * NIMG, 4.3), sharey=True)
if NIMG == 1:
    axes = [axes]

for ax, img in zip(axes, range(NIMG)):
    for sn, col, mk in zip(NOISES, C, MARK):
        rec = next(r for r in d["records"] if r["img"] == img and r["noise"] == sn)
        Ls = np.array([p["L"] for p in rec["profile"]])
        ax.plot(TH, Ls, color=col, lw=2, marker=mk, ms=5.5, mec=SURFACE, mew=1.1,
                label=f"$\\sigma$={sn}", zorder=3)
    ax.axvline(TSTAR, color=MUTED, ls="--", lw=1.2, zorder=1)
    ax.text(TSTAR, ax.get_ylim()[1], r"  $\theta^\star$", color=MUTED, fontsize=9,
            va="top", ha="left")
    ax.axvline(0.0, color="#e34948", ls=":", lw=1.4, zorder=1)
    ax.text(0.02, ax.get_ylim()[0], " no-blur trap", color="#e34948", fontsize=8.5,
            va="bottom", ha="left")
    ax.set_title(f"image {img}", color=INK, fontsize=11.5, fontweight="bold", loc="left")
    ax.set_xlabel(r"kernel parameter $\theta$ (Gaussian $\sigma$)", fontsize=9.5)
    ax.grid(True, color=MUTED, alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

axes[0].set_ylabel(r"profile  $\ell_y^{\mathrm{opt}}(\theta)=\min_x L(x,\theta;y)$",
                   fontsize=9.5)
axes[-1].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK2,
                title="noise", title_fontsize=9)

fig.suptitle("Marginalized profile across noise: two basins, but no clean collapse threshold",
             fontsize=12, color=INK, y=0.99)
fig.text(0.5, 0.015,
         "Lower = more likely. Red dotted: no-blur trap (Lemma 12). Grey dashed: true kernel. "
         "The local basin sits near $\\theta^\\star$ but is displaced; at high noise the minimum drifts to over-blur ($\\theta>\\theta^\\star$).",
         ha="center", fontsize=8.6, color=MUTED)
fig.tight_layout(rect=[0, 0.045, 1, 0.94])
p = os.path.join(OUT, "phaseC_profiles.png")
fig.savefig(p, dpi=170)
plt.close(fig)
print("[saved]", p)
