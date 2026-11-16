"""Phase B 可视化：势能 q = -log p 随模糊强度的变化，四族对比。

形式选择：四族的 theta 量纲不同（motion 是整数长度 0-9，其余 0-3），
不能用同一根 x 轴 -> 小倍数面板（2x2），每面板单序列（无需图例，标题即标签）。
另附一张归一化对比图（x = theta/theta_max），四序列 -> 图例 + 直接标签。

配色：dataviz 参考调色板前 4 槽（已过 validator 全部硬门禁）。
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(PROJ, "results")
os.makedirs(OUTDIR, exist_ok=True)

# ---- dataviz 参考调色板（light 模式，前 4 槽，validator 全过）----
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8880"
MARKERS = ["o", "s", "^", "D"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.8,
})

d = json.load(open(os.path.join(OUTDIR, "phaseB_nimg20.json")))
recs = d["records"]
FAMS = ["gaussian", "defocus", "airy", "motion"]
LABEL = {"gaussian": "Gaussian", "defocus": "Defocus",
         "airy": "Airy", "motion": "Motion"}
XNAME = {"gaussian": r"Gaussian $\sigma$", "defocus": r"Defocus radius",
         "airy": r"Airy radius", "motion": "Motion length (px)"}


def load(fam):
    rows = [r for r in recs if r["family"] == fam]
    imgs = sorted({r["img"] for r in rows})
    th = sorted({r["theta"] for r in rows})
    M = np.array([[next(r["nll"] for r in rows if r["img"] == i and r["theta"] == t)
                   for t in th] for i in imgs])
    return np.array(th, float), M


# ================= 图 1：2x2 小倍数 =================
fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
for ax, fam, col, mk in zip(axes.ravel(), FAMS, C, MARKERS):
    th, M = load(fam)
    mean, std = M.mean(0), M.std(0)
    ax.fill_between(th, mean - std, mean + std, color=col, alpha=0.16, lw=0)
    ax.plot(th, mean, color=col, lw=2, marker=mk, ms=6, mec=SURFACE, mew=1.2,
            zorder=3)
    delta = mean[-1] - mean[0]
    mono = np.mean([np.all(np.diff(M[i]) <= 1e-9) for i in range(len(M))])
    ax.set_title(f"{LABEL[fam]}", color=INK, fontsize=11.5, fontweight="bold",
                 loc="left", pad=8)
    ax.text(0.0, 1.015, f"$\\Delta$ = {delta:+,.0f}   monotone {mono*100:.0f}%",
            transform=ax.transAxes, fontsize=8.6, color=INK2, va="bottom")
    ax.set_xlabel(XNAME[fam], fontsize=9)
    ax.set_ylabel(r"potential  $q=-\log p(x)$", fontsize=9)
    ax.grid(True, color=MUTED, alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.suptitle("Blurry images are more likely: potential vs. blur level (20 FFHQ-64 samples, mean ± std)",
             fontsize=11.5, color=INK, y=0.985)
fig.text(0.5, 0.012, "Lower potential = more likely. Curves go down, so blurred images are more probable than the originals.",
         ha="center", fontsize=8.8, color=MUTED)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
p1 = os.path.join(OUTDIR, "phaseB_families.png")
fig.savefig(p1, dpi=170)
plt.close(fig)

# ================= 图 2：归一化对比（四序列）=================
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for fam, col, mk in zip(FAMS, C, MARKERS):
    th, M = load(fam)
    xn = th / th.max()
    mean = M.mean(0)
    ax.plot(xn, mean, color=col, lw=2, marker=mk, ms=6.5, mec=SURFACE, mew=1.2,
            label=LABEL[fam], zorder=3)
    # 直接标签：曲线右端
    ax.annotate(LABEL[fam], xy=(xn[-1], mean[-1]), xytext=(6, 0),
                textcoords="offset points", va="center", fontsize=9.5,
                color=col, fontweight="bold")
ax.set_xlabel(r"Normalized blur level  $\theta\,/\,\theta_{\max}$", fontsize=10)
ax.set_ylabel(r"potential  $q=-\log p(x)$", fontsize=10)
ax.set_title("All four families drop on net; monotonicity is kernel-dependent", color=INK, fontsize=12,
             fontweight="bold", loc="left", pad=10)
ax.grid(True, color=MUTED, alpha=0.22, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=9.5, loc="lower left", labelcolor=INK2)
ax.set_xlim(-0.02, 1.14)
fig.text(0.5, 0.015, "Airy rises at small radius (ringing sidelobes redistribute high frequencies); Defocus rebounds at radius 2.0.",
         ha="center", fontsize=8.8, color=MUTED)
fig.tight_layout(rect=[0, 0.04, 1, 1])
p2 = os.path.join(OUTDIR, "phaseB_compare.png")
fig.savefig(p2, dpi=170)
plt.close(fig)

print("[saved]", p1)
print("[saved]", p2)
