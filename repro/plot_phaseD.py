"""Phase D 可视化：高斯谱近似下的证据后验 p(theta|y)。

形式：2x1。
  上：后验 p(theta|y) 随 theta，多条曲线（每噪声一条）—— 4 序列 -> 图例+直接标签
  下：径向功率谱 S(rho)（对数轴），叠加 theta* 的核响应 |h(rho)|^2
      —— 展示机制：核在哪些频率上把信号压到噪声以下

配色：dataviz 参考调色板前 4 槽（validator 全过）。
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
fn = "phaseD_spec_t0.0.json" if len(os.sys.argv) < 2 else os.sys.argv[1]
d = json.load(open(os.path.join(OUT, fn)))

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8880"
MARK = ["o", "s", "^", "D"]
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans",
                     "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.linewidth": 0.8})

recs = d["records"]
TH = np.array(d["cfg"]["thetas"])
NOISES = d["cfg"]["noises"]
TSTAR = d["cfg"]["sigma_star"]
img0 = sorted({r["img"] for r in recs})[0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.8, 7.2),
                               gridspec_kw={"height_ratios": [1.35, 1]})

for sn, col, mk in zip(NOISES, C, MARK):
    r = next(r for r in recs if r["img"] == img0 and r["noise"] == sn)
    p = np.array(r["post"])
    ax1.plot(TH, p, color=col, lw=2, marker=mk, ms=5.5, mec=SURFACE, mew=1.1,
             label=f"$\\sigma_n$={sn}", zorder=3)
    ax1.annotate(f"{r['mode']:.2f}", xy=(r["mode"], p.max()), xytext=(5, 0),
                 textcoords="offset points", fontsize=8.5, color=col,
                 fontweight="bold", va="center")

ax1.axvline(TSTAR, color=MUTED, ls="--", lw=1.3, zorder=1)
ax1.text(TSTAR, ax1.get_ylim()[1], r"  $\theta^\star$", color=MUTED, fontsize=9.5,
         va="top")
ax1.axvline(0.0, color="#e34948", ls=":", lw=1.5, zorder=1)
ax1.text(0.03, ax1.get_ylim()[1] * 0.55, "no-blur\n(weight $\\approx$ 0)",
         color="#e34948", fontsize=8.5, va="top")
ax1.set_xlabel(r"kernel parameter $\theta$ (Gaussian $\sigma$)", fontsize=10)
ax1.set_ylabel(r"posterior  $p(\theta\,|\,y)$  (evidence, Gaussian-spectrum approx.)",
               fontsize=9.5)
ax1.set_title(f"Evidence posterior on the kernel — image {img0}", color=INK,
              fontsize=12, fontweight="bold", loc="left", pad=10)
ax1.grid(True, color=MUTED, alpha=0.22, lw=0.6)
ax1.set_axisbelow(True)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
ax1.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=INK2)

# ---- 下：径向谱 + 核响应 ----
rho = np.array(d["spectrum_rho"])
S = np.array(d["spectrum_S"])
m = rho > 0.5
ax2.loglog(rho[m], S[m], color="#4a3aa7", lw=2, label="prior power spectrum $S(\\rho)$",
           zorder=3)
for th, col in zip([0.5, 1.0, 1.5, 2.5], ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]):
    # 高斯核的径向傅里叶响应
    h2 = np.exp(-2 * (np.pi * th * rho) ** 2 / 1.0)  # 近似：|h|^2 ~ exp(-c sigma^2 rho^2)
    ax2.loglog(rho[m], (h2 * S)[m], color=col, lw=1.6, ls="--",
               label=f"$|\\hat h|^2 S$,  $\\theta$={th}", zorder=2)
ax2.axhline(0.02 ** 2, color="#e34948", ls=":", lw=1.4)
ax2.text(rho[m][0], 0.02 ** 2 * 1.3, "noise floor $\\sigma_n^2$ ($\\sigma_n$=0.02)",
         color="#e34948", fontsize=8.5, va="bottom")
ax2.set_xlabel(r"radial frequency $\rho$ (cycles / image)", fontsize=10)
ax2.set_ylabel("power", fontsize=10)
ax2.set_title("Mechanism: the kernel suppresses high frequencies below the noise floor",
              color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=8)
ax2.grid(True, which="both", color=MUTED, alpha=0.2, lw=0.5)
ax2.set_axisbelow(True)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
ax2.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=INK2, ncol=1)

fig.text(0.5, 0.012,
         "Lower panel: where $|\\hat h|^2 S$ drops below $\\sigma_n^2$, the data must be "
         "explained by noise — the Occam factor penalizes models that cannot.",
         ha="center", fontsize=8.6, color=MUTED)
fig.tight_layout(rect=[0, 0.035, 1, 1])
p = os.path.join(OUT, "phaseD_evidence.png")
fig.savefig(p, dpi=170)
plt.close(fig)
print("[saved]", p)
