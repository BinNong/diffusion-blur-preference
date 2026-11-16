# -*- coding: utf-8 -*-
"""plot_overblur_terms.py —— 论文图：过模糊的两分量竞争（零 GPU）

只读两个已落地的裁判/诊断 JSON，不重算任何 bootstrap：
  results/fig_overblur_mech.json    第一组 20 张 EDM 生成池图（§3.9(by)）
  results/fig_overblur_indep.json   同一池索引 20–39 的 20 张图（§3.9(bz3)）

画的是固定网格点上的差分 ΔX = X(θ=2.5) − X(θ=1.5)，不是 profile 在最优点处的分解。
实线 = 索引 0–19，虚线 = 索引 20–39；两组只是索引不相交，不是真实数据独立队列。
圆点 = data 项，三角 = 先验项 q。产出 results/fig_overblur_terms.{pdf,png}。

本机无 matplotlib，用法（服务器）：
  cd /data/experiment/de_conv_1
  ./venv2/bin/python repro/plot_overblur_terms.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

a = json.load(open(os.path.join(OUT, "fig_overblur_mech.json")))
b = json.load(open(os.path.join(OUT, "fig_overblur_indep.json")))

rows = a["per_sigma"]
sig = [r["sigma_n"] for r in rows]
dd0 = [r["d_data"] for r in rows]
dq0 = [r["d_q"] for r in rows]
dd1 = b["new"]["dd"]
dq1 = b["new"]["dq"]
assert len(dd1) == len(sig) and len(dq1) == len(sig)
assert [c["sigma_n"] for c in b["o2_cells"]] == sig

x = list(range(len(sig)))
amber, blue, grey = "#BA7517", "#185FA5", "#888780"
fig, ax = plt.subplots(figsize=(7.2, 3.3))

ax.axhline(0.0, color=grey, lw=0.9, ls=":", zorder=1)
ax.axvline(sig.index(0.172), color="#5F5E5A", lw=0.9, ls="--", alpha=0.7, zorder=1)

ax.plot(x, dd0, "-o", color=amber, lw=1.8, ms=5, label="data term, first pool")
ax.plot(x, dd1, "--o", color=amber, lw=1.8, ms=5, mfc="white",
        label="data term, disjoint pool")
ax.plot(x, dq0, "-^", color=blue, lw=1.8, ms=6, label="prior term, first pool")
ax.plot(x, dq1, "--^", color=blue, lw=1.8, ms=6, mfc="white",
        label="prior term, disjoint pool")

ax.annotate("prior term first negative", xy=(2, dq1[2]), xytext=(2.35, -2850),
            fontsize=8, color="#5F5E5A",
            arrowprops=dict(arrowstyle="-", color="#5F5E5A", lw=0.7))

ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in sig], fontsize=8)
ax.set_xlabel(r"observation noise $\sigma_n$", fontsize=9)
ax.set_ylabel(r"$\Delta X = X(2.5)-X(1.5)$  (nats)", fontsize=9)
ax.set_ylim(-3200, 6100)
ax.tick_params(axis="y", labelsize=8)
ax.grid(axis="y", color="#E8E6E0", lw=0.6, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="upper right", ncol=2, fontsize=7.5, frameon=False)

fig.tight_layout()
for ext in ("pdf", "png"):
    p = os.path.join(OUT, "fig_overblur_terms." + ext)
    fig.savefig(p, dpi=200 if ext == "png" else None)
    print("[saved]", p)
print("first pool  : ddata %.1f -> %.1f, dq %.1f -> %.1f" % (dd0[0], dd0[-1], dq0[0], dq0[-1]))
print("disjoint    : ddata %.1f -> %.1f, dq %.1f -> %.1f" % (dd1[0], dd1[-1], dq1[0], dq1[-1]))
