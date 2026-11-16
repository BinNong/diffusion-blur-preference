"""图 1（论文 Figure 1）：模糊偏好 Λ 对保留谱功率 u 的依赖 —— 8 个核、120 图 mean±sd

论文版只读 results/sweep_pshape_n120_BCV.json（t=0.002、nimg=120、putils 真核）；
默认的 sweep_profile_shape.json 是 20 图旧回归输入。
**零 GPU**：参考功率谱用 putils 在 CPU 上重算，并用 JSON 里记录的 kunif_ref 做守卫
（守卫不过就说明 helper 抄错了，图不画）。

面板 (a)  Λ vs u，双对数。4 个典型核族（gaussian/defocus/airy/motion）画彩色，
          另外 4 个（gaussian σ=2 / defocus 2 / motion 7 / ideal low-pass）画细灰线。
          y 轴下端画四次 Hutchinson 重复的试点标准差 388 nats（tdep_kfix.json 的 correct 臂；
          424 nats 是 buggy 臂，见 §3.9(ap1)-5），不把它当作通用误差界。
面板 (b)  8 个参考核的径向功率谱 |ĥ(f)|² —— 说明 (a) 里"各族几乎重合"是有内容的：
          这 8 个形状彼此很不相同。

u = 1 − kunif = ⟨|ĥ|²⟩ 是算子**保留**的谱功率，u→0 = 模糊越重，所以 Λ 随 u 下降。

产出：
  results/fig_phenomenon.json —— 图上每个数（论文表格可追溯）
  results/fig_phenomenon.png / .pdf

用法： venv2/bin/python repro/plot_fig1.py
"""

import argparse
import json
import os
import sys

import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))
OUT = os.path.join(_PROJ, "results")

import torch  # noqa: E402
from putils.convolution import (  # noqa: E402
    AiryDisc, DefocusBlur, GaussianBlur, MotionBlur)

# ---- 与 sweep_profile_shape.py 逐字对齐的常量（memory: A/B 脚本 helper 逐字复制）----
H = W = 64
KS = 15
DEV = torch.device("cpu")          # 只构造核 + 一次 FFT；CPU 即可，且构成独立确认
SWEEP = "sweep_profile_shape.json"

# 4 个上色族 + 8 个全族（REFS 的顺序即配色顺序：固定序分配，不循环）
FAMS4 = [("gaussian", 1.0), ("defocus", 1.0), ("airy", 4.0), ("motion", 3.0)]
FAMS_ALL = [("gaussian", 1.0), ("gaussian", 2.0), ("defocus", 1.0),
            ("defocus", 2.0), ("airy", 4.0), ("motion", 3.0),
            ("motion", 7.0), ("lowpass", 0.25)]
LABEL = {("gaussian", 1.0): "Gaussian $\\sigma$=1",
         ("defocus", 1.0): "Defocus $r$=1",
         ("airy", 4.0): "Airy $a$=4",
         ("motion", 3.0): "Motion $L$=3"}

# Okabe–Ito 色盲安全四色（固定顺序），另配 4 个不同 marker 作二级编码
COLOR = {("gaussian", 1.0): "#0072B2", ("defocus", 1.0): "#D55E00",
         ("airy", 4.0): "#009E73", ("motion", 3.0): "#CC79A7"}
MARK = {("gaussian", 1.0): "o", ("defocus", 1.0): "s",
        ("airy", 4.0): "^", ("motion", 3.0): "D"}
GRAY = "#D3D6DA"          # 面板 (a) 里另外 4 个核（统一一条图例项）

HUTCH_SD = 388.0        # nats，tdep_kfix.json res.correct.hutch.sdiff=388.05
                        # （424.19 是 res.buggy，即错核那一臂；见 §3.9(ap1)-5）
KUNIF_REF_TOL = 1e-9    # 守卫容差：同一构造在 CPU 上应逐位复现


def mode_grid():
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    return np.sqrt(FX ** 2 + FY ** 2), FX, FY


def make_ref(fam, th):
    """参考剖面 ref2(f) = 真实核的 |h(f)|^2，(H, W//2+1) 的 numpy。
    与 sweep_profile_shape.py 的 make_ref 逐字相同（只把 DEV 换成 cpu）。"""
    if fam == "lowpass":
        R, FX, FY = mode_grid()
        Rr = R[:, :W // 2 + 1]
        return (Rr <= th * R.max()).astype(np.float64)
    cls = {"gaussian": GaussianBlur, "defocus": DefocusBlur,
           "airy": AiryDisc, "motion": MotionBlur}[fam]
    k = cls(kernel_size=KS, device=DEV, dtype=torch.float64).kernel(float(th))
    kk = k[None, None] if k.ndim == 2 else k
    kh, kw = kk.shape[-2:]
    pad = torch.zeros((1, 1, H, W), dtype=torch.float64, device=DEV)
    pad[:, :, (H - kh) // 2:(H - kh) // 2 + kh,
        (W - kw) // 2:(W - kw) // 2 + kw] = kk.to(torch.float64)
    return torch.fft.rfft2(pad, dim=(-2, -1)).abs().square()[0, 0]\
        .cpu().numpy().astype(np.float64)


def radial_profile(ref2, nb=24):
    """把 (H, W//2+1) 的功率谱按归一化半径分桶求均值 → (r, ⟨|ĥ|²⟩)。"""
    R = mode_grid()[0]
    Rr = R[:, :W // 2 + 1] / R.max()
    x, v = Rr.ravel(), ref2.ravel()
    idx = np.clip(np.digitize(x, np.linspace(0.0, 1.0, nb + 1)) - 1, 0, nb - 1)
    xs, ys = [], []
    for b in range(nb):
        m = idx == b
        if m.sum():
            xs.append(float(x[m].mean()))
            ys.append(float(v[m].mean()))
    return np.array(xs), np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="fn", default=SWEEP)
    ap.add_argument("--nb", type=int, default=24)
    args = ap.parse_args()
    with open(os.path.join(OUT, args.fn)) as fh:
        d = json.load(fh)
    rows = d["rows"]
    cfg = d["cfg"]
    print(f"[cfg] t={cfg['t']} nimg={cfg['nimg']} N={cfg['N']} ckpt={cfg['ckpt']}")

    # ------------------------------------------------------------------ 守卫
    # 每个参考核在 CPU 上重算 kunif_ref = 1 − ⟨|ĥ|²⟩，须与 JSON 记录逐位一致。
    print("\n===== 守卫：CPU 重算参考核的 kunif_ref 对账 JSON =====")
    refs = {}
    bad = 0
    for fam, th in FAMS_ALL:
        ref2 = make_ref(fam, th)
        kr = float(1.0 - ref2.mean())
        rec = [r for r in rows if r["meta"].get("family") == fam
               and r["meta"].get("theta") == th and "kunif_ref" in r["meta"]]
        kr_json = rec[0]["meta"]["kunif_ref"] if rec else None
        ok = kr_json is not None and abs(kr - kr_json) <= KUNIF_REF_TOL
        bad += (not ok)
        refs[(fam, th)] = ref2
        print(f"  {fam}/{th:<5g} CPU {kr:.12f}  JSON {kr_json}  "
              f"|Δ|={abs(kr - kr_json) if kr_json else float('nan'):.2e}  "
              f"{'✅' if ok else '❌'}")
    if bad:
        print(f"  ⇒ ❌ {bad} 个参考核对不上——helper 与 sweep 脚本不一致，不出图。")
        return
    print("  ⇒ ✅ 全部对上：make_ref 在 CPU 上与 sweep 脚本一致（独立确认）")

    # ------------------------------------------------------------------ 取数
    fam_pts = {}
    for fam, th in FAMS_ALL:
        sel = [r for r in rows if r["part"] in ("A", "C", "V")
               and r["meta"].get("family") == fam and r["meta"].get("theta") == th]
        pts = sorted((1.0 - r["kunif"], r["lam_mean"], r["lam_std"], r["part"])
                     for r in sel)
        fam_pts[(fam, th)] = pts
        print(f"  {LABEL.get((fam, th), fam + '/' + str(th)):<18s} "
              f"{len(pts)} 点，u∈[{pts[0][0]:.4f},{pts[-1][0]:.4f}]，"
              f"Λ∈[{min(p[1] for p in pts):.0f},{max(p[1] for p in pts):.0f}]")

    # 同 u 处各族的离散度（论文要报的数）。按 u 分桶（间隔 <0.03 视为同档），
    # 每档只数**不同族**，至少 4 个族才报。
    buckets = []
    for fam, th in FAMS_ALL:
        for u, lm, sd, _ in fam_pts[(fam, th)]:
            for b in buckets:
                if abs(b[0] - u) < 0.03:
                    b[1][(fam, th)] = lm
                    break
            else:
                buckets.append((u, {(fam, th): lm}))
    spread = []
    for uc, m in sorted(buckets):
        if len(m) >= 4:
            vs = list(m.values())
            spread.append((uc, max(vs) / min(vs), len(m)))
    print("\n===== 同 u 处各族 Λ 的离散度（跨族 max/min）=====")
    for u, sp, n in spread:
        print(f"  u≈{u:.3f}：{n} 个不同族，max/min = {sp:.3f}×")
    if spread:
        rs = sorted(s[1] for s in spread)
        print(f"  ⇒ {len(spread)} 个 u 档，max/min ∈ [{rs[0]:.3f}, {rs[-1]:.3f}]，"
              f"中位数 {rs[len(rs) // 2]:.3f}")
        print(f"  ⇒ 只有 4 个族的档（最低 u 端）："
              f"{[(round(s[0], 3), round(s[1], 3)) for s in spread if s[2] == 4]}")

    # ------------------------------------------------------------------ 落盘
    fig = {
        "cfg": {"t": cfg["t"], "nimg": cfg["nimg"], "ckpt": cfg["ckpt"],
                "sweep": args.fn, "hutch_sd": HUTCH_SD, "nb": args.nb},
        "series": [{
            "family": f, "theta": t, "label": LABEL.get((f, t), f + "/" + str(t)),
            "highlight": (f, t) in FAMS4,
            "kunif_ref": 1.0 - float(refs[(f, t)].mean()),
            "points": [{"u": u, "lam": lm, "sd": sd, "part": p}
                       for u, lm, sd, p in fam_pts[(f, t)]],
            "radial": [{"r": float(a), "p": float(b)}
                       for a, b in zip(*radial_profile(refs[(f, t)], args.nb))],
        } for f, t in FAMS_ALL],
        "spread": [{"u": u, "max_over_min": sp, "n": n} for u, sp, n in spread],
    }
    with open(os.path.join(OUT, "fig_phenomenon.json"), "w") as fh:
        json.dump(fig, fh, indent=1)
    print(f"\n[写出] results/fig_phenomenon.json")

    # ------------------------------------------------------------------ 画图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8.0,
        "axes.linewidth": 0.7, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0, "lines.linewidth": 1.6,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    })
    figp, (axa, axb) = plt.subplots(
        1, 2, figsize=(7.16, 3.05), gridspec_kw={"width_ratios": [1.5, 1.0],
                                                 "wspace": 0.30})

    def kill_minor_labels(ax):
        """对数轴的次要刻度标签默认会自动出现（范围 <2 decade 时），一律关掉。"""
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # ---- (a) Λ vs u ----
    for fam, th in FAMS_ALL:
        pts = fam_pts[(fam, th)]
        u = [p[0] for p in pts]
        lm = [p[1] for p in pts]
        sd = [p[2] for p in pts]
        hi = (fam, th) in FAMS4
        if hi:
            c = COLOR[(fam, th)]
            axa.errorbar(u, lm, yerr=sd, fmt="none", ecolor=c, elinewidth=0.9,
                         capsize=2.0, capthick=0.9, zorder=3)
            axa.plot(u, lm, "-", color=c, lw=1.6, zorder=4)
            axa.plot(u, lm, MARK[(fam, th)], color=c, ms=4.6, mec="white",
                     mew=0.7, zorder=5, label=LABEL[(fam, th)])
        else:
            axa.plot(u, lm, "-", color=GRAY, lw=0.9, zorder=2,
                     label="4 further kernels" if (fam, th) == FAMS_ALL[1] else None)

    axa.axhline(HUTCH_SD, color="#8A8F94", lw=0.8, ls=(0, (4, 2.5)), zorder=1)
    axa.text(0.023, HUTCH_SD * 1.14, "Hutchinson pilot repeatability\n"
             f"4-repeat sd($\\Lambda$) = {HUTCH_SD:.0f} nats", fontsize=6.4,
             color="#5A5F64", ha="left", va="bottom", linespacing=1.25)
    axa.set_xscale("log")
    axa.set_yscale("log")
    axa.set_xlim(0.02, 0.95)
    axa.set_ylim(2.5e2, 7.0e4)
    axa.set_xticks([0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8])
    axa.set_xticklabels(["0.03", "0.05", "0.1", "0.2", "0.3", "0.5", "0.8"])
    axa.set_yticks([3e2, 1e3, 3e3, 1e4, 3e4])
    axa.set_yticklabels(["300", "1k", "3k", "10k", "30k"])
    axa.set_xlabel("retained spectral power  $u=\\langle|\\hat h|^2\\rangle$")
    axa.set_ylabel("$\\Lambda = \\log p_t(h\\star x)-\\log p_t(x)$  (nats)")
    axa.set_title("(a)  blur preference vs. blur strength, "
                  "$t$=0.002, %d images" % cfg["nimg"], pad=3.5)
    axa.grid(True, which="major", axis="y", color="#E6E8EA", lw=0.6, zorder=0)
    axa.set_axisbelow(True)
    axa.annotate("", xy=(0.026, 0.44), xytext=(0.10, 0.44),
                 xycoords=axa.get_xaxis_transform(),
                 arrowprops=dict(arrowstyle="<-", lw=0.7, color="#5A5F64"))
    axa.text(0.108, 0.44, "stronger blur", fontsize=6.6, color="#5A5F64",
             va="center", ha="left", transform=axa.get_xaxis_transform())
    # 本图的定量结论：同 u 处跨 8 个物理核的离散度（不留给正文）
    rs = sorted(s[1] for s in spread) if spread else [float("nan")] * 2
    axa.text(0.975, 0.86, f"at matched $u$, 8 kernels agree\nto within "
             f"{rs[0]:.2f}–{rs[-1]:.2f}$\\times$",
             transform=axa.transAxes, fontsize=6.8, color="#3A3F44",
             ha="right", va="center", linespacing=1.35)
    # 图例不放面板级：面板 (a) 右上角要留给定量结论，且 (b) 的四条曲线
    # 必须能用同一个图例说明——挂在本面板下方会被读成只属于 (a)。
    # 改为图级横排，居中跨两个面板（与 plot_fig2.py 同一做法）。

    # ---- (b) 参考功率谱：只画 4 个上色核（灰线锯齿重、6 个 decade 上不可读）----
    for fam, th in FAMS4:
        s = [x for x in fig["series"]
             if x["family"] == fam and x["theta"] == th][0]
        r = [p["r"] for p in s["radial"]]
        p = [p["p"] for p in s["radial"]]
        axb.plot(r, p, "-", color=COLOR[(fam, th)], lw=1.5, zorder=4)
    axb.set_yscale("log")
    axb.set_xlim(0, 1.0)
    axb.set_ylim(3e-7, 2.0)
    axb.set_xlabel("normalised radial frequency  $r/r_{\\max}$")
    axb.set_ylabel("radially binned $\\langle|\\hat h|^2\\rangle$")
    axb.set_title("(b)  the four reference kernel spectra (colours as in (a))",
                  pad=3.5)
    axb.grid(True, which="major", axis="y", color="#E6E8EA", lw=0.6, zorder=0)
    axb.set_axisbelow(True)
    kill_minor_labels(axa)
    kill_minor_labels(axb)

    figp.align_ylabels([axa, axb])
    # 图级图例：5 项一排，居中于整张图下方，两个面板共用
    h, l = axa.get_legend_handles_labels()
    figp.subplots_adjust(bottom=0.215)
    figp.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=5,
                frameon=False, fontsize=7.0, handlelength=1.9,
                columnspacing=1.6, labelspacing=0.35)
    for ext in ("png", "pdf"):
        figp.savefig(os.path.join(OUT, f"fig_phenomenon.{ext}"),
                     dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("[写出] results/fig_phenomenon.png / .pdf")


if __name__ == "__main__":
    main()
