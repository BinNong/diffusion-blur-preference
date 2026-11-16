"""twod 定律的图：1x3。
(a) C vs twod——统一定律覆盖两个 regime（28 基线 + 8 新族）；
(b) C vs tw_lev——底座剖面上的失败（对照）；
(c) twod-tw_lev 平面按 dyn 上色——两个 regime 如何塌缩到同一根轴。

用法： venv2/bin/python plot_twod.py
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
    "ptod", os.path.join(PROJ, "repro", "predict_twod.py"))
ptod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ptod)
hd = ptod.hd

BLUE, ORANGE, GREEN, PURPLE, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#8a8f98"
INK = "#2f333a"
NEW = ["W6", "TK80", "RQ400", "TK95", "W15", "RQ200", "EF100", "LG150"]
PED = ["RQ400", "RQ200", "EF100", "LG150"]


def main():
    prof, meta = ptod.load_all()
    r = hd.halfplane()
    seg_u, seg_b = [], []
    for a, b in sorted(hd.P1_RANGES, reverse=True):
        bs = []
        for k in hd.SMOOTH_OLD:
            nm = f"{k[0]}/{k[1]:g}"
            va, vb = hd.interp(prof[nm], a), hd.interp(prof[nm], b)
            if va and vb and va > 0 and vb > 0:
                bs.append(math.log(vb / va) / math.log(b / a))
        if len(bs) >= 5:
            seg_u.append(math.sqrt(a * b)); seg_b.append(float(np.mean(bs)))
    seg_u = np.array(seg_u); seg_b = np.array(seg_b)

    def b_of(u):
        if u <= seg_u[0]:
            return seg_b[0]
        if u >= seg_u[-1]:
            return seg_b[-1]
        lx = math.log(u)
        for i in range(len(seg_u) - 1):
            if seg_u[i] <= u <= seg_u[i + 1]:
                w = (lx - math.log(seg_u[i])) / \
                    (math.log(seg_u[i + 1]) - math.log(seg_u[i]))
                return seg_b[i] * (1 - w) + seg_b[i + 1] * w
        return seg_b[-1]

    Ucom = None
    for nm in prof:
        s = {round(c[0], 4) for c in prof[nm]}
        Ucom = s if Ucom is None else (Ucom & s)
    Ucom = sorted(Ucom, reverse=True)
    Uall = sorted({round(c[0], 3) for nm in prof for c in prof[nm]}, reverse=True)
    C, mid = {}, {}
    for nm in prof:
        C[nm] = {}
        for u in Uall:
            v = hd.interp(prof[nm], u)
            if v and v > 0:
                C[nm][u] = v * u ** (-b_of(u))
    for u in Uall:
        vs = [C[f"{k[0]}/{k[1]:g}"][u] for k in hd.SMOOTH_OLD
              if u in C.get(f"{k[0]}/{k[1]:g}", {})]
        if len(vs) >= 5:
            mid[u] = float(np.exp(np.mean([math.log(v) for v in vs])))

    def crel(nm):
        vs = [C[nm][u] / mid[u] for u in Ucom if u in C[nm] and u in mid]
        return float(np.median(vs)) if vs else float("nan")

    desc = {}
    for nm in prof:
        m = meta[nm]
        a = (1.0 - hd.real_g2(m["fam"], m["th"])) if m["kind"] == "real" \
            else (1.0 - hd.make_profile(m["kind"], m["par"], m["scale"], r))
        a = a * (0.60 / float(a.mean()))
        d = hd.descriptors(a, r)
        d["dyn"] = float(d["_amax"] - d["_amin"]) / float(a.mean())
        d["twod"] = d["tw_lev"] / d["dyn"] if d["dyn"] > 1e-9 else np.nan
        desc[nm] = d

    names = sorted(prof)
    y = np.array([crel(n) for n in names])
    xt = np.array([desc[n]["twod"] for n in names])
    xl = np.array([desc[n]["tw_lev"] for n in names])
    ft, x0t, p_ = ptod.fit_sat(xt, y)
    fl, x0l, pl = ptod.fit_sat(xl, y)

    # 新族
    p = os.path.join(OUT, "sweep_twod.json")
    new = {}
    if os.path.exists(p):
        d = json.load(open(p))
        nprof = defaultdict(list)
        nmeta = {}
        for rec in d["rows"]:
            m = rec["meta"]
            nprof[m["fid"]].append((1.0 - rec["kunif"], rec["lam_mean"]))
            nmeta[m["fid"]] = m
        for k in nprof:
            nprof[k].sort(key=lambda c: -c[0])
        for fid in NEW:
            if fid not in nprof:
                continue
            m = nmeta[fid]
            a = 1.0 - hd.make_profile(m["kind"], m["par"], m["scale"], r)
            a = a * (0.60 / float(a.mean()))
            dd = hd.descriptors(a, r)
            dyn = float(dd["_amax"] - dd["_amin"]) / float(a.mean())
            vs = []
            for u in Ucom:
                v = hd.interp(nprof[fid], u)
                if v and v > 0 and u in mid:
                    vs.append(v * u ** (-b_of(u)) / mid[u])
            new[fid] = dict(twod=dd["tw_lev"] / dyn, twl=dd["tw_lev"], dyn=dyn,
                            c=float(np.median(vs)) if vs else float("nan"))

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4))
    for ax in axes:
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # ---------------- (a) C vs twod ----------------
    ax = axes[0]
    tt = np.logspace(math.log10(0.010), math.log10(40.0), 400)
    cc = ft["coef"][0] + ft["coef"][1] / (1.0 + (tt / x0t) ** (-p_))
    ax.fill_between(tt, cc * 10 ** -0.038, cc * 10 ** 0.038, color=GRAY,
                    alpha=0.16, lw=0, label=r"law $\pm$0.038 dex (LOO tol.)")
    ax.plot(tt, cc, color=GRAY, lw=2.0, zorder=2,
            label=r"$C=%.3f+%.3f/[1+(twod/%.2f)^{-%.2f}]$, $R^2$=%.3f"
                  % (ft["coef"][0], ft["coef"][1], x0t, p_, ft["r2"]))
    ax.scatter(xt, y, s=42, color=BLUE, zorder=3, edgecolor="white", lw=0.8,
               label="28 measured profiles")
    if new:
        for fid in NEW:
            if fid not in new:
                continue
            mk = "D" if fid in PED else "o"
            ax.scatter([new[fid]["twod"]], [new[fid]["c"]], s=72, color=ORANGE,
                       zorder=4, marker=mk, edgecolor="white", lw=0.9)
            dx, dy = (8, 3)
            if fid == "TK80":
                dx, dy = (-42, 6)
            if fid == "TK95":
                dx, dy = (6, -12)
            if fid == "RQ400":
                dx, dy = (-14, -16)
            if fid == "W15":
                dx, dy = (-30, 8)
            ax.annotate(fid, (new[fid]["twod"], new[fid]["c"]),
                        textcoords="offset points", xytext=(dx, dy), fontsize=7.4,
                        color=INK)
        ax.scatter([], [], s=72, color=ORANGE, marker="o", edgecolor="white",
                   lw=0.9, label="4 new passband")
        ax.scatter([], [], s=72, color=ORANGE, marker="D", edgecolor="white",
                   lw=0.9, label="4 new pedestal")
    ax.set_xscale("log")
    ax.set_xlabel(r"$tw_{\rm od}=tw_{\rm lev}/dyn$")
    ax.set_ylabel(r"$C$(shape) / smooth-family mean")
    ax.set_title("(a) one law, both regimes", fontsize=10.5, loc="left")
    ax.legend(fontsize=7.0, loc="lower right", framealpha=0.93)

    # ---------------- (b) C vs tw_lev ----------------
    ax = axes[1]
    ax.plot(xl, fl["coef"][0] + fl["coef"][1] / (1.0 + (xl / x0l) ** (-pl)),
            color=GREEN, lw=2.0, ls="--", zorder=2,
            label=r"$tw_{\rm lev}$ law, $R^2$=%.3f" % fl["r2"])
    ax.scatter(xl, y, s=42, color=BLUE, zorder=3, edgecolor="white", lw=0.8,
               label="28 measured profiles")
    if new:
        for fid in NEW:
            if fid not in new:
                continue
            ax.scatter([new[fid]["twl"]], [new[fid]["c"]], s=72, color=ORANGE,
                       zorder=4, marker="D" if fid in PED else "o",
                       edgecolor="white", lw=0.9)
            ax.annotate(fid, (new[fid]["twl"], new[fid]["c"]),
                        textcoords="offset points", xytext=(7, 3), fontsize=7.4,
                        color=INK)
    ax.set_xscale("log")
    ax.set_xlabel(r"relative transition width  $tw_{\rm lev}$")
    ax.set_ylabel(r"$C$(shape) / smooth-family mean")
    ax.set_title("(b) $tw_{\\rm lev}$ misses the pedestals", fontsize=10.5,
                 loc="left")
    ax.legend(fontsize=7.2, loc="lower right", framealpha=0.93)

    # ---------------- (c) 残差 vs twod：定律是否在整个范围都充分 ----------------
    ax = axes[2]
    rt = np.log10(y / (ft["coef"][0] + ft["coef"][1] /
                       (1.0 + (xt / x0t) ** (-p_))))
    ax.axhline(0.0, color=GRAY, lw=1.2, zorder=2)
    for v in (0.038, -0.038):
        ax.axhline(v, color=PURPLE, lw=0.9, ls=":", zorder=2)
    ax.scatter(xt, rt, s=44, color=BLUE, zorder=3, edgecolor="white", lw=0.8,
               label="28 measured profiles")
    if new:
        for fid in NEW:
            if fid not in new:
                continue
            x_ = new[fid]["twod"]
            r_ = math.log10(new[fid]["c"] / (ft["coef"][0] + ft["coef"][1] /
                                             (1.0 + (x_ / x0t) ** (-p_))))
            ax.scatter([x_], [r_], s=74, color=ORANGE, zorder=4,
                       marker="D" if fid in PED else "o", edgecolor="white",
                       lw=0.9)
            ax.annotate(fid, (x_, r_), textcoords="offset points",
                        xytext=(7, 3), fontsize=7.4, color=INK)
        ax.scatter([], [], s=74, color=ORANGE, marker="o", edgecolor="white",
                   lw=0.9, label="4 new passband")
        ax.scatter([], [], s=74, color=ORANGE, marker="D", edgecolor="white",
                   lw=0.9, label="4 new pedestal")
    ax.set_xscale("log")
    ax.set_xlabel(r"$tw_{\rm od}=tw_{\rm lev}/dyn$")
    ax.set_ylabel(r"$\log_{10}(C_{\rm meas}/C_{\rm law})$  [dex]")
    ax.set_title("(c) residuals: no trend across 4 decades", fontsize=10.5,
                 loc="left")
    ax.set_ylim(-0.045, 0.052)
    ax.legend(fontsize=7.0, loc="lower right", framealpha=0.93)
    ax.text(0.011, 0.045, r"$\pm$0.038 dex = LOO tolerance", fontsize=7.0,
            color=PURPLE, va="top")

    # 均匀算子（g2 常数）——底座平台的极限，单独标注在 (a) 与 (c)
    pu = os.path.join(OUT, "sweep_uniform.json")
    uni = {}
    if os.path.exists(pu):
        du = json.load(open(pu))
        up = defaultdict(list)
        for rec in du["rows"]:
            up[rec["meta"]["fid"]].append((1.0 - rec["kunif"], rec["lam_mean"]))
        vs = []
        for u in Ucom:
            # 均匀算子每个 u 只有一个点，必须精确命中（hd.interp 需要两个点）
            cur = up["UNI0.4" if abs(u - 0.4) < 1e-6 else "UNI0.65"]
            v = None
            for k, val in cur:
                if abs(k - u) < 1e-9:
                    v = val
                    break
            if v is None:
                v = hd.interp(cur, u)
            if v and v > 0 and u in mid:
                vs.append(v * u ** (-b_of(u)) / mid[u])
        if vs:
            uni["c"] = float(np.median(vs))
            uni["twl"] = float("nan")
    if uni:
        axes[0].scatter([30.0], [uni["c"]], s=150, color=PURPLE, zorder=6,
                        marker="*", edgecolor="white", lw=0.9)
        axes[0].annotate("uniform\n$g_2\equiv c$", (30.0, uni["c"]),
                         textcoords="offset points", xytext=(-58, -4),
                         fontsize=7.6, color=PURPLE, ha="left")
        axes[2].scatter([30.0], [math.log10(uni["c"] / (ft["coef"][0] +
                                                        ft["coef"][1] /
                                                        (1.0 + (30.0 / x0t) ** (-p_))))],
                        s=150, color=PURPLE, zorder=6, marker="*",
                        edgecolor="white", lw=0.9)
        axes[2].annotate("uniform", (30.0, math.log10(
            uni["c"] / (ft["coef"][0] + ft["coef"][1] /
                        (1.0 + (30.0 / x0t) ** (-p_))))),
            textcoords="offset points", xytext=(-46, 10), fontsize=7.6,
            color=PURPLE)

    fig.suptitle("Blur preference: the prefactor $C$ in "
                 r"$\Lambda = C\,u^{\,b(u)}$: $C$ vs the profile's own roll-off steepness",
                 fontsize=11.5, y=1.01)
    fig.tight_layout()
    o = os.path.join(OUT, "fig_twod.png")
    fig.savefig(o, dpi=170, bbox_inches="tight")
    print("[saved]", o)


if __name__ == "__main__":
    main()
