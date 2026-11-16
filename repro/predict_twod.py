"""为 twod 定律的预官检验挑选对抗性新族，并计算预官值（只看剖面定义，不涉及 Λ）。

背景（validate_twod.py）
------------------------
28 个已测剖面上，饱和形式 C = a + b/(1+(twod/x0)^(-p))：
    twod = tw_lev/dyn    R2=0.911  Spear=0.85  LOO=0.0144 dex  最紧bin=1.008
    tw_lev               R2=0.827  Spear=0.69  LOO=0.0199 dex
    B2                   R2=0.457  Spear=-0.79 LOO=0.0389 dex  最紧bin=1.475
V1 语义单调性 32/32 通过。V2 只用 23 个有通带剖面拟合、预言 5 个底座剖面，残差 <=0.038 dex
（EF60 从 tw_lev 的 +0.064 dex 降到 +0.018 dex）。

但 twod 是**看着全部 28 个剖面设计出来的**，所以 V2 只是弱样本外。要变成结论必须再跑预官。

物理读法：1/twod = r50 * dyn/w_abs = 特征半径 x 剖面在自身尺度上的总变化量（以 <a> 为单位）。
硬截止 => 变化量巨大 => twod→0 => C→0.70；缓坡 => twod→∞ => C→1.065。
尺度不变（tw_lev 与 dyn 都尺度不变）。

本脚本：
  1. 在 28 个已测剖面上冻结 twod 定律与 tw_lev 定律。
  2. 对一批**从未测过**的候选族算 twod / tw_lev / dyn / phi 与两条定律的预言。
  3. 标出**对抗性**族：两条定律预言分歧大、且 twod 落在敏感区（0.2-2.5）。

用法： venv2/bin/python predict_twod.py
"""

import importlib.util
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
sys.path.insert(0, os.path.join(PROJ, "code", "blind_deblur_diffusion"))

_spec = importlib.util.spec_from_file_location(
    "hd", os.path.join(PROJ, "repro", "hunt_descriptor.py"))
hd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hd)

_spec2 = importlib.util.spec_from_file_location(
    "pt", os.path.join(PROJ, "repro", "predict_tw.py"))
pt = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(pt)

FILES = ["sweep_profile_shape.json", "sweep_law_test.json",
         "sweep_steepness.json", "sweep_b2test.json", "sweep_twlaw.json"]
KREF = 0.92
T = 0.60
PED = {"RP75", "LG35", "RP35", "EF60", "LG60"}

# 候选新族：(id, kind, par, 备注)
CAND = [
    # --- 边宽扫（erf/logistic 的宽边会自然产生底座，twod 与 tw_lev 在这里分歧最大）---
    ("EF10", "erf", 0.10, "erf 窄边"),
    ("EF20", "erf", 0.20, "erf 边 w=0.2"),
    ("EF100", "erf", 1.00, "erf 宽边（底座）"),
    ("EF150", "erf", 1.50, "erf 极宽边（近均匀）"),
    ("LG10", "logistic", 0.10, "logistic 窄边"),
    ("LG100", "logistic", 1.00, "logistic 宽边（底座）"),
    ("LG150", "logistic", 1.50, "logistic 极宽边"),
    # --- ramp：q 大 => 衰减集中在 DC 附近，q 小 => 近均匀底座 ---
    ("RQ25", "ramp", 0.25, "幂律斜坡 q=0.25"),
    ("RQ40", "ramp", 0.40, "幂律斜坡 q=0.40"),
    ("RQ100", "ramp", 1.00, "幂律斜坡 q=1"),
    ("RQ200", "ramp", 2.00, "幂律斜坡 q=2"),
    ("RQ400", "ramp", 4.00, "幂律斜坡 q=4"),
    # --- tukey：全新形状类的更多参数 ---
    ("TK15", "tukey", 0.15, "升余弦 过渡占 15%"),
    ("TK40", "tukey", 0.40, "升余弦 过渡占 40%"),
    ("TK60", "tukey", 0.60, "升余弦 过渡占 60%"),
    ("TK80", "tukey", 0.80, "升余弦 过渡占 80%"),
    ("TK95", "tukey", 0.95, "升余弦 过渡占 95%"),
    # --- 填已测类的参数缺口 ---
    ("W15", "weibull", 1.5, "Weibull p=1.5"),
    ("W6", "weibull", 6.0, "Weibull p=6"),
    ("W12", "weibull", 12.0, "Weibull p=12"),
    ("B3n", "butterworth", 3.0, "Butterworth n=3"),
    ("B6n", "butterworth", 6.0, "Butterworth n=6"),
    ("B12n", "butterworth", 12.0, "Butterworth n=12"),
    ("TS12", "twoscale", 1.2, "双尺度 R=1.2"),
    ("TS40", "twoscale", 4.0, "双尺度 R=4"),
    ("TS80", "twoscale", 8.0, "双尺度 R=8"),
]


def load_all():
    prof, meta = defaultdict(list), {}
    for fn in FILES:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for rec in d["rows"]:
            m = rec["meta"]
            nm = m.get("fid") or (f"{m['family']}/{m['theta']:g}"
                                  if "family" in m else None)
            if nm is None:
                continue
            prof[nm].append((1.0 - rec["kunif"], rec["lam_mean"]))
            if nm not in meta:
                if "kind" in m and m["kind"] != "real":
                    meta[nm] = dict(kind=m["kind"], par=float(m["par"]),
                                    scale=float(m["scale"]),
                                    kref=float(m["kunif_ref"]))
                else:
                    meta[nm] = dict(kind="real", fam=m["family"],
                                    th=float(m["theta"]))
    for k in prof:
        prof[k].sort(key=lambda c: -c[0])
    return dict(prof), meta


def fit_sat(x, y):
    bg = None
    for x0 in np.linspace(0.02, 8.0, 200):
        for p in np.linspace(0.2, 4.0, 60):
            sig = 1.0 / (1.0 + (x / x0) ** (-p))
            f = hd.ols(np.column_stack([np.ones(len(x)), sig]), y)
            if f and (bg is None or f["rmse"] < bg[0]):
                bg = (f["rmse"], f, x0, p)
    return bg[1], bg[2], bg[3]


def main():
    prof, meta = load_all()
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
        a = a * (T / float(a.mean()))
        d = hd.descriptors(a, r)
        d["dyn"] = float(d["_amax"] - d["_amin"]) / float(a.mean())
        d["twod"] = d["tw_lev"] / d["dyn"] if d["dyn"] > 1e-9 else np.nan
        desc[nm] = d

    names = sorted(prof)
    y = np.array([crel(n) for n in names])
    xt = np.array([desc[n]["twod"] for n in names])
    xl = np.array([desc[n]["tw_lev"] for n in names])
    ft, x0t, pt_ = fit_sat(xt, y)
    fl, x0l, pl = fit_sat(xl, y)
    l1, _ = hd.loo(np.column_stack(
        [np.ones(len(xt)), 1.0 / (1.0 + (xt / x0t) ** (-pt_))]), y)
    print("=" * 112)
    print(f" 基准：{len(names)} 个已测剖面；共同 u 网格 " +
          " ".join(f"{u:.3g}" for u in Ucom))
    print("=" * 112)
    print(f"\n twod 定律: C = {ft['coef'][0]:.3f} + {ft['coef'][1]:.3f}"
          f"/(1+(twod/{x0t:.3f})^(-{pt_:.2f}))，R2={ft['r2']:.3f}，"
          f"LOO {l1/float(np.mean(np.abs(y)))/math.log(10):.4f} dex")
    print(f" tw_lev 定律: C = {fl['coef'][0]:.3f} + {fl['coef'][1]:.3f}"
          f"/(1+(tw/{x0l:.3f})^(-{pl:.2f}))，R2={fl['r2']:.3f}")

    def law_t(x):
        return ft["coef"][0] + ft["coef"][1] / (1.0 + (x / x0t) ** (-pt_))

    def law_l(x):
        return fl["coef"][0] + fl["coef"][1] / (1.0 + (x / x0l) ** (-pl))

    # ---- 候选族 ----
    print("\n" + "=" * 112)
    print(" 候选新族：描述量与两条定律的预言")
    print("=" * 112)
    print(f" {'族':<6} | {'kind':<12} | {'par':>5} | {'twod':>7} | {'tw_lev':>7} | "
          f"{'dyn':>6} | {'phi':>6} | {'C(twod)':>8} | {'C(tw_lev)':>9} | {'分歧dex':>8}")
    print(" " + "-" * 104)
    rows = []
    for fid, kind, par, note in CAND:
        try:
            sc = pt.tune(kind, par, r, KREF)
        except Exception as e:
            print(f" {fid:<6} | tune 失败: {e}")
            continue
        a = 1.0 - hd.make_profile(kind, par, sc, r)
        if a.mean() <= 0 or a.max() > 1.0 + 1e-9:
            print(f" {fid:<6} | 剖面越界，跳过")
            continue
        a = a * (T / float(a.mean()))
        d = hd.descriptors(a, r)
        dyn = float(d["_amax"] - d["_amin"]) / float(a.mean())
        phi = float(d["_amin"]) / float(a.mean())
        twod = d["tw_lev"] / dyn if dyn > 1e-9 else np.nan
        if not np.isfinite(twod) or twod <= 0:
            print(f" {fid:<6} | twod 无定义，跳过")
            continue
        ct, cl = law_t(twod), law_l(d["tw_lev"])
        dis = abs(math.log10(ct / cl))
        rows.append((fid, kind, par, note, twod, d["tw_lev"], dyn, phi, ct, cl, dis))

    # 已测 28 个的 twod 覆盖（找缺口）
    xt_sorted = sorted(xt)
    print("\n 已测 28 个的 twod 覆盖: " + "  ".join(f"{v:.2f}" for v in xt_sorted))
    for fid, kind, par, note, twod, twl, dyn, phi, ct, cl, dis in rows:
        print(f" {fid:<6} | {kind:<12} | {par:>5.2f} | {twod:>7.3f} | {twl:>7.3f} | "
              f"{dyn:>6.3f} | {phi:>6.3f} | {ct:>8.3f} | {cl:>9.3f} | {dis:>8.3f}")

    # ---- 对抗性排序 ----
    print("\n" + "=" * 112)
    print(" 按对抗性排序（两条定律分歧降序；分歧大 => 能区分 twod 与 tw_lev）")
    print("=" * 112)
    for fid, kind, par, note, twod, twl, dyn, phi, ct, cl, dis in \
            sorted(rows, key=lambda t: -t[10]):
        sens = "敏感区" if 0.2 <= twod <= 2.5 else "饱和/低端"
        print(f" {fid:<6} twod={twod:>7.3f} ({sens:<7}) dyn={dyn:.3f} phi={phi:.3f} "
              f"C(twod)={ct:.3f}  C(tw_lev)={cl:.3f}  分歧={dis:.3f} dex   {note}")


if __name__ == "__main__":
    main()
