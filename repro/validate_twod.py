"""验证 twod = tw_lev/dyn：语义单调性、分 regime 表现、逐剖面残差。

动机（hunt2.py）
----------------
28 个已测剖面上，用**饱和逻辑斯蒂**形式（定律本来就是饱和的，线性 R2 会冤枉它）：
    twod = tw_lev/dyn   R2sat=0.911  Spear=0.85  LOO=0.0144 dex  最紧bin=1.008
    tw_lev              R2sat=0.827  Spear=0.69  LOO=0.0199 dex  最紧bin=1.008
    B2                  R2sat=0.457  Spear=-0.79 LOO=0.0389 dex  最紧bin=1.475
且 twod 的拟合饱和值 C_inf = 1.065，正好被当初打穿 tw_lev 的 EF60(1.067)/LG60(1.070) 锚住。

物理读法：1/twod = r50 * (dyn/w_abs) = 特征半径 x 剖面在自身尺度上的总变化量（以 <a> 为单位）。
硬截止 => 变化量巨大 => twod→0 => C→0.69；缓坡 => twod→∞ => C→1.065。
尺度不变（tw_lev 与 dyn 都尺度不变）。

本脚本只做**离线验证**（不碰 GPU）：
  V1 语义单调性：7 个族扫形状参数，twod 是否朝期望方向动。
  V2 分 regime：只在 22 个有通带剖面上拟合，再去预言 5 个底座剖面（这是真正的样本外）。
  V3 逐剖面残差：twod 定律在 28 个剖面上哪里还不行。
  V4 与 tw_lev 的直接对照：同一拟合形式、同一数据。

用法： venv2/bin/python validate_twod.py
"""

import importlib.util
import json
import math
import os
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
sys_path = os.path.join(PROJ, "code", "blind_deblur_diffusion")
import sys
sys.path.insert(0, sys_path)

_spec = importlib.util.spec_from_file_location(
    "pt", os.path.join(PROJ, "repro", "predict_tw.py"))
pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pt)
hd = pt.hd


def h2_load_all():
    """hunt2 的加载逻辑（5 个数据文件 => 28 个剖面）。"""
    import json as _json
    from collections import defaultdict as _dd
    prof, meta = _dd(list), {}
    for fn in pt.os.listdir(os.path.join(PROJ, "results")):
        pass
    for fn in ["sweep_profile_shape.json", "sweep_law_test.json",
               "sweep_steepness.json", "sweep_b2test.json", "sweep_twlaw.json"]:
        fp = os.path.join(OUT, fn)
        if not os.path.exists(fp):
            continue
        d = _json.load(open(fp))
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

PED = {"RP75", "LG35", "RP35", "EF60", "LG60"}     # phi > 0.3 的底座剖面


def fit_sat(x, y):
    """C = a + b/(1+(x/x0)^(-p))；网格搜 x0,p。返回 (fit, x0, p)。"""
    bg = None
    for x0 in np.linspace(0.02, 8.0, 200):
        for p in np.linspace(0.2, 4.0, 60):
            sig = 1.0 / (1.0 + (x / x0) ** (-p))
            X = np.column_stack([np.ones(len(x)), sig])
            f = hd.ols(X, y)
            if f and (bg is None or f["rmse"] < bg[0]):
                bg = (f["rmse"], f, x0, p)
    return bg[1], bg[2], bg[3]


def main():
    prof, meta = h2_load_all()
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

    araw, desc = {}, {}
    for nm in prof:
        m = meta[nm]
        if m["kind"] == "real":
            a = 1.0 - hd.real_g2(m["fam"], m["th"])
        else:
            a = 1.0 - hd.make_profile(m["kind"], m["par"], m["scale"], r)
        a = a * (0.60 / float(a.mean()))
        araw[nm] = a
        d = hd.descriptors(a, r)
        m_ = float(a.mean())
        d["dyn"] = float(d["_amax"] - d["_amin"]) / m_
        d["phi"] = float(d["_amin"]) / m_
        d["twod"] = d["tw_lev"] / d["dyn"] if d["dyn"] > 1e-9 else np.nan
        desc[nm] = d

    names = sorted(prof)
    y = np.array([crel(n) for n in names])

    # ---------------- V1 语义单调性 ----------------
    print("=" * 104)
    print(" V1 语义检查：twod 是否随'过渡带变宽'单调变化（不涉及任何 Λ）")
    print("=" * 104)
    sweeps = [("erf 边宽 w", "erf", [0.10, 0.20, 0.35, 0.60, 1.00, 1.50], +1),
              ("logistic 边宽 w", "logistic", [0.10, 0.20, 0.35, 0.60, 1.00, 1.50], +1),
              ("tukey 过渡占比", "tukey", [0.10, 0.30, 0.50, 0.70, 0.90], +1),
              ("weibull 陡峭度 p", "weibull", [0.5, 1, 2, 4, 8, 16, 32], -1),
              ("butterworth 陡峭度 n", "butterworth", [1, 2, 4, 8, 16], -1),
              ("ramp 幂 q", "ramp", [0.25, 0.5, 1.0, 2.0, 4.0], -1),
              ("twoscale 尺度比 R", "twoscale", [1.2, 1.5, 2.0, 3.0, 5.0], +1)]
    tot = [0, 0]
    for label, kind, pars, direction in sweeps:
        vals = []
        for par in pars:
            sc = pt.tune(kind, par, r, 0.92)
            a = 1.0 - hd.make_profile(kind, par, sc, r)
            a = a * (0.60 / float(a.mean()))
            d = hd.descriptors(a, r)
            d["dyn"] = float(d["_amax"] - d["_amin"]) / float(a.mean())
            d["twod"] = d["tw_lev"] / d["dyn"] if d["dyn"] > 1e-9 else np.nan
            vals.append((par, d["twod"]))
        seq = [(p, v) for p, v in vals if np.isfinite(v)]
        bad = sum(1 for i in range(len(seq) - 1)
                  if (seq[i + 1][1] - seq[i][1]) * direction < -1e-9)
        tot[1] += max(len(seq) - 1, 0); tot[0] += max(len(seq) - 1, 0) - bad
        print(f" {label:<20} twod: " + "  ".join(f"{p:g}→{v:.3f}" for p, v in seq)
              + f"   违例 {bad} {'✅' if bad == 0 else '❌'}")
    print(f"\n => twod 语义单调性 {tot[0]}/{tot[1]} 通过"
          f"{'（与 tw_lev 的 32/32 同级）' if tot[0] == tot[1] else ''}")

    # ---------------- V2 分 regime：只用有通带剖面拟合，预言底座剖面 ----------------
    print("\n" + "=" * 104)
    print(" V2 真正的样本外：只用 22 个有通带剖面拟合，再去预言 5 个底座剖面")
    print("=" * 104)
    pas = [n for n in names if n not in PED]
    ped = [n for n in names if n in PED]
    xp = np.array([desc[n]["twod"] for n in pas])
    yp = np.array([crel(n) for n in pas])
    f_p, x0, p = fit_sat(xp, yp)
    print(f" 有通带 {len(pas)} 个拟合: C = {f_p['coef'][0]:.3f} + "
          f"{f_p['coef'][1]:.3f}/(1+(twod/{x0:.3f})^(-{p:.2f}))，R2={f_p['r2']:.3f}")
    print(f" 外推饱和值 C_inf = {f_p['coef'][0]+f_p['coef'][1]:.3f}")
    print(f"\n {'底座剖面':<10} | {'twod':>7} | {'C实测':>7} | {'C预言':>7} | {'残差dex':>8}")
    print(" " + "-" * 52)
    for n in sorted(ped, key=lambda n: desc[n]["twod"]):
        x_ = desc[n]["twod"]
        cp = f_p["coef"][0] + f_p["coef"][1] / (1.0 + (x_ / x0) ** (-p))
        d = math.log10(crel(n) / cp)
        print(f" {n:<10} | {x_:>7.3f} | {crel(n):>7.3f} | {cp:>7.3f} | {d:>+8.3f}")
    print(" " + "-" * 52)
    print(" 对照：当初 tw_lev 在同一检验上 EF60 残差 +0.064 dex（16%，6-10σ 失败）")

    # ---------------- V3 逐剖面残差 ----------------
    print("\n" + "=" * 104)
    print(" V3 twod 定律在全部 28 个剖面上的逐点残差（in-sample）")
    print("=" * 104)
    xa = np.array([desc[n]["twod"] for n in names])
    f_a, x0a, pa = fit_sat(xa, y)
    print(f" 全 28 个拟合: C = {f_a['coef'][0]:.3f} + "
          f"{f_a['coef'][1]:.3f}/(1+(twod/{x0a:.3f})^(-{pa:.2f}))，R2={f_a['r2']:.3f}")
    l1, l2 = hd.loo(np.column_stack(
        [np.ones(len(xa)), 1.0 / (1.0 + (xa / x0a) ** (-pa))]), y)
    print(f" LOO RMSE {l1/float(np.mean(np.abs(y)))/math.log(10):.4f} dex，"
          f"LOO 最大 {l2/float(np.mean(np.abs(y)))/math.log(10):.4f} dex")
    print(f"\n {'剖面':<12} | {'twod':>7} | {'C实测':>7} | {'C定律':>7} | {'残差dex':>8}")
    print(" " + "-" * 56)
    for n in sorted(names, key=lambda n: desc[n]["twod"]):
        x_ = desc[n]["twod"]
        cl = f_a["coef"][0] + f_a["coef"][1] / (1.0 + (x_ / x0a) ** (-pa))
        d = math.log10(crel(n) / cl)
        flag = "  <== 底座" if n in PED else ""
        print(f" {n:<12} | {x_:>7.3f} | {crel(n):>7.3f} | {cl:>7.3f} | {d:>+8.3f}{flag}")

    # ---------------- V4 与 tw_lev 直接对照 ----------------
    print("\n" + "=" * 104)
    print(" V4 同一形式下 twod vs tw_lev（全 28 个 + 只用有通带 22 个）")
    print("=" * 104)
    print(f" {'数据':<14} | {'描述量':<8} | {'R2':>7} | {'LOOdex':>8} | {'C_inf':>7}")
    print(" " + "-" * 56)
    for tag, ns in [("全 28 个", names), ("有通带 22 个", pas)]:
        yy = np.array([crel(n) for n in ns])
        for key in ["twod", "tw_lev", "wod", "w_abs"]:
            xx = np.array([desc[n][key] for n in ns])
            if (xx <= 0).any() or not np.isfinite(xx).all():
                continue
            ff, x0_, p_ = fit_sat(xx, yy)
            ll, _ = hd.loo(np.column_stack(
                [np.ones(len(xx)), 1.0 / (1.0 + (xx / x0_) ** (-p_))]), yy)
            print(f" {tag:<14} | {key:<8} | {ff['r2']:>7.3f} | "
                  f"{ll/float(np.mean(np.abs(yy)))/math.log(10):>8.4f} | "
                  f"{ff['coef'][0]+ff['coef'][1]:>7.3f}")
    print("\n 结论要等预官检验：这一步全是 in-sample。")


if __name__ == "__main__":
    main()
