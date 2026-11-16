"""离线搜第三批描述量：目标是吃掉 kref=0.92 底座剖面的 +0.016 dex 系统偏移。

背景（diag_offset.py）
---------------------
twod 定律的残差按 (knif_ref, regime) 分组：
    kref=0.92 底座  n=6  均值 +0.0160 dex  **全正**
    kref=0.60 底座  n=3  均值 -0.0172 dex  混合
    kref=0.92 通带  n=17 均值 +0.0019 dex  混合
    real      通带  n=8  均值 -0.0010 dex  混合
=> 偏移既不是「新 vs 训练」也不是「底座 vs 通带」单独能解释，而是 **kref=0.92 的底座**。
   且这 6 个的 C 几乎是平台（1.049-1.102，均值 1.070）而 twod 跨 0.89->26.6。

物理线索：kref=0.92 的算子都经过 beta=0.60/0.92=0.652 压缩，所以**传输 g2 有一个
0.348 的地板**——算子永远至少通过 34.8% 的功率。kref=0.60 的算子 beta=1，g2 可以到 0。
所以候选描述量都围绕「传输的相对位置」建：
  gfloor = (1-amax)/<g2>          传输地板的相对高度
  gceil  = (1-amin)/<g2>          传输顶棚的相对高度
  grange = (amax-amin)/<g2>       = dyn*<a>/<g2>，传输的相对动态范围
以及把它们与 tw_lev 组合的各种形式。

评价标准（按重要性）：
  1. kref=0.92 底座组的残差均值是否回到 ~0 且符号混合；
  2. 有通带组（27 个）是否不被破坏；
  3. R2sat / LOO / 匹配处散度。
注意：这是**探索性**的——任何胜出者都只是 in-sample，要变成结论必须再跑预官。

用法： venv2/bin/python hunt3.py
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
    "ptod", os.path.join(PROJ, "repro", "predict_twod.py"))
ptod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ptod)
hd = ptod.hd

FILES = ["sweep_profile_shape.json", "sweep_law_test.json",
         "sweep_steepness.json", "sweep_b2test.json", "sweep_twlaw.json",
         "sweep_twod.json"]


def main():
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
                                    th=float(m["theta"]), kref=float("nan"))
    for k in prof:
        prof[k].sort(key=lambda c: -c[0])

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

    rows = []
    for nm in sorted(prof):
        m = meta[nm]
        a = (1.0 - hd.real_g2(m["fam"], m["th"])) if m["kind"] == "real" \
            else (1.0 - hd.make_profile(m["kind"], m["par"], m["scale"], r))
        a = a * (0.60 / float(a.mean()))
        d = hd.descriptors(a, r)
        ma = float(a.mean())
        amin, amax = float(d["_amin"]), float(d["_amax"])
        dyn = (amax - amin) / ma
        phi = amin / ma
        mg = 1.0 - ma
        gfloor = (1.0 - amax) / mg if mg > 1e-9 else np.nan
        gceil = (1.0 - amin) / mg if mg > 1e-9 else np.nan
        grange = (amax - amin) / mg if mg > 1e-9 else np.nan
        if dyn <= 1e-9 or not np.isfinite(crel(nm)):
            continue
        twod = d["tw_lev"] / dyn
        rows.append(dict(
            nm=nm, c=crel(nm), kref=m["kref"],
            twl=d["tw_lev"], dyn=dyn, phi=phi, twod=twod,
            gfloor=gfloor, gceil=gceil, grange=grange,
            # 候选描述量
            d_twod=twod,
            d_twod_phi=twod / (1.0 + phi),
            d_twod_phi2=twod / (1.0 - phi) if phi < 1 else np.nan,
            d_twod_gf=twod * (1.0 + gfloor),
            d_twod_gc=twod * gceil if gceil > 1e-9 else np.nan,
            d_twod_gr=twod / grange if grange > 1e-9 else np.nan,
            d_twl_gf=d["tw_lev"] * (1.0 + gfloor),
            d_gr=d["tw_lev"] / grange if grange > 1e-9 else np.nan,
        ))
    names = [t["nm"] for t in rows]
    y = np.array([t["c"] for t in rows])
    kr92ped = np.array([(np.isfinite(t["kref"]) and abs(t["kref"] - 0.92) < 0.02
                         and t["phi"] > 0.3) for t in rows])
    kr60ped = np.array([(np.isfinite(t["kref"]) and abs(t["kref"] - 0.60) < 0.02
                         and t["phi"] > 0.3) for t in rows])
    ped = np.array([t["phi"] > 0.3 for t in rows])
    pas = ~ped
    print("=" * 108)
    print(f" {len(rows)} 个剖面；kref=0.92 底座 {kr92ped.sum()} 个，"
          f"kref=0.60 底座 {kr60ped.sum()} 个，有通带 {pas.sum()} 个")
    print("=" * 108)

    # ---- 表 1：候选描述量总表 ----
    keys = ["d_twod", "d_twod_phi", "d_twod_phi2", "d_twod_gf", "d_twod_gc",
            "d_twod_gr", "d_twl_gf", "d_gr"]
    print("\n" + "=" * 108)
    print(" 1. 候选描述量横评（饱和形式）")
    print("=" * 108)
    print(f" {'描述量':<12} | {'R2sat':>7} | {'LOOdex':>8} | {'最紧bin':>8} | "
          f"{'k92底座均值':>11} | {'符号':>5} | {'k60底座':>8} | {'通带均值':>9}")
    print(" " + "-" * 92)
    out = []
    for k in keys:
        x = np.array([t[k] for t in rows])
        if not np.isfinite(x).all() or (x <= 0).any() or x.std() == 0:
            print(f" {k:<12} | NaN / 非正 / 无跨度")
            continue
        f, x0, p = ptod.fit_sat(x, y)
        l1, _ = hd.loo(np.column_stack(
            [np.ones(len(x)), 1.0 / (1.0 + (x / x0) ** (-p))]), y)
        resid = np.log10(y / (f["coef"][0] + f["coef"][1] /
                              (1.0 + (x / x0) ** (-p))))
        order = np.argsort(x); xs, ys = x[order], y[order]
        best = (np.inf, np.nan)
        for w in [0.10, 0.15, 0.20, 0.30]:
            for i in range(len(xs)):
                j = i
                while j < len(xs) and xs[j] - xs[i] <= w:
                    j += 1
                if j - i >= 3:
                    s2 = float(ys[i:j].max() / ys[i:j].min())
                    if s2 < best[0]:
                        best = (s2, w)
        m92 = float(resid[kr92ped].mean()) if kr92ped.any() else float("nan")
        m60 = float(resid[kr60ped].mean()) if kr60ped.any() else float("nan")
        mpa = float(resid[pas].mean())
        allpos = bool((resid[kr92ped] > 0).all()) if kr92ped.any() else False
        print(f" {k:<12} | {f['r2']:>7.3f} | "
              f"{l1/float(np.mean(np.abs(y)))/math.log(10):>8.4f} | {best[0]:>8.3f} | "
              f"{m92:>+11.4f} | {'全正' if allpos else '混合':>5} | {m60:>+8.4f} | "
              f"{mpa:>+9.4f}")
        out.append((k, f["r2"], m92, allpos, mpa, l1, best[0]))
    print(" " + "-" * 92)
    print(" 目标：k92底座均值回到 ~0 且符号混合，同时通带均值不被推离 0。")

    # ---- 表 2：加一个 phi 一次项 ----
    print("\n" + "=" * 108)
    print(" 2. 在 twod 定律上加一个 phi（或 gfloor）的一次项")
    print("=" * 108)
    xt = np.array([t["d_twod"] for t in rows])
    f0, x00, p0 = ptod.fit_sat(xt, y)
    sig = 1.0 / (1.0 + (xt / x00) ** (-p0))
    ph = np.array([t["phi"] for t in rows])
    gf = np.array([t["gfloor"] for t in rows])
    print(f" {'模型':<22} | {'R2':>7} | {'LOOdex':>8} | {'k92底座':>9} | "
          f"{'符号':>5} | {'通带':>8}")
    print(" " + "-" * 72)
    for lab, X in [
            ("twod only", np.column_stack([np.ones(len(xt)), sig])),
            ("twod + phi", np.column_stack([np.ones(len(xt)), sig, ph])),
            ("twod + gfloor", np.column_stack([np.ones(len(xt)), sig, gf])),
            ("twod + phi + gfloor",
             np.column_stack([np.ones(len(xt)), sig, ph, gf])),
            ("sig(phi) only",
             np.column_stack([np.ones(len(xt)), 1.0 / (1.0 + (ph / 0.5) ** (-3.0))]))]:
        f = hd.ols(X, y)
        if not f:
            continue
        l1, _ = hd.loo(X, y)
        resid = y - X @ f["coef"]
        r92 = float(resid[kr92ped].mean()) if kr92ped.any() else float("nan")
        rpa = float(resid[pas].mean())
        allpos = bool((resid[kr92ped] > 0).all()) if kr92ped.any() else False
        print(f" {lab:<22} | {f['r2']:>7.3f} | "
              f"{l1/float(np.mean(np.abs(y)))/math.log(10):>8.4f} | {r92:>+9.4f} | "
              f"{'全正' if allpos else '混合':>5} | {rpa:>+8.4f}")
    print(" " + "-" * 72)
    print(" 注意：加参数必然提高 R2，所以只看 R2 会被骗；要看的是 k92底座那组。")

    # ---- 表 3：kref=0.92 底座的逐点 ----
    print("\n" + "=" * 108)
    print(" 3. kref=0.92 底座剖面逐点：C 是不是平台？")
    print("=" * 108)
    sel = [t for t in rows if kr92ped[names.index(t["nm"])]]
    print(f" {'剖面':<12} | {'twod':>8} | {'phi':>6} | {'gfloor':>7} | {'C':>6}")
    print(" " + "-" * 50)
    for t in sorted(sel, key=lambda t: t["d_twod"]):
        print(f" {t['nm']:<12} | {t['d_twod']:>8.3f} | {t['phi']:>6.3f} | "
              f"{t['gfloor']:>7.3f} | {t['c']:>6.3f}")
    if sel:
        cs = np.array([t["c"] for t in sel])
        print(" " + "-" * 50)
        print(f" C: 均值 {cs.mean():.3f}，标准差 {cs.std():.3f}，"
              f"极差 {cs.max()-cs.min():.3f}（{cs.max()/cs.min():.2f}x）")
        print(" 若极差 << twod 跨度的效应，则这一组确实是平台，twod 曲线不适用于它。")
    print("\n 这一步全是 in-sample；胜出者要变成结论必须再跑预官。")


if __name__ == "__main__":
    main()
