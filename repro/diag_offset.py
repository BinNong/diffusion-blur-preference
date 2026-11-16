"""诊断：底座剖面的 +0.018 dex 系统偏移，究竟是「新 vs 训练」还是「knif_ref 混淆」？

背景（RESULTS §3.9(q5)）
-----------------------
twod 定律预官 8 个新族：4 个新底座剖面残差全为正（+0.044/+0.004/+0.007/+0.016，
均值 +0.018 dex），4 个新有通带剖面均值 +0.003 dex。当时归因为"描述量被看着数据设计出来"。

但还有一个没排除的混淆：**训练集里的底座剖面一半在 knif_ref=0.60（b2test 组：EF35/LG35/
RP35/RP75），另一半在 0.92（EF60/LG60）；而 8 个新族全在 0.92。** 同一个 erf 族在
knif_ref=0.60 与 0.92 是**不同的形状**（sc 被调到不同位置），所以如果描述量漏掉了某个
与 knif_ref 相关的维度，"新 vs 训练"和"0.92 vs 0.60"就分不开。

本脚本把 36 个已测剖面的残差按 (knif_ref, regime) 分组，看偏移到底跟哪个变量走。
不涉及任何新拟合的定律（只用已冻结的 twod 定律）。

用法： venv2/bin/python diag_offset.py
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
NEWFAM = {"W6", "TK80", "RQ400", "TK95", "W15", "RQ200", "EF100", "LG150",
          "TK30", "TK50", "EF60", "W3", "LG60", "TS3"}
# twod 定律（冻结，28 个训练剖面）
FT = (0.704, 0.361, 0.421, 1.23)


def law(x):
    a, b, x0, p = FT
    return a + b / (1.0 + (x / x0) ** (-p))


def main():
    prof, meta = defaultdict(list), {}
    for fn in FILES:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            print("[缺少]", fn)
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
        dyn = float(d["_amax"] - d["_amin"]) / float(a.mean())
        phi = float(d["_amin"]) / float(a.mean())
        twod = d["tw_lev"] / dyn if dyn > 1e-9 else np.nan
        if not np.isfinite(twod) or twod <= 0 or not np.isfinite(crel(nm)):
            continue
        resid = math.log10(crel(nm) / law(twod))
        rows.append(dict(nm=nm, kref=m["kref"], phi=phi, dyn=dyn, twod=twod,
                         twl=d["tw_lev"], c=crel(nm), resid=resid,
                         new=nm in NEWFAM, kind=m["kind"]))
    print("=" * 104)
    print(f" {len(rows)} 个剖面（共同 u 网格 " +
          " ".join(f"{u:.3g}" for u in Ucom) + "）")
    print("=" * 104)
    print(f"\n {'剖面':<14} | {'knif_ref':>8} | {'新?':>4} | {'phi':>6} | {'twod':>7} | "
          f"{'C':>6} | {'残差dex':>8}")
    print(" " + "-" * 74)
    for t in sorted(rows, key=lambda t: (t["kref"], t["phi"])):
        kr = "%.2f" % t["kref"] if np.isfinite(t["kref"]) else "real"
        print(f" {t['nm']:<14} | {kr:>8} | {'新' if t['new'] else '':>4} | "
              f"{t['phi']:>6.3f} | {t['twod']:>7.3f} | {t['c']:>6.3f} | "
              f"{t['resid']:>+8.3f}")

    # ---- 分组统计 ----
    print("\n" + "=" * 104)
    print(" 分组残差均值：偏移到底跟哪个变量走？")
    print("=" * 104)
    groups = [
        ("新 8 族 vs 训练 28", lambda t: "新" if t["new"] else "训练"),
        ("knif_ref 0.92 vs 0.60 vs real",
         lambda t: ("real" if not np.isfinite(t["kref"])
                    else "kref=%.2f" % t["kref"])),
        ("regime：phi>0.3 vs <=0.3",
         lambda t: "底座" if t["phi"] > 0.3 else "有通带"),
        ("(knif_ref, regime) 交叉",
         lambda t: ("real/通带" if not np.isfinite(t["kref"])
                    else "kref=%.2f/%s" % (t["kref"],
                                           "底座" if t["phi"] > 0.3 else "通带"))),
    ]
    for label, key in groups:
        g = defaultdict(list)
        for t in rows:
            g[key(t)].append(t["resid"])
        print(f"\n [{label}]")
        for k in sorted(g):
            v = np.array(g[k])
            print(f"   {k:<16} n={len(v):>2}  均值 {v.mean():>+7.4f} dex  "
                  f"标准差 {v.std():>6.4f}  "
                  f"全{'正' if (v > 0).all() else '异号'}")
    print("\n 若偏移主要由 knif_ref 解释（而不是「新 vs 训练」），")
    print(" 说明描述量漏掉了一个与参考强度相关的维度——那比「再加一个 phi 项」更根本。")

    # ---- 在 knif_ref=0.92 子集内重新拟合，看底座偏移是否消失 ----
    print("\n" + "=" * 104)
    print(" 只在 knif_ref=0.92 的 33 个剖面上重新拟合 twod 定律，再看残差分组")
    print("=" * 104)
    sub = [t for t in rows if np.isfinite(t["kref"]) and abs(t["kref"] - 0.92) < 0.02]
    if len(sub) >= 10:
        x = np.array([t["twod"] for t in sub]); y = np.array([t["c"] for t in sub])
        f, x0, p = ptod.fit_sat(x, y)
        print(f" n={len(sub)}；C = {f['coef'][0]:.3f} + {f['coef'][1]:.3f}"
              f"/(1+(twod/{x0:.3f})^(-{p:.2f}))，R2={f['r2']:.3f}")
        for lab, sel in [("有通带", lambda t: t["phi"] <= 0.3),
                         ("底座", lambda t: t["phi"] > 0.3),
                         ("新底座", lambda t: t["phi"] > 0.3 and t["new"])]:
            v = [math.log10(t["c"] / (f["coef"][0] + f["coef"][1] /
                                      (1.0 + (t["twod"] / x0) ** (-p))))
                 for t in sub if sel(t)]
            if v:
                v = np.array(v)
                print(f"   {lab:<8} n={len(v):>2}  均值 {v.mean():>+7.4f} dex  "
                      f"全{'正' if (v > 0).all() else '异号'}")
        print("\n 注意：这一步是 in-sample（只用于诊断偏移来源，不作为定律）。")


if __name__ == "__main__":
    main()
