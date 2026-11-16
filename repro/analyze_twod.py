"""twod 定律预官检验的裁判：U1-U5，判据跑前写定在 sweep_law_test.py docstring 第五组。

预官来源：sweep_law_test.py docstring 第五组（2026-10-18 14:59，跑前写定；
          md5 = 21c6b4bcbe06e1345818894b61b41789）。

冻结的定律（28 个已测剖面，共同 u 网格 {0.4, 0.65}）：
    twod  :  C = 0.704 + 0.361/(1+(twod/0.421)^(-1.23))     R2=0.911
    tw_lev:  C = 0.718 + 0.315/(1+(tw/0.381)^(-1.68))       R2=0.827
其中 twod = tw_lev/dyn，dyn = (amax-amin)/<a>。

8 个新族（knif_ref=0.92，targets 0.35/0.60/0.85 => 落在共同 u 网格上）：
  族      kind        par   twod     dyn    phi    C(twod)预官  C(tw_lev)预官
  W6      weibull     6.00   0.453  1.087  0.000      0.893        0.909
  TK80    tukey       0.80   0.724  1.087  0.000      0.943        0.961
  RQ400   ramp        4.00   0.894  0.547  0.540      0.963        0.908   <- 底座
  TK95    tukey       0.95   0.983  1.087  0.000      0.971        0.986
  W15     weibull     1.50   1.786  1.087  0.000      1.013        1.014
  RQ200   ramp        2.00   3.456  0.259  0.828      1.040        0.972   <- 底座
  EF100   erf         1.00   4.528  0.212  0.861      1.047        0.978   <- 底座
  LG150   logistic    1.50  26.603  0.052  0.973      1.063        1.001   <- 底座(外推)

判据：
  U1 前缀有界：8 个新族 C 落在 twod 预官的 ±0.038 dex 内（= 训练数据 LOO 最大残差）。
  U2 结构判别：4 个底座剖面的平均 |残差| 在 twod 下必须小于在 tw_lev 下。
  U3a Spearman(twod, C) >= 0.90；U3b C(W6) < C(TK80) 且 C(W6) < C(LG150)。
  U4 两个 tukey 族同样满足 U1。
  U5 8 个新族的 b(u) 仍落在旧范围。

用法： venv2/bin/python analyze_twod.py [--data sweep_twod.json]
"""

import argparse
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

NEW = ["W6", "TK80", "RQ400", "TK95", "W15", "RQ200", "EF100", "LG150"]
PED = ["RQ400", "RQ200", "EF100", "LG150"]
PAS = ["W6", "TK80", "TK95", "W15"]
TUKEY = ["TK80", "TK95"]
TOL = 0.038           # dex；= twod 定律在 28 个训练剖面上的 LOO 最大残差


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sweep_twod.json")
    args = ap.parse_args()

    # ---------- 基准：28 个已测剖面 ----------
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
    ft, x0t, pt_ = ptod.fit_sat(xt, y)
    fl, x0l, pl = ptod.fit_sat(xl, y)
    l1, l2 = hd.loo(np.column_stack(
        [np.ones(len(xt)), 1.0 / (1.0 + (xt / x0t) ** (-pt_))]), y)
    tol_calc = l2 / float(np.mean(np.abs(y))) / math.log(10)

    print("=" * 108)
    print(f" 基准：{len(names)} 个已测剖面；共同 u 网格 " +
          " ".join(f"{u:.3g}" for u in Ucom))
    print("=" * 108)
    print(f"\n twod 定律: C = {ft['coef'][0]:.3f} + {ft['coef'][1]:.3f}"
          f"/(1+(twod/{x0t:.3f})^(-{pt_:.2f}))，R2={ft['r2']:.3f}")
    print(f"   LOO RMSE {l1/float(np.mean(np.abs(y)))/math.log(10):.4f} dex，"
          f"LOO 最大 {tol_calc:.4f} dex")
    print(f"   预官容差 {TOL} dex（跑前写定）=> "
          f"{'一致' if abs(tol_calc - TOL) < 0.01 else '不一致，需说明'}")
    print(f" tw_lev 定律: C = {fl['coef'][0]:.3f} + {fl['coef'][1]:.3f}"
          f"/(1+(tw/{x0l:.3f})^(-{pl:.2f}))，R2={fl['r2']:.3f}")

    def law_t(x):
        return ft["coef"][0] + ft["coef"][1] / (1.0 + (x / x0t) ** (-pt_))

    def law_l(x):
        return fl["coef"][0] + fl["coef"][1] / (1.0 + (x / x0l) ** (-pl))

    # ---------- 新族 ----------
    d = json.load(open(os.path.join(OUT, args.data)))
    nprof = defaultdict(list)
    nmeta = {}
    for rec in d["rows"]:
        m = rec["meta"]
        nprof[m["fid"]].append((1.0 - rec["kunif"], rec["lam_mean"]))
        nmeta[m["fid"]] = m
    for k in nprof:
        nprof[k].sort(key=lambda c: -c[0])
    print(f"\n 载入 {args.data}: {len(nprof)} 个新族；cfg nimg={d['cfg']['nimg']} "
          f"t={d['cfg']['t']:g} knif_ref={d['cfg']['kunif_ref_target']}")

    res = {}
    print("\n" + "=" * 108)
    print(" 1. 新族的描述量与实测 C")
    print("=" * 108)
    print(f" {'族':<6} | {'kind':<10} | {'twod':>7} | {'tw_lev':>7} | {'dyn':>6} | "
          f"{'phi':>6} | {'u=0.65':>8} | {'u=0.40':>8} | {'C实测':>7} | "
          f"{'C(twod)':>8} | {'C(tw_lev)':>9}")
    print(" " + "-" * 104)
    for fid in NEW:
        if fid not in nprof:
            print(f" {fid:<6} | 数据里没有")
            continue
        m = nmeta[fid]
        a = 1.0 - hd.make_profile(m["kind"], m["par"], m["scale"], r)
        a = a * (0.60 / float(a.mean()))
        dd = hd.descriptors(a, r)
        dyn = float(dd["_amax"] - dd["_amin"]) / float(a.mean())
        phi = float(dd["_amin"]) / float(a.mean())
        twod = dd["tw_lev"] / dyn if dyn > 1e-9 else np.nan
        cs_ = {}
        for u in Ucom:
            v = hd.interp(nprof[fid], u)
            if v and v > 0 and u in mid:
                cs_[u] = v * u ** (-b_of(u)) / mid[u]
        cm = float(np.median(list(cs_.values()))) if cs_ else float("nan")
        res[fid] = dict(twod=twod, twl=dd["tw_lev"], dyn=dyn, phi=phi, cm=cm,
                        cs_=cs_)
        row = [f"{cs_[u]:>8.3f}" if u in cs_ else f"{'--':>8}" for u in Ucom]
        print(f" {fid:<6} | {m['kind']:<10} | {twod:>7.3f} | {dd['tw_lev']:>7.3f} | "
              f"{dyn:>6.3f} | {phi:>6.3f} | " + " | ".join(row) +
              f" | {cm:>7.3f} | {law_t(twod):>8.3f} | {law_l(dd['tw_lev']):>9.3f}")
    print(" " + "-" * 104)

    # ---------- 判据 ----------
    print("\n" + "=" * 108)
    print(" 2. 判据")
    print("=" * 108)

    # U1
    print(f"\n U1 前缀有界（容差 {TOL} dex = twod 定律在 28 个训练剖面上的 LOO 最大残差）:")
    u1 = {}
    for fid in NEW:
        if fid not in res:
            continue
        ct = law_t(res[fid]["twod"])
        dd = math.log10(res[fid]["cm"] / ct)
        u1[fid] = abs(dd) <= TOL
        print(f"   {fid:<6} twod={res[fid]['twod']:>7.3f}  C实测 {res[fid]['cm']:.3f}"
              f" vs C(twod) {ct:.3f}  残差 {dd:>+7.3f} dex  "
              f"{'✅' if u1[fid] else '❌'}")
    p1 = all(u1.values()) and len(u1) == len(NEW)

    # U2
    print("\n U2 结构判别：4 个底座剖面的平均 |残差|，twod 定律 vs tw_lev 定律")
    print(f" {'族':<6} | {'|残差| twod':>12} | {'|残差| tw_lev':>13}")
    print(" " + "-" * 40)
    et, el = [], []
    for fid in PED:
        if fid not in res:
            continue
        dt = abs(math.log10(res[fid]["cm"] / law_t(res[fid]["twod"])))
        dl = abs(math.log10(res[fid]["cm"] / law_l(res[fid]["twl"])))
        et.append(dt); el.append(dl)
        print(f" {fid:<6} | {dt:>12.4f} | {dl:>13.4f}")
    print(" " + "-" * 40)
    mt = float(np.mean(et)) if et else float("nan")
    ml = float(np.mean(el)) if el else float("nan")
    p2 = (len(et) == len(PED)) and mt < ml
    print(f"   平均 |残差|: twod {mt:.4f} dex  vs  tw_lev {ml:.4f} dex"
          f"  => U2 {'通过' if p2 else '未过'}")

    # U3
    print("\n U3 排序（统计公平性跑前已声明：只要求间距 > 0.02 dex 的两条 + 秩相关）:")
    order = sorted([f for f in NEW if f in res], key=lambda f: res[f]["twod"])
    cms = [res[f]["cm"] for f in order]
    sp = hd.spearman(np.array([res[f]["twod"] for f in order]), np.array(cms))
    ninv = sum(1 for i in range(len(cms) - 1) if cms[i] >= cms[i + 1])
    for f in order:
        print(f"   {f:<6} twod={res[f]['twod']:>7.3f}  C={res[f]['cm']:.3f}"
              f"  (预官 {law_t(res[f]['twod']):.3f})")
    print(f"   相邻逆序 {ninv} 处；Spearman = {sp:+.3f}")
    p3a = sp >= 0.90
    b1 = res["W6"]["cm"] < res["TK80"]["cm"] if "W6" in res and "TK80" in res else False
    b2 = res["W6"]["cm"] < res["LG150"]["cm"] if "W6" in res and "LG150" in res else False
    p3b = b1 and b2
    print(f"   U3a Spearman >= 0.90 : {'✅' if p3a else '❌'}")
    print(f"   U3b C(W6)={res.get('W6',{}).get('cm',float('nan')):.3f} < "
          f"C(TK80)={res.get('TK80',{}).get('cm',float('nan')):.3f} : "
          f"{'✅' if b1 else '❌'}")
    print(f"       C(W6) < C(LG150)={res.get('LG150',{}).get('cm',float('nan')):.3f} : "
          f"{'✅' if b2 else '❌'}")
    p3 = p3a and p3b

    # U4
    print("\n U4 形状无关性（两个 tukey 族同样满足 U1）:")
    p4 = all(u1.get(f, False) for f in TUKEY)
    for f in TUKEY:
        if f in res:
            dd = math.log10(res[f]["cm"] / law_t(res[f]["twod"]))
            print(f"   {f}: 残差 {dd:+.3f} dex  {'✅' if abs(dd) <= TOL else '❌'}")
    print(f"   => U4 {'通过' if p4 else '未过'}")

    # U5
    print("\n U5 指数不受影响（8 个新族的 b(u) 仍落在旧范围）:")
    print(f" {'族':<6} | " + " | ".join(f"u∈[{a:g},{b:g}]" for a, b in
                                        sorted(hd.P1_RANGES, reverse=True)))
    print(" " + "-" * 62)
    p5 = True
    for fid in NEW:
        if fid not in nprof:
            continue
        row, okf = [], True
        for (a, b) in sorted(hd.P1_RANGES, reverse=True):
            va, vb = hd.interp(nprof[fid], a), hd.interp(nprof[fid], b)
            lo, hi = hd.P1_RANGES[(a, b)]
            if va and vb and va > 0 and vb > 0:
                bb = math.log(vb / va) / math.log(b / a)
                row.append(f"{bb:>6.2f} {'✓' if lo <= bb <= hi else '✗'}")
                okf = okf and (lo <= bb <= hi)
            else:
                row.append(f"{'--':>6}  ")
        p5 = p5 and okf
        print(f" {fid:<6} | " + " | ".join(row))
    print(" " + "-" * 62)
    print(f" => U5 {'通过' if p5 else '未过'}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 108)
    print(" 3. 汇总")
    print("=" * 108)
    print(f" U1 前缀有界  : {'✅' if p1 else '❌'}   "
          f"（{sum(u1.values())}/{len(u1)} 个族在 ±{TOL} dex 内）")
    print(f" U2 结构判别  : {'✅' if p2 else '❌'}   "
          f"（底座剖面平均|残差| twod {mt:.4f} vs tw_lev {ml:.4f} dex）")
    print(f" U3 排序      : {'✅' if p3 else '❌'}   "
          f"（Spearman {sp:+.3f}，逆序 {ninv} 处）")
    print(f" U4 形状无关  : {'✅' if p4 else '❌'}")
    print(f" U5 指数      : {'✅' if p5 else '❌'}")
    allp = all([p1, p2, p3, p4, p5])
    print("\n " + ("全部通过 => 「C 由 twod = tw_lev/dyn 决定」同时覆盖有通带与底座两个"
                   " regime；第四组对 tw_lev 的部分撤回升级为「有通带只是 twod 的特例」。"
                   if allp else
                   "有未通过项 => 记录失败。按项目规矩，不为失败的预言改阈值；"
                   "失败的条目通常比通过的更有信息量。"))


if __name__ == "__main__":
    main()
