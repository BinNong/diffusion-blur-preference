"""tw_lev 定律预官检验的裁判：T1-T6，判据跑前写定在 sweep_law_test.py docstring 第四组。

预官来源：sweep_law_test.py docstring 第四组（2026-10-18 12:56，跑前写定；
          脚本 md5 = 3738455f29ae239c27dbaf3cbce4043c）。

定律（22 个已测剖面，C 取共同 u 网格 {0.4, 0.65} 的中位数）：
    C = 0.696 + 0.377 / (1 + (tw/0.47)^(-1.09))        R2=0.909
    LOO RMSE 0.0147 dex，LOO 最大残差 0.0358 dex（旧惯例 0.0452 dex）

6 个新族（knif_ref=0.92，targets 0.35/0.60/0.85 => 落在共同 u 网格上）：
  族      kind        par   tw_lev   C_pred(tw)   C_pred(B2)   C_pred(supp20)
  TK30    tukey       0.30   0.209      0.806        0.918          0.821
  TK50    tukey       0.50   0.395      0.866        0.927          0.859
  EF60    erf         0.60   0.686      0.922        1.016          1.032
  W3      weibull     3.00   0.959      0.954        0.952          0.936
  LG60    logistic    0.60   1.133      0.968        1.021          1.032
  TS3     twoscale    3.00   2.549      1.021        0.988          1.010

判据：
  T1 排序（无需容差）：C 随 tw_lev 单调递增。
  T2 前缀有界：|log10(C_meas/C_pred(tw))| <= 0.045 dex（= 训练数据 LOO 最大残差取大）。
  T3 对抗性判别：TK30 与 EF60 的 C 比 B2 预言低 >=5%。
  T4 膝部：TK50 与 EF60 的 C 严格落在 B8 的 0.834 与 W2 的 0.983 之间。
  T5 形状无关性：两个 tukey 族同样满足 T2。
  T6 指数不受影响：新族 b(u) 仍落在旧范围。

用法： venv2/bin/python analyze_twlaw.py [--data sweep_twlaw.json]
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
    "hd", os.path.join(PROJ, "repro", "hunt_descriptor.py"))
hd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hd)

SMOOTH_OLD = [("gaussian", 1.0), ("gaussian", 2.0), ("defocus", 1.0),
              ("defocus", 2.0), ("airy", 4.0), ("motion", 3.0), ("motion", 7.0)]
P1_RANGES = {(0.65, 0.40): (-1.68, -1.31), (0.40, 0.30): (-0.91, -0.82),
             (0.30, 0.15): (-0.90, -0.68)}
BASE_FILES = ["sweep_profile_shape.json", "sweep_law_test.json",
              "sweep_steepness.json", "sweep_b2test.json"]

# 跑前写定的预官（sweep_law_test.py docstring 第四组）
PRED = {          # fid: (kind, par, tw_lev, C_tw, C_B2, C_supp20)
    "TK30": ("tukey", 0.30, 0.209, 0.806, 0.918, 0.821),
    "TK50": ("tukey", 0.50, 0.395, 0.866, 0.927, 0.859),
    "EF60": ("erf", 0.60, 0.686, 0.922, 1.016, 1.032),
    "W3": ("weibull", 3.00, 0.959, 0.954, 0.952, 0.936),
    "LG60": ("logistic", 0.60, 1.133, 0.968, 1.021, 1.032),
    "TS3": ("twoscale", 3.00, 2.549, 1.021, 0.988, 1.010),
}
TOL = 0.045          # dex；= 训练数据 LOO 最大残差（两惯例取大）
T4_LO, T4_HI = 0.834, 0.983     # B8 与 W2 的实测 C
NEW = ["TK30", "TK50", "EF60", "W3", "LG60", "TS3"]
TUKEY = ["TK30", "TK50"]


def load_json(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sweep_twlaw.json")
    args = ap.parse_args()

    # ---------- 基准：22 个已测剖面 ----------
    tb = hd.build_tables()
    names, desc = tb["names"], tb["desc"]
    crel, crel_all = tb["crel"], tb["crel_all"]
    r, Ucom, Uall = tb["r"], tb["Ucom"], tb["Uall"]
    C, mid, b_of = tb["C"], tb["mid"], tb["b_of"]
    print("=" * 110)
    print(f" 基准：{len(names)} 个已测剖面，共同 u 网格 " +
          " ".join(f"{u:.3g}" for u in Ucom))
    print("=" * 110)

    # 拟合法律（与预官同一形式、同一数据）
    y = np.array([crel(n) for n in names])
    x = np.array([desc[n]["tw_lev"] for n in names])
    best = None
    for t0 in np.linspace(0.05, 3.0, 120):
        for p in np.linspace(0.3, 4.0, 76):
            X = np.column_stack([np.ones(len(x)),
                                 1.0 / (1.0 + (x / t0) ** (-p))])
            f = hd.ols(X, y)
            if f and (best is None or f["rmse"] < best[0]):
                best = (f["rmse"], f, t0, p)
    _, f, t0, p = best
    c0, c1 = f["coef"]

    def law(tw):
        return c0 + c1 / (1.0 + (tw / t0) ** (-p))

    # B2 / supp20 对照定律
    xb = np.array([desc[n]["B2"] for n in names])
    xs = np.array([desc[n]["supp20"] for n in names])
    fb = hd.ols(np.column_stack([np.ones(len(xb)), xb]), y)
    fs = hd.ols(np.column_stack([np.ones(len(xs)), xs]), y)

    # LOO 残差（容差来源，不硬编码）
    X = np.column_stack([np.ones(len(x)), 1.0 / (1.0 + (x / t0) ** (-p))])
    l1, l2 = hd.loo(X, y)
    tol_lin = l2 / float(np.mean(np.abs(y))) / math.log(10)
    print(f"\n 拟合 C = {c0:.3f} + {c1:.3f}/(1+(tw/{t0:.2f})^(-{p:.2f}))，"
          f"R2={f['r2']:.3f}")
    print(f" LOO RMSE {l1/float(np.mean(np.abs(y)))/math.log(10):.4f} dex，"
          f"LOO 最大残差 {tol_lin:.4f} dex")
    print(f" 预官容差 {TOL} dex（跑前写定 = 两 C 惯例 LOO 最大的大者）；"
          f"本轮重算 {tol_lin:.4f} dex => "
          f"{'一致' if abs(tol_lin - TOL) < 0.02 else '不一致，需说明'}")
    print(f" 对照 B2 : C = {fb['coef'][0]:.3f} + ({fb['coef'][1]:.3f})*B2，"
          f"R2={fb['r2']:.3f}")
    print(f" 对照 supp20 : C = {fs['coef'][0]:.3f} + ({fs['coef'][1]:.3f})*supp20，"
          f"R2={fs['r2']:.3f}")

    # ---------- 新族数据 ----------
    d = load_json(args.data)
    if d is None:
        print(f"\n[缺少] {args.data}")
        return
    prof = defaultdict(list)
    meta = {}
    for rec in d["rows"]:
        m = rec["meta"]
        nm = m["fid"]
        prof[nm].append((1.0 - rec["kunif"], rec["lam_mean"]))
        if nm not in meta:
            meta[nm] = dict(kind=m["kind"], par=float(m["par"]),
                            scale=float(m["scale"]), kref=float(m["kunif_ref"]))
    for k in prof:
        prof[k].sort(key=lambda c: -c[0])
    print(f"\n 载入 {args.data}: {len(prof)} 个新族 " + "  ".join(sorted(prof)))
    print(f" cfg: nimg={d['cfg']['nimg']} t={d['cfg']['t']:g} "
          f"knif_ref={d['cfg']['kunif_ref_target']} targets={d['cfg']['targets']}")

    # 新族的 tw_lev（只由剖面定义决定，与 Λ 无关）与实测 C
    print("\n" + "=" * 110)
    print(" 1. 新族的 tw_lev 与实测 C")
    print("=" * 110)
    print(f" {'族':<6} | {'kind':<10} | {'tw_lev(预官)':>12} | {'tw_lev(重算)':>12} | "
          f"{'u=0.65':>8} | {'u=0.40':>8} | {'C实测':>7} | {'C(tw)预官':>10}")
    print(" " + "-" * 92)
    res = {}
    for fid in NEW:
        if fid not in prof:
            print(f" {fid:<6} | 数据里没有这个族")
            continue
        kind, par, tw_p, ct, cb, cs = PRED[fid]
        m = meta[fid]
        a = 1.0 - hd.make_profile(m["kind"], m["par"], m["scale"], r)
        a = a * (0.60 / float(a.mean()))
        dd = hd.descriptors(a, r)
        tw_m = dd["tw_lev"]
        cs_ = {}
        for u in Ucom:
            v = hd.interp(prof[fid], u)
            if v and v > 0 and u in mid:
                cs_[u] = v * u ** (-b_of(u)) / mid[u]
        cm = float(np.median(list(cs_.values()))) if cs_ else float("nan")
        res[fid] = dict(tw_p=tw_p, tw_m=tw_m, ct=ct, cb=cb, cs=cs, cm=cm,
                        cs_=cs_, b2=dd["B2"], sup=dd["supp20"])
        row = [f"{cs_[u]:>8.3f}" if u in cs_ else f"{'--':>8}" for u in Ucom]
        print(f" {fid:<6} | {kind:<10} | {tw_p:>12.3f} | {tw_m:>12.3f} | " +
              " | ".join(row) + f" | {cm:>7.3f} | {ct:>10.3f}")
    print(" " + "-" * 92)
    ok_tw = all(abs(res[f]["tw_m"] - res[f]["tw_p"]) < 0.02 for f in res)
    print(f" tw_lev 重算 vs 预官: " + "  ".join(
        f"{f}:{res[f]['tw_m']:.3f}/{res[f]['tw_p']:.3f}" for f in sorted(res)) +
        f"  => {'一致' if ok_tw else '不一致'}")

    # ---------- 判据 ----------
    print("\n" + "=" * 110)
    print(" 2. 判据")
    print("=" * 110)

    # T1 排序（跑前修正案：严格单调在噪声下不公平，改为秩相关 + 三条可分辨不等式）
    order = sorted(res, key=lambda f: res[f]["tw_m"])
    cms = [res[f]["cm"] for f in order]
    mono = all(cms[i] < cms[i + 1] for i in range(len(cms) - 1))
    sp = hd.spearman(np.array([res[f]["tw_m"] for f in order]),
                     np.array(cms))
    n_inv = sum(1 for i in range(len(cms) - 1) if cms[i] >= cms[i + 1])
    print("\n T1 排序（跑前修正案；严格单调只有 ~35% 概率即使定律正确也通过，不公平）:")
    for f in order:
        print(f"   {f:<6} tw={res[f]['tw_m']:>6.3f}  C={res[f]['cm']:>6.3f}"
              f"  (预官 {res[f]['ct']:.3f})")
    print(f"   相邻逆序 {n_inv} 处；Spearman = {sp:+.3f}")
    print(f"   T1a 秩相关 >= 0.94 : {'✅' if sp >= 0.94 else '❌'}")
    pairs = [("TK30", "EF60"), ("TK30", "TS3"), ("EF60", "TS3")]
    t1b = {}
    for a, b in pairs:
        if a in res and b in res:
            t1b[(a, b)] = res[a]["cm"] < res[b]["cm"]
            print(f"   T1b C({a})={res[a]['cm']:.3f} < C({b})={res[b]['cm']:.3f}"
                  f"  {'✅' if t1b[(a, b)] else '❌'}")
    p1b = all(t1b.values()) and len(t1b) == 3
    print(f"   T1c 仅记录：EF60/W3 = "
          f"{'C(EF60)<C(W3)' if res.get('EF60', {}).get('cm', 1) < res.get('W3', {}).get('cm', 0) else 'C(EF60)>=C(W3)'}"
          f"；W3/LG60 = "
          f"{'C(W3)<C(LG60)' if res.get('W3', {}).get('cm', 1) < res.get('LG60', {}).get('cm', 0) else 'C(W3)>=C(LG60)'}"
          f"（间距低于噪声，不作判据）")
    p1 = (sp >= 0.94) and p1b
    print(f"   => T1 {'通过' if p1 else '未通过'}"
          f"（T1a {'✓' if sp >= 0.94 else '✗'}，T1b {'✓' if p1b else '✗'}）"
          f"；参考：严格单调 {'成立' if mono else '不成立'}")

    # T2 前缀有界
    print(f"\n T2 前缀有界（容差 {TOL} dex = 训练数据 LOO 最大残差）:")
    t2 = {}
    for fid in sorted(res, key=lambda f: res[f]["tw_m"]):
        d = math.log10(res[fid]["cm"] / res[fid]["ct"])
        t2[fid] = abs(d) <= TOL
        print(f"   {fid:<6} C实测 {res[fid]['cm']:.3f} vs C(tw)预官 {res[fid]['ct']:.3f}"
              f"  残差 {d:>+7.3f} dex  {'✅' if t2[fid] else '❌'}")
    p2 = all(t2.values())

    # T3 对抗性判别 vs B2
    print("\n T3 对抗性判别（TK30/EF60 的 C 须比 B2 预言低 >=5%）:")
    t3 = {}
    for fid in ["TK30", "EF60"]:
        if fid not in res:
            continue
        drop = 1.0 - res[fid]["cm"] / res[fid]["cb"]
        t3[fid] = drop >= 0.05
        print(f"   {fid}: C实测 {res[fid]['cm']:.3f} vs C(B2)预官 {res[fid]['cb']:.3f}"
              f"  低 {drop:>+6.1%}  {'✅' if t3[fid] else '❌'}")
    p3 = all(t3.values()) and len(t3) == 2

    # T4 膝部（边界跑前写定 = B8 与 W2 的实测 C；脚本从基准重算并核对）
    steep = [crel(n) for n in names if desc[n]["tw_lev"] < 0.40]
    sat = [crel(n) for n in names if desc[n]["tw_lev"] > 1.30]
    sat_lp = [crel(n) for n in names
              if desc[n]["tw_lev"] > 1.30 and tb["meta"][n]["kind"] != "real"
              and tb["meta"][n].get("kind") not in ("ramp", "logistic", "erf")]
    print(f"\n T4 膝部（跑前写定：{T4_LO} < C < {T4_HI}）:")
    print(f"   基准核对：tw<0.40 的'陡峭簇' C 最高 = {max(steep):.3f}"
          f"（预官写 B8=0.834）")
    print(f"             tw>1.30 的'饱和簇' C 最低 = {min(sat):.3f}"
          f"（若只算低通族则 {min(sat_lp):.3f}，预官写 W2=0.983）")
    print(f"   => 预官把饱和簇下界写成 W2 的 0.983，但 RP75（高通, tw=2.07）其实更低 "
          f"{min(sat):.3f}；")
    print(f"      两个新族的预言值都在两者之下，所以这个定义分歧不影响判定。")
    t4 = {}
    for fid in ["TK50", "EF60"]:
        if fid not in res:
            continue
        v = res[fid]["cm"]
        t4[fid] = T4_LO < v < T4_HI
        print(f"   {fid}: C = {v:.3f}  {'✅' if t4[fid] else '❌'}"
              f"（更严的饱和簇下界 {min(sat):.3f} 下也 "
              f"{'通过' if v < min(sat) else '不通过'}）")
    p4 = all(t4.values()) and len(t4) == 2

    # T5 形状无关性（tukey 也要过 T2）
    print("\n T5 形状无关性（两个 tukey 族——全新形状类——同样满足 T2）:")
    p5 = all(t2.get(f, False) for f in TUKEY)
    for f in TUKEY:
        if f in res:
            d = math.log10(res[f]["cm"] / res[f]["ct"])
            print(f"   {f}: 残差 {d:>+.3f} dex  {'✅' if abs(d) <= TOL else '❌'}")
    print(f"   => T5 {'通过' if p5 else '未通过'}")

    # T6 指数
    print("\n T6 指数不受影响（新族 b(u) 仍落在旧范围）:")
    print(f" {'族':<6} | " + " | ".join(f"u∈[{a:g},{b:g}]" for a, b in
                                        sorted(P1_RANGES, reverse=True)))
    print(" " + "-" * 62)
    t6 = True
    for fid in NEW:
        if fid not in prof:
            continue
        row, okf = [], True
        for (a, b) in sorted(P1_RANGES, reverse=True):
            va, vb = hd.interp(prof[fid], a), hd.interp(prof[fid], b)
            lo, hi = P1_RANGES[(a, b)]
            if va and vb and va > 0 and vb > 0:
                bb = math.log(vb / va) / math.log(b / a)
                row.append(f"{bb:>6.2f} {'✓' if lo <= bb <= hi else '✗'}")
                okf = okf and (lo <= bb <= hi)
            else:
                row.append(f"{'--':>6}  ")
        t6 = t6 and okf
        print(f" {fid:<6} | " + " | ".join(row))
    print(" " + "-" * 62)
    print(f" => T6 {'通过' if t6 else '未通过'}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 110)
    print(" 3. 汇总")
    print("=" * 110)
    print(f" T1 排序      : {'✅' if mono else '❌'}")
    print(f" T2 前缀有界  : {'✅' if p2 else '❌'}   （{sum(t2.values())}/{len(t2)} 个族在 ±{TOL} dex 内）")
    print(f" T3 对抗性    : {'✅' if p3 else '❌'}   （TK30/EF60 比 B2 预言低 >=5%）")
    print(f" T4 膝部      : {'✅' if p4 else '❌'}")
    print(f" T5 形状无关  : {'✅' if p5 else '❌'}")
    print(f" T6 指数      : {'✅' if t6 else '❌'}")
    allp = all([mono, p2, p3, p4, p5, t6])
    print("\n " + ("全部通过 => 「C(shape) 由衰减剖面的相对过渡带宽度 tw_lev 决定」"
                   "是一条有预言力的定律，且强于被推翻的 B2。"
                   if allp else
                   "有未通过项 => 记录失败。按项目规矩，不为失败的预言改阈值；"
                   "失败的条目通常比通过的更有信息量。"))


if __name__ == "__main__":
    main()
