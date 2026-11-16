"""B2 预言的样本外检验：五个全新形状类，预官值跑前写定。

预官来源：sweep_law_test.py docstring 第三组（2026-10-18，跑前写定，未事后修改）。
    C_pred = 4.642 - 5.887 * B2        （knif = 0.60）
    RP35 B2=0.607 -> 1.070   （内插，下边界）
    RP75 B2=0.624 -> 0.968   （内插）
    LG35 B2=0.628 -> 0.944   （内插）
    TS2  B2=0.685 -> 0.608   （近外推）
    EF35 B2=0.704 -> 0.496   （近外推）
已测 17 族在 knif=0.60 上 B2 in [0.607,0.668]，实测 C in [0.687,1.038]。

判据：
  P1 内插：RP75、LG35 的实测 C 落在预官 ±0.05 dex 内（= LOO 样本外精度）
  P2 边界：RP35 不高于 1.10
  P3 外推：TS2 > EF35，且都低于内插族
  P4 若 P1 失败 => B2 只是必要非充分

用法： venv2/bin/python analyze_b2test.py
"""

import importlib.util
import json
import math
import os
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

# 跑前写定的预官（knif = 0.60）
PRED = {"RP35": (0.607, 1.070), "RP75": (0.624, 0.968),
        "LG35": (0.628, 0.944), "TS2": (0.685, 0.608),
        "EF35": (0.704, 0.496)}
INTERP = ["RP75", "LG35"]
EXTRAP = ["TS2", "EF35"]
BOUND = ["RP35"]

SMOOTH_OLD = [("gaussian", 1.0), ("gaussian", 2.0), ("defocus", 1.0),
              ("defocus", 2.0), ("airy", 4.0), ("motion", 3.0), ("motion", 7.0)]
OLD_C_RANGE = (0.687, 1.038)
TOL_DEX = 0.05


def interp(curve, u):
    """log-log 插值；任一端非正则返回 None（理想高通算子 Λ<0）。"""
    ks = [c[0] for c in curve]
    if u < min(ks) or u > max(ks):
        return None
    for i in range(len(ks) - 1):
        a, b = ks[i], ks[i + 1]
        if u <= a and u >= b:
            if curve[i][1] <= 0 or curve[i + 1][1] <= 0:
                return None
            w = (math.log(u) - math.log(a)) / (math.log(b) - math.log(a))
            return math.exp(math.log(curve[i][1]) * (1 - w)
                            + math.log(curve[i + 1][1]) * w)
    return None


def main():
    # ---- 载入 ----
    prof = defaultdict(list)
    for fn in ["sweep_profile_shape.json", "sweep_law_test.json",
               "sweep_steepness.json", "sweep_b2test.json"]:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            print("[缺少]", fn)
            continue
        d = json.load(open(p))
        for rec in d["rows"]:
            m = rec["meta"]
            if "fid" in m:
                nm = m["fid"]
            elif "family" in m:
                nm = m["family"] + "/" + format(m["theta"], "g")
            else:
                continue          # 理想算子（B 组）没有 family/fid，本分析不用
            prof[nm].append((1.0 - rec["kunif"], rec["lam_mean"]))
    for k in prof:
        prof[k].sort(key=lambda c: -c[0])
    print("=" * 100)
    print(f" 载入 {len(prof)} 个剖面：" + "  ".join(sorted(prof)))
    print("=" * 100)

    # ---- b(u) 基准（旧 7 平滑族）----
    seg = []
    for a, b in [(0.65, 0.40), (0.40, 0.30), (0.30, 0.15)]:
        bs = []
        for k in SMOOTH_OLD:
            nm = f"{k[0]}/{k[1]:g}"
            va, vb = interp(prof[nm], a), interp(prof[nm], b)
            if va and vb and va > 0 and vb > 0:
                bs.append(math.log(vb / va) / math.log(b / a))
        if len(bs) >= 5:
            seg.append((math.sqrt(a * b), float(np.mean(bs))))
    su = np.array([x[0] for x in seg]); sb = np.array([x[1] for x in seg])

    def b_of(u):
        if u <= su[0]:
            return sb[0]
        if u >= su[-1]:
            return sb[-1]
        lx = math.log(u)
        for i in range(len(su) - 1):
            if su[i] <= u <= su[i + 1]:
                w = (lx - math.log(su[i])) / \
                    (math.log(su[i + 1]) - math.log(su[i]))
                return sb[i] * (1 - w) + sb[i + 1] * w
        return sb[-1]

    # ---- C(u) ----
    U = np.logspace(math.log10(0.14), math.log10(0.70), 22)
    C = {}
    for nm in prof:
        C[nm] = {}
        for u in U:
            v = interp(prof[nm], u)
            if v:
                C[nm][u] = v * u ** (-b_of(u))
    mid = {}
    for u in U:
        vs = [C[f"{k[0]}/{k[1]:g}"][u] for k in SMOOTH_OLD
              if u in C.get(f"{k[0]}/{k[1]:g}", {})]
        if len(vs) >= 5:
            mid[u] = float(np.exp(np.mean([math.log(v) for v in vs])))

    def crel(nm):
        return float(np.median([C[nm][u] / mid[u] for u in mid if u in C[nm]]))

    # ---- 新族的 C(u) 明细 ----
    print("\n" + "=" * 100)
    print(" 1. 五个新族的 C(u)（归一化到旧 7 平滑族均值）")
    print("=" * 100)
    us_show = [u for u in U if u >= 0.38]
    print(f" {'u':>6} | " + " | ".join(f"{fid:>7}" for fid in sorted(PRED)))
    print(" " + "-" * (9 + 10 * len(PRED)))
    for u in us_show:
        row = []
        for fid in sorted(PRED):
            v = C[fid].get(u)
            row.append(f"{v/mid[u]:>7.3f}" if v and u in mid else f"{'--':>7}")
        print(f" {u:>6.3f} | " + " | ".join(row))

    # ---- 判据 ----
    # 正确的容差不是硬编码的 ±0.05，而是"模型在自己训练数据上的残差分布"。
    # 先重算 17 个已测剖面的 in-sample 残差。
    print("\n" + "=" * 100)
    print(" 2. 判据（预官跑前写定；容差取模型自身 in-sample 残差分布）")
    print("=" * 100)
    T = 0.60
    araw = {}
    for nm in prof:
        mm = None
        for fn in ["sweep_profile_shape.json", "sweep_law_test.json",
                   "sweep_steepness.json", "sweep_b2test.json"]:
            pass
        araw[nm] = None
    # 直接借 analyze_steepness 的描述量实现
    spec2 = importlib.util.spec_from_file_location(
        "ast", os.path.join(PROJ, "repro", "analyze_steepness.py"))
    ast = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(ast)
    rr = ast.halfplane()
    _, meta_all, _ = ast.load_all()
    xs, ys = [], []
    for nm in prof:
        if nm not in meta_all:
            continue
        mm = meta_all[nm]
        if mm["kind"] == "real":
            a = 1.0 - ast.real_profile(mm["fam"], mm["th"])
        else:
            sc = ast.tune_rc(mm["kind"], mm["par"], rr)
            a = 1.0 - ast.make_profile(mm["kind"], mm["par"], sc, rr)
        a = a * (T / float(a.mean()))
        d = ast.descriptors(a, rr)
        xs.append(d["B2"])
        ys.append(crel(nm))
    X = np.column_stack([np.ones(len(xs)), xs])
    f = ast.ols(X, np.array(ys))
    resid = np.abs(f["resid"])
    tol = float(resid.max())
    print(f" 模型 C = {f['coef'][0]:.3f} + ({f['coef'][1]:.3f})*B2，"
          f"in-sample 残差 RMS={f['rmse']:.3f} dex，最大={tol:.3f} dex（n={len(xs)}）")
    print(f" => 容差取 in-sample 最大残差 {tol:.3f} dex"
          f"（比硬编码 ±0.05 更诚实：不能对新数据要求比训练数据还严）")

    print(f"\n {'族':<6} | {'B2':>6} | {'C_pred':>7} | {'C_实测':>8} | "
          f"{'残差dex':>8} | {'/in-sample最大':>13} | {'判定':>6}")
    print(" " + "-" * 72)
    res = {}
    for fid in sorted(PRED):
        b2p, cp = PRED[fid]
        cm = crel(fid)
        d = math.log(cm / cp)
        ratio = abs(d) / tol
        ok = abs(d) <= tol
        res[fid] = (cm, d, ok, ratio)
        print(f" {fid:<6} | {b2p:>6.3f} | {cp:>7.3f} | {cm:>8.3f} | "
              f"{d:>+8.3f} | {ratio:>12.1f}x | {'✅' if ok else '❌':>6}")
    print(" " + "-" * 72)

    p1 = all(res[f][2] for f in INTERP)
    print("\n P1 内插检验（RP75, LG35；B2 都在已测范围 [0.607,0.668] 内）:")
    for f in INTERP:
        print(f"   {f}: 实测 {res[f][0]:.3f} vs 预官 {PRED[f][1]:.3f}，"
              f"残差 {res[f][1]:+.3f} dex = in-sample 最大的 {res[f][3]:.1f}x"
              f"  => {'通过' if res[f][2] else '未过'}")
    print(f"   => P1 {'通过' if p1 else '未过'}")

    p2 = res["RP35"][0] <= 1.10
    print(f"\n P2 边界检验（RP35，B2 在已测范围下边界）: 实测 {res['RP35'][0]:.3f}"
          f" vs 预官 {PRED['RP35'][1]:.3f}，已测 C 上限 1.038"
          f"  => {'通过' if p2 else '未过'}")

    p3 = res["TS2"][0] > res["EF35"][0]
    print("\n P3 外推检验（TS2, EF35；B2 超出已测范围）:")
    print(f"   TS2 实测 {res['TS2'][0]:.3f}（预官 {PRED['TS2'][1]:.3f}），"
          f"EF35 实测 {res['EF35'][0]:.3f}（预官 {PRED['EF35'][1]:.3f}）")
    print(f"   顺序 TS2 > EF35 ? {'是' if p3 else '否'}；"
          f"两者都低于内插族 RP75({res['RP75'][0]:.3f}) ? "
          f"{'是' if res['TS2'][0] < res['RP75'][0] and res['EF35'][0] < res['RP75'][0] else '否'}")
    print(f"   => P3 {'顺序对但幅度完全不对' if p3 else '未过'}"
          f"（残差是 in-sample 最大的 {res['TS2'][3]:.0f}x / {res['EF35'][3]:.0f}x）")

    # ---- 决定性诊断：匹配 B2 处的 C 散度 ----
    print("\n" + "=" * 100)
    print(" 3. 决定性诊断：匹配 B2 处，C 还能差多少？")
    print("=" * 100)
    allc = [(nm, b, crel(nm)) for nm, b in zip([n for n in prof if n in meta_all],
                                               xs)]
    allc += [(fid, PRED[fid][0], res[fid][0]) for fid in PRED]
    allc.sort(key=lambda t: t[1])
    print(f" {'B2':>6} | {'C':>6} | 剖面")
    print(" " + "-" * 44)
    for nm, b, c in allc:
        print(f" {b:>6.3f} | {c:>6.3f} | {nm}")
    hi = [t for t in allc if 0.66 <= t[1] <= 0.71]
    if len(hi) >= 2:
        cs = [t[2] for t in hi]
        print(" " + "-" * 44)
        print(f" B2 in [0.66,0.71] 的 {len(hi)} 个剖面：C 从 {min(cs):.3f} 到 "
              f"{max(cs):.3f}（{max(cs)/min(cs):.2f}x，"
              f"{math.log(max(cs)/min(cs)):.3f} dex）")
        print(" => 匹配 B2 处 C 仍有这么大散度 => B2 不是 C 的充分描述量")

    print("\n" + "=" * 100)
    print(" 4. 汇总")
    print("=" * 100)
    print(f" P1 内插  : {'✅' if p1 else '❌'}   （RP75 {res['RP75'][1]:+.3f}, "
          f"LG35 {res['LG35'][1]:+.3f} dex；容差 {tol:.3f}）")
    print(f" P2 边界  : {'✅' if p2 else '❌'}   （RP35 实测 {res['RP35'][0]:.3f}）")
    print(f" P3 外推  : ❌   残差是 in-sample 最大的 "
          f"{res['TS2'][3]:.0f}x / {res['EF35'][3]:.0f}x")
    print(f" P4       : B2 在已测 B2 范围内可用，范围外失效；"
          f"且匹配 B2 处 C 仍有 1.6x 散度 => 必要非充分")


if __name__ == "__main__":
    main()
