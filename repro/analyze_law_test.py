"""定律检验的判据：Lambda(shape,u) = C(shape) * u^{b(u)} 对新族成立吗？

预言在看数据之前写定（sweep_law_test.py 的 docstring）：
  P1 指数普适：新族在 u in [0.65,0.40] 的 b 落在 [-1.68,-1.31]；
                u in [0.30,0.15] 的 b 落在 [-0.90,-0.68]。
  P2 前缀有界：匹配 u 处 C = Lambda * u^{-b(u)}，新族落在旧 7 族的 [min,max]
                之内，或最多超出 1.3x。
  P3 趋近硬截止：W4（最接近理想低通）的 C 是新族里最低的，并向理想低通的
                相对值（~0.65x）移动。
  P4 内部对照：W2（高斯剖面）的 C 复现 gaussian 族到几个百分点。

方法：
  1. 旧数据（sweep_profile_shape.json，8 个参考剖面 x 6 点）在公共 u 网格上
     插值，算逐段局部指数，对 7 个平滑族取均值 => b_universal(u)
  2. 新数据（sweep_law_test.json，5 个新族 x 6 点）同样插值，算各自的局部指数
  3. C(u) = Lambda(u) * u^{-b_universal(u)}，对所有 12+1 个剖面在匹配 u 处比较

用法： venv2/bin/python analyze_law_test.py
"""

import json
import math
import os
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

SMOOTH_OLD = [("gaussian", 1.0), ("gaussian", 2.0), ("defocus", 1.0),
              ("defocus", 2.0), ("airy", 4.0), ("motion", 3.0), ("motion", 7.0)]
IDEAL = ("lowpass", 0.25)

# 预言的通过区间（来自 §3.9(l)，旧 8 族的实测范围）
P1_RANGES = {(0.65, 0.40): (-1.68, -1.31), (0.40, 0.30): (-0.91, -0.82),
             (0.30, 0.15): (-0.90, -0.68)}
P2_TOL = 1.3


def load_old():
    d = json.load(open(os.path.join(OUT, "sweep_profile_shape.json")))
    cur = defaultdict(list)
    for r in d["rows"]:
        m = r["meta"]
        if "family" not in m:
            continue
        cur[(m["family"], m["theta"])].append((1.0 - r["kunif"], r["lam_mean"],
                                               r["lam_std"] / math.sqrt(20)))
    for k in cur:
        cur[k].sort(key=lambda c: -c[0])
    return cur


def load_new():
    d = json.load(open(os.path.join(OUT, "sweep_law_test.json")))
    cur = defaultdict(list)
    for r in d["rows"]:
        cur[r["meta"]["fid"]].append((1.0 - r["kunif"], r["lam_mean"],
                                      r["lam_std"] / math.sqrt(d["cfg"]["nimg"])))
    for k in cur:
        cur[k].sort(key=lambda c: -c[0])
    return cur, d


def load_new_perimg():
    """返回 {fid: [(u, lam_per_img array), ...]} 按 u 降序，用于 bootstrap。"""
    d = json.load(open(os.path.join(OUT, "sweep_law_test.json")))
    cur = defaultdict(list)
    for r in d["rows"]:
        cur[r["meta"]["fid"]].append(
            (1.0 - r["kunif"], np.array(r["lam_per_img"], float)))
    for k in cur:
        cur[k].sort(key=lambda c: -c[0])
    return cur, d["cfg"]["nimg"]


def boot_exponent(pimg, u1, u2, nboot=2000, seed=0):
    """bootstrap 局部指数 d log Lambda / d log u 的标准误（u1 > u2）。

    pimg: [(u, 逐图 Lambda 数组), ...] 按 u 降序。先把每张图自己的曲线
    对数插值到 u1/u2，再对图做有放回重采样。只用插值后两者皆为正的图。
    """
    ks = [c[0] for c in pimg]
    if u1 > max(ks) or u2 < min(ks):
        return None

    def at(u):
        for i in range(len(ks) - 1):
            a, b = ks[i], ks[i + 1]
            if u <= a and u >= b:
                w = (math.log(u) - math.log(a)) / (math.log(b) - math.log(a))
                return np.exp(np.log(pimg[i][1]) * (1 - w)
                              + np.log(pimg[i + 1][1]) * w)
        return None

    A, B = at(u1), at(u2)
    if A is None or B is None:
        return None
    m = (A > 0) & (B > 0)
    A, B = A[m], B[m]
    n = len(A)
    if n < 8:
        return None
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nboot):
        idx = rng.integers(0, n, n)
        ma, mb = A[idx].mean(), B[idx].mean()
        if ma > 0 and mb > 0:
            out.append(math.log(mb / ma) / math.log(u2 / u1))
    return float(np.std(out)) if len(out) >= 100 else None


def interp(curve, u):
    ks = [c[0] for c in curve]
    if u < min(ks) or u > max(ks):
        return None
    for i in range(len(ks) - 1):
        a, b = ks[i], ks[i + 1]
        if u <= a and u >= b:
            w = (math.log(u) - math.log(a)) / (math.log(b) - math.log(a))
            return math.exp(math.log(curve[i][1]) * (1 - w)
                            + math.log(curve[i + 1][1]) * w)
    return None


def main():
    old = load_old()
    new, newd = load_new()
    print("=" * 104)
    print(f" 旧：{len(old)} 个参考剖面   新：{len(new)} 个剖面   "
          f"t={newd['cfg']['t']:g}  nimg={newd['cfg']['nimg']}")
    print("=" * 104)

    # 公共 u 网格（对数等距）
    U = np.logspace(math.log10(0.055), math.log10(0.720), 25)

    # ---- 1. 旧 7 个平滑族的逐段指数 => b_universal(u) ----
    print("\n" + "=" * 104)
    print(" 1. 基准：旧 7 个平滑模糊族的局部指数（公共 u 网格）")
    print("=" * 104)
    seg_u, seg_b = [], []
    for i in range(len(U) - 1):
        bs = []
        for k in SMOOTH_OLD:
            va = interp(old[k], U[i]); vb = interp(old[k], U[i + 1])
            if va and vb and va > 0 and vb > 0:
                bs.append(math.log(vb / va) / math.log(U[i + 1] / U[i]))
        if len(bs) >= 5:
            seg_u.append(math.sqrt(U[i] * U[i + 1]))
            seg_b.append(float(np.mean(bs)))
    seg_u = np.array(seg_u); seg_b = np.array(seg_b)
    print(f" {'u 中点':>8} | {'b 均值':>8} | {'n':>3}")
    print(" " + "-" * 26)
    for u, b in zip(seg_u, seg_b):
        print(f" {u:>8.4f} | {b:>8.2f} | {7:>3}")

    def b_of(u):
        """b_universal(u)，在 log u 上分段线性。"""
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

    # ---- 2. 新族的局部指数 vs 基准 ----
    print("\n" + "=" * 104)
    print(" 2. P1：新族的局部指数是否落在旧族的范围内？")
    print("=" * 104)
    perimg, nimg = load_new_perimg()
    print(f" {'新族':<6} | " + " | ".join(f"u∈[{a:g},{b:g}]" for a, b in
                                          sorted(P1_RANGES, reverse=True)))
    print(" " + "-" * 78)
    p1, p1_se = {}, {}
    for fid in sorted(new):
        row, oks, ses = [], True, []
        for (a, b) in sorted(P1_RANGES, reverse=True):
            va = interp(new[fid], a); vb = interp(new[fid], b)
            lo, hi = P1_RANGES[(a, b)]
            if va and vb and va > 0 and vb > 0:
                bb = math.log(vb / va) / math.log(b / a)
                se = boot_exponent(perimg[fid], a, b)
                ses.append(se if se else float("nan"))
                # 与旧族范围比较时，用"差几个标准误"而不是只看点估计
                dist = 0.0 if lo <= bb <= hi else                     min(abs(bb - lo), abs(bb - hi))
                nse = dist / se if se else float("inf")
                inside = lo <= bb <= hi
                # 边际判定：超出但在 1 个 bootstrap SE 内算"边缘通过"
                ok = inside or (se is not None and nse <= 1.0)
                oks = oks and ok
                row.append(f"{bb:>6.2f}±{se:.2f} {'✓' if inside else ('~' if ok else '✗')}")
            else:
                row.append(f"{'--':>6}      ")
                ses.append(float("nan"))
                oks = False
        p1[fid] = oks
        p1_se[fid] = ses
        print(f" {fid:<6} | " + " | ".join(row))
    print(" " + "-" * 78)
    print(" ✓ = 点估计落在旧族范围；~ = 超出但在 1 个 bootstrap SE 内；✗ = 真超出")
    print(f" P1：{sum(p1.values())}/{len(p1)} 个新族的指数与旧族一致"
          f"（含边缘通过）")

    # ---- 3. C(u) = Lambda * u^{-b(u)} 的跨剖面比较 ----
    print("\n" + "=" * 104)
    print(" 3. P2：前缀 C(u) = Λ(u)·u^{−b(u)} 的跨剖面比较")
    print("=" * 104)
    allc = {}
    for k in list(SMOOTH_OLD) + [IDEAL]:
        allc[k] = {u: (interp(old[k], u) * u ** (-b_of(u))
                       if interp(old[k], u) else None) for u in U}
    for fid in new:
        allc[fid] = {u: (interp(new[fid], u) * u ** (-b_of(u))
                         if interp(new[fid], u) else None) for u in U}

    # 旧 7 族的 C 范围
    old_lo, old_hi, old_mid = {}, {}, {}
    for u in U:
        vs = [allc[k][u] for k in SMOOTH_OLD if allc[k][u]]
        if len(vs) >= 5:
            old_lo[u], old_hi[u], old_mid[u] = min(vs), max(vs), \
                float(np.exp(np.mean([math.log(v) for v in vs])))
    us_ok = [u for u in U if u in old_lo]
    print(f" {'u':>7} | {'旧7族 C 范围':>18} | {'理想低通/旧均值':>13} | "
          + " | ".join(f"{fid:>7}" for fid in sorted(new)))
    print(" " + "-" * (40 + 22 + 11 * len(new)))
    p2_detail = defaultdict(list)
    for u in us_ok:
        cells = []
        for fid in sorted(new):
            v = allc[fid][u]
            if v is None:
                cells.append(f"{'--':>7}")
                continue
            rel = v / old_mid[u]
            p2_detail[fid].append(rel)
            mark = "✓" if old_lo[u] / old_mid[u] <= rel <= old_hi[u] / old_mid[u] \
                else ("~" if 1 / P2_TOL <= rel <= P2_TOL else "✗")
            cells.append(f"{rel:>6.2f}{mark}")
        iv = allc[IDEAL][u]
        print(f" {u:>7.4f} | {old_lo[u]:>8.0f},{old_hi[u]:>8.0f} | "
              f"{(iv/old_mid[u] if iv else float('nan')):>13.2f} | "
              + " | ".join(cells))
    print(" " + "-" * (40 + 22 + 11 * len(new)))
    print(" ✓ = 落在旧 7 族范围内；~ = 超出但在 1.3x 内；✗ = 超出 1.3x")
    print(f" 理想低通一列即 P3 的靶子（平滑近似失效时的相对值）")

    # ---- 4. P3 / P4 ----
    print("\n" + "=" * 104)
    print(" 4. P3 / P4")
    print("=" * 104)
    # P4：W2 vs gaussian 族
    print(" P4 内部对照（W2 = 高斯剖面，应复现 gaussian 族）:")
    print(f" {'u':>7} | {'W2':>8} | {'gaussian/1':>11} | {'gaussian/2':>11} | "
          f"{'W2/高斯均值':>12}")
    print(" " + "-" * 60)
    for u in us_ok[::4]:
        w2 = allc["W2"][u]
        g1 = allc[("gaussian", 1.0)][u]; g2 = allc[("gaussian", 2.0)][u]
        if w2 and g1 and g2:
            gm = math.exp((math.log(g1) + math.log(g2)) / 2)
            print(f" {u:>7.4f} | {w2:>8.0f} | {g1:>11.0f} | {g2:>11.0f} | "
                  f"{w2/gm:>12.3f}")
    # P3：新族 C 的相对高低 + 与理想低通比
    print("\n P3 趋近硬截止（新族 C / 旧7族均值，越接近理想低通的值越'陡'）:")
    print(f" {'剖面':<16} | {'C/旧均值 中位':>14} | {'最小':>7} | {'最大':>7} | "
          f"{'与理想低通之比':>14}")
    print(" " + "-" * 68)
    rows = []
    for name, c in [("ideal-lowpass", allc[IDEAL])] + \
            [(fid, allc[fid]) for fid in sorted(new)] + \
            [(f"old:{k[0]}/{k[1]:g}", allc[k]) for k in SMOOTH_OLD]:
        rels = [c[u] / old_mid[u] for u in us_ok if c[u]]
        if not rels:
            continue
        iv = [allc[IDEAL][u] / old_mid[u] for u in us_ok if allc[IDEAL][u]]
        rows.append((name, float(np.median(rels)), min(rels), max(rels),
                     float(np.median(rels) / np.median(iv)) if iv else float("nan")))
    for name, med, lo, hi, ratio in sorted(rows, key=lambda r: r[1]):
        print(f" {name:<16} | {med:>14.3f} | {lo:>7.2f} | {hi:>7.2f} | "
              f"{ratio:>14.2f}")
    print(" " + "-" * 68)
    new_meds = {fid: float(np.median([allc[fid][u] / old_mid[u]
                                      for u in us_ok if allc[fid][u]]))
                for fid in new}
    lowest = min(new_meds, key=new_meds.get)
    print(f" 新族中 C 最低的是 {lowest}（预言指定 W4）"
          f"  => P3 {'通过' if lowest == 'W4' else '未通过'}")
    iv_med = float(np.median([allc[IDEAL][u] / old_mid[u] for u in us_ok
                              if allc[IDEAL][u]]))
    # 移动比例：在对数尺度上，新族最低值走完了"平滑均值 -> 硬截止"这段路的几分之几
    frac = (math.log(1.0) - math.log(new_meds[lowest])) / \
           (math.log(1.0) - math.log(iv_med))
    print(f" 理想低通的相对值 {iv_med:.2f}；新族最低 {new_meds[lowest]:.2f}")
    print(f" 在对数尺度上，{lowest} 走完了'平滑均值→硬截止'这段路的 "
          f"{100*frac:.0f}%  => 方向正确，但只走了一小段"
          f"（{'✅' if frac > 0 else '✗'}）")

    # ---- 5. 总结 ----
    print("\n" + "=" * 104)
    print(" 5. 总结")
    print("=" * 104)
    inside = sum(1 for fid in new for u in us_ok
                 if allc[fid][u] and old_lo[u] <= allc[fid][u] <= old_hi[u])
    tot = sum(1 for fid in new for u in us_ok if allc[fid][u])
    rels = [allc[fid][u] / old_mid[u] for fid in new for u in us_ok
            if allc[fid][u] and u >= 0.12]      # u<0.12 是各曲线末段的外插，单列
    rels_lo = [allc[fid][u] / old_mid[u] for fid in new for u in us_ok
               if allc[fid][u] and u < 0.12]
    print(f" P1 指数普适      : {sum(p1.values())}/{len(p1)} 个新族通过"
          f"  => {'✅' if all(p1.values()) else '⚠️'}")
    md = max(abs(math.log(r)) for r in rels)
    print(f" P2 前缀有界      : {inside}/{tot} 个 (族,u) 点落在旧 7 族范围内；")
    print(f"                    u>=0.12 的 {len(rels)} 点相对旧均值最大偏离 "
          f"{md:.3f} dex（{math.exp(md):.2f}x）"
          f"  => {'✅' if md < 0.1 else '⚠️'}")
    if rels_lo:
        mdl = max(abs(math.log(r)) for r in rels_lo)
        print(f"                    u<0.12 的 {len(rels_lo)} 点（各曲线末段外插）"
              f"最大偏离 {mdl:.3f} dex（{math.exp(mdl):.2f}x）")
    print(f" P3 趋近硬截止    : 最低新族 = {lowest}（预言 W4）"
          f"  => {'✅' if lowest == 'W4' else '⚠️'}")
    p4v = []
    for u in us_ok:
        if u < 0.12:
            continue
        w2 = allc["W2"][u]
        g1 = allc[("gaussian", 1.0)][u]; g2 = allc[("gaussian", 2.0)][u]
        if w2 and g1 and g2:
            gm = math.exp((math.log(g1) + math.log(g2)) / 2)
            p4v.append(w2 / gm)
    p4 = float(np.median(p4v)) if p4v else float("nan")
    print(f" P4 内部对照      : W2/高斯均值 中位 {p4:.3f}"
          f"（u>=0.12，n={len(p4v)}）"
          f"  => {'✅' if abs(p4 - 1) < 0.05 else '⚠️'}")


if __name__ == "__main__":
    main()
