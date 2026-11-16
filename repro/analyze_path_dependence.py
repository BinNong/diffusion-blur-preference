"""路径依赖的直接量化：同一个 kunif，沿不同路径得到的 Lambda 差多少？

背景（RESULTS §3.9(h4)）：alpha 路径给 d log Lambda/d log u = -0.98，
sigma 路径给 -0.43。当时只用两个点估 sigma 路径，很粗。

本脚本用 alpha 扫描的数据做**直接**比较：
  gaussian sigma=1 的 alpha 扫描 与 gaussian sigma=2 的 alpha 扫描
是两条穿过同一片 (kunif, Lambda) 区域的不同路径。在公共 kunif 区间上，
把两条曲线都插值到同一组 kunif 网格，逐点比 Lambda 的比值。
  比值恒定 => 路径无关（kunif 是完备描述量）
  比值随 kunif 变 => 路径依赖被直接量化

同时检查已有的其它描述量（kappa, width, tail）在匹配 kunif 处是否能解释差异。

用法： python3 analyze_path_dependence.py
"""

import json
import math
import os
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")


def loglog_interp(curve, k):
    """curve: [(kunif, lam, rec)] 按 kunif 升序；在 log-log 上线性插值。
    返回 dict(kunif, lam, rec)。"""
    ks = [c[0] for c in curve]
    if k < ks[0] or k > ks[-1]:
        return None
    lo = max([i for i, v in enumerate(ks) if v <= k] or [0])
    hi = min([i for i, v in enumerate(ks) if v >= k] or [len(ks) - 1])
    if lo == hi:
        return dict(kunif=k, lam=curve[lo][1], rec=curve[lo][2])
    w = (math.log(k) - math.log(ks[lo])) / (math.log(ks[hi]) - math.log(ks[lo]))
    return dict(kunif=k,
                lam=math.exp(math.log(curve[lo][1]) * (1 - w)
                             + math.log(curve[hi][1]) * w),
                rec=(curve[lo][2] if w < 0.5 else curve[hi][2]))


def main():
    p = os.path.join(OUT, "sweep_alpha.json")
    if not os.path.exists(p):
        print("[缺少]", p)
        return
    d = json.load(open(p))
    recs = d["records"]
    t0 = min(r["t"] for r in recs)
    nimg = d["cfg"]["nimg"]

    def snr(r):
        return r["lam_mean"] / r["lam_std"] * math.sqrt(nimg) \
            if r["lam_std"] > 0 else float("inf")

    good = [r for r in recs if r["t"] == t0 and r["lam_mean"] > 0
            and snr(r) >= 3.0]

    # 每条"路径"= 一个核的 alpha 扫描。注意：必须按 alpha 排序后才能判断
    # kunif 是否随 alpha 单调——按 kunif 排序会让单调性检查失效。
    paths = defaultdict(list)
    for r in good:
        paths[(r["family"], r["theta"])].append(r)
    for k in paths:
        paths[k].sort(key=lambda r: r["alpha"])

    print("=" * 104)
    print(" 每条路径的 kunif 覆盖")
    print("=" * 104)
    for k in sorted(paths):
        c = paths[k]
        print(f" {k[0]+'/'+format(k[1],'g'):<18} kunif {c[0]['kunif']:.4f} - "
              f"{c[-1]['kunif']:.4f}  ({math.log(c[-1]['kunif']/c[0]['kunif']):.3f} dex)"
              f"   Lambda {c[0]['lam_mean']:.0f} - {c[-1]['lam_mean']:.0f}")

    # ---- 两两比较：在公共 kunif 区间上插值后比 Lambda ----
    print()
    print("=" * 104)
    print(" 路径成对比较（只取 kunif 随 alpha 单调的路径）")
    print("=" * 104)
    keys = sorted(paths)
    valid = []
    for k in keys:
        c = paths[k]
        if all(c[i]["kunif"] < c[i + 1]["kunif"] for i in range(len(c) - 1)):
            valid.append(k)
        else:
            print(f" [排除] {k[0]}/{k[1]:g}：kunif 随 alpha 非单调，不是有效路径")
    print()
    print(f" {'路径A':<16} | {'路径B':<16} | {'公共kunif区间':>16} | {'B/A 起':>7} | "
          f"{'B/A 末':>7} | {'极差':>6}")
    print(" " + "-" * 88)
    summary = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            ka, kb = valid[i], valid[j]
            ca, cb = paths[ka], paths[kb]
            lo = max(c[0]["kunif"] for c in (ca, cb))
            hi = min(c[-1]["kunif"] for c in (ca, cb))
            if hi <= lo * 1.02:
                continue
            grid = [lo * (hi / lo) ** (m / 6.0) for m in range(7)]
            rows = []
            for k in grid:
                qa = loglog_interp([(c["kunif"], c["lam_mean"], c) for c in ca], k)
                qb = loglog_interp([(c["kunif"], c["lam_mean"], c) for c in cb], k)
                if qa is None or qb is None:
                    continue
                rows.append((k, qa["lam"], qb["lam"], qa["rec"], qb["rec"]))
            if len(rows) < 3:
                continue
            ratios = [b / a for _, a, b, _, _ in rows]
            spread = max(ratios) / min(ratios)
            # 路径依赖集中在高频段还是低频段？把公共区间分两半再看极差
            half = len(rows) // 2
            rlo = rows[:half + 1]
            rhi = rows[half:]
            def _sp(rs):
                rr = [b / a for _, a, b, _, _ in rs]
                return max(rr) / min(rr) if rr else float("nan")
            summary.append((spread, ka, kb, lo, hi, ratios, rows,
                            _sp(rlo), _sp(rhi)))

    print(f" {'路径A':<16} | {'路径B':<16} | {'B/A 起':>7} | {'B/A 末':>7} | "
          f"{'极差':>6} | {'低半段极差':>10} | {'高半段极差':>10}")
    print(" " + "-" * 92)
    for spread, ka, kb, lo, hi, ratios, rows, slo, shi in sorted(summary):
        print(f" {ka[0]+'/'+format(ka[1],'g'):<16} | {kb[0]+'/'+format(kb[1],'g'):<16} | "
              f"{ratios[0]:>7.3f} | {ratios[-1]:>7.3f} | {spread:>6.2f} | "
              f"{slo:>10.2f} | {shi:>10.2f}")
    print(" " + "-" * 92)
    if summary:
        sp = [s[0] for s in summary]
        sl = [s[7] for s in summary]
        sh = [s[8] for s in summary]
        print(f" {len(summary)} 对可比路径；全段极差中位数 "
              f"{sorted(sp)[len(sp)//2]:.2f}x，最大 {max(sp):.2f}x")
        print(f" 低半段(kunif 小)极差中位数 {sorted(sl)[len(sl)//2]:.2f}x；"
              f"高半段(kunif 大)极差中位数 {sorted(sh)[len(sh)//2]:.2f}x")
        print(" => 若高半段远大于低半段，路径依赖集中在强模糊端（kunif->1 的饱和区）")

    # 打印极差最大的三对的细节
    for spread, ka, kb, lo, hi, ratios, rows, slo, shi in sorted(
            summary, key=lambda s: -s[0])[:3]:
        print(f"\n 细节：{ka[0]}/{ka[1]:g} vs {kb[0]}/{kb[1]:g}（极差 {spread:.2f}x）")
        print(f" {'kunif':>8} | {'L_A':>9} | {'L_B':>9} | {'B/A':>6} | "
              f"{'kappa_A':>8} | {'kappa_B':>8} | {'tail_A':>10} | {'tail_B':>10}")
        print(" " + "-" * 84)
        for k, la, lb, ra, rb in rows:
            print(f" {k:>8.4f} | {la:>9.0f} | {lb:>9.0f} | {lb/la:>6.3f} | "
                  f"{ra['kappa']:>8.4f} | {rb['kappa']:>8.4f} | "
                  f"{ra['tail']:>10.2e} | {rb['tail']:>10.2e}")

    # ---- 全池：log Lambda 对 log u 回归，加 kappa / width 是否改善 ----
    print()
    print("=" * 104)
    print(" 全池回归：加描述量能否解释路径差异？")
    print("=" * 104)

    def ols(cols, ys):
        k = len(cols)
        X = [[1.0] + [c(r) for c in cols] for r in good]
        n = len(X)
        A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k + 1)]
             for a in range(k + 1)]
        v = [sum(X[i][a] * ys[i] for i in range(n)) for a in range(k + 1)]
        M = [A[r][:] + [v[r]] for r in range(k + 1)]
        for cc in range(k + 1):
            pp = max(range(cc, k + 1), key=lambda r: abs(M[r][cc]))
            if abs(M[pp][cc]) < 1e-12:
                return None
            M[cc], M[pp] = M[pp], M[cc]
            for r in range(k + 1):
                if r != cc:
                    f = M[r][cc] / M[cc][cc]
                    for q in range(cc, k + 2):
                        M[r][q] -= f * M[cc][q]
        coef = [M[r][k + 1] / M[r][r] for r in range(k + 1)]
        res = [ys[i] - sum(coef[a] * X[i][a] for a in range(k + 1))
               for i in range(n)]
        my = sum(ys) / n
        sst = sum((y - my) ** 2 for y in ys)
        ssr = sum(e * e for e in res)
        return coef, (1 - ssr / sst if sst > 0 else float("nan")), \
            math.sqrt(ssr / n)

    ys = [math.log(r["lam_mean"]) for r in good]
    U = lambda r: math.log(1.0 - r["kunif"])
    K = lambda r: math.log(r["kappa"])
    tests = [
        ("log u", [U]),
        ("log u + log kappa", [U, K]),
        ("log kappa", [K]),
    ]
    for name, cols in tests:
        f = ols(cols, ys)
        if f is None:
            print(f" {name:<22} | 跳过（共线或样本不足）")
            continue
        coef, r2, sd = f
        print(f" {name:<22} | R^2={r2:.4f}  残差={sd:.3f}dex  系数="
              f"{[round(c, 3) for c in coef]}")

    # 残差是否按"路径"（核）聚集
    f = ols([U], ys)
    if f:
        coef, r2, sd = f
        g = defaultdict(list)
        for r, y in zip(good, ys):
            pred = coef[0] + coef[1] * U(r)
            g[(r["family"], r["theta"])].append(y - pred)
        print(f"\n 只用 log u 拟合后，各路径的平均残差（dex）：")
        allm = sum(y - (coef[0] + coef[1] * U(r))
                   for r, y in zip(good, ys)) / len(good)
        for k in sorted(g):
            m = sum(g[k]) / len(g[k])
            print(f"   {k[0]+'/'+format(k[1],'g'):<18} n={len(g[k]):>2}  "
                  f"残差={m:+.3f}  倍数={math.exp(m-allm):.2f}")
        print(" 倍数偏离 1 越多，说明该路径越不能被 log u 单独解释")


if __name__ == "__main__":
    main()
