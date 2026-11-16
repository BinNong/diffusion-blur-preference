"""纯 stdlib 分析：幂律拟合的杠杆结构与局部斜率（无需 numpy）

动机：sweep_lambda_theta 的 theta 档位从 0.5/1.0 起跳，导致 kunif 几乎全部
堆积在 0.92-0.99 的饱和区。若 log Lambda = a + b log(kunif) 的斜率几乎由
最弱模糊那一个点撑起来，则"指数"不是核的性质，而是 theta 网格的性质。

输出：
  1. 每个 (模型, 族) 的 (theta, kunif, Lambda) 表 + 相邻点局部斜率
  2. 全局拟合的杠杆（hat 值）与 Cook 距离 —— 谁在决定指数
  3. Lambda 随模糊强度的峰值位置（幂律只在峰值左侧成立）
  4. 跨族在相同 kunif 处的 Lambda 比值（真正该比的量）

用法： python3 analyze_local_slope.py
"""

import json
import math
import os
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")


def ols(xs, ys):
    """一元线性回归，返回 (a, b, r2, resid)。"""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0:
        return None
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return a, b, r2, [y - (a + b * x) for x, y in zip(xs, ys)]


def leverage(xs):
    """一元回归的 hat 值 h_i = 1/n + (x_i-mx)^2/Sxx。"""
    n = len(xs)
    mx = sum(xs) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return [float("nan")] * n
    return [1.0 / n + (x - mx) ** 2 / sxx for x in xs]


def load(fn):
    d = json.load(open(os.path.join(OUT, fn)))
    return d["records"]


def group(recs, tmin=None):
    g = defaultdict(list)
    for r in recs:
        g[(r["family"], round(r["theta"], 4))].append(r)
    ts = sorted({r["t"] for r in recs})
    t0 = tmin if tmin is not None else min(ts)
    out = {}
    for k, v in g.items():
        rr = [x for x in v if x["t"] == t0]
        if rr:
            out[k] = rr[0]
    return out, t0


def main():
    files = {
        "ffhq": "sweep_lambda_theta_ffhq.json",
        "afhq": "sweep_lambda_theta_afhqv2-64x64-uncond-vp.json",
        "imnet": "sweep_lambda_theta_imagenet-64x64-cond-adm.json",
    }
    data = {}
    for tag, fn in files.items():
        if not os.path.exists(os.path.join(OUT, fn)):
            continue
        recs = load(fn)
        if "kunif" not in recs[0]:
            # 族扫描没记 kunif，从 spectral_functionals.json 取
            continue
        data[tag] = group(recs)

    # spectral_functionals.json 同时有 kunif 和 lam，用它做主表
    sf = json.load(open(os.path.join(OUT, "spectral_functionals.json")))["rows"]
    by = defaultdict(dict)
    for r in sf:
        by[r["tag"]][(r["family"], round(r["theta"], 4))] = r

    print("#" * 108)
    print("# 1. 每个 (模型, 族)：kunif 覆盖范围 + 相邻点局部斜率 d log Lambda / d log kunif")
    print("#" * 108)
    print(f"{'模型/族':<22} | {'kunif 范围':>17} | {'跨度(dex)':>9} | "
          f"{'Lambda 范围':>20} | {'峰值 theta':>9}")
    print("-" * 108)

    local = {}
    for tag in sorted(by):
        fams = sorted({f for (f, _) in by[tag]})
        for fam in fams:
            pts = sorted([(r["kunif"], r["lam"], r["theta"])
                          for (f, th), r in by[tag].items()
                          if f == fam and r["lam"] > 0 and r["kunif"] > 1e-9],
                         key=lambda p: p[0])
            if len(pts) < 3:
                continue
            kmin, kmax = pts[0][0], pts[-1][0]
            print(f"{tag+'/'+fam:<22} | {kmin:>8.4f} - {kmax:>7.4f} | "
                  f"{math.log(kmax)-math.log(kmin):>9.3f} | "
                  f"{pts[0][1]:>9.0f} - {pts[-1][1]:>9.0f} | "
                  f"{max(pts, key=lambda p: p[1])[2]:>9g}")
            sl = []
            for i in range(len(pts) - 1):
                dk = math.log(pts[i + 1][0] / pts[i][0])
                dl = math.log(pts[i + 1][1] / pts[i][1])
                if abs(dk) > 1e-9:
                    sl.append((pts[i][0], pts[i + 1][0], dl / dk))
            local[(tag, fam)] = sl

    print()
    print("#" * 108)
    print("# 2. 局部斜率表（kunif 区间 -> 斜率）。斜率随 kunif 大幅变化 => 不是幂律")
    print("#" * 108)
    for (tag, fam), sl in sorted(local.items()):
        if not sl:
            continue
        s = "  ".join(f"[{a:.3f}->{b:.3f}]:{v:+.2f}" for a, b, v in sl)
        rng = max(v for _, _, v in sl) - min(v for _, _, v in sl)
        print(f"  {tag+'/'+fam:<20} 极差 {rng:5.2f}   {s}")

    print()
    print("#" * 108)
    print("# 3. 全局幂律拟合的杠杆：谁在决定指数 b？")
    print("#" * 108)
    print(f"{'模型/族':<22} | {'n':>2} | {'b':>6} | {'R^2':>6} | "
          f"{'最大 hat 的点':>14} | {'hat':>6} | {'去掉它后 b':>10} | {'变化':>7}")
    print("-" * 108)
    for tag in sorted(by):
        fams = sorted({f for (f, _) in by[tag]})
        for fam in fams:
            pts = sorted([(r["kunif"], r["lam"], r["theta"])
                          for (f, th), r in by[tag].items()
                          if f == fam and r["lam"] > 0 and r["kunif"] > 1e-9],
                         key=lambda p: p[0])
            if len(pts) < 4:
                continue
            xs = [math.log(p[0]) for p in pts]
            ys = [math.log(p[1]) for p in pts]
            fit = ols(xs, ys)
            if fit is None:
                continue
            a, b, r2, _ = fit
            hs = leverage(xs)
            i = max(range(len(pts)), key=lambda j: hs[j])
            rest = [p for j, p in enumerate(pts) if j != i]
            xs2 = [math.log(p[0]) for p in rest]
            ys2 = [math.log(p[1]) for p in rest]
            f2 = ols(xs2, ys2)
            b2 = f2[1] if f2 else float("nan")
            print(f"{tag+'/'+fam:<22} | {len(pts):>2} | {b:>6.2f} | {r2:>6.3f} | "
                  f"{('theta=' + format(pts[i][2], 'g')):>14} | {hs[i]:>6.3f} | "
                  f"{b2:>10.2f} | {b2-b:>+7.2f}")

    print()
    print("#" * 108)
    print("# 4. 跨族：在相同 kunif 处 Lambda 相差多少（插值到公共网格）")
    print("#" * 108)
    for tag in sorted(by):
        fams = sorted({f for (f, _) in by[tag]})
        curves = {}
        for fam in fams:
            pts = sorted([(r["kunif"], r["lam"]) for (f, th), r in by[tag].items()
                          if f == fam and r["lam"] > 0 and r["kunif"] > 1e-9])
            if len(pts) >= 2:
                curves[fam] = pts
        if len(curves) < 2:
            continue
        allk = sorted({k for c in curves.values() for k, _ in c})
        common = [k for k in allk if all(
            min(k for k, _ in c) <= k <= max(k for k, _ in c) for c in curves.values())]
        if not common:
            continue
        print(f"\n  [{tag}] 公共 kunif 网格 {len(common)} 点，"
              f"{common[0]:.3f} - {common[-1]:.3f}")
        grid = sorted(set(round(k, 3) for k in
                          [common[0] + (common[-1] - common[0]) * i / 8
                           for i in range(9)]))
        hdr = f"  {'kunif':>7} | " + " | ".join(f"{f:>9}" for f in curves)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for k in grid:
            cells = []
            for fam, c in curves.items():
                ks = [p[0] for p in c]
                if k < min(ks) or k > max(ks):
                    cells.append(f"{'--':>9}")
                    continue
                # 线性插值 log-log
                lo = max([j for j, kk in enumerate(ks) if kk <= k] or [0])
                hi = min([j for j, kk in enumerate(ks) if kk >= k] or [len(ks) - 1])
                if lo == hi:
                    cells.append(f"{c[lo][1]:>9.0f}")
                    continue
                lk = (math.log(k) - math.log(ks[lo])) / \
                     (math.log(ks[hi]) - math.log(ks[lo]))
                v = math.exp(math.log(c[lo][1]) * (1 - lk) +
                             math.log(c[hi][1]) * lk)
                cells.append(f"{v:>9.0f}")
            print(f"  {k:>7.3f} | " + " | ".join(cells))

    print()
    print("#" * 108)
    print("# 5. Butterworth / tail 受控实验的同样检查（kunif 是否饱和）")
    print("#" * 108)
    for fn in ["sweep_butterworth.json", "sweep_tail.json"]:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            continue
        recs = load(fn)
        g, t0 = group(recs)
        print(f"\n  [{fn}] t={t0:g}")
        print(f"  {'参数':<26} | {'kunif':>8} | {'宽度':>7} | {'Lambda':>10} | "
              f"{'Lambda_Gauss':>12}")
        for k in sorted(g, key=lambda k: g[k]["kunif"]):
            r = g[k]
            lab = ", ".join(f"{kk}={r[kk]:g}" for kk in
                            ("order", "rho_c", "sigma", "delta") if kk in r)
            print(f"  {lab:<26} | {r['kunif']:>8.4f} | {r.get('width', float('nan')):>7.3f} | "
                  f"{r['lam_mean']:>10.1f} | {r['lam_gauss_mean']:>12.1f}")


if __name__ == "__main__":
    main()
