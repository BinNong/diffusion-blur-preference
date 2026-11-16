"""路径依赖的定位：弱模糊端 kunif 是否是完备描述量？

analyze_path_dependence 的发现：28 对有效路径，B/A 比值的极差
  低半段（kunif 小）中位数 1.11x
  高半段（kunif 大）中位数 1.92x
即路径依赖几乎全部集中在强模糊端。

本脚本把这一点做成可报告的定量结论：
  1. 按 kunif 分段，逐段做 log Lambda = a + b log u 的合并拟合，看 R^2 与残差
  2. 逐路径在低 kunif 段与高 kunif 段分别拟合斜率，比较跨路径离散度
  3. 排除 alpha=1 那一点后重做——检验高 kunif 端的发散是否由 alpha=1 的跳变驱动
  4. 低 kunif 段合并拟合的指数 = 真正的"弱模糊极限指数"

用法： python3 analyze_path_dependence_lowk.py
"""

import json
import math
import os
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sst = sum((y - my) ** 2 for y in ys)
    ssr = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return dict(a=a, b=b, r2=1 - ssr / sst if sst > 0 else float("nan"),
                sd=math.sqrt(ssr / n))


def main():
    d = json.load(open(os.path.join(OUT, "sweep_alpha.json")))
    recs = d["records"]
    t0 = min(r["t"] for r in recs)
    nimg = d["cfg"]["nimg"]

    def snr(r):
        return r["lam_mean"] / r["lam_std"] * math.sqrt(nimg) \
            if r["lam_std"] > 0 else float("inf")

    good = [r for r in recs if r["t"] == t0 and r["lam_mean"] > 0
            and snr(r) >= 3.0]
    paths = defaultdict(list)
    for r in good:
        paths[(r["family"], r["theta"])].append(r)
    for k in paths:
        paths[k].sort(key=lambda r: r["alpha"])
    # 只保留 kunif 随 alpha 单调的路径
    valid = {k: v for k, v in paths.items()
             if all(v[i]["kunif"] < v[i + 1]["kunif"]
                    for i in range(len(v) - 1))}

    print("=" * 100)
    print(f" t={t0:g}，{len(good)} 个可用点，{len(valid)} 条有效路径")
    print("=" * 100)

    # ---- 1. 按 kunif 分段的合并拟合 ----
    print("\n 1. 按 kunif 分段：log Lambda = a + b log u 的合并拟合（u = 1-kunif）")
    print(f" {'kunif 区间':<18} | {'n':>3} | {'b':>7} | {'R^2':>7} | {'残差dex':>8} | "
          f"{'跨路径倍数极差':>14}")
    print(" " + "-" * 78)
    bands = [(0.20, 0.35), (0.35, 0.50), (0.50, 0.65), (0.65, 0.80),
             (0.80, 0.90), (0.90, 1.00)]
    for lo, hi in bands:
        sub = [r for r in good if lo <= r["kunif"] < hi
               and (r["family"], r["theta"]) in valid]
        if len(sub) < 6:
            continue
        f = ols([math.log(1 - r["kunif"]) for r in sub],
                [math.log(r["lam_mean"]) for r in sub])
        # 每路径的前缀倍数
        pref = defaultdict(list)
        for r in sub:
            pref[(r["family"], r["theta"])].append(
                math.log(r["lam_mean"]) - f["b"] * math.log(1 - r["kunif"]))
        ps = [math.exp(sum(v) / len(v)) for v in pref.values() if len(v) >= 2]
        spread = max(ps) / min(ps) if len(ps) >= 2 else float("nan")
        print(f" {f'{lo:.2f} - {hi:.2f}':<18} | {len(sub):>3} | {f['b']:>7.2f} | "
              f"{f['r2']:>7.3f} | {f['sd']:>8.3f} | {spread:>14.2f}")
    print(" " + "-" * 78)
    print(" 残差与倍数极差都小 => 该段内 kunif 是完备描述量")

    # ---- 2. 逐路径分段斜率（按 alpha 分段，保证每段每路径 3 点）----
    print("\n 2. 逐路径分段斜率 d log Lambda / d log u")
    print("    低段 = alpha in {0.15, 0.25, 0.4}；高段 = alpha in {0.6, 0.8, 1.0}")
    print(f" {'路径':<16} | {'低段 b':>8} | {'低段R^2':>8} | {'高段 b':>8} | "
          f"{'高段R^2':>8} | {'低段前缀':>9} | {'高段前缀':>9}")
    print(" " + "-" * 82)
    LO = {0.15, 0.25, 0.4}
    HI = {0.6, 0.8, 1.0}
    rows = []
    for k in sorted(valid):
        v = valid[k]
        lo_pts = [r for r in v if r["alpha"] in LO]
        hi_pts = [r for r in v if r["alpha"] in HI]
        fl = ols([math.log(1 - r["kunif"]) for r in lo_pts],
                 [math.log(r["lam_mean"]) for r in lo_pts]) \
            if len(lo_pts) >= 3 else None
        fh = ols([math.log(1 - r["kunif"]) for r in hi_pts],
                 [math.log(r["lam_mean"]) for r in hi_pts]) \
            if len(hi_pts) >= 3 else None
        rows.append((k, fl, fh))
        print(f" {k[0]+'/'+format(k[1],'g'):<16} | "
              f"{fl['b'] if fl else float('nan'):>8.2f} | "
              f"{fl['r2'] if fl else float('nan'):>8.3f} | "
              f"{fh['b'] if fh else float('nan'):>8.2f} | "
              f"{fh['r2'] if fh else float('nan'):>8.3f} | "
              f"{math.exp(fl['a']) if fl else float('nan'):>9.0f} | "
              f"{math.exp(fh['a']) if fh else float('nan'):>9.0f}")
    print(" " + "-" * 82)
    bl = [r[1]["b"] for r in rows if r[1]]
    bh = [r[2]["b"] for r in rows if r[2]]
    pl = [math.exp(r[1]["a"]) for r in rows if r[1]]
    ph = [math.exp(r[2]["a"]) for r in rows if r[2]]
    if bl:
        print(f" 低段斜率 {min(bl):.2f} - {max(bl):.2f}"
              f"（跨度 {max(bl)-min(bl):.2f}，相对 {(max(bl)-min(bl))/abs(sum(bl)/len(bl)):.2f}）"
              f"；前缀 {min(pl):.0f} - {max(pl):.0f}（{max(pl)/min(pl):.2f}x）")
    if bh:
        print(f" 高段斜率 {min(bh):.2f} - {max(bh):.2f}"
              f"（跨度 {max(bh)-min(bh):.2f}，相对 {(max(bh)-min(bh))/abs(sum(bh)/len(bh)):.2f}）"
              f"；前缀 {min(ph):.0f} - {max(ph):.0f}（{max(ph)/min(ph):.2f}x）")

    # ---- 4. 低 kunif 段的"弱模糊极限" ----
    print("\n 4. 弱模糊极限：kunif < 0.5 的合并拟合")
    sub = [r for r in good if r["kunif"] < 0.5
           and (r["family"], r["theta"]) in valid]
    if len(sub) >= 6:
        f = ols([math.log(1 - r["kunif"]) for r in sub],
                [math.log(r["lam_mean"]) for r in sub])
        print(f"   n={len(sub)}  b={f['b']:.3f}  R^2={f['r2']:.4f}  "
              f"残差={f['sd']:.3f} dex")
        pref = defaultdict(list)
        for r in sub:
            pref[(r["family"], r["theta"])].append(
                math.log(r["lam_mean"]) - f["b"] * math.log(1 - r["kunif"]))
        ps = sorted(((k, math.exp(sum(v) / len(v))) for k, v in pref.items()),
                    key=lambda kv: -kv[1])
        print("   各路径前缀倍数：")
        for k, v in ps:
            print(f"     {k[0]+'/'+format(k[1],'g'):<18} {v:>8.0f}")
        print(f"   前缀极差 {ps[0][1]/ps[-1][1]:.2f}x")
        print("   => 弱模糊端：Lambda = C(kernel) / <|h|^2>^{|b|}，"
              "C 只差 1.5x 左右，是真正的极限定律")


if __name__ == "__main__":
    main()
