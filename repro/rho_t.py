"""ρ(t,u) = Λ(t,u)/Λ_0(u)：偏好衰减因子的 u 依赖性（离线，零 GPU）。

E1 已证恒等式 Λ(t,u) = Λ_0(u) − G(t,u)，所以 ρ = 1 − G/Λ_0。
param_t.py 的 E3/E4/E6 显示：G 的主体 ∝ Λ_0(u)（即 ρ 近似与 u 无关），
残差 2–4% 承载 b(u) 的全部 t 依赖性。本脚本把 ρ(t,u) 的完整表格打出来，
并检验 (a) ρ 的 u 散度随 t 怎么变；(b) ρ(t) 在 u 上平均后是否 ≈ 1 − k·t 再饱和。

用法： venv2/bin/python rho_t.py
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
DENSE = ["W2", "W8", "TK50"]
T0 = 0.002


def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    prof = defaultdict(dict)
    for rec in d["rows"]:
        prof[rec["meta"]["fid"]][round(1.0 - rec["kunif"], 4)] = rec["lam_mean"]
    return d, {k: dict(v) for k, v in prof.items()}


def grid(prof, fids, lo=0.13, hi=0.86):
    sets = [set(prof[f]) for f in fids]
    ug_all = sorted(set.intersection(*sets), reverse=True)
    return [u for u in ug_all if lo <= u <= hi]


def main():
    ra = load("sweep_bdense.json")
    pa = dict(ra[1])
    rl = load("sweep_blow.json")
    if rl is not None:
        for fid, dd in rl[1].items():
            pa.setdefault(fid, {}).update(dd)
    cells = [(T0, pa)]
    for fn in sorted(glob.glob(os.path.join(OUT, "sweep_bt*.json"))):
        r = load(os.path.basename(fn))
        if r is None:
            continue
        cells.append((float(r[0]["cfg"]["t"]), dict(r[1])))
    cells.sort(key=lambda c: c[0])
    ts = [t for t, _ in cells]
    C = dict(cells)
    ug = grid(pa, DENSE)

    LAM = {f: {t: np.array([C[t][f][round(u, 4)] for u in ug]) for t in ts}
           for f in DENSE}
    RHO = {f: {t: LAM[f][t] / LAM[f][T0] for t in ts} for f in DENSE}

    print("=" * 104)
    print(" 1. ρ(t,u) = Λ(t,u)/Λ_0(u)（三族各自 + 均值 + u 散度）")
    print("=" * 104)
    print(f" {'u':>6} | " + " | ".join(
        f"{'t=' + format(t, 'g'):>22}" for t in ts[1:]))
    print(" " + "-" * (10 + 25 * (len(ts) - 1)))
    rho_mean = {}
    for i, u in enumerate(ug):
        row = []
        for t in ts[1:]:
            v = [RHO[f][t][i] for f in DENSE]
            row.append(f"{np.mean(v):>6.3f}[{min(v):.3f},{max(v):.3f}]")
        rho_mean[u] = {t: float(np.mean([RHO[f][t][i] for f in DENSE]))
                       for t in ts}
        print(f" {u:>6.3f} | " + " | ".join(row))
    print(" " + "-" * (10 + 25 * (len(ts) - 1)))
    print(" 方括号是三族的范围。若每列的方括号很窄，则 ρ 近似与 u 无关（=> b 近似 t 不变）。")

    print("\n" + "=" * 104)
    print(" 2. ρ 的 u 散度随 t 的变化（正是 b(u) t 依赖性的来源）")
    print("=" * 104)
    print(f" {'t':>7} | {'ρ(u=0.85)':>11} | {'ρ(u=0.14)':>11} | "
          f"{'高/低比':>9} | {'1-ρ 均值':>10} | {'(1-ρ)/t':>9}")
    print(" " + "-" * 68)
    for t in ts[1:]:
        hi_ = float(np.mean([RHO[f][t][0] for f in DENSE]))
        lo_ = float(np.mean([RHO[f][t][-1] for f in DENSE]))
        one_minus = 1.0 - float(np.mean([np.mean(RHO[f][t]) for f in DENSE]))
        print(f" {t:>7g} | {hi_:>11.4f} | {lo_:>11.4f} | {hi_/lo_:>9.3f} | "
              f"{one_minus:>10.4f} | {one_minus/t:>9.2f}")
    print(" " + "-" * 68)
    print(" 「高/低比」= ρ(u=0.85)/ρ(u=0.14)。它偏离 1 越多，b(u) 的 t 依赖性越大。")

    print("\n" + "=" * 104)
    print(" 3. 1−ρ 对 t：线性区与饱和区")
    print("=" * 104)
    ta = np.array(ts[1:])
    om = np.array([1.0 - float(np.mean([np.mean(RHO[f][t]) for f in DENSE]))
                   for t in ts[1:]])
    print(f" {'t':>7} | {'1-ρ':>8} | {'(1-ρ)/t':>9} | {'线性外推 1-23t':>14} | "
          f"{'差':>8}")
    print(" " + "-" * 56)
    for i, t in enumerate(ts[1:]):
        lin = 23.0 * t
        print(f" {t:>7g} | {om[i]:>8.4f} | {om[i]/t:>9.2f} | {lin:>14.4f} | "
              f"{om[i]-lin:>+8.4f}")
    print(" " + "-" * 56)
    print(" 若 (1-ρ)/t 在小 t 近似常数、大 t 掉下来，说明是「线性衰减 + 饱和」。")
    # 饱和型拟合：1-ρ = A*t/(B+t)
    from scipy.optimize import curve_fit
    try:
        popt, _ = curve_fit(lambda t, A, B: A * t / (B + t), ta, om,
                            p0=(1.0, 0.02), bounds=([0.01, 1e-4], [5, 1]),
                            maxfev=200000)
        pred = popt[0] * ta / (popt[1] + ta)
        r = float(np.sqrt(np.mean((om - pred) ** 2)))
        print(f"\n 饱和型 1-ρ = A·t/(B+t)：A={popt[0]:.4f}  B={popt[1]:.5f}  "
              f"rms={r:.4f}（1-ρ 的量级 {om.mean():.3f}）")
        print(f"   => ρ(t) ≈ 1 − {popt[0]:.3f}·t/(t+{popt[1]:.4f})，"
              f"渐近 1-ρ → {popt[0]:.3f}")
    except Exception as e:
        print(f" [饱和型拟合失败：{e}]")

    print("\n" + "=" * 104)
    print(" 4. 结论句")
    print("=" * 104)
    print(f" ρ 从 t=0.002 的 1.0 降到 t=0.05 的 "
          f"{float(np.mean([np.mean(RHO[f][0.05]) for f in DENSE])):.3f}；")
    print(f" 同时 ρ 的 u 散度（u=0.85 vs u=0.14）从 "
          f"{float(np.mean([RHO[f][0.005][0] for f in DENSE]))/float(np.mean([RHO[f][0.005][-1] for f in DENSE])):.3f}"
          f"（t=0.005）张到 "
          f"{float(np.mean([RHO[f][0.05][0] for f in DENSE]))/float(np.mean([RHO[f][0.05][-1] for f in DENSE])):.3f}"
          f"（t=0.05）。")
    print(" 所以：**偏好随 t 近线性衰减、速率近似与 u 无关；残差的 u 结构随 t 放大，"
          "这就是 b(u) 的 t 依赖性。**")


if __name__ == "__main__":
    main()
