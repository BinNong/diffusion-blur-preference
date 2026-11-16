"""离线选形：为 B2 预言检验挑新的剖面族（不涉及任何 Λ 测量，无循环）。

在 knif=T 上算每个候选族的 B2，并用已在 17 个剖面上拟合的
    C = 4.310 - 5.887 * B2        (T = 0.45)
给出预言值。选族只看剖面定义，所以不构成事后挑选。

用法： venv2/bin/python pick_families.py [--target 0.45]
"""
import argparse
import importlib.util
import json
import math
import os

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

spec = importlib.util.spec_from_file_location(
    "ast", os.path.join(PROJ, "repro", "analyze_steepness.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

r = m.halfplane()


def rc(frac, kind="cos"):
    """升余弦/三次 Hermite 过渡：平通带 + S 边 + 平阻带。"""
    r1 = (1.0 - frac) / 2.0
    r2 = r1 + frac
    g2 = np.ones_like(r)
    msk = (r > r1) & (r < r2)
    t = (r[msk] - r1) / (r2 - r1)
    g2[msk] = 0.5 * (1 + np.cos(np.pi * t)) if kind == "cos" \
        else (3 * t ** 2 - 2 * t ** 3)
    g2[r >= r2] = 0.0
    return 1.0 - g2


def ramp(q):
    """a = (r/rmax)^q，调 rmax 命中目标 knif。"""
    lo, hi = 1e-3, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ((r / mid) ** q).mean() < T:
            hi = mid
        else:
            lo = mid
    return (r / (0.5 * (lo + hi))) ** q


def logistic(w):
    """logistic 边：a = 1/(1+exp(-(r-r0)/w))，r0 使 knif 命中。"""
    lo, hi = 1e-3, 3.0
    for _ in range(60):
        r0 = 0.5 * (lo + hi)
        a = 1.0 / (1.0 + np.exp(-(r - r0) / w))
        if a.mean() < T:
            hi = r0
        else:
            lo = r0
    r0 = 0.5 * (lo + hi)
    return 1.0 / (1.0 + np.exp(-(r - r0) / w))


def erf_edge(w):
    """erf 边：a = 0.5(1+erf((r-r0)/w))。"""
    from math import erf as _erf
    lo, hi = 1e-3, 3.0
    for _ in range(60):
        r0 = 0.5 * (lo + hi)
        a = 0.5 * (1.0 + np.vectorize(_erf)((r - r0) / w))
        if a.mean() < T:
            hi = r0
        else:
            lo = r0
    r0 = 0.5 * (lo + hi)
    return 0.5 * (1.0 + np.vectorize(_erf)((r - r0) / w))


def notch(w):
    """平通带 + 高斯阻带边。"""
    g2 = np.exp(-((np.maximum(r - 0.35, 0.0)) / w) ** 2)
    return 1.0 - g2


def main():
    global T
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.45)
    args = ap.parse_args()
    T = args.target

    cands = []
    for f in [0.25, 0.3, 0.4, 0.5, 0.7, 0.9]:
        cands.append(("RC-cos f=%.2f" % f, rc(f, "cos")))
    for f in [0.3, 0.5, 0.7]:
        cands.append(("RC-cubic f=%.2f" % f, rc(f, "cubic")))
    for q in [0.35, 0.5, 0.75, 1.0, 1.5, 3.0]:
        cands.append(("ramp q=%.2f" % q, ramp(q)))
    for w in [0.1, 0.25, 0.5]:
        cands.append(("notch w=%.2f" % w, notch(w)))
    for w in [0.10, 0.20, 0.35]:
        cands.append(("logistic w=%.2f" % w, logistic(w)))
    for w in [0.10, 0.20, 0.35]:
        cands.append(("erf w=%.2f" % w, erf_edge(w)))
    for ratio in [1.5, 2.0, 3.0, 10.0]:
        sc = m.tune_rc("twoscale", ratio, r)
        cands.append(("twoscale R=%.1f" % ratio,
                      1.0 - m.make_profile("twoscale", ratio, sc, r)))

    # 已在 17 个剖面上拟合的系数。标度协变：c0 恒定，c1 = -3.532/T
    c0 = 4.642
    c1 = -3.532 / T
    print("目标 knif = %.4f；拟合式 C = %.3f + (%.3f)*B2（17 个已测剖面）"
          % (T, c0, c1))
    lo = 0.607 * (T / 0.60)
    hi = 0.668 * (T / 0.60)
    print("已测 17 族在 knif=%.4f 上的 B2 范围 = [%.3f, %.3f]（内插区）"
          % (T, lo, hi))
    print("%-18s %10s %9s %9s %s" % ("候选新族", "knif_orig", "B2", "C_pred",
                                      "内插/外推"))
    print("-" * 50)
    for nm, a in cands:
        asc = a * (T / a.mean())
        b = float((asc ** 2).mean() / asc.mean())
        tag = "内插" if lo <= b <= hi else ("近外推" if b < hi * 1.15
                                            else "远外推")
        print("%-18s %10.4f %9.3f %9.3f %s"
              % (nm, a.mean(), b, c0 + c1 * b, tag))


if __name__ == "__main__":
    main()
