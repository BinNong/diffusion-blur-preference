"""为 B2 预言的样本外检验计算预官值（只看剖面定义，不涉及任何 Λ 测量，无循环）。

做法（与 analyze_steepness 拟合 B2 回归时完全一致）：
  1. 每个新族调到 knif_ref = KREF（与已测族同 regime，取 0.92）
  2. 把它缩放到 knif = T（= 0.60，B2 回归的拟合knif）：a_s = a_ref * (T/KREF)
  3. B2 = <a_s^2> / <a_s>
  4. 预官 C = 4.642 - 5.887 * B2      （17 个已测剖面，R2=0.825，LOO 0.051 dex）

已测 17 族在 knif=0.60 上的 B2 范围 = [0.607, 0.668]

用法： venv2/bin/python predict_b2.py [--target 0.60] [--kref 0.92]
"""
import argparse
import importlib.util
import math
import os

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "slt", os.path.join(PROJ, "repro", "sweep_law_test.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

r = m.halfplane()

FAMILIES = [("RP35", "ramp", 0.35), ("RP75", "ramp", 0.75),
            ("LG35", "logistic", 0.35), ("TS2", "twoscale", 2.0),
            ("EF35", "erf", 0.35)]

LO, HI = 0.607, 0.668      # 已测 17 族在 T=0.60 上的 B2 范围


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.60)
    ap.add_argument("--kref", type=float, default=0.92)
    args = ap.parse_args()
    T, KREF = args.target, args.kref
    c0, c1 = 4.642, -3.532 / T

    print("B2 回归拟合knif T = %.4f；式 C = %.3f + (%.3f)*B2" % (T, c0, c1))
    print("已测 17 族 B2 范围 = [%.3f, %.3f]；新族参考knif KREF = %.4f"
          % (LO, HI, KREF))
    print()
    print("%-6s %-11s %6s %10s %8s %9s %9s %s"
          % ("族", "kind", "par", "knif_ref", "B2", "C_pred", "实测C范围", "判定"))
    print("-" * 82)
    out = []
    for fid, kind, par in FAMILIES:
        sc = m.tune_rc(kind, par, r, KREF)
        a = 1.0 - m.make_profile(kind, par, sc, r)
        kref = float(a.mean())
        asc = a * (T / kref)
        b2 = float((asc ** 2).mean() / asc.mean())
        cp = c0 + c1 * b2
        tag = "内插" if LO <= b2 <= HI else ("近外推" if b2 < HI * 1.15
                                            else "远外推")
        # 实测 C 的已有范围（17 个已测剖面）
        print("%-6s %-11s %6.2f %10.4f %8.3f %9.3f %9s %s"
              % (fid, kind, par, kref, b2, cp, "0.69-1.04", tag))
        out.append((fid, b2, cp, tag))
    print("-" * 82)
    print("内插族的 C_pred 都应落在 0.69-1.04（已测 C 的范围）内；"
          "近外推族允许偏离但方向须继续下降。")
    print()
    print("预官（写进 sweep_law_test.py docstring 后不再改动）:")
    for fid, b2, cp, tag in out:
        print("  %-6s B2=%.3f  C_pred=%.3f  (%s)" % (fid, b2, cp, tag))


if __name__ == "__main__":
    main()
