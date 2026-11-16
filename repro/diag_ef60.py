"""诊断：tw_lev 定律为什么在 EF60/LG60 上失败（预官检验后，找失败机制）。

预官结果（analyze_twlaw.py）：
  族      tw_lev   C实测   C(tw)预官   残差dex
  TK30    0.209   0.821     0.806     +0.008  OK
  TK50    0.395   0.833     0.866     -0.017  OK
  EF60    0.686   1.067     0.922     +0.064  FAIL (16%, ~10 sigma)
  W3      0.959   0.996     0.954     +0.018  OK
  LG60    1.133   1.070     0.968     +0.043  marginal
  TS3     2.549   1.016     1.021     -0.002  OK

怀疑：tw_lev 的 levels 取径向剖面 min+{0.1,0.5,0.9}*range。当边宽到剖面在域内
走不完自己的 transition 时（erf/logistic w=0.6 的 sc 被推到 1.2，超出 r 的域 [0,1]），
a_ref 有一个高"底座"，range 被压缩，10-90% 带挤成薄片 => tw 被严重低估。

本脚本打印径向剖面本身 + 一批描述量，定位失败机制。不涉及新拟合的定律。

用法： venv2/bin/python diag_ef60.py
"""

import importlib.util
import json
import os
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
_spec = importlib.util.spec_from_file_location(
    "hd", os.path.join(PROJ, "repro", "hunt_descriptor.py"))
hd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hd)

NEW = ["TK30", "TK50", "EF60", "W3", "LG60", "TS3"]
CMEAS = {"TK30": 0.821, "TK50": 0.833, "EF60": 1.067,
         "W3": 0.996, "LG60": 1.070, "TS3": 1.016}
BASE_SHOW = ["W4", "W2", "W1", "EF35", "LG35", "RP35", "M2", "lowpass/0.25",
             "W8", "defocus/1"]


def main():
    tb = hd.build_tables()
    r, names, desc, crel = tb["r"], tb["names"], tb["desc"], tb["crel"]

    # 新族的剖面
    d = json.load(open(os.path.join(OUT, "sweep_twlaw.json")))
    meta = {}
    for rec in d["rows"]:
        meta[rec["meta"]["fid"]] = rec["meta"]
    anew = {}
    for fid in NEW:
        m = meta[fid]
        a = 1.0 - hd.make_profile(m["kind"], m["par"], m["scale"], r)
        anew[fid] = a * (0.60 / float(a.mean()))

    print("=" * 118)
    print(" 1. 径向均值剖面 a_bar(r)（缩到 knif=0.60）")
    print("=" * 118)
    rs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hdr = " ".join(f"{v:>6.2f}" for v in rs)
    print(f" {'剖面':<14} | {'sc':>7} | " + hdr)
    print(" " + "-" * (34 + 7 * len(rs)))
    for fid in NEW + BASE_SHOW:
        a = anew[fid] if fid in anew else tb["araw"][fid]
        rr, pr = hd.radial(a, r, nb=64)
        row = []
        for v in rs:
            j = int(np.argmin(np.abs(rr - v)))
            row.append(f"{pr[j]:>6.3f}")
        sc = meta[fid]["scale"] if fid in meta else float("nan")
        print(f" {fid:<14} | {sc:>7.3f} | " + " ".join(row))
    print(" " + "-" * (34 + 7 * len(rs)))
    print(" 看 a_bar 的**底座**（r=0 处的值）：EF60/LG60 的底座远高于 W4/W8 等。")

    print("\n" + "=" * 118)
    print(" 2. 描述量对照（新族 + 基线代表）")
    print("=" * 118)
    keys = ["tw_lev", "tw_mass", "tw_massw", "supp50", "supp80", "supp20",
            "PR", "B2", "rbar"]
    print(f" {'剖面':<14} | {'C':>6} | " + " | ".join(f"{k:>8}" for k in keys) +
          f" | {'amin':>6} | {'amax':>6} | {'dyn':>6}")
    print(" " + "-" * (28 + 11 * len(keys) + 26))
    for fid in NEW + BASE_SHOW:
        a = anew[fid] if fid in anew else tb["araw"][fid]
        dd = hd.descriptors(a, r) if fid in anew else desc[fid]
        c = CMEAS[fid] if fid in anew else crel(fid)
        m = float(a.mean())
        dyn = (float(a.max()) - float(a.min())) / m
        print(f" {fid:<14} | {c:>6.3f} | " +
              " | ".join(f"{dd[k]:>8.3f}" for k in keys) +
              f" | {float(a.min()):>6.3f} | {float(a.max()):>6.3f} | {dyn:>6.3f}")
    print(" " + "-" * (28 + 11 * len(keys) + 26))
    print(" dyn = (amax-amin)/<a>：相对动态范围。EF60/LG35 的 dyn 远小于 1，")
    print("      即剖面是「高底座 + 缓坡」而不是「从 0 走到满幅」。")

    print("\n" + "=" * 118)
    print(" 3. tw_lev 的 levels 到底取在哪里（EF60 vs W4）")
    print("=" * 118)
    for fid in ["EF60", "LG60", "W4", "TK50"]:
        a = anew[fid] if fid in anew else tb["araw"][fid]
        rr, pr = hd.radial(a, r, nb=200)
        lo, hi = float(pr.min()), float(pr.max())
        print(f"\n {fid}: a_bar 的 range = [{lo:.3f}, {hi:.3f}]，mean = {a.mean():.3f}")
        for frac in (0.1, 0.5, 0.9):
            lv = lo + frac * (hi - lo)
            cs = hd._cross(rr, pr, lv)
            print(f"   level {frac:.1f}*range = {lv:.3f} -> 穿越半径 "
                  + (", ".join(f"{c:.3f}" for c in cs) if cs else "无"))


if __name__ == "__main__":
    main()
