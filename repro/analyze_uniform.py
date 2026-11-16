"""均匀传输算子的裁判：V1-V3，判据跑前写定在 sweep_law_test.py docstring 第六组。

预官来源：sweep_law_test.py docstring 第六组（2026-10-18 17:24，跑前写定；
          md5 = 90e8901207666049bb71eb25a15f8230）。

背景：hunt3.py / diag_offset.py 发现 kref=0.92 的 6 个底座剖面是一个**平台**——
twod 跨 0.894->26.603（30x）而 C 只在 1.049->1.102（1.05x）。它们的传输 g2 全被限制在
[0.35, 0.68] 窄带内（beta 压缩给的地板）。物理读数：传输被限制在窄带内的算子，其模糊
偏好退化为**均匀振幅缩放**的偏好。本实验直接测均匀算子（g2 = c 常数）来判决。

预官：
  V1 平台读数：均匀算子的 C 落在 1.070 的 +/-0.038 dex 内（容差 = twod 定律在 28 个
     训练剖面上的 LOO 最大残差 0.0383 dex，与第五组同源）。
  V2 指数普适：均匀算子在 u in [0.65,0.40] 的 b(u) 落在 [-1.68,-1.31]。
     （实现注意：均匀算子每个 u 只有一个点，必须精确命中取值，不能靠插值。）
  V3 判别：均匀算子的 C 必须 >= 1.05（平台 6 个值的范围是 [1.049,1.102]；
     twod 定律在已测最大 twod=26.6 处给 1.063、在 twod=0.894 处只给 0.963）。

用法： venv2/bin/python analyze_uniform.py [--data sweep_uniform.json]
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
    "ptod", os.path.join(PROJ, "repro", "predict_twod.py"))
ptod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ptod)
hd = ptod.hd

TOL = 0.038
PLATEAU = 1.070
V3_MIN = 1.05
B_RANGE = (-1.68, -1.31)
UNI = ["UNI0.4", "UNI0.6", "UNI0.65"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sweep_uniform.json")
    args = ap.parse_args()

    # ---------- 基准：b(u) 与 mid(u) ----------
    prof, meta = ptod.load_all()
    seg_u, seg_b = [], []
    for a, b in sorted(hd.P1_RANGES, reverse=True):
        bs = []
        for k in hd.SMOOTH_OLD:
            nm = f"{k[0]}/{k[1]:g}"
            va, vb = hd.interp(prof[nm], a), hd.interp(prof[nm], b)
            if va and vb and va > 0 and vb > 0:
                bs.append(math.log(vb / va) / math.log(b / a))
        if len(bs) >= 5:
            seg_u.append(math.sqrt(a * b)); seg_b.append(float(np.mean(bs)))
    seg_u = np.array(seg_u); seg_b = np.array(seg_b)

    def b_of(u):
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

    Uall = sorted({round(c[0], 3) for nm in prof for c in prof[nm]}, reverse=True)
    C, mid = {}, {}
    for nm in prof:
        C[nm] = {}
        for u in Uall:
            v = hd.interp(prof[nm], u)
            if v and v > 0:
                C[nm][u] = v * u ** (-b_of(u))
    for u in Uall:
        vs = [C[f"{k[0]}/{k[1]:g}"][u] for k in hd.SMOOTH_OLD
              if u in C.get(f"{k[0]}/{k[1]:g}", {})]
        if len(vs) >= 5:
            mid[u] = float(np.exp(np.mean([math.log(v) for v in vs])))
    Ucom = sorted({round(u, 4) for u in mid}, reverse=True)
    print("=" * 100)
    print(f" 基准：{len(prof)} 个已测剖面；共同 u 网格 " +
          " ".join(f"{u:.3g}" for u in Ucom))
    print("=" * 100)

    # ---------- 均匀算子 ----------
    d = json.load(open(os.path.join(OUT, args.data)))
    up = defaultdict(list)
    for rec in d["rows"]:
        m = rec["meta"]
        up[m["fid"]].append((1.0 - rec["kunif"], rec["lam_mean"]))
    for k in up:
        up[k].sort(key=lambda c: -c[0])
    print(f"\n 载入 {args.data}: nimg={d['cfg']['nimg']} t={d['cfg']['t']:g}；"
          f"{len(up)} 个均匀算子 " + "  ".join(sorted(up)))

    print("\n" + "=" * 100)
    print(" 1. 均匀算子的 Λ(u) 与 C")
    print("=" * 100)
    print(f" {'算子':<8} | {'g2':>5} | {'kunif':>6} | {'u':>5} | {'Λ':>9} | "
          f"{'Λ_G':>8} | {'C=Λu^-b/mid':>12}")
    print(" " + "-" * 74)
    cs_ = {}
    for fid in UNI:
        if fid not in up:
            continue
        g2 = float(fid.replace("UNI", ""))
        for u, lam in up[fid]:
            uu = round(u, 4)
            if uu not in mid:
                print(f" {fid:<8} | {g2:>5.2f} | {1-u:>6.3f} | {u:>5.3f} | "
                      f"{lam:>9.0f} | {'--':>8} | {'(非共同网格点)':>12}")
                continue
            cv = lam * uu ** (-b_of(uu)) / mid[uu]
            cs_[uu] = cv
            print(f" {fid:<8} | {g2:>5.2f} | {1-u:>6.3f} | {u:>5.3f} | "
                  f"{lam:>9.0f} | {'--':>8} | {cv:>12.3f}")
    if not cs_:
        print("\n[没有共同网格上的点]")
        return
    cuni = float(np.median(list(cs_.values())))
    drift = float(max(cs_.values()) / min(cs_.values())) if len(cs_) > 1 \
        else float("nan")
    print(" " + "-" * 74)
    print(f" C（共同网格中位）= {cuni:.3f}；u 漂移 = {drift:.3f}"
          f"（越接近 1 说明 C 确实是形状-only）")

    # ---------- 判据 ----------
    print("\n" + "=" * 100)
    print(" 2. 判据")
    print("=" * 100)

    # V1
    d1 = math.log10(cuni / PLATEAU)
    p1 = abs(d1) <= TOL
    print(f"\n V1 平台读数：C = {cuni:.3f} vs 平台均值 {PLATEAU:.3f}，"
          f"残差 {d1:>+7.3f} dex（容差 {TOL}）  {'✅' if p1 else '❌'}")

    # V2
    print(f"\n V2 指数普适（u in [0.65, 0.40]，旧范围 [{B_RANGE[0]}, {B_RANGE[1]}]）:")
    a, b = 0.65, 0.40
    if a in cs_ and b in cs_:
        pass
    # 均匀算子每个 u 只有一个点，hd.interp 需要两个点才能括住，会返回 None。
    # 这里改成"精确命中就直接取该点的值"——只修实现，不改判据。
    def exact(curve, uu):
        for k, v in curve:
            if abs(k - uu) < 1e-9:
                return v
        return hd.interp(curve, uu)

    va = exact(up["UNI0.65"], a) if "UNI0.65" in up else None
    vb = exact(up["UNI0.4"], b) if "UNI0.4" in up else None
    if va and vb and va > 0 and vb > 0:
        bb = math.log(vb / va) / math.log(b / a)
        p2 = B_RANGE[0] <= bb <= B_RANGE[1]
        print(f"   Λ(0.65)={va:.0f}，Λ(0.40)={vb:.0f} => b = {bb:+.3f}  "
              f"{'✅' if p2 else '❌'}")
    else:
        p2 = False
        print("   [插值失败]")

    # V3
    p3 = cuni >= V3_MIN
    print(f"\n V3 判别（平台 vs twod 曲线外推）：C = {cuni:.3f} >= {V3_MIN} ? "
          f"{'是' if p3 else '否'}  {'✅' if p3 else '❌'}")
    print(f"   参照：twod 定律在已测最大 twod=26.6 处给 1.063，在 twod=0.894 处给 0.963；")
    print(f"         平台 6 个已测值的范围是 [1.049, 1.102]。")

    # ---------- 汇总 ----------
    print("\n" + "=" * 100)
    print(" 3. 汇总")
    print("=" * 100)
    print(f" V1 平台读数  : {'✅' if p1 else '❌'}   （C={cuni:.3f}，残差 {d1:+.3f} dex）")
    print(f" V2 指数普适  : {'✅' if p2 else '❌'}")
    print(f" V3 判别      : {'✅' if p3 else '❌'}")
    allp = all([p1, p2, p3])
    print("\n " + ("全部通过 => 「传输被限制在窄带内的算子，其模糊偏好退化为均匀振幅缩放的"
                   "偏好，C≈1.070」成立，这也解释了 twod 曲线为什么在底座剖面失效。"
                   if allp else
                   "有未通过项 => 记录失败。按项目规矩，不为失败的预言改阈值。"))


if __name__ == "__main__":
    main()
