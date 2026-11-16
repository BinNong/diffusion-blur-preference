"""b(u) 密网格的裁判：B1-B5 + 闭式候选，方法与判据跑前写定在 sweep_law_test.py 第七组。

预官来源：sweep_law_test.py docstring 第七组（2026-10-18 18:11，跑前写定；
          md5 = 18b734da587a79541ecca1346ee8972a）。

b(u) 提取方式（跑前定死）：
    对每个族，log Λ 作为 log u 的**5 次多项式**（13 点、6 参数），
    b(u) = d logΛ / d log u 取其解析导数；SE 由对 20 张图的有放回重采样（2000 次）给出。

判据：
  B1 结构：b(u) 对 u 单调不增，且 b(u) > -2 处处成立。
  B2 低 u 极限：u < 0.26 时 b(u) 落在 [-0.95, -0.65]。
  B3 普适性：3 个族在每一个密网格 u 上，b 的极差 <= 0.50。
  B4 插值噪声假说：RQ400/EF100 在 [0.40,0.30] 档用直接测得的 L(0.40)/L(0.30) 算 b，
     落在旧范围 [-0.91,-0.82] 内。
  B5 无新结构：log Λ vs log u 光滑单调，无新拐点/振荡。
  闭式候选（按预官顺序）：(i) b = b_inf + A*u^q；(ii) b = b_inf + A/(1+u/u0)^p；
     (iii) b = -2 + A/u^q。

用法： venv2/bin/python analyze_bdense.py
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
    "hd", os.path.join(PROJ, "repro", "hunt_descriptor.py"))
hd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hd)

DENSE = ["W2", "W8", "TK50"]
GAP = ["RQ400", "EF100"]
# 注意：sweep 的 --targets 是 **kunif**，不是 u；u = 1 - kunif。
# 第七组我把两者搞混了，所以密网格实际覆盖的是 u in [0.30, 0.85]（kunif 0.15..0.70）。
# 这里改成从数据自动探测网格，避免再犯。
UDENSE = None
DEG = 5
NBOOT = 2000


def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    prof = defaultdict(list)
    for rec in d["rows"]:
        prof[rec["meta"]["fid"]].append((1.0 - rec["kunif"], rec["lam_mean"],
                                         rec.get("lam_per_img")))
    for k in prof:
        prof[k].sort(key=lambda c: -c[0])
    return d, dict(prof)


def poly_b(lus, llam, deg=DEG):
    """logΛ 对 log u 的 deg 次多项式；返回 (coef, 导数函数)。"""
    c = np.polyfit(lus, llam, deg)

    def b(u):
        x = math.log(u)
        d = np.polyder(c)
        return float(np.polyval(d, x))
    return c, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dense", default="sweep_bdense.json")
    ap.add_argument("--gap", default="sweep_bgap.json")
    args = ap.parse_args()

    rd = load(args.dense)
    if rd is None:
        print("[缺少]", args.dense)
        return
    dd, pd = rd
    # 合并 sweep_blow.json（低 u 段，kunif 0.74..0.86 => u 0.26..0.14）
    blow = load("sweep_blow.json")
    if blow is not None:
        for fid, rows in blow[1].items():
            pd.setdefault(fid, []).extend(rows)
        for k in pd:
            pd[k].sort(key=lambda c: -c[0])
        print(" [已合并 sweep_blow.json：补上 u in [0.14, 0.26]]")
    # 自动探测：三个族都有数据的 u（只取 A 组算子，即 K=target 的那些）
    sets = []
    for fid in DENSE:
        sets.append({round(u, 4) for u, _, _ in pd[fid]})
    ug_all = sorted(set.intersection(*sets), reverse=True)
    print(" 实际 u 覆盖（三族共有）: " + " ".join(f"{u:.3f}" for u in ug_all))
    # 用全部共同点（sweep_bdense 覆盖 u in [0.30,0.85]，sweep_blow 补 u in [0.14,0.26]，
    # 再加上 beta 组的 0.724/0.356/0.08）；只在端点各留一点余量避免多项式外冲。
    ug = [u for u in ug_all if 0.10 <= u <= 0.86]
    globals()["UDENSE"] = ug
    print(f" 密网格主体（{len(ug)} 点）: " + " ".join(f"{u:.3f}" for u in ug))
    print("=" * 104)
    print(f" 密网格：{args.dense}；nimg={dd['cfg']['nimg']} t={dd['cfg']['t']:g}")
    print("=" * 104)

    # ---- 表 1：Λ(u) 与 log-log 局部斜率 ----
    print("\n" + "=" * 104)
    print(" 1. 三个族的 Λ(u) 与 log-log 局部斜率（相邻两点）")
    print("=" * 104)
    print(f" {'u':>6} | " + " | ".join(f"{f:>16}" for f in DENSE))
    print(" " + "-" * (10 + 19 * len(DENSE)))
    lam = {}
    for fid in DENSE:
        lam[fid] = {round(u, 4): v for u, v, _ in pd[fid]}
    for u in UDENSE:
        row = []
        for fid in DENSE:
            v = lam[fid].get(round(u, 4))
            row.append(f"{v:>16.0f}" if v else f"{'--':>16}")
        print(f" {u:>6.3f} | " + " | ".join(row))
    print(" " + "-" * (10 + 19 * len(DENSE)))
    print(" 相邻两点的局部斜率 b = log(Λ2/Λ1)/log(u2/u1)：")
    print(f" {'u区间':<14} | " + " | ".join(f"{f:>10}" for f in DENSE))
    print(" " + "-" * (16 + 13 * len(DENSE)))
    for i in range(len(UDENSE) - 1):
        u1, u2 = UDENSE[i], UDENSE[i + 1]
        row = []
        for fid in DENSE:
            v1, v2 = lam[fid].get(round(u1, 4)), lam[fid].get(round(u2, 4))
            if v1 and v2 and v1 > 0 and v2 > 0:
                row.append(f"{math.log(v2/v1)/math.log(u2/u1):>10.3f}")
            else:
                row.append(f"{'--':>10}")
        print(f" [{u1:.3f},{u2:.3f}] | " + " | ".join(row))

    # ---- 多项式拟合 + bootstrap ----
    print("\n" + "=" * 104)
    print(f" 2. b(u)：logΛ 对 log u 的 {DEG} 次多项式导数（bootstrap {NBOOT} 次）")
    print("=" * 104)
    fits = {}
    for fid in DENSE:
        us = sorted(lam[fid])
        lus = np.array([math.log(u) for u in us])
        llam = np.array([math.log(lam[fid][u]) for u in us])
        c, b = poly_b(lus, llam, DEG)
        # bootstrap：对 20 张图有放回重采样
        per = {}
        for u, v, pi in pd[fid]:
            if pi:
                per[round(u, 4)] = np.array(pi)
        rng = np.random.default_rng(0)
        boots = []
        common = [u for u in us if round(u, 4) in per]
        if len(common) == len(us):
            P = np.array([per[round(u, 4)] for u in us])     # (nu, nimg)
            nimg = P.shape[1]
            for _ in range(NBOOT):
                idx = rng.integers(0, nimg, nimg)
                lm = np.log(P[:, idx].mean(1))
                try:
                    cb = np.polyfit(lus, lm, DEG)
                    boots.append(np.polyval(np.polyder(cb), lus))
                except Exception:
                    pass
        fits[fid] = dict(c=c, b=b, us=us,
                         se=np.array(boots).std(0) if boots else None)
    print(f" {'u':>6} | " + " | ".join(f"{f:>14}" for f in DENSE) +
          f" | {'极差':>7}")
    print(" " + "-" * (10 + 17 * len(DENSE) + 10))
    spread = []
    for u in UDENSE:
        bs, ses = [], []
        for fid in DENSE:
            v = fits[fid]["b"](u)
            bs.append(v)
            if fits[fid]["se"] is not None:
                j = int(np.argmin(np.abs(np.array(fits[fid]["us"]) - u)))
                ses.append(fits[fid]["se"][j])
        rng_ = max(bs) - min(bs)
        spread.append(rng_)
        cells = []
        for k, fid in enumerate(DENSE):
            s = f"{bs[k]:>7.3f}"
            if ses:
                s += f"±{ses[k]:.3f}"
            cells.append(f"{s:>14}")
        print(f" {u:>6.3f} | " + " | ".join(cells) + f" | {rng_:>7.3f}")
    print(" " + "-" * (10 + 17 * len(DENSE) + 10))
    print(f" 极差最大 {max(spread):.3f}（判据 B3 要求 <= 0.50）")

    # ---- 判据 ----
    print("\n" + "=" * 104)
    print(" 3. 判据")
    print("=" * 104)
    bcurves = {fid: np.array([fits[fid]["b"](u) for u in UDENSE])
               for fid in DENSE}
    # B1
    # UDENSE 按 u 降序；b 随 u 减小而增大 => 相邻差应 >= 0
    mono = all((np.diff(bcurves[fid]) >= -1e-6).mean() >= 0.9 for fid in DENSE)
    gt2 = all((bcurves[fid] > -2.0).all() for fid in DENSE)
    print(f"\n B1 结构：b 对 u 单调不增（>=80% 的相邻差 <=0）且 b > -2")
    for fid in DENSE:
        d = np.diff(bcurves[fid])
        print(f"   {fid:<5} b 从 {bcurves[fid][0]:>6.3f}（u={UDENSE[0]:.2f}）到 "
              f"{bcurves[fid][-1]:>6.3f}（u={UDENSE[-1]:.2f}）；相邻差 >=0 占 "
              f"{(d >= -1e-6).mean():.0%}；最小 b = {bcurves[fid].min():.3f}")
    print(f"   => 单调 {'✅' if mono else '❌'}；b > -2 {'✅' if gt2 else '❌'}")
    # B2
    lo = [u for u in UDENSE if u < 0.26]
    b2ok = True
    print(f"\n B2 低 u 极限（u<0.26 时 b 落在 [-0.95,-0.65]）:")
    for fid in DENSE:
        vs = [fits[fid]["b"](u) for u in lo]
        ok = all(-0.95 <= v <= -0.65 for v in vs)
        b2ok = b2ok and ok
        print(f"   {fid:<5} " + "  ".join(f"{u:.2f}→{v:+.3f}" for u, v in zip(lo, vs))
              + f"   {'✅' if ok else '❌'}")
    # B3
    print(f"\n B3 普适性（每个 u 上 3 族 b 极差 <= 0.50）: "
          f"{'✅' if max(spread) <= 0.50 else '❌'}（最大 {max(spread):.3f}）")
    # B5
    print(f"\n B5 无新结构（logΛ vs log u 光滑单调）:")
    b5 = True
    for fid in DENSE:
        us = sorted(lam[fid])
        lv = [math.log(lam[fid][u]) for u in us]
        d2 = np.diff(lv, 2)
        mono_uv = all(lv[i] > lv[i + 1] for i in range(len(lv) - 1))
        print(f"   {fid:<5} Λ 随 u 单调下降: {'是' if mono_uv else '否'}；"
              f"二阶差分最大 {np.abs(d2).max():.4f}"
              f"（相对一阶差分量级 {np.abs(np.diff(lv)).mean():.4f}）")
        b5 = b5 and mono_uv
    print(f"   => B5 {'✅' if b5 else '❌'}")

    # ---- B4：插值噪声假说 ----
    print("\n" + "=" * 104)
    print(" 4. B4 插值噪声假说（RQ400/EF100 的 [0.40,0.30] 档不再依赖插值）")
    print("=" * 104)
    rg = load(args.gap)
    print(f" {'族':<6} | {'L(0.40)':>9} | {'L(0.35)':>9} | {'L(0.30)':>9} | "
          f"{'b[.4,.3]':>9} | {'旧范围':>12} | {'判定':>5}")
    print(" " + "-" * 78)
    b4 = True
    if rg is None:
        print(" [缺少]", args.gap)
        b4 = False
    else:
        dg, pg = rg
        # L(0.40) 在 sweep_twod.json（默认 targets 0.35/0.60/0.85 => u=0.65/0.40/0.15）
        ro = load("sweep_twod.json")
        if ro is not None:
            for fid, rows in ro[1].items():
                pg.setdefault(fid, []).extend(rows)
            for k in pg:
                pg[k].sort(key=lambda c: -c[0])
        for fid in GAP:
            if fid not in pg:
                continue
            L = {round(u, 4): v for u, v, _ in pg[fid]}
            v40, v35, v30 = (L.get(0.4), L.get(0.35), L.get(0.3))
            if v40 and v30 and v40 > 0 and v30 > 0:
                b = math.log(v30 / v40) / math.log(0.3 / 0.4)
                ok = -0.91 <= b <= -0.82
                b4 = b4 and ok
                print(f" {fid:<6} | {v40:>9.0f} | "
                      f"{v35 if v35 else float('nan'):>9.0f} | {v30:>9.0f} | "
                      f"{b:>9.3f} | {'[-0.91,-0.82]':>12} | "
                      f"{'✅' if ok else '❌':>5}")
            else:
                print(f" {fid:<6} | 缺 L(0.40) 或 L(0.30)")
                b4 = False
            # 旧值（插值）对照
            if ro is not None and fid in ro[1]:
                Lo = {round(u, 4): v for u, v, _ in ro[1][fid]}
                a1, a2 = Lo.get(0.356), Lo.get(0.15)
                if a1 and a2 and a1 > 0 and a2 > 0:
                    bold = math.log(a2 / a1) / math.log(0.15 / 0.356)
                    print(f"       （旧插值法给 b = {bold:+.3f}，"
                          f"靠 0.356 与 0.15 跨 0.2）")
        print(" " + "-" * 78)
        print(f" => B4 {'通过：贴边违规确实是插值噪声，b(u) 普适性恢复' if b4 else '未过'}")

    # ---- 闭式候选 ----
    print("\n" + "=" * 104)
    print(" 5. b(u) 的闭式候选（按预官顺序）")
    print("=" * 104)
    print(f" {'形式':<26} | " + " | ".join(f"{f:>16}" for f in DENSE))
    print(" " + "-" * (28 + 19 * len(DENSE)))
    # 注意 curve_fit 把参数按位置传，必须用 *p
    forms = [
        ("(i) b = b_inf + A*u^q",
         lambda u, *p: p[0] + p[1] * u ** p[2], [(-0.6, -4.0, 2.0)],
         ([-1e3, -50, 0.1], [1e3, 50, 12])),
        ("(ii) b = b_inf + A/(1+u/u0)^p",
         lambda u, *p: p[0] + p[1] / (1.0 + u / p[2]) ** p[3],
         [(-0.6, -4.0, 0.3, 2.0)], ([-1e3, -50, 1e-3, 0.1],
                                    [1e3, 50, 5, 12])),
        ("(iii) b = -2 + A/u^q",
         lambda u, *p: -2.0 + p[0] / u ** p[1], [(3.0, 2.0)],
         ([-50, 0.1], [50, 12])),
        ("(iv) b = -b0 - c/(u0-u)^q  (u<u0)",
         lambda u, *p: -p[0] - p[1] / (p[2] - u) ** p[3],
         [(0.6, 1.0, 1.2, 1.0)], ([1e-3, 1e-3, 0.9], [50, 50, 3])),
        ("(v) b = -A*exp((u/u0)^q)",
         lambda u, *p: -p[0] * np.exp((u / p[1]) ** p[2]),
         [(1.0, 0.6, 2.0)], ([1e-3, 1e-3, 0.1], [50, 5, 12])),
        ("(vi) b = -A*(u/u0)^q / (1+(u/u0)^q)",
         lambda u, *p: -p[0] * (u / p[1]) ** p[2] /
         (1.0 + (u / p[1]) ** p[2]),
         [(4.0, 0.6, 2.0)], ([1e-3, 1e-3, 0.1], [50, 5, 12])),
    ]
    from scipy.optimize import curve_fit
    for label, fn, p0s, bnds in forms:
        cells = []
        for fid in DENSE:
            us = np.array([u for u in sorted(lam[fid]) if 0.13 <= u <= 0.86])
            bv = np.array([fits[fid]["b"](u) for u in us])
            best = None
            for p0 in p0s:
                try:
                    popt, _ = curve_fit(fn, us, bv, p0=p0, bounds=bnds,
                                        maxfev=100000)
                    r = float(np.sqrt(np.mean((bv - fn(us, *popt)) ** 2)))
                    if best is None or r < best[0]:
                        best = (r, popt)
                except Exception:
                    pass
            if best:
                cells.append(f"rms={best[0]:.3f} p={np.round(best[1],3)}")
            else:
                cells.append(f"{'拟合失败':>16}")
        print(f" {label:<26} | " + " | ".join(f"{c:>16}" for c in cells))
    print(" " + "-" * (28 + 19 * len(DENSE)))
    print(" rms 是 b(u) 的拟合残差（单位同 b）。三个族上同时 rms 最小且接近 bootstrap SE 的，")
    print(" 才是候选闭式。")

    # ---- 汇总 ----
    print("\n" + "=" * 104)
    print(" 6. 汇总")
    print("=" * 104)
    print(f" B1 结构      : {'✅' if (mono and gt2) else '❌'}")
    print(f" B2 低 u 极限 : {'✅' if b2ok else '❌'}")
    print(f" B3 普适性    : {'✅' if max(spread) <= 0.50 else '❌'}"
          f"（极差最大 {max(spread):.3f}）")
    print(f" B4 插值噪声  : {'✅' if b4 else '❌'}")
    print(f" B5 无新结构  : {'✅' if b5 else '❌'}")


if __name__ == "__main__":
    main()
