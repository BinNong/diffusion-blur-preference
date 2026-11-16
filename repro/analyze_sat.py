"""第十组判据：ρ(t) 的饱和区与 b(u) t 依赖性的上界（离线，零 GPU）。

预官来源：sweep_law_test.py docstring 第十组（2026-10-20，跑前写定；
          md5 ea96967cc7057636eb0a78b6f0f1b014）。判据在看到任何 bs*.json 之前写定。

与 analyze_bt.py 的唯一差别：**多了三个更高的 t**（0.08/0.12/0.16）。提取流程逐字对齐：
  * 多项式 logΛ~logu（DEG=5）在**全部共有 u** 上拟合，闭式 b=b_inf+A*u^q 在 [LO,HI]
    子集上拟合（第七/八组的规矩：多项式用全网格、闭式用子集）
  * 逐族拟合作出 b_inf，再跨族平均
  * 不夹非正 Λ；Λ 先对图取均值再取 log

用法： venv2/bin/python analyze_sat.py
"""

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
DENSE = ["W2", "W8", "TK50"]
T0 = 0.002
LO, HI = 0.13, 0.86
DEG = 5
SKIP = {"sweep_bdense.json", "sweep_blow.json", "sweep_bgap.json",
        "sweep_b2test.json", "sweep_butterworth.json"}


def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if "rows" not in d or "cap" not in d.get("cfg", {}):
        return None
    prof = defaultdict(dict)
    for rec in d["rows"]:
        prof[rec["meta"]["fid"]][round(1.0 - rec["kunif"], 4)] = rec
    return d, {k: dict(v) for k, v in prof.items()}


def grid(prof, fids, lo=LO, hi=HI):
    """返回 (ug_all, ug)：多项式用全部共有 u，闭式用 [lo,hi] 子集。"""
    sets = [set(prof[f]) for f in fids]
    ug_all = sorted(set.intersection(*sets), reverse=True)
    return ug_all, [u for u in ug_all if lo <= u <= hi]


def bfun_of(prof, fids, ug_all):
    """逐族返回 b(u)=d log(mean Λ)/d log u 的函数（多项式在 ug_all 上拟合）。"""
    out = {}
    for fid in fids:
        mu = np.array([prof[fid][round(u, 4)]["lam_mean"] for u in ug_all])
        if not (mu > 0).all():
            out[fid] = None
            continue
        c = np.polyfit(np.log(np.array(ug_all)), np.log(mu), DEG)
        d = np.polyder(c)
        out[fid] = (lambda dd: (lambda u: float(np.polyval(dd, math.log(u)))))(d)
    return out


def closed_form(ug, bf):
    """b = b_inf + A*u^q 在 ug 上拟合。"""
    if bf is None:
        return None
    ua = np.array(ug)
    bv = np.array([bf(u) for u in ug])
    f1 = lambda u, *p: p[0] + p[1] * u ** p[2]
    try:
        popt, _ = curve_fit(f1, ua, bv, p0=(-0.7, -6.0, 3.0),
                            bounds=([-1e3, -50, 0.1], [1e3, 50, 12]),
                            maxfev=100000)
    except Exception:
        return None
    rms = float(np.sqrt(np.mean((bv - f1(ua, *popt)) ** 2)))
    return dict(p=popt, rms=rms)


def cell(prof):
    """一个 t 的全部量：ug_all / ug / 逐族 bfun / 逐族闭式 / b_inf 跨族均值。"""
    ug_all, ug = grid(prof, DENSE)
    assert ug_all[:len(ug)] == ug, "ug 不是 ug_all 的前缀，索引会错位"
    bfs = bfun_of(prof, DENSE, ug_all)
    cfs = {f: closed_form(ug, bfs[f]) for f in DENSE}
    if any(v is None for v in cfs.values()):
        return None
    bin_ = float(np.mean([cfs[f]["p"][0] for f in DENSE]))
    return dict(ug_all=ug_all, ug=ug, bfs=bfs, cfs=cfs, bin_=bin_)


def main():
    ra = load("sweep_bdense.json")
    if ra is None:
        print("[错误] 缺 sweep_bdense.json")
        return
    base = dict(ra[1])
    rl = load("sweep_blow.json")
    if rl is not None:
        for fid, dd in rl[1].items():
            base.setdefault(fid, {}).update(dd)

    cells = [(T0, base)]
    new = []
    for fn in sorted(glob.glob(os.path.join(OUT, "sweep_bs*.json"))):
        r = load(os.path.basename(fn))
        if r is None:
            continue
        cells.append((float(r[0]["cfg"]["t"]), dict(r[1])))
        new.append(float(r[0]["cfg"]["t"]))
    # 第八组的四个 t 也一起载入，好让 S1/S2/S3 的曲线连续
    for fn in sorted(glob.glob(os.path.join(OUT, "sweep_bt*.json"))):
        r = load(os.path.basename(fn))
        if r is None:
            continue
        cells.append((float(r[0]["cfg"]["t"]), dict(r[1])))
    cells.sort(key=lambda c: c[0])
    ts = [t for t, _ in cells]
    C = dict(cells)

    print("=" * 100)
    print(f" 第十组判据：{len(ts)} 个 t（新增 {sorted(new) if new else '无'}）")
    print("=" * 100)

    E = {}
    for t in ts:
        e = cell(C[t])
        if e is None:
            print(f"[错误] t={t:g} 的闭式拟合失败（可能有非正 Λ）")
            continue
        E[t] = e
    if T0 not in E:
        print("[错误] 基线失败")
        return
    b0 = E[T0]["bin_"]
    print(f" 基线 b_∞(t={T0:g}) = {b0:.4f}；"
          f"逐族：" + " ".join(f"{f}={E[T0]['cfs'][f]['p'][0]:.3f}" for f in DENSE))

    # ---------------- S6 拟合机器守门 ----------------
    print("\n" + "=" * 100)
    print(" S6 拟合机器守门：闭式 rms（第八组四个新 t 上 ≤0.044）")
    print("=" * 100)
    print(f" {'t':>7} | {'b_∞':>8} | {'A':>9} | {'q':>7} | {'rms':>7} | 判定")
    print(" " + "-" * 62)
    ok6 = True
    for t in ts:
        if t not in E:
            continue
        p = E[t]["cfs"][DENSE[0]]["p"]
        rms = float(np.mean([E[t]["cfs"][f]["rms"] for f in DENSE]))
        good = rms <= 0.10
        ok6 &= good
        print(f" {t:>7g} | {E[t]['bin_']:>8.4f} | {p[1]:>9.3f} | {p[2]:>7.3f} | "
              f"{rms:>7.4f} | {'✅' if good else '❌'}")
    print(" " + "-" * 62)
    print(f" => S6 {'✅ 通过' if ok6 else '❌ 失败'}")

    # ---------------- S5 信噪比守门 ----------------
    print("\n" + "=" * 100)
    print(" S5 信噪比守门：逐格 SNR=Λ̄·√n/s；SNR<3 的格计数")
    print("=" * 100)
    print(f" {'t':>7} | {'SNR 最小':>9} | {'SNR 中位':>9} | {'SNR<3':>7} | "
          f"{'Λ 最小':>9} | {'s/√n 中位':>10}")
    print(" " + "-" * 70)
    snr = {}
    for t in ts:
        v = []
        for f in DENSE:
            for u in E[t]["ug_all"]:
                r = C[t][f][round(u, 4)]
                v.append(r["lam_mean"] * np.sqrt(len(r["lam_per_img"]))
                         / max(r["lam_std"], 1e-9))
        v = np.array(v)
        snr[t] = v
        lmin = min(C[t][f][round(u, 4)]["lam_mean"]
                   for f in DENSE for u in E[t]["ug_all"])
        smed = float(np.median([C[t][f][round(u, 4)]["lam_std"] / np.sqrt(20)
                                for f in DENSE for u in E[t]["ug_all"]]))
        print(f" {t:>7g} | {v.min():>9.2f} | {np.median(v):>9.2f} | "
              f"{int((v < 3).sum()):>7d} | {lmin:>9.0f} | {smed:>10.1f}")
    print(" " + "-" * 70)

    # ---------------- S1 ----------------
    print("\n" + "=" * 100)
    print(" S1：(1-ρ̄)/t 在 t≥0.02 严格下降；三个新点全部 <13.9")
    print("=" * 100)
    ugc = E[T0]["ug"]
    LAM = {f: {t: np.array([C[t][f][round(u, 4)]["lam_mean"] for u in ugc])
               for t in ts} for f in DENSE}
    RHO = {f: {t: LAM[f][t] / LAM[f][T0] for t in ts} for f in DENSE}
    om = {t: 1.0 - float(np.mean([np.mean(RHO[f][t]) for f in DENSE]))
          for t in ts}
    print(f" {'t':>7} | {'1-ρ̄':>8} | {'(1-ρ̄)/t':>10} | {'单调?':>6} | 判定")
    print(" " + "-" * 56)
    prev, ok1 = None, True
    for t in ts:
        if t == T0:
            continue
        mono = "" if prev is None else ("是" if om[t] > prev else "否")
        good = bool(om[t] / t < 13.9) if t >= 0.02 else True
        ok1 &= good
        print(f" {t:>7g} | {om[t]:>8.4f} | {om[t]/t:>10.2f} | {mono:>6} | "
              f"{'✅' if good else '❌'}")
        prev = om[t]
    print(" " + "-" * 56)
    sat = None
    ta = np.array([t for t in ts if t != T0])
    oa = np.array([om[t] for t in ta])
    try:
        p, _ = curve_fit(lambda t, A, B: A * t / (B + t), ta, oa, p0=(1.0, 0.02),
                         bounds=([0.01, 1e-4], [5, 1]), maxfev=400000)
        pred = p[0] * ta / (p[1] + ta)
        rms = float(np.sqrt(np.mean((oa - pred) ** 2)))
        sat = dict(A=float(p[0]), B=float(p[1]), rms=rms)
        print(f"\n {len(ta)} 点饱和型 1-ρ̄=A·t/(B+t)：A={p[0]:.4f}  B={p[1]:.5f}  "
              f"rms={rms:.4f}（1-ρ̄ 量级 {oa.mean():.3f}）")
        print(f"   => A {'≤1，物理' if p[0] <= 1 else '>1，无物理意义'}"
              f"{'；渐近 1-ρ̄→%.3f' % p[0] if p[0] <= 1 else ''}")
    except Exception as e:
        print(f"\n [饱和型拟合失败：{e}]")
    print(f" => S1 {'✅ 通过' if ok1 else '❌ 失败'}")

    # ---------------- S2 ----------------
    print("\n" + "=" * 100)
    print(" S2：ρ(hi)/ρ(lo) 单调增，t=0.16 处落入 [1.895, 高斯极限上界]")
    print("=" * 100)
    L0 = {u: float(np.mean([np.log(LAM[f][T0][i]) for f in DENSE]))
          for i, u in enumerate(ugc)}
    uhi, ulo = max(ugc), min(ugc)
    g_lim = ((1 - uhi) / math.exp(L0[uhi])) / ((1 - ulo) / math.exp(L0[ulo]))
    print(f" 高斯极限上界 = [(1-{uhi})/Λ_0({uhi})]/[(1-{ulo})/Λ_0({ulo})] = "
          f"{(1-uhi)/math.exp(L0[uhi]):.3e}/{((1-ulo)/math.exp(L0[ulo])):.3e} = "
          f"{g_lim:.3f}")
    print(f" {'t':>7} | {'ρ(hi)':>8} | {'ρ(lo)':>8} | {'高/低比':>8} | {'单调?':>6} | 判定")
    print(" " + "-" * 66)
    prev, ok2, s2 = None, True, []
    for t in ts:
        if t == T0:
            continue
        hi_ = float(np.mean([RHO[f][t][0] for f in DENSE]))
        lo_ = float(np.mean([RHO[f][t][-1] for f in DENSE]))
        r = hi_ / lo_
        s2.append((t, r))
        mono = "" if prev is None else ("是" if r > prev else "否")
        good = bool(1.895 <= r <= g_lim) if t in new else True
        ok2 &= good
        print(f" {t:>7g} | {hi_:>8.4f} | {lo_:>8.4f} | {r:>8.3f} | {mono:>6} | "
              f"{'✅' if good else '❌'}")
        prev = r
    print(" " + "-" * 66)
    print(f" => S2 {'✅ 通过' if ok2 else '❌ 失败'}")

    # ---------------- S3 ----------------
    print("\n" + "=" * 100)
    print(" S3：Δb_∞ 单调增、新 t 处 ∈[0.316, -b_∞(t0)]（高斯极限上界），且 Δb_∞/t<6.32")
    print("=" * 100)
    print(f" {'t':>7} | {'b_∞':>8} | {'Δb_∞':>8} | {'Δb_∞/t':>8} | {'单调?':>6} | 判定")
    print(" " + "-" * 66)
    prev, ok3, dbinf = None, True, {}
    for t in ts:
        if t not in E:
            continue
        d = E[t]["bin_"] - b0
        dbinf[t] = d
        mono = "" if prev is None else ("是" if d > prev else "否")
        good = bool(0.316 <= d <= -b0 and d / t < 6.32) if t in new else True
        ok3 &= good
        print(f" {t:>7g} | {E[t]['bin_']:>8.4f} | {d:>+8.4f} | {d/t:>8.2f} | "
              f"{mono:>6} | {'✅' if good else '❌'}")
        prev = d
    print(" " + "-" * 66)
    print(f" => S3 {'✅ 通过' if ok3 else '❌ 失败'}")

    # ---------------- S4 ----------------
    print("\n" + "=" * 100)
    print(" S4：放大倍数 A=Λ/Λ_Gauss 继续下降，t=0.16 处 <5")
    print("=" * 100)
    print(f" {'t':>7} | {'A 最小':>8} | {'A 中位':>8} | {'A 最大':>8} | 判定")
    print(" " + "-" * 56)
    ok4, amps = True, {}
    for t in ts:
        v = np.array([C[t][f][round(u, 4)]["lam_mean"]
                      / C[t][f][round(u, 4)]["lam_gauss"]
                      for f in DENSE for u in E[t]["ug_all"]
                      if C[t][f][round(u, 4)]["lam_gauss"] > 0])
        amps[t] = v
        good = bool(v.max() < 5) if t in new else True
        ok4 &= good
        print(f" {t:>7g} | {v.min():>8.2f} | {np.median(v):>8.2f} | {v.max():>8.2f} | "
              f"{'✅' if good else '❌'}")
    print(" " + "-" * 56)
    print(f" => S4 {'✅ 通过' if ok4 else '❌ 失败'}")

    with open(os.path.join(OUT, "fig_sat.json"), "w") as fh:
        json.dump(dict(
            ts=ts, ug=ugc, new=new, b0=b0,
            bin_={str(t): E[t]["bin_"] for t in E},
            dbinf={str(t): float(dbinf[t]) for t in dbinf},
            rms={str(t): float(np.mean([E[t]["cfs"][f]["rms"] for f in DENSE]))
                 for t in E},
            one_minus={str(t): om[t] for t in ts},
            slope={str(t): om[t] / t for t in ts if t != T0},
            sat_fit=sat, s2=[[t, r] for t, r in s2], gauss_limit=float(g_lim),
            amp={str(t): dict(lo=float(amps[t].min()), med=float(np.median(amps[t])),
                              hi=float(amps[t].max())) for t in ts},
            snr={str(t): dict(lo=float(snr[t].min()), med=float(np.median(snr[t])),
                              n_bad=int((snr[t] < 3).sum())) for t in ts},
            pass_=dict(S1=bool(ok1), S2=bool(ok2), S3=bool(ok3), S4=bool(ok4),
                       S6=bool(ok6))), fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'fig_sat.json')}")


if __name__ == "__main__":
    main()
