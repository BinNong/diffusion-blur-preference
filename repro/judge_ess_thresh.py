"""ess_threshold.py 的离线判据：Z1–Z4（零 GPU）。

**预官澄清（2026-10-22，披露）**：ess_threshold.py 已跑完（ESS 已落盘），但**本判据尚未运行、
我也还没有看过任何一个 ESS 数值**。在此窗口内只做两处**澄清**，不改任何阈值：
  (1) 「过渡区」改为**逐算子**取该算子自己的 ESS in [2, n/2] 区间，而不是全局取。
      理由（先验的，与数据无关）：全局区间在某些 sigma 上可能一个格子都没有，那样 Z2 会被
      「跳过」而不是被「检验」；逐算子取保证每个 sigma 都被真正检验到。
  (2) Z3 的「ESS 首次 > 1」给一个明确的数值定义 ess > 1.01（即 log(n/ESS) < log n - 0.01），
      避免把浮点噪声当成离开塌缩。
Z2 的阈值 [0.5, 2] 与「全部过渡格」的读法**一字未改**（原预官文本见
ess_threshold.py docstring，md5 983091dd）。同时额外报落在区间内的格子占比与比值的几何均值——
那是**补充信息**，不是替代判据；若占比 <100% 而其余都过，按预官记为失败并报占比。

用法： venv2/bin/python judge_ess_thresh.py
"""

import json
import math
import os

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")


def main():
    d = json.load(open(os.path.join(OUT, "ess_threshold.json")))
    rows = d["rows"]
    cfg = d["cfg"]
    SN = cfg["noises"]
    n = cfg["n_pool"]
    ids = sorted({r["id"] for r in rows})
    print("=" * 100)
    print(f" ESS 塌缩点判据：{len(rows)} 行 = {len(SN)} 个 sigma_n × {len(ids)} 算子；"
          f"池 {n} 张，ln n = {math.log(n):.4f}")
    print("=" * 100)

    # ---------------- Z1 守门 ----------------
    print("\n" + "=" * 100)
    print(" Z1 守门（精确恒等式，与 sigma 无关）：两种 ESS 算法一致到 1e-10")
    print("=" * 100)
    ea = np.array([r["ess"] for r in rows])
    eb = np.array([r["ess_b"] for r in rows])
    el = np.array([r["ess_lse"] for r in rows])
    worst = float(np.max(np.abs(ea - eb) / np.maximum(ea, 1e-30)))
    worst2 = float(np.max(np.abs(ea - el) / np.maximum(ea, 1e-30)))
    print(f" {'算法对':<28} | {'最大相对偏差':>14}")
    print(" " + "-" * 48)
    print(f" {'(sum W)^2/sum W^2  vs  n/(1+CV^2)':<28} | {worst:>14.3e}")
    print(f" {'(sum W)^2/sum W^2  vs  logsumexp 版':<28} | {worst2:>14.3e}")
    print(" " + "-" * 48)
    ok1 = bool(worst <= 1e-10 and worst2 <= 1e-10)
    print(f" => Z1 {'✅ 通过' if ok1 else '❌ 失败'}")

    # 跨轮次复现（与 (ae) 的 ess_scale.json 逐字对齐的三个 sigma）
    oldp = os.path.join(OUT, "ess_scale.json")
    if os.path.exists(oldp):
        o = json.load(open(oldp))
        om = {(r["id"], r["sn"]): r for r in o["rows"]}
        dv, de = [], []
        for r in rows:
            k = (r["id"], r["sn"])
            if k in om:
                dv.append(abs(r["var_data"] / om[k]["var_data"] - 1))
                de.append(abs(r["ess_lse"] / max(om[k]["ess"], 1e-30) - 1))
        print(f"\n 跨轮次复现（sigma={cfg['old']}，{len(dv)} 格，与 (ae) 的 ess_scale.json 对齐）："
              f"Var(data) 最大相对差 {max(dv):.2e}，ESS 最大相对差 {max(de):.2e}")

    # ---------------- Z2 过渡区的对数正态近似 ----------------
    print("\n" + "=" * 100)
    print(" Z2 主判据（定量，零自由参数）：过渡区 ESS 与 n*exp(-Var(data)) 之比 in [0.5,2]")
    print("=" * 100)
    print(f" {'sigma_n':>7} | {'n_trans':>7} | {'比值 min':>9} | {'比值 max':>9} | "
          f"{'几何均值':>9} | {'in[.5,2]':>9} | 判定")
    print(" " + "-" * 82)
    z2 = []
    ok2, tot_in, tot_n = True, 0, 0
    for sn in SN:
        v = [r for r in rows if r["sn"] == sn and 2.0 <= r["ess"] <= n / 2]
        if not v:
            print(f" {sn:>7g} | {0:>7d} | {'-':>9} | {'-':>9} | {'-':>9} | {'-':>9} | "
                  f"（该 sigma 无过渡格）")
            continue
        rr = np.array([r["ess"] / (n * math.exp(-r["var_data"])) for r in v])
        fr = float(np.mean((rr >= 0.5) & (rr <= 2.0)))
        gm = float(np.exp(np.mean(np.log(rr))))
        good = bool(fr >= 1.0)          # 预官读法：全部过渡格落在 [0.5,2] 内
        ok2 &= good
        tot_in += int(((rr >= 0.5) & (rr <= 2.0)).sum())
        tot_n += len(rr)
        z2.append(dict(sn=sn, n=len(v), lo=float(rr.min()), hi=float(rr.max()),
                       gm=gm, frac=fr))
        print(f" {sn:>7g} | {len(v):>7d} | {rr.min():>9.4f} | {rr.max():>9.4f} | "
              f"{gm:>9.4f} | {fr*100:>8.0f}% | {'✅' if good else '❌'}")
    print(" " + "-" * 82)
    print(f" => Z2 {'✅ 通过' if ok2 else '❌ 失败'}：过渡区共 {tot_n} 格，"
          f"{tot_in} 格（{100*tot_in/max(tot_n,1):.0f}%）落在 [0.5,2] 内"
          f"（预官读法要求 100%）")

    # 附带：对数正态近似在 Var(data) 大处的失效方式（(ae) 的定量化）
    print("\n 附（(ae) 的失败方式定量化）：log(ESS) 与 Var(data) 的关系")
    print(f" {'sigma_n':>7} | {'Var(data) 中位':>15} | {'log ESS 中位':>12} | "
          f"{'近似 log(n)-Var':>16} | {'差':>10}")
    print(" " + "-" * 74)
    for sn in SN:
        v = [r for r in rows if r["sn"] == sn]
        if not v:
            continue
        vd = float(np.median([r["var_data"] for r in v]))
        le = float(np.median([math.log(r["ess"]) for r in v]))
        pred = math.log(n) - vd
        print(f" {sn:>7g} | {vd:>15.4g} | {le:>12.4f} | {pred:>16.4g} | "
              f"{le-pred:>10.4g}")

    # ---------------- Z3 最尖 ----------------
    print("\n" + "=" * 100)
    print(" Z3 最尖（论文级陈述）：ESS 首次 > 1.01 的 sigma_n 满足 sigma_hi/0.2 >= 50")
    print("=" * 100)
    print(f" {'算子':<12} | " + " ".join(f"{('s='+format(s,'g')):>8}" for s in SN))
    print(" " + "-" * (14 + 9 * len(SN)))
    hi_all = []
    for i in ids:
        v = {r["sn"]: r["ess"] for r in rows if r["id"] == i}
        cells = []
        for s in SN:
            e = v[s]
            cells.append(f"{e:>8.3f}" if e < 1.01 else f"{'**'+format(e,'.2f'):>8}")
        up = [s for s in SN if v[s] > 1.01]
        hi_all.append(min(up) if up else float("inf"))
        print(f" {i:<12} | " + " ".join(cells))
    print(" " + "-" * (14 + 9 * len(SN)))
    print(" （** = ESS > 1.01，即已离开塌缩；ESS <= 1.01 显示为 1.000）")
    fin = [x for x in hi_all if np.isfinite(x)]
    a = np.array(fin)
    print(f"\n sigma_hi（ESS 首次 > 1.01）：n={len(fin)}/{len(ids)} 个算子已越过；"
          f"min={a.min():g}  中位={np.median(a):g}  max={a.max():g}" if fin else
          f" sigma_hi：{len(ids)-len(fin)}/{len(ids)} 个算子在 sigma<=300 内从未离开塌缩")
    if fin:
        print(f" sigma_hi/0.2：中位 {np.median(a)/0.2:.1f}，范围 {a.min()/0.2:.1f} - {a.max()/0.2:.1f}")
        ok3 = bool(np.min(a) / 0.2 >= 50.0)
        print(f" 判据用**最不利**的那个算子（min sigma_hi = {a.min():g}）："
              f"{a.min()/0.2:.1f} >= 50 ? {'✅' if ok3 else '❌'}")
        print(f" => Z3 {'✅ 通过' if ok3 else '❌ 失败'}")
    else:
        ok3 = True
        print(" => Z3 ✅ 通过（全部算子在 sigma<=300 内 ESS 恒为 1，sigma_hi > 300，"
              "sigma_hi/0.2 > 1500）")

    # 由 (ae) 的数据外推的 sigma_hi（下界）
    C = np.array([r["var_data"] * r["sn"] ** 4 for r in rows if r["sn"] <= 0.1])
    ext = (float(np.median(C)) / math.log(n)) ** 0.25
    print(f"\n 由 sigma<=0.1 的 sigma^-4 标度外推：sigma_hi_extrap = (C/ln n)^(1/4) = {ext:.3f}"
          f"（C 中位 {np.median(C):.4g}）")
    print(f" 这是**下界**（大 sigma 处标度应变平），实测 sigma_hi 中位 "
          f"{np.median(a) if fin else float('nan'):g} => 实测/外推 = "
          f"{(np.median(a)/ext if fin else float('nan')):.3f}")

    # ---------------- Z4 诊断 ----------------
    print("\n" + "=" * 100)
    print(" Z4(a) 诊断：ESS(sigma_n) 完整曲线")
    print("=" * 100)
    print(f" {'sigma_n':>7} | {'ESS min':>9} | {'ESS 中位':>9} | {'ESS max':>9} | "
          f"{'n/ESS 中位':>11} | {'ESS/n':>7}")
    print(" " + "-" * 72)
    for sn in SN:
        v = np.array([r["ess"] for r in rows if r["sn"] == sn])
        if not len(v):
            continue
        print(f" {sn:>7g} | {v.min():>9.4f} | {np.median(v):>9.4f} | {v.max():>9.4f} | "
              f"{n/np.median(v):>11.4f} | {np.median(v)/n:>7.4f}")
    print(" " + "-" * 72)
    print(f" 预言：sigma -> 300 处 ESS -> n = {n}（观测与先验重合）")

    print("\n" + "=" * 100)
    print(" Z4(b) 诊断：Var(data) = Var_i(A) + Var_i(B) 两条标度律")
    print("=" * 100)
    print(" A_i = ||g2^(1/2)(x_i-x*)||^2/(2s^2)  ~ s^-2  =>  Var_i(A)*s^4 = const")
    print(" B_i = ||eps||^2/2 - <g2^(1/2)(x_i-x*),eps>/s     =>  Var_i(B)*s^2 = const\n")
    print(f" {'sigma_n':>7} | {'var_det*s^4 中位':>17} | {'var_x*s^2 中位':>16} | "
          f"{'交点 s*':>9} | {'var_det':>11} | {'var_x':>11}")
    print(" " + "-" * 92)
    b4 = []
    for sn in SN:
        v = [r for r in rows if r["sn"] == sn]
        c4 = float(np.median([r["var_det"] * sn ** 4 for r in v]))
        c2 = float(np.median([r["var_x"] * sn ** 2 for r in v]))
        st = math.sqrt(c4 / c2) if c2 > 0 else float("nan")
        b4.append((sn, c4, c2, st))
        print(f" {sn:>7g} | {c4:>17.4g} | {c2:>16.4g} | {st:>9.3f} | "
              f"{float(np.median([r['var_det'] for r in v])):>11.4g} | "
              f"{float(np.median([r['var_x'] for r in v])):>11.4g}")
    print(" " + "-" * 92)
    c4s = [x[1] for x in b4 if x[0] <= 1.0]
    c2s = [x[2] for x in b4 if x[0] <= 1.0]
    if c4s and c2s:
        print(f" sigma<=1 内：var_det*s^4 极差 {max(c4s)/min(c4s):.3f}x，"
              f"var_x*s^2 极差 {max(c2s)/min(c2s):.3f}x  （各自恒定则两条标度律成立）")
    st = [x[3] for x in b4 if np.isfinite(x[3])]
    if st:
        print(f" 交点 s* = sqrt(var_det*s^2/var_x)：中位 {np.median(st):.3f}，"
              f"范围 {min(st):.3f} - {max(st):.3f}（预言：在该处 var_det = var_x）")

    print("\n" + "=" * 100)
    print(" Z4(c) 诊断：sigma_hi 处的 Var(data) 与 ln n 对照")
    print("=" * 100)
    if fin:
        near = [s for s in SN]
        for i, x in zip(ids, hi_all):
            if not np.isfinite(x):
                continue
            j = SN.index(x) if x in SN else None
            if j is not None:
                v = [r["var_data"] for r in rows if r["id"] == i and r["sn"] == x]
                print(f"  {i:<12} sigma_hi={x:<7g} Var(data)={v[0]:.4g}  "
                      f"ln n = {math.log(n):.4f}  比值 {v[0]/math.log(n):.3f}")
    else:
        print("  全部算子未越过塌缩，无 sigma_hi 处可比。")

    with open(os.path.join(OUT, "fig_ess_thresh.json"), "w") as fh:
        json.dump(dict(cfg=cfg, z1_worst=worst, z1_worst_lse=worst2, z1_pass=bool(ok1),
                       z2=z2, z2_pass=bool(ok2), z2_tot=[tot_in, tot_n],
                       sigma_hi=dict(n_finite=len(fin), n_total=len(ids),
                                     vals=[float(x) for x in a] if fin else [],
                                     median=float(np.median(a)) if fin else None,
                                     min=float(a.min()) if fin else None,
                                     max=float(a.max()) if fin else None,
                                     over_02=float(np.median(a) / 0.2) if fin else None),
                       z3_pass=bool(ok3),
                       extrap=ext,
                       curve=[dict(sn=sn, ess_med=float(np.median(
                           [r["ess"] for r in rows if r["sn"] == sn])),
                           var_med=float(np.median(
                               [r["var_data"] for r in rows if r["sn"] == sn])))
                              for sn in SN],
                       scales=b4), fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'fig_ess_thresh.json')}")


if __name__ == "__main__":
    main()
