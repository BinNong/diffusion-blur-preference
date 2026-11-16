"""裁判：第十七组 (am) —— Phase C 的 marginalized profile 在正确高斯核上重测

只读 results/phaseC_kfix.json + results/phaseC_sweep.json，零 GPU。
判据 N0–N5 预官在 repro/repro_phaseC_kfix.py 的 docstring 里
（md5 64688016826fdffd248bc33549388a0c，2026-10-23 20:52:46，早于运行）。

用法： venv2/bin/python judge_phaseC_kfix.py [--in phaseC_kfix.json]
"""

import argparse
import json
import math
import os

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_PROJ, "results")

# 预官值（离线算出，N0(d) 对账用）
KUNIF_BUGGY_15 = 0.967782
KUNIF_CORRECT_15 = 0.962855
SIGMA_MATCH_PRE = {1.5: 1.6154, 2.0: 2.1530, 2.5: 2.6871}
# 预官的通过/否掉线
N1_TOL = 600.0            # nats，Hutchinson 对差值的底实测 424
N2_PASS, N2_REFUTE = 6, 4
N3_LO, N3_HI = 1.33, 1.63  # 旧跑 1.479 ± 0.15
N4_MAX_ZERO = 4


def load(fn):
    with open(os.path.join(OUT, fn)) as fh:
        return json.load(fh)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def prof_map(rec):
    return {round(e["theta"], 4): e for e in rec["profile"]}


def argmin_theta(rec):
    p = sorted(rec["profile"], key=lambda e: e["theta"])
    return p[min(range(len(p)), key=lambda k: p[k]["L"])]["theta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="fn", default="phaseC_kfix.json")
    args = ap.parse_args()
    d = load(args.fn)
    cfg = d["cfg"]
    old = load("phaseC_sweep.json")
    oldmap = {(r["img"], round(r["noise"], 4)): r for r in old["records"]}
    recA = [r for r in d["records"] if r["arm"] == "buggy"]
    recB = [r for r in d["records"] if r["arm"] == "correct"]
    print(f"[cfg] θ*={cfg['tstar']} n_inner={cfg['n_inner']} θ 网格 {len(cfg['thetas'])} 点；"
          f"臂 A（错核）{len(recA)} 格、臂 B（正确核）{len(recB)} 格")

    # ------------------------------------------------------------------ N0
    print("\n===== N0 核守卫 =====")
    g = cfg.get("kernel_guard", {})
    print(f"  错核：不对称 {g.get('sym_buggy'):.3e}、峰 {g.get('peak_buggy')}"
          f"（应 [6,6]、>1e-7 —— 这就是 bug 的判据）")
    print(f"  正确核：不对称 {g.get('sym_correct'):.3e}、峰 {g.get('peak_correct')}、"
          f"Σk={g.get('sum_correct'):.9f}、与 putils 逐比特相同={g.get('bitident_correct')}")
    print(f"  kunif(σ=1.5)：错核 {g.get('kunif_buggy_15'):.6f}（预官 {KUNIF_BUGGY_15:.6f}）、"
          f"正确核 {g.get('kunif_correct_15'):.6f}（预官 {KUNIF_CORRECT_15:.6f}）")
    sm = g.get("sigma_match", {})
    print(f"  kunif 匹配的正确 σ：{ {k: round(v, 4) for k, v in sm.items()} }"
          f"（预官 {SIGMA_MATCH_PRE}）")
    ok0 = (g.get("sym_correct", 9) <= 1e-7 and g.get("peak_correct") == [7, 7]
           and g.get("bitident_correct") is True and g.get("sym_buggy", 0) > 1e-7
           and abs(g.get("kunif_buggy_15", 0) - KUNIF_BUGGY_15) < 1e-4
           and abs(g.get("kunif_correct_15", 0) - KUNIF_CORRECT_15) < 1e-4)
    print(f"  ⇒ {'✅ 守卫通过' if ok0 else '❌ 守卫失败'}")

    # ------------------------------------------------------------------ N1
    print("\n===== N1 定位验证：臂 A（错核）须复现 phaseC_sweep.json =====")
    print(f"  容差 |ΔL| ≤ {N1_TOL:.0f} nats 且 argmin θ 完全相同"
          f"（Hutchinson 对差值的底实测 424 nats，§3.9(al) M3）")
    print(f"  {'img':>3} {'sn':>7} {'argmin旧':>8} {'argminA':>8} {'max|ΔL|':>9} "
          f"{'超差θ数':>7}  判定")
    n1_ok, n1_cells = 0, 0
    for r in sorted(recA, key=lambda r: (r["img"], r["noise"])):
        k = (r["img"], round(r["noise"], 4))
        if k not in oldmap:
            continue
        o = oldmap[k]
        pm, po = prof_map(r), prof_map(o)
        ds = [(th, pm[th]["L"] - po[th]["L"]) for th in po if th in pm]
        mx = max(abs(v) for _, v in ds)
        nbad = sum(1 for _, v in ds if abs(v) > N1_TOL)
        a_o, a_a = argmin_theta(o), argmin_theta(r)
        ok = (mx <= N1_TOL) and (a_o == a_a)
        n1_ok += ok
        n1_cells += 1
        print(f"  {r['img']:>3} {r['noise']:>7} {a_o:>8} {a_a:>8} {mx:>9.1f} "
              f"{nbad:>5}/{len(ds)}  {'✅' if ok else '❌'}")
    print(f"  ⇒ {n1_ok}/{n1_cells} 格在容差内且 argmin 相同"
          f"{'（✅ 定位正确，臂 B 的数字有参照）' if n1_ok == n1_cells and n1_cells else ''}")
    if n1_cells and n1_ok < n1_cells:
        print("  ❌ 定位验证不过——**臂 B 的数字不报物理**，先查脚本配置。")

    # ------------------------------------------------------------------ N2
    print("\n===== N2 主判据：过度模糊在正确核上存活（σ_n ≥ 0.172 上 argmin θ > θ*）=====")
    hi = [r for r in recB if r["noise"] >= 0.172]
    n_over = sum(1 for r in hi if argmin_theta(r) > cfg["tstar"])
    print(f"  {'img':>3} {'sn':>7} {'argmin θ':>9} {'因子 θ/θ*':>10} "
          f"{'L(θ=0)':>11} {'L(θ*)':>11} {'θ=0胜':>6}")
    for r in sorted(recB, key=lambda r: (r["img"], r["noise"])):
        pm = prof_map(r)
        a = argmin_theta(r)
        L0 = pm[min(pm)]["L"]
        Ls = min(abs(th - cfg["tstar"]) for th in pm)
        Lstar = pm[[th for th in pm if abs(th - cfg["tstar"]) == Ls][0]]["L"]
        print(f"  {r['img']:>3} {r['noise']:>7} {a:>9} {a/cfg['tstar']:>10.3f} "
              f"{L0:>11.1f} {Lstar:>11.1f} {'Y' if L0 < Lstar else 'N':>6}")
    print(f"  ⇒ σ_n ≥ 0.172 的 {len(hi)} 格中 argmin θ > θ* 的有 **{n_over}** 个"
          f"（通过线 {N2_PASS}、否掉线 {N2_REFUTE}；旧跑错核是 8/8）")
    if n_over >= N2_PASS:
        v2 = "✅ 通过"
    elif n_over <= N2_REFUTE:
        v2 = "❌ 否掉——过度模糊主要是错核的伪影"
    else:
        v2 = "⚠️ 灰区，记为不可判定"
    print(f"  ⇒ {v2}")

    # ------------------------------------------------------------------ N3
    print("\n===== N3 最尖的一条：过度模糊因子 θ_argmin/θ* 对 kunif 映射不变 =====")
    old_hi = [r for r in old["records"] if r["noise"] >= 0.172]
    f_old = [argmin_theta(r) / old["cfg"]["sigma_star"] for r in old_hi]
    f_new = [argmin_theta(r) / cfg["tstar"] for r in hi]
    print(f"  旧跑（错核，{len(f_old)} 格）：因子 mean = {mean(f_old):.4f}，"
          f"范围 [{min(f_old):.3f}, {max(f_old):.3f}]")
    print(f"  本组（正确核，{len(f_new)} 格）：因子 mean = {mean(f_new):.4f}，"
          f"范围 [{min(f_new):.3f}, {max(f_new):.3f}]")
    inb = N3_LO <= mean(f_new) <= N3_HI
    print(f"  预官带 = 旧值 ± 0.15 = [{N3_LO}, {N3_HI}]  ⇒ "
          f"{'✅ 落在带内——因子不变' if inb else '❌ 掉出带——过度模糊幅度依赖核的精细结构'}")
    print("  kunif 匹配版（把本组 argmin θ 换算成等 kunif 的错核 σ，应与旧跑 argmin 一致）：")
    print(f"  {'img':>3} {'sn':>7} {'本组argmin':>10} {'旧跑argmin':>10} {'差':>7}")
    for r in sorted(hi, key=lambda r: (r["img"], r["noise"])):
        k = (r["img"], round(r["noise"], 4))
        if k not in oldmap:
            continue
        a_new, a_old = argmin_theta(r), argmin_theta(oldmap[k])
        # 等 kunif 换算：本组的 σ_new 对应错核的 σ_old（用线性近似斜率 1.077）
        conv = a_new / 1.0770
        print(f"  {r['img']:>3} {r['noise']:>7} {a_new:>10} {a_old:>10} "
              f"{conv-a_old:>+7.3f}")

    # ------------------------------------------------------------------ N4
    print("\n===== N4 盆地结构 =====")
    nz = [r for r in recB if argmin_theta(r) == 0.0]
    print(f"  臂 B 的 {len(recB)} 格中 argmin θ = 0 的有 **{len(nz)}** 个"
          f"（预官通过线 ≤ {N4_MAX_ZERO}）⇒ "
          f"{'✅' if len(nz) <= N4_MAX_ZERO else '❌ θ=0 深井占主导'}")
    old_nz = [r for r in old["records"] if argmin_theta(r) == 0.0]
    print(f"  旧跑 24 格中 argmin θ = 0 的有 {len(old_nz)} 个："
          f"{[(r['img'], r['noise']) for r in old_nz]}")
    print(f"  本组 argmin θ = 0 的格子：{[(r['img'], r['noise']) for r in nz]}")
    print("  basin collapse 阈值：旧跑结论是「未出现」（全局极小迁移到过度模糊而非塌掉）。"
          "本组逐格看 argmin θ 是否始终 ≥ 0 且 profile 是否有第二盆地——见 N5 全表。")

    # ------------------------------------------------------------------ N5
    print("\n===== N5 诊断（无阈值）=====")
    print("  (a) 重叠噪声档上本组与旧跑的 argmin θ 对照：")
    print(f"  {'img':>3} {'sn':>7} {'旧跑(错核)':>10} {'本组(正确)':>11} {'差':>7}")
    for r in sorted(recB, key=lambda r: (r["img"], r["noise"])):
        k = (r["img"], round(r["noise"], 4))
        if k not in oldmap:
            continue
        print(f"  {r['img']:>3} {r['noise']:>7} {argmin_theta(oldmap[k]):>10} "
              f"{argmin_theta(r):>11} {argmin_theta(r)-argmin_theta(oldmap[k]):>+7.3f}")
    print("  (b) 绝对 L 的偏移：同 (img, sn, θ) 上本组减旧跑（只在重叠档）")
    ds = []
    for r in recB:
        k = (r["img"], round(r["noise"], 4))
        if k not in oldmap:
            continue
        pm, po = prof_map(r), prof_map(oldmap[k])
        for th in po:
            if th in pm:
                ds.append(pm[th]["L"] - po[th]["L"])
    if ds:
        print(f"      n={len(ds)}，mean = {mean(ds):+.1f} nats，范围 "
              f"[{min(ds):+.1f}, {max(ds):+.1f}]；|L| 本身量级 "
              f"{min(abs(x) for x in ds):.0f}–{max(abs(x) for x in ds):.0f}")
    print("  (c) kunif 匹配映射（N0 已算，σ=1.5/2.0/2.5 三点）：")
    print(f"      错核 σ → 等 kunif 的正确 σ：{ {k: round(v, 4) for k, v in sm.items()} }；"
          f"近似线性斜率 ≈ 1.077，用于 (a) 的换算")

    print("\n===== 汇总 =====")
    print(f"  N0 核守卫 {'✅' if ok0 else '❌'}")
    print(f"  N1 定位验证 {n1_ok}/{n1_cells}")
    print(f"  N2 过度模糊存活 {n_over}/{len(hi)} ⇒ {v2}")
    print(f"  N3 因子 mean {mean(f_new):.4f}（旧 {mean(f_old):.4f}，带 "
          f"[{N3_LO},{N3_HI}]）⇒ {'✅' if inb else '❌'}")
    print(f"  N4 argmin=0 的格子 {len(nz)}/{len(recB)}（通过线 ≤{N4_MAX_ZERO}）")


if __name__ == "__main__":
    main()
