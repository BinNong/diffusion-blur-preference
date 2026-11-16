"""evidence_scale2.py 的离线判据：W1–W5（零 GPU）。

预官与 evidence_scale2.py 同一份（跑前写定）。判据在看到任何输出之前写定。

用法： venv2/bin/python judge_evidence2.py
"""

import json
import os

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
BIG = 2000.0


def mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def main():
    d = json.load(open(os.path.join(OUT, "evidence_scale2.json")))
    rows = d["rows"]
    cfg = d["cfg"]
    blur = [r for r in rows if r["tag"] == "blur"]
    print("=" * 100)
    print(f" evidence_scale2 判据：{len(rows)} 格（{cfg['nimg']} 图 × "
          f"{len(cfg['noises'])} 噪声 × θ*={cfg['thetas']} + 对照）")
    print("=" * 100)

    # ---------------- W1 ----------------
    print("\n" + "=" * 100)
    print(" W1 守门：E_y = -(Λ_noisy + data)")
    print("=" * 100)
    worst, wat = 0.0, ""
    for r in rows:
        e = abs(r["E_y"] + r["lam"] + r["data"]) / max(1.0, abs(r["E_y"]))
        if e > worst:
            worst, wat = e, f"img{r['img']} σ={r['sn']:g} θ*={r['theta']:g}"
    print(f" 最大相对误差 = {worst:.3e}   ({wat})")
    if worst > 1e-6:
        print(" ❌ W1 未过 —— 后面全部作废")
        return
    print(" ✅ W1 通过")

    # ---------------- W2 ----------------
    print("\n" + "=" * 100)
    print(f" W2：Λ_clean≥{BIG:.0f} 的格子上 mean(|E_hqs|/Λ_clean)∈[1.00,1.25] 且 SE≤0.03")
    print("=" * 100)
    big = [r for r in blur if r["lam_clean"] is not None
           and r["lam_clean"] >= BIG and r.get("E_hqs") is not None]
    ratios = [abs(r["E_hqs"]) / r["lam_clean"] for r in big]
    m, se = mean_se(ratios)
    print(f" n = {len(big)} 格（Λ_clean 从 {min(r['lam_clean'] for r in big):.0f} 到 "
          f"{max(r['lam_clean'] for r in big):.0f}，"
          f"{max(r['lam_clean'] for r in big)/min(r['lam_clean'] for r in big):.1f} 倍）")
    print(f" 比值：mean={m:.4f}  SE={se:.4f}  min={min(ratios):.4f}  "
          f"max={max(ratios):.4f}  极差={max(ratios)-min(ratios):.4f}")
    ok2 = (1.00 <= m <= 1.25) and (se <= 0.03)
    print(f" => W2 {'✅ 通过' if ok2 else '❌ 失败'}")

    # ---------------- W3 ----------------
    print("\n" + "=" * 100)
    print(" W3：跨核强度普适 —— 两个 θ* 上的 mean 比值之差 ≤ 2·SE_combined")
    print("=" * 100)
    print(f" {'θ*':>5} | {'n':>4} | {'mean 比值':>10} | {'SE':>8} | "
          f"{'min':>7} | {'max':>7}")
    print(" " + "-" * 60)
    per = {}
    for th in cfg["thetas"]:
        v = [abs(r["E_hqs"]) / r["lam_clean"] for r in big if r["theta"] == th]
        if len(v) < 2:
            continue
        mm, ss = mean_se(v)
        per[th] = (mm, ss, len(v))
        print(f" {th:>5g} | {len(v):>4d} | {mm:>10.4f} | {ss:>8.4f} | "
              f"{min(v):>7.4f} | {max(v):>7.4f}")
    print(" " + "-" * 60)
    if len(per) >= 2:
        ths = sorted(per)
        (m1, s1, n1), (m2, s2, n2) = per[ths[0]], per[ths[-1]]
        diff = abs(m1 - m2)
        comb = float(np.sqrt(s1 ** 2 + s2 ** 2))
        t = diff / comb if comb > 0 else float("inf")
        print(f" θ*={ths[0]:g} vs {ths[-1]:g}：|Δmean|={diff:.4f}  2·SE_comb={2*comb:.4f}  "
              f"t={t:.2f}")
        ok3 = diff <= 2 * comb
        print(f" => W3 {'✅ 通过' if ok3 else '❌ 失败'}："
              f"{'比值跨核强度不可区分' if ok3 else '比值依赖核强度，=Λ_clean 要改写成 =k(θ)·Λ_clean'}")
    else:
        ok3 = False
        print(" => W3 ❌ 失败（θ* 分组不足）")

    # ---------------- W4 ----------------
    print("\n" + "=" * 100)
    print(" W4 机理直接验证：(i) [q(x)-q(h⋆x*)]/Λ_clean ∈[-0.15,0.05]；"
          "(ii) data_term(x)/data ∈[0.5,2.0]")
    print("=" * 100)
    print(f" {'img':>3} | {'σ_n':>6} | {'θ*':>5} | {'Λ_clean':>9} | {'Δq/Λ_cl':>9} | "
          f"{'data_t/data':>11} | 判定")
    print(" " + "-" * 72)
    ok4, n4 = True, 0
    for r in sorted(big, key=lambda r: (r["img"], r["sn"], r["theta"])):
        a = r["dq_over_lam"]
        b = r["xdata_over_data"]
        good = (a is not None and b is not None and -0.15 <= a <= 0.05
                and 0.5 <= b <= 2.0)
        ok4 &= good
        n4 += 1
        print(f" {r['img']:>3} | {r['sn']:>6g} | {r['theta']:>5g} | "
              f"{r['lam_clean']:>9.1f} | {a:>+9.4f} | {b:>11.3f} | "
              f"{'✅' if good else '❌'}")
    print(" " + "-" * 72)
    print(f" => W4 {'✅ 通过' if (ok4 and n4) else '❌ 失败'}（{n4} 格）")

    # ---------------- W5 诊断 ----------------
    print("\n" + "=" * 100)
    print(" W5 诊断（无阈值）：与 (z) 的 img0-2 复现性；θ*=0 对照")
    print("=" * 100)
    old_p = os.path.join(OUT, "evidence_scale.json")
    if os.path.exists(old_p):
        old = {(r["img"], r["sn"], r["theta"]): r
               for r in json.load(open(old_p))["rows"] if r["tag"] == "blur"}
        print(f" {'img':>3} | {'σ_n':>6} | {'Λ_cl (z)':>10} | {'Λ_cl (新)':>10} | "
              f"{'E_hqs (z)':>10} | {'E_hqs (新)':>10} | {'差':>8}")
        print(" " + "-" * 76)
        nd = 0
        for r in sorted(big, key=lambda r: (r["img"], r["sn"], r["theta"])):
            k = (r["img"], r["sn"], r["theta"])
            if k not in old or r["img"] >= 3:
                continue
            o = old[k]
            de = r["E_hqs"] - o["E_hqs"]
            nd += 1
            print(f" {r['img']:>3} | {r['sn']:>6g} | {o['lam_clean']:>10.1f} | "
                  f"{r['lam_clean']:>10.1f} | {o['E_hqs']:>10.1f} | "
                  f"{r['E_hqs']:>10.1f} | {de:>+8.1f}")
        print(" " + "-" * 76)
        print(f" 复现格数 {nd}；ΔE_hqs 应 ≈0（同 seed、同图、同代码路径）。")
    sh = [r for r in rows if r["tag"] == "sharp"]
    print(f"\n θ*=0 对照：{len(sh)} 格，Λ_noisy 均值 "
          f"{np.mean([r['lam'] for r in sh]):+.1f}，"
          f"E_y<0 的格数 {sum(1 for r in sh if r['E_y'] < 0)}/{len(sh)}")

    with open(os.path.join(OUT, "fig_evidence2.json"), "w") as fh:
        json.dump(dict(w1=worst, w1_pass=bool(worst <= 1e-6),
                       w2=dict(n=len(big), mean=m, se=se, lo=min(ratios),
                               hi=max(ratios)), w2_pass=bool(ok2),
                       w3={str(k): dict(mean=v[0], se=v[1], n=v[2])
                           for k, v in per.items()},
                       w3_pass=bool(ok3),
                       w4_pass=bool(ok4 and n4), w4_n=int(n4),
                       w5_sharp=dict(n=len(sh),
                                     lam_mean=float(np.mean([r["lam"] for r in sh])),
                                     n_neg=int(sum(1 for r in sh if r["E_y"] < 0)))),
                  fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'fig_evidence2.json')}")


if __name__ == "__main__":
    main()
