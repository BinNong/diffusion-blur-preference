"""evidence_scale.py 的离线判据：V1–V6（零 GPU）。

预官与 evidence_scale.py 同一份（md5 6816fb1e8aeab77d86c29a6351be6b82，2026-10-20
11:49:31，早于运行）。判据在看到任何输出之前写定；本脚本只负责算，不负责改阈值。

用法： venv2/bin/python judge_evidence.py
"""

import json
import os

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

# §3.2 的表：EDM FFHQ-64、Gaussian 核 σ=1.5、单张配对图、t = σ_n
REF32 = {0.01: 22210, 0.02: 13290, 0.05: 3530, 0.10: 1054, 0.20: 50}
TOL32 = 0.15


def main():
    d = json.load(open(os.path.join(OUT, "evidence_scale.json")))
    rows = d["rows"]
    cfg = d["cfg"]
    print("=" * 104)
    print(f" evidence_scale 判据：{len(rows)} 格（{cfg['nimg']} 图 × "
          f"{len(cfg['noises'])} 噪声 × 2 真值）")
    print("=" * 104)

    # ---------------- V1 守门 ----------------
    print("\n" + "=" * 104)
    print(" V1 守门：E_y = -(Λ + data)（恒等式）")
    print("=" * 104)
    worst, wat = 0.0, ""
    for r in rows:
        e = abs(r["E_y"] + r["lam"] + r["data"]) / max(1.0, abs(r["E_y"]))
        if e > worst:
            worst, wat = e, f"img{r['img']} σ={r['sn']:g} {r['tag']}"
    print(f" 最大相对误差 = {worst:.3e}   ({wat})")
    if worst > 1e-6:
        print(" ❌ V1 未过 —— 后面全部作废")
        return
    print(" ✅ V1 通过")

    # ---------------- V2 ----------------
    print("\n" + "=" * 104)
    print(" V2 Lemma 12 在真最优处必然成立：每格 E_y < 0")
    print("=" * 104)
    pos = [r for r in rows if r["E_y"] >= 0]
    print(f" E_y < 0 的格子：{len(rows)-len(pos)}/{len(rows)}")
    for r in pos:
        print(f"   反例 img{r['img']} σ={r['sn']:g} {r['tag']}: Λ={r['lam']:+.1f} "
              f"data={r['data']:.1f} E_y={r['E_y']:+.1f}")
    print(f" => V2 {'✅ 通过' if not pos else '❌ 失败'}")

    # ---------------- V3 ----------------
    print("\n" + "=" * 104)
    print(" V3：σ_n=0.02 处 E_hqs > 0 而 E_y < 0，且 gap = E_hqs - E_y ≥ 1000")
    print("=" * 104)
    print(f" {'img':>3} | {'Λ':>9} | {'data':>8} | {'E_y':>10} | {'E_hqs':>10} | "
          f"{'gap':>10} | {'E_hqs/Λ':>8}")
    print(" " + "-" * 78)
    v3 = []
    for r in rows:
        if r["sn"] != 0.02 or r["tag"] != "blur":
            continue
        v3.append(r)
        print(f" {r['img']:>3} | {r['lam']:>9.1f} | {r['data']:>8.1f} | "
              f"{r['E_y']:>10.1f} | {r['E_hqs']:>10.1f} | {r['gap']:>10.1f} | "
              f"{r['E_hqs']/r['lam']:>8.3f}")
    print(" " + "-" * 78)
    ok3 = bool(v3) and all(r["E_hqs"] > 0 and r["E_y"] < 0 and r["gap"] >= 1000
                           for r in v3)
    print(f" => V3 {'✅ 通过' if ok3 else '❌ 失败'}："
          f"{'6600 是 HQS 的优化漏洞，不是物理退化' if ok3 else 'gap 不够大或符号不符'}")

    # ---------------- V4 ----------------
    print("\n" + "=" * 104)
    print(f" V4：干净图 Λ_clean 复现 §3.2 的表（容差 ±{TOL32*100:.0f}%）")
    print("=" * 104)
    print(f" {'σ_n':>6} | {'§3.2 参考':>10} | {'实测均值':>10} | {'逐图':>26} | "
          f"{'相对差':>8} | 判定")
    print(" " + "-" * 84)
    ok4, n4 = True, 0
    for sn in sorted(REF32):
        v = [r["lam_clean"] for r in rows if r["sn"] == sn and r["tag"] == "blur"]
        if not v:
            continue
        m = float(np.mean(v))
        rel = (m - REF32[sn]) / REF32[sn]
        good = abs(rel) <= TOL32
        ok4 &= good
        n4 += 1
        vs = " ".join(f"{x:+.0f}" for x in v)
        print(f" {sn:>6g} | {REF32[sn]:>10d} | {m:>10.1f} | {vs:>26} | "
              f"{rel:>+8.1%} | {'✅' if good else '❌'}")
    print(" " + "-" * 84)
    print(f" => V4 {'✅ 通过' if (ok4 and n4) else '❌ 失败'}（{n4} 个噪声档）")

    # ---------------- V5 ----------------
    print("\n" + "=" * 104)
    print(" V5 符号对照：同一张图、同一束 η 下 Λ(θ*=1.5) > 0 > Λ(θ*=0)")
    print("=" * 104)
    print(f" {'img':>3} | {'σ_n':>6} | {'Λ(blur)':>10} | {'Λ(sharp)':>10} | "
          f"{'差':>10} | 判定")
    print(" " + "-" * 62)
    ok5, n5 = True, 0
    for i in sorted({r["img"] for r in rows}):
        for sn in sorted({r["sn"] for r in rows}):
            a = [r for r in rows if r["img"] == i and r["sn"] == sn
                 and r["tag"] == "blur"]
            b = [r for r in rows if r["img"] == i and r["sn"] == sn
                 and r["tag"] == "sharp"]
            if not (a and b):
                continue
            la, lb = a[0]["lam"], b[0]["lam"]
            good = la > 0 > lb
            ok5 &= good
            n5 += 1
            print(f" {i:>3} | {sn:>6g} | {la:>+10.1f} | {lb:>+10.1f} | "
                  f"{la-lb:>+10.1f} | {'✅' if good else '❌'}")
    print(" " + "-" * 62)
    print(f" => V5 {'✅ 通过' if (ok5 and n5) else '❌ 失败'}（{n5} 对）")

    # ---------------- V6 诊断 ----------------
    print("\n" + "=" * 104)
    print(" V6 诊断（无阈值）")
    print("=" * 104)
    print(" (a) gap 与 E_hqs/Λ 随 σ_n 的走向（跨图均值）")
    print(f" {'σ_n':>6} | {'Λ 均值':>9} | {'data 均值':>10} | {'|E_y| 均值':>11} | "
          f"{'gap 均值':>9} | {'gap/|E_y|':>10} | {'E_hqs/Λ':>8}")
    print(" " + "-" * 84)
    diag = []
    for sn in sorted({r["sn"] for r in rows}):
        b = [r for r in rows if r["sn"] == sn and r["tag"] == "blur"]
        if not b or "gap" not in b[0]:
            continue
        lm = float(np.mean([r["lam"] for r in b]))
        dm = float(np.mean([r["data"] for r in b]))
        em = float(np.mean([abs(r["E_y"]) for r in b]))
        gm = float(np.mean([r["gap"] for r in b]))
        rm = float(np.mean([r["E_hqs"] / r["lam"] for r in b]))
        diag.append(dict(sn=sn, lam=lm, data=dm, absEy=em, gap=gm, ratio=gm / em,
                         eoverlam=rm))
        print(f" {sn:>6g} | {lm:>9.1f} | {dm:>10.1f} | {em:>11.1f} | {gm:>9.1f} | "
              f"{gm/em:>10.3f} | {rm:>8.3f}")
    print(" " + "-" * 84)
    print(" (b) |E_y| - data 是否等于 Λ（V1 已逐格验证，这里报跨图一致性）")
    print(f" {'σ_n':>6} | {'|E_y|-data 均值':>16} | {'Λ 均值':>10} | {'相对差':>8}")
    print(" " + "-" * 50)
    for sn in sorted({r["sn"] for r in rows}):
        b = [r for r in rows if r["sn"] == sn and r["tag"] == "blur"]
        if not b:
            continue
        a = float(np.mean([abs(r["E_y"]) - r["data"] for r in b]))
        lm = float(np.mean([r["lam"] for r in b]))
        print(f" {sn:>6g} | {a:>16.1f} | {lm:>10.1f} | {(a-lm)/lm:>+8.2%}")
    print(" " + "-" * 50)

    with open(os.path.join(OUT, "fig_evidence.json"), "w") as fh:
        json.dump(dict(v1=worst, v1_pass=bool(worst <= 1e-6),
                       v2_pos=[dict(img=r["img"], sn=r["sn"], tag=r["tag"],
                                    lam=r["lam"], data=r["data"], E_y=r["E_y"])
                               for r in pos],
                       v3=[dict(img=r["img"], lam=r["lam"], data=r["data"],
                                E_y=r["E_y"], E_hqs=r["E_hqs"], gap=r["gap"],
                                ratio=r["E_hqs"] / r["lam"]) for r in v3],
                       v3_pass=bool(ok3),
                       v4=[dict(sn=sn, ref=REF32[sn],
                                meas=float(np.mean([r["lam_clean"] for r in rows
                                                    if r["sn"] == sn
                                                    and r["tag"] == "blur"])))
                           for sn in sorted(REF32)
                           if any(r["sn"] == sn and r["tag"] == "blur" for r in rows)],
                       v4_pass=bool(ok4 and n4),
                       v5_pass=bool(ok5 and n5), v5_n=int(n5),
                       v6=diag), fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'fig_evidence.json')}")


if __name__ == "__main__":
    main()
