# -*- coding: utf-8 -*-
"""judge_phaseC_pool20.py —— 20 真实池图过模糊的裁判：N0–N4（零 GPU，只读一个 JSON）

预官：2026-10-29，md5 `7846e2072f502b54c35d430c1f1b602c`（repro/repro_phaseC_pool20.py），
      判据原文见该脚本 docstring 与 RESULTS §3.9(bt)。**本裁判在跑之前写好。**

N1 σ_n≥0.172 上 argmin θ > θ*=1.5 的比例（旧跑 2 图 8/8；<16/20 则论文必须改）
N2 过模糊因子 θ_argmin/θ* 的 mean 与范围（旧跑 1.4896、[1.333,1.667]；mean 出 [1.30,1.65] 则必须改）
N3 θ∈[2.0,2.5] 整段相对 θ* 的优势区间与优于比例（旧跑 1211–2781 nats、39/40）
N4 诊断：低噪声档 argmin θ；top-5 L 跨度（旧跑 247–1031 nats）

用法（服务器上）：
  cd /data/experiment/de_conv_1
  ./venv2/bin/python repro/judge_phaseC_pool20.py
"""

import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
FN = "phaseC_pool20.json"
TSTAR = 1.5
HI = 0.172            # 预官：σ_n ≥ 0.172 才算「高噪声」
SEG = [2.0, 2.125, 2.25, 2.375, 2.5]      # θ∈[2.0,2.5] 的 5 个网格点
# 旧跑（2 图、先验样本、正确核）的已发布值
PUB = dict(n_hi=8, n_hi_ok=8, fac_mean=1.4896, fac_lo=1.3333, fac_hi=1.6667,
           adv_lo=1211.4, adv_hi=2781.1, npair=40, npair_ok=39,
           top5_lo=246.7, top5_hi=1030.7, lo_argmin={0.0415: 1.625, 0.0845: 1.625})


def load():
    p = os.path.join(OUT, FN)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    print("=" * 100)
    print(" 20 真实池图的过模糊裁判：N0–N4（预官 md5 7846e207…，2026-10-29）")
    print("=" * 100)
    d = load()
    if d is None:
        print("  [缺少数据] %s 尚未落地——判据已跑前写定，补齐后重跑即可。" % FN)
        return
    recs = d["records"]
    cfg = d["cfg"]
    ths = cfg["thetas"]
    print("  cfg: nimg=%s  image_source=%s" % (cfg.get("nimg"), cfg.get("image_source")))
    g = cfg.get("kernel_guard", {})
    print("  N0 核守卫: %s" % json.dumps(g, ensure_ascii=False))
    ok0 = (g.get("sym_correct", 9) <= 1e-7 and g.get("peak_correct") == [7, 7]
           and g.get("bitident_correct") is True
           and abs(g.get("kunif_correct_15", 0) - 0.962855) < 1e-4)
    print("  => N0 %s" % ("PASS" if ok0 else "FAIL（不过则下面一个数字都不该被引用）"))
    print("  记录数 %d；噪声档 %s；图 %s"
          % (len(recs), sorted(set(r["noise"] for r in recs)),
             sorted(set(r["img"] for r in recs))))

    hi = [r for r in recs if r["noise"] >= HI]
    lo = [r for r in recs if r["noise"] < HI]
    print("\n  高噪声（σ_n≥%g）格子 %d 个；低噪声格子 %d 个" % (HI, len(hi), len(lo)))

    def argmin(r):
        L = [p["L"] for p in sorted(r["profile"], key=lambda z: z["theta"])]
        return ths[int(min(range(len(L)), key=lambda k: L[k]))]

    # ---------------- N1 ----------------
    nok = sum(1 for r in hi if argmin(r) > TSTAR)
    print("\n----- N1 主判据一：σ_n≥%g 上 argmin θ > θ*=1.5 的比例 -----" % HI)
    print("  旧跑（2 图）%d/%d；本组（%d 图）%d/%d"
          % (PUB["n_hi_ok"], PUB["n_hi"], cfg["nimg"], nok, len(hi)))
    print("  逐格 argmin θ：%s" % sorted(set(argmin(r) for r in hi)))
    print("  未过阈的格子：%s"
          % [(r["img"], r["noise"], argmin(r)) for r in hi if argmin(r) <= TSTAR] or "无")
    n1 = nok >= 16
    print("  => N1 %s（要求 >= 16/%d）" % ("PASS" if n1 else "FAIL", len(hi)))

    # ---------------- N2 ----------------
    fac = [argmin(r) / TSTAR for r in hi]
    print("\n----- N2 主判据二：过模糊因子 θ_argmin/θ* -----")
    print("  旧跑 mean=%.4f 范围 [%.4f, %.4f]" % (PUB["fac_mean"], PUB["fac_lo"], PUB["fac_hi"]))
    print("  本组 mean=%.4f 范围 [%.4f, %.4f]  n=%d"
          % (mean(fac), min(fac), max(fac), len(fac)))
    print("  高出真值 %.1f%%–%.1f%%" % (100 * (min(fac) - 1), 100 * (max(fac) - 1)))
    n2 = 1.30 <= mean(fac) <= 1.65
    print("  => N2 %s（要求 mean ∈ [1.30, 1.65]）" % ("PASS" if n2 else "FAIL"))

    # ---------------- N3 ----------------
    print("\n----- N3 主判据三：θ∈[2.0,2.5] 整段相对 θ* 的优势 -----")
    adv, npair, npok = [], 0, 0
    for r in hi:
        L = [p["L"] for p in sorted(r["profile"], key=lambda z: z["theta"])]
        Ls = L[ths.index(TSTAR)]
        for t in SEG:
            if t not in ths:
                continue
            npair += 1
            a = Ls - L[ths.index(t)]
            adv.append(a)
            if a > 0:
                npok += 1
    print("  旧跑：优势 %d–%d nats，%d/%d 个（格, 点）组合优于 θ*"
          % (PUB["adv_lo"], PUB["adv_hi"], PUB["npair_ok"], PUB["npair"]))
    print("  本组：优势 %.1f–%.1f nats，%d/%d 个（格, 点）组合优于 θ*"
          % (min(adv), max(adv), npok, npair))
    print("  => N3：优于比例 %.1f%%（旧跑 %.1f%%）"
          % (100.0 * npok / npair, 100.0 * PUB["npair_ok"] / PUB["npair"]))

    # ---------------- N4 诊断 ----------------
    print("\n----- N4 诊断（无阈值）-----")
    print("  低噪声档 argmin θ（旧跑 img0：0.0415→1.625、0.0845→1.625；img1：均 1.5）：")
    for sn in sorted(set(r["noise"] for r in lo)):
        v = sorted(set(argmin(r) for r in lo if r["noise"] == sn))
        print("    σ_n=%-7g argmin θ 取值 %s（θ*=1.5）" % (sn, v))
    sp = []
    for r in recs:
        L = sorted(p["L"] for p in r["profile"])[:5]
        sp.append((r["img"], r["noise"], L[-1] - L[0]))
    v = [x[2] for x in sp]
    print("  top-5 L 跨度：旧跑 %d–%d nats；本组 %.1f–%.1f nats（全部 %d 格）"
          % (PUB["top5_lo"], PUB["top5_hi"], min(v), max(v), len(v)))
    vh = [x[2] for x in sp if x[1] >= HI]
    print("    其中高噪声 %d 格：%.1f–%.1f nats" % (len(vh), min(vh), max(vh)))

    print("\n" + "=" * 100)
    print(" 汇总：N0 %s | N1 %s (%d/%d) | N2 %s (mean %.4f) | N3 优于 %.1f%% | "
          "优势 %.0f–%.0f nats"
          % ("PASS" if ok0 else "FAIL", "PASS" if n1 else "FAIL", nok, len(hi),
             "PASS" if n2 else "FAIL", mean(fac), 100.0 * npok / npair,
             min(adv), max(adv)))
    print("=" * 100)


if __name__ == "__main__":
    main()
