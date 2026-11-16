# -*- coding: utf-8 -*-
"""judge_evidence_scale2_kfix.py —— 正确核重跑的裁判：K1–K4（零 GPU，只读两个 JSON）

预官：2026-10-28，md5 `744bc4f4d785d53706e26f2dd2c85c78`（repro/evidence_scale2_kfix.py），
      判据原文见该脚本 docstring 与 RESULTS §3.9(bn)。**本裁判在跑之前写好。**

修订（跑前、看到任何正确核数字之前）：预官 K1 写「SE_buggy = lam_std/sqrt(nimg)（错核文件
自带）」，但 `evidence_scale2.json` 的行里**没有 lam_std 这一列**（只有
E_y/data/img/lam/lam_clean/qx/qy/sn/tag/theta），所以 z 分数算不出来。
K1 本来就不设通过线（预官原文：「只报 median|z| 与 max|z|，不设通过线——本组不是复现实验，
是换核后的第一次有效测量」），故改为报**可算的**量：逐格 ΔΛ_clean 与其相对值、以及
|Δ| 相对 Hutchinson 差分噪声底（424 nats，§3.9(al) M3）的位置。不影响任何判定。

K2/K3/K4 与预官逐字相同。

用法（服务器上）：
  cd /data/experiment/de_conv_1
  ./venv2/bin/python repro/judge_evidence_scale2_kfix.py
"""

import json
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
THR = 2000.0          # 预官 K2/K3 的门槛：Λ_clean >= 2000 nats
HUTCH_DIFF = 424.0    # Hutchinson 对**差值**的噪声底（§3.9(al) M3）
PUB_SPAN = 14.606     # 错核侧的跨度（[2130.8, 31121.8]）
PUB_N = 76            # 错核侧的格数
PUB_RATIO = {1.0: 1.2284, 1.5: 1.0767, 2.0: 1.0215}   # §3.9(ac) 错核侧逐 θ*
PAIR = [("ffhq", "evidence_scale2.json", "evidence_scale2_kfix.json"),
        ("afhqv2", "evidence_scale2_afq.json", "evidence_scale2_afq_kfix.json")]


def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d["cfg"], d["rows"]


def key(r):
    return (r.get("img"), round(float(r.get("sn")), 6), round(float(r.get("theta")), 6))


def cells(rows):
    """(img, sn, theta) -> lam_clean / E_hqs，只取有 lam_clean 的行。"""
    out = {}
    for r in rows:
        if isinstance(r.get("lam_clean"), (int, float)):
            out[key(r)] = r
    return out


def sel(c, thr=THR):
    return {k: v for k, v in c.items() if v["lam_clean"] >= thr}


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    print("=" * 100)
    print(" 正确核重跑的裁判：K1–K4（预官 md5 744bc4f4…，2026-10-28）")
    print("=" * 100)
    res = {}
    for name, fbug, fkix in PAIR:
        print("\n" + "=" * 100)
        print(" 先验臂：%s   错核 %s  vs  正确核 %s" % (name, fbug, fkix))
        print("=" * 100)
        a = load(fbug)
        b = load(fkix)
        if a is None:
            print("  [缺] %s" % fbug)
            return
        if b is None:
            print("  [缺少数据] %s 尚未落地——判据已跑前写定，补齐后重跑即可。" % fkix)
            return
        ca, cb = cells(a[1]), cells(b[1])
        print("  错核 %d 格、正确核 %d 格；共同键 %d 个" % (len(ca), len(cb), len(set(ca) & set(cb))))
        common = sorted(set(ca) & set(cb))
        if not common:
            print("  [无共同格子] 无法比较——检查 img/sn/theta 是否一致。")
            return

        # ---------------- K1 配对一致性（无通过线） ----------------
        d = [cb[k]["lam_clean"] - ca[k]["lam_clean"] for k in common]
        rel = [(cb[k]["lam_clean"] - ca[k]["lam_clean"]) / ca[k]["lam_clean"]
               for k in common if abs(ca[k]["lam_clean"]) > 1e-9]
        within = sum(1 for x in d if abs(x) <= HUTCH_DIFF)
        print("\n----- K1 配对一致性（只报数，不设通过线）-----")
        print("  ΔΛ_clean = 正确核 − 错核：n=%d  mean=%+.1f  范围 [%+.1f, %+.1f]"
              % (len(d), mean(d), min(d), max(d)))
        print("  相对差：mean %+.3f%%  范围 [%+.3f%%, %+.3f%%]"
              % (100 * mean(rel), 100 * min(rel), 100 * max(rel)))
        print("  |Δ| ≤ Hutchinson 差分底 %.0f nats 的格子：%d/%d（%.1f%%）"
              % (HUTCH_DIFF, within, len(d), 100.0 * within / len(d)))

        # ---------------- K2 格数 ----------------
        sa, sb = sel(ca), sel(cb)
        print("\n----- K2 主判据一：Λ_clean ≥ %.0f nats 的格数 -----" % THR)
        print("  错核 %d 格（预官记 %d）；正确核 %d 格" % (len(sa), PUB_N, len(sb)))
        near = sorted(v["lam_clean"] for v in cb.values() if 1900 <= v["lam_clean"] <= 2400)
        print("  正确核侧门槛附近（[1900,2400]）的值：%s"
              % ([round(x, 1) for x in near] or "无"))
        print("  错核侧同区间的值：%s"
              % ([round(x, 1) for x in sorted(v["lam_clean"] for v in ca.values()
                                              if 1900 <= v["lam_clean"] <= 2400)] or "无"))
        k2 = len(sb) in (75, 76)
        print("  => K2 %s（要求 ∈ {75,76}）" % ("PASS" if k2 else "FAIL"))

        # ---------------- K3 跨度 ----------------
        print("\n----- K3 主判据二：该子集的 Λ_clean 跨度 -----" )
        if len(sb) >= 2:
            lo, hi = min(v["lam_clean"] for v in sb.values()), max(v["lam_clean"] for v in sb.values())
            span = hi / lo
            print("  正确核：[%.1f, %.1f] = %.3f×（错核 %.3f×）" % (lo, hi, span, PUB_SPAN))
            dev = abs(span - PUB_SPAN) / PUB_SPAN
            print("  与错核跨度之差 %.2f%%（阈值 5%%）" % (100 * dev))
            k3 = dev <= 0.05
            print("  => K3 %s" % ("PASS" if k3 else "FAIL"))
        else:
            span, k3 = float("nan"), False
            print("  格子不足 2 个，无法算跨度 => K3 FAIL")

        # ---------------- K4 诊断（无阈值） ----------------
        print("\n----- K4 诊断（无阈值）：逐 θ* 的 mean|E_hqs/Λ_clean| -----")
        print("  %-6s | %-22s | %-22s | %s" % ("θ*", "错核（预官记）", "正确核", "差"))
        r4 = {}
        for th in (1.0, 1.5, 2.0):
            va = [abs(v["E_hqs"]) / v["lam_clean"] for v in sa.values()
                  if abs(float(v.get("theta", -1)) - th) < 1e-9 and v["lam_clean"] > 0]
            vb = [abs(v["E_hqs"]) / v["lam_clean"] for v in sb.values()
                  if abs(float(v.get("theta", -1)) - th) < 1e-9 and v["lam_clean"] > 0]
            ma, mb = mean(va), mean(vb)
            r4[th] = (ma, mb)
            print("  %-6g | %-22s | %-22s | %+.4f"
                  % (th, "%.4f (n=%d)" % (ma, len(va)), "%.4f (n=%d)" % (mb, len(vb)),
                     (mb - ma) if va and vb else float("nan")))
        print("  （预官记的错核值：%s）"
              % {k: PUB_RATIO[k] for k in sorted(PUB_RATIO)})

        res[name] = dict(n_bug=len(sa), n_kix=len(sb), k2=bool(k2),
                         span_kix=float(span), span_bug=PUB_SPAN, k3=bool(k3),
                         d_mean=float(mean(d)), d_min=float(min(d)), d_max=float(max(d)),
                         within=within, n_common=len(common), r4=r4)

    # ---------------- 跨先验：0.008–0.021 是否仍成立 ----------------
    if "ffhq" in res and "afhqv2" in res:
        print("\n" + "=" * 100)
        print(" 跨先验：逐 θ* 的 mean 比值之差是否仍落在 0.008–0.021")
        print("=" * 100)
        for th in (1.0, 1.5, 2.0):
            a = res["ffhq"]["r4"].get(th, (float("nan"),) * 2)[1]
            b = res["afhqv2"]["r4"].get(th, (float("nan"),) * 2)[1]
            if a == a and b == b:
                print("  θ*=%-4g ffhq %.4f  afhqv2 %.4f  |差| = %.4f  %s"
                      % (th, a, b, abs(a - b),
                         "在 0.008–0.021 内" if 0.008 <= abs(a - b) <= 0.021 else "**在外**"))

    print("\n" + "=" * 100)
    print(" 汇总")
    print("=" * 100)
    for name, r in res.items():
        print("  %-7s 格数 %d（错核 %d） K2=%s  跨度 %.3f×（错核 %.3f×） K3=%s  "
              "ΔΛ_clean mean %+.1f [%+.1f, %+.1f]"
              % (name, r["n_kix"], r["n_bug"], "PASS" if r["k2"] else "FAIL",
                 r["span_kix"], r["span_bug"], "PASS" if r["k3"] else "FAIL",
                 r["d_mean"], r["d_min"], r["d_max"]))


if __name__ == "__main__":
    main()
