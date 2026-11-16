# -*- coding: utf-8 -*-
"""judge_evidence3_kfix.py —— evidence3 正确核重跑的裁判：K1–K3（零 GPU，只读两个 JSON）

预官：2026-10-29 09:41，md5 `04afda226c98c66c3f21f10dcf963d65`（repro/sweep_evidence3_kfix.py），
      判据原文见该脚本 docstring 与 RESULTS §3.9(bs)。**本裁判在跑之前写好。**

K1 逐格配对（只报数，不设通过线）：ΔΛ_clean = 正确核 − 错核，同一 (arm, img, sn)。
K2 主判据：Λ_clean ≥ 2000 nats 的格数与该子集的跨度（错核给 76 格、12.98×）。
   与 12.98× 差 >5% 则论文 §7 的「13.0×」必须改。
K3 诊断（无阈值）：逐 σ_n 的 mean|E_hqs/Λ_clean| 对照；恒等式自检 |pred−E_hqs| 是否仍为 0。

用法（服务器上）：
  cd /data/experiment/de_conv_1
  ./venv2/bin/python repro/judge_evidence3_kfix.py
"""

import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
THR = 2000.0
HUTCH_DIFF = 424.0
PUB_N = 76          # 错核侧 Λ_clean>=2000 的格数
PUB_SPAN = 12.979   # 错核侧该子集跨度
PAIR = ("evidence3.json", "evidence3_kfix.json")


def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    return json.load(open(p))["rows"]


def key(r):
    return (r.get("arm"), r.get("img"), round(float(r.get("sn")), 6))


def cells(rows):
    out = {}
    for r in rows:
        if isinstance(r.get("lam_clean"), (int, float)):
            out[key(r)] = r
    return out


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main():
    print("=" * 100)
    print(" evidence3 正确核重跑的裁判：K1–K3（预官 md5 04afda22…，2026-10-29 09:41）")
    print("=" * 100)
    a, b = load(PAIR[0]), load(PAIR[1])
    if a is None:
        print("  [缺] %s" % PAIR[0])
        return
    if b is None:
        print("  [缺少数据] %s 尚未落地——判据已跑前写定，补齐后重跑即可。" % PAIR[1])
        return
    ca, cb = cells(a), cells(b)
    common = sorted(set(ca) & set(cb))
    print("  错核 %d 格、正确核 %d 格；共同键 %d 个" % (len(ca), len(cb), len(common)))
    if not common:
        print("  [无共同格子] 检查 arm/img/sn 是否一致。")
        return

    # ---------------- K1 逐格配对 ----------------
    d = [cb[k]["lam_clean"] - ca[k]["lam_clean"] for k in common]
    within = sum(1 for x in d if abs(x) <= HUTCH_DIFF)
    print("\n----- K1 逐格配对（只报数，不设通过线）-----")
    print("  ΔΛ_clean = 正确核 − 错核：n=%d  mean=%+.1f  范围 [%+.1f, %+.1f]"
          % (len(d), mean(d), min(d), max(d)))
    print("  |Δ| ≤ Hutchinson 差分底 %.0f nats 的格子：%d/%d（%.1f%%）"
          % (HUTCH_DIFF, within, len(d), 100.0 * within / len(d)))

    # ---------------- K2 格数与跨度 ----------------
    print("\n----- K2 主判据：Λ_clean ≥ %.0f nats 的格数与该子集跨度 -----" % THR)
    sa = {k: v for k, v in ca.items() if v["lam_clean"] >= THR}
    sb = {k: v for k, v in cb.items() if v["lam_clean"] >= THR}
    print("  错核 %d 格（预官记 %d）；正确核 %d 格" % (len(sa), PUB_N, len(sb)))
    for nm, s in (("错核", sa), ("正确核", sb)):
        if len(s) >= 2:
            lc = [v["lam_clean"] for v in s.values()]
            print("    %s：Λ_clean ∈ [%.1f, %.1f]，跨度 %.3f×"
                  % (nm, min(lc), max(lc), max(lc) / min(lc)))
    print("  门槛附近（[1900,2400]）：错核 %s ；正确核 %s"
          % ([round(v["lam_clean"], 1) for v in sorted(sa.values(),
              key=lambda z: z["lam_clean"]) if 1900 <= v["lam_clean"] <= 2400],
             [round(v["lam_clean"], 1) for v in sorted(sb.values(),
              key=lambda z: z["lam_clean"]) if 1900 <= v["lam_clean"] <= 2400]))
    if len(sb) >= 2:
        lo = min(v["lam_clean"] for v in sb.values())
        hi = max(v["lam_clean"] for v in sb.values())
        span = hi / lo
        dev = abs(span - PUB_SPAN) / PUB_SPAN
        print("  与错核跨度 %.3f× 之差 %.2f%%（阈值 5%%）" % (PUB_SPAN, 100 * dev))
        k2 = dev <= 0.05
        print("  => K2 %s" % ("PASS" if k2 else "FAIL（论文 §7 的 13.0× 必须改）"))
    else:
        span, k2 = float("nan"), False
        print("  格子不足 2 个 => K2 FAIL")

    # ---------------- K3 诊断 ----------------
    print("\n----- K3 诊断（无阈值）-----")
    print("  %-8s | %-22s | %-22s | %s" % ("σ_n", "错核 mean|E_hqs/Λ_cl|", "正确核", "差"))
    for sn in sorted(set(round(float(k[2]), 6) for k in common)):
        va = [abs(v["E_hqs"]) / v["lam_clean"] for k, v in sa.items()
              if abs(round(float(k[2]), 6) - sn) < 1e-9 and v["lam_clean"] > 0]
        vb = [abs(v["E_hqs"]) / v["lam_clean"] for k, v in sb.items()
              if abs(round(float(k[2]), 6) - sn) < 1e-9 and v["lam_clean"] > 0]
        if va and vb:
            print("  %-8g | %-22s | %-22s | %+.4f"
                  % (sn, "%.4f (n=%d)" % (mean(va), len(va)),
                     "%.4f (n=%d)" % (mean(vb), len(vb)), mean(vb) - mean(va)))
    # 恒等式自检：E_hqs == dq - lam_clean + (data_term - data)
    err = []
    for k in common:
        r = cb[k]
        if all(isinstance(r.get(f), (int, float)) for f in ("dq", "data_term", "data")):
            err.append(abs((r["dq"] - r["lam_clean"] + (r["data_term"] - r["data"]))
                           - r["E_hqs"]))
    if err:
        print("  恒等式自检 |pred − E_hqs|：max = %.3e（预期 0，逐项恒等、与核无关）"
              % max(err))

    print("\n" + "=" * 100)
    print(" 汇总：格数 %d（错核 %d）；跨度 %.3f×（错核 %.3f×，差 %.2f%%）；"
          "ΔΛ_clean mean %+.1f [%+.1f, %+.1f]；K2 %s"
          % (len(sb), len(sa), span, PUB_SPAN,
             100 * abs(span - PUB_SPAN) / PUB_SPAN if span == span else float("nan"),
             mean(d), min(d), max(d), "PASS" if k2 else "FAIL"))
    print("=" * 100)


if __name__ == "__main__":
    main()
