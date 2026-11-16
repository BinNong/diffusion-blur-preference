# -*- coding: utf-8 -*-
"""diag_evidence3_existing_kfix.py —— 零 GPU 诊断：§3.9(bs) 那条链想产出的数据，可能已经存在

动机（2026-10-30，零 GPU 文档/数据自查）
-----------------------------------------
§3.9(bs) 预官了 `sweep_evidence3_kfix.py`（约 3.4 h GPU，10-29 09:54 随链发出），
目的是给论文 §7 的「the twenty-image run gives $13.0\times$」补一个**正确核**的值。
(bs) 与 (bm2) 都记载：「`evidence4.json` 没有正确核版本」。

本诊断只用**本地已有的两个 JSON** 复算 K1–K3，结论是那句记载不成立：

  * `evidence3.json`  cfg 无 `kernel_guard`            → 手写错核（§3.9(aj) 那轮）
  * `evidence4.json`  cfg.kernel_guard sym=0, peak=[7,7], kunif=0.962854
                                                       → putils 正确核
  而 `evidence4.json` 正是 §3.9(ak)（预官 2026-10-23 16:49:56，md5 `6cf1b0ef…`）
  「只改核的来源、其余与 (aj)/(z) 逐字相同」的那一轮产出——**同一个设计的正确核重跑**。
  两文件 115 个 (arm, img, sn) 格子**逐一相同**，θ 全为 0.0，噪声档一致，
  所以它们是天然的逐格配对，差别只来自核。

**本脚本不是 (bs) 的裁判，不替代其预官。** 它只回答一个问题：
在花 3.4 h GPU 之前，正确核的那一侧是不是已经躺在仓库里了。

判据（沿用 §3.9(bs) 的 K1–K3 定义，逐条照抄，阈值不改）
------------------------------------------------------
K1  逐格 ΔΛ_clean = evidence4 − evidence3（正确核 − 错核）：只报 mean / 范围 /
    |Δ| ≤ 424 nats 差分底的占比，不设通过线。
K2  Λ_clean ≥ 2000 的格数与该子集跨度。错核基线 76 格、12.979×（(bs) 预官值）。
    与 12.979× 差 >5% ⇒ 论文 §7 的「13.0×」必须改。
K3  诊断：逐 σ_n 的 mean|E_hqs/Λ_clean| 对照；恒等式自检 |pred − E_hqs| 是否仍为 0。
    恒等式：E_hqs = Δq − Λ_clean + (data_term − data)。

用法（本地，零 GPU，只用标准库）：
  python3 repro/diag_evidence3_existing_kfix.py
"""

import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

THR = 2000.0
HUTCH_DIFF = 424.0        # Hutchinson 差分底（错核臂的值，§3.9(ap)）
PUB_N = 76                # (bs) 预官：错核侧 Λ_clean>=2000 的格数
PUB_SPAN = 12.979         # (bs) 预官：错核侧该子集跨度
TOL = 0.05                # (bs) 预官：跨度差 >5% 则论文 §7 必须改

BUGGY = "evidence3.json"      # 无 kernel_guard → 手写错核
CORRECT = "evidence4.json"    # kernel_guard: sym=0, peak=[7,7] → putils 正确核


def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def cells(rows):
    """与 repro/judge_evidence3_kfix.py 的 cells() 同一个键：(arm, img, round(sn,6))。

    每个键有 tag='blur' / 'sharp' 两行；两行的 lam_clean 逐一相同（已实测 115/115 格
    零差异），所以字典后者覆盖前者不影响 K1/K2。E_hqs 只存在于 'blur' 行，K3 另行过滤。
    """
    out = {}
    for r in rows:
        if isinstance(r.get("lam_clean"), (int, float)):
            out[(r.get("arm"), r.get("img"), round(float(r.get("sn")), 6))] = r
    return out


def with_ehqs(rows):
    """直接从原始行里取有 E_hqs 的格子（tag='blur' 行，每键一条）。"""
    out = {}
    for r in rows:
        if isinstance(r.get("lam_clean"), (int, float)) and isinstance(r.get("E_hqs"), (int, float)):
            out[(r.get("arm"), r.get("img"), round(float(r.get("sn")), 6))] = r
    return out


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def subset_span(cells, thresh=THR):
    sel = [r["lam_clean"] for r in cells.values() if r["lam_clean"] >= thresh]
    if not sel:
        return 0, None, None, None
    return len(sel), min(sel), max(sel), max(sel) / min(sel)


def main():
    print("=" * 100)
    print(" 诊断：§3.9(bs) 要的正确核数据是否已存在（零 GPU，只读本地两个 JSON）")
    print("=" * 100)

    da, db = load(BUGGY), load(CORRECT)
    if da is None or db is None:
        print("  [缺少数据] %s 或 %s 不在本地 results/。" % (BUGGY, CORRECT))
        return

    ga = da["cfg"].get("kernel_guard")
    gb = db["cfg"].get("kernel_guard")
    print("\n----- 核来源核对（不过则本诊断无意义）-----")
    print("  %-18s kernel_guard = %s" % (BUGGY, "缺失（手写错核）" if ga is None else ga))
    print("  %-18s kernel_guard = %s" % (CORRECT, "缺失（手写错核）" if gb is None
                                         else "sym=%g peak=%s kunif=%.6f"
                                              % (gb.get("sym", -1), gb.get("peak_rc"),
                                                 gb.get("kunif", -1))))
    if gb is None or abs(gb.get("sym", -1)) > 1e-7 or list(gb.get("peak_rc", [])) != [7, 7]:
        print("  [中止] %s 不是正确核，逐格配对没有意义。" % CORRECT)
        return
    print("  → 两侧设计核对：%d vs %d 格" % (len(da["rows"]), len(db["rows"])))

    ca, cb = cells(da["rows"]), cells(db["rows"])
    common = sorted(set(ca) & set(cb))
    print("  错核 %d 格、正确核 %d 格；共同键 %d 个" % (len(ca), len(cb), len(common)))
    if set(ca) != set(cb):
        print("  [注意] 键集合不完全相同，配对只按交集算。")

    # ---------------- K1 逐格配对 ----------------
    d = [cb[k]["lam_clean"] - ca[k]["lam_clean"] for k in common]
    within = sum(1 for x in d if abs(x) <= HUTCH_DIFF)
    print("\n----- K1 逐格配对（只报数，不设通过线）-----")
    print("  ΔΛ_clean = 正确核 − 错核：n=%d  mean=%+.1f  范围 [%+.1f, %+.1f]"
          % (len(d), mean(d), min(d), max(d)))
    print("  |Δ| ≤ %g nats 差分底：%d/%d（%.1f%%）" % (HUTCH_DIFF, within, len(d),
                                                      100.0 * within / len(d)))

    # ---------------- K2 格数与跨度 ----------------
    print("\n----- K2 格数与跨度（预官阈值：与 %.3f× 差 >%.0f%% 则论文 §7 必须改）-----"
          % (PUB_SPAN, 100 * TOL))
    for name, c in ((BUGGY, ca), (CORRECT, cb)):
        n, lo, hi, sp = subset_span(c)
        print("  %-18s Λ_clean≥2000：%3d 格  跨度 %.4f×  [%.1f, %.1f]"
              % (name, n, sp, lo, hi))
    n_b, lo_b, hi_b, sp_b = subset_span(ca)
    n_c, lo_c, hi_c, sp_c = subset_span(cb)
    dev = (sp_c - PUB_SPAN) / PUB_SPAN
    print("  格数 %d → %d（预官基线 %d）" % (n_b, n_c, PUB_N))
    print("  跨度 %.4f× → %.4f×，相对预官基线 %+.2f%%" % (sp_b, sp_c, 100 * dev))
    if abs(dev) > TOL:
        print("  ⇒ K2 触发：论文 §7 的「13.0×」必须改成 %.1f×（%.0f–%.0f nats）。"
              % (sp_c, lo_c, hi_c))
    else:
        print("  ⇒ K2 通过：跨度变化在 %.0f%% 内，论文 §7 的「13.0×」可据实更新为 %.1f×"
              "（%.0f–%.0f nats）。" % (100 * TOL, sp_c, lo_c, hi_c))

    # ---------------- K3 诊断 ----------------
    print("\n----- K3 诊断（无阈值）-----")
    print("  逐 σ_n 的 mean|E_hqs/Λ_clean|（只在 |Λ_clean|≥500 nats 的格子上，同 (ak) L5 口径）：")
    ea, eb = with_ehqs(da["rows"]), with_ehqs(db["rows"])
    sns = sorted(set(k[2] for k in ea))
    print("    %-8s %14s %14s %8s" % ("σ_n", BUGGY, CORRECT, "n"))
    for sn in sns:
        va = [abs(r["E_hqs"] / r["lam_clean"]) for k, r in ea.items()
              if k[2] == sn and abs(r["lam_clean"]) >= 500]
        vb = [abs(r["E_hqs"] / r["lam_clean"]) for k, r in eb.items()
              if k[2] == sn and abs(r["lam_clean"]) >= 500]
        print("    %-8.4g %14.4f %14.4f %8d" % (sn, mean(va), mean(vb), len(vb)))
    va_all = [abs(r["E_hqs"] / r["lam_clean"]) for r in ea.values() if abs(r["lam_clean"]) >= 500]
    vb_all = [abs(r["E_hqs"] / r["lam_clean"]) for r in eb.values() if abs(r["lam_clean"]) >= 500]
    va_pool = [abs(r["E_hqs"] / r["lam_clean"]) for k, r in ea.items()
               if k[0] == "pool" and r["lam_clean"] >= THR]
    vb_pool = [abs(r["E_hqs"] / r["lam_clean"]) for k, r in eb.items()
               if k[0] == "pool" and r["lam_clean"] >= THR]
    print("    %-8s %14.4f %14.4f %8d   ← 两臂合计，|Λ_clean|≥500"
          % ("all", mean(va_all), mean(vb_all), len(vb_all)))
    print("    %-8s %14.4f %14.4f %8d   ← 仅 pool 臂且 Λ_clean≥2000，与 (ak) L5 同口径"
          "（该处发布值 1.1055±0.0080, n=67）"
          % ("pool", mean(va_pool), mean(vb_pool), len(vb_pool)))

    print("\n  恒等式自检 |pred − E_hqs|，pred = Δq − Λ_clean + (data_term − data)：")
    for name, c in ((BUGGY, ea), (CORRECT, eb)):
        err = [abs(r["dq"] - r["lam_clean"] + (r["data_term"] - r["data"]) - r["E_hqs"])
               for r in c.values()]
        print("    %-18s n=%3d  max=%.3e  mean=%.3e" % (name, len(err), max(err), mean(err)))


if __name__ == "__main__":
    main()
