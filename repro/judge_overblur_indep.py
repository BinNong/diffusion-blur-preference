# -*- coding: utf-8 -*-
r"""第十九组预官裁判：过模糊机理的独立复现（零 GPU，纯标准库）

预官时间：2026-11-08 16:41:32 +0800（在看到任何 `overblur_indep.json` 之前）
判据原文与 md5 记录在 RESULTS §3.9(bz)；本 docstring 是同一份判据的权威副本。
**本裁判在跑之前写好；试跑时目标数据尚不存在，必须报「缺少数据」并退出。**

================================================================================
为什么有这一组
================================================================================
§3.9(by) 用 `phaseC_pool20.json` 自带的 L/data/q 分解，零 GPU 拿到了一个过模糊机理的
**定量候选**：数据项惩罚按 sigma^-1.355 衰减、先验项在 sigma in (0.0845, 0.172) 符号
翻转、反事实拿掉先验只剩 1.7% 过模糊。但 (by) 是 **post-hoc**——设计它时数据和
N0–N4 判定都已可见，所以它不能进论文。本组把同一套读数搬到**尚未产生的独立数据**上，
判据跑前写定，是把机理升格为论文结论的唯一路径。

================================================================================
全额披露：写判据时已经可见的东西
================================================================================
`phaseC_pool20.json` 的全部 120 条 profile、(bt) 的 N0–N4 判定、(by) 的全部读数与
配对 bootstrap SE、论文 §8 的现有文字、以及 §3.9(z) 的 theta=0 陷阱机制。
**本组判据针对的是尚未产生的独立数据**；所有阈值由已发布的 (by) 基线残差分布
（现场重算，固定 seed）推出，不是从新数据里挑的，也不是硬编码容差。

================================================================================
数据契约（新文件 results/overblur_indep.json）
================================================================================
cfg  : nimg=20；pool_offset>=20 或 image_source 明确为池图第 20–39 张；tstar=1.5；
       thetas 为 (bt) 的 21 点网格；noises 为 (bt) 的 6 档；kernel_guard 存在。
records: 每条含 arm / img / noise / theta_star / profile[{theta, L, data, q}]。
实现规格（待在服务器实现，不得改动本判据）：复制 `repro_phaseC_pool20.py`，
只改三处 —— ①池图切片 `pool[:NIMG]` -> `pool[20:40]`；②`cfg` 增加 `pool_offset=20`；
③默认 `--out overblur_indep.json`。其余（正确核、TSTAR=1.5、21 点网格、6 噪声档、
N_INNER=240、HQS 两初值、L 的定义与落盘字段）逐字不变。
成本：120 条 profile × 7.6 min ≈ 16 GPU 小时，与 (bt) 同量级。

================================================================================
判据 O0–O6（跑前写定；容差全部现场由基线 bootstrap 给出）
================================================================================
O0 守门（任一不过则中止，一个数字都不报）：
  (a) 恒等式 max|L - (data + q)| <= 1e-6；
  (b) kernel_guard 通过（对称、峰 [7,7]、bitident、kunif(1.5) 与 0.962855 一致到 1e-4）；
  (c) 图池独立性：cfg.pool_offset>=20，或新 img 集合与基线 img 集合无交集；
  (d) 噪声档、theta 网格、tstar 与基线逐字相同。

O1 决定性｜先验项是必要条件：
  反事实（只保留 data 曲线求 argmin）的过模糊比例 <= cf_base + 2*SE_cf，
  且 (真实过模糊比例 - 反事实比例) >= 2*SE_diff。
  失败后果：「先验项是必要条件」不成立，机理段整体撤回，不报最接近的通过变体。

O2 决定性｜先验项符号翻转：
  sigma<=0.0845 的档 mean(dq) > +2*SE_dq(该档)；
  sigma>=0.172 的档 mean(dq) < -2*SE_dq(该档)；
  且 dq<0 的第一个噪声档必须是 0.172。
  **sigma=0.0845 一格声明 RECORD ONLY**：基线 dq=+1218.8、2*SE=959.0，
  预言边际只有 259.8 nats = 0.54 SE，按 CLAUDE.md「边际须超过 2 sigma」不足，
  故该格只报数、不计入通过数。
  失败后果：翻转点若移动超过一个噪声档，「翻转与高噪声阈值重合」判为巧合，机理降级。

O3 决定性｜衰减指数：新数据拟合的 p 落在
  [p_base - 2*SE_p, p_base + 2*SE_p]，p 由 log(mean dData) 对 log(sigma) 回归得到。
  同时报告纯 1/sigma^2 的 p=2 与该区间的距离（诊断，不设阈值）。
  失败后果：「不是纯 1/sigma^2」不成立。

O4 零判据｜H0「dq 与 log(sigma) 无关」：
  报 dq 对 log(sigma) 的回归斜率、其 bootstrap SE，并把 **2*SE 作为可探测下限**
  与判定一起打印。|slope| <= 2*SE 时判「不可分辨」，不得表述为「通过」。

O5 纯诊断（无阈值）：sign(dL[1.5->2.5]) 预测 argmin>theta* 的命中率与混淆矩阵、
各档 mean argmin、dL 幅度分布。

O6 纯诊断（无阈值）：dq 在高噪声档（sigma>=0.2455）的饱和均值与基线之差；
若新数据额外带 theta*=2.0 臂，报告翻转档位是否随 theta* 移动。

================================================================================
失败后果总表与修订规则
================================================================================
O1/O2/O3 任一失败 => 对应机理陈述不得进论文，并在 RESULTS 记录失败原值。
O4 若「不可分辨」=> 先验项的噪声依赖性无法与常数模型区分，机理降级为描述性。
修订只允许在 unblinding 之前，必须显式披露新 md5、修订时已可见的内容与修订理由。
判据阈值 O1–O4 全部由基线现场 bootstrap 重算（seed=20261108, B=2000），不得改 seed。

用法：
  python3 repro/judge_overblur_indep.py
输出：results/fig_overblur_indep.json（仅在目标数据存在且 O0 通过时写出）
"""

import json
import math
import os
import random
import statistics as st

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
BASE = "phaseC_pool20.json"
NEW = "overblur_indep.json"
DST = "fig_overblur_indep.json"
BOOT_B = 2000
BOOT_SEED = 20261108
HI = 0.172
TSTAR = 1.5
TEND = 2.5


def prof(rec, key):
    return {p["theta"]: p[key] for p in rec["profile"]}


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def sd(v):
    return st.pstdev(v) if len(v) > 1 else 0.0


def slope_vs_logsigma(xs, ys):
    lx = [math.log(x) for x in xs]
    mx, my = mean(lx), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(lx, ys)) / sum((x - mx) ** 2 for x in lx)


def per_cell(recs, ths, sns, imgs):
    """返回 (i, s) -> (ddata, dq, dL, cf_over, true_over, sign_hit)。"""
    by = {(r["img"], r["noise"]): r for r in recs}
    q = {}
    for i in imgs:
        for s in sns:
            r = by[(i, s)]
            L, D, Q = prof(r, "L"), prof(r, "data"), prof(r, "q")
            am_l = min(ths, key=lambda t: L[t])
            am_d = min(ths, key=lambda t: D[t])
            q[(i, s)] = (D[TEND] - D[TSTAR], Q[TEND] - Q[TSTAR], L[TEND] - L[TSTAR],
                         1 if am_d > TSTAR else 0,
                         1 if am_l > TSTAR else 0,
                         1 if ((L[TEND] - L[TSTAR]) < 0) == (am_l > TSTAR) else 0)
    return q


def stats_for(q, idx, sns):
    dd = [mean(q[(i, s)][0] for i in idx) for s in sns]
    dq = [mean(q[(i, s)][1] for i in idx) for s in sns]
    cf = mean(q[(i, s)][3] for i in idx for s in sns)
    tr = mean(q[(i, s)][4] for i in idx for s in sns)
    hit = mean(q[(i, s)][5] for i in idx for s in sns)
    p = -slope_vs_logsigma(sns, [math.log(v) for v in dd])
    sq = slope_vs_logsigma(sns, dq)
    return dict(p=p, dq=dq, dd=dd, cf=cf, true=tr, hit=hit, slope_dq=sq,
                diff=tr - cf)


def baseline_se(base_recs, ths, sns, imgs):
    q = per_cell(base_recs, ths, sns, imgs)
    base = stats_for(q, imgs, sns)
    random.seed(BOOT_SEED)
    keys = ["p", "cf", "diff", "hit", "slope_dq"]
    draws = {k: [] for k in keys}
    draws["dq"] = [[] for _ in sns]
    for _ in range(BOOT_B):
        idx = [random.choice(imgs) for _ in imgs]
        s = stats_for(q, idx, sns)
        for k in keys:
            draws[k].append(s[k])
        for j, v in enumerate(s["dq"]):
            draws["dq"][j].append(v)
    se = {k: sd(draws[k]) for k in keys}
    se["dq"] = [sd(v) for v in draws["dq"]]
    return base, se


def main():
    newpath = os.path.join(OUT, NEW)
    if not os.path.exists(newpath):
        print("  [缺少数据] %s 尚未落地——预官窗口仍然闭合，未计算任何新数据的数字。" % NEW)
        return

    base_d = json.load(open(os.path.join(OUT, BASE)))
    new_d = json.load(open(newpath))
    brecs, nrecs = base_d["records"], new_d["records"]
    ths = base_d["cfg"]["thetas"]
    sns = sorted(set(r["noise"] for r in brecs))
    bimgs = sorted(set(r["img"] for r in brecs))
    nimgs = sorted(set(r["img"] for r in nrecs))

    print("=" * 96)
    print(" 第十九组：过模糊机理的独立复现（预官 2026-11-08 16:41:32 +0800）")
    print("=" * 96)

    # ---------------- O0 ----------------
    ident = max(abs(p["L"] - p["data"] - p["q"]) for r in nrecs for p in r["profile"])
    g = new_d["cfg"].get("kernel_guard", {})
    guard_ok = (g.get("sym_correct", 9) <= 1e-7 and g.get("peak_correct") == [7, 7]
                and g.get("bitident_correct") is True
                and abs(g.get("kunif_correct_15", 0) - 0.962855) < 1e-4)
    offset = new_d["cfg"].get("pool_offset", 0)
    indep_ok = offset >= 20 or (not set(nimgs) & set(bimgs))
    grid_ok = (new_d["cfg"].get("thetas") == ths
               and sorted(set(r["noise"] for r in nrecs)) == sns
               and all(r.get("theta_star", TSTAR) == TSTAR for r in nrecs))
    print("  O0 (a) 恒等式 max|L-(data+q)| = %.3e" % ident)
    print("  O0 (b) 核守卫 = %s   (c) 图池独立 = %s (offset=%s)   (d) 网格一致 = %s"
          % (guard_ok, indep_ok, offset, grid_ok))
    if not (ident <= 1e-6 and guard_ok and indep_ok and grid_ok):
        print("  => O0 FAIL：中止，一个数字都不报。")
        return
    print("  => O0 PASS\n")

    # ---------------- 容差：现场由基线 bootstrap 给出 ----------------
    base, se = baseline_se(brecs, ths, sns, bimgs)
    print("  基线（已发布 (by)，20 图配对 bootstrap seed=%d B=%d）：" % (BOOT_SEED, BOOT_B))
    print("    p=%.4f SE=%.4f | cf=%.4f SE=%.4f | diff=%.4f SE=%.4f | dq斜率=%.1f SE=%.1f"
          % (base["p"], se["p"], base["cf"], se["cf"], base["diff"], se["diff"],
             base["slope_dq"], se["slope_dq"]))
    for j, s in enumerate(sns):
        print("    sigma=%.4f  dq=%+8.1f  2SE=%7.1f" % (s, base["dq"][j], 2 * se["dq"][j]))
    print("")

    q = per_cell(nrecs, ths, sns, nimgs)
    new = stats_for(q, nimgs, sns)
    verdicts = {}

    # ---------------- O1 ----------------
    lim_cf = base["cf"] + 2 * se["cf"]
    lim_diff = 2 * se["diff"]
    o1 = (new["cf"] <= lim_cf) and (new["diff"] >= lim_diff)
    verdicts["O1"] = o1
    print("  O1 反事实比例 %.4f (上限 %.4f)；真实−反事实 %.4f (下限 %.4f) => %s"
          % (new["cf"], lim_cf, new["diff"], lim_diff, "PASS" if o1 else "FAIL"))

    # ---------------- O2 ----------------
    o2_cells, first_neg = [], None
    for j, s in enumerate(sns):
        thresh = 2 * se["dq"][j]
        v = new["dq"][j]
        if v < 0 and first_neg is None:
            first_neg = s
        if s == 0.0845:
            tag = "RECORD ONLY"
        elif s < HI:
            tag = "PASS" if v > thresh else "FAIL"
        else:
            tag = "PASS" if v < -thresh else "FAIL"
        o2_cells.append(dict(sigma_n=s, dq=v, two_se=thresh, verdict=tag))
        print("  O2 sigma=%.4f  dq=%+8.1f  2SE=%7.1f  => %s" % (s, v, thresh, tag))
    scored = [c for c in o2_cells if c["verdict"] != "RECORD ONLY"]
    o2 = all(c["verdict"] == "PASS" for c in scored) and first_neg == HI
    verdicts["O2"] = o2
    print("  O2 首个负档 = %s (要求 %s) => %s" % (first_neg, HI, "PASS" if o2 else "FAIL"))

    # ---------------- O3 ----------------
    lo, hi = base["p"] - 2 * se["p"], base["p"] + 2 * se["p"]
    o3 = lo <= new["p"] <= hi
    verdicts["O3"] = o3
    print("  O3 p=%.4f  预官区间 [%.4f, %.4f]  p=2 距上端 %.4f => %s"
          % (new["p"], lo, hi, 2 - hi, "PASS" if o3 else "FAIL"))

    # ---------------- O4 ----------------
    floor = 2 * se["slope_dq"]
    o4 = abs(new["slope_dq"]) > floor
    verdicts["O4"] = o4
    print("  O4 零判据 dq~log(sigma) 斜率=%+.1f  可探测下限(2SE)=%.1f => %s"
          % (new["slope_dq"], floor,
             "拒绝零假设" if o4 else "不可分辨（不得表述为通过）"))

    # ---------------- O5 / O6 ----------------
    conf = dict(tp=0, fp=0, tn=0, fn=0)
    for (i, s), row in q.items():
        actual, pred = row[4], (row[2] < 0)
        conf[("tp" if actual else "fp") if pred else ("fn" if actual else "tn")] += 1
    by_new = {(r["img"], r["noise"]): r for r in nrecs}
    am = {(i, s): min(ths, key=lambda t: prof(by_new[(i, s)], "L")[t])
          for i in nimgs for s in sns}
    mean_argmin = {str(s): mean(am[(i, s)] for i in nimgs) for s in sns}
    sat = mean(new["dq"][j] for j, s in enumerate(sns) if s >= 0.2455)
    sat_base = mean(base["dq"][j] for j, s in enumerate(sns) if s >= 0.2455)
    print("  O5 sign(dL) 命中 %.4f，混淆 %s" % (new["hit"], conf))
    print("  O5 各档 mean argmin：%s" % {k: round(v, 3) for k, v in mean_argmin.items()})
    print("  O6 高噪声 dq 饱和：新 %+.1f  基线 %+.1f  差 %+.1f（诊断，无阈值）"
          % (sat, sat_base, sat - sat_base))

    res = dict(kind="pre-registered O0-O6, baseline-derived tolerances",
               baseline=dict(p=base["p"], se_p=se["p"], cf=base["cf"], se_cf=se["cf"],
                             diff=base["diff"], se_diff=se["diff"],
                             slope_dq=base["slope_dq"], se_slope_dq=se["slope_dq"],
                             dq=base["dq"], two_se_dq=[2 * x for x in se["dq"]]),
               new=dict(p=new["p"], cf=new["cf"], diff=new["diff"], hit=new["hit"],
                        dq=new["dq"], dd=new["dd"], slope_dq=new["slope_dq"],
                        mean_argmin=mean_argmin, saturation_dq=sat),
               o2_cells=o2_cells, o4_detectable_floor=floor,
               o5_confusion=conf, verdicts=verdicts)
    with open(os.path.join(OUT, DST), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("\n  汇总：%s" % " | ".join("%s %s" % (k, "PASS" if v else "FAIL")
                                   for k, v in verdicts.items()))
    print("  [saved] %s" % os.path.join(OUT, DST))


if __name__ == "__main__":
    main()
