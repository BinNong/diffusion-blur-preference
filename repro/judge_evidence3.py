"""裁判：第十四组 (aj) —— §3.9(z)/(ab) 的符号类论断在 20 张真实池图上重测

只读 results/evidence3.json + results/evidence_scale.json，零 GPU。
判据 K0–K6 全部预官在 repro/sweep_evidence3.py 的 docstring 里
（md5 34178383c57993bf8f8ed915ab381e0c，2026-10-23 14:40:13，早于主臂运行）。

用法： venv2/bin/python judge_evidence3.py [--in evidence3.json]
"""

import argparse
import json
import math
import os
import random

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_PROJ, "results")

# §3.2 的表（RESULTS 第 187–192 行）：12 张真实池图、另一个脚本、另一条代码路径。
# 索引是 t = σ_n，Λ_clean = q_t(x*) − q_t(h_1.5 ⋆ x*)。
REF32 = {0.01: 22210.0, 0.02: 13290.0, 0.05: 3530.0, 0.10: 1054.0, 0.20: 50.0}
NBOOT = 4000


def load(fn):
    with open(os.path.join(OUT, fn)) as fh:
        return json.load(fh)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def sd(v):
    if len(v) < 2:
        return float("nan")
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def se(v):
    return sd(v) / math.sqrt(len(v)) if len(v) > 1 else float("nan")


def skew(v):
    if len(v) < 3:
        return float("nan")
    m, s = mean(v), sd(v)
    if s <= 0:
        return float("nan")
    n = len(v)
    return (n / ((n - 1.0) * (n - 2.0))) * sum(((x - m) / s) ** 3 for x in v)


def med(v):
    s = sorted(v)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="fn", default="evidence3.json")
    args = ap.parse_args()
    d = load(args.fn)
    rows = d["rows"]
    cfg = d["cfg"]
    ref = load("evidence_scale.json")           # (z) 的原始行，K1 的参照
    refmap = {(r["img"], r["sn"], r["tag"]): r for r in ref["rows"]}

    pool = [r for r in rows if r["arm"] == "pool"]
    reg = [r for r in rows if r["arm"] == "reg"]
    pb = [r for r in pool if r["tag"] == "blur"]      # Λ_noisy(θ*=1.5)
    ps = [r for r in pool if r["tag"] == "sharp"]     # Λ_noisy(θ*=0)
    print(f"[cfg] nimg_pool={cfg['nimg_pool']} nimg_reg={cfg['nimg_reg']} "
          f"θ*={cfg['tstar']} σ={cfg['noises']} n_inner={cfg['n_inner']}")
    print(f"[rows] 总 {len(rows)}（pool {len(pool)} = blur {len(pb)} + sharp {len(ps)}；"
          f"reg {len(reg)}）")

    # ------------------------------------------------------------------ K0
    print("\n===== K0 守门：E_y = −(Λ_noisy + data) 恒等式 =====")
    worst = 0.0
    for r in rows:
        e = abs(r["E_y"] + r["lam"] + r["data"]) / max(1.0, abs(r["E_y"]))
        worst = max(worst, e)
    print(f"  两臂 {len(rows)} 行，最大相对残差 = {worst:.3e}   "
          f"{'✅ ≤1e−6' if worst <= 1e-6 else '❌ 全废'}")
    worst3 = 0.0
    for r in pb:
        if "E_hqs" not in r:
            continue
        e = abs(r["E_hqs"] - (r["dq"] - r["lam_clean"]
                              + (r["data_term"] - r["data"]))) \
            / max(1.0, abs(r["E_hqs"]))
        worst3 = max(worst3, e)
    print(f"  三项恒等式 E_hqs = Δq − Λ_clean + (data_term − data)："
          f"最大相对残差 = {worst3:.3e}（(ab) 报 1.6e−14）")

    # ------------------------------------------------------------------ K1
    print("\n===== K1 回归臂：3 张先验样本须复现 (z) 到 2% 内 =====")
    bad = []
    print(f"  {'img':>3} {'σ':>5} {'tag':>6} {'量':>8} {'(z)':>11} {'本组':>11} {'相对差':>9}")
    for r in reg:
        k = (r["img"], r["sn"], r["tag"])
        if k not in refmap:
            continue
        o = refmap[k]
        for f in ("lam", "lam_clean", "E_y"):
            a, b = o[f], r[f]
            rel = abs(b - a) / max(1.0, abs(a))
            flag = "" if rel <= 0.02 else "  ← 超 2%"
            if rel > 0.02:
                bad.append((k, f, a, b, rel))
            if f != "lam_clean" or r["tag"] == "blur":
                print(f"  {r['img']:>3} {r['sn']:>5g} {r['tag']:>6} {f:>8} "
                      f"{a:>11.1f} {b:>11.1f} {rel:>8.2%}{flag}")
        if "E_hqs" in r and "E_hqs" in o:
            a, b = o["E_hqs"], r["E_hqs"]
            rel = abs(b - a) / max(1.0, abs(a))
            print(f"  {r['img']:>3} {r['sn']:>5g} {r['tag']:>6} {'E_hqs':>8} "
                  f"{a:>11.1f} {b:>11.1f} {rel:>8.2%}"
                  f"{'  ← 超 2%（(ab4) 已测定 E_hqs 数值地板 1–2%）' if rel > 0.02 else ''}")
    print(f"  ⇒ {'✅ 全部 ≤2%（除可能豁免的 E_hqs）' if not bad else '❌ 超差项：' + str(bad)}")

    # ------------------------------------------------------------------ K2
    print("\n===== K2 跨实现复现（**已修正，见下方披露**）：同图同核、不同脚本 =====")
    # 参照：results/sweep_lambda_theta.json 的 gaussian σ=1.5 记录。
    #   * 12 张图 = "first nimg of 256-pool seed0"，即 pool[:12]，是本组 pool[:20] 的子集；
    #   * 核逐字相同（putils GaussianBlur：linspace(-6,7,15)、exp(-grid/2σ²) 归一化、KS=15）；
    #   * 同 ckpt、同 t_spans=[80,t]、Λ = logp(模糊) − logp(清晰)，与本组 Λ_clean 同定义；
    #   * 只有脚本与批组成不同 ⇒ 这是一次真正的**跨实现**复现，不是同代码自说自话。
    lt = load("sweep_lambda_theta.json")
    ltmap = {(r["family"], round(r["theta"], 6), r["t"]): r
             for r in lt["records"]}
    nsub = min(12, cfg["nimg_pool"])
    print(f"  用 pool[:{nsub}]（本组 {cfg['nimg_pool']} 图的子集）对 "
          f"sweep_lambda_theta.json 的 gaussian σ=1.5 记录；容差 2%（(ab4) 测定的数值地板）")
    print(f"  {'t':>5} {'参照(12图)':>11} {'本组(子集)':>11} {'相对差':>8}  判定")
    n_ok2, det32 = 0, []
    for tt in (0.02, 0.05):
        r = ltmap.get(("gaussian", 1.5, tt))
        v = [x["lam_clean"] for x in pb if x["sn"] == tt and x["img"] < nsub]
        if r is None or not v:
            continue
        m = mean(v)
        rel = (m - r["lam_mean"]) / r["lam_mean"]
        ok = abs(rel) <= 0.02
        n_ok2 += ok
        print(f"  {tt:>5g} {r['lam_mean']:>11.1f} {m:>11.1f} {rel:>+8.2%}  "
              f"{'✅' if ok else '❌'}  (n={len(v)})")
    print(f"  ⇒ {n_ok2}/2 档在 2% 内")
    print("  **披露（预官后、主臂运行前）**：预官里把 K2 的参照写成「§3.2 是 12 张真实图」，"
          "这是错的——\n    §3.2 的表是**单张配对先验样本**（§3.2 标题「单张配对图」、§3.4「本张图 "
          "t=0.002 处为 +33898」）。\n    单图参照不是可用的容差：它在 t=0.05 上与 12 图均值差 "
          "26%（3530 vs 4440）。故把 K2 改成上面的跨实现复现\n    （**加强**而非放松：参照从 1 张图"
          "换成 12 张同池真实图、容差从 15% 收到 2%），§3.2 的单图表降为 K6 诊断。")
    print("  **修改时已可见什么（如实记录）**：冒烟测试（--smoke，nimg=2，仅代码验证用途）的"
          "**两臂**都已有输出——\n    回归臂 img0/img1 全部 10 格（已 4 位有效数字复现 (z)），"
          "以及主臂 img0 全部 5 格、img1 部分的 Λ_clean\n    （img0：23691/15412/5172/1746/462）。"
          "**20 图主臂尚未开始**。修改的动因不是这些冒烟值，而是\n    (i) 我自己预官里的参照出处"
          "写错了，(ii) 已发布文件 sweep_lambda_theta.json 显示单图参照在 t=0.05 上偏离 12 图均值 "
          "26%。\n    但读者无法核验动机，故按实际可见量披露；若认为该暴露已污染 K2，K2 应整条作废，"
          "此时请以 K1（同 seed 逐字复现 (z)）作为实现自检。")
    print("  §3.2 单图表（降为诊断，不设阈值）：")
    for sn in cfg["noises"]:
        v = [r["lam_clean"] for r in pb if r["sn"] == sn]
        if not v:
            continue
        m = mean(v)
        rel = (m - REF32[sn]) / REF32[sn]
        det32.append((sn, m, rel))
        print(f"    σ={sn:<5g} §3.2(单图)={REF32[sn]:>8.0f}  本组20图均值={m:>8.0f}  "
              f"{rel:>+8.1%}")
    print(f"    ⇒ {sum(1 for _, _, r in det32 if abs(r) <= 0.15)}/"
          f"{len(det32)} 档在 ±15% 内（(z) 在 3 张先验样本上报 4/5）——"
          f"这正是「单图参照不可用」的量化")

    # ------------------------------------------------------------------ K3
    print("\n===== K3 主判据一：模糊偏好不随观测噪声存活（Λ_noisy < 0 的比例）=====")
    imgs = sorted({r["img"] for r in pb})

    def frac_neg(rs):
        return (sum(1 for r in rs if r["lam"] < 0) / len(rs)) if rs else float("nan")

    def boot_frac(rs, key="lam"):
        """按图像整簇重采样（σ 档在同一张图内相关，不能当独立单元）。"""
        by = {}
        for r in rs:
            by.setdefault(r["img"], []).append(r)
        ks = list(by)
        out = []
        rnd = random.Random(12345)
        for _ in range(NBOOT):
            pick = [by[rnd.choice(ks)] for _ in ks]
            flat = [x for g in pick for x in g]
            out.append(sum(1 for x in flat if x[key] < 0) / len(flat))
        out.sort()
        return out[int(0.025 * NBOOT)], out[int(0.975 * NBOOT)]

    for tag, sel, line in (("K3a 全部 100 对", pb, 0.85),
                           ("K3b σ≥0.02", [r for r in pb if r["sn"] >= 0.02], 0.95),
                           ("K3c σ=0.01（只记录）", [r for r in pb if r["sn"] == 0.01], None)):
        v = frac_neg(sel)
        lo, hi = boot_frac(sel)
        if line is None:
            print(f"  {tag}: {v:.3f}（{sum(1 for r in sel if r['lam']<0)}/"
                  f"{len(sel)}）  95% 聚类自助 CI [{lo:.3f}, {hi:.3f}]  只记录不判")
        else:
            ok = v >= line
            print(f"  {tag}: {v:.3f}（{sum(1 for r in sel if r['lam']<0)}/{len(sel)}）  "
                  f"95% 聚类自助 CI [{lo:.3f}, {hi:.3f}]  通过线 {line:.2f}  "
                  f"{'✅' if ok else '❌'}")
    print(f"  (z) 的原始值：全部 0.933（14/15）、σ≥0.02 为 1.000（12/12）、σ=0.01 为 0.667（2/3）")

    print("\n  —— 估计量脆弱性（预官要求必须报）——")
    print(f"  {'img':>3} {'负值':>5} {'mean/std':>9} {'偏度':>7} {'中位数':>9} {'均值':>9} "
          f"{'中位/均值':>9} {'σ剖面':>18}")
    for i in imgs:
        v = [r["lam"] for r in pb if r["img"] == i]
        seq = [r["lam"] for r in sorted([r for r in pb if r["img"] == i],
                                        key=lambda r: r["sn"])]
        mono = "单调不增" if all(seq[j] >= seq[j + 1] for j in range(len(seq) - 1)) \
            else "非单调"
        neg = sum(1 for x in v if x < 0)
        print(f"  {i:>3} {neg:>4}/{len(v)} {mean(v)/sd(v):>9.2f} {skew(v):>7.2f} "
              f"{med(v):>9.0f} {mean(v):>9.0f} {med(v)/mean(v):>9.3f}  {mono:>18}")
    allv = [r["lam"] for r in pb]
    print(f"  合计：负值 {sum(1 for x in allv if x<0)}/{len(allv)}、"
          f"mean/std={mean(allv)/sd(allv):.2f}、偏度={skew(allv):.2f}、"
          f"中位数/均值={med(allv)/mean(allv):.3f}")

    print("\n===== K3d 伴生判据：Λ_noisy(θ*=1.5) > Λ_noisy(θ*=0) 的比例 =====")
    smap = {(r["img"], r["sn"]): r["lam"] for r in ps}
    pairs = [(r, smap[(r["img"], r["sn"])]) for r in pb if (r["img"], r["sn"]) in smap]
    n_ord = sum(1 for a, b in pairs if a["lam"] > b)
    v = n_ord / len(pairs)
    byi = {}
    for a, b in pairs:
        byi.setdefault(a["img"], []).append(1.0 if a["lam"] > b else 0.0)
    ks = list(byi)
    rnd = random.Random(12346)
    bo = []
    for _ in range(NBOOT):
        pick = [byi[rnd.choice(ks)] for _ in ks]
        flat = [x for g in pick for x in g]
        bo.append(sum(flat) / len(flat))
    bo.sort()
    print(f"  {n_ord}/{len(pairs)} = {v:.3f}  95% 聚类自助 CI "
          f"[{bo[int(0.025*NBOOT)]:.3f}, {bo[int(0.975*NBOOT)]:.3f}]  通过线 0.95  "
          f"{'✅' if v >= 0.95 else '❌'}   ((z) 报 15/15)")

    # ------------------------------------------------------------------ K4
    print("\n===== K4 主判据二：Lemma 12 的成立与否由 Λ_clean 的符号决定 =====")
    cells = [r for r in pb if "E_hqs" in r]
    good = [r for r in cells if (r["lam_clean"] > 0) == (r["E_hqs"] < 0)]
    neg_lc = [r for r in cells if r["lam_clean"] < 0]
    neg_ok = [r for r in neg_lc if r["E_hqs"] > 0]
    v = len(good) / len(cells)
    byi = {}
    for r in cells:
        byi.setdefault(r["img"], []).append(
            1.0 if (r["lam_clean"] > 0) == (r["E_hqs"] < 0) else 0.0)
    ks = list(byi)
    rnd = random.Random(12347)
    bo = []
    for _ in range(NBOOT):
        pick = [byi[rnd.choice(ks)] for _ in ks]
        flat = [x for g in pick for x in g]
        bo.append(sum(flat) / len(flat))
    bo.sort()
    print(f"  对应 {len(good)}/{len(cells)} = {v:.3f}  95% 聚类自助 CI "
          f"[{bo[int(0.025*NBOOT)]:.3f}, {bo[int(0.975*NBOOT)]:.3f}]  "
          f"通过线 0.95、否掉线 0.85  "
          f"{'✅' if v >= 0.95 else ('❌ 否掉' if v < 0.85 else '⚠️ 灰区')}")
    print(f"  反向半边：Λ_clean<0 的格子 {len(neg_lc)} 个，其中 E_hqs>0 的 "
          f"{len(neg_ok)} 个（(z) 是 1/1；数量少，只报不设阈值）")
    for r in neg_lc:
        print(f"    img{r['img']} σ={r['sn']:<5g} Λ_cl={r['lam_clean']:>+9.1f} "
              f"E_hqs={r['E_hqs']:>+9.1f}")

    # ------------------------------------------------------------------ K5
    print("\n===== K5 主判据三：Λ_clean≥2000 nats 上 mean(|E_hqs|/Λ_clean) =====")
    sel = [r for r in cells if r["lam_clean"] >= 2000.0]
    rs = [r["ratio"] for r in sel]
    m, s = mean(rs), se(rs)
    byi = {}
    for r in sel:
        byi.setdefault(r["img"], []).append(r["ratio"])
    ks = list(byi)
    rnd = random.Random(12348)
    bo = []
    for _ in range(NBOOT):
        pick = [byi[rnd.choice(ks)] for _ in ks]
        flat = [x for g in pick for x in g]
        bo.append(mean(flat))
    bo.sort()
    ok = (1.00 <= m <= 1.25) and s <= 0.03
    print(f"  n={len(rs)}（{len(ks)} 张图），Λ_clean 跨 "
          f"{min(r['lam_clean'] for r in sel):.0f}–{max(r['lam_clean'] for r in sel):.0f} "
          f"（{max(r['lam_clean'] for r in sel)/min(r['lam_clean'] for r in sel):.1f} 倍）")
    print(f"  mean = {m:.4f}、SE = {s:.4f}  95% 聚类自助 CI "
          f"[{bo[int(0.025*NBOOT)]:.4f}, {bo[int(0.975*NBOOT)]:.4f}]")
    print(f"  与 (ab) 的 W2 同一条预谱带宽：mean ∈ [1.00, 1.25] 且 SE ≤ 0.03  "
          f"{'✅' if ok else '❌'}   ((ab) 报 1.1057 ± 0.0125，n=76)")

    # ------------------------------------------------------------------ K6
    print("\n===== K6 诊断（无阈值）=====")
    print("  (a) |E_hqs|/Λ_clean 随 Λ_clean 的走向（按 Λ_clean 分箱）")
    bnds = [(0, 500), (500, 2000), (2000, 8000), (8000, 20000), (20000, 1e9)]
    for lo, hi in bnds:
        g = [r for r in cells if lo <= r["lam_clean"] < hi]
        if not g:
            continue
        rr = [r["ratio"] for r in g]
        print(f"    Λ_cl ∈ [{lo:>6g},{hi if hi < 1e8 else 999999:>6g})  n={len(rr):>3}  "
              f"mean 比值={mean(rr):>7.3f}  范围 [{min(rr):.3f},{max(rr):.3f}]  "
              f"负 E_hqs {sum(1 for r in g if r['E_hqs']<0)}/{len(g)}")
    print("  (b) 三项恒等式的两个修正项")
    dqr = [r["dq"] / r["lam_clean"] for r in sel if r["lam_clean"]]
    dtr = [r["data_term"] / r["data"] for r in sel]
    dvc = [r["data"] / r["lam_clean"] for r in sel if r["lam_clean"]]
    print(f"    −Δq/Λ_clean：mean={mean(dqr):+.4f}  范围 "
          f"[{min(dqr):+.4f},{max(dqr):+.4f}]  ((ab) 在 θ*=1.5 上 +0.0030)")
    print(f"    data_term/data：mean={mean(dtr):.4f}  范围 "
          f"[{min(dtr):.3f},{max(dtr):.3f}]  ((ab) 报 [0.71,1.03])")
    print(f"    data/Λ_clean：mean={mean(dvc):.4f}  范围 "
          f"[{min(dvc):.4f},{max(dvc):.4f}]")
    print("  (c) 图源效应：20 张真实池图 vs 先验样本")
    for nm, g in (("pool(20图)", pb), ("reg(先验样本)", [r for r in reg if r["tag"] == "blur"])):
        if not g:
            continue
        lc = [r["lam_clean"] for r in g]
        lm = [r["lam"] for r in g]
        print(f"    {nm:<14} Λ_clean mean={mean(lc):>8.0f} 范围 "
              f"[{min(lc):.0f},{max(lc):.0f}]；Λ_noisy mean={mean(lm):>8.0f} "
              f"负值 {sum(1 for x in lm if x<0)}/{len(lm)}")
    print("  (d) data 项的尺度不变性（应恒为 N/2=6144 附近）")
    dv = sorted({round(r["data"], 1) for r in rows})
    print(f"    data 取值 {len(dv)} 个：{dv[0]:.1f} … {dv[-1]:.1f}")

    print("\n===== 汇总 =====")
    print(f"  K0 恒等式 {'✅' if worst <= 1e-6 else '❌'}（{worst:.2e}）；"
          f"三项恒等式 {worst3:.2e}")
    print(f"  K1 回归臂 {'✅' if not bad else '❌'}")
    print(f"  K2 跨实现复现 {n_ok2}/2（§3.2 单图表 {sum(1 for _,_,r in det32 if abs(r)<=0.15)}"
          f"/{len(det32)} 在 ±15% 内，仅诊断）")
    print(f"  K3a {frac_neg(pb):.3f} / K3b "
          f"{frac_neg([r for r in pb if r['sn']>=0.02]):.3f} / "
          f"K3c {frac_neg([r for r in pb if r['sn']==0.01]):.3f} / "
          f"K3d {n_ord}/{len(pairs)}")
    print(f"  K4 符号对应 {len(good)}/{len(cells)}")
    print(f"  K5 mean 比值 {m:.4f} ± {s:.4f}（n={len(rs)}）")


if __name__ == "__main__":
    main()
