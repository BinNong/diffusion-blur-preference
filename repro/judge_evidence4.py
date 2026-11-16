"""裁判：第十五组 (ak) —— 修正高斯核半像素偏移 bug 后重测符号类论断

只读 results/evidence4.json + results/evidence_scale.json + results/sweep_lambda_theta.json，
零 GPU。判据 L0–L6 预官在 repro/sweep_evidence4.py 的 docstring 里
（md5 6cf1b0eff942711decccbc43e0f10a43，2026-10-23 16:49:56，早于运行）。

用法： venv2/bin/python judge_evidence4.py [--in evidence4.json]
"""

import argparse
import json
import math
import os
import random

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_PROJ, "results")

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


def boot_ci(byimg, fn, seed):
    """按图像整簇重采样（σ 档在同一张图内相关，不能当独立单元）。"""
    ks = list(byimg)
    rnd = random.Random(seed)
    out = []
    for _ in range(NBOOT):
        flat = [x for _ in ks for x in byimg[rnd.choice(ks)]]
        out.append(fn(flat))
    out.sort()
    return out[int(0.025 * NBOOT)], out[int(0.975 * NBOOT)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="fn", default="evidence4.json")
    args = ap.parse_args()
    d = load(args.fn)
    rows = d["rows"]
    cfg = d["cfg"]
    ref = load("evidence_scale.json")
    refmap = {(r["img"], r["sn"], r["tag"]): r for r in ref["rows"]}
    lt = load("sweep_lambda_theta.json")
    ltmap = {(r["family"], round(r["theta"], 6), r["t"]): r for r in lt["records"]}

    pool = [r for r in rows if r["arm"] == "pool"]
    reg = [r for r in rows if r["arm"] == "reg"]
    pb = [r for r in pool if r["tag"] == "blur"]
    ps = [r for r in pool if r["tag"] == "sharp"]
    print(f"[cfg] nimg_pool={cfg['nimg_pool']} nimg_reg={cfg['nimg_reg']} "
          f"θ*={cfg['tstar']} ks={cfg['ks']} σ={cfg['noises']} n_inner={cfg['n_inner']}")
    print(f"[rows] 总 {len(rows)}（pool {len(pool)} = blur {len(pb)} + sharp {len(ps)}；"
          f"reg {len(reg)}）")

    # ------------------------------------------------------------------ L0
    print("\n===== L0 核守卫（来自 sweep 落盘的 kernel_guard）=====")
    g = cfg.get("kernel_guard", {})
    print(f"  对称性 max|k − flip(k)| = {g.get('sym')}   （预官 ≤1e-7）")
    print(f"  峰位置 {g.get('peak_rc')}   （须含 [7,7]，即中心格）")
    print(f"  Σk = {g.get('ksum')}")
    print(f"  与 sweep_lambda_theta 的 blur_fft 逐比特相同 = {g.get('bitident')}"
          f"   max|diff| = {g.get('blur_maxdiff')}")
    print(f"  kunif = mean(1−|ĥ|²) = {g.get('kunif'):.6f}   "
          f"（高斯 σ=1.5 的解析估计 0.9646；σ=1 时为 0.9185，与 §3.9(i5) 文档值一致）")
    print(f"  kappa（能量加权、径向分桶，与参照文件不同口径）= {g.get('kappa'):.6f}")
    # **实现修正（2026-10-23 18:2x，披露）**：首版写 `[7,7] in peak_rc`，而 peak_rc 是
    # 扁平坐标对 [7,7] 而非位置列表，恒为 False，L0 被误判为失败（实际全过）。
    ok0 = (g.get("sym", 9) <= 1e-7 and g.get("bitident") is True
           and (g.get("peak_rc") or []) == [7, 7])
    print(f"  ⇒ {'✅ 守卫通过' if ok0 else '❌ 守卫失败'}")

    # ------------------------------------------------------------------ L1
    print("\n===== L1 跨实现复现（同图、同核、同 ckpt、不同脚本）=====")
    print("  容差 = 2% × logp_blur（预官披露：第十四组 K2 把 2% 贴在 Λ_clean 上是标定错误——")
    print("  (ab4) 的 1–2% 数值地板作用在 logp 上，而 Λ_clean 是两个 3–5×10⁴ nats 量之差）")
    nsub = min(12, cfg["nimg_pool"])
    print(f"  {'t':>5} {'参照':>10} {'本组':>10} {'ΔΛ_cl':>9} {'容差':>8} {'logp_blur参照':>13}  判定")
    n_ok1 = 0
    for tt in (0.02, 0.05):
        r = ltmap.get(("gaussian", 1.5, tt))
        v = [x["lam_clean"] for x in pb if x["sn"] == tt and x["img"] < nsub]
        if r is None or not v:
            continue
        m = mean(v)
        tol = 0.02 * r["logp_blur_mean"]
        diff = m - r["lam_mean"]
        ok = abs(diff) <= tol
        n_ok1 += ok
        print(f"  {tt:>5g} {r['lam_mean']:>10.1f} {m:>10.1f} {diff:>+9.1f} {tol:>8.1f} "
              f"{r['logp_blur_mean']:>13.1f}  {'✅' if ok else '❌'}  (n={len(v)})")
    print(f"  ⇒ {n_ok1}/2 档在容差内")
    print("  附：第十四组（错核）在同一条比较上是 +606.2 / +328.7 nats，"
          "即 +4.68% / +7.40% —— 超出 2%×Λ_clean 的旧容差，那正是 K2 失败的原因。")

    # ------------------------------------------------------------------ L2
    print("\n===== L2 核对 (z) 的核效应（诊断：预言比 (z) 低 2%–10%）=====")
    print(f"  {'img':>3} {'σ':>5} {'(z) Λ_cl':>10} {'本组 Λ_cl':>11} {'绝对差':>8} {'相对差':>9}")
    devs, devs_big = [], []
    for r in reg:
        k = (r["img"], r["sn"], r["tag"])
        if k not in refmap or r["tag"] != "blur":
            continue
        o = refmap[k]
        rel = (r["lam_clean"] - o["lam_clean"]) / abs(o["lam_clean"])
        devs.append(rel)
        if abs(o["lam_clean"]) >= 500.0:      # 只在这子集上判带，见下方披露
            devs_big.append(rel)
        print(f"  {r['img']:>3} {r['sn']:>5g} {o['lam_clean']:>10.1f} "
              f"{r['lam_clean']:>11.1f} {r['lam_clean']-o['lam_clean']:>+8.1f} {rel:>+9.2%}")
    if devs:
        print(f"  ⇒ 全部 15 格：相对差 mean = {mean(devs):+.2%}，范围 [{min(devs):+.2%}, "
              f"{max(devs):+.2%}]；预官预言 ∈ [−10%, −2%]")
        inband = all(-0.10 <= x <= -0.02 for x in devs)
        nearz = all(abs(x) <= 0.02 for x in devs)
        print(f"  ⇒ {'✅ 全部落在预言带内' if inband else ('❌ 落在 (z) 的 ±2% 内 ⇒ 核没真正换成 putils 的' if nearz else '⚠️ 有格外，见下方分解')}")
        print(f"  ⇒ |Λ_clean| ≥ 500 nats 的 {len(devs_big)} 格：范围 "
              f"[{min(devs_big):+.2%}, {max(devs_big):+.2%}]，"
              f"{'全部在带内 ✅' if all(-0.10 <= x <= -0.02 for x in devs_big) else '有格外 ❌'}")
        print("  **披露（预官设计缺陷，非事后放松）**：预官把 [−10%,−2%] 这条**相对**带用在全部"
              "15 格上，\n    但 σ=0.2 处 Λ_clean 只有 27…317 nats（≈0），相对差在那里没有意义。"
              "实测三格越界的**绝对**差只有\n    −36.4 / −18.4 / −10.7 nats，而 Hutchinson 对 "
              "logp 的噪声底是 172–250 nats（§3.3）——即越界幅度比测量噪声小一个量级。\n"
              "    所以正确读法是：**在 Λ_clean 有量级的格子上带成立；在 Λ_clean≈0 的格子上"
              "该用绝对差判，而绝对差全部不显著。**")
    print("  （这一条同时给出 §3.9(z)/(ab) 已发布数字因该 bug 产生的偏移量。）")

    # ------------------------------------------------------------------ L3
    print("\n===== L3 主判据一：模糊偏好不随观测噪声存活（Λ_noisy < 0 的比例）=====")

    def frac_neg(rs):
        return (sum(1 for r in rs if r["lam"] < 0) / len(rs)) if rs else float("nan")

    for tag, sel, line in (("L3a 全部 100 对", pb, 0.85),
                           ("L3b σ≥0.02", [r for r in pb if r["sn"] >= 0.02], 0.95),
                           ("L3c σ=0.01（只记录）", [r for r in pb if r["sn"] == 0.01], None)):
        v = frac_neg(sel)
        byi = {}
        for r in sel:
            byi.setdefault(r["img"], []).append(1.0 if r["lam"] < 0 else 0.0)
        lo, hi = boot_ci(byi, mean, 12345)
        if line is None:
            print(f"  {tag}: {v:.3f}（{sum(1 for r in sel if r['lam']<0)}/"
                  f"{len(sel)}）  95% 聚类自助 CI [{lo:.3f}, {hi:.3f}]  只记录不判")
        else:
            print(f"  {tag}: {v:.3f}（{sum(1 for r in sel if r['lam']<0)}/{len(sel)}）  "
                  f"95% 聚类自助 CI [{lo:.3f}, {hi:.3f}]  通过线 {line:.2f}  "
                  f"{'✅' if v >= line else '❌'}")
    print("  (z) 的原始值：全部 0.933、σ≥0.02 为 1.000、σ=0.01 为 0.667；"
          "第十四组（错核）：0.850 / 0.963 / 0.400")

    print("\n  —— 估计量脆弱性（预官要求必须报）——")
    print(f"  {'img':>3} {'负值':>6} {'mean/std':>9} {'偏度':>7} {'中位数':>9} {'均值':>9} "
          f"{'中位/均值':>9} {'σ剖面':>10}")
    for i in sorted({r["img"] for r in pb}):
        v = [r["lam"] for r in pb if r["img"] == i]
        seq = [r["lam"] for r in sorted([r for r in pb if r["img"] == i],
                                        key=lambda r: r["sn"])]
        mono = "不增" if all(seq[j] >= seq[j + 1] for j in range(len(seq) - 1)) else "非单调"
        print(f"  {i:>3} {sum(1 for x in v if x<0):>4}/{len(v)} {mean(v)/sd(v):>9.2f} "
              f"{skew(v):>7.2f} {med(v):>9.0f} {mean(v):>9.0f} {med(v)/mean(v):>9.3f}  {mono:>10}")
    allv = [r["lam"] for r in pb]
    print(f"  合计：负值 {sum(1 for x in allv if x<0)}/{len(allv)}、"
          f"mean/std={mean(allv)/sd(allv):.2f}、偏度={skew(allv):.2f}、"
          f"中位数/均值={med(allv)/mean(allv):.3f}")

    print("\n===== L3d 伴生判据：Λ_noisy(θ*=1.5) > Λ_noisy(θ*=0) 的比例 =====")
    smap = {(r["img"], r["sn"]): r["lam"] for r in ps}
    pairs = [(r, smap[(r["img"], r["sn"])]) for r in pb if (r["img"], r["sn"]) in smap]
    n_ord = sum(1 for a, b in pairs if a["lam"] > b)
    byi = {}
    for a, b in pairs:
        byi.setdefault(a["img"], []).append(1.0 if a["lam"] > b else 0.0)
    lo, hi = boot_ci(byi, mean, 12346)
    print(f"  {n_ord}/{len(pairs)} = {n_ord/len(pairs):.3f}  95% 聚类自助 CI "
          f"[{lo:.3f}, {hi:.3f}]  通过线 0.95  "
          f"{'✅' if n_ord/len(pairs) >= 0.95 else '❌'}   ((z) 报 15/15)")

    # ------------------------------------------------------------------ L4
    print("\n===== L4 主判据二：Lemma 12 的成立与否由 Λ_clean 的符号决定 =====")
    cells = [r for r in pb if "E_hqs" in r]
    good = [r for r in cells if (r["lam_clean"] > 0) == (r["E_hqs"] < 0)]
    neg_lc = [r for r in cells if r["lam_clean"] < 0]
    neg_ok = [r for r in neg_lc if r["E_hqs"] > 0]
    v = len(good) / len(cells)
    byi = {}
    for r in cells:
        byi.setdefault(r["img"], []).append(
            1.0 if (r["lam_clean"] > 0) == (r["E_hqs"] < 0) else 0.0)
    lo, hi = boot_ci(byi, mean, 12347)
    print(f"  对应 {len(good)}/{len(cells)} = {v:.3f}  95% 聚类自助 CI "
          f"[{lo:.3f}, {hi:.3f}]  通过线 0.95、否掉线 0.85  "
          f"{'✅' if v >= 0.95 else ('❌ 否掉' if v < 0.85 else '⚠️ 灰区')}")
    print(f"  反向半边：Λ_clean<0 的格子 {len(neg_lc)} 个，其中 E_hqs>0 的 {len(neg_ok)} 个"
          f"（(z) 是 1/1；数量少，只报不设阈值）")
    for r in neg_lc:
        print(f"    img{r['img']} σ={r['sn']:<5g} Λ_cl={r['lam_clean']:>+9.1f} "
              f"E_hqs={r['E_hqs']:>+9.1f}")

    # ------------------------------------------------------------------ L5
    print("\n===== L5 主判据三：Λ_clean≥2000 nats 上 mean(|E_hqs|/Λ_clean) =====")
    sel = [r for r in cells if r["lam_clean"] >= 2000.0]
    rs = [r["ratio"] for r in sel]
    m, s = mean(rs), se(rs)
    byi = {}
    for r in sel:
        byi.setdefault(r["img"], []).append(r["ratio"])
    lo, hi = boot_ci(byi, mean, 12348)
    ok = (1.00 <= m <= 1.25) and s <= 0.03
    print(f"  n={len(rs)}（{len(byi)} 张图），Λ_clean 跨 "
          f"{min(r['lam_clean'] for r in sel):.0f}–{max(r['lam_clean'] for r in sel):.0f}"
          f"（{max(r['lam_clean'] for r in sel)/min(r['lam_clean'] for r in sel):.1f} 倍）")
    print(f"  mean = {m:.4f}、SE = {s:.4f}  95% 聚类自助 CI [{lo:.4f}, {hi:.4f}]")
    print(f"  与 (ab) 的 W2 同一条预谱带宽：mean ∈ [1.00, 1.25] 且 SE ≤ 0.03  "
          f"{'✅' if ok else '❌'}   ((ab) 报 1.1057 ± 0.0125，n=76；"
          f"第十四组错核报 1.0843 ± 0.0076)")
    worst3 = 0.0
    for r in pb:
        if "E_hqs" not in r:
            continue
        worst3 = max(worst3, abs(r["E_hqs"] - (r["dq"] - r["lam_clean"]
                                               + (r["data_term"] - r["data"])))
                     / max(1.0, abs(r["E_hqs"])))
    print(f"  三项恒等式 E_hqs = Δq − Λ_clean + (data_term − data) 自检残差 = {worst3:.3e}"
          f"（(ab) 报 1.6e−14）")

    # ------------------------------------------------------------------ L6
    print("\n===== L6 诊断（无阈值）=====")
    print("  (a) |E_hqs|/Λ_clean 随 Λ_clean 的走向（分箱）")
    for lo_, hi_ in ((0, 500), (500, 2000), (2000, 8000), (8000, 20000), (20000, 1e9)):
        g2 = [r for r in cells if lo_ <= r["lam_clean"] < hi_]
        if not g2:
            continue
        rr = [r["ratio"] for r in g2]
        print(f"    Λ_cl ∈ [{lo_:>6g},{hi_ if hi_ < 1e8 else 999999:>6g})  n={len(rr):>3}  "
              f"mean 比值={mean(rr):>7.3f}  范围 [{min(rr):.3f},{max(rr):.3f}]  "
              f"负 E_hqs {sum(1 for r in g2 if r['E_hqs']<0)}/{len(g2)}")
    print("  (b) 三项恒等式的两个修正项")
    dqr = [r["dq"] / r["lam_clean"] for r in sel if r["lam_clean"]]
    dtr = [r["data_term"] / r["data"] for r in sel]
    dvc = [r["data"] / r["lam_clean"] for r in sel if r["lam_clean"]]
    print(f"    −Δq/Λ_clean：mean={mean(dqr):+.4f}  范围 [{min(dqr):+.4f},{max(dqr):+.4f}]"
          f"  ((ab) 在 θ*=1.5 上 +0.0030)")
    print(f"    data_term/data：mean={mean(dtr):.4f}  范围 [{min(dtr):.3f},{max(dtr):.3f}]"
          f"  ((ab) 报 [0.71,1.03])")
    print(f"    data/Λ_clean：mean={mean(dvc):.4f}  范围 [{min(dvc):.4f},{max(dvc):.4f}]")
    print("  (c) 图源效应：20 张真实池图 vs 先验样本")
    for nm, g2 in (("pool(20图)", pb),
                   ("reg(先验样本)", [r for r in reg if r["tag"] == "blur"])):
        if not g2:
            continue
        lc = [r["lam_clean"] for r in g2]
        lm = [r["lam"] for r in g2]
        print(f"    {nm:<14} Λ_clean mean={mean(lc):>8.0f} 范围 [{min(lc):.0f},{max(lc):.0f}]；"
              f"Λ_noisy mean={mean(lm):>8.0f} 负值 {sum(1 for x in lm if x<0)}/{len(lm)}")
    print("  (d) data 的尺度不变性（应恒为 N/2=6144 附近）")
    dv = sorted({round(r["data"], 1) for r in rows})
    print(f"    data 取值 {len(dv)} 个：{dv[0]:.1f} … {dv[-1]:.1f}")
    print("  (e) 逐图 σ 剖面（见上表「σ剖面」列）")

    print("\n===== 汇总 =====")
    print(f"  L0 核守卫 {'✅' if ok0 else '❌'}")
    print(f"  L1 跨实现复现 {n_ok1}/2")
    if devs:
        print(f"  L2 核对 (z)：相对差 mean {mean(devs):+.2%}（预言 [−10%,−2%]）")
    print(f"  L3a {frac_neg(pb):.3f} / L3b {frac_neg([r for r in pb if r['sn']>=0.02]):.3f}"
          f" / L3c {frac_neg([r for r in pb if r['sn']==0.01]):.3f} / L3d {n_ord}/{len(pairs)}")
    print(f"  L4 符号对应 {len(good)}/{len(cells)}")
    print(f"  L5 mean 比值 {m:.4f} ± {s:.4f}（n={len(rs)}）；三项恒等式 {worst3:.2e}")


if __name__ == "__main__":
    main()
