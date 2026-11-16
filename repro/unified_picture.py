"""统一图景的定量检验：二阶（功率谱）方法到底看不见 Λ 的哪一部分。（离线，零 GPU）

预官（2026-10-20，跑前写定）
================================================================================
背景（RESULTS §五点五）
---------------------
§五点五 提出"统一图景"：扩散先验盲解卷积的全部困难可归结为先验对模糊图的对数密度
偏好 Λ 这一个已测量的量。原表用三个点对 Λ(σ_n)：

    高斯谱近似在 θ=0 的误差   -25933 / -14313     Λ = 13290 / ~3000
    PnP-HQS 不动点偏差          ~6600              Λ = ~700
    先验重要性采样 ESS          1/64               Λ = ~20

按项目规矩这张表有两个问题：(a) 只有 3 个可比点，隐含比值 1.95 / 4.8 / 9.4 **漂了
5 倍且单调发散**；(b) ESS=1/64 那一行不在同一量纲上。而 §3.9(l5) 已把图景修正为
更强也更好检验的版本：

    **二阶方法失效，是因为它把一个对径向位置极敏感的量，换成了一个只认总衰减的标量。**

本脚本把修正版变成可判定命题。出发点是一条**恒等式**（不是模型）
-------------------------------------------------------------
每个 sweep 文件都存了三个量：`lam_mean`（精确 Λ）、`lam_gauss`（高斯参照 Λ_G）、
`cfg.cap`（= Σ_f wg(f)，其中 wg = Zx/(2(S+t²))，Zx 是图像的周期图、S 是图池周期图）。
由 sweep_law_test.py 第 735 行：

    lam_gauss = Σ_f (1 - g2(f)) * wg(f)

若 wg 在频谱上平坦，则 lam_gauss = cap * kunif（kunif = mean(1-g2) = 1-u）。
于是

    Λ_ng := Λ - Λ_G

恰好就是**二阶方法看不见的那部分偏好**。三个输入量全都已在文件里，零 GPU 可算，
覆盖 16 个文件 671 个算子行。

**写判据前已经看到过的东西（披露，防事后合理化）**：
  - RESULTS §3.9(l5) 的表：ideal-lp/bp/hp 在 4 个 K 上的 Λ 与 Λ_Gauss（已发布结果）。
  - 各 sweep 文件 cfg 里的 cap 值（1081.7 → 784.9，随 t 下降）。
  - 各文件的 cfg 与 meta 字段结构。
**没有看到过的东西**：Λ_ng、A=Λ/Λ_G、r=Λ_G/(cap·kunif)、任何散度、符号统计、
|Λ_G/Λ| 的 (t,u) 表。本判据全部由恒等式与 §3.9(l5) 的已发布数字推出。

预官判据
--------
U0 守门（恒等式）：每行 lam_mean == logp_blur_mean - logp_sharp_mean，
   相对误差 ≤ 1e-12。不过则后面全部作废。（这条在 §3.9(x) 的 E1 上抓过 1.16 nats
   的跨文件 bug，是唯一能抓住它的东西。）

U1 二阶盲性（**可能失败**）：Λ_G/(cap·kunif) 在同一文件内应近似常数——共同偏移
   来自"20 张图 vs 64 张池"的谱采样差异，与算子无关。真正要判的是除掉共同偏移后
   **跨算子形状的残差散度**：
     (a) profile_shape B 组（ideal-lp/bp/hp，K、Σa、剖面二值性全部相同）内，
         Λ_G 除掉共同偏移后的残差散度 ≤ 2%；
     (b) 同组 Λ 的相对散度 ≥ 10 × (a) 的散度（以百分点计）。
   两者合起来就是"二阶只认总衰减、对位置盲"的定量版。
   为什么用比值而不是"Λ 散度 ≥100%"：K=0.80 处 Λ 三者同号且只差 1.44×，绝对阈值
   会在那里假失败；比值判据尺度不变，是这里该用的形式（项目规矩：描述量跨度/C 跨度
   相差很大时该换解释而不是描述量）。
   **失败方式**：若 (a) 的残差散度 >2%，则 Λ_G 并非完全盲于位置，"盲性"论断必须
   加限定——这是本判据唯一有价值的失败方式。

U2 幅度（**只记录，不判**）：A = Λ/Λ_G 的分布。§3.9(h6) 已报 A 从 3.2 到 22-37；
   此处只把同一个量铺到全部 671 行并报分位数。设通过/失败没有信息量。

U3 位置判决（**最尖的一条，可能失败**）：profile_shape B 组每个 K 上，
       |A(hp)| ≥ 5 × max(|A(lp)|, |A(bp)|)   且配对 |t| ≥ 2
   其中 hp = 衰减在内圈（含 DC），lp = 外圈，bp = 环带；三者 kunif 相同到 0.004。
   lp vs bp 的差**预先降级为只记录**：K=0.60 处两行 Λ 差 94 nats 而各自的 SE 约
   500-530 nats，信噪比 ≈0.13——即使论断完全正确也测不出（项目规矩：预官间距
   <2σ 的符号只记录、不作判据）。
   **新意**：§3.9(l5) 只在 K=0.60 一个点上报了这一刀；本判据把它推到全部 4 个 K，
   并用逐图的 lam_per_img 做**配对**检验（20 张图两边共用，SE 不会被高估）。

U4 符号（**可能失败**）：Λ_ng 与 Λ 同号的比例 ≥ 95%。报每一个反例的 (t, 族, u)。
   预言反例集中在 u 小（强模糊）且 t 大处。

U5 诊断（无阈值）：(a) cap(t)/cap(0.002) 与 ρ(t)=Λ(t)/Λ(0.002) 的对照——两者都随
   t 下降但速率不同，其比就是 Λ_G/Λ 的 t 增长率；(b) |Λ_G/Λ| 的 (t,u) 表；(c) 落盘
   results/fig_unified.json。

本检验**不**证明什么（诚实声明）
------------------------------
本检验只覆盖统一图景的**机理半**（二阶方法看不见什么）。统一图景的**幅度半**
（"任何算法的失败幅度都由 Λ 定标"）仍然是 3 个点的巧合、隐含比值漂 5 倍，本脚本
不碰它——检验它需要 PnP-HQS 与采样器的失败幅度数据，目前不存在。

用法： venv2/bin/python unified_picture.py
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

FILES = [
    "sweep_b2test.json", "sweep_bdense.json", "sweep_bgap.json", "sweep_blow.json",
    "sweep_bt0005.json", "sweep_bt001.json", "sweep_bt002.json", "sweep_bt005.json",
    # 第十组（2026-10-20）补的三个高 t；加进来是为了把 U2/U5 延伸到新区间。
    # 判据 U0–U5 的阈值未改（U2/U5 本来就是「只记录/无阈值」），只是数据变多。
    "sweep_bs008.json", "sweep_bs012.json", "sweep_bs016.json",
    "sweep_law_test.json", "sweep_m_afq_afq.json", "sweep_m_ffq_afq.json",
    "sweep_profile_shape.json", "sweep_steepness.json", "sweep_twlaw.json",
    "sweep_twod.json", "sweep_uniform.json",
]
DENSE = ["W2", "W8", "TK50"]
T0 = 0.002
KINDS = ("lp", "bp", "hp")


def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if "rows" not in d or "cap" not in d.get("cfg", {}):
        return None
    return d


def fid_of(r):
    return r["meta"].get("fid") or r["meta"].get("family") or r["id"].split("@")[0]


def main():
    datas = []
    for fn in FILES:
        d = load(fn)
        if d is None:
            print(f"[skip] {fn}")
            continue
        datas.append((fn, d))
    byname = dict(datas)
    nrow = sum(len(d["rows"]) for _, d in datas)
    print("=" * 100)
    print(f" 统一图景检验：载入 {len(datas)} 个文件 / {nrow} 个算子行")
    print("=" * 100)

    # ---------------- U0 守门 ----------------
    print("\n" + "=" * 100)
    print(" U0 守门：lam_mean == logp_blur_mean - logp_sharp_mean（恒等式）")
    print("=" * 100)
    worst, worst_at = 0.0, ""
    for fn, d in datas:
        for r in d["rows"]:
            a = r["lam_mean"]
            b = r["logp_blur_mean"] - r["logp_sharp_mean"]
            e = abs(a - b) / max(abs(a), 1.0)
            if e > worst:
                worst, worst_at = e, fn + ":" + r["id"]
    print(f" 最大相对误差 = {worst:.3e}   ({worst_at})")
    if worst > 1e-12:
        print(" ❌ U0 未过 —— 后面全部作废")
        return
    print(" ✅ U0 通过")

    # ---------------- U1 二阶盲性 ----------------
    print("\n" + "=" * 100)
    print(" U1 二阶盲性：Λ_G 是否只认总衰减（kunif），而对衰减的径向位置盲")
    print("=" * 100)
    print(" Λ_G^flat = cap·kunif 是「wg 在频谱上平坦」时的解析预言。")
    print(" r = Λ_G/Λ_G^flat 在同一文件内应近似常数（共同偏移 = 谱采样差），")
    print(" 跨算子形状的残差散度才是要判的东西。\n")
    print(f" {'文件':<26} | {'t':>7} | {'cap':>9} | {'r 均值':>8} | {'r 标准差':>9} | "
          f"{'r 极差':>8}")
    print(" " + "-" * 88)
    for fn, d in datas:
        cap = d["cfg"]["cap"]
        rs = np.array([r["lam_gauss"] / (cap * r["kunif"]) for r in d["rows"]
                       if r["kunif"] > 1e-9])
        print(f" {fn:<26} | {d['cfg']['t']:>7g} | {cap:>9.2f} | {rs.mean():>8.4f} | "
              f"{rs.std():>9.4f} | {rs.max()-rs.min():>8.4f}")
    print(" " + "-" * 88)

    ps = byname["sweep_profile_shape.json"]
    grp = defaultdict(dict)
    for r in ps["rows"]:
        m = r["meta"]
        if m.get("kind") in KINDS:
            grp[round(float(m["K"]), 4)][m["kind"]] = r
    print("\n" + "-" * 100)
    print(" B 组（ideal-lp / bp / hp；K、Σa、剖面二值性全部相同，唯一差别是衰减的半径）")
    print("-" * 100)
    print(f" {'K':>5} | {'kunif lp/bp/hp':>21} | {'Λ_G lp/bp/hp':>27} | "
          f"{'Λ_G 残差散度':>13} | {'Λ lp/bp/hp':>29} | {'Λ 相对散度':>12}")
    print(" " + "-" * 118)
    u1 = []
    for K in sorted(grp):
        g = grp[K]
        if not all(k in g for k in KINDS):
            continue
        ku = [g[k]["kunif"] for k in KINDS]
        lg = np.array([g[k]["lam_gauss"] for k in KINDS])
        lam = np.array([g[k]["lam_mean"] for k in KINDS])
        res = lg / lg.mean() - 1.0
        sg = float(res.max() - res.min())
        alam = np.abs(lam)
        sl = float((alam.max() - alam.min()) / alam.mean())
        u1.append((K, sg, sl))
        print(f" {K:>5.2f} | {ku[0]:.4f}/{ku[1]:.4f}/{ku[2]:.4f} | "
              f"{lg[0]:>9.1f}{lg[1]:>8.1f}{lg[2]:>8.1f} | {sg*100:>12.2f}% | "
              f"{lam[0]:>+10.0f}{lam[1]:>+9.0f}{lam[2]:>+9.0f} | {sl*100:>11.1f}%")
    print(" " + "-" * 118)
    # 实现修正（2026-10-20 11:2x，披露）：首版把分数 sl 与百分数 10*sg*100 直接相比，
    # 单位不一致导致 U1(b) 假失败。判据本身（Λ 散度 ≥ 10×Λ_G 散度）未改。
    ok_a = bool(u1) and all(sg <= 0.02 for _, sg, _ in u1)
    ok_b = bool(u1) and all(sl >= 10 * sg for _, sg, sl in u1)
    for K, sg, sl in u1:
        print(f"   K={K:.2f}: Λ_G 残差散度 {sg*100:5.2f}% (≤2%)   "
              f"Λ 相对散度 {sl*100:6.1f}% (≥{10*sg*100:5.1f}%)   "
              f"倍数 {sl*100/(sg*100):6.1f}×")
    print(f" => U1(a) {'✅' if ok_a else '❌'}   U1(b) {'✅' if ok_b else '❌'}   "
          f"U1 {'✅ 通过' if (ok_a and ok_b) else '❌ 失败'}")

    # ---------------- U2 幅度（只记录） ----------------
    print("\n" + "=" * 100)
    print(" U2 幅度（只记录，不判）：|Λ_G/Λ| 的分布")
    print("=" * 100)
    allr = np.array([abs(r["lam_gauss"] / r["lam_mean"])
                     for _, d in datas for r in d["rows"] if r["lam_gauss"] > 0])
    print(f" n = {len(allr)}（lam_gauss>0 的行，共 {nrow} 行）")
    print(" " + " | ".join(f"P{q}={np.percentile(allr, q):.4f}"
                           for q in (50, 75, 90, 95, 99, 100)))
    print(f" 中位数 {np.median(allr):.4f} => 二阶参照平均只解释 Λ 的 "
          f"{100*np.median(allr):.1f}%")

    # ---------------- U3 位置判决 ----------------
    print("\n" + "=" * 100)
    print(" U3 位置判决：|A(hp)| ≥ 5×max(|A(lp)|,|A(bp)|) 且配对 |t|≥2")
    print("=" * 100)
    print(f" {'K':>5} | {'A(lp)':>8} | {'A(bp)':>8} | {'A(hp)':>9} | {'倍数':>7} | "
          f"{'配对 t':>8} | 判定")
    print(" " + "-" * 80)
    u3 = []
    for K in sorted(grp):
        g = grp[K]
        if not all(k in g for k in KINDS):
            continue
        a = {k: g[k]["lam_mean"] / g[k]["lam_gauss"] for k in KINDS}
        ref = "lp" if abs(a["lp"]) >= abs(a["bp"]) else "bp"
        dh = np.array(g["hp"]["lam_per_img"]) / g["hp"]["lam_gauss"]
        dr = np.array(g[ref]["lam_per_img"]) / g[ref]["lam_gauss"]
        df = dh - dr
        tt = df.mean() / (df.std(ddof=1) / np.sqrt(len(df)))
        ratio = abs(a["hp"]) / max(abs(a["lp"]), abs(a["bp"]))
        good = bool(ratio >= 5.0 and abs(tt) >= 2.0)
        u3.append((K, ratio, abs(tt), good))
        print(f" {K:>5.2f} | {a['lp']:>8.2f} | {a['bp']:>8.2f} | {a['hp']:>9.2f} | "
              f"{ratio:>7.2f} | {tt:>8.2f} | {'✅' if good else '❌'}  (基准 {ref})")
    print(" " + "-" * 80)
    print(" 附：lp vs bp 只记录不判（预官已声明信噪比 ≈0.13，测不出）")
    for K in sorted(grp):
        g = grp[K]
        if "lp" in g and "bp" in g:
            d = np.array(g["lp"]["lam_per_img"]) - np.array(g["bp"]["lam_per_img"])
            tt = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
            print(f"   K={K:.2f}: Λ(lp)-Λ(bp) = {d.mean():+7.1f} ± "
                  f"{d.std(ddof=1)/np.sqrt(len(d)):6.1f}   t={tt:+.2f}")
    npass = sum(1 for _, _, _, g_ in u3 if g_)
    print(f" => U3 {npass}/{len(u3)} 个 K 通过"
          f"（{'✅' if u3 and npass == len(u3) else '❌'}）")

    # ---------------- U4 符号 ----------------
    print("\n" + "=" * 100)
    print(" U4 符号：Λ_ng = Λ - Λ_G 与 Λ 同号的比例 ≥ 95%")
    print("=" * 100)
    same, bad = 0, []
    for fn, d in datas:
        for r in d["rows"]:
            ng = r["lam_mean"] - r["lam_gauss"]
            if ng * r["lam_mean"] > 0:
                same += 1
            else:
                bad.append((d["cfg"]["t"], fid_of(r), round(1 - r["kunif"], 4),
                            r["lam_mean"], r["lam_gauss"], ng, fn))
    tot = same + len(bad)
    print(f" 同号 {same}/{tot} = {100*same/tot:.2f}%   反号 {len(bad)}")
    if bad:
        print(f"\n 反号的全部 {len(bad)} 行（t, 族, u, Λ, Λ_G, Λ_ng, 文件）：")
        for t, f, u, lm, lg, ng, fn in bad:
            print(f"   t={t:<6g} {f:<10} u={u:<6} Λ={lm:>+9.0f} Λ_G={lg:>8.1f} "
                  f"Λ_ng={ng:>+9.0f}  {fn}")
    print(f" => U4 {'✅ 通过' if 100*same/tot >= 95 else '❌ 失败'}")

    # ---------------- U5 诊断 ----------------
    base = byname["sweep_bdense.json"]
    bc = base["cfg"]
    pb = defaultdict(dict)
    for r in base["rows"]:
        pb[r["meta"]["fid"]][round(1 - r["kunif"], 4)] = r["lam_mean"]
    for r in byname["sweep_blow.json"]["rows"]:      # 与 rho_t.py 同序：blow 覆盖
        pb[r["meta"]["fid"]].setdefault(round(1 - r["kunif"], 4), r["lam_mean"])
    cap0 = bc["cap"]

    print("\n" + "=" * 100)
    print(" U5(a) 诊断：cap(t)/cap(t0) 与 ρ(t)=Λ(t)/Λ(t0)（两者都随 t 降，速率不同）")
    print("=" * 100)
    print(f" {'t':>7} | {'cap':>9} | {'cap/cap0':>9} | {'ρ 均值':>8} | "
          f"{'ρ 的 u 散度':>12} | {'(Λ_G/Λ) 增长':>13}")
    print(" " + "-" * 78)
    diag = []
    for fn, d in datas:
        c = d["cfg"]
        if c["t"] == T0 or c.get("ckpt") != bc.get("ckpt") \
                or c.get("pool") != bc.get("pool"):
            continue
        rhos = [(round(1 - r["kunif"], 4), r["lam_mean"] / pb[r["meta"]["fid"]][u])
                for r in d["rows"] if r["meta"].get("fid") in DENSE
                for u in [round(1 - r["kunif"], 4)]
                if u in pb.get(r["meta"]["fid"], {})]
        if not rhos:
            continue
        rv = np.array([v for _, v in rhos])
        cr = c["cap"] / cap0
        diag.append(dict(t=c["t"], cap=c["cap"], cap_ratio=float(cr),
                         rho=float(rv.mean()), u_spread=float(rv.max() / rv.min()),
                         growth=float(cr / rv.mean())))
        print(f" {c['t']:>7g} | {c['cap']:>9.2f} | {cr:>9.4f} | {rv.mean():>8.4f} | "
              f"{rv.max()/rv.min():>12.3f} | {cr/rv.mean():>13.3f}")
    print(" " + "-" * 78)
    print(" 「(Λ_G/Λ) 增长」= [cap(t)/cap0]/ρ(t)；>1 表示二阶参照的占比随 t 上升。")

    print("\n" + "=" * 100)
    print(" U5(b) 诊断：|Λ_G/Λ| 的 (t,u) 表（三族均值）")
    print("=" * 100)
    tf = sorted([(d["cfg"]["t"], d) for _, d in datas
                 if d["cfg"]["t"] != T0
                 and d["cfg"].get("ckpt") == bc.get("ckpt")
                 and d["cfg"].get("pool") == bc.get("pool")], key=lambda x: x[0])
    us = sorted({round(1 - r["kunif"], 4) for _, d in tf for r in d["rows"]
                 if r["meta"].get("fid") in DENSE}, reverse=True)
    print(f" {'u':>6} | " + " ".join(f"{'t='+format(t, 'g'):>9}" for t, _ in tf))
    print(" " + "-" * (9 + 10 * len(tf)))
    for u in us:
        cells = []
        for t, d in tf:
            v = [abs(r["lam_gauss"] / r["lam_mean"]) for r in d["rows"]
                 if r["meta"].get("fid") in DENSE
                 and round(1 - r["kunif"], 4) == u and r["lam_mean"] != 0]
            cells.append(f"{np.mean(v):>9.4f}" if v else f"{'-':>9}")
        print(f" {u:>6.3f} | " + " ".join(cells))
    print(" " + "-" * (9 + 10 * len(tf)))

    with open(os.path.join(OUT, "fig_unified.json"), "w") as fh:
        json.dump(dict(u0=worst,
                       u1=[dict(K=K, spread_g=sg, spread_l=sl) for K, sg, sl in u1],
                       u1_pass=bool(ok_a and ok_b),
                       u2=dict(n=int(len(allr)), median=float(np.median(allr)),
                               p90=float(np.percentile(allr, 90)),
                               p99=float(np.percentile(allr, 99))),
                       u3=[dict(K=K, ratio=ratio, t=tt, ok=g_)
                           for K, ratio, tt, g_ in u3],
                       u4=dict(same=same, total=tot, pct=100 * same / tot,
                               bad=[dict(t=t, fid=f, u=u, lam=lm, lam_g=lg,
                                         lam_ng=ng, fn=fn)
                                    for t, f, u, lm, lg, ng, fn in bad]),
                       u5=diag), fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'fig_unified.json')}")


if __name__ == "__main__":
    main()
