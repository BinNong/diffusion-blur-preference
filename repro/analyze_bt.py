"""b(u) 是否随噪声级 t 变？裁判：T1-T6，判据跑前写定在 sweep_law_test.py 第八组。

预官来源：sweep_law_test.py docstring 第八组（2026-10-18 21:44，跑前写定；
          md5 = 5864b1cbae505a75b28fd0674772c53c）。

与第七组（analyze_bdense.py）唯一的差别：**只改 t**。同样的 3 个族
（W2 / W8 / TK50）、同样 20 张图、同样的 u 网格、同样的提取方式
（logΛ 对 log u 的 5 次多项式，b(u) 取其解析导数；SE 由对 20 张图的有放回
重采样给出）。基线是 t=0.002 的 sweep_bdense + sweep_blow。

判据（第八组，跑前写定）：
  T1 守门：每个 t、每个族，闭式 b=b_inf+A*u^q 的拟合 rms <= 2 倍基线的 rms
     （预官基线 rms：W2 0.016 / W8 0.047 / TK50 0.037 => 上限 0.032/0.094/0.074）。
  T2 |b_inf(t)-b_inf(0.002)| <= 2*SE_Delta（配对 bootstrap）。
  T3 |q(t)-q(0.002)| <= 2*SE_Delta（配对 bootstrap）。
  T4 无模型：D(u)=logΛ_t(u)-logΛ_0(u) 对 log u 的斜率 <= 2*SE（配对 bootstrap）。
     若 b(u) 与 t 无关，则 Λ=C(t)u^{b(u)}，D(u) 必须与 u 无关。
  T5 方向性（附带、非判决）：若 T2/T3 失败，检查符号是否符合高斯收敛预言
     Δb_inf>0、Δq<0。预官已记录该预言的量级（~0.05 dex）与噪声同量级，
     所以符号不符不能作为反证。
  T5b 定量高斯收敛预言（修正案 2026-10-18 22:02，跑前写定，md5 e31c14db...）：
     Λ = Λ_Gauss + Λ_ng，b_Gauss(u) = -u/(1-u) 与 t 无关，b_ng(u) 与 t 无关
     （待检验），b_ng 由 t=0.002 基线自校准 => b_pred(t,u) 被完全预言
     （只用已测的 Λ 电平，不含任何新拟合）。判据：19 个 u 上 |b_测-b_pred| 的
     rms <= 0.10 且均值偏差 <= 2*SE_bootstrap。
  T6 必须报告可探测下限 2*SE_Delta(b_inf) 与 2*SE_Delta(q)，否则"通过"不可报告。

配对 bootstrap 的实现要点：同一批图像重采样下标同时用于基线和新 t 的整套拟合，
所以两条 b(u) 曲线的相关性被正确处理——两次独立 bootstrap 再相减会高估 SE。

用法： venv2/bin/python analyze_bt.py
"""

import argparse
import importlib.util
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
sys.path.insert(0, os.path.join(PROJ, "code", "blind_deblur_diffusion"))

_spec = importlib.util.spec_from_file_location(
    "hd", os.path.join(PROJ, "repro", "hunt_descriptor.py"))
hd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hd)

DENSE = ["W2", "W8", "TK50"]
DEG = 5
NBOOT = 2000
# 预官写定的基线 rms（T1 的上限来源）；脚本会重算并核对
PRE_RMS = {"W2": 0.016, "W8": 0.047, "TK50": 0.037}
# 新 t 的数据文件：默认用 glob 自动发现（文件名不可靠，t 一律从文件内 cfg 读）
NEWFILES = None


def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    prof = defaultdict(list)
    gauss = defaultdict(dict)
    for rec in d["rows"]:
        fid = rec["meta"]["fid"]
        prof[fid].append((1.0 - rec["kunif"], rec["lam_mean"],
                          rec.get("lam_per_img")))
        gauss[fid][round(1.0 - rec["kunif"], 4)] = rec.get("lam_gauss")
    for k in prof:
        prof[k].sort(key=lambda c: -c[0])
    return d, dict(prof), {k: dict(v) for k, v in gauss.items()}


def grid(prof, fids, lo=0.13, hi=0.86):
    """三族共有的 u 网格（自动探测，避免再把 --targets 当成 u）。

    返回 (ug_all, ug)：ug_all 是全部共有 u（含 u=0.08 的 @raw 点），ug 是闭式拟合
    用的子集 [lo, hi]。**多项式必须在 ug_all 上拟合**——第七组
    （analyze_bdense.py）就是在全部共有 u 上拟合 5 次多项式、再在 [0.13,0.86] 上拟
    闭式；这里少一个点就会改变 b(u)，T2/T3 的跨 t 比较就不再是同方法的。
    """
    sets = [{round(u, 4) for u, _, _ in prof[f]} for f in fids]
    ug_all = sorted(set.intersection(*sets), reverse=True)
    return ug_all, [u for u in ug_all if lo <= u <= hi]


def matrices(prof, fids, ug_all):
    """(nimg, nu) 的 **原始 Λ** 矩阵；缺点的族返回 None。

    两个坑，都踩过：
    1. 第七组的 b(u) 拟合用 log(**均值** Λ)（analyze_bdense.py:
       `np.polyfit(log u, log(lam_mean), DEG)`），不是 mean(log Λ)。必须逐字对齐。
    2. **不能把非正 Λ 夹到 1e-300**。个别图完全可能Λ<=0（先验偏爱清晰图），夹一下
       再把 log 传给 mean-of-logs，会把 -23 nats 变成 log=-690，于是一个图就制造出
       -34 dex 的假 D(u)——第一版就这么错了，D(u=0.85) 报出 -34.86 dex。
       这里原样保留，取 log 的地方各自显式处理。
    """
    out = {}
    for fid in fids:
        d = {round(u, 4): pi for u, _, pi in prof[fid]}
        cols, ok = [], True
        for u in ug_all:
            pi = d.get(round(u, 4))
            if not pi:
                ok = False
                break
            cols.append(np.array(pi, dtype=float))
        out[fid] = np.array(cols).T if ok else None     # (nimg, nu)
    return out


def Dvec(P0, P1, nug):
    """逐图配对差 D(u)=logΛ_t(u)-logΛ_0(u) 再对图取均值（预官 T4 的算子）。

    只取两侧 Λ 都为正的图；Λ<=0 时 log 无定义，排除并报告排除了几张。
    同时返回 log(均值) 版本作对照（那个永远有定义，且与 b(u) 的估计量一致）。
    """
    A, B = P0[:, :nug], P1[:, :nug]
    m = (A > 0) & (B > 0)
    la = np.log(np.where(m, A, 1.0))
    lb = np.log(np.where(m, B, 1.0))
    cnt = m.sum(0)
    D = np.where(cnt > 0, (lb - la).sum(0) / np.maximum(cnt, 1), np.nan)
    Dl = np.log(B.mean(0)) - np.log(A.mean(0))
    return D, Dl, cnt


def bcurve(P, us_all):
    """logΛ 对 log u 的 DEG 次多项式（Λ 先对图取均值再取 log，同第七组）；
    拟合用全部共有 u（us_all），返回的 b(u) 可在任意 u 上求值。"""
    if P is None:
        return None
    la = np.log(np.array(us_all))
    mu = P.mean(0)
    if not (mu > 0).all():
        bad = [round(float(u), 4) for u, v in zip(us_all, mu) if v <= 0]
        print(f"   [警告] 有 {len(bad)} 个 u 的均值 Λ <= 0，无法取 log：{bad}")
        return None
    try:
        c = np.polyfit(la, np.log(mu), DEG)
    except Exception:
        return None
    d = np.polyder(c)
    return lambda u: float(np.polyval(d, math.log(u)))


def closed_form(us, bfun):
    """预官顺序里的第一个也是第七组胜出者：b = b_inf + A*u^q。"""
    from scipy.optimize import curve_fit
    f1 = lambda u, *p: p[0] + p[1] * u ** p[2]          # curve_fit 按位置传参
    ua = np.array(us)
    bv = np.array([bfun(u) for u in us])
    try:
        popt, _ = curve_fit(f1, ua, bv, p0=(-0.7, -6.0, 3.0),
                            bounds=([-1e3, -50, 0.1], [1e3, 50, 12]),
                            maxfev=100000)
    except Exception:
        return None
    rms = float(np.sqrt(np.mean((bv - f1(ua, *popt)) ** 2)))
    return dict(p=popt, rms=rms, bv=bv)


def gauss_b(u):
    """高斯先验的闭式指数（推导见 docstring）：b_Gauss(u) = -u/(1-u)，与 t 无关。"""
    return -u / (1.0 - u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None)
    ap.add_argument("--dump", default=None,
                    help="把关键数值存成 JSON 供 plot_bt.py 画图（避免重算 bootstrap）")
    args = ap.parse_args()

    # ---------- 基线 t=0.002 ----------
    rb = load("sweep_bdense.json")
    if rb is None:
        print("[缺少] sweep_bdense.json")
        return
    pb = dict(rb[1])
    gb = dict(rb[2])
    rl = load("sweep_blow.json")
    if rl is not None:
        for fid, rows in rl[1].items():
            pb.setdefault(fid, []).extend(rows)
        for k in pb:
            pb[k].sort(key=lambda c: -c[0])
        for fid, gg in rl[2].items():
            gb.setdefault(fid, {}).update(gg)
    print("=" * 104)
    print(" 0. 基线（t=0.002）与数据文件")
    print("=" * 104)
    print(f" sweep_bdense.json + sweep_blow.json：nimg={rb[0]['cfg']['nimg']} "
          f"t={rb[0]['cfg']['t']:g}")
    ug_all, ug = grid(pb, DENSE)
    print(f" 全部共有 u（{len(ug_all)} 点）: " + " ".join(f"{u:.3f}" for u in ug_all))
    print(f" 闭式拟合子集（{len(ug)} 点）: " + " ".join(f"{u:.3f}" for u in ug))
    Pb = matrices(pb, DENSE, ug_all)
    if any(Pb[f] is None for f in DENSE):
        print("[错误] 基线有点缺失")
        return
    lus = np.log(np.array(ug))          # 闭式/D(u) 用的 19 点子网格
    lus_all = np.log(np.array(ug_all))  # 多项式拟合用全部共有 u（同第七组）
    nug = len(ug)
    # 护栏：下面多处用 P[:, i]（i < len(ug)）直接索引 ug，这要求 ug 恰好是 ug_all
    # 的前 len(ug) 个（即 ug_all 只可能在低端多出 u=0.08 那个 @raw 点）。
    assert ug_all[:len(ug)] == ug, "ug 不是 ug_all 的前缀，P 的列索引会错位"

    base = {}
    for fid in DENSE:
        bf = bcurve(Pb[fid], ug_all)
        cf = closed_form(ug, bf)
        base[fid] = dict(b=bf, cf=cf, P=Pb[fid],
                         G=np.array([gb.get(fid, {}).get(round(u, 4), np.nan)
                                     for u in ug]))
        if cf is None:
            print(f"[错误] 基线 {fid} 闭式拟合失败")
            return
        print(f" {fid:<5} b_inf={cf['p'][0]:>7.3f}  A={cf['p'][1]:>7.3f}  "
              f"q={cf['p'][2]:>6.3f}  rms={cf['rms']:.3f}"
              f"（预官 rms {PRE_RMS[fid]:.3f}）")

    # ---------- 新 t ----------
    import glob
    files = args.files if args.files else sorted(
        glob.glob(os.path.join(OUT, "sweep_bt*.json")))
    files = [os.path.basename(f) for f in files]
    news = []
    for fn in files:
        r = load(fn)
        if r is None:
            print(f" [跳过] {fn}（不存在）")
            continue
        t = float(r[0]["cfg"]["t"])
        prof = dict(r[1])
        gpr = dict(r[2])
        ug_all2, ug2 = grid(prof, DENSE)
        assert ug_all2[:len(ug2)] == ug2, \
            f"{fn}: ug 不是 ug_all 的前缀，P 的列索引会错位"
        same = (len(ug2) == len(ug)) and all(abs(a - b) < 1e-9
                                            for a, b in zip(ug, ug2))
        print(f"\n [载入] {fn}: nimg={r[0]['cfg']['nimg']} t={t:g}；"
              f"全部共有 u {len(ug_all2)} 个、闭式子网格 {len(ug2)} 个；"
              f"子网格与基线{'一致' if same else '不一致!'}")
        if not same:
            print("   新子网格: " + " ".join(f"{u:.3f}" for u in ug2))
            print("   [注意] 子网格不一致时仍按各自网格拟合闭式，但 T2/T3/T4 的配对")
            print("           bootstrap 只在共有 u 上做；若差异大则结果不可比。")
        P = matrices(prof, DENSE, ug_all2)
        news.append(dict(fn=fn, t=t, ug=ug2, ug_all=ug_all2, P=P, G=gpr))

    if not news:
        print("\n[没有新 t 的数据]")
        return

    # ---------- 每个 t 的 b(u) 与闭式 ----------
    print("\n" + "=" * 104)
    print(" 1. 各 t 的 b(u)（5 次多项式解析导数）与闭式 b=b_inf+A*u^q")
    print("=" * 104)
    print(f" {'t':>7} | {'族':<5} | {'b_inf':>8} | {'A':>8} | {'q':>7} | "
          f"{'rms':>7} | {'T1上限':>7} | {'T1':>4}")
    print(" " + "-" * 76)
    curves = {}
    for nd in news:
        for fid in DENSE:
            bf = bcurve(nd["P"][fid], nd["ug_all"])
            cf = closed_form(nd["ug"], bf) if bf else None
            curves[(nd["t"], fid)] = dict(b=bf, cf=cf, P=nd["P"][fid])
            lim = 2.0 * PRE_RMS[fid]
            ok = cf is not None and cf["rms"] <= lim
            if cf is None:
                print(f" {nd['t']:>7g} | {fid:<5} | {'拟合失败':>8}")
            else:
                print(f" {nd['t']:>7g} | {fid:<5} | {cf['p'][0]:>8.3f} | "
                      f"{cf['p'][1]:>8.3f} | {cf['p'][2]:>7.3f} | "
                      f"{cf['rms']:>7.3f} | {lim:>7.3f} | "
                      f"{'✅' if ok else '❌':>4}")

    # ---------- b(u) 逐点对照 ----------
    print("\n" + "=" * 104)
    print(" 2. b(u) 逐点：基线 vs 各 t（含高斯极限 b_Gauss=-u/(1-u)）")
    print("=" * 104)
    hdr = f" {'u':>6} | {'b_Gauss':>8} | " + " | ".join(
        f"{'t=' + format(nd['t'], 'g'):>16}" for nd in news)
    print(hdr)
    print(" " + "-" * (18 + 19 * len(news) + 6))
    for fid in DENSE:
        print(f" --- {fid} ---")
        for u in ug:
            row = [f"{gauss_b(u):>8.3f}"]
            for nd in news:
                c = curves[(nd["t"], fid)]
                if c["b"] is None:
                    row.append(f"{'--':>16}")
                else:
                    v0 = base[fid]["b"](u)
                    v1 = c["b"](u)
                    row.append(f"{v1:>7.3f} ({v1 - v0:>+6.3f})")
            print(f" {u:>6.3f} | " + " | ".join(row))
        print(" " + "-" * (18 + 19 * len(news) + 6))
    print(" 括号内是相对基线 t=0.002 的变化 Δb。")

    # ---------- 配对 bootstrap ----------
    print("\n" + "=" * 104)
    print(" 3. 配对 bootstrap（同一批图像重采样同时用于基线与新 t；2000 次）")
    print("=" * 104)
    rng = np.random.default_rng(0)
    nimg = Pb[DENSE[0]].shape[0]
    res = {}
    for nd in news:
        t = nd["t"]
        for fid in DENSE:
            P0, P1 = base[fid]["P"], curves[(t, fid)]["P"]
            if P0 is None or P1 is None or P0.shape != P1.shape:
                continue
            db, dq, sl, rngD = [], [], [], []
            for _ in range(NBOOT):
                idx = rng.integers(0, nimg, nimg)
                c0 = c1 = None
                try:
                    cc = np.polyfit(lus_all, np.log(P0[idx].mean(0)), DEG)
                    d = np.polyder(cc)
                    c0 = closed_form(
                        ug, lambda u, d=d: float(np.polyval(d, math.log(u))))
                except Exception:
                    c0 = None
                try:
                    cc = np.polyfit(lus_all, np.log(P1[idx].mean(0)), DEG)
                    d = np.polyder(cc)
                    c1 = closed_form(
                        ug, lambda u, d=d: float(np.polyval(d, math.log(u))))
                except Exception:
                    c1 = None
                if c0 is None or c1 is None:
                    continue
                db.append(c1["p"][0] - c0["p"][0])
                dq.append(c1["p"][2] - c0["p"][2])
                # T4 的 D(u)：逐图配对差再对图取均值（预官原文），只用两侧 Λ>0 的图
                D, _, _ = Dvec(P0[idx], P1[idx], nug)
                good = np.isfinite(D)
                if good.sum() >= 5:
                    try:
                        sl.append(float(np.polyfit(lus[good], D[good], 1)[0]))
                    except Exception:
                        pass
                    rngD.append(float(np.nanmax(D[good]) - np.nanmin(D[good])))
            res[(t, fid)] = dict(
                db=np.array(db), dq=np.array(dq), sl=np.array(sl),
                rD=np.array(rngD))
    print(f" {'t':>7} | {'族':<5} | {'Δb_inf':>8} | {'SE':>7} | {'2SE':>7} | "
          f"{'T2':>4} | {'Δq':>7} | {'SE':>7} | {'2SE':>7} | {'T3':>4} | "
          f"{'D斜率':>8} | {'2SE':>7} | {'T4':>4}")
    print(" " + "-" * 108)
    verdict = {}
    for nd in news:
        t = nd["t"]
        for fid in DENSE:
            r = res.get((t, fid))
            if r is None or len(r["db"]) < 100:
                print(f" {t:>7g} | {fid:<5} | bootstrap 失败")
                verdict[(t, fid)] = None
                continue
            mb, sb = float(r["db"].mean()), float(r["db"].std())
            mq, sq = float(r["dq"].mean()), float(r["dq"].std())
            ms, ss = float(r["sl"].mean()), float(r["sl"].std())
            p2 = abs(mb) <= 2 * sb
            p3 = abs(mq) <= 2 * sq
            p4 = abs(ms) <= 2 * ss
            verdict[(t, fid)] = (p2, p3, p4)
            print(f" {t:>7g} | {fid:<5} | {mb:>+8.3f} | {sb:>7.3f} | "
                  f"{2*sb:>7.3f} | {'✅' if p2 else '❌':>4} | {mq:>+7.3f} | "
                  f"{sq:>7.3f} | {2*sq:>7.3f} | {'✅' if p3 else '❌':>4} | "
                  f"{ms:>+8.4f} | {2*ss:>7.4f} | {'✅' if p4 else '❌':>4}")
    print(" " + "-" * 108)
    print(" Δb_inf / Δq 是（新 t）-（t=0.002）；2SE 就是 T6 要求的可探测下限。")

    # ---------- D(u) 的形状 ----------
    print("\n" + "=" * 104)
    print(" 4. D(u) = logΛ_t(u) - logΛ_0.002(u)（逐图配对差再取均值）：H0 要求与 u 无关")
    print("=" * 104)
    print(" 实现说明：个别图的 Λ 可能 <= 0（先验偏爱清晰图），此时 log 无定义，")
    print(" 该图在该 u 上被排除并计数；括号外另给 log(均值) 版本作对照。")
    for nd in news:
        t = nd["t"]
        print(f"\n --- t={t:g} ---")
        print(f" {'u':>6} | " + " | ".join(f"{f:>10}" for f in DENSE) +
              f" | {'三族极差':>9} | {'排除图数':>8}")
        print(" " + "-" * (10 + 13 * len(DENSE) + 22))
        for i, u in enumerate(ug):
            row, vals, nex = [], [], 0
            for fid in DENSE:
                P1 = curves[(t, fid)]["P"]
                P0 = base[fid]["P"]
                D, Dl, cnt = Dvec(P0, P1, nug)
                d = float(D[i])
                vals.append(d)
                nex += int(P0.shape[0] - cnt[i])
                row.append(f"{d:>10.4f}")
            print(f" {u:>6.3f} | " + " | ".join(row) +
                  f" | {max(vals)-min(vals):>9.4f} | {nex:>8}")
        print(" " + "-" * (10 + 13 * len(DENSE) + 22))
        rr = [res.get((t, f)) for f in DENSE]
        rr = [r for r in rr if r is not None and len(r["rD"]) > 100]
        if rr:
            mr = float(np.mean([r["rD"].mean() for r in rr]))
            sr = float(np.mean([r["rD"].std() for r in rr]))
            print(f" D 的全 u 跨度：均值 {mr:+.4f} dex，bootstrap SD {sr:.4f}"
                  f"（H0 下应 ≈ 0；2SD = {2*sr:.4f}）")

    # ---------- T5b：定量高斯收敛预言 ----------
    print("\n" + "=" * 104)
    print(" 4b. T5b 定量高斯收敛预言（修正案 2026-10-18 22:02，md5 e31c14db...）")
    print("=" * 104)
    print(" 模型：Λ = Λ_Gauss + Λ_ng，b_Gauss(u) = -u/(1-u) 与 t 无关，b_ng(u) 与 t 无关；")
    print("       b_ng 由 t=0.002 基线自校准，于是新 t 的斜率被完全预言。")
    print(f" {'t':>7} | {'族':<5} | {'rms|b-b_pred|':>14} | {'均值偏差':>9} | "
          f"{'b 范围':>16} | {'b_pred 范围':>16} | {'T5b':>5}")
    print(" " + "-" * 92)
    t5b = {}
    for nd in news:
        t = nd["t"]
        for fid in DENSE:
            G0 = base[fid]["G"]
            L0 = np.array([np.mean(base[fid]["P"][:, i]) for i in range(len(ug))])
            b0 = np.array([base[fid]["b"](u) for u in ug])
            bG = np.array([gauss_b(u) for u in ug])
            Gn = np.array([nd["G"].get(fid, {}).get(round(u, 4), np.nan)
                           for u in ug])
            Ln = np.array([np.mean(curves[(t, fid)]["P"][:, i])
                           for i in range(len(ug))])
            if not (np.isfinite(G0).all() and np.isfinite(Gn).all()):
                print(f" {t:>7g} | {fid:<5} | 缺 lam_gauss")
                t5b[(t, fid)] = None
                continue
            # b_ng 自校准（t=0.002 处 Λ_ng/Λ_G ≈ 74）
            ng0 = L0 - G0
            if (ng0 <= 0).any():
                print(f" {t:>7g} | {fid:<5} | 基线 Λ_ng <= 0，无法自校准")
                t5b[(t, fid)] = None
                continue
            bng = (L0 * b0 - G0 * bG) / ng0
            ngn = Ln - Gn
            bpred = (Gn * bG + ngn * bng) / Ln
            bmeas = np.array([curves[(t, fid)]["b"](u) for u in ug])
            dd = bmeas - bpred
            r = res.get((t, fid))
            se = float(r["db"].std()) if r is not None and len(r["db"]) > 100 \
                else float("nan")
            rms = float(np.sqrt(np.mean(dd ** 2)))
            okrms = rms <= 0.10
            okmean = (np.isfinite(se) and abs(float(dd.mean())) <= 2 * se)
            ok = bool(okrms and okmean)
            t5b[(t, fid)] = dict(rms=rms, mean=float(dd.mean()), ok=ok,
                                 okrms=bool(okrms), okmean=bool(okmean),
                                 bpred=bpred, dd=dd)
            print(f" {t:>7g} | {fid:<5} | {rms:>14.3f} | {dd.mean():>+9.3f} | "
                  f"[{bmeas.min():>6.2f},{bmeas.max():>6.2f}]"
                  f" | [{bpred.min():>6.2f},{bpred.max():>6.2f}] | "
                  f"{'✅' if ok else '❌':>5}")
    print(" " + "-" * 92)
    print(" 容差跑前写定：rms <= 0.10 且 |均值| <= 2*SE_bootstrap。")

    # ---------- 汇总 ----------
    print("\n" + "=" * 104)
    print(" 5. 汇总")
    print("=" * 104)
    print(f" {'t':>7} | {'T1':>4} | {'T2':>4} | {'T3':>4} | {'T4':>4} | {'T5b':>5} | "
          f"{'b_inf':>8} | {'q':>7} | {'Δb_inf':>8} | {'Δq':>7} | "
          f"{'2SE(b)':>7} | {'2SE(q)':>7} | {'T5b rms':>8}")
    print(" " + "-" * 108)
    allpass = True
    for nd in news:
        t = nd["t"]
        for fid in DENSE:
            cf = curves[(t, fid)]["cf"]
            p1 = cf is not None and cf["rms"] <= 2 * PRE_RMS[fid]
            r = res.get((t, fid))
            if r is None or len(r["db"]) < 100 or cf is None:
                print(f" {t:>7g} | {'--':>4}")
                allpass = False
                continue
            p2, p3, p4 = verdict[(t, fid)]
            e5 = t5b.get((t, fid))
            p5 = bool(e5 and e5["ok"])
            allpass = allpass and p1 and p2 and p3 and p4 and p5
            b0 = base[fid]["cf"]["p"][0]
            q0 = base[fid]["cf"]["p"][2]
            print(f" {t:>7g} | {'✅' if p1 else '❌':>4} | "
                  f"{'✅' if p2 else '❌':>4} | {'✅' if p3 else '❌':>4} | "
                  f"{'✅' if p4 else '❌':>4} | {'✅' if p5 else '❌':>5} | "
                  f"{cf['p'][0]:>8.3f} | "
                  f"{cf['p'][2]:>7.3f} | {cf['p'][0]-b0:>+8.3f} | "
                  f"{cf['p'][2]-q0:>+7.3f} | {2*float(r['db'].std()):>7.3f} | "
                  f"{2*float(r['dq'].std()):>7.3f} | "
                  f"{(e5['rms'] if e5 else float('nan')):>8.3f}")
    print(" " + "-" * 108)

    # T5 方向性（附带、非判决）
    print("\n T5 方向性（附带，非判决；预官已记其量级与噪声同量级）:")
    for nd in news:
        t = nd["t"]
        for fid in DENSE:
            r = res.get((t, fid))
            cf = curves[(t, fid)]["cf"]
            if r is None or len(r["db"]) < 100 or cf is None:
                continue
            mb, sb = float(r["db"].mean()), float(r["db"].std())
            mq, sq = float(r["dq"].mean()), float(r["dq"].std())
            sig = abs(mb) > 2 * sb
            d = "符合" if mb > 0 else "不符"
            d2 = "符合" if mq < 0 else "不符"
            print(f"   t={t:<6g} {fid:<5} Δb_inf={mb:>+7.3f}"
                  f"（预言 >0，{d}{'，且 >2SE' if sig else '，未达 2SE'}）；"
                  f"Δq={mq:>+7.3f}（预言 <0，{d2}）")
    print("\n " + ("T1–T4 + T5b 全部通过 => b(u) 与 t 无关，且其全部 t 漂移可由"
                   "「高斯分量 + 非高斯分量」两分量混合算出。" if allpass else
                   "有未通过项 => 记录失败。按项目规矩，不为失败的预言改阈值。"))
    print(" 注意：即使全通过，也必须连同 T6 的可探测下限一起报告——"
          "「测不到」不等于「没有」。")

    # ---------- 落盘供画图 ----------
    if args.dump:
        out = dict(ug=ug, dense=DENSE, deg=DEG, nboot=NBOOT,
                   pre_rms=PRE_RMS, base={}, new=[])
        for fid in DENSE:
            out["base"][fid] = dict(
                p=[float(v) for v in base[fid]["cf"]["p"]],
                rms=float(base[fid]["cf"]["rms"]),
                b=[base[fid]["b"](u) for u in ug])
        for nd in news:
            t = nd["t"]
            ent = dict(t=t, fn=nd["fn"], fam={})
            for fid in DENSE:
                c = curves[(t, fid)]
                r = res.get((t, fid))
                e = dict(p=([float(v) for v in c["cf"]["p"]]
                            if c["cf"] else None),
                         rms=(float(c["cf"]["rms"]) if c["cf"] else None),
                         b=([c["b"](u) for u in ug] if c["b"] else None))
                if r is not None and len(r["db"]) > 100:
                    e.update(se_binf=float(r["db"].std()),
                             se_q=float(r["dq"].std()),
                             d_binf=float(r["db"].mean()),
                             d_q=float(r["dq"].mean()),
                             se_slope=float(r["sl"].std()),
                             d_slope=float(r["sl"].mean()))
                P1, P0 = c["P"], base[fid]["P"]
                D, Dl, cnt = Dvec(P0, P1, nug)
                e["D"] = [float(v) for v in D]
                e["Dlogmean"] = [float(v) for v in Dl]
                e["nexcl"] = [int(P0.shape[0] - v) for v in cnt]
                if e5 is not None:
                    e["bpred"] = [float(v) for v in e5["bpred"]]
                    e["dbpred"] = [float(v) for v in e5["dd"]]
                ent["fam"][fid] = e
            out["new"].append(ent)
        p = os.path.join(OUT, args.dump)
        json.dump(out, open(p, "w"))
        print(f"\n [已存] {p}")


if __name__ == "__main__":
    main()
