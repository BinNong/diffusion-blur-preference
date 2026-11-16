"""b(u) 是否随**先验**（模型）变？裁判：M1-M7，判据跑前写定在 sweep_law_test.py 第九组。

预官来源：sweep_law_test.py docstring 第九组（2026-10-19 12:19，跑前写定；
          md5 = 774d9acdc4f5d875dafad74930191925）。

设计：2x2 因子表缺一格（ffhq+ffhq = 第八组基线）。
  格子 A  ffhq 先验 + ffhq 图池   = sweep_bdense + sweep_blow（已有，t=0.002）
  格子 B  afhqv2 先验 + afhqv2 图池 = sweep_m_afq_afq.json（新）
  格子 C  ffhq 先验 + afqhv2 图池  = sweep_m_ffq_afq.json（新）
**格子 B − 格子 C = 固定图像下的先验效应**（同一批 20 张 afhqv2 图，只换先验）——配对。
格子 C − 格子 A = 固定先验下的图像效应（两组图不同，故 SE 按独立合成）。

判据：
  M1 守门：两格闭式 rms <= 2 倍基线（0.032/0.094/0.074）。
  M2 主判据：固定图像下 |b∞^B − b∞^C| 与配对 bootstrap 2SE 比。预期测得出（物理预期）。
  M3 同上对 q。
  M4 无模型：D_prior(u)=logΛ_B(u)−logΛ_C(u) 对 log u 的斜率 <= 2SE。
  M5 图像对照：报格子 C − 格子 A 的 b∞ 变化，防止"先验效应"其实只是"任何扰动都动"。
  M6 符号翻转（附带）：若 b_ng 与先验无关，则固定图像时
       Δb(u) = (b_Gauss(u) − b_ng,C(u)) · Λ_Gauss · (1/Λ_B − 1/Λ_C)
     而 b_Gauss−b_ng 在低 u 为 +0.52、高 u 为 −1.39（反号）=> Δb(u) 应低/高 u 反号。
     b_ng,C 用**格子 C 自校准**（B/C 同图同池，Λ_Gauss 完全相同）。
  M7 可探测下限：每格报 2SE(b∞)、2SE(q)。

用法： venv2/bin/python analyze_model.py
"""

import argparse
import glob
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
PRE_RMS = {"W2": 0.016, "W8": 0.047, "TK50": 0.037}


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
    sets = [{round(u, 4) for u, _, _ in prof[f]} for f in fids]
    ug_all = sorted(set.intersection(*sets), reverse=True)
    return ug_all, [u for u in ug_all if lo <= u <= hi]


def matrices(prof, fids, ug_all):
    """原始 Λ 矩阵 (nimg, nu)。不取 log、不夹非正值——非正 Λ 单独显式处理。"""
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
        out[fid] = np.array(cols).T if ok else None
    return out


def bcurve(P, us_all):
    if P is None:
        return None
    mu = P.mean(0)
    if not (mu > 0).all():
        bad = [round(float(u), 4) for u, v in zip(us_all, mu) if v <= 0]
        print(f"   [警告] 有 {len(bad)} 个 u 的均值 Λ<=0：{bad}")
        return None
    try:
        c = np.polyfit(np.log(np.array(us_all)), np.log(mu), DEG)
    except Exception:
        return None
    d = np.polyder(c)
    return lambda u: float(np.polyval(d, math.log(u)))


def closed_form(us, bfun):
    from scipy.optimize import curve_fit
    f1 = lambda u, *p: p[0] + p[1] * u ** p[2]
    ua = np.array(us)
    bv = np.array([bfun(u) for u in us])
    try:
        popt, _ = curve_fit(f1, ua, bv, p0=(-0.7, -6.0, 3.0),
                            bounds=([-1e3, -50, 0.1], [1e3, 50, 12]),
                            maxfev=100000)
    except Exception:
        return None
    return dict(p=popt, rms=float(np.sqrt(np.mean((bv - f1(ua, *popt)) ** 2))),
                bv=bv)


def gauss_b(u):
    return -u / (1.0 - u)


def boot_cell(P, ug, ug_all, rng):
    """一次 bootstrap：返回该格的闭式参数（失败返回 None）。"""
    try:
        cc = np.polyfit(np.log(np.array(ug_all)), np.log(P.mean(0)), DEG)
        d = np.polyder(cc)
        return closed_form(ug, lambda u, d=d: float(np.polyval(d, math.log(u))))
    except Exception:
        return None


def boot_b(P, ug_all, u):
    """一次 bootstrap 的 b(u)（单个点）。失败返回 nan。"""
    try:
        cc = np.polyfit(np.log(np.array(ug_all)), np.log(P.mean(0)), DEG)
        return float(np.polyval(np.polyder(cc), math.log(u)))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cellB", default="sweep_m_afq_afq.json")
    ap.add_argument("--cellC", default="sweep_m_ffq_afq.json")
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    # ---------- 三格数据 ----------
    ra = load("sweep_bdense.json")
    if ra is None:
        print("[缺少] sweep_bdense.json")
        return
    pa, ga = dict(ra[1]), dict(ra[2])
    rl = load("sweep_blow.json")
    if rl is not None:
        for fid, rows in rl[1].items():
            pa.setdefault(fid, []).extend(rows)
        for k in pa:
            pa[k].sort(key=lambda c: -c[0])
        for fid, gg in rl[2].items():
            ga.setdefault(fid, {}).update(gg)

    cells = {}
    for name, fn in [("A(ffq+ffq)", None), ("B(afq+afq)", args.cellB),
                     ("C(ffq+afq)", args.cellC)]:
        if fn is None:
            prof, gpr = pa, ga
            cfg = dict(nimg=20, t=0.002, ckpt="edm-ffhq-64x64-uncond-vp.pkl",
                       pool="/tmp/pool_ffhq64_256_s0.pt")
        else:
            r = load(fn)
            if r is None:
                print(f"[缺少] {fn}")
                return
            prof, gpr, cfg = dict(r[1]), dict(r[2]), r[0]["cfg"]
        ug_all, ug = grid(prof, DENSE)
        assert ug_all[:len(ug)] == ug, f"{name}: ug 不是 ug_all 前缀"
        P = matrices(prof, DENSE, ug_all)
        cells[name] = dict(cfg=cfg, ug_all=ug_all, ug=ug, P=P, G=gpr)
        print(f"[{name}] {os.path.basename(str(cfg.get('pool')))}  "
              f"ckpt={cfg.get('ckpt')}  t={cfg.get('t')}  "
              f"{len(ug)} 个拟合 u 点")

    A, B, C = (cells["A(ffq+ffq)"], cells["B(afq+afq)"],
               cells["C(ffq+afq)"])
    for nm, c in [("A", A), ("B", B), ("C", C)]:
        if any(c["P"][f] is None for f in DENSE):
            print(f"[错误] 格子 {nm} 有点缺失")
            return

    # ---------- 表 1：各格的 b(u) 与闭式 ----------
    print("\n" + "=" * 104)
    print(" 1. 各格的 b(u)（5 次多项式解析导数）与闭式 b=b_inf+A*u^q")
    print("=" * 104)
    print(f" {'格子':<12} | {'族':<5} | {'b_inf':>8} | {'A':>8} | {'q':>7} | "
          f"{'rms':>7} | {'T1上限':>7} | {'M1':>4}")
    print(" " + "-" * 82)
    fit = {}
    for nm, c in [("A(ffq+ffq)", A), ("B(afq+afq)", B), ("C(ffq+afq)", C)]:
        for fid in DENSE:
            bf = bcurve(c["P"][fid], c["ug_all"])
            cf = closed_form(c["ug"], bf) if bf else None
            fit[(nm, fid)] = dict(b=bf, cf=cf)
            lim = 2 * PRE_RMS[fid]
            ok = cf is not None and cf["rms"] <= lim
            if cf is None:
                print(f" {nm:<12} | {fid:<5} | {'拟合失败':>8}")
            else:
                print(f" {nm:<12} | {fid:<5} | {cf['p'][0]:>8.3f} | "
                      f"{cf['p'][1]:>8.3f} | {cf['p'][2]:>7.3f} | "
                      f"{cf['rms']:>7.3f} | {lim:>7.3f} | {'✅' if ok else '❌':>4}")

    # ---------- 表 2：b(u) 逐点 ----------
    print("\n" + "=" * 104)
    print(" 2. b(u) 逐点（括号内为相对格子 A 的变化）")
    print("=" * 104)
    ug = A["ug"]
    for fid in DENSE:
        print(f" --- {fid} ---")
        print(f" {'u':>6} | {'b_Gauss':>8} | {'A':>8} | {'B':>16} | {'C':>16}")
        print(" " + "-" * 62)
        for u in ug:
            bA = fit[("A(ffq+ffq)", fid)]["b"](u)
            bB = fit[("B(afq+afq)", fid)]["b"](u)
            bC = fit[("C(ffq+afq)", fid)]["b"](u)
            print(f" {u:>6.3f} | {gauss_b(u):>8.3f} | {bA:>8.3f} | "
                  f"{bB:>8.3f} ({bB-bA:>+6.3f}) | {bC:>8.3f} ({bC-bA:>+6.3f})")
        print(" " + "-" * 62)

    # ---------- 表 3：配对 bootstrap ----------
    print("\n" + "=" * 104)
    print(" 3. 配对 bootstrap（2000 次；B−C 同图同池故真正配对，C−A 图不同故独立合成）")
    print("=" * 104)
    rng = np.random.default_rng(0)
    nimg = A["P"][DENSE[0]].shape[0]
    res = {}
    for fid in DENSE:
        PA, PB, PC = (A["P"][fid], B["P"][fid], C["P"][fid])
        dBC_b, dBC_q, dCA_b, dCA_q, dBA_b, sl = [], [], [], [], [], []
        dBC_curve = []
        for _ in range(NBOOT):
            idx = rng.integers(0, nimg, nimg)
            idxA = rng.integers(0, nimg, nimg)     # A 的图与 B/C 不同，独立重采样
            cA = boot_cell(PA[idxA], ug, A["ug_all"], rng)
            cB = boot_cell(PB[idx], ug, B["ug_all"], rng)
            cC = boot_cell(PC[idx], ug, C["ug_all"], rng)
            if cA is None or cB is None or cC is None:
                continue
            dBC_b.append(cB["p"][0] - cC["p"][0])
            dBC_q.append(cB["p"][2] - cC["p"][2])
            dCA_b.append(cC["p"][0] - cA["p"][0])
            dCA_q.append(cC["p"][2] - cA["p"][2])
            dBA_b.append(cB["p"][0] - cA["p"][0])
            # 每个 u 上的先验效应（配对，同一批图）
            bB = np.array([boot_b(PB[idx], B["ug_all"], u) for u in ug])
            bC = np.array([boot_b(PC[idx], C["ug_all"], u) for u in ug])
            if np.isfinite(bB).all() and np.isfinite(bC).all():
                dBC_curve.append(bB - bC)
            # M4：固定图像下 D(u)=logΛ_B−logΛ_C 的斜率
            DB = np.log(PB[idx]).mean(0)[:len(ug)] - \
                np.log(PC[idx]).mean(0)[:len(ug)]
            g = np.isfinite(DB)
            if g.sum() >= 5:
                try:
                    sl.append(float(np.polyfit(np.log(np.array(ug))[g],
                                               DB[g], 1)[0]))
                except Exception:
                    pass
        res[fid] = dict(dBC_b=np.array(dBC_b), dBC_q=np.array(dBC_q),
                        dCA_b=np.array(dCA_b), dCA_q=np.array(dCA_q),
                        dBA_b=np.array(dBA_b), sl=np.array(sl),
                        dBC_curve=(np.array(dBC_curve)
                                   if dBC_curve else None))
    print(f" {'族':<5} | {'Δb∞(B−C)':>10} | {'SE':>7} | {'2SE':>7} | {'M2':>4} | "
          f"{'Δq(B−C)':>9} | {'2SE':>7} | {'M3':>4} | {'D斜率':>8} | {'2SE':>7} | "
          f"{'M4':>4}")
    print(" " + "-" * 96)
    verdict = {}
    for fid in DENSE:
        r = res[fid]
        if len(r["dBC_b"]) < 100:
            print(f" {fid:<5} | bootstrap 失败")
            verdict[fid] = None
            continue
        mb, sb = float(r["dBC_b"].mean()), float(r["dBC_b"].std())
        mq, sq = float(r["dBC_q"].mean()), float(r["dBC_q"].std())
        ms, ss = float(r["sl"].mean()), float(r["sl"].std())
        p2 = abs(mb) > 2 * sb          # M2 预期"测得出"
        p3 = abs(mq) > 2 * sq
        p4 = abs(ms) > 2 * ss          # M4 零假设是先验无关，测得出即否掉
        verdict[fid] = (p2, p3, p4)
        print(f" {fid:<5} | {mb:>+10.3f} | {sb:>7.3f} | {2*sb:>7.3f} | "
              f"{'测得出' if p2 else '测不出':>4} | {mq:>+9.3f} | {2*sq:>7.3f} | "
              f"{'测得出' if p3 else '测不出':>4} | {ms:>+8.4f} | {2*ss:>7.4f} | "
              f"{'测得出' if p4 else '测不出':>4}")
    print(" " + "-" * 96)
    print(" M2/M3/M4 的预期都是**测得出**（物理预期：b(u) 已证明对 t 敏感，"
          "没有理由对先验特殊地不敏感）。")

    # ---------- 表 4：M5 图像对照 ----------
    print("\n" + "=" * 104)
    print(" 3b. 每个 u 上的先验效应 Δb(u)=b_B(u)−b_C(u)（配对 bootstrap 的 2SE 带）")
    print("=" * 104)
    print(f" {'u':>6} | " + " | ".join(f"{f:>18}" for f in DENSE))
    print(" " + "-" * (10 + 21 * len(DENSE)))
    curve_est = {}
    for fid in DENSE:
        r = res[fid]
        bB = np.array([fit[("B(afq+afq)", fid)]["b"](u) for u in ug])
        bC = np.array([fit[("C(ffq+afq)", fid)]["b"](u) for u in ug])
        est = bB - bC
        sd = (r["dBC_curve"].std(0) if r["dBC_curve"] is not None
              else np.full(len(ug), np.nan))
        curve_est[fid] = (est, sd)
    for i, u in enumerate(ug):
        row = []
        for fid in DENSE:
            est, sd = curve_est[fid]
            row.append(f"{est[i]:>+8.3f} ±{2*sd[i]:.3f}")
        print(f" {u:>6.3f} | " + " | ".join(row))
    print(" " + "-" * (10 + 21 * len(DENSE)))
    print(" H0（先验无关）要求每一行都在 0 的 ±2SE 内。")

    # ---------- 表 4：M5 图像对照 ----------
    print("\n" + "=" * 104)
    print(" 4. M5 图像对照：固定先验、只换图池（格子 C − 格子 A），以及总效应 B − A")
    print("=" * 104)
    print(f" {'族':<5} | {'Δb∞(C−A) 图像':>15} | {'SE(独立合成)':>13} | "
          f"{'Δb∞(B−A) 总':>13} | {'图像/先验 比':>12}")
    print(" " + "-" * 74)
    for fid in DENSE:
        r = res[fid]
        if len(r["dCA_b"]) < 100:
            continue
        mc, sc = float(r["dCA_b"].mean()), float(r["dCA_b"].std())
        mba, sba = float(r["dBA_b"].mean()), float(r["dBA_b"].std())
        # C−A 的 SE：A 与 C 图不同 => 独立合成（bootstrap 里已用独立 idx）
        ratio = abs(mc) / abs(float(r["dBC_b"].mean())) \
            if abs(float(r["dBC_b"].mean())) > 1e-9 else float("nan")
        print(f" {fid:<5} | {mc:>+15.3f} | {sc:>13.3f} | {mba:>+13.3f} | "
              f"{ratio:>12.2f}")
    print(" " + "-" * 74)
    print(" 若「图像效应」与「先验效应」同量级或更大，结论必须写成"
          "「b(u) 对一切扰动都敏感」，不能只说「依赖先验」。")

    # ---------- 表 5：M6 符号翻转 ----------
    print("\n" + "=" * 104)
    print(" 5. M6（附带）：若 b_ng 与先验无关，Δb(u) 应低 u 与高 u 反号")
    print("=" * 104)
    print(" 模型（固定图像时 Λ_Gauss 完全相同，b_ng,C 由格子 C 自校准）:")
    print("   Δb(u) = (b_Gauss(u) − b_ng,C(u)) · Λ_Gauss · (1/Λ_B − 1/Λ_C)")
    print(f" {'族':<5} | {'u':>6} | {'Δb 实测':>9} | {'Δb 模型':>9} | "
          f"{'b_G−b_ng':>9} | {'1/Λ_B−1/Λ_C':>13} | {'符号':>6}")
    print(" " + "-" * 76)
    for fid in DENSE:
        PB, PC, GC = B["P"][fid], C["P"][fid], C["G"].get(fid, {})
        Lb = np.array([PB[:, i].mean() for i in range(len(ug))])
        Lc = np.array([PC[:, i].mean() for i in range(len(ug))])
        bC = np.array([fit[("C(ffq+afq)", fid)]["b"](u) for u in ug])
        bB = np.array([fit[("B(afq+afq)", fid)]["b"](u) for u in ug])
        bG = np.array([gauss_b(u) for u in ug])
        Gn = np.array([GC.get(round(u, 4), np.nan) for u in ug])
        if not np.isfinite(Gn).all() or (Lc - Gn <= 0).any():
            print(f" {fid:<5} | 缺 lam_gauss 或 Λ_ng<=0，跳过")
            continue
        bng = (Lc * bC - Gn * bG) / (Lc - Gn)
        pred = (bG - bng) * Gn * (1.0 / Lb - 1.0 / Lc)
        meas = bB - bC
        for i in [0, len(ug) // 2, len(ug) - 1]:
            u = ug[i]
            sg = "一致" if (meas[i] * pred[i] > 0 or
                            abs(meas[i]) < 1e-6 or abs(pred[i]) < 1e-6) else "反号"
            print(f" {fid:<5} | {u:>6.3f} | {meas[i]:>+9.3f} | {pred[i]:>+9.3f} | "
                  f"{bG[i]-bng[i]:>+9.3f} | {1/Lb[i]-1/Lc[i]:>13.3e} | {sg:>6}")
        lo, hi = meas[0], meas[-1]
        print(f"   => {fid}: Δb 在 u={ug[0]:.2f} 处 {lo:+.3f}、"
              f"u={ug[-1]:.2f} 处 {hi:+.3f}；"
              f"{'低/高 u 反号，与 M6 模型的方向特征一致' if lo*hi < 0 else '未反号'}")
        print(" " + "-" * 76)

    # ---------- 汇总 ----------
    print("\n" + "=" * 104)
    print(" 6. 汇总")
    print("=" * 104)
    print(f" {'族':<5} | {'M1':>4} | {'M2':>6} | {'M3':>6} | {'M4':>6} | "
          f"{'b∞ B':>8} | {'b∞ C':>8} | {'b∞ A':>8} | {'Δ(B−C)':>9} | "
          f"{'2SE':>7} | {'2SE(q)':>8}")
    print(" " + "-" * 90)
    allpass = True
    for fid in DENSE:
        cB = fit[("B(afq+afq)", fid)]["cf"]
        cC = fit[("C(ffq+afq)", fid)]["cf"]
        cA = fit[("A(ffq+ffq)", fid)]["cf"]
        r = res[fid]
        p1 = (cB is not None and cB["rms"] <= 2 * PRE_RMS[fid]
              and cC is not None and cC["rms"] <= 2 * PRE_RMS[fid])
        ok = verdict[fid] is not None and all(verdict[fid])
        allpass = allpass and p1 and ok
        v = verdict[fid]
        print(f" {fid:<5} | {'✅' if p1 else '❌':>4} | "
              f"{('测得出' if v[0] else '测不出') if v else '--':>6} | "
              f"{('测得出' if v[1] else '测不出') if v else '--':>6} | "
              f"{('测得出' if v[2] else '测不出') if v else '--':>6} | "
              f"{cB['p'][0]:>8.3f} | {cC['p'][0]:>8.3f} | {cA['p'][0]:>8.3f} | "
              f"{float(r['dBC_b'].mean()):>+9.3f} | "
              f"{2*float(r['dBC_b'].std()):>7.3f} | "
              f"{2*float(r['dBC_q'].std()):>8.3f}")
    print(" " + "-" * 90)
    print("\n " + ("M1–M4 全过 => b(u,t) 读的是先验谱，不是 ODE 框架——第八组想要的 payoff。"
                   if allpass else
                   "有未通过项 => 记录结果。按项目规矩，不为预言改阈值。"))
    print(" 若 M2/M4 测不出 => 在所测精度内 b(u) 与先验无关（附 2SE 下限），"
          "则 b(u) 的 t 依赖性来自 ODE 积分本身——同样有信息量。")
    print(" 即使测得出，也必须连同 M5 的图像效应一起报告。")

    # ---------- 落盘供画图 ----------
    if args.dump:
        out = dict(ug=ug, dense=DENSE, pre_rms=PRE_RMS,
                   cells={}, verdict={})
        for nm in ["A(ffq+ffq)", "B(afq+afq)", "C(ffq+afq)"]:
            c = cells[nm]
            out["cells"][nm] = dict(
                ckpt=c["cfg"].get("ckpt"), pool=c["cfg"].get("pool"),
                fam={fid: dict(
                    p=[float(v) for v in fit[(nm, fid)]["cf"]["p"]],
                    rms=float(fit[(nm, fid)]["cf"]["rms"]),
                    b=[fit[(nm, fid)]["b"](u) for u in ug],
                    lam=[float(c["P"][fid][:, i].mean())
                         for i in range(len(ug))],
                    lg=[float(c["G"].get(fid, {}).get(round(u, 4), np.nan))
                        for u in ug],
                ) for fid in DENSE})
        for fid in DENSE:
            r = res[fid]
            est, sd = curve_est[fid]
            out["verdict"][fid] = dict(
                d_bc=float(r["dBC_b"].mean()), se_bc=float(r["dBC_b"].std()),
                d_q=float(r["dBC_q"].mean()), se_q=float(r["dBC_q"].std()),
                d_ca=float(r["dCA_b"].mean()), se_ca=float(r["dCA_b"].std()),
                d_ba=float(r["dBA_b"].mean()),
                d_slope=float(r["sl"].mean()), se_slope=float(r["sl"].std()),
                curve=[float(v) for v in est], curve_2se=[float(2 * v)
                                                          for v in sd])
        p = os.path.join(OUT, args.dump)
        json.dump(out, open(p, "w"))
        print(f"\n [已存] {p}")


if __name__ == "__main__":
    main()
