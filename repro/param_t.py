"""把机理变成可预言的公式：Λ(t,u) = Λ_0(u) − G(t,u)（离线，零 GPU）。

预官（2026-10-20，跑前写定；md5 由运行记录给出）
================================================================================
背景：§3.9(w) 用恒等式 Λ = Λ_B − Λ_S 定位了机理（Λ 只是 Λ_B 的 1.3–2.6%，两项几乎同速
衰减，差额全在差分项且随算子强度单调增长）。但那是定性机理 + 一个量级核对，不是公式。

本步把它变成公式。恒等式（从 likelihood_fn 的结构读出的，不是模型）
--------------------------------------------------------------------
  prior.py 的 likelihood_fn 给出
      log p_t(y) = A(y) + ∫_t^{σ_max} ∇·f ds
  其中 A(y) = -(d/2)log(2π σ_max²) - ||z_T||²/(2σ_max²) **与 t 无关**
  （σ_max = 80 固定）。所以 log p_t 的全部 t 依赖性来自散度积分的下限。

  定义 Δ(y; t) ≡ log p_{t0}(y) − log p_t(y) = ∫_{t0}^{t} ∇·f ds（t0 = 0.002），则
      Λ(t,u) = Λ(t0,u) − G(t,u),     G(t,u) = Δ_B(t,u) − Δ_S(t)
  即"偏好的衰减 = 模糊图与清晰图各自散度积分增量之差"。**这是恒等式。**

高斯对照（可从已存数据算，无需重跑）
------------------------------------
  若先验是 N(0,S)，则 log p_t(y) − log p_{t0}(y) =
     ½ Σ_f log((S_f+t²)/(S_f+t0²))            ← 归一化项（与 y 无关）
   + ½ Σ_f |Y_f|² [1/(S_f+t0²) − 1/(S_f+t²)]  ← 数据拟合项
  第二项正是 Λ_Gauss(t0,u) − Λ_Gauss(t,u)（已存）。第一项需要 S_f，从图池算。

**判据（跑前写定）**：
  E1 守门：恒等式 Λ(t,u) = Λ_0(u) − G(t,u) 在存过的均值上成立到 1e-6 相对误差。
     （纯实现自检；不过则后面全无效。）
  E2 可分性：G(t,u) 是否 = g(t)·h(u)？对每个族的 5×19 矩阵做秩 1 逼近，
     要求残差 rms <= 2 ×（Λ 的相对 2SE）。秩 1 是"两函数参数化"的前提。
  E3 h(u) 的形式（若 E2 过）：候选 (a) h ∝ 1−u（算子总衰减，物理意义最明确）；
     (b) h ∝ u^γ；(c) h ∝ Λ_0(u)（**这一项先验上被排除**——若 h ∝ Λ_0 则 Λ(t,u)=Λ_0(u)ρ(t)，
     b(u) 就与 t 无关，而 §3.9(t) 已实测否掉；列为反向对照）。
     报三个的 rms，最小的胜出。
  E4 g(t) 的形式（若 E2 过）：候选 (a) g ∝ log(t/t0)；(b) g ∝ t^α；(c) g ∝ t²；
     (d) g ∝ d_eff(t) = Σ_f S_f/(S_f+t²)（先验在 σ=t 处的"有效维度"，物理动机：
     散度积分沿路径累积，能累积的长度由多少频率还在噪声地板之上决定）。
  E5 高斯对照（**不依赖 E2 是否过，独立判决**）：
     E5a Δ_S(t) 与高斯归一化项 ½Σ_f log((S_f+t²)/(S_f+t0²)) 的比；
     E5b G(t,u) 与 [Λ_Gauss(t0,u) − Λ_Gauss(t,u)] 的比。
     若 E5b 的比在全部 (t,u) 上近似常数 => 非高斯部分的衰减是高斯部分衰落的固定倍数，
     这是一条强的结构陈述；若比随 t 或 u 系统变化 => 两者不是简单比例。
  E6 若 E2/E3/E4 都成：Λ(t,u) = Λ_0(u) − g(t)h(u) 是**两函数**参数化
     （u 侧两个函数 Λ_0 与 h，t 侧一个函数 g）。在全部 5 t × 19 u × 3 族上检验，
     容差 = 2 × Λ 的相对 bootstrap SE。与 §3.9(v) 否掉的四参数 family 对比。

若 E1–E4 + E6 过 => 拿到 b(u,t) 背后第一条两函数参数化，且它的每一项有物理名字。
若 E2 不过 => G 不可分离，记录失败并报秩 1 残差随 t 的变化。
若 E5b 成立而 E2 不过 => 至少拿到"非高斯衰落 ∝ 高斯衰落"这条。

用法： venv2/bin/python param_t.py
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
T0 = 0.002


def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    prof = defaultdict(dict)
    for rec in d["rows"]:
        fid = rec["meta"]["fid"]
        prof[fid][round(1.0 - rec["kunif"], 4)] = dict(
            lam=rec["lam_mean"], lg=rec["lam_gauss"],
            ls=rec["logp_sharp_mean"], lb=rec["logp_blur_mean"])
    return d, {k: dict(v) for k, v in prof.items()}


def grid(prof, fids, lo=0.13, hi=0.86):
    sets = [set(prof[f]) for f in fids]
    ug_all = sorted(set.intersection(*sets), reverse=True)
    return [u for u in ug_all if lo <= u <= hi]


def pool_spectrum():
    """图池的平均功率谱 S_f（与 sweep_law_test.py 同一定义）。"""
    import torch
    from edm import load_local_model  # noqa: F401  (只是确保路径可用)
    pool = torch.load("/tmp/pool_ffhq64_256_s0.pt", map_location="cpu")
    with torch.no_grad():
        P = (torch.fft.rfft2(pool.to(torch.float64), dim=(-2, -1)).abs() ** 2) \
            / (64 * 64)
        S = P.mean(1).mean(0)          # (33, 33)
    return S.numpy(), int(np.prod(S.shape))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="fig_param.json")
    args = ap.parse_args()

    ra = load("sweep_bdense.json")
    if ra is None:
        print("[缺少] sweep_bdense.json")
        return
    pa = dict(ra[1])
    rl = load("sweep_blow.json")
    if rl is not None:
        for fid, dd in rl[1].items():
            pa.setdefault(fid, {}).update(dd)

    cells = [(T0, pa)]
    for fn in sorted(glob.glob(os.path.join(OUT, "sweep_bt*.json"))):
        r = load(os.path.basename(fn))
        if r is None:
            continue
        cells.append((float(r[0]["cfg"]["t"]), dict(r[1])))
    cells.sort(key=lambda c: c[0])
    ts = [t for t, _ in cells]
    C = dict(cells)
    ug = grid(pa, DENSE)
    print("=" * 100)
    print(f" 0. {len(ts)} 个 t: " + " ".join(f"{t:g}" for t in ts) +
          f"；{len(ug)} 个 u；3 族")
    print("=" * 100)

    # ---------- 取数 ----------
    LAM = {f: {t: np.array([C[t][f][round(u, 4)]["lam"] for u in ug])
               for t in ts} for f in DENSE}
    LG = {f: {t: np.array([C[t][f][round(u, 4)]["lg"] for u in ug])
              for t in ts} for f in DENSE}
    # 注意：**不能**用一个全局的 LS[t]。基线是 sweep_bdense + sweep_blow 两次独立
    # 运行合并的，两者的 logp_sharp_mean 相差 1.16 nats（34839.131 vs 34840.295，
    #  Hutchinson 噪声 ~68 nats 的 1/60，完全正常）。取全局值会让恒等式
    # Λ_0 − G = Λ 多出一项 [ls_picked − ls_own]，E1 因此报 2.2e-3 的假失败。
    # 改成逐 u 带自己的 ls（bdense/blow 的 u 不重叠，除了 0.724/0.356/0.08 三点，
    # 那三点按项目既定规则"后者覆盖前者"取 blow 的，自洽）。
    # 每个 (族, t, u) 都带自己的 ls。文件内 ls 是单一值，但**基线的两个来源文件
    # （bdense / blow）ls 相差 1.16 nats**，所以 t=T0 那一侧也必须逐行取，
    # 否则恒等式在 t=T0 处多一个常数偏移（实测 -1.2 nat，E1 于是报 7.6e-4）。
    LSP = {f: {t: np.array([C[t][f][round(u, 4)]["ls"] for u in ug])
               for t in ts} for f in DENSE}
    LS = {t: float(C[t][DENSE[0]][round(ug[0], 4)]["ls"]) for t in ts}
    LB = {f: {t: np.array([C[t][f][round(u, 4)]["lb"] for u in ug])
              for t in ts} for f in DENSE}

    # ---------- E1 ----------
    print("\n" + "=" * 100)
    print(" 1. E1 守门：恒等式 Λ(t,u) = Λ_0(u) − G(t,u)，G = Δ_B − Δ_S")
    print("=" * 100)
    G = {f: {t: np.zeros(len(ug)) for t in ts} for f in DENSE}
    dS = {t: LS[T0] - LS[t] for t in ts}      # 只用于展示
    worst = 0.0
    for f in DENSE:
        for t in ts:
            dB = LB[f][T0] - LB[f][t]
            dS_u = LSP[f][T0] - LSP[f][t]             # 逐 u 的清晰图衰减
            G[f][t] = dB - dS_u
            resid = np.abs((LAM[f][T0] - G[f][t]) - LAM[f][t])
            worst = max(worst, float(np.max(resid / np.abs(LAM[f][t]))))
    print(f" 最大相对误差 = {worst:.3e}  =>  E1 {'✅' if worst < 1e-6 else '❌'}")

    print(f"\n {'t':>7} | {'Δ_S':>10} | " + " | ".join(
        f"{'G ' + f + ' (u=.14)':>18}" for f in DENSE) + " | "
          + " | ".join(f"{'G/Λ0 ' + f:>10}" for f in DENSE))
    print(" " + "-" * (20 + 21 * len(DENSE) + 13 * len(DENSE)))
    for t in ts:
        row = [f"{dS[t]:>10.1f}"]
        for f in DENSE:
            row.append(f"{G[f][t][-1]:>18.1f}")
        for f in DENSE:
            row.append(f"{G[f][t][-1]/LAM[f][T0][-1]:>10.4f}")
        print(f" {t:>7g} | " + " | ".join(row))
    print(" " + "-" * (20 + 21 * len(DENSE) + 13 * len(DENSE)))
    print(' G/Λ0 是「衰减比例」；若它随 u 基本不变，则 Λ(t,u)≈Λ_0(u)ρ(t)（b 就与 t 无关，已被否掉）。')

    # ---------- E2 可分性 ----------
    print("\n" + "=" * 100)
    print(" 2. E2 可分性：G(t,u) = g(t)·h(u) ？（5×19 矩阵的秩 1 残差）")
    print("=" * 100)
    rank1 = {}
    for f in DENSE:
        M = np.array([G[f][t] for t in ts])          # (5, nu)
        U, s, Vt = np.linalg.svd(M, full_matrices=False)
        r1 = (s[0] * np.outer(U[:, 0], Vt[0, :]))
        res = M - r1
        rel = float(np.sqrt(np.mean(res ** 2)) / np.sqrt(np.mean(M ** 2)))
        rank1[f] = dict(rel=rel, g=U[:, 0] * s[0], h=Vt[0, :])
        print(f" {f:<5}: 奇异值 {np.round(s, 1)}")
        print(f"       秩 1 残差（相对 rms）= {rel:.4f}；"
              f"第二奇异值/第一 = {s[1]/s[0]:.4f}")
    print(" 若第二奇异值可忽略，则 G 可分离。")

    # ---------- E3 h(u) ----------
    print("\n" + "=" * 100)
    print(" 3. E3 h(u) 的形式（用 t=0.05 那一列，信号最大）")
    print("=" * 100)
    tmax = ts[-1]
    ua = np.array(ug)
    print(f" {'u':>6} | " + " | ".join(f"{'G/Λ0 ' + f:>11}" for f in DENSE) +
          f" | {'1-u':>7} | {'G/(1-u) 均值':>13}")
    print(" " + "-" * (10 + 14 * len(DENSE) + 24))
    for i, u in enumerate(ug):
        row = []
        for f in DENSE:
            row.append(f"{G[f][tmax][i]/LAM[f][T0][i]:>11.4f}")
        gm = np.mean([G[f][tmax][i] / LAM[f][T0][i] for f in DENSE])
        print(f" {u:>6.3f} | " + " | ".join(row) +
              f" | {1-u:>7.3f} | {gm/(1-u):>13.2f}")
    print(" " + "-" * (10 + 14 * len(DENSE) + 24))
    # 三种候选的拟合质量（对三族平均的 G/Λ0）
    Gm = np.array([[np.mean([G[f][tmax][i] for f in DENSE])
                    for i in range(len(ug))]])
    Gm = Gm[0]
    l0m = np.array([np.mean([LAM[f][T0][i] for f in DENSE])
                    for i in range(len(ug))])
    cands = {
        "(a) h ∝ 1-u": 1.0 - ua,
        "(b) h ∝ u^gamma": None,
        "(c) h ∝ Λ_0(u)（反向对照）": l0m,
    }
    from scipy.optimize import curve_fit
    print(f" {'候选':<28} | {'rms':>10} | {'参数':>22}")
    print(" " + "-" * 66)
    best = None
    for lab, x in cands.items():
        if x is None:
            try:
                popt, _ = curve_fit(lambda u, g: u ** g, ua, Gm,
                                    p0=(0.5,), bounds=([0.05], [5]),
                                    maxfev=100000)
                r = float(np.sqrt(np.mean((Gm - ua ** popt[0]) ** 2)))
                ps = f"gamma={popt[0]:.3f}"
            except Exception:
                r, ps = float("nan"), "拟合失败"
        else:
            try:
                popt, _ = curve_fit(lambda u, k: k * x, ua, Gm, p0=(1.0,),
                                    maxfev=100000)
                r = float(np.sqrt(np.mean((Gm - popt[0] * x) ** 2)))
                ps = f"k={popt[0]:.4f}"
            except Exception:
                r, ps = float("nan"), "拟合失败"
        print(f" {lab:<28} | {r:>10.5f} | {ps:>22}")
        if best is None or (np.isfinite(r) and r < best[1]):
            best = (lab, r)
    print(" " + "-" * 66)
    print(f" => E3 胜出：{best[0]}（rms {best[1]:.5f}）")

    # ---------- E4 g(t) ----------
    print("\n" + "=" * 100)
    print(" 4. E4 g(t) 的形式（h 已由 E3 定，g(t) = G/h 在三族上平均）")
    print("=" * 100)
    if best[0].startswith("(a)"):
        h = 1.0 - ua
    elif best[0].startswith("(b)"):
        h = ua ** float(best[1].split("gamma=")[1].rstrip(") "))
    else:
        h = l0m
    gt = np.array([np.mean([np.mean(G[f][t] / h) for f in DENSE]) for t in ts])
    ta = np.array(ts)
    print(f" {'t':>7} | {'g(t)':>12} | {'g/log(t/t0)':>12} | {'g/t^2':>12} | "
          f"{'g/t':>12}")
    print(" " + "-" * 66)
    for i, t in enumerate(ts):
        lt = math.log(t / T0)
        print(f" {t:>7g} | {gt[i]:>12.2f} | {gt[i]/lt if lt > 0 else float('nan'):>12.2f} | "
              f"{gt[i]/t**2:>12.1f} | {gt[i]/t:>12.1f}")
    print(" " + "-" * 66)
    # 有效维度 d_eff(t) = Σ_f S_f/(S_f+t²)
    try:
        S, d = pool_spectrum()
        deff = np.array([float(np.sum(S / (S + t ** 2))) for t in ts])
        print(f" {'t':>7} | {'d_eff(t)':>12} | {'g/d_eff':>12}")
        print(" " + "-" * 40)
        for i, t in enumerate(ts):
            print(f" {t:>7g} | {deff[i]:>12.2f} | {gt[i]/deff[i]:>12.4f}")
        print(" " + "-" * 40)
    except Exception as e:
        deff = None
        print(f" [图池谱计算失败：{e}]")
    from scipy.optimize import curve_fit as cf2
    print(f" {'候选':<26} | {'rms':>10} | {'参数':>20}")
    print(" " + "-" * 62)
    gbest = None
    gcands = [
        ("(a) g = k*log(t/t0)", lambda t, k: k * np.log(t / T0), (1.0,)),
        ("(b) g = k*t^alpha", lambda t, k, a: k * t ** a, (1.0, 0.5)),
        ("(c) g = k*t^2", lambda t, k: k * t ** 2, (1.0,)),
    ]
    if deff is not None:
        gcands.append(("(d) g = k*d_eff(t)",
                       lambda t, k: k * np.interp(t, ta, deff), (1.0,)))
    for lab, fn, p0 in gcands:
        try:
            popt, _ = cf2(fn, ta, gt, p0=p0, maxfev=200000)
            r = float(np.sqrt(np.mean((gt - fn(ta, *popt)) ** 2)))
            ps = str(np.round(popt, 4))
        except Exception:
            r, ps = float("nan"), "拟合失败"
        print(f" {lab:<26} | {r:>10.3f} | {ps:>20}")
        if gbest is None or (np.isfinite(r) and r < gbest[1]):
            gbest = (lab, r, fn, popt if np.isfinite(r) else None)
    print(" " + "-" * 62)
    print(f" => E4 胜出：{gbest[0]}（rms {gbest[1]:.3f}）")

    # ---------- E5 高斯对照 ----------
    print("\n" + "=" * 100)
    print(" 5. E5 高斯对照（独立于 E2，直接判决）")
    print("=" * 100)
    try:
        S, d = pool_spectrum()
    except Exception:
        S = None
    if S is not None:
        norm = np.array([0.5 * float(np.sum(np.log((S + t ** 2) /
                                                   (S + T0 ** 2))))
                         for t in ts])
        print(f" {'t':>7} | {'Δ_S 实测':>10} | {'高斯归一化项':>13} | {'比':>7}")
        print(" " + "-" * 48)
        for i, t in enumerate(ts):
            print(f" {t:>7g} | {dS[t]:>10.1f} | {norm[i]:>13.1f} | "
                  f"{dS[t]/norm[i]:>7.3f}")
        print(" " + "-" * 48)
        print(' E5a：Δ_S 是清晰图对数密度的衰减；高斯归一化项是「纯先验几何」的那部分。')
    print(f"\n E5b：G(t,u) 与 [Λ_Gauss(t0,u) − Λ_Gauss(t,u)] 的比")
    print(f" {'t':>7} | " + " | ".join(f"{f:>20}" for f in DENSE))
    print(" " + "-" * (10 + 23 * len(DENSE)))
    for t in ts[1:]:
        row = []
        for f in DENSE:
            dg = LG[f][T0] - LG[f][t]
            r = G[f][t] / np.where(np.abs(dg) > 1e-9, dg, np.nan)
            row.append(f"[{np.nanmin(r):.2f},{np.nanmax(r):.2f}]")
        print(f" {t:>7g} | " + " | ".join(row))
    print(" " + "-" * (10 + 23 * len(DENSE)))
    print(" 方括号是该族 19 个 u 上的比的范围。范围窄 => 非高斯衰落 ∝ 高斯衰落。")

    # ---------- E6 ----------
    print("\n" + "=" * 104)
    print(" 6. E6：两函数参数化 Λ(t,u) = Λ_0(u) − g(t)h(u) 的检验")
    print("=" * 104)
    if gbest[3] is None:
        print(" [g(t) 拟合失败，跳过 E6]")
        return
    gfn = gbest[2]
    print(f" {'t':>7} | {'族':<5} | {'rms(Λ)':>10} | {'相对 rms':>10} | "
          f"{'Λ 量级':>10}")
    print(" " + "-" * 60)
    allrel = []
    for f in DENSE:
        for t in ts:
            pred = LAM[f][T0] - gfn(t, *gbest[3]) * h
            r = float(np.sqrt(np.mean((pred - LAM[f][t]) ** 2)))
            rel = r / float(np.sqrt(np.mean(LAM[f][t] ** 2)))
            allrel.append(rel)
            print(f" {t:>7g} | {f:<5} | {r:>10.1f} | {rel:>10.4f} | "
                  f"{float(np.sqrt(np.mean(LAM[f][t]**2))):>10.1f}")
    print(" " + "-" * 60)
    print(f" 全部相对 rms 最大 {max(allrel):.4f}；Λ 的相对测量噪声约 0.01–0.04。")
    print("\n " + ("E6 过 => 两函数参数化成立，比 §3.9(v) 否掉的四参数 family 强。"
                   if max(allrel) < 0.04 else
                   "E6 未过 0.04（测量噪声量级）=> 记录残差，不为预言改阈值。"))

    if args.dump:
        out = dict(ts=ts, ug=ug, dense=DENSE,
                   G={f: {str(t): [float(v) for v in G[f][t]] for t in ts}
                      for f in DENSE},
                   lam={f: {str(t): [float(v) for v in LAM[f][t]]
                            for t in ts} for f in DENSE},
                   rank1={f: dict(rel=rank1[f]["rel"],
                                  g=[float(v) for v in rank1[f]["g"]],
                                  h=[float(v) for v in rank1[f]["h"]])
                          for f in DENSE},
                   h_best=best[0], h=[float(v) for v in h],
                   g=[float(v) for v in gt],
                   g_best=gbest[0], g_p=[float(v) for v in gbest[3]],
                   rel_rms=allrel)
        p = os.path.join(OUT, args.dump)
        json.dump(out, open(p, "w"))
        print(f"\n [已存] {p}")


if __name__ == "__main__":
    main()
