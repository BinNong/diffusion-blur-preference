"""把 b(u,t) 的 t 敏感性定位到 Λ 的哪一项（离线，零 GPU）。

预官（2026-10-20，跑前写定；md5 由运行记录给出）
================================================================================
动机：§3.9(v) 证明 b(u,t) 的四个参数不是 t 的简单函数，且控制变量是 t 本身而非高斯占比
（P5）。剩下的机理问题是：**t 到底改写了 Λ 的哪一部分**？

分解（恒等式，不是模型）
------------------------
  Λ(t,u) = logp_blur(t,u) − logp_sharp(t)
  记 Λ_B(t,u) = logp_blur_mean（**随 u 变**），Λ_S(t) = logp_sharp_mean（**不随 u 变**，
  同一 t 下所有算子同一值——它只是"清晰原图在先验下的对数密度"）。
  则
      b(t,u) = d logΛ / d log u = S_B(t,u) / Λ(t,u)
  其中 S_B(t,u) = d log Λ_B / d log u 是**模糊图对数密度的 u 斜率**。

  所以 t 敏感性有且只有两个入口：分子 S_B（模糊图自己的对数密度怎么随算子变），
  和分母 Λ（偏好本身在缩小）。这一条是恒等式，无需假设。

**判据（跑前写定）**：
  D1 分子是否 t 无关：在每个 u 上，把 5 个 t 的 S_B 拟成常数，要求 rms <= 2*SE_bootstrap。
     （SE 用同一套对 20 张图的重采样；注意 logp_sharp 的逐图值未存，用其均值代替——
      它是 u 无关的加性项，对**斜率**的影响是二阶的，如实披露。）
  D2 若 D1 过，则 b(t,u) = S_B(u)/Λ(t,u) 是"一个 u 的函数 + 实测 Λ"的预言：
     在 4 个新 t × 19 个 u 上检验，容差 = 2*SE_bootstrap(b)。
  D3 无论 D1 过否，把 Δb/b 分解（一阶）：
         Δb/b ≈ ΔS_B/S_B − ΔΛ/Λ
     报两项各自的量级，说明 t 敏感性主要走哪个入口。
  D4 另外报 Λ_S(t) 与 Λ_B(t,u) 各自的 t 衰减（5 个 t 点），看哪一项掉得快。

若 D1+D2 过 => b(u,t) 归约为"模糊图对数密度的 u 斜率"这一个函数加实测 Λ，
   四参数 family 塌成 1 个函数——这是机理上的大简化。
若 D1 不过 => 分子自己也随 t 变，t 同时改写了两边；D3 会给出各自的量级。
若 D4 显示 Λ_S 掉得比 Λ_B 快 => "偏好缩小"主要是清晰图那一侧在掉，可继续追。

用法： venv2/bin/python decomp_t.py
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


def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    prof = defaultdict(list)
    for rec in d["rows"]:
        fid = rec["meta"]["fid"]
        prof[fid].append((1.0 - rec["kunif"], rec["lam_mean"],
                          rec.get("lam_per_img"),
                          rec.get("logp_sharp_mean")))
    for k in prof:
        prof[k].sort(key=lambda c: -c[0])
    return d, dict(prof)


def grid(prof, fids, lo=0.13, hi=0.86):
    sets = [{round(u, 4) for u, _, _, _ in prof[f]} for f in fids]
    ug_all = sorted(set.intersection(*sets), reverse=True)
    return ug_all, [u for u in ug_all if lo <= u <= hi]


def fit_slope(vals, us_all, lam_key):
    """把 log(vals) 对 log u 拟 DEG 次多项式，返回 (值向量, 斜率函数)。"""
    v = np.array(vals, dtype=float)
    if not (v > 0).all():
        return v, None
    try:
        c = np.polyfit(np.log(np.array(us_all)), np.log(v), DEG)
    except Exception:
        return v, None
    d = np.polyder(c)
    return v, (lambda u: float(np.polyval(d, math.log(u))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="fig_decomp.json")
    args = ap.parse_args()

    ra = load("sweep_bdense.json")
    if ra is None:
        print("[缺少] sweep_bdense.json")
        return
    pa = dict(ra[1])
    rl = load("sweep_blow.json")
    if rl is not None:
        for fid, rows in rl[1].items():
            pa.setdefault(fid, []).extend(rows)
        for k in pa:
            pa[k].sort(key=lambda c: -c[0])

    cells = [(0.002, pa)]
    for fn in sorted(glob.glob(os.path.join(OUT, "sweep_bt*.json"))):
        r = load(os.path.basename(fn))
        if r is None:
            continue
        cells.append((float(r[0]["cfg"]["t"]), dict(r[1])))
    cells.sort(key=lambda c: c[0])
    print("=" * 100)
    print(f" 0. {len(cells)} 个 t: " + " ".join(f"{t:g}" for t, _ in cells))
    print("=" * 100)

    # 每格：ug / 每族的 Λ(u)、Λ_B(u)、Λ_S、b(u)、S_B(u)
    D = {}
    for t, prof in cells:
        ug_all, ug = grid(prof, DENSE)
        assert ug_all[:len(ug)] == ug
        ent = dict(ug=ug, ug_all=ug_all, fam={})
        LS = None
        for fid in DENSE:
            dd = {round(u, 4): (lm, pi, ls) for u, lm, pi, ls in prof[fid]}
            lam = np.array([dd[round(u, 4)][0] for u in ug_all])
            P = np.array([dd[round(u, 4)][1] for u in ug_all])   # (nu, nimg)
            ls = dd[round(ug_all[0], 4)][2]
            LS = float(ls) if ls is not None else LS
            lamB = lam + (LS if LS is not None else 0.0)
            _, fb = fit_slope(lam, ug_all, "lam")
            _, fB = fit_slope(lamB, ug_all, "lamB")
            ent["fam"][fid] = dict(lam=lam, lamB=lamB, P=P, b=fb, SB=fB)
        ent["LS"] = LS
        D[t] = ent
        print(f" [t={t:g}] {len(ug)} 个拟合 u；Λ_S = logp_sharp_mean = "
              f"{LS if LS is None else round(LS, 1)}")

    # ---------- 表 1：Λ_S 与 Λ_B 的 t 衰减（D4） ----------
    print("\n" + "=" * 100)
    print(" 1. D4：Λ_S（清晰图）与 Λ_B（模糊图）各自的 t 衰减")
    print("=" * 100)
    ts = [t for t, _ in cells]
    print(f" {'t':>7} | {'Λ_S':>10} | " + " | ".join(
        f"{'Λ_B ' + f + ' (u=0.85)':>20}" for f in DENSE))
    print(" " + "-" * (20 + 23 * len(DENSE)))
    for t in ts:
        e = D[t]
        row = [f"{e['LS']:>10.1f}" if e["LS"] is not None else f"{'--':>10}"]
        for fid in DENSE:
            row.append(f"{e['fam'][fid]['lamB'][0]:>20.1f}")
        print(f" {t:>7g} | " + " | ".join(row))
    print(" " + "-" * (20 + 23 * len(DENSE)))
    base = D[0.002]
    print(" 相对 t=0.002 的比值：")
    print(f" {'t':>7} | {'Λ_S/Λ_S0':>10} | " + " | ".join(
        f"{'Λ_B ' + f:>12}" for f in DENSE))
    for t in ts:
        e = D[t]
        r0 = e["LS"] / base["LS"] if e["LS"] and base["LS"] else float("nan")
        row = [f"{r0:>10.4f}"]
        for fid in DENSE:
            row.append(f"{e['fam'][fid]['lamB'][0]/base['fam'][fid]['lamB'][0]:>12.4f}")
        print(f" {t:>7g} | " + " | ".join(row))

    # ---------- 表 2：b 与 S_B（D1/D3 的原料） ----------
    print("\n" + "=" * 100)
    print(" 2. b(t,u) = S_B(t,u)/Λ(t,u) 的两项（W2；括号内为相对 t=0.002 的变化）")
    print("=" * 100)
    ug = base["ug"]
    for fid in DENSE:
        print(f" --- {fid} ---")
        print(f" {'u':>6} | {'b(0.002)':>10} | " + " | ".join(
            f"{'t=' + format(t, 'g'):>20}" for t in ts[1:]))
        print(" " + "-" * (18 + 23 * (len(ts) - 1)))
        for i, u in enumerate(ug):
            b0 = base["fam"][fid]["b"](u)
            s0 = base["fam"][fid]["SB"](u)
            l0 = base["fam"][fid]["lam"][i]
            row = [f"{b0:>10.3f}"]
            for t in ts[1:]:
                e = D[t]
                bt = e["fam"][fid]["b"](u)
                st = e["fam"][fid]["SB"](u)
                lt = e["fam"][fid]["lam"][i]
                row.append(f"{bt:>7.3f}(S{st/s0:>5.2f}/L{lt/l0:>5.2f})")
            print(f" {u:>6.3f} | " + " | ".join(row))
        print(" " + "-" * (18 + 23 * (len(ts) - 1)))
    print(" S = S_B/S_B(0.002)（分子），L = Λ/Λ(0.002)（分母）。")
    print(" 恒等式预测 b_t/b_0 = (S_t/S_0)/(L_t/L_0)。")

    # ---------- 表 3：D3 分解 ----------
    print("\n" + "=" * 100)
    print(" 3. D3：Δb/b 的一阶分解 = ΔS_B/S_B − ΔΛ/Λ")
    print("=" * 100)
    print(f" {'t':>7} | {'族':<5} | {'u':>6} | {'Δb/b 实测':>11} | "
          f"{'分子项':>9} | {'分母项':>9} | {'两项和':>9} | {'残差':>8}")
    print(" " + "-" * 84)
    for t in ts[1:]:
        for fid in DENSE:
            for i in [0, len(ug) // 2, len(ug) - 1]:
                u = ug[i]
                b0 = base["fam"][fid]["b"](u)
                s0 = base["fam"][fid]["SB"](u)
                l0 = base["fam"][fid]["lam"][i]
                e = D[t]
                bt = e["fam"][fid]["b"](u)
                st = e["fam"][fid]["SB"](u)
                lt = e["fam"][fid]["lam"][i]
                num = st / s0 - 1.0
                den = -(lt / l0 - 1.0)
                tot = num + den
                act = bt / b0 - 1.0
                print(f" {t:>7g} | {fid:<5} | {u:>6.3f} | {act:>+11.4f} | "
                      f"{num:>+9.4f} | {den:>+9.4f} | {tot:>+9.4f} | "
                      f"{act-tot:>+8.4f}")
    print(" " + "-" * 84)
    print(' 若「残差」远小于两项，说明一阶分解够用，t 敏感性的入口可归因。')

    # ---------- 表 4：D1/D2 ----------
    print("\n" + "=" * 100)
    print(" 4. D1/D2：分子 S_B(t,u) 是否 t 无关；若是，b 能否由 S_B(u)/Λ(t,u) 预言")
    print("=" * 100)
    rng = np.random.default_rng(0)
    nimg = base["fam"][DENSE[0]]["P"].shape[1]
    se_b = {}
    for fid in DENSE:
        for _ in range(NBOOT):
            idx = rng.integers(0, nimg, nimg)
            for t in ts:
                e = D[t]
                P = e["fam"][fid]["P"][:, idx]
                lam = P.mean(0)
                LS = e["LS"] if e["LS"] is not None else 0.0
                lamB = lam + LS
                _, fb = fit_slope(lam, e["ug_all"], "lam")
                _, fB = fit_slope(lamB, e["ug_all"], "lamB")
                if fb is None or fB is None:
                    continue
                for i, u in enumerate(e["ug"]):
                    se_b[(fid, t, round(u, 4))] = se_b.get(
                        (fid, t, round(u, 4)), [])
                    se_b[(fid, t, round(u, 4))].append(fb(u))
    seb = {k: float(np.std(v)) for k, v in se_b.items() if len(v) > 100}

    print(f" {'族':<5} | {'u':>6} | {'S_B 极差/均值':>14} | {'2SE(b)':>8} | "
          f"{'D1':>4} | {'b 预言 rms':>11} | {'D2':>4}")
    print(" " + "-" * 66)
    d1all, d2all = True, True
    for fid in DENSE:
        for i, u in enumerate(ug):
            sv = [D[t]["fam"][fid]["SB"](u) for t in ts]
            sv = np.array(sv)
            spread = float((sv.max() - sv.min()) / abs(np.mean(sv)))
            ses = [seb.get((fid, t, round(u, 4)), np.nan) for t in ts]
            thr = 2 * np.nanmin(ses) if np.isfinite(ses).any() else np.nan
            ok1 = spread <= (thr / abs(np.mean(
                [D[t]["fam"][fid]["b"](u) for t in ts]))
                if np.isfinite(thr) else False)
            d1all = d1all and ok1
            # D2：用 t=0.002 的 S_B 预言其它 t 的 b
            s0 = base["fam"][fid]["SB"](u)
            errs = []
            for t in ts[1:]:
                e = D[t]
                pred = s0 / e["fam"][fid]["lam"][i]
                act = e["fam"][fid]["b"](u)
                se = seb.get((fid, t, round(u, 4)), np.nan)
                if np.isfinite(se) and se > 0:
                    errs.append(abs(pred - act) / se)
            rms_se = float(np.mean(errs)) if errs else float("nan")
            ok2 = (rms_se <= 2.0) if np.isfinite(rms_se) else False
            d2all = d2all and ok2
            print(f" {fid:<5} | {u:>6.3f} | {spread:>14.4f} | {thr:>8.3f} | "
                  f"{'✅' if ok1 else '❌':>4} | {rms_se:>11.3f} | "
                  f"{'✅' if ok2 else '❌':>4}")
    print(" " + "-" * 66)
    print(" D1 的判据：S_B 的相对极差 <= b 的相对 2SE（用 2SE/|b| 折算）。")
    print(" D2 的判据：|b_预测 - b_实测|/SE 的均值 <= 2。")
    print("\n " + ("D1+D2 全过 => b(u,t) 归约为 S_B(u)/Λ(t,u)，四参数 family 塌成一个函数。"
                   if (d1all and d2all) else
                   "有未过项 => 记录结果。按项目规矩，不为预言改阈值。"))

    if args.dump:
        out = dict(ts=ts, ug=ug, dense=DENSE,
                   LS={str(t): D[t]["LS"] for t in ts},
                   fam={})
        for t in ts:
            for fid in DENSE:
                e = D[t]["fam"][fid]
                out["fam"][f"{t}|{fid}"] = dict(
                    lam=[float(v) for v in e["lam"]],
                    lamB=[float(v) for v in e["lamB"]],
                    b=[e["b"](u) for u in ug],
                    SB=[e["SB"](u) for u in ug])
        p = os.path.join(OUT, args.dump)
        json.dump(out, open(p, "w"))
        print(f"\n [已存] {p}")


if __name__ == "__main__":
    main()
