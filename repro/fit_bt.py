"""b(u,t) 的统一参数化：把 b∞, A, q, C 四个参数的 t 依赖性拟合成闭式。

预官（2026-10-20，跑前写定；本文件第一次运行前的 md5 由运行记录给出）
================================================================================
背景：第八组证明 b(u) 随 t 大幅漂移（b∞ 从 -0.705 升到 -0.402，5 个 t 点），
第九组证明这个漂移**不是**先验谱造成的（换先验只动 Λ 的 0.7-3.5%）。所以它应该是
概率流 ODE 结构的函数——有机会闭式化。本脚本只做离线拟合，不跑 GPU。

已测的 5 个 t：0.002 / 0.005 / 0.01 / 0.02 / 0.05（3 族 × 19 个 u 点 × 20 图）。
提取方式与第七/八/九组逐字对齐：logΛ 对 log u 的 5 次多项式（在全部共有 u 上拟合），
b(u) 取其解析导数，闭式 b = b_inf + A*u^q；SE 由对 20 张图的 2000 次有放回重采样给出。
C(t,族) 定义为 median_u[ Λ(t,u) * u^{-b(u,t)} ]（定律说它与 u 无关，可自检）。

**候选形式（按预官顺序，每个参数都试）**：
  (i)   f(t) = f0 + k*log10(t)          [2 参数：每十倍 t 改变 k]
  (ii)  f(t) = f0 + k*t^alpha           [3 参数：幂律]
  (iii) f(t) = f0 + k*t                 [2 参数：线性]
  (iv)  f(t) = f_inf + (f0-f_inf)/(1+(t/t0)^alpha)   [4 参数：饱和到 f_inf]
  注意：只有 5 个点，(iv) 有 4 个参数，**必然能拟上**，所以它不算证据，只用来读渐近线。

**判据**：
  P1 守门：5 个 t 上闭式都能拟（rms <= 2 倍基线 0.016/0.047/0.037）。不过则不外推。
  P2 形式：每个参数、每个族，最好的**2 参数**形式（(i) 或 (iii)）的 rms
     <= 0.5 * min_over_t(2SE)。即拟合必须落在噪声内，否则就是"没拟上"。
     (iv) 的 rms 一并报但不参与判决（参数太多）。
  P3 跨族一致：胜出形式必须 3 个族相同；且拟合出的参数在 3 族间相差 <= 2SE。
  P4 可推导的渐近线（高斯极限，第八组已推导）：t->inf 时
        b_Gauss(u) = -u/(1-u)  =>  b_inf -> 0,  A -> -1,  q -> 1
     所以 (iv) 拟合出的 f_inf 必须与这三个数相符（b_inf: 0；A: -1；q: 1）。
     若明显不符 => 要么形式错，要么 5 个 t 点离渐近线太远、f_inf 不可辨（要明说）。
  P5 控制变量是不是"高斯占比"而不是 t 本身：
     定义逐点高斯占比 f_G(t,u) = Λ_Gauss(t,u)/Λ(t,u)。
     检验 Δb(u,t) = b(u,t) - b(u,0.002) 在**按 f_G 画图时是否塌缩成一条曲线**
     （跨 4 个 t × 19 个 u = 76 点/族）。判据：按 f_G 画的残差 rms 明显小于按 t 画的；
     若两者差不多 => 控制变量是 t（或 log t）本身，不是高斯占比。
     （这一步是探索性，跑前声明：它不设通过/失败阈值，只报告两个 rms 的比。）

若 P2/P3 过 => 拿到 b(u,t) 的第一个参数化，且它跨形状族一致——这比"5 个点各自拟"强。
若 P2 不过 => 四个参数里至少有一个不是简单的 t 函数，记录失败并报是哪一个。
若 P5 显示按 f_G 塌缩 => 机理是"高斯/非高斯占比"，可继续推；否则是别的。

用法： venv2/bin/python fit_bt.py
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
GAUSS_LIMIT = {"b_inf": 0.0, "A": -1.0, "q": 1.0}


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


def bfun_of(P, us_all):
    mu = P.mean(0)
    if not (mu > 0).all():
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


def c_of(P, ug, bfun):
    """C(t,族) = 在 u 网格上的几何均值 exp(mean_u log[Λ(u)·u^{-b(u)}])。

    自检量 csp = 同一个 C(u) 在 u 上的 max/min。第一版我用的是 median 且**没有**
    按 u 归一，结果 csp 达 4–8.5×——这不是定律失效，而是我在重复项目里已有的定义时
    漏了一步：analyze_uniform/hunt2/hunt3 都是 `C[u] / mid[u]`（mid[u] = 该 u 上
    跨族几何均值），把**共同的** u 剖面除掉之后再报"u 漂移"（均匀算子那个 1.006 就是
    除掉之后的）。这里两者都报：cgeo 用于 t 依赖性（共同的 u 偏移在比值里抵消），
    csp 作为诊断量。另外补一个按 mid(u) 归一后的 u 漂移 crel_drift，与旧定义对齐。
    """
    vals = []
    for i, u in enumerate(ug):
        lam = float(P[:, i].mean())
        vals.append(lam * u ** (-bfun(u)))
    v = np.array(vals)
    return (float(np.exp(np.mean(np.log(v)))), float(v.max() / v.min()),
            np.log(v))


# ---------- 预官的四个候选形式 ----------
FORMS = [
    ("(i)   f0 + k*log10 t", lambda t, *p: p[0] + p[1] * np.log10(t),
     [(-0.7, 1.0)], ([-1e3, -1e3], [1e3, 1e3])),
    ("(ii)  f0 + k*t^alpha", lambda t, *p: p[0] + p[1] * t ** p[2],
     [(-0.7, 1.0, 0.5)], ([-1e3, -1e3, 0.05], [1e3, 1e3, 5])),
    ("(iii) f0 + k*t", lambda t, *p: p[0] + p[1] * t,
     [(-0.7, 1.0)], ([-1e3, -1e3], [1e3, 1e3])),
    ("(iv)  f_inf + (f0-f_inf)/(1+(t/t0)^a)",
     lambda t, *p: p[0] + (p[1] - p[0]) / (1.0 + (t / p[2]) ** p[3]),
     [(0.0, -0.7, 0.01, 1.0)], ([-1e3, -1e3, 1e-4, 0.05],
                                [1e3, 1e3, 1.0, 5])),
]
TWO_PAR = {"(i)   f0 + k*log10 t", "(iii) f0 + k*t"}


def fit_forms(ts, ys, ses):
    from scipy.optimize import curve_fit
    ts = np.array(ts)
    ys = np.array(ys)
    out = []
    for label, fn, p0s, bnds in FORMS:
        best = None
        for p0 in p0s:
            try:
                popt, _ = curve_fit(fn, ts, ys, p0=p0, bounds=bnds,
                                    maxfev=200000, sigma=ses, absolute_sigma=True)
                r = float(np.sqrt(np.mean((ys - fn(ts, *popt)) ** 2)))
                if best is None or r < best[0]:
                    best = (r, popt)
            except Exception:
                pass
        out.append((label, best))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="fig_fit.json")
    args = ap.parse_args()

    # ---------- 5 个 t 的数据 ----------
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

    cells = [(0.002, pa, ga)]
    for fn in sorted(glob.glob(os.path.join(OUT, "sweep_bt*.json"))):
        r = load(os.path.basename(fn))
        if r is None:
            continue
        cells.append((float(r[0]["cfg"]["t"]), dict(r[1]), dict(r[2])))
    cells.sort(key=lambda c: c[0])
    print("=" * 104)
    print(f" 0. {len(cells)} 个 t: " + " ".join(f"{t:g}" for t, _, _ in cells))
    print("=" * 104)

    data = {}
    for t, prof, gpr in cells:
        ug_all, ug = grid(prof, DENSE)
        assert ug_all[:len(ug)] == ug
        P = matrices(prof, DENSE, ug_all)
        data[t] = dict(ug=ug, ug_all=ug_all, P=P, G=gpr)
        if any(P[f] is None for f in DENSE):
            print(f"[错误] t={t:g} 有点缺失")
            return

    # ---------- 表 1：各 t 的四个参数 ----------
    print("\n" + "=" * 104)
    print(" 1. 各 t 的 b_inf / A / q / C（闭式拟合 + C 的 u 散度自检）")
    print("=" * 104)
    print(f" {'t':>7} | {'族':<5} | {'b_inf':>8} | {'A':>8} | {'q':>7} | "
          f"{'rms':>7} | {'C':>10} | {'C散度':>7} | {'P1':>4}")
    print(" " + "-" * 84)
    est = {}
    for t, _, _ in cells:
        d = data[t]
        for fid in DENSE:
            bf = bfun_of(d["P"][fid], d["ug_all"])
            cf = closed_form(d["ug"], bf) if bf else None
            r3 = c_of(d["P"][fid], d["ug"], bf) if bf else (np.nan, np.nan, None)
            C, csp, lv = r3
            est[(t, fid)] = dict(cf=cf, b=bf, C=C, csp=csp, lv=lv)
            ok = cf is not None and cf["rms"] <= 2 * PRE_RMS[fid]
            if cf is None:
                print(f" {t:>7g} | {fid:<5} | {'拟合失败':>8}")
            else:
                print(f" {t:>7g} | {fid:<5} | {cf['p'][0]:>8.3f} | "
                      f"{cf['p'][1]:>8.3f} | {cf['p'][2]:>7.3f} | "
                      f"{cf['rms']:>7.3f} | {C:>10.1f} | {csp:>7.3f} | "
                      f"{'✅' if ok else '❌':>4}")

    # ---------- 表 1b：C 的 u 漂移，按项目旧定义（除掉跨族共同剖面后） ----------
    print("\n" + "=" * 104)
    print(" 1b. C 的 u 漂移：旧定义是 C(u)/mid(u)（mid(u)=该 u 上跨族几何均值），")
    print("     把**共同的** u 剖面除掉之后再报——均匀算子当年那个 1.006 就是除掉后的。")
    print("=" * 104)
    print(f" {'t':>7} | {'族':<5} | {'C_geo':>10} | {'原始散度':>9} | "
          f"{'除掉共同剖面后':>13}")
    print(" " + "-" * 60)
    for t, _, _ in cells:
        d = data[t]
        L = np.array([est[(t, f)]["lv"] for f in DENSE])     # (3, nu) log C(u)
        mid = L.mean(0)                                       # 跨族均值（log 域）
        for k, fid in enumerate(DENSE):
            dd = L[k] - mid
            print(f" {t:>7g} | {fid:<5} | {est[(t, fid)]['C']:>10.1f} | "
                  f"{est[(t, fid)]['csp']:>9.3f} | "
                  f"{float(np.exp(dd.max() - dd.min())):>13.4f}")
    print(" " + "-" * 60)
    print(" 若第三列接近 1（像旧定义的 1.006），说明定律结构与旧结论一致，")
    print(" 前面的「原始散度 4-8.5x」只是共同的 u 剖面，不是定律失效。")

    # ---------- 表 2：配对 bootstrap 拿四个参数的 SE ----------
    print("\n" + "=" * 104)
    print(f" 2. 配对 bootstrap（{NBOOT} 次，同一批图同时用于所有 t）")
    print("=" * 104)
    rng = np.random.default_rng(0)
    nimg = data[0.002]["P"][DENSE[0]].shape[0]
    boots = {fid: {k: [] for k in ["b_inf", "A", "q", "C"]} for fid in DENSE}
    for fid in DENSE:
        for _ in range(NBOOT):
            idx = rng.integers(0, nimg, nimg)
            for t, _, _ in cells:
                d = data[t]
                P = d["P"][fid][idx]
                bf = bfun_of(P, d["ug_all"])
                if bf is None:
                    continue
                cf = closed_form(d["ug"], bf)
                if cf is None:
                    continue
                C, _, _ = c_of(P, d["ug"], bf)
                boots[fid]["b_inf"].append((t, cf["p"][0]))
                boots[fid]["A"].append((t, cf["p"][1]))
                boots[fid]["q"].append((t, cf["p"][2]))
                boots[fid]["C"].append((t, C))
    se = {}
    print(f" {'参数':<6} | {'族':<5} | " + " | ".join(
        f"{'t=' + format(t, 'g'):>13}" for t, _, _ in cells))
    print(" " + "-" * (16 + 16 * len(cells)))
    for key in ["b_inf", "A", "q", "C"]:
        for fid in DENSE:
            arr = np.array(boots[fid][key])
            row = []
            for t, _, _ in cells:
                v = arr[arr[:, 0] == t, 1]
                row.append(f"{v.std():>13.3f}")
                se[(key, fid, t)] = float(v.std())
            print(f" {key:<6} | {fid:<5} | " + " | ".join(row))
    print(" " + "-" * (16 + 16 * len(cells)))

    # ---------- 表 3：拟合候选形式 ----------
    print("\n" + "=" * 104)
    print(" 3. 候选形式拟合（P2：最好的 2 参数形式 rms <= 0.5*min(2SE)）")
    print("=" * 104)
    ts = [t for t, _, _ in cells]
    fitres = {}
    for key in ["b_inf", "A", "q", "C"]:
        print(f"\n --- {key} ---")
        print(f" {'形式':<34} | " + " | ".join(f"{f:>22}" for f in DENSE))
        print(" " + "-" * (38 + 25 * len(DENSE)))
        rows = {}
        for label, _fn, _p0, _bd in FORMS:
            cells_out = []
            oks = []
            for fid in DENSE:
                ys = [est[(t, fid)]["cf"]["p"][{"b_inf": 0, "A": 1,
                                                 "q": 2}[key]]
                      if key != "C" else est[(t, fid)]["C"] for t in ts]
                ses = [se.get((key, fid, t), np.nan) for t in ts]
                fl = fit_forms(ts, ys, ses)
                got = [r for lab, r in fl if lab == label][0]
                if got is None:
                    cells_out.append(f"{'拟合失败':>22}")
                    oks.append(False)
                    continue
                r, popt = got
                m2 = 0.5 * min(2 * s for s in ses if np.isfinite(s))
                ok = (r <= m2)
                oks.append(bool(ok))
                cells_out.append(f"rms={r:.4f} p={np.round(popt, 3)}")
                rows.setdefault(fid, {})[label] = dict(rms=r,
                                                       p=[float(v) for v in popt])
            print(f" {label:<34} | " + " | ".join(f"{c:>22}" for c in cells_out))
            rows["_ok"] = rows.get("_ok", {})
            rows["_ok"][label] = all(oks)
        fitres[key] = rows
        # 判决：2 参数形式里 rms 最小且通过 P2 的
        cand = [(lab, min(rows[f][lab]["rms"] for f in DENSE))
                for lab, _f, _p, _b in FORMS if lab in TWO_PAR]
        best2 = min(cand, key=lambda x: x[1])
        print(f"   => 2 参数形式里最好的是 {best2[0]}（三族最大 rms {best2[1]:.4f}）；"
              f"P2 {'✅' if best2[1] <= 0.5 * min(2 * se.get((key, f, t), 9) for f in DENSE for t in ts) else '❌'}")
        same = len({lab for lab, _f, _p, _b in FORMS if lab in TWO_PAR
                    and all(rows[f][lab]["rms"] == min(
                        rows[f][l2]["rms"] for l2, _f2, _p2, _b2 in FORMS if l2 in TWO_PAR)
                        for f in DENSE)}) >= 1
        print(f"   => P3 跨族一致：三族的最优 2 参数形式"
              f"{'相同' if same else '不完全相同'}"
              f"（各自最优："
              + " / ".join(f"{f}:" + min((l2 for l2, _f2, _p2, _b2 in FORMS if l2 in TWO_PAR),
                                          key=lambda l2: rows[f][l2]['rms']).split()[0]
                           for f in DENSE) + "）")

    # ---------- 表 4：P4 渐近线 ----------
    print("\n" + "=" * 104)
    print(" 4. P4 可推导的渐近线（高斯极限：b_inf->0, A->-1, q->1）")
    print("=" * 104)
    print(f" {'参数':<6} | {'高斯极限':>9} | " + " | ".join(
        f"{f + ' f_inf':>16}" for f in DENSE))
    print(" " + "-" * (20 + 19 * len(DENSE)))
    for key in ["b_inf", "A", "q"]:
        row = []
        for fid in DENSE:
            r = fitres[key].get(fid, {}).get("(iv)  f_inf + (f0-f_inf)/(1+(t/t0)^a)")
            row.append(f"{r['p'][0]:>16.3f}" if r else f"{'--':>16}")
        print(f" {key:<6} | {GAUSS_LIMIT[key]:>9.2f} | " + " | ".join(row))
    print(" " + "-" * (20 + 19 * len(DENSE)))
    print(" 注意：5 个 t 点最大只到 0.05，离高斯极限（t~0.2，放大倍数 0.39）还远，")
    print("       所以 f_inf 很可能不可辨——若与高斯极限差很多，先说的是'范围不够'。")

    # ---------- 表 5：P5 控制变量 ----------
    print("\n" + "=" * 104)
    print(" 5. P5 控制变量是 t 还是高斯占比 f_G？（探索性，不设阈值）")
    print("=" * 104)
    ug = data[0.002]["ug"]
    lx = np.log(np.array(ug))
    for fid in DENSE:
        b0 = np.array([est[(0.002, fid)]["b"](u) for u in ug])
        pts_t, pts_g, dv = [], [], []
        for t, _, _ in cells:
            if t == 0.002:
                continue
            d = data[t]
            bt = np.array([est[(t, fid)]["b"](u) for u in ug])
            Lb = np.array([d["P"][fid][:, i].mean() for i in range(len(ug))])
            L0 = np.array([data[0.002]["P"][fid][:, i].mean()
                           for i in range(len(ug))])
            Gt = np.array([d["G"].get(fid, {}).get(round(u, 4), np.nan)
                           for u in ug])
            G0 = np.array([data[0.002]["G"].get(fid, {}).get(round(u, 4), np.nan)
                           for u in ug])
            fG = Gt / Lb
            pts_t.append((t, bt - b0))
            pts_g.append((fG, bt - b0))
        # 按 t 分组画残差：每族各自对 log t 做线性回归，看跨 t 的 rms
        # 按 f_G：把 4 个 t 的 76 点合起来对 log f_G 回归
        for lab, arr in [("按 t（各自线性回归）", pts_t),
                         ("按 f_G（合并回归）", pts_g)]:
            if lab.startswith("按 t"):
                num = den = 0.0
                for t, dd in arr:
                    c = np.polyfit(lx, dd, 1)
                    num += float(np.sum((dd - np.polyval(c, lx)) ** 2))
                    den += len(dd)
                print(f"   {fid:<5} {lab}: rms = {math.sqrt(num/den):.4f}")
            else:
                X = np.concatenate([np.log10(np.array(a)) for _, a in arr])
                Y = np.concatenate([np.array(a) for _, a in arr])
                g = np.isfinite(X) & np.isfinite(Y)
                c = np.polyfit(X[g], Y[g], 1)
                r = float(np.sqrt(np.mean((Y[g] - np.polyval(c, X[g])) ** 2)))
                print(f"   {fid:<5} {lab}: rms = {r:.4f}  "
                      f"（f_G 从 {10**X[g].min():.4f} 到 {10**X[g].max():.4f}）")
    print(" " + "-" * 60)
    print(" 若「按 f_G」的 rms 明显小于「按 t」的 => 控制变量是高斯/非高斯占比；")
    print(" 两者差不多 => 控制变量是 t 本身（或 log t）。")

    # ---------- 汇总 ----------
    print("\n" + "=" * 104)
    print(" 6. 汇总")
    print("=" * 104)
    allp2 = True
    for key in ["b_inf", "A", "q", "C"]:
        cand = [(lab, min(fitres[key][f][lab]["rms"] for f in DENSE))
                for lab, _f, _p, _b in FORMS if lab in TWO_PAR]
        best2 = min(cand, key=lambda x: x[1])
        thr = 0.5 * min(2 * se.get((key, f, t), 9) for f in DENSE for t in ts)
        ok = best2[1] <= thr
        allp2 = allp2 and ok
        print(f" {key:<6}: 最优 2 参数形式 {best2[0]:<26} rms={best2[1]:.4f} "
              f"阈值={thr:.4f}  P2 {'✅' if ok else '❌'}")
    print("\n " + ("P2 全过 => 四个参数都是 t 的简单函数，b(u,t) 拿到第一个参数化。"
                   if allp2 else
                   "有参数不是 t 的简单函数 => 记录是哪一个，不为预言改阈值。"))

    if args.dump:
        out = dict(cells=[t for t, _, _ in cells], dense=DENSE,
                   pre_rms=PRE_RMS, gauss_limit=GAUSS_LIMIT,
                   est={f"{t}|{fid}": dict(
                       p=[float(v) for v in est[(t, fid)]["cf"]["p"]],
                       rms=float(est[(t, fid)]["cf"]["rms"]),
                       C=float(est[(t, fid)]["C"]),
                       csp=float(est[(t, fid)]["csp"]))
                       for t, _, _ in cells for fid in DENSE},
                   se={f"{k}|{fid}|{t}": v for (k, fid, t), v in se.items()},
                   fit={k: {f: {lab: dict(rms=v["rms"], p=v["p"])
                                for lab, v in fitres[k][f].items()}
                           for f in DENSE} for k in fitres})
        p = os.path.join(OUT, args.dump)
        json.dump(out, open(p, "w"))
        print(f"\n [已存] {p}")


if __name__ == "__main__":
    main()
