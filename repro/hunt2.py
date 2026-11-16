"""第二轮描述量搜猎：把 6 个新族并入，找能同时覆盖「有通带」与「底座剖面」的判据。

背景（RESULTS §3.9(p)）
----------------------
$tw_{\\rm lev}$ 在 22 个已测剖面上 R²=0.909、匹配处散度 1.030，预官检验 5/6 通过，
但 EF60/LG60 决定性地失败（+0.064 dex = 16%，6–10σ）。失败机制：**底座剖面**——
过渡中心被推出频域、衰减无通带，$tw_{\\rm lev}$ 的 levels 取剖面自己的 range，
于是量的是平坦剖面的斜率而不是截止锐度。

现在共有 **28 个已测剖面**（22 基线 + sweep_twlaw.json 的 6 个新族）。
本轮在离线数据上搜一遍：**能不能找到一个描述量同时覆盖两个 regime？**
搜到的任何胜出者都只是 in-sample（28 个全都已测），要变成结论必须再跑预官。

新增候选（都在 hunt_descriptor 的尺度不变框架内）：
  w_abs  绝对过渡带宽度 r90-r10（不除以 r50）
  phi    相对底座 amin/<a>（0 = 有通带，>0 = 无通带）
  dyn    相对动态范围 (amax-amin)/<a>
  以及 tw_lev / w_abs / phi / dyn 的两两与三元组合

用法： venv2/bin/python hunt2.py
"""

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

ALL_FILES = ["sweep_profile_shape.json", "sweep_law_test.json",
             "sweep_steepness.json", "sweep_b2test.json", "sweep_twlaw.json"]
NEWFAM = {"TK30", "TK50", "EF60", "W3", "LG60", "TS3"}


def load_all():
    prof, meta = defaultdict(list), {}
    for fn in ALL_FILES:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            print("[缺少]", fn)
            continue
        d = json.load(open(p))
        for rec in d["rows"]:
            m = rec["meta"]
            nm = m.get("fid") or (f"{m['family']}/{m['theta']:g}"
                                  if "family" in m else None)
            if nm is None:
                continue
            prof[nm].append((1.0 - rec["kunif"], rec["lam_mean"]))
            if nm not in meta:
                if "kind" in m and m["kind"] != "real":
                    meta[nm] = dict(kind=m["kind"], par=float(m["par"]),
                                    scale=float(m["scale"]),
                                    kref=float(m["kunif_ref"]))
                else:
                    meta[nm] = dict(kind="real", fam=m["family"],
                                    th=float(m["theta"]))
    for k in prof:
        prof[k].sort(key=lambda c: -c[0])
    return dict(prof), meta


def main():
    prof, meta = load_all()
    r = hd.halfplane()
    print("=" * 116)
    print(f" 载入 {len(prof)} 个已测剖面（22 基线 + 6 新族）")
    print("=" * 116)

    # ---- b(u) 基准（旧 7 平滑族，与既有分析一致）----
    seg_u, seg_b = [], []
    for a, b in sorted(hd.P1_RANGES, reverse=True):
        bs = []
        for k in hd.SMOOTH_OLD:
            nm = f"{k[0]}/{k[1]:g}"
            va, vb = hd.interp(prof[nm], a), hd.interp(prof[nm], b)
            if va and vb and va > 0 and vb > 0:
                bs.append(math.log(vb / va) / math.log(b / a))
        if len(bs) >= 5:
            seg_u.append(math.sqrt(a * b)); seg_b.append(float(np.mean(bs)))
    seg_u = np.array(seg_u); seg_b = np.array(seg_b)

    def b_of(u):
        if u <= seg_u[0]:
            return seg_b[0]
        if u >= seg_u[-1]:
            return seg_b[-1]
        lx = math.log(u)
        for i in range(len(seg_u) - 1):
            if seg_u[i] <= u <= seg_u[i + 1]:
                w = (lx - math.log(seg_u[i])) / \
                    (math.log(seg_u[i + 1]) - math.log(seg_u[i]))
                return seg_b[i] * (1 - w) + seg_b[i + 1] * w
        return seg_b[-1]

    Ucom = None
    for nm in prof:
        s = {round(c[0], 4) for c in prof[nm]}
        Ucom = s if Ucom is None else (Ucom & s)
    Ucom = sorted(Ucom, reverse=True)
    Uall = sorted({round(c[0], 3) for nm in prof for c in prof[nm]}, reverse=True)

    C, mid = {}, {}
    for nm in prof:
        C[nm] = {}
        for u in Uall:
            v = hd.interp(prof[nm], u)
            if v and v > 0:
                C[nm][u] = v * u ** (-b_of(u))
    for u in Uall:
        vs = [C[f"{k[0]}/{k[1]:g}"][u] for k in hd.SMOOTH_OLD
              if u in C.get(f"{k[0]}/{k[1]:g}", {})]
        if len(vs) >= 5:
            mid[u] = float(np.exp(np.mean([math.log(v) for v in vs])))

    def crel(nm):
        vs = [C[nm][u] / mid[u] for u in Ucom if u in C[nm] and u in mid]
        return float(np.median(vs)) if vs else float("nan")

    # ---- 剖面重建 + 描述量 ----
    araw, desc = {}, {}
    for nm in prof:
        m = meta[nm]
        if m["kind"] == "real":
            a = 1.0 - hd.real_g2(m["fam"], m["th"])
        else:
            a = 1.0 - hd.make_profile(m["kind"], m["par"], m["scale"], r)
        a = a * (0.60 / float(a.mean()))
        araw[nm] = a
        d = hd.descriptors(a, r)
        m_ = float(a.mean())
        d["w_abs"] = float(abs(d["_r90"] - d["_r10"]))
        d["phi"] = float(d["_amin"]) / m_
        d["dyn"] = float(d["_amax"] - d["_amin"]) / m_
        # 派生量：把"过渡带宽度"除以"相对动态范围"= 剖面滚降陡度的倒数
        # 动机：tw_lev 对底座剖面失效，是因为 levels 取剖面自己的 range；
        #       除掉 dyn 后，两个 regime 可能落到同一根轴上。
        d["twod"] = d["tw_lev"] / d["dyn"] if d["dyn"] > 1e-9 else np.nan
        d["wod"] = d["w_abs"] / d["dyn"] if d["dyn"] > 1e-9 else np.nan
        desc[nm] = d

    names = sorted(prof)
    y = np.array([crel(n) for n in names])

    # ---- 表 1：全部 28 个剖面 ----
    print(f"\n 共同 u 网格: " + " ".join(f"{u:.3g}" for u in Ucom))
    print("\n" + "=" * 116)
    print(" 1. 28 个已测剖面（新族标 *）")
    print("=" * 116)
    print(f" {'剖面':<14} | {'C':>6} | {'tw_lev':>7} | {'twod':>7} | {'wod':>7} | "
          f"{'phi':>6} | {'dyn':>6} | {'amin':>6} | {'amax':>6} | {'r50':>6}")
    print(" " + "-" * 88)
    for nm in sorted(names, key=lambda n: crel(n)):
        d = desc[nm]
        print(f" {nm+'*' if nm in NEWFAM else '':<14} | {crel(nm):>6.3f} | "
              f"{d['tw_lev']:>7.3f} | {d['twod']:>7.3f} | {d['wod']:>7.3f} | "
              f"{d['phi']:>6.3f} | {d['dyn']:>6.3f} | {d['_amin']:>6.3f} | "
              f"{d['_amax']:>6.3f} | {d['_r50']:>6.3f}")
    print(" " + "-" * 88)
    print(" phi = 相对底座 amin/<a>（0 = 有通带）；dyn = 相对动态范围")

    # ---- 表 2：单描述量横评 ----
    print("\n" + "=" * 116)
    print(" 2. 单描述量横评（28 个剖面）")
    print("=" * 116)
    keys = ["tw_lev", "twod", "wod", "w_abs", "phi", "dyn", "B2", "supp20",
            "rbar", "PR"]
    print(f" {'描述量':<8} | {'跨度':>8} | {'R2lin':>7} | {'R2sat':>7} | {'Spear':>6} | "
          f"{'LOOdex':>8} | {'最紧bin':>8}")
    print(" " + "-" * 74)
    sat_fits = {}
    for k in keys:
        x = np.array([desc[n][k] for n in names])
        if not np.isfinite(x).all() or x.std() == 0 or (x <= 0).any():
            print(f" {k:<8} | NaN / 无跨度 / 非正（饱和形式要求 x>0）")
            continue
        f = hd.ols(np.column_stack([np.ones(len(x)), x]), y)
        l1, _ = hd.loo(np.column_stack([np.ones(len(x)), x]), y)
        sp = hd.spearman(x, y)
        # 饱和逻辑斯蒂形式：C = a + b/(1+(x/x0)^(-p))，网格搜 x0,p
        bg = None
        for x0 in np.linspace(0.02, 6.0, 150):
            for p in np.linspace(0.2, 4.0, 60):
                sig = 1.0 / (1.0 + (x / x0) ** (-p))
                X = np.column_stack([np.ones(len(x)), sig])
                ff = hd.ols(X, y)
                if ff and (bg is None or ff["rmse"] < bg[0]):
                    bg = (ff["rmse"], ff, x0, p)
        _, fs, x0, p = bg
        l1s, _ = hd.loo(np.column_stack(
            [np.ones(len(x)), 1.0 / (1.0 + (x / x0) ** (-p))]), y)
        sat_fits[k] = (fs, x0, p)
        order = np.argsort(x); xs, ys = x[order], y[order]
        best = (np.inf, np.nan)
        for w in [0.10, 0.15, 0.20, 0.30]:
            for i in range(len(xs)):
                j = i
                while j < len(xs) and xs[j] - xs[i] <= w:
                    j += 1
                if j - i >= 3:
                    s2 = float(ys[i:j].max() / ys[i:j].min())
                    if s2 < best[0]:
                        best = (s2, w)
        print(f" {k:<8} | {x.max()-x.min():>8.3f} | {f['r2']:>7.3f} | {fs['r2']:>7.3f} | "
              f"{sp:>6.2f} | {l1s/float(np.mean(np.abs(y)))/math.log(10):>8.4f} | "
              f"{best[0]:>8.3f}")
    print(" " + "-" * 74)
    print(" R2lin = 线性拟合；R2sat = 饱和逻辑斯蒂 C = a + b/(1+(x/x0)^(-p))。")
    print(" 定律本来就是饱和的，所以 R2sat 才是可比的指标。")
    if sat_fits:
        bk = max(sat_fits, key=lambda k: sat_fits[k][0]["r2"])
        fs, x0, p = sat_fits[bk]
        print(f" R2sat 最高: {bk}  R2={fs['r2']:.3f}，"
              f"x0={x0:.3f}，p={p:.2f}，C_inf={fs['coef'][0]+fs['coef'][1]:.3f}")

    # ---- 表 3：两两 / 三元组合 ----
    print("\n" + "=" * 116)
    print(" 3. 组合描述量（按 R2 降序）")
    print("=" * 116)
    combos = [("twod+phi", ["twod", "phi"]),
              ("wod+phi", ["wod", "phi"]),
              ("twod+dyn", ["twod", "dyn"]),
              ("tw_lev+phi", ["tw_lev", "phi"]),
              ("w_abs+phi", ["w_abs", "phi"]),
              ("tw_lev+dyn", ["tw_lev", "dyn"]),
              ("w_abs+dyn", ["w_abs", "dyn"]),
              ("tw_lev+phi+dyn", ["tw_lev", "phi", "dyn"]),
              ("w_abs+phi+dyn", ["w_abs", "phi", "dyn"]),
              ("tw_lev+w_abs", ["tw_lev", "w_abs"]),
              ("phi+dyn", ["phi", "dyn"])]
    rows = []
    for nm_, ks in combos:
        X = np.column_stack([np.ones(len(names))] +
                            [[desc[n][k] for n in names] for k in ks])
        if not np.isfinite(X).all():
            print(f" {nm_:<16} | 有 NaN")
            continue
        f = hd.ols(X, y)
        if not f:
            continue
        l1, _ = hd.loo(X, y)
        # 匹配处散度：用拟合值的残差代替（多维没法简单分 bin）
        rows.append((f["r2"], nm_, f, l1 / float(np.mean(np.abs(y))) / math.log(10)))
    for r2, nm_, f, l1 in sorted(rows, reverse=True):
        print(f" {nm_:<16} | R2={r2:>7.3f} | LOO={l1:>7.4f} dex | "
              f"残差RMS={f['rmse']:>7.4f}")
    if rows:
        r2, nm_, f, l1 = max(rows)
        print(f"\n 最优 [{nm_}] R2={r2:.3f}，LOO {l1:.4f} dex")
        print(" 系数: " + "  ".join(f"{c:+.4f}" for c in f["coef"]))

    # ---- 表 4：分组检查 ----
    print("\n" + "=" * 116)
    print(" 4. 分 regime 检查：有通带 vs 底座剖面")
    print("=" * 116)
    ped = [n for n in names if desc[n]["phi"] > 0.3]
    pas = [n for n in names if desc[n]["phi"] <= 0.3]
    print(f" 底座剖面（phi>0.3，{len(ped)} 个）: " + "  ".join(
        f"{n}({crel(n):.3f})" for n in sorted(ped, key=lambda n: crel(n))))
    print(f" 有通带（phi<=0.3，{len(pas)} 个）: C 范围 "
          f"{min(crel(n) for n in pas):.3f} - {max(crel(n) for n in pas):.3f}")
    print("\n 注意：这一步是**探索性**的——28 个剖面全都已测。")
    print(" 胜出者要变成结论，必须挑**对抗性新族**、先写预官值再跑 GPU。")


if __name__ == "__main__":
    main()
