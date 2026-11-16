"""为 tw_lev 定律的预官检验挑选对抗性新族，并计算预官值（只看剖面定义，不涉及任何 Λ 测量）。

背景（hunt_descriptor.py 的输出）
--------------------------------
22 个已测剖面上，**相对过渡带宽度**
    tw_lev = (r at a=0.9*range - r at a=0.1*range) / (r at a=0.5*range)
（range = 径向均值剖面 max-min；尺度不变，所以不受 beta 压缩/knif_ref 差异影响）
预测 C 的能力全面超过 B2：

    描述量    R2      LOO dex   匹配描述量处 C 的 max/min
    tw_lev   0.909   0.036     1.030
    B2       0.825   0.051     1.475     <-- 就是死在这条上（§3.9(o)）

定律形式（两种 C 取值惯例给出同一个形式）：
    C = C_lo + (C_hi - C_lo) / (1 + (tw/t0)^(-p))
物理读法：衰减在频域上"滚过去"的带宽与它自身的径向尺度可相比时（tw >~ 1），
C ≈ 1（线性/高斯 regime）；只有滚降带远窄于尺度（tw << 1，算子像尖锐谱投影）时 C 才下降。

本脚本做什么
------------
1. 重建 22 个已测剖面的 (tw_lev, C)，拟合并 bootstrap 出预言不确定度。
2. 对一批**候选新族**（含一个全新形状类 tukey 升余弦边）算 tw_lev / B2 / supp20，
   并分别按三条定律给出预言。
3. 标出 (a) tw_lev 落在已测缺口 [0.4,0.7] 或 [0.95,1.25] 的，(b) 三条定律预言互相矛盾的
   ——后者才是对抗性检验，能区分"是真的定律"和"又一个能拟合的标量"。

用法： venv2/bin/python predict_tw.py
"""

import importlib.util
import math
import os

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "hd", os.path.join(PROJ, "repro", "hunt_descriptor.py"))
hd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hd)

KREF = 0.92          # 与第一/二组一致：新族参考强度定到 knif_ref
T = 0.60             # B2 回归的拟合 knif（描述量统一到这个 knif 上算）


def tukey(par, sc, r):
    """升余弦（Tukey）边：平通带 + cos 过渡 + 平阻带。
    par = 过渡带占 [0, sc] 的比例；sc = g2 归零的半径。
    g2 = 1 (r<=r1), 0.5(1+cos(pi*(r-r1)/(sc-r1))) (r1<r<sc), 0 (r>=sc)"""
    r1 = sc * (1.0 - par)
    g = np.zeros_like(r)
    m = r <= r1
    g[m] = 1.0
    m2 = (r > r1) & (r < sc)
    t = (r[m2] - r1) / max(sc - r1, 1e-12)
    g[m2] = 0.5 * (1.0 + np.cos(np.pi * t))
    return g


def make(kind, par, sc, r):
    if kind == "tukey":
        return tukey(par, sc, r)
    return hd.make_profile(kind, par, sc, r)


INC = {"ramp", "logistic", "erf"}


def tune(kind, par, r, target):
    def kunif(sc):
        return float(1.0 - make(kind, par, sc, r).mean())
    inc = kind in INC
    lo, hi = 1e-4, 200.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        k = kunif(mid)
        if (k > target) if inc else (k < target):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# 候选新族：(id, kind, par, 备注)
CAND = [
    # --- 全新形状类：升余弦边（真实滤波器的 canonical 形状，22 个已测族里没有）---
    ("TK30", "tukey", 0.30, "升余弦边，过渡带占 30%"),
    ("TK50", "tukey", 0.50, "升余弦边，过渡带占 50%"),
    ("TK70", "tukey", 0.70, "升余弦边，过渡带占 70%"),
    # --- 填已测类的参数缺口 ---
    ("W3", "weibull", 3.0, "Weibull p=3（2 与 4 之间）"),
    ("B2n", "butterworth", 2.0, "Butterworth n=2（1 与 8 之间）"),
    ("B4n", "butterworth", 4.0, "Butterworth n=4"),
    ("TS15", "twoscale", 1.5, "双尺度 R=1.5（TS2=2, M2=5 之间）"),
    ("TS3", "twoscale", 3.0, "双尺度 R=3"),
    ("LG60", "logistic", 0.60, "logistic 宽边（LG35 的宽版）"),
    ("EF60", "erf", 0.60, "erf 宽边（EF35 的宽版）"),
    ("RP50", "ramp", 0.50, "幂律斜坡 q=0.5"),
]


def main():
    tb = hd.build_tables()
    names, desc = tb["names"], tb["desc"]
    crel, crel_all = tb["crel"], tb["crel_all"]
    r = tb["r"]

    # ---------- 1. 已测 22 个剖面的定律拟合 ----------
    print("=" * 112)
    print(" 1. 已测 22 个剖面上三条候选定律的拟合（供对照）")
    print("=" * 112)
    fits = {}
    for tag, cf in [("共同u网格", crel), ("旧惯例", crel_all)]:
        y = np.array([cf(n) for n in names])
        row = {}
        # tw_lev 饱和逻辑斯蒂
        x = np.array([desc[n]["tw_lev"] for n in names])
        best = None
        for t0 in np.linspace(0.05, 3.0, 120):
            for p in np.linspace(0.3, 4.0, 76):
                X = np.column_stack([np.ones(len(x)),
                                     1.0 / (1.0 + (x / t0) ** (-p))])
                f = hd.ols(X, y)
                if f and (best is None or f["rmse"] < best[0]):
                    best = (f["rmse"], f, t0, p)
        _, f, t0, p = best
        row["tw"] = (f, t0, p)
        # B2 线性
        xb = np.array([desc[n]["B2"] for n in names])
        row["B2"] = hd.ols(np.column_stack([np.ones(len(xb)), xb]), y)
        # supp20 线性
        xs = np.array([desc[n]["supp20"] for n in names])
        row["supp20"] = hd.ols(np.column_stack([np.ones(len(xs)), xs]), y)
        fits[tag] = row
        print(f"\n [{tag}]")
        print(f"   tw_lev : C = {f['coef'][0]:.3f} + ({f['coef'][1]:.3f})*sig(tw;"
              f"t0={t0:.2f},p={p:.2f})   R2={f['r2']:.3f}")
        print(f"   B2     : C = {row['B2']['coef'][0]:.3f} + "
              f"({row['B2']['coef'][1]:.3f})*B2          R2={row['B2']['r2']:.3f}")
        print(f"   supp20 : C = {row['supp20']['coef'][0]:.3f} + "
              f"({row['supp20']['coef'][1]:.3f})*supp20     R2={row['supp20']['r2']:.3f}")

    # 用「共同u网格」的拟合作主预言；bootstrap 300 次估预言不确定度
    rng = np.random.default_rng(0)
    y = np.array([crel(n) for n in names])
    x = np.array([desc[n]["tw_lev"] for n in names])
    f0, t0, p0 = fits["共同u网格"]["tw"]
    sig = 1.0 / (1.0 + (x / t0) ** (-p0))

    def pred(tw, coef=None):
        c = f0["coef"] if coef is None else coef
        return c[0] + c[1] / (1.0 + (tw / t0) ** (-p0))

    boots = []
    for _ in range(400):
        idx = rng.integers(0, len(y), len(y))
        Xb = np.column_stack([np.ones(len(idx)), sig[idx]])
        fb = hd.ols(Xb, y[idx])
        if fb:
            boots.append(fb["coef"])
    boots = np.array(boots)

    # ---------- 2. 候选新族的描述量与三条定律的预言 ----------
    print("\n" + "=" * 112)
    print(" 2. 候选新族：描述量与三条定律的预言（tw_lev 定律用 bootstrap 68% 区间）")
    print("=" * 112)
    print(f" {'族':<6} | {'kind':<12} | {'par':>5} | {'tw_lev':>7} | {'B2':>6} | "
          f"{'supp20':>7} | {'C(tw)':>7} | {'68%半宽':>8} | {'C(B2)':>7} | "
          f"{'C(supp20)':>9} | {'最大分歧':>8}")
    print(" " + "-" * 108)
    rows = []
    for fid, kind, par, note in CAND:
        sc = tune(kind, par, r, KREF)
        a = 1.0 - make(kind, par, sc, r)
        a = a * (T / float(a.mean()))
        d = hd.descriptors(a, r)
        tw = d["tw_lev"]
        if not np.isfinite(tw):
            print(f" {fid:<6} | {kind:<12} | {par:>5.2f} | tw_lev 无定义，跳过")
            continue
        pb = np.array([b[0] + b[1] / (1.0 + (tw / t0) ** (-p0)) for b in boots])
        lo68, hi68 = np.percentile(pb, 16), np.percentile(pb, 84)
        ct = pred(tw)
        cb = fits["共同u网格"]["B2"]["coef"][0] + \
            fits["共同u网格"]["B2"]["coef"][1] * d["B2"]
        cs = fits["共同u网格"]["supp20"]["coef"][0] + \
            fits["共同u网格"]["supp20"]["coef"][1] * d["supp20"]
        dis = max(abs(ct - cb), abs(ct - cs)) / ct
        rows.append((fid, kind, par, note, tw, d, ct, (hi68 - lo68) / 2, cb, cs, dis))

    # 已测 22 族的 tw_lev 覆盖（用于找缺口）
    xt = sorted(desc[n]["tw_lev"] for n in names)
    print("\n 已测 22 族的 tw_lev 覆盖: " + "  ".join(f"{v:.2f}" for v in xt))
    for fid, kind, par, note, tw, d, ct, hw, cb, cs, dis in rows:
        print(f" {fid:<6} | {kind:<12} | {par:>5.2f} | {tw:>7.3f} | {d['B2']:>6.3f} | "
              f"{d['supp20']:>7.3f} | {ct:>7.3f} | {hw:>8.3f} | {cb:>7.3f} | "
              f"{cs:>9.3f} | {dis:>7.1%}")

    # ---------- 3. 缺口与对抗性 ----------
    print("\n" + "=" * 112)
    print(" 3. 缺口定位与对抗性选族")
    print("=" * 112)
    gaps = []
    for i in range(len(xt) - 1):
        if xt[i + 1] - xt[i] > 0.25:
            gaps.append((xt[i], xt[i + 1]))
    print(" tw_lev 缺口（相邻已测值差 > 0.25）: " +
          ("  ".join(f"[{a:.2f},{b:.2f}]" for a, b in gaps) if gaps else "无"))
    print("\n 按对抗性排序（最大分歧降序）；分歧大 => 能区分定律，不只是再加一个点")
    for fid, kind, par, note, tw, d, ct, hw, cb, cs, dis in \
            sorted(rows, key=lambda t: -t[10]):
        ingap = any(a < tw < b for a, b in gaps)
        print(f"   {fid:<6} tw={tw:>6.3f} {'[在缺口内]' if ingap else '':<11} "
              f"C(tw)={ct:.3f}±{hw:.3f}  C(B2)={cb:.3f}  C(supp20)={cs:.3f}  "
              f"分歧={dis:>5.1%}   {note}")


if __name__ == "__main__":
    main()
