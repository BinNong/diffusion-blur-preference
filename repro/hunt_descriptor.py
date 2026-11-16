"""描述量搜猎：找一个能区分「衰减铺开」与「二值化」的判据（预官版）。

背景（RESULTS §3.9(o)）
----------------------
B2 = <a^2>/<a> 在 17 个剖面上 R2=0.825、LOO 0.051 dex、有物理诠释，但五个全新形状类的
预官检验否定了它：匹配 B2 in [0.66,0.71] 处 C 有 1.61x 散度、关系非单调。
保留下来的正面结果只有定性方向 + 「B2 in [0.607,0.630] 的铺开剖面 C 紧聚 0.94-1.04」。

已测 22 个剖面按 C 恰好分成两组：
    C < 0.90（5 个）: W8 0.839, B8 0.837, W16 0.799, W32 0.757, lowpass 0.682
    C >= 0.90（17 个）: 其余全部（含 TS2 1.096, EF35 0.976, RP35 1.043, W4 0.946）
=> 需要一个判据把这 5 个挑出来，且在匹配判值处 C 必须紧聚（B2 就死在这条上）。

为什么换尺度不变的描述量
------------------------
B2 只齐次一次（B2(c*a)=c*B2(a)），所以必须靠"标度协变"技巧才能跨组比较；而且实验里
各组 knif_ref 不同（0.92 vs 0.60）、beta 不同，绝对电平的描述量（旧 sharp 用 0.1/0.5/0.9，
beta 压缩后 a 的上限只有 0.652，0.9 永远够不着 => sharp 对 beta 缩放族全是 NaN）不可比。
本脚本的描述量全部**尺度不变**：a -> c*a 时描述量不变。衰减剖面的形状本来就是 beta 不变量
（a_op = beta*a_ref 精确成立），所以尺度不变量才是无混淆的正确选择。

做法
----
1. 从四个 sweep JSON 重建 22 条实测 Λ(u) 曲线；C 在**共同 u 网格**上重算
   （各组 u 窗口不同，直接取中位数会把 u 漂移混进 shape 效应）。
2. 用 JSON 里存的 kind/par/scale/kunif_ref **精确**重建每条剖面的 a_ref(f)
   （与 sweep_law_test.py 的构造逐行一致；真实核用 putils 的核再 rfft2）。
3. 算一批尺度不变描述量。
4. 每个描述量报四项：in-sample R2、LOO RMSE、Spearman、以及
   **决定性判据——匹配描述量处 C 的散度**（分 bin 后报最紧 bin 的 max/min）。

注意：第 3-4 步是**探索性**的（22 个剖面全都已测，任何胜出者都是 in-sample）。
所以脚本最后按胜出描述量挑 3 个**对抗性新族**并写出预官值，真正的检验要跑 GPU。

用法： venv2/bin/python hunt_descriptor.py
"""

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
sys.path.insert(0, os.path.join(PROJ, "code", "blind_deblur_diffusion"))

SMOOTH_OLD = [("gaussian", 1.0), ("gaussian", 2.0), ("defocus", 1.0),
              ("defocus", 2.0), ("airy", 4.0), ("motion", 3.0), ("motion", 7.0)]
P1_RANGES = {(0.65, 0.40): (-1.68, -1.31), (0.40, 0.30): (-0.91, -0.82),
             (0.30, 0.15): (-0.90, -0.68)}
FILES = ["sweep_profile_shape.json", "sweep_law_test.json",
         "sweep_steepness.json", "sweep_b2test.json"]
T = 0.60            # 描述量统一缩到 knif=T（尺度不变量不受影响，只为和非不变量对齐）


# ---------------------------------------------------------------- 剖面重建
def halfplane():
    fx = np.fft.fftfreq(64) * 64
    fy = np.fft.fftfreq(64) * 64
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)[:, :64 // 2 + 1]
    return R / R.max()


def make_profile(kind, par, sc, r):
    """与 sweep_law_test.make_profile 逐行一致。"""
    if kind == "weibull":
        return np.exp(-(r / sc) ** par)
    if kind == "butterworth":
        return 1.0 / (1.0 + (r / sc) ** (2 * par))
    if kind == "twoscale":
        return 0.5 * np.exp(-(r / sc) ** 2) + 0.5 * np.exp(-(r / (sc * par)) ** 2)
    if kind == "ramp":
        return np.clip((r / sc) ** par, 0.0, 1.0)
    if kind == "logistic":
        return 1.0 / (1.0 + np.exp(-(r - sc) / par))
    if kind == "erf":
        from math import erf
        return 0.5 * (1.0 + np.vectorize(erf)((r - sc) / par))
    if kind == "tukey":
        # 升余弦边（与 sweep_law_test.make_profile 一致）：par = 过渡带占 [0,sc] 的比例
        r1 = sc * (1.0 - par)
        g = np.zeros_like(r)
        g[r <= r1] = 1.0
        m = (r > r1) & (r < sc)
        t = (r[m] - r1) / max(sc - r1, 1e-12)
        g[m] = 0.5 * (1.0 + np.cos(np.pi * t))
        return g
    raise ValueError(kind)


def real_g2(fam, th):
    """真实核的 |h|^2（与 sweep_profile_shape.make_ref 一致）。"""
    import torch
    from putils.convolution import GaussianBlur, DefocusBlur, AiryDisc, MotionBlur
    if fam == "lowpass":
        R = halfplane() * halfplane().max()
        return (R <= th * R.max()).astype(np.float64)
    cls = {"gaussian": GaussianBlur, "defocus": DefocusBlur,
           "airy": AiryDisc, "motion": MotionBlur}[fam]
    k = cls(kernel_size=15, device="cpu", dtype=torch.float64).kernel(float(th))
    kk = k[None, None] if k.ndim == 2 else k
    kh, kw = kk.shape[-2:]
    pad = torch.zeros((1, 1, 64, 64), dtype=torch.float64)
    pad[:, :, (64 - kh) // 2:(64 - kh) // 2 + kh,
        (64 - kw) // 2:(64 - kw) // 2 + kw] = kk.to(torch.float64)
    return torch.fft.rfft2(pad, dim=(-2, -1)).abs().square()[0, 0].numpy()


# ---------------------------------------------------------------- 数据
def load():
    prof, meta = defaultdict(list), {}
    for fn in FILES:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            print("[缺少]", fn)
            continue
        d = json.load(open(p))
        for rec in d["rows"]:
            m = rec["meta"]
            if "fid" in m:
                nm = m["fid"]
            elif "family" in m:
                nm = f"{m['family']}/{m['theta']:g}"
            else:
                continue          # B 组理想算子没有 family/fid
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


def interp(curve, u):
    ks = [c[0] for c in curve]
    if u < min(ks) or u > max(ks):
        return None
    for i in range(len(ks) - 1):
        a, b = ks[i], ks[i + 1]
        if u <= a and u >= b:
            if abs(b - a) < 1e-12:
                return None          # 相邻 u 相同（不同组的浮点差），跳过
            if curve[i][1] <= 0 or curve[i + 1][1] <= 0:
                return None
            w = (math.log(u) - math.log(a)) / (math.log(b) - math.log(a))
            return math.exp(math.log(curve[i][1]) * (1 - w)
                            + math.log(curve[i + 1][1]) * w)
    return None


# ---------------------------------------------------------------- 描述量
def radial(a, r, nb=200):
    """径向均值剖面。返回 (rr, prof)，只保留非空 bin。"""
    edges = np.linspace(0, r.max() + 1e-9, nb + 1)
    idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, nb - 1).reshape(r.shape)
    rr, pr = [], []
    for b in range(nb):
        msk = idx == b
        if msk.any():
            rr.append(float(r[msk].mean()))
            pr.append(float(a[msk].mean()))
    return np.array(rr), np.array(pr)


def _cross(rr, prof, lv):
    """所有穿越 lv 的半径（线性插值）。"""
    out = []
    for i in range(len(prof) - 1):
        x, y = prof[i] - lv, prof[i + 1] - lv
        if x * y <= 0 and x != y:
            out.append(rr[i] + (lv - prof[i]) * (rr[i + 1] - rr[i])
                       / (prof[i + 1] - prof[i]))
    return out


def _q(cum, rm, f):
    i = int(np.searchsorted(cum, f))
    return float(rm[min(i, len(rm) - 1)])


def descriptors(a, r):
    """a: 衰减剖面（任意幅度）；r: 归一半径。全部尺度不变。"""
    m = float(a.mean())
    o = {}
    # --- 旧描述量（留着对照）---
    o["B2"] = float((a ** 2).mean() / m)                    # 齐次一次，非尺度不变
    o["B3"] = float((a ** 3).mean() / m)
    o["rbar"] = float((r * a).sum() / a.sum())
    rr, pr = radial(a, r)
    # --- 新的尺度不变量 ---
    # (1) 电平法相对过渡宽度：levels 取径向剖面 min+{0.1,0.5,0.9}*range
    lo, hi = float(pr.min()), float(pr.max())
    if hi - lo > 1e-12:
        cs = [_cross(rr, pr, lo + f * (hi - lo)) for f in (0.1, 0.5, 0.9)]
        if all(len(c) >= 1 for c in cs):
            r1, r5, r9 = cs[0][0], cs[1][0], cs[2][0]
            o["tw_lev"] = float(abs(r9 - r1) / r5) if r5 > 1e-9 else np.nan
            o["_r10"], o["_r50"], o["_r90"] = r1, r5, r9
        else:
            o["tw_lev"] = np.nan
    else:
        o["tw_lev"] = np.nan
    # (1b) 同上但 levels 相对**剖面自身均值**（所有剖面都缩到同一 knif，所以这是共同参照）；
    #      不随剖面 floor 高低而改变含义。max(a) < 0.9*mean 时该项无定义。
    cs2 = [_cross(rr, pr, f * m) for f in (0.1, 0.5, 0.9)]
    if all(len(c) >= 1 for c in cs2):
        q1, q5, q9 = cs2[0][0], cs2[1][0], cs2[2][0]
        o["tw_mean"] = float(abs(q9 - q1) / q5) if q5 > 1e-9 else np.nan
    else:
        o["tw_mean"] = np.nan
    o["_amin"], o["_amax"] = lo, hi
    # (2) 质量法相对过渡宽度：p(r) ∝ |dā/dr|，取 10/50/90 分位半径
    d = np.abs(np.diff(pr))
    rm = 0.5 * (rr[1:] + rr[:-1])
    s = float(d.sum())
    if s > 0:
        cum = np.cumsum(d / s)
        r1, r5, r9 = _q(cum, rm, 0.1), _q(cum, rm, 0.5), _q(cum, rm, 0.9)
        o["tw_mass"] = float(abs(r9 - r1) / r5) if r5 > 1e-9 else np.nan
        # (3) 同 (2) 但按 bin 内模数加权（大半径模更多）
        edges = np.linspace(0, r.max() + 1e-9, 201)
        idx = np.clip(np.digitize(r.ravel(), edges) - 1, 0, 199).reshape(r.shape)
        cnt = np.array([(idx == b).sum() for b in range(200)], float)
        cnt = cnt[:len(d)]
        dw = d * np.maximum(cnt[:len(d)], 1.0)
        sw = float(dw.sum())
        if sw > 0:
            cumw = np.cumsum(dw / sw)
            w1, w5, w9 = _q(cumw, rm, 0.1), _q(cumw, rm, 0.5), _q(cumw, rm, 0.9)
            o["tw_massw"] = float(abs(w9 - w1) / w5) if w5 > 1e-9 else np.nan
        else:
            o["tw_massw"] = np.nan
    else:
        o["tw_mass"] = o["tw_massw"] = np.nan
    # (4) 支撑集测度：衰减超过自身均值 0.5 倍的频率占比
    o["supp50"] = float((a > 0.5 * m).mean())
    o["supp80"] = float((a > 0.8 * m).mean())
    o["supp20"] = float((a > 0.2 * m).mean())
    # (5) 传输 g2=1-a 的集中度（参与比）：铺开 -> 大，尖峰 -> 小
    g = 1.0 - a
    o["PR"] = float((g.sum() ** 2) / (g.size * (g ** 2).sum()))
    # (6) 与图像功率谱的重叠（1/f^2 代理）：衰减落在强信号区的比例
    S = 1.0 / (r ** 2 + 1e-4)
    o["wS"] = float((a * S).sum() / a.sum())
    o["wS_r"] = float((a / (r + 1e-3)).sum() / a.sum())
    # (7) 各向异性 + 振铃（沿用旧定义，尺度不变）
    nb2 = 64
    e2 = np.linspace(0, r.max() + 1e-9, nb2 + 1)
    i2 = np.clip(np.digitize(r.ravel(), e2) - 1, 0, nb2 - 1).reshape(r.shape)
    arad = np.zeros_like(a)
    for b in range(nb2):
        msk = i2 == b
        if msk.any():
            arad[msk] = a[msk].mean()
    vt = float(((a - m) ** 2).mean())
    o["aniso"] = math.sqrt(float(((a - arad) ** 2).mean()) / vt) if vt > 0 else 0.0
    prf = np.array([a[i2 == b].mean() if (i2 == b).any() else np.nan
                    for b in range(nb2)])
    prf = prf[np.isfinite(prf)]
    o["ring"] = float((np.diff(np.sign(np.diff(prf))) != 0).sum()) if len(prf) > 3 else 0.0
    return o


# ---------------------------------------------------------------- 统计
def ols(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float)
    A = X.T @ X
    try:
        coef = np.linalg.solve(A, X.T @ y)
    except np.linalg.LinAlgError:
        return None
    res = y - X @ coef
    sst = float(((y - y.mean()) ** 2).sum())
    return dict(coef=coef, r2=1 - float((res ** 2).sum()) / sst if sst > 0 else np.nan,
                resid=res, rmse=float(math.sqrt(float((res ** 2).mean()) / len(y))))


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum())))


def loo(X, y):
    errs = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        f = ols(X[m], y[m])
        if f is not None:
            errs.append(float(X[i] @ f["coef"] - y[i]))
    return (float(math.sqrt(np.mean(np.array(errs) ** 2))),
            float(np.max(np.abs(errs)))) if errs else (np.nan, np.nan)


def build_tables():
    """载入数据、重算 C、重建剖面、算描述量。返回 predict_tw.py 也要用的全部对象。"""
    prof, meta = load()
    r = halfplane()

    # ---- b(u) 基准：旧 7 个平滑族 ----
    seg_u, seg_b = [], []
    for a, b in sorted(P1_RANGES, reverse=True):
        bs = []
        for k in SMOOTH_OLD:
            nm = f"{k[0]}/{k[1]:g}"
            va, vb = interp(prof[nm], a), interp(prof[nm], b)
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
                w = (lx - math.log(seg_u[i])) / (math.log(seg_u[i + 1]) - math.log(seg_u[i]))
                return seg_b[i] * (1 - w) + seg_b[i + 1] * w
        return seg_b[-1]

    # ---- 共同 u 网格：所有 22 个剖面都有数据的 u（round 掉浮点差）----
    common = None
    for nm in prof:
        s = {round(c[0], 4) for c in prof[nm]}
        common = s if common is None else (common & s)
    Ucom = sorted(common, reverse=True)
    # 每个剖面的全部 u（用于和 analyze_b2test 的"逐剖面中位数"惯例对照）
    Uall = sorted({round(c[0], 3) for nm in prof for c in prof[nm]}, reverse=True)

    # ---- C(shape)：共同网格重算 + 逐剖面全 u 中位数（旧惯例）对照 ----
    C = {}
    for nm in prof:
        C[nm] = {}
        for u in Uall:
            v = interp(prof[nm], u)
            if v and v > 0:
                C[nm][u] = v * u ** (-b_of(u))
    mid = {}
    for u in Uall:
        vs = [C[f"{k[0]}/{k[1]:g}"][u] for k in SMOOTH_OLD
              if u in C.get(f"{k[0]}/{k[1]:g}", {})]
        if len(vs) >= 5:
            mid[u] = float(np.exp(np.mean([math.log(v) for v in vs])))

    def crel(nm):
        """共同 u 网格上的 C（主定义）。"""
        vs = [C[nm][u] / mid[u] for u in Ucom if u in C[nm] and u in mid]
        return float(np.median(vs)) if vs else np.nan

    def crel_all(nm):
        """逐剖面全部可用 u 的中位数（= analyze_b2test 的旧惯例，用作对照）。"""
        vs = [C[nm][u] / mid[u] for u in mid if u in C[nm]]
        return float(np.median(vs)) if vs else np.nan

    def cspread(nm):
        vs = [C[nm][u] / mid[u] for u in Ucom if u in C[nm] and u in mid]
        return (float(np.max(vs) / np.min(vs)) if len(vs) > 1 else np.nan)

    # ---- 剖面重建 ----
    araw = {}
    for nm in prof:
        m = meta[nm]
        if m["kind"] == "real":
            a = 1.0 - real_g2(m["fam"], m["th"])
        else:
            a = 1.0 - make_profile(m["kind"], m["par"], m["scale"], r)
        araw[nm] = a * (T / float(a.mean()))

    names = sorted(prof)
    desc = {nm: descriptors(araw[nm], r) for nm in names}
    return dict(prof=prof, meta=meta, r=r, names=names, desc=desc,
                crel=crel, crel_all=crel_all, cspread=cspread,
                Ucom=Ucom, Uall=Uall, b_of=b_of, araw=araw, C=C, mid=mid)


def main():
    tb = build_tables()
    prof, meta, r, names, desc = (tb["prof"], tb["meta"], tb["r"], tb["names"],
                                  tb["desc"])
    crel, crel_all, cspread = tb["crel"], tb["crel_all"], tb["cspread"]
    Ucom, Uall, C, mid = tb["Ucom"], tb["Uall"], tb["C"], tb["mid"]
    print("=" * 108)
    print(f" 载入 {len(prof)} 个剖面：" + "  ".join(sorted(prof)))
    print("=" * 108)
    print(f"\n 共同 u 网格（{len(Ucom)} 个）: " + "  ".join(f"{u:.4g}" for u in Ucom))
    print(" 各剖面 u 覆盖数: " + "  ".join(
        f"{nm}={len(prof[nm])}" for nm in sorted(prof)))
    print(f" 全体 u 网格（{len(Uall)} 个）: " + "  ".join(f"{u:.4g}" for u in Uall))

    # ---- 表 1：C 与 u 稳定性 ----
    print("\n" + "=" * 108)
    print(" 1. C(shape)（共同 u 网格，归一化到旧 7 平滑族均值）与 u 稳定性")
    print("=" * 108)
    print(f" {'剖面':<14} | {'kind':<12} | " +
          " | ".join(f"u={u:.3g}" for u in Ucom) +
          f" | {'C中位':>7} | {'u漂移':>6} | {'C旧惯例':>8}")
    print(" " + "-" * (52 + 14 * len(Ucom)))
    for nm in sorted(prof, key=lambda n: crel(n)):
        row = [f"{C[nm][u]/mid[u]:>7.3f}" if u in C[nm] and u in mid else f"{'--':>7}"
               for u in Ucom]
        print(f" {nm:<14} | {meta[nm]['kind']:<12} | " + " | ".join(row) +
              f" | {crel(nm):>7.3f} | {cspread(nm):>6.3f} | {crel_all(nm):>8.3f}")
    print(" " + "-" * (52 + 14 * len(Ucom)))
    print(" u漂移 = max/min over 共同网格；若接近 1 说明 C 确实是形状-only（定律自洽）")
    print(" C旧惯例 = analyze_b2test 的逐剖面全-u 中位数。两列差得大 => 之前的 C 有 u 窗口混淆")

    # ---- 表 2：描述量总表 ----
    print("\n" + "=" * 108)
    print(" 2. 描述量总表（剖面已缩到 knif=%.2f；尺度不变量不受缩放影响）" % T)
    print("=" * 108)
    keys = ["tw_lev", "tw_mean", "tw_mass", "tw_massw", "supp50", "supp80",
            "supp20", "PR", "wS", "wS_r", "rbar", "B2", "B3", "aniso", "ring"]
    y = np.array([crel(n) for n in names])
    print(f" {'剖面':<14} | {'C':>6} | " + " | ".join(f"{k:>8}" for k in keys))
    print(" " + "-" * (32 + 11 * len(keys)))
    for nm in sorted(names, key=lambda n: crel(n)):
        print(f" {nm:<14} | {crel(nm):>6.3f} | " +
              " | ".join(f"{desc[nm][k]:>8.3f}" for k in keys))
    print(" " + "-" * (32 + 11 * len(keys)))
    print(" tw_*  = 相对过渡带宽度（电平法 / 质量法 / 模数加权质量法）")
    print(" supp* = 衰减超过自身均值 {0.2,0.5,0.8} 倍的频率占比")
    print(" PR    = 传输 g2 的参与比 (Sg2)^2/(N*Sg2^2)，铺开->1 尖峰->0")
    print(" wS    = 衰减按 1/r^2 加权的重心（落在强信号区的程度）")

    # ---- 表 3：每个描述量的四项指标 ----
    print("\n" + "=" * 108)
    print(" 3. 候选描述量横评（22 个剖面）")
    print("=" * 108)
    print(f" {'描述量':<9} | {'跨度':>8} | {'R2':>7} | {'Spear':>6} | "
          f"{'LOO dex':>8} | {'max dex':>8} | {'最紧bin散度':>10} | {'bin宽度':>8}")
    print(" " + "-" * 82)
    rows = []
    for k in keys:
        x = np.array([desc[n][k] for n in names])
        if not np.isfinite(x).all() or x.std() == 0:
            print(f" {k:<9} | {'-- 有 NaN 或无跨度 --':>40}")
            continue
        f = ols(np.column_stack([np.ones(len(x)), x]), y)
        l1, l2 = loo(np.column_stack([np.ones(len(x)), x]), y)
        sp = spearman(x, y)
        # 匹配描述量处的 C 散度：按 x 排序后取宽度为 w 的滑窗，报最紧的
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        best = (np.inf, np.nan)
        for w in [0.10, 0.15, 0.20, 0.30]:
            for i in range(len(xs)):
                j = i
                while j < len(xs) and xs[j] - xs[i] <= w:
                    j += 1
                if j - i >= 3:
                    sp2 = float(ys[i:j].max() / ys[i:j].min())
                    if sp2 < best[0]:
                        best = (sp2, w)
        rows.append((k, f["r2"], sp, l1, best[0], best[1]))
        print(f" {k:<9} | {x.max()-x.min():>8.3f} | {f['r2']:>7.3f} | {sp:>6.2f} | "
              f"{l1:>8.3f} | {l2:>8.3f} | {best[0]:>10.3f} | {best[1]:>8.2f}")
    print(" " + "-" * 82)
    print(" 「最紧bin散度」= 在描述量跨度 <= bin宽度 且含 >=3 个剖面的窗口里，C 的 max/min。")
    print(" 这是 B2 当初失败的那条判据（B2 在 0.05 宽的窗里仍有 1.6x）。")

    # ---- 表 4：胜出者能否把 5 个低 C 剖面分开 ----
    print("\n" + "=" * 108)
    print(" 4. 决定性检验：胜出描述量能否分开 C<0.90 的 5 个剖面？")
    print("=" * 108)
    low = {nm for nm in names if crel(nm) < 0.90}
    print(f" C<0.90 的 {len(low)} 个: " + "  ".join(
        f"{nm}({crel(nm):.3f})" for nm in sorted(low, key=lambda n: crel(n))))
    if rows:
        # 按「最紧bin散度」升序 = 匹配处最紧的排前面
        rows.sort(key=lambda t: t[4])
        for k, r2, sp, l1, sp2, w in rows[:3]:
            x = np.array([desc[n][k] for n in names])
            thr_lo, thr_hi = x[[n in low for n in names]].min(), \
                x[[n in low for n in names]].max()
            xh = x[[n not in low for n in names]]
            gap = (thr_hi, xh.min()) if thr_hi < xh.min() else (xh.max(), thr_lo)
            print(f"\n [{k}] R2={r2:.3f} Spear={sp:.2f} LOO={l1:.3f} dex，"
                  f"最紧bin散度={sp2:.3f}（bin宽 {w}）")
            print(f"   低C组的 {k} 范围 [{thr_lo:.3f}, {thr_hi:.3f}]；"
                  f"高C组范围 [{xh.min():.3f}, {xh.max():.3f}]")
            sep = "完全分开" if thr_hi < xh.min() or thr_lo > xh.max() else "有重叠"
            print(f"   => {sep}")
            print("   按 %s 排序:" % k)
            for nm in sorted(names, key=lambda n: desc[n][k]):
                tag = "  <== C<0.90" if nm in low else ""
                print(f"      {nm:<14} {k}={desc[nm][k]:>8.3f}  C={crel(nm):.3f}{tag}")

    print("\n" + "=" * 108)
    print(" 5. 小结")
    print("=" * 108)
    print(" * 这一步是**探索性**的：22 个剖面全都已测，任何胜出者都只是 in-sample。")
    print(" * 要变成结论，必须按胜出描述量挑**对抗性新族**、先写预官值再跑 GPU。")
    print(" * 对抗性选族标准：描述量值落在低C组与高C组的**重叠区**，且形状类与已测 22 个不同。")

    # ---- 6. tw_lev 定律的拟合与两种 C 惯例的稳健性 ----
    print("\n" + "=" * 108)
    print(" 6. tw_lev -> C 的定律拟合（两种 C 惯例都试，看结论是否依赖惯例）")
    print("=" * 108)
    KEY = "tw_lev"
    for tag, cf in [("共同u网格", crel), ("逐剖面全u中位(旧惯例)", crel_all)]:
        yy = np.array([cf(n) for n in names])
        xx = np.array([desc[n][KEY] for n in names])
        ok = np.isfinite(yy) & np.isfinite(xx) & (xx > 0)
        xx, yy = xx[ok], yy[ok]
        print(f"\n [{tag}]  n={len(xx)}")
        print(f" {'形式':<26} | {'R2':>7} | {'残差dex':>8} | {'最大dex':>8}")
        print(" " + "-" * 58)
        best = None
        forms = []
        # (a) 线性 in tw
        forms.append(("C = a + b*tw", np.column_stack([np.ones(len(xx)), xx])))
        # (b) 线性 in log tw
        forms.append(("C = a + b*log(tw)",
                      np.column_stack([np.ones(len(xx)), np.log(xx)])))
        # (c) 饱和逻辑斯蒂：C = Clo + (Chi-Clo)/(1+(tw/t0)^-p)，网格搜 t0,p
        bg = None
        for t0 in np.linspace(0.05, 3.0, 60):
            for p in np.linspace(0.3, 4.0, 38):
                X = np.column_stack([np.ones(len(xx)),
                                     1.0 / (1.0 + (xx / t0) ** (-p))])
                f = ols(X, yy)
                if f and (bg is None or f["rmse"] < bg[0]):
                    bg = (f["rmse"], f, t0, p)
        if bg:
            _, f, t0, p = bg
            forms.append((f"C = Clo+(Chi-Clo)/(1+(tw/{t0:.2f})^-{p:.2f})",
                          np.column_stack([np.ones(len(xx)),
                                           1.0 / (1.0 + (xx / t0) ** (-p))])))
        # (d) C = a + b/tw^p
        bg2 = None
        for p in np.linspace(0.1, 3.0, 30):
            X = np.column_stack([np.ones(len(xx)), xx ** (-p)])
            f = ols(X, yy)
            if f and (bg2 is None or f["rmse"] < bg2[0]):
                bg2 = (f["rmse"], f, p)
        if bg2:
            _, f, p = bg2
            forms.append((f"C = a + b*tw^-{p:.2f}",
                          np.column_stack([np.ones(len(xx)), xx ** (-p)])))
        def todex(e_lin):
            """线性 C 误差 -> dex（C~1 时 0.01 的误差约 0.0043 dex）。"""
            return float(e_lin) / float(np.mean(np.abs(yy))) / math.log(10)

        for nm, X in forms:
            f = ols(X, yy)
            if not f:
                continue
            l1, l2 = loo(X, yy)
            print(f" {nm:<26} | {f['r2']:>7.3f} | {f['rmse']:>8.4f} | {todex(l2):>8.4f}"
                  f"   (LOO {todex(l1):.4f} dex)")
            if best is None or f["rmse"] < best[0]:
                best = (f["rmse"], nm, f, todex(l2))
        if best:
            print(f" => 最优形式 [{best[1]}] in-sample 残差 {todex(best[0]):.4f} dex，"
                  f"LOO 最大 {best[3]:.4f} dex（预官容差从两者取大）")
    print("\n 若两种惯例给出同一个胜出形式和同量级残差 => tw_lev 定律不依赖 C 的取值惯例。")


if __name__ == "__main__":
    main()
