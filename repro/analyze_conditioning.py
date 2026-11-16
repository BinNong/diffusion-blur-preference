"""修正分析：kunif 作为回归量的条件数灾难，以及正确的比较方式

背景（本文件要修正的问题）：
  dump_spectral.py / sweep_butterworth.py 把 log Lambda 回归到 log(kunif) 上，
  报告"幂律指数 b_f 是核的性质"。但 kunif = <1-|h|^2>_f 对高斯核有闭式
      kunif = 1 - 1/(pi sigma)^2
  即在 sigma >= 1 后饱和到 0.92-0.99。整个 theta 网格只覆盖 kunif 的
  0.13-0.52 dex，且最弱模糊那一个点独占 74%-98% 的杠杆。
  后果：
    (1) 指数由拟合区间决定，不是核的性质
    (2) 跨族比较指数无效——各族 theta->kunif 映射不同，杠杆结构不同
    (3) Butterworth 里 corr(width, b) = -0.957 很大程度上是
        "kunif 范围随 n 压缩"的机械后果

本文件做四件事：
  A. 杠杆诊断（hat 值、去掉最优点后指数怎么变）
  B. 局部斜率 d log Lambda / d log kunif（证明不是幂律）
  C. 改用良态轴：高斯等效宽度 sigma_eff = 1/(pi sqrt(1-kunif))
  D. 正确的跨族比较：在匹配 kunif 的实测点上做配对 t 检验

用法： python3 analyze_conditioning.py
"""

import json
import math
import os
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")


# ---------------------------------------------------------------- 基础工具
def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sst = sum((y - my) ** 2 for y in ys)
    ssr = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return dict(a=a, b=b, r2=1 - ssr / sst if sst > 0 else float("nan"))


def hat(xs):
    n = len(xs)
    mx = sum(xs) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return [float("nan")] * n
    return [1.0 / n + (x - mx) ** 2 / sxx for x in xs]


def sigma_eff(kunif):
    """高斯等效宽度：与给定 kunif 相同平均衰减的连续高斯宽度。
    高斯核 kunif = 1 - 1/(pi sigma)^2  =>  sigma = 1/(pi sqrt(1-kunif))。"""
    if kunif >= 1.0:
        return float("inf")
    return 1.0 / (math.pi * math.sqrt(1.0 - kunif))


def paired(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = [x - y for x, y in zip(a, b)]
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 0 else float("nan")
    return m, se, (m / se if se and se > 0 else float("nan")), n


def boot_slope(per_img, xs, reps=400, seed=1):
    """对图像重采样，返回斜率的 bootstrap SE。per_img: (n_cfg, n_img)。"""
    n_cfg, n_img = len(per_img), len(per_img[0])
    rng = _Rng(seed)
    out = []
    for _ in range(reps):
        idx = [rng.below(n_img) for _ in range(n_img)]
        ys = [sum(r[i] for i in idx) / n_img for r in per_img]
        ok = [i for i in range(n_cfg) if ys[i] > 0 and xs[i] > 0]
        if len(ok) < 4:
            continue
        f = ols([xs[i] for i in ok], [math.log(ys[i]) for i in ok])
        if f:
            out.append(f["b"])
    if len(out) < 20:
        return float("nan")
    mu = sum(out) / len(out)
    return math.sqrt(sum((x - mu) ** 2 for x in out) / (len(out) - 1))


class _Rng:
    """极简 LCG，避免依赖 numpy。"""

    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def below(self, n):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s % n


# ---------------------------------------------------------------- 数据
def load_family_sweep():
    """返回 {tag: {(family, theta): record}}，record 带 kunif（来自 spectral_functionals）。"""
    sf = json.load(open(os.path.join(OUT, "spectral_functionals.json")))["rows"]
    kun = {(r["tag"], r["family"], round(r["theta"], 4)): r["kunif"] for r in sf}
    out = {}
    for tag, fn in [("ffhq", "sweep_lambda_theta_ffhq.json"),
                    ("afhq", "sweep_lambda_theta_afhqv2-64x64-uncond-vp.json"),
                    ("imnet", "sweep_lambda_theta_imagenet-64x64-cond-adm.json")]:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            continue
        recs = json.load(open(p))["records"]
        t0 = min(r["t"] for r in recs)
        d = {}
        for r in recs:
            if r["t"] != t0:
                continue
            key = (r["family"], round(r["theta"], 4))
            k = kun.get(("ffhq.json" if tag == "ffhq" else
                         ("afhqv2-64x" if tag == "afhq" else "imagenet-6"),
                         r["family"], round(r["theta"], 4)))
            r = dict(r)
            r["kunif"] = k if k is not None else float("nan")
            d[key] = r
        out[tag] = d
    return out


def main():
    fam = load_family_sweep()
    tagname = {"ffhq": "ffhq.json", "afhq": "afhqv2-64x"}

    # ============================================================ A. 杠杆
    print("=" * 112)
    print(" A. kunif 作为回归量的杠杆诊断")
    print("=" * 112)
    print(f"{'模型/族':<20} | {'n':>2} | {'b(kunif)':>9} | {'R^2':>6} | "
          f"{'kunif 跨度':>10} | {'hat(最弱)':>9} | {'去掉后 b':>8} | {'Δb':>6}")
    print("-" * 112)
    for tag in sorted(fam):
        for f in sorted({k[0] for k in fam[tag]}):
            pts = sorted([(r["kunif"], r["lam_mean"], r["theta"], r["lam_per_img"])
                          for k, r in fam[tag].items()
                          if k[0] == f and r["kunif"] > 1e-9 and r["lam_mean"] > 0],
                         key=lambda p: p[0])
            if len(pts) < 4:
                continue
            xs = [math.log(p[0]) for p in pts]
            ys = [math.log(p[1]) for p in pts]
            fit = ols(xs, ys)
            if fit is None:
                continue
            hs = hat(xs)
            i = max(range(len(pts)), key=lambda j: hs[j])
            rest = [p for j, p in enumerate(pts) if j != i]
            f2 = ols([math.log(p[0]) for p in rest], [math.log(p[1]) for p in rest])
            span = math.log(pts[-1][0] / pts[0][0])
            print(f"{tag+'/'+f:<20} | {len(pts):>2} | {fit['b']:>9.2f} | {fit['r2']:>6.3f} | "
                  f"{span:>10.3f} | {hs[i]:>9.3f} | "
                  f"{f2['b'] if f2 else float('nan'):>8.2f} | "
                  f"{(f2['b']-fit['b']) if f2 else float('nan'):>+6.2f}")
    print("-" * 112)
    print(" hat 接近 1 => 该点独占杠杆，指数基本由它一个点决定")
    print(" kunif 跨度 < 0.2 dex => 回归量几乎恒定，斜率数值上无意义")

    # ============================================================ B. 局部斜率
    print()
    print("=" * 112)
    print(" B. 局部斜率 d log Lambda / d log kunif（随 kunif 大幅变化 => 不是幂律）")
    print("=" * 112)
    for tag in sorted(fam):
        for f in sorted({k[0] for k in fam[tag]}):
            pts = sorted([(r["kunif"], r["lam_mean"], r["theta"])
                          for k, r in fam[tag].items()
                          if k[0] == f and r["kunif"] > 1e-9 and r["lam_mean"] > 0],
                         key=lambda p: p[0])
            if len(pts) < 3:
                continue
            sl = []
            for i in range(len(pts) - 1):
                dk = math.log(pts[i + 1][0] / pts[i][0])
                if abs(dk) > 1e-9:
                    sl.append(math.log(pts[i + 1][1] / pts[i][1]) / dk)
            if sl:
                print(f"  {tag+'/'+f:<20} 极差 {max(sl)-min(sl):5.1f}   "
                      + "  ".join(f"{v:+6.2f}" for v in sl))

    # ============================================================ C. sigma_eff
    print()
    print("=" * 112)
    print(" C. 改用良态轴：高斯等效宽度 sigma_eff = 1/(pi sqrt(1-kunif))")
    print("=" * 112)
    print(f"{'模型/族':<20} | {'n':>2} | {'b(sigma_eff)':>12} | {'boot SE':>8} | "
          f"{'R^2':>6} | {'sigma_eff 范围':>16}")
    print("-" * 112)
    for tag in sorted(fam):
        for f in sorted({k[0] for k in fam[tag]}):
            pts = sorted([(r["kunif"], r["lam_mean"], r["theta"], r["lam_per_img"])
                          for k, r in fam[tag].items()
                          if k[0] == f and r["kunif"] > 1e-9 and r["lam_mean"] > 0],
                         key=lambda p: p[0])
            if len(pts) < 4:
                continue
            se = [sigma_eff(p[0]) for p in pts]
            if not all(math.isfinite(v) and v > 0 for v in se):
                continue
            xs = [math.log(v) for v in se]
            ys = [math.log(p[1]) for p in pts]
            fit = ols(xs, ys)
            if fit is None:
                continue
            L = [len(p[3]) for p in pts]
            bse = boot_slope([p[3] for p in pts], xs) if len(set(L)) == 1 \
                else float("nan")
            print(f"{tag+'/'+f:<20} | {len(pts):>2} | {fit['b']:>12.2f} | {bse:>8.2f} | "
                  f"{fit['r2']:>6.3f} | {se[0]:>7.3f} - {se[-1]:>7.3f}")
    print("-" * 112)
    print(" sigma_eff 把饱和区 [0.99,1] 拉开，跨度从 0.13 dex 变成 ~1 dex")

    # ============================================================ D. 匹配比较
    print()
    print("=" * 112)
    print(" D. 正确的跨族比较：匹配 kunif 的实测点，配对 t 检验（同 20 张图）")
    print("=" * 112)
    for tag in ["ffhq", "afhq"]:
        if tag not in fam:
            continue
        fams = sorted({k[0] for k in fam[tag]})
        print(f"\n  [{tag}] 每行：以 defocus 为基准，找其他族中 kunif 最近的实测点")
        print(f"  {'对照':<20} | {'kunif差':>7} | {'Lambda 比':>9} | "
              f"{'配对差':>8} | {'SE':>7} | {'t':>7}")
        print("  " + "-" * 72)
        base = sorted([(r["kunif"], r["lam_mean"], r["theta"], r["lam_per_img"])
                       for k, r in fam[tag].items()
                       if k[0] == "defocus" and r["kunif"] > 1e-9
                       and r["lam_mean"] > 0], key=lambda p: p[0])
        for bp in base:
            for f in fams:
                if f == "defocus":
                    continue
                cand = [(r["kunif"], r["lam_mean"], r["theta"], r["lam_per_img"])
                        for k, r in fam[tag].items()
                        if k[0] == f and r["kunif"] > 1e-9 and r["lam_mean"] > 0]
                if not cand:
                    continue
                q = min(cand, key=lambda p: abs(p[0] - bp[0]))
                dk = (q[0] - bp[0]) / bp[0]
                if abs(dk) > 0.05:
                    continue
                m, se_, tt, n = paired(q[3], bp[3])
                print(f"  {f+'/'+format(q[2],'g')+' vs def/'+format(bp[2],'g'):<20} | "
                      f"{100*dk:>+6.1f}% | {q[1]/bp[1]:>9.3f} | {m:>8.0f} | "
                      f"{se_:>7.0f} | {tt:>+7.1f}")
        print("  kunif 差 > 5% 的行已略去（模糊强度不可比）")

    # ============================================================ E. Butterworth
    print()
    print("=" * 112)
    print(" E. Butterworth 受控实验的重新审视：kunif 范围随 n 压缩")
    print("=" * 112)
    p = os.path.join(OUT, "sweep_butterworth.json")
    if os.path.exists(p):
        recs = json.load(open(p))["records"]
        t0 = min(r["t"] for r in recs)
        rr = [r for r in recs if r["t"] == t0]
        print(f"  t={t0:g}，每阶 5 个 rho_c")
        print(f"  {'n':>3} | {'b(kunif)':>9} | {'kunif 跨度':>10} | {'logL 跨度':>9} | "
              f"{'跨度比':>7} | {'宽度范围':>14}")
        print("  " + "-" * 76)
        for n in sorted({r["order"] for r in rr}):
            pts = sorted([(r["kunif"], r["lam_mean"], r["width"]) for r in rr
                          if r["order"] == n], key=lambda p: p[0])
            xs = [math.log(p[0]) for p in pts]
            ys = [math.log(p[1]) for p in pts]
            fit = ols(xs, ys)
            span_k = xs[-1] - xs[0]
            span_l = ys[-1] - ys[0]
            print(f"  {n:>3} | {fit['b']:>9.2f} | {span_k:>10.3f} | {span_l:>9.3f} | "
                  f"{span_l/span_k:>7.2f} | {pts[0][2]:>6.2f} - {pts[-1][2]:>5.2f}")
        print("  " + "-" * 76)
        print("  logL 跨度基本恒定(~1.0)，kunif 跨度随 n 单调压缩 => 斜率机械上升")
        print()
        print("  固定 rho_c、只改 n 时 Lambda 的变化（控制住截止频率）：")
        print(f"  {'rho_c':>6} | " + " | ".join(f"{'n='+str(n):>14}" for n in
                                                sorted({r['order'] for r in rr})))
        print("  " + "-" * 92)
        for rc in sorted({r["rho_c"] for r in rr}):
            cells = []
            for n in sorted({r["order"] for r in rr}):
                q = [r for r in rr if r["order"] == n and r["rho_c"] == rc]
                cells.append(f"{q[0]['lam_mean']:>8.0f}@{q[0]['kunif']:.3f}"
                             if q else f"{'--':>14}")
            print(f"  {rc:>6g} | " + " | ".join(f"{c:>14}" for c in cells))
        print("  每格 = Lambda@kunif。固定 rho_c 下 Lambda 对 n 非单调（在 n=2 附近最高）")


if __name__ == "__main__":
    main()
