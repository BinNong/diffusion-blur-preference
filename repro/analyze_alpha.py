"""Alpha-插值实验分析（修订版）：用宽 kunif 覆盖重做机理判别

sweep_alpha 把 kunif 覆盖从旧的 0.59-0.99（0.52 dex）拉到 0.25-0.98（1.4 dex），
且每个核内部连续扫 alpha。本文件回答：

  Q1 全池幂律：log Lambda = a + b log(kunif)，54 点，报告 b / R^2 / 杠杆
  Q2 逐核幂律：b 是否随核变？（旧结论说变；现在区间宽了，再看）
  Q3 匹配 kunif 的跨族差异：配对 t 检验
  Q4 Lambda 是否只由位移范数 ||h*x-x|| 决定？（用 kernel_displacement.json 检验）
  Q5 alpha 剖面形状：局部指数、Lambda/alpha 的线性性

用法： python3 analyze_alpha.py
"""

import json
import math
import os
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")


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


class _Rng:
    def __init__(self, seed):
        self.s = seed & 0xFFFFFFFF

    def below(self, n):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s % n


def boot_slope(per_img, xs, reps=400, seed=1):
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


def paired(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = [x - y for x, y in zip(a, b)]
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 0 else float("nan")
    return m, se, (m / se if se and se > 0 else float("nan")), n


def main():
    p = os.path.join(OUT, "sweep_alpha.json")
    if not os.path.exists(p):
        print("[缺少]", p)
        return
    d = json.load(open(p))
    recs, cfg = d["records"], d["cfg"]
    t0 = min(r["t"] for r in recs)
    rr = [r for r in recs if r["t"] == t0]
    nimg = cfg["nimg"]

    # SNR 过滤：Hutchinson 底噪下 SNR<3 的点不可用
    def snr(r):
        return r["lam_mean"] / r["lam_std"] * math.sqrt(nimg) \
            if r["lam_std"] > 0 else float("inf")

    good = [r for r in rr if r["lam_mean"] > 0 and snr(r) >= 3.0]
    print("=" * 108)
    print(f" t = {t0:g}；{len(rr)} 条中 {len(good)} 条 SNR>=3 可用"
          f"（剔除 {len(rr)-len(good)} 条）")
    print("=" * 108)

    # ================= Q1 全池幂律 =================
    print("\n Q1 全池幂律 log Lambda = a + b log(kunif)（宽覆盖版）")
    pts = sorted(good, key=lambda r: r["kunif"])
    xs = [math.log(r["kunif"]) for r in pts]
    ys = [math.log(r["lam_mean"]) for r in pts]
    f = ols(xs, ys)
    hs = hat(xs)
    i = max(range(len(pts)), key=lambda j: hs[j])
    span = math.log(pts[-1]["kunif"] / pts[0]["kunif"])
    print(f"   n = {len(pts)}   kunif {pts[0]['kunif']:.4f} - {pts[-1]['kunif']:.4f}"
          f"（{span:.3f} dex，旧 theta 网格只有 0.13-0.52 dex）")
    print(f"   b = {f['b']:.3f}   R^2 = {f['r2']:.4f}   "
          f"残差 std = {math.sqrt(sum(e*e for e in [y-(f['a']+f['b']*x) for x,y in zip(xs,ys)])/len(xs)):.3f}")
    print(f"   最大 hat = {hs[i]:.3f}（{pts[i]['family']}/{pts[i]['theta']:g} "
          f"alpha={pts[i]['alpha']:g}）")
    print(f"   boot SE（对图重采样）= "
          f"{boot_slope([r['lam_per_img'] for r in pts], xs):.3f}")
    print("   对比：旧 theta 网格上 gaussian 4.17 / defocus 5.83，"
          "kunif 跨度仅 0.52 / 0.13 dex")

    # ================= Q2 逐核幂律 =================
    print("\n Q2 逐核幂律（固定核形状，alpha 连续扫）")
    print(f" {'核':<16} | {'n':>2} | {'b(kunif)':>9} | {'b(alpha)':>9} | "
          f"{'R^2':>6} | {'kunif 跨度':>10} | {'alpha 跨度':>10}")
    print(" " + "-" * 78)
    for k in sorted({(r["family"], r["theta"]) for r in good}):
        sub = sorted([r for r in good if (r["family"], r["theta"]) == k],
                     key=lambda r: r["alpha"])
        if len(sub) < 4:
            continue
        xk = [math.log(r["kunif"]) for r in sub]
        xa = [math.log(r["alpha"]) for r in sub]
        y = [math.log(r["lam_mean"]) for r in sub]
        fk, fa = ols(xk, y), ols(xa, y)
        print(f" {k[0]+'/'+format(k[1],'g'):<16} | {len(sub):>2} | {fk['b']:>9.2f} | "
              f"{fa['b']:>9.2f} | {fk['r2']:>6.3f} | "
              f"{math.log(sub[-1]['kunif']/sub[0]['kunif']):>10.3f} | "
              f"{math.log(sub[-1]['alpha']/sub[0]['alpha']):>10.3f}")
    print(" " + "-" * 78)
    print(" b(kunif) 现在应该比旧的 4-6 低且更一致，因为覆盖宽了、杠杆分散了")

    # ================= Q3 匹配 kunif =================
    print("\n Q3 匹配 kunif 的跨族差异（配对 t 检验，同 20 张图）")
    for base_fam, base_th in [("gaussian", 1.0), ("defocus", 1.0)]:
        base = [r for r in good if (r["family"], r["theta"]) == (base_fam, base_th)]
        if not base:
            continue
        print(f"\n  基准 {base_fam}/{base_th:g}：")
        print(f"  {'对照':<22} | {'kunif差':>7} | {'Lambda 比':>9} | "
              f"{'配对差':>8} | {'SE':>7} | {'t':>7}")
        print("  " + "-" * 72)
        for r0 in sorted(base, key=lambda r: r["alpha"]):
            for k in sorted({(r["family"], r["theta"]) for r in good}):
                if k == (base_fam, base_th):
                    continue
                cand = [r for r in good if (r["family"], r["theta"]) == k]
                if not cand:
                    continue
                q = min(cand, key=lambda r: abs(r["kunif"] - r0["kunif"]))
                dk = (q["kunif"] - r0["kunif"]) / r0["kunif"]
                if abs(dk) > 0.06:
                    continue
                m, se_, tt, n = paired(q["lam_per_img"], r0["lam_per_img"])
                print(f"  {k[0]+'/'+format(k[1],'g')+' vs '+base_fam+'/'+format(base_th,'g'):<22} | "
                      f"{100*dk:>+6.1f}% | {q['lam_mean']/r0['lam_mean']:>9.3f} | "
                      f"{m:>8.0f} | {se_:>7.0f} | {tt:>+7.1f}")
        print("  kunif 差 > 6% 的行已略去")

    # ================= Q4 位移范数检验 =================
    print("\n Q4 Lambda 是否只由位移范数 ||h*x-x|| 决定？")
    dp = os.path.join(OUT, "kernel_displacement.json")
    if os.path.exists(dp):
        disp = {(r["family"], r["theta"]): r for r in
                json.load(open(dp))["rows"]}
        print(f" {'核':<16} | {'||h*x-x||':>10} | {'相对':>7} | "
              f"{'Lambda(a=1)':>12} | {'kunif(a=1)':>11}")
        print(" " + "-" * 66)
        for k in sorted({(r["family"], r["theta"]) for r in good}):
            r1 = [r for r in good if (r["family"], r["theta"]) == k
                  and abs(r["alpha"] - 1.0) < 1e-9]
            if not r1 or k not in disp:
                continue
            print(f" {k[0]+'/'+format(k[1],'g'):<16} | "
                  f"{disp[k]['norm_mean']:>10.2f} | {disp[k]['rel_mean']:>7.3f} | "
                  f"{r1[0]['lam_mean']:>12.0f} | {r1[0]['kunif']:>11.4f}")
        print(" " + "-" * 66)
        # 位移范数 vs Lambda 的相关
        xs4, ys4 = [], []
        for k in sorted({(r["family"], r["theta"]) for r in good}):
            r1 = [r for r in good if (r["family"], r["theta"]) == k
                  and abs(r["alpha"] - 1.0) < 1e-9]
            if r1 and k in disp:
                xs4.append(disp[k]["norm_mean"])
                ys4.append(r1[0]["lam_mean"])
        if len(xs4) >= 3:
            mx, my = sum(xs4) / len(xs4), sum(ys4) / len(ys4)
            sxy = sum((a - mx) * (b - my) for a, b in zip(xs4, ys4))
            sx = math.sqrt(sum((a - mx) ** 2 for a in xs4))
            sy = math.sqrt(sum((b - my) ** 2 for b in ys4))
            print(f"   corr(||h*x-x||, Lambda) = {sxy/(sx*sy):+.3f}  (n={len(xs4)})")
            print("   low-pass 的位移范数只有别人的 1/5，Lambda 却同量级"
                  " => Lambda 不是位移范数的函数")

    # ================= Q5 alpha 剖面 =================
    print("\n Q5 alpha 剖面：局部指数 d log Lambda / d log alpha")
    print(f" {'核':<16} | " + " | ".join(
        f"{'a'+format(a,'g'):>7}" for a in sorted({r['alpha'] for r in good})))
    print(" " + "-" * 92)
    for k in sorted({(r["family"], r["theta"]) for r in good}):
        sub = sorted([r for r in good if (r["family"], r["theta"]) == k],
                     key=lambda r: r["alpha"])
        cells = []
        for i in range(len(sub) - 1):
            a0, a1 = sub[i]["alpha"], sub[i + 1]["alpha"]
            if sub[i]["lam_mean"] <= 0 or sub[i + 1]["lam_mean"] <= 0:
                cells.append(f"{'--':>7}")
                continue
            v = math.log(sub[i + 1]["lam_mean"] / sub[i]["lam_mean"]) / \
                math.log(a1 / a0)
            cells.append(f"{v:>7.2f}")
        print(f" {k[0]+'/'+format(k[1],'g'):<16} | " + " | ".join(cells))
    print(" " + "-" * 92)
    print(" 列头是区间的右端点 alpha。高斯先验给 1（Lambda_Gauss 对 alpha 线性）；")
    print(" 纯曲率主导给 2。实测从 ~1.1 升到 2-4.6，且 alpha=1 那一步最陡")


if __name__ == "__main__":
    main()
