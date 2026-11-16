"""分段斜率差的显著性：强模糊端"按核强度分两组"是真的还是拟合伪影？

背景（RESULTS §3.9 / 阶段一小结 §4.3）：
  * 低段（alpha<=0.4）逐路径斜率 -1.54 - -1.86，前缀 1.62x
  * 高段（alpha>=0.6）逐路径斜率 -0.75 - -1.29，前缀 2.89x，
    "按核强度分两组"
"分两组"这个说法来自斜率的目视分组。但每段每路径只有 3 个点，
斜率的标准差可能本身就很大；而前缀是外插到 u=1（数据在 u~0.08-0.19），
外插 1 dex，斜率的微小误差会被放大成前缀的巨大差异。

本脚本按项目判据（analyze_conditioning / regression-conditioning-checklist）
把这件事做定量：
  1. 逐路径分段拟合，报斜率、截距、以及两者的标准误
  2. 逐对路径的斜率差 + 合并标准误 -> t 统计量
  3. 前缀差 = 外插量，报它的标准误（说明 2.89x 是否在噪声内）
  4. 改用"数据区间内插"而不是"外插到 u=1"重做跨路径比较
  5. 低段 vs 高段的斜率差是否显著（同一路径内的配对比较）

用法： venv2/bin/python analyze_segment_significance.py
"""

import json
import math
import os
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

LO = {0.15, 0.25, 0.4}
HI = {0.6, 0.8, 1.0}


def ols_se(x, y):
    """返回 b, a, SE(b), SE(a), R2, hat。n=3 时自由度为 1。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    mx, my = x.mean(), y.mean()
    sxx = float(((x - mx) ** 2).sum())
    sxy = float(((x - mx) * (y - my)).sum())
    b = sxy / sxx
    a = my - b * mx
    res = y - (a + b * x)
    dof = max(n - 2, 1)
    s2 = float((res ** 2).sum()) / dof
    seb = math.sqrt(s2 / sxx)
    sea = math.sqrt(s2 * (1.0 / n + mx * mx / sxx))
    sst = float(((y - my) ** 2).sum())
    r2 = 1 - float((res ** 2).sum()) / sst if sst > 0 else float("nan")
    # 在 x=0（u=1）处的预测标准误 = 外插量的不确定度
    sep0 = math.sqrt(s2 * (1.0 / n + (0.0 - mx) ** 2 / sxx))
    # 杠杆：每个点的 hat
    hat = 1.0 / n + (x - mx) ** 2 / sxx
    return dict(b=b, a=a, seb=seb, sea=sea, r2=r2, sep0=sep0, hat=hat,
                n=n, span=float(x.max() - x.min()), s=math.sqrt(s2))


def main():
    d = json.load(open(os.path.join(OUT, "sweep_alpha.json")))
    recs = d["records"]
    t0 = min(r["t"] for r in recs)
    nimg = d["cfg"]["nimg"]

    def snr(r):
        return r["lam_mean"] / r["lam_std"] * math.sqrt(nimg) \
            if r["lam_std"] > 0 else float("inf")

    good = [r for r in recs if r["t"] == t0 and r["lam_mean"] > 0
            and snr(r) >= 3.0]
    paths = defaultdict(list)
    for r in good:
        paths[(r["family"], r["theta"])].append(r)
    for k in paths:
        paths[k].sort(key=lambda r: r["alpha"])
    valid = {k: v for k, v in paths.items()
             if all(v[i]["kunif"] < v[i + 1]["kunif"]
                    for i in range(len(v) - 1))}

    print("=" * 108)
    print(f" t={t0:g}，{len(good)} 个可用点，{len(valid)} 条有效路径")
    print("=" * 108)

    # 每条的"核强度"：alpha=1 处的 kunif
    strength = {}
    for k, v in valid.items():
        a1 = [r for r in v if r["alpha"] == 1.0]
        strength[k] = a1[0]["kunif"] if a1 else float("nan")

    for seg, S in [("低段 alpha<=0.4", LO), ("高段 alpha>=0.6", HI)]:
        print(f"\n{'='*108}\n {seg}：逐路径拟合 log Lambda = a + b log u\n{'='*108}")
        print(f" {'路径':<16} | {'kunif(1)':>8} | {'b':>7} | {'SE(b)':>7} | "
              f"{'前缀 exp(a)':>11} | {'SE(前缀)':>9} | {'u 跨度dex':>9} | "
              f"{'R2':>6} | {'maxhat':>7}")
        print(" " + "-" * 104)
        fits = {}
        for k in sorted(valid, key=lambda k: strength[k]):
            pts = [r for r in valid[k] if r["alpha"] in S]
            if len(pts) < 3:
                continue
            x = [math.log(1 - r["kunif"]) for r in pts]
            y = [math.log(r["lam_mean"]) for r in pts]
            f = ols_se(x, y)
            fits[k] = (f, x)
            print(f" {k[0]+'/'+format(k[1],'g'):<16} | {strength[k]:>8.4f} | "
                  f"{f['b']:>7.2f} | {f['seb']:>7.2f} | "
                  f"{math.exp(f['a']):>11.0f} | "
                  f"{math.exp(f['a'])*f['sep0']:>9.0f} | "
                  f"{f['span']:>9.2f} | {f['r2']:>6.3f} | {f['hat'].max():>7.3f}")
        if not fits:
            continue
        bs = [f[0]["b"] for f in fits.values()]
        print(" " + "-" * 104)
        print(f" 斜率 {min(bs):.2f} - {max(bs):.2f}；"
              f"SE(b) 中位数 {np.median([f[0]['seb'] for f in fits.values()]):.2f}"
              f"  => 斜率极差 / 典型 SE = "
              f"{(max(bs)-min(bs))/np.median([f[0]['seb'] for f in fits.values()]):.1f}")
        # 前缀（外插到 u=1）的不确定度
        pr = [(k, math.exp(f["a"]), math.exp(f["a"]) * f["sep0"])
              for k, (f, x) in fits.items()]
        lo = min(p[1] for p in pr); hi = max(p[1] for p in pr)
        se_typ = np.median([p[2] for p in pr])
        print(f" 前缀极差 {hi/lo:.2f}x；前缀 SE 中位数 {se_typ:.0f} "
              f"（即 {100*se_typ/np.median([p[1] for p in pr]):.0f}%）")
        print(f" => 若前缀极差与 SE 同量级，'前缀散开'就是外插伪影")

        # 逐对斜率差
        ks = sorted(fits)
        print(f"\n 逐对斜率差（只报 |t|>2 的，即统计上可分辨的）:")
        print(f" {'路径A':<16} | {'路径B':<16} | {'db':>7} | {'SE(db)':>7} | "
              f"{'t':>6}")
        print(" " + "-" * 74)
        nt = 0
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                fa = fits[ks[i]][0]; fb = fits[ks[j]][0]
                db = fa["b"] - fb["b"]
                sdb = math.sqrt(fa["seb"] ** 2 + fb["seb"] ** 2)
                t = db / sdb if sdb > 0 else float("nan")
                if abs(t) > 2:
                    nt += 1
                    print(f" {ks[i][0]+'/'+format(ks[i][1],'g'):<16} | "
                          f"{ks[j][0]+'/'+format(ks[j][1],'g'):<16} | "
                          f"{db:>7.2f} | {sdb:>7.2f} | {t:>6.2f}")
        tot = len(ks) * (len(ks) - 1) // 2
        print(" " + "-" * 74)
        print(f" {tot} 对中 {nt} 对 |t|>2。")
        print(f" 注：每段每路径只有 3 点、自由度 1，SE 本身极不稳定；")
        print(f"     |t|>2 的数目若接近 {int(0.05*tot)}（5% 假阳性），"
              f"说明斜率差异整体不显著。")

        # 按核强度分两组？
        if seg.startswith("高段"):
            med = np.median([strength[k] for k in fits])
            g1 = [k for k in fits if strength[k] >= med]
            g2 = [k for k in fits if strength[k] < med]
            if g1 and g2:
                b1 = [fits[k][0]["b"] for k in g1]
                b2 = [fits[k][0]["b"] for k in g2]
                print(f"\n 按 kunif(alpha=1) 中位数 {med:.4f} 分两组:")
                print(f"   强组 {[k[0]+'/'+format(k[1],'g') for k in g1]}")
                print(f"        b = {[round(v,2) for v in b1]}  "
                      f"均值 {np.mean(b1):.2f}")
                print(f"   弱组 {[k[0]+'/'+format(k[1],'g') for k in g2]}")
                print(f"        b = {[round(v,2) for v in b2]}  "
                      f"均值 {np.mean(b2):.2f}")
                s1 = np.mean([fits[k][0]["seb"] for k in g1])
                s2 = np.mean([fits[k][0]["seb"] for k in g2])
                dd = np.mean(b1) - np.mean(b2)
                sdd = math.sqrt(s1 ** 2 / len(g1) + s2 ** 2 / len(g2))
                print(f"   组间差 {dd:+.2f}，SE {sdd:.2f}，t = {dd/sdd:+.2f}"
                      f"  => {'可分' if abs(dd/sdd) > 2 else '不可分（在噪声内）'}")

    # ---- 5. 同一路径内 低段 vs 高段 的斜率差（配对）----
    print(f"\n{'='*108}\n 同一路径内：高段斜率 - 低段斜率（配对比较）\n{'='*108}")
    print(f" {'路径':<16} | {'b_低':>7} | {'b_高':>7} | {'差':>7} | "
          f"{'SE(差)':>7} | {'t':>6}")
    print(" " + "-" * 66)
    ds = []
    for k in sorted(valid):
        pl = [r for r in valid[k] if r["alpha"] in LO]
        ph = [r for r in valid[k] if r["alpha"] in HI]
        if len(pl) < 3 or len(ph) < 3:
            continue
        fl = ols_se([math.log(1 - r["kunif"]) for r in pl],
                    [math.log(r["lam_mean"]) for r in pl])
        fh = ols_se([math.log(1 - r["kunif"]) for r in ph],
                    [math.log(r["lam_mean"]) for r in ph])
        dd = fh["b"] - fl["b"]
        sdd = math.sqrt(fl["seb"] ** 2 + fh["seb"] ** 2)
        t = dd / sdd if sdd > 0 else float("nan")
        ds.append((dd, sdd))
        print(f" {k[0]+'/'+format(k[1],'g'):<16} | {fl['b']:>7.2f} | "
              f"{fh['b']:>7.2f} | {dd:>7.2f} | {sdd:>7.2f} | {t:>6.2f}")
    if ds:
        md = np.mean([d for d, s in ds])
        ms = math.sqrt(np.mean([s ** 2 for d, s in ds]) / len(ds))
        print(" " + "-" * 66)
        print(f" 平均斜率变化 {md:+.2f}，典型 SE {ms:.2f}，"
              f"t = {md/ms:+.2f}"
              f"  => 斜率随模糊增强而变平{'显著' if abs(md/ms) > 2 else '不显著'}")


if __name__ == "__main__":
    main()
