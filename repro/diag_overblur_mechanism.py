# -*- coding: utf-8 -*-
"""diag_overblur_mechanism.py —— 过模糊失效模式的两分量诊断（零 GPU，纯标准库）

*** 事后诊断（post-hoc），不是预注册实验 ***
设计本诊断时 `results/phaseC_pool20.json` 与其 N0–N4 判定已经可见，因此这里
**不设任何通过线、不产生 PASS/FAIL、不作为论文结论**。它的用途只有一个：
为下一轮「跑前写定判据」的正式实验提供候选机制与效应量。按项目惯例，
纯诊断判据可以只报数（CLAUDE.md「Keep one or two pure diagnostic criteria」）。

动机：RESULTS §3.9(br)/(bx) 把「为什么噪声越大越倾向过度模糊」列为仍未解释的
开放问题。`phaseC_pool20.json` 的每条 profile 同时存了 L / data / q 三个字段，
且 L = data + q，所以这个分解不需要重跑 GPU。

读数口径（与 judge_phaseC_pool20.py 完全一致）：
  theta* = 1.5；高噪声 = sigma_n >= 0.172；区间端点取 theta = 2.5；
  ΔX = X(2.5) − X(1.5)，对 20 张图取算术平均。
  argmin 取 21 点网格上 L 的最小者。

用法：
  python3 repro/diag_overblur_mechanism.py
输出：results/fig_overblur_mech.json + 终端表格
"""

import json
import math
import os
import statistics as st

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
SRC = "phaseC_pool20.json"
DST = "fig_overblur_mech.json"
TSTAR = 1.5
TEND = 2.5
HI = 0.172


def profile(rec, key):
    return {p["theta"]: p[key] for p in rec["profile"]}


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def linfit_loglog(xs, ys):
    """拟合 log(y) = c - p*log(x)，返回 (p, R^2)。"""
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = mean(lx), mean(ly)
    sxx = sum((x - mx) ** 2 for x in lx)
    slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx
    p = -slope
    sst = sum((y - my) ** 2 for y in ly)
    sse = sum((y - (my + slope * (x - mx))) ** 2 for x, y in zip(lx, ly))
    return p, (1.0 - sse / sst if sst else float("nan"))


def main():
    d = json.load(open(os.path.join(OUT, SRC)))
    recs = d["records"]
    ths = d["cfg"]["thetas"]
    sns = sorted(set(r["noise"] for r in recs))

    # 恒等式自检：L = data + q 必须逐点成立
    ident = max(abs(p["L"] - p["data"] - p["q"])
                for r in recs for p in r["profile"])
    print("  [identity] max |L - (data + q)| = %.3e" % ident)
    if ident > 1e-6:
        raise SystemExit("  ABORT: L = data + q does not hold; nothing below is interpretable")

    per_sigma, dd_at_sigma = [], []
    for sn in sns:
        rs = [r for r in recs if r["noise"] == sn]
        dd = st.mean(profile(r, "data")[TEND] - profile(r, "data")[TSTAR] for r in rs)
        dq = st.mean(profile(r, "q")[TEND] - profile(r, "q")[TSTAR] for r in rs)
        dl = st.mean(profile(r, "L")[TEND] - profile(r, "L")[TSTAR] for r in rs)
        am = st.mean(min(ths, key=lambda t: profile(r, "L")[t]) for r in rs)
        dat = st.mean(profile(r, "data")[TSTAR] for r in rs)
        ratio = st.mean(abs(profile(r, "q")[TEND] - profile(r, "q")[TSTAR]) /
                        abs(profile(r, "data")[TEND] - profile(r, "data")[TSTAR])
                        for r in rs
                        if abs(profile(r, "data")[TEND] - profile(r, "data")[TSTAR]) > 1e-9)
        per_sigma.append(dict(sigma_n=sn, n=len(rs), mean_argmin=am,
                              mean_data_at_tstar=dat, d_data=dd, d_q=dq, d_L=dl,
                              mean_abs_dq_over_ddata=ratio,
                              prior_sign="rewards over-blur" if dq < 0 else "penalises over-blur"))
        dd_at_sigma.append(dd)

    p, r2 = linfit_loglog(sns, dd_at_sigma)

    # 诊断 A：sign(ΔL[1.5->2.5]) 能否预测该配置是否 argmin > theta*
    conf = dict(tp=0, fp=0, tn=0, fn=0)
    # 诊断 B：反事实——只保留 data 项（拿掉先验），还会过模糊吗
    cf_over = 0
    for r in recs:
        L, D = profile(r, "L"), profile(r, "data")
        actual = min(ths, key=lambda t: L[t]) > TSTAR
        pred = L[TEND] - L[TSTAR] < 0
        conf[("tp" if actual else "fp") if pred else ("fn" if actual else "tn")] += 1
        cf_over += (min(ths, key=lambda t: D[t]) > TSTAR)
    n = len(recs)
    sign_hits = conf["tp"] + conf["tn"]

    print("\n  sigma_n | mean argmin | data(theta*) | dData | dQ | dL | |dQ/dData| | prior")
    for row in per_sigma:
        print("  %7.4f | %11.3f | %12.1f | %+8.1f | %+8.1f | %+8.1f | %8.2f | %s"
              % (row["sigma_n"], row["mean_argmin"], row["mean_data_at_tstar"],
                 row["d_data"], row["d_q"], row["d_L"], row["mean_abs_dq_over_ddata"],
                 row["prior_sign"]))
    print("\n  [A] dData ~ sigma^-p : p=%.3f  R2=%.4f  (pure 1/sigma^2 => p=2)" % (p, r2))
    print("  [B] sign(dL) predicts argmin>theta* : %d/%d = %.1f%%  conf=%s"
          % (sign_hits, n, 100.0 * sign_hits / n, conf))
    print("  [C] counterfactual, prior removed   : %d/%d = %.1f%% still over-blur"
          % (cf_over, n, 100.0 * cf_over / n))
    flip = [row["sigma_n"] for row in per_sigma if row["d_q"] < 0]
    print("  [D] prior-term sign flips between the last positive and first negative sigma;"
          " first negative = %s" % (min(flip) if flip else None))

    res = dict(
        kind="post-hoc diagnostic, no pre-registration, no thresholds, not a paper claim",
        source=SRC, identity_max_abs=ident, tstar=TSTAR, tend=TEND, hi_noise=HI,
        per_sigma=per_sigma, decay_exponent_p=p, decay_r2=r2,
        sign_prediction=dict(hits=sign_hits, total=n, confusion=conf),
        counterfactual_data_only=dict(over_blur=cf_over, total=n),
        prior_sign_first_negative_sigma=min(flip) if flip else None,
    )
    with open(os.path.join(OUT, DST), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print("\n  [saved] %s" % os.path.join(OUT, DST))


if __name__ == "__main__":
    main()
