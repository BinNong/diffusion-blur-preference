"""跨先验判据：三项恒等式的结构是否先验无关（离线，零 GPU）。

预官来源：evidence_scale2.py docstring「跨先验检验」（2026-10-20 20:59，跑前写定；
          md5 809c68dcd8e4ee1cba7ec78add684da7）。判据在看到 afhqv2 输出之前写定。

用法： venv2/bin/python judge_xprior.py
"""

import json
import os

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
BIG = 2000.0
FILES = [("ffhq", "evidence_scale2.json"), ("afhqv2", "evidence_scale2_afq.json")]


def mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    blur = [r for r in d["rows"] if r["tag"] == "blur"]
    big = [r for r in blur if r["lam_clean"] is not None and r["lam_clean"] >= BIG
           and r.get("E_hqs") is not None]
    return d, blur, big


def main():
    D = {}
    for name, fn in FILES:
        r = load(fn)
        if r is None:
            print(f"[错误] 缺 {fn}")
            return
        D[name] = r
    print("=" * 100)
    print(" 跨先验判据：三项恒等式的结构是否先验无关")
    print("=" * 100)
    for name, _ in FILES:
        d, blur, big = D[name]
        print(f" {name:<7} ckpt={d['cfg']['ckpt']}  "
              f"{len(blur)} 模糊格 / {len(big)} 格 Λ_clean≥{BIG:.0f}")

    # ---------------- X1 ----------------
    print("\n" + "=" * 100)
    print(" X1 守门：afhqv2 上 E_y = -(Λ_noisy + data) 仍成立")
    print("=" * 100)
    for name, _ in FILES:
        d = D[name][0]
        w = max(abs(r["E_y"] + r["lam"] + r["data"]) / max(1.0, abs(r["E_y"]))
                for r in d["rows"])
        print(f" {name:<7} 最大相对误差 = {w:.3e}  {'✅' if w <= 1e-6 else '❌'}")
        if w > 1e-6:
            print(" ❌ X1 未过 —— 后面作废")
            return

    # ---------------- X2 ----------------
    print("\n" + "=" * 100)
    print(" X2：afhqv2 上 mean(|E_hqs|/Λ_clean)∈[1.00,1.25] 且 SE≤0.03")
    print("=" * 100)
    print(f" {'先验':<7} | {'n':>4} | {'mean':>8} | {'SE':>8} | {'min':>7} | "
          f"{'max':>7} | {'Λ_cl 跨度':>10}")
    print(" " + "-" * 76)
    stats = {}
    for name, _ in FILES:
        big = D[name][2]
        v = [abs(r["E_hqs"]) / r["lam_clean"] for r in big]
        m, se = mean_se(v)
        lc = [r["lam_clean"] for r in big]
        stats[name] = (m, se, len(v))
        print(f" {name:<7} | {len(v):>4d} | {m:>8.4f} | {se:>8.4f} | {min(v):>7.4f} | "
              f"{max(v):>7.4f} | {max(lc)/min(lc):>9.1f}×")
    print(" " + "-" * 76)
    m2, se2, _ = stats["afhqv2"]
    ok2 = (1.00 <= m2 <= 1.25) and (se2 <= 0.03)
    print(f" => X2 {'✅ 通过' if ok2 else '❌ 失败'}")

    # ---------------- X3 ----------------
    print("\n" + "=" * 100)
    print(" X3 最尖：两个先验的 mean 比值之差 ≤ 2·SE_combined")
    print("=" * 100)
    (m1, s1, n1), (m2b, s2b, n2) = stats["ffhq"], stats["afhqv2"]
    diff = abs(m1 - m2b)
    comb = float(np.sqrt(s1 ** 2 + s2b ** 2))
    t = diff / comb if comb > 0 else float("inf")
    print(f" ffhq  mean = {m1:.4f} ± {s1:.4f}  (n={n1})")
    print(f" afhqv2 mean = {m2b:.4f} ± {s2b:.4f}  (n={n2})")
    print(f" |Δmean| = {diff:.4f}   2·SE_comb = {2*comb:.4f}   t = {t:.2f}")
    ok3 = diff <= 2 * comb
    print(f" => X3 {'✅ 通过' if ok3 else '❌ 失败'}："
          f"{'比值跨先验不可区分，退化机制先验无关' if ok3 else '比值依赖先验——与 §3.9(u) 的「Λ 先验无关」并置，这是需要解释的新结果'}")

    # ---------------- X4 ----------------
    print("\n" + "=" * 100)
    print(" X4 结构：-Δq/Λ_clean 随 θ* 单调下降（ffhq 实测 +0.1105/+0.0030/-0.0318）")
    print("=" * 100)
    print(f" {'先验':<7} | " + " | ".join(f"{'θ*='+format(th,'g'):>12}"
                                          for th in (1.0, 1.5, 2.0)) + " | 单调?")
    print(" " + "-" * 76)
    ok4 = True
    for name, _ in FILES:
        big = D[name][2]
        vals = []
        for th in (1.0, 1.5, 2.0):
            v = [-r["dq_over_lam"] for r in big if r["theta"] == th]
            vals.append(float(np.mean(v)) if v else float("nan"))
        mono = all(vals[i] > vals[i + 1] for i in range(len(vals) - 1)
                   if not (np.isnan(vals[i]) or np.isnan(vals[i + 1])))
        ok4 &= bool(mono)
        print(f" {name:<7} | " + " | ".join(f"{v:>12.4f}" for v in vals)
              + f" | {'是' if mono else '否'}")
    print(" " + "-" * 76)
    print(f" => X4 {'✅ 通过' if ok4 else '❌ 失败'}")

    # ---------------- X5 诊断 ----------------
    print("\n" + "=" * 100)
    print(" X5 诊断（无阈值）")
    print("=" * 100)
    print(" (a) data_term/data 范围（ffhq 是 [0.71,1.03]）")
    for name, _ in FILES:
        v = [r["xdata_over_data"] for r in D[name][2]]
        print(f"   {name:<7} [{min(v):.3f}, {max(v):.3f}]  中位 {np.median(v):.3f}")
    print("\n (b) 逐 θ* 的比值与两项修正")
    for name, _ in FILES:
        big = D[name][2]
        print(f"   {name}:")
        print(f"     {'θ*':>5} | {'n':>4} | {'mean 比值':>10} | {'-Δq/Λ':>9} | "
              f"{'数据项修正':>10}")
        for th in (1.0, 1.5, 2.0):
            v = [r for r in big if r["theta"] == th]
            if not v:
                continue
            mr = float(np.mean([abs(r["E_hqs"]) / r["lam_clean"] for r in v]))
            a = float(np.mean([-r["dq_over_lam"] for r in v]))
            b = float(np.mean([-(r["xdata_over_data"] - 1) * r["data"]
                               / r["lam_clean"] for r in v]))
            print(f"     {th:>5g} | {len(v):>4d} | {mr:>10.4f} | {a:>9.4f} | "
                  f"{b:>10.4f}")
    print("\n (c) 恒等式自检 E_hqs = Δq - Λ_clean + (data_term - data)")
    for name, _ in FILES:
        w = 0.0
        for r in D[name][1]:
            rec = (r["dq_over_lam"] * r["lam_clean"] - r["lam_clean"]
                   + (r["xdata_over_data"] - 1) * r["data"])
            w = max(w, abs(rec - r["E_hqs"]) / max(abs(r["E_hqs"]), 1))
        print(f"   {name:<7} 最大相对误差 = {w:.3e}")

    with open(os.path.join(OUT, "fig_xprior.json"), "w") as fh:
        json.dump(dict(
            stats={k: dict(mean=v[0], se=v[1], n=v[2]) for k, v in stats.items()},
            x2_pass=bool(ok2), x3_pass=bool(ok3), x3_t=float(t),
            x3_diff=float(diff), x4_pass=bool(ok4)), fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'fig_xprior.json')}")


if __name__ == "__main__":
    main()
