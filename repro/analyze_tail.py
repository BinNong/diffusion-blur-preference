"""解耦实验分析：指数是否依赖阻带底板 delta（固定 sigma => 固定过渡带宽度）？

对每个 delta，用 4 个 sigma 的点拟合 log Lambda = a + b log(kunif) -> b_delta。
若 b_delta 随 delta 不变 => 阻带尾不是决定因素（defocus 的离群另有原因）
若 b_delta 随 delta 变化 => 阻带尾是第二个决定因素

同时报告每个 delta 下 sigma 扫描覆盖的 kunif 范围（确认宽度确实被固定住了）。

用法： python analyze_tail.py
"""

import json
import os

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")


def fit(ks, ys):
    A = np.vstack([np.ones(len(ks)), np.log(ks)]).T
    c, *_ = np.linalg.lstsq(A, np.log(ys), rcond=None)
    pred = A @ c
    r2 = 1 - ((np.log(ys) - pred) ** 2).sum() / \
        ((np.log(ys) - np.log(ys).mean()) ** 2).sum()
    return c[1], r2, float(np.std(np.log(ys) - pred))


def main():
    d = json.load(open(os.path.join(OUT, "sweep_tail.json")))
    recs, cfg = d["records"], d["cfg"]
    ts = sorted({r["t"] for r in recs})
    dels = sorted({r["delta"] for r in recs})
    sigs = sorted({r["sigma"] for r in recs})

    for t in ts:
        print("=" * 100)
        print(f" t = {t:g}：指数 b_delta 是否依赖阻带底板 delta")
        print("=" * 100)
        print(f"{'delta':>10} | {'指数 b':>8} | {'R^2':>7} | {'残差 std':>9} | "
              f"{'kunif 范围':>20} | {'宽度范围':>18} | {'Lambda 范围':>22}")
        print("-" * 100)
        rows = []
        for dl in dels:
            rr = sorted([r for r in recs if r["delta"] == dl and r["t"] == t],
                        key=lambda r: r["kunif"])
            if len(rr) < 3:
                continue
            k = np.array([r["kunif"] for r in rr])
            l = np.array([r["lam_mean"] for r in rr])
            b, r2, rs = fit(k, l)
            w = [r["width"] for r in rr]
            rows.append((dl, b))
            print(f"{dl:>10g} | {b:>8.2f} | {r2:>7.3f} | {rs:>9.3f} | "
                  f"{k.min():.4f} - {k.max():.4f} | "
                  f"{min(w):.2f} - {max(w):.2f} | "
                  f"{l.min():.0f} - {l.max():.0f}")
        print("-" * 100)
        if len(rows) >= 2:
            bs = np.array([r[1] for r in rows])
            print(f"  b_delta 范围: {bs.min():.2f} - {bs.max():.2f}"
                  f"（极差 {bs.max()-bs.min():.2f}，相对 {100*(bs.max()-bs.min())/bs.mean():.1f}%）")
            print(f"  delta 跨了 {dels[0]:g} -> {dels[-1]:g}（{dels[-1]/max(dels[1],1e-30):.0e} 倍阻带尾）")
        # 每个 sigma 下 delta 的直接影响
        print(f"\n  固定 sigma、只改 delta 时 Lambda 的变化（t={t:g}）：")
        for sg in sigs:
            rr = sorted([r for r in recs if r["sigma"] == sg and r["t"] == t],
                        key=lambda r: r["delta"])
            if len(rr) < 2:
                continue
            l0 = rr[0]["lam_mean"]
            ch = [100 * (r["lam_mean"] - l0) / abs(l0) for r in rr]
            print(f"    sigma={sg:<5} Lambda={l0:>9.0f}  相对变化: "
                  + " ".join(f"{c:+6.2f}%" for c in ch))
        print()

    print("[判据]")
    print("  * b_delta 极差若远小于其 bootstrap 不确定度 => 阻带尾不是决定因素")
    print("  * 注意：defocus 与 gaussian 的阻带差异只有 ~10-90 倍（delta ~ 1e-8），")
    print("    而本实验 delta 取到 1e-3，已覆盖并远超该量级")


if __name__ == "__main__":
    main()
