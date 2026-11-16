"""Butterworth 受控实验的分析：指数 b_n 是否随谱过渡带宽度变化？

输入 sweep_butterworth.json（每阶 n 扫 5 个 rho_c，2 个噪声水平）。
对每个 (n, t)：
  * 用 5 个 rho_c 的点拟合 log Lambda = a + b log(kunif)  -> b_n, R^2
  * bootstrap（对 20 张图重采样）给 b_n 的标准误
  * 报告该 n 下 Lambda 的逐图 std（检查方差是否随锐度增长——若是，
    跨 n 比较指数会受方差混淆）
最后：corr(过渡带宽度, b_n)，以及按 t 分层。

判据：
  * b_n 随 width 单调变化 => 机理成立：指数由滚降锐度决定
  * b_n 不随 width 变化 => 观察到的 -0.75 是偶然

用法： python analyze_butterworth.py
"""

import json
import os

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")


def load():
    d = json.load(open(os.path.join(OUT, "sweep_butterworth.json")))
    return d["records"], d["cfg"]


def fit_slope(ks, ys):
    A = np.vstack([np.ones(len(ks)), np.log(ks)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(ys), rcond=None)
    pred = A @ coef
    r2 = 1 - ((np.log(ys) - pred) ** 2).sum() / ((np.log(ys) - np.log(ys).mean()) ** 2).sum()
    return coef[1], r2, float(np.std(np.log(ys) - pred))


def main():
    recs, cfg = load()
    ts = sorted({r["t"] for r in recs})
    orders = sorted({r["order"] for r in recs})
    print("=" * 104)
    print(" Butterworth 受控实验：指数 b_n vs 谱过渡带宽度")
    print("=" * 104)
    print(f" 每阶 {len(cfg['rcs'])} 个 rho_c，{cfg['nimg']} 张图，"
          f"bootstrap 200 次（对图重采样）")

    for t in ts:
        print(f"\n{'':=<104}")
        print(f" t = {t:g}")
        print(f"{'':=<104}")
        print(f"{'阶数 n':>6} | {'过渡带宽度':>10} | {'指数 b_n':>9} | {'boot SE':>8} | "
              f"{'R^2':>7} | {'残差 std':>9} | {'Lambda 逐图 std':>15} | "
              f"{'kunif 范围':>18}")
        print("-" * 104)
        rows = []
        for n in orders:
            rr = sorted([r for r in recs if r["order"] == n and r["t"] == t],
                        key=lambda r: r["kunif"])
            if len(rr) < 3:
                continue
            ks = np.array([r["kunif"] for r in rr])
            ys = np.array([r["lam_mean"] for r in rr])
            b, r2, rs = fit_slope(ks, ys)
            # bootstrap：对 20 张图重采样，重算每点均值再拟合
            per = np.array([r["lam_per_img"] for r in rr])     # (npts, nimg)
            nimg = per.shape[1]
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(200):
                idxr = rng.integers(0, nimg, nimg)
                m = per[:, idxr].mean(axis=1)
                ok = m > 0
                if ok.sum() >= 3:
                    bs.append(fit_slope(ks[ok], m[ok])[0])
            se = float(np.std(bs)) if len(bs) > 10 else float("nan")
            width = float(np.mean([r["width"] for r in rr]))
            std_lam = float(np.mean([r["lam_std"] for r in rr]))
            rows.append((n, width, b, se))
            print(f"{n:>6} | {width:>10.3f} | {b:>9.2f} | {se:>8.2f} | {r2:>7.3f} | "
                  f"{rs:>9.3f} | {std_lam:>15.0f} | "
                  f"{ks.min():.3f} - {ks.max():.3f}")
        print("-" * 104)
        if len(rows) >= 3:
            w = np.array([r[1] for r in rows])
            b = np.array([r[2] for r in rows])
            c = float(np.corrcoef(w, b)[0, 1])
            print(f"  corr(过渡带宽度, 指数 b_n) = {c:+.3f}   (n={len(rows)} 个阶数)")
            # 斜率 b_n 对 1/n 的回归（width ~ 2.2*rho_c/n，固定 rho_c 集合下 1/n 是宽度的代理）
            inv = 1.0 / np.array([r[0] for r in rows], dtype=float)
            A = np.vstack([np.ones(len(inv)), inv]).T
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
            print(f"  b_n 对 1/n 的线性拟合: 截距 {coef[0]:.2f}, 斜率 {coef[1]:.2f}"
                  f"   (R^2 = "
                  f"{1 - ((b - A @ coef) ** 2).sum() / ((b - b.mean()) ** 2).sum():.3f})")
            print(f"  b_n 范围: {b.min():.2f} (n={rows[int(np.argmin(b))][0]}) 到 "
                  f"{b.max():.2f} (n={rows[int(np.argmax(b))][0]})")
        # 方差混淆检查
        print("\n  [方差混淆检查] Lambda 逐图 std 随阶数的变化：")
        for n in orders:
            rr = [r for r in recs if r["order"] == n and r["t"] == t]
            if rr:
                print(f"    n={n:<3d} 平均 std = "
                      f"{np.mean([r['lam_std'] for r in rr]):>8.0f}   "
                      f"平均 Lambda = {np.mean([r['lam_mean'] for r in rr]):>9.0f}   "
                      f"变异系数 = "
                      f"{np.mean([r['lam_std'] for r in rr]) / max(np.mean([r['lam_mean'] for r in rr]), 1):.2f}")

    print("\n[判据]")
    print("  * b_n 随过渡带宽度单调变化 => 机理成立")
    print("  * 若同时看到逐图 std 随 n 快速增长，则跨 n 的指数比较受方差混淆，"
          "需更多图像")


if __name__ == "__main__":
    main()
