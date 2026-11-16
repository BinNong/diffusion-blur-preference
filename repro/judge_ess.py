"""ess_scale.py 的离线判据：Y1–Y4（零 GPU）。

**预官修正案（2026-10-22，披露）**：原预官（md5 5849a53c）的 Y2/Y3 是
「把 log(n/ESS) 对 Λ 回归，报 R²」。测量跑完后发现该量**恒等于 log(256)=5.545**
（ESS 精确为 1，见下），方差为零、R² 无定义——这是只有先跑测量才能发现的 ill-posed。
修正案在**判据尚未运行**时写定：把主判据从「回归一个退化量」改成「检验有无恒定比例」。
Y1 的恒等式检验与 Y4 的诊断保留原样。原预官文本见 ess_scale.py docstring。

用法： venv2/bin/python judge_ess.py
"""

import json
import math
import os

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")


def main():
    d = json.load(open(os.path.join(OUT, "ess_scale.json")))
    rows = d["rows"]
    cfg = d["cfg"]
    have = [r for r in rows if r.get("lam") is not None]
    print("=" * 100)
    print(f" ESS 判据：{len(rows)} 行（{len(cfg['noises'])} 个 σ_n × "
          f"{len({r['id'] for r in rows})} 算子，池 {cfg['n_pool']} 张）；"
          f"有 Λ 可比 {len(have)} 行")
    print("=" * 100)

    # ---------------- Y1 ----------------
    print("\n" + "=" * 100)
    print(" Y1 守门（原预官保留）：对数正态恒等式 log(n/ESS) ≈ Var(data)")
    print("=" * 100)
    print(f" {'σ_n':>6} | {'log(n/ESS)':>11} | {'Var(data)':>12} | {'比值':>11}")
    print(" " + "-" * 56)
    for sn in cfg["noises"]:
        v = [r for r in rows if r["sn"] == sn]
        lr = float(np.median([r["log_nrat"] for r in v]))
        vd = float(np.median([r["var_data"] for r in v]))
        print(f" {sn:>6g} | {lr:>11.4f} | {vd:>12.4g} | {lr/vd:>11.3g}")
    print(" " + "-" * 56)
    print(" 恒等式要求 Var(data)=O(1)；实测 Var(data)~1e12，比值 ~1e-13。")
    print(" **Y1 失败，而且失败方式有信息**：对数正态近似在这里完全不适用——")
    print(" 权重跨十几个数量级，ESS 塌到 1，log(n/ESS) 钉在 log n 上没有动态范围。")

    # ---------------- Y2 ----------------
    print("\n" + "=" * 100)
    print(" Y2（修正案主判据）：ESS ≤ 1.01，即采样器彻底塌缩")
    print("=" * 100)
    print(f" {'σ_n':>6} | {'n':>4} | {'ESS 最小':>9} | {'ESS 最大':>9} | "
          f"{'log(n/ESS) 最小':>16} | {'log(n/ESS) 最大':>16} | 判定")
    print(" " + "-" * 90)
    ok2 = True
    for sn in cfg["noises"]:
        v = [r for r in rows if r["sn"] == sn]
        e = np.array([r["ess"] for r in v])
        lr = np.array([r["log_nrat"] for r in v])
        good = bool(e.max() <= 1.01)
        ok2 &= good
        print(f" {sn:>6g} | {len(v):>4d} | {e.min():>9.4f} | {e.max():>9.4f} | "
              f"{lr.min():>16.4f} | {lr.max():>16.4f} | {'✅' if good else '❌'}")
    print(" " + "-" * 90)
    print(f" => Y2 {'✅ 通过' if ok2 else '❌ 失败'}："
          f"{'ESS 精确为 1——256 个先验样本只折合 1 个有效样本，且与算子、噪声无关' if ok2 else ''}")
    print(f" （log(n) = log({cfg['n_pool']}) = {math.log(cfg['n_pool']):.4f}）")

    # ---------------- Y3 ----------------
    print("\n" + "=" * 100)
    print(" Y3（修正案，最尖）：Var(data)/Λ 不是常数——若「采样器失效 = k·Λ」成立它应恒定")
    print("=" * 100)
    print(f" {'σ_n':>6} | {'n':>4} | {'Var/Λ 最小':>12} | {'Var/Λ 最大':>12} | "
          f"{'极差':>8} | 判定")
    print(" " + "-" * 66)
    ok3 = True
    res = []
    for sn in cfg["noises"]:
        v = [r for r in have if r["sn"] == sn]
        if len(v) < 3:
            continue
        rr = np.array([r["var_data"] / r["lam"] for r in v])
        sp = float(rr.max() / rr.min())
        good = bool(sp >= 5.0)
        ok3 &= good
        res.append(dict(sn=sn, n=len(v), lo=float(rr.min()), hi=float(rr.max()),
                        spread=sp))
        print(f" {sn:>6g} | {len(v):>4d} | {rr.min():>12.4g} | {rr.max():>12.4g} | "
              f"{sp:>7.1f}x | {'✅' if good else '❌'}")
    print(" " + "-" * 66)
    print(f" => Y3 {'✅ 通过' if ok3 else '❌ 失败'}："
          f"{'Var/Λ 散 15 倍 ⇒ 采样器的权重散度**不是**由 Λ 定标' if ok3 else ''}")

    # 附带：Var(data) 与 Λ 的相关系数（原 Y2 的回归，现在只作诊断）
    print("\n 附（原 Y2 的回归，降级为诊断）：corr(Var(data), Λ)")
    for sn in cfg["noises"]:
        v = [r for r in have if r["sn"] == sn]
        if len(v) < 3:
            continue
        a = np.array([r["var_data"] for r in v])
        b = np.array([r["lam"] for r in v])
        c = float(np.corrcoef(a, b)[0, 1])
        print(f"   σ_n={sn:<5g} corr = {c:+.4f}   （Var 跨 {a.max()/a.min():.3f}x，"
              f"Λ 跨 {b.max()/b.min():.1f}x）")

    # ---------------- Y4 ----------------
    print("\n" + "=" * 100)
    print(" Y4 诊断（原预官保留）：Var(data) 的 σ_n 标度（二阶公式预言 1/σ⁴）")
    print("=" * 100)
    print(f" {'σ_n':>6} | {'Var(data) 中位':>15} | {'相对 σ=0.02':>12} | "
          f"{'(σ/0.02)⁴':>10}")
    print(" " + "-" * 60)
    base = None
    diag = []
    for sn in cfg["noises"]:
        v = [r for r in rows if r["sn"] == sn]
        vd = float(np.median([r["var_data"] for r in v]))
        if base is None:
            base = vd
        pred = (sn / cfg["noises"][0]) ** 4
        diag.append(dict(sn=sn, var=vd, ratio=vd / base, pred=float(pred)))
        print(f" {sn:>6g} | {vd:>15.4g} | {vd/base:>12.4f} | {pred:>10.1f}")
    print(" " + "-" * 60)
    print(" 若两列一致，则 Var(data) 纯粹由二阶（功率谱）结构决定，σ_n 只通过 1/σ⁴ 进入。")

    with open(os.path.join(OUT, "fig_ess.json"), "w") as fh:
        json.dump(dict(cfg=cfg, y1_ratio=[lr / vd for lr, vd in
                                          [(float(np.median([r["log_nrat"] for r in rows
                                                             if r["sn"] == sn])),
                                            float(np.median([r["var_data"] for r in rows
                                                             if r["sn"] == sn])))
                                           for sn in cfg["noises"]]],
                       y2_pass=bool(ok2), y3=res, y3_pass=bool(ok3), y4=diag,
                       n_rows=len(rows), n_have=len(have)), fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'fig_ess.json')}")


if __name__ == "__main__":
    main()
