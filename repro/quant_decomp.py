"""量化 D4 的机理：Λ = Λ_B − Λ_S 是两个大量之差，两者的 t 衰减几乎同速。

恒等式（不是模型）：
    Λ(t,u) = Λ_B(t,u) − Λ_S(t)
    Λ_B = logp_blur_mean（随 u 变），Λ_S = logp_sharp_mean（不随 u 变）
令 r_B(t,u) = Λ_B(t,u)/Λ_B(0.002,u)，r_S(t) = Λ_S(t)/Λ_S(0.002)。
取 r(t) = r_S(t)（清晰图那一侧的共同模衰减），则
    Λ(t,u) = r(t)·Λ(0.002,u) + Λ_B(0.002,u)·[r_B(t,u) − r(t)]
             └── 共同模部分 ──┘   └───── 差分部分 ─────┘
两项分开报，看 Λ 的 t 衰减有多少来自"两边一起缩"、多少来自"模糊图比清晰图缩得快"。

用法： venv2/bin/python quant_decomp.py
"""

import glob
import json
import os
from collections import defaultdict

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
DENSE = ["W2", "W8", "TK50"]


def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    prof = defaultdict(dict)
    for rec in d["rows"]:
        fid = rec["meta"]["fid"]
        prof[fid][round(1.0 - rec["kunif"], 4)] = (
            rec["lam_mean"], rec["logp_sharp_mean"])
    return d, {k: dict(v) for k, v in prof.items()}


def grid(prof, fids, lo=0.13, hi=0.86):
    sets = [set(prof[f]) for f in fids]
    ug_all = sorted(set.intersection(*sets), reverse=True)
    return [u for u in ug_all if lo <= u <= hi]


def main():
    ra = load("sweep_bdense.json")
    pa = dict(ra[1])
    rl = load("sweep_blow.json")
    if rl is not None:
        for fid, dd in rl[1].items():
            pa.setdefault(fid, {}).update(dd)

    cells = [(0.002, pa)]
    for fn in sorted(glob.glob(os.path.join(OUT, "sweep_bt*.json"))):
        r = load(os.path.basename(fn))
        if r is None:
            continue
        cells.append((float(r[0]["cfg"]["t"]), dict(r[1])))
    cells.sort(key=lambda c: c[0])
    ts = [t for t, _ in cells]

    ug = grid(pa, DENSE)
    LS = {}
    for t, prof in cells:
        LS[t] = float(prof[DENSE[0]][round(ug[0], 4)][1])
    rS = {t: LS[t] / LS[0.002] for t in ts}

    print("=" * 100)
    print(" 1. r_S(t) = Λ_S(t)/Λ_S(0.002)（清晰图对数密度的共同模衰减，u 无关）")
    print("=" * 100)
    print(f" {'t':>7} | {'Λ_S':>10} | {'r_S':>8} | {'1-r_S':>8}")
    print(" " + "-" * 42)
    for t in ts:
        print(f" {t:>7g} | {LS[t]:>10.1f} | {rS[t]:>8.4f} | {1-rS[t]:>8.4f}")

    print("\n" + "=" * 100)
    print(" 2. r_B(t,u) = Λ_B(t,u)/Λ_B(0.002,u)（模糊图对数密度的衰减，随 u 变）")
    print("=" * 100)
    print(f" {'u':>6} | " + " | ".join(f"{f:>18}" for f in DENSE) +
          f" | {'r_S':>8} | {'r_B-r_S 均值':>13}")
    print(" " + "-" * (10 + 21 * len(DENSE) + 24))
    rB = {}
    for u in ug:
        row = []
        diffs = []
        for f in DENSE:
            b0 = cells[0][1][f][round(u, 4)][0] + LS[0.002]
            vs = []
            for t in ts:
                bt = dict(cells)[t][f][round(u, 4)][0] + LS[t]
                vs.append(bt / b0)
            rB[(f, u)] = vs
            row.append(f"{vs[-1]:>10.4f}(t=.05)")
            diffs.append(vs[-1] - rS[0.05])
        print(f" {u:>6.3f} | " + " | ".join(row) + f" | {rS[0.05]:>8.4f} | "
              f"{sum(diffs)/len(diffs):>+13.4f}")
    print(" " + "-" * (10 + 21 * len(DENSE) + 24))
    print(" 括号内是 t=0.05 的值。若 r_B < r_S，说明模糊图的对数密度比清晰图掉得快。")

    print("\n" + "=" * 100)
    print(" 3. Λ(t,u) 的衰减分解：共同模部分 vs 差分部分")
    print("=" * 100)
    print(f" {'t':>7} | {'族':<5} | {'u':>6} | {'Λ(0.002,u)':>12} | "
          f"{'Λ(t,u)':>10} | {'共同模':>9} | {'差分':>9} | "
          f"{'共同模占比':>11}")
    print(" " + "-" * 92)
    for t in ts[1:]:
        for f in DENSE:
            for u in [ug[0], ug[len(ug) // 2], ug[-1]]:
                l0 = cells[0][1][f][round(u, 4)][0]
                lt = dict(cells)[t][f][round(u, 4)][0]
                b0 = l0 + LS[0.002]
                com = rS[t] * l0
                diff = b0 * (rB[(f, u)][ts.index(t)] - rS[t])
                tot = com + diff
                share = com / tot if abs(tot) > 1e-9 else float("nan")
                print(f" {t:>7g} | {f:<5} | {u:>6.3f} | {l0:>12.1f} | "
                      f"{lt:>10.1f} | {com:>9.1f} | {diff:>9.1f} | "
                      f"{share:>11.3f}")
    print(" " + "-" * 92)
    print(" 「共同模」= 两边一起缩（r_S·Λ）；「差分」= 模糊图比清晰图多缩的部分。")
    print(" 两者相加精确等于 Λ(t,u)（恒等式），占比说明衰减主要走哪条路。")

    print("\n" + "=" * 100)
    print(" 4. 汇总：Λ 是大量之差，所以它对 t 极敏感")
    print("=" * 100)
    for t in ts:
        lb = cells[ts.index(t)][1][DENSE[0]][round(ug[0], 4)][0] + LS[t]
        lam = cells[ts.index(t)][1][DENSE[0]][round(ug[0], 4)][0]
        print(f" t={t:<6g} Λ_B={lb:>9.1f}  Λ_S={LS[t]:>9.1f}  "
              f"Λ={lam:>8.1f}  Λ/Λ_B={lam/lb:>7.4f}  "
              f"r_S={rS[t]:.4f}  Λ/Λ(0.002)={lam/cells[0][1][DENSE[0]][round(ug[0],4)][0]:.4f}")
    print("\n 关键比：Λ 只占 Λ_B 的 ~2.6%，而 r_S 与 r_B 只差 ~0.9%；")
    print("        所以共同模衰减拿掉 ~26%，差分再拿掉 ~36%，合计 Λ 掉 62%。")
    print("        放大倍数 = (Λ_B/Λ) × (r_S − r_B) ≈ 39 × 0.009 ≈ 0.36，量级对得上。")


if __name__ == "__main__":
    main()
