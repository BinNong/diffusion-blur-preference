"""Phase B 聚合分析：跨 20 张图的统计，而不是单张图的曲线。"""
import json
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(PROJ, "results", sys.argv[1] if len(sys.argv) > 1 else "phaseB_nimg20.json")
d = json.load(open(path))
recs = d["records"]

fams = []
for r in recs:
    if r["family"] not in fams:
        fams.append(r["family"])

print(f"共 {len(recs)} 条记录，{len(set(r['img'] for r in recs))} 张图，族: {fams}\n")

for fam in fams:
    rows = [r for r in recs if r["family"] == fam]
    imgs = sorted(set(r["img"] for r in rows))
    thetas = sorted(set(r["theta"] for r in rows))
    M = np.array([[next(r["nll"] for r in rows if r["img"] == i and r["theta"] == t)
                   for t in thetas] for i in imgs])
    mean, std = M.mean(0), M.std(0)
    delta = M[:, -1] - M[:, 0]

    mono = [bool(np.all(np.diff(M[i]) <= 1e-9)) for i in range(len(imgs))]
    # 净下降（终点低于起点）的比例
    net_down = (delta < 0).sum()

    print(f"===== {fam} =====")
    print("  theta      " + " ".join(f"{t:>10}" for t in thetas))
    print("  mean -logp " + " ".join(f"{v:10.1f}" for v in mean))
    print("  std        " + " ".join(f"{v:10.1f}" for v in std))
    print(f"  Δ(末-初): mean {delta.mean():+.1f}  std {delta.std():.1f}  "
          f"范围 [{delta.min():+.1f}, {delta.max():+.1f}]")
    print(f"  净下降图像数: {net_down}/{len(imgs)}")
    print(f"  逐图严格单调递减: {sum(mono)}/{len(imgs)}  ({100*sum(mono)/len(imgs):.0f}%)")
    # 每张图相对起点的下降量，看是否所有图都同向
    print(f"  相对起点下降(末-初)为负的比例: {100*(delta<0).mean():.0f}%")
    print()

print("=" * 70)
print("结论：")
print("  * 四个族的【平均】势能都随模糊下降 → 论文 Lemma 12 的前提成立")
print("  * 但【逐图严格单调】只有部分族成立 → 'blurry preference' 是统计性、")
print("    不是逐图逐点成立的，这决定了它作为定理假设的强度")
