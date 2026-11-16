import json

import numpy as np

d = json.load(open("/data/experiment/de_conv_1/results/fig_bt.json"))
ug = np.array(d["ug"])
lx = np.log(ug)
print("D(u) 两种估计量的斜率对照（H0 要求两者都为 0）:")
print("  mean-of-logs = 预官 T4 的算子（逐图配对差再取均值，排除 Λ<=0 的图）")
print("  log-of-means  = 与 b(u) 同一估计量，永远有定义")
for e in sorted(d["new"], key=lambda x: x["t"]):
    for fid in d["dense"]:
        f = e["fam"][fid]
        A = np.array(f["D"], dtype=float)
        B = np.array(f["Dlogmean"], dtype=float)
        g = np.isfinite(A)
        sa = float(np.polyfit(lx[g], A[g], 1)[0])
        sb = float(np.polyfit(lx, B, 1)[0])
        nex = int(sum(f["nexcl"]))
        tot = 20 * len(ug)
        print(f"  t={e['t']:<6g} {fid:<5} mean-of-logs 斜率={sa:+.4f}  "
              f"log-of-means 斜率={sb:+.4f}  排除 {nex}/{tot} 个 (图,u) 对")
