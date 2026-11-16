import json

import numpy as np

d = json.load(open("/data/experiment/de_conv_1/results/fig_model.json"))
C = d["cells"]
fams = d["dense"]
ug = np.array(d["ug"])
lx = np.log(ug)

for lab, (x, y) in [("prior B/C", ("B(afq+afq)", "C(ffq+afq)")),
                    ("images A/C", ("A(ffq+ffq)", "C(ffq+afq)"))]:
    rs = []
    for f in fams:
        a = np.array(C[x]["fam"][f]["lam"], dtype=float)
        b = np.array(C[y]["fam"][f]["lam"], dtype=float)
        rs.append(a / b)
    R = np.array(rs)
    print(f"{lab}: Lambda ratio in [{R.min():.4f}, {R.max():.4f}]  "
          f"=> {(1-R.max())*100:+.1f}% .. {(1-R.min())*100:+.1f}%")
    for k, f in enumerate(fams):
        sl = float(np.polyfit(lx, R[k], 1)[0])
        print(f"    {f}: ratio u=0.85 -> {R[k][0]:.4f}, u=0.14 -> {R[k][-1]:.4f}"
              f"   slope vs log u = {sl:+.5f}")
