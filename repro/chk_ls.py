import json

for fn in ["sweep_bdense.json", "sweep_blow.json", "sweep_bgap.json",
           "sweep_bt0005.json", "sweep_bt001.json", "sweep_bt002.json",
           "sweep_bt005.json"]:
    d = json.load(open("/data/experiment/de_conv_1/results/" + fn))
    ls = {}
    for r in d["rows"]:
        ls.setdefault(r["logp_sharp_mean"], []).append(r["meta"]["fid"])
    print(f"{fn:22s} n={len(d['rows']):3d}  logp_sharp_mean 取值数={len(ls)}")
    for v, fids in ls.items():
        print(f"    {v:.6f}  x{len(fids)}  ({','.join(sorted(set(fids))[:4])})")
    # 自检：lam_mean == lb - ls ?
    worst = 0.0
    for r in d["rows"]:
        a = r["lam_mean"]
        b = r["logp_blur_mean"] - r["logp_sharp_mean"]
        worst = max(worst, abs(a - b) / max(abs(a), 1e-9))
    print(f"    max |lam - (lb-ls)|/|lam| = {worst:.3e}")
