import json

FILES = ["sweep_bdense.json", "sweep_bt0005.json", "sweep_bt001.json",
         "sweep_bt002.json", "sweep_bt005.json"]
TARGET = 0.85

for fn in FILES:
    d = json.load(open("/data/experiment/de_conv_1/results/" + fn))
    t = d["cfg"]["t"]
    for r in d["rows"]:
        fid = r["meta"]["fid"]
        u = round(1.0 - r["kunif"], 4)
        if fid == "TK50" and abs(u - TARGET) < 1e-6:
            pi = r["lam_per_img"]
            neg = sum(1 for v in pi if v <= 0)
            print(f"{fn:22s} t={t:<6g} lam={r['lam_mean']:12.2f} "
                  f"std={r['lam_std']:10.2f} min={min(pi):12.2f} "
                  f"max={max(pi):12.2f} nneg={neg}")
            if neg:
                bad = [(i, v) for i, v in enumerate(pi) if v <= 0]
                print("      negative/zero per-image:", bad)
                lp = r.get("logp_blur_mean")
                print(f"      logp_blur_mean={lp}")
