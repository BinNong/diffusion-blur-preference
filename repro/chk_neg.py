import json

FILES = ["sweep_bdense.json", "sweep_bt0005.json", "sweep_bt001.json",
         "sweep_bt002.json", "sweep_bt005.json"]
BASE = "/data/experiment/de_conv_1/results/"

print(f"{'file':<22} {'t':>6} | {'算子':<12} {'u':>6} | {'Λ_mean':>10} "
      f"{'n_Λ<=0':>7} | {'最负':>9}")
print("-" * 84)
tot = 0
for fn in FILES:
    d = json.load(open(BASE + fn))
    t = d["cfg"]["t"]
    for r in d["rows"]:
        pi = r["lam_per_img"]
        neg = [(i, v) for i, v in enumerate(pi) if v <= 0]
        if not neg:
            continue
        tot += len(neg)
        fid = r["meta"]["fid"]
        u = round(1.0 - r["kunif"], 4)
        print(f"{fn:<22} {t:>6g} | {fid:<12} {u:>6.3f} | "
              f"{r['lam_mean']:>10.1f} {len(neg):>7} | {min(v for _, v in neg):>9.2f}")
print("-" * 84)
print(f"合计 {tot} 个非正 Λ（跨 4 个新 t + 基线）")
