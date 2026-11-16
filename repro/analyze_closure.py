"""Offline summaries for the review-closure experiments; no model evaluation.

Usage: ./venv2/bin/python repro/analyze_closure.py --kind {real,adm64,adm128,profile,local}
Requires the complete expected row/record counts. Reports all settings, not selected subsets.
Cluster intervals resample whole images. These are descriptive, not universal claims.
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": int(values.size), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values)>1 else 0.0,
            "se": float(values.std(ddof=1)/np.sqrt(values.size)) if len(values)>1 else 0.0,
            "positive": int((values>0).sum()),
            "min": float(values.min()), "max": float(values.max())}


def clean_analysis(filename, expected, output):
    with open(os.path.join(PROJ, "results", filename)) as handle:
        data = json.load(handle)
    rows = data["rows"]
    if len(rows) != expected:
        raise RuntimeError("Incomplete input: %d != %d" % (len(rows), expected))
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["t"]) if "t" in row else str(row["sigma"]),
                row["family"], float(row["theta"]))].append(row)
    result = {"source": filename, "n": len(rows), "generated": False,
              "groups": {}, "clustered": {}}
    for key, selected in sorted(groups.items()):
        t, family, theta = key
        result["groups"]["|".join(map(str, key))] = {
            "t": t, "family": family, "theta": theta,
            **summary([row["lambda"] for row in selected])}
    cluster_groups = defaultdict(list)
    for row in rows:
        cluster_groups[(row.get("t", row.get("sigma")), row["img"])].append(row["lambda"])
    images = sorted({image for _, image in cluster_groups})
    for t in sorted({key[0] for key in cluster_groups}):
        selected = {(tt, image): values for (tt, image), values in cluster_groups.items() if tt == t}
        rng = random.Random(20261111)
        def statistic(sample):
            values = [v for image in sample for v in selected[(t, image)]]
            return [float(np.mean(values)), float(np.mean(np.asarray(values)>0))]
        observed = statistic(images)
        boot = [statistic([rng.choice(images) for _ in images]) for _ in range(100000)]
        boot = np.asarray(boot)
        result["clustered"][str(t)] = {"n_images": len(images), "mean": observed[0],
            "positive_proportion": observed[1],
            "mean_cluster95": np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
            "positive_cluster95": np.percentile(boot[:, 1], [2.5, 97.5]).tolist()}
    with open(os.path.join(PROJ, "results", output), "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result, indent=2))


def profile_analysis():
    with open(os.path.join(PROJ, "results", "real_bsds_profile.json")) as handle:
        data = json.load(handle)
    records = data["records"]
    if len(records) != 16:
        raise RuntimeError("Incomplete real profile input")
    rows = []
    for record in records:
        profile = record["profile"]
        optimum = min(profile, key=lambda p: p["L"])
        truth = next(p for p in profile if p["theta"] == record["theta_star"])
        top = sorted(p["L"] for p in profile)[:5]
        rows.append({"img": record["img"], "noise": record["noise"],
                     "theta_star": record["theta_star"], "theta_opt": optimum["theta"],
                     "factor": optimum["theta"]/record["theta_star"],
                     "absolute_relative_error": abs(optimum["theta"]-record["theta_star"])/record["theta_star"],
                     "delta_data": optimum["data"]-truth["data"],
                     "delta_q": optimum["q"]-truth["q"],
                     "top5_spread": top[-1]-top[0]})
    result = {"source": "real_bsds_profile.json", "n": len(rows),
              "over": sum(r["theta_opt"]>r["theta_star"] for r in rows),
              "under": sum(r["theta_opt"]<r["theta_star"] for r in rows),
              "equal": sum(r["theta_opt"]==r["theta_star"] for r in rows),
              "mean_factor": float(np.mean([r["factor"] for r in rows])),
              "mean_absolute_relative_error": float(np.mean([r["absolute_relative_error"] for r in rows])),
              "mean_delta_data": float(np.mean([r["delta_data"] for r in rows])),
              "mean_delta_q": float(np.mean([r["delta_q"] for r in rows])),
              "top5_spread_range": [min(r["top5_spread"] for r in rows),
                                    max(r["top5_spread"] for r in rows)], "rows": rows}
    with open(os.path.join(PROJ, "results", "real_profile_analysis.json"), "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result, indent=2))


def local_analysis():
    with open(os.path.join(PROJ, "results", "local_quadratic.json")) as handle:
        data = json.load(handle)
    if len(data["rows"]) != 4:
        raise RuntimeError("Incomplete local quadratic input")
    rows = []
    for row in data["rows"]:
        item = {"source": row["source"], "t": row["t"],
                "actual_delta_q": row["actual_delta_q_at_1"], "predictions": {}}
        for half, prediction in row["predictions"].items():
            item["predictions"][half] = {
                "predicted_delta_q": prediction["predicted_delta_q_at_1"],
                "signed_error": prediction["absolute_error"],
                "absolute_error": abs(prediction["absolute_error"])}
        rows.append(item)
    result = {"source": "local_quadratic.json", "scope": "single directional Taylor test per source/t",
              "rows": rows}
    with open(os.path.join(PROJ, "results", "local_quadratic_analysis.json"), "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["real", "adm64", "adm128", "profile", "local"], required=True)
    args = parser.parse_args()
    if args.kind == "real":
        clean_analysis("real_bsds_edm.json", 384, "real_bsds_edm_analysis.json")
    elif args.kind == "adm64":
        clean_analysis("adm_imagenet64_uncond_bsds.json", 96, "adm64_analysis.json")
    elif args.kind == "adm128":
        clean_analysis("adm_imagenet128_cond_bsds.json", 32, "adm128_analysis.json")
    elif args.kind == "profile":
        profile_analysis()
    else:
        local_analysis()


if __name__ == "__main__":
    main()
