"""Post-hoc decomposition at each empirical profile-MAP argmin for review 1."""

import argparse
import json
import os
import random


PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def percentile(values, probability):
    ordered = sorted(values)
    return ordered[min(int(probability * len(ordered)), len(ordered)-1)]


def decompose(data):
    rows = []
    for record in data["records"]:
        if record["noise"] < 0.172:
            continue
        profile = sorted(record["profile"], key=lambda point: point["theta"])
        optimum = min(profile, key=lambda point: point["L"])
        truth = next(point for point in profile if point["theta"] == record["theta_star"])
        row = {"img": record["img"], "noise": record["noise"],
               "theta_opt": optimum["theta"], "theta_star": record["theta_star"],
               "delta_data": optimum["data"]-truth["data"],
               "delta_q": optimum["q"]-truth["q"],
               "delta_L": optimum["L"]-truth["L"]}
        if abs(row["delta_data"]+row["delta_q"]-row["delta_L"]) > 1e-8:
            raise ValueError("Stored profile terms do not add up")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="overblur_optimum.json")
    parser.add_argument("--repeats", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    output = {"cfg": {"noise_threshold": 0.172, "repeats": args.repeats,
                      "seed": args.seed,
                      "analysis_status": "post-hoc, requested during review"},
              "pools": {}}

    for name, filename in [("main", "phaseC_pool20.json"),
                           ("independent", "overblur_indep.json")]:
        with open(os.path.join(PROJ, "results", filename)) as handle:
            rows = decompose(json.load(handle))
        images = sorted({row["img"] for row in rows})
        by_image = {image: [row for row in rows if row["img"] == image]
                    for image in images}

        def sample_rows(sample):
            return [row for image in sample for row in by_image[image]]

        def statistics(selected):
            return {"mean_delta_data": sum(r["delta_data"] for r in selected)/len(selected),
                    "mean_delta_q": sum(r["delta_q"] for r in selected)/len(selected),
                    "mean_delta_L": sum(r["delta_L"] for r in selected)/len(selected),
                    "negative_delta_q_proportion": sum(r["delta_q"] < 0 for r in selected)/len(selected),
                    "prior_dominant_proportion": sum(-r["delta_q"] > r["delta_data"]
                                                     for r in selected)/len(selected)}

        observed = statistics(rows)
        bootstraps = [statistics(sample_rows([rng.choice(images) for _ in images]))
                      for _ in range(args.repeats)]
        intervals = {key: {"low": percentile([b[key] for b in bootstraps], 0.025),
                           "high": percentile([b[key] for b in bootstraps], 0.975)}
                     for key in observed}
        output["pools"][name] = {"source": filename, "n": len(rows),
                                 "observed": observed, "cluster_95_intervals": intervals,
                                 "rows": rows}
        print(name, json.dumps(observed, indent=2))
        print("cluster intervals", json.dumps(intervals, indent=2))

    path = os.path.join(PROJ, "results", args.output)
    with open(path, "w") as handle:
        json.dump(output, handle, indent=1)
    print("[saved]", path)


if __name__ == "__main__":
    main()
