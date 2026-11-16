"""Offline analysis for the review-1 multi-kernel over-blur pilot."""

import json
import os
import random


PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def percentile(values, probability):
    values = sorted(values)
    return values[min(int(probability*len(values)), len(values)-1)]


def main():
    with open(os.path.join(PROJ, "results", "overblur_generalization_completed.json")) as handle:
        data = json.load(handle)
    rows = []
    for record in data["records"]:
        profile = sorted(record["profile"], key=lambda point: point["theta"])
        optimum = min(profile, key=lambda point: point["L"])
        truth = next(point for point in profile if point["theta"] == record["theta_star"])
        rows.append({"family": record["family"], "theta_star": record["theta_star"],
                     "img": record["img"], "noise": record["noise"],
                     "theta_opt": optimum["theta"],
                     "overblur": float(optimum["theta"] > record["theta_star"]),
                     "signed_factor": optimum["theta"]/record["theta_star"],
                     "absolute_relative_error": abs(optimum["theta"]-record["theta_star"])
                                                /record["theta_star"],
                     "delta_data": optimum["data"]-truth["data"],
                     "delta_q": optimum["q"]-truth["q"]})
    rng = random.Random(20261110)
    result = {"cfg": {"source": "overblur_generalization_completed.json",
                      "analysis_status": "diagnostic, pre-registered before collection",
                      "repeats": 100000, "seed": 20261110},
              "groups": {}}
    groups = [(name, [r for r in rows if r["family"] == name])
              for name in sorted({r["family"] for r in rows})]
    groups += [(f"gaussian_theta_{theta:g}",
                [r for r in rows if r["family"] == "gaussian" and r["theta_star"] == theta])
               for theta in sorted({r["theta_star"] for r in rows if r["family"] == "gaussian"})]
    groups += [(f"noise_{noise:g}", [r for r in rows if r["noise"] == noise])
               for noise in sorted({r["noise"] for r in rows})]
    groups.append(("pooled", rows))

    for name, selected in groups:
        images = sorted({r["img"] for r in selected})
        by_image = {image: [r for r in selected if r["img"] == image] for image in images}

        def statistics(sample):
            values = [r for image in sample for r in by_image[image]]
            return {"n": len(values),
                    "overblur_proportion": sum(r["overblur"] for r in values)/len(values),
                    "underblur_proportion": sum(r["signed_factor"] < 1 for r in values)/len(values),
                    "mean_signed_factor": sum(r["signed_factor"] for r in values)/len(values),
                    "mean_absolute_relative_error": sum(r["absolute_relative_error"]
                                                        for r in values)/len(values),
                    "mean_delta_data": sum(r["delta_data"] for r in values)/len(values),
                    "mean_delta_q": sum(r["delta_q"] for r in values)/len(values),
                    "negative_delta_q_proportion": sum(r["delta_q"] < 0
                                                       for r in values)/len(values)}

        observed = statistics(images)
        bootstraps = [statistics([rng.choice(images) for _ in images]) for _ in range(100000)]
        intervals = {key: {"low": percentile([b[key] for b in bootstraps], 0.025),
                           "high": percentile([b[key] for b in bootstraps], 0.975)}
                     for key in observed if key != "n"}
        result["groups"][name] = {"observed": observed, "cluster_95_intervals": intervals}
        print(name, json.dumps(observed, indent=2))

    result["rows"] = rows
    path = os.path.join(PROJ, "results", "overblur_generalization_analysis.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=1)
    print("[saved]", path)


if __name__ == "__main__":
    main()
