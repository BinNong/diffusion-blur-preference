"""Reviewer-requested post-hoc cluster bootstrap for the 20-image over-blur study."""

import argparse
import json
import os
import random


PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def percentile(values, probability):
    values = sorted(values)
    return values[min(int(probability * len(values)), len(values) - 1)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="phaseC_pool20.json")
    parser.add_argument("--output", default="overblur_cluster.json")
    parser.add_argument("--repeats", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261109)
    args = parser.parse_args()

    with open(os.path.join(PROJ, "results", args.input)) as handle:
        data = json.load(handle)
    thetas = data["cfg"]["thetas"]
    by_image = {}
    for record in data["records"]:
        if record["noise"] < 0.172:
            continue
        profile = sorted(record["profile"], key=lambda point: point["theta"])
        best = min(profile, key=lambda point: point["L"])
        by_image.setdefault(record["img"], []).append(
            best["theta"] / record["theta_star"])

    images = sorted(by_image)
    if not images or any(len(by_image[image]) != 4 for image in images):
        raise ValueError("Expected four high-noise configurations per image")

    def statistics(sample):
        factors = [factor for image in sample for factor in by_image[image]]
        return {
            "overblur_proportion": sum(f > 1.0 for f in factors) / len(factors),
            "mean_overblur_factor": sum(factors) / len(factors),
        }

    observed = statistics(images)
    rng = random.Random(args.seed)
    bootstraps = [statistics([rng.choice(images) for _ in images])
                  for _ in range(args.repeats)]
    intervals = {
        key: {
            "level": 0.95,
            "method": "percentile bootstrap over whole-image clusters",
            "low": percentile([row[key] for row in bootstraps], 0.025),
            "high": percentile([row[key] for row in bootstraps], 0.975),
        }
        for key in observed
    }
    result = {
        "cfg": {
            "input": args.input,
            "noise_threshold": 0.172,
            "n_images": len(images),
            "n_configurations": sum(map(len, by_image.values())),
            "repeats": args.repeats,
            "seed": args.seed,
            "analysis_status": "post-hoc, requested during review",
        },
        "observed": observed,
        "confidence_intervals": intervals,
        "per_image_factors": {str(image): by_image[image] for image in images},
    }
    output = os.path.join(PROJ, "results", args.output)
    with open(output, "w") as handle:
        json.dump(result, handle, indent=1)
    print(json.dumps({"observed": observed, "confidence_intervals": intervals}, indent=2))
    print("[saved]", output)


if __name__ == "__main__":
    main()
