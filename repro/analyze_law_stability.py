r"""Offline judge for sweep_law_stability.py; post-hoc review2 diagnostics.

Use the original degree-5/all-common-u derivative and unweighted closed-form fit. Report
per-replicate curves and parameters, between-probe SD/range, and same-probe tolerance changes.
With four probes, these are variability summaries rather than precise confidence intervals.
"""

import argparse
import json
import os

import numpy as np

from analyze_review2 import PARAMETERS, fit_once

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def statistics(values):
    array = np.asarray(values, dtype=np.float64)
    return {"n": int(array.size), "mean": float(array.mean()),
            "sd": float(array.std(ddof=1)) if array.size > 1 else None,
            "min": float(array.min()), "max": float(array.max())}


def derivative(u, per_image):
    mean = per_image.mean(axis=0)
    if np.any(mean <= 0):
        return None
    coefficients = np.polyfit(np.log(u), np.log(mean), 5)
    return np.polyval(np.polyder(coefficients), np.log(u))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="law_stability.json")
    parser.add_argument("--out", default="law_stability_analysis.json")
    args = parser.parse_args()
    with open(os.path.join(PROJ, "results", args.source)) as handle:
        raw = json.load(handle)
    families = raw["cfg"]["families"]
    replicates = raw["cfg"]["replicates"]
    n_images = len(raw["cfg"]["images"])
    rows = {}
    for row in raw["rows"]:
        if len(row["operators"]) != 20:
            raise RuntimeError("Incomplete operator curve")
        u = np.asarray([operator["u_rfft"] for operator in row["operators"]])
        per_image = np.asarray([operator["lambda"] for operator in row["operators"]]).T
        if per_image.shape != (n_images, 20):
            raise RuntimeError("Malformed per-image matrix")
        profile = {row["family"]: (u, per_image)}
        fit = fit_once(profile, row["family"])
        curve = derivative(u, per_image)
        if fit is None or curve is None:
            raise RuntimeError("Non-positive curve or failed fit")
        key = (row["family"], row["replicate"], row["rtol"])
        rows[key] = {"fit": fit, "b_curve": curve, "u": u,
                     "per_image_lambda": per_image}
    output = {"cfg": {"source": args.source, "analysis": "post-hoc review2",
                      "raw_config": raw["cfg"]}, "rows": [], "probe_variability": {},
              "tolerance": {}}
    for (family, repeat, tolerance), value in rows.items():
        output["rows"].append({"family": family, "replicate": repeat,
                               "rtol": tolerance, "fit": value["fit"],
                               "u": value["u"].tolist(),
                               "b_curve": value["b_curve"].tolist()})
    for family in families:
        selected = [rows[(family, repeat, 1e-5)] for repeat in range(replicates)]
        curves = np.stack([value["b_curve"] for value in selected])
        entry = {"parameters": {}, "three_family_b_curve": {}}
        for parameter in PARAMETERS:
            entry["parameters"][parameter] = statistics(
                [value["fit"][parameter] for value in selected])
        entry["b_curve_sd_by_u"] = curves.std(axis=0, ddof=1).tolist() if replicates > 1 else []
        entry["b_curve_max_abs_sd"] = (float(curves.std(axis=0, ddof=1).max())
                                       if replicates > 1 else 0.0)
        output["probe_variability"][family] = entry
    means = []
    for repeat in range(replicates):
        fits = {family: rows[(family, repeat, 1e-5)]["fit"] for family in families}
        means.append({parameter: float(np.mean([fits[family][parameter]
                                                for family in families]))
                      for parameter in PARAMETERS})
    output["three_family_probe_variability"] = {
        parameter: statistics([mean[parameter] for mean in means]) for parameter in PARAMETERS}
    for family in families:
        default = rows[(family, 0, 1e-5)]
        strict = rows[(family, 0, 1e-6)]
        lambda_difference = default["per_image_lambda"] - strict["per_image_lambda"]
        output["tolerance"][family] = {
            "parameter_default_minus_strict": {
                parameter: default["fit"][parameter] - strict["fit"][parameter]
                for parameter in PARAMETERS},
            "b_curve_default_minus_strict": (
                default["b_curve"] - strict["b_curve"]).tolist(),
            "max_abs_lambda_difference_nats": float(np.abs(lambda_difference).max()),
            "rms_lambda_difference_nats": float(np.sqrt(np.mean(lambda_difference ** 2))),
            "strict_rms": strict["fit"]["rms"], "default_rms": default["fit"]["rms"]}
    output_path = os.path.join(PROJ, "results", args.out)
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=1)
    print("[probe variability]", json.dumps(output["three_family_probe_variability"], indent=1))
    print("[tolerance]", json.dumps(output["tolerance"], indent=1))
    print("[saved]", output_path)


if __name__ == "__main__":
    main()
