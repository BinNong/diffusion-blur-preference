"""Offline summary of direct score-gradient profiles for review2; no success threshold.

This is a two-image/noise pilot per source, not a replacement 79/80 population estimate.
Compare like-for-like PnP and score-grid selections, objective decrease, gradient/drift,
top-five spread, and directional ODE gradient consistency. Do not infer global optimality.
"""

import hashlib
import json
import math
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJ, "results")


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def pnp_argmin(source, index, noise):
    filename = "phaseC_pool20.json" if source == "generated" else "real_bsds_profile.json"
    with open(os.path.join(RESULTS, filename)) as handle:
        data = json.load(handle)
    for record in data["records"]:
        if record["img"] == index and abs(record["noise"] - noise) < 1e-10:
            best = min(record["profile"], key=lambda row: row["L"])
            return {"theta": best["theta"], "L": best["L"],
                    "top5_spread": max(row["L"] for row in sorted(
                        record["profile"], key=lambda row: row["L"])[:5]) - best["L"]}
    raise RuntimeError("Missing like-for-like PnP record")


def summarise_record(source, record):
    profile = sorted(record["profile"], key=lambda row: row["L"])
    runs = record["runs"]
    best = profile[0]
    check = record["directional_gradient_check"]
    relative_gradient_error = abs(check["difference"]) / max(
        abs(check["ode_central_difference_per_rms"]), 1e-30)
    return {
        "img": record["img"], "noise": record["noise"],
        "theta_star": 1.5, "score_grid_theta": best["theta"],
        "score_grid_factor": best["theta"] / 1.5,
        "direction": "over" if best["theta"] > 1.5 else "under" if best["theta"] < 1.5 else "equal",
        "top5_spread_nats": profile[4]["L"] - best["L"],
        "pnp": pnp_argmin(source, record["img"], record["noise"]),
        "all_runs": len(runs),
        "objective_decreased_runs": sum(row["L_final_minus_initial"] < 0 for row in runs),
        "max_abs_objective_increase_nats": max(0.0, max(
            row["L_final_minus_initial"] for row in runs)),
        "final_gradient_rms": {
            "median": sorted(row["gradient_rms_final"] for row in runs)[len(runs) // 2],
            "max": max(row["gradient_rms_final"] for row in runs)},
        "final_drift_rms_last50": {
            "median": sorted(row["image_drift_rms_last50"] for row in runs)[len(runs) // 2],
            "max": max(row["image_drift_rms_last50"] for row in runs)},
        "max_abs_pixel": max(row["max_abs_pixel"] for row in runs),
        "directional_gradient_check": dict(check, absolute_relative_error=relative_gradient_error),
    }


def main():
    output = {"cfg": {"analysis": "post-hoc direct score profile review2",
                      "global_optimality": False, "images_per_source": 2,
                      "noise_levels": [0.172, 0.5]}, "sources": {}}
    for source in ("generated", "real"):
        filename = "score_profile_%s.json" % source
        path = os.path.join(RESULTS, filename)
        with open(path) as handle:
            raw = json.load(handle)
        if raw["cfg"]["smoke"] or len(raw["records"]) != 4:
            raise RuntimeError("Expected four non-smoke records for " + source)
        rows = [summarise_record(source, record) for record in raw["records"]]
        recheck_path = os.path.join(RESULTS, "score_profile_%s_recheck.json" % source)
        recheck_summary = None
        if os.path.exists(recheck_path):
            with open(recheck_path) as handle:
                recheck = json.load(handle)
            if len(recheck["records"]) != 4 or any(len(r["replicates"]) != 4
                                                   for r in recheck["records"]):
                raise RuntimeError("Incomplete Hutchinson recheck for " + source)
            max_sd = 0.0
            for row, replicated in zip(rows, recheck["records"]):
                point_sds = []
                for index in range(len(raw["records"][0]["runs"])):
                    values = [rep["q_values"][index] for rep in replicated["replicates"]]
                    mean = sum(values) / 4.0
                    point_sds.append(math.sqrt(sum((value - mean) ** 2
                                                   for value in values) / 3.0))
                max_sd = max(max_sd, max(point_sds))
                row["hutchinson_recheck"] = {
                    "probe_argmins": [rep["summary"]["argmin_theta"]
                                      for rep in replicated["replicates"]],
                    "mean_of_four_argmin": replicated["mean_of_four_probes"],
                    "max_endpoint_q_sd_nats": max(point_sds),
                    "original_probe_q_difference":
                        replicated["original_probe_check"]["difference"]}
            recheck_summary = {
                "path": os.path.basename(recheck_path), "max_endpoint_q_sd_nats": max_sd,
                "probe_argmins": [row["hutchinson_recheck"]["probe_argmins"] for row in rows],
                "mean_of_four_argmins": [row["hutchinson_recheck"]["mean_of_four_argmin"]
                                         for row in rows]}
        output["sources"][source] = {
            "source_sha256": sha256(path),
            "config": raw["cfg"], "records": rows,
            "over": sum(row["direction"] == "over" for row in rows),
            "under": sum(row["direction"] == "under" for row in rows),
            "equal": sum(row["direction"] == "equal" for row in rows),
            "mean_factor": sum(row["score_grid_factor"] for row in rows) / len(rows),
            "objective_decreased_total": sum(row["objective_decreased_runs"] for row in rows),
            "runs_total": sum(row["all_runs"] for row in rows),
            "max_final_gradient_rms": max(row["final_gradient_rms"]["max"] for row in rows),
            "max_directional_relative_error": max(
                row["directional_gradient_check"]["absolute_relative_error"] for row in rows),
            "hutchinson_recheck": recheck_summary,
        }
    path = os.path.join(RESULTS, "score_profile_analysis.json")
    with open(path, "w") as handle:
        json.dump(output, handle, indent=1)
    for source, summary in output["sources"].items():
        print("[", source, "]", {key: value for key, value in summary.items()
                                  if key not in {"config", "records", "source_sha256"}})
        for row in summary["records"]:
            print(row["img"], row["noise"], "score", row["score_grid_theta"],
                  "PnP", row["pnp"]["theta"], "top5", row["top5_spread_nats"],
                  "gradient", row["final_gradient_rms"],
                  "gradcheck relerr", row["directional_gradient_check"]["absolute_relative_error"])
    print("[saved]", path)


if __name__ == "__main__":
    main()
