r"""Post-hoc Hutchinson replication of direct score-profile endpoints (review2).

Motivation: first completed score profile has near-tied low-theta grid points. Registered after
that record was visible, before any replication. No operator, image, noise, or outcome threshold
is selected. Evaluate EVERY final image from both starts at all 21 grid points, four probes per
profile, without reoptimising or using theta_star to choose a point.

R0: load score_profile_SOURCE.json and its saved final images. Reconstruct the original probe
    sequence from seed 20261113 and check q of the first endpoint against its stored value.
    A mismatch is reported; it must not be described as a recovered original vector.
R1: probes 1--3 use independent deterministic seeds. Probe 0 is the reconstructed sequence,
    whether or not R0 matches. Every probe is shared across all theta/start endpoints within a
    profile and saved verbatim. RK45 rtol=atol=1e-5.
R2: for each replicate independently repeat the original final-L start selection and grid
    argmin. Also average L across the four probes before selection. Report every argmin, top-5
    spread and per-point SD. No global optimisation or trace-free claim is made.

Run: ./venv2/bin/python repro/recheck_score_profile.py --source generated
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn


def choose_profile(runs, q_values):
    selected = []
    for theta in sorted({row["theta"] for row in runs}):
        candidates = [dict(row, q=q_values[index], L=row["data"] + q_values[index])
                      for index, row in enumerate(runs) if row["theta"] == theta]
        selected.append(min(candidates, key=lambda row: row["L"]))
    ordered = sorted(selected, key=lambda row: row["L"])
    return selected, {"argmin_theta": ordered[0]["theta"],
                      "top5_spread": ordered[4]["L"] - ordered[0]["L"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("generated", "real"), default="generated")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    input_name = "score_profile_%s.json" % args.source
    output_name = args.out or ("score_profile_%s_recheck.json" % args.source)
    input_path = os.path.join(PROJ, "results", input_name)
    output_path = os.path.join(PROJ, "results", output_name)
    with open(input_path) as handle:
        raw = json.load(handle)
    if raw["cfg"]["smoke"] or len(raw["records"]) != 4:
        raise RuntimeError("R0 requires four complete non-smoke profiles")
    endpoints = torch.load(input_path.replace(".json", "_x.pt"), map_location="cpu")
    if isinstance(endpoints, list):
        endpoints = torch.cat(endpoints, dim=0)
    runs_per_profile = len(raw["records"][0]["runs"])
    if len(endpoints) != len(raw["records"]) * runs_per_profile:
        raise RuntimeError("R0 endpoint count mismatch")
    config = {"source": args.source, "input": input_name,
              "input_sha256": hashlib.sha256(open(input_path, "rb").read()).hexdigest(),
              "replicates": 4, "method": "RK45", "rtol": 1e-5, "atol": 1e-5,
              "batch_size": args.batch_size, "analysis": "post-hoc after first score record",
              "preregistered": "2026-11-13"}
    with open(__file__, "rb") as handle:
        config["script_sha256"] = hashlib.sha256(handle.read()).hexdigest()
    if os.path.exists(output_path):
        with open(output_path) as handle:
            result = json.load(handle)
        if result["cfg"] != config:
            raise RuntimeError("Existing recheck does not match frozen configuration")
    else:
        result = {"cfg": config, "records": []}

    def save():
        with open(output_path, "w") as handle:
            json.dump(result, handle, indent=1)

    torch.manual_seed(20261113)
    torch.cuda.manual_seed(20261113)
    model = load_local_model("edm-ffhq-64x64-uncond-vp.pkl").cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x, t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)
    template = endpoints[0:1].cuda()
    original_probes = [torch.randint_like(template, low=0, high=2).float() * 2.0 - 1.0
                       for _ in raw["records"]]
    original_randint = torch.randint_like
    start = time.time()

    for record_index, record in enumerate(raw["records"]):
        output_record = next((row for row in result["records"]
                              if row["record_index"] == record_index), None)
        if (output_record is not None and len(output_record["replicates"]) == 4
                and "mean_of_four_probes" in output_record):
            continue
        noise = record["noise"]
        runs = record["runs"]
        images = endpoints[record_index * runs_per_profile:(record_index + 1) * runs_per_profile]
        if output_record is None:
            output_record = {"record_index": record_index, "img": record["img"], "noise": noise,
                             "original_argmin_theta": record["argmin_theta"],
                             "original_probe_check": None, "replicates": []}
            result["records"].append(output_record)
            save()
        for replicate in range(4):
            if any(row["replicate"] == replicate for row in output_record["replicates"]):
                continue
            if replicate == 0:
                epsilon = original_probes[record_index]
            else:
                generator = torch.Generator(device="cuda").manual_seed(
                    20261114 + 1000 * replicate + record_index)
                epsilon = torch.randint(2, template.shape, generator=generator, device="cuda",
                                        dtype=torch.int64).float() * 2.0 - 1.0

            def fixed_randint(tensor, low=0, high=2):
                return ((epsilon + 1.0) / 2.0).to(tensor.dtype).expand(tensor.shape)

            @torch.no_grad()
            def q_values(batch):
                torch.randint_like = fixed_randint
                try:
                    values = likelihood_fn(batch, ode_flow, t_spans=[80.0, float(noise)],
                                           rtol=1e-5, atol=1e-5, method="RK45")
                    return [float(value) for value in values.reshape(-1)]
                finally:
                    torch.randint_like = original_randint

            q_all = []
            for offset in range(0, runs_per_profile, args.batch_size):
                batch = images[offset:offset + args.batch_size].cuda()
                q_all.extend(q_values(batch))
                print("[R1]", args.source, record_index, replicate, offset, "/",
                      runs_per_profile, "elapsed", round(time.time() - start), flush=True)
            if replicate == 0:
                output_record["original_probe_check"] = {
                    "stored_q_first_endpoint": runs[0]["q"], "recomputed_q_first_endpoint": q_all[0],
                    "difference": q_all[0] - runs[0]["q"]}
            selected, summary = choose_profile(runs, q_all)
            output_record["replicates"].append({
                "replicate": replicate, "probe_seed": "reconstructed" if replicate == 0
                else 20261114 + 1000 * replicate + record_index,
                "probe": epsilon.to(torch.int8).cpu().reshape(-1).tolist(),
                "q_values": q_all, "summary": summary,
                "selected": selected})
            save()
        mean_q = [sum(replicate["q_values"][index]
                      for replicate in output_record["replicates"]) / 4.0
                  for index in range(runs_per_profile)]
        _, mean_summary = choose_profile(runs, mean_q)
        output_record["mean_of_four_probes"] = mean_summary
        save()
        print("[R2]", args.source, record_index, "original", record["argmin_theta"],
              "replicates", [row["summary"]["argmin_theta"]
                             for row in output_record["replicates"]],
              "mean", mean_summary, flush=True)
    print("[saved]", output_path, flush=True)


if __name__ == "__main__":
    main()
