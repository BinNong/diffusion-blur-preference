"""Post-hoc B1 check of the reconstructed score-profile Rademacher sequence.

Run only after score_profile_*_recheck.json are complete. For every one of the eight profiles,
recompute the first stored endpoint with batch size one and its reconstructed original probe.
Compare against both the original B1 q and the recheck's batched q. This distinguishes a seed
mismatch from a batching/numerical difference; no outcome threshold is used.
"""

import json
import os
import sys

import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn


def main():
    output = {"analysis": "post-hoc B1 original-probe check", "sources": {}}
    for source in ("generated", "real"):
        stem = "score_profile_%s" % source
        with open(os.path.join(PROJ, "results", stem + ".json")) as handle:
            raw = json.load(handle)
        with open(os.path.join(PROJ, "results", stem + "_recheck.json")) as handle:
            recheck = json.load(handle)
        if len(recheck["records"]) != 4 or any(len(r["replicates"]) != 4
                                               for r in recheck["records"]):
            raise RuntimeError("Incomplete recheck")
        endpoints = torch.load(os.path.join(PROJ, "results", stem + "_x.pt"),
                               map_location="cpu")
        if isinstance(endpoints, list):
            endpoints = torch.cat(endpoints, dim=0)
        torch.manual_seed(20261113)
        torch.cuda.manual_seed(20261113)
        model = load_local_model("edm-ffhq-64x64-uncond-vp.pkl").cuda().eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x, t, *a, **k))
        ode_flow, _ = get_ode_flow(prior)
        template = endpoints[0:1].cuda()
        probes = [torch.randint_like(template, low=0, high=2).float() * 2.0 - 1.0
                  for _ in raw["records"]]
        original_randint = torch.randint_like
        rows = []
        for index, record in enumerate(raw["records"]):
            epsilon = probes[index]

            def fixed_randint(tensor, low=0, high=2):
                return ((epsilon + 1.0) / 2.0).to(tensor.dtype)

            torch.randint_like = fixed_randint
            endpoint_index = index * len(record["runs"])
            endpoint = endpoints[endpoint_index:endpoint_index + 1].cuda()
            with torch.no_grad():
                q = float(likelihood_fn(endpoint, ode_flow,
                                        t_spans=[80.0, float(record["noise"])],
                                        rtol=1e-5, atol=1e-5, method="RK45").item())
            torch.randint_like = original_randint
            row = {"source": source, "record_index": index, "img": record["img"],
                   "noise": record["noise"], "original_q": record["runs"][0]["q"],
                   "b1_recomputed_q": q, "b1_minus_original": q - record["runs"][0]["q"],
                   "batched_q": recheck["records"][index]["replicates"][0]["q_values"][0]}
            row["batched_minus_b1"] = row["batched_q"] - q
            rows.append(row)
            print("[B1]", source, index, row, flush=True)
        output["sources"][source] = {
            "rows": rows,
            "max_abs_b1_minus_original": max(abs(row["b1_minus_original"]) for row in rows),
            "max_abs_batched_minus_b1": max(abs(row["batched_minus_b1"]) for row in rows)}
        del model
        torch.cuda.empty_cache()
    path = os.path.join(PROJ, "results", "score_profile_probe_b1.json")
    with open(path, "w") as handle:
        json.dump(output, handle, indent=1)
    print("[saved]", path, flush=True)


if __name__ == "__main__":
    main()
