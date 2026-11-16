r"""Shared-probe Hutchinson and ODE-tolerance audit of the scaling curve.

Pre-registered for review2 on 2026-11-13 before any result from this script.
L0: reconstruct exactly the 20 common W2/W8/TK50 operators used by the N=120 dense files.
    Use generated-pool images 0:4, t=0.002 and the released EDM/FFHQ checkpoint.
L1: four independent Rademacher replicates. Within one replicate, share the same probe across
    sharp and every blurred image and every u; across replicates draw independent probes.
    Evaluate four images as one batch. Save all q_sharp and q_blur values and the probe vectors.
L2: default RK45 rtol=atol=1e-5 for four replicates. Repeat the complete curve for replicate 0
    with the SAME probe and RK45 rtol=atol=1e-6. This tests fit stability under integration
    tolerance, not every solver or image. Existing DOP853 checks remain separately scoped.
L3: no outcome threshold. Offline analysis must report replicate-to-replicate b(u),
    b_inf/A/q variability, and default-minus-stricter-tolerance fit differences. Do not treat
    image bootstrap, probe variance and solver error as interchangeable.

Run: ./venv2/bin/python repro/sweep_law_stability.py --out law_stability.json
Smoke uses one image, one family, three u and one replicate; never report it as science.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "code", "blind_deblur_diffusion"))
sys.path.insert(0, os.path.join(PROJ, "repro"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn
from analyze_review2 import full_coordinate

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
FAMILIES = ("W2", "W8", "TK50")


def operators():
    result = {family: {} for family in FAMILIES}
    for filename in ("sweep_bdense_n120.json", "sweep_blow_n120.json"):
        with open(os.path.join(PROJ, "results", filename)) as handle:
            rows = json.load(handle)["rows"]
        for record in rows:
            family = record["meta"]["fid"]
            if family not in result:
                continue
            key = round(1.0 - float(record["kunif"]), 4)
            _, power = full_coordinate(record)
            result[family].setdefault(key, power)
    for family in FAMILIES:
        ordered = sorted(result[family].items(), reverse=True)
        if len(ordered) != 20:
            raise RuntimeError("L0 expected 20 unique operators")
        result[family] = [(key, value) for key, value in ordered]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="law_stability.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and not os.path.basename(args.out).startswith("smoke_"):
        args.out = "smoke_" + os.path.basename(args.out)
    output = os.path.join(PROJ, "results", args.out)
    all_operators = operators()
    if args.smoke:
        all_operators = {"W2": all_operators["W2"][:3]}
    n_images = 1 if args.smoke else 4
    replicates = 1 if args.smoke else 4
    settings = [(repeat, 1e-5) for repeat in range(replicates)] + [(0, 1e-6)]
    config = {"checkpoint": CKPT, "pool": "/tmp/pool_ffhq64_256_s0.pt",
              "images": list(range(n_images)), "t": 0.002, "families": list(all_operators),
              "replicates": replicates, "settings": settings, "trace_seed": 20261113,
              "method": "RK45", "preregistered": "2026-11-13", "smoke": args.smoke}
    with open(__file__, "rb") as handle:
        config["script_sha256"] = hashlib.sha256(handle.read()).hexdigest()
    if os.path.exists(output):
        with open(output) as handle:
            result = json.load(handle)
        if result["cfg"] != config:
            raise RuntimeError("Existing output does not match the frozen configuration")
    else:
        result = {"cfg": config, "trace_vectors": [], "rows": []}

    def save():
        with open(output, "w") as handle:
            json.dump(result, handle, indent=1)

    torch.manual_seed(config["trace_seed"])
    torch.cuda.manual_seed(config["trace_seed"])
    model = load_local_model(CKPT).cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x, t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)
    images = torch.load(config["pool"], map_location="cuda")[:n_images].float()
    vectors = [torch.randint_like(images, low=0, high=2).float() * 2.0 - 1.0
               for _ in range(replicates)]
    encoded = [vector.reshape(n_images, -1).to(torch.int8).cpu().tolist() for vector in vectors]
    if result["trace_vectors"] and result["trace_vectors"] != encoded:
        raise RuntimeError("L1 saved probe mismatch")
    result["trace_vectors"] = encoded
    save()
    original_randint = torch.randint_like
    start = time.time()

    for family, family_operators in all_operators.items():
        for repeat, tolerance in settings:
            matches = [row for row in result["rows"] if row["family"] == family
                       and row["replicate"] == repeat and row["rtol"] == tolerance]
            if matches and len(matches[0]["operators"]) == len(family_operators):
                continue
            epsilon = vectors[repeat]

            def fixed_randint(tensor, low=0, high=2):
                return ((epsilon + 1.0) / 2.0).to(tensor.dtype)

            @torch.no_grad()
            def q_values(image):
                torch.randint_like = fixed_randint
                try:
                    values = likelihood_fn(image, ode_flow, t_spans=[80.0, 0.002],
                                           rtol=tolerance, atol=tolerance, method="RK45")
                    return [float(value) for value in values.reshape(-1)]
                finally:
                    torch.randint_like = original_randint

            if matches:
                row = matches[0]
            else:
                row = {"family": family, "replicate": repeat, "method": "RK45",
                       "rtol": tolerance, "atol": tolerance, "q_sharp": q_values(images),
                       "operators": []}
                result["rows"].append(row)
                save()
            for u_rfft, power in family_operators[len(row["operators"]):]:
                gate = torch.from_numpy(power).cuda()[None, None]
                blurred = torch.fft.irfft2(torch.fft.rfft2(images.double(), dim=(-2, -1))
                                           * gate.sqrt(), s=(64, 64), dim=(-2, -1)).float()
                q_blur = q_values(blurred)
                q_sharp = row["q_sharp"]
                norm_sharp = images.double().square().sum(dim=(1, 2, 3))
                norm_blurred = blurred.double().square().sum(dim=(1, 2, 3))
                terminal = ((norm_sharp - norm_blurred) / (2.0 * 80.0 ** 2)).cpu().tolist()
                if min(terminal) < -1e-7 or max(terminal) > 0.96 + 1e-7:
                    raise RuntimeError("L0 terminal guard failed")
                row["operators"].append({"u_rfft": u_rfft, "q_blur": q_blur,
                                         "lambda": [sharp - blur for sharp, blur
                                                    in zip(q_sharp, q_blur)],
                                         "terminal": terminal})
                save()
                print("[L1/L2]", family, repeat, tolerance, len(row["operators"]), "/",
                      len(family_operators), "elapsed", round(time.time() - start), flush=True)
                del blurred, gate
                torch.cuda.empty_cache()
    print("[saved]", output, "rows", len(result["rows"]), flush=True)


if __name__ == "__main__":
    main()
