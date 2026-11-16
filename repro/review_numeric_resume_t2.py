"""Resume only the t=0.2 ODE block after the 2026-11-10 SSH reset.

The completed terminal and Hutchinson blocks are preserved. All six t=0.2 settings are
recomputed with one new shared Rademacher vector, replacing the incomplete block so solver
comparisons never mix trace vectors.
"""

import json
import os
import sys
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "repro"))
from review_numeric_checks import (CKPT, POOL, DEV, DT, SIGMA_MAX,
                                   apply_power, kernel_power)
from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn


def main():
    path = os.path.join(_PROJ, "results", "review_numeric.json")
    with open(path) as handle:
        result = json.load(handle)
    if not result.get("terminal") or len(result.get("hutchinson", {})) != 2:
        raise RuntimeError("Completed terminal and Hutchinson blocks are required")

    torch.manual_seed(20261111)
    torch.cuda.manual_seed(20261111)
    model = load_local_model(CKPT).to(DEV).eval()
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x.to(DT), t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)
    pool = torch.load(POOL, map_location=DEV).to(DT)
    sharp = pool[255:256].to(DT)
    blurred = apply_power(sharp.to(torch.float64), kernel_power("gaussian", 1.5)).to(DT)
    epsilon = torch.randint_like(sharp, low=0, high=2).to(DT)*2-1
    original_randint = torch.randint_like

    def fixed_randint_like(tensor, low=0, high=2):
        return ((epsilon+1.0)/2.0).to(tensor.dtype)

    @torch.no_grad()
    def q_value(z, tolerance, method):
        return float(likelihood_fn(z, ode_flow, t_spans=[SIGMA_MAX, 0.2],
                                   rtol=tolerance, atol=tolerance, method=method).item())

    settings = [("RK45", tol) for tol in [1e-3, 1e-4, 1e-5, 1e-6]]
    settings += [("DOP853", tol) for tol in [1e-5, 1e-6]]
    rows = []
    try:
        torch.randint_like = fixed_randint_like
        for method, tolerance in settings:
            q_sharp = q_value(sharp, tolerance, method)
            q_blur = q_value(blurred, tolerance, method)
            rows.append({"method": method, "rtol": tolerance, "atol": tolerance,
                         "q_sharp": q_sharp, "q_blur": q_blur,
                         "lambda": q_sharp-q_blur})
            print("[N2-resume]", method, tolerance, rows[-1]["lambda"], flush=True)
    finally:
        torch.randint_like = original_randint
    reference = next(row["lambda"] for row in rows
                     if row["method"] == "DOP853" and row["rtol"] == 1e-6)
    for row in rows:
        row["difference_from_reference"] = row["lambda"]-reference
    result["ode"]["0.2"] = {"reference": "DOP853, rtol=atol=1e-6", "rows": rows,
                              "resume_seed": 20261111,
                              "resume_reason": "SSH reset; whole t=0.2 block recomputed"}
    with open(path, "w") as handle:
        json.dump(result, handle, indent=1)
    print("[saved]", path, flush=True)


if __name__ == "__main__":
    main()
