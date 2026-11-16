"""Numerical robustness audit for the learned ADM VP-ODE, before full sweeps.

One real BSDS test image and Gaussian 1, shared Rademacher within each sharp/blur pair.
Compare default RK45 1e-5 at sigma {0.005,0.05,0.2} with DOP853 1e-6 at sigma 0.2.
Report both sides and all differences; no pass threshold is selected after seeing results.
"""

import json
import os
import sys
import time

import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "repro"))
from adm_prior import load_adm
from sweep_adm_bsds import physical_power


def main():
    vp, model, metadata = load_adm("imagenet64_uncond")
    sharp = torch.load(os.path.join(_PROJ, "data", "bsds300_real_test_64.pt"),
                       map_location="cpu")[:1].cuda()
    half, _ = physical_power("gaussian", 1.0, 64)
    blurred = torch.fft.irfft2(torch.fft.rfft2(sharp, dim=(-2, -1))*half.sqrt().cuda(),
                               s=(64, 64), dim=(-2, -1)).to(sharp.dtype)
    torch.manual_seed(20261111)
    epsilon = torch.randint_like(sharp, low=0, high=2).to(torch.float32)*2-1
    result = {"cfg": {**metadata, "input": "BSDS300 test image 0", "kernel": "Gaussian 1",
                      "shared_rademacher": True, "analysis_status": "diagnostic before full sweep"},
              "rows": []}
    start = time.time()
    settings = [(t, "RK45", 1e-5) for t in (0.005, 0.05, 0.2)]
    settings += [(0.2, "DOP853", 1e-6)]
    for t, method, tolerance in settings:
        qs = float(vp.potential(sharp, t, epsilon=epsilon, method=method,
                                rtol=tolerance, atol=tolerance)[0])
        qb = float(vp.potential(blurred, t, epsilon=epsilon, method=method,
                                rtol=tolerance, atol=tolerance)[0])
        row = {"sigma": t, "method": method, "tolerance": tolerance,
               "q_sharp": qs, "q_blur": qb, "lambda": qs-qb,
               "elapsed": time.time()-start}
        result["rows"].append(row)
        with open(os.path.join(_PROJ, "results", "adm_numeric.json"), "w") as handle:
            json.dump(result, handle, indent=1)
        print("[numeric]", row, flush=True)
    reference = result["rows"][-1]["lambda"]
    result["default_minus_dop853_at_02"] = result["rows"][-2]["lambda"]-reference
    with open(os.path.join(_PROJ, "results", "adm_numeric.json"), "w") as handle:
        json.dump(result, handle, indent=1)
    print("[saved] adm_numeric.json", flush=True)


if __name__ == "__main__":
    main()
