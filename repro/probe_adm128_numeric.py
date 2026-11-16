"""DOP853 confirmation of the native-128 sign reversal; no new physical sweep.

Uses real BSDS image 0, Gaussian 1, fixed conditional label 0, and the same Rademacher
seed as the completed sweep. Existing default RK45 rows are reused unchanged; only DOP853
rtol=atol=1e-6 at sigma 0.05 and 0.2 is newly evaluated. No outcome threshold is imposed.
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
    with open(os.path.join(_PROJ, "results", "adm_imagenet128_cond_bsds.json")) as handle:
        completed = json.load(handle)
    defaults = {row["sigma"]: row for row in completed["rows"]
                if row["img"] == 0 and row["family"] == "gaussian" and row["theta"] == 1.0}
    if set(defaults) != {0.05, 0.2}:
        raise RuntimeError("Expected completed default rows at sigma 0.05/0.2")
    vp, model, metadata = load_adm("imagenet128_cond")
    sharp = torch.load(os.path.join(_PROJ, "data", "bsds300_real_test_128.pt"),
                       map_location="cpu")[:1].cuda()
    half, _ = physical_power("gaussian", 1.0, 128)
    blurred = torch.fft.irfft2(torch.fft.rfft2(sharp, dim=(-2, -1))*half.sqrt().cuda(),
                               s=(128, 128), dim=(-2, -1)).to(sharp.dtype)
    torch.manual_seed(20261111)
    epsilon = torch.randint_like(sharp, low=0, high=2).to(torch.float32)*2-1
    result = {"cfg": {**metadata, "source_default": "adm_imagenet128_cond_bsds.json",
                      "analysis_status": "diagnostic DOP853 confirmation"},
              "rows": []}
    output = os.path.join(_PROJ, "results", "adm128_numeric.json")
    if os.path.exists(output):
        with open(output) as handle:
            previous = json.load(handle)
        result["rows"] = [row for row in previous.get("rows", [])
                          if row.get("sigma") in (0.05, 0.2)]
    start = time.time()
    for t in (0.05, 0.2):
        if any(row["sigma"] == t for row in result["rows"]):
            continue
        qs = float(vp.potential(sharp, t, epsilon=epsilon, method="DOP853",
                                rtol=1e-6, atol=1e-6)[0])
        qb = float(vp.potential(blurred, t, epsilon=epsilon, method="DOP853",
                                rtol=1e-6, atol=1e-6)[0])
        row = {"sigma": t, "q_sharp": qs, "q_blur": qb, "lambda": qs-qb,
               "default_lambda": defaults[t]["lambda"],
               "default_minus_dop853": defaults[t]["lambda"]-(qs-qb),
               "elapsed": time.time()-start}
        result["rows"].append(row)
        with open(os.path.join(_PROJ, "results", "adm128_numeric.json"), "w") as handle:
            json.dump(result, handle, indent=1)
        print("[ADM128-numeric]", row, flush=True)
    print("[saved] adm128_numeric.json", flush=True)


if __name__ == "__main__":
    main()
