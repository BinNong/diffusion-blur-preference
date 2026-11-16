r"""Review-response real-image blur-preference sweep on BSDS300 (EDM/FFHQ).

Pre-registered 2026-11-11 before any result was produced. This is a diagnostic
external-validity experiment, not a universal sign criterion.

R0: input must be the prepared 100-image BSDS300 test tensor, not an EDM sample pool.
    All pixels finite and in [-1,1]. Exact terminal terms are in [0,0.96] nats.
R1: for the first 32 sorted test images, measure paired Lambda for eight raw reference
    kernels at t=0.002. Use Gaussian 1/2, Defocus 1/2, Airy 4, Motion 3/7, ideal low-pass.
R2: repeat four representative kernels (Gaussian 1/2, Defocus 1, Motion 3) at t=0.2.
    Report image-wise mean/SD/SE, sign proportions and terminal terms. No outcome threshold.
R3: use default RK45 rtol=atol=1e-5 and a separate Hutchinson draw for each density;
    do not assume stochastic errors cancel. The old 32-repeat numerical audit remains limited
    to its original generated image; no universal precision claim is made.

Smoke uses one image and only Gaussian 1/2 at both t, writing a separate output.
Run: ./venv2/bin/python repro/sweep_real_bsds.py --out real_bsds_edm.json
"""

import argparse
import json
import os
import sys
import time

import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "repro"))
from review_numeric_checks import REFS, apply_power, kernel_power
from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
POOL = "data/bsds300_real_test_64.pt"
SIGMA_MAX = 80.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="real_bsds_edm.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.out == "real_bsds_edm.json":
        args.out = "smoke_real_bsds_edm.json"
    output = os.path.join(_PROJ, "results", args.out)
    pool_path = os.path.join(_PROJ, POOL)
    images = torch.load(pool_path, map_location="cpu")
    if images.shape != (100, 3, 64, 64) or not torch.isfinite(images).all():
        raise RuntimeError("R0 invalid real-image pool")
    if float(images.min()) < -1 or float(images.max()) > 1:
        raise RuntimeError("R0 invalid pixel range")
    with open(os.path.join(_PROJ, "results", "bsds300_real_manifest.json")) as handle:
        provenance = json.load(handle)
    if provenance.get("generated") is not False or provenance.get("n") != 100:
        raise RuntimeError("R0 provenance guard failed")

    selected = [("gaussian", 1.0), ("gaussian", 2.0)] if args.smoke else REFS
    configurations = []
    for t in (0.002, 0.2):
        for family, theta in selected:
            if t == 0.2 and (family, theta) not in {
                    ("gaussian", 1.0), ("gaussian", 2.0),
                    ("defocus", 1.0), ("motion", 3.0)}:
                continue
            g2 = kernel_power(family, theta).to(torch.float64)[0, 0].cpu()
            g2_full = kernel_power(family, theta, full=True).to(torch.float64)[0, 0].cpu()
            configurations.append({"t": t, "family": family, "theta": theta,
                                   "g2": g2, "g2_full": g2_full,
                                   "u_rfft": float(g2.mean()),
                                   "u_full": float(g2_full.mean())})
    operators = [{key: value for key, value in cfg.items() if not key.startswith("g2")}
                 for cfg in configurations]
    result = {"cfg": {"checkpoint": CKPT, "pool": POOL, "generated": False,
                      "nimg": 1 if args.smoke else 32, "operators": operators,
                      "rtol": 1e-5, "atol": 1e-5, "method": "RK45",
                      "preregistered": "2026-11-11", "smoke": args.smoke},
              "rows": []}
    with open(output, "w") as handle:
        json.dump(result, handle, indent=1)

    torch.manual_seed(20261111)
    torch.cuda.manual_seed(20261111)
    model = load_local_model(CKPT).cuda().eval()
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x, t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)

    @torch.no_grad()
    def q_value(image, t):
        return float(likelihood_fn(image, ode_flow, t_spans=[SIGMA_MAX, float(t)],
                                   rtol=1e-5, atol=1e-5, method="RK45").item())

    start = time.time()
    for index in range(result["cfg"]["nimg"]):
        sharp = images[index:index+1].cuda()
        norm_sharp = sharp.to(torch.float64).square().sum()
        for cfg in configurations:
            blurred = apply_power(sharp, cfg["g2"].cuda()).to(torch.float32)
            norm_blurred = blurred.to(torch.float64).square().sum()
            terminal = float((norm_sharp-norm_blurred)/(2*SIGMA_MAX**2))
            if not 0 <= terminal <= 0.96+1e-7:
                raise RuntimeError("R0 terminal guard failed")
            q_sharp = q_value(sharp, cfg["t"])
            q_blur = q_value(blurred, cfg["t"])
            row = {"img": index, "source_filename": provenance["records"][index]["filename"],
                   **{key: value for key, value in cfg.items() if not key.startswith("g2")},
                   "lambda": q_sharp-q_blur, "q_sharp": q_sharp, "q_blur": q_blur,
                   "terminal": terminal}
            result["rows"].append(row)
            with open(output, "w") as handle:
                json.dump(result, handle, indent=1)
            print("[R1/R2]", index, cfg["family"], cfg["theta"], cfg["t"],
                  "Lambda", row["lambda"], "elapsed", round(time.time()-start), flush=True)
            del blurred
            torch.cuda.empty_cache()
        del sharp
        torch.cuda.empty_cache()
    print("[saved]", output, "rows", len(result["rows"]), flush=True)


if __name__ == "__main__":
    main()
