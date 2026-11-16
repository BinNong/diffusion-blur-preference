r"""Non-EDM and native-128 review pilots on real BSDS300 images.

Pre-registered 2026-11-11 before any learned-ADM result. No sign or magnitude pass criterion.

A0: analytic Gaussian VP-ODE potential must agree with the exact density to <=1e-3 nats.
A1: official checkpoint must load with strict=True. The 64x64 arm is unconditional
    improved-diffusion ImageNet; the native 128x128 arm is guided-diffusion ImageNet,
    evaluated as a fixed-label (0) conditional prior without classifier guidance. The latter
    is not an unconditional-prior claim. Both use the documented continuous VP-ODE surrogate,
    not a discrete DDPM variational bound.
A2: 64x64: first 8 BSDS test images, Gaussian 1/2, Defocus 1, Motion 3,
    physical sigma in {0.005,0.05,0.2}. Native 128x128: first 4 images, same kernels,
    sigma in {0.05,0.2}. These endpoints respect the discrete checkpoints' coverage.
    One Rademacher vector is shared within each sharp/blur pair, not across all pairs.
    Report every paired value, mean/SD/SE and sign proportion; no universal claim.
A3: smoke runs one image, Gaussian 1, sigma=0.2, only to test model and memory paths.
    Save every completed row so interruption cannot invalidate prior rows.

Run: ./venv2/bin/python repro/sweep_adm_bsds.py --model imagenet64_uncond
     ./venv2/bin/python repro/sweep_adm_bsds.py --model imagenet128_cond
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "improved-diffusion"))
sys.path.insert(0, os.path.join(_PROJ, "code", "guided-diffusion"))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))
sys.path.insert(0, os.path.join(_PROJ, "repro"))
from adm_prior import analytic_guard, load_adm
from putils.convolution import DefocusBlur, GaussianBlur, MotionBlur


def physical_power(family, theta, size):
    cls = {"gaussian": GaussianBlur, "defocus": DefocusBlur, "motion": MotionBlur}[family]
    kernel = cls(kernel_size=15, device="cpu", dtype=torch.float64).kernel(theta)
    if kernel.ndim == 2:
        kernel = kernel[None, None]
    padded = torch.zeros((1, 1, size, size), dtype=torch.float64)
    padded[:, :, (size-15)//2:(size-15)//2+15, (size-15)//2:(size-15)//2+15] = kernel
    half = torch.fft.rfft2(padded, dim=(-2, -1)).abs().square()
    full = torch.fft.fft2(padded, dim=(-2, -1)).abs().square()
    return half, full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["imagenet64_uncond", "imagenet128_cond"], required=True)
    parser.add_argument("--out")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    size = 64 if args.model == "imagenet64_uncond" else 128
    default_out = "adm_%s_bsds.json" % args.model
    output_name = args.out or default_out
    if args.smoke and not args.out:
        output_name = "smoke_"+default_out
    output = os.path.join(_PROJ, "results", output_name)
    analytic = analytic_guard()
    if analytic[2] > 1e-3:
        raise RuntimeError("A0 analytic VP guard failed: %r" % (analytic,))
    vp, model, metadata = load_adm(args.model)
    images = torch.load(os.path.join(_PROJ, "data", "bsds300_real_test_%d.pt" % size),
                        map_location="cpu")
    operators = [("gaussian", 1.0), ("gaussian", 2.0), ("defocus", 1.0), ("motion", 3.0)]
    ts = [0.005, 0.05, 0.2] if size == 64 else [0.05, 0.2]
    if args.smoke:
        operators, ts = operators[:1], [0.2]
    nimg = 1 if args.smoke else (8 if size == 64 else 4)
    result = {"cfg": {**metadata, "analytic_guard": {"numerical": analytic[0],
                      "exact": analytic[1], "absolute_error": analytic[2]},
                      "pool": "BSDS300 official test", "generated": False,
                      "nimg": nimg, "image_size": size, "operators": operators,
                      "target_sigmas": ts, "rtol": 1e-5, "atol": 1e-5,
                      "solver": "RK45", "preregistered": "2026-11-11",
                      "smoke": args.smoke}, "rows": []}
    with open(output, "w") as handle:
        json.dump(result, handle, indent=1)
    start = time.time()
    for index in range(nimg):
        sharp = images[index:index+1].cuda()
        for family, theta in operators:
            half, full = physical_power(family, theta, size)
            blurred = torch.fft.irfft2(torch.fft.rfft2(sharp, dim=(-2, -1))*half.sqrt().cuda(),
                                       s=(size, size), dim=(-2, -1)).to(sharp.dtype)
            for t in ts:
                tau = vp.tau_for_sigma(t)
                torch.manual_seed(20261111+index)
                epsilon = torch.randint_like(sharp, low=0, high=2).to(torch.float32)*2-1
                q_sharp = float(vp.potential(sharp, t, epsilon=epsilon)[0])
                q_blur = float(vp.potential(blurred, t, epsilon=epsilon)[0])
                row = {"img": index, "family": family, "theta": theta,
                       "sigma": t, "tau": tau, "u_rfft": float(half.mean()),
                       "u_full": float(full.mean()), "lambda": q_sharp-q_blur,
                       "q_sharp": q_sharp, "q_blur": q_blur}
                result["rows"].append(row)
                with open(output, "w") as handle:
                    json.dump(result, handle, indent=1)
                print("[A2]", args.model, index, family, theta, t,
                      "Lambda", row["lambda"], "elapsed", round(time.time()-start), flush=True)
                torch.cuda.empty_cache()
        del sharp
        torch.cuda.empty_cache()
    print("[saved]", output, "rows", len(result["rows"]), flush=True)


if __name__ == "__main__":
    main()
