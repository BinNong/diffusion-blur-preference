r"""Real-BSDS profile-MAP pilot for review comment 6 (EDM/FFHQ).

Pre-registered 2026-11-11 before results. Diagnostic only; no success threshold.

P0: input is the official BSDS300 test pool, not an EDM-generated pool. Correct symmetric
    15x15 Gaussian from the library, theta*=1.5, 21-point grid 0:0.125:2.5.
P1: first 8 sorted test images, noise levels {0.172,0.5}. At each grid point use 240
    plug-and-play HQS iterations, FFT data step, clipping [-1,1], and two starts (y and
    h_theta*y), retaining the lower L=D+q_sigma. The argmin is numerical, not certified global.
P2: save every L/D/q value, empirical argmin, true-point differences, and top-five spread.
    Report over/under/equal counts and kernel errors without causal or universal claims.

Run: ./venv2/bin/python repro/sweep_real_profile.py --out real_bsds_profile.json
"""

import argparse
import json
import os
import sys
import time

import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "repro"))
from sweep_overblur_generalization import Blur
from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
N_INNER = 240
NOISES = [0.172, 0.5]
GRID = [round(0.125*i, 4) for i in range(21)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="real_bsds_profile.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.out == "real_bsds_profile.json":
        args.out = "smoke_real_bsds_profile.json"
    images = torch.load(os.path.join(_PROJ, "data", "bsds300_real_test_64.pt"),
                        map_location="cpu")
    if images.shape != (100, 3, 64, 64):
        raise RuntimeError("P0 invalid real pool")
    blur = Blur("gaussian")
    kernel = blur.kernel(1.5)
    symmetry = float((kernel[0, 0]-kernel[0, 0].flip(0).flip(1)).abs().max())
    if symmetry > 1e-6:
        raise RuntimeError("P0 kernel symmetry failed")
    torch.manual_seed(20261111)
    torch.cuda.manual_seed(20261111)
    model = load_local_model(CKPT).cuda().eval()
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x, t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)

    def hqs(theta, y, noise, initial):
        aty = blur.AT(y, theta)
        denominator = blur.eig(theta, 64, 64)+1
        z = initial.clone()
        for _ in range(N_INNER):
            denoised = model(z, noise)
            v = 2*denoised-z
            x = torch.fft.irfft2(torch.fft.rfft2(aty+v, dim=(-2, -1))/denominator,
                                 s=(64, 64), dim=(-2, -1))
            z = torch.clamp(z+x-denoised, -1, 1)
        return model(z, noise)

    @torch.no_grad()
    def q_value(x, noise):
        return float(likelihood_fn(x, ode_flow, t_spans=[80.0, float(noise)]).item())

    output = os.path.join(_PROJ, "results", args.out)
    result = {"cfg": {"checkpoint": CKPT, "pool": "BSDS300 official test",
                      "generated": False, "nimg": 1 if args.smoke else 8,
                      "noises": NOISES[:1] if args.smoke else NOISES,
                      "theta_star": 1.5, "kernel_size": 15, "grid": GRID,
                      "n_inner": N_INNER, "starts": ["y", "h_theta*y"],
                      "kernel_symmetry": symmetry, "preregistered": "2026-11-11",
                      "smoke": args.smoke}, "records": []}
    with open(output, "w") as handle:
        json.dump(result, handle, indent=1)
    start = time.time()
    for index in range(result["cfg"]["nimg"]):
        sharp = images[index:index+1].cuda()
        clean = blur.A(sharp, 1.5)
        for noise in result["cfg"]["noises"]:
            generator = torch.Generator(device="cuda").manual_seed(
                20261111+1000*index+int(noise*100000))
            y = clean+noise*torch.randn(clean.shape, generator=generator, device="cuda")
            profile = []
            grid = [0.0, 1.5, 2.5] if args.smoke else GRID
            for theta in grid:
                best = None
                for initial in (y, blur.A(y, theta)):
                    x = hqs(theta, y, noise, initial)
                    residual = blur.A(x, theta)-y
                    data = float(residual.square().sum()/(2*noise**2))
                    q = q_value(x, noise)
                    if best is None or data+q < best["L"]:
                        best = {"theta": theta, "L": data+q, "data": data, "q": q}
                    del x
                    torch.cuda.empty_cache()
                profile.append(best)
                with open(output, "w") as handle:
                    json.dump(result, handle, indent=1)
            optimum = min(profile, key=lambda point: point["L"])
            record = {"img": index, "noise": noise, "theta_star": 1.5, "profile": profile}
            result["records"].append(record)
            with open(output, "w") as handle:
                json.dump(result, handle, indent=1)
            print("[P2]", index, noise, "theta_opt", optimum["theta"],
                  "elapsed", round(time.time()-start), flush=True)
        del sharp
        torch.cuda.empty_cache()
    print("[saved]", output, "records", len(result["records"]), flush=True)


if __name__ == "__main__":
    main()
