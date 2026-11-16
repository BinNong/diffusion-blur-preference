r"""Review-1 over-blur generalization to new strengths and kernel families.

Pre-registered 2026-11-10 before any result was produced. This is a diagnostic pilot,
not a universal pass/fail claim. Eight previously unused FFHQ pool images (40:48),
two high noise levels (0.172, 0.5), 240 HQS iterations and two starts are fixed.

Four settings:
  Gaussian theta*=1.0, grid 0:0.1:2.0;
  Gaussian theta*=2.0, grid 0:0.175:3.5;
  Defocus theta*=1.5, grid 0:0.15:3.0;
  Motion theta*=3.0, grid 0:0.25:5.0.

G0 guards: finite normalized kernels, symmetric Gaussian/defocus, 0<=g2<=1 within 1e-6.
G1: report each family's over-blur proportion and relative kernel error, plus whole-image
    cluster intervals. No minimum proportion was selected in advance.
G2: at each empirical argmin report Delta D and Delta q relative to the true-strength grid
    point. This is a numerical decomposition, not proof of a global inner minimum.

Run: ./venv2/bin/python repro/sweep_overblur_generalization.py --out overblur_generalization.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))
from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn
from putils.convolution import DefocusBlur, GaussianBlur, MotionBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
POOL = "/tmp/pool_ffhq64_256_s0.pt"
DEV = torch.device("cuda")
DT = torch.float32
N_INNER = 240
NIMG = 8
POOL_OFFSET = 40
NOISES = [0.172, 0.5]
SETTINGS = [
    ("gaussian", 1.0, [round(0.1*i, 4) for i in range(21)]),
    ("gaussian", 2.0, [round(0.175*i, 4) for i in range(21)]),
    ("defocus", 1.5, [round(0.15*i, 4) for i in range(21)]),
    ("motion", 3.0, [round(0.25*i, 4) for i in range(21)]),
]


class Blur:
    def __init__(self, family):
        cls = {"gaussian": GaussianBlur, "defocus": DefocusBlur,
               "motion": MotionBlur}[family]
        self.operator = cls(kernel_size=15, device=DEV, dtype=DT)

    def kernel(self, theta):
        return self.operator.dirac if theta == 0 else self.operator.kernel(theta)

    def transform(self, kernel, height, width):
        pad = torch.zeros((1, 1, height, width), device=DEV, dtype=DT)
        kh, kw = kernel.shape[-2:]
        top, left = (height-kh)//2, (width-kw)//2
        pad[:, :, top:top+kh, left:left+kw] = kernel
        return torch.fft.rfft2(pad, dim=(-2, -1))

    def A(self, x, theta):
        k = self.transform(self.kernel(theta), x.shape[-2], x.shape[-1])
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))*k,
                                s=x.shape[-2:], dim=(-2, -1)).to(DT)

    def AT(self, x, theta):
        k = self.transform(self.kernel(theta), x.shape[-2], x.shape[-1])
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))*k.conj(),
                                s=x.shape[-2:], dim=(-2, -1)).to(DT)

    def eig(self, theta, height, width):
        return self.transform(self.kernel(theta), height, width).abs().square()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="overblur_generalization.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.out == "overblur_generalization.json":
        args.out = "smoke_overblur_generalization.json"
    output = os.path.join(_PROJ, "results", args.out)
    result = {"cfg": {"checkpoint": CKPT, "pool": POOL, "pool_offset": POOL_OFFSET,
                      "nimg": NIMG, "noises": NOISES, "n_inner": N_INNER,
                      "settings": [(f, t, g) for f, t, g in SETTINGS],
                      "preregistered": "2026-11-10"}, "records": []}

    def save():
        with open(output, "w") as handle:
            json.dump(result, handle, indent=1)

    torch.manual_seed(20261110)
    torch.cuda.manual_seed(20261110)
    model = load_local_model(CKPT).to(DEV).eval()
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x.to(DT), t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)
    pool = torch.load(POOL, map_location=DEV).to(DT)
    images = pool[POOL_OFFSET:POOL_OFFSET+NIMG]

    @torch.no_grad()
    def hqs(blur, theta, y, noise, initial):
        aty = blur.AT(y, theta)
        denominator = blur.eig(theta, y.shape[-2], y.shape[-1])+1
        z = initial.clone()
        for _ in range(N_INNER):
            denoised = model(z, noise).to(DT)
            v = 2*denoised-z
            x = torch.fft.irfft2(torch.fft.rfft2(aty+v, dim=(-2, -1))/denominator,
                                 s=y.shape[-2:], dim=(-2, -1))
            z = torch.clamp(z+x-denoised, -1, 1)
        return model(z, noise).to(DT)

    @torch.no_grad()
    def q_value(x, noise):
        return float(likelihood_fn(x, ode_flow, t_spans=[80.0, float(noise)]).item())

    start = time.time()
    settings = SETTINGS[:1] if args.smoke else SETTINGS
    noises = NOISES[:1] if args.smoke else NOISES
    image_indices = range(1) if args.smoke else range(NIMG)
    for family, truth, full_grid in settings:
        blur = Blur(family)
        kernel = blur.kernel(truth)
        eigenvalues = blur.eig(truth, 64, 64)
        symmetry = float((kernel[0, 0]-kernel[0, 0].flip(0).flip(1)).abs().max())
        guard = {"kernel_sum": float(kernel.sum()), "symmetry": symmetry,
                 "g2_min": float(eigenvalues.min()), "g2_max": float(eigenvalues.max())}
        if (abs(guard["kernel_sum"]-1) > 1e-5 or guard["g2_min"] < -1e-6
                or guard["g2_max"] > 1+1e-6
                or (family in {"gaussian", "defocus"} and symmetry > 1e-6)):
            raise RuntimeError("G0 guard failed: %s %s" % (family, guard))
        grid = [0.0, truth, full_grid[-1]] if args.smoke else full_grid
        print("[G0]", family, truth, guard, flush=True)
        for i in image_indices:
            sharp = images[i:i+1]
            clean = blur.A(sharp, truth)
            for noise in noises:
                generator = torch.Generator(device=DEV).manual_seed(
                    1000*(POOL_OFFSET+i)+int(noise*100000))
                y = (clean+noise*torch.randn(clean.shape, generator=generator,
                                             device=DEV, dtype=DT)).to(DT)
                profile = []
                for theta in grid:
                    best = None
                    for initial in (y, blur.A(y, theta)):
                        x = hqs(blur, theta, y, noise, initial)
                        residual = blur.A(x, theta)-y
                        data = float(residual.square().sum()/(2*noise**2))
                        q = q_value(x, noise)
                        if best is None or data+q < best["L"]:
                            best = {"theta": theta, "L": data+q, "data": data, "q": q}
                        del x
                        torch.cuda.empty_cache()
                    profile.append(best)
                record = {"family": family, "img": POOL_OFFSET+i, "noise": noise,
                          "theta_star": truth, "profile": profile}
                result["records"].append(record)
                save()
                optimum = min(profile, key=lambda point: point["L"])
                print("[G1]", family, "img", POOL_OFFSET+i, "noise", noise,
                      "theta_opt", optimum["theta"], "truth", truth,
                      "elapsed", round(time.time()-start), flush=True)
    print("[saved]", output, "elapsed", time.time()-start, flush=True)


if __name__ == "__main__":
    main()
