r"""Review-1 numerical audit: terminal terms, 32 Hutchinson repeats, ODE convergence.

Pre-registered 2026-11-10 before running this script. The three checks are diagnostic;
no pass threshold was chosen after seeing results.

N0: for 120 FFHQ pool images and eight raw reference kernels, compute the exact terminal
term T=(||x||^2-||h*x||^2)/(2*80^2), the retired approximation T_u=(1-u)||x||^2/(2*80^2),
and their difference. Guard: every T must lie in [0,0.96], and the FFT energy identity must
match the Parseval sum to relative error <=1e-8.

N1: at pool[255], Gaussian sigma=1.5, t in {0.002,0.2}, draw 32 independent sharp/blur
likelihood pairs. Report the mean, sample SD and standard error of Lambda. This changes the
old four-repeat pilot into a 32-repeat measurement; it is not a universal error bound.

N2: with one fixed Rademacher vector shared by all evaluations of a pair, scan RK45
rtol=atol in {1e-3,1e-4,1e-5,1e-6} and DOP853 in {1e-5,1e-6}. Report Lambda differences
against DOP853 1e-6. Because likelihood_fn uses scipy.solve_ivp, this is a tolerance and
solver check, not a claimed fixed-step scan.

Run: ./venv2/bin/python repro/review_numeric_checks.py --out review_numeric.json
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn
from putils.convolution import AiryDisc, DefocusBlur, GaussianBlur, MotionBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
POOL = "/tmp/pool_ffhq64_256_s0.pt"
DEV = torch.device("cuda")
DT = torch.float32
NIMG = 120
K_REP = 32
SIGMA_MAX = 80.0
TS = [0.002, 0.2]
REFS = [("gaussian", 1.0), ("gaussian", 2.0), ("defocus", 1.0),
        ("defocus", 2.0), ("airy", 4.0), ("motion", 3.0),
        ("motion", 7.0), ("lowpass", 0.25)]


def apply_power(x, g2):
    y = torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1)) * g2.sqrt(),
                         s=x.shape[-2:], dim=(-2, -1))
    return y


def kernel_power(family, theta, full=False):
    if family == "lowpass":
        freq = np.fft.fftfreq(64) * 64
        fx, fy = np.meshgrid(freq, freq, indexing="ij")
        radius = np.sqrt(fx ** 2 + fy ** 2)
        spectrum = radius if full else radius[:, :33]
        g2 = (spectrum <= theta * radius.max()).astype(np.float64)
        return torch.from_numpy(g2).to(DEV)[None, None]
    cls = {"gaussian": GaussianBlur, "defocus": DefocusBlur,
           "airy": AiryDisc, "motion": MotionBlur}[family]
    kernel = cls(kernel_size=15, device=DEV, dtype=torch.float64).kernel(float(theta))
    if kernel.ndim == 2:
        kernel = kernel[None, None]
    pad = torch.zeros((1, 1, 64, 64), device=DEV, dtype=torch.float64)
    kh, kw = kernel.shape[-2:]
    pad[:, :, (64-kh)//2:(64-kh)//2+kh, (64-kw)//2:(64-kw)//2+kw] = kernel
    transform = torch.fft.fft2 if full else torch.fft.rfft2
    return transform(pad, dim=(-2, -1)).abs().square()


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": int(values.size), "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)),
            "se": float(values.std(ddof=1) / math.sqrt(values.size)),
            "min": float(values.min()), "median": float(np.median(values)),
            "max": float(values.max())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="review_numeric.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = os.path.join(_PROJ, "results", args.out)
    result = {"cfg": {"checkpoint": CKPT, "pool": POOL, "nimg": NIMG,
                      "hutchinson_repeats": K_REP, "ts": TS,
                      "sigma_max": SIGMA_MAX, "preregistered": "2026-11-10"},
              "terminal": {}, "hutchinson": {}, "ode": {}}

    def save():
        with open(output, "w") as handle:
            json.dump(result, handle, indent=1)

    start = time.time()
    pool = torch.load(POOL, map_location=DEV).to(DT)
    images = pool[:NIMG].to(torch.float64)
    terminal_rows = []
    for family, theta in REFS:
        g2 = kernel_power(family, theta)
        blurred = apply_power(images, g2)
        norm_x = images.square().sum(dim=(1, 2, 3))
        norm_y = blurred.square().sum(dim=(1, 2, 3))
        exact = (norm_x - norm_y) / (2 * SIGMA_MAX ** 2)
        u = float(g2.mean())
        approximate = (1-u) * norm_x / (2 * SIGMA_MAX ** 2)
        spectrum_x = torch.fft.fft2(images, dim=(-2, -1)).abs().square()
        g2_full = kernel_power(family, theta, full=True)
        parseval = ((1-g2_full) * spectrum_x).sum(dim=(1, 2, 3)) / (64 * 64)
        parseval = parseval / (2 * SIGMA_MAX ** 2)
        relative = ((exact-parseval).abs() / parseval.clamp_min(1e-300)).max()
        if relative > 1e-8 or float(exact.min()) < -1e-7 or float(exact.max()) > 0.96+1e-7:
            raise RuntimeError("N0 terminal guard failed: %s/%s, relative=%g, min=%g, max=%g" %
                               (family, theta, float(relative), float(exact.min()),
                                float(exact.max())))
        row = {"family": family, "theta": theta, "u": u,
               "exact": summary(exact.cpu().numpy()),
               "retired_u_approximation": summary(approximate.cpu().numpy()),
               "approximation_minus_exact": summary((approximate-exact).cpu().numpy()),
               "parseval_max_relative_error": float(relative)}
        terminal_rows.append(row)
        print("[N0]", family, theta, "T max", row["exact"]["max"],
              "approx error max abs", max(abs(row["approximation_minus_exact"]["min"]),
                                           abs(row["approximation_minus_exact"]["max"])),
              flush=True)
    result["terminal"] = {"rows": terminal_rows,
                          "max_exact": max(r["exact"]["max"] for r in terminal_rows),
                          "max_abs_approximation_error": max(
                              max(abs(r["approximation_minus_exact"]["min"]),
                                  abs(r["approximation_minus_exact"]["max"]))
                              for r in terminal_rows)}
    save()

    torch.manual_seed(20261110)
    torch.cuda.manual_seed(20261110)
    model = load_local_model(CKPT).to(DEV).eval()
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x.to(DT), t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)
    sharp = pool[255:256].to(DT)
    blurred = apply_power(sharp.to(torch.float64), kernel_power("gaussian", 1.5)).to(DT)

    @torch.no_grad()
    def q_value(z, t, rtol=1e-5, atol=1e-5, method="RK45"):
        return float(likelihood_fn(z, ode_flow, t_spans=[SIGMA_MAX, float(t)],
                                   rtol=rtol, atol=atol, method=method).item())

    for t in TS:
        values = []
        for repeat in range(K_REP):
            q_sharp = q_value(sharp, t)
            q_blur = q_value(blurred, t)
            values.append(q_sharp-q_blur)
            print("[N1]", t, repeat+1, "/", K_REP, "Lambda", values[-1],
                  "elapsed", round(time.time()-start), flush=True)
        result["hutchinson"][str(t)] = {"values": values, "summary": summary(values)}
        save()

    settings = [("RK45", tol) for tol in [1e-3, 1e-4, 1e-5, 1e-6]]
    settings += [("DOP853", tol) for tol in [1e-5, 1e-6]]
    for t in TS:
        epsilon = torch.randint_like(sharp, low=0, high=2).to(DT)*2-1
        original_randint = torch.randint_like

        def fixed_randint_like(tensor, low=0, high=2):
            return ((epsilon+1.0)/2.0).to(tensor.dtype)

        rows = []
        try:
            for method, tolerance in settings:
                torch.randint_like = fixed_randint_like
                q_sharp = q_value(sharp, t, tolerance, tolerance, method)
                q_blur = q_value(blurred, t, tolerance, tolerance, method)
                rows.append({"method": method, "rtol": tolerance, "atol": tolerance,
                             "q_sharp": q_sharp, "q_blur": q_blur,
                             "lambda": q_sharp-q_blur})
                print("[N2]", t, method, tolerance, "Lambda", rows[-1]["lambda"],
                      flush=True)
        finally:
            torch.randint_like = original_randint
        reference = next(r["lambda"] for r in rows
                         if r["method"] == "DOP853" and r["rtol"] == 1e-6)
        for row in rows:
            row["difference_from_reference"] = row["lambda"]-reference
        result["ode"][str(t)] = {"reference": "DOP853, rtol=atol=1e-6", "rows": rows}
        save()

    print("[saved]", output, "elapsed", time.time()-start, flush=True)


if __name__ == "__main__":
    main()
