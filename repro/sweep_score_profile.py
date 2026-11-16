r"""Direct score-gradient profile optimisation for review2; pre-registered 2026-11-13.

M0: use the corrected symmetric 15x15 Gaussian, theta*=1.5 and the original 21-point grid.
    The denoiser relation is grad q_t(x)=(x-D_t(x))/t^2, with t=sigma_n. For each fixed theta,
    optimise the declared surrogate L=D_theta+q_sigma_n over x by Adam; do not substitute a
    denoiser call for prox(q_t). Do not clip x: the declared objective is unconstrained.
M1: two starts (y and h_theta*y), 600 steps, initial lr=0.003, cosine decay to 5% of that
    value, identical budgets at all grid points. Retain the lower FINAL L from the two starts;
    do not select an iterate by looking at the truth. Adam need not be monotone and no global
    optimality certificate is claimed.
M2: evaluate both initial and final q with RK45 rtol=atol=1e-5. One fixed Rademacher vector
    is shared by all sharp/blur ODE evaluations and all theta/start/iterate values within a
    profile. Save L, D, q, L_final-L_initial, gradient RMS and the last-50-step image drift.
M3: at theta=0/start=y, use one ODE central difference along -grad L with perturbation RMS
    0.003 and compare it with the score-gradient prediction. This is a directional diagnostic,
    not a proof of accuracy in every direction. Report, but do not threshold, the discrepancy.
M4: default pilot uses generated-pool images 0 and 1 at sigma_n={0.172,0.5}; a separate real
    invocation uses BSDS images 0 and 1. No outcome threshold or universal over/under-blur
    claim is registered. Smoke is only a 20-step software test and cannot be used as science.

The smoothed-prior objective is not asserted to be the exact posterior for the original clean
p_data under y=h_theta*x0+eta; that would require a different generative-model derivation.

Run: ./venv2/bin/python repro/sweep_score_profile.py --source generated \
       --out score_profile_generated.json
     ./venv2/bin/python repro/sweep_score_profile.py --source real \
       --out score_profile_real.json
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "code", "blind_deblur_diffusion"))
sys.path.insert(0, os.path.join(PROJ, "repro"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn
from sweep_overblur_generalization import Blur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
GRID = [round(0.125 * index, 4) for index in range(21)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("generated", "real"), default="generated")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--noises", type=float, nargs="+", default=[0.172, 0.5])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.steps < 60 and not args.smoke:
        raise ValueError("M1 requires at least 60 steps outside smoke")
    output_name = args.out or ("score_profile_%s.json" % args.source)
    if args.smoke and not os.path.basename(output_name).startswith("smoke_"):
        output_name = "smoke_" + os.path.basename(output_name)
    output = os.path.join(PROJ, "results", output_name)
    image_output = output.replace(".json", "_x.pt")
    indices = args.indices[:1] if args.smoke else args.indices
    noises = args.noises[:1] if args.smoke else args.noises
    grid = [0.0, 1.5, 2.5] if args.smoke else GRID
    steps = 20 if args.smoke else args.steps
    config = {"checkpoint": CKPT, "source": args.source, "indices": indices,
              "noises": noises, "theta_star": 1.5, "grid": grid, "steps": steps,
              "learning_rate": args.lr, "final_learning_rate_fraction": 0.05,
              "batch_size": args.batch_size, "starts": ["y", "h_theta*y"],
              "optimizer": "Adam on manually supplied score gradient, unconstrained x",
              "trace_sharing": "one Rademacher vector per image/noise profile",
              "rtol": 1e-5, "atol": 1e-5, "method": "RK45",
              "preregistered": "2026-11-13", "smoke": args.smoke}
    with open(__file__, "rb") as handle:
        config["script_sha256"] = hashlib.sha256(handle.read()).hexdigest()
    tensor_records = []
    if os.path.exists(output):
        with open(output) as handle:
            result = json.load(handle)
        if result["cfg"] != config or not os.path.exists(image_output):
            raise RuntimeError("Existing output does not match the frozen configuration")
        tensor_records = torch.load(image_output, map_location="cpu")
    else:
        result = {"cfg": config, "records": []}

    def save():
        with open(output, "w") as handle:
            json.dump(result, handle, indent=1)
        torch.save(tensor_records, image_output)

    torch.manual_seed(20261113)
    torch.cuda.manual_seed(20261113)
    blur = Blur("gaussian")
    kernel = blur.kernel(1.5)
    symmetry = float((kernel[0, 0] - kernel[0, 0].flip(0).flip(1)).abs().max())
    eigenvalues = blur.eig(1.5, 64, 64)
    if symmetry > 1e-6 or abs(float(kernel.sum()) - 1.0) > 1e-6:
        raise RuntimeError("M0 kernel guard failed")
    if float(eigenvalues.min()) < -1e-6 or float(eigenvalues.max()) > 1.0 + 1e-6:
        raise RuntimeError("M0 spectrum guard failed")
    if args.source == "generated":
        images = torch.load("/tmp/pool_ffhq64_256_s0.pt", map_location="cpu")
    else:
        images = torch.load(os.path.join(PROJ, "data", "bsds300_real_test_64.pt"),
                            map_location="cpu")
    model = load_local_model(CKPT).cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x, t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)
    start_time = time.time()

    for index in indices:
        sharp = images[index:index + 1].cuda().float()
        clean = blur.A(sharp, 1.5)
        for noise in noises:
            if any(record["img"] == index and record["noise"] == noise
                   for record in result["records"]):
                continue
            seed = (1000 * index if args.source == "generated" else 20261111 + 1000 * index)
            seed += int(noise * 100000)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            observation = clean + noise * torch.randn(clean.shape, generator=generator,
                                                       device="cuda")
            epsilon = torch.randint_like(sharp, low=0, high=2).float() * 2.0 - 1.0
            original_randint = torch.randint_like

            def fixed_randint(tensor, low=0, high=2):
                return ((epsilon + 1.0) / 2.0).to(tensor.dtype)

            @torch.no_grad()
            def q_value(image):
                torch.randint_like = fixed_randint
                try:
                    return float(likelihood_fn(image, ode_flow, t_spans=[80.0, float(noise)],
                                               rtol=1e-5, atol=1e-5, method="RK45").item())
                finally:
                    torch.randint_like = original_randint

            @torch.no_grad()
            def values(image, theta):
                residual = blur.A(image, theta) - observation
                data = float(residual.square().sum() / (2.0 * noise ** 2))
                denoised = model(image, noise)
                gradient = (blur.AT(residual, theta) + image - denoised) / noise ** 2
                return data, gradient

            runs = [(theta, start) for theta in grid for start in ("y", "blur")]
            profile = []
            for offset in range(0, len(runs), args.batch_size):
                chunk = runs[offset:offset + args.batch_size]
                initial = torch.cat([observation if start == "y" else blur.A(observation, theta)
                                     for theta, start in chunk], dim=0)
                image = torch.nn.Parameter(initial.clone())
                optimizer = torch.optim.Adam([image], lr=args.lr)
                initial_metrics = []
                for position, (theta, _) in enumerate(chunk):
                    data, gradient = values(image[position:position + 1].detach(), theta)
                    q_initial = q_value(image[position:position + 1].detach())
                    initial_metrics.append((data, q_initial, float(gradient.square().mean().sqrt())))
                history_step = max(0, steps - 51)
                for step in range(steps):
                    fraction = 0.5 * (1.0 + math.cos(math.pi * step / max(steps - 1, 1)))
                    lr = args.lr * (0.05 + 0.95 * fraction)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    gradients = []
                    for position, (theta, _) in enumerate(chunk):
                        _, gradient = values(image[position:position + 1].detach(), theta)
                        gradients.append(gradient)
                    image.grad = torch.cat(gradients, dim=0)
                    optimizer.step()
                    if step == history_step:
                        previous = image.detach().clone()
                    if not torch.isfinite(image).all():
                        raise RuntimeError("M1 non-finite iterate")
                for position, (theta, start_name) in enumerate(chunk):
                    final = image[position:position + 1].detach()
                    data, gradient = values(final, theta)
                    q_final = q_value(final)
                    data_initial, q_initial, gradient_initial = initial_metrics[position]
                    drift = float((final - previous[position:position + 1]).square().mean().sqrt())
                    row = {"theta": theta, "start": start_name, "data": data, "q": q_final,
                           "L": data + q_final, "data_initial": data_initial,
                           "q_initial": q_initial, "L_initial": data_initial + q_initial,
                           "L_final_minus_initial": data + q_final - data_initial - q_initial,
                           "gradient_rms_final": float(gradient.square().mean().sqrt()),
                           "gradient_rms_initial": gradient_initial,
                           "image_drift_rms_last50": drift,
                           "drift_window_steps": steps - history_step,
                           "max_abs_pixel": float(final.abs().max())}
                    profile.append(row)
                    tensor_records.append(final.cpu())
                print("[M1]", args.source, index, noise, offset, "/", len(runs),
                      "elapsed", round(time.time() - start_time), flush=True)
                del image, optimizer
                torch.cuda.empty_cache()

            check_image = tensor_records[len(result["records"]) * len(runs)].cuda()
            _, gradient = values(check_image, runs[0][0])
            direction = -gradient / gradient.norm().clamp_min(1e-30)
            displacement = 0.003 * math.sqrt(direction.numel()) * direction
            q_plus = q_value(check_image + displacement)
            q_minus = q_value(check_image - displacement)
            data_plus = float(((blur.A(check_image + displacement, 0.0) - observation)
                               .square().sum() / (2.0 * noise ** 2)))
            data_minus = float(((blur.A(check_image - displacement, 0.0) - observation)
                                .square().sum() / (2.0 * noise ** 2)))
            predicted = float((gradient * direction).sum() * math.sqrt(direction.numel()))
            measured = (q_plus + data_plus - q_minus - data_minus) / (2.0 * 0.003)
            check = {"theta": 0.0, "start": "y", "perturbation_rms": 0.003,
                     "score_predicted_derivative_per_rms": predicted,
                     "ode_central_difference_per_rms": measured,
                     "difference": measured - predicted}
            selected = {}
            for theta in grid:
                candidates = [row for row in profile if row["theta"] == theta]
                selected[theta] = min(candidates, key=lambda row: row["L"])
            best = min(selected.values(), key=lambda row: row["L"])
            record = {"img": index, "noise": noise, "observation_seed": seed,
                      "runs": profile, "profile": list(selected.values()),
                      "argmin_theta": best["theta"], "directional_gradient_check": check}
            result["records"].append(record)
            save()
            print("[M2]", args.source, index, noise, "argmin", best["theta"],
                  "gradient_check", check, "elapsed", round(time.time() - start_time), flush=True)
            del observation
            torch.cuda.empty_cache()
        del sharp, clean
    print("[saved]", output, "records", len(result["records"]), flush=True)


if __name__ == "__main__":
    main()
