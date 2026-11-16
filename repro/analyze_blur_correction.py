r"""Fixed spectral-bias correction for the existing Gaussian profile-MAP grids.

Pre-registered 2026-11-10 before evaluating the corrected argmins. The correction is
L_corr(theta)=L(theta)+Lambda_G(theta,t), with coefficient exactly one and no fitted or
oracle parameter. Lambda_G=sum_f (1-g2_theta(f))*S(f)/(2*(S(f)+t^2)), where S is the
mean Fourier power of FFHQ pool images 128:256 only. Test images 0:40 are not used to
estimate S. The correction removes the stationary spectral-Gaussian prediction of blur
preference; it is not claimed to remove the non-Gaussian component.

C0: the coefficient and image split must remain as above. Theta=0 has zero correction.
C1: report baseline/corrected over-blur proportions and absolute relative kernel error on
    the main pool (0:20) and disjoint pool (20:40). The desired direction is lower error,
    but no outcome is hidden if the correction fails.
C2: whole-image cluster bootstrap CI for corrected-minus-baseline absolute relative error.

Run on the server: ./venv2/bin/python repro/analyze_blur_correction.py
"""

import json
import os
import random
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))
from putils.convolution import GaussianBlur

OUT = os.path.join(_PROJ, "results")
POOL = "/tmp/pool_ffhq64_256_s0.pt"
SEED = 20261110
REPEATS = 100000


def percentile(values, probability):
    values = sorted(values)
    return values[min(int(probability*len(values)), len(values)-1)]


def main():
    pool = torch.load(POOL, map_location="cpu").to(torch.float64)
    calibration = pool[128:256]
    spectrum = torch.fft.rfft2(calibration, dim=(-2, -1)).abs().square()/(64*64)
    s = spectrum.mean(dim=(0, 1))
    blur = GaussianBlur(kernel_size=15, device="cpu", dtype=torch.float64)
    rng = random.Random(SEED)
    result = {"cfg": {"coefficient": 1.0, "calibration_images": "128:256",
                      "test_images": "0:40", "repeats": REPEATS, "seed": SEED,
                      "analysis_status": "pre-registered correction, existing profile data"},
              "pools": {}}

    for name, filename in [("main", "phaseC_pool20.json"),
                           ("independent", "overblur_indep.json")]:
        with open(os.path.join(OUT, filename)) as handle:
            records = json.load(handle)["records"]
        rows = []
        for record in records:
            if record["noise"] < 0.172:
                continue
            t = record["noise"]
            profile = sorted(record["profile"], key=lambda point: point["theta"])
            corrected_values = []
            biases = []
            for point in profile:
                theta = point["theta"]
                if theta == 0:
                    bias = 0.0
                else:
                    kernel = blur.kernel(theta)
                    pad = torch.zeros((1, 1, 64, 64), dtype=torch.float64)
                    pad[:, :, 24:39, 24:39] = kernel
                    g2 = torch.fft.rfft2(pad, dim=(-2, -1)).abs().square()[0, 0]
                    bias = float(((1-g2)*s/(2*(s+t*t))).sum())
                biases.append(bias)
                corrected_values.append(point["L"]+bias)
            baseline = min(profile, key=lambda point: point["L"])["theta"]
            corrected = profile[min(range(len(profile)), key=lambda i: corrected_values[i])]["theta"]
            truth = record["theta_star"]
            base_error = abs(baseline-truth)/truth
            corrected_error = abs(corrected-truth)/truth
            rows.append({"img": record["img"], "noise": t, "theta_star": truth,
                         "theta_baseline": baseline, "theta_corrected": corrected,
                         "baseline_error": base_error, "corrected_error": corrected_error,
                         "error_difference": corrected_error-base_error,
                         "bias_at_truth": biases[profile.index(next(
                             point for point in profile if point["theta"] == truth))],
                         "bias_at_baseline": biases[[p["theta"] for p in profile].index(baseline)],
                         "bias_max": max(biases)})

        images = sorted({row["img"] for row in rows})
        by_image = {image: [row for row in rows if row["img"] == image] for image in images}

        def statistics(sample):
            selected = [row for image in sample for row in by_image[image]]
            return {"n": len(selected),
                    "baseline_overblur": sum(r["theta_baseline"] > r["theta_star"]
                                             for r in selected)/len(selected),
                    "corrected_overblur": sum(r["theta_corrected"] > r["theta_star"]
                                              for r in selected)/len(selected),
                    "baseline_mean_error": sum(r["baseline_error"] for r in selected)/len(selected),
                    "corrected_mean_error": sum(r["corrected_error"] for r in selected)/len(selected),
                    "mean_error_difference": sum(r["error_difference"] for r in selected)/len(selected),
                    "mean_bias_at_truth": sum(r["bias_at_truth"] for r in selected)/len(selected),
                    "mean_bias_at_baseline": sum(r["bias_at_baseline"] for r in selected)/len(selected)}

        observed = statistics(images)
        differences = [statistics([rng.choice(images) for _ in images])["mean_error_difference"]
                       for _ in range(REPEATS)]
        result["pools"][name] = {"source": filename, "observed": observed,
                                 "error_difference_cluster_95_interval": {
                                     "low": percentile(differences, 0.025),
                                     "high": percentile(differences, 0.975)},
                                 "rows": rows}
        print(name, json.dumps(observed, indent=2))
        print("error difference CI", percentile(differences, 0.025),
              percentile(differences, 0.975), flush=True)

    path = os.path.join(OUT, "blur_correction.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=1)
    print("[saved]", path)


if __name__ == "__main__":
    main()
