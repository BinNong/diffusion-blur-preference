r"""CPU physics guards for review2 reproducibility; no model checkpoint or GPU required.

K1: all eight physical reference powers are finite, in [0,1], and their conjugate-weighted
    rFFT coordinate equals a directly reconstructed full 64x64 FFT mean. Gaussian, defocus,
    Airy and motion kernels are normalised and centre-symmetric.
K2: for a deterministic real RGB input, circular filtering and Parseval's identity agree with
    the full-grid spectrum; rFFT and full-FFT filtering agree pointwise.
K3: all 69 raw dense rows (60 unique physical configurations) reconstruct their stored rFFT
    coordinate and a valid full-grid coordinate, independently checked in full_coordinate.

Run: ./venv2/bin/python repro/validate_review2_physics.py
"""

import json
import os
import sys

import numpy as np
import torch

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJ, "code", "blind_deblur_diffusion"))
sys.path.insert(0, os.path.join(PROJ, "repro"))

import review_numeric_checks as reference
from analyze_review2 import full_coordinate
from putils.convolution import AiryDisc, DefocusBlur, GaussianBlur, MotionBlur


def main():
    reference.DEV = torch.device("cpu")
    generator = torch.Generator(device="cpu").manual_seed(20261113)
    image = torch.randn((2, 3, 64, 64), generator=generator, dtype=torch.float64)
    rows = []
    for family, theta in reference.REFS:
        power = reference.kernel_power(family, theta).cpu()
        full = reference.kernel_power(family, theta, full=True).cpu()
        if not torch.isfinite(power).all() or not torch.isfinite(full).all():
            raise RuntimeError("K1 non-finite spectrum")
        if float(power.min()) < -1e-12 or float(power.max()) > 1.0 + 1e-12:
            raise RuntimeError("K1 non-contractive spectrum")
        weights = torch.full((1, 33), 2.0, dtype=torch.float64)
        weights[0, (0, 32)] = 1.0
        weighted = float((weights * power[0, 0]).sum() / (64 * 64))
        direct = float(full.mean())
        coordinate_error = abs(weighted - direct)
        if coordinate_error > 1e-12:
            raise RuntimeError("K1 conjugate-weight mismatch")
        blurred = reference.apply_power(image, power)
        blurred_full = torch.fft.ifft2(torch.fft.fft2(image, dim=(-2, -1))
                                      * full.sqrt(), dim=(-2, -1)).real
        filter_error = float((blurred - blurred_full).abs().max())
        spectrum_image = torch.fft.fft2(image, dim=(-2, -1)).abs().square()
        exact = (image.square().sum(dim=(1, 2, 3))
                 - blurred.square().sum(dim=(1, 2, 3)))
        parseval = ((1.0 - full) * spectrum_image).sum(dim=(1, 2, 3)) / (64 * 64)
        parseval_relative = float(((exact - parseval).abs()
                                   / exact.abs().clamp_min(1e-300)).max())
        if filter_error > 1e-9 or parseval_relative > 1e-10:
            raise RuntimeError("K2 filtering/Parseval failure")
        row = {"family": family, "theta": theta, "u_rfft": float(power.mean()),
               "u_full": direct, "full_filter_max_abs": filter_error,
               "parseval_max_relative": parseval_relative}
        if family != "lowpass":
            cls = {"gaussian": GaussianBlur, "defocus": DefocusBlur,
                   "airy": AiryDisc, "motion": MotionBlur}[family]
            kernel = cls(kernel_size=15, device="cpu", dtype=torch.float64).kernel(theta)
            if kernel.ndim == 4:
                kernel = kernel[0, 0]
            sum_error = abs(float(kernel.sum()) - 1.0)
            symmetry = float((kernel - kernel.flip(0).flip(1)).abs().max())
            centre = int(kernel[7, 7].argmax()) if kernel[7, 7].ndim else 0
            centre_peak = bool(kernel[7, 7] >= kernel.max())
            if sum_error > 1e-10 or symmetry > 1e-10 or not centre_peak:
                raise RuntimeError("K1 kernel normalisation/centre guard failed: %s" % family)
            row.update(kernel_sum_error=sum_error, flip_symmetry=symmetry,
                       centre_has_peak=centre_peak)
        rows.append(row)
    synthetic = 0
    for filename in ("sweep_bdense_n120.json", "sweep_blow_n120.json"):
        with open(os.path.join(PROJ, "results", filename)) as handle:
            records = json.load(handle)["rows"]
        for record in records:
            u_full, power = full_coordinate(record)
            expected = 1.0 - float(record["kunif"])
            if abs(float(power.mean()) - expected) > 1e-12 or not 0.0 <= u_full <= 1.0:
                raise RuntimeError("K3 synthetic coordinate failure")
            synthetic += 1
    output = {"cfg": {"date": "2026-11-13", "device": "cpu",
                      "torch": torch.__version__, "seed": 20261113,
                      "synthetic_rows": synthetic},
              "reference_kernels": rows,
              "guards": {"K1_K2_reference_kernels": len(rows), "K3_synthetic_rows": synthetic,
                         "pass": True}}
    path = os.path.join(PROJ, "results", "review2_physics_guards.json")
    with open(path, "w") as handle:
        json.dump(output, handle, indent=1)
    print("[PASS] K1/K2", len(rows), "reference kernels; K3", synthetic,
          "synthetic rows; saved", path)


if __name__ == "__main__":
    main()
