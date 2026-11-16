r"""Full-FFT audit of Gaussian/spectral references; no diffusion-model evaluations.

Pre-registered 2026-11-11 before computing corrected reference values. This is a
post-hoc correction of a mathematical convention, not a new physical sweep.

For real images, a non-redundant rFFT stores each non-self conjugate mode once.
Use c(f)=1 for columns 0 and W/2 and c(f)=2 otherwise. Full-FFT Parseval sums are
sum_f c(f)P_rfft(f). All Gaussian references here sum all three RGB channels,
with shared channel covariance, rather than averaging channels before comparing
with the full-image log density. Preserve every old result; write new files only.

A0: c sums to H*W, and weighted rFFT energy equals full-FFT energy to 1e-10 relative.
A1: reconstructed operators reproduce every stored kunif to 1e-7 absolute.
A2: preserve measured Lambda and the original table; recompute only the Gaussian/spectral
    reference, its full-grid u, and the fixed unit-coefficient correction. No fitted parameter
    and no threshold selected after seeing results. Report all corrected values, including
    any changed argmin. No PSNR/SSIM claim is licensed by this audit.

Run: ./venv2/bin/python repro/analyze_fullfft_references.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ)
sys.path.insert(0, os.path.join(_PROJ, "repro"))
from review_numeric_checks import REFS, kernel_power
from sweep_law_test import halfplane, make_profile
from sweep_profile_shape import ideal_profile, make_ref, mode_grid
from putils.convolution import GaussianBlur

OUT = os.path.join(_PROJ, "results")
H = W = 64
FILES = [
    "sweep_b2test.json", "sweep_bdense.json", "sweep_bgap.json", "sweep_blow.json",
    "sweep_bt0005.json", "sweep_bt001.json", "sweep_bt002.json", "sweep_bt005.json",
    "sweep_bs008.json", "sweep_bs012.json", "sweep_bs016.json", "sweep_law_test.json",
    "sweep_m_afq_afq.json", "sweep_m_ffq_afq.json", "sweep_profile_shape.json",
    "sweep_steepness.json", "sweep_twlaw.json", "sweep_twod.json", "sweep_uniform.json",
]


def weights(dtype=torch.float64):
    result = torch.full((H, W//2+1), 2.0, dtype=dtype)
    result[:, 0] = 1.0
    result[:, W//2] = 1.0
    return result


def spectrum(images):
    return torch.fft.rfft2(images.to(torch.float64), dim=(-2, -1)).abs().square()/(H*W)


def operator_for_row(row):
    meta = row["meta"]
    if "family" in meta:
        reference = make_ref(meta["family"], meta["theta"])
    elif meta.get("kind") in {"lp", "hp", "bp"}:
        radius, _, _ = mode_grid()
        attenuation, _ = ideal_profile(meta["kind"], radius, meta["K"]*H*(W//2+1))
        reference = 1.0-attenuation
    elif meta["kind"] == "uniform":
        return np.full((H, W//2+1), float(meta["par"]), dtype=np.float64)
    else:
        reference = make_profile(meta["kind"], meta["par"], meta["scale"], halfplane())
    beta = float(meta.get("beta", 1.0))
    return (1.0-beta)+beta*reference


def power_sums(images, t):
    c = weights()
    full = spectrum(images).mean(dim=(0, 1))
    selected = spectrum(images[:20]).mean(dim=(0, 1))
    if selected.shape != full.shape or full.shape != c.shape:
        raise RuntimeError("A0 spectrum shape mismatch")
    weight = 3.0*c*selected/(2.0*(full+t*t))
    cap = float(weight.sum())
    return weight, cap


def summary(values):
    return {"n": len(values), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}


def table_audit(pool):
    c = weights()
    sharp = pool[255:256]
    g2 = kernel_power("gaussian", 1.5).to(torch.float64)[0, 0].cpu()
    blurred = torch.fft.irfft2(torch.fft.rfft2(sharp.to(torch.float64), dim=(-2, -1))
                              * g2.sqrt(), s=(H, W), dim=(-2, -1))
    sharp_power = spectrum(sharp)
    blurred_power = spectrum(blurred)
    calibration = spectrum(pool[:128])
    radial, _, _ = mode_grid()
    radial = radial[:, :W//2+1]
    edges = np.linspace(0, radial.max()+1e-6, 33)
    indices = np.clip(np.digitize(radial.ravel(), edges)-1, 0, 31).reshape(radial.shape)
    pooled = calibration.mean(dim=(0, 1))
    radial_spectrum = torch.zeros(32, dtype=torch.float64)
    for index in range(32):
        mask = torch.from_numpy(indices == index)
        if mask.any():
            radial_spectrum[index] = pooled[mask].mean()
    with open(os.path.join(OUT, "tdep_kfix.json")) as handle:
        original = json.load(handle)
    rows = []
    for row in original["res"]["correct"]["rows"]:
        t = float(row["t"])
        variances = radial_spectrum[torch.from_numpy(indices)]+t*t
        preference = float((c*(sharp_power[0]-blurred_power[0])/(2*variances)).sum())
        rows.append({"t": t, "lambda_measured_unchanged": row["lam"],
                     "lambda_gauss_retired_rfft_channel_average": row["lam_gauss"],
                     "lambda_gauss_full_rgb": preference,
                     "ratio_full": row["lam"]/preference})
    return {"source": "tdep_kfix.json (measured Lambda unchanged)",
            "calibration": "pool[0:128], same 32 radial bins as retired reference",
            "model": "zero-mean real RGB stationary Gaussian, independent channels with shared covariance",
            "rows": rows}


def row_audit(pool_cache):
    rows = []
    maximum_kunif_error = 0.0
    for filename in FILES:
        with open(os.path.join(OUT, filename)) as handle:
            data = json.load(handle)
        cfg = data["cfg"]
        pool_name = cfg["pool"]
        if pool_name not in pool_cache:
            pool_cache[pool_name] = torch.load(pool_name, map_location="cpu")
        pool = pool_cache[pool_name]
        weight, cap = power_sums(pool, float(cfg["t"]))
        for row in data["rows"]:
            g2_numpy = operator_for_row(row)
            error = abs(float(1.0-g2_numpy.mean())-float(row["kunif"]))
            maximum_kunif_error = max(maximum_kunif_error, error)
            if error > 1e-7:
                raise RuntimeError("A1 operator mismatch: %s %s error=%g" %
                                   (filename, row["id"], error))
            g2 = torch.from_numpy(g2_numpy).to(torch.float64)
            full_preference = float(((1.0-g2)*weight).sum())
            rows.append({"source": filename, "id": row["id"], "t": cfg["t"],
                         "u_rfft": 1.0-float(g2.mean()),
                         "u_full": float((weights()*g2).sum()/weights().sum()),
                         "lambda": row["lam_mean"], "lambda_gauss_retired": row["lam_gauss"],
                         "lambda_gauss_full_rgb": full_preference,
                         "full_cap": cap, "kunif_reconstruction_error": error})
    if len(rows) != 851:
        raise RuntimeError("Expected the published 851 reference rows, got %d" % len(rows))
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["t"])].append(row)
    by_t = {}
    for t, selected in sorted(grouped.items(), key=lambda item: float(item[0])):
        by_t[t] = summary([abs(r["lambda_gauss_full_rgb"]/r["lambda"])
                           for r in selected if r["lambda_gauss_full_rgb"] > 0])
    return rows, by_t, maximum_kunif_error


def correction_audit(pool):
    c = weights()
    calibration_spectrum = spectrum(pool[128:256]).mean(dim=(0, 1))
    blur = GaussianBlur(kernel_size=15, device="cpu", dtype=torch.float64)
    result = {}
    for name, filename in [("main", "phaseC_pool20.json"),
                           ("disjoint", "overblur_indep.json")]:
        with open(os.path.join(OUT, filename)) as handle:
            records = json.load(handle)["records"]
        rows = []
        for record in records:
            if record["noise"] < 0.172:
                continue
            t = float(record["noise"])
            profile = sorted(record["profile"], key=lambda point: point["theta"])
            biases = []
            for point in profile:
                theta = point["theta"]
                if theta == 0:
                    biases.append(0.0)
                    continue
                kernel = blur.kernel(theta)
                padded = torch.zeros((1, 1, H, W), dtype=torch.float64)
                padded[:, :, 24:39, 24:39] = kernel
                g2 = torch.fft.rfft2(padded, dim=(-2, -1)).abs().square()[0, 0]
                biases.append(float((3.0*c*(1.0-g2)*calibration_spectrum /
                                     (2.0*(calibration_spectrum+t*t))).sum()))
            corrected = [point["L"]+bias for point, bias in zip(profile, biases)]
            optimum = profile[min(range(len(profile)), key=lambda i: corrected[i])]["theta"]
            baseline = min(profile, key=lambda point: point["L"])["theta"]
            truth = record["theta_star"]
            base_error = abs(baseline-truth)/truth
            corrected_error = abs(optimum-truth)/truth
            rows.append({"img": record["img"], "noise": t, "theta_star": truth,
                         "theta_baseline": baseline, "theta_corrected_full_rgb": optimum,
                         "bias_at_truth": biases[[p["theta"] for p in profile].index(truth)],
                         "bias_at_baseline": biases[[p["theta"] for p in profile].index(baseline)],
                         "baseline_error": base_error, "corrected_error": corrected_error,
                         "error_difference": corrected_error-base_error,
                         "changed": optimum != baseline})
        images = sorted({row["img"] for row in rows})

        def statistics(sample):
            selected = [row for image in sample for row in rows if row["img"] == image]
            return {"n": len(selected),
                    "baseline_overblur": sum(r["theta_baseline"] > r["theta_star"]
                                             for r in selected)/len(selected),
                    "corrected_overblur": sum(r["theta_corrected_full_rgb"] > r["theta_star"]
                                              for r in selected)/len(selected),
                    "baseline_mean_error": float(np.mean([r["baseline_error"] for r in selected])),
                    "corrected_mean_error": float(np.mean([r["corrected_error"] for r in selected])),
                    "mean_error_difference": float(np.mean([r["error_difference"] for r in selected])),
                    "n_changed_argmins": sum(r["changed"] for r in selected)}

        import random
        rng = random.Random(20261110)
        differences = [statistics([rng.choice(images) for _ in images])["mean_error_difference"]
                       for _ in range(100000)]
        differences.sort()
        result[name] = {"source": filename, "observed": statistics(images),
                        "error_difference_cluster_95_interval": {
                            "method": "post-hoc percentile bootstrap over whole-image clusters",
                            "low": differences[2500], "high": differences[97500]},
                        "mean_bias_at_truth": float(np.mean([r["bias_at_truth"] for r in rows])),
                        "mean_bias_at_baseline": float(np.mean([r["bias_at_baseline"] for r in rows])),
                        "rows": rows}
    return {"model": "full RGB stationary spectral Gaussian; unit coefficient; calibration pool[128:256]",
            "analysis_status": "post-hoc formula correction; no tuning; cluster intervals are post-hoc",
            "pools": result}


def main():
    pool = torch.load("/tmp/pool_ffhq64_256_s0.pt", map_location="cpu")
    c = weights()
    if int(c.sum()) != H*W:
        raise RuntimeError("A0 weight total failed")
    probe = pool[:2]
    full = torch.fft.fft2(probe.to(torch.float64), dim=(-2, -1)).abs().square()/(H*W)
    half = spectrum(probe)
    relative = float(((full.sum(dim=(-2, -1))-(c*half).sum(dim=(-2, -1))).abs() /
                      full.sum(dim=(-2, -1)).clamp_min(1e-300)).max())
    if relative > 1e-10:
        raise RuntimeError("A0 Parseval guard failed: %g" % relative)
    terminal = []
    for family, theta in REFS:
        g2 = kernel_power(family, theta).to(torch.float64)[0, 0].cpu()
        half_u = float(g2.mean())
        full_u = float((c*g2).sum()/c.sum())
        terminal.append({"family": family, "theta": theta,
                         "u_rfft": half_u, "u_full": full_u,
                         "rfft_minus_full": half_u-full_u})
    rows, by_t, maximum_kunif_error = row_audit({})
    overall = summary([abs(row["lambda_gauss_full_rgb"]/row["lambda"])
                       for row in rows if row["lambda_gauss_full_rgb"] > 0])
    profile_path = os.path.join(OUT, "sweep_profile_shape.json")
    with open(profile_path) as handle:
        profile_data = json.load(handle)
    weight, _ = power_sums(pool, profile_data["cfg"]["t"])
    position = []
    for k in (0.2, 0.4, 0.6, 0.8):
        selected = [row for row in profile_data["rows"]
                    if row["meta"].get("kind") in {"lp", "hp", "bp"}
                    and row["meta"]["K"] == k]
        gauss = []
        measured = []
        for row in selected:
            g2 = torch.from_numpy(operator_for_row(row)).to(torch.float64)
            gauss.append(float(((1.0-g2)*weight).sum()))
            measured.append(abs(row["lam_mean"]))
        gauss = np.asarray(gauss)
        position.append({"K": k,
                         "gauss_relative_spread": float(np.ptp(gauss/gauss.mean()-1)),
                         "lambda_relative_spread": float((np.max(measured)-np.min(measured))/np.mean(measured))})
    caps = {}
    for label, selected in [("0:20", pool[:20]), ("20:40", pool[20:40])]:
        all_spectrum = spectrum(pool).mean(dim=(0, 1))
        z = spectrum(selected).mean(dim=(0, 1))
        if z.shape != all_spectrum.shape:
            raise RuntimeError("A0 cap shape mismatch")
        cap = float((3.0*c*z/(2.0*(all_spectrum+0.002**2))).sum())
        caps[label] = cap
    result = {"cfg": {"status": "post-hoc mathematical audit, no model evaluations",
                      "fft": "full Hermitian weighting, all RGB channels",
                      "row_count": len(rows), "parseval_max_relative_error": relative,
                      "max_kunif_reconstruction_error": maximum_kunif_error},
              "u_audit": terminal, "table": table_audit(pool),
              "spectral_rows": rows, "spectral_overall": overall,
              "spectral_by_t": by_t, "position_sensitivity": position,
              "low_t_full_rgb_caps": caps,
              "theoretical_full_rgb_mode_count": 3*H*W,
              "theoretical_low_t_cap_if_Z_equals_S": 3*H*W/2}
    with open(os.path.join(OUT, "fullfft_reference_audit.json"), "w") as handle:
        json.dump(result, handle, indent=1)
    correction = correction_audit(pool)
    with open(os.path.join(OUT, "blur_correction_fullfft.json"), "w") as handle:
        json.dump(correction, handle, indent=1)
    print("A0/A1 passed", relative, maximum_kunif_error)
    print("TABLE", json.dumps(result["table"]["rows"], indent=2))
    print("SPECTRAL", json.dumps({"overall": overall, "by_t": by_t,
                                  "position": position, "caps": caps}, indent=2))
    print("CORRECTION", json.dumps({name: {key: value for key, value in group.items()
                                           if key != "rows"}
                                    for name, group in correction["pools"].items()}, indent=2))
    print("[saved] fullfft_reference_audit.json and blur_correction_fullfft.json")


if __name__ == "__main__":
    main()
