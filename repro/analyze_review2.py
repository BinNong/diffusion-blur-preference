r"""Review-2 post-hoc statistical audit; no new image measurement and no GPU.

Criteria fixed in this file on 2026-11-13 after reading review2, before executing the analysis.
S1: use the image, not the image--kernel pair, as the independent unit for real-BSDS sign
    intervals. Report two-sided 95% Clopper--Pearson intervals per kernel and for the event
    that every tested kernel is positive on an image. Keep pair counts only as raw counts.
S2: reproduce the original N=120, t=0.002 estimator: arithmetic mean Lambda per operator,
    an unweighted degree-5 polynomial of log(mean Lambda) against log(u) on ALL common u,
    its analytic derivative, and unweighted nonlinear least squares b_inf+A*u^q on
    0.13<=u<=0.86. Do not move the polynomial onto the closed-form subset.
S3: resample the same 120 image indices across all three families (B=2000, seed=20261113)
    and report percentile 95% intervals per family and for the arithmetic three-family mean.
    These are image-sampling intervals, not Hutchinson-repeat or solver intervals.
S4: report first-20, first-60 and last-60 image fits; polynomial degrees 3/4/5/6; and
    closed-form windows [0.13,0.86], [0.14,0.74], [0.18,0.65]. These are sensitivity
    analyses with no success threshold and must not be described as confidence intervals.
S5: reconstruct the three synthetic power spectra and replace the unweighted rFFT mean by
    u_full=sum(c_f*g2_f)/sum(c_f), c_f=1 in rFFT columns 0 and 32 and 2 otherwise. Repeat
    S2/S3 on exactly the same stored Lambda values. Do not interpolate to the old u grid.

Run on the server: ./venv2/bin/python repro/analyze_review2.py --nboot 2000
"""

import argparse
import hashlib
import json
import math
import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import beta as beta_distribution

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(PROJ, "results")
DENSE = ("W2", "W8", "TK50")
PARAMETERS = ("b_inf", "A", "q")
FAMILY_KIND_PAR = {"W2": ("weibull", 2.0), "W8": ("weibull", 8.0),
                   "TK50": ("tukey", 0.5)}
BPROBE = (0.15, 0.35, 0.65)


def make_profile(kind, par, scale, radius):
    if kind == "weibull":
        return np.exp(-(radius / scale) ** par)
    if kind == "tukey":
        inner = scale * (1.0 - par)
        result = np.zeros_like(radius)
        result[radius <= inner] = 1.0
        middle = (radius > inner) & (radius < scale)
        phase = (radius[middle] - inner) / (scale - inner)
        result[middle] = 0.5 * (1.0 + np.cos(np.pi * phase))
        return result
    raise ValueError(kind)


def halfplane():
    frequency = np.fft.fftfreq(64) * 64
    fx, fy = np.meshgrid(frequency, frequency, indexing="ij")
    radius = np.sqrt(fx ** 2 + fy ** 2)[:, :33]
    return radius / radius.max()


def full_coordinate(record):
    meta = record["meta"]
    kind, par = FAMILY_KIND_PAR[meta["fid"]]
    reference = make_profile(kind, par, float(meta["scale"]), halfplane())
    power = (1.0 - float(meta["beta"])) + float(meta["beta"]) * reference
    weights = np.full((1, 33), 2.0)
    weights[0, (0, 32)] = 1.0
    full_sum = power.shape[0] * weights.sum()
    u = float((weights * power).sum() / full_sum)
    completed = np.concatenate((power, power[:, -2:0:-1]), axis=1)
    if completed.shape != (64, 64) or abs(float(completed.mean()) - u) > 1e-12:
        raise RuntimeError("S5 conjugate-weight reconstruction failed")
    if not 0.0 <= u <= 1.0:
        raise RuntimeError("S5 retained power outside [0,1]")
    return u, power


def load_dense(full=False):
    profiles = {family: {} for family in DENSE}
    guards = []
    for filename in ("sweep_bdense_n120.json", "sweep_blow_n120.json"):
        with open(os.path.join(RESULTS, filename)) as handle:
            data = json.load(handle)
        if data["cfg"]["nimg"] != 120 or data["cfg"]["t"] != 0.002:
            raise RuntimeError("S2 wrong image count or noise level")
        for record in data["rows"]:
            family = record["meta"]["fid"]
            if family not in profiles:
                continue
            original_u = 1.0 - float(record["kunif"])
            values = np.asarray(record["lam_per_img"], dtype=np.float64)
            if values.shape != (120,) or not np.isfinite(values).all():
                raise RuntimeError("S2 missing or non-finite per-image values")
            if full:
                u, power = full_coordinate(record)
                rfft_u = float(power.mean())
                guards.append(abs(rfft_u - original_u))
            else:
                u = original_u
            key = round(original_u, 4)
            if key in profiles[family] and abs(profiles[family][key][0] - u) > 1e-10:
                raise RuntimeError("S2 inconsistent duplicate coordinates")
            profiles[family][key] = (u, values)
    if full and max(guards) > 1e-12:
        raise RuntimeError("S5 reconstructed rFFT spectrum mismatch")
    result = {}
    for family, entries in profiles.items():
        ordered = sorted(entries.values(), key=lambda item: -item[0])
        result[family] = (np.asarray([item[0] for item in ordered]),
                          np.stack([item[1] for item in ordered], axis=1))
    counts = {len(result[family][0]) for family in DENSE}
    if counts != {20}:
        raise RuntimeError("S2 expected 20 common operators, including the raw reference")
    return result


def fit_once(profiles, family, indices=None, degree=5, window=(0.13, 0.86)):
    u, values = profiles[family]
    if indices is not None:
        values = values[indices]
    means = values.mean(axis=0)
    if not np.all(means > 0):
        return None
    coefficients = np.polyfit(np.log(u), np.log(means), degree)
    derivative = np.polyder(coefficients)
    selected = (u >= window[0]) & (u <= window[1])
    us = u[selected]
    bv = np.polyval(derivative, np.log(us))

    def model(x, b_inf, amplitude, exponent):
        return b_inf + amplitude * x ** exponent

    try:
        parameters, _ = curve_fit(model, us, bv, p0=(-0.7, -6.0, 3.0),
                                  bounds=([-1000, -50, 0.1], [1000, 50, 12]),
                                  maxfev=100000)
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    residual = bv - model(us, *parameters)
    return {"b_inf": float(parameters[0]), "A": float(parameters[1]),
            "q": float(parameters[2]), "rms": float(np.sqrt(np.mean(residual ** 2))),
            "b_at": {str(x): float(np.polyval(derivative, math.log(x))) for x in BPROBE}}


def mean_fit(fits):
    valid = [fits[family] for family in DENSE if fits[family] is not None]
    if len(valid) != len(DENSE):
        return None
    return {key: float(np.mean([fit[key] for fit in valid])) for key in PARAMETERS}


def interval(values):
    array = np.asarray(values, dtype=np.float64)
    return {"n": int(array.size), "sd": float(array.std(ddof=1)),
            "ci95": [float(v) for v in np.percentile(array, [2.5, 97.5])],
            "median": float(np.median(array))}


def bootstrap(profiles, resamples):
    samples = {family: {key: [] for key in PARAMETERS} for family in DENSE}
    samples["three_family_mean"] = {key: [] for key in PARAMETERS}
    failed = 0
    for indices in resamples:
        fits = {family: fit_once(profiles, family, indices) for family in DENSE}
        if any(fit is None for fit in fits.values()):
            failed += 1
            continue
        for family in DENSE:
            for key in PARAMETERS:
                samples[family][key].append(fits[family][key])
        mean = mean_fit(fits)
        for key in PARAMETERS:
            samples["three_family_mean"][key].append(mean[key])
    return {"requested": len(resamples), "failed": failed,
            "fits": {group: {key: interval(values) for key, values in parameters.items()}
                     for group, parameters in samples.items()}}


def clopper_pearson(successes, total):
    lower = 0.0 if successes == 0 else float(beta_distribution.ppf(0.025, successes,
                                                                    total-successes+1))
    upper = 1.0 if successes == total else float(beta_distribution.ppf(0.975, successes+1,
                                                                        total-successes))
    return [lower, upper]


def real_sign_intervals():
    with open(os.path.join(RESULTS, "real_bsds_edm.json")) as handle:
        rows = json.load(handle)["rows"]
    result = {}
    for noise in (0.002, 0.2):
        subset = [row for row in rows if row["t"] == noise]
        kernels = sorted({(row["family"], row["theta"]) for row in subset})
        per_kernel = []
        for kernel in kernels:
            selected = [row for row in subset
                        if (row["family"], row["theta"]) == kernel]
            successes = sum(row["lambda"] > 0 for row in selected)
            per_kernel.append({"family": kernel[0], "theta": kernel[1],
                               "positive": successes, "n_images": len(selected),
                               "clopper_pearson95": clopper_pearson(successes, len(selected))})
        indices = sorted({row["img"] for row in subset})
        unanimous = sum(all(row["lambda"] > 0 for row in subset if row["img"] == index)
                        for index in indices)
        result[str(noise)] = {
            "per_kernel": per_kernel,
            "all_tested_kernels_positive_on_same_image": {
                "positive": unanimous, "n_images": len(indices),
                "clopper_pearson95": clopper_pearson(unanimous, len(indices))},
            "raw_positive_pairs": sum(row["lambda"] > 0 for row in subset),
            "raw_pairs": len(subset)}
    return result


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nboot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--out", default="review2_statistics.json")
    args = parser.parse_args()
    if args.nboot < 100:
        raise ValueError("S3 requires at least 100 bootstrap replicates")

    primary = load_dense(full=False)
    expected = {"W2": (-0.7115386522279126, -7.360020959843477, 3.325755269948318),
                "W8": (-0.7429181064822714, -7.203132105124114, 3.4894541596213706),
                "TK50": (-0.7348145719171694, -7.111165631824027, 3.4497714135609105)}
    for family in DENSE:
        reproduced = fit_once(primary, family)
        for key, target in zip(PARAMETERS, expected[family]):
            if reproduced is None or abs(reproduced[key] - target) > 1e-6:
                raise RuntimeError("S2 failed to reproduce the frozen N=120 fit")
    full = load_dense(full=True)
    rng = np.random.default_rng(args.seed)
    resamples = rng.integers(0, 120, size=(args.nboot, 120))
    output = {"cfg": {"analysis": "post-hoc review2", "date": "2026-11-13",
                      "nboot": args.nboot, "seed": args.seed,
                      "criteria": __doc__},
              "real_image_sign": real_sign_intervals(),
              "rfft": {}, "full_fft": {}}
    for label, profiles in (("rfft", primary), ("full_fft", full)):
        fits = {family: fit_once(profiles, family) for family in DENSE}
        if any(fit is None for fit in fits.values()):
            raise RuntimeError("S2/S5 primary fit failed")
        output[label]["point_fits"] = fits
        output[label]["three_family_mean"] = mean_fit(fits)
        output[label]["image_bootstrap"] = bootstrap(profiles, resamples)
    output["rfft"]["image_subsets"] = {}
    for label, indices in (("first_20", np.arange(20)), ("first_60", np.arange(60)),
                           ("last_60", np.arange(60, 120))):
        fits = {family: fit_once(primary, family, indices) for family in DENSE}
        output["rfft"]["image_subsets"][label] = {
            "per_family": fits, "three_family_mean": mean_fit(fits)}
    sensitivity = []
    for degree in (3, 4, 5, 6):
        for window in ((0.13, 0.86), (0.14, 0.74), (0.18, 0.65)):
            fits = {family: fit_once(primary, family, degree=degree, window=window)
                    for family in DENSE}
            sensitivity.append({"degree": degree, "window": list(window),
                                "per_family": fits, "three_family_mean": mean_fit(fits)})
    output["rfft"]["fit_sensitivity"] = sensitivity
    sources = ["real_bsds_edm.json", "sweep_bdense_n120.json", "sweep_blow_n120.json"]
    output["source_sha256"] = {name: sha256(os.path.join(RESULTS, name)) for name in sources}
    output_path = os.path.join(RESULTS, args.out)
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=1)
    print("[real-BSDS]", json.dumps(output["real_image_sign"], indent=1))
    for label in ("rfft", "full_fft"):
        print("[point]", label, output[label]["three_family_mean"])
        print("[bootstrap mean]", label,
              output[label]["image_bootstrap"]["fits"]["three_family_mean"])
    print("[saved]", output_path)


if __name__ == "__main__":
    main()
