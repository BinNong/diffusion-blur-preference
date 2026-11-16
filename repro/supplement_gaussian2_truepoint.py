"""Add the accidentally omitted theta=2.0 true-strength point to the Gaussian-2 arm.

The original 21-point grid did not contain 2.0. This post-collection correction evaluates
only the missing true point with the frozen 240-iteration/two-start protocol and writes a
new file. Existing points are not edited or removed, and the corrected argmin is recomputed
transparently by the offline judge.
"""

import json
import os
import sys
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "repro"))
from sweep_overblur_generalization import Blur, CKPT, POOL, DEV, DT, N_INNER
from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn


def main():
    source = os.path.join(_PROJ, "results", "overblur_generalization.json")
    with open(source) as handle:
        result = json.load(handle)
    model = load_local_model(CKPT).to(DEV).eval()
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x.to(DT), t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)
    pool = torch.load(POOL, map_location=DEV).to(DT)
    blur = Blur("gaussian")

    @torch.no_grad()
    def hqs(theta, y, noise, initial):
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

    count = 0
    for record in result["records"]:
        if record["family"] != "gaussian" or record["theta_star"] != 2.0:
            continue
        if any(point["theta"] == 2.0 for point in record["profile"]):
            continue
        image = pool[record["img"]:record["img"]+1]
        clean = blur.A(image, 2.0)
        noise = record["noise"]
        generator = torch.Generator(device=DEV).manual_seed(
            1000*record["img"]+int(noise*100000))
        y = (clean+noise*torch.randn(clean.shape, generator=generator,
                                     device=DEV, dtype=DT)).to(DT)
        best = None
        for initial in (y, blur.A(y, 2.0)):
            x = hqs(2.0, y, noise, initial)
            residual = blur.A(x, 2.0)-y
            data = float(residual.square().sum()/(2*noise**2))
            q = q_value(x, noise)
            if best is None or data+q < best["L"]:
                best = {"theta": 2.0, "L": data+q, "data": data, "q": q}
        record["profile"].append(best)
        record["profile"].sort(key=lambda point: point["theta"])
        record["true_point_supplement"] = "added 2026-11-11 to repair omitted theta=2.0"
        count += 1
        print("[supplement]", count, "img", record["img"], "noise", noise,
              "L", best["L"], flush=True)
    if count != 16:
        raise RuntimeError("Expected 16 missing Gaussian-2 true points, got %d" % count)
    result["cfg"]["true_point_supplement"] = {
        "family": "gaussian", "theta_star": 2.0, "n": count,
        "reason": "original 0:0.175:3.5 grid omitted the true value"}
    path = os.path.join(_PROJ, "results", "overblur_generalization_completed.json")
    with open(path, "w") as handle:
        json.dump(result, handle, indent=1)
    print("[saved]", path, flush=True)


if __name__ == "__main__":
    main()
