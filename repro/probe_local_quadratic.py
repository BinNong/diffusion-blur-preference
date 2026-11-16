r"""Directional local-quadratic test of the ODE log-density along actual blur.

Pre-registered 2026-11-11 before results. Diagnostic, no sign or error threshold.

Q0: use EDM/FFHQ, Gaussian sigma=1.5, t in {0.002,0.2}; two input sources are the
    previously used generated pool[255] and real BSDS300 test image 0. Both are identified.
Q1: for delta=h*x-x, evaluate q_t(x+s*delta) at s={-1,-0.5,-0.25,0,0.25,0.5,1}.
    One fixed Rademacher vector is shared across all points within a line, so the line
    comparison is not dominated by independent trace draws. No pixel clipping is applied.
Q2: estimate the directional gradient and Hessian curvature from s=+-0.5,
    g=(q(+0.5)-q(-0.5))/(2*0.5), H=(q(+0.5)+q(-0.5)-2q(0))/0.5^2.
    Predict q(1)-q(0) by g+0.5H and compare with the independently evaluated q(1).
    Also report the +-0.25 estimate. This is a one-direction Taylor test, not a full-Hessian
    approximation and not a theorem about all local second-order methods.

Run: ./venv2/bin/python repro/probe_local_quadratic.py
"""

import json
import os
import sys
import time

import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "repro"))
from review_numeric_checks import apply_power, kernel_power
from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn


def main():
    generated = torch.load("/tmp/pool_ffhq64_256_s0.pt", map_location="cpu")[255:256]
    real = torch.load(os.path.join(_PROJ, "data", "bsds300_real_test_64.pt"),
                      map_location="cpu")[:1]
    inputs = [("edm_generated_255", generated), ("bsds_real_0", real)]
    model = load_local_model("edm-ffhq-64x64-uncond-vp.pkl").cuda().eval()
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x, t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)
    g2 = kernel_power("gaussian", 1.5)
    result = {"cfg": {"checkpoint": "edm-ffhq-64x64-uncond-vp.pkl", "kernel": "Gaussian 1.5",
                      "ts": [0.002, 0.2], "s": [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
                      "rtol": 1e-5, "atol": 1e-5, "method": "RK45",
                      "preregistered": "2026-11-11", "scope": "one blur direction only"},
              "rows": []}
    start = time.time()
    for source, image in inputs:
        sharp = image.cuda()
        delta = apply_power(sharp, g2).to(torch.float32)-sharp
        for t in result["cfg"]["ts"]:
            torch.manual_seed(20261111)
            epsilon = torch.randint_like(sharp, low=0, high=2).to(torch.float32)*2-1
            original_randint = torch.randint_like

            def fixed_randint_like(tensor, low=0, high=2):
                return ((epsilon+1.0)/2.0).to(tensor.dtype)

            values = {}
            try:
                torch.randint_like = fixed_randint_like
                for s in result["cfg"]["s"]:
                    with torch.no_grad():
                        q = float(likelihood_fn(sharp+s*delta, ode_flow,
                                               t_spans=[80.0, float(t)]).item())
                    values[str(s)] = q
                    print("[Q1]", source, t, s, q, round(time.time()-start), flush=True)
            finally:
                torch.randint_like = original_randint
            q0 = values["0.0"]
            qp, qm = values["1.0"], values["-1.0"]
            predictions = {}
            for half in (0.25, 0.5):
                plus, minus = values[str(half)], values[str(-half)]
                gradient = (plus-minus)/(2*half)
                curvature = (plus+minus-2*q0)/(half*half)
                predicted = gradient+0.5*curvature
                predictions[str(half)] = {"gradient": gradient, "curvature": curvature,
                                          "predicted_delta_q_at_1": predicted,
                                          "absolute_error": predicted-(qp-q0)}
            row = {"source": source, "t": t, "q": values,
                   "actual_delta_q_at_1": qp-q0, "actual_lambda": q0-qp,
                   "actual_delta_q_at_minus1": qm-q0, "predictions": predictions}
            result["rows"].append(row)
            with open(os.path.join(_PROJ, "results", "local_quadratic.json"), "w") as handle:
                json.dump(result, handle, indent=1)
            print("[Q2]", source, t, "actual", qp-q0,
                  "prediction", predictions["0.5"], flush=True)
        del sharp, delta
        torch.cuda.empty_cache()
    print("[saved] local_quadratic.json", flush=True)


if __name__ == "__main__":
    main()
