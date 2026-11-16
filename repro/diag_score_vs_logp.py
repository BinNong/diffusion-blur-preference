"""诊断：score 投影 <grad log p_t(y_alpha), delta1> 是否等于 dLambda/dalpha？

微积分基本定理说  Lambda(alpha) = int_0^alpha <grad log p_t(y_a), delta1> da
（因为 dy_a/dalpha = delta1）。所以左边对 alpha 求导就应该等于右边的被积函数。

但 probe_score_integral 发现线积分复现不出精确 Lambda。这个脚本直接比
    * 精确 log p_t 在 alpha = 0, .2, .4, .6, .8, 1.0 的值（ODE + Hutchinson）
    * 同一点的 score 投影 <grad log p_t, delta1>
若两者对不上，就逐点看从哪个 alpha 开始背离，从而定位是 score 错还是 logp 错。

用法： python diag_score_vs_logp.py --smoke
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn
from putils.convolution import GaussianBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
H = W = 64
KS = 15
POOL = "/tmp/pool_ffhq64_256_s0.pt"


def blur(x, k):
    if k.ndim == 2:
        k = k[None, None]
    kh, kw = k.shape[-2:]
    pad = torch.zeros((1, x.shape[1], H, W), device=x.device, dtype=x.dtype)
    top, left = (H - kh) // 2, (W - kw) // 2
    pad[:, :, top:top + kh, left:left + kw] = k.to(x.dtype)
    return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))
                            * torch.fft.rfft2(pad, dim=(-2, -1)),
                            s=(H, W), dim=(-2, -1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=4)
    ap.add_argument("--t", type=float, default=0.002)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    nimg = 1 if args.smoke else args.nimg

    model = load_local_model(CKPT).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x.to(DT), t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)

    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:nimg].clone()
    k = GaussianBlur(kernel_size=KS, device=DEV, dtype=DT).kernel(args.sigma)
    y1 = blur(xs, k)
    d1 = y1 - xs
    t2 = args.t ** 2

    @torch.no_grad()
    def logp(z):
        """分批，避免 ODE 求解器一次性持有全批状态导致 OOM。"""
        out = []
        for i in range(0, z.shape[0], 4):
            nll = likelihood_fn(z[i:i + 4].to(DT), ode_flow,
                                t_spans=[80.0, args.t])
            out.extend([-float(v) for v in nll.reshape(-1)])
            torch.cuda.empty_cache()
        return np.array(out)

    @torch.no_grad()
    def score_proj(z, d):
        """<grad log p_t(z), d> = <D(z,t)-z, d>/t^2"""
        return float((model(z, args.t) - z).flatten().dot(d.flatten())) / t2

    A = args.alphas
    print(f"t={args.t:g}  sigma={args.sigma:g}  nimg={nimg}")
    print(f"{'alpha':>6} | {'log p_t':>12} | {'Lambda':>10} | "
          f"{'<g,d1>':>12} | {'dLambda/da(数值)':>16}")
    print("-" * 70)
    lp = np.zeros((nimg, len(A)))
    sp = np.zeros((nimg, len(A)))
    for j, a in enumerate(A):
        ya = (1.0 - a) * xs + a * y1
        lp[:, j] = logp(ya)
        for i in range(nimg):
            sp[i, j] = score_proj(ya[i:i + 1], d1[i:i + 1])
        torch.cuda.empty_cache()

    lam = lp - lp[:, :1]
    for j, a in enumerate(A):
        cells = [f"{a:>6g} | {lp[:,j].mean():>12.1f} | {lam[:,j].mean():>10.1f} | "
                 f"{sp[:,j].mean():>12.1f} |"]
        if j > 0:
            da = A[j] - A[j - 1]
            cells.append(f"{(lam[:,j].mean()-lam[:,j-1].mean())/da:>16.1f}")
        else:
            cells.append(f"{'--':>16}")
        print(" ".join(cells))
    print("-" * 70)
    print("判据：第三列 <g,d1> 应当等于第四列 dLambda/dalpha（数值微分）。")
    print("逐图版本：")
    for i in range(nimg):
        print(f"  img{i}: " + "  ".join(
            f"a={a:g}:L={lam[i,j]:+9.0f},g={sp[i,j]:+9.0f}"
            for j, a in enumerate(A)))


if __name__ == "__main__":
    main()
