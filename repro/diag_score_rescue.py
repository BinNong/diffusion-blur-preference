"""救活 score？在无回绕的纯谱算子上重跑 score vs 精确 Lambda 的对比。

背景（RESULTS §3.9(i)）：pointwise score 沿 alpha 线段剧烈振荡，根因是 15 边核的
循环卷积回绕（振荡比 2.01 / 3 次变号），换成纯谱高斯后变平滑（0.30 / 0 次）。
但"平滑"不等于"对"——还没验过 score 投影是否真的等于 dLambda/dalpha。

本脚本用无回绕的纯谱高斯（th 选成与 GaussianBlur(sigma=1) 的 kunif 匹配 = 1.4137，
另加一个 th=2.0 作第二形状），在同一条 alpha 线段上同时测：
    * 精确 Lambda(alpha)        （ODE + Hutchinson，粗网格）
    * <grad log p_t, delta1>    （细网格）
然后比三件事：
    A. score 投影 vs 数值导数 dLambda/dalpha     （应 = 1）
    B. score 的累计积分 vs Lambda                （微积分基本定理）
    C. score 剖面是否还振荡                       （振荡比、变号次数）

用法： python diag_score_rescue.py --smoke
      python diag_score_rescue.py --nimg 20
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

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")
H = W = 64
POOL = "/tmp/pool_ffhq64_256_s0.pt"

# 无回绕谱算子的 th；1.4137 与 GaussianBlur(sigma=1) 的 kunif=0.9185 匹配
KERNELS = [("specgauss", 1.4137), ("specgauss", 2.0)]
# score 用细网格（便宜，只前向），Lambda 用同一网格但只取部分点算 ODE
GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def spectral_mask(th):
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    f2 = (FX ** 2 + FY ** 2)[:, :W // 2 + 1] / (H * W)
    amp = np.sqrt(np.exp(-2 * np.pi ** 2 * th ** 2 * f2))
    return torch.from_numpy(amp).to(DEV)[None, None].to(torch.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=20)
    ap.add_argument("--t", type=float, default=0.002)
    ap.add_argument("--kernels", nargs="*", default=None)
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    nimg = 2 if args.smoke else args.nimg
    kers = KERNELS if args.kernels is None else \
        [(s.split(":")[0], float(s.split(":")[1])) for s in args.kernels]

    model = load_local_model(CKPT).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x.to(DT), t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)

    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:nimg].clone()
    t2 = args.t ** 2
    A = GRID

    @torch.no_grad()
    def logp(z):
        out = []
        for i in range(0, z.shape[0], 4):
            nll = likelihood_fn(z[i:i + 4].to(DT), ode_flow,
                                t_spans=[80.0, args.t])
            out.extend([-float(v) for v in nll.reshape(-1)])
            torch.cuda.empty_cache()
        return np.array(out)

    @torch.no_grad()
    def score_proj(z, d):
        return ((model(z, args.t) - z) / t2).flatten(1).mul(
            d.flatten(1)).sum(1).cpu().numpy()

    lp_sharp = logp(xs)
    rows = []
    for fam, th in kers:
        amp = spectral_mask(th)
        with torch.no_grad():
            y1 = torch.fft.irfft2(torch.fft.rfft2(xs.to(torch.float64),
                                                  dim=(-2, -1)) * amp,
                                  s=(H, W), dim=(-2, -1)).to(DT)
        d1 = y1 - xs
        kunif1 = float((1.0 - amp.to(torch.float64) ** 2).mean())

        print(f"\n{'='*104}\n [{fam}:{th:g}]  kunif(1) = {kunif1:.4f}   "
              f"t={args.t:g}\n{'='*104}")
        print(f"{'alpha':>6} | {'Lambda':>10} | {'<g,d1>':>10} | "
              f"{'dL/da 数值':>11} | {'比值':>7} | {'累计积分':>10} | {'积分/L':>7}")
        print("-" * 82)

        sp = np.zeros((nimg, len(A)))
        lp = np.zeros((nimg, len(A)))
        for j, a in enumerate(A):
            ya = (1.0 - a) * xs + a * y1
            sp[:, j] = score_proj(ya, d1)
            if a in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                lp[:, j] = logp(ya)
            torch.cuda.empty_cache()
        lam = lp - lp[:, :1]

        # 累计积分（梯形）
        cum = np.zeros((nimg, len(A)))
        for j in range(1, len(A)):
            cum[:, j] = cum[:, j - 1] + 0.5 * (sp[:, j] + sp[:, j - 1]) \
                * (A[j] - A[j - 1])

        for j, a in enumerate(A):
            if j == 0:
                print(f"{a:>6g} | {lam[:,j].mean():>10.1f} | {sp[:,j].mean():>10.0f} | "
                      f"{'--':>11} | {'--':>7} | {cum[:,j].mean():>10.1f} | {'--':>7}")
                continue
            dlam = (lam[:, j].mean() - lam[:, j - 1].mean()) / (A[j] - A[j - 1])
            ratio = sp[:, j].mean() / dlam if abs(dlam) > 1e-9 else float("nan")
            ir = cum[:, j].mean() / lam[:, j].mean() \
                if abs(lam[:, j].mean()) > 1e-9 else float("nan")
            print(f"{a:>6g} | {lam[:,j].mean():>10.1f} | {sp[:,j].mean():>10.0f} | "
                  f"{dlam:>11.0f} | {ratio:>7.2f} | {cum[:,j].mean():>10.1f} | "
                  f"{ir:>7.2f}")

        v = sp.mean(0)
        dv = np.diff(v)
        print("-" * 82)
        print(f" 振荡比 RMS(相邻差)/|<proj>| = "
              f"{np.sqrt((dv**2).mean())/np.abs(v).mean():.2f}   变号次数 = "
              f"{int((np.diff(np.sign(v)) != 0).sum())}")
        print(f" alpha=1 处：Lambda={lam[:,-1].mean():.1f}，"
              f"累计积分={cum[:,-1].mean():.1f}，"
              f"积分/Lambda = {cum[:,-1].mean()/lam[:,-1].mean():.3f}")
        for j, a in enumerate(A):
            for i in range(nimg):
                rows.append(dict(family=fam, theta=float(th), alpha=float(a),
                                 t=args.t, img=i, lam=float(lam[i, j]),
                                 proj=float(sp[i, j]), cum=float(cum[i, j])))
        del y1, d1
        torch.cuda.empty_cache()

    outfn = os.path.join(OUT, f"diag_score_rescue_t{args.t:g}.json")
    with open(outfn, "w") as f:
        json.dump(dict(cfg=dict(nimg=nimg, kernels=kers, t=args.t,
                                grid=A, ckpt=CKPT), rows=rows), f, indent=1)
    print(f"\n[saved] {outfn}")


if __name__ == "__main__":
    main()
