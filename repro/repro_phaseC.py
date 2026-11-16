"""
Phase C v3 —— marginalized profile，细噪声扫描

复现 Morse-Bott 论文 Section 5.1/5.3/Fig.9-10：
    ell_y^opt(theta) = min_x  L(x, theta; y),
    L = (1/2 sigma^2)|| h_theta * x - y ||^2 + q(x)

【v1 失败】直接对 L 做梯度下降数值上不可行：
    grad q = -(1/T_EPS^2)(D(x,T_EPS) - x)，1/T_EPS^2 = 2.5e5，
    x 离开图像流形后这一项爆炸。改用作者的 PnP-HQS 内循环后稳定。

【v2 -> v3 的改动，依据 diag_phaseC.py 的诊断】
  * 诊断 A：n_inner=240 已收敛（60->240->600 波动 0.3%-1.5%）=> 固定 N_INNER=240
  * 诊断 B：多初值只值 1-3%，但论文用多初值 => 保留 2 个初值取最优
  * v2 与论文的最大分歧（论文说高噪声 basin collapse，我看到全局极小迁移到
    过度模糊）不是数值 artifact（诊断 A/B 已排除），最可能是噪声取点太稀。
    => 噪声改为 12 个对数间隔水平（0.01 -> 0.5），theta 网格细化到 0.125 步。

用法： python repro_phaseC.py --sweep [--nimg 2]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn, ode_sampler

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
IMG_LO, IMG_HI = -1.0, 1.0
OUT = os.path.join(_PROJ, "results")
N_INNER = 240


class BlurOp:
    def __init__(self, ks=15):
        lin = torch.linspace(-(ks // 2) + 1, ks // 2, ks, device=DEV, dtype=DT)
        self.grid = lin[None] ** 2 + lin[:, None] ** 2

    def kernel(self, s):
        if s == 0:
            return None
        k = torch.exp(-self.grid / (2.0 * s ** 2))
        return (k / k.sum())[None, None]

    def _K(self, k, C, H, W):
        pad = torch.zeros((1, C, H, W), device=DEV, dtype=DT)
        kh, kw = k.shape[-2:]
        t, l = (H - kh) // 2, (W - kw) // 2
        pad[:, :, t:t + kh, l:l + kw] = k
        return torch.fft.rfft2(pad, dim=(-2, -1))

    def A(self, x, s):
        k = self.kernel(s)
        if k is None:
            return x
        B, C, H, W = x.shape
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1)) * self._K(k, C, H, W),
                                s=(H, W), dim=(-2, -1))

    def AT(self, x, s):
        k = self.kernel(s)
        if k is None:
            return x
        B, C, H, W = x.shape
        return torch.fft.irfft2(torch.conj(self._K(k, C, H, W))
                                * torch.fft.rfft2(x, dim=(-2, -1)), s=(H, W), dim=(-2, -1))

    def eig(self, s, C, H, W):
        k = self.kernel(s)
        if k is None:
            return torch.ones((1, C, H, W // 2 + 1), device=DEV, dtype=DT)
        return self._K(k, C, H, W).abs() ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--nimg", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-star", type=float, default=1.5)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.sweep:
        noises = [round(float(v), 4) for v in
                  np.logspace(np.log10(0.01), np.log10(0.5), 12)]
        thetas = [round(0.125 * i, 4) for i in range(21)]     # 0 .. 2.5
    else:
        noises = [0.02, 0.05, 0.10, 0.20, 0.40]
        thetas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
    nimg = args.nimg

    n_evals = nimg * len(noises) * len(thetas) * 2
    print(f"[cfg] nimg={nimg} noises({len(noises)})={noises}", flush=True)
    print(f"[cfg] thetas({len(thetas)}) step={thetas[1]-thetas[0]} "
          f"theta*={args.sigma_star} n_inner={N_INNER} inits=2", flush=True)
    print(f"[cfg] 预计 {n_evals} 次 q 求值 x ~7s ≈ {n_evals*7/60:.0f} 分钟", flush=True)

    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    blur = BlurOp()

    @torch.no_grad()
    def hqs(theta, y, sn, z0):
        B, C, H, W = y.shape
        ATy = blur.AT(y, theta)
        denom = blur.eig(theta, C, H, W) + 1.0
        z = z0.clone()
        for _ in range(N_INNER):
            xh = denoiser(z, sn).to(DT)
            v = 2 * xh - z
            xn = torch.fft.irfft2(torch.fft.rfft2(ATy + v, dim=(-2, -1)) / denom,
                                  s=(H, W), dim=(-2, -1))
            z = torch.clamp(z + xn - xh, IMG_LO, IMG_HI)
        return denoiser(z, sn).to(DT)

    @torch.no_grad()
    def q_value(x):
        return float(likelihood_fn(x.to(DT), ode_flow, t_spans=t_spans).item())

    samples, _ = ode_sampler(ode_flow, shape=(nimg, 3, 64, 64), t_spans=t_spans,
                             device=DEV)
    print(f"[sample] {nimg} sharp images", flush=True)

    records = []
    t0 = time.time()
    for i in range(nimg):
        x_star = samples[i:i + 1].to(DT)
        y_clean = blur.A(x_star, args.sigma_star)
        for sn in noises:
            g = torch.Generator(device=DEV).manual_seed(1000 * i + int(sn * 100000))
            y = (y_clean + sn * torch.randn(y_clean.shape, generator=g, device=DEV,
                                            dtype=DT)).to(DT)
            prof = []
            for th in thetas:
                best = None
                for z0 in (y, blur.A(y, th)):
                    x = hqs(th, y, sn, z0)
                    resid = blur.A(x, th) - y
                    data = float((resid ** 2).sum().item() / (2 * sn ** 2))
                    q = q_value(x)
                    if best is None or data + q < best["L"]:
                        best = dict(theta=th, L=data + q, data=data, q=q)
                    del x
                    torch.cuda.empty_cache()
                prof.append(best)
            records.append(dict(img=i, noise=sn, theta_star=args.sigma_star,
                                profile=prof))

            Ls = np.array([p["L"] for p in prof])
            ths = np.array([p["theta"] for p in prof])
            ig = int(np.argmin(Ls))
            istar = int(np.argmin(np.abs(ths - args.sigma_star)))
            local = bool(0 < istar < len(Ls) - 1
                         and Ls[istar] <= Ls[istar - 1] and Ls[istar] <= Ls[istar + 1])
            near = np.abs(ths - args.sigma_star) <= 0.4
            ibasin = int(np.arange(len(ths))[near][np.argmin(Ls[near])])
            print(f"  img{i} sn={sn:<6} 全局极小 theta={ths[ig]:<6} | "
                  f"盆地在 theta={ths[ibasin]:<6} | "
                  f"L(0)-L(basin)={Ls[0]-Ls[ibasin]:+8.0f} | "
                  f"basin胜={'Y' if Ls[ibasin] < Ls[0] else 'N'} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    path = os.path.join(OUT, "phaseC_sweep.json")
    with open(path, "w") as f:
        json.dump(dict(cfg=dict(nimg=nimg, noises=noises, thetas=thetas,
                                sigma_star=args.sigma_star, n_inner=N_INNER,
                                n_init=2, seed=args.seed),
                       records=records), f, indent=1)
    print(f"\n[saved] {path}  ({time.time()-t0:.0f}s)", flush=True)
    free, total = torch.cuda.mem_get_info()
    print(f"[gpu] free {free/2**30:.2f}/{total/2**30:.2f} GiB, "
          f"peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
