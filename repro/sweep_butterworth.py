"""过渡带宽度受控实验：径向 Butterworth 核族。

动机：三模型分析显示幂律指数 b_f 与核的谱过渡带宽度相关 -0.75（n=5, p~0.14），
但那是观察性的、核之间差异太多。这里做受控实验：

    |h(rho)|^2 = 1 / (1 + (rho/rho_c)^{2n})

  * n    -> 滚降锐度。n=1 宽过渡，n->inf 趋于理想低通（硬阶跃）
  * rho_c-> 半功率频率

固定 n、扫 rho_c（5 个点）=> 可以拟合该 n 下的指数 b_n。
过渡带宽度（|h|^2 从 0.1 到 0.9 的径向频率宽度）有闭式：
    width = rho_c * (9^{1/(2n)} - 9^{-1/(2n)})
于是得到 5 个 (width, b_n) 点，全部来自同一族、同一支撑、同一各向同性，
唯一系统变化的是滚降锐度。

若 b_n 随 width 单调变化 => 机理成立（指数由过渡带锐度决定）。
若不变 => -0.75 是偶然，需换描述量。

谱掩模直接乘（等价于以 IFFT 掩模为像素核的循环卷积，与 blur_fft 一致）。

用法：
  python sweep_butterworth.py --smoke
  python sweep_butterworth.py --nimg 12 --orders 1 2 4 8 16 --rcs 4 6 8 11 15
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
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
H = W = 64
NB = 48
POOL = "/tmp/pool_ffhq64_256_s0.pt"


def mode_radii():
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)
    return R[:, :W // 2 + 1]                      # (H, W//2+1)


def radial_bins(nbin=NB):
    Rr = mode_radii()
    edges = np.linspace(0, Rr.max() + 1e-6, nbin + 1)
    idx = np.clip(np.digitize(Rr.ravel(), edges) - 1, 0, nbin - 1)
    return idx.reshape(Rr.shape), nbin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=12)
    ap.add_argument("--orders", nargs="*", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--rcs", nargs="*", type=float,
                    default=[4.0, 6.0, 8.0, 11.0, 15.0])
    ap.add_argument("--ts", nargs="*", type=float, default=[0.002, 0.02])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.smoke:
        orders, rcs, ts, nimg = [1, 4], [6.0, 11.0], [0.002], 2
    else:
        orders, rcs, ts, nimg = args.orders, args.rcs, args.ts, args.nimg

    print(f"[cfg] orders={orders} rcs={rcs} ts={ts} nimg={nimg}", flush=True)

    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    idx, _ = radial_bins()
    idx_t = torch.from_numpy(idx).to(DEV)

    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:nimg].clone()
    print(f"[pool] cached {tuple(pool.shape)}, using first {nimg}", flush=True)

    Rr = mode_radii()
    rho = torch.from_numpy(Rr).to(DEV)[None, None].to(torch.float64)   # (1,1,H,W//2+1)

    with torch.no_grad():
        P = (torch.fft.rfft2(pool, dim=(-2, -1)).abs() ** 2) / (H * W)
        Pc = P.mean(1)
        Srad = torch.zeros(NB, device=DEV, dtype=torch.float64)
        for b in range(NB):
            m = (idx_t == b).reshape(-1)
            if m.any():
                Srad[b] = Pc.reshape(P.shape[0], -1)[:, m].mean()
        Sfull = Srad[idx_t][None, None]
        Zx = (torch.fft.rfft2(xs, dim=(-2, -1)).abs() ** 2).mean(1, keepdim=True) / (H * W)

    B = args.batch

    @torch.no_grad()
    def logp_chunked(z, t):
        out = []
        for i in range(0, z.shape[0], B):
            nll = likelihood_fn(z[i:i + B].to(DT), ode_flow,
                                t_spans=[80.0, float(t)])
            out.extend([-float(v) for v in nll.reshape(-1)])
            torch.cuda.empty_cache()
        return np.array(out)

    lp_sharp = {t: logp_chunked(xs, t) for t in ts}
    for t in ts:
        print(f"[sharp] t={t} mean log p = {lp_sharp[t].mean():.1f}", flush=True)

    records = []
    for n in orders:
        for rc in rcs:
            with torch.no_grad():
                x = (rho / rc) ** (2 * n)
                h2 = 1.0 / (1.0 + x)                      # (1,1,H,W//2+1)
                d = 1.0 - h2
                kunif = float(d.mean())
                kappa = float((d * Sfull).sum() / Sfull.sum())
                # 闭式过渡带宽度（|h|^2 从 0.1 到 0.9）
                width = float(rc * (9.0 ** (1.0 / (2 * n)) - 9.0 ** (-1.0 / (2 * n))))
                Zy = None
                yb = torch.fft.irfft2(torch.fft.rfft2(xs.to(torch.float64),
                                                      dim=(-2, -1)) * h2,
                                      s=(H, W), dim=(-2, -1)).to(DT)
                Zy = (torch.fft.rfft2(yb, dim=(-2, -1)).abs() ** 2).mean(
                    1, keepdim=True) / (H * W)

            for t in ts:
                lp = logp_chunked(yb, t)
                lam = lp - lp_sharp[t]
                Sf = Sfull + t ** 2
                lam_g = float((((Zx - Zy) / (2 * Sf)).sum(dim=(1, 2, 3))).mean())
                rec = dict(family="butterworth", order=int(n), rho_c=float(rc),
                           theta=float(rc), t=float(t), kappa=kappa, kunif=kunif,
                           width=width,
                           lam_mean=float(lam.mean()), lam_std=float(lam.std()),
                           lam_min=float(lam.min()), lam_max=float(lam.max()),
                           lam_gauss_mean=lam_g,
                           logp_blur_mean=float(lp.mean()),
                           logp_sharp_mean=float(lp_sharp[t].mean()),
                           lam_per_img=[float(v) for v in lam])
                records.append(rec)
                print(f"  n={n:<3d} rho_c={rc:<5} t={t:<6} kunif={kunif:.4f} "
                      f"kappa={kappa:.4f} width={width:7.3f} "
                      f"Lambda={lam.mean():+10.1f} +/-{lam.std():7.1f} "
                      f"Gauss={lam_g:+8.1f}", flush=True)
                del lp
                torch.cuda.empty_cache()
            del yb
            torch.cuda.empty_cache()

    fn = os.path.join(OUT, "sweep_butterworth.json")
    with open(fn, "w") as f:
        json.dump(dict(cfg=dict(orders=orders, rcs=rcs, ts=ts, nimg=nimg,
                                ckpt=CKPT, pool="first nimg of 256-pool seed0"),
                       records=records), f, indent=1)
    print(f"\n[saved] {fn}", flush=True)


if __name__ == "__main__":
    main()
