"""解耦实验：谱过渡带宽度 vs 阻带尾，哪个决定幂律指数？

背景：
  * 受控 Butterworth 实验：指数随过渡带宽度单调下降，corr = -0.957
  * 但 Butterworth 里宽度与尾部衰减速率是锁在一起的（width ~ 1/n，尾 ~ 1/rho^{2n}）
  * defocus 是唯一离群点：与 gaussian 过渡带宽度几乎相同（9.27 vs 6.86 修正后），
    指数却高 1.58。诊断显示两者通过带一致到 1-2%，但阻带相差 10-87 倍
    （高斯下溢到 0，Bessel 尾按 1/rho^6 存活）

本实验把两者解耦：
    |h(rho)|^2 = (1-delta) * exp(-2 pi^2 sigma^2 f^2) + delta

  * sigma  -> 过渡带宽度（delta <= 0.01 时穿越点几乎不动）
  * delta  -> 阻带底板（深阻带里 |h|^2 = delta）
  * DC 处恰好为 1，全程 <= 1，是合法的低通算子
  * 像素空间等价于 (1-delta)*高斯模糊 + delta*恒等（部分模糊）

固定 sigma、扫 delta => 若指数随 delta 变化，则阻带尾是第二个决定因素；
若不变，则 defocus 的离群另有原因。

用法： python sweep_tail.py --smoke
      python sweep_tail.py --sigmas 1.0 1.5 2.0 3.0 --deltas 0 1e-6 1e-5 1e-4 1e-3
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


def mode_grid():
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    return FX / H, FY / H, np.sqrt(FX ** 2 + FY ** 2)      # f_x, f_y, rho


def radial_bins(nbin=NB):
    _, _, R = mode_grid()
    Rr = R[:, :W // 2 + 1]
    edges = np.linspace(0, Rr.max() + 1e-6, nbin + 1)
    idx = np.clip(np.digitize(Rr.ravel(), edges) - 1, 0, nbin - 1)
    return idx.reshape(Rr.shape), nbin, Rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=20)
    ap.add_argument("--sigmas", nargs="*", type=float,
                    default=[1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--deltas", nargs="*", type=float,
                    default=[0.0, 1e-6, 1e-5, 1e-4, 1e-3])
    ap.add_argument("--ts", nargs="*", type=float, default=[0.002, 0.02])
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    if args.smoke:
        sigs, dels, ts, nimg = [1.5], [0.0, 1e-4], [0.002], 2
    else:
        sigs, dels, ts, nimg = args.sigmas, args.deltas, args.ts, args.nimg

    print(f"[cfg] sigmas={sigs} deltas={dels} ts={ts} nimg={nimg}", flush=True)

    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    idx, _, Rr = radial_bins()
    idx_t = torch.from_numpy(idx).to(DEV)

    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:nimg].clone()
    print(f"[pool] cached {tuple(pool.shape)}, using first {nimg}", flush=True)

    FX, FY, R = mode_grid()
    f2 = (FX ** 2 + FY ** 2)[:, :W // 2 + 1]               # (H, W//2+1)
    f2_t = torch.from_numpy(f2).to(DEV)[None, None].to(torch.float64)

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

    def cross_first(prof, rho, level):
        for i in range(len(prof) - 1):
            a, b = prof[i] - level, prof[i + 1] - level
            if a * b <= 0 and a != b:
                return rho[i] + (level - prof[i]) * (rho[i + 1] - rho[i]) / \
                    (prof[i + 1] - prof[i])
        return float("nan")

    rho_b = np.array([Rr.ravel()[idx.ravel() == b].mean() if (idx == b).any() else 0.0
                      for b in range(NB)])

    records = []
    for sg in sigs:
        for dl in dels:
            with torch.no_grad():
                g = torch.exp(-2 * np.pi ** 2 * sg ** 2 * f2_t)
                h2 = (1.0 - dl) * g + dl
                d = 1.0 - h2
                kunif = float(d.mean())
                kappa = float((d * Sfull).sum() / Sfull.sum())
                # 过渡带宽度（径向分桶后取首次穿越）
                h2n = h2[0, 0].cpu().numpy()
                hb = np.array([h2n[idx == b].mean() if (idx == b).any() else 0.0
                               for b in range(NB)])
                r9 = cross_first(hb, rho_b, 0.1)
                r1 = cross_first(hb, rho_b, 0.9)
                width = float(r9 - r1) if (np.isfinite(r9) and np.isfinite(r1)) \
                    else float("nan")
                tail = float(h2n[idx == NB - 1].mean()) if (idx == NB - 1).any() \
                    else float("nan")
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
                records.append(dict(family="gauss_floor", sigma=float(sg),
                                    delta=float(dl), theta=float(sg), t=float(t),
                                    kappa=kappa, kunif=kunif, width=width, tail=tail,
                                    lam_mean=float(lam.mean()), lam_std=float(lam.std()),
                                    lam_min=float(lam.min()), lam_max=float(lam.max()),
                                    lam_gauss_mean=lam_g,
                                    logp_blur_mean=float(lp.mean()),
                                    logp_sharp_mean=float(lp_sharp[t].mean()),
                                    lam_per_img=[float(v) for v in lam]))
                print(f"  sigma={sg:<5} delta={dl:<8g} t={t:<6} kunif={kunif:.4f} "
                      f"width={width:7.3f} tail={tail:.2e} "
                      f"Lambda={lam.mean():+10.1f} +/-{lam.std():7.1f} "
                      f"Gauss={lam_g:+8.1f}", flush=True)
                del lp
                torch.cuda.empty_cache()
            del yb
            torch.cuda.empty_cache()

    fn = os.path.join(OUT, "sweep_tail.json")
    with open(fn, "w") as f:
        json.dump(dict(cfg=dict(sigmas=sigs, deltas=dels, ts=ts, nimg=nimg,
                                ckpt=CKPT, pool="first nimg of 256-pool seed0"),
                       records=records), f, indent=1)
    print(f"\n[saved] {fn}", flush=True)


if __name__ == "__main__":
    main()
