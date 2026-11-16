"""
Phase C v2 —— marginalized profile 与 basin collapse

复现 Morse-Bott 论文 Section 5.1/5.3/Fig.9-10：
    ell_y^opt(theta) = min_x  L(x, theta; y),
    L = (1/2 sigma^2)|| h_theta * x - y ||^2 + q(x)

【v1 失败的原因】直接对 L 做梯度下降数值上不可行：
    grad q = -(1/T_EPS^2)(D(x,T_EPS) - x)，而 1/T_EPS^2 = 2.5e5，
    x 离开图像流形后这一项爆炸，任何固定步长都发散。

【v2 的做法】改用作者官方代码 putils/optimization.py 里的 PnP-HQS 内循环
    （核固定，只优化 x）：
        x_hat = D(z, sigma_prox)                       # 先验 prox：一次去噪器调用
        z     = z + prox_l2(kernel, y, 2*x_hat - z, gamma) - x_hat
        z     = clamp(z, -0.2, 1.2)
    prox_l2 解 (A^T A + I/gamma) x = A^T y + z/gamma（共轭梯度）。
    这一步固有稳定——去噪器输出有界，数据步是凸二次问题。
    收敛后用作者的 likelihood_fn 算真实的 q(x_hat)，得到 profile 值。

用法： python repro_phaseC.py --smoke / [--nimg 3]
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
OUT = os.path.join(_PROJ, "results")
IMG_LO, IMG_HI = -1.0, 1.0        # EDM 模型输出在 [-1,1]；作者用 [-0.2,1.2] 是给 [0,1] 量纲的 score_sde


# --------------------------------------------------------------- 模糊算子
class BlurOp:
    def __init__(self, kernel_size=15):
        lin = torch.linspace(-(kernel_size // 2) + 1, kernel_size // 2, kernel_size,
                             device=DEV, dtype=DT)
        self.grid = lin[None] ** 2 + lin[:, None] ** 2

    def kernel(self, sigma):
        if sigma == 0:
            return None
        k = torch.exp(-self.grid / (2.0 * sigma ** 2))
        return (k / k.sum())[None, None]

    def _K(self, k, C, H, W):
        pad = torch.zeros((1, C, H, W), device=DEV, dtype=DT)
        kh, kw = k.shape[-2:]
        t, l = (H - kh) // 2, (W - kw) // 2
        pad[:, :, t:t + kh, l:l + kw] = k
        return torch.fft.rfft2(pad, dim=(-2, -1))

    def A(self, x, sigma):
        k = self.kernel(sigma)
        if k is None:
            return x
        B, C, H, W = x.shape
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1)) * self._K(k, C, H, W),
                                s=(H, W), dim=(-2, -1))

    def AT(self, x, sigma):
        k = self.kernel(sigma)
        if k is None:
            return x
        B, C, H, W = x.shape
        return torch.fft.irfft2(torch.conj(self._K(k, C, H, W))
                                * torch.fft.rfft2(x, dim=(-2, -1)), s=(H, W), dim=(-2, -1))

    def ATA_eig(self, sigma, C, H, W):
        """A^T A 在傅里叶域对角：|K|^2。用于解析解数据 prox。"""
        k = self.kernel(sigma)
        if k is None:
            return torch.ones((1, C, H, W // 2 + 1), device=DEV, dtype=DT)
        return self._K(k, C, H, W).abs() ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-star", type=float, default=1.5)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.smoke:
        nimg, noises = 1, [0.05, 0.30]
        thetas = [0.0, 0.5, 1.0, 1.5, 2.0]
        n_inner, cg_iter = 60, 60
    else:
        nimg, noises = args.nimg, [0.02, 0.05, 0.10, 0.20, 0.40]
        thetas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
        n_inner, cg_iter = 80, 100

    print(f"[cfg] nimg={nimg} noises={noises} theta*={args.sigma_star} "
          f"thetas={thetas} n_inner={n_inner}", flush=True)

    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    blur = BlurOp()
    print("[model] ready", flush=True)

    # ---------------- PnP-HQS：核固定，只优化 x ----------------
    @torch.no_grad()
    def hqs_x(theta, y, sn, n_inner):
        """返回 min 附近的 x_hat。

        HQS 分裂：f = (1/2 sigma^2)||Ax-y||^2, g = q。
          prox_{gamma f}(v) 解 (A^T A/sigma^2 + I/gamma) x = A^T y/sigma^2 + v/gamma
          prox_{gamma g}(z) ≈ 去噪器 D(z, sqrt(gamma))
        取 gamma = sigma^2（理论一致选择）=> 去噪器噪声水平恰为 sigma，
        且数据步化为 x_new = (A^T y + v)/(A^T A + I)（傅里叶域逐频率）。
        """
        B, C, H, W = y.shape
        ATy = blur.AT(y, theta)
        eig = blur.ATA_eig(theta, C, H, W)          # |K|^2
        denom = eig + 1.0                            # gamma = sigma^2 已约掉
        sigma_prox = sn                              # sqrt(gamma) = sigma
        z = y.clone()
        for _ in range(n_inner):
            x_hat = denoiser(z, sigma_prox).to(DT)   # 先验 prox
            v = 2 * x_hat - z
            x_new = torch.fft.irfft2(
                torch.fft.rfft2(ATy + v, dim=(-2, -1)) / denom, s=(H, W), dim=(-2, -1))
            z = torch.clamp(z + x_new - x_hat, IMG_LO, IMG_HI)
        return denoiser(z, sigma_prox).to(DT)

    @torch.no_grad()
    def q_value(x):
        return float(likelihood_fn(x.to(DT), ode_flow, t_spans=t_spans).item())

    # ---------------- 采样清晰图 ----------------
    samples, _ = ode_sampler(ode_flow, shape=(nimg, 3, 64, 64), t_spans=t_spans,
                             device=DEV)
    print(f"[sample] {nimg} sharp images", flush=True)

    records = []
    t0 = time.time()
    for i in range(nimg):
        x_star = samples[i:i + 1].to(DT)
        y_clean = blur.A(x_star, args.sigma_star)
        for sn in noises:
            g = torch.Generator(device=DEV).manual_seed(1000 * i + int(sn * 10000))
            y = (y_clean + sn * torch.randn(y_clean.shape, generator=g, device=DEV,
                                            dtype=DT)).to(DT)
            prof = []
            for th in thetas:
                t1 = time.time()
                xhat = hqs_x(th, y, sn, n_inner)
                resid = blur.A(xhat, th) - y
                rms = float(resid.pow(2).mean().sqrt().item())   # 应与 sn 同量级
                data = float((resid ** 2).sum().item() / (2 * sn ** 2))
                qv = q_value(xhat)
                prof.append(dict(theta=th, L=data + qv, data=data, q=qv))
                print(f"  img{i} noise={sn:<5} theta={th:<5} "
                      f"L*={data+qv:11.1f} (data={data:9.1f} q={qv:11.1f} rms={rms:.3f}) "
                      f"[{time.time()-t1:.0f}s]", flush=True)
                del xhat
                torch.cuda.empty_cache()
            records.append(dict(img=i, noise=sn, theta_star=args.sigma_star,
                                profile=prof))

            Ls = np.array([p["L"] for p in prof])
            ths = np.array([p["theta"] for p in prof])
            ig = int(np.argmin(Ls))
            istar = int(np.argmin(np.abs(ths - args.sigma_star)))
            if 0 < istar < len(Ls) - 1:
                local = bool(Ls[istar] <= Ls[istar - 1] and Ls[istar] <= Ls[istar + 1])
            else:
                local = bool(Ls[istar] <= Ls[min(istar + 1, len(Ls) - 1)])
            print(f"    => 全局极小 theta={ths[ig]} | theta*={args.sigma_star} "
                  f"局部极小: {local} | L(0)={Ls[0]:.0f} L(theta*)={Ls[istar]:.0f} "
                  f"差 {Ls[istar]-Ls[0]:+.0f}", flush=True)

    tag = "smoke" if args.smoke else "full"
    path = os.path.join(OUT, f"phaseC_{tag}.json")
    with open(path, "w") as f:
        json.dump(dict(cfg=dict(nimg=nimg, noises=noises, thetas=thetas,
                                sigma_star=args.sigma_star, n_inner=n_inner,
                                seed=args.seed), records=records), f, indent=1)
    print(f"\n[saved] {path}  ({time.time()-t0:.0f}s)", flush=True)
    free, total = torch.cuda.mem_get_info()
    print(f"[gpu] free {free/2**30:.2f}/{total/2**30:.2f} GiB, "
          f"peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
