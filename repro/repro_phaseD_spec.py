"""
Phase D v3 —— 高斯谱近似下的证据 p(theta|y)

动机：Phase D v2 证明在 12288 维空间用先验做重要性采样 ESS=1，不可行。
      但有一个便宜的近似能抓住核心机制。

模型： y = A_theta * x + eta,  x ~ p_t,  eta ~ N(0, sigma_n^2 I)
近似： 把 p_t 换成同功率谱的高斯 N(0, Sigma_t)。A_theta 是循环卷积（傅里叶对角），
      于是 y 也是高斯，对数证据有闭式：

  log p(y|theta) ≈ -sum_w [ |yhat_w|^2 / (2 (|h_w|^2 S_w + sigma_n^2))
                            + 0.5*log(2*pi*(|h_w|^2 S_w + sigma_n^2)) ]

关键机制（Occam 因子）：
  * 似然项 |yhat|^2/(2(...))：被核削弱的频率，方差小，y 在那里有能量就被重罚
  * 归一化项 0.5*log(...)：模型必须为每个频率"付费"
  两者共同实现"协方差匹配"——谁的隐含协方差 |h_theta|^2 S + sigma^2
  最接近 y 的真实协方差，谁就赢。这正是可能区分 theta* 与 theta=0 的机制。

谱估计：自然图像谱随频率衰减且近似各向同性，故把周期图按【径向频率】平均
        （64x64 有 4096 个频率/通道，64 张图就有 26 万个周期图值，
         按径向分箱后每箱数千个样本，估计稳定）。

用法： python repro_phaseD_spec.py [--nsample 64]
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
from edm.prior import ScorePrior, get_ode_flow, ode_sampler

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
H = W = 64


class BlurOp:
    def __init__(self, ks=15):
        lin = torch.linspace(-(ks // 2) + 1, ks // 2, ks, device=DEV, dtype=DT)
        self.grid = lin[None] ** 2 + lin[:, None] ** 2

    def kernel(self, s):
        if s == 0:
            return None
        k = torch.exp(-self.grid / (2.0 * s ** 2))
        return (k / k.sum())[None, None]

    def Khat(self, s):
        """核的傅里叶响应，zero-pad 到 64x64 后 rfft2。theta=0 -> 全 1。"""
        k = self.kernel(s)
        if k is None:
            return torch.ones((1, 1, H, W // 2 + 1), device=DEV, dtype=DT)
        pad = torch.zeros((1, 1, H, W), device=DEV, dtype=DT)
        kh, kw = k.shape[-2:]
        t, l = (H - kh) // 2, (W - kw) // 2
        pad[:, :, t:t + kh, l:l + kw] = k
        return torch.fft.rfft2(pad, dim=(-2, -1))

    def A(self, x, s):
        k = self.kernel(s)
        if k is None:
            return x
        B, C, Hh, Ww = x.shape
        pad = torch.zeros((1, C, Hh, Ww), device=DEV, dtype=DT)
        kh, kw = k.shape[-2:]
        t, l = (Hh - kh) // 2, (Ww - kw) // 2
        pad[:, :, t:t + kh, l:l + kw] = k
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))
                                * torch.fft.rfft2(pad, dim=(-2, -1)),
                                s=(Hh, Ww), dim=(-2, -1))


def radial_bins(nbin=48):
    """径向频率分箱：返回每个 rfft2 频率点的径向索引和频率值。"""
    fx = np.fft.fftfreq(H) * H          # cycles per image
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)      # (H, W)
    Rr = R[:, :W // 2 + 1]              # rfft2 只取非负 fy
    edges = np.linspace(0, Rr.max() + 1e-6, nbin + 1)
    idx = np.digitize(Rr.ravel(), edges) - 1
    idx = np.clip(idx, 0, nbin - 1)
    rho = np.array([Rr.ravel()[idx == b].mean() if (idx == b).any() else 0.0
                    for b in range(nbin)])
    return idx.reshape(Rr.shape), rho, nbin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsample", type=int, default=64)
    ap.add_argument("--nimg", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma-star", type=float, default=1.5)
    ap.add_argument("--t", type=float, default=0.0, help="先验平滑；0 = 用 p_0 样本谱")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    noises = [0.02, 0.06, 0.12, 0.25]
    thetas = [round(0.125 * i, 4) for i in range(21)]
    S = args.nsample

    print(f"[cfg] S={S} t={args.t} noises={noises} theta*={args.sigma_star}", flush=True)

    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    blur = BlurOp()
    idx, rho, NB = radial_bins()

    # ---------- 1. 生成样本并估计径向功率谱 S(rho) ----------
    t0 = time.time()
    pool, _ = ode_sampler(ode_flow, shape=(S + args.nimg, 3, 64, 64), t_spans=t_spans,
                          device=DEV)
    pool = pool.to(DT)
    x_stars = pool[S:]
    xs = pool[:S]
    print(f"[pool] {S}+{args.nimg} 张，{time.time()-t0:.0f}s", flush=True)

    with torch.no_grad():
        # 周期图：|FFT|^2 / (H*W)  —— 每频率每通道的功率
        P = (torch.fft.rfft2(xs, dim=(-2, -1)).abs() ** 2) / (H * W)   # (S,C,H,W/2+1)
        Pc = P.mean(1)                                                  # 平均通道 -> (S,H,W/2+1)
        # 按径向分箱平均
        Srad = torch.zeros(NB, device=DEV, dtype=DT)
        cnt = torch.zeros(NB, device=DEV, dtype=DT)
        flat = Pc.reshape(-1)
        for b in range(NB):
            m = torch.from_numpy((idx == b)).to(DEV)
            if m.any():
                Srad[b] = flat.reshape(S, -1)[:, m.reshape(-1)].mean()
                cnt[b] = m.sum()
        # 若先验平滑 t>0，功率谱加 t^2（加性高斯噪声）
        if args.t > 0:
            Srad = Srad + args.t ** 2
    print(f"[spectrum] 径向谱已估计，S 范围 [{float(Srad.min()):.2e}, "
          f"{float(Srad.max()):.2e}]", flush=True)
    # 把径向谱广播回 (1,1,H,W/2+1)
    Sfull = Srad[torch.from_numpy(idx).to(DEV)][None, None]

    # ---------- 2. 对每个 theta 算证据 ----------
    records = []
    for i in range(args.nimg):
        x_star = x_stars[i:i + 1].to(DT)
        y_clean = blur.A(x_star, args.sigma_star)
        for sn in noises:
            gg = torch.Generator(device=DEV).manual_seed(1000 * i + int(sn * 100000))
            y = (y_clean + sn * torch.randn(y_clean.shape, generator=gg, device=DEV,
                                            dtype=DT)).to(DT)
            with torch.no_grad():
                Yh = torch.fft.rfft2(y, dim=(-2, -1))                  # (1,C,H,W/2+1)
                Yp = (Yh.abs() ** 2).mean(1, keepdim=True) / (H * W)   # 平均通道
                logps = []
                for th in thetas:
                    h2 = blur.Khat(th).abs() ** 2                      # (1,1,H,W/2+1)
                    var = h2 * Sfull + sn ** 2
                    lp = float(-(Yp / (2 * var)).sum()
                               - 0.5 * torch.log(2 * np.pi * var).sum())
                    logps.append(lp)
            logps = np.array(logps)
            lp = logps - logps.max()
            post = np.exp(lp)
            post /= post.sum()
            cdf = np.cumsum(post)
            q = [float(thetas[min(np.searchsorted(cdf, p), len(thetas) - 1)])
                 for p in (0.05, 0.5, 0.95)]
            mode_th = thetas[int(np.argmax(post))]
            near = np.abs(np.array(thetas) - args.sigma_star) <= 0.3
            # 后验熵（衡量确定性）
            ent = float(-(post[post > 0] * np.log(post[post > 0])).sum())
            rec = dict(img=i, noise=sn, t=args.t, theta_star=args.sigma_star,
                       thetas=thetas, logp=logps.tolist(), post=post.tolist(),
                       mode=mode_th, q05=q[0], q50=q[1], q95=q[2],
                       mass_near=float(post[near].sum()), entropy=ent,
                       w0_rel=float(post[0] / post.max()))
            records.append(rec)
            print(f"  img{i} sn={sn:<6} 证据后验: mode={mode_th:.3f} "
                  f"90%CI=[{q[0]:.2f},{q[2]:.2f}] | theta*±0.3 质量="
                  f"{rec['mass_near']:.2f} | 熵={ent:.2f} | w(0)/w(max)={rec['w0_rel']:.3f}",
                  flush=True)
            del y
            torch.cuda.empty_cache()

    path = os.path.join(OUT, f"phaseD_spec_t{args.t}.json")
    with open(path, "w") as f:
        json.dump(dict(cfg=dict(S=S, t=args.t, noises=noises, thetas=thetas,
                                sigma_star=args.sigma_star, seed=args.seed),
                       spectrum_rho=rho.tolist(), spectrum_S=Srad.tolist(),
                       records=records), f)
    print(f"\n[saved] {path}", flush=True)


if __name__ == "__main__":
    main()
