"""验证高斯谱近似：在 theta=0 处与精确值对比。

theta=0 时模型是 y = x + eta, x~p_t, eta~N(0, sigma_n^2 I)，
所以 p(y) = (p_t * N(0,sigma_n^2))(y) = p_{sqrt(t^2+sigma_n^2)}(y)
—— 这可以用作者的 likelihood_fn 【精确】算出（~4s）。

高斯谱近似在同一点给出闭式：
    log p(y|0) ≈ -sum_w [ |yhat_w|^2/(2(S_w+sigma_n^2)) + 0.5 log(2pi(S_w+sigma_n^2)) ]

两者之差就是近似在该点的误差。判据：
  * 若 |误差| << |log 证据比(theta=0 vs theta*)|，则"否决 theta=0"的结论不受近似影响
  * 若误差与证据比同量级或更大，则结论不可靠

附带做二阶统计校验：生成 y_s = A_theta* x_s + eta_s，
比较其经验谱与模型预测谱 |h_theta*|^2 S + sigma_n^2。
高斯近似只用到二阶矩，所以这项校验通过 => 近似在二阶结构上是精确的，
误差全部来自高阶（非高斯）结构。

用法： python validate_spec_approx.py
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
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)
    Rr = R[:, :W // 2 + 1]
    edges = np.linspace(0, Rr.max() + 1e-6, nbin + 1)
    idx = np.clip(np.digitize(Rr.ravel(), edges) - 1, 0, nbin - 1)
    rho = np.array([Rr.ravel()[idx == b].mean() if (idx == b).any() else 0.0
                    for b in range(nbin)])
    return idx.reshape(Rr.shape), rho, nbin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsample", type=int, default=64)
    ap.add_argument("--sigma-star", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    noises = [0.02, 0.06, 0.12, 0.25]
    thetas = [0.0, 0.75, 1.5, 2.25]
    S = args.nsample
    TSTAR = args.sigma_star

    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    blur = BlurOp()
    idx, rho, NB = radial_bins()
    idx_t = torch.from_numpy(idx).to(DEV)

    t0 = time.time()
    pool, _ = ode_sampler(ode_flow, shape=(S + 1, 3, 64, 64), t_spans=t_spans, device=DEV)
    pool = pool.to(DT)
    x_star = pool[S:S + 1]
    xs = pool[:S]
    print(f"[pool] {S}+1 张，{time.time()-t0:.0f}s", flush=True)

    with torch.no_grad():
        P = (torch.fft.rfft2(xs, dim=(-2, -1)).abs() ** 2) / (H * W)
        Pc = P.mean(1)
        Srad = torch.zeros(NB, device=DEV, dtype=DT)
        for b in range(NB):
            m = (idx_t == b).reshape(-1)
            if m.any():
                Srad[b] = Pc.reshape(S, -1)[:, m].mean()
        Sfull = Srad[idx_t][None, None]
    print(f"[spectrum] S in [{float(Srad.min()):.2e}, {float(Srad.max()):.2e}]", flush=True)

    def gauss_logp(y, sn, th):
        with torch.no_grad():
            Yh = torch.fft.rfft2(y, dim=(-2, -1))
            Yp = (Yh.abs() ** 2).mean(1, keepdim=True) / (H * W)
            var = blur.Khat(th).abs() ** 2 * Sfull + sn ** 2
            return float(-(Yp / (2 * var)).sum() - 0.5 * torch.log(2 * np.pi * var).sum())

    @torch.no_grad()
    def exact_logp_y_theta0(y, sn):
        """精确：p(y) = p_{sigma_n}(y)（t=0）。"""
        return -float(likelihood_fn(y.to(DT), ode_flow,
                                    t_spans=[80.0, float(sn)]).item())

    y_clean = blur.A(x_star, TSTAR)
    print("\n" + "=" * 96, flush=True)
    print("A. theta=0 处：高斯近似 vs 精确值", flush=True)
    print("=" * 96, flush=True)
    print(f"{'sn':>6} | {'精确 log p(y|0)':>17} | {'近似 log p(y|0)':>17} | "
          f"{'误差(近似-精确)':>16} | {'证据比 log w(0)/w(t*)':>23}", flush=True)
    print("-" * 96, flush=True)

    results = []
    for sn in noises:
        gg = torch.Generator(device=DEV).manual_seed(1000 + int(sn * 100000))
        y = (y_clean + sn * torch.randn(y_clean.shape, generator=gg, device=DEV,
                                        dtype=DT)).to(DT)
        ex = exact_logp_y_theta0(y, sn)
        ap0 = gauss_logp(y, sn, 0.0)
        aps = gauss_logp(y, sn, TSTAR)
        err = ap0 - ex
        ratio = ap0 - aps          # log w(0) - log w(theta*)  <0 表示 theta* 赢
        results.append(dict(sn=sn, exact=ex, approx=ap0, err=err, ratio=ratio))
        print(f"{sn:>6} | {ex:>17.1f} | {ap0:>17.1f} | {err:>+16.1f} | {ratio:>+23.1f}",
              flush=True)
        del y
        torch.cuda.empty_cache()

    print("\n" + "=" * 96, flush=True)
    print("B. 二阶统计校验：模型预测谱 vs 经验谱（theta=theta*）", flush=True)
    print("=" * 96, flush=True)
    with torch.no_grad():
        for sn in noises:
            gg = torch.Generator(device=DEV).manual_seed(2000 + int(sn * 100000))
            ys = blur.A(xs, TSTAR) + sn * torch.randn(xs.shape, generator=gg,
                                                      device=DEV, dtype=DT)
            emp = ((torch.fft.rfft2(ys, dim=(-2, -1)).abs() ** 2) / (H * W)).mean((0, 1))
            pred = (blur.Khat(TSTAR).abs() ** 2 * Sfull)[0, 0] + sn ** 2
            # 只在信号功率远高于数值噪声的频段比较
            m = pred > 1e-6
            rel = ((emp[m] - pred[m]).abs() / pred[m]).mean()
            print(f"  sn={sn:<6} 平均相对偏差(在 pred>1e-6 的 {int(m.sum())}/"
                  f"{pred.numel()} 个频率上) = {float(rel):.3f}", flush=True)
            del ys
            torch.cuda.empty_cache()

    print("\n[判据]", flush=True)
    print("  A: |误差| << |证据比| => 结论稳健；否则结论受近似误差威胁", flush=True)
    print("  B: 平均相对偏差小 => 二阶结构精确，误差全来自高阶（非高斯）结构", flush=True)

    with open(os.path.join(OUT, "validate_spec.json"), "w") as f:
        json.dump(dict(cfg=dict(S=S, sigma_star=TSTAR, noises=noises),
                       partA=results), f, indent=1)
    print("\n[saved] results/validate_spec.json", flush=True)


if __name__ == "__main__":
    main()
