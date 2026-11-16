"""对照实验：score 沿线段的振荡是"核回绕伪影"还是"1/t^2 放大模型误差"？

diag_score_vs_logp 发现 t=0.002 时 score 投影 <g,d1> 沿 alpha 剧烈振荡
（alpha=0 处 +9812，alpha=0.05 处 -2588，alpha=0.2 处 +502 ...），
而精确 Lambda 平滑单调。两种可能：
  (a) 15 边核的循环卷积把核回绕到图像对边，产生接缝；模型对接缝响应剧烈
  (b) score = (D-x)/t^2，t=0.002 时放大 25 万倍，模型的微小误差被放大成振荡

判别：用**纯谱高斯**（h2 = exp(-2 pi^2 sigma^2 f^2)，没有核截断、没有回绕），
沿同样的 alpha 线段看 score 是否还振荡。
  * 谱高斯也振荡 => 情形 (b)，与核实现无关
  * 谱高斯不振荡  => 情形 (a)，alpha 实验的算子带伪影，需要加注释

同时报 t=0.02 的同样测量作第二个对照。

用法： python diag_score_control.py --smoke
"""

import argparse
import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from putils.convolution import GaussianBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
H = W = 64
KS = 15
POOL = "/tmp/pool_ffhq64_256_s0.pt"
ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]


def blur_kernel(x, k):
    if k.ndim == 2:
        k = k[None, None]
    kh, kw = k.shape[-2:]
    pad = torch.zeros((1, x.shape[1], H, W), device=x.device, dtype=x.dtype)
    top, left = (H - kh) // 2, (W - kw) // 2
    pad[:, :, top:top + kh, left:left + kw] = k.to(x.dtype)
    return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))
                            * torch.fft.rfft2(pad, dim=(-2, -1)),
                            s=(H, W), dim=(-2, -1))


def blur_spectral(x, sg):
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    f2 = (FX ** 2 + FY ** 2)[:, :W // 2 + 1] / (H * W)
    h2 = torch.from_numpy(np.exp(-2 * np.pi ** 2 * sg ** 2 * f2))
    return torch.fft.irfft2(torch.fft.rfft2(x.to(torch.float64), dim=(-2, -1))
                            * h2.to(DEV)[None, None].to(torch.float64),
                            s=(H, W), dim=(-2, -1)).to(DT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=6)
    ap.add_argument("--ts", nargs="*", type=float, default=[0.002, 0.02])
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    nimg = 2 if args.smoke else args.nimg

    model = load_local_model(CKPT).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:nimg].clone()

    k = GaussianBlur(kernel_size=KS, device=DEV, dtype=DT).kernel(1.0)
    variants = [("15边核(有回绕)", blur_kernel(xs, k)),
                ("纯谱高斯(无回绕)", blur_spectral(xs, 1.0))]

    for t in args.ts:
        t2 = t * t
        print(f"\n{'='*96}\n t = {t:g}   (1/t^2 = {1/t2:.0f})\n{'='*96}")
        for name, y1 in variants:
            d1 = y1 - xs
            print(f"\n [{name}]")
            print(f" {'alpha':>6} | {'<g,d1>':>11} | {'相邻变化':>10}")
            print(" " + "-" * 36)
            prev = None
            for a in ALPHAS:
                ya = (1.0 - a) * xs + a * y1
                with torch.no_grad():
                    g = (model(ya, t) - ya) / t2
                    proj = (g.flatten(1) * d1.flatten(1)).sum(1).mean().item()
                ch = "" if prev is None else f"{proj-prev:>+10.0f}"
                print(f" {a:>6g} | {proj:>11.0f} | {ch}")
                prev = proj
                torch.cuda.empty_cache()
        # 振荡度量：相邻点差分的均方根 / 均值幅度
        print(f"\n [振荡度量] t={t:g}")
        for name, y1 in variants:
            d1 = y1 - xs
            vals = []
            for a in ALPHAS:
                ya = (1.0 - a) * xs + a * y1
                with torch.no_grad():
                    g = (model(ya, t) - ya) / t2
                    vals.append((g.flatten(1) * d1.flatten(1)).sum(1).mean().item())
                torch.cuda.empty_cache()
            v = np.array(vals)
            dv = np.diff(v)
            print(f"   {name:<18} |<proj>|mean={np.abs(v).mean():>9.0f}  "
                  f"RMS(相邻差)={np.sqrt((dv**2).mean()):>9.0f}  "
                  f"比值={np.sqrt((dv**2).mean())/np.abs(v).mean():>6.2f}  "
                  f"变号次数={int((np.diff(np.sign(v)) != 0).sum())}")


if __name__ == "__main__":
    main()
