"""计算每个核把图像挪多远：||h*x - x||，以及它在 alpha 参数化下的缩放

动机：sweep_alpha 显示各族沿 alpha 的 Lambda 剖面形状惊人地相似
（b(alpha) 大多在 1.35-1.65）。一个自然的解释是：
      Lambda 主要依赖 位移的大小 ||alpha*(h*x - x)||，而不依赖方向
若成立，把 Lambda 对 位移范数 作图，所有核应当塌缩到同一条曲线——
这将把"核族依赖"整个消解掉。

本脚本只做纯张量运算，不需要模型/GPU 密集计算，但需要先验样本池。

输出 results/kernel_displacement.json：
  {kernel: {theta, per_img_norm, mean_norm, per_img_rel, mean_rel}}

用法： python dump_displacement.py [--nimg 20]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from putils.convolution import GaussianBlur, DefocusBlur, AiryDisc, MotionBlur

DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
H = W = 64
KS = 15
POOL = "/tmp/pool_ffhq64_256_s0.pt"

KERNELS = [
    ("gaussian", 1.0), ("gaussian", 2.0),
    ("defocus", 1.0), ("defocus", 2.0),
    ("airy", 2.0), ("airy", 4.0),
    ("motion", 3.0), ("motion", 7.0),
    ("lowpass", 0.25),
]


def blur(x, k, gate):
    if gate is not None:
        return torch.fft.irfft2(torch.fft.rfft2(x.to(torch.float64), dim=(-2, -1))
                                * gate, s=(H, W), dim=(-2, -1)).to(DT)
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
    ap.add_argument("--nimg", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(0)
    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:args.nimg].clone()
    nx = xs.flatten(1).norm(dim=1)                       # 每张图的范数

    def make(fam, th):
        if fam == "lowpass":
            fx = np.fft.fftfreq(H) * H
            fy = np.fft.fftfreq(W) * W
            FX, FY = np.meshgrid(fx, fy, indexing="ij")
            R = np.sqrt(FX ** 2 + FY ** 2)
            g = torch.from_numpy((R <= th * R.max()).astype(np.float64))
            return None, g[:, :W // 2 + 1].to(DEV)[None, None]
        cls = {"gaussian": GaussianBlur, "defocus": DefocusBlur,
               "airy": AiryDisc, "motion": MotionBlur}[fam]
        return cls(kernel_size=KS, device=DEV, dtype=DT).kernel(float(th)), None

    rows = []
    for fam, th in KERNELS:
        k, gate = make(fam, th)
        y1 = blur(xs, k, gate)
        d = (y1 - xs).flatten(1).norm(dim=1)             # 每张图的位移范数
        # 也记谱域位移：||(h-1) X||，与像素域应相等（Parseval）
        if gate is not None:
            # gate 是 |h|^2 的半平面谱门；位移谱 = (sqrt(gate) - 1) X
            ph = gate.to(torch.float64).sqrt()
            dd = torch.fft.irfft2(torch.fft.rfft2(xs.to(torch.float64),
                                                  dim=(-2, -1)) * (ph - 1.0),
                                  s=(H, W), dim=(-2, -1))
        else:
            if k.ndim == 2:
                k = k[None, None]
            kh, kw = k.shape[-2:]
            pad = torch.zeros((1, 1, H, W), device=DEV, dtype=torch.float64)
            pad[:, :, (H - kh) // 2:(H - kh) // 2 + kh,
                (W - kw) // 2:(W - kw) // 2 + kw] = k.to(torch.float64)
            ph = torch.fft.rfft2(pad, dim=(-2, -1))       # (1,1,H,W//2+1)
            dd = torch.fft.irfft2(torch.fft.rfft2(xs.to(torch.float64),
                                                  dim=(-2, -1)) * (ph - 1.0),
                                  s=(H, W), dim=(-2, -1))
        dspec = dd.flatten(1).norm(dim=1)
        rows.append(dict(family=fam, theta=float(th),
                         norm=[float(v) for v in d],
                         norm_mean=float(d.mean()),
                         rel=[float(v) for v in (d / nx)],
                         rel_mean=float((d / nx).mean()),
                         spec_norm_mean=float(dspec.mean()),
                         parseval_ratio=float((d / dspec).mean())))
        print(f"  {fam+'/'+format(th,'g'):<16} ||h*x-x|| = {d.mean():8.3f}   "
              f"相对 = {(d/nx).mean():.4f}   Parseval 校验 = "
              f"{(d/dspec).mean():.4f}", flush=True)
        del y1, dd
        torch.cuda.empty_cache()

    fn = os.path.join(OUT, "kernel_displacement.json")
    with open(fn, "w") as f:
        json.dump(dict(cfg=dict(nimg=args.nimg, ks=KS,
                                note="||h*x-x|| per image, first nimg of "
                                     "256-pool seed0"), rows=rows), f, indent=1)
    print(f"\n[saved] {fn}")


if __name__ == "__main__":
    main()
