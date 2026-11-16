"""
Phase B —— 复现 Morse-Bott 论文 Section 4.1 / Fig.6-7
"Blurry images are more likely"

对每张图像 x，计算 q(h_theta * x) = -log p(x_blurred) 随模糊强度 theta 的变化。
论文结论：势能随模糊度【严格递减】，即模糊图比原图更"可能"。

严格复用作者代码：
  - edm/prior.py 的 ScorePrior / get_ode_flow / likelihood_fn / ode_sampler
  - putils/convolution.py 的四个模糊族（Gaussian / Defocus / Airy / Motion）
唯一自补的部分：模糊施加（原 notebook 用 deepinv.physics.BlurFFT，这里用
等价的 FFT 循环卷积，避免安装 deepinv 这一大堆依赖）。

用法：
  python repro_phaseB.py --smoke          # 冒烟：2 张图，只测 Gaussian，4 档
  python repro_phaseB.py --nimg 20        # 正式：20 张图，四个模糊族
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repro/ 的上一级
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn, ode_sampler
from putils.convolution import GaussianBlur, DefocusBlur, AiryDisc, MotionBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")


# ---------------------------------------------------------------- 模糊施加
def blur_fft(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """与 deepinv.physics.BlurFFT 等价：每通道用同一核做循环（FFT）卷积，输出同尺寸。"""
    if kernel.ndim == 2:
        kernel = kernel[None, None]
    assert kernel.ndim == 4 and kernel.shape[:2] == (1, 1), kernel.shape
    B, C, H, W = x.shape
    kh, kw = kernel.shape[-2:]
    pad = torch.zeros((1, C, H, W), device=x.device, dtype=x.dtype)  # (1,C,·) 才能对 B 广播
    top, left = (H - kh) // 2, (W - kw) // 2
    pad[:, :, top:top + kh, left:left + kw] = kernel.to(x.dtype)
    X = torch.fft.rfft2(x, dim=(-2, -1))
    K = torch.fft.rfft2(pad, dim=(-2, -1))
    return torch.fft.irfft2(X * K, s=(H, W), dim=(-2, -1))


# 每个模糊族的参数名与扫描范围（theta 越大越模糊，theta=0 -> Dirac）
FAMILIES = {
    "gaussian": (GaussianBlur,  "sigma",  [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
    "defocus":  (DefocusBlur,   "radius", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
    "airy":     (AiryDisc,      "radius", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
    "motion":   (MotionBlur,    "length", [0, 1, 3, 5, 7, 9]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：2 张图，只 Gaussian，4 档")
    ap.add_argument("--nimg", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--families", nargs="*", default=None)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.smoke:
        nimg, fams, levels = 2, ["gaussian"], [0.0, 0.5, 1.0, 2.0]
    else:
        nimg, fams = args.nimg, (args.families or list(FAMILIES))
        levels = None

    print(f"[cfg] nimg={nimg} families={fams} ckpt={CKPT}", flush=True)

    # ---------------- 模型与 ODE flow ----------------
    t0 = time.time()
    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k).to(torch.float64)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    print(f"[model] loaded in {time.time()-t0:.1f}s  t_spans={t_spans}", flush=True)

    # ---------------- 采样 nimg 张图（避免依赖 FFHQ 数据集）----------------
    t0 = time.time()
    samples, nfe = ode_sampler(ode_flow, shape=(nimg, 3, 64, 64),
                               t_spans=t_spans, device=DEV)
    print(f"[sample] {nimg} images, nfe={nfe}, {time.time()-t0:.1f}s", flush=True)

    # ---------------- 势能扫描 ----------------
    records = []
    for i in range(nimg):
        x = samples[i:i + 1].to(DT)
        for fam in fams:
            cls, pname, default_levels = FAMILIES[fam]
            lv = levels if levels is not None else default_levels
            kernel_fn = cls(kernel_size=15, device=DEV, dtype=DT)
            for th in lv:
                if th == 0:
                    blurred = x
                else:
                    k = kernel_fn.kernel(th)
                    blurred = blur_fft(x, k)
                t1 = time.time()
                with torch.no_grad():
                    nll = likelihood_fn(blurred, ode_flow, t_spans=t_spans)
                torch.cuda.synchronize()
                dt = time.time() - t1
                val = float(nll.item())
                records.append(dict(img=i, family=fam, param=pname,
                                    theta=float(th), nll=val, sec=round(dt, 2)))
                print(f"  img{i} {fam:9s} theta={th:<5} -logp={val:12.4f}  ({dt:.1f}s)",
                      flush=True)
                del blurred
                torch.cuda.empty_cache()

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 72, flush=True)
    print("势能随模糊度变化（-log p，越小=越可能）：", flush=True)
    summary = {}
    for fam in fams:
        rows = [r for r in records if r["family"] == fam and r["img"] == 0]
        if not rows:
            continue
        thetas = [r["theta"] for r in rows]
        vals = [r["nll"] for r in rows]
        d = vals[-1] - vals[0]
        mono = all(b <= a + 1e-9 for a, b in zip(vals, vals[1:]))
        summary[fam] = dict(thetas=thetas, nll_img0=vals,
                            delta=round(d, 4), monotone_decreasing=mono)
        print(f"  {fam:9s} " + " ".join(f"{v:10.3f}" for v in vals)
              + f"   Δ={d:+.3f}  单调递减={mono}", flush=True)

    tag = "smoke" if args.smoke else f"nimg{nimg}"
    path = os.path.join(OUT, f"phaseB_{tag}.json")
    with open(path, "w") as f:
        json.dump(dict(cfg=dict(nimg=nimg, families=fams, ckpt=CKPT,
                                seed=args.seed, levels=levels),
                       records=records, summary=summary), f, indent=1)
    print(f"\n[saved] {path}", flush=True)

    free, total = torch.cuda.mem_get_info()
    print(f"[gpu] free {free/2**30:.2f} GiB / {total/2**30:.2f} GiB, "
          f"peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
