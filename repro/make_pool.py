"""生成先验样本池（供 sweep_law_test.py --pool 用）。

背景：第八组证明 b(u) 随噪声级 t 变。第九组要问 b(u) 是否随**先验**（模型）变，
设计是 2x2 因子表（见 sweep_law_test.py docstring 第九组）。其中"固定图像、只换先验"
这一格需要**另一个模型的先验去判同一批图**，所以图池要能独立指定。

用法： venv2/bin/python make_pool.py --ckpt edm-afhqv2-64x64-uncond-vp.pkl \
            --out /tmp/pool_afhqv2_256_s0.pt [--n 256] [--seed 0]

注意：
  * 池子会缓存，已存在就直接用（不会重新采样，保证与 ffhq 池同源可比）。
  * 采样前固定 torch 种子，保证可复现。
  * ffhq 池是 /tmp/pool_ffhq64_256_s0.pt（256 张，2026-10-11 生成）。
"""

import argparse
import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, ode_sampler

DEV = torch.device("cuda")
DT = torch.float32
H = W = 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=32,
                    help="每块采几张；256 一起采会 OOM（4060 8G）")
    args = ap.parse_args()

    if os.path.exists(args.out):
        p = torch.load(args.out, map_location=DEV).to(DT)
        print(f"[pool] 已存在 {args.out}: {tuple(p.shape)}（不重新采样）")
        return

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = load_local_model(args.ckpt).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x.to(DT), t, *a, **k))
    ode_flow, t_spans = get_ode_flow(prior)
    print(f"[cfg] ckpt={args.ckpt} n={args.n} seed={args.seed} "
          f"chunk={args.chunk}", flush=True)

    # 分块采样：一次性采 256 张会把 RK45 的整个状态放在显存里，4060 8G 直接 OOM
    # （2026-10-19 实测：Tried to allocate 512 MiB，池子根本没生成）。
    chunks = []
    done = 0
    while done < args.n:
        c = min(args.chunk, args.n - done)
        torch.manual_seed(args.seed + done)      # 每块独立但确定
        torch.cuda.manual_seed(args.seed + done)
        pc, _ = ode_sampler(ode_flow, shape=(c, 3, H, W), t_spans=t_spans,
                            device=DEV)
        chunks.append(pc.to(DT))
        done += c
        torch.cuda.empty_cache()
        print(f"[sample] {done}/{args.n}", flush=True)
    pool = torch.cat(chunks, 0)
    torch.save(pool.cpu(), args.out)
    print(f"[saved] {args.out} {tuple(pool.shape)} "
          f"range [{float(pool.min()):.3f}, {float(pool.max()):.3f}]")

    # 顺带打印径向功率谱的几个数，便于和 ffhq 池对照（先验谱形状的粗读）
    with torch.no_grad():
        P = (torch.fft.rfft2(pool.to(torch.float64), dim=(-2, -1)).abs() ** 2) \
            / (H * W)
        S = P.mean(1).mean(0)                       # (3, 33, 33)? 取径向
        fy = np.fft.fftfreq(H) * H
        fx = np.fft.fftfreq(W) * W
        FX, FY = np.meshgrid(fx, fy, indexing="ij")
        R = np.sqrt(FX ** 2 + FY ** 2)[:, :W // 2 + 1]
        edges = np.linspace(0, R.max() + 1e-6, 9)
        idx = np.clip(np.digitize(R.ravel(), edges) - 1, 0, 8).reshape(R.shape)
        Sc = S.mean(0).cpu().numpy()
        rad = [float(Sc[(idx == b)].mean()) if (idx == b).any() else float("nan")
               for b in range(9)]
        print("[spectrum] 径向平均功率（8 个环，内→外）:")
        print("   " + " ".join(f"{v:.4g}" for v in rad))
        print(f"   外环/内环 = {rad[-1] / rad[0]:.4g}"
              f"（ffhq 池对照见 make_pool 日志）")


if __name__ == "__main__":
    main()
