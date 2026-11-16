"""冒烟：ImageNet-64 条件 ADM 检查点能否接入现有框架？

要验证三件事：
  1. load_local_model 能加载 edm-imagenet-64x64-cond-adm.pkl
  2. 传入 class_labels 的去噪器能让 ode_sampler 跑通（类别标签烘进 closure）
  3. likelihood_fn 在该模型上返回有限值

用法： python smoke_imagenet.py --class-idx 207
"""

import argparse
import os
import sys
import time

import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn, ode_sampler

CKPT = "edm-imagenet-64x64-cond-adm.pkl"
DEV = torch.device("cuda")
DT = torch.float32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-idx", type=int, default=207)
    args = ap.parse_args()

    t0 = time.time()
    model = load_local_model(CKPT).to(DEV)
    print(f"[load] {time.time()-t0:.1f}s  params="
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    def denoiser(x, t, *a, **k):
        lab = torch.full((x.shape[0],), int(args.class_idx),
                         device=x.device, dtype=torch.long)
        onehot = torch.nn.functional.one_hot(lab, 1000).to(DT)
        return model(x.to(DT), t, class_labels=onehot, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    print(f"[flow] t_spans={t_spans}", flush=True)

    t0 = time.time()
    pool, nfe = ode_sampler(ode_flow, shape=(4, 3, 64, 64), t_spans=t_spans,
                            device=DEV)
    print(f"[sample] 4 images, nfe={nfe}, {time.time()-t0:.1f}s, "
          f"range=[{float(pool.min()):.2f},{float(pool.max()):.2f}]", flush=True)

    x = pool[:1].to(DT)
    t0 = time.time()
    with torch.no_grad():
        v = likelihood_fn(x, ode_flow, t_spans=[80.0, 0.002])
    print(f"[logp] -log p = {float(v.reshape(-1)[0]):.1f}  ({time.time()-t0:.1f}s)",
          flush=True)
    print("[smoke] OK" if abs(float(v.reshape(-1)[0])) < 1e6 else "[smoke] 值异常")


if __name__ == "__main__":
    main()
