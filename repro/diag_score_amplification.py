"""量化 1/t^2 放大：去噪器残差 ||D(x*,t) - x*|| 随 t 怎么衰减？

背景：score 恒等式  grad log p_t = (D - x)/t^2  对*精确*去噪器成立。
对模型去噪器 D_model = D_true + dD，score 的相对误差是
      ||dD|| / ||D_true - x||
而对流形上的干净样本，D_true(x,t) - x = -t^2 * grad log p_t(x) = O(t^2)。
所以只要模型的绝对误差 dD 不随 t 同步衰减，相对误差就按 1/t^2 爆炸。

本脚本测 ||D(x*,t) - x*|| 随 t 的衰减速率：
  * 若 ~ t^2  => 去噪器足够准，score 可用
  * 若衰减更慢（有误差地板）=> 小 t 处 score 不可用，局部二阶方法必然失败

同时测同一批点的 score 投影 <grad log p_t, delta>，与精确 Lambda 的数值导数对比。

用法： python diag_score_amplification.py
"""

import argparse
import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn
from putils.convolution import GaussianBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
H = W = 64
KS = 15
POOL = "/tmp/pool_ffhq64_256_s0.pt"


def blur(x, k):
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
    ap.add_argument("--nimg", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--ts", nargs="*", type=float,
                    default=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    model = load_local_model(CKPT).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x.to(DT), t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)

    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:args.nimg].clone()
    k = GaussianBlur(kernel_size=KS, device=DEV, dtype=DT).kernel(args.sigma)
    y1 = blur(xs, k)
    d1 = (y1 - xs).flatten(1)
    nx = xs.flatten(1).norm(dim=1)

    @torch.no_grad()
    def logp(z, t):
        nll = likelihood_fn(z.to(DT), ode_flow, t_spans=[80.0, float(t)])
        return np.array([-float(v) for v in nll.reshape(-1)])

    print(f"nimg={args.nimg}  gaussian sigma={args.sigma:g}")
    print(f"{'t':>7} | {'||D(x*)-x*||':>13} | {'/t^2':>10} | {'/t':>9} | "
          f"{'<g,d1>':>11} | {'dLambda/da':>11} | {'比值':>7}")
    print("-" * 82)
    rows = []
    for t in args.ts:
        with torch.no_grad():
            D = model(xs, t)
            r = (D - xs).flatten(1).norm(dim=1)          # 每张图的去噪器残差
            g = ((D - xs) / (t * t)).flatten(1)
            proj = (g * d1).sum(1) / d1.norm(dim=1) ** 0  # <g, d1>
        rmean = float(r.mean())
        # 精确 Lambda 在 alpha=0.25 与 0 处，数值导数
        ya = 0.25 * y1 + 0.75 * xs
        lp0, lp1 = logp(xs, t), logp(ya, t)
        dlam = float(((lp1 - lp0) / 0.25).mean())
        ratio = proj.mean() / dlam if abs(dlam) > 1e-9 else float("nan")
        print(f"{t:>7g} | {rmean:>13.5f} | {rmean/t**2:>10.1f} | "
              f"{rmean/t:>9.2f} | {proj.mean():>11.1f} | {dlam:>11.1f} | "
              f"{ratio:>7.2f}", flush=True)
        rows.append(dict(t=float(t), resid=rmean, resid_over_t2=rmean / t ** 2,
                         resid_over_t=rmean / t, proj=float(proj.mean()),
                         dlam=dlam, ratio=ratio))
        torch.cuda.empty_cache()

    print("-" * 82)
    print("判据：")
    print("  * ||D-x*|| 列：若 ~ t^2（除以 t^2 后近似常数）则去噪器精确，score 可用")
    print("  * 若除以 t^2 后仍随 t 下降 => 有误差地板，score 相对误差 ~ 1/t^2 爆炸")
    print("  * 比值列：score 投影 / 精确数值导数，应 = 1。偏离 1 的程度就是 score 的误差")
    rs = np.array([r["resid"] for r in rows])
    ts = np.array([r["t"] for r in rows])
    ok = (rs > 0) & (ts > 0)
    if ok.sum() >= 3:
        p2 = np.polyfit(np.log(ts[ok]), np.log(rs[ok]), 1)[0]
        print(f"\n  ||D-x*|| ~ t^{p2:.2f}   （完美去噪器应为 t^2）")


if __name__ == "__main__":
    main()
