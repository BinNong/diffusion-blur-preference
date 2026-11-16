"""第十三组：把 (ah) 的「径向位置 3.2 倍」坐实——固定 φ₀ 扫 m，图像数提高 6 倍（GPU）

预官（2026-10-23，跑前写定）
================================================================================
背景（RESULTS §3.9(ah)）
------------------------
(ah) 撤回了我自己在 (ag) 的「$\phi$ 是第二描述量」（同类 $tw_{\rm od}$ 配对在第二个 $u$ 上**反号**），
但留下一个方向一致、**幅度未坐实**的新观察：在 $u$=0.65、kunif=0.35、**固定 $\phi_0$=0.15** 时，
只改过渡形状 $m$（等价于把过渡带的径向位置从 $r\approx0.66$ 搬到 $r\approx0.23$）：

  | 算子 | $m$ | $r_1$ | $tw_{\rm od}$ | $C_{\\rm rel}$ | SNR | 逐图负值 |
  |---|---|---|---|---|---|---|
  | PX15m020 | 0.20 | 0.662 | 0.124 | **0.299** | **3.7** | **5/20** |
  | PX15m100 | 1.00 | 0.477 | 0.233 | 0.858 | 21.4 | 0/20 |
  | PX15m200 | 2.00 | 0.227 | 0.260 | **0.960** | 29.6 | 0/20 |

$C_{\\rm rel}$ 跨 **3.2 倍**，而 $tw_{\rm od}$ 只跨 2.1 倍、$tw_{\rm od}$ 定律只预言 1.08 倍。
**但 H3 的残差与 SNR 单调相关**（SNR 18.5/22.0 → +0.071/+0.072；9.4 → −0.134；6.6 → −0.232；
3.7 → −0.410），而 PX15m020 的 mean/std = **0.80**、5/20 张图 $\Lambda$ 为负。
所以这个 3.2 倍**完全由一个 SNR=3.7 的点主导**，按项目规矩只能记为「方向一致、幅度未坐实」。

本组把它坐实
------------
**只改一个变量：图像数 20 → 120**（池有 256 张，够）。其余逐项与 (ah) 对齐：
同族、同 kunif=0.35（$u$=0.65）、同 $t$=0.002、同 ffhq 池、同配对 $\Lambda$、同 $b(u)$/mid。
$\phi_0$ 固定为 0.15（(ah) 的关键 $\phi$），$m\in\{0.20, 0.50, 1.00, 2.00\}$——4 个算子，
其中 $m$=1.00 与 2.00 是 (ah) 已测点的**同算子重复**，用作跨轮次复现的锚。

**功效（跑前算好）**：PX15m020 的逐图 std = 855，$n$=120 时 SE$(\Lambda)$=855/√120=**78**
（$n$=20 时是 191），即 SE$(\log_{10}C)$ 从 0.121 降到 **0.048 dex**。
(ah) 的值 0.299 与「真实值是 0.60」相差 0.302 dex ⇒ **6.3σ**，可判定。

**跑前已算出的描述量（与 (ah) 逐字同法）**：

  | 算子 | $\phi_0$ | $m$ | $r_1$ | amin | amax | $tw_{\rm od}$ | $\phi$ | $C$ 预言 |
  |---|---|---|---|---|---|---|---|---|
  | PY15m020 | 0.15 | 0.20 | 0.6624 | 0.150 | 1.000 | 0.124 | 0.429 | 0.770 |
  | PY15m050 | 0.15 | 0.50 | 0.5950 | 0.150 | 1.000 | 0.191 | 0.429 | 0.805 |
  | PY15m100 | 0.15 | 1.00 | 0.4768 | 0.150 | 1.000 | 0.233 | 0.429 | 0.824 |
  | PY15m200 | 0.15 | 2.00 | 0.2268 | 0.150 | 1.000 | 0.260 | 0.429 | 0.835 |

  （$r_1$/amin/amax/$tw_{\rm od}$ 由离线脚本算出，冒烟时逐项核对。）

预官判据
--------
J0 守门（**必须全过**）：
   (a) 每算子 kunif 命中 0.35，重建对账 ≤5e−3；
   (b) amin=0.150、amax=1.000（属「$g_2$ 能到 0 但到不了 1」类），逐项报出；
   (c) 逐格 SNR ≥ **10**（$n$=120 的预期功效；(ah) 的守门是 3，这里提高，因为本组的目的就是
       把 SNR 不达标的格子救回来）；
   (d) **跨轮次复现**：$m$=1.00 与 $m$=2.00 两点（(ah) 已测、同算子、不同图像数）的 $C_{\\rm rel}$
       须与 (ah) 的值在 **2×SE_comb** 内一致。不过则说明两次测量有一次不对，先查测量再报物理。

J1 主判据（**坐实 (ah) 的 3.2 倍**）：$C_{\\rm rel}(m{=}0.20)/C_{\\rm rel}(m{=}2.00)$。
   **预言 ≤ 0.50**（即至少 2 倍的下降）。(ah) 报 0.299/0.960 = **0.311**。
   **失败方式**：若实测 ≥ **0.70**（即真实倍数 ≤1.43），则 (ah) 的 3.2 倍被否掉，
   「径向位置是一等描述量」在本类的定量版只能降到 ~1.4 倍，与 §3.9(l5) 的定性发现之间
   出现实质差距。阈值 0.50 与 0.70 分别对应「确认」和「否掉」，中间是灰区（会明确记为不可判定）。

J2 判别（$m$-曲线的形状）：$C_{\\rm rel}$ 在 4 个 $m$ 上单调不减，且
   $C_{\\rm rel}(m{=}0.20)/C_{\\rm rel}(m{=}1.00)\le 0.60$。(ah) 只有 3 个点且最低点 SNR=3.7，
   **本组第一次能把这条曲线的低 $m$ 端测准**。不预设曲线形状（对数/幂/饱和都没依据），只查单调性。

J3 诊断（无阈值，但**必须报**）：
   (a) 逐图的 mean/std、负值个数、偏度——把 (ah) 的估计量脆弱性定量化；
   (b) SE 的 $1/\\sqrt n$ 标度自检：$m$=1.00/2.00 两点的 SE(120 图) / SE(20 图) 应 ≈ √(20/120)=0.408；
   (c) 4 个点的 $\\Lambda$ 与 $\\Lambda_{\\rm Gauss}$：$\\Lambda_G/({\\rm cap}\\cdot\\text{knif})$ 应 ≈1
       （(ah) 实测 1.003–1.010；我预官时曾把它推反，这里只作记录不改判据）。

诚实声明
--------
* 只测 $t$=0.002、ffhq 先验、$u$=0.65 一个点、$\phi_0$=0.15 一个 $\phi$。
* 本组**不检验** $m$-效应是否随 $\phi_0$ 变——那要另一个 $\phi_0$ 的序列（(ah) 的 $\phi_0$=0.20
  只有 2 个低 SNR 点，本组不补）。
* 若 J1 落在 [0.50, 0.70] 灰区，结论记为「不可判定」并报需要多少图像才能判定
  （按 SE ∝ 1/√n 外推）。

用法： venv2/bin/python sweep_pedestal3.py [--smoke]
"""

import argparse
import json
import math
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
POOL = "/tmp/pool_ffhq64_256_s0.pt"
KUNIF_TARGET = 0.35
PHI0 = 0.15
NIMG = 120                     # **本组唯一改的变量**：(ah) 用 20

# (id, phi0, m)：phi0 固定 0.15，扫 m
SPEC = [
    ("PY15m020", 0.15, 0.20), ("PY15m050", 0.15, 0.50),
    ("PY15m100", 0.15, 1.00), ("PY15m200", 0.15, 2.00),
]


def halfplane():
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)[:, :W // 2 + 1]
    return R / R.max()


def atten(phi0, r1, m, r):
    s = np.zeros_like(r)
    msk = r >= r1
    s[msk] = ((r[msk] - r1) / max(1.0 - r1, 1e-12)) ** m
    return phi0 + (1.0 - phi0) * np.clip(s, 0.0, 1.0)


def kunif_of(phi0, r1, m, r):
    """kunif = <a>。atten() 返回衰减 a，**不能**再写 1-mean（那是 u）。"""
    return float(atten(phi0, r1, m, r).mean())


def solve_r1(phi0, m, r, target=KUNIF_TARGET):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if kunif_of(phi0, mid, m, r) < target:      # <a> 随 r1 递减
            hi = mid
        else:
            lo = mid
    r1 = 0.5 * (lo + hi)
    assert abs(kunif_of(phi0, r1, m, r) - target) < 2e-3, "r1 解不收敛"
    assert 1e-3 < r1 < 0.999, f"r1={r1:.4f} 贴边"
    return r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=NIMG)
    ap.add_argument("--t", type=float, default=0.002)
    ap.add_argument("--out", default="sweep_pedestal3.json")
    args = ap.parse_args()
    if args.smoke and args.out == "sweep_pedestal3.json":
        args.out = "smoke_pedestal3.json"
    nimg = 2 if args.smoke else args.nimg

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    r = halfplane()
    ops = []
    for oid, phi0, m in SPEC:
        r1 = solve_r1(phi0, m, r)
        a = atten(phi0, r1, m, r)
        g2 = 1.0 - a
        ops.append((oid, g2, dict(fid=oid, kind="pedestal", par=float(m),
                                   phi0=float(phi0), r1=float(r1),
                                   K=float(KUNIF_TARGET), beta=1.0,
                                   kunif_ref=float(KUNIF_TARGET))))
        print(f"[build] {oid}: phi0={phi0:.2f} m={m:.2f} r1={r1:.4f} "
              f"kunif={1.0-g2.mean():.4f} amin={a.min():.4f} amax={a.max():.4f}",
              flush=True)
    print(f"[cfg] nimg={nimg} t={args.t:g} kunif={KUNIF_TARGET} "
          f"{len(ops)} 个算子", flush=True)

    model = load_local_model(CKPT).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x.to(DT), t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)

    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:nimg].clone()
    with torch.no_grad():
        P = (torch.fft.rfft2(pool.to(torch.float64), dim=(-2, -1)).abs() ** 2) / (H * W)
        Sfull = P.mean(1).mean(0)
        Zx = (torch.fft.rfft2(xs.to(torch.float64), dim=(-2, -1)).abs() ** 2) \
            .mean(1, keepdim=True).mean(0)[0] / (H * W)
    Sf = Sfull + args.t ** 2
    wg = Zx / (2.0 * Sf)
    cap = float((Zx / (2.0 * Sf)).sum())
    B = 4

    @torch.no_grad()
    def logp(z):
        out = []
        for i in range(0, z.shape[0], B):
            nll = likelihood_fn(z[i:i + B].to(DT), ode_flow, t_spans=[80.0, args.t])
            out.extend([-float(v) for v in nll.reshape(-1)])
            torch.cuda.empty_cache()
        return np.array(out)

    t0 = time.time()
    lp_sharp = logp(xs)
    print(f"[sharp] t={args.t:g} n={nimg} mean log p = {lp_sharp.mean():.1f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    rows = []
    for i, (oid, g2, meta) in enumerate(ops):
        g = torch.from_numpy(g2).to(DEV)[None, None].to(torch.float64)
        with torch.no_grad():
            y = torch.fft.irfft2(torch.fft.rfft2(xs.to(torch.float64),
                                                 dim=(-2, -1)) * g.sqrt(),
                                 s=(H, W), dim=(-2, -1)).to(DT)
        tt = time.time()
        lp = logp(y)
        lam = lp - lp_sharp
        lam_g = float(((1.0 - g2) * wg.cpu().numpy()).sum())
        kunif = float(1.0 - g2.mean())
        rows.append(dict(id=oid, meta=meta, kunif=kunif,
                         lam_mean=float(lam.mean()), lam_std=float(lam.std()),
                         lam_per_img=[float(v) for v in lam],
                         lam_gauss=lam_g,
                         logp_blur_mean=float(lp.mean()),
                         logp_sharp_mean=float(lp_sharp.mean())))
        snr = lam.mean() * math.sqrt(nimg) / max(lam.std(), 1e-9)
        print(f"  [{i+1}/{len(ops)}] {oid:<10} knif={kunif:.4f} "
              f"Lambda={lam.mean():>9.0f}+-{lam.std()/math.sqrt(nimg):>6.0f} "
              f"SNR={snr:>5.1f}  neg={int((lam<0).sum())}/{nimg}  "
              f"L_G={lam_g:>7.1f}  ({time.time()-tt:.0f}s)", flush=True)
        del y, g
        torch.cuda.empty_cache()

    outfn = os.path.join(OUT, args.out)
    json.dump(dict(cfg=dict(nimg=nimg, t=args.t, ckpt=CKPT, pool=POOL,
                            kunif_target=KUNIF_TARGET, phi0=PHI0, cap=cap,
                            spec=[dict(id=s[0], phi0=s[1], m=s[2]) for s in SPEC]),
                    rows=rows), open(outfn, "w"), indent=1)
    print(f"\n[saved] {outfn}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
