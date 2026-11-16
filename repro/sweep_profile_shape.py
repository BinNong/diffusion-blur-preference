"""剖面形状受控实验：在匹配 kunif 处，Lambda 依赖剖面的"形状"吗？

动机（阶段一计划 §7.1）
---------------------
强模糊端的跨核差异原因未知。两条前置发现决定了本实验的设计：

 1. profile_response G1b：Lambda_Gauss = Sum_f a(f) Zx_f/(2 Sf)，
    而 Zx_f ≈ Sfull_f（同一统计量的 20 图 / 256 图版本），所以
        Lambda_Gauss ≈ (N/2) * kunif
    即高斯对照对剖面形状完全盲，只认总衰减。它在 N/2=1056 nats 处打满。
 2. sweep_alpha G3：alpha-插值出的剖面 a(f)=1-|(1-a)+a*h|^2 是"单值台面"
    ——每个 alpha 下约 90% 的频率上 a(f) 取同一个值 p=2a-a^2。
    所以 alpha 路径几乎不扫形状，只扫台面高度。它是坏的形状探针。

因此本实验换一种完全受控的算子构造：零相位谱门 g(f)=|h(f)|，
    功率剖面 g2(f) = (1-beta) + beta * ref2(f)
    => 衰减剖面 a(f) = beta * a_ref(f)     （形状精确保持，只缩放振幅）
对每个参考剖面选 beta 使命中公共目标 kunif。于是：
    * 同一目标下，所有参考的 kunif 相同、Lambda_Gauss 相同（≈ N/2 * K）
    * 但剖面形状完全不同（高斯 /  Airy / defocus / motion / 理想低通）
    => 该目标下 Lambda_real 的跨参考散度 = 纯粹的形状效应，
       且与高斯响应无关（高斯响应在这一组里是常数）。

三组实验：
  A  matched：8 个参考剖面 x 3 个目标 kunif —— 形状不同、kunif 与 Lambda_Gauss 全同
  B  where ：理想低通 / 高通 / 带通 x 4 个目标，只改"衰减放在哪个半径"
  C  amp   ：固定形状扫 beta —— 固定形状下的非线性响应
  V  raw   ：beta=1 的零相位真实核，与 sweep_alpha 的 alpha=1 对照（校验）

输出 results/sweep_profile_shape.json

用法： venv2/bin/python sweep_profile_shape.py --smoke
      venv2/bin/python sweep_profile_shape.py --nimg 40 --parts A B C
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
from putils.convolution import GaussianBlur, DefocusBlur, AiryDisc, MotionBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
H = W = 64
KS = 15
POOL = "/tmp/pool_ffhq64_256_s0.pt"
NB = 48

# 参考剖面：真实核的 |h|^2（零相位化后）
REFS = [("gaussian", 1.0), ("gaussian", 2.0), ("defocus", 1.0),
        ("defocus", 2.0), ("airy", 4.0), ("motion", 3.0),
        ("motion", 7.0), ("lowpass", 0.25)]
TARGETS = [0.35, 0.60, 0.85]
BETAS = [0.3, 0.7]
B_KINDS = ["lp", "hp", "bp"]
B_TARGETS = [0.2, 0.4, 0.6, 0.8]


def mode_grid():
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    return np.sqrt(FX ** 2 + FY ** 2), FX, FY


def make_ref(fam, th):
    """参考剖面 ref2(f) = 真实核的 |h(f)|^2，(H, W//2+1) 的 numpy。"""
    if fam == "lowpass":
        R, FX, FY = mode_grid()
        Rr = R[:, :W // 2 + 1]
        return (Rr <= th * R.max()).astype(np.float64)
    cls = {"gaussian": GaussianBlur, "defocus": DefocusBlur,
           "airy": AiryDisc, "motion": MotionBlur}[fam]
    k = cls(kernel_size=KS, device=DEV, dtype=torch.float64).kernel(float(th))
    kk = k[None, None] if k.ndim == 2 else k
    kh, kw = kk.shape[-2:]
    pad = torch.zeros((1, 1, H, W), dtype=torch.float64, device=DEV)
    pad[:, :, (H - kh) // 2:(H - kh) // 2 + kh,
        (W - kw) // 2:(W - kw) // 2 + kw] = kk.to(torch.float64)
    return torch.fft.rfft2(pad, dim=(-2, -1)).abs().square()[0, 0]\
        .cpu().numpy().astype(np.float64)


def ideal_profile(kind, r, target_sum):
    """理想低通/高通/带通的衰减剖面 a(f)，使 Sum_f a ≈ target_sum。

    kind: 'lp' 衰减高频(r>rc) / 'hp' 衰减低频(r<rc) / 'bp' 衰减环带(rc<r<1)
    二分半径命中目标模式数。cnt(rc) 对 'lp'/'bp' 单调降，对 'hp' 单调升。
    """
    Rr = r[:, :W // 2 + 1] / r.max()

    def cnt(rc):
        if kind == "lp":
            return float((Rr > rc).sum())
        if kind == "hp":
            return float((Rr < rc).sum())
        return float(((Rr > rc) & (Rr < 1.0)).sum())

    lo, hi = 1e-4, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        c = cnt(mid)
        # 'lp'/'bp': c 随 rc 增而减；'hp': 随 rc 增而增
        too_big = (c > target_sum) if kind in ("lp", "bp") else (c < target_sum)
        if too_big:
            lo = mid
        else:
            hi = mid
    rc = 0.5 * (lo + hi)
    if kind == "lp":
        a = (Rr > rc).astype(np.float64)
    elif kind == "hp":
        a = (Rr < rc).astype(np.float64)
    else:
        a = ((Rr > rc) & (Rr < 1.0)).astype(np.float64)
    return a, dict(kind=kind, radius=float(rc), n_att=int(a.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=40)
    ap.add_argument("--t", type=float, default=0.002)
    ap.add_argument("--parts", nargs="*", default=["A", "B", "C", "V"])
    ap.add_argument("--refs", nargs="*", default=None,
                    help="'family:theta' 限制参考剖面")
    ap.add_argument("--targets", nargs="*", type=float, default=None)
    ap.add_argument("--out", default="sweep_profile_shape.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    nimg = 2 if args.smoke else args.nimg
    refs = [(s.split(":")[0], float(s.split(":")[1])) for s in args.refs] \
        if args.refs else REFS
    targets = args.targets if args.targets else TARGETS
    if args.smoke:
        refs, targets, parts = refs[:1], [0.6], ["A"]

    R, FX, FY = mode_grid()
    Rr = R[:, :W // 2 + 1]
    N = Rr.size

    # ---------------- 算子表 ----------------
    # 每项: dict(id, g2(f) numpy, tag, part, meta)
    ops = []
    for fam, th in refs:
        ref2 = make_ref(fam, th)
        kunif_ref = float(1.0 - ref2.mean())
        for K in targets:
            beta = K / kunif_ref
            if beta > 1.0 + 1e-12:
                continue
            g2 = (1.0 - beta) + beta * ref2
            ops.append(dict(id=f"{fam}/{th:g}@K{K:g}", g2=g2, part="A",
                            meta=dict(family=fam, theta=float(th), K=float(K),
                                      beta=float(beta), kunif_ref=kunif_ref)))
        if "V" in args.parts:
            # 校验：beta=1 就是零相位版的真实核，应与 sweep_alpha 的 alpha=1 对照
            ops.append(dict(id=f"{fam}/{th:g}@raw", g2=ref2.copy(), part="V",
                            meta=dict(family=fam, theta=float(th), K=kunif_ref,
                                      beta=1.0, kunif_ref=kunif_ref)))
        if "C" in args.parts:
            for beta in BETAS:
                if beta * kunif_ref > 0.98:
                    continue
                g2 = (1.0 - beta) + beta * ref2
                ops.append(dict(id=f"{fam}/{th:g}@b{beta:g}", g2=g2, part="C",
                                meta=dict(family=fam, theta=float(th),
                                          beta=float(beta),
                                          kunif_ref=kunif_ref)))
    if "B" in args.parts:
        for K in B_TARGETS:
            tgt = K * N
            for kind in B_KINDS:
                a, info = ideal_profile(kind, R, tgt)
                g2 = 1.0 - a
                ops.append(dict(id=f"ideal-{kind}@K{K:g}", g2=g2, part="B",
                                meta=dict(K=float(K), **info)))
    # 去重（同一 g2 可能出现）
    seen = {}
    for o in ops:
        seen[o["id"]] = o
    ops = [seen[k] for k in sorted(seen)]
    print(f"[cfg] nimg={nimg} t={args.t:g} parts={args.parts} "
          f"refs={len(refs)} targets={targets} => {len(ops)} 个算子",
          flush=True)

    # ---------------- 模型 ----------------
    model = load_local_model(CKPT).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    prior = ScorePrior(denoiser=lambda x, t, *a, **k: model(x.to(DT), t, *a, **k))
    ode_flow, _ = get_ode_flow(prior)

    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:nimg].clone()
    with torch.no_grad():
        P = (torch.fft.rfft2(pool.to(torch.float64), dim=(-2, -1)).abs() ** 2) \
            / (H * W)
        Sfull = P.mean(1).mean(0)                        # (H, W//2+1) 全 256 图
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
            nll = likelihood_fn(z[i:i + B].to(DT), ode_flow,
                                t_spans=[80.0, args.t])
            out.extend([-float(v) for v in nll.reshape(-1)])
            torch.cuda.empty_cache()
        return np.array(out)

    t_start = time.time()
    lp_sharp = logp(xs)
    print(f"[sharp] t={args.t:g} mean log p = {lp_sharp.mean():.1f} "
          f"({time.time()-t_start:.0f}s)", flush=True)

    rows = []
    for oi, o in enumerate(ops):
        g2 = torch.from_numpy(o["g2"]).to(DEV)[None, None].to(torch.float64)
        with torch.no_grad():
            y = torch.fft.irfft2(torch.fft.rfft2(xs.to(torch.float64),
                                                 dim=(-2, -1)) * g2.sqrt(),
                                 s=(H, W), dim=(-2, -1)).to(DT)
        t0 = time.time()
        lp = logp(y)
        lam = lp - lp_sharp
        lam_g = float(((1.0 - o["g2"]) * wg.cpu().numpy()).sum())
        kunif = float(1.0 - o["g2"].mean())
        rows.append(dict(id=o["id"], part=o["part"], meta=o["meta"],
                         kunif=kunif, lam_mean=float(lam.mean()),
                         lam_std=float(lam.std()), lam_min=float(lam.min()),
                         lam_max=float(lam.max()),
                         lam_per_img=[float(v) for v in lam],
                         lam_gauss=lam_g, logp_blur_mean=float(lp.mean()),
                         logp_sharp_mean=float(lp_sharp.mean())))
        snr = lam.mean() / max(lam.std(), 1e-9) * math.sqrt(nimg)
        print(f"  [{oi+1}/{len(ops)}] {o['id']:<26} kunif={kunif:.4f} "
              f"Lambda={lam.mean():>9.0f}+-{lam.std()/math.sqrt(nimg):>6.0f} "
              f"SNR={snr:>5.1f}  L_G={lam_g:>7.1f}  A={lam.mean()/lam_g:>6.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        del y, g2
        torch.cuda.empty_cache()

    outfn = os.path.join(OUT, args.out)
    json.dump(dict(cfg=dict(nimg=nimg, t=args.t, parts=args.parts, refs=refs,
                            targets=targets, ckpt=CKPT, pool=POOL,
                            N=N, cap=cap,
                            note="g2=(1-b)+b*ref2 => a=b*a_ref，形状精确保持"),
                    rows=rows), open(outfn, "w"), indent=1)
    print(f"\n[saved] {outfn}   总耗时 {time.time()-t_start:.0f}s")



if __name__ == "__main__":
    main()
