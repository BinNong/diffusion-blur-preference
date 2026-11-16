r"""第十九组：过模糊机理在同源不相交图池的复现（GPU，约 16 小时）

图源勘误（2026-11-11，跑后仅修正文档与未来输出的元数据，不改冻结判据或算法）：
`make_pool.py` 通过 EDM `ode_sampler` 生成此池，旧称“真实池图/held-out”错误。
“独立”仅指池索引不相交，不是独立真实数据队列或噪声重复；见 RESULTS §3.9(ce)。

预官 2026-11-08 16:41:32 +0800；判据原文见 RESULTS §3.9(bz) 与
repro/judge_overblur_indep.py 的 docstring（md5 3afae722…）。
**本脚本只实现已冻结的数据契约，不得改动任何判据或物理设置。**

与第十八组 repro_phaseC_pool20.py 逐字相同，只改：
  (a) 池图切片 pool[:20] -> pool[20:40]（同源不相交索引，第 20–39 张）；
  (b) cfg 增加 pool_offset=20，并把 image_source 文字改为 offset 口径；
  (c) 默认 --out overblur_indep.json（冒烟输出名同步改）。
其余逐字不变：正确核（putils）、theta*=1.5、21 点 theta 网格、6 噪声档、
N_INNER=240、HQS 两初值、L 的定义与落盘字段（含 data / q）。

注意：噪声实现仍用 1000*i + int(sn*100000) 的固定 seed，与第十八组相同；
两组只有图像索引不相交，不是独立真实数据或独立噪声重复，见 §3.9(ce)。

用法：
  cd /data/experiment/de_conv_1
  ./venv2/bin/python repro/sweep_overblur_indep.py --out overblur_indep.json
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
from putils.convolution import GaussianBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
IMG_LO, IMG_HI = -1.0, 1.0
N_INNER = 240
TSTAR = 1.5
KS = 15

# 与 repro_phaseC_kfix.py 逐字相同的网格/协议
NOISES_OLD = [round(float(v), 4) for v in np.logspace(np.log10(0.01), np.log10(0.5), 12)]
THETAS = [round(0.125 * i, 4) for i in range(21)]      # 0 .. 2.5
# 缓存图池由 make_pool.py 生成，不是真实数据集；本组使用索引 20–39（§3.9(ce)）。
POOL = "/tmp/pool_ffhq64_256_s0.pt"
POOL_OFFSET = 20
NIMG = 20
ARM_B_IMG = list(range(NIMG))
ARM_B_NOISES = [0.0415, 0.0845, 0.1720, 0.2455, 0.3504, 0.5000]

# §3.9(aj)/(al) 定位的 kunif（预官时离线算出，N0(d) 用）
KUNIF_BUGGY_15 = 0.967782
KUNIF_CORRECT_15 = 0.962855
# kunif 匹配映射：错核 σ → 等 kunif 的正确 σ（预官时离线算出）
SIGMA_MATCH = {1.5: 1.6154, 2.0: 2.1530, 2.5: 2.6871}


class BlurBuggy:
    """臂 A 用：repro_phaseC.py 的 BlurOp 逐字复制（不对称、峰 [6,6]）。"""

    def __init__(self, ks=KS):
        lin = torch.linspace(-(ks // 2) + 1, ks // 2, ks, device=DEV, dtype=DT)
        self.grid = lin[None] ** 2 + lin[:, None] ** 2

    def kernel(self, s):
        if s == 0:
            return None
        k = torch.exp(-self.grid / (2.0 * s ** 2))
        return (k / k.sum())[None, None]

    def _K(self, k, C, H, W):
        pad = torch.zeros((1, C, H, W), device=DEV, dtype=DT)
        kh, kw = k.shape[-2:]
        t, l = (H - kh) // 2, (W - kw) // 2
        pad[:, :, t:t + kh, l:l + kw] = k
        return torch.fft.rfft2(pad, dim=(-2, -1))

    def A(self, x, s):
        k = self.kernel(s)
        if k is None:
            return x
        B, C, H, W = x.shape
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1)) * self._K(k, C, H, W),
                                s=(H, W), dim=(-2, -1))

    def AT(self, x, s):
        k = self.kernel(s)
        if k is None:
            return x
        B, C, H, W = x.shape
        return torch.fft.irfft2(torch.conj(self._K(k, C, H, W))
                                * torch.fft.rfft2(x, dim=(-2, -1)), s=(H, W), dim=(-2, -1))

    def eig(self, s, C, H, W):
        k = self.kernel(s)
        if k is None:
            return torch.ones((1, C, H, W // 2 + 1), device=DEV, dtype=DT)
        return self._K(k, C, H, W).abs() ** 2


class BlurCorrect:
    """臂 B 用：核一律取自 putils.convolution.GaussianBlur（对称、峰 [7,7]）。
    其余（_K / A / AT / eig）与 BlurBuggy 逐字相同，只换 kernel() 的来源。"""

    def __init__(self, ks=KS):
        self.ks = ks
        self.gb = GaussianBlur(kernel_size=ks, device=DEV, dtype=DT)

    def kernel(self, s):
        if s == 0:
            return self.gb.dirac
        return self.gb.kernel(s)

    def _K(self, k, C, H, W):
        pad = torch.zeros((1, C, H, W), device=DEV, dtype=DT)
        kh, kw = k.shape[-2:]
        t, l = (H - kh) // 2, (W - kw) // 2
        pad[:, :, t:t + kh, l:l + kw] = k
        return torch.fft.rfft2(pad, dim=(-2, -1))

    def A(self, x, s):
        k = self.kernel(s)
        B, C, H, W = x.shape
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1)) * self._K(k, C, H, W),
                                s=(H, W), dim=(-2, -1))

    def AT(self, x, s):
        k = self.kernel(s)
        B, C, H, W = x.shape
        return torch.fft.irfft2(torch.conj(self._K(k, C, H, W))
                                * torch.fft.rfft2(x, dim=(-2, -1)), s=(H, W), dim=(-2, -1))

    def eig(self, s, C, H, W):
        if s == 0:
            return torch.ones((1, C, H, W // 2 + 1), device=DEV, dtype=DT)
        return self._K(self.kernel(s), C, H, W).abs() ** 2


def kunif_of(blur, s, C=3):
    h2 = blur.eig(s, C, 64, 64)[0].to(torch.float64)
    return float((1.0 - h2).mean())


def sigma_for_kunif(blur, target, lo=0.3, hi=4.0):
    """二分找与 target 等 kunif 的 σ。"""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if kunif_of(blur, mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="overblur_indep.json")
    args = ap.parse_args()
    if args.smoke and args.out == "overblur_indep.json":
        args.out = "smoke_overblur_indep.json"

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)

    bB = BlurCorrect()

    # ---------------- N0 核守卫（零 GPU）----------------
    kB = bB.kernel(TSTAR)
    symB = float((kB[0, 0] - kB[0, 0].flip(0).flip(1)).abs().max())
    pkB = [int(v) for v in torch.nonzero(kB[0, 0] == kB[0, 0].max()).flatten().tolist()]
    bitid = True
    xprobe = torch.load("/tmp/pool_ffhq64_256_s0.pt", map_location=DEV).to(DT)[:2]
    ref = GaussianBlur(kernel_size=KS, device=DEV, dtype=DT).kernel(TSTAR)
    # 逐比特对账：bB 的模糊结果 vs 直接用 putils 核做的循环卷积
    pad = torch.zeros((1, 3, 64, 64), device=DEV, dtype=DT)
    kh, kw = ref.shape[-2:]
    t_, l_ = (64 - kh) // 2, (64 - kw) // 2
    pad[:, :, t_:t_ + kh, l_:l_ + kw] = ref
    Khat = torch.fft.rfft2(pad, dim=(-2, -1))
    yB = bB.A(xprobe, TSTAR)
    yref = torch.fft.irfft2(torch.fft.rfft2(xprobe, dim=(-2, -1)) * Khat,
                            s=(64, 64), dim=(-2, -1)).to(DT)
    bitB = bool(torch.equal(yB, yref))
    kuB = kunif_of(bB, TSTAR)
    guard = dict(sym_correct=symB, peak_correct=pkB, sum_correct=float(kB.sum()),
                 bitident_correct=bitB, kunif_correct_15=kuB)
    print("[N0 核守卫]", json.dumps(guard, ensure_ascii=False), flush=True)
    print(f"[N0] 正确核：不对称 {symB:.3e}、峰 {pkB}、Σk={float(kB.sum()):.9f}、"
          f"与 putils 逐比特相同={bitB}；kunif(σ=1.5)={kuB:.6f}（预官 "
          f"{KUNIF_CORRECT_15:.6f}）", flush=True)
    ok0 = (symB <= 1e-7 and pkB == [7, 7] and bitB
           and abs(kuB - KUNIF_CORRECT_15) < 1e-4)
    print(f"[N0] {'✅ 守卫通过' if ok0 else '❌ 守卫失败，不启动任何一臂'}", flush=True)
    if not ok0:
        raise SystemExit(2)

    @torch.no_grad()
    def hqs(blur, theta, y, sn, z0):
        B, C, H, W = y.shape
        ATy = blur.AT(y, theta)
        denom = blur.eig(theta, C, H, W) + 1.0
        z = z0.clone()
        for _ in range(N_INNER):
            xh = denoiser(z, sn).to(DT)
            v = 2 * xh - z
            xn = torch.fft.irfft2(torch.fft.rfft2(ATy + v, dim=(-2, -1)) / denom,
                                  s=(H, W), dim=(-2, -1))
            z = torch.clamp(z + xn - xh, IMG_LO, IMG_HI)
        return denoiser(z, sn).to(DT)

    @torch.no_grad()
    def q_value(x):
        return float(likelihood_fn(x.to(DT), ode_flow, t_spans=t_spans).item())

    # 缓存图池由 make_pool.py 通过 EDM ode_sampler 生成，不是真实数据集（§3.9(ce)）。
    # 过模糊研究的对象是 profile L(θ)，真实图上的外部有效性仍需另测。
    pool = torch.load(POOL, map_location=DEV).to(DT)
    samples = pool[POOL_OFFSET:POOL_OFFSET + NIMG].clone()
    print(f"[pool] 载入 {NIMG} 张 EDM 生成池图 {POOL}", flush=True)

    def run_arm(blur, arm, imgs, noises, thetas, t0):
        recs = []
        for i in imgs:
            x_star = samples[i:i + 1].to(DT)
            y_clean = blur.A(x_star, TSTAR)
            for sn in noises:
                g = torch.Generator(device=DEV).manual_seed(1000 * i + int(sn * 100000))
                y = (y_clean + sn * torch.randn(y_clean.shape, generator=g, device=DEV,
                                                dtype=DT)).to(DT)
                prof = []
                for th in thetas:
                    best = None
                    for z0 in (y, blur.A(y, th)):
                        x = hqs(blur, th, y, sn, z0)
                        resid = blur.A(x, th) - y
                        data = float((resid ** 2).sum().item() / (2 * sn ** 2))
                        q = q_value(x)
                        if best is None or data + q < best["L"]:
                            best = dict(theta=th, L=data + q, data=data, q=q)
                        del x
                        torch.cuda.empty_cache()
                    prof.append(best)
                recs.append(dict(arm=arm, img=i, noise=sn, theta_star=TSTAR,
                                 profile=prof))
                Ls = np.array([p["L"] for p in prof])
                ths = np.array([p["theta"] for p in prof])
                ig = int(np.argmin(Ls))
                istar = int(np.argmin(np.abs(ths - TSTAR)))
                print(f"  [{arm} img{i} sn={sn:<6}] argmin θ={ths[ig]:<6} "
                      f"L(0)={Ls[0]:>10.1f} L(θ*)={Ls[istar]:>10.1f} "
                      f"θ=0胜={'Y' if Ls[0] < Ls[istar] else 'N'}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
        return recs

    t0 = time.time()
    # 冒烟用 5 点 θ 网格 + 1 张图 + 2 个噪声，只验代码路径，不产物理数
    thetas = [0.0, 0.75, 1.5, 2.0, 2.5] if args.smoke else THETAS
    smoke_noises = [0.05, 0.3] if args.smoke else None
    aB_noises = smoke_noises if args.smoke else ARM_B_NOISES
    aB_imgs = [0] if args.smoke else ARM_B_IMG

    recB = run_arm(bB, "correct", aB_imgs, aB_noises, thetas, t0)

    outfn = os.path.join(OUT, args.out)
    json.dump(dict(cfg=dict(nimg=NIMG, tstar=TSTAR, ks=KS, n_inner=N_INNER,
                            pool_offset=POOL_OFFSET,
                            image_source="EDM prior-generated pool (offset %d, %d images of %s; corrected provenance 2026-11-11)" % (POOL_OFFSET, NIMG, POOL),
                            thetas=THETAS, noises_old=NOISES_OLD,
                            arm_b=dict(imgs=ARM_B_IMG, noises=ARM_B_NOISES),
                            kernel_guard=guard),
                   records=recB), open(outfn, "w"), indent=1)
    print(f"\n[saved] {outfn}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
