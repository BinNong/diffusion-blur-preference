r"""第十八组：过模糊失效模式搬到 **20 张 EDM 生成池图 + 正确核**（GPU，约 15.5 小时）

图源勘误（2026-11-11，跑后仅修正文档与未来输出的元数据，不改冻结判据或算法）：
`make_pool.py` 通过 EDM `ode_sampler` 生成此池，旧称“真实池图/held-out”错误。
两图池均为先验样本；原始 JSON 保留原标签，撤回记录见 RESULTS §3.9(ce)。

预官（2026-10-29，跑前写定；md5 见 RESULTS §3.9(bt)）
================================================================================
为什么有这一组
---------------
§3.9(br) 核账时发现论文 §8 的过模糊研究只有 **2 张图**（`phaseC_kfix.json` 的
`cfg.nimg=2`，8 个「configuration」= 2 图 × 4 高噪声档），而且那 2 张是
**ode_sampler 生成的先验样本**，不是真实图。§3.9(aj) 早已指出先验样本对
$q_t(x^\star)$ 类论断是方法学让步。过模糊是论文四大贡献之一，却立在 2 张先验样本上——
审稿人一句「two images, and prior samples at that」就能打回来。

本组**只改两件事**：
  (a) **图像输入**：旧跑即时生成的先验样本 → 缓存 EDM 生成池
      `/tmp/pool_ffhq64_256_s0.pt` 的前 20 张（旧“真实池图”称呼已撤回）；
  (b) **图数**：2 → 20。
其余逐字与 `repro_phaseC_kfix.py` 相同：正确核（putils）、$\theta^\blur$=1.5、
21 点 $\theta$ 网格、6 个噪声档、`N_INNER=240`、HQS 两个初值、$L$ 的定义与落盘字段。
所以与旧跑的差别只有「缓存图池 + 图数」，可以直接比较「过模糊是否还在」。
（旧跑的臂 A 是错核定位验证，图源一变它的参照 `phaseC_sweep.json` 就不可比，故本组不跑臂 A；
核不变性那条论断仍由 §3.9(am) 的 2 图双核跑支撑，论文里照实标注样本量。）

预官判据（在看到任何数字之前写定）
----------------------------------
N0 核守卫（不过则一个数字都不报）：正确核必须对称（不对称度 ≤1e-7）、峰 (7,7)、
   与 putils 逐比特相同、kunif(σ=1.5) 与预官值 0.962855 一致到 1e-4。

N1 主判据一：**σ_n ≥ 0.172 的格子上 argmin θ > θ*=1.5 的比例**。
   旧跑（2 图）是 8/8。要求报出 20 图版的比例；**若 < 16/20，论文 §8 的「8/8」必须
   改成新值**，并据实说明这是 20 张真实图上的比例。

N2 主判据二：**过模糊因子 θ_argmin/θ*** 在该子集上的 mean 与范围。
   旧跑（2 图）mean=1.4896、范围 [1.333, 1.667]（即高出真值 33–67%）。
   要求报出 20 图版；若 mean 落在 [1.30, 1.65] 之外，论文的「33–67%」必须改。

N3 主判据三：**θ∈[2.0,2.5] 整段相对 θ\*** 的优势区间。
   旧跑（2 图）是 1211–2781 nats（39/40 个（格, 点）组合优于 θ\*）。
   要求报出 20 图版的最小/最大优势，以及优于 θ\* 的（格, 点）组合占比。

N4 诊断（无阈值）：低噪声档（σ_n = 0.0415、0.0845）上 argmin θ 是否仍 ≈ θ\*
   （旧跑 img0 给 1.625/1.625、img1 给 1.5/1.5）——即过模糊是**噪声依赖的**而非恒定偏置；
   以及 top-5 的 L 跨度（旧跑 247–1031 nats，与 Hutchinson 底同量级）。

失败后果：N1/N2/N3 任一项超出上列范围，论文 §8 对应数字必须改，并在 RESULTS 记录
改了多少、原值是什么。不报「最接近的通过变体」。

用法：
  cd /data/experiment/de_conv_1
  ./venv2/bin/python repro/repro_phaseC_pool20.py --out phaseC_pool20.json
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
# **本组只改两件事**：(a) 图源从先验样本换成**真实池图**；(b) 图数 2 → 20。
POOL = "/tmp/pool_ffhq64_256_s0.pt"
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
    ap.add_argument("--out", default="phaseC_kfix.json")
    args = ap.parse_args()
    if args.smoke and args.out == "phaseC_kfix.json":
        args.out = "smoke_phaseC_kfix.json"

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
    samples = pool[:NIMG].clone()
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
                            image_source="EDM prior-generated pool (first %d of %s; corrected provenance 2026-11-11)" % (NIMG, POOL),
                            thetas=THETAS, noises_old=NOISES_OLD,
                            arm_b=dict(imgs=ARM_B_IMG, noises=ARM_B_NOISES),
                            kernel_guard=guard),
                   records=recB), open(outfn, "w"), indent=1)
    print(f"\n[saved] {outfn}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
