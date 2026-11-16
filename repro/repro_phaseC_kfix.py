r"""第十七组：Phase C 的 marginalized profile 在**正确高斯核**上重测（GPU，约 90 分钟）

预官（2026-10-23，跑前写定）
================================================================================
为什么有这一组（开放问题 13 + 开放问题 6）
--------------------------------------
§3.9(aj)–(al) 定位到一个 13 个脚本共有的高斯核半像素偏移 bug，并已量化它对 §3.2 的 Λ(t) 表
和 §3.9(z)/(ab) 的 Λ_clean 的影响。**尚未清的最后一项是 Phase C / D**——它用的是同一个错核。

Phase C 的产出里有一条**活着的开放问题**（本文开放问题 6）：
    「过度模糊失效模式：Phase C 和 D 都观察到高噪声下估计核系统性大于真值。」
从 `results/phaseC_sweep.json`（错核、2 图 × 12 噪声 × 21 个 θ）直接读出：

    噪声 ↑ ⇒ argmin_θ 从 ~1.25–1.5 单调升到 ~2.0–2.5（真值 θ\*=1.5）
    σ_n ≥ 0.172 的 8 个格子 argmin θ 全部 > θ\*：2.25/2.375/2.0/2.5（img0）、
    2.0/1.75/2.5/2.375（img1）；过度模糊因子 θ_argmin/θ\* 的均值 = **1.479**

本组把它在正确核上重测。

**一个跑前就必须算清楚的前提：两个核的 kunif 不同，所以「θ\*=1.5」在两边不是同一个模糊强度。**

    σ=1.5：kunif(错核) = 0.967782，kunif(正确核) = 0.962855   ⇒ 错核**更弱**
    kunif 匹配的正确 σ：错核 1.5 ≡ 正确 **1.6154**；错核 2.0 ≡ 正确 2.1530；
                        错核 2.5 ≡ 正确 2.6871（近似线性，斜率 ≈1.077）

所以**正确的比较量是过度模糊因子 θ_argmin/θ\***（它对 kunif 映射近似不变），
而不是 θ_argmin 的绝对值。

设计（与 repro_phaseC.py 逐字对齐，只换核）
------------------------------------------
  * ckpt edm-ffhq-64x64-uncond-vp；θ\* = 1.5；N_INNER = 240；2 个初值取最优
  * 图：`ode_sampler(shape=(2,3,64,64))`、seed 0 —— 与旧跑**同一批图**（先验样本）
  * θ 网格 21 点（0 … 2.5，步长 0.125）—— 与旧跑相同，保证 argmin 分辨率可比
  * 噪声种子 `1000*i + int(sn*100000)` —— 逐字复制
  * **臂 A（定位验证，错核）**：img0、噪声取旧跑极值 {0.01, 0.5}、全 θ 网格。
    目的是证明"我把旧脚本的配置复现对了"，否则臂 B 的数字没有参照。
  * **臂 B（重测，正确核）**：2 张图、6 个对数间隔噪声
    {0.0415, 0.0845, 0.1720, 0.2455, 0.3504, 0.5000}（旧跑 12 档的中–高段；
    低四档 0.01–0.0291 的答案是"θ=0 或 θ\* 胜"，对过度模糊论断没有信息量，
    砍掉以把 GPU 压到约 90 分钟）。

预官判据（在看到任何数字之前写定）
----------------------------------
N0 核守卫（**零 GPU，不过则不启动任何一臂**）：
   (a) 正确核关于中心格对称 max|k−flip(k)| ≤ 1e−7、峰 [7,7]、Σk=1；
   (b) 与 `putils.GaussianBlur(15).kernel(1.5)` 逐比特相同；
   (c) **错核必须不满足 (a)**（峰 [6,6]、不对称 3.159e−02）——这就是 bug 的判据；
   (d) 报出两个核在 σ=1.5 的 kunif 与 kunif 匹配映射（预官值 0.967782 / 0.962855 /
       1.6154），用于 N3 的换算。

N1 定位验证（**必须过；不过则臂 B 一个数字都不报**）：
   臂 A（错核）在 img0、sn ∈ {0.01, 0.5} 上，逐 θ 的 L 与 `phaseC_sweep.json` 的差
   **|ΔL| ≤ 600 nats**，且 argmin θ **完全相同**。
   容差依据：Hutchinson 对 logp 的噪声底实测 172–250 nats（§3.3）、对差值是 424 nats
   （§3.9(al) M3），600 ≈ 1.4× 该底。**两臂的全局 RNG 序列不同**（臂 A 只跑 2 个噪声档），
   所以不要求逐位相同，只要求在噪声底内。
   **失败方式**：argmin θ 不同或 |ΔL| 普遍超差 ⇒ 我的脚本与旧跑配置不一致，先查再报。

N2 主判据（过度模糊在正确核上存活）：臂 B 中 σ_n ≥ 0.172 的 8 个格子上，
   argmin θ > θ\* = 1.5 的比例。旧跑（错核）**8/8**。
   **通过线 ≥ 6/8**；**否掉线 ≤ 4/8**（则"过度模糊"主要是错核的伪影）。
   同时报全部 12 个格子的 argmin θ 表。

N3 最尖的一条（**过度模糊因子对 kunif 映射不变**）：臂 B 在 σ_n ≥ 0.172 的 8 个格子上，
   mean(θ_argmin/θ\*) 落在旧跑值 **1.479 ± 0.15** 即 [1.33, 1.63] 内。
   机理预言：该现象应由模糊强度（kunif）决定，而 kunif 映射近似线性（斜率 1.077），
   所以因子不变。**失败方式**：掉出 [1.33, 1.63] ⇒ 过度模糊幅度依赖核的精细结构
   （不对称性），旧跑的比例数要按新核重报。
   同时报 kunif 匹配版：把臂 B 的 argmin θ 换算成"等 kunif 的错核 σ"，应与旧跑 argmin 一致。

N4 盆地结构（诊断 + 一条通过线）：逐格报 L(θ=0) − L(θ 最接近 θ\*) 与"盆地胜/θ=0 胜"。
   旧跑在低中噪声有 3 格 argmin=0（img0 sn=0.0592；img1 sn=0.0143/0.0204）。
   **通过线**：臂 B 的 12 格里 argmin=0 的格子数 ≤ 4（即 θ=0 深井不占主导）；
   并报"basin collapse 阈值是否出现"（旧跑结论：未出现）。

N5 诊断（无阈值）：(a) 逐格 argmin θ 与 L 的全表；(b) 臂 B 与旧跑在**重叠噪声档**
   （0.172/0.2455/0.3504/0.5）上的 argmin 差；(c) 绝对 L 的偏移量级；
   (d) 两个核在 θ 网格各点的 kunif，用于 N3 的换算。

诚实声明
--------
* **旧跑的结果在写本预官时已全部可见**（argmin 表、1.479 这个数都是先读出来的）。
  本组的 N2/N3 阈值是从旧跑值推出的，**没有放宽空间**：N2 通过线 6/8 低于旧跑 8/8，
  N3 带宽 ±0.15 覆盖旧跑全部 8 个格子的散度（1.167–1.667）。
* 只测 θ\* = 1.5 一个核强度、2 张先验样本图、ffhq 一个先验、64×64。
* 臂 B 的噪声档是旧跑 12 档的子集（砍掉低四档），所以 N2/N3 只在中–高噪声上有陈述。
* HQS 的设置（240 内循环、2 初值）与旧跑相同；`diag_phaseC.py` 已诊断 n_inner 60→240→600
  只波动 0.3–1.5%、多初值只值 1–3%。

用法： venv2/bin/python repro_phaseC_kfix.py [--smoke]
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

# 旧跑（repro_phaseC.py --sweep）的配置，逐字复制
NOISES_OLD = [round(float(v), 4) for v in np.logspace(np.log10(0.01), np.log10(0.5), 12)]
THETAS = [round(0.125 * i, 4) for i in range(21)]      # 0 .. 2.5
NIMG = 2
# 臂 A：定位验证（错核），img0，旧跑的极值噪声
ARM_A_IMG = [0]
ARM_A_NOISES = [0.01, 0.5]
# 臂 B：重测（正确核），旧跑 12 档的中–高段
ARM_B_IMG = [0, 1]
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

    bA, bB = BlurBuggy(), BlurCorrect()

    # ---------------- N0 核守卫（零 GPU）----------------
    kA, kB = bA.kernel(TSTAR), bB.kernel(TSTAR)
    symA = float((kA[0, 0] - kA[0, 0].flip(0).flip(1)).abs().max())
    symB = float((kB[0, 0] - kB[0, 0].flip(0).flip(1)).abs().max())
    pkA = [int(v) for v in torch.nonzero(kA[0, 0] == kA[0, 0].max()).flatten().tolist()]
    pkB = [int(v) for v in torch.nonzero(kB[0, 0] == kB[0, 0].max()).flatten().tolist()]
    bitid = bool(torch.equal(bA.A(torch.zeros(1, 3, 64, 64, device=DEV), 0.0),
                             torch.zeros(1, 3, 64, 64, device=DEV)))
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
    kuA, kuB = kunif_of(bA, TSTAR), kunif_of(bB, TSTAR)
    match = {s: sigma_for_kunif(bB, kunif_of(bA, s)) for s in (1.5, 2.0, 2.5)}
    guard = dict(sym_buggy=symA, peak_buggy=pkA, sym_correct=symB, peak_correct=pkB,
                 sum_correct=float(kB.sum()), bitident_correct=bitB,
                 kunif_buggy_15=kuA, kunif_correct_15=kuB, sigma_match=match)
    print("[N0 核守卫]", json.dumps(guard, ensure_ascii=False), flush=True)
    print(f"[N0] 错核：不对称 {symA:.3e}、峰 {pkA}（应为 [6,6]）；"
          f"正确核：不对称 {symB:.3e}、峰 {pkB}、Σk={float(kB.sum()):.9f}、"
          f"与 putils 逐比特相同={bitB}", flush=True)
    print(f"[N0] kunif(σ=1.5)：错核 {kuA:.6f}、正确核 {kuB:.6f}（预官 "
          f"{KUNIF_BUGGY_15:.6f} / {KUNIF_CORRECT_15:.6f}）；"
          f"kunif 匹配的正确 σ = {match}", flush=True)
    ok0 = (symB <= 1e-7 and pkB == [7, 7] and bitB and symA > 1e-7
           and abs(kuA - KUNIF_BUGGY_15) < 1e-4
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

    # 与旧跑同一批图：manual_seed(0) → load → ode_sampler(shape=(NIMG,...))
    samples, _ = ode_sampler(ode_flow, shape=(NIMG, 3, 64, 64), t_spans=t_spans,
                             device=DEV)
    print(f"[sample] {NIMG} sharp images（与旧跑同 seed 同 shape）", flush=True)

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
    aA_noises = smoke_noises if args.smoke else ARM_A_NOISES
    aB_noises = smoke_noises if args.smoke else ARM_B_NOISES
    aA_imgs = [0] if args.smoke else ARM_A_IMG
    aB_imgs = [0] if args.smoke else ARM_B_IMG

    recA = run_arm(bA, "buggy", aA_imgs, aA_noises, thetas, t0)
    recB = run_arm(bB, "correct", aB_imgs, aB_noises, thetas, t0)

    outfn = os.path.join(OUT, args.out)
    json.dump(dict(cfg=dict(nimg=NIMG, tstar=TSTAR, ks=KS, n_inner=N_INNER,
                            thetas=THETAS, noises_old=NOISES_OLD,
                            arm_a=dict(imgs=ARM_A_IMG, noises=ARM_A_NOISES),
                            arm_b=dict(imgs=ARM_B_IMG, noises=ARM_B_NOISES),
                            kernel_guard=guard),
                   records=recA + recB), open(outfn, "w"), indent=1)
    print(f"\n[saved] {outfn}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
