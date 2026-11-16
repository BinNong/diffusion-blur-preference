r"""第十六组：量化高斯核半像素偏移 bug 对 §3.2 的 Λ(t) 表的影响（GPU，约 4 分钟）

预官（2026-10-23，跑前写定）
================================================================================
为什么有这一组
------------
第十五组 (ak) 定位到一个真 bug：`check_baseline.py` / `check_tdependence.py` /
`evidence_scale*.py` 等 **13 个脚本**手写了高斯核的坐标网格

    lin = torch.linspace(-(ks // 2) + 1, ks // 2, ks)     # ks=15 ⇒ linspace(-6, 7, 15)

而 `putils.convolution.GaussianBlur` 的 `-ks//2 + 1` 因 Python 优先级（一元负号比 `//`
绑得紧）实际是 `(-ks)//2 + 1 = -7`，给出整数网格 [-7..7]。手写版因此**非整数、不对称、
峰偏到第 6 格**——一个不对称的核不是高斯模糊。**§3.2 的 Λ(t) 表正是 `check_tdependence.py`
产出的**，所以论文"最硬的资产"带着这个 bug。

本组把影响量化：同一张图、同一配置、同一 t 网格，**两个核各跑一遍**。

设计（与 check_tdependence.py 逐字对齐，只加第二个核）
--------------------------------------------------
  * 图：pool[255]（check_tdependence 的 `x_star = pool[255:256]`，注释写明"与 check_baseline
    用同一张 x*"）；池 `/tmp/pool_ffhq64_256_s0.pt`；ckpt edm-ffhq-64x64-uncond-vp
  * 核 A（buggy）：逐字复制 check_tdependence 的 `-(ks//2)+1` 网格
  * 核 B（correct）：`putils.convolution.GaussianBlur(kernel_size=15).kernel(1.5)`
  * TSTAR = 1.5；TS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]；t_spans=[80, t]
  * 高斯对照同 check_tdependence（径向分桶 S + t² 下限）
  * Hutchinson 噪声底：t=0.002 处每核重复 K_REP 次

预官判据
--------
M0 核守卫：核 B 必须关于中心格对称（max|k−flip(k)| ≤ 1e-7）、峰在 [7,7]、Σk=1；
   核 A 必须**不**满足对称性（这就是 bug 的判据）。不过则全废。

M1 定位验证（**必须过**）：用**核 A** 复现 §3.2 的表，容差 **5%**。
   为什么是 5% 而不是更紧：§3.3 已测定 Hutchinson 对 logp 的标准差 172–250 nats，
   差值的不确定度约 300 nats，在 t=0.02 处（Λ≈13290）就是 2.3%，所以 5% ≈ 2σ。
   **失败方式**：若核 A 复现不了 §3.2，说明我定位错了脚本/图/配置，M2 的"修正"无效，
   必须先查定位再报任何数字。

M2 修正量（**无阈值，这是本组的产出**）：报核 B 的 Λ(t) 表与逐档 (B−A)/A。
   同时明确：这不是"新结果"，而是把已有结果换到正确的算子上。

M3 诊断（无阈值）：Hutchinson 噪声底（两核各自重复 K_REP 次），用于判断 M2 的逐档差
   是否超过测量噪声。

诚实声明
--------
* 只测**一张图**（pool[255]）——与 §3.2 同图，所以这是同一测量的修正，不是新的统计陈述。
  要给出修正后的 20 图均值是另一个实验。
* 不碰 §3.9(l)/(m) 的主定律：那条用零相位谱门，不经过这个核。

用法： venv2/bin/python check_tdependence_fixed.py
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
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn
from putils.convolution import GaussianBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
POOL = "/tmp/pool_ffhq64_256_s0.pt"
TSTAR = 1.5
KS = 15
TS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
K_REP = 4
S_N = 128
NB = 32

# §3.2 的表（RESULTS 第 187–192 行），pool[255]、TSTAR=1.5、check_tdependence.py 产出
REF32 = {0.002: 33898.0, 0.005: 27792.0, 0.01: 22210.0, 0.02: 13290.0,
         0.05: 3530.0, 0.10: 1054.0, 0.20: 50.0}


def kernel_buggy(s, ks=KS):
    """核 A：check_tdependence.py 逐字复制（不对称，峰的位置偏一格）。"""
    lin = torch.linspace(-(ks // 2) + 1, ks // 2, ks, device=DEV, dtype=DT)
    grid = lin[None] ** 2 + lin[:, None] ** 2
    k = torch.exp(-grid / (2.0 * s ** 2))
    return (k / k.sum())[None, None]


def kernel_correct(s, ks=KS):
    """核 B：putils.convolution.GaussianBlur（对称、峰在中心格）。"""
    return GaussianBlur(kernel_size=ks, device=DEV, dtype=DT).kernel(s)


def khat(k, C, H, W):
    pad = torch.zeros((1, C, H, W), device=DEV, dtype=DT)
    kh, kw = k.shape[-2:]
    t, l = (H - kh) // 2, (W - kw) // 2
    pad[:, :, t:t + kh, l:l + kw] = k
    return torch.fft.rfft2(pad, dim=(-2, -1))


def apply_blur(x, k):
    C, Hh, Ww = x.shape[1], x.shape[2], x.shape[3]
    return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))
                            * khat(k, C, Hh, Ww), s=(Hh, Ww),
                            dim=(-2, -1)).to(DT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tdep_kfix.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, _ = get_ode_flow(prior)

    pool = torch.load(POOL, map_location=DEV).to(DT)
    x_star = pool[255:256].clone()
    xs = pool[:S_N].clone()

    # ---------------- M0 核守卫 ----------------
    kA, kB = kernel_buggy(TSTAR), kernel_correct(TSTAR)
    symA = float((kA[0, 0] - kA[0, 0].flip(0).flip(1)).abs().max())
    symB = float((kB[0, 0] - kB[0, 0].flip(0).flip(1)).abs().max())
    pkB = [int(v) for v in torch.nonzero(kB[0, 0] == kB[0, 0].max()).flatten().tolist()]
    print("[M0] 核 A（buggy）对称性 max|k−flip(k)| = %.3e；峰位置 %s"
          % (symA, [int(v) for v in torch.nonzero(kA[0, 0] == kA[0, 0].max())
                    .flatten().tolist()]))
    print("[M0] 核 B（correct）对称性 = %.3e；峰位置 %s；Σk = %.9f"
          % (symB, pkB, float(kB.sum())))
    # **实现修正（2026-10-23 18:2x，披露）**：首版写 `[7,7] in pkB`，而 pkB 是
    # torch.nonzero(...).flatten().tolist() 出来的**扁平坐标对** [7,7]，不是位置列表，
    # 所以 `[7,7] in [7,7]` 恒为 False，M0 被误判为失败。改为直接比对坐标对。
    ok0 = symB <= 1e-7 and pkB == [7, 7] and symA > 1e-7
    print("[M0] %s" % ("✅ 核 B 对称且峰在中心、核 A 不对称（即 bug 的判据）"
                       if ok0 else "❌ 守卫失败，不报任何物理"))
    if not ok0:
        raise SystemExit(2)

    # ---------------- 径向谱（与 check_tdependence 同法）----------------
    fx = np.fft.fftfreq(64) * 64
    fy = np.fft.fftfreq(64) * 64
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)
    # **与 check_tdependence.py 的 radial_bins() 逐字对齐**：分桶建在 rfft 半边网格上。
    # 首版用了全网格 (64,64)，与 Pc 的 (S_N, 64*33) 列数不匹配而崩。
    Rr = R[:, :64 // 2 + 1]
    edges = np.linspace(0, Rr.max() + 1e-6, NB + 1)
    idx_t = np.clip(np.digitize(Rr.ravel(), edges) - 1, 0, NB - 1).reshape(Rr.shape)
    with torch.no_grad():
        P = (torch.fft.rfft2(xs.to(torch.float64), dim=(-2, -1)).abs() ** 2) / (64 * 64)
        Pc = P.mean(1)
        Srad = torch.zeros(NB, device=DEV, dtype=torch.float64)
        for b in range(NB):
            m = (idx_t == b).reshape(-1)
            if m.any():
                Srad[b] = Pc.reshape(S_N, -1)[:, m].mean()

    @torch.no_grad()
    def exact_logp(z, t):
        return -float(likelihood_fn(z.to(DT), ode_flow,
                                    t_spans=[80.0, float(t)]).item())

    def gauss_logp(z_pow, t):
        Sf = (Srad + t ** 2)[idx_t][None, None]
        return float(-(z_pow / (2 * Sf)).sum()
                     - 0.5 * torch.log(2 * np.pi * Sf).sum())

    t0 = time.time()
    res = {}
    for nm, k in (("buggy", kA), ("correct", kB)):
        y = apply_blur(x_star, k)
        Zx = (torch.fft.rfft2(x_star.to(torch.float64), dim=(-2, -1)).abs() ** 2
              ).mean(1, keepdim=True) / (64 * 64)
        Zy = (torch.fft.rfft2(y.to(torch.float64), dim=(-2, -1)).abs() ** 2
              ).mean(1, keepdim=True) / (64 * 64)
        rows = []
        for t in TS:
            ey, ex = exact_logp(y, t), exact_logp(x_star, t)
            dg = gauss_logp(Zy, t) - gauss_logp(Zx, t)
            rows.append(dict(t=t, logp_blur=ey, logp_sharp=ex, lam=ey - ex,
                             lam_gauss=dg, amp=(ey - ex) / dg if abs(dg) > 1e-6
                             else float("nan")))
        # M3：Hutchinson 噪声底
        vy = [exact_logp(y, 0.002) for _ in range(K_REP)]
        vx = [exact_logp(x_star, 0.002) for _ in range(K_REP)]
        res[nm] = dict(rows=rows,
                       hutch=dict(sy=float(np.std(vy)), sx=float(np.std(vx)),
                                  sdiff=float(np.sqrt(np.var(vy) + np.var(vx)))))
        print("[%s] Hutchinson@t=0.002: sd(logp_y)=%.1f sd(logp_x)=%.1f "
              "sd(Λ)=%.1f  (%d reps)" % (nm, res[nm]["hutch"]["sy"],
                                         res[nm]["hutch"]["sx"],
                                         res[nm]["hutch"]["sdiff"], K_REP), flush=True)

    print("\n" + "=" * 96)
    print("Λ(t) = log p_t(高斯σ=1.5⋆x*) − log p_t(x*)，x* = pool[255]")
    print("=" * 96)
    print(f"{'t':>7} | {'§3.2(核A)':>10} | {'核A本组':>10} | {'核B本组':>10} | "
          f"{'B−A':>9} | {'(B−A)/A':>9} | {'B vs §3.2':>10}")
    n_ok = 0
    for rA, rB in zip(res["buggy"]["rows"], res["correct"]["rows"]):
        t = rA["t"]
        d = rB["lam"] - rA["lam"]
        relA = (rA["lam"] - REF32[t]) / REF32[t]
        ok = abs(relA) <= 0.05
        n_ok += ok
        print(f"{t:>7} | {REF32[t]:>10.0f} | {rA['lam']:>+10.1f} | {rB['lam']:>+10.1f} | "
              f"{d:>+9.1f} | {d/rA['lam']:>+9.2%} | {relA:>+10.2%} "
              f"{'✅' if ok else '❌'}")
    print(f"\n[M1] 核 A 复现 §3.2：{n_ok}/{len(TS)} 档在 5% 内"
          f"（Hutchinson 对差值的不确定度 ≈ {res['buggy']['hutch']['sdiff']:.0f} nats）")
    print(f"[M2] 修正量：Λ(t) 换到正确核后系统性变"
          f"{'低' if mean_of(res, 'correct') < mean_of(res, 'buggy') else '高'}，"
          f"逐档见上表。")
    print(f"[总耗时] {time.time()-t0:.0f}s")

    outfn = os.path.join(OUT, args.out)
    json.dump(dict(cfg=dict(ckpt=CKPT, pool=POOL, tstar=TSTAR, ks=KS, ts=TS,
                            k_rep=K_REP, img_index=255, ref32=REF32,
                            sym_buggy=symA, sym_correct=symB),
                   res=res), open(outfn, "w"), indent=1)
    print(f"[saved] {outfn}")


def mean_of(res, nm):
    return sum(r["lam"] for r in res[nm]["rows"]) / len(res[nm]["rows"])


if __name__ == "__main__":
    main()
