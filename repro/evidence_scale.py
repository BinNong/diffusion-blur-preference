"""全局 MAP 退化的大小由什么定标？——「θ=0 vs θ* 证据」的精确计算（GPU）。

预官（2026-10-20，跑前写定）
================================================================================
要检验什么（RESULTS §五点五 / §3.9(y6) / 开放问题 12）
--------------------------------------------------
统一图景的**幅度半**说「任何算法的失败幅度都由 Λ 定标」。原表三个点：
    高斯谱近似在 θ=0 的误差   -25933 / -14313     Λ = 13290 / ~3000
    PnP-HQS 不动点偏差          ~6600              Λ ≈ 700
    先验重要性采样 ESS          1/64               Λ ≈ 20
隐含比值 1.95 / 4.8 / 9.4 漂了 5 倍，且 ESS 那行不在同一量纲。§3.9(y) 已把机理半做定
（二阶只捕获 Λ 的 15.3%），但幅度半一直没检验，因为**三个「失败幅度」根本不是同一个量**。

本实验先把其中最容易定义清楚的那一个做成精确量：**全局 MAP 的退化幅度**。

定义（与 test_lemma12.py 逐字对齐）
----------------------------------
    L(x, θ, y) = ||h_θ ⋆ x - y||^2 / (2σ_n^2) + q_t(x)
    L_star     = L(x*, θ*, y) = ||η||^2/(2σ_n^2) + q_t(x*)      （可精确计算，无优化器）
    best       = min_x L(x, 0, y)                                 （θ=0 处的最优）
    E          = best - L_star                                    （nats，可正可负）

Lemma 12 预言 E ≤ 0（θ=0 永远不差于真值）。§五点五 把实测 E=+6600 当成「PnP-HQS 失败
幅度」，但 test_lemma12.py 自己的 [说明] 已写：「若违反且差距大，优先怀疑 HQS 在 θ=0
未收敛到真极小」。

**出发点（想判据时已想清楚）**：x = y 本身就是一个**可行的** θ=0 点（数据项为 0），于是
    E_y := L(y) - L_star = q_t(y) - ||η||^2/(2σ_n^2) - q_t(x*) = -(Λ + ||η||^2/(2σ_n^2))
其中 Λ(t) = log p_t(y) - log p_t(x*) = q_t(x*) - q_t(y)。
所以 |E_y| = Λ(σ_n) + ||η||^2/(2σ_n^2)：**退化幅度 = 模糊偏好 + 一个与模糊强度无关的
数据项地板**。这就是幅度半的精确形式，也是本实验要判的东西。

**设计**：3 张池图 × 5 个观测噪声 σ_n ∈ {0.01, 0.02, 0.05, 0.10, 0.20} × 两个真值模糊
θ* ∈ {1.5, 0}，同一束 η 在两个 θ* 间复用。q 一律在 t = σ_n 上求值——原脚本用默认
t_spans=[80, 0.002]，与 HQS 去噪器的噪声 sn 不一致；统一到 t=σ_n 后，Λ、L_star、HQS
三者指同一个先验尺度。

预官判据（在看到任何数字之前写定）
----------------------------------
V1 守门（恒等式）：每格 |E_y + Λ + data| ≤ 1e-6·max(1,|E_y|)。不过则全废。

V2 Lemma 12 在真最优处必然成立：每格 E_y < 0。这是恒等式的推论（Λ>0、data>0），属守门；
   若出现 E_y ≥ 0，说明 Λ ≤ 0（过模糊区）或恒等式错了。

V3 6600 是优化漏洞（**可能失败**）：σ_n=0.02 处 E_hqs > 0 而 E_y < 0，且
   gap = E_hqs - E_y ≥ 1000 nats。另预言 gap 随 σ_n **增大**而增大（噪声越大、去噪器
   越过度平滑，HQS 越走不到 x=y）——这一条只记录不判，因为方向没有先验把握。
   **失败方式**：若 gap ≈ 0，则 6600 是物理退化而非优化漏洞，§五点五 的表成立而本实验
   的出发点要改写。

V4 跨测量复现（**可能失败**）：干净图上的 Λ_clean(σ_n) = q(x*) - q(h_1.5⋆x*) 应复现 §3.2
   的表（0.02→13290、0.05→3530、0.10→1054、0.20→50），容差 ±15%（该表 12 图、此处 3 图）。
   复现不了则本脚本的 q/Λ 映射有错。带噪声的 Λ 不参与此判据（定义不同，只记录）。

V5 符号对照（**可能失败**）：同一张图、同一束 η 下，Λ(θ*=1.5) > 0 > Λ(θ*=0)。
   即「模糊 + 噪声」比清晰图更可能，而「清晰图 + 噪声」更不可能。这是模糊偏好现象的
   核心对照；若 Λ(θ*=0) 也 >0，说明噪声本身也在提升概率，现象的归因就要改写。

V6 诊断（无阈值）：gap/|E_y| 与 E_hqs/Λ 随 σ_n 的走向（后者就是 §五点五 想要的那个
   比值，现在有 5 个噪声档而不是 1 个）。

本实验**不**证明什么（诚实声明）
------------------------------
只覆盖统一图景幅度半的**一个算法**（全局 MAP / HQS）。另两个（高斯谱近似、采样器 ESS）
仍未检验；ESS=1/64 不在同一量纲，本实验不碰它。

用法： venv2/bin/python evidence_scale.py
"""

import json
import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn, ode_sampler

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
IMG_LO, IMG_HI = -1.0, 1.0
TSTAR = 1.5
NIMG = 3
NOISES = [0.01, 0.02, 0.05, 0.10, 0.20]
N_INNER = 240
ETA_SEED = 777


class BlurOp:
    def __init__(self, ks=15):
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
        B, C, Hh, Ww = x.shape
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))
                                * self._K(k, C, Hh, Ww), s=(Hh, Ww),
                                dim=(-2, -1)).to(DT)

    def eig(self, s, C, H, W):
        """|ĥ|²。注意 s=0 也必须返回 4-D (1,C,H,W//2+1)，否则与 rfft2 的结果广播不了
        （test_lemma12.py 原版在这里返回 3-D，θ=0 时会炸）。"""
        k = self.kernel(s)
        if k is None:
            return torch.ones((1, C, H, W // 2 + 1), device=DEV, dtype=DT)
        return self._K(k, C, H, W).abs() ** 2


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans_def = get_ode_flow(prior)   # ode_sampler 要 tensor，list 会炸
    blur = BlurOp()

    @torch.no_grad()
    def q_at(z, t):
        """q_t(z) = -log p_t(z)。likelihood_fn 返回 NLL，原样取即 q。"""
        return float(likelihood_fn(z.to(DT), ode_flow,
                                   t_spans=[80.0, float(t)]).item())

    @torch.no_grad()
    def q_batch(zs, t):
        """一批张量共用一次 ODE 积分，返回各自的 q_t。"""
        out = []
        for i in range(0, len(zs), 4):
            v = likelihood_fn(torch.cat([z.to(DT) for z in zs[i:i + 4]], 0),
                              ode_flow, t_spans=[80.0, float(t)])
            out.extend([float(x) for x in v.reshape(-1)])
            torch.cuda.empty_cache()
        return out

    @torch.no_grad()
    def hqs_theta0(y, sn, z0, n_inner=N_INNER):
        """PnP-HQS 解 θ=0 的子问题，返回 (数据项, x)。"""
        C, Hh, Ww = y.shape[1], y.shape[2], y.shape[3]
        denom = blur.eig(0.0, C, Hh, Ww) + 1.0          # θ=0: eig≡1 => denom≡2
        z = z0.clone()
        for _ in range(n_inner):
            xh = denoiser(z, sn).to(DT)
            v = 2 * xh - z
            xn = torch.fft.irfft2(torch.fft.rfft2(y + v, dim=(-2, -1)) / denom,
                                  s=(Hh, Ww), dim=(-2, -1))
            z = torch.clamp(z + xn - xh, IMG_LO, IMG_HI)
        x = denoiser(z, sn).to(DT)
        d = float(((x - y) ** 2).sum().item() / (2 * sn ** 2))
        return d, x

    samples, _ = ode_sampler(ode_flow, shape=(NIMG + 1, 3, 64, 64),
                             t_spans=t_spans_def, device=DEV)
    xs = samples[:NIMG].to(DT)
    print(f"载入 {NIMG} 张池图；σ_n ∈ {NOISES}；θ* ∈ {{1.5, 0}}；"
          f"n_inner={N_INNER}", flush=True)

    rows = []
    for i in range(NIMG):
        x_star = xs[i:i + 1]
        for sn in NOISES:
            g = torch.Generator(device=DEV).manual_seed(ETA_SEED + i)
            eta = sn * torch.randn(x_star.shape, generator=g, device=DEV, dtype=DT)
            data = float((eta ** 2).sum().item() / (2 * sn ** 2))
            yb = (blur.A(x_star, TSTAR) + eta).to(DT)     # θ*=1.5 的观测
            y0 = (x_star + eta).to(DT)                    # θ*=0   的观测
            xb = blur.A(x_star, TSTAR).to(DT)             # 干净模糊图（对 §3.2 用）
            qx, qb, qb_n, q0_n = q_batch([x_star, xb, yb, y0], sn)
            lam_clean = qx - qb                           # 与 §3.2 同定义
            for tag, y, qy in (("blur", yb, qb_n), ("sharp", y0, q0_n)):
                lam = qx - qy
                L_star = data + qx
                L_y = qy                                  # x=y 是可行的 θ=0 点
                E_y = L_y - L_star
                rec = dict(img=i, sn=sn, tag=tag, theta=TSTAR if tag == "blur" else 0.0,
                           data=data, qx=qx, qy=qy, lam=lam, lam_clean=lam_clean,
                           L_star=L_star, L_y=L_y, E_y=E_y)
                if tag == "blur":
                    best = None
                    for z0 in (yb, blur.A(yb, 0.0), torch.randn_like(yb) * 0.5):
                        d, x = hqs_theta0(yb, sn, z0)
                        L = d + q_at(x, sn)
                        best = L if best is None else min(best, L)
                        del z0, x
                        torch.cuda.empty_cache()
                    rec["best"], rec["E_hqs"] = best, best - L_star
                    rec["gap"] = rec["E_hqs"] - E_y
                rows.append(rec)
                extra = (f"  E_hqs={rec['E_hqs']:>+10.1f}  gap={rec['gap']:>+9.1f}"
                         if tag == "blur" else "")
                print(f"  [img{i} σ={sn:<5g} {tag:<5s}] Λ={lam:>+9.1f} "
                      f"Λ_clean={lam_clean:>+9.1f} data={data:>8.1f} "
                      f"E_y={E_y:>+10.1f}{extra}", flush=True)

    with open(os.path.join(OUT, "evidence_scale.json"), "w") as fh:
        json.dump(dict(cfg=dict(nimg=NIMG, noises=NOISES, tstar=TSTAR,
                                n_inner=N_INNER, eta_seed=ETA_SEED, ckpt=CKPT),
                       rows=rows), fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'evidence_scale.json')}")


if __name__ == "__main__":
    main()
