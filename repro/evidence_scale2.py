"""第 2 步：把「退化幅度 = Λ_clean」做厚——更多图 × 两个核强度 + 机理直接验证（GPU）。

预官（2026-10-20，跑前写定）
================================================================================
背景（§3.9(z)）
---------------
§3.9(z) 在 3 张图 × 5 噪声 × 单一核强度 θ*=1.5 上得到：
  * |E_hqs|/Λ_clean = 1.019–1.238（Λ_clean≥2000 nats 的 9 格，跨 2394–23007）
  * Lemma 12 的成立与否由 Λ_clean 的符号精确决定：15/15
  * 机理（赛后推导）：θ=0 处优化器把模糊观测解释成「模糊的内容图」，先验恰好吃掉
    Λ_clean、数据项只付噪声地板，于是 E_hqs ≈ −Λ_clean。
最后一条**只是从数值反推的，没有直接验证**。且样本只有 3 张图、一个核强度。

本步做三件事：
  (a) 图数 3→8，把比值的标准误压下来；
  (b) 加第二个核强度 θ*=2.0，检验比值是否**跨核强度普适**（这是新内容，§3.9(z) 完全没有）；
  (c) **直接验证机理**：把 θ=0 处 HQS 的最优解 x 的 (数据项, q(x), ‖x−h⋆x*‖/‖h⋆x*‖)
      落盘。若 q(x)≈q(h⋆x*) 且数据项≈噪声地板，则 E_hqs≈−Λ_clean 就是直接的而非反推的。

定义与 §3.9(z) 逐字对齐：
    L(x,θ,y)=‖h_θ⋆x−y‖²/(2σ_n²)+q_t(x)，t=σ_n
    L_star=‖η‖²/(2σ_n²)+q_t(x*)，E_hqs=min_x L(x,0,y)−L_star
    Λ_clean=q_t(x*)−q_t(h_θ⋆x*)
同一束 η 在两个 θ* 间复用（与 §3.9(z) 同seed）。

预官判据（在看到任何数字之前写定）
----------------------------------
W1 守门：每格 |E_y+Λ_noisy+data| ≤ 1e-6·max(1,|E_y|)。（与 (z) 的 V1 同一条恒等式。）

W2 比值收敛（**可能失败**）：Λ_clean≥2000 nats 的格子上，
       mean(|E_hqs|/Λ_clean) ∈ [1.00, 1.25]   且   该均值的 SE ≤ 0.03
   (z) 的 9 格给 1.019–1.238；此处要求均值落在同一区间且 SE 收得比 (z) 更紧。
   **失败方式**：若均值掉出 [1.00,1.25]，说明 (z) 的 12.4% 带是小样本的运气。

W3 跨核强度普适（**最尖的一条，(z) 完全没有**）：
       |mean_ratio(θ*=1.0) − mean_ratio(θ*=2.0)| ≤ 2·SE_combined
   即两个核强度上的比值在统计上不可区分。**失败方式**：若差别 >2SE，则退化幅度不是
   只看 Λ_clean，还依赖核强度，「=Λ_clean」要改写成「=k(θ)·Λ_clean」。

W4 机理直接验证（**最尖的一条，(z) 只是反推**）：对 θ*=1.5 与 2.0 各自的最优 HQS 解 x，
       (i)  q(x) − q(h_θ⋆x*) ∈ [−0.15·Λ_clean, +0.05·Λ_clean]
       (ii) data_term(x)/data ∈ [0.5, 2.0]
   两条合起来就是「优化器走到了模糊内容图附近」的直接证据。
   **失败方式**：若 (i) 严重为负（q(x) ≫ q(h⋆x*)），说明 θ=0 解比模糊图更「像先验样本」，
   机理要改写；若 (ii) 远离 1，说明数据项不是噪声地板。

W5 诊断（无阈值）：θ*=0 对照格上的 E_y 与 Λ_noisy；以及 (z) 已测的 img0-2 在新代码下
   的复现性（同一 seed、同一图，应与 (z) 逐字相同——这是一次免费的实现回归检查）。

本实验**不**证明什么（诚实声明）
------------------------------
仍只覆盖一个算法族（全局 MAP / HQS）与一个先验（ffhq）。统一图景的另外两个算法
（高斯谱近似、采样器 ESS）仍未检验。

====================  跨先验检验（2026-10-20 20:4x，跑前写定）==========================
§3.9(ab) 的三项恒等式只在 **ffhq 一个先验**上验过。§3.9(u) 已证 Λ 几乎不依赖先验
（固定图像下换整个先验只动 0.7–3.5%），但**恒等式的两个修正项是否也先验无关，完全没测**。
本组在 afhqv2 先验 + afhqv2 图池上重跑同一设计（8 图 × 5 噪声 × 3 核强度 = 120 模糊格）。

要比的是**无量纲的比值** |E_hqs|/Λ_clean 与两个修正项，不是 Λ_clean 的绝对值
（两边图池不同，绝对值不可比)）。

X1 守门：afhqv2 上 W1 恒等式仍成立到 1e-6。
X2 主判据：afhqv2 上 mean(|E_hqs|/Λ_clean) ∈ [1.00, 1.25] 且 SE ≤ 0.03
   ——与 ffhq 用同一条预谱带宽。
X3 最尖的一条：afhqv2 与 ffhq 的 mean 比值之差 ≤ 2·SE_combined。
   **失败方式**：若差别 >2SE，则退化幅度**依赖先验**——这与 §3.9(u) 的「Λ 先验无关」
   并置，意味着「Λ 无关但 Λ 的退化机制有关」，是一个需要解释的新结果。
X4 结构：两个修正项在 afhqv2 上保持同一 θ* 序位——-Δq/Λ_clean 随 θ* 单调下降。
   （ffhq 实测 +0.1105 / +0.0030 / -0.0318。）
X5 诊断：afhqv2 上 data_term/data 的范围（ffhq 是 [0.71,1.03]）；以及逐 θ* 的比值。

用法：
  venv2/bin/python evidence_scale2.py --ckpt edm-afhqv2-64x64-uncond-vp.pkl \
      --pool /tmp/pool_afhqv2-64x64-vp_256_s0.pt --out evidence_scale2_afq.json
"""

import argparse
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
POOL = "/tmp/pool_ffhq64_256_s0.pt"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
IMG_LO, IMG_HI = -1.0, 1.0
NIMG = 8
NOISES = [0.01, 0.02, 0.05, 0.10, 0.20]
THETAS = [1.0, 1.5, 2.0]
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
        """|ĥ|²。s=0 也必须返回 4-D，否则与 rfft2 结果广播不了。"""
        k = self.kernel(s)
        if k is None:
            return torch.ones((1, C, H, W // 2 + 1), device=DEV, dtype=DT)
        return self._K(k, C, H, W).abs() ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT, help="换先验（跨先验检验用）")
    ap.add_argument("--pool", default=POOL, help="图池 .pt")
    ap.add_argument("--nimg", type=int, default=NIMG)
    ap.add_argument("--out", default="evidence_scale2.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = load_local_model(args.ckpt).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans_def = get_ode_flow(prior)   # ode_sampler 要 tensor
    blur = BlurOp()

    @torch.no_grad()
    def q_at(z, t):
        return float(likelihood_fn(z.to(DT), ode_flow,
                                   t_spans=[80.0, float(t)]).item())

    @torch.no_grad()
    def q_batch(zs, t):
        out = []
        for i in range(0, len(zs), 4):
            v = likelihood_fn(torch.cat([z.to(DT) for z in zs[i:i + 4]], 0),
                              ode_flow, t_spans=[80.0, float(t)])
            out.extend([float(x) for x in v.reshape(-1)])
            torch.cuda.empty_cache()
        return out

    @torch.no_grad()
    def hqs_theta0(y, sn, z0, n_inner=N_INNER):
        """返回 (数据项, q_t(x), ‖x−y‖/‖y‖)。"""
        C, Hh, Ww = y.shape[1], y.shape[2], y.shape[3]
        denom = blur.eig(0.0, C, Hh, Ww) + 1.0
        z = z0.clone()
        for _ in range(n_inner):
            xh = denoiser(z, sn).to(DT)
            v = 2 * xh - z
            xn = torch.fft.irfft2(torch.fft.rfft2(y + v, dim=(-2, -1)) / denom,
                                  s=(Hh, Ww), dim=(-2, -1))
            z = torch.clamp(z + xn - xh, IMG_LO, IMG_HI)
        x = denoiser(z, sn).to(DT)
        d = float(((x - y) ** 2).sum().item() / (2 * sn ** 2))
        rel = float((x - y).norm().item() / y.norm().item())
        return d, x, rel

    nimg = args.nimg
    samples, _ = ode_sampler(ode_flow, shape=(nimg + 1, 3, 64, 64),
                             t_spans=t_spans_def, device=DEV)
    xs = samples[:nimg].to(DT)
    print(f"ckpt={args.ckpt}  pool={args.pool}\n"
          f"载入 {nimg} 张池图；σ_n ∈ {NOISES}；θ* ∈ {THETAS} ∪ {{0}}；"
          f"n_inner={N_INNER}", flush=True)

    rows = []
    for i in range(nimg):
        x_star = xs[i:i + 1]
        for sn in NOISES:
            g = torch.Generator(device=DEV).manual_seed(ETA_SEED + i)
            eta = sn * torch.randn(x_star.shape, generator=g, device=DEV, dtype=DT)
            data = float((eta ** 2).sum().item() / (2 * sn ** 2))
            zs, tags = [x_star], ["x"]
            for th in THETAS:
                zs.append(blur.A(x_star, th))          # 干净模糊图
                tags.append(f"b{th}")
            zs.append(x_star + eta)                     # θ*=0 的观测
            tags.append("y0")
            qs = q_batch(zs, sn)
            Q = dict(zip(tags, qs))
            qx = Q["x"]
            # θ*=0 对照
            qy0 = Q["y0"]
            rows.append(dict(img=i, sn=sn, tag="sharp", theta=0.0, data=data,
                             qx=qx, qy=qy0, lam=qx - qy0, lam_clean=None,
                             E_y=qy0 - data - qx))
            for th in THETAS:
                yb = (blur.A(x_star, th) + eta).to(DT)
                qb = Q[f"b{th}"]
                lam_clean = qx - qb
                qy = q_at(yb, sn)
                lam = qx - qy
                L_star = data + qx
                E_y = qy - data - qx
                best, bd, bq, brel = None, None, None, None
                for z0 in (yb, blur.A(yb, 0.0), torch.randn_like(yb) * 0.5):
                    d, x, rel = hqs_theta0(yb, sn, z0)
                    q = q_at(x, sn)
                    L = d + q
                    if best is None or L < best:
                        best, bd, bq, brel = L, d, q, rel
                    del z0, x
                    torch.cuda.empty_cache()
                rows.append(dict(img=i, sn=sn, tag="blur", theta=th, data=data,
                                 qx=qx, qy=qy, lam=lam, lam_clean=lam_clean,
                                 E_y=E_y, best=best, E_hqs=best - L_star,
                                 gap=(best - L_star) - E_y,
                                 x_data=bd, x_q=bq,
                                 x_rel_y=brel,
                                 # 机理量：HQS 解 vs 干净模糊图
                                 dq_vs_blur=bq - qb,
                                 dq_over_lam=(bq - qb) / lam_clean
                                 if abs(lam_clean) > 1e-9 else None,
                                 xdata_over_data=bd / data if data > 0 else None))
                print(f"  [img{i} σ={sn:<5g} θ*={th:<4g}] Λ_cl={lam_clean:>+9.1f} "
                      f"E_hqs={best-L_star:>+10.1f} 比值="
                      f"{abs(best-L_star)/lam_clean:>6.3f}  "
                      f"Δq/Λ_cl={(bq-qb)/lam_clean if abs(lam_clean)>1e-9 else 0:>+6.3f}",
                      flush=True)

    with open(os.path.join(OUT, args.out), "w") as fh:
        json.dump(dict(cfg=dict(nimg=nimg, noises=NOISES, thetas=THETAS,
                                n_inner=N_INNER, eta_seed=ETA_SEED,
                                ckpt=args.ckpt, pool=args.pool),
                       rows=rows), fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, args.out)}")


if __name__ == "__main__":
    main()
