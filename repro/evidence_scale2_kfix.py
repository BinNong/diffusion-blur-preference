"""evidence_scale2.py 的正确核重跑：把 §7 那两个「错核值」换成正确核的值（GPU，约 2 h）

预官（2026-10-28，跑前写定；md5 见 RESULTS §3.9(bn)）
================================================================================
为什么有这一组
---------------
§3.9(bm) 核账时发现：论文 §7 引的 **76 个配置**与 **Λ_clean 跨 14.6×** 出自
`evidence_scale2.json`，而那个文件是**错核**跑的（手写高斯网格 `-(ks//2)+1=-6`
vs putils 的 `-ks//2+1=-7`，§3.9(aj)/(ak) 定位的 bug）。错核使 Λ_clean 偏高
4.0–8.5%，所以这两个数**没有正确核的对应值**。§9 因此把「Every quantity … was
re-measured on the corrected kernel」改成了 Almost every，并点名这两个是例外。

本组**只改一件事：核的来源**（BlurOp.kernel() 改从 putils.convolution.GaussianBlur 取）。
_NIMG / NOISES / THETAS / N_INNER / ETA_SEED / 图池 / ckpt / HQS 协议 / 落盘字段_
全部与 evidence_scale2.py 逐字相同，所以两组数可直接逐格配对比较。

已经量化过的稳健性（§3.9(bm2)，不用重跑就能算）：
  * 把 120 个 Λ_clean 整体乘 f（正确值≈错核值×f，f∈[0.915,0.96]）：
      f=0.96 / 0.95 / 0.97 → ≥2000 的格数仍 **76**，跨度仍 **14.606×**（严格不变）
      f=0.915          → 格数 **75**，跨度 12.588×
  * **15/15 完全稳健**：乘正因子保号（那组 Λ_clean = {−136.6, +27.4, +316.7, …}）
  * 恒等式自检与核无关（逐项恒等，实测 |pred−E_hqs| = 0.000e+00）
所以本组真正要回答的只有两个数：**76 会不会变 75**、**14.6× 会变多少**。

预官判据（在看到任何数字之前写定）
----------------------------------
K0 核守卫（不过则一个数字都不报）：核必须对称（不对称度 == 0）、峰在 (7,7)、
   与 putils 逐比特相同；且必须不是错核（峰 (6,6)、不对称 3.159e−02）。

K1 配对一致性（统计门，非判决）：同一 (img, sn, θ*) 上
       z = (Λ_clean_kfix − Λ_clean_buggy) / SE_buggy
   其中 SE_buggy = lam_std/sqrt(nimg)（错核文件自带）。若两组均值无偏，z 的 SD 应 ≈ 0.91
   （推导同 judge_law_n120 的 R1）。报 median|z| 与 max|z|，**不设通过线**——本组不是
   复现实验，是换核后的第一次有效测量。

K2 主判据一：**Λ_clean ≥ 2000 nats 的格数**。错核给 76。
       要求在 f=0.96（等价错核偏高 4.2%）下仍为 76，即格数 ∈ {75, 76} 都算通过，
       但必须**逐格报出**门槛 2000 附近那几个值（错核侧是 1943.5/1958.2/1962.6/2130.8），
       因为格数只可能由它们改变。

K3 主判据二：该子集的 **Λ_clean 跨度**（max/min）。错核给 14.606×（[2130.8, 31121.8]）。
       要求报出正确核的值；若与 14.606× 之差 > 5%，论文 §7 的「14.6×」必须改成新值。

K4 诊断（无阈值）：逐 θ* 的 mean(|E_hqs|/Λ_clean) 与 §3.9(ac) 的 1.2284/1.0767/1.0215
       对照；以及跨先验那一臂（--ckpt afhqv2）与 ffhq 的差是否仍落在 0.008–0.021。
       注意：0.008–0.021 本身是**两个错核跑之比**，同核相比是合法比较，且 §3.9(ak) 已实测
       比值本身对核稳定（1.1057±0.0125 → 1.1055±0.0080）；此处只是补正确核的对应值。

失败后果：K2 若给 75、或 K3 与 14.606× 差 >5%，论文 §7 那两个数必须改，并在 RESULTS
记录改了多少、原值是什么。不报「最接近的通过变体」。

用法：
  cd /data/experiment/de_conv_1
  ./venv2/bin/python repro/evidence_scale2_kfix.py --out evidence_scale2_kfix.json
  ./venv2/bin/python repro/evidence_scale2_kfix.py --ckpt edm-afhqv2-64x64-uncond-vp.pkl \
      --pool /tmp/pool_afhqv2-64x64-vp_256_s0.pt --out evidence_scale2_afq_kfix.json
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
from putils.convolution import GaussianBlur

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
    """核一律取自 putils.convolution.GaussianBlur（对称、峰 [7,7]）。

    **本脚本唯一改动就是 kernel() 的来源**：原 evidence_scale2.py 用
        lin = torch.linspace(-(ks // 2) + 1, ks // 2, ks)   # ks=15 => [-6..7]
    而 putils 的 -ks//2+1 因 Python 优先级实为 (-ks)//2+1 = -7，给出 [-7..7]。
    手写版因此非整数、不对称、峰偏到第 6 格——§3.9(aj)/(ak) 定位的真 bug。
    _K / A / AT / eig 与 evidence_scale2.py 逐字相同，只换 kernel()。
    """

    def __init__(self, ks=15):
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


def kernel_guard(blur):
    """N0 核守卫：核必须对称、峰在 (7,7)、与 putils 逐比特相同；且必须不是错核。
    不过则中止，一个数字都不报。"""
    k = blur.kernel(1.5)[0, 0].to(torch.float64)
    asym = float((k - k.flip(0).flip(1)).abs().max())
    peak = tuple(int(v) for v in (k == k.max()).nonzero()[0].tolist())
    ref = GaussianBlur(kernel_size=15, device=DEV, dtype=DT).kernel(1.5)[0, 0].to(torch.float64)
    same = bool(torch.equal(k, ref))
    print("[N0 核守卫] 不对称=%.3e（须 0）  峰=%s（须 (7,7)）  与 putils 逐比特相同=%s"
          % (asym, peak, same), flush=True)
    if not (asym == 0.0 and peak == (7, 7) and same):
        raise SystemExit("[N0 FAIL] 核不是正确核——中止，不产出任何数字。")
    print("[N0 PASS]", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT, help="换先验（跨先验检验用）")
    ap.add_argument("--pool", default=POOL, help="图池 .pt")
    ap.add_argument("--nimg", type=int, default=NIMG)
    ap.add_argument("--out", default="evidence_scale2_kfix.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = load_local_model(args.ckpt).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans_def = get_ode_flow(prior)   # ode_sampler 要 tensor
    blur = BlurOp()
    kernel_guard(blur)

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
                                ckpt=args.ckpt, pool=args.pool,
                                kernel="putils.convolution.GaussianBlur(15)",
                                kernel_note="evidence_scale2.py 的 kfix 版：只换 kernel() 来源，"
                                            "其余逐字相同；修 §3.9(aj)/(ak) 的手写网格 bug"),
                       rows=rows), fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, args.out)}")


if __name__ == "__main__":
    main()
