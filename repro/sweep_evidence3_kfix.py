r"""sweep_evidence3.py 的正确核重跑：给 §7 的「twenty-image run gives 13.0×」一个正确核的值（GPU，约 1.5 h）

预官（2026-10-29，跑前写定；md5 见 RESULTS §3.9(bs)）
================================================================================
为什么有这一组
---------------
§3.9(bm) 核账时发现论文 §7 引的数出自错核跑。§3.9(bo) 已用 `evidence_scale2_kfix.py`
把**八图跨强度**那一臂重跑完（76 格 → 75、14.606× → 12.925×）。但 §7 还有一句
「the twenty-image run gives $13.0\times$」——那个 13.0× 出自 `evidence3.json`（**错核**，
20 张池图 × 5 噪声 × 单一 θ*=1.5，Λ_clean≥2000 的 76 格跨 12.98×）。
恒等式的**精确性**与核无关（逐项恒等，实测 |pred−E_hqs|=0），但**跨度**是核相关的，
所以这一句还没有正确核的对应值。

本组**只改一件事：核的来源**（BlurOp.kernel() 改从 putils.convolution.GaussianBlur 取）。
两个臂（reg 先验样本 + pool 真实池图）的**顺序、图数、seed、批组成全部不变**，
所以与 evidence3.json 可**逐格配对**，差别只来自核。
（原脚本的 K1「同 seed 复现 §3.9(z)」因此作废——它的目的是抓实现错误，而本组故意换了核；
新的对照就是与 evidence3.json 的逐格差，由裁判算。）

预官判据（在看到任何数字之前写定）
----------------------------------
K0 核守卫（不过则一个数字都不报）：核必须对称（不对称度 == 0）、峰在 (7,7)、
   与 putils 逐比特相同；且必须不是错核（峰 (6,6)、不对称 3.159e−02）。

K1 逐格配对（只报数，不设通过线）：同一 (arm, img, sn) 上
       ΔΛ_clean = 正确核 − 错核
   报 mean、范围、以及 |Δ| 相对 Hutchinson 差分噪声底（424 nats）的占比。
   预期方向：正确核系统性偏低（八图臂实测 mean −388.7 nats）。

K2 主判据：**Λ_clean ≥ 2000 nats 的格数与该子集的跨度**（错核给 76 格、12.98×）。
   要求报出正确核的值。**若跨度与 12.98× 差 > 5%，论文 §7 的「13.0×」必须改成新值。**

K3 诊断（无阈值）：逐 σ_n 的 mean|E_hqs/Λ_clean| 与错核侧对照；以及恒等式自检
   |pred − E_hqs| 是否仍为 0（预期是，因为逐项恒等、与核无关）。

失败后果：K2 若与 12.98× 差 >5%，论文 §7 那句必须改，并在 RESULTS 记录改了多少、
原值是什么。不报「最接近的通过变体」。

用法：
  cd /data/experiment/de_conv_1
  ./venv2/bin/python repro/sweep_evidence3_kfix.py --out evidence3_kfix.json
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
from edm.prior import ScorePrior, get_ode_flow, likelihood_fn, ode_sampler
from putils.convolution import GaussianBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
IMG_LO, IMG_HI = -1.0, 1.0
TSTAR = 1.5
POOL = "/tmp/pool_ffhq64_256_s0.pt"
NOISES = [0.01, 0.02, 0.05, 0.10, 0.20]
N_INNER = 240
ETA_SEED = 777
NIMG_POOL = 20                 # 主臂：真实池图（本项目主线标准图数）
NIMG_REG = 3                   # 回归臂：(z) 的 img0-2（先验样本）


class BlurOp:
    """核一律取自 putils.convolution.GaussianBlur（对称、峰 [7,7]）。

    **本脚本唯一改动就是 kernel() 的来源**：原 sweep_evidence3.py 用手写网格
        lin = torch.linspace(-(ks // 2) + 1, ks // 2, ks)   # ks=15 => [-6..7]
    而 putils 的 -ks//2+1 因 Python 优先级实为 (-ks)//2+1 = -7，给出 [-7..7]；
    手写版非整数、不对称、峰偏到第 6 格——§3.9(aj)/(ak) 定位的真 bug。
    _K / A / AT / eig 与 sweep_evidence3.py 逐字相同，只换 kernel()。
    两个臂（reg 先验样本 + pool 真实池图）的**顺序、图数、seed、批组成全部不变**，
    所以与 evidence3.json 可逐格配对，差别只来自核。
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
        k = self.kernel(s)
        if k is None:
            return torch.ones((1, C, H, W // 2 + 1), device=DEV, dtype=DT)
        return self._K(k, C, H, W).abs() ** 2


def kernel_guard(blur):
    """N0 核守卫：核必须对称、峰 (7,7)、与 putils 逐比特相同；且必须不是错核。
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
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="evidence3_kfix.json")
    args = ap.parse_args()
    if args.smoke and args.out == "evidence3_kfix.json":
        args.out = "smoke_evidence3.json"
    nimg_pool = 2 if args.smoke else NIMG_POOL
    nimg_reg = 2 if args.smoke else NIMG_REG

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans_def = get_ode_flow(prior)   # ode_sampler 要 tensor，list 会炸
    blur = BlurOp()
    kernel_guard(blur)

    @torch.no_grad()
    def q_at(z, t):
        """q_t(z) = -log p_t(z)。likelihood_fn 返回 NLL，原样取即 q。"""
        return float(likelihood_fn(z.to(DT), ode_flow,
                                   t_spans=[80.0, float(t)]).item())

    @torch.no_grad()
    def q_batch(zs, t):
        """一批张量共用一次 ODE 积分，返回各自的 q_t。**批组成必须与 (z) 一致**（4 个一组），
        否则回归臂 K1 的浮点对账会超出 (ab4) 测定的 1–2% 数值地板。"""
        out = []
        for i in range(0, len(zs), 4):
            v = likelihood_fn(torch.cat([z.to(DT) for z in zs[i:i + 4]], 0),
                              ode_flow, t_spans=[80.0, float(t)])
            out.extend([float(x) for x in v.reshape(-1)])
            torch.cuda.empty_cache()
        return out

    @torch.no_grad()
    def hqs_theta0(y, sn, z0, n_inner=N_INNER):
        """PnP-HQS 解 θ=0 的子问题，返回 (数据项, x)。与 (z)/(ab) 逐字相同。"""
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

    def run_arm(xs, arm, nimg, t0):
        """(z) 的协议，逐字复制。xs: (nimg,3,64,64)。"""
        rows = []
        for i in range(nimg):
            x_star = xs[i:i + 1]
            for sn in NOISES:
                g = torch.Generator(device=DEV).manual_seed(ETA_SEED + i)
                eta = sn * torch.randn(x_star.shape, generator=g, device=DEV, dtype=DT)
                data = float((eta ** 2).sum().item() / (2 * sn ** 2))
                yb = (blur.A(x_star, TSTAR) + eta).to(DT)     # θ*=1.5 的观测
                y0 = (x_star + eta).to(DT)                    # θ*=0   的观测
                xb = blur.A(x_star, TSTAR).to(DT)             # 干净模糊图
                qx, qb, qb_n, q0_n = q_batch([x_star, xb, yb, y0], sn)
                lam_clean = qx - qb
                for tag, y, qy in (("blur", yb, qb_n), ("sharp", y0, q0_n)):
                    lam = qx - qy
                    L_star = data + qx
                    L_y = qy                                  # x=y 是可行的 θ=0 点
                    E_y = L_y - L_star
                    rec = dict(arm=arm, img=i, sn=sn, tag=tag,
                               theta=TSTAR if tag == "blur" else 0.0,
                               data=data, qx=qx, qy=qy, lam=lam, lam_clean=lam_clean,
                               L_star=L_star, L_y=L_y, E_y=E_y)
                    if tag == "blur":
                        # **q_at 每个解只调一次**：(z) 是 3 次/格，多调会改变 Hutchinson
                        # 的全局 RNG 消耗，回归臂 K1 就不可比了。落盘只需要标量，不必留张量。
                        best, cands = None, []
                        for z0 in (yb, blur.A(yb, 0.0), torch.randn_like(yb) * 0.5):
                            d, x = hqs_theta0(yb, sn, z0)
                            q_opt = q_at(x, sn)
                            cands.append((d, q_opt))
                            L = d + q_opt
                            best = L if best is None else min(best, L)
                            del z0, x
                            torch.cuda.empty_cache()
                        d_j, q_j = min(cands, key=lambda c: c[0] + c[1])
                        rec["best"], rec["E_hqs"] = best, best - L_star
                        rec["gap"] = rec["E_hqs"] - E_y
                        rec["dq"] = q_j - qb             # Δq = q(x) - q(h⋆x*)
                        rec["data_term"] = d_j
                        rec["ratio"] = (abs(rec["E_hqs"]) / lam_clean
                                        if lam_clean else float("nan"))
                    rows.append(rec)
                    extra = (f"  E_hqs={rec['E_hqs']:>+10.1f}  ratio={rec['ratio']:>6.3f}"
                             if tag == "blur" else "")
                    print(f"  [{arm} img{i} σ={sn:<5g} {tag:<5s}] Λ={lam:>+9.1f} "
                          f"Λ_cl={lam_clean:>+9.1f} data={data:>7.1f} "
                          f"E_y={E_y:>+10.1f}{extra}  ({time.time()-t0:.0f}s)",
                          flush=True)
                    torch.cuda.empty_cache()
        return rows

    t0 = time.time()

    # ---- 臂 A：回归臂（先验样本，复现 (z) 的 img0-2）。必须在主臂之前：Hutchinson 吃全局 RNG。
    samples, _ = ode_sampler(ode_flow, shape=(nimg_reg + 1, 3, 64, 64),
                             t_spans=t_spans_def, device=DEV)
    reg_xs = samples[:nimg_reg].to(DT)
    print(f"[reg] 重新生成 (z) 的 {nimg_reg} 张先验样本（同 seed）", flush=True)
    rows = run_arm(reg_xs, "reg", nimg_reg, t0)
    del reg_xs, samples
    torch.cuda.empty_cache()

    # ---- 臂 B：主臂（20 张真实池图）
    pool = torch.load(POOL, map_location=DEV).to(DT)
    xs = pool[:nimg_pool].clone()
    print(f"[pool] 载入 {nimg_pool} 张真实池图 {POOL}；σ_n ∈ {NOISES}；"
          f"θ* = {TSTAR}；n_inner={N_INNER}", flush=True)
    rows += run_arm(xs, "pool", nimg_pool, t0)

    outfn = os.path.join(OUT, args.out)
    json.dump(dict(cfg=dict(nimg_pool=nimg_pool, nimg_reg=nimg_reg, ckpt=CKPT,
                            pool=POOL, noises=NOISES, tstar=TSTAR,
                            n_inner=N_INNER, eta_seed=ETA_SEED,
                            kernel="putils.convolution.GaussianBlur(15)",
                            kernel_note="sweep_evidence3.py 的 kfix 版：只换 kernel() 来源、"
                                        "两臂顺序与 seed 不变；修 §3.9(aj)/(ak) 的手写网格 bug"),
                   rows=rows), open(outfn, "w"), indent=1)
    print(f"\n[saved] {outfn}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
