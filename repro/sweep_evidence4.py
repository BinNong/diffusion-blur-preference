r"""第十五组：修正高斯核的**半像素偏移 bug** 后重测 §3.9(z)/(ab) 的符号类论断（GPU）

预官（2026-10-23，跑前写定）
================================================================================
为什么有这一组（第十四组 (aj) 的结局）
------------------------------------
第十四组 `sweep_evidence3.py` 在 20 张**真实池图**上重测了 §3.9(z)/(ab) 的符号类论断，
并按要求带了一条**跨实现复现**判据 K2：用 pool[:12] 对 `sweep_lambda_theta.json` 的
gaussian σ=1.5 记录（同池、同核、同 ckpt、不同脚本）。**K2 失败了**：

    t=0.02  参照 12962.6  本组 13568.7   +4.68%
    t=0.05  参照  4440.4  本组  4769.1   +7.40%

逐图比对显示**是同一批图**（差值 sd=429 而跨图 sd=3025，若图不同差值 sd 会是 ~4280），
所以差异不在图。再把对数密度拆开：logp_sharp 一致到 **0.08%**，logp_blur 差 **+1.27%**
——偏移全在模糊图那一项，指向**模糊核本身**。

根因（已定位，零 GPU 可复现）
--------------------------
`evidence_scale.py` 手写了核的坐标网格：

    lin = torch.linspace(-(ks // 2) + 1, ks // 2, ks)      # ks=15 ⇒ linspace(-6, 7, 15)

而 `putils.convolution.GaussianBlur` 写的是：

    lin = torch.linspace(-kernel_size//2 + 1, kernel_size//2, kernel_size)

**Python 里一元负号比 `//` 绑得紧**，所以 `-ks//2` 是 `(-ks)//2 = -8`，putils 实际拿到
`linspace(-7, 7, 15)`——整数网格 [-7..7]，**关于中心格对称、峰在中心**。而手写版是
`linspace(-6, 7, 15)`，步长 13/14，**非整数、不对称、峰偏到第 6 格**：

    kA[7,:] = […, 0.056641, 0.070736, 0.056641, …]     对称，峰 0.0707 在 index 7
    kB[7,:] = […, 0.055389, 0.054578, 0.036660, …]     不对称，峰 0.0554 在 index 6

**一个不对称的核不是高斯模糊。** 所以第十四组测的不是"高斯 σ=1.5 的模糊偏好"，而是一个
半像素偏移的畸变算子的模糊偏好——它的数字不适用于本实验的目的。

这个手写网格在项目里存在于 **13 个脚本**（check_baseline / check_tdependence /
diag_gauss_bias / diag_phaseC / diag_theta0 / evidence_scale / evidence_scale2 /
repro_phaseC / repro_phaseC_v2_backup / repro_phaseD / repro_phaseD_spec /
test_lemma12 / validate_spec_approx），即 §3.2 的 Λ(t) 表、§3.9(z)/(ab)、Phase C/D 全都
用它。§3.9(l)/(m) 的主定律用**零相位谱门**、不经过这个核，不受影响。影响量化见 L2。

本组做什么
----------
**只改一件事：核改为直接取自 `putils.convolution.GaussianBlur`**，不再手写网格。
其余与第十四组逐字相同：20 张真实池图（pool[:20]，本项目主线标准图数）、
σ_n ∈ {0.01, 0.02, 0.05, 0.10, 0.20}、t = σ_n、θ* = 1.5、同 ETA_SEED=777+i、
HQS（θ=0、3 初值 × 240 迭代取最小）、同 b(u) 无关的量。回归臂仍跑 (z) 的 3 张先验样本。

预官判据（在看到任何数字之前写定）
----------------------------------
L0 核守卫（**零 GPU，跑前就跑；不过则不启动主臂**）：
   (a) 核关于中心格对称，max|k − flip(k)| ≤ 1e-7；
   (b) 本组的模糊结果与 `sweep_lambda_theta.py` 的 `blur_fft(x, GaussianBlur(15).kernel(1.5))`
       **逐比特相同**（同池前 4 张）。这条正是能抓住上述 bug 的检查；
   (c) 报出 kunif = mean(1−|ĥ|²) 与 kappa（能量加权），留作后续对照。
   **失败方式**：任何一条不过 ⇒ 核仍不对，主臂不启动。

L1 跨实现复现（**本组的实现自检，取代第十四组 K2 的地位**）：
   pool[:12] 的 Λ_clean 在 t=0.02/0.05 上对 `sweep_lambda_theta.json` 的 gaussian σ=1.5
   记录，容差 **2% × logp_blur**（t=0.02 为 912 nats、t=0.05 为 606 nats）。
   **容差为什么这么定（披露）**：第十四组的 K2 把容差设成 2% × **Λ_clean**，那是标定错误——
   (ab4) 测定的 1–2% 数值地板是作用在 **logp** 上的（ODE 批组成改变浮点求和顺序），
   而 Λ_clean 是两个 3×10⁴–5×10⁴ nats 量之差，1% 的 logp 地板到 Λ_clean 上就是
   4–7%。这是 CLAUDE.md 里那条"Λ 是两个大数的小差"陷阱的又一次实例。
   **失败方式**：超出容差 ⇒ 仍有实现差异，先查再报物理。

L2 核对 (z) 的核效应（**诊断 + 定量，不是通过/失败判据**）：
   回归臂（3 张先验样本、同 seed）的 Λ_clean 与 `evidence_scale.json` 比。
   预言：比 (z) **低 2%–10%**（第十四组用错核时比 putils 高 +4.68%/+7.40%，换向后即低）。
   **若落在 (z) 的 ±2% 内，说明核没真正换成 putils 的**，那是失败，要查实现。
   这条同时给出 §3.9(z)/(ab) 已发布数字因该 bug 产生的偏移量。

L3 主判据一（模糊偏好不随观测噪声存活）：Λ_noisy < 0 的配对比例。**阈值与第十四组逐字相同**：
     L3a 全部 100 对：通过线 ≥ 0.85；否掉线 < 0.70；灰区记为不可判定。
     L3b σ ≥ 0.02 的 80 对：通过线 ≥ 0.95。
     L3c σ = 0.01 的 20 对：**只记录不判**。
     L3d 伴生排序 Λ_noisy(θ*=1.5) > Λ_noisy(θ*=0)：通过线 ≥ 0.95。
   **必须同时报**逐图的负值个数、mean/std、偏度、中位数/均值。

L4 主判据二（Lemma 12 符号对应）：(Λ_clean>0 ⟺ E_hqs<0) 的格子比例。
   通过线 ≥ 0.95；否掉线 < 0.85。分报 Λ_clean<0 的反向半边（只报数量与符号）。

L5 主判据三（幅度半）：Λ_clean ≥ 2000 nats 的格子上 mean(|E_hqs|/Λ_clean) ∈ [1.00, 1.25]
   且 SE ≤ 0.03（与 (ab) 的 W2 同一条预谱带宽）。同时报三项恒等式自检残差。

L6 诊断（无阈值）：(a) 比值随 Λ_clean 的走向；(b) 两个修正项 −Δq/Λ_clean 与 data_term/data；
   (c) 图源效应（真实池图 vs 先验样本）；(d) data 的尺度不变性；(e) 逐图 σ 剖面单调性。

关于"重跑"的诚实声明（**必须读**）
--------------------------------
* 第十四组的全部结果在写本预官时**已经可见**（K3a=0.850 恰好压在通过线上、K4=97/100、
  K5=1.0843±0.0076）。**本组的 L3/L4/L5 阈值与第十四组逐字相同，没有放宽**，所以不存在
  "重跑到过为止"的空间：若本组 K3a 低于 0.85，就照实报失败。
* 但也要说清：第十四组测的是一个**非预期的算子**（不对称核），所以它的数字不是本实验的
  数字，本组不是"同一次测量的重掷"，而是**换成正盼的算子后第一次有效测量**。第十四组的
  数字会全文保留并标注为"错核算子的测量"——它本身有信息量：符号类论断在错核上也成立。
* 仍未覆盖：一个算法族（全局 MAP / PnP-HQS）、一个先验（ffhq）、一个分辨率（64×64）、
  θ* 只取 1.5；E_hqs 依赖优化器设置。

用法： venv2/bin/python sweep_evidence4.py [--smoke]
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
KS = 15
POOL = "/tmp/pool_ffhq64_256_s0.pt"
NOISES = [0.01, 0.02, 0.05, 0.10, 0.20]
N_INNER = 240
ETA_SEED = 777
NIMG_POOL = 20
NIMG_REG = 3


class BlurOp:
    """高斯模糊核。**核一律取自 putils.convolution.GaussianBlur，不再手写网格。**

    理由（(ak1) 披露）：手写 `-(ks//2)+1` 在 ks=15 时是 -6，而 putils 的 `-ks//2+1`
    因 Python 优先级是 `(-ks)//2+1 = -7`。前者给出 linspace(-6,7,15)（非整数、不对称、
    峰偏移），后者给出整数网格 [-7..7]（对称、峰在中心）。**不对称的核不是高斯模糊。**
    """

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
        B, C, Hh, Ww = x.shape
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))
                                * self._K(k, C, Hh, Ww), s=(Hh, Ww),
                                dim=(-2, -1)).to(DT)

    def eig(self, s, C, H, W):
        if s == 0:
            return torch.ones((1, C, H, W // 2 + 1), device=DEV, dtype=DT)
        return self._K(self.kernel(s), C, H, W).abs() ** 2


def blur_fft_ref(x, kernel):
    """sweep_lambda_theta.py 的模糊实现，逐字复制，仅用于 L0(b) 的逐比特对账。"""
    if kernel.ndim == 2:
        kernel = kernel[None, None]
    B, C, Hh, Ww = x.shape
    kh, kw = kernel.shape[-2:]
    pad = torch.zeros((1, C, Hh, Ww), device=x.device, dtype=x.dtype)
    top, left = (Hh - kh) // 2, (Ww - kw) // 2
    pad[:, :, top:top + kh, left:left + kw] = kernel.to(x.dtype)
    return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))
                            * torch.fft.rfft2(pad, dim=(-2, -1)),
                            s=(Hh, Ww), dim=(-2, -1))


def guard_kernel(blur, pool):
    """L0：核守卫。返回 (ok, report_dict)。不过则调用方不得启动主臂。"""
    rep = {}
    k = blur.kernel(TSTAR)[0, 0]
    rep["sym"] = float((k - k.flip(0).flip(1)).abs().max())
    rep["peak_rc"] = [int(v) for v in torch.nonzero(k == k.max()).flatten().tolist()]
    rep["ksum"] = float(k.sum())
    h2 = blur.eig(TSTAR, 3, 64, 64)[0].to(torch.float64)          # (H, W//2+1)
    rep["kunif"] = float((1.0 - h2).mean())
    # 能量加权 kappa（与 sweep_lambda_theta.py 同定义，用全部 256 张估）
    with torch.no_grad():
        Sfull = (torch.fft.rfft2(pool.to(torch.float64), dim=(-2, -1)).abs() ** 2
                 / (64 * 64)).mean(1).mean(0)
    rep["kappa"] = float(((1.0 - h2) * Sfull).sum() / Sfull.sum())
    xs = pool[:4].clone()
    yA = blur.A(xs, TSTAR)
    yB = blur_fft_ref(xs, GaussianBlur(kernel_size=KS, device=DEV, dtype=DT).kernel(TSTAR))
    rep["bitident"] = bool(torch.equal(yA, yB))
    rep["blur_maxdiff"] = float((yA - yB).abs().max())
    ok = (rep["sym"] <= 1e-7) and rep["bitident"] and abs(rep["ksum"] - 1.0) < 1e-6
    return ok, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="evidence4.json")
    args = ap.parse_args()
    if args.smoke and args.out == "evidence4.json":
        args.out = "smoke_evidence4.json"
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

    pool_all = torch.load(POOL, map_location=DEV).to(DT)
    ok, rep = guard_kernel(blur, pool_all)
    print("[L0 核守卫]", json.dumps(rep, ensure_ascii=False), flush=True)
    print(f"[L0] 对称性 max|k−flip(k)| = {rep['sym']:.2e}（≤1e-7）；"
          f"峰位置 {rep['peak_rc']}（应含 [7,7]）；Σk = {rep['ksum']:.9f}；"
          f"与 sweep_lambda_theta 的 blur_fft 逐比特相同 = {rep['bitident']}"
          f"（max|diff|={rep['blur_maxdiff']:.2e}）", flush=True)
    print(f"[L0] kunif = mean(1−|ĥ|²) = {rep['kunif']:.6f}；"
          f"kappa（能量加权）= {rep['kappa']:.6f}", flush=True)
    if not ok:
        print("[L0] ❌ 核守卫不过——**不启动主臂**。", flush=True)
        raise SystemExit(2)

    @torch.no_grad()
    def q_at(z, t):
        """q_t(z) = -log p_t(z)。likelihood_fn 返回 NLL，原样取即 q。"""
        return float(likelihood_fn(z.to(DT), ode_flow,
                                   t_spans=[80.0, float(t)]).item())

    @torch.no_grad()
    def q_batch(zs, t):
        """一批张量共用一次 ODE 积分。批组成固定为 4（与 (z) 一致）。"""
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
                        # q_at 每个解只调一次（与 (z) 同），否则 Hutchinson 的全局 RNG
                        # 消耗改变，回归臂 L2 的浮点对账不可比。
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
                        rec["dq"] = q_j - qb             # Δq = q(x) − q(h⋆x*)
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

    # ---- 臂 A：回归臂（先验样本，(z) 的 img0-2）。必须在主臂之前：Hutchinson 吃全局 RNG。
    samples, _ = ode_sampler(ode_flow, shape=(nimg_reg + 1, 3, 64, 64),
                             t_spans=t_spans_def, device=DEV)
    reg_xs = samples[:nimg_reg].to(DT)
    print(f"[reg] 重新生成 (z) 的 {nimg_reg} 张先验样本（同 seed）", flush=True)
    rows = run_arm(reg_xs, "reg", nimg_reg, t0)
    del reg_xs, samples
    torch.cuda.empty_cache()

    # ---- 臂 B：主臂（20 张真实池图）
    xs = pool_all[:nimg_pool].clone()
    print(f"[pool] 载入 {nimg_pool} 张真实池图 {POOL}；σ_n ∈ {NOISES}；"
          f"θ* = {TSTAR}；n_inner={N_INNER}", flush=True)
    rows += run_arm(xs, "pool", nimg_pool, t0)

    outfn = os.path.join(OUT, args.out)
    json.dump(dict(cfg=dict(nimg_pool=nimg_pool, nimg_reg=nimg_reg, ckpt=CKPT,
                            pool=POOL, noises=NOISES, tstar=TSTAR, ks=KS,
                            n_inner=N_INNER, eta_seed=ETA_SEED, kernel_guard=rep),
                   rows=rows), open(outfn, "w"), indent=1)
    print(f"\n[saved] {outfn}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
