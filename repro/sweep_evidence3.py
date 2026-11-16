r"""第十四组：§3.9(z)/(ab) 的**符号类论断**在 20 张**真实池图**上重测（GPU）

预官（2026-10-23，跑前写定；md5 见 RESULTS §3.9(aj) 的记录）
================================================================================
背景（RESULTS §3.9(z) / (ab)）
------------------------------
§3.9(z) 在 **3 张图** × 5 个 σ_n × θ*=1.5 上、§3.9(ab) 在 **8 张图** × 5 σ_n × 3 个 θ* 上
得到目前全文最强的两条论断：

  (S1) **模糊偏好是干净图的性质，不随观测噪声存活**：Λ_noisy = q_t(y) − q_t(x*) 在
       15 对里 14 对为负（t = σ_n）。
  (S2) **Lemma 12 的成立与否由 Λ_clean 的符号精确决定**：15/15。
  外加幅度半：Λ_clean ≥ 2000 nats 的格子上 mean(|E_hqs|/Λ_clean) = 1.1057 ± 0.0125（(ab) 的 76 格）。

两条符号论断有两个共同的弱点，本组专门修第一个：
  (i) **图像是 prior sample**。(z)/(ab) 的 xs 来自 ode_sampler（代码里 `samples, _ =
      ode_sampler(...)`），即**先验自己生成的图**。对 S1/S2 这类关于 q_t(x*) 的论断，用
      on-manifold 样本当"内容图"是一个实质性的方法学让步：先验样本的 q_t(x*) 系统性地高。
      (z) 的 V4 已间接说明影响不大（3 张先验样本的 Λ_clean 比 §3.2 的 12 张真实图低 6.9%），
      但**带噪声的 Λ_noisy 从来没有在真实图上测过**，而它正是 S1 的对象。
  (ii) **有效样本量 = 图像数**。15 对来自 3 张图，5 个 σ 是同一张图的重复，所以 14/15 的
      二项标准误被低估了约 √5；真正的独立单元是 3 张图。

本组的设计（**只改一件事：图像来源，先验样本 → 真实池图**）
------------------------------------------------------------
其余逐项与 (z) 对齐：同 ckpt（edm-ffhq-64x64-uncond-vp）、同 BlurOp（ks=15 高斯、s=1.5）、
同 σ_n ∈ {0.01, 0.02, 0.05, 0.10, 0.20}、同 t = σ_n、同 ETA_SEED=777+i、同 HQS
（θ=0、3 个初值 × 240 迭代、取最小）、同 b(u) 无关的量。θ* 只取 1.5（(ab) 已在 8 图上
扫过 θ* ∈ {1.0, 1.5, 2.0} 并给出三项恒等式，本组不加这条轴，把 GPU 全花在图像数上）。
图像数取 **20**——这是本项目其余全部主线 sweep 的标准图数，使 Λ_clean 与 §3.2/§3.9(l)
可直接对照。

**回归臂（K1）**：用同一 seed 重新生成 (z) 的 img0-2（manual_seed(0) → load → ode_sampler
shape=(4,3,64,64)），跑逐字相同的协议，与 results/evidence_scale.json 对账。这一臂必须在
主臂之前跑（likelihood_fn 的 Hutchinson 噪声吃全局 RNG，顺序变了就不可比），它的作用是
把"新代码写错了"从"物理结论变了"里分离出来。

预官判据（在看到任何数字之前写定）
----------------------------------
K0 守门（恒等式）：每格 |E_y + Λ_noisy + data| ≤ 1e-6·max(1,|E_y|)。两臂所有行都要过。
   不过则全废，先查代码。

K1 回归臂（**实现自检，不是物理判据**）：3 张先验样本的 Λ、Λ_clean、E_y、E_hqs 与
   evidence_scale.json 的对应值相对差 ≤ **2%**。(ab4) 已测定本代码路径的数值地板是
   1–2%（q_batch 的批组成改变浮点求和顺序）。**失败方式**：>2% 则先查实现，不报物理；
   但允许单项 E_hqs 超出 2%——(ab4) 测的正是 E_hqs 的分歧（−316…+219 nats）。

K2 跨测量复现（**可能失败**）：20 张真实池图的 Λ_clean(σ_n)（5 档均值）对 §3.2 的表
   （0.01→22210、0.02→13290、0.05→3530、0.10→1054、0.20→50），容差 ±15%。
   **注意这不是代码判据**：两边图源不同（§3.2 是 12 张真实图，本组是另外 20 张真实图），
   系统性偏移本身是"图源效应"的度量而非错误。若超差，报差异并说明来源，不据此否 K0/K1。

K3 主判据一（S1：模糊偏好不随观测噪声存活）：Λ_noisy < 0 的配对比例。
   分三档报，因为 (z) 的原始数据在三档上行为不同（σ=0.01 是 1/3 正，σ≥0.02 是 0/12 负）：
     K3a 全部 100 对：(z) 报 0.933。**通过线 ≥ 0.85**；**否掉线 < 0.70**（则 S1 是 3 张图
         的运气，要降级为"在部分图上成立"）；[0.70, 0.85] 记为不可判定并按 SE ∝ 1/√n 外推
         需要多少张图。
     K3b σ ≥ 0.02 的 80 对：(z) 报 12/12。**通过线 ≥ 0.95**。
     K3c σ = 0.01 的 20 对：(z) 报 2/3。**只记录不判**——(z) 在此档没有给出可用先验，
         且量级上这一档正是 Λ_clean 与噪声代价交叉的地方，预期是灰的。
   **必须同时报的估计量脆弱性**：逐图的 Λ_noisy 负值个数、mean/std、偏度、中位数 vs 均值
   ——按 §3.9(ai) 的教训，均值不稳健时必须连中位数一起报。

K3d 主判据一之伴生（排序）：Λ_noisy(θ*=1.5) > Λ_noisy(θ*=0) 的配对比例。(z) 报 15/15。
   **通过线 ≥ 0.95**。这条比 K3a 稳：它不依赖 Λ_clean 与噪声代价谁大，只依赖"模糊比不模糊好"。

K4 主判据二（S2：Lemma 12 的符号对应）：(Λ_clean > 0 且 E_hqs < 0) 或 (Λ_clean < 0 且
   E_hqs > 0) 的格子比例。(z) 报 15/15。**通过线 ≥ 0.95**；**否掉线 < 0.85**。
   必须分报：Λ_clean < 0 的格子里 E_hqs > 0 的比例（(z) 是 1/1）——这一格数量少但是
   S2 的"反向"半边，只报数量与符号，不设阈值。

K5 主判据三（幅度半）：Λ_clean ≥ 2000 nats 的格子上 mean(|E_hqs|/Λ_clean)。
   **与 (ab) 的 W2 用同一条预谱带宽**：mean ∈ [1.00, 1.25] 且 SE ≤ 0.03。
   **失败方式**：mean 掉出 [1.00, 1.25] ⇒ (z)/(ab) 的 12.4% 带是小样本运气。
   同时报三项恒等式 E_hqs = Δq − Λ_clean + (data_term − data) 的自检残差（(ab) 报 1.6e−14）。

K6 诊断（无阈值，**必须报**）：
   (a) |E_hqs|/Λ_clean 随 Λ_clean 的走向——(z) 的表显示它随 Λ_clean 减小而升到 9.882；
   (b) data/Λ_clean 与两项修正 −Δq/Λ_clean、data_term/data 的分布；
   (c) **图源效应**：20 张真实池图 vs 3 张先验样本的 Λ_clean 与 Λ_noisy 分布差异
       （这是本组相对 (z)/(ab) 唯一的结构性变化，必须量化）；
   (d) 逐图 σ 剖面：每张图的 Λ_noisy 是否随 σ 单调下降。

本实验**不**证明什么（诚实声明）
------------------------------
* 仍只覆盖一个算法族（全局 MAP / PnP-HQS）、一个先验（ffhq）、一个分辨率（64×64）。
* θ* 只取 1.5；θ* 普适性已由 (ab) 在 8 张先验样本上判过（W3 失败并被三项恒等式解释）。
* E_hqs 依赖优化器设置（3 个初值 × 240 迭代取最小），不是真最优。
* 不测另外两个算法（高斯谱近似、采样器 ESS）。
* 20 张图的二项标准误仍受图间相关性影响：5 个 σ 是同一张图的重复，K3a 的有效独立单元
  是 20 而不是 100，这一点在读 K3a 时必须计入。

用法： venv2/bin/python sweep_evidence3.py [--smoke]
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
    """与 evidence_scale.py 逐字相同（高斯核、零相位谱门、s=0 时恒等）。"""

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
        k = self.kernel(s)
        if k is None:
            return torch.ones((1, C, H, W // 2 + 1), device=DEV, dtype=DT)
        return self._K(k, C, H, W).abs() ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="evidence3.json")
    args = ap.parse_args()
    if args.smoke and args.out == "evidence3.json":
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
                            n_inner=N_INNER, eta_seed=ETA_SEED),
                   rows=rows), open(outfn, "w"), indent=1)
    print(f"\n[saved] {outfn}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
