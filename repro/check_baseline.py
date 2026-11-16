"""检验 40274 nats 这个数字是否只是谱估计基线的伪影。

背景（自查中发现的不一致）：
  * diag_gauss_bias.py 的高斯基线用了 Srad + TMIN^2（加了下限）
  * validate_spec_approx.py 的高斯基线用了 Srad（没加）
  * 两者估计谱只用了 64 个样本
  * 径向分桶后 1/S 的偏差不可忽略

而 Delta(y) - Delta(x*) = [exact(y)-exact(x*)] - [gauss(y)-gauss(x*)]，
第二项完全由谱估计决定。所以必须检验它对谱估计的敏感性。

本脚本：
  A. 用嵌套子集 K=16/64/256 估计径向谱，重算 Delta(y)-Delta(x*)
  B. 理论对照：对 M 张先验样本，计算高斯偏好的经验均值
     理论值 = 0.5 * sum_{rfft}(1 - |h_f|^2)   （当 |X_f|^2/S_f ~ 1 时）
     若单张 x* 的值远离 M 张的均值 -> x*  atypical，或谱估计有偏
  C. 直接报告 exact(y) - exact(x*)（基线无关量）

用法： python check_baseline.py
"""

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
H = W = 64
TMIN = 0.002
TSTAR = 1.5
NB = 48


class BlurOp:
    def __init__(self, ks=15):
        lin = torch.linspace(-(ks // 2) + 1, ks // 2, ks, device=DEV, dtype=DT)
        self.grid = lin[None] ** 2 + lin[:, None] ** 2

    def kernel(self, s):
        if s == 0:
            return None
        k = torch.exp(-self.grid / (2.0 * s ** 2))
        return (k / k.sum())[None, None]

    def A(self, x, s):
        k = self.kernel(s)
        if k is None:
            return x
        B, C, Hh, Ww = x.shape
        pad = torch.zeros((1, C, Hh, Ww), device=DEV, dtype=DT)
        kh, kw = k.shape[-2:]
        t, l = (Hh - kh) // 2, (Ww - kw) // 2
        pad[:, :, t:t + kh, l:l + kw] = k
        return torch.fft.irfft2(torch.fft.rfft2(x, dim=(-2, -1))
                                * torch.fft.rfft2(pad, dim=(-2, -1)),
                                s=(Hh, Ww), dim=(-2, -1))


def radial_bins(nbin=NB):
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)
    Rr = R[:, :W // 2 + 1]
    edges = np.linspace(0, Rr.max() + 1e-6, nbin + 1)
    idx = np.clip(np.digitize(Rr.ravel(), edges) - 1, 0, nbin - 1)
    return idx.reshape(Rr.shape), nbin


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    S_TOT = 256
    CHUNK = 32
    M_TEST = 24          # 用于经验均值对照的先验样本数

    model = load_local_model(CKPT).to(DEV)

    def denoiser(x, t, *a, **k):
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    blur = BlurOp()
    idx, _ = radial_bins()
    idx_t = torch.from_numpy(idx).to(DEV)

    # ---- 分块生成先验样本池（带磁盘缓存，避免重跑 7 分钟）----
    CACHE = f"/tmp/pool_ffhq64_{S_TOT}_s0.pt"
    if os.path.exists(CACHE):
        pool = torch.load(CACHE, map_location=DEV).to(DT)
        print(f"[pool] cached {tuple(pool.shape)} <- {CACHE}", flush=True)
    else:
        chunks = []
        chunks = []
        for i in range(S_TOT // CHUNK):
            c, _ = ode_sampler(ode_flow, shape=(CHUNK, 3, H, W), t_spans=t_spans,
                               device=DEV)
            chunks.append(c.to(DT))
            torch.cuda.empty_cache()
            print(f"[pool] {len(chunks)*CHUNK}/{S_TOT}", flush=True)
        pool = torch.cat(chunks, 0)
        del chunks
        torch.cuda.empty_cache()
        torch.save(pool.cpu(), CACHE)
        pool = pool.to(DEV)
        print(f"[pool] cached -> {CACHE}", flush=True)
    x_star = pool[S_TOT - 1:S_TOT]          # 最后一张当 x*
    xs_all = pool[:S_TOT - 1]
    xs_test = pool[:M_TEST]                 # 用于经验均值
    y_clean = blur.A(x_star, TSTAR)
    print(f"[pool] done, {pool.shape}", flush=True)

    # ---- 径向谱（嵌套子集）----
    with torch.no_grad():
        P = (torch.fft.rfft2(xs_all, dim=(-2, -1)).abs() ** 2) / (H * W)
        Pc = P.mean(1)                       # (S-1, H, W//2+1)

        def srad_from(k):
            k = min(k, Pc.shape[0])
            Pk = Pc[:k]
            out = torch.zeros(NB, device=DEV, dtype=DT)
            for b in range(NB):
                m = (idx_t == b).reshape(-1)
                if m.any():
                    out[b] = Pk.reshape(k, -1)[:, m].mean()
            return out

        KS = (16, 64, min(256, Pc.shape[0]))
        Srad = {k: srad_from(k) for k in KS}

        # |h|^2 的径向均值（用于理论预测）
        k = blur.kernel(TSTAR)
        pad = torch.zeros((1, 1, H, W), device=DEV, dtype=DT)
        kh = k.shape[-2]
        t = (H - kh) // 2
        pad[:, :, t:t + kh, t:t + kh] = k
        h2 = torch.fft.rfft2(pad, dim=(-2, -1)).abs() ** 2
        h2r = torch.zeros(NB, device=DEV, dtype=DT)
        for b in range(NB):
            m = (idx_t == b).reshape(-1)
            if m.any():
                h2r[b] = h2.reshape(-1)[m].mean()

        n_mode = torch.zeros(NB, device=DEV, dtype=DT)
        for b in range(NB):
            n_mode[b] = (idx_t == b).sum()

    # 理论高斯偏好：0.5 * sum_f (1-|h_f|^2)，按实际模式数加权
    theo = float(0.5 * (n_mode * (1.0 - h2r)).sum())
    print(f"\n[理论] 0.5*sum_f(1-|h_f|^2) = {theo:.0f} nats"
          f"   (mode 总数 {int(n_mode.sum())})", flush=True)

    # ---- 精确 log p_t（基线无关）----
    @torch.no_grad()
    def exact_logp(z):
        return -float(likelihood_fn(z.to(DT), ode_flow,
                                    t_spans=[80.0, TMIN]).item())

    ex_y = exact_logp(y_clean)
    ex_x = exact_logp(x_star)
    print(f"\n[精确] log p_t(y_blur) = {ex_y:.1f}", flush=True)
    print(f"[精确] log p_t(x*)     = {ex_x:.1f}", flush=True)
    print(f"[精确] exact(y)-exact(x*) = {ex_y - ex_x:+.1f}  <== 基线无关", flush=True)

    # ---- 高斯基线：随 K 与是否加 TMIN^2 变化 ----
    def gauss_logp(z, Srad, floor):
        with torch.no_grad():
            Z = torch.fft.rfft2(z, dim=(-2, -1))
            Zp = (Z.abs() ** 2).mean(1, keepdim=True) / (H * W)
            Sf = (Srad + floor)[idx_t][None, None]
            return float(-(Zp / (2 * Sf)).sum()
                         - 0.5 * torch.log(2 * np.pi * Sf).sum())

    print("\n" + "=" * 96)
    print("Delta(y) - Delta(x*)  对谱估计的敏感性")
    print("=" * 96)
    print(f"{'K':>5} | {'floor':>10} | {'gauss(y)':>11} | {'gauss(x*)':>11} | "
          f"{'g(y)-g(x*)':>12} | {'D(y)-D(x*)':>12}")
    print("-" * 96)
    for K in KS:
        for floor in (0.0, TMIN ** 2):
            gy = gauss_logp(y_clean, Srad[K], floor)
            gx = gauss_logp(x_star, Srad[K], floor)
            dd = (ex_y - gy) - (ex_x - gx)
            tag = "TMIN^2" if floor > 0 else "0"
            print(f"{K:>5} | {tag:>10} | {gy:>11.1f} | {gx:>11.1f} | "
                  f"{gy-gx:>+12.1f} | {dd:>+12.1f}", flush=True)

    # ---- 经验均值对照：M 张先验样本各自的高斯偏好 ----
    print("\n" + "=" * 96)
    print("B. 单张 vs M 张均值（检验 x* 是否 atypical / 谱估计是否有偏）")
    print("=" * 96)
    S256 = Srad[KS[-1]]
    with torch.no_grad():
        # 每张测试图：高斯偏好 = 0.5*sum_f (1-|h_f|^2)*|X_f|^2/S_f
        prefs = []
        for i in range(M_TEST):
            xi = xs_test[i:i + 1]
            yb = blur.A(xi, TSTAR)
            Zx = (torch.fft.rfft2(xi, dim=(-2, -1)).abs() ** 2).mean(1, keepdim=True) / (H * W)
            Zy = (torch.fft.rfft2(yb, dim=(-2, -1)).abs() ** 2).mean(1, keepdim=True) / (H * W)
            Sf = S256[idx_t][None, None]
            prefs.append(float(((Zx - Zy) / (2 * Sf)).sum()))
            del xi, yb
    prefs = np.array(prefs)
    # x* 自己的偏好
    with torch.no_grad():
        Zx = (torch.fft.rfft2(x_star, dim=(-2, -1)).abs() ** 2).mean(1, keepdim=True) / (H * W)
        Zy = (torch.fft.rfft2(y_clean, dim=(-2, -1)).abs() ** 2).mean(1, keepdim=True) / (H * W)
        Sf = S256[idx_t][None, None]
        pref_xstar = float(((Zx - Zy) / (2 * Sf)).sum())
    print(f"  M={M_TEST} 张先验样本的高斯偏好: mean={prefs.mean():+.1f}  "
          f"std={prefs.std():.1f}  min={prefs.min():+.1f}  max={prefs.max():+.1f}")
    print(f"  x* 自己的高斯偏好            : {pref_xstar:+.1f}")
    print(f"  理论值 0.5*sum(1-|h|^2)      : {theo:+.1f}")
    print(f"  x* 相对 M 张均值的偏离       : {pref_xstar - prefs.mean():+.1f}"
          f"  ({(pref_xstar-prefs.mean())/max(prefs.std(),1e-9):+.2f} sigma)")

    print("\n[判据]")
    print("  * Delta(y)-Delta(x*) 若随 K/floor 大幅变动 -> 40274 是谱估计伪影，需重算")
    print("  * 若稳定 -> 该结论稳健，且 exact(y)-exact(x*) 是其主要成分")
    print("  * 若 x* 的高斯偏好远离 M 张均值 -> 单图结论不可外推")


if __name__ == "__main__":
    main()
