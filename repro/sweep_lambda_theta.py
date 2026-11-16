"""Lambda(theta)：模糊偏好随模糊强度与核族的变化，及其机理判别。

目的：判定 Lambda 的机理是"谱的"还是"流形厚度的"。
  谱的故事：Lambda 只由二阶统计决定 -> 各族在 (Lambda_Gauss, Lambda) 平面上
            落在同一条直线 Lambda = c * Lambda_Gauss 上
  流形的故事：Lambda 依赖扰动在频域的方向/形状 -> 同一 killed energy 下各族散开

五个族：
  gaussian / defocus / airy / motion  （作者 putils.convolution，与 Phase B 一致）
  lowpass                              （理想径向硬门，纯谱扰动的参照）

对每个 (族, theta, t)：
  * 批量算 log p_t(整批模糊图) 与 log p_t(整批原图)   -> Lambda 逐图 + 均值/std
  * 闭式高斯大对照 Lambda_Gauss（用同一套估计谱 S）
  * kappa(theta) = 被核杀掉的先验能量占比（族无关的公共横轴）

likelihood_fn 支持批量（内部 state 是 (B*N+B,) 展平向量），冒烟模式会验证：
批量结果与逐张结果的差应在 Hutchinson 噪声量级（~几百 nats）内。

用法：
  python sweep_lambda_theta.py --smoke
  python sweep_lambda_theta.py --nimg 12 --ts 0.002 0.02
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
from putils.convolution import GaussianBlur, DefocusBlur, AiryDisc, MotionBlur

CKPT = "edm-ffhq-64x64-uncond-vp.pkl"
DEV = torch.device("cuda")
DT = torch.float32
OUT = os.path.join(_PROJ, "results")
H = W = 64
NB = 48
KS = 15
def pool_cache_for(ckpt, n=256):
    tag = os.path.splitext(os.path.basename(ckpt))[0].replace("edm-", "").replace("-uncond", "")
    return f"/tmp/pool_{tag}_{n}_s0.pt"

# theta 越大越模糊；覆盖 kappa 从 ~0 到 ~0.95
FAMILIES = {
    "gaussian": (GaussianBlur, "sigma", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]),
    "defocus": (DefocusBlur, "radius", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]),
    "airy": (AiryDisc, "radius", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]),
    "motion": (MotionBlur, "length", [1, 3, 5, 7, 9, 13]),
    "lowpass": (None, "cutoff", [0.50, 0.35, 0.25, 0.18, 0.12, 0.08]),
}


def blur_fft(x, kernel):
    """循环（FFT）卷积，pad 形状 (1,C,H,W) 才能对 batch 广播。"""
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


def khat_of(kernel):
    """(1,1,H,W//2+1) 的核频谱。"""
    if kernel.ndim == 2:
        kernel = kernel[None, None]
    kh, kw = kernel.shape[-2:]
    pad = torch.zeros((1, 1, H, W), device=DEV, dtype=torch.float64)
    top, left = (H - kh) // 2, (W - kw) // 2
    pad[:, :, top:top + kh, left:left + kw] = kernel.to(torch.float64)
    return torch.fft.rfft2(pad, dim=(-2, -1))


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--nimg", type=int, default=12)
    ap.add_argument("--ts", nargs="*", type=float, default=[0.002, 0.02])
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4,
                    help="likelihood_fn 的批大小；显存预算 2.5GiB 下 B=4 峰值 ~1.5GiB")
    ap.add_argument("--ckpt", default=CKPT, help="换模型做跨模型复现")
    ap.add_argument("--class-idx", type=int, default=None,
                    help="条件模型（如 imagenet adm）的类别标签")
    ap.add_argument("--levels", nargs="*", type=float, default=None,
                    help="自定义 theta 档位（默认用各族内置范围）")
    ap.add_argument("--poolsize", type=int, default=256)
    ap.add_argument("--poolchunk", type=int, default=32,
                    help="生成先验样本池时的批大小；大模型要调小")
    ap.add_argument("--ks", type=int, default=KS,
                    help="核边长；强模糊时 15 会因循环卷积回绕而产生伪影，可用 31 复跑")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.smoke:
        nimg, fams = 2, ["gaussian"]
        levels, ts = [1.0, 2.0], [0.002]
    else:
        nimg = args.nimg
        fams = args.families or list(FAMILIES)
        levels, ts = args.levels, args.ts

    print(f"[cfg] nimg={nimg} families={fams} ts={ts}", flush=True)

    CK = args.ckpt
    model = load_local_model(CK).to(DEV)

    def denoiser(x, t, *a, **k):
        if args.class_idx is not None:
            lab = torch.full((x.shape[0],), int(args.class_idx),
                             device=x.device, dtype=torch.long)
            onehot = torch.nn.functional.one_hot(lab, 1000).to(DT)
            return model(x.to(DT), t, class_labels=onehot, *a, **k)
        return model(x.to(DT), t, *a, **k)

    prior = ScorePrior(denoiser=denoiser)
    ode_flow, t_spans = get_ode_flow(prior)
    idx, _ = radial_bins()
    idx_t = torch.from_numpy(idx).to(DEV)

    # ---------------- 先验样本池（缓存）----------------
    POOL_CACHE = pool_cache_for(CK, args.poolsize)
    if os.path.exists(POOL_CACHE):
        pool = torch.load(POOL_CACHE, map_location=DEV).to(DT)
        print(f"[pool] cached {tuple(pool.shape)} <- {POOL_CACHE}", flush=True)
    else:
        PC = args.poolchunk
        chunks = []
        for i in range(args.poolsize // PC):
            c, _ = ode_sampler(ode_flow, shape=(PC, 3, H, W), t_spans=t_spans,
                               device=DEV)
            chunks.append(c.to(DT))
            torch.cuda.empty_cache()
            print(f"[pool] {(i + 1) * PC}/{args.poolsize}", flush=True)
        pool = torch.cat(chunks, 0)
        del chunks
        torch.cuda.empty_cache()
        torch.save(pool.cpu(), POOL_CACHE)
        pool = pool.to(DEV)
    xs = pool[:nimg].clone()
    print(f"[pool] using first {nimg}", flush=True)

    # ---------------- 径向谱（用全部 256 张估，噪声小）----------------
    with torch.no_grad():
        P = (torch.fft.rfft2(pool, dim=(-2, -1)).abs() ** 2) / (H * W)
        Pc = P.mean(1)
        Srad = torch.zeros(NB, device=DEV, dtype=torch.float64)
        for b in range(NB):
            m = (idx_t == b).reshape(-1)
            if m.any():
                Srad[b] = Pc.reshape(P.shape[0], -1)[:, m].mean()
        # 逐频率的 S（径向分桶展开）
        Sfull = Srad[idx_t][None, None]                       # (1,1,H,W//2+1)
        # 各图功率（通道平均）
        Zx = (torch.fft.rfft2(xs, dim=(-2, -1)).abs() ** 2).mean(1, keepdim=True) / (H * W)

    @torch.no_grad()
    def logp_batch_vec(z, t):
        """返回 list[float]，长度 = batch。"""
        nll = likelihood_fn(z.to(DT), ode_flow, t_spans=[80.0, float(t)])
        return [-float(v) for v in nll.reshape(-1)]

    B = args.batch

    def logp_chunked(z, t):
        out = []
        for i in range(0, z.shape[0], B):
            out.extend(logp_batch_vec(z[i:i + B], t))
            torch.cuda.empty_cache()
        return np.array(out)

    # ---------------- 冒烟：验证批量 vs 逐张 ----------------
    if args.smoke:
        print("\n[smoke] 批量 vs 逐张（差应在 Hutchinson 噪声 ~几百 nats 内）")
        kb = GaussianBlur(kernel_size=args.ks, device=DEV, dtype=DT).kernel(1.0)
        yb = blur_fft(xs, kb)
        for t in ts:
            b = logp_chunked(yb, t)
            s = [logp_batch_vec(yb[i:i + 1], t)[0] for i in range(yb.shape[0])]
            for i, (bb, ss) in enumerate(zip(b, s)):
                print(f"  t={t} img{i}: batch={bb:12.1f}  single={ss:12.1f}  "
                      f"diff={bb-ss:+9.1f}")
        print("[smoke] OK")
        return

    # ---------------- 清晰基准（分批）----------------
    lp_sharp = {}
    for t in ts:
        lp_sharp[t] = logp_chunked(xs, t)
        print(f"[sharp] t={t}  mean log p = {lp_sharp[t].mean():.1f} "
              f"+/- {lp_sharp[t].std():.1f}", flush=True)

    # ---------------- 扫描 ----------------
    records = []
    for fam in fams:
        cls, pname, default_levels = FAMILIES[fam]
        lv = levels if levels is not None else default_levels
        kernel_fn = cls(kernel_size=args.ks, device=DEV, dtype=DT) if cls is not None else None
        for th in lv:
            # --- 核与 |h|^2 ---
            if fam == "lowpass":
                cut = float(th)
                fx = np.fft.fftfreq(H) * H
                fy = np.fft.fftfreq(W) * W
                FX, FY = np.meshgrid(fx, fy, indexing="ij")
                R = np.sqrt(FX ** 2 + FY ** 2)
                gate = (R <= cut * R.max()).astype(np.float64)
                h2 = torch.from_numpy(gate[:, :W // 2 + 1]).to(DEV)[None, None]
                yb = torch.fft.irfft2(torch.fft.rfft2(xs.to(torch.float64),
                                                      dim=(-2, -1)) * h2,
                                      s=(H, W), dim=(-2, -1)).to(DT)
            else:
                k = kernel_fn.kernel(th)
                if k is None:
                    yb = xs.clone()
                else:
                    yb = blur_fft(xs, k)
                h2 = khat_of(k).abs() ** 2                  # (1,1,H,W//2+1)

            # kappa = 被杀掉的先验能量占比
            with torch.no_grad():
                kappa = float(((1.0 - h2) * Sfull).sum() / Sfull.sum())
                Zy = (torch.fft.rfft2(yb, dim=(-2, -1)).abs() ** 2).mean(
                    1, keepdim=True) / (H * W)

            for t in ts:
                lp = logp_chunked(yb, t)
                lam = lp - lp_sharp[t]
                # 高斯大对照（逐图）
                Sf = (Sfull + t ** 2)
                lam_g = float((((Zx - Zy) / (2 * Sf)).sum(dim=(1, 2, 3))).mean())
                lam_g_std = float((((Zx - Zy) / (2 * Sf)).sum(dim=(1, 2, 3))).std())
                rec = dict(family=fam, param=pname, theta=float(th), t=float(t),
                           kappa=kappa,
                           lam_mean=float(lam.mean()), lam_std=float(lam.std()),
                           lam_min=float(lam.min()), lam_max=float(lam.max()),
                           lam_gauss_mean=lam_g, lam_gauss_std=lam_g_std,
                           logp_blur_mean=float(lp.mean()),
                           logp_sharp_mean=float(lp_sharp[t].mean()),
                           lam_per_img=[float(v) for v in lam])
                records.append(rec)
                print(f"  {fam:9s} {pname:<7s}={str(th):<5} t={t:<6} "
                      f"kappa={kappa:5.3f}  Lambda={lam.mean():+10.1f}"
                      f" +/-{lam.std():7.1f}  Gauss={lam_g:+9.1f}"
                      f"  x{lam.mean()/lam_g if abs(lam_g)>1e-6 else float('nan'):6.2f}",
                      flush=True)
                del lp
                torch.cuda.empty_cache()
            del yb
            torch.cuda.empty_cache()

    tag = os.path.splitext(os.path.basename(CK))[0].replace("edm-", "")
    outfn = os.path.join(OUT, f"sweep_lambda_theta_{tag}.json")
    with open(outfn, "w") as f:
        json.dump(dict(cfg=dict(nimg=nimg, ts=ts, families=fams, ckpt=CK,
                                class_idx=args.class_idx, ks=args.ks,
                                pool="first nimg of 256-pool seed0"),
                       records=records), f, indent=1)
    print(f"\n[saved] {outfn}", flush=True)


if __name__ == "__main__":
    main()
