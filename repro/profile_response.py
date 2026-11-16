"""剖面响应拟合 v2：Lambda 是不是"衰减剖面 a(f) 的一个泛函"？

背景（阶段一计划 §7.1）：强模糊端逐路径斜率散开 50% 且按核强度分两组；
kunif / kappa / 过渡带宽度 / 阻带尾 / 各向异性 / 位移范数 全部不能解释。

v1 的发现（本脚本沿用并修正）：
  G1  重算剖面 a(f) = 1-|h_a(f)|^2，复现已存 kunif / kappa / Lambda_Gauss 到 1e-7
      ——注意 Lambda_Gauss 必须用 20 图平均谱（逐图算再平均），用单张图会差 19%
  G1b Lambda_Gauss 的结构上限 = Sum_f Zx_f/(2 Sf) ≈ N/2 = 1056 nats，实测已打满
      => 高斯响应不可能解释 Lambda（最大差 36 倍），必须找非线性的剖面泛函

v2 的主检验（G4）：
      Lambda ≈ Sum_f phi(a(f))          phi 自由、分段常数、非负
  即"对数密度落差 = 逐模式的衰减函数之和"。这是高斯对照（phi(a)=a*const）
  的最直接非高斯推广，参数只有 K 个且是直方图回归（条件数好）。
  若样本外也成立 => 一条定律：给定任意算子剖面即可预测 Lambda。

同时保留的对照定律：
  L0  Lambda = C u^b                    （kunif 幂律，现任经验式）
  L1  Lambda = C U1^b
  L2  Lambda = Sum_k c_k U_k,  U_k=Sum_f a^k   （矩基）
  L3  Lambda = Sum_j phi_j n_j(a)       （本脚本主检验）
  L4  Lambda = Sum_j phi_j n_j(a) w0    （先验谱加权版）
  L5  Lambda = Sum_{j,r} phi_{jr} n_{jr}(a)  （衰减值 x 半径 的二维直方图）

用法： venv2/bin/python profile_response.py [--nimg 20] [--snr 3]
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from scipy.optimize import nnls

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from putils.convolution import GaussianBlur, DefocusBlur, AiryDisc, MotionBlur

OUT = os.path.join(_PROJ, "results")
H = W = 64
KS = 15
NB = 48
POOL = "/tmp/pool_ffhq64_256_s0.pt"

KERNELS = [("gaussian", 1.0), ("gaussian", 2.0),
           ("defocus", 1.0), ("defocus", 2.0),
           ("airy", 2.0), ("airy", 4.0),
           ("motion", 3.0), ("motion", 7.0),
           ("lowpass", 0.25)]
ALPHAS = [0.15, 0.25, 0.4, 0.6, 0.8, 1.0]

DEV = torch.device("cpu")
DT = torch.float32

# 衰减值 a 的分箱（直方图回归用；对数间隔，因为弱模糊端 a 很小）
A_EDGES = [0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0001]
# 半径分箱（Nyquist=32 归一后）
R_EDGES = [0.0, 0.06, 0.12, 0.20, 0.32, 0.50, 0.75, 1.05]


# ---------------------------------------------------------------- 基础工具
def mode_grid():
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    return np.sqrt(FX ** 2 + FY ** 2)


def radial_bins(nbin=NB):
    R = mode_grid()
    Rr = R[:, :W // 2 + 1]
    edges = np.linspace(0, Rr.max() + 1e-6, nbin + 1)
    idx = np.clip(np.digitize(Rr.ravel(), edges) - 1, 0, nbin - 1)
    return idx.reshape(Rr.shape), nbin, Rr


def make_kernel(fam, th):
    """与 sweep_alpha.make_kernel 一致。"""
    if fam == "lowpass":
        fx = np.fft.fftfreq(H) * H
        fy = np.fft.fftfreq(W) * W
        FX, FY = np.meshgrid(fx, fy, indexing="ij")
        R = np.sqrt(FX ** 2 + FY ** 2)
        gate = torch.from_numpy((R <= th * R.max()).astype(np.float64))
        return None, gate[:, :W // 2 + 1]
    cls = {"gaussian": GaussianBlur, "defocus": DefocusBlur,
           "airy": AiryDisc, "motion": MotionBlur}[fam]
    k = cls(kernel_size=KS, device=DEV, dtype=DT).kernel(float(th))
    return k, None


def h2_of_alpha(alpha, k, gate):
    """(H, W//2+1) 的 |h_alpha|^2。与 sweep_alpha 一致。"""
    if gate is not None:
        h2 = ((1.0 - alpha) + alpha * gate.to(torch.float64)) ** 2
        return h2.numpy().astype(np.float64)
    kk = k[None, None] if k.ndim == 2 else k
    kh, kw = kk.shape[-2:]
    pad = torch.zeros((1, 1, H, W), dtype=torch.float64)
    top, left = (H - kh) // 2, (W - kw) // 2
    pad[:, :, top:top + kh, left:left + kw] = kk.to(torch.float64)
    delta = torch.zeros_like(pad)
    delta[:, :, 0, 0] = 1.0
    ha = (1.0 - alpha) * delta + alpha * pad
    return torch.fft.rfft2(ha, dim=(-2, -1)).abs().square()[0, 0].numpy()


def hist_features(A, w0, a_edges, r_edges=None, Rr=None):
    """把每条剖面 a(f) 编成直方图特征。

    r_edges=None：只按 a 的值分箱 -> (n, K)
    否则：按 (a 值, 半径) 二维分箱 -> (n, K*J)
    w0：可选的逐频率权重（None 时为计数）
    """
    n = len(A)
    aidx = np.clip(np.digitize(A.ravel(), a_edges) - 1, 0,
                   len(a_edges) - 2).reshape(A.shape)
    K = len(a_edges) - 1
    w = np.ones_like(A) if w0 is None else np.broadcast_to(w0, A.shape)
    if r_edges is None:
        F = np.zeros((n, K))
        for j in range(K):
            F[:, j] = (w * (aidx == j)).sum(axis=(1, 2))
        return F
    ridx = np.clip(np.digitize(Rr.ravel(), r_edges) - 1, 0,
                   len(r_edges) - 2).reshape(Rr.shape)
    J = len(r_edges) - 1
    F = np.zeros((n, K * J))
    for j in range(K):
        for q in range(J):
            F[:, j * J + q] = (w * (aidx == j) * (ridx == q)).sum(axis=(1, 2))
    return F


def ols(X, y, w=None):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    if w is None:
        w = np.ones(n)
    A_ = X.T @ (X * w[:, None]); b = X.T @ (w * y)
    try:
        coef = np.linalg.solve(A_, b)
    except np.linalg.LinAlgError:
        return None
    res = y - X @ coef
    sst = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    ssr = float(np.sum(w * res ** 2))
    Hm = X @ np.linalg.pinv(X.T @ X) @ X.T
    return dict(coef=coef, r2=1 - ssr / sst if sst > 0 else float("nan"),
                resid=res, hat=np.diag(Hm), pred=X @ coef,
                rmse=float(math.sqrt(ssr / n)))


def fit_report(tag, X, y, w, npar):
    f = ols(X, y, w)
    if f is None:
        return None
    return dict(tag=tag, r2=f["r2"], rmse=f["rmse"], npar=npar,
                maxhat=float(f["hat"].max()), coef=f["coef"],
                X=X, y=y, w=w)


def law_table(laws, y, w, mask_fit, kern_key, tag_os):
    """对每条定律报 in-sample R2 + 两种样本外误差。"""
    print(f"\n {'定律':<26} | {'参数':>4} | {'R2':>7} | {'maxhat':>7} | "
          f"{'弱->强 dex':>10} | {'留一核 dex':>10}")
    print(" " + "-" * 82)
    rows = []
    tr = np.where(mask_fit)[0]
    te = np.where(~mask_fit)[0]

    def aug(X, idx):
        """按需补截距列。"""
        if L_cur.get("has_int"):
            return X[idx]
        return np.column_stack([np.ones(len(idx))] + [X[idx][:, j]
                               for j in range(X.shape[1])])

    for L in laws:
        L_cur = L
        X = L["X"]
        f = ols(aug(X, np.arange(len(y))), y, w)
        if f is None:
            continue
        e1 = None
        if len(te) > 2:
            ff = ols(aug(X, tr), y[tr], w[tr])
            if ff is not None:
                e1 = float(math.sqrt((((aug(X, te) @ ff["coef"])
                                       - y[te]) ** 2).mean()))
        errs = []
        for u in sorted(set(kern_key)):
            t2 = np.array([i for i, k in enumerate(kern_key) if k != u])
            s2 = np.array([i for i, k in enumerate(kern_key) if k == u])
            if len(s2) < 3:
                continue
            ff = ols(aug(X, t2), y[t2], w[t2])
            if ff is not None:
                errs.append(float(math.sqrt((((aug(X, s2) @ ff["coef"])
                                              - y[s2]) ** 2).mean())))
        e2 = float(np.median(errs)) if errs else float("nan")
        print(f" {L['tag']:<26} | {L['npar']:>4} | {f['r2']:>7.4f} | "
              f"{f['hat'].max():>7.3f} | "
              f"{(f'{e1:.3f}' if e1 is not None else '--'):>10} | {e2:>10.3f}")
        rows.append(dict(tag=L["tag"], npar=L["npar"], r2=f["r2"],
                         rmse=f["rmse"], maxhat=float(f["hat"].max()),
                         os_weak=e1, os_kernel=e2))
    print(" " + "-" * 82)
    print(" dex = log10 单位的误差；0.01 dex ≈ 2.3%")
    return rows


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nimg", type=int, default=20)
    ap.add_argument("--snr", type=float, default=3.0)
    ap.add_argument("--data", default="sweep_alpha.json")
    args = ap.parse_args()

    src = json.load(open(os.path.join(OUT, args.data)))
    recs = src["records"]
    t0 = min(r["t"] for r in recs)
    nimg = src["cfg"]["nimg"]
    lam = np.array([r["lam_mean"] for r in recs])
    lam_std = np.array([r["lam_std"] for r in recs])
    lam_g = np.array([r["lam_gauss_mean"] for r in recs])
    kunif_saved = np.array([r["kunif"] for r in recs])
    kappa_saved = np.array([r["kappa"] for r in recs])
    kern_key = [(r["family"], r["theta"]) for r in recs]
    alpha = np.array([r["alpha"] for r in recs])
    snr = lam / np.maximum(lam_std, 1e-9) * math.sqrt(nimg)
    keep = snr >= args.snr

    # ---------------- 图像池谱 ----------------
    pool = torch.load(POOL, map_location="cpu").to(torch.float64)
    xs = pool[:nimg].clone()
    idx, NB_, Rr = radial_bins()
    idx_flat = idx.ravel()
    with torch.no_grad():
        P = (torch.fft.rfft2(pool, dim=(-2, -1)).abs() ** 2) / (H * W)
        Pc = P.mean(1).numpy()
        Srad = np.zeros(NB)
        Pf = Pc.reshape(len(pool), -1)
        for b in range(NB):
            m = idx_flat == b
            if m.any():
                Srad[b] = Pf[:, m].mean()
        Sfull = Srad[idx]
        Zx = (torch.fft.rfft2(xs, dim=(-2, -1)).abs() ** 2).mean(1, keepdim=True) \
            .numpy().mean(0)[0] / (H * W)

    Sf = Sfull + t0 ** 2
    w0 = Sfull / Sf
    wg = Zx / (2.0 * Sf)
    Rrn = Rr / Rr.max()                      # 半径归一到 [0,1]

    print("=" * 104)
    print(f" t={t0:g}  nimg={nimg}  记录={len(recs)}  核={len(set(kern_key))}  "
          f"SNR>={args.snr} 保留 {int(keep.sum())} 点")
    print("=" * 104)

    # ---------------- 剖面 ----------------
    A = np.zeros((len(recs), H, W // 2 + 1))
    kcache = {}
    for i, r in enumerate(recs):
        kk = (r["family"], r["theta"])
        if kk not in kcache:
            kcache[kk] = make_kernel(*kk)
        k, gate = kcache[kk]
        A[i] = 1.0 - h2_of_alpha(r["alpha"], k, gate)

    kunif_new = A.mean(axis=(1, 2))
    kappa_new = (A * Sfull[None]).sum(axis=(1, 2)) / Sfull.sum()
    lamg_new = (A * wg[None]).sum(axis=(1, 2))

    print("\n" + "=" * 104)
    print(" G1  三重验证：重算剖面 vs 已存值")
    print("=" * 104)
    ok = True
    for nm, new, old in [("kunif", kunif_new, kunif_saved),
                         ("kappa", kappa_new, kappa_saved),
                         ("Lambda_Gauss", lamg_new, lam_g)]:
        rel = np.abs(new - old) / np.maximum(np.abs(old), 1e-12)
        good = rel.max() < 2e-3
        ok = ok and good
        print(f" {nm:<12} 最大相对误差 {rel.max():.3e}  "
              f"中位数 {np.median(rel):.3e}  {'OK' if good else '!! 不一致'}")
    if not ok:
        print(" => 关卡未过，停止")
        sys.exit(1)
    print(" => 剖面 a(f) 复现三个独立量，机器可信")

    cap = float((Zx / (2.0 * Sf)).sum())
    print(f"\n G1b  Lambda_Gauss 上限 = Sum_f Zx_f/(2Sf) = {cap:.1f}"
          f"（N/2 = {H*(W//2+1)/2:.0f}）")
    print(f"      实测 Lambda_Gauss 最大 {lamg_new.max():.1f}"
          f"（打满 {100*lamg_new.max()/cap:.1f}%）")
    print(f"      实测 Lambda 最大 {lam.max():.0f} = 上限的 "
          f"{lam.max()/cap:.1f} 倍 => 高斯响应原理上不可能解释 Lambda")

    # ---------------- 描述量 ----------------
    u_kunif = 1.0 - kunif_new
    U = {k: (A ** k).sum(axis=(1, 2)) for k in (1, 2, 3, 4)}
    U1w = (A * w0[None]).sum(axis=(1, 2))

    print("\n" + "=" * 104)
    print(" G2  描述量（全部 54 点）")
    print("=" * 104)
    print(f" {'描述量':<26} | {'跨度 dex':>8} | {'corr(log,L)':>12}")
    print(" " + "-" * 54)
    ll_all = np.log(lam)
    for nm, v in [("U1=Sum a (加权 w0)", U1w), ("U1=Sum a (未加权)", U[1]),
                  ("U2=Sum a^2", U[2]), ("U3=Sum a^3", U[3]),
                  ("U4=Sum a^4", U[4]), ("u=1-kunif", u_kunif),
                  ("kappa", kappa_new), ("Lambda_Gauss", lamg_new)]:
        lv = np.log(np.maximum(v, 1e-300))
        print(f" {nm:<26} | {lv.max()-lv.min():>8.2f} | "
              f"{np.corrcoef(lv, ll_all)[0,1]:>12.3f}")

    # ---------------- 只用 SNR 足够的点做定律拟合 ----------------
    y = np.log(lam[keep])
    se = np.maximum(lam_std[keep] / math.sqrt(nimg), 1.0)
    wts = 1.0 / np.maximum(se, 0.3 * np.median(se)) ** 2
    kk_keep = [kern_key[i] for i in np.where(keep)[0]]
    al_keep = alpha[keep]
    weak = al_keep <= 0.4

    # ---------------- 定律库 ----------------
    laws = []
    def add(tag, cols, npar, has_int=False):
        X = np.column_stack(cols) if len(cols) else np.zeros((len(y), 0))
        laws.append(dict(tag=tag, X=np.asarray(X, float), npar=npar,
                         has_int=has_int))

    add("L0 kunif 幂律", [np.log(u_kunif[keep])], 2)
    add("L1 U1 幂律", [np.log(U1w[keep])], 2)
    add("L2 矩基 U1..U3", [np.log(U1w[keep]), np.log(U[2][keep]),
                        np.log(U[3][keep])], 4)
    add("L2b 矩基 U1..U4", [np.log(U1w[keep]), np.log(U[2][keep]),
                         np.log(U[3][keep]), np.log(U[4][keep])], 5)

    # L3：phi(a) 直方图（NNLS，非负）
    F_a = hist_features(A[keep], None, A_EDGES)
    Ka = len(A_EDGES) - 1
    # L4：phi(a) 加权
    F_aw = hist_features(A[keep], w0, A_EDGES)
    # L5：phi(a,r) 二维
    F_ar = hist_features(A[keep], w0, A_EDGES, R_EDGES, Rrn)

    print("\n" + "=" * 104)
    print(" G3  衰减值直方图（每条剖面的 a(f) 分布，10 分位摘要）")
    print("=" * 104)
    print(f" {'a 箱':<14} | {'频率占比(均值)':>14} | " +
          " | ".join(f"a={a:g}" for a in ALPHAS))
    print(" " + "-" * 90)
    cnt = F_a / (H * (W // 2 + 1))
    ki = np.where(keep)[0]
    al_k = alpha[ki]
    for j in range(Ka):
        row = " | ".join(f"{cnt[al_k == a, j].mean():.3f}" for a in ALPHAS)
        print(f" [{A_EDGES[j]:.2f},{A_EDGES[j+1]:.2f}){'':<3} | "
              f"{cnt[:, j].mean():>14.3f} | {row}")

    # NNLS 拟合（无截距；phi>=0）
    def nnls_fit(F, y, w):
        sw = np.sqrt(w)
        coef, _ = nnls(F * sw[:, None], y * sw)
        pred = F @ coef
        res = y - pred
        ssr = float(np.sum(w * res ** 2))
        sst = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
        Hm = F @ np.linalg.pinv(F.T @ (F * w[:, None])) @ (F * w[:, None]).T
        return dict(coef=coef, r2=1 - ssr / sst if sst > 0 else float("nan"),
                    hat=np.diag(Hm), rmse=float(math.sqrt(ssr / len(y))),
                    pred=pred)

    print("\n" + "=" * 104)
    print(" G4  主检验：Lambda ≈ Sum_f phi(a(f))，phi 自由非负分段常数")
    print("=" * 104)
    fL3 = nnls_fit(F_a, y, wts)
    fL4 = nnls_fit(F_aw, y, wts)
    fL5 = nnls_fit(F_ar, y, wts)
    for nm, F, f in [("L3 计数直方图", F_a, fL3),
                     ("L4 先验谱加权直方图", F_aw, fL4),
                     ("L5 (a x 半径) 二维", F_ar, fL5)]:
        print(f" {nm:<22} K={F.shape[1]:>3}  R2={f['r2']:.4f}  "
              f"残差={f['rmse']:.4f}dex  maxhat={f['hat'].max():.3f}  "
              f"phi 范围 [{f['coef'].min():.1f}, {f['coef'].max():.1f}]")
    print(f"\n phi_j（每模式贡献，单位 nats/模式）：")
    print(f" {'a 箱':<14} | {'phi_3':>12} | {'phi_4(加权)':>12}")
    print(" " + "-" * 46)
    for j in range(Ka):
        print(f" [{A_EDGES[j]:.2f},{A_EDGES[j+1]:.2f}){'':<3} | "
              f"{fL3['coef'][j]:>12.3e} | {fL4['coef'][j]:>12.3e}")
    print(" 注意：G3 显示每个 alpha 下约 90% 的频率落在同一个 a 箱，")
    print("       所以这个直方图近似于'按 alpha 拟合'，不是关于核的定律。")
    print("       真正的手术台是 G5 的样本外 + 下面的新实验。")

    # ---------------- 样本外（含 NNLS 三条）----------------
    print("\n" + "=" * 104)
    print(" G5  样本外检验")
    print("=" * 104)

    def os_report(tag, build, npar):
        """build(train_mask) -> (Ftr, Fte)；返回 (弱->强, 留一核中位数)"""
        e1 = e2 = None
        tr = np.where(weak)[0]; te = np.where(~weak)[0]
        if len(te) > 2:
            Ftr, Fte = build(weak)
            sw = np.sqrt(wts[tr])
            coef, _ = nnls(Ftr * sw[:, None], y[tr] * sw)
            e1 = float(math.sqrt((((Fte @ coef) - y[te]) ** 2).mean()))
        errs = []
        for u in sorted(set(kk_keep)):
            m = np.array([k != u for k in kk_keep])
            if m.sum() < 8 or (~m).sum() < 3:
                continue
            Ftr, Fte = build(m)
            sw = np.sqrt(wts[m])
            coef, _ = nnls(Ftr * sw[:, None], y[m] * sw)
            errs.append(float(math.sqrt((((Fte @ coef) - y[~m]) ** 2).mean())))
        e2 = float(np.median(errs)) if errs else float("nan")
        print(f" {tag:<26} | {'--':>4} | {'--':>7} | {'--':>7} | "
              f"{(f'{e1:.3f}' if e1 is not None else '--'):>10} | {e2:>10.3f}")
        return e1, e2

    laws_nnls = [
        ("L3 phi(a) 直方图", lambda m: (F_a[m], F_a[~m])),
        ("L4 phi(a) 加权直方图", lambda m: (F_aw[m], F_aw[~m])),
        ("L5 phi(a,r) 二维", lambda m: (F_ar[m], F_ar[~m])),
    ]
    rows = law_table(laws, y, wts, weak, kk_keep, "os")
    for nm, build in laws_nnls:
        F, f = (F_a, fL3) if nm.startswith("L3") else \
               ((F_aw, fL4) if nm.startswith("L4") else (F_ar, fL5))
        e1, e2 = os_report(nm, build, F.shape[1])
        rows.append(dict(tag=nm, npar=F.shape[1], r2=f["r2"], rmse=f["rmse"],
                         maxhat=float(f["hat"].max()), os_weak=e1,
                         os_kernel=e2))
    print(" " + "-" * 82)
    print(" dex = log10 单位误差；0.01 dex ≈ 2.3%")

    # ---------------- G6：放大因子 ----------------
    print("\n" + "=" * 104)
    print(" G6  放大因子 A = Lambda / Lambda_Gauss")
    print("=" * 104)
    Aamp = lam / lamg_new
    print(f" {'kernel':<16} | " + " | ".join(f"a={a:g}" for a in ALPHAS))
    print(" " + "-" * 100)
    for kk in sorted(set(kern_key)):
        ii = sorted([i for i, k in enumerate(kern_key) if k == kk],
                    key=lambda i: alpha[i])
        print(f" {kk[0]+'/'+format(kk[1],'g'):<16} | " +
              " | ".join(f"{Aamp[i]:>7.1f}" for i in ii))
    print(f"\n A 的范围 {Aamp.min():.1f} - {Aamp.max():.1f}"
          f"（{Aamp.max()/Aamp.min():.1f}x）")
    la = np.log(Aamp)
    for nm, v in [("log U1", np.log(U1w)), ("log u", np.log(u_kunif)),
                  ("log U2/U1", np.log(U[2] / U[1])),
                  ("log U3/U2", np.log(U[3] / U[2]))]:
        b, a0 = np.polyfit(v, la, 1)
        print(f" log A ~ {nm:<10}: 斜率={b:+.3f}  R2={np.corrcoef(v,la)[0,1]**2:.3f}")

    # ---------------- 存盘 ----------------
    outfn = os.path.join(OUT, "profile_response.json")
    json.dump(dict(
        cfg=dict(nimg=nimg, t=t0, snr_min=args.snr, n_keep=int(keep.sum()),
                 a_edges=A_EDGES, r_edges=R_EDGES, data=args.data),
        gates=dict(kunif=float(np.abs(kunif_new-kunif_saved).max()
                               / max(np.abs(kunif_saved).max(), 1e-9)),
                   kappa=float(np.abs(kappa_new-kappa_saved).max()
                               / max(np.abs(kappa_saved).max(), 1e-9)),
                   lamgauss=float(np.abs(lamg_new-lam_g).max()
                                  / max(np.abs(lam_g).max(), 1e-9))),
        cap=cap, lam_gauss_max=float(lamg_new.max()),
        lam_max=float(lam.max()),
        phi_a=[float(v) for v in fL3["coef"]],
        phi_aw=[float(v) for v in fL4["coef"]],
        r2=dict(L3=fL3["r2"], L4=fL4["r2"], L5=fL5["r2"]),
        laws=rows,
        rows=[dict(family=r["family"], theta=r["theta"], alpha=r["alpha"],
                   lam=float(lam[i]), lam_gauss=float(lamg_new[i]),
                   amp=float(Aamp[i]), u_kunif=float(u_kunif[i]),
                   U1=float(U[1][i]), U2=float(U[2][i]), U3=float(U[3][i]),
                   U1w=float(U1w[i]), snr=float(snr[i]),
                   hist=[float(v) for v in F_a[j]])
              for j, (i, r) in enumerate((i, r) for i, r in enumerate(recs)
                                         if keep[i])],
    ), open(outfn, "w"), indent=1)
    print(f"\n[saved] {outfn}")


if __name__ == "__main__":
    main()
