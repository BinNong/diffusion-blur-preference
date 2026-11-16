"""机理判别：Lambda 能由哪个谱泛函解释？

已知（FFHQ + AFHQv2，5 族 x 6 强度）：
  * kappa = sum_f (1-|h|^2) S_f / sum_f S_f  （被杀掉的先验能量占比）
  * 同 kappa 下各族 Lambda 相差最多 2.2 倍 -> 不是纯谱效应

问题：是否存在别的谱泛函能解释 Lambda？候选：
  kappa1  = sum (1-|h|^2) S / sum S                      （能量被杀占比）
  kappa2  = sum (1-|h|^2)^2 S / sum S                    （二阶，重罚部分衰减）
  kappa4  = sum (1-|h|^2)^4 S / sum S
  kinf    = max_f (1-|h|^2)                              （最坏衰减）
  khard   = sum 1[|h|^2<0.1] S / sum S                   （被"关掉"的频段能量）
  klog    = sum (1-|h|^2) S log(1/S) / sum S log(1/S)    （按先验集中度加权）
  kaniso  = 各向异性度量：同一半径环上 |h|^2 的变异系数均值

对每个候选泛函 f，用全部 (族, theta) 样本做
    log Lambda = a + b log f
的线性回归，报告 R^2。R^2 接近 1 => Lambda 由该泛函决定（谱的故事）；
所有 R^2 都低 => Lambda 有真正非谱的依赖（核形状/流形几何）。

用法： python dump_spectral.py [sweep_lambda_theta_ffhq.json ...]
"""

import json
import os
import sys

import numpy as np
import torch

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "code", "blind_deblur_diffusion"))

from putils.convolution import GaussianBlur, DefocusBlur, AiryDisc, MotionBlur

DEV = torch.device("cuda")
DT = torch.float32
H = W = 64
NB = 48
KS = 15
POOL = "/tmp/pool_ffhq64_256_s0.pt"
OUT = os.path.join(_PROJ, "results")

FAMILIES = {
    "gaussian": (GaussianBlur, "sigma", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]),
    "defocus": (DefocusBlur, "radius", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]),
    "airy": (AiryDisc, "radius", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]),
    "motion": (MotionBlur, "length", [1, 3, 5, 7, 9, 13]),
}


def radial_bins(nbin=NB):
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)
    Rr = R[:, :W // 2 + 1]
    edges = np.linspace(0, Rr.max() + 1e-6, nbin + 1)
    idx = np.clip(np.digitize(Rr.ravel(), edges) - 1, 0, nbin - 1)
    return idx.reshape(Rr.shape), nbin, Rr


def khat(kernel):
    if kernel.ndim == 2:
        kernel = kernel[None, None]
    kh, kw = kernel.shape[-2:]
    pad = torch.zeros((1, 1, H, W), device=DEV, dtype=torch.float64)
    t, l = (H - kh) // 2, (W - kw) // 2
    pad[:, :, t:t + kh, l:l + kw] = kernel.to(torch.float64)
    return torch.fft.rfft2(pad, dim=(-2, -1)).abs() ** 2      # (1,1,H,W//2+1)


def main():
    files = sys.argv[1:] or ["sweep_lambda_theta_ffhq.json"]
    idx, _, Rr = radial_bins()
    idx_t = torch.from_numpy(idx).to(DEV)

    pool = torch.load(POOL, map_location=DEV).to(torch.float64)
    with torch.no_grad():
        P = (torch.fft.rfft2(pool, dim=(-2, -1)).abs() ** 2) / (H * W)
        Pc = P.mean(1)
        Srad = torch.zeros(NB, device=DEV, dtype=torch.float64)
        for b in range(NB):
            m = (idx_t == b).reshape(-1)
            if m.any():
                Srad[b] = Pc.reshape(P.shape[0], -1)[:, m].mean()
    S = Srad[idx_t].cpu().numpy()                           # (H, W//2+1)
    Stot = S.sum()

    # 各向异性度量：按半径分箱，桶内 |h|^2 的变异系数
    rbin = np.clip(np.digitize(Rr.ravel(),
                               np.linspace(0, Rr.max() + 1e-6, 25)) - 1, 0, 23)
    rbin = rbin.reshape(Rr.shape)

    # 低通硬门：本地实现（作者 putils 里没有），作为最强的界外检验
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    RR = np.sqrt(FX ** 2 + FY ** 2)
    LOWPASS = [0.50, 0.35, 0.25, 0.18, 0.12, 0.08]

    def lowpass_h2(cut):
        gate = (RR <= cut * RR.max()).astype(np.float64)
        return gate[:, :W // 2 + 1]                      # |h|^2 in {0,1}

    rows = []
    for si, fn in enumerate(files):
        tag = os.path.basename(fn).replace("sweep_lambda_theta_", "")[:10]
        # 先低通族（界外检验）
        for cut in LOWPASS:
            h2 = lowpass_h2(cut)
            d = 1.0 - h2
            rows.append(dict(src=si, tag=tag, family="lowpass", theta=float(cut),
                             kappa1=float((d * S).sum() / Stot),
                             kappa2=float(((d ** 2) * S).sum() / Stot),
                             kappa4=float(((d ** 4) * S).sum() / Stot),
                             kinf=float(d.max()),
                             khard=float(((d > 0.9) * S).sum() / Stot),
                             kunif=float(d.mean()),
                             kprec=float((d / (S + 1e-12)).sum()
                                         / (1.0 / (S + 1e-12)).sum()),
                             kaniso=0.0))
        for fam, (cls, pname, levels) in FAMILIES.items():
            kf = cls(kernel_size=KS, device=DEV, dtype=torch.float64)
            for th in levels:
                k = kf.kernel(th)
                h2 = khat(k)[0, 0].cpu().numpy()
                d = 1.0 - h2
                # 各向异性：每个半径桶内 d 的 std/mean
                cv = []
                for b in range(24):
                    m = (rbin == b)
                    if m.sum() > 3 and d[m].mean() > 1e-3:
                        cv.append(d[m].std() / d[m].mean())
                kaniso = float(np.mean(cv)) if cv else 0.0
                rows.append(dict(src=si, tag=tag, family=fam, theta=float(th),
                                 kappa1=float((d * S).sum() / Stot),
                                 kappa2=float(((d ** 2) * S).sum() / Stot),
                                 kappa4=float(((d ** 4) * S).sum() / Stot),
                                 kinf=float(d.max()),
                                 khard=float(((d > 0.9) * S).sum() / Stot),
                                 kunif=float(d.mean()),
                                 kprec=float((d / (S + 1e-12)).sum()
                                             / (1.0 / (S + 1e-12)).sum()),
                                 kaniso=kaniso))

    # 汇入 Lambda
    recs = []
    for si, fn in enumerate(files):
        for r in json.load(open(os.path.join(OUT, fn)))["records"]:
            r["_src"] = si
            recs.append(r)
    tmin = min(r["t"] for r in recs)
    print(f"  用 t={tmin:g} 的记录（共 {len(recs)} 条，"
          f"{sum(1 for r in recs if r['t'] == tmin)} 条对应该 t）")
    by = {(r["_src"], r["family"], round(r["theta"], 4)): r for r in recs
          if r["t"] == tmin}
    X, Y, lab = [], [], []
    for row in rows:
        r = by.get((row["src"], row["family"], row["theta"]))
        if r is None:
            continue
        row["lam"] = r["lam_mean"]
        row["lam_gauss"] = r["lam_gauss_mean"]
        row["lam_per_img"] = r.get("lam_per_img", [])
        X.append(row)
        Y.append(row["lam"])
        lab.append(f"{row['family']}:{row['theta']:g}")

    Y = np.array(Y)
    print("=" * 100)
    print(" 机理判别：log Lambda 对 log(谱泛函) 的回归（全部 族 x theta 样本）")
    print("=" * 100)
    print(f"{'泛函':>10} | {'R^2':>7} | {'斜率':>7} | {'残差 std':>10} |  含义")
    print("-" * 100)
    keys = ["kappa1", "kappa2", "kappa4", "kinf", "khard", "kunif",
            "kprec", "kaniso"]
    meaning = {
        "kappa1": "被杀掉的先验能量占比",
        "kappa2": "同上，平方（重罚部分衰减）",
        "kappa4": "同上，四次方",
        "kinf": "最坏频率处的衰减",
        "khard": "被完全关掉(|h|^2<0.1)的频段能量",
        "kunif": "平均衰减（不加权，按模式数）",
        "kprec": "按先验精度 1/S 加权的平均衰减",
        "kaniso": "各向异性：环上衰减的变异系数",
    }
    print(f"  样本数 {len(X)}，其中 Lambda>0 的 {int((Y > 0).sum())} 个"
          f"（Airy 小半径处 Lambda<0，取对数前剔除）\n")
    for key in keys:
        v = np.array([r[key] for r in X], dtype=float)
        ok = np.isfinite(v) & (v > 1e-9) & (Y > 0)
        if ok.sum() < 5 or np.ptp(v[ok]) < 1e-12:
            print(f"{key:>10} | {'跳过（取值近乎恒定或样本不足）':>40}")
            continue
        A = np.vstack([np.ones(ok.sum()), np.log(v[ok])]).T
        coef, res, *_ = np.linalg.lstsq(A, np.log(Y[ok]), rcond=None)
        pred = A @ coef
        ss = 1 - ((np.log(Y[ok]) - pred) ** 2).sum() / \
            ((np.log(Y[ok]) - np.log(Y[ok]).mean()) ** 2).sum()
        print(f"{key:>10} | {ss:>7.3f} | {coef[1]:>7.2f} | "
              f"{float(np.std(np.log(Y[ok]) - pred)):>10.3f} |  {meaning[key]}")
    print("-" * 100)
    print(" R^2 高 => Lambda 由该谱泛函决定；全部低 => 有真正非谱的依赖")

    # ---- 幂律的稳健性：留一族法 + 去掉 Gaussian ----
    print("\n" + "=" * 100)
    print(" 幂律 log Lambda = a + b log(kunif) 的稳健性")
    print("=" * 100)
    print(f"{'子集':<28} | {'n':>3} | {'b':>6} | {'R^2':>7} | {'残差 std':>9}")
    print("-" * 100)

    def fit(sub, key="kunif"):
        k = np.array([r[key] for r in sub], dtype=float)
        y = np.log(np.array([r["lam"] for r in sub], dtype=float))
        ok = np.isfinite(k) & (k > 1e-9)
        if ok.sum() < 4:
            return None
        A = np.vstack([np.ones(ok.sum()), np.log(k[ok])]).T
        coef, *_ = np.linalg.lstsq(A, y[ok], rcond=None)
        pred = A @ coef
        r2 = 1 - ((y[ok] - pred) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
        return int(ok.sum()), coef[1], r2, float(np.std(y[ok] - pred))

    pos = [r for r in X if r["lam"] > 0 and r["kunif"] > 1e-9]
    bexp = {}
    bse = {}
    ALLFAM = list(FAMILIES) + ["lowpass"]
    TAGS = sorted({r["tag"] for r in pos})
    subsets = [("全部", pos),
               ("仅拟合用过的 4 族", [r for r in pos if r["family"] != "lowpass"]),
               ("仅 lowpass（界外）", [r for r in pos if r["family"] == "lowpass"])] + [
        (f"去掉 {f}", [r for r in pos if r["family"] != f]) for f in ALLFAM] + [
        (f"仅 {f}", [r for r in pos if r["family"] == f]) for f in ALLFAM] + [
        (f"仅 {t}/{f}", [r for r in pos if r["tag"] == t and r["family"] == f])
        for t in TAGS for f in ALLFAM]
    for name, sub in subsets:
        out = fit(sub)
        if out is None:
            continue
        n, b, r2, rs = out
        print(f"{name:<28} | {n:>3} | {b:>6.2f} | {r2:>7.3f} | {rs:>9.3f}")
        if name.startswith("仅 "):
            bexp[name[2:].strip()] = b
            # 同时给观察族算 bootstrap SE（对 20/12 张图重采样）
    print("-" * 100)
    print(" 去掉 Gaussian 后 R^2 仍高 => 幂律不是 Gaussian 族自己的曲线")

    # ---- 观察族指数的 bootstrap 标准误（对图重采样）----
    def boot_se(sub):
        """按 lam_per_img 长度分组（不同模型样本数不同），各组分别 bootstrap 后取均值。"""
        groups = {}
        for r in sub:
            groups.setdefault(len(r["lam_per_img"]), []).append(r)
        ses = []
        for g in groups.values():
            if len(g) < 3:
                continue
            per = np.array([r["lam_per_img"] for r in g])
            kk = np.array([r["kunif"] for r in g])
            ni = per.shape[1]
            rng2 = np.random.default_rng(1)
            out = []
            for _ in range(200):
                m = per[:, rng2.integers(0, ni, ni)].mean(axis=1)
                ok = m > 0
                if ok.sum() >= 3 and np.ptp(kk[ok]) > 1e-12:
                    A = np.vstack([np.ones(ok.sum()), np.log(kk[ok])]).T
                    c, *_ = np.linalg.lstsq(A, np.log(m[ok]), rcond=None)
                    out.append(c[1])
            if len(out) > 10:
                ses.append(float(np.std(out)))
        return float(np.mean(ses)) if ses else float("nan")

    for k in list(bexp):
        if "/" in k:
            tg, fm = k.split("/", 1)
            sub2 = [r for r in pos if r["tag"] == tg and r["family"] == fm]
        else:
            sub2 = [r for r in pos if r["family"] == k]
        if len(sub2) >= 3 and all("lam_per_img" in r for r in sub2):
            bse[k] = boot_se(sub2)

    # ---- 族内回归：log Lambda = a_f + b_f log kappa1，看族间偏移 ----
    print("\n" + "=" * 100)
    print(" 族内回归 log Lambda = a_f + b_f log(kappa1)，族间偏移 a_f 才是非谱成分")
    print("=" * 100)
    print(f"{'族':>10} | {'n':>3} | {'b_f':>6} | {'a_f':>7} | {'族内 R^2':>9} | "
          f"{'残差 std':>9}")
    print("-" * 100)
    offs = {}
    for (tag, fam) in sorted({(r["tag"], r["family"]) for r in X}):
        rr = [r for r in X if r["tag"] == tag and r["family"] == fam
              and r["lam"] > 0 and r["kappa1"] > 1e-6]
        if len(rr) < 3:
            continue
        k = np.array([r["kappa1"] for r in rr])
        y = np.log(np.array([r["lam"] for r in rr]))
        A = np.vstack([np.ones(len(k)), np.log(k)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        offs[f"{tag}/{fam}"] = coef[0]
        print(f"{tag+'/'+fam:>18} | {len(rr):>3} | {coef[1]:>6.2f} | "
              f"{coef[0]:>7.2f} | "
              f"{r2:>9.3f} | {float(np.std(y - pred)):>9.3f}")
    print("-" * 100)
    if offs:
        base = np.mean(list(offs.values()))
        print("  相对均值的偏移（对数，即倍数）：")
        for f, a in sorted(offs.items(), key=lambda kv: -kv[1]):
            print(f"    {f:>22}: {a - base:+.2f}  ({np.exp(a - base):.2f}x)")
    print("  偏移若明显大于族内残差 -> Lambda 有真正非谱的族依赖")

    # ---- 指数 b_f 与核描述量的相关（5 个族，n 很小，只看方向）----
    print("\n" + "=" * 100)
    print(" 族指数 b_f 与核描述量的关系（n=5，只看方向，不作统计推断）")
    print("=" * 100)
    desc = {}
    for fam, (cls, pname, levels) in FAMILIES.items():
        kf = cls(kernel_size=KS, device=DEV, dtype=torch.float64)
        bs = []
        for th in levels:
            h2 = khat(kf.kernel(th))[0, 0].cpu().numpy().ravel()
            d = 1.0 - h2
            # 半衰减频率：d = 0.5 处的径向频率
            rho = Rr.ravel()
            half = np.interp(-0.5, -d[np.argsort(rho)], rho[np.argsort(rho)]) \
                if d.max() > 0.5 else 0.0
            bs.append((th, d, h2, half))
        desc[fam] = bs
    # 低通族
    lp = []
    for cut in LOWPASS:
        h2 = lowpass_h2(cut).ravel()
        lp.append((cut, 1.0 - h2, h2, cut * Rr.max()))
    desc["lowpass"] = lp

    rho_flat = Rr.ravel()
    idx_np = idx

    def cross_first(prof, rho, level):
        """prof 随 rho 递减，返回首次穿越 level 的半径（相邻点线性插值）。
        np.interp 要求 x 坐标递增，剖面递减时结果无定义，故不能用。"""
        for i in range(len(prof) - 1):
            a, b = prof[i] - level, prof[i + 1] - level
            if a * b <= 0 and a != b:
                return rho[i] + (level - prof[i]) * (rho[i + 1] - rho[i]) / \
                    (prof[i + 1] - prof[i])
        return float("nan")

    def width_of(prof, rho):
        r9 = cross_first(prof, rho, 0.1)     # |h|^2 = 0.1
        r1 = cross_first(prof, rho, 0.9)     # |h|^2 = 0.9
        if not (np.isfinite(r9) and np.isfinite(r1)):
            return float("nan")
        return float(r9 - r1)

    def fam_desc(fam):
        """核描述量：谱过渡带宽度 rho(0.9)-rho(0.1)、|h|^2 零点比例、
        核像素支撑、theta 扫描覆盖的 kunif 动态范围。

        宽度给两个版本：
          width_raw  —— 直接在频率上按首次穿越定义（振荡核不成立）
          width_smooth —— 先把 |h|^2 做径向分桶平均（抹掉振荡），再取穿越宽度
        """
        dd = desc[fam]
        widths, wsm, zeros, supps, kranges = [], [], [], [], []
        for th, d, h2, half in dd:
            # 逐频率（按半径排序后取首次穿越）
            o = np.argsort(rho_flat)
            w = width_of(h2[o], rho_flat[o])
            if np.isfinite(w):
                widths.append(w)
            # 径向分桶平均后的 |h|^2（48 桶，抹掉振荡），再取穿越宽度
            hb = np.zeros(NB)
            rb = np.zeros(NB)
            for b in range(NB):
                mv = (idx_np == b).ravel()
                if mv.any():
                    hb[b] = h2[mv].mean()
                    rb[b] = rho_flat[mv].mean()
            ws = width_of(hb, rb)
            if np.isfinite(ws):
                wsm.append(ws)
            zeros.append(float((h2 < 1e-6).mean()))
            kranges.append(float(d.mean()))
        if fam != "lowpass":
            cls = FAMILIES[fam][0]
            kf = cls(kernel_size=KS, device=DEV, dtype=torch.float64)
            kk = kf.kernel(FAMILIES[fam][2][-1]).cpu().numpy().ravel()
            supps.append(float((np.abs(kk) > 1e-4).sum()))
        return dict(width=float(np.mean(widths)) if widths else np.nan,
                    width_smooth=float(np.mean(wsm)) if wsm else np.nan,
                    zeros=float(np.mean(zeros)),
                    support=float(supps[0]) if supps else np.nan,
                    kunif_range=float(max(kranges) - min(kranges)))

    print(f"{'模型/族':>26} | {'kunif 指数 b':>12} | {'boot SE':>8} | "
          f"{'宽度(raw)':>10} | {'宽度(smooth)':>12} | {'零点比例':>8} | "
          f"{'kunif 范围':>11}")
    print("-" * 100)
    tab = []
    for key in sorted(bexp, key=lambda k: -bexp[k]):
        fam = key.split("/")[-1]
        if fam not in desc:
            continue
        dd = fam_desc(fam)
        tab.append((key, bexp[key], dd, fam))
        print(f"{key:>26} | {bexp[key]:>12.2f} | {bse.get(key, float('nan')):>8.2f} | "
              f"{dd['width']:>10.3f} | {dd['width_smooth']:>12.3f} | "
              f"{dd['zeros']:>8.3f} | {dd['kunif_range']:>11.3f}")
    print("-" * 100)
    for key, name in [("width", "过渡带宽度(raw)"), ("width_smooth", "过渡带宽度(smooth)"),
                      ("zeros", "零点比例"), ("support", "像素支撑"),
                      ("kunif_range", "kunif 范围")]:
        v = np.array([t[2][key] for t in tab], dtype=float)
        b = np.array([t[1] for t in tab], dtype=float)
        ok = np.isfinite(v) & np.isfinite(b)
        if ok.sum() >= 3 and np.ptp(v[ok]) > 1e-12:
            c = float(np.corrcoef(v[ok], b[ok])[0, 1])
            print(f"  corr(指数 b, {name:<12}) = {c:+.2f}   (n={int(ok.sum())})")
    # 同一族跨模型：指数是否稳定（= 核的性质）还是随模型变（= 先验的性质）
    print("\n  同一族跨模型的指数（回答：指数是核的性质还是先验的性质）")
    for fam in ALLFAM:
        vs = [(k.split("/")[0], bexp[k]) for k in bexp if k.endswith("/" + fam)]
        if len(vs) < 2:
            continue
        arr = np.array([v for _, v in vs])
        print(f"    {fam:>9}: " + "  ".join(f"{t.split('-')[0]}={v:.2f}"
                                            for t, v in vs)
              + f"   | 均值 {arr.mean():.2f}, 极差 {arr.max()-arr.min():.2f}"
              + f", 相对离散 {(arr.std()/arr.mean()):.2f}")
    # 核层面（每核取跨模型均值）的相关 —— 有效 n = 核的个数，不是行数
    print("\n  核层面相关（每核取跨模型均值，有效 n = 核数，避免重复计量）")
    kw, kb = [], []
    for fam in ALLFAM:
        vs = [bexp[k] for k in bexp if k.endswith("/" + fam)]
        if len(vs) < 2:
            continue
        dd = fam_desc(fam)
        kw.append(dd["width_smooth"])
        kb.append(float(np.mean(vs)))
    if len(kw) >= 3:
        for key, name, vv in [("width_smooth", "过渡带宽度(平滑)", kw)]:
            c = float(np.corrcoef(vv, kb)[0, 1])
            print(f"    corr(指数, {name}) = {c:+.2f}  (n={len(kw)} 个核)")
            for fam, w, b in zip([f for f in ALLFAM
                                  if len([k for k in bexp
                                          if k.endswith('/' + f)]) >= 2], kw, kb):
                print(f"      {fam:>9}: 宽度 {w:8.3f}  指数 {b:.2f}")

    # 逐族看：同 kappa1 下 Lambda 的离散（控制谱之后的残余）
    print("\n 各族在 kappa1 相近处的 Lambda（谱控制后的残余差异）")
    for fam in FAMILIES:
        rr = [r for r in X if r["family"] == fam]
        if not rr:
            continue
        rr.sort(key=lambda r: r["kappa1"])
        print(f"  [{fam}]")
        for r in rr:
            print(f"    theta={r['theta']:<5g} kappa1={r['kappa1']:.4f} "
                  f"lam={r['lam']:>9.0f}  kappa2={r['kappa2']:.4f} "
                  f"kinf={r['kinf']:.3f} khard={r['khard']:.4f} "
                  f"kaniso={r['kaniso']:.3f}")

    with open(os.path.join(OUT, "spectral_functionals.json"), "w") as f:
        json.dump(dict(S_note="radial-binned prior spectrum, 256 FFHQ samples",
                       rows=X), f, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'spectral_functionals.json')}")


if __name__ == "__main__":
    main()
