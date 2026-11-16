"""sweep_pedestal.py 的离线判据：G0–G4（零 GPU）。

预官见 sweep_pedestal.py docstring（md5 `0a397a9c`，2026-10-23 08:07:12，早于测量）。

判据复用 predict_twod.py 的同一套机器（b(u) 普适曲线、共同 u 网格、mid(u)、tw_od 定律的
饱和拟合），保证新点与 28 个基线剖面**逐点同尺**。新算子的剖面由 meta 里的
(phi0, r1, m) 精确重建——这是确定性重构，并用重建的 kunif 与 sweep 落盘值对账（G0a）。

用法： venv2/bin/python judge_pedestal.py
"""

import importlib.util
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")
sys.path.insert(0, os.path.join(PROJ, "code", "blind_deblur_diffusion"))

_spec = importlib.util.spec_from_file_location(
    "ptod", os.path.join(PROJ, "repro", "predict_twod.py"))
ptod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ptod)
hd = ptod.hd

BASELINE = ["sweep_profile_shape.json", "sweep_law_test.json",
            "sweep_steepness.json", "sweep_b2test.json", "sweep_twlaw.json",
            "sweep_twod.json"]
TOL_INSAMPLE = 0.038          # tw_od 定律在 28 个训练剖面上的最大逐点残差（dex）
SNR_MIN = 3.0


# ---------------------------------------------------------------- 剖面重建
def halfplane():
    H = W = 64
    fx = np.fft.fftfreq(H) * H
    fy = np.fft.fftfreq(W) * W
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    R = np.sqrt(FX ** 2 + FY ** 2)[:, :W // 2 + 1]
    return R / R.max()


def pedestal(phi0, r1, m, r):
    s = np.zeros_like(r)
    msk = r >= r1
    s[msk] = ((r[msk] - r1) / max(1.0 - r1, 1e-12)) ** m
    return phi0 + (1.0 - phi0) * np.clip(s, 0.0, 1.0)


def load_baseline():
    prof, meta = defaultdict(list), {}
    for fn in BASELINE:
        p = os.path.join(OUT, fn)
        if not os.path.exists(p):
            continue
        for rec in json.load(open(p))["rows"]:
            m = rec["meta"]
            nm = m.get("fid") or (f"{m['family']}/{m['theta']:g}"
                                  if "family" in m else None)
            if nm is None:
                continue
            prof[nm].append((1.0 - rec["kunif"], rec["lam_mean"]))
            if nm not in meta:
                meta[nm] = (dict(kind=m["kind"], par=float(m["par"]),
                                 scale=float(m["scale"]), kref=float(m["kunif_ref"]))
                            if "kind" in m and m["kind"] != "real"
                            else dict(kind="real", fam=m["family"],
                                      th=float(m["theta"]), kref=float("nan")))
    for k in prof:
        prof[k].sort(key=lambda c: -c[0])
    return prof, meta


def main():
    r = halfplane()
    prof, meta = load_baseline()

    # ---- b(u) 普适曲线与 mid(u)（与 predict_twod.py / diag_k60.py 同法）----
    seg_u, seg_b = [], []
    for a, b in sorted(hd.P1_RANGES, reverse=True):
        bs = []
        for k in hd.SMOOTH_OLD:
            nm = f"{k[0]}/{k[1]:g}"
            va, vb = hd.interp(prof[nm], a), hd.interp(prof[nm], b)
            if va and vb and va > 0 and vb > 0:
                bs.append(math.log(vb / va) / math.log(b / a))
        if len(bs) >= 5:
            seg_u.append(math.sqrt(a * b))
            seg_b.append(float(np.mean(bs)))
    seg_u = np.array(seg_u)
    seg_b = np.array(seg_b)

    def b_of(u):
        if u <= seg_u[0]:
            return seg_b[0]
        if u >= seg_u[-1]:
            return seg_b[-1]
        lx = math.log(u)
        for i in range(len(seg_u) - 1):
            if seg_u[i] <= u <= seg_u[i + 1]:
                w = (lx - math.log(seg_u[i])) / \
                    (math.log(seg_u[i + 1]) - math.log(seg_u[i]))
                return seg_b[i] * (1 - w) + seg_b[i + 1] * w
        return seg_b[-1]

    Ucom = None
    for nm in prof:
        ss = {round(c[0], 4) for c in prof[nm]}
        Ucom = ss if Ucom is None else (Ucom & ss)
    Ucom = sorted(Ucom, reverse=True)
    Uall = sorted({round(c[0], 3) for nm in prof for c in prof[nm]}, reverse=True)
    C = {}
    for nm in prof:
        C[nm] = {}
        for u in Uall:
            v = hd.interp(prof[nm], u)
            if v and v > 0:
                C[nm][u] = v * u ** (-b_of(u))
    mid = {}
    for u in Uall:
        vs = [C[f"{k[0]}/{k[1]:g}"][u] for k in hd.SMOOTH_OLD
              if u in C.get(f"{k[0]}/{k[1]:g}", {})]
        if len(vs) >= 5:
            mid[u] = float(np.exp(np.mean([math.log(v) for v in vs])))
    U0 = 0.40
    assert U0 in mid, f"共同网格没有 u={U0}：{[u for u in Uall]}"
    assert U0 in [round(u, 3) for u in Ucom], f"u={U0} 不在共同网格 {Ucom}"
    b0, mid0 = b_of(U0), mid[U0]

    # ---- tw_od 定律：在全部基线剖面上重新拟合（不硬编码系数）----
    brows = []
    for nm in sorted(prof):
        m = meta[nm]
        a = (1.0 - hd.real_g2(m["fam"], m["th"])) if m["kind"] == "real" \
            else (1.0 - hd.make_profile(m["kind"], m["par"], m["scale"], r))
        a = a * (0.60 / float(a.mean()))
        d = hd.descriptors(a, r)
        ma = float(a.mean())
        dyn = (d["_amax"] - d["_amin"]) / ma
        if dyn <= 1e-9:
            continue
        vs = [C[nm][u] / mid[u] for u in Ucom if u in C[nm] and u in mid]
        if not vs:
            continue
        brows.append(dict(nm=nm, c=float(np.median(vs)),
                          twod=d["tw_lev"] / dyn, phi=d["_amin"] / ma,
                          amin=d["_amin"], amax=d["_amax"]))
    bx = np.array([t["twod"] for t in brows])
    by = np.array([t["c"] for t in brows])
    f, x0, p_ = ptod.fit_sat(bx, by)
    cf = f["coef"]

    def Claw(t):
        return cf[0] + cf[1] / (1.0 + (t / x0) ** (-p_))

    resid_b = np.log10(by / Claw(bx))
    tol = float(np.max(np.abs(resid_b)))

    # ---- 惯例匹配的第二条定律（披露：跑后新增，因为发现预官定律的 C 惯例与
    #      新点的单-u 值不一致，差 ~±2%，与效应量可比）----
    # 预官定律的 C 是「对 Ucom 取中位数」，而新算子每个只有一个 u=0.40。
    # 这里在**同一估计量**（u=0.40 的 C）上重新拟合全部基线剖面，容差取自它自己
    # 的残差分布。两条定律都报；若结论一致则无歧义，若不一致则明确记为惯例敏感。
    c40 = {}
    for nm in prof:
        v = hd.interp(prof[nm], U0)
        if v and v > 0:
            c40[nm] = v * U0 ** (-b0) / mid0
    b40 = [t for t in brows if t["nm"] in c40]
    x40 = np.array([t["twod"] for t in b40])
    y40 = np.array([c40[t["nm"]] for t in b40])
    f40, x040, p040 = ptod.fit_sat(x40, y40)
    cf40 = f40["coef"]

    def Claw40(t):
        return cf40[0] + cf40[1] / (1.0 + (t / x040) ** (-p040))

    resid40 = np.log10(y40 / Claw40(x40))
    tol40 = float(np.max(np.abs(resid40)))

    print("=" * 104)
    print(f"  pedestal 判据：{len(brows)} 个基线剖面；tw_od 定律 "
          f"C = {cf[0]:.4f} + {cf[1]:.4f}/(1+(twod/{x0:.4f})^(-{p_:.3f}))  R2={f['r2']:.3f}")
    print(f"  基线最大逐点残差 = {tol:.4f} dex  <=  G1 的容差（预官惯例：C 对 Ucom 取中位数）")
    print(f"  [惯例匹配] 同一批剖面在 u={U0} 的单点 C 上重拟合："
          f"C = {cf40[0]:.4f} + {cf40[1]:.4f}/(1+(twod/{x040:.4f})^(-{p040:.3f}))  R2={f40['r2']:.3f}")
    print(f"  [惯例匹配] 基线最大逐点残差 = {tol40:.4f} dex（n={len(b40)}）")
    print(f"  u = {U0}:  b(u) = {b0:.4f},  mid(u) = {mid0:.1f}")
    print("=" * 104)

    # ---- 新数据 ----
    d = json.load(open(os.path.join(OUT, "sweep_pedestal.json")))
    rows, cfg = d["rows"], d["cfg"]
    print(f"\n 载入 {len(rows)} 个新算子；nimg={cfg['nimg']} t={cfg['t']:g}")

    new = []
    for rec in rows:
        m = rec["meta"]
        a = pedestal(float(m["phi0"]), float(m["r1"]), float(m["par"]), r)
        ma = float(a.mean())
        dd = hd.descriptors(a, r)
        dyn = (dd["_amax"] - dd["_amin"]) / ma
        u = round(1.0 - rec["kunif"], 4)
        Cabs = rec["lam_mean"] * U0 ** (-b0)
        Crel = Cabs / mid0
        snr = rec["lam_mean"] * math.sqrt(cfg["nimg"]) / max(rec["lam_std"], 1e-9)
        cls = "有通带" if dd["_amin"] < 0.05 else \
              ("本类" if dd["_amax"] >= 0.99 else "窄带")
        new.append(dict(id=rec["id"], phi0=float(m["phi0"]), m=float(m["par"]),
                        r1=float(m["r1"]), kunif=rec["kunif"], u=u,
                        kunif_reco=ma, amin=dd["_amin"], amax=dd["_amax"],
                        twl=dd["tw_lev"], dyn=dyn,
                        twod=dd["tw_lev"] / dyn if dyn > 1e-9 else float("nan"),
                        phi=dd["_amin"] / ma, lam=rec["lam_mean"],
                        lam_se=rec["lam_std"] / math.sqrt(cfg["nimg"]),
                        snr=snr, Cabs=Cabs, Crel=Crel, cls=cls,
                        pred=Claw(dd["tw_lev"] / dyn) if dyn > 1e-9 else float("nan"),
                        lg=rec["lam_gauss"]))

    # ---------------- G0 守门 ----------------
    print("\n" + "=" * 104)
    print(" G0 守门")
    print("=" * 104)
    print(f" {'算子':<9} | {'kunif落盘':>9} {'kunif重建':>9} {'差':>9} | {'amin':>6} {'amax':>6} | "
          f"{'SNR':>6} | {'C_rel':>7} | {'C预言':>7} | {'残差dex':>8} | 类")
    print(" " + "-" * 118)
    ok_a = True
    keep = []
    for t in new:
        dk = abs(t["kunif_reco"] - t["kunif"])
        ok_a &= bool(dk <= 5e-3)
        inclass = bool(t["amin"] >= 0.05 and t["amax"] >= 0.99)
        # 分类本身不设通过/失败标准，只用于分组（G0b 只报计数）
        res = math.log10(t["Crel"] / t["pred"]) if t["pred"] > 0 else float("nan")
        t["resid"] = res
        t["inclass"] = inclass
        flag = "" if (inclass or t["cls"] == "有通带") else "  <= 非目标类"
        print(f" {t['id']:<9} | {t['kunif']:>9.4f} {t['kunif_reco']:>9.4f} {dk:>9.1e} | "
              f"{t['amin']:>6.3f} {t['amax']:>6.3f} | {t['snr']:>6.1f} | {t['Crel']:>7.3f} | "
              f"{t['pred']:>7.3f} | {res:>+8.4f} | {t['cls']}{flag}")
        keep.append(t)
    print(" " + "-" * 118)
    n_cls = sum(1 for t in keep if t["inclass"])
    n_pb = sum(1 for t in keep if t["cls"] == "有通带")
    n_low = [t["id"] for t in keep if t["snr"] < SNR_MIN]
    print(f" => G0(a) kunif 重建对账 {'✅' if ok_a else '❌'}（最大差 "
          f"{max(abs(t['kunif_reco']-t['kunif']) for t in keep):.1e}）")
    print(f" => G0(b) 分类：本类 {n_cls} 个，有通带对照 {n_pb} 个，"
          f"其他 {len(keep)-n_cls-n_pb} 个（若有则计数剔除）")
    print(f" => G0(c) SNR<{SNR_MIN:g} 的格子：{len(n_low)} 个 {n_low}"
          f"（{'✅ 无' if not n_low else '⚠️ 计数剔除'}）")

    cls = [t for t in keep if t["inclass"]]
    pb = [t for t in keep if t["cls"] == "有通带"]
    # G0(d) 桥接：phi0=0 的对照算子
    if pb:
        rp = np.array([t["resid"] for t in pb])
        okd = bool(np.max(np.abs(rp)) <= tol)
        print(f" => G0(d) 桥接（{len(pb)} 个有通带对照的 tw_od 定律残差）："
              f"max|残差| = {np.max(np.abs(rp)):.4f} dex  vs 容差 {tol:.4f}  "
              f"{'✅' if okd else '❌'}")
        print(f"     对照组自身在基线上的最大残差是 {tol:.4f} dex，所以这条检验的是"
              f"「新装置与旧数据同尺」，不是定律本身。")
    else:
        okd = False

    # ---------------- G1 最尖 ----------------
    print("\n" + "=" * 104)
    print(f" G1 最尖：本类新算子上 tw_od 定律的残差，预官 max|残差| <= {tol:.4f} dex")
    print("=" * 104)
    print(f" {'算子':<9} | {'tw_od':>7} | {'phi':>6} | {'C_rel':>7} | {'预官预言':>8} | "
          f"{'残差':>8} | {'匹配预言':>8} | {'残差':>8} | {'sig':>6}")
    print(" " + "-" * 104)
    rc, rc40 = [], []
    for t in sorted(cls, key=lambda t: t["twod"]):
        t["pred40"] = Claw40(t["twod"])
        t["resid40"] = math.log10(t["Crel"] / t["pred40"])
        se_rel = t["lam_se"] / max(abs(t["lam"]), 1e-9)
        n_sig = abs(t["resid40"]) / (se_rel / math.log(10))
        rc.append(t["resid"])
        rc40.append(t["resid40"])
        print(f" {t['id']:<9} | {t['twod']:>7.3f} | {t['phi']:>6.3f} | {t['Crel']:>7.3f} | "
              f"{t['pred']:>8.3f} | {t['resid']:>+8.4f} | {t['pred40']:>8.3f} | "
              f"{t['resid40']:>+8.4f} | {n_sig:>5.1f}")
    print(" " + "-" * 104)
    rc = np.array(rc)
    rc40 = np.array(rc40)
    ok1 = bool(np.max(np.abs(rc)) <= tol)
    ok1b = bool(np.max(np.abs(rc40)) <= tol40)
    allpos = bool((rc > 0).all())
    print(f" => G1（预官惯例）{'✅ 通过' if ok1 else '❌ 失败'}：max|残差| = "
          f"{np.max(np.abs(rc)):.4f} dex（容差 {tol:.4f}）；均值 {rc.mean():+.4f} dex；"
          f"{'**全部为正**' if allpos else '符号混合'}")
    print(f" => G1（惯例匹配）{'✅ 通过' if ok1b else '❌ 失败'}：max|残差| = "
          f"{np.max(np.abs(rc40)):.4f} dex（容差 {tol40:.4f}）；均值 {rc40.mean():+.4f} dex；"
          f"全部同号 = {bool((rc40>0).all()) or bool((rc40<0).all())}")
    print(f"     两种惯例{'给出一致结论' if ok1==ok1b else '**结论不一致，须记为惯例敏感**'}")
    print(f"     旧 3 个类成员的残差是 -0.012 / -0.034 / +0.027 dex（均值 -0.006）。")
    if allpos and rc.mean() > 0.025:
        print(f"     **预官的失败方式发生**：残差系统性为正且均值 > +0.025 dex"
              f"（2.5x 单点噪声）=> 本类有 C 的**地板**，是独立 regime。")
    lo = min(cls, key=lambda t: t["twod"])
    print(f"     最低 tw_od = {lo['twod']:.3f}（{lo['id']}）：定律预言 {lo['pred']:.3f}，"
          f"实测 {lo['Crel']:.3f}。若存在 C=0.900 的地板，残差应为 "
          f"+{0.900-lo['pred']:.3f} dex。")

    # ---------------- G2 跨类配对 ----------------
    print("\n" + "=" * 104)
    print(" G2 判别：同类 tw_od（相差 <=15%）的 (本类, 有通带) 配对")
    print("=" * 104)
    print(f" {'本类':<9} | {'tw_od':>7} | {'C_rel':>7} | {'对照':<9} | {'tw_od':>7} | "
          f"{'C_rel':>7} | {'比值':>7} | {'定律预言比':>10}")
    print(" " + "-" * 96)
    pairs = []
    for a in cls:
        for b in pb:
            if abs(a["twod"] / b["twod"] - 1) <= 0.15:
                ratio = a["Crel"] / b["Crel"]
                pr = a["pred"] / b["pred"]
                pairs.append((a, b, ratio, pr))
                print(f" {a['id']:<9} | {a['twod']:>7.3f} | {a['Crel']:>7.3f} | "
                      f"{b['id']:<9} | {b['twod']:>7.3f} | {b['Crel']:>7.3f} | "
                      f"{ratio:>7.3f} | {pr:>10.3f}")
    print(" " + "-" * 96)
    if pairs:
        dv = np.array([math.log10(p[2] / p[3]) for p in pairs])
        print(f" => 实测比/定律比 的偏离：{' '.join(f'{v:+.4f}' for v in dv)} dex，"
              f"均值 {dv.mean():+.4f}")
        ok2 = bool(abs(dv.mean()) <= 0.025)
        print(f" => G2 {'✅ 通过' if ok2 else '❌ 失败'}（预官：|均值| <= 0.025 dex；"
              f"同时注意预官间距只有 ~0.017 dex ≈ 1.7x 单点噪声，"
              f"若实际间距 <2sigma 按规矩降级为只记录）")
    else:
        ok2 = None
        print(" => 无配对，G2 不可判定")

    # ---------------- G3 类内的 tw_od 效应 ----------------
    print("\n" + "=" * 104)
    print(" G3 判别：本类新点内部，C 是否随 tw_od 变")
    print("=" * 104)
    tx = np.array([t["twod"] for t in cls])
    ty = np.array([t["Crel"] for t in cls])
    if len(cls) >= 4:
        sp = float(np.corrcoef(np.argsort(np.argsort(tx)),
                               np.argsort(np.argsort(ty)))[0, 1])
        ratio = float(ty[np.argmax(tx)] / ty[np.argmin(tx)])
        ok3 = bool(sp >= 0.9 and ratio >= 1.06)
        print(f" Spearman(tw_od, C) = {sp:+.3f}（预官 >= 0.9）")
        print(f" C(max tw_od)/C(min tw_od) = {ratio:.3f}x（预官 >= 1.06x；定律预言 1.165x）")
        print(f" => G3 {'✅ 通过' if ok3 else '❌ 失败'}")
    else:
        sp, ratio, ok3 = float("nan"), float("nan"), None
        print(" 点数不足，G3 不可判定")
    byp = defaultdict(list)
    for t in cls:
        byp[round(t["phi"], 3)].append(t)
    print("\n 固定 phi 的类内配对（预官设计的解耦对）：")
    for k, v in sorted(byp.items()):
        if len(v) >= 2:
            v = sorted(v, key=lambda t: t["twod"])
            print(f"  phi={k}: " + "  ".join(
                f"{t['id']}(tw_od={t['twod']:.3f}, C={t['Crel']:.3f}, 预言={t['pred']:.3f})"
                for t in v) +
                f"   C 比 {v[-1]['Crel']/v[0]['Crel']:.3f}x，定律预言比 "
                f"{v[-1]['pred']/v[0]['pred']:.3f}x")
    ph = np.array([t["phi"] for t in cls])
    print(f"\n 类内 corr(phi, tw_od) = {np.corrcoef(ph, tx)[0, 1]:+.3f}"
          f"（离线设计时算得 -0.925；越接近 0 越能解耦）")

    # ---------------- G4 诊断 ----------------
    print("\n" + "=" * 104)
    print(" G4 诊断（无阈值）")
    print("=" * 104)
    print(f" (a) 几何约束（预官）：单体单调边 a=phi0+(1-phi0)E(r)、E 从 0 到 1，")
    for p_ in (0.0, 0.2, 0.4):
        print(f"     phi0={p_:.1f} 时线性边的最低均值 = {0.75+0.25*p_:.4f} > 0.60 "
              f"=> 不可能同时 amin>0 且 amax=1")
    print(f" (b) 本类新点的 tw_od 覆盖 {tx.min():.3f}-{tx.max():.3f}，"
          f"phi 覆盖 {ph.min():.3f}-{ph.max():.3f}")
    print(f" (c) 本类新点把 tw_od 从旧最低 0.599 向下延伸 {0.599/tx.min():.2f}x；"
          f"旧 3 个成员在 0.599/1.934/6.282，新点在 {tx.min():.3f}-{tx.max():.3f}")
    print(f" (d) Lambda_Gauss 的结构核对：实测 L_G / (cap*kunif) 应接近 1")
    cap = cfg.get("cap", float("nan"))
    if np.isfinite(cap):
        for t in new[:3]:
            print(f"     {t['id']}: L_G={t['lg']:.1f}, cap*kunif={cap*t['kunif']:.1f}, "
                  f"比 {t['lg']/(cap*t['kunif']):.4f}")
    print(f" (e) 单点 C 的相对噪声（由 Lambda 的 SE 传播）中位 = "
          f"{np.median([t['lam_se']/max(abs(t['lam']),1e-9) for t in cls])/math.log(10):.4f} dex"
          f" => 可探测的 C 极差约 1.0{np.median([t['lam_se']/max(abs(t['lam']),1e-9) for t in cls])/math.log(10)*100:.0f}%")

    with open(os.path.join(OUT, "fig_pedestal.json"), "w") as fh:
        json.dump(dict(cfg=cfg, law=dict(coef=[float(c) for c in cf], x0=float(x0),
                                         p=float(p_), r2=float(f["r2"]), tol=tol),
                       b0=b0, mid0=mid0,
                       new=[{k: v for k, v in t.items()} for t in keep],
                       g0=dict(kunif_ok=bool(ok_a), n_cls=n_cls, n_pb=n_pb,
                               low_snr=n_low, bridge_ok=okd),
                       g1=dict(pass_=bool(ok1), max_res=float(np.max(np.abs(rc))),
                               mean_res=float(rc.mean()), all_pos=allpos,
                               pass_matched=bool(ok1b), tol_matched=tol40,
                               max_res_matched=float(np.max(np.abs(rc40))),
                               mean_res_matched=float(rc40.mean()),
                               law_matched=dict(coef=[float(c) for c in cf40],
                                                x0=float(x040), p=float(p040))),
                       g2=dict(pass_=ok2, n_pairs=len(pairs),
                               dev=[float(v) for v in dv] if pairs else []),
                       g3=dict(spearman=sp, ratio=ratio, pass_=ok3),
                       ), fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'fig_pedestal.json')}")


if __name__ == "__main__":
    main()
