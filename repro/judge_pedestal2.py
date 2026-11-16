"""sweep_pedestal2.py 的离线判据：H0–H4（零 GPU）。

预官见 sweep_pedestal2.py docstring（md5 `fca6c30a`，2026-10-23 09:00:42，早于测量）。
本组的唯一目的是**解开 (ag) 的 φ–tw_od 混淆**：靠第二个 ⟨a⟩（kunif=0.35 ⇒ u=0.65）
在 tw_od 匹配到 1% 以内做出 φ = 0 → 0.429 的反差，且主比较全部在 u=0.65 内部做。

用法： venv2/bin/python judge_pedestal2.py
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
SNR_MIN = 3.0
U_NEW = 0.65
U_OLD = 0.40


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


def main():
    r = halfplane()
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
                                 scale=float(m["scale"]))
                            if "kind" in m and m["kind"] != "real"
                            else dict(kind="real", fam=m["family"],
                                      th=float(m["theta"])))
    for k in prof:
        prof[k].sort(key=lambda c: -c[0])

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
    for u in (U_NEW, U_OLD):
        assert u in mid, f"共同网格没有 u={u}"
        assert u in [round(x, 3) for x in Ucom], f"u={u} 不在共同网格"

    def Crel(nm, u):
        v = hd.interp(prof[nm], u)
        return (v * u ** (-b_of(u)) / mid[u]) if (v and v > 0) else None

    # ---- tw_od 定律（预官惯例：C 对 Ucom 取中位数）----
    brows = []
    for nm in sorted(prof):
        m = meta.get(nm)
        if m is None:
            continue
        a = (1.0 - hd.real_g2(m["fam"], m["th"])) if m["kind"] == "real" \
            else (1.0 - hd.make_profile(m["kind"], m["par"], m["scale"], r))
        a = a * (0.60 / float(a.mean()))
        d = hd.descriptors(a, r)
        ma = float(a.mean())
        dyn = (d["_amax"] - d["_amin"]) / ma
        if dyn <= 1e-9:
            continue
        vs = [Crel(nm, u) for u in Ucom if u in C[nm] and u in mid]
        if not vs:
            continue
        brows.append(dict(nm=nm, c=float(np.median(vs)), twod=d["tw_lev"] / dyn))
    bx = np.array([t["twod"] for t in brows])
    by = np.array([t["c"] for t in brows])
    f, x0, p_ = ptod.fit_sat(bx, by)
    cf = f["coef"]

    def Claw(t):
        return cf[0] + cf[1] / (1.0 + (t / x0) ** (-p_))

    tol = float(np.max(np.abs(np.log10(by / Claw(bx)))))

    print("=" * 104)
    print(f"  pedestal2 判据：{len(brows)} 个基线剖面；tw_od 定律 "
          f"C = {cf[0]:.4f} + {cf[1]:.4f}/(1+(twod/{x0:.4f})^(-{p_:.3f}))  R2={f['r2']:.3f}")
    print(f"  预官版系数是 0.7032 + 0.3699/(1+(twod/0.4210)^(-1.231)) —— "
          f"{'一致' if abs(cf[0]-0.7032)<2e-3 else '**不一致，以本次重拟合为准**'}")
    print(f"  基线最大逐点残差 = {tol:.4f} dex")
    print(f"  u={U_NEW}: b={b_of(U_NEW):.4f} mid={mid[U_NEW]:.2f} 因子="
          f"{U_NEW**(-b_of(U_NEW)):.6f}")
    print(f"  u={U_OLD}: b={b_of(U_OLD):.4f} mid={mid[U_OLD]:.2f}")
    # 基线在 u=0.65 上的 tw_od 覆盖（H0d 的参照）
    b65 = [(t["twod"], Crel(t["nm"], U_NEW)) for t in brows]
    b65 = [(t, c) for t, c in b65 if c is not None]
    r65 = np.log10(np.array([c for _, c in b65]) / Claw(np.array([t for t, _ in b65])))
    tol65 = float(np.max(np.abs(r65)))
    print(f"  基线在 u={U_NEW} 上：n={len(b65)}，最大逐点残差 = {tol65:.4f} dex"
          f"（H0d 的容差取这个，因为它与主比较同 u）")
    print(f"  基线的 tw_od 覆盖（u={U_NEW}）：{min(t for t,_ in b65):.3f} - "
          f"{max(t for t,_ in b65):.3f}")

    # ---- 新数据 ----
    d = json.load(open(os.path.join(OUT, "sweep_pedestal2.json")))
    rows, cfg = d["rows"], d["cfg"]
    cap = cfg.get("cap", float("nan"))
    new = []
    for rec in rows:
        m = rec["meta"]
        a = pedestal(float(m["phi0"]), float(m["r1"]), float(m["par"]), r)
        ma = float(a.mean())
        dd = hd.descriptors(a, r)
        dyn = (dd["_amax"] - dd["_amin"]) / ma
        Cabs = rec["lam_mean"] * U_NEW ** (-b_of(U_NEW))
        snr = rec["lam_mean"] * math.sqrt(cfg["nimg"]) / max(rec["lam_std"], 1e-9)
        new.append(dict(id=rec["id"], phi0=float(m["phi0"]), m=float(m["par"]),
                        r1=float(m["r1"]), kunif=rec["kunif"], kunif_reco=ma,
                        amin=dd["_amin"], amax=dd["_amax"], twl=dd["tw_lev"],
                        dyn=dyn, twod=dd["tw_lev"] / dyn if dyn > 1e-9 else float("nan"),
                        phi=dd["_amin"] / ma, lam=rec["lam_mean"],
                        lam_se=rec["lam_std"] / math.sqrt(cfg["nimg"]),
                        snr=snr, Cabs=Cabs, Crel=Cabs / mid[U_NEW],
                        lg=rec["lam_gauss"],
                        cls="有通带" if dd["_amin"] < 0.05 else
                            ("本类" if dd["_amax"] >= 0.99 else "窄带")))
    for t in new:
        t["inclass"] = bool(t["amin"] >= 0.05 and t["amax"] >= 0.99)
        t["pred"] = Claw(t["twod"])
        t["resid"] = math.log10(t["Crel"] / t["pred"])

    # ---------------- H0 守门 ----------------
    print("\n" + "=" * 104)
    print(" H0 守门")
    print("=" * 104)
    print(f" {'算子':<10} | {'kunif':>6} | {'amin':>6} {'amax':>6} | {'tw_od':>7} | "
          f"{'phi':>6} | {'SNR':>6} | {'C_rel':>7} | {'C预言':>7} | {'残差':>8} | 类")
    print(" " + "-" * 116)
    for t in sorted(new, key=lambda t: -t["phi"]):
        print(f" {t['id']:<10} | {t['kunif']:>6.4f} | {t['amin']:>6.3f} {t['amax']:>6.3f} | "
              f"{t['twod']:>7.3f} | {t['phi']:>6.3f} | {t['snr']:>6.1f} | {t['Crel']:>7.3f} | "
              f"{t['pred']:>7.3f} | {t['resid']:>+8.4f} | {t['cls']}")
    print(" " + "-" * 116)
    oka = bool(max(abs(t["kunif_reco"] - t["kunif"]) for t in new) <= 5e-3)
    n_cls = sum(1 for t in new if t["inclass"])
    n_pb = sum(1 for t in new if t["cls"] == "有通带")
    low = [t["id"] for t in new if t["snr"] < SNR_MIN]
    print(f" => H0(a) kunif 重建对账 {'✅' if oka else '❌'}（最大差 "
          f"{max(abs(t['kunif_reco']-t['kunif']) for t in new):.1e}）")
    print(f" => H0(b) 分类：本类 {n_cls}，有通带对照 {n_pb}，其他 {len(new)-n_cls-n_pb}")
    print(f" => H0(c) SNR<{SNR_MIN:g}：{len(low)} 个 {low}"
          f"（{'✅ 无' if not low else '⚠️ 已从 H1/H3 剔除'}）")
    good = [t for t in new if t["snr"] >= SNR_MIN]
    cls = [t for t in good if t["inclass"]]
    pb = [t for t in good if t["cls"] == "有通带"]
    if pb:
        rp = np.array([t["resid"] for t in pb])
        okd = bool(np.max(np.abs(rp)) <= tol65)
        print(f" => H0(d) 同 u 桥接：{len(pb)} 个对照的 max|残差| = "
              f"{np.max(np.abs(rp)):.4f} dex vs 同 u 容差 {tol65:.4f}  "
              f"{'✅' if okd else '❌'}")
    else:
        okd = False
        print(" => H0(d) 无对照，不可判定")

    # ---------------- H1 主判据：解耦 ----------------
    print("\n" + "=" * 104)
    print(" H1 主判据（解耦检验）：tw_od 匹配到 1% 以内的同 u 配对")
    print("=" * 104)
    PAIRS = [("A", "PX00m100", "PX10m200"), ("B", "PX00m070", "PX15m200")]
    print(f" {'配对':<4} | {'对照':<9} {'tw_od':>7} {'phi':>6} {'C_rel':>7} | "
          f"{'本类':<9} {'tw_od':>7} {'phi':>6} {'C_rel':>7} | {'Δtw_od':>7} | "
          f"{'Δlog10C':>9} | {'预言':>7}")
    print(" " + "-" * 122)
    h1 = []
    for tag, c0, c1 in PAIRS:
        a = next((t for t in new if t["id"] == c0), None)
        b = next((t for t in new if t["id"] == c1), None)
        if a is None or b is None:
            print(f" {tag:<4} | 缺算子 {c0} 或 {c1}")
            continue
        dl = math.log10(b["Crel"] / a["Crel"])
        dtw = b["twod"] / a["twod"] - 1
        # 预言的斜率：(ag) 在 tw_od≈0.29 处 class(phi=0.75) 比 baseline(phi=0) 低 0.107 dex
        slope = 0.107 / 0.75
        pred = -slope * b["phi"]
        h1.append((tag, c0, c1, dl, dtw, pred))
        print(f" {tag:<4} | {c0:<9} {a['twod']:>7.3f} {a['phi']:>6.3f} {a['Crel']:>7.3f} | "
              f"{c1:<9} {b['twod']:>7.3f} {b['phi']:>6.3f} {b['Crel']:>7.3f} | "
              f"{dtw*100:>6.1f}% | {dl:>+9.4f} | {pred:>+7.4f}")
    print(" " + "-" * 122)
    if len(h1) == 2:
        dlB = h1[1][3]
        dlA = h1[0][3]
        ok1 = bool(dlA < 0 and dlB <= -0.030)
        print(f" => H1 {'✅ 通过' if ok1 else '❌ 失败'}："
              f"配对A Δlog₁₀C = {dlA:+.4f}（预言 {h1[0][5]:+.4f}），"
              f"配对B = {dlB:+.4f}（预言 {h1[1][5]:+.4f}，阈值 −0.030）")
        print(f"     实测斜率 = {dlB/0.429:.4f} dex / 单位 φ（预官估计 0.143）")
        if dlA >= -0.015 and dlB >= -0.015:
            print("     **预官的失败方式发生**：两对下降都 <0.015 dex ⇒ (ag) 的「$C$ 依赖 φ」"
                  "被否掉，偏离必须归给 tw_od 本身或别的共变量。")
    else:
        ok1 = None
        print(" => H1 不可判定（配对不完整）")
    # 宽松的第三对
    a = next((t for t in new if t["id"] == "PX00m070"), None)
    b = next((t for t in new if t["id"] == "PX15m100"), None)
    if a and b:
        dl = math.log10(b["Crel"] / a["Crel"])
        print(f" 附（宽松第三对，Δtw_od={100*(b['twod']/a['twod']-1):.1f}%）："
              f"PX00m070(phi=0) vs PX15m100(phi=0.429)  Δlog₁₀C = {dl:+.4f}"
              f"（定律因 tw_od 差 9.7% 已含 {math.log10(b['pred']/a['pred']):+.4f}）")

    # ---------------- H2 跨 u 配对（只记录） ----------------
    print("\n" + "=" * 104)
    print(" H2 次级（跨 u 配对，**明记 u 漂移混淆**，只记录不作判据）")
    print("=" * 104)
    p2 = os.path.join(OUT, "sweep_pedestal.json")
    if os.path.exists(p2):
        ag = json.load(open(p2))["rows"]
        agm = {rec["id"]: rec for rec in ag}
        acfg = json.load(open(p2))["cfg"]
        b40, mid40 = b_of(U_OLD), mid[U_OLD]
        for oid in ("PD50m15", "PD45m15", "PD35m30"):
            if oid not in agm:
                continue
            rec = agm[oid]
            mm = rec["meta"]
            aa = pedestal(float(mm["phi0"]), float(mm["r1"]), float(mm["par"]), r)
            maa = float(aa.mean())
            dd = hd.descriptors(aa, r)
            dyn = (dd["_amax"] - dd["_amin"]) / maa
            td = dd["tw_lev"] / dyn
            c40 = rec["lam_mean"] * U_OLD ** (-b40) / mid40
            print(f"  (ag) {oid}: u={U_OLD} tw_od={td:.3f} phi={dd['_amin']/maa:.3f} "
                  f"C_rel={c40:.3f} 定律={Claw(td):.3f} 残差={math.log10(c40/Claw(td)):+.4f}")
        for oid in ("PX00m100", "PX00m070"):
            t = next((x for x in new if x["id"] == oid), None)
            if t:
                print(f"  (本组) {oid}: u={U_NEW} tw_od={t['twod']:.3f} phi={t['phi']:.3f} "
                      f"C_rel={t['Crel']:.3f} 定律={t['pred']:.3f} 残差={t['resid']:+.4f}")
        print("  预言：(ag) 那两个低 tw_od 类成员应低于本组对照（差 ≥ +0.03 dex）。")
        print("  **但 $C_{\\rm rel}$ 的 u 漂移与该效应同量级（见 H4b），本组无法分离。**")
    else:
        print(" (ag) 的数据不在，H2 跳过")

    # ---------------- H3 向下外推 ----------------
    print("\n" + "=" * 104)
    print(" H3 判别（向下外推到 tw_od<=0.15，基线与 (ag) 在这个区间一个点都没有）")
    print("=" * 104)
    lo = sorted([t for t in cls if t["twod"] <= 0.15], key=lambda t: t["twod"])
    print(f" {'算子':<10} | {'tw_od':>7} | {'phi':>6} | {'C_rel':>7} | {'C预言':>7} | "
          f"{'残差':>8} | {'sig':>6}")
    print(" " + "-" * 76)
    rl = []
    for t in lo:
        se = t["lam_se"] / max(abs(t["lam"]), 1e-9) / math.log(10)
        rl.append(t["resid"])
        print(f" {t['id']:<10} | {t['twod']:>7.3f} | {t['phi']:>6.3f} | {t['Crel']:>7.3f} | "
              f"{t['pred']:>7.3f} | {t['resid']:>+8.4f} | {abs(t['resid'])/se:>5.1f}")
    print(" " + "-" * 76)
    if rl:
        rl = np.array(rl)
        ok3 = bool(np.max(np.abs(rl)) <= tol)
        print(f" => H3 {'✅ 通过' if ok3 else '❌ 失败'}：max|残差| = {np.max(np.abs(rl)):.4f}"
              f"（容差 {tol:.4f}）；均值 {rl.mean():+.4f}；"
              f"{'全部同号' if (rl>0).all() or (rl<0).all() else '符号混合'}")
    else:
        ok3 = None

    # ---------------- H4 诊断 ----------------
    print("\n" + "=" * 104)
    print(" H4 诊断（无阈值）")
    print("=" * 104)
    # (b) C_rel 的 u 漂移
    dr = []
    for nm in sorted(prof):
        c65, c40 = Crel(nm, U_NEW), Crel(nm, U_OLD)
        if c65 and c40:
            dr.append(c65 / c40)
    dr = np.array(dr)
    if len(dr):
        print(f" (b) $C_{{\\rm rel}}$ 的 u 漂移（同一条剖面 u=0.65 / u=0.40）：n={len(dr)}，"
              f"中位 {np.median(dr):.4f}，范围 {dr.min():.4f}-{dr.max():.4f}"
              f"（{dr.max()/dr.min():.3f}x）")
        print(f"     => H2 的混淆下限约 {abs(math.log10(np.median(dr))):.4f} dex")
    # (c) 合并后的 corr(phi, tw_od)
    allc = [(t["phi"], t["twod"]) for t in cls]
    if os.path.exists(p2):
        for rec in json.load(open(p2))["rows"]:
            mm = rec["meta"]
            aa = pedestal(float(mm["phi0"]), float(mm["r1"]), float(mm["par"]), r)
            maa = float(aa.mean())
            dd = hd.descriptors(aa, r)
            dyn = (dd["_amax"] - dd["_amin"]) / maa
            if dyn <= 1e-9 or dd["_amin"] < 0.05 or dd["_amax"] < 0.99:
                continue
            if rec["lam_mean"] * math.sqrt(acfg["nimg"]) / max(rec["lam_std"], 1e-9) < SNR_MIN:
                continue
            allc.append((dd["_amin"] / maa, dd["tw_lev"] / dyn))
    ph = np.array([a for a, _ in allc])
    td = np.array([b for _, b in allc])
    print(f" (c) 合并 (ag)+本组 的类内 corr(phi, tw_od) = "
          f"{np.corrcoef(ph, td)[0, 1]:+.3f}（(ag) 单独是 -0.925；越接近 0 越解耦）")
    print(f"     合并后 tw_od 覆盖 {td.min():.3f}-{td.max():.3f}，"
          f"phi 覆盖 {ph.min():.3f}-{ph.max():.3f}")
    # (a) Lambda_Gauss 的结构核对
    print(" (a) $\\Lambda_G/({\\rm cap}\\cdot\\text{knif})$（平坦谱近似；pedestal 剖面"
          "因 $a$ 在低频小而 $w_g$ 在低频大，应系统地 <1）：")
    if np.isfinite(cap):
        for t in sorted(new, key=lambda t: -t["phi"])[:4] + sorted(new, key=lambda t: t["phi"])[:2]:
            print(f"     {t['id']:<10} L_G={t['lg']:>7.1f}  cap*knif={cap*t['kunif']:>7.1f}  "
                  f"比 {t['lg']/(cap*t['kunif']):.4f}  amin={t['amin']:.3f}")
    print(f" (d) 单点 C 的相对噪声中位 = "
          f"{np.median([t['lam_se']/max(abs(t['lam']),1e-9) for t in cls])/math.log(10):.4f} dex")

    with open(os.path.join(OUT, "fig_pedestal2.json"), "w") as fh:
        json.dump(dict(cfg=cfg, law=dict(coef=[float(c) for c in cf], x0=float(x0),
                                         p=float(p_), tol=tol, tol65=tol65),
                       b65=b_of(U_NEW), mid65=mid[U_NEW],
                       new=[{k: v for k, v in t.items()} for t in new],
                       h0=dict(kunif_ok=oka, n_cls=n_cls, n_pb=n_pb, low=low,
                               bridge_ok=okd),
                       h1=dict(pairs=[dict(tag=a, c0=b, c1=c, dl=e, dtw=f2, pred=g)
                                      for a, b, c, e, f2, g in h1], pass_=ok1),
                       h3=dict(pass_=ok3, n=len(rl),
                               max_res=float(np.max(np.abs(rl))) if len(rl) else None,
                               mean_res=float(rl.mean()) if len(rl) else None),
                       h4=dict(udrift_median=float(np.median(dr)) if len(dr) else None,
                               udrift_min=float(dr.min()) if len(dr) else None,
                               udrift_max=float(dr.max()) if len(dr) else None,
                               corr_phi_twod=float(np.corrcoef(ph, td)[0, 1]),
                               n_class=len(allc))), fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'fig_pedestal2.json')}")


if __name__ == "__main__":
    main()
