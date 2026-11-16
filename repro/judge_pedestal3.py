"""sweep_pedestal3.py 的离线判据：J0–J3（零 GPU）。

预官见 sweep_pedestal3.py docstring（md5 `b0d046d0`，2026-10-23 10:18:50，早于测量）。
本组的唯一目的：把 §3.9(ah) 的「径向位置 3.2 倍」坐实——固定 φ₀=0.15 扫 m，图像数 20→120。

用法： venv2/bin/python judge_pedestal3.py
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
SNR_MIN = 10.0
U = 0.65
OLD_NIMG = 20
OLD_FILE = "sweep_pedestal2.json"


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
    assert U in mid and U in [round(x, 3) for x in Ucom]
    b0, mid0 = b_of(U), mid[U]

    def Crel(nm, u):
        v = hd.interp(prof[nm], u)
        return (v * u ** (-b_of(u)) / mid[u]) if (v and v > 0) else None

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
    b65 = [(t["twod"], Crel(t["nm"], U)) for t in brows]
    b65 = [(t, c) for t, c in b65 if c is not None]
    tol65 = float(np.max(np.abs(np.log10(
        np.array([c for _, c in b65]) / Claw(np.array([t for t, _ in b65]))))))

    print("=" * 100)
    print(f"  pedestal3 判据：{len(brows)} 个基线剖面；tw_od 定律 "
          f"C = {cf[0]:.4f} + {cf[1]:.4f}/(1+(twod/{x0:.4f})^(-{p_:.3f}))  R2={f['r2']:.3f}")
    print(f"  基线最大逐点残差 = {tol:.4f} dex；同 u={U} 上 = {tol65:.4f} dex")
    print(f"  u={U}: b={b0:.4f} mid={mid0:.2f} 因子={U**(-b0):.6f}")

    # ---- 新数据 ----
    d = json.load(open(os.path.join(OUT, "sweep_pedestal3.json")))
    rows, cfg = d["rows"], d["cfg"]
    nimg = cfg["nimg"]
    cap = cfg.get("cap", float("nan"))
    new = {}
    for rec in rows:
        m = rec["meta"]
        a = pedestal(float(m["phi0"]), float(m["r1"]), float(m["par"]), r)
        ma = float(a.mean())
        dd = hd.descriptors(a, r)
        dyn = (dd["_amax"] - dd["_amin"]) / ma
        v = np.array(rec["lam_per_img"])
        Cabs = rec["lam_mean"] * U ** (-b0)
        new[rec["id"]] = dict(
            id=rec["id"], m=float(m["par"]), phi0=float(m["phi0"]),
            r1=float(m["r1"]), kunif=rec["kunif"], kunif_reco=ma,
            amin=dd["_amin"], amax=dd["_amax"], twl=dd["tw_lev"], dyn=dyn,
            twod=dd["tw_lev"] / dyn if dyn > 1e-9 else float("nan"),
            phi=dd["_amin"] / ma, lam=rec["lam_mean"],
            lam_se=rec["lam_std"] / math.sqrt(nimg),
            snr=rec["lam_mean"] * math.sqrt(nimg) / max(rec["lam_std"], 1e-9),
            Crel=Cabs / mid0, lg=rec["lam_gauss"],
            pmean=float(v.mean()), pstd=float(v.std(ddof=1)),
            pmin=float(v.min()), pmax=float(v.max()),
            neg=int((v < 0).sum()),
            skew=float(np.mean(((v - v.mean()) / max(v.std(ddof=1), 1e-9)) ** 3)))
        new[rec["id"]]["pred"] = Claw(new[rec["id"]]["twod"])
        new[rec["id"]]["resid"] = math.log10(new[rec["id"]]["Crel"] /
                                            new[rec["id"]]["pred"])
        new[rec["id"]]["ms"] = float(np.median(v))

    order = ["PY15m020", "PY15m050", "PY15m100", "PY15m200"]
    print("\n" + "=" * 100)
    print(f" 新数据：nimg={nimg}，{len(new)} 个算子")
    print("=" * 100)
    print(f" {'算子':<10} | {'m':>5} | {'kunif':>6} | {'amin':>6} {'amax':>6} | "
          f"{'r1':>6} | {'tw_od':>7} | {'phi':>6} | {'SNR':>6} | {'C_rel':>7} | "
          f"{'C预言':>7} | {'残差':>8}")
    print(" " + "-" * 122)
    for oid in order:
        t = new[oid]
        print(f" {oid:<10} | {t['m']:>5.2f} | {t['kunif']:>6.4f} | {t['amin']:>6.3f} "
              f"{t['amax']:>6.3f} | {t['r1']:>6.4f} | {t['twod']:>7.3f} | {t['phi']:>6.3f} | "
              f"{t['snr']:>6.1f} | {t['Crel']:>7.3f} | {t['pred']:>7.3f} | "
              f"{t['resid']:>+8.4f}")
    print(" " + "-" * 122)

    # ---------------- J0 守门 ----------------
    print("\n" + "=" * 100)
    print(" J0 守门")
    print("=" * 100)
    oka = bool(max(abs(t["kunif_reco"] - t["kunif"]) for t in new.values()) <= 5e-3)
    okb = all(bool(t["amin"] >= 0.05 and t["amax"] >= 0.99) for t in new.values())
    low = [oid for oid in order if new[oid]["snr"] < SNR_MIN]
    print(f" => J0(a) kunif 重建对账 {'✅' if oka else '❌'}（最大差 "
          f"{max(abs(t['kunif_reco']-t['kunif']) for t in new.values()):.1e}）")
    print(f" => J0(b) amin/amax 分类 {'✅ 全部属本类' if okb else '❌ 有算子不在本类'}"
          f"（amin={new[order[0]]['amin']:.3f}, amax={new[order[0]]['amax']:.3f}）")
    print(f" => J0(c) SNR≥{SNR_MIN:g}：{len(low)} 个不达标 {low}"
          f"（{'✅' if not low else '⚠️ 计数剔除'}）")
    # J0(d) 跨轮次复现
    print(f" => J0(d) 跨轮次复现（(ah) 同算子、nimg={OLD_NIMG}）：")
    okd = True
    oldp = os.path.join(OUT, OLD_FILE)
    if os.path.exists(oldp):
        od = json.load(open(oldp))
        om = {rec["id"]: rec for rec in od["rows"]}
        ocfg = json.load(open(oldp))["cfg"]
        on = ocfg["nimg"]
        for oid in ("PY15m100", "PY15m200"):
            # (ah) 的算子名是 PX15m100 / PX15m200
            oid_old = oid.replace("PY", "PX")
            if oid_old not in om or oid not in new:
                continue
            o = om[oid_old]
            t = new[oid]
            se_o = o["lam_std"] / math.sqrt(on) / max(abs(o["lam_mean"]), 1e-9) \
                / math.log(10)
            se_n = t["lam_se"] / max(abs(t["lam"]), 1e-9) / math.log(10)
            sec = math.sqrt(se_o ** 2 + se_n ** 2)
            dv = math.log10(t["Crel"] / (o["lam_mean"] * U ** (-b0) / mid0))
            good = bool(abs(dv) <= 2 * sec)
            okd &= good
            print(f"     {oid}: (ah) C_rel={o['lam_mean']*U**(-b0)/mid0:.4f} "
                  f"(SE {se_o:.4f})  vs  本组 {t['Crel']:.4f} (SE {se_n:.4f})  "
                  f"差 {dv:+.4f} dex，2SE={2*sec:.4f}  {'✅' if good else '❌'}")
        # SE 的 1/sqrt(n) 标度自检
        print(f"     SE 标度自检：")
        for oid in ("PY15m100", "PY15m200"):
            oid_old = oid.replace("PY", "PX")
            if oid_old not in om or oid not in new:
                continue
            o = om[oid_old]
            t = new[oid]
            ratio = (t["lam_se"]) / (o["lam_std"] / math.sqrt(on))
            print(f"       {oid}: SE(120)/SE(20) = {ratio:.4f}，"
                  f"√(20/120) = {math.sqrt(OLD_NIMG/nimg):.4f}  "
                  f"{'✅' if abs(ratio-math.sqrt(OLD_NIMG/nimg))<0.02 else '⚠️'}")
    else:
        okd = False
        print("     (ah) 的数据不在，J0(d) 不可判定")

    # ---------------- J1 主判据 ----------------
    print("\n" + "=" * 100)
    print(" J1 主判据（坐实 (ah) 的 3.2 倍）：C_rel(m=0.20)/C_rel(m=2.00)")
    print("=" * 100)
    c20, c200 = new["PY15m020"]["Crel"], new["PY15m200"]["Crel"]
    ratio = c20 / c200
    se20 = new["PY15m020"]["lam_se"] / max(new["PY15m020"]["lam"], 1e-9) / math.log(10)
    se200 = new["PY15m200"]["lam_se"] / max(new["PY15m200"]["lam"], 1e-9) / math.log(10)
    ser = math.sqrt(se20 ** 2 + se200 ** 2)
    ah_ratio = 0.299 / 0.960
    print(f" (ah) 报的比值 = {ah_ratio:.4f}（0.299 / 0.960）")
    print(f" 本组实测比值 = {ratio:.4f}  ± {ser:.4f} dex"
          f"（= {c20:.3f} / {c200:.3f}）")
    print(f" 即倍数 = {1/ratio:.2f}x（(ah) 报 3.21x）")
    if ratio <= 0.50:
        v1 = "✅ 通过（≥2 倍）"
    elif ratio >= 0.70:
        v1 = "❌ 失败（≤1.43 倍）"
    else:
        v1 = "⚠️ 灰区 [0.50, 0.70]，记为不可判定"
    print(f" => J1 {v1}")
    # **实现修正（2026-10-23，披露）**：首版的一致性检查只用了本组的 SE，漏了 (ah) 的 SE。
    # (ah) 的 m=0.20 点 SE 是 0.121 dex（20 张图、5/20 负），是本组同点（0.032）的 4 倍，
    # 漏掉它会把"一致"误判成"不一致"。比值的 SE 必须两边都算。
    if os.path.exists(oldp):
        od0 = json.load(open(oldp))
        om0 = {rec["id"]: rec for rec in od0["rows"]}
        oc0 = od0["cfg"]
        se_ah = []
        for oid_n, oid_o in (("PY15m020", "PX15m020"), ("PY15m200", "PX15m200")):
            if oid_o in om0 and oid_n in new:
                o = om0[oid_o]
                se_ah.append(o["lam_std"] / math.sqrt(oc0["nimg"])
                             / max(abs(o["lam_mean"]), 1e-9) / math.log(10))
        if len(se_ah) == 2:
            ser_ah = math.sqrt(se_ah[0] ** 2 + se_ah[1] ** 2)
            ser_c = math.sqrt(ser ** 2 + ser_ah ** 2)
            dl = math.log10(ratio / ah_ratio)
            print(f"    与 (ah) 的一致性：Δlog₁₀(比值) = {dl:+.4f} dex；"
                  f"本组 SE = {ser:.4f}，(ah) SE = {ser_ah:.4f}，合并 2SE = {2*ser_c:.4f} dex "
                  f"⇒ {'一致' if abs(dl) <= 2*ser_c else '**不一致**'}")
            print(f"    （(ah) 的 m=0.20 点：C_rel={om0['PX15m020']['lam_mean']*U**(-b0)/mid0:.3f}，"
                  f"SE={se_ah[0]:.4f} dex；本组 {new['PY15m020']['Crel']:.3f}，SE={se20:.4f} dex）")
        else:
            print("     (ah) 的 SE 不全，一致性检查降级为只报差值")
    else:
        print("     (ah) 数据不在，一致性检查不可判定")
    if 0.50 < ratio < 0.70:
        need = (ser / 0.10) ** 2 * nimg
        print(f"    灰区判定所需图像数（按 SE ∝ 1/√n，要把 SE 压到 0.10/2=0.05 dex）："
              f"约 {need:.0f} 张")

    # ---------------- J2 m-曲线形状 ----------------
    print("\n" + "=" * 100)
    print(" J2 判别：C_rel 在 4 个 m 上单调不减，且 C(m=0.20)/C(m=1.00) ≤ 0.60")
    print("=" * 100)
    ms = np.array([new[o]["m"] for o in order])
    cs = np.array([new[o]["Crel"] for o in order])
    mono = bool(np.all(np.diff(cs) >= 0))
    r21 = new["PY15m020"]["Crel"] / new["PY15m100"]["Crel"]
    print(f" {'m':>6} | {'r1':>7} | {'tw_od':>7} | {'C_rel':>7} | {'SE(log10)':>10} | 残差")
    print(" " + "-" * 62)
    for o in order:
        t = new[o]
        se = t["lam_se"] / max(abs(t["lam"]), 1e-9) / math.log(10)
        print(f" {t['m']:>6.2f} | {t['r1']:>7.4f} | {t['twod']:>7.3f} | {t['Crel']:>7.3f} | "
              f"{se:>10.4f} | {t['resid']:>+8.4f}")
    print(" " + "-" * 62)
    ok2 = bool(mono and r21 <= 0.60)
    print(f" => 单调性 {'✅' if mono else '❌'}；C(0.20)/C(1.00) = {r21:.4f}"
          f"（阈值 0.60，(ah) 报 0.299/0.858 = {0.299/0.858:.4f}）")
    print(f" => J2 {'✅ 通过' if ok2 else '❌ 失败'}")
    print(f"    总跨度 C(2.00)/C(0.20) = {c200/c20:.3f}x；"
          f"tw_od 跨度 {new['PY15m200']['twod']/new['PY15m020']['twod']:.3f}x；"
          f"定律预言跨度 {new['PY15m200']['pred']/new['PY15m020']['pred']:.3f}x")

    # ---------------- J3 诊断 ----------------
    print("\n" + "=" * 100)
    print(" J3 诊断（把 (ah) 的估计量脆弱性定量化）")
    print("=" * 100)
    print(f" {'算子':<10} | {'mean':>7} | {'std':>6} | {'mean/std':>9} | {'min':>7} | "
          f"{'max':>7} | {'负值':>6} | {'偏度':>6} | {'中位':>7}")
    print(" " + "-" * 90)
    for o in order:
        t = new[o]
        print(f" {o:<10} | {t['pmean']:>7.0f} | {t['pstd']:>6.0f} | "
              f"{t['pmean']/max(abs(t['pstd']),1e-9):>9.3f} | {t['pmin']:>7.0f} | "
              f"{t['pmax']:>7.0f} | {t['neg']:>4d}/{nimg} | {t['skew']:>6.2f} | "
              f"{t['ms']:>7.0f}")
    print(" " + "-" * 90)
    print(f" (ah) 的对应值：PX15m020 mean/std=0.803、5/20 负；PX15m100 4.657、0/20；"
          f"PX15m200 6.452、0/20")
    print(f" => 若本组 PY15m020 的 mean/std 仍 <2 且负值占比仍 >10%，说明该算子的逐图 "
          f"$\\Lambda$ 分布本身是重尾的，均值作为描述量在此不稳——要在论文里改报中位数或截尾均值。")
    print("\n $\\Lambda_G/({\\rm cap}\\cdot\\text{knif})$（平坦谱近似；只记录）：")
    if np.isfinite(cap):
        for o in order:
            t = new[o]
            print(f"   {o}: L_G={t['lg']:>7.1f}  cap*knif={cap*t['kunif']:>7.1f}  "
                  f"比 {t['lg']/(cap*t['kunif']):.4f}")
    print(f" （(ah) 实测 1.003–1.010；我预官时曾把它推反成 <1，已披露。）")

    with open(os.path.join(OUT, "fig_pedestal3.json"), "w") as fh:
        json.dump(dict(cfg=cfg, law=dict(coef=[float(c) for c in cf], x0=float(x0),
                                         p=float(p_), tol=tol, tol65=tol65),
                       b0=b0, mid0=mid0,
                       new={k: v for k, v in new.items()},
                       j0=dict(kunif_ok=oka, class_ok=okb, low_snr=low, repro_ok=okd),
                       j1=dict(ratio=float(ratio), se=float(ser), ah_ratio=ah_ratio,
                               verdict=v1),
                       j2=dict(mono=mono, r21=float(r21), pass_=bool(ok2),
                               span=float(c200 / c20)),
                       j3=dict(per_img={o: dict(mean=new[o]["pmean"], std=new[o]["pstd"],
                                                neg=new[o]["neg"], skew=new[o]["skew"])
                                        for o in order})), fh, indent=1)
    print(f"\n[saved] {os.path.join(OUT, 'fig_pedestal3.json')}")


if __name__ == "__main__":
    main()
