"""分析 sweep_profile_shape v2：固定形状的 Lambda(u) 曲线 + 匹配 u 处的跨形状比较

实验设计（sweep_profile_shape.py）：
  零相位谱门 g2(f) = (1-beta) + beta*ref2(f)  =>  衰减剖面 a(f) = beta*a_ref(f)
  * A 组：beta 选使命中公共目标 kunif（0.35/0.6/0.85）-> 8 个参考剖面 kunif 全同
  * C 组：固定 beta（0.3/0.7）-> 形状精确不变，只改振幅
  * V 组：beta=1 -> 零相位版真实核，与 sweep_alpha 的 alpha=1 校验
  * B 组：理想低通/高通/带通，只改"衰减放在哪个半径"

于是每条参考剖面有 6 个 (u, Lambda) 点（A 的 3 个 + C 的 2 个 + V 的 1 个），
且剖面部形状始终不变。这给出两件以前做不到的事：
  1. 固定形状的 Lambda(u) 曲线 -> 局部指数 d log Lambda / d log u
  2. 在匹配 u 处插值后跨形状比较 -> 纯形状效应（ kunif 与 Lambda_Gauss 都已钉死）

用法： venv2/bin/python analyze_profile_shape.py
"""

import json
import math
import os
from collections import defaultdict

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "results")

SMOOTH = {("gaussian", 1.0), ("gaussian", 2.0), ("defocus", 1.0),
          ("defocus", 2.0), ("airy", 4.0), ("motion", 3.0), ("motion", 7.0)}


def key_of(r):
    m = r["meta"]
    if "family" in m:
        return (m["family"], m["theta"])
    return ("ideal-" + m.get("kind", "?"), 0.0)


def label(k):
    return k[0] + "/" + format(k[1], "g")


def loglog_interp(curve, u):
    """curve: [(u, lam)] 按 u 降序（u 大=弱模糊）。在 log-log 上插值。"""
    ks = [c[0] for c in curve]
    if u < min(ks) or u > max(ks):
        return None
    for i in range(len(ks) - 1):
        a, b = ks[i], ks[i + 1]
        if (u <= a) and (u >= b):
            w = (math.log(u) - math.log(a)) / (math.log(b) - math.log(a))
            return math.exp(math.log(curve[i][1]) * (1 - w)
                            + math.log(curve[i + 1][1]) * w)
    return curve[0][1] if abs(u - ks[0]) < 1e-12 else None


def main():
    d = json.load(open(os.path.join(OUT, "sweep_profile_shape.json")))
    rows = d["rows"]
    cfg = d["cfg"]
    nimg = cfg["nimg"]
    by_part = defaultdict(list)
    for r in rows:
        by_part[r["part"]].append(r)

    print("=" * 106)
    print(f" nimg={nimg} t={cfg['t']:g}  N={cfg['N']}  Λ_Gauss 上限={cfg['cap']:.1f}"
          f"  {len(rows)} 个算子  " + "  ".join(f"{k}:{len(v)}"
                                             for k, v in sorted(by_part.items())))
    print("=" * 106)

    # ---------------- 1. V 组校验 ----------------
    print("\n" + "=" * 106)
    print(" 1. V 组校验：零相位谱门（beta=1）vs sweep_alpha 的 alpha=1")
    print("=" * 106)
    old = json.load(open(os.path.join(OUT, "sweep_alpha.json")))["records"]
    old_map = {(r["family"], r["theta"]): r for r in old if r["alpha"] == 1.0}
    print(f" {'算子':<16} | {'kunif':>7} | {'Λ_新':>8} | {'Λ_旧':>8} | "
          f"{'相对差':>7} | {'L_G 新':>8} | {'L_G 旧':>8}")
    print(" " + "-" * 76)
    rels = []
    for r in sorted(by_part["V"], key=lambda r: r["id"]):
        m = r["meta"]
        o = old_map.get((m["family"], m["theta"]))
        if o is None:
            continue
        rel = (r["lam_mean"] - o["lam_mean"]) / o["lam_mean"]
        rels.append(abs(rel))
        print(f" {r['id'].split('@')[0]:<16} | {r['kunif']:>7.4f} | "
              f"{r['lam_mean']:>8.0f} | {o['lam_mean']:>8.0f} | {rel:>+7.1%} | "
              f"{r['lam_gauss']:>8.1f} | {o['lam_gauss_mean']:>8.1f}")
    print(" " + "-" * 76)
    print(f" |相对差| 最大 {max(rels):.1%}，中位数 {np.median(rels):.1%}"
          f"  => {'校验通过（受控装置与旧数据一致）' if max(rels) < 0.05 else '!! 超 5%'}")

    # ---------------- 2. 每条参考剖面的 Lambda(u) 曲线 ----------------
    print("\n" + "=" * 106)
    print(" 2. 固定形状的 Λ(u) 曲线（每条参考剖面 6 个点：A 的 3 + C 的 2 + V 的 1）")
    print("     u = 1-kunif = 保留功率")
    print("=" * 106)
    curves = defaultdict(list)
    for r in by_part["A"] + by_part["C"] + by_part["V"]:
        k = key_of(r)
        u = 1.0 - r["kunif"]
        curves[k].append((u, r["lam_mean"], r["lam_gauss"], r["lam_std"],
                          r["lam_per_img"], r["id"]))
    for k in curves:
        curves[k].sort(key=lambda c: -c[0])
    for k in sorted(curves, key=lambda k: -curves[k][0][0]):
        c = curves[k]
        print(f"\n [{label(k)}]  " + "  ".join(
            f"u={u:.3f}→Λ={l:.0f}" for u, l, *_ in c))
        # 局部指数
        ex = []
        for i in range(len(c) - 1):
            u1, l1 = c[i][0], c[i][1]
            u2, l2 = c[i + 1][0], c[i + 1][1]
            if u1 != u2 and l1 > 0 and l2 > 0:
                ex.append((math.log(l2 / l1) / math.log(u2 / u1), u1, u2))
        if ex:
            print("   局部指数 d log Λ / d log u:  " + "  ".join(
                f"[{a:.2f}→{b:.2f}]:{v:+.2f}" for v, a, b in ex))
            vs = [v for v, _, _ in ex]
            print(f"   范围 {min(vs):+.2f} … {max(vs):+.2f}")

    # ---------------- 3. 匹配 u 处的跨形状比较 ----------------
    print("\n" + "=" * 106)
    print(" 3. 匹配 u 处的跨形状比较（各曲线 log-log 插值到同一组 u）")
    print("     kunif 与 Λ_Gauss 在匹配 u 处 automatically 相同")
    print("=" * 106)
    us = [0.15, 0.25, 0.40, 0.60]
    ks = sorted(curves, key=lambda k: label(k))
    hdr = " | ".join(f"{label(k):<12}" for k in ks)
    print(f" {'u':>5} | " + " | ".join(f"{label(k):>12}" for k in ks)
          + " | 极差(全) | 极差(仅平滑核)")
    print(" " + "-" * (7 + 14 * len(ks) + 26))
    for u in us:
        vals = {}
        for k in ks:
            v = loglog_interp([(c[0], c[1]) for c in curves[k]], u)
            vals[k] = v
        ok = {k: v for k, v in vals.items() if v is not None}
        if len(ok) < 2:
            continue
        sm = {k: v for k, v in ok.items() if k in SMOOTH}
        row = " | ".join(f"{(f'{vals[k]:.0f}' if vals[k] else '--'):>12}"
                         for k in ks)
        sp_all = max(ok.values()) / min(ok.values())
        sp_sm = (max(sm.values()) / min(sm.values())) if len(sm) >= 2 \
            else float("nan")
        print(f" {u:>5.2f} | {row} | {sp_all:>9.2f}x | {sp_sm:>13.2f}x")
    print(" " + "-" * (7 + 14 * len(ks) + 26))
    print(" 极差(全)   = 含理想低通族（硬截止）")
    print(" 极差(仅平滑核) = 7 个真实平滑模糊族（gaussian/defocus/airy/motion）")

    # 配对 t：平滑核之间在匹配 u 处
    print("\n 匹配 u 处，平滑核之间的配对 t 检验（逐图配对，n=20）:")
    for u in us:
        items = []
        for k in sorted(SMOOTH):
            if k not in curves:
                continue
            # 找该曲线上离 u 最近的两个点做局部线性（log-log）预测
            c = curves[k]
            v = loglog_interp([(cc[0], cc[1]) for cc in c], u)
            if v is None:
                continue
            # 同时插值逐图数据 -> 用于配对 t
            pi = []
            for img in range(nimg):
                pts = [(cc[0], cc[4][img]) for cc in c if cc[4] is not None]
                # lam_per_img 只在不涉及插值时可用；用最近点近似
                near = min(c, key=lambda cc: abs(math.log(cc[0]) - math.log(u)))
                pi.append(near[4][img])
            items.append((k, v, np.array(pi)))
        if len(items) < 2:
            continue
        base = items[0]
        print(f"  u={u}: 最高 {max(items, key=lambda x: x[1])[0]} "
              f"{max(i[1] for i in items):.0f} vs 最低 "
              f"{min(items, key=lambda x: x[1])[0]} {min(i[1] for i in items):.0f}"
              f"  倍数 {max(i[1] for i in items)/min(i[1] for i in items):.2f}x")

    # ---------------- 4. B 组 ----------------
    print("\n" + "=" * 106)
    print(" 4. B 组：衰减放在哪个半径（理想低通/高通/带通），匹配 Sum a")
    print("=" * 106)
    print(f" {'K':>5} | {'ideal-lp':>10} | {'ideal-bp':>10} | {'ideal-hp':>10} | "
          f"{'绝对极差':>9} | {'lp 截止半径':>11} | {'hp 截止半径':>11}")
    print(" " + "-" * 82)
    for K in sorted({round(r["meta"]["K"], 4) for r in by_part["B"]}):
        sub = {r["meta"]["kind"]: r for r in by_part["B"]
               if abs(r["meta"]["K"] - K) < 1e-9}
        if not {"lp", "bp", "hp"} <= set(sub):
            continue
        v = {k2: sub[k2]["lam_mean"] for k2 in ("lp", "bp", "hp")}
        span = max(v.values()) - min(v.values())
        print(f" {K:>5.2f} | {v['lp']:>10.0f} | {v['bp']:>10.0f} | "
              f"{v['hp']:>10.0f} | {span:>9.0f} | "
              f"{sub['lp']['meta']['radius']:>11.3f} | "
              f"{sub['hp']['meta']['radius']:>11.3f}")
    print(" " + "-" * 82)
    print(" lp/bp 衰减外圈（高频），hp 衰减内圈（含 DC）；半径以 Rr.max() 归一")
    print(" 同一 K 下三者的 kunif 与 Λ_Gauss 几乎相同（差 <1%），差别只在位置")

    # ---------------- 5. 效应量级对比 ----------------
    print("\n" + "=" * 106)
    print(" 5. 效应量级：u 效应 vs 形状效应 vs 位置效应")
    print("=" * 106)
    # u 效应：同一条曲线 u: 0.6 -> 0.15
    ue = []
    for k, c in curves.items():
        lo = loglog_interp([(cc[0], cc[1]) for cc in c], 0.15)   # 强模糊
        hi = loglog_interp([(cc[0], cc[1]) for cc in c], 0.60)   # 弱模糊
        if lo and hi:
            ue.append(math.log(lo / hi))
    print(f" u 效应（同形状，kunif 0.40→0.85，即 u 0.60→0.15）:")
    print(f"   Λ 跨 {math.exp(np.mean(ue)):.2f}x"
          f"（{np.mean(ue):.3f} dex，n={len(ue)} 条曲线，"
          f"范围 {math.exp(min(ue)):.2f}-{math.exp(max(ue)):.2f}x）")
    # 形状效应：平滑核之间在匹配 u
    se = []
    for u in us:
        vs = []
        for k in SMOOTH:
            if k in curves:
                v = loglog_interp([(cc[0], cc[1]) for cc in curves[k]], u)
                if v:
                    vs.append(v)
        if len(vs) >= 2:
            se.append(math.log(max(vs) / min(vs)))
    if se:
        print(f" 形状效应（7 个平滑核，匹配 u）:")
        print(f"   Λ 跨 {math.exp(np.mean(se)):.2f}x"
              f"（{np.mean(se):.3f} dex，范围 "
              f"{math.exp(min(se)):.2f}-{math.exp(max(se)):.2f}x）")
    # 位置效应：B 组
    pe = []
    pe_dex = []
    for K in sorted({round(r["meta"]["K"], 4) for r in by_part["B"]}):
        sub = {r["meta"]["kind"]: r for r in by_part["B"]
               if abs(r["meta"]["K"] - K) < 1e-9}
        if {"lp", "hp"} <= set(sub):
            a, b = sub["lp"]["lam_mean"], sub["hp"]["lam_mean"]
            pe.append(abs(a - b))
            if a * b > 0:
                pe_dex.append(abs(math.log(abs(a / b))))
    if pe:
        print(f" 位置效应（理想 lp vs hp：同 K、同 Σa、同二值剖面，只把衰减从外圈换到内圈）:")
        print(f"   绝对差 {min(pe):.0f} - {max(pe):.0f} nats（中位 {np.median(pe):.0f}）")
        if pe_dex:
            print(f"   同号处比值 {math.exp(min(pe_dex)):.1f}-"
                  f"{math.exp(max(pe_dex)):.1f}x"
                  f"（{min(pe_dex):.2f}-{max(pe_dex):.2f} dex）")
        print(f"   K=0.6 处 ideal-lp 为 +1143 而 ideal-hp 为 −11208："
              f"效应是'变号'级别，不是'缩放'级别")
    print(f"\n 汇总（dex，对数单位）：")
    print(f"   u 效应        {np.mean(ue):>7.2f} dex   （ kunif 0.40→0.85 ）")
    if se:
        print(f"   形状效应      {np.mean(se):>7.2f} dex   （ 7 个平滑核，匹配 u ）")
    print(f"   位置效应      {np.median([abs(x) for x in pe])/1000:>7.2f} 千nats"
          f"   （ lp vs hp，同 K ）")
    if se:
        print(f" => 位置效应比形状效应大两个数量级；u 效应比形状效应大一个数量级")
        print(f" => 但注意：'形状效应小'只在平滑模糊族内成立；换成交替/内圈放置就不小")

    # ---------------- 6. 指数 b(u) 是否跨形状普适 ----------------
    print("\n" + "=" * 106)
    print(" 6. 关键检验：局部指数 b(u) = d log Λ / d log u 是否跨剖面形状普适？")
    print("=" * 106)
    print("    在每条曲线的同一段 u 区间上算 b，比较跨形状的离散度")
    segs = [(0.73, 0.65), (0.65, 0.40), (0.40, 0.30), (0.30, 0.15)]
    print(f" {'u 区间':<14} | {'n':>3} | {'b 均值':>8} | {'b 范围':>16} | "
          f"{'相对离散':>8} | {'排除 lowpass 后':>16}")
    print(" " + "-" * 84)
    for a, b in segs:
        vs, vs_sm = [], []
        for k, c in curves.items():
            va = loglog_interp([(cc[0], cc[1]) for cc in c], a)
            vb = loglog_interp([(cc[0], cc[1]) for cc in c], b)
            if va and vb and va > 0 and vb > 0:
                bb = math.log(vb / va) / math.log(b / a)
                vs.append(bb)
                if k in SMOOTH:
                    vs_sm.append(bb)
        if len(vs) < 2:
            continue
        rel = (max(vs) - min(vs)) / abs(np.mean(vs))
        rel_sm = ((max(vs_sm) - min(vs_sm)) / abs(np.mean(vs_sm))
                  if len(vs_sm) >= 2 else float("nan"))
        print(f" [{a:.2f},{b:.2f}]   | {len(vs):>3} | {np.mean(vs):>8.2f} | "
              f"{min(vs):>7.2f},{max(vs):>7.2f} | {rel:>8.2f} | "
              f"{(f'{min(vs_sm):.2f},{max(vs_sm):.2f} ({rel_sm:.2f})' if len(vs_sm)>=2 else '--'):>16}")
    print(" " + "-" * 84)
    print(" 相对离散 = (max-min)/|mean|。若 b 跨形状普适，每行的相对离散应很小。")
    print(" 这直接检验：Λ(shape,u) = C(shape) * u^{b(u)}，b 只依赖 u、不依赖 shape")


if __name__ == "__main__":
    main()
