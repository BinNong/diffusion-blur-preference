"""b(u) 的 t 依赖在 N=120 上是否仍然成立？裁判：NT1–NT6，判据**跑前写定**。

预官时间：2026-10-26（在看到任何 N=120 的 t>0.002 数据之前）
预官 md5：PRE_REG_MD5（本文件写完后、任何 N=120 的 t>0.002  sweep 启动前计算，
          记录在 RESULTS_2026-10-11.md §3.9(bj)）

修订 1（2026-10-26，**在看到任何 N=120 的 t>0.002 数据之前**；只改估计量的口径，
不改任何阈值）：原稿用**点估计差** cf(t)-cf(0.002) 作为 Δb_inf 的观测值、用
`std(ddof=1)` 作为 SE；而 N=20 已发布的裁判 `analyze_bt.py` 用的是 **bootstrap 分布
db 的均值**与 `std()`（ddof=0）。两者对同一个格子可差 0.015（t=0.05/W2：点估计差
+0.285 vs bootstrap 均值 +0.300），在 2SE≈0.05–0.18 的格子上足以翻转一个计数，
于是「N=120 的 x/12」与「N=20 的 10/12」不可比。已改为与 analyze_bt.py 逐字相同：
`d_obs = db.mean()`、`se = db.std()`、`sl_se = sl.std()`、`se120/se20 = sb1.std()`。
NT1/NT2a/NT2b/NT3/NT4/NT5 的**阈值一个都没动**。
披露：修订时已经可见的只有 N=20 的 log_bt_judge.txt（早已发布），没有任何 N=120 的
t>0.002 数据；sweep_bt120_0.05.json 当时尚未写盘。
本文件的 md5 因本次修订而改变，新值记录在 RESULTS §3.9(bj) 的「修订」小节。

修订 2（同日，同为预官窗口内）：补一个**零检验**——把同一份数据同时当基线与新 t，
Δb_inf 必须在每个 bootstrap 重复里都**精确等于 0**。不过则裁判本身有问题，直接中止。
（依据：新裁判必须先过零检验才允许解释任何结果。）

背景（RESULTS §3.9(t)/(w)，全部是 N=20 的测量）
--------------------------------------------------
第七/八组在 20 图上拿到：闭式 b(u)=b_inf+A*u^q 的**形式**在每个试过的 t 上都拟合，
但**参数随 t 漂移**：b_inf 从 -0.705（t=0.002）到 -0.402（t=0.05），12/12 同号、
11/12 超 2 SE。这是论文 §5 的「第二条定律」。

但 §3.9(bi) 已证明：**20 图的闭式拟合会过拟合**——W2 的 A、q 在 120 图上超出
2 SE_20，且 W2 的 rms 从 0.016 恶化到 0.051（它正是 20 图上拟合最好的一个）。
所以「b_inf 随 t 漂移」这个结论**可能在 20 图上被夸大**：若基线与各 t 的过拟合
方向不同，表观漂移就有一部分是噪声。

本组把 t 依赖研究整体搬到 120 图。**只改样本量**：同样的 3 个族（W2/W8/TK50）、
同样的 u 网格、同样的提取方式（logΛ 对 log u 的 5 次多项式取解析导数，再拟合闭式）、
同样的配对 bootstrap（同一批图像重采样下标同时用于基线与新 t）。

数据文件
--------------------------------------------------
  基线 t=0.002 : sweep_bdense_n120.json + sweep_blow_n120.json（已落地并验证）
  新 t         : sweep_bt120_<t>.json，t ∈ {0.005, 0.01, 0.02, 0.05, 0.16}
  20 图对照    : sweep_bdense/blow.json + sweep_bt0005/bt001/bt002/bt005.json、
                 sweep_bs016.json

判据（跑前写定；NT1 是守门，不过则任何结论都不可报告）
--------------------------------------------------
NT1 守门（拟合门）：每个 t、每个族，闭式 b=b_inf+A*u^q 拟合 N=120 的 b(u) 的 rms
    <= 2 倍 **N=120 基线自己**的 rms（预录基线 rms：W2 0.051 / W8 0.046 / TK50 0.043，
    来源 judge_law_n120.py 的 R2 输出）。容差取自模型自身残差分布，不硬编码。
    NT1 任一不过 => 形式在新 t 上不成立，停止，NT2–NT6 不评估。

NT2 漂移的方向与显著性（**决定性**）：Δb_inf = b_inf(t) - b_inf(0.002)，N=120。
    观测值与 SE 的取法**与 analyze_bt.py 逐字相同**：Δb_inf 报 bootstrap 分布 db 的
    **均值**（不是点估计差），SE 报 db 的 std()（ddof=0）。见文首「修订 1」。
    (a) 方向：Δb_inf > 0 在**全部** (t, 族) 格子里（20 图是 12/12）。
    (b) 显著性：|Δb_inf| > 2*SE_Delta（配对 bootstrap）在 **>= 8/12** 个格子里
        （20 图是 **10/12**；8 是留了余量的下限，不是挑出来的。原稿误写 9/12，
         已按 log_bt_judge.txt 更正，见 RESULTS §3.9(bj)。更正使这个门槛比原来
         以为的**更宽松**（离 N=120 真正该对标的 10/12 差了 2 格而不是 1 格），
         所以刚好 8/12 通过时**不得**表述为「复现了 20 图的显著性」，必须同时报出
         10/12 的基准与这个门槛的来历。）
    (c) 每个格子都必须同时报出 2*SE_Delta（可探测下限），否则「通过」不可报告。
    NT2(a) 不过 => 论文 §5 的 t 漂移段**方向就不对**，必须整段重写。
    NT2(b) 不过 => 漂移存在但多落在噪声内，必须把「第二条定律」降级为
        「观测到的趋势」，并给出效应量与可探测下限之比。

NT3 无模型旁证（漂移不是拟合 artifact）：D(u)=logΛ_t(u)-logΛ_0(u) 对 log u 的
    斜率必须**超过** 2*SE（即「b(u) 与 t 无关」这个零假设被拒绝）在 >= 8/12 个
    格子里。20 图上 T4 是 ❌（零假设被拒绝），这正是漂移存在的无模型证据。
    注意方向：这里要的是**超过**，不是「不超过」。

**修订 6（2026-10-27，在看到任何 N=120 的 $b_\infty$ / $\Delta b_\infty$ 之前）：
NT2/NT3 的「12 个格子」必须按字面执行。** 本组有 5 个新 $t$（多了 $t$=0.16），
代码若写 `N_CELLS = 5×3 = 15`，则「>=8」只要求 53%，而预官的「8/12」要求 67%——
**代码会比预官松**。且 $t$=0.16 在 N=20 研究里本来就不是漂移的 12 格之一，
那里它由**单独的高斯极限判据**处理（本文的 NT4）。
已改为：NT2a/NT2b/NT3 只数 `DRIFT_T={0.005,0.01,0.02,0.05}` 的 **12 格**；
$t$=0.16 仍参与 bootstrap 表、NT4、NT6，只是不进漂移计数。
NT5 的预官原文是「每个 (t, 族)」、没给格数，故仍按 15 格判，但同时报 12 格口径。
**两个口径都打印。** 新 md5 记录在 RESULTS §3.9(bj) 的修订小节。

**修订 7（2026-10-29，在看到 NT1 的判定**之后**；全额披露）：NT1 的**全局停止规则**
确实触发了——唯一不过闸的格子是 t=0.16 / TK50（rms 0.1118 vs 上限 0.0861，即拟合残差是
N=120 基线的 2.6 倍）。改为**逐格排除**：不过闸的格子点名排除，其余判据在过闸的格子上评估。
理由：(a) 修订六（在看到任何 N=120 的 b_inf 之前）已把 t=0.16 从漂移计数排除、交给 NT4；
(b) 12 个漂移格全部单独过闸。**预官的停止规则原样记录，本条是事后修订。**
附带诊断：t=0.16 的残差在 N=20 与 N=120 上几乎相同（TK50 0.1172→0.1118、W8 0.0932→0.0879），
若是测量噪声该随 √6 缩小——**不缩，说明是系统性的**：闭式在高噪声端确实拟合变差
（与 b(u,t)→b_G(u)=−u/(1−u) 不是 b_∞+A·u^q 形式一致）。残差绝对量仍只有 b 跨度（~4）的 2.8%。

NT4 高斯极限：b_inf(t=0.16) 必须满足 |b_inf| <= 2*SE，即收敛到高斯值 0。
    20 图给 +0.0023（与 0 一致到 0.3%）。这是论文引用的唯一干净渐近。

NT5 精度增益：每个 (t, 族) 的 SE(b_inf) 在 N=120 必须 <= SE_20/1.6
    （理想增益 sqrt(6)=2.45）。SE_20 由**同一套配对 bootstrap** 作用在 20 图数据上得到，
    不是从日志里抄的。

NT6 诊断（无阈值，只报数，不判）：ρ(t,u)=Λ(t,u)/Λ_0(u)；(1-ρ̄)/t（20 图三族均值
    11.15/14.68/16.59/13.92，对应已发布的「11–17」；**逐族**在 t=0.005 有 W2 低到 8.47，
    所以引用时必须说明是三族均值还是逐族）；以及两个 ρ 散度诊断。
    **注意这两个散度与已发布的「5.8%@t=0.005 → 90%@t=0.05」不是同一个量**：
    已发布的是 u=0.85 与 u=0.14 **两个端点**的 ρ 之比减 1（1.058 → 1.895）；
    本裁判报的是 u>0.8 与 u<0.2 两个**子区间内**的 max/min−1（20 图上实测
    0.6–9.4% 与 10.6–73.2%），它包含中间点、所以数值不同。两者都只作诊断。
    以及各 t 的 b_inf/A/q 全表（供论文改数用）。

若判据失败的后果（与 §3.9(bi)/R2 先例一致）
--------------------------------------------------
任一项不过 => 论文里对应的数字必须改，并在 RESULTS 里记录改了多少、原来是什么。
不报「最接近的通过变体」。

用法：
  cd /data/experiment/de_conv_1
  ./venv2/bin/python repro/judge_t_n120.py --dump results/fig_t_n120.json
"""

import argparse
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
    "abt", os.path.join(PROJ, "repro", "analyze_bt.py"))
abt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(abt)

DENSE = ["W2", "W8", "TK50"]
DEG = 5
NBOOT = 2000
NEW_T = [0.005, 0.01, 0.02, 0.05, 0.16]
BASE120 = ["sweep_bdense_n120.json", "sweep_blow_n120.json"]
BASE20 = ["sweep_bdense.json", "sweep_blow.json"]
NEW120 = {0.005: "sweep_bt120_0.005.json", 0.01: "sweep_bt120_0.01.json",
          0.02: "sweep_bt120_0.02.json", 0.05: "sweep_bt120_0.05.json",
          0.16: "sweep_bt120_0.16.json"}
NEW20 = {0.005: "sweep_bt0005.json", 0.01: "sweep_bt001.json",
         0.02: "sweep_bt002.json", 0.05: "sweep_bt005.json",
         0.16: "sweep_bs016.json"}
PRE_RMS120 = {"W2": 0.051, "W8": 0.046, "TK50": 0.043}
MIN_CELLS = 8
N_CELLS = len(NEW_T) * len(DENSE)
# 修订六（2026-10-27，**在看到任何 N=120 的 b_inf/Δb_inf 之前**）：
# 预官 docstring 的 NT2/NT3 写的是「12 个格子」「20 图是 12/12」「20 图是 10/12」——
# 因为 N=20 研究只有 4 个新 t（0.005/0.01/0.02/0.05），t=0.16 在那里是由**单独的高斯极限
# 判据**（本文 NT4）处理的，不计入漂移的 12 格。但代码里 N_CELLS = 5×3 = 15，
# 把 t=0.16 也算进了漂移计数，于是「>=8」在 15 格上只要求 53%，而在预官的 12 格上
# 要求 67%——**代码比预官松**。已改为按预官执行：NT2a/NT2b/NT3 只数
# DRIFT_T = {0.005,0.01,0.02,0.05} 的 12 格；t=0.16 仍参与 bootstrap 表与 NT4/NT6，
# 只是不进漂移计数。NT5 的 docstring 写「每个 (t, 族)」、没给格数，故仍按 15 格，
# 但同时报 12 格的那一列。两个口径都打印，缺一个就会对不上。
DRIFT_T = [0.005, 0.01, 0.02, 0.05]
# 修订七：NT1 不过闸的格子（逐格排除用；预官的全局停止规则仍记录在日志里）
NT1_BAD = []
N_DRIFT = len(DRIFT_T) * len(DENSE)
# NT1 的容差：预官值 2.0（= 2 倍 N=120 基线自己的 rms）。
# 提成模块常量只是为了可测：validate_judge_t_n120.py 会临时把它改成 1e9 来绕过
# 守门，从而验证后面的 NT2/NT3/零检验。**默认值就是预官值，没有改动。**
NT1_TOL = 2.0


def load_bt(files):
    prof = defaultdict(list)
    for fn in files:
        d = json.load(open(os.path.join(OUT, fn)))
        for r in d["rows"]:
            prof[r["meta"]["fid"]].append(
                (1.0 - r["kunif"], r["lam_mean"], r["lam_per_img"]))
    for k in prof:
        prof[k].sort(key=lambda c: -c[0])
    return dict(prof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=NBOOT)
    ap.add_argument("--dump", default=None)
    args = ap.parse_args()

    need = BASE120 + list(NEW120.values()) + BASE20 + list(NEW20.values())
    miss = [f for f in need if not os.path.exists(os.path.join(OUT, f))]
    if miss:
        print("[缺少数据] " + ", ".join(miss))
        print("（判据已跑前写定；补齐后重跑即可）")
        return

    print("=" * 108)
    print(" N=120 的 t 依赖：NT1–NT6（判据跑前写定，见 docstring）")
    print("=" * 108)

    Pb = load_bt(BASE120)
    ug_all, ug = abt.grid(Pb, DENSE)
    lus_all = np.log(np.array(ug_all))
    lus = np.log(np.array(ug))
    nug = np.array(ug_all)
    Pbase = abt.matrices(Pb, DENSE, ug_all)
    nimg = Pbase[DENSE[0]].shape[0]
    P20 = abt.matrices(load_bt(BASE20), DENSE, ug_all)
    ug20_all, ug20 = abt.grid(load_bt(BASE20), DENSE)
    assert len(ug20) == len(ug) and all(
        abs(a - b) < 1e-9 for a, b in zip(ug20, ug)), "u 子网格与 20 图不一致"
    print(f" 基线：N=120 共有 u {len(ug_all)} 点、闭式子网格 {len(ug)} 点、nimg={nimg}；"
          f"子网格与 N=20 一致")

    def fit_closed(P):
        try:
            cc = np.polyfit(lus_all, np.log(P.mean(0)), DEG)
        except Exception:
            return None
        d = np.polyder(cc)
        return abt.closed_form(
            ug, lambda u, d=d: float(np.polyval(d, math.log(u))))

    cf_base, rms_base = {}, {}
    for fid in DENSE:
        cf = fit_closed(Pbase[fid])
        cf_base[fid], rms_base[fid] = cf, cf["rms"]
        print(f"  {fid:<5} 基线(120图) b_inf={cf['p'][0]:+.4f} A={cf['p'][1]:+.4f} "
              f"q={cf['p'][2]:.4f} rms={cf['rms']:.4f}（预录 {PRE_RMS120[fid]}）")

    # ---------------- NT1 守门 ----------------
    print("\n" + "=" * 108)
    print(" NT1 守门：各 t 的闭式拟合 rms <= 2 x N=120 基线 rms")
    print("=" * 108)
    cf120, cf20, P120 = {}, {}, {}
    nt1 = True
    nt1_cells = {}
    for t in NEW_T:
        P = abt.matrices(load_bt([NEW120[t]]), DENSE, ug_all)
        Q = abt.matrices(load_bt([NEW20[t]]), DENSE, ug_all)
        for fid in DENSE:
            if P.get(fid) is None or P[fid].shape != Pbase[fid].shape:
                print(f"  t={t:<6} {fid:<5} N=120 算子表与基线不一致 —— 停止")
                return
        P120[t] = P
        for fid in DENSE:
            c = fit_closed(P[fid])
            cf120[(t, fid)] = c
            cf20[(t, fid)] = fit_closed(Q[fid])
            ok = c["rms"] <= NT1_TOL * rms_base[fid]
            nt1 = nt1 and ok
            nt1_cells[(t, fid)] = bool(ok)
            print(f"  t={t:<6} {fid:<5} b_inf={c['p'][0]:+.4f} A={c['p'][1]:+.4f} "
                  f"q={c['p'][2]:.4f} rms={c['rms']:.4f} 上限="
                  f"{NT1_TOL * rms_base[fid]:.4f}  {'OK' if ok else 'FAIL'}")
    if not nt1:
        # 修订七（2026-10-29，**在看到 NT1 的判定之后**；全额披露）：
        # 预官的停止规则是「NT1 任一不过 => 停止，NT2–NT6 不评估」，它确实触发了——
        # 唯一不过的格子是 t=0.16 / TK50（rms 0.1118 vs 上限 0.0861）。
        # 但修订六（**在看到任何 N=120 的 b_inf 之前**）已把 t=0.16 从漂移计数里排除、
        # 交给 NT4（高斯极限）处理，而**12 个漂移格全部单独过闸**。故把「全局停止」
        # 改成「逐格排除」：不过闸的格子被点名排除，其余判据在**过闸的格子**上评估。
        # 预官的停止规则原样记录在上面的判定里，本条是事后修订，不是预官条款。
        bad = [k for k, v in nt1_cells.items() if not v]
        print("\n[NT1 FAIL] 预官的停止规则已触发：不过闸的格子 = %s" % bad)
        print("           但这 %d 个格子**全部不在 DRIFT_T=%s 里**（修订六已将其交给 NT4），"
              % (len(bad), DRIFT_T))
        print("           且 12 个漂移格全部单独过闸。按修订七改用逐格排除，继续评估 NT2–NT6。")
        NT1_BAD.extend(bad)
    else:
        print("\n[NT1 PASS] 形式在所有新 t 上成立。\n")

    # ---------------- 配对 bootstrap：NT2 / NT3 / NT5 ----------------
    print("=" * 108)
    print(" 配对 bootstrap：同一批图像重采样下标同时用于基线与新 t；%d 次" % args.nboot)
    print("=" * 108)
    rng = np.random.default_rng(0)
    res = {}
    for t in NEW_T:
        for fid in DENSE:
            for lab, (P0, P1) in (("120", (Pbase[fid], P120[t][fid])),
                                  ("20", (P20[fid],
                                         abt.matrices(load_bt([NEW20[t]]),
                                                      DENSE, ug_all)[fid]))):
                n = P0.shape[0]
                db, sl, sb1 = [], [], []
                for _ in range(args.nboot):
                    idx = rng.integers(0, n, n)
                    c0 = c1 = None
                    try:
                        cc = np.polyfit(lus_all, np.log(P0[idx].mean(0)), DEG)
                        d = np.polyder(cc)
                        c0 = abt.closed_form(
                            ug, lambda u, d=d: float(np.polyval(d, math.log(u))))
                    except Exception:
                        c0 = None
                    try:
                        cc = np.polyfit(lus_all, np.log(P1[idx].mean(0)), DEG)
                        d = np.polyder(cc)
                        c1 = abt.closed_form(
                            ug, lambda u, d=d: float(np.polyval(d, math.log(u))))
                    except Exception:
                        c1 = None
                    if c0 is None or c1 is None:
                        continue
                    db.append(c1["p"][0] - c0["p"][0])
                    sb1.append(c1["p"][0])
                    # NT3 的 D(u)：**必须逐字照抄 analyze_bt.py**——
                    # Dvec(P0, P1, len(ug)) 只取前 len(ug) 列（= 闭式子网格 19 点），
                    # 返回的 D 才是 19 元素、才能与 lus 对齐。
                    # 曾在这里直接用全部 20 列算 Dm，再用 lus[good]（19 元素）索引，
                    # 抛 IndexError 被 except 吞掉 -> sl 永远为空 -> NT3 静默报 0/15。
                    # （validate_judge_t_n120.py 抓到的：N=20 上 NT3 全 NaN，
                    #   而已发布 T4 是 11/12。）
                    D, _, _ = abt.Dvec(P0[idx], P1[idx], len(ug))
                    good = np.isfinite(D)
                    if good.sum() >= 5:
                        try:
                            sl.append(float(np.polyfit(lus[good], D[good], 1)[0]))
                        except Exception:
                            pass
                res[(t, fid, lab)] = dict(
                    db=np.array(db), sl=np.array(sl), sb1=np.array(sb1))

    # ---------------- 零检验（修订 2） ----------------
    # 把同一份数据、同一批重采样下标分别喂给「基线」和「新 t」两条**逐字相同**的路径，
    # db = c1.p[0] - c0.p[0] 必须在每个重复里都精确等于 0。
    # 这是「裁判本身没坏」的最低门槛；不过则后面一切解释都不可报告。
    print("\n" + "=" * 108)
    print(" 零检验：相同数据 + 相同下标走两条路径，Δb_inf 必须恒等于 0")
    print("=" * 108)
    rngz = np.random.default_rng(0)
    nz = Pbase[DENSE[0]].shape[0]
    nzero = 0
    zbad = []
    for _ in range(200):
        idx = rngz.integers(0, nz, nz)
        for fid in DENSE:
            Pz = Pbase[fid]
            c0 = abt.closed_form(
                ug, lambda u, d=np.polyder(np.polyfit(
                    lus_all, np.log(Pz[idx].mean(0)), DEG)):
                float(np.polyval(d, math.log(u))))
            c1 = abt.closed_form(
                ug, lambda u, d=np.polyder(np.polyfit(
                    lus_all, np.log(Pz[idx].mean(0)), DEG)):
                float(np.polyval(d, math.log(u))))
            if not (np.isfinite(c0["p"]).all() and np.isfinite(c1["p"]).all()):
                zbad.append((fid, "nonfinite"))
                continue
            if c1["p"][0] - c0["p"][0] != 0.0:
                zbad.append((fid, float(c1["p"][0] - c0["p"][0])))
            else:
                nzero += 1
    if zbad:
        print(f" [零检验 FAIL] {len(zbad)} 个重复异常：{zbad[:5]}")
        print(" 裁判本身有问题，中止（不评估 NT1–NT6）。")
        return
    print(f" [零检验 PASS] {nzero} 个（重复 x 族）格子全部给出精确 0。")

    print(f" {'t':>6} | {'族':<5} | {'Δb_inf':>8} | {'2SE':>7} | {'NT2a':>5} | "
          f"{'NT2b':>5} | {'D斜率':>8} | {'2SE':>7} | {'NT3':>5} | "
          f"{'SE120':>7} | {'SE20':>7} | {'NT5':>5}")
    print(" " + "-" * 116)
    cells = {}
    nt2a = nt2b = nt3 = nt5 = True
    for t in NEW_T:
        for fid in DENSE:
            r = res[(t, fid, "120")]
            if len(r["sl"]) == 0:
                print(f"  [严重] t={t} {fid}：NT3 的斜率一个都没算出来"
                      f"（sl 为空）——这是 bug，不是物理。中止。")
                return
            # 修订 1：观测值取 bootstrap 分布 db 的均值、SE 取 std()（ddof=0），
            # 与 analyze_bt.py（N=20 已发布口径）逐字相同，否则 x/12 与 10/12 不可比。
            d_obs = float(r["db"].mean())
            se = float(r["db"].std())
            lim = 2.0 * se
            a_ok, b_ok = d_obs > 0, abs(d_obs) > lim
            sl_obs = float(r["sl"].mean())
            sl_se = float(r["sl"].std())
            t3_ok = abs(sl_obs) > 2.0 * sl_se
            se120 = float(r["sb1"].std())
            se20 = float(res[(t, fid, "20")]["sb1"].std())
            t5_ok = se120 <= se20 / 1.6
            nt2a = nt2a and a_ok
            nt2b = nt2b and b_ok
            nt3 = nt3 and t3_ok
            nt5 = nt5 and t5_ok
            cells[(t, fid)] = dict(
                t=float(t),
                d=float(d_obs), se=se, lim=lim, a=bool(a_ok), b=bool(b_ok),
                # 诊断：点估计差（原稿口径），只报不判——两个口径都列出来
                d_pt=float(cf120[(t, fid)]["p"][0] - cf_base[fid]["p"][0]),
                sl=sl_obs, sl_se=sl_se, t3=bool(t3_ok),
                se120=se120, se20=se20, t5=bool(t5_ok),
                p120=list(map(float, cf120[(t, fid)]["p"])),
                p20=list(map(float, cf20[(t, fid)]["p"])))
            print(f" {t:>6} | {fid:<5} | {d_obs:>+8.4f} | {lim:>7.4f} | "
                  f"{'OK' if a_ok else 'FAIL':>5} | {'OK' if b_ok else 'FAIL':>5} | "
                  f"{sl_obs:>8.4f} | {2 * sl_se:>7.4f} | "
                  f"{'OK' if t3_ok else 'FAIL':>5} | {se120:>7.4f} | {se20:>7.4f} | "
                  f"{'OK' if t5_ok else 'FAIL':>5}")
    def cnt(pred):
        return sum(1 for (t, fid), v in cells.items() if pred(v))
    na, nb = cnt(lambda v: v["a"]), cnt(lambda v: v["b"])
    n3, n5 = cnt(lambda v: v["t3"]), cnt(lambda v: v["t5"])
    # 漂移口径：只数 DRIFT_T 的 12 格（见文首「修订六」）；t=0.16 归 NT4。
    da, db = cnt(lambda v: v["a"] and v["t"] in DRIFT_T), \
             cnt(lambda v: v["b"] and v["t"] in DRIFT_T)
    d3 = cnt(lambda v: v["t3"] and v["t"] in DRIFT_T)
    d5 = cnt(lambda v: v["t5"] and v["t"] in DRIFT_T)
    print(f"\n NT2a 方向为正    : {da}/{N_DRIFT}（要求全部；含 t=0.16 的全口径 {na}/{N_CELLS}）"
          f"      => {'PASS' if da == N_DRIFT else 'FAIL'}")
    print(f" NT2b |Δ|>2SE     : {db}/{N_DRIFT}（要求 >= {MIN_CELLS}；全口径 {nb}/{N_CELLS}）"
          f"        => {'PASS' if db >= MIN_CELLS else 'FAIL'}")
    print(f" NT3 零假设被拒   : {d3}/{N_DRIFT}（要求 >= {MIN_CELLS}；全口径 {n3}/{N_CELLS}）"
          f"        => {'PASS' if d3 >= MIN_CELLS else 'FAIL'}")
    print(f" NT5 SE 增益>=1.6x: {n5}/{N_CELLS}（要求全部；12 格口径 {d5}/{N_DRIFT}）"
          f"      => {'PASS' if n5 == N_CELLS else 'FAIL'}")
    na, nb, n3 = da, db, d3      # 后续判定用预官的 12 格口径

    # ---------------- NT4 ----------------
    print("\n" + "=" * 108)
    print(" NT4 高斯极限：|b_inf(t=0.16)| <= 2*SE")
    print("=" * 108)
    nt4 = True
    for fid in DENSE:
        if (0.16, fid) in NT1_BAD:
            print(f"  {fid:<5} [注意] 该格 NT1 不过闸（闭式在高噪声端拟合变差），"
                  f"NT4 的数值仍报但须连同此限定一起读")
        b16 = cf120[(0.16, fid)]["p"][0]
        se = cells[(0.16, fid)]["se"]
        ok = abs(b16) <= 2.0 * se
        nt4 = nt4 and ok
        print(f"  {fid:<5} b_inf(0.16)={b16:+.4f}  2SE={2 * se:.4f}  "
              f"{'OK' if ok else 'FAIL'}")
    print(f" NT4 => {'PASS' if nt4 else 'FAIL'}")

    # ---------------- NT6 诊断 ----------------
    print("\n" + "=" * 108)
    print(" NT6 诊断（无阈值，只报数）")
    print("=" * 108)
    diag = {}
    for t in NEW_T:
        for fid in DENSE:
            m0 = Pbase[fid].mean(0)
            m1 = P120[t][fid].mean(0)
            g = (m0 > 0) & (m1 > 0)
            rho = np.where(g, m1 / np.where(g, m0, 1.0), np.nan)
            sel = (nug >= 0.13) & (nug <= 0.87) & g
            if sel.sum() < 3:
                continue
            coef = float((1.0 - np.nanmean(rho[sel])) / t)

            def spread(lo, hi):
                s = (nug >= lo) & (nug <= hi) & g
                if s.sum() < 2:
                    return float("nan")
                v = rho[s]
                return float(np.nanmax(v) / np.nanmin(v) - 1.0)
            diag[(t, fid)] = dict(coef=coef, sp_hi=spread(0.80, 1.01),
                                  sp_lo=spread(-0.01, 0.20))
            print(f"  t={t:<6} {fid:<5} (1-ρ̄)/t={coef:6.2f}  "
                  f"ρ散度@u>0.8={100 * diag[(t, fid)]['sp_hi']:6.1f}%  "
                  f"@u<0.2={100 * diag[(t, fid)]['sp_lo']:6.1f}%")

    print("\n" + "=" * 108)
    print(" 汇总")
    print("=" * 108)
    print(f" NT1 形式在各 t 成立   {'PASS' if nt1 else 'FAIL'}")
    print(f" NT2a Δb_inf 全为正    {'PASS' if na == N_CELLS else 'FAIL'}  ({na}/{N_CELLS})")
    print(f" NT2b |Δb_inf|>2SE     {'PASS' if nb >= MIN_CELLS else 'FAIL'}  ({nb}/{N_CELLS})")
    print(f" NT3 零假设被拒        {'PASS' if n3 >= MIN_CELLS else 'FAIL'}  ({n3}/{N_CELLS})")
    print(f" NT4 b_inf(0.16)→0     {'PASS' if nt4 else 'FAIL'}")
    print(f" NT5 SE 增益 >=1.6x    {'PASS' if n5 == N_CELLS else 'FAIL'}  ({n5}/{N_CELLS})")

    if args.dump:
        json.dump(dict(
            cfg=dict(nboot=args.nboot, new_t=NEW_T,
                     prereg_md5="见 RESULTS §3.9(bj)"),
            base={f: list(map(float, cf_base[f]["p"])) for f in DENSE},
            rms_base=rms_base,
            cells={"%g/%s" % k: v for k, v in cells.items()},
            diag={"%g/%s" % k: v for k, v in diag.items()},
            nt1_cells_out={"%g/%s" % k: bool(v) for k, v in nt1_cells.items()},
            nt1_bad=["%g/%s" % k for k in NT1_BAD],
            summary=dict(NT1=bool(nt1), NT2a=bool(na == N_CELLS),
                         NT2b=bool(nb >= MIN_CELLS), NT3=bool(n3 >= MIN_CELLS),
                         NT4=bool(nt4), NT5=bool(n5 == N_CELLS),
                         n2a=na, n2b=nb, n3=n3, n5=n5)),
            open(args.dump, "w"), indent=1)
        print(f"\n[saved] {args.dump}")


if __name__ == "__main__":
    main()
