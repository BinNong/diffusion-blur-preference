"""验证 tw 描述量的**含义**：在合成剖面上检查它是否随"过渡带变宽"而单调增大。

动机
----
hunt_descriptor.py 发现 tw_lev = (r90-r10)/r50 在 22 个已测剖面上预测 C 很好
（R2=0.909，匹配处散度 1.030，全面超过 B2 的 1.475）。但 predict_tw.py 暴露一个问题：
erf 族里 EF35（边宽 w=0.35）tw_lev=0.926，而 EF60（w=0.60，边更宽）只有 0.686
——更宽的边反而算出更"陡"。原因是 tw_lev 的 levels 取径向剖面 min+{0.1,0.5,0.9}*range，
当边宽到剖面走不完自己的transition时，range 被压缩，10-90% 带就挤在一起。

本脚本不涉及任何 Λ，只检查描述量本身的语义：
  对每个族扫形状参数，tw 必须随"过渡带变宽"单调变化（方向先写定）。
哪个变体通过这个检查，才配拿去预测 C。

用法： venv2/bin/python validate_tw.py
"""

import importlib.util
import os

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "pt", os.path.join(PROJ, "repro", "predict_tw.py"))
pt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt)
hd = pt.hd

r = hd.halfplane()

# (族, kind, par 扫描, 期望方向): +1 = tw 随 par 增大而增大（par 是"宽度"）
#                              -1 = tw 随 par 增大而减小（par 是"陡峭度"）
SWEEPS = [
    ("erf      边宽 w", "erf", [0.10, 0.20, 0.35, 0.60, 1.00, 1.50], +1),
    ("logistic 边宽 w", "logistic", [0.10, 0.20, 0.35, 0.60, 1.00, 1.50], +1),
    ("tukey    过渡占比", "tukey", [0.10, 0.30, 0.50, 0.70, 0.90], +1),
    ("weibull  陡峭度 p", "weibull", [0.5, 1, 2, 4, 8, 16, 32], -1),
    ("butterw  陡峭度 n", "butterworth", [1, 2, 4, 8, 16], -1),
    ("ramp     幂 q", "ramp", [0.25, 0.5, 1.0, 2.0, 4.0], -1),
    ("twoscale 尺度比 R", "twoscale", [1.2, 1.5, 2.0, 3.0, 5.0], +1),
]

KEYS = ["tw_lev", "tw_mean", "tw_mass", "tw_massw", "B2", "supp20"]


def main():
    print("=" * 108)
    print(" tw 描述量的语义检查：扫形状参数，看 tw 是否朝期望方向单调变化")
    print("=" * 108)
    print(" 期望方向由物理先写定：par 是'宽度'=> tw 应随 par 增大；"
          "par 是'陡峭度'=> tw 应随 par 减小")
    print(" （全部在 knif_ref=0.92 下调好 sc，再缩到 knif=0.60；尺度不变量不受缩放影响）\n")

    verdict = {k: [0, 0] for k in KEYS}     # [通过数, 检查数]
    for label, kind, pars, direction in SWEEPS:
        print("-" * 108)
        print(f" {label}   (kind={kind}, 期望 tw 随 par {'增大' if direction > 0 else '减小'})")
        print(f" {'par':>6} | " + " | ".join(f"{k:>9}" for k in KEYS))
        prev = None
        vals = []
        for par in pars:
            sc = pt.tune(kind, par, r, 0.92)
            a = 1.0 - pt.make(kind, par, sc, r)
            a = a * (0.60 / float(a.mean()))
            d = hd.descriptors(a, r)
            row = []
            for k in KEYS:
                v = d[k]
                row.append(f"{v:>9.3f}" if np.isfinite(v) else f"{'NaN':>9}")
            print(f" {par:>6.2f} | " + " | ".join(row))
            vals.append((par, d))
        # 单调性：只比较相邻且都有定义的点
        for k in KEYS:
            seq = [(p, d[k]) for p, d in vals if np.isfinite(d[k])]
            bad = 0
            for i in range(len(seq) - 1):
                dv = seq[i + 1][1] - seq[i][1]
                if dv * direction < -1e-9:
                    bad += 1
            verdict[k][1] += max(len(seq) - 1, 0)
            verdict[k][0] += max(len(seq) - 1, 0) - bad
            if len(seq) >= 2:
                print(f"   => {k:<9} 定义 {len(seq)}/{len(pars)} 个点，"
                      f"单调性违例 {bad} 处"
                      f"{'  ✅' if bad == 0 and len(seq) == len(pars) else ''}")

    print("\n" + "=" * 108)
    print(" 汇总：语义检查通过率（违例越少、定义点越全越好）")
    print("=" * 108)
    print(f" {'描述量':<10} | {'通过':>6} | {'检查数':>6} | {'通过率':>6}")
    print(" " + "-" * 40)
    for k in KEYS:
        p, n = verdict[k]
        print(f" {k:<10} | {p:>6d} | {n:>6d} | {p/n:>6.1%}" if n
              else f" {k:<10} | 无从检查")
    print("\n 只有同时通过 (i) 语义单调性 和 (ii) 对 C 的预测力（hunt_descriptor.py 表 3）")
    print(" 的变体，才可以用来做预官。")


if __name__ == "__main__":
    main()
