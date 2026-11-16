# -*- coding: utf-8 -*-
"""validate_judge_t_n120.py —— 新裁判必须先复现已发布基线（零 GPU，只读 results/*.json）

规则来源：本项目记忆里的硬规矩「新裁判必须先复现已发布基线」「新裁判要过零检验」。

做法：把 `judge_t_n120.py` 的 **N=120 那一臂整体换成 N=20 数据**
（BASE120 -> BASE20，NEW120 -> NEW20），NEW_T 保持 5 档不变，然后原样跑 main()。
这样跑出来的表必须与早已发布的 `results/log_bt_judge.txt` 一致：

  基线 b_inf : W2 -0.666 / W8 -0.736 / TK50 -0.713，rms 0.016 / 0.047 / 0.037
  t=0.05    : W2 -0.381 / W8 -0.389 / TK50 -0.436
  12 个已发布格子（t in {0.005,0.01,0.02,0.05}）：Δb_inf 同号 12/12、超 2SE 10/12
    唯二未过 2SE 的是 t=0.005 的 W8(+0.065 vs 2SE 0.068) 与 TK50(+0.032 vs 0.053)
  t=0.16    : b_inf -> +0.0023（高斯极限）
  NT5 必然 FAIL（两臂是同一份数据，SE 增益 = 1.0 < 1.6）——这是**预期的**，
  正说明 SE 增益那一列确实在比两个不同的样本量。

t=0.16 那一行不在已发布的 12 格里，只报不校（已发布研究把它单独作高斯极限处理）。

用法（服务器上）：
  cd /data/experiment/de_conv_1
  ./venv2/bin/python repro/validate_judge_t_n120.py
"""

import importlib.util
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = os.path.join(PROJ, "repro", "judge_t_n120.py")
    spec = importlib.util.spec_from_file_location("jt_val", p)
    jt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jt)

    print("=" * 100)
    print(" 验证：把裁判的 N=120 臂换成 N=20 数据，必须复现 log_bt_judge.txt")
    print("=" * 100)
    print(" 交换前 BASE120=%s" % jt.BASE120)
    jt.BASE120 = list(jt.BASE20)
    jt.NEW120 = dict(jt.NEW20)
    print(" 交换后 BASE120=%s" % jt.BASE120)
    print(" 交换后 NEW120=%s" % jt.NEW120)
    print(" NEW_T 不变：%s（5 档；已发布的 12 格只用前 4 档）\n" % jt.NEW_T)

    # 第一遍：容差用预官的 2.0。此时 rms_base 由 N=20 基线算出（W2 只有 0.016），
    # 所以 NT1 会在 0.02/W2 与 0.05/W2 上 FAIL —— 这正是已发布 T1 的两个 ❌，
    # 是**应该**出现的，不是缺陷。它会挡住后面的 NT2/NT3。
    sys.argv = ["jt_val", "--nboot", "2000"]
    jt.main()

    # 第二遍：把 NT1 容差临时放到 1e9 绕过守门（只为验证 NT2/NT3/零检验；
    # 阈值本身没动，默认值仍是预官的 2.0）。
    print("\n\n" + "#" * 100)
    print("# 第二遍：绕过 NT1 守门（NT1_TOL=1e9），验证零检验与 NT2/NT3/NT5")
    print("#" * 100 + "\n")
    jt.NT1_TOL = 1e9
    jt.main()

    print("\n" + "=" * 100)
    print(" 人工核对点（对照 log_bt_judge.txt）：")
    print("   1) 零检验必须 PASS（相同数据+相同下标 -> Δb_inf 恒精确 0）")
    print("   2) 基线 b_inf = -0.666 / -0.736 / -0.713，rms = 0.016 / 0.047 / 0.037")
    print("   3) 第一遍 NT1 必须**恰好在 0.02/W2 与 0.05/W2 上 FAIL**（原 T1 的两个 ❌）：")
    print("      因为 rms_base 现在由 N=20 基线算出（W2=0.016 -> 上限 0.032），")
    print("      而这两格 rms 0.036/0.043 超过它。真实跑 N=120 时 rms_base 是")
    print("      0.051/0.046/0.043，上限宽一倍，W2 那两格就会过。")
    print("   4) 第二遍前 12 格：NT2a 应 12/12 全 OK；NT2b 应 10/12，")
    print("      唯二 FAIL 在 t=0.005 的 W8(+0.065 vs 2SE 0.068) 与 TK50(+0.032 vs 0.053)")
    print("   5) t=0.16 的 b_inf：第一遍给 -0.149/+0.230/-0.075（N=20，未发布过这一格）")
    print("   6) NT5 必然 FAIL（两臂同数据，增益 1.0 < 1.6）——预期，见下")
    print("=" * 100)


if __name__ == "__main__":
    main()
