#!/bin/bash
# 链：#2 evidence3 正确核重跑 → #1 过模糊搬到 20 真实池图
#
# 预官：
#   #2  repro/sweep_evidence3_kfix.py   md5 04afda226c98c66c3f21f10dcf963d65（RESULTS §3.9(bs)）
#       裁判 repro/judge_evidence3_kfix.py  md5 5caa4e2e609b27f6effd836f864c090d
#   #1  repro/repro_phaseC_pool20.py    md5 7846e2072f502b54c35d430c1f1b602c（RESULTS §3.9(bt)）
#       裁判 repro/judge_phaseC_pool20.py   md5 eade11c2b860d90475845c0db50b4043
#
# 顺序：#2 已在跑（本脚本只等它落地后接裁判），#1 排在 #2 之后，因为 GPU 只有一个。
# 每步自己的日志；崩了只损失当前步。
#
# 用法（服务器上）：nohup bash repro/run_kfix_chain.sh > results/log_kfix_chain.txt 2>&1 &
set -u
cd /data/experiment/de_conv_1 || exit 1
PY=./venv2/bin/python

say() { echo ""; echo "########## $(date '+%F %T') $* ##########"; }

say "启动；等 #2（evidence3_kfix.json）落地"
for i in $(seq 1 300); do
    sleep 60
    if [ -f /data/experiment/de_conv_1/results/evidence3_kfix.json ]; then
        say "#2 已落地（第 $i 分钟）"
        break
    fi
    if [ "$i" -eq 300 ]; then say "等超 5 小时，#2 仍未落地——中止"; exit 1; fi
done

say "跑 #2 的裁判（零 GPU）"
$PY repro/judge_evidence3_kfix.py

say "#1：过模糊搬到 20 真实池图（120 条 profile × 7.6 min ≈ 15.2 h）"
$PY repro/repro_phaseC_pool20.py --out phaseC_pool20.json || say "#1 失败"

say "跑 #1 的裁判（零 GPU）"
$PY repro/judge_phaseC_pool20.py

say "全部结束"
