#!/bin/bash
# 第十九组顺序链：独立图池 sweep -> 预官裁判 O0-O6
# 单链、单 GPU 任务；不要与本链并行启动第二个 sweep。
P=/data/experiment/de_conv_1
SWEEP_LOG=$P/results/log_overblur_indep.txt
JUDGE_LOG=$P/results/log_overblur_indep_judge.txt

cd "$P" || exit 1

echo "########## $(date '+%Y-%m-%d %H:%M:%S %z') chain start ##########" >> "$SWEEP_LOG"
./venv2/bin/python repro/sweep_overblur_indep.py --out overblur_indep.json >> "$SWEEP_LOG" 2>&1
RC=$?
echo "########## $(date '+%Y-%m-%d %H:%M:%S %z') sweep exit=$RC ##########" >> "$SWEEP_LOG"

if [ "$RC" -ne 0 ]; then
    echo "sweep failed; judge skipped" >> "$SWEEP_LOG"
    exit "$RC"
fi

echo "########## $(date '+%Y-%m-%d %H:%M:%S %z') judge start ##########" >> "$JUDGE_LOG"
./venv2/bin/python repro/judge_overblur_indep.py >> "$JUDGE_LOG" 2>&1
JRC=$?
echo "########## $(date '+%Y-%m-%d %H:%M:%S %z') judge exit=$JRC ##########" >> "$JUDGE_LOG"
exit "$JRC"
