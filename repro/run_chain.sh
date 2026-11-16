#!/bin/bash
# 链式执行：等第十组（ρ 饱和区）跑完 → 判第十组 → 跑第2步（degeneracy 做厚）→ 判第2步
# 这样两步都不需要人盯。2026-10-20 14:51 建立。
cd /data/experiment/de_conv_1
PY=./venv2/bin/python
L=/data/experiment/de_conv_1/results

echo "=== $(date '+%F %T') 等待第十组（run_sat.sh）完成 ==="
while ! grep -q "ALL DONE" $L/log_sat.txt 2>/dev/null; do sleep 60; done
echo "=== $(date '+%F %T') 第十组完成，跑判据 analyze_sat.py ==="
$PY repro/analyze_sat.py 2>&1 | tee $L/log_sat_judge.txt

echo "=== $(date '+%F %T') 第2步：degeneracy 做厚（8 图 × 5 噪声 × 3 核强度）==="
$PY repro/evidence_scale2.py 2>&1 | tee $L/log_evidence2.txt
echo "=== $(date '+%F %T') 第2步判据 ==="
$PY repro/judge_evidence2.py 2>&1 | tee $L/log_evidence2_judge.txt
echo "=== CHAIN DONE $(date '+%F %T') ==="
