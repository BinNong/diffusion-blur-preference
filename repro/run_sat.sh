#!/bin/bash
# 第十组：ρ(t) 的饱和区。预官 2026-10-20 14:44，sweep_law_test.py md5 ea96967cc7057636eb0a78b6f0f1b014
# 只改 t；族/u 网格/图数/提取方式与第八组逐字对齐。
# 顺序：0.16 先跑（信号最小、最可能 SNR 崩，早失败早收手），然后 0.12 / 0.08。
cd /data/experiment/de_conv_1
PY=./venv2/bin/python
SW=repro/sweep_law_test.py
TG="--targets 0.15 0.18 0.22 0.26 0.30 0.34 0.38 0.42 0.46 0.52 0.58 0.65 0.70 0.74 0.78 0.82 0.86"
FAM="--families W2 W8 TK50"
for T in 0.16 0.12 0.08; do
  TAG=$(echo $T | tr -d '.')
  OUT=sweep_bs${TAG}.json
  echo "======== $(date '+%F %T')  t=$T  out=$OUT ========"
  $PY $SW --nimg 20 --t $T $TG $FAM --out $OUT
  echo "-------- $(date '+%F %T')  t=$T done, rc=$? ---------"
done
echo "ALL DONE $(date '+%F %T')"
