#!/bin/bash
# 第八组：b(u) 是否随噪声级 t 变？预官 2026-10-18 21:44，md5 5864b1cbae505a75b28fd0674772c53c
# 只改 t；族/u 网格/图数/提取方式与第七组逐项对齐。
# 顺序：0.05 先跑（杠杆最大：Λ_ng 相对 t=0.002 掉到 0.096），然后 0.005 / 0.01 / 0.02。
cd /data/experiment/de_conv_1
PY=./venv2/bin/python
SW=repro/sweep_law_test.py
TG="--targets 0.15 0.18 0.22 0.26 0.30 0.34 0.38 0.42 0.46 0.52 0.58 0.65 0.70 0.74 0.78 0.82 0.86"
FAM="--families W2 W8 TK50"
for T in 0.05 0.005 0.01 0.02; do
  TAG=$(echo $T | tr -d '.')
  OUT=sweep_bt${TAG}.json
  echo "======== $(date '+%F %T')  t=$T  out=$OUT ========"
  $PY $SW --nimg 20 --t $T $TG $FAM --out $OUT
  echo "-------- $(date '+%F %T')  t=$T done, rc=$? ---------"
done
echo "ALL DONE $(date '+%F %T')"
