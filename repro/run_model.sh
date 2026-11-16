#!/bin/bash
# 第九组：b(u) 是否随**先验**（模型）变？预官 2026-10-19 12:19，md5 774d9acd...
# 2x2 因子表缺一格（ffhq+ffhq 即第八组基线）。这里跑剩下两格，都用 afhqv2 图池，
# 于是「格子B - 格子C」= 固定图像下的先验效应（配对）。
cd /data/experiment/de_conv_1
PY=./venv2/bin/python
SW=repro/sweep_law_test.py
POOL=/tmp/pool_afhqv2-64x64-vp_256_s0.pt
TG="--targets 0.15 0.18 0.22 0.26 0.30 0.34 0.38 0.42 0.46 0.52 0.58 0.65 0.70 0.74 0.78 0.82 0.86"
FAM="--families W2 W8 TK50"
for CELL in "afq_afq:edm-afhqv2-64x64-uncond-vp.pkl" "ffq_afq:edm-ffhq-64x64-uncond-vp.pkl"; do
  NAME=${CELL%%:*}
  CK=${CELL##*:}
  OUT=sweep_m_${NAME}.json
  echo "======== $(date '+%F %T')  格子 $NAME  ckpt=$CK  pool=afhqv2  out=$OUT ========"
  $PY $SW --nimg 20 --t 0.002 --ckpt $CK --pool $POOL $TG $FAM --out $OUT
  echo "-------- $(date '+%F %T')  格子 $NAME done, rc=$? ---------"
done
echo "ALL DONE $(date '+%F %T')"
