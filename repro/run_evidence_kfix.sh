#!/bin/bash
# 正确核重跑 §7 那两个「错核值」：等 GPU 空出来 → 两臂 → 离线裁判
#
# 预官：2026-10-28，md5 744bc4f4d785d53706e26f2dd2c85c78（repro/evidence_scale2_kfix.py）
#       判据 K0–K4 原文见该脚本 docstring 与 RESULTS §3.9(bn)
#       离线裁判 repro/judge_evidence_scale2_kfix.py 的 K1 容差修订见其 docstring
#
# 为什么等：t=0.16 那一档（repro/run_t_n120.sh 的最后一档）还在跑，GPU 只有一个，
#   不能抢。哨兵文件是 results/sweep_bt120_0.16.json。
#
# 用法（服务器上）：nohup bash repro/run_evidence_kfix.sh > results/log_evidence_kfix.txt 2>&1 &
set -u
cd /data/experiment/de_conv_1 || exit 1
PY=./venv2/bin/python
LOG=results/log_evidence_kfix.txt
SENT=/data/experiment/de_conv_1/results/sweep_bt120_0.16.json

say() { echo ""; echo "########## $(date '+%F %T') $* ##########"; }

say "启动；等 t=0.16 落地（哨兵 $SENT）"
# 最多等 6 小时（360 分钟 × 60 s）
for i in $(seq 1 360); do
    sleep 60
    if [ -f "$SENT" ]; then
        say "哨兵出现（第 $i 分钟）——GPU 应已空出"
        break
    fi
    if [ "$i" -eq 360 ]; then
        say "等超 6 小时，哨兵仍未出现——中止（不抢 GPU）"
        exit 1
    fi
done

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
say "GPU 空闲 ${FREE} MiB"
if [ "$FREE" -lt 1200 ]; then
    say "空闲显存不足 1200 MiB，中止（不抢占别人的任务）"
    exit 1
fi

say "臂 1（ffhq，8 图 × 5 噪声 × 3 强度 = 120 格，约 2h）"
$PY repro/evidence_scale2_kfix.py --out evidence_scale2_kfix.json || say "臂 1 失败"

say "臂 2（afhqv2，约 2h）"
$PY repro/evidence_scale2_kfix.py --ckpt edm-afhqv2-64x64-uncond-vp.pkl \
    --pool /tmp/pool_afhqv2-64x64-vp_256_s0.pt --out evidence_scale2_afq_kfix.json \
    || say "臂 2 失败"

say "跑预官裁判（零 GPU）："
say "  ./venv2/bin/python repro/judge_evidence_scale2_kfix.py"
$PY repro/judge_evidence_scale2_kfix.py

say "全部结束"
