#!/bin/bash
# 第十一组：定律的样本量升级 20 -> 120 张图（跑前写定 2026-10-24 21:34，
#         预官 md5 cdcc8e4e630aed35759f999d57296a99，见 repro/judge_law_n120.py）
#
# 原则：每个命令自己的 --out，崩了只损失一个文件；链条顺序按价值排
#   （D2 样本外检验 > D3 闭式 > D1 图 1）。算子表/族/targets/kunif_ref/t 与
#   已发布运行逐字相同，只把 --nimg 20 改成 --nimg 120。
#
# 用法（服务器上）：  nohup bash repro/run_law_n120.sh > results/log_law_n120.txt 2>&1 &
set -u
cd /data/experiment/de_conv_1 || exit 1
PY=./venv2/bin/python
LOG=results/log_law_n120.txt

say() { echo ""; echo "########## $(date '+%F %T') $* ##########"; }

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
say "启动；GPU 空闲 ${FREE} MiB"
# 阈值 1200：实测本任务 nimg=2 时占 1040 MiB，而显存与 nimg 无关（logp 按 B=4 批，
# 随 nimg 变的只有 xs/y/lp 三个数组，120 张 64x64 float64 合计约 12 MB）。留 160 MiB 余量。
if [ "$FREE" -lt 1200 ]; then
    say "空闲显存不足 1200 MiB，中止（不抢占别人的任务）"
    exit 1
fi

say "D2: sweep_law_n120.json  (5 新族 x 6 算子 = 30 算子，约 4.1h)"
# 必须显式给 --families！NEW_FAMILIES 已经长到 28 个（第一~六组累加），
# 当年产出 sweep_law_test.json 时它只有 5 个。不带这个参数会跑出 168 个算子，
# 而且算子表和已发布文件不一致，裁判的 R0/R4 全部失真。这个坑是探针抓到的。
$PY repro/sweep_law_test.py --nimg 120 --families W1 W2 W4 B1 M2 \
    --out sweep_law_n120.json || say "D2 失败"

say "D3a: sweep_bdense_n120.json  (3 族 x 16 算子 = 48 算子，约 6.6h)"
$PY repro/sweep_law_test.py --nimg 120 \
    --targets 0.15 0.18 0.22 0.26 0.30 0.34 0.38 0.42 0.46 0.52 0.58 0.65 0.70 \
    --families W2 W8 TK50 --out sweep_bdense_n120.json || say "D3a 失败"

say "D3b: sweep_blow_n120.json  (3 族 x 7 算子 = 21 算子，约 2.9h)"
$PY repro/sweep_law_test.py --nimg 120 \
    --targets 0.74 0.78 0.82 0.86 \
    --families W2 W8 TK50 --out sweep_blow_n120.json || say "D3b 失败"

say "D1a: sweep_pshape_n120_A.json  (parts A，约 3.3h)"
$PY repro/sweep_profile_shape.py --nimg 120 --parts A \
    --out sweep_pshape_n120_A.json || say "D1a 失败"

say "D1b: sweep_pshape_n120_BCV.json  (parts B C V，约 4.9h)"
$PY repro/sweep_profile_shape.py --nimg 120 --parts B C V \
    --out sweep_pshape_n120_BCV.json || say "D1b 失败"

say "全部命令结束；跑裁判："
say "  ./venv2/bin/python repro/judge_law_n120.py --dump results/fig_law_n120.json"
