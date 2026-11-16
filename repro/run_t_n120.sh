#!/bin/bash
# 第九组：b(u) 的 t 依赖搬到 N=120（判据跑前写定，见 repro/judge_t_n120.py docstring）
#
# 预官：2026-10-26 15:35:11，md5 eed7e7969129e30febf1a512e007f286
#       （在看到任何 N=120 的 t>0.002 数据之前）
#
# 规模（实测，来自 log_law_n120.txt 的墙钟）：N=120 每算子 9.4–10.1 分钟
#   （48 算子 7h31m / 21 算子 3h32m / 60 算子 9h53m），所以每档 60 算子约 9.8 h，
#   五档约 49 h。**tuning 每族实际给 20 个算子而不是 17 个**（17 个 target + 3 个
#   派生点，含 u=0.08 的 @raw），所以一档是 3 族 x 20 = 60 算子，不是 51。
#   （本注释在启动后修正：原写「51 算子，约 7.9h」两处都错。已执行的命令行逐字未变，
#    由 log_t_n120.txt 的 [cfg] 行证实地跑到 60 个算子。）
#
# 原则：每个命令自己的 --out，崩了只损失一个文件；顺序按价值排——
#   t=0.05 漂移最大、最能判决 NT2，放第一个；t=0.16 是高斯极限渐近，放最后。
#   targets / families / kunif_ref / 提取方式与已发布的 N=20 t 研究逐字相同，
#   只把 --nimg 20 改成 --nimg 120。
#
# 用法（服务器上）：nohup bash repro/run_t_n120.sh > results/log_t_n120.txt 2>&1 &
set -u
cd /data/experiment/de_conv_1 || exit 1
PY=./venv2/bin/python
LOG=results/log_t_n120.txt

say() { echo ""; echo "########## $(date '+%F %T') $* ##########"; }

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
say "启动；GPU 空闲 ${FREE} MiB"
# 阈值 1200：实测本任务 nimg=120 占 1778 MiB（显存与 nimg 无关，120 张只多 ~12 MB 数组）
if [ "$FREE" -lt 1200 ]; then
    say "空闲显存不足 1200 MiB，中止（不抢占别人的任务）"
    exit 1
fi

# 与 N=20 的 t 研究逐字相同的 17 个 target（kunif，不是 u）
TARGETS="0.15 0.18 0.22 0.26 0.30 0.34 0.38 0.42 0.46 0.52 0.58 0.65 0.70 0.74 0.78 0.82 0.86"

for T in 0.05 0.02 0.01 0.005 0.16; do
    say "t=${T}: sweep_bt120_${T}.json  (3 族 x 20 算子 = 60 算子，约 9.8h)"
    $PY repro/sweep_law_test.py --nimg 120 --t "$T" --targets $TARGETS \
        --families W2 W8 TK50 --out "sweep_bt120_${T}.json" || say "t=${T} 失败"
done

say "全部 t 档结束；跑预官裁判："
say "  ./venv2/bin/python repro/judge_t_n120.py --dump results/fig_t_n120.json"
