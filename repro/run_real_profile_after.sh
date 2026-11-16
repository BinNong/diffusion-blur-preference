#!/usr/bin/env bash
set -euo pipefail
cd /data/experiment/de_conv_1
WAIT_PID="${1:?usage: run_real_profile_after.sh PID_OF_REAL_SWEEP}"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
./venv2/bin/python repro/sweep_real_profile.py --smoke > /data/experiment/de_conv_1/results/log_real_bsds_profile_smoke.txt 2>&1
./venv2/bin/python repro/sweep_real_profile.py --out real_bsds_profile.json > /data/experiment/de_conv_1/results/log_real_bsds_profile.txt 2>&1
