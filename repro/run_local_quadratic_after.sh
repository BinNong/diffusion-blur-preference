#!/usr/bin/env bash
set -euo pipefail
cd /data/experiment/de_conv_1
WAIT_PID="${1:?usage: run_local_quadratic_after.sh PID_OF_PROFILE_CHAIN}"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
./venv2/bin/python repro/probe_local_quadratic.py > /data/experiment/de_conv_1/results/log_local_quadratic.txt 2>&1
