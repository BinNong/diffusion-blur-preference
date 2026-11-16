#!/usr/bin/env bash
set -euo pipefail
cd /data/experiment/de_conv_1
./venv2/bin/python repro/probe_adm_numeric.py > /data/experiment/de_conv_1/results/log_adm_numeric.txt 2>&1
./venv2/bin/python repro/sweep_adm_bsds.py --model imagenet64_uncond > /data/experiment/de_conv_1/results/log_adm64_bsds.txt 2>&1
./venv2/bin/python repro/sweep_adm_bsds.py --model imagenet128_cond --smoke > /data/experiment/de_conv_1/results/log_adm128_smoke.txt 2>&1
./venv2/bin/python repro/sweep_adm_bsds.py --model imagenet128_cond > /data/experiment/de_conv_1/results/log_adm128_bsds.txt 2>&1
