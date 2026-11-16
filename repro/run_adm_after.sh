#!/usr/bin/env bash
set -euo pipefail
cd /data/experiment/de_conv_1
WAIT_PID="${1:?usage: run_adm_after.sh PID_OF_LOCAL_QUADRATIC_CHAIN}"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
for ((i=0;i<600;i++)); do
  if [[ -f ckpt/adm_imagenet64_uncond_100M_1500K.pt ]]; then break; fi
  sleep 120
done
test -f ckpt/adm_imagenet64_uncond_100M_1500K.pt
./venv2/bin/python repro/sweep_adm_bsds.py --model imagenet64_uncond --smoke > /data/experiment/de_conv_1/results/log_adm64_smoke.txt 2>&1
./venv2/bin/python repro/sweep_adm_bsds.py --model imagenet64_uncond > /data/experiment/de_conv_1/results/log_adm64_bsds.txt 2>&1
for ((i=0;i<600;i++)); do
  if [[ -f ckpt/adm_imagenet128_cond_128x128_diffusion.pt ]]; then break; fi
  sleep 120
done
test -f ckpt/adm_imagenet128_cond_128x128_diffusion.pt
./venv2/bin/python repro/sweep_adm_bsds.py --model imagenet128_cond --smoke > /data/experiment/de_conv_1/results/log_adm128_smoke.txt 2>&1
./venv2/bin/python repro/sweep_adm_bsds.py --model imagenet128_cond > /data/experiment/de_conv_1/results/log_adm128_bsds.txt 2>&1
