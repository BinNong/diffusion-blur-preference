#!/usr/bin/env bash
# ============================================================================
#  de_conv_1 — 环境修复 + 模型加载冒烟测试
#
#  背景：sparktts 里是 torch 2.14.0+cu130，与驱动 575.57.08 (CUDA 12.9) 不兼容，
#        CUDA 不可用。改装 PyPI 版 torch 2.7.1（默认 cu128，驱动 575 可跑）。
#  权重已由 02b_fetch.sh 下载到 ckpt/，这里软链到作者代码期望的位置
#        code/blind_deblur_diffusion/edm/edm_ckpt/
# ============================================================================
set -uo pipefail

DIR="/data/experiment/de_conv_1"
LOG="$DIR/setup_report.txt"
PY="$DIR/venv/bin/python"

hr(){ printf '\n===== %s =====\n' "$1"; }

{
hr "0. 当前状态"
echo "python: $($PY -V 2>&1)"
$PY -c 'import torch;print("torch:",torch.__version__,"| cuda:",torch.cuda.is_available())' 2>&1 | grep -v Warning

hr "1. 安装 torch 2.7.1（清华镜像，PyPI 版默认 cu128）"
$PY -m pip install -q --no-progress-meter \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 2>&1 | tail -10
echo "--- 安装后 ---"
$PY -c 'import torch;print("torch:",torch.__version__,"| cuda:",torch.cuda.is_available(),"| dev:",torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")' 2>&1 | grep -v Warning

hr "2. 权重就位（软链到 edm/edm_ckpt/）"
mkdir -p "$DIR/code/blind_deblur_diffusion/edm/edm_ckpt"
for f in "$DIR"/ckpt/*.pkl; do
    b=$(basename "$f")
    ln -sf "$f" "$DIR/code/blind_deblur_diffusion/edm/edm_ckpt/$b"
done
ls -la "$DIR/code/blind_deblur_diffusion/edm/edm_ckpt/"

hr "3. 冒烟测试：加载 FFHQ-64 并跑一次 score"
cd "$DIR/code/blind_deblur_diffusion" || exit 1
$PY - <<'PYEOF' 2>&1 | grep -viE '^\s*$|warning'
import sys, time, torch
sys.path.insert(0, ".")
from edm import load_local_model
from edm.prior import ScorePrior, get_ode_flow

device = torch.device("cuda")
torch.manual_seed(0)

t0 = time.time()
model = load_local_model("edm-ffhq-64x64-uncond-vp.pkl").to(device)
print(f"加载耗时 {time.time()-t0:.1f}s")

def denoiser(x, t, *a, **k):
    return model(x.to(torch.float32), t, *a, **k).to(torch.float64)

prior = ScorePrior(denoiser=denoiser)
ode_flow, t_spans = get_ode_flow(prior)
print("t_spans =", t_spans)

x = torch.randn(1, 3, 64, 64, device=device)
for s in (0.002, 1.0, 80.0):
    t0 = time.time()
    with torch.no_grad():
        d = denoiser(x, s)
    torch.cuda.synchronize()
    print(f"  sigma={s:<7} denoiser out shape={tuple(d.shape)} mean|d|={float(d.abs().mean()):.4f}  ({(time.time()-t0)*1000:.0f} ms)")

free, total = torch.cuda.mem_get_info()
print(f"显存: free {free/2**30:.2f} GiB / total {total/2**30:.2f} GiB")
print(f"峰值分配: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
print("SMOKE OK")
PYEOF

hr "DONE"
echo "报告: $LOG"
} 2>&1 | tee "$LOG"
