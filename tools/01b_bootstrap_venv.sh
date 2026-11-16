#!/usr/bin/env bash
# ============================================================================
#  de_conv_1 — 01_bootstrap.sh 的补丁版
#  修正：conda 在非交互 SSH 下不在 PATH，需先 source。
#        另外 GPU 当前仅剩约 2.5 GiB（被 Janus 项目占用 5.2 GiB），
#        显存测试改为如实报告可用余量，并自动找出可行的 Hessian 规模。
# ============================================================================
set -uo pipefail

PROJ="${PROJ:-de_conv_1}"
EXP_ROOT="${EXP_ROOT:-/data/experiment}"
DIR="$EXP_ROOT/$PROJ"
LOG="$DIR/bootstrap_report.txt"
CONDA=/home/vr301/miniconda3/bin/conda

hr(){ printf '\n===== %s =====\n' "$1"; }

{
hr "0. conda 初始化"
export PATH="/home/vr301/miniconda3/bin:$PATH"
echo "conda: $("$CONDA" --version 2>&1)"
"$CONDA" env list

hr "1. 当前 GPU 余量（重要：Janus 项目正占用 5.2 GiB）"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "(none)"

hr "2. sparktts 环境内容"
"$CONDA" run -n sparktts python -c '
import sys
print("python    :", sys.version.split()[0])
print("executable:", sys.executable)
for m in ("torch","numpy","scipy","matplotlib","PIL","torchvision","einops"):
    try:
        mod=__import__(m); print(f"{m:11s}:", getattr(mod,"__version__","?"))
    except Exception as e:
        print(f"{m:11s}: MISSING ({type(e).__name__})")
import torch
print("cuda avail :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device     :", torch.cuda.get_device_name(0))
    print("torch cuda :", torch.version.cuda)
' 2>&1

hr "3. 建立虚拟环境（从 sparktts 克隆）"
if [ -d "$DIR/venv" ]; then
    echo "venv 已存在"
else
    "$CONDA" create -p "$DIR/venv" --clone sparktts -y 2>&1 | tail -15
fi

hr "4. venv 自检 + 显存可行性（关键）"
"$DIR/venv/bin/python" - <<'PY' 2>&1
import sys, torch
print("python    :", sys.version.split()[0])
print("executable:", sys.executable)
for m in ("torch","numpy","scipy","matplotlib","PIL","torchvision","einops"):
    try:
        mod=__import__(m); print(f"{m:11s}:", getattr(mod,"__version__","?"))
    except Exception as e:
        print(f"{m:11s}: MISSING ({type(e).__name__})")
if not torch.cuda.is_available():
    print("!! CUDA 不可用"); raise SystemExit
free, total = torch.cuda.mem_get_info()
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"free {free/2**30:.2f} GiB / total {total/2**30:.2f} GiB")

print("\n--- 稠密 Hessian 规模上限（论文 Fig.4 用 N=12288）---")
for N in (3072, 6144, 8192, 12288):
    for dt in (torch.float32, torch.float64):
        need = N*N*torch.tensor([],dtype=dt).element_size()
        verdict = "OK" if need < free*0.5 else ("TIGHT" if need < free*0.8 else "NO")
        print(f"  N={N:6d}  {str(dt).split('.')[-1]:8s}  need {need/2**30:6.2f} GiB  -> {verdict}")

print("\n--- 实测 GPU eigvalsh ---")
for N in (3072, 6144, 8192):
    try:
        A = torch.randn(N, N, device="cuda"); A = (A + A.T) * 0.5
        s = torch.linalg.eigvalsh(A); torch.cuda.synchronize()
        print(f"  N={N:6d} GPU eigvalsh : OK")
        del A, s; torch.cuda.empty_cache()
    except Exception as e:
        print(f"  N={N:6d} GPU eigvalsh : FAIL -> {type(e).__name__}")

print("\n--- 实测 CPU eigvalsh（显存不够时的退路；注意机器仅 15 GiB RAM）---")
import time
for N in (6144, 12288):
    try:
        t0=time.time()
        A = torch.randn(N, N, dtype=torch.float32); A = (A + A.T) * 0.5
        s = torch.linalg.eigvalsh(A)
        print(f"  N={N:6d} CPU eigvalsh : OK  ({time.time()-t0:.1f}s)")
        del A, s
    except Exception as e:
        print(f"  N={N:6d} CPU eigvalsh : FAIL -> {type(e).__name__}: {e}")

print("\n--- 模型前向/反向显存占用（EDM 64x64, 63M 参数）---")
try:
    x = torch.randn(1, 3, 64, 64, device="cuda", requires_grad=True)
    y = (x * 1.7).sum()
    y.backward()
    torch.cuda.synchronize()
    print(f"  一次 backward 后峰值: {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
except Exception as e:
    print("  FAIL:", e)
PY

hr "DONE"
echo "报告: $LOG"
} 2>&1 | tee "$LOG"
echo "[tee exit: $?]"
