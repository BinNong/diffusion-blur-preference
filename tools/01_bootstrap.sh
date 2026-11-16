#!/usr/bin/env bash
# ============================================================================
#  de_conv_1 — 服务器环境侦察 + 引导脚本
#
#  用途：在 172.16.65.85 上一次性完成
#        ① 环境侦察（GPU / torch / conda / 磁盘 / 出网）
#        ② 建立项目目录与虚拟环境（从 conda env sparktts 克隆）
#        ③ 自检（能否 import torch、能否看到 GPU）
#
#  本脚本【不下载任何模型权重】——模型下载是独立的、较重的步骤，
#  由 02_fetch_models.sh 单独执行。
#
#  用法： bash 01_bootstrap.sh
# ============================================================================
set -uo pipefail

PROJ="${PROJ:-de_conv_1}"
EXP_ROOT="${EXP_ROOT:-/data/experiment}"
DIR="$EXP_ROOT/$PROJ"
LOG="$DIR/bootstrap_report.txt"

hr(){ printf '\n===== %s =====\n' "$1"; }

{
hr "0. 基本信息"
echo "date      : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "host      : $(hostname)"
echo "user      : $(whoami)"
echo "uname     : $(uname -a)"
echo "shell     : $SHELL"

hr "1. GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,cuda_version \
               --format=csv,noheader
    echo "--- utilization ---"
    nvidia-smi --query-gpu=utilization.gpu,temperature.gpu --format=csv,noheader
    echo "--- processes on GPU ---"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "(none)"
else
    echo "!! nvidia-smi NOT FOUND"
fi

hr "2. CUDA toolkit"
ls -d /usr/local/cuda* 2>/dev/null || echo "(no /usr/local/cuda*)"
nvcc --version 2>/dev/null | tail -2 || echo "(nvcc not on PATH)"

hr "3. Conda"
if command -v conda >/dev/null 2>&1; then
    echo "conda: $(conda --version)"
    echo "--- envs ---"
    conda env list
    echo "--- sparktts python ---"
    conda run -n sparktts python -c '
import sys
print("python    :", sys.version.split()[0])
print("executable:", sys.executable)
for m in ("torch","numpy","scipy","matplotlib","PIL","torchvision"):
    try:
        mod=__import__(m); print(f"{m:11s}:", getattr(mod,"__version__","?"))
    except Exception as e:
        print(f"{m:11s}: MISSING ({type(e).__name__})")
try:
    import torch
    print("cuda avail :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device     :", torch.cuda.get_device_name(0))
        print("capability :", torch.cuda.get_device_capability(0))
        print("torch cuda :", torch.version.cuda)
        x=torch.randn(1024,1024,device="cuda"); y=x@x
        torch.cuda.synchronize()
        print("matmul ok  :", float(y.sum())==float(y.sum()))
except Exception as e:
    print("torch check failed:", e)
' 2>&1
else
    echo "!! conda NOT FOUND"
fi

hr "4. 磁盘"
df -h "$EXP_ROOT" 2>/dev/null || df -h /
echo "--- /data/experiment 内容 ---"
ls -la "$EXP_ROOT" 2>/dev/null | head -30

hr "5. 出网能力（决定能否直接 git clone / 下权重）"
for u in https://github.com https://huggingface.co https://pypi.org https://raw.githubusercontent.com; do
    printf '%-40s ' "$u"
    curl -s -o /dev/null -w 'HTTP %{http_code}  %{time_total}s\n' --max-time 15 "$u" 2>/dev/null \
        || echo "FAILED"
done
echo "--- git ---"
git --version 2>/dev/null || echo "(git missing)"
echo "--- HF 镜像环境变量 ---"
env | grep -iE 'HF_|HUGGINGFACE|PIP_INDEX|PROXY' || echo "(none set)"

hr "6. 已有相关代码/权重（避免重复下载）"
find "$EXP_ROOT" -maxdepth 3 -type d \( -name 'edm' -o -name 'score_sde*' -o -name '*blind*' -o -name '*.pth' -o -name 'checkpoints' \) 2>/dev/null | head -20 || true
find "$EXP_ROOT" -maxdepth 4 -name '*.pth' -o -maxdepth 4 -name '*.pt' 2>/dev/null | head -20 || true

hr "7. 建立项目目录"
if [ -d "$DIR" ]; then
    echo "目录已存在: $DIR"
    ls -la "$DIR"
else
    mkdir -p "$DIR" && echo "已创建: $DIR"
fi

hr "8. 建立虚拟环境（从 sparktts 克隆）"
if [ -d "$DIR/venv" ]; then
    echo "venv 已存在: $DIR/venv"
else
    echo "conda create -p $DIR/venv --clone sparktts ..."
    conda create -p "$DIR/venv" --clone sparktts -y 2>&1 | tail -20
fi

hr "9. venv 自检"
if [ -x "$DIR/venv/bin/python" ]; then
    "$DIR/venv/bin/python" - <<'PY' 2>&1
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
    # 关键可行性测试：12288x12288 float32 = 576 MB（论文 Fig.4 的 Hessian 规模）
    try:
        N=12288
        A=torch.randn(N,N,device="cuda")
        A=(A+A.T)*0.5
        s=torch.linalg.eigvalsh(A)
        torch.cuda.synchronize()
        print(f"eigvalsh {N}x{N} float32 : OK (min={float(s[0]):.4g}, max={float(s[-1]):.4g})")
        del A,s; torch.cuda.empty_cache()
    except Exception as e:
        print(f"eigvalsh {N}x{N} float32 : FAILED -> {type(e).__name__}: {e}")
        try:
            N=6144
            A=torch.randn(N,N,device="cuda"); A=(A+A.T)*0.5
            s=torch.linalg.eigvalsh(A); torch.cuda.synchronize()
            print(f"eigvalsh {N}x{N} float32 : OK (fallback scale)")
        except Exception as e2:
            print(f"eigvalsh {N}x{N} float32 : FAILED too -> {e2}")
PY
else
    echo "!! venv python 不存在，clone 可能失败"
fi

hr "10. 显存实测（决定 Hessian 规模上限）"
if [ -x "$DIR/venv/bin/python" ]; then
"$DIR/venv/bin/python" - <<'PY' 2>&1
import torch
if not torch.cuda.is_available():
    print("no cuda"); raise SystemExit
free, total = torch.cuda.mem_get_info()
print(f"free {free/2**30:.2f} GiB / total {total/2**30:.2f} GiB")
for N in (4096, 6144, 8192, 12288):
    for dt in (torch.float32, torch.float64):
        need = N*N*torch.tensor([],dtype=dt).element_size()
        ok = "OK " if need < free*0.6 else "TIGHT"
        print(f"  N={N:6d} {str(dt).split('.')[-1]:8s} need {need/2**30:7.2f} GiB  -> {ok}")
PY
fi

hr "DONE"
echo "报告已写入: $LOG"

} 2>&1 | tee "$LOG"
