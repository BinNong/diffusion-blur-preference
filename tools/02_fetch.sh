#!/usr/bin/env bash
# ============================================================================
#  de_conv_1 — 拉取代码与预训练权重（服务器端）
#
#  包含：
#    ① NVlabs/edm                 —— EDM 框架（论文用的 FFHQ-64 / AFHQ-64 / ImageNet-64）
#    ② mh-nguyen712/blind_deblur_diffusion —— Morse-Bott 论文作者官方代码
#                                       （含他们实现势能的确切方式，比我们猜要可靠）
#    ③ 三个 EDM 64x64 预训练权重
#
#  说明：论文的所有诊断实验（Fig.4 Hessian 谱、Fig.6/7 势能-模糊曲线、
#        Fig.9/10 marginalized profile）都在 64x64 上做，N=12288。
#        FFHQ-256（N=196608）只用于 Fig.6/7 的一个子图，且 256x256 的稠密
#        Hessian 在 8GB 卡上不可行，故本阶段【不下载】FFHQ-256。
#
#  用法： PROJ=de_conv_1 bash 02_fetch.sh
# ============================================================================
set -uo pipefail

PROJ="${PROJ:-de_conv_1}"
EXP_ROOT="${EXP_ROOT:-/data/experiment}"
DIR="$EXP_ROOT/$PROJ"
LOG="$DIR/fetch_report.txt"

hr(){ printf '\n===== %s =====\n' "$1"; }

dl(){ # dl <url> <dest>
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then
        echo "已存在，跳过: $dest ($(du -h "$dest" | cut -f1))"
        return 0
    fi
    echo "下载: $url"
    if curl -fL --retry 3 --retry-delay 2 --connect-timeout 20 \
            -o "$dest.part" "$url"; then
        mv "$dest.part" "$dest"
        echo "  -> 完成 $(du -h "$dest" | cut -f1)"
        return 0
    else
        echo "  -> 失败: $url"
        rm -f "$dest.part"
        return 1
    fi
}

{
hr "0. 前置检查"
[ -d "$DIR" ] || { echo "!! 先跑 01_bootstrap.sh"; exit 1; }
PY="$DIR/venv/bin/python"
[ -x "$PY" ] || { echo "!! venv 不存在，先跑 01_bootstrap.sh"; exit 1; }
echo "python: $PY"
cd "$DIR" || exit 1

hr "1. 克隆代码仓库"
mkdir -p "$DIR/code"
if [ -d "$DIR/code/edm/.git" ]; then
    echo "edm 已存在"
else
    git clone --depth 1 https://github.com/NVlabs/edm.git "$DIR/code/edm" 2>&1 | tail -3
fi
if [ -d "$DIR/code/blind_deblur_diffusion/.git" ]; then
    echo "blind_deblur_diffusion 已存在"
else
    git clone --depth 1 https://github.com/mh-nguyen712/blind_deblur_diffusion.git \
        "$DIR/code/blind_deblur_diffusion" 2>&1 | tail -3
fi

hr "2. 作者代码结构（看他们如何实现势能）"
find "$DIR/code/blind_deblur_diffusion" -maxdepth 2 -type f -name '*.py' 2>/dev/null | head -20
echo "--- README 摘要 ---"
head -60 "$DIR/code/blind_deblur_diffusion/README.md" 2>/dev/null || echo "(no README)"

hr "3. EDM 64x64 预训练权重"
mkdir -p "$DIR/ckpt"
BASE="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"
for m in edm-ffhq-64x64-uncond-vp edm-afhqv2-64x64-uncond-vp edm-imagenet-64x64-cond-adm; do
    dl "$BASE/$m.pkl" "$DIR/ckpt/$m.pkl"
done

hr "4. 权重校验（能否被 torch.load + 结构识别）"
"$PY" - <<'PY' 2>&1
import os, sys, pickle, glob, torch
d = os.path.join(os.environ.get("PROJ_DIR","."), "ckpt")
if not os.path.isdir(d):
    d = "ckpt"
for p in sorted(glob.glob(os.path.join(d, "*.pkl"))):
    sz = os.path.getsize(p)/2**20
    try:
        with open(p,"rb") as f: obj = pickle.load(f)
        keys = list(obj.keys()) if isinstance(obj, dict) else []
        ema = obj.get("ema") if isinstance(obj, dict) else None
        nparam = sum(x.numel() for x in ema.parameters()) if ema is not None else -1
        print(f"OK   {os.path.basename(p):42s} {sz:7.1f} MiB  keys={keys}  ema_params={nparam/1e6:.1f}M")
    except Exception as e:
        print(f"FAIL {os.path.basename(p):42s} {sz:7.1f} MiB  -> {type(e).__name__}: {e}")
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY

hr "5. EDM 冒烟测试（加载模型 + 跑一次 score）"
"$PY" - <<'PY' 2>&1
import os, sys, pickle, torch
sys.path.insert(0, "code/edm")
try:
    import dnnlib, torch_utils
except Exception as e:
    print("EDM import 失败（可能需要把 edm 目录加入 sys.path 或用其内部模块）:", e)
    raise SystemExit(0)
cands = [p for p in ["ckpt/edm-ffhq-64x64-uncond-vp.pkl",
                     "ckpt/edm-afhqv2-64x64-uncond-vp.pkl",
                     "ckpt/edm-imagenet-64x64-cond-adm.pkl"] if os.path.exists(p)]
if not cands:
    print("没有可用权重"); raise SystemExit(0)
p = cands[0]
print("使用:", p)
with open(p,"rb") as f: obj = pickle.load(f)
net = obj["ema"].cuda().eval()
x = torch.randn(1,3,64,64, device="cuda")
sigma = torch.tensor([1.0], device="cuda")
with torch.no_grad():
    if net.label_dim:   # 条件模型
        lbl = torch.zeros(1, dtype=torch.long, device="cuda")
        out = net(x, sigma, class_labels=lbl)
    else:
        out = net(x, sigma)
torch.cuda.synchronize()
print("score 输出 shape:", tuple(out.shape), " mean|out| =", float(out.abs().mean()))
print("SMOKE OK")
PY

hr "DONE"
echo "报告已写入: $LOG"
} 2>&1 | tee "$LOG"
