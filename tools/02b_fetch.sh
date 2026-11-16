#!/usr/bin/env bash
# ============================================================================
#  de_conv_1 — 拉取代码与预训练权重（服务器端，v2）
#
#  已验证可用的源（服务器白名单式出网）：
#    - codeload.github.com          : GitHub 仓库 tarball（git 协议走 ghproxy 太慢，改用 tarball）
#    - nvlabs-fi-cdn.nvidia.com     : EDM 官方预训练权重
#    - pypi.tuna.tsinghua.edu.cn    : pip（~/.pip/pip.conf 已配置）
#
#  用法： bash 02b_fetch.sh
# ============================================================================
set -uo pipefail

PROJ="${PROJ:-de_conv_1}"
DIR="/data/experiment/$PROJ"
LOG="$DIR/fetch_report.txt"
CODE="$DIR/code"
CKPT="$DIR/ckpt"

hr(){ printf '\n===== %s =====\n' "$1"; }

{
hr "0. 目录"
mkdir -p "$CODE" "$CKPT"
cd "$CODE" || exit 1
echo "工作目录: $CODE"

hr "1. 代码 tarball"
fetch_tar(){ # fetch_tar <user/repo> <destname>
    local repo="$1" name="$2"
    if [ -d "$name" ] && [ -n "$(ls -A "$name" 2>/dev/null)" ]; then
        echo "已存在，跳过: $name"; return 0
    fi
    local url="https://codeload.github.com/$repo/tar.gz/refs/heads/main"
    echo "下载 $repo (main) ..."
    if ! curl -fL --retry 3 --connect-timeout 20 --max-time 300 -o "/tmp/$name.tgz" "$url"; then
        echo "  main 分支失败，试 master"
        url="https://codeload.github.com/$repo/tar.gz/refs/heads/master"
        curl -fL --retry 3 --connect-timeout 20 --max-time 300 -o "/tmp/$name.tgz" "$url" || { echo "  失败"; return 1; }
    fi
    mkdir -p "$name" && tar xzf "/tmp/$name.tgz" -C "$name" --strip-components=1
    echo "  -> $(find "$name" -type f | wc -l) 个文件"
}

fetch_tar "mh-nguyen712/blind_deblur_diffusion" "blind_deblur_diffusion"
fetch_tar "NVlabs/edm" "edm"

hr "2. 作者代码结构（看他们如何实现势能）"
find blind_deblur_diffusion -maxdepth 2 -type f 2>/dev/null | head -30
echo "--- README 前 50 行 ---"
head -50 blind_deblur_diffusion/README.md 2>/dev/null || echo "(无 README)"

hr "3. EDM 64x64 预训练权重"
cd "$CKPT" || exit 1
BASE="https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"
for m in edm-ffhq-64x64-uncond-vp edm-afhqv2-64x64-uncond-vp edm-imagenet-64x64-cond-adm; do
    if [ -s "$m.pkl" ]; then
        echo "已存在: $m.pkl ($(du -h "$m.pkl" | cut -f1))"
    else
        echo "下载 $m ..."
        curl -fL --retry 3 --connect-timeout 20 --max-time 1800 -o "$m.pkl" "$BASE/$m.pkl" \
            && echo "  -> $(du -h "$m.pkl" | cut -f1)" \
            || { echo "  失败"; rm -f "$m.pkl"; }
    fi
done
ls -la

hr "DONE"
echo "报告: $LOG"
} 2>&1 | tee "$LOG"
