#!/usr/bin/env bash
set -euo pipefail

# 国内镜像与下载优化（优先在进程最早阶段设置）
export HUGGINGFACE_HUB_ENDPOINT=${HUGGINGFACE_HUB_ENDPOINT:-https://hf-mirror.com}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# 传输/容错优化
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export HF_HUB_TIMEOUT=${HF_HUB_TIMEOUT:-120}

# 缓存目录：优先内存盘，其次 /tmp（避免长期落盘）
if [ -d "/dev/shm" ]; then
  CACHE_DIR="/dev/shm/hf_cache"
else
  CACHE_DIR="/tmp/hf_cache"
fi
mkdir -p "$CACHE_DIR"
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$CACHE_DIR}
# v5 兼容：使用 HF_HOME 统一缓存主目录
export HF_HOME=${HF_HOME:-$CACHE_DIR}

# 如需代理请取消以下两行注释
# export HTTPS_PROXY=http://your-proxy:port
# export HTTP_PROXY=http://your-proxy:port

echo "[start.sh] HF endpoint: $HUGGINGFACE_HUB_ENDPOINT"
echo "[start.sh] Cache dir: $CACHE_DIR (HF_HOME)"

exec python3 run.py


