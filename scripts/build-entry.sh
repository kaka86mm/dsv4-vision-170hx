#!/bin/bash
# 常驻编译容器入口: 重启自动续编 (ccache+ninja增量), 写满本本见 docs/runbook §3
set -x
export http_proxy="${BUILD_PROXY:-http://<代理机>}"
export https_proxy="$http_proxy"
export no_proxy=localhost,127.0.0.1,.ustc.edu.cn
export PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/lib/ccache:/root/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda CCACHE_DIR=/ccache
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=48
export CMAKE_BUILD_TYPE=Release VLLM_TARGET_DEVICE=cuda
cd /src

# [坑] pip 构建子进程会换 HOME, 必须 --system 级
git config --system --add safe.directory /src 2>/dev/null || true

[ -x /opt/venv/bin/pip ] || {
  apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv git build-essential \
    curl ca-certificates ccache cmake ninja-build
  python3.12 -m venv /opt/venv
  pip install --no-cache-dir -U pip "setuptools>=77.0.3,<81.0.0" wheel \
    "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" ninja "cmake>=3.26.1" \
    "packaging>=24.2" jinja2
  RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static \
    RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup \
    /tmp/rustup-init -y --default-toolchain 1.95 --profile minimal
  git config --global http.version HTTP/1.1
  git config --global http.postBuffer 524288000
  git config --global http.lowSpeedLimit 1000
  git config --global http.lowSpeedTime 60
  git config --global url.https://gh-proxy.com/https://github.com/.insteadOf https://github.com/
  pip install --no-cache-dir --no-index --find-links /tmp/wheels \
    torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0
}

for i in 1 2 3 4 5; do
  pip install --no-cache-dir -e . --no-build-isolation \
    && { echo BUILD_OK > /src/.build_status; exit 0; }
  echo "compile retry $i"; sleep 15
done
echo BUILD_FAILED > /src/.build_status
