#!/bin/bash
# DSV4-Flash-Vision-Exp 生产启动脚本 (4x CMP 170HX, PP4)
# 用法: 改 MODEL_DIR 后 sudo ./launch.sh
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/<user>/models/dsv4-flash-vision-exp}"
IMAGE="${IMAGE:-dsv4-vision:sm80}"
PORT="${PORT:-8096}"

docker stop -t 60 dsv4-vision >/dev/null 2>&1 || true
docker rm dsv4-vision >/dev/null 2>&1 || true

docker run -d --name dsv4-vision --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple \
  -e VLLM_PP_LAYER_PARTITION=12,11,11,9 \
  -v "$MODEL_DIR":/model:ro \
  --ipc=host -p "$PORT":8000 \
  --entrypoint /opt/venv/bin/vllm "$IMAGE" serve /model \
  --served-model-name dsv4-vision \
  --pipeline-parallel-size 4 \
  --max-model-len 524288 --gpu-memory-utilization 0.93 \
  --kv-cache-dtype fp8 --block-size 256 \
  --max-num-batched-tokens 2048 --max-num-seqs 16 \
  --trust-remote-code --disable-custom-all-reduce \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --tokenizer-mode deepseek_v4 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32,64],"max_cudagraph_capture_size":64}' \
  --hf-overrides '{"head_dtype": "float32", "architectures": ["DeepseekV4ForConditionalGeneration"]}' \
  --host 0.0.0.0 --port 8000

echo "dsv4-vision on :$PORT (PP4 512k DSpark-n3, ~10min 启动)"
echo "就绪检查: curl -s localhost:$PORT/health"
