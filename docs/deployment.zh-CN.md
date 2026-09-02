[English](deployment.md) | [简体中文](deployment.zh-CN.md)

# DSV4-Flash-Vision-Exp 视觉服务部署 Runbook（4× CMP 170HX）

> 目标：在裸的 4× CMP 170HX 机器上，从零部署 DeepSeek-V4-Flash-Vision-Exp 多模态生产服务（文本+图像，PP4，512k 窗口，DSpark 投机解码）。
> 注意：显存只够跑一个模型（权重 167GB 占满 4×64GB）。本服务独占 GPU；跑它之前停掉机器上其他模型容器。
> 机器底座准备（解锁/驱动/Gen2/230W/Docker）如未做过，按 `machine-prep-reference.md` 第 0-3 节执行，本 runbook 从模型下载开始。
> 全部命令实战验证于 2026-09-01/02。遇 [坑] 按提示处理。

## 0. 环境确认

```bash
# 机器底座就位（没做过 → 先走 machine-prep-reference.md §0-3）：
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,power.limit --format=csv,noheader
# 期望: 4× "NVIDIA CMP 170HX, 65536 MiB", gen=2, 230W
docker info >/dev/null && echo docker-ok
sudo docker images | grep nvidia/cuda:13.0.2 || sudo docker pull nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
# GPU 上如有其他模型容器，先停（互斥）:
docker ps --format "{{.Names}}" | grep -E "vllm|glm|dsv4" && docker stop <名字>
```

## 1. 模型下载（~157GB）

```bash
python3 -m venv ~/hfenv 2>/dev/null; ~/hfenv/bin/pip install -q -U huggingface_hub
~/hfenv/bin/pip uninstall -y -q hf_xet hf_transfer   # [坑] Xet协议与镜像不兼容→401
mkdir -p ~/models/dsv4-flash-vision-exp
HF_ENDPOINT=https://hf-mirror.com nohup ~/hfenv/bin/hf download \
  deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --local-dir ~/models/dsv4-flash-vision-exp > /tmp/dl-vision.log 2>&1 &
# 日志出现 "✓ Downloaded" 即完整（hf自带校验）
```

## 1.5 构建物料预备（cp312 轮子 + rustup-init）

```bash
mkdir -p ~/tools/vision-wheels
cat > /tmp/dlwheels.sh <<'EOF'
#!/bin/bash
cd ~/tools/vision-wheels
export http_proxy=http://<代理机> https_proxy=http://<代理机>
for i in $(seq 1 30); do
  ~/hfenv/bin/pip download --no-cache-dir --python-version 3.12 --implementation cp \
    --abi cp312 --only-binary=:all: -d . \
    torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 && { echo DONE > status.txt; exit 0; }
  echo "retry $i"; sleep 10
done
EOF
chmod +x /tmp/dlwheels.sh && /tmp/dlwheels.sh   # ~3.3GB, 重试环扛烂网
# rustup-init (USTC 镜像)
curl -sL -o ~/tools/rustup-init \
  https://mirrors.ustc.edu.cn/rust-static/rustup/dist/x86_64-unknown-linux-gnu/rustup-init
chmod +x ~/tools/rustup-init
```
[坑] 宿主 pip 默认按本机 Python 版本下轮子 — 必须带 `--python-version 3.12 --abi cp312`（容器是 3.12）。

## 2. 源码树组装（vision-v3 配方）

四个部分的叠加，顺序固定：

```bash
mkdir -p ~/tools && cd ~/tools
git clone --branch dsv4-vision-exp --single-branch \
  https://github.com/wtdcode/vllm-backport.git vllm-backport-vision
cd vllm-backport-vision
git config user.email "m@local" && git config user.name "m"

# 2.1 PR #54566 前四个提交（视觉支持基线，截止 9327439714）
git fetch https://gh-proxy.com/https://github.com/vllm-project/vllm.git pull/54566/head:pr54566
git cherry-pick edafe3dbe^..9327439714
# 冲突处理：测试文件与同语义 gate-bias 条件 → 一律 git checkout --theirs

# 2.2 cg 修复 + MTP 启用（从 PR 后续提交单摘）
git cherry-pick 5ab628dd1     # fix breakable cg — 仅2行config，修复图模式下文本乱码
git cherry-pick 2de7255a2     # enable mtp — VL草稿器管线
# [坑] 不要用 PR 最新头（含 upstream main merge，带来 DeepGEMM 硬门，sm80 不可用）

# 2.3 一键打全部 sm80 补丁（Ampere后端+选择器+PP中继+末阶嵌入）
#    sm80-patches.py 自动从 patches/ampere/ 拷入后端文件 — 无需手动获取。
python3 /path/to/dsv4-vision-170hx/scripts/sm80-patches.py .
# 预期输出: "applied: ['sm80-selector', 'pp-relay-empty', 'pp-relay-recv', 'pp-relay-send', 'embed-last-rank']"
git add -A && git commit -m "sm80: ampere+selector+PP relay+embed"
```

## 3. 构建（常驻编译容器，重启自动续编）

```bash
cat > ~/tools/vision-build-entry.sh <<'EOF'
#!/bin/bash
set -x
export http_proxy=http://<代理机> https_proxy=http://<代理机> no_proxy=localhost,127.0.0.1,.ustc.edu.cn
export PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/lib/ccache:/root/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda CCACHE_DIR=/ccache
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=48
export CMAKE_BUILD_TYPE=Release VLLM_TARGET_DEVICE=cuda
cd /src
git config --system --add safe.directory /src   # [坑] 必须--system级（pip子进程换HOME）
[ -x /opt/venv/bin/pip ] || {
  apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv git build-essential curl ca-certificates ccache cmake ninja-build
  python3.12 -m venv /opt/venv
  pip install --no-cache-dir -U pip "setuptools>=77.0.3,<81.0.0" wheel "setuptools-scm>=8.0" "setuptools-rust>=1.9.0" ninja "cmake>=3.26.1" "packaging>=24.2" jinja2
  /tmp/rustup-init -y --default-toolchain 1.95 --profile minimal
  RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static /tmp/rustup-init -y --default-toolchain 1.95 --profile minimal
  pip install --no-cache-dir --no-index --find-links /tmp/wheels torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0
}
for i in 1 2 3 4 5; do
  pip install --no-cache-dir -e . --no-build-isolation && { echo BUILD_OK > /src/.build_status; exit 0; }
  echo "retry $i"; sleep 15
done
echo BUILD_FAILED > /src/.build_status
EOF
chmod +x ~/tools/vision-build-entry.sh

docker run -d --name vision-build --restart unless-stopped \
  -v ~/tools/vllm-backport-vision:/src \
  -v ~/tools/vision-wheels:/tmp/wheels:ro \
  -v ~/tools/vision-ccache:/ccache \
  -v ~/tools/rustup-init:/tmp/rustup-init:ro \
  -v ~/tools/vision-build-entry.sh:/entry.sh:ro \
  --entrypoint bash nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 \
  -c "while [ ! -f /src/.build_status ]; do bash /entry.sh >> /ccache/build.log 2>&1; sleep 30; done; sleep infinity"
# 盯: tail -f ~/tools/vision-ccache/build.log；满载时 ~230 编译进程 / load 47

# 构建完成后固化镜像（顺序铁律：start → exec安装 → stop → commit）
docker start vision-build
docker exec vision-build bash -c "export PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/lib/ccache:/root/.cargo/bin:\$PATH CUDA_HOME=/usr/local/cuda CCACHE_DIR=/ccache SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=48; cd /src && pip uninstall -y vllm; pip install --no-build-isolation --no-cache-dir ."
docker exec vision-build bash -c "export PATH=/opt/venv/bin:\$PATH; python3 -c 'import vllm; print(vllm.__file__)'"   # 必须指向 site-packages（非editable）
docker stop vision-build
docker commit vision-build dsv4-vision:sm80
```

**[坑] 构建期网络**：GitHub 克隆走 gh-proxy（`git config --global url.https://gh-proxy.com/https://github.com/.insteadOf …`）+ `git config http.version HTTP/1.1`；rustup 必须 USTC 镜像直连（不走代理）。

## 4. 启动（生产参数）

```bash
cat > ~/tools/launchVision.sh <<'EOF'
#!/bin/bash
docker stop -t 60 dsv4-vision >/dev/null 2>&1; docker rm dsv4-vision >/dev/null 2>&1
docker run -d --name dsv4-vision --restart unless-stopped \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0,1,2,3 \
  -e HF_HUB_OFFLINE=1 -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple \
  -v /home/<user>/models/dsv4-flash-vision-exp:/model:ro \
  --ipc=host -p 8096:8000 \
  --entrypoint /opt/venv/bin/vllm dsv4-vision:sm80 serve /model \
  --served-model-name dsv4-vision \
  --pipeline-parallel-size 4 \
  -e VLLM_PP_LAYER_PARTITION=12,11,11,9 \
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
EOF
chmod +x ~/tools/launchVision.sh && sudo ~/tools/launchVision.sh
```

**参数红线**：
- `--entrypoint /opt/venv/bin/vllm` 必须显式（commit 镜像默认 entrypoint 是 bash）
- PP=4 禁 TP（Gen2x4 无 P2P）；`VLLM_PP_LAYER_PARTITION=12,11,11,9`（rank3 扛草稿器+嵌入+lm_head，少给层 → KV 池 3.3 倍）
- **DSpark n=3**：VL checkpoint 草稿器是 3 层 nextn（n 必须整除 3）；n=6 单流快 18% 但并发聚合崩（对话 C16 差 3 倍）—— 并发场景一律 n=3
- `--kv-cache-dtype fp8`（不是 fp8_ds_mla，那会撞 fp8e4nv）
- 图模式必须 FULL_AND_PIECEWISE + NCCL 钉死（enforce-eager 会 3.3 tok/s）
- util 0.93 上限（0.95 有捕获 OOM 风险）

## 5. 验收清单

```bash
# 5.1 就绪（~10min：载权重+warmup）
curl -s localhost:8096/health        # 200
docker logs dsv4-vision 2>&1 | grep "KV cache size"
# 期望: ~4,719,096 tokens；"Maximum concurrency for 524,288: 9.00x"

# 5.2 文本冒烟（正确性关键 — cg修复验证）
curl -s localhost:8096/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv4-vision","messages":[{"role":"user","content":"中国的首都是哪座城市？直接回答"}],
       "max_tokens":40,"chat_template_kwargs":{"thinking":false},"temperature":0}'
# 期望: "北京"（若返回无关内容=图模式损坏，检查 5ab628dd1 是否摘入）

# 5.3 视觉（base64）
# content blocks: [{"type":"text",...},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]
# 期望: 官方 carrots.jpeg → "胡萝卜"

# 5.4 工具调用
# 带 tools 参数 → tool_calls: [{function: {name:..., arguments:...}}], finish_reason: "tool_calls"
```

## 6. 性能参考（实测 @230W，512k 窗口，DSpark n=3）

以下全部可用 `bench/bench.py` 在空载引擎上复现。DSpark 吞吐随草稿接受率线性放大，而接受率强依赖生成内容——bench.py 每个场景都会打印接受率，不同口径/提示词的数字不可比。

| 场景 | 数值 |
|---|---|
| 单流解码（浅层） | 44–49 tok/s（高接受率内容更快） |
| 单流解码（85k-178k 深度） | **90-96 tok/s，与深度无关** |
| TTFT | 1k=1.1s / 85k=16s / 163k=32s / 366k=94s（prefill ~4k t/s） |
| 视觉问答 | 首问 ~1.4s / 热问 0.4-0.5s |
| 对话聚合（无共享前缀，接受率 ~1.1） | C16=236 / C32=225 t/s |
| Agent 聚合（3.2k 共享前缀，接受率 ~1.33） | C8=207 / C16=258（峰值）/ C32=242 t/s |
| 并发甜点 | ≤8 交互 / ≤16 agent；>16 排队劣化 |
| KV 池 | 471.9 万 token；满 512k 窗口 9 路 |
| DSpark 接受长度 | 本基准口径 1.1–1.4 tok/draft（n=3）；内容更可预测时可达 ~2.3 |

## 7. 运维

| 症状 | 处置 |
|---|---|
| 容器重启循环 + `No module named vllm` | 镜像被坏 commit 覆盖 → 重走第3节安装+commit（顺序：start→exec→stop→commit） |
| 文本返回无关内容 | 图模式 cg bug → 确认 5ab628dd1 在树上；验证：echo 测试看服务器侧 prompt |
| `fp8e4nv not supported` | 用了非 Ampere 后端 → 确认 sm80 选择器补丁在 |
| `vision MoE routing requires input_ids` | PP 中继未生效 → 检查 3 处 dsv4_img_ids hunks |
| `requires DeepGEMM` | 用了 PR merge 后的树 → 回到 vision-v3 配方（不要 merge upstream main） |
| 深度"解码慢" | **先检查测量方法**：必须 TTFT 分离（流式首 content 时刻）+ usage 计数；total/tokens 是错的 |

## 8. 接入 fleet

```yaml
# 网关机 litellm config.yaml 追加（注意零缩进风格）：
- model_name: dsv4-vision
  litellm_params:
    model: openai/dsv4-vision
    api_base: http://<本机IP>:8096/v1
    api_key: none
    request_timeout: 3600
    stream_timeout: 3600
```

## 9. 使用守则

- 图像 + 常规对话（<32k）：全速
- 深上下文（≤400k）**长写完全可用**（88 tok/s 深度平坦），只是首字等 prefill（85k=16s / 366k=94s）
- 每图 ≤384 token；min_pixels 147456（过小图被放大）；宽高比 ≤8
- 精度敏感判断开 thinking（返回字段名是 `reasoning` 非 `reasoning_content`）
- 单机互斥：显存只够一个模型。需要纯文本高并发（C64 聚合 1600+ t/s）时换 `dsv4-flash-deploy-runbook.md` 的文本服务；需要图像/深上下文（88 tok/s 平到 400k）时换本服务

## 10. 已知边界

- MTP method 不可用（checkpoint 无 mtp_block 权重，只有 DSpark 草稿头）
- 1M 窗口未开（KV 池 472 万 < 1M 单请求装不下；需 8 卡或等上游压缩存储）
- DSpark n=6 仅适合 ≤2 路纯交互实例
