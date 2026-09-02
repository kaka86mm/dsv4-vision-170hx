[English](deployment.md) | [简体中文](deployment.zh-CN.md)

# Deployment Guide — DSV4-Flash-Vision-Exp on 4× CMP 170HX

> Goal: deploy DeepSeek-V4-Flash-Vision-Exp as a production multimodal service (text + image, PP4, 512k context, DSpark speculative decoding) on bare 4× CMP 170HX hardware.
> Note: VRAM fits exactly one model (167GB weights fill 4×64GB). This service owns the GPUs exclusively — stop any other model container before launching.
> If the machine has not been prepared (unlock / driver / Gen2 / 230W / Docker), follow `machine-setup.zh-CN.md` §0–3 first; this guide starts from model download.
> All commands verified on real hardware, 2026-09. `[坑]` markers flag known pitfalls.

## 0. Environment check

```bash
# Machine base in place (if not → machine-setup.zh-CN.md §0-3):
nvidia-smi --query-gpu=index,name,memory.total,pcie.link.gen.current,power.limit --format=csv,noheader
# Expect: 4× "NVIDIA CMP 170HX, 65536 MiB", gen=2, 230W
docker info >/dev/null && echo docker-ok
sudo docker images | grep nvidia/cuda:13.0.2 || sudo docker pull nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
# Stop other model containers (GPU exclusive):
docker ps --format "{{.Names}}" | grep -E "vllm|glm|dsv4" && docker stop <name>
```

## 1. Model download (~157GB)

```bash
python3 -m venv ~/hfenv 2>/dev/null; ~/hfenv/bin/pip install -q -U huggingface_hub
~/hfenv/bin/pip uninstall -y -q hf_xet hf_transfer   # [坑] Xet protocol incompatible with mirrors → 401
mkdir -p ~/models/dsv4-flash-vision-exp
HF_ENDPOINT=https://hf-mirror.com nohup ~/hfenv/bin/hf download \
  deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --local-dir ~/models/dsv4-flash-vision-exp > /tmp/dl-vision.log 2>&1 &
# "✓ Downloaded" in the log = complete (hf verifies checksums)
```

## 1.5 Build material preparation (cp312 wheels + rustup-init)

```bash
mkdir -p ~/tools/vision-wheels
cat > /tmp/dlwheels.sh <<'EOF'
#!/bin/bash
cd ~/tools/vision-wheels
export http_proxy=http://<proxy> https_proxy=http://<proxy>
for i in $(seq 1 30); do
  ~/hfenv/bin/pip download --no-cache-dir --python-version 3.12 --implementation cp \
    --abi cp312 --only-binary=:all: -d . \
    torch==2.13.0 torchaudio==2.11.0 torchvision==0.28.0 && { echo DONE > status.txt; exit 0; }
  echo "retry $i"; sleep 10
done
EOF
chmod +x /tmp/dlwheels.sh && /tmp/dlwheels.sh   # ~3.3GB, retry loop survives flaky networks
curl -sL -o ~/tools/rustup-init \
  https://mirrors.ustc.edu.cn/rust-static/rustup/dist/x86_64-unknown-linux-gnu/rustup-init
chmod +x ~/tools/rustup-init
```
[坑] Host pip downloads wheels for the host Python version — you must pass `--python-version 3.12 --abi cp312` (the build container runs 3.12).

## 2. Source tree assembly (vision-v3 recipe)

Four layered parts, in order:

```bash
mkdir -p ~/tools && cd ~/tools
git clone --branch dsv4-vision-exp --single-branch \
  https://github.com/wtdcode/vllm-backport.git vllm-backport-vision
cd vllm-backport-vision
git config user.email "m@local" && git config user.name "m"

# 2.1 PR #54566 first four commits (vision support baseline, up to 9327439714)
git fetch https://gh-proxy.com/https://github.com/vllm-project/vllm.git pull/54566/head:pr54566
git cherry-pick edafe3dbe^..9327439714
# Conflicts (test files, semantically-identical gate-bias condition): git checkout --theirs

# 2.2 cg fix + MTP enable (cherry-pick individually from later PR commits)
git cherry-pick 5ab628dd1     # fix breakable cg — 2-line config fix for graph-mode text corruption
git cherry-pick 2de7255a2     # enable mtp — VL drafter plumbing
# [坑] Do NOT use the PR's latest head (contains upstream main merge → DeepGEMM hard gate, unusable on sm80)

# 2.3 The sm80 kit (three pieces, without them sm80 cannot run)
mkdir -p vllm/models/deepseek_v4/ampere
# Take ampere/{__init__,ampere_sparse}.py from wtdcode master branch into the above dir
# (a thin 34-line wrapper: ROCm Triton path + fp8_sm80 software encode/decode)
# Then apply the selector + PP relay patches:
python3 scripts/sm80-patches.py   # from this repository, run in the source tree root
git add -A && git commit -m "sm80: ampere+selector+PP relay"

# 2.4 DSpark last-rank embedding patch (required for speculative decoding)
#    — included in sm80-patches.py (embed-on-last-rank when spec is active, +1GB)
git commit -am "sm80: embed on last rank for drafter"
```

## 3. Build (persistent container, auto-resumes across reboots)

```bash
# Copy scripts/build-entry.sh from this repo to ~/tools/vision-build-entry.sh
# (edit BUILD_PROXY at the top), then:
docker run -d --name vision-build --restart unless-stopped \
  -v ~/tools/vllm-backport-vision:/src \
  -v ~/tools/vision-wheels:/tmp/wheels:ro \
  -v ~/tools/vision-ccache:/ccache \
  -v ~/tools/rustup-init:/tmp/rustup-init:ro \
  -v ~/tools/vision-build-entry.sh:/entry.sh:ro \
  --entrypoint bash nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 \
  -c "while [ ! -f /src/.build_status ]; do bash /entry.sh >> /ccache/build.log 2>&1; sleep 30; done; sleep infinity"
# Monitor: tail -f ~/tools/vision-ccache/build.log (~230 compile processes at full load)

# After BUILD_OK — bake the image (strict order: start → exec install → stop → commit)
docker start vision-build
docker exec vision-build bash -c "export PATH=/opt/venv/bin:/usr/local/cuda/bin:/usr/lib/ccache:/root/.cargo/bin:\$PATH CUDA_HOME=/usr/local/cuda CCACHE_DIR=/ccache SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=48; cd /src && pip uninstall -y vllm; pip install --no-build-isolation --no-cache-dir ."
docker exec vision-build bash -c "export PATH=/opt/venv/bin:\$PATH; python3 -c 'import vllm; print(vllm.__file__)'"   # must point to site-packages (NOT editable)
docker stop vision-build
docker commit vision-build dsv4-vision:sm80
```

**[坑] Build-time networking**: GitHub clones via gh-proxy (`git config --global url.https://gh-proxy.com/https://github.com/.insteadOf …`) + `git config http.version HTTP/1.1`; rustup must use the USTC mirror directly (no proxy).

## 4. Launch (production parameters)

Use `scripts/launch.sh` from this repository, or the equivalent `docker run` with:

**Parameter red lines**:
- `--entrypoint /opt/venv/bin/vllm` must be explicit (committed images default to bash)
- PP=4, never TP (Gen2 x4, no P2P); `VLLM_PP_LAYER_PARTITION=12,11,11,9` (rank 3 carries drafter + embedding + lm_head → fewer layers → 3.3× KV pool)
- **DSpark n=3**: the VL checkpoint ships a 3-layer nextn drafter (n must divide 3); n=6 is 18% faster single-stream but collapses under concurrency (3× worse at C=16)
- `--kv-cache-dtype fp8` (not fp8_ds_mla — that hits fp8e4nv)
- Graph mode must be FULL_AND_PIECEWISE + pinned NCCL (enforce-eager = 3.3 tok/s)
- util ≤ 0.93 (0.95 risks capture OOM)

## 5. Acceptance checklist

```bash
# 5.1 Readiness (~10min: weight load + warmup)
curl -s localhost:8096/health        # 200
docker logs dsv4-vision 2>&1 | grep "KV cache size"
# Expect: ~4,719,096 tokens; "Maximum concurrency for 524,288: 9.00x"

# 5.2 Text smoke (correctness — the cg-fix verification)
curl -s localhost:8096/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"dsv4-vision","messages":[{"role":"user","content":"What is the capital of China? Answer directly."}],
       "max_tokens":40,"chat_template_kwargs":{"thinking":false},"temperature":0}'
# Expect: "Beijing" (unrelated content = graph-mode corruption → check 5ab628dd1)

# 5.3 Vision (base64)
# content blocks: [{"type":"text",...},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}]

# 5.4 Tool calling
# With tools param → tool_calls: [{function: {name:..., arguments:...}}], finish_reason: "tool_calls"

# Or run the full suite:
python3 bench/bench.py
```

## 6. Performance reference (measured @230W, 512k, DSpark n=3)

| Scenario | Value |
|---|---|
| Single-stream decode (shallow) | 49 tok/s (58 with n=6) |
| Single-stream decode (85k–366k depth) | **88–92 tok/s, depth-independent** |
| TTFT | 1k=1.1s / 85k=16s / 163k=32s / 366k=94s (prefill ~4k t/s) |
| Vision QA | first ~1.4s / hot 0.4–0.5s |
| Agent aggregate | C8=379 / C16=497 (peak) / C32=404 t/s |
| Chat aggregate | C32=439 t/s |
| Concurrency sweet spot | ≤8 interactive / ≤16 agent; >16 queueing degrades |
| KV pool | 4.719M tokens; 9 concurrent full-512k sessions |
| DSpark acceptance length | 2.28 (n=3) / 2.67 (n=6) |

## 7. Operations

| Symptom | Remedy |
|---|---|
| Restart loop + `No module named vllm` | Image overwritten by a bad commit → redo §3 install+commit (order: start→exec→stop→commit) |
| Text returns unrelated content | Graph-mode cg bug → verify 5ab628dd1 is in the tree |
| `fp8e4nv not supported` | Non-Ampere backend selected → verify the sm80 selector patch |
| `vision MoE routing requires input_ids` | PP relay not effective → check the 3 dsv4_img_ids hunks |
| `requires DeepGEMM` | Tree includes the upstream main merge → return to the vision-v3 recipe |
| Deep decode "slow" | **Check measurement method first**: TTFT must be separated (streaming first-content time) + usage-counted tokens; total/tokens is wrong |

## 8. Gateway integration

```yaml
# litellm config.yaml (zero-indent list style):
- model_name: dsv4-vision
  litellm_params:
    model: openai/dsv4-vision
    api_base: http://<host-ip>:8096/v1
    api_key: none
    request_timeout: 3600
    stream_timeout: 3600
```

## 9. Usage guidelines

- Images + regular conversation (<32k): full speed
- Deep context (≤400k) **long-form generation fully usable** (88 tok/s depth-flat); only the first token waits for prefill
- Per image ≤384 tokens; min_pixels 147456 (smaller images get upscaled); aspect ratio ≤8
- For precision-sensitive reasoning enable thinking (response field is `reasoning`, not `reasoning_content`)
- Single-model exclusive: the same hardware can alternatively run the text-only 0731 recipe (see machine-setup.zh-CN.md §4+); switching = stop one container, start the other (~10min)

## 10. Known limits

- MTP method unavailable (checkpoint ships DSpark drafter only, no mtp_block weights)
- 1M window not enabled (KV pool 4.72M < 1M single request; needs 8 cards or upstream compressed KV storage)
- DSpark n=6 only for ≤2-way pure-interactive instances
