[English](README.md) | [简体中文](README.zh-CN.md)

# dsv4-vision-170hx

Production deployment of **DeepSeek-V4-Flash-Vision-Exp** (multimodal MoE, 284B total / 13B active + 32-layer ViT) on **4× unlocked NVIDIA CMP 170HX** — GA100 mining cards repurposed as 64GB sm_80 inference GPUs, connected via PCIe Gen2 x4 with no P2P.

This repository contains everything needed to reproduce the deployment from bare metal: source patches, build tooling, launch configuration, benchmark suite, and the complete troubleshooting record.

## Why this exists

The CMP 170HX is an A100-class die crippled for mining: compute fuse-locked, memory strap-limited, PCIe capped at Gen1 x4. After software unlocking (64GB HBM2e, full SM count, Gen2 retrain), it becomes one of the cheapest ways to accumulate large-memory inference capacity — but every mainstream serving stack assumes Hopper or newer. This repo documents the four hard blockers that stop DeepSeek-V4-Vision from running on sm_80 and the patches that resolve them, so the same recipe applies to any A100/A800 fleet.

## Measured performance

All numbers from a 4× CMP 170HX box (PP4, 512k context, DSpark speculative decoding n=3, FULL CUDA graphs):

| Workload | Result |
|---|---|
| Single-stream decode (shallow) | 49 tok/s |
| Single-stream decode (85k–366k depth) | **88–92 tok/s — flat across depth** |
| Time to first token | 1k=1.1s / 85k=16s / 366k=94s (prefill ~4k tok/s) |
| Agent concurrency (3.2k shared prefix) | C8=379 / **C16=497 peak** / C32=404 tok/s aggregate |
| Vision QA (hot) | 0.4s per query (≤384 tokens per image) |
| KV cache pool | 4.72M tokens (9 concurrent full-window sessions) |
| Tool calling | verified (deepseek_v4 parser) |

The depth-flat decode curve is a property of the architecture's sparse attention (top-512 selection): per-token cost does not scale with context length. Deep-context sessions are limited by prefill latency, not decode throughput.

## Repository layout

```
docs/
  deployment.md          # Full deployment guide (EN) — [中文版](deployment.zh-CN.md) (bare metal → production service)
  machine-setup.zh-CN.md # Hardware unlocking (中文), driver, PCIe Gen2, power management
  benchmarks-0731.zh-CN.md # Reference benchmarks (中文) of the text-only model on identical hardware
scripts/
  launch.sh              # Production launch script with all tuning parameters
  build-entry.sh         # Persistent build container entrypoint (resumable compilation)
  sm80-patches.py        # Source patches: sm80 backend routing, PP input relay,
                         #   drafter embedding on last pipeline rank
bench/
  bench.py               # Acceptance benchmark (correctness / single-stream /
                         #   depth-separated decode / concurrency)
```

## The four sm_80 blockers

| Blocker | Symptom | Fix |
|---|---|---|
| No native FP8 on GA100 | `fp8e4nv not supported` at kernel compile | Route to the Ampere attention backend (ROCm Triton path + `fp8_sm80` software encode/decode) |
| Vision expert routing needs `input_ids` under pipeline parallelism | `vision MoE routing requires input_ids` on non-first PP ranks | Relay `input_ids` through `IntermediateTensors` broadcast (3 patches) |
| DSpark drafter aliases the target embedding on the last PP rank | `needs the target's embedding on the last stage` | Build `embed_tokens` on the last rank when speculative decoding is active (+1 GB) |
| CUDA-graph corruption with multimodal models | Text queries return content from unrelated requests | Upstream PR #54566 `fix breakable cg` (2-line config change), cherry-picked |

## Quick start

```bash
# 1. Hardware setup (unlock / driver / Gen2 / 230W / Docker)
#    → follow docs/machine-setup.md sections 0–3
#
# 2. Download the checkpoint (~157 GB)
HF_ENDPOINT=https://hf-mirror.com hf download \
  deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --local-dir ~/models/dsv4-flash-vision-exp
#
# 3. Assemble the source tree and apply patches
#    → follow docs/deployment.md §2 (vision-v3 recipe)
python3 scripts/sm80-patches.py
#
# 4. Build (persistent container, survives reboots via ccache)
bash scripts/build-entry.sh   # inside the build container
#
# 5. Launch
sudo scripts/launch.sh
#
# 6. Verify
python3 bench/bench.py --quick
```

## Key configuration decisions

- **Pipeline parallel (PP=4), never tensor parallel.** Gen2 x4 without P2P makes TP all-reduce latency-dominated; PP moves activations 3 times per step instead of synchronizing 86 times.
- **Layer partition `12,11,11,9`.** The last rank carries the DSpark drafter, output head, and embedding patch — giving it fewer layers rebalances memory and grows the KV pool 3.3×.
- **DSpark n=3, not n=5 or n=6.** The vision checkpoint ships a 3-layer nextn drafter; the engine requires `num_speculative_tokens` divisible by 3. n=6 is 18% faster single-stream but collapses under concurrency (3× worse aggregate at C=16).
- **FULL_AND_PIECEWISE CUDA graphs + pinned NCCL (`Ring`/`Simple`).** Enforce-eager mode runs at 3.3 tok/s — 15× slower than graph mode.
- **`--entrypoint /opt/venv/bin/vllm` explicit.** Images committed from build containers inherit `bash` as the entrypoint.

## Benchmark methodology rules

These rules are encoded in `bench/bench.py`; violating any of them produces misleading numbers (each was learned from a real misdiagnosis):

1. **Separate TTFT from decode time.** Never divide total time by token count — prefill dominates at depth.
2. **Count tokens from `usage`, not stream chunks.** Speculative decoding emits multiple tokens per chunk.
3. **Discard the first request after engine restart.** Triton JIT compilation inflates it.
4. **Randomize the first token of benchmark prompts.** Prefix caching returns stale results for repeated prompts.

## Hardware requirements

- 4× CMP 170HX (8GB hynix variant) unlocked to 64GB — or any 4× 64GB sm_80 GPUs (A100/A800)
- ≥3TB free disk (checkpoint 157GB + build artifacts + images)
- ≥200GB system RAM
- Docker with NVIDIA Container Toolkit
- Power: 230W software cap per card (250W+ causes transient spikes that trigger Xid 43)

## License

MIT — see [LICENSE](LICENSE). The patched components originate from [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) and [vllm-project/vllm PR #54566](https://github.com/vllm-project/vllm/pull/54566); respect their licenses when redistributing.
