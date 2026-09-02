[English](README.md) | [简体中文](README.zh-CN.md)

# dsv4-vision-170hx

在 **4× 解锁 NVIDIA CMP 170HX** 上生产部署 **DeepSeek-V4-Flash-Vision-Exp**（多模态 MoE，284B 总参 / 13B 激活 + 32 层 ViT）—— GA100 矿卡改造为 64GB sm_80 推理卡，PCIe Gen2 x4 互联、无 P2P。

本仓库包含从裸机复现该部署的全部内容：源码补丁、构建工具链、启动配置、基准套件与完整排障记录。

## 为什么做这个

CMP 170HX 是一颗为挖矿残化的 A100 级核心：算力熔丝锁定、显存 strap 限制、PCIe 钉死 Gen1 x4。软件解锁（64GB HBM2e、完整 SM、Gen2 重训）之后，它是堆积大显存推理容量最便宜的路径之一 —— 但所有主流推理栈都默认 Hopper 或更新架构。本仓库记录了 DeepSeek-V4-Vision 跑在 sm_80 上的四个硬阻塞及其修复，同一配方适用于任何 A100/A800 机队。

## 实测性能

以下数据全部由 [`bench/bench.py`](../bench/bench.py) 在空载引擎上实测（PP4、512k 上下文、DSpark n=3、FULL CUDA graph）。DSpark 投机吞吐随草稿接受率线性放大，而**接受率强依赖生成内容**（本口径 1.1–1.4 tok/draft）——对比数字必须连同接受率一起看，不同测试口径/提示词的结果不可比。

| 负载 | 结果 |
|---|---|
| 单流解码（浅层） | 44–49 tok/s |
| 单流解码（85k–178k 深度） | **90–96 tok/s — 与深度无关** |
| 首字延迟 (TTFT) | 93k=17.6s / 178k=36.4s（prefill ~4k tok/s） |
| 对话并发（无共享前缀） | C16=236 / C32=225 t/s 聚合（接受率 ~1.1） |
| Agent 并发（3.2k 共享前缀） | C8=207 / **C16=258 峰值** / C32=242 t/s 聚合（接受率 ~1.33） |
| 视觉问答（热） | 0.4s/问（每图 ≤384 token） |
| KV 缓存池 | 472 万 token（9 路满窗口并发） |
| 工具调用 | ✓（deepseek_v4 parser） |

深度平坦的解码曲线源自架构的稀疏注意力（top-512 选取）：单 token 成本不随上下文长度增长。深上下文会话的瓶颈是 prefill 延迟，而非解码吞吐。

## 仓库结构

```
docs/
  deployment.zh-CN.md    # 完整部署指南（裸机 → 生产服务）
  machine-setup.zh-CN.md # 硬件解锁、驱动、Gen2、功耗 (中文)
  benchmarks-0731.zh-CN.md # 同硬件纯文本模型(0731)基准参考
patches/
  ampere/               # Ampere attention backend (auto-copied by sm80-patches.py)
scripts/
  launch.sh              # 生产启动脚本（全部调优参数）
  build-entry.sh         # 常驻编译容器入口（断点续编）
  sm80-patches.py        # 源码补丁：sm80 后端路由、PP input 中继、
                         #   末阶草稿器嵌入
bench/
  bench.py               # 验收基准（正确性 / 单流 / 深度分离解码 / 并发）
```

## 四个 sm_80 硬阻塞

| 阻塞 | 症状 | 修复 |
|---|---|---|
| GA100 无原生 FP8 | 内核编译期 `fp8e4nv not supported` | 路由到 Ampere 注意力后端（ROCm Triton 路径 + `fp8_sm80` 软件编解码） |
| 流水线并行下视觉专家路由需要 `input_ids` | 非首 PP 阶 `vision MoE routing requires input_ids` | 经 `IntermediateTensors` 广播中继 `input_ids`（3 处补丁） |
| DSpark 草稿器在末 PP 阶别名目标嵌入表 | `needs the target's embedding on the last stage` | 投机解码开启时末阶也构建 `embed_tokens`（+1GB） |
| 多模态模型的 CUDA graph 损坏 | 文本请求返回无关请求的内容 | 单摘上游 PR #54566 `fix breakable cg`（2 行配置改动） |

## 快速开始

```bash
# 1. 硬件准备（解锁 / 驱动 / Gen2 / 230W / Docker）
#    → 按 docs/machine-setup.zh-CN.md §0–3 执行
#
# 2. 下载 checkpoint（~157 GB）
HF_ENDPOINT=https://hf-mirror.com hf download \
  deepseek-ai/DeepSeek-V4-Flash-Vision-Exp \
  --local-dir ~/models/dsv4-flash-vision-exp
#
# 3. 组装源码树并打补丁
#    → 按 docs/deployment.zh-CN.md §2（vision-v3 配方）
python3 scripts/sm80-patches.py
#
# 4. 构建（常驻容器，重启经 ccache 自动续编）
bash scripts/build-entry.sh   # 在构建容器内
#
# 5. 启动
sudo scripts/launch.sh
#
# 6. 验收
python3 bench/bench.py --quick
```

## 关键配置决策

- **流水线并行（PP=4），禁用张量并行。** Gen2 x4 无 P2P 下 TP 的 all-reduce 被延迟支配；PP 每步传 3 次激活值而非同步 86 次。
- **层配平 `12,11,11,9`。** 末阶独扛 DSpark 草稿器、输出头与嵌入补丁 —— 少给它层以配平显存，KV 池扩大 3.3 倍。
- **DSpark n=3，不是 5 或 6。** 视觉 checkpoint 的草稿器是 3 层 nextn；引擎要求 `num_speculative_tokens` 整除 3。n=6 单流快 18% 但并发聚合崩（C=16 差 3 倍）。
- **FULL_AND_PIECEWISE CUDA graph + 钉死 NCCL（`Ring`/`Simple`）。** enforce-eager 只有 3.3 tok/s —— 比图模式慢 15 倍。
- **显式 `--entrypoint /opt/venv/bin/vllm`。** 从构建容器 commit 的镜像默认入口是 bash。

## 基准测量铁律

以下规则已编码进 `bench/bench.py`；违反任何一条都会产出误导数字（每条都来自真实误诊）：

1. **TTFT 与解码时间必须分离。** 绝不用总时间除以 token 数 —— 深处 prefill 占绝对主导。
2. **token 数用 `usage` 字段，不数流式 chunk。** 投机解码每个 chunk 携带多 token。
3. **丢弃引擎重启后的第一个请求。** Triton JIT 编译使其虚高。
4. **基准 prompt 首 token 随机化。** 前缀缓存会对重复 prompt 返回假快结果。

## 硬件要求

- 4× CMP 170HX（8GB hynix 版）解锁至 64GB —— 或任意 4× 64GB sm_80 GPU（A100/A800）
- ≥3TB 空闲磁盘（checkpoint 157GB + 构建产物 + 镜像）
- ≥200GB 系统内存
- Docker + NVIDIA Container Toolkit
- 功耗：每卡 230W 软帽（250W+ 瞬态尖峰触发 Xid 43）

## 许可

MIT —— 见 [LICENSE](LICENSE)。补丁组件源自 [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) 与 [vllm-project/vllm PR #54566](https://github.com/vllm-project/vllm/pull/54566)；再分发时请遵守其许可。
