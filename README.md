# DSV4-Flash-Vision-Exp on 4× CMP 170HX

DeepSeek-V4-Flash-Vision-Exp（多模态 284B MoE + 32 层 ViT）在 4× 解锁 CMP 170HX（GA100 sm80, PCIe Gen2 x4）上的生产部署全套。

**实测性能**（PP4 · 512k 窗口 · DSpark n=3 投机解码 · FULL cudagraph）：

| 场景 | 数值 |
|---|---|
| 单流解码（浅层） | 49 tok/s |
| 单流解码（85k–366k 深度） | **88–92 tok/s，深度无关**（稀疏注意力只读 top-512） |
| TTFT | 1k=1.1s / 85k=16s / 366k=94s（prefill ~4k t/s） |
| Agent 并发聚合 | C8=379 / **C16=497（峰值）** / C32=404 t/s |
| 视觉问答 | 热问 0.4s（≤384 tok/图） |
| KV 池 | 472 万 token（满 512k 窗口 9 路） |
| 工具调用 | ✓ deepseek_v4 parser |

## 仓库结构

```
docs/
  dsv4-vision-deploy-runbook.md      # 主 runbook（从文本服务基础上加装视觉服务）
  PREREQUISITE-text-service-runbook.md  # 前置：文本服务 runbook（裸机→生产）
  dsv4-4x170hx-acceptance-report.md  # 文本服务验收报告
scripts/
  launch.sh          # 生产启动（PP4/512k/DSpark-n3/FULL图，含全部参数红线）
  build-entry.sh     # 常驻编译容器入口（重启自动续编，ccache 增量）
  sm80-patches.py    # sm80 三件套+DSpark嵌入补丁（选择器/PP中继/末阶嵌入）
bench/
  bench.py           # 验收基准（正确性/单流/深度TTFT分离/并发）
```

## 快速开始

```bash
# 0. 前置：完成 docs/PREREQUISITE-text-service-runbook.md（解锁/驱动/230W/Docker）
# 1. 下载模型 (~157GB)
HF_ENDPOINT=https://hf-mirror.com hf download deepseek-ai/DeepSeek-V4-Flash-Vision-Exp --local-dir ~/models/dsv4-flash-vision-exp
# 2. 组装源码树 + 打补丁 + 构建 —— 见 docs/dsv4-vision-deploy-runbook.md §2-3
python3 scripts/sm80-patches.py    # 在源码树上执行
# 3. 启动
sudo scripts/launch.sh
# 4. 验收
python3 bench/bench.py --quick
```

## 四个 sm80 硬阻塞与解法（本仓库核心技术点）

| 阻塞 | 解法 |
|---|---|
| sm80 无原生 FP8（`fp8e4nv` 编译崩） | Ampere 后端（ROCm Triton 路径 + `fp8_sm80` 软件编解码） |
| bias_vl 视觉路由需 input_ids，PP 非首阶没有 | PP input_ids 中继（IntermediateTensors 广播，3 处 hunks） |
| DSpark 草稿器在 PP 末阶别名嵌入表 | spec 开启时末阶也构建 embed_tokens（+1GB） |
| 图模式下文本返回无关内容 | 上游 PR #54566 的 `fix breakable cg`（2 行 config）单摘 |

## 测量铁律（深度解码测试防坑）

1. TTFT 必须流式分离（首 content 时刻），绝不用 total/tokens 当解码速率
2. token 数用 usage（DSpark 多 token/chunk）
3. 重启后首个请求丢弃（JIT 冷启）
4. prompt 首 token 随机化（防前缀缓存）

## 关键参数红线

- PP=4 禁 TP（Gen2 x4 无 P2E）；`VLLM_PP_LAYER_PARTITION=12,11,11,9`（KV 池 3.3×）
- DSpark **n=3**（VL 草稿器 3 层 nextn，n 须整除 3；n=6 单流 +18% 但并发聚合崩）
- `--entrypoint /opt/venv/bin/vllm` 显式（commit 镜像默认 bash）
- util ≤0.93；enforce-eager 会 3.3 tok/s（禁用）
