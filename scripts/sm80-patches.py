#!/usr/bin/env python3
"""sm80 三件套 + DSpark 末阶嵌入补丁 — 在 vision-v3 源码树上执行。

四个补丁解决 CMP 170HX (sm80) 跑 DSV4-Vision 的四个硬阻塞:
  1. sm80 选择器   — fp8e4nv 崩溃 → Ampere 后端 (ROCm Triton + 软件编解码)
  2. PP 中继 x3    — bias_vl 路由需要 input_ids, PP 非首阶没有 → 广播
  3. 末阶嵌入      — DSpark 草稿器别名目标嵌入表 → spec 开启时末阶也建

用法: cd <vision-v3 源码树> && python3 sm80-patches.py
前置: ampere/{__init__,ampere_sparse}.py 已从 wtdcode master 放入 (见 runbook §2.3)
"""
import re, sys

MODEL_PY = "vllm/models/deepseek_v4/nvidia/model.py"

c = open(MODEL_PY).read()
applied = []

# --- 1. sm80 选择器 ---
SEL_ANCHOR = """    backend = vllm_config.attention_config.backend
    device_capability = current_platform.get_device_capability()
"""
SEL_BLOCK = SEL_ANCHOR + """    if device_capability is not None and device_capability.major == 8:
        # SM8x: Triton refuses native fp8e4nv converts below SM89; route to
        # the Ampere backend (ROCm Triton path + fp8_sm80 software enc/dec).
        from vllm.models.deepseek_v4.ampere.ampere_sparse import (
            DeepseekV4AmpereMLAAttention,
        )

        return DeepseekV4AmpereMLAAttention
"""
if "AmpereMLA" not in c:
    assert c.count(SEL_ANCHOR) == 1, "selector anchor not found"
    c = c.replace(SEL_ANCHOR, SEL_BLOCK); applied.append("sm80-selector")

# --- 2a. 空 intermediate 加 img ids ---
EMPTY_OLD = """                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
            }"""
EMPTY_NEW = """                "hidden_states": torch.zeros(
                    (batch_size, self.hc_mult, self.config.hidden_size),
                    dtype=dtype,
                    device=device,
                ),
                "dsv4_img_ids": torch.zeros(
                    (batch_size,), dtype=torch.int64, device=device
                ),
            }"""
if "dsv4_img_ids" not in c:
    assert c.count(EMPTY_OLD) == 1, "empty-intermediates anchor"
    c = c.replace(EMPTY_OLD, EMPTY_NEW); applied.append("pp-relay-empty")

# --- 2b. 非首阶取回 ---
RECV_OLD = """        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
"""
RECV_NEW = RECV_OLD + """            if input_ids is None:
                input_ids = intermediate_tensors["dsv4_img_ids"]
"""
if 'intermediate_tensors["dsv4_img_ids"]' not in c:
    assert c.count(RECV_OLD) == 1, "recv anchor"
    c = c.replace(RECV_OLD, RECV_NEW); applied.append("pp-relay-recv")

# --- 2c. 首阶发送 ---
SEND_OLD = """        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
"""
SEND_NEW = """        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "dsv4_img_ids": input_ids.to(torch.int64),
                }
            )
"""
if "dsv4_img_ids\": input_ids" not in c:
    assert c.count(SEND_OLD) == 1, "send anchor"
    c = c.replace(SEND_OLD, SEND_NEW); applied.append("pp-relay-send")

# --- 3. DSpark 末阶嵌入 ---
EMBED_OLD = """        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()"""
EMBED_NEW = """        # DSpark drafter (last PP rank) aliases the target embedding table.
        _spec = getattr(vllm_config, "speculative_config", None) is not None
        if get_pp_group().is_first_rank or (
            _spec and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()"""
if "_spec and get_pp_group().is_last_rank" not in c:
    assert c.count(EMBED_OLD) == 1, "embed anchor"
    c = c.replace(EMBED_OLD, EMBED_NEW); applied.append("embed-last-rank")

open(MODEL_PY, "w").write(c)
print("applied:", applied if applied else "all already present")
