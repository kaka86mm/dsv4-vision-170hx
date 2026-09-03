# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E8M0 block-scale handling of the Triton block-scaled fp8 GEMM.

Lives outside test_block_fp8.py because that module skips below SM90 while
the bug this guards manifests on SM89 (Ada), where the fp8 kernel selector
prefers the native route over Marlin and DeepSeek-V4's exponent-only E8M0
scales reach the Triton kernel (wtdcode/vllm-backport#14).
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    w8a8_triton_block_scaled_mm,
)
from vllm.platforms import current_platform

if not current_platform.is_cuda() or current_platform.get_device_capability() < (8, 9):
    pytest.skip(
        "Triton fp8 GEMM needs CUDA SM89+ (fp8e4nv loads)", allow_module_level=True
    )


@torch.inference_mode()
def test_w8a8_block_fp8_matmul_e8m0_scales():
    # DeepSeek-V4-style checkpoints store block scales in exponent-only
    # E8M0, which Triton cannot bind directly; the kernel must upcast them
    # to fp32 before launch on every platform (the upcast used to be gated
    # to ROCm/XPU, crashing CUDA SM89).
    M, N, K = 83, 512, 7168
    block_size = [128, 128]
    out_dtype = torch.bfloat16
    torch.manual_seed(0)

    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    n_tiles = (N + block_size[0] - 1) // block_size[0]
    k_tiles = (K + block_size[1] - 1) // block_size[1]

    # Power-of-two scales round-trip E8M0 <-> fp32 exactly, so the E8M0 run
    # must match the fp32 run bit for bit.
    As = torch.exp2(torch.randint(-8, 0, (M, k_tiles)).to(torch.float32))
    Bs = torch.exp2(torch.randint(-8, 0, (n_tiles, k_tiles)).to(torch.float32))

    ref_out = w8a8_triton_block_scaled_mm(A_fp8, B_fp8, As, Bs, block_size, out_dtype)
    out = w8a8_triton_block_scaled_mm(
        A_fp8,
        B_fp8,
        As.to(torch.float8_e8m0fnu),
        Bs.to(torch.float8_e8m0fnu),
        block_size,
        out_dtype,
    )
    assert torch.equal(out, ref_out)
