#!/usr/bin/env python3
"""第 33 章 §5、§6：量化算子族与 RoPE 的端到端调用样例。

覆盖四类调用，全部走 TileKernels 的公开 API（因此同一份脚本在两个后端都能跑，
后端由 tile_kernels.config.is_ascend() 判定）：

  * quant.cast               —— 逐元素 dtype 转换（可选随机舍入）
  * quant.per_token_cast     —— per-token 量化，返回 (out, sf)
  * quant.per_token_cast_back—— 反量化，回到 bf16
  * quant.swiglu_forward_and_per_token_cast —— SwiGLU + 量化融合
  * transform.apply_rotary   —— 原地 RoPE

运行前提：已安装 tile-kernels + TileLang v0.1.15；NVIDIA SM90/SM100
或 Ascend 950 + CANN >= 9.2.0。
验证级别：仅静态核对（py_compile 通过；逐条 API 与语料比对）。
"""

from __future__ import annotations

import torch

import tile_kernels
from tile_kernels.config import get_device
from tile_kernels.testing.numeric import calc_diff


def demo_cast_and_quant(hidden: int = 2048) -> None:
    """per-token 量化 → 反量化，检查相对误差（口径见 testing/numeric.py:33-38）。"""
    x = torch.randn((512, hidden), dtype=torch.bfloat16, device=get_device())

    # 逐元素 cast（不带缩放）。
    y = tile_kernels.quant.cast(x, torch.float8_e4m3fn)
    print(f'cast: {tuple(y.shape)} dtype={y.dtype}')

    # per-token 量化：sf_block = (1, num_per_channels)，C 必须是 16/32/64/128。
    q, sf = tile_kernels.quant.per_token_cast(x, 'e4m3', 128)
    print(f'per_token_cast: data={tuple(q.shape)} sf={tuple(sf.shape)}')

    # 反量化回 bf16，与原值比较相对误差。
    x_back = tile_kernels.quant.per_token_cast_back((q, sf), 'bf16', 128)
    print(f'cast_back diff = {float(calc_diff(x_back, x)):.3e}')

    # 只算尺度、不算数据（MoE aux loss 之类场景会这么用）。
    sf_only = tile_kernels.quant.per_token_cast_with_sf_only(x, 'e4m3', 128)
    print(f'sf_only: {tuple(sf_only.shape)}')


def demo_fused_swiglu(hidden: int = 2048) -> None:
    """SwiGLU 前向 + per-token 量化的融合路径（§5.5）。"""
    # 融合 kernel 的输入是「左右两半拼在一起」的 2*hidden。
    x = torch.randn((512, 2 * hidden), dtype=torch.bfloat16, device=get_device())
    out, sf = tile_kernels.quant.swiglu_forward_and_per_token_cast(x, 'e4m3', 128)
    print(f'swiglu+cast: out={tuple(out.shape)} sf={tuple(sf.shape)}')

    # 同一算法的非融合版本（只做 SwiGLU，不量化）用于对照。
    ref = tile_kernels.quant.swiglu_forward(x, 'bf16')
    print(f'swiglu only: {tuple(ref.shape)}')


def demo_rope(seqlen: int = 2001, nheads: int = 32, dim: int = 64) -> None:
    """原地 RoPE（§6.1）。rot_dim 只支持 64 与 128。"""
    device = get_device()
    query = torch.randn(seqlen, nheads, dim, dtype=torch.bfloat16, device=device)
    key = torch.randn(seqlen, nheads, dim, dtype=torch.bfloat16, device=device)
    cos_sin = torch.randn(8 * 1024, dim, dtype=torch.float32, device=device)
    positions = torch.randint(0, 8 * 1024, (seqlen,), dtype=torch.int32, device=device)

    # 与 tests/transform/test_rope.py:36-46 相同的调用形态。
    # interleaved=True 走 GPT-J 奇偶配对，False 走 NeoX 前后半配对。
    tile_kernels.transform.apply_rotary(
        query,
        key=key,
        cos_sin_cache=cos_sin,
        positions=positions,
        interleaved=False,
        conjugate=False,
    )

    cos, sin = cos_sin.chunk(2, dim=-1)
    ref = tile_kernels.torch.rotary_embedding_ref(query, cos, sin, positions)
    print(f'rope diff = {float(calc_diff(query, ref)):.3e}')


if __name__ == '__main__':
    demo_cast_and_quant()
    demo_fused_swiglu()
    demo_rope()
