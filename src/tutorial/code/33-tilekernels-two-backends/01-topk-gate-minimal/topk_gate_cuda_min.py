#!/usr/bin/env python3
"""第 33 章 §4.2：top-k gate 的 CUDA 版最小可读实现。

改写自 corpus 快照
`.workspace/research/tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/moe/topk_gate_cuda.py`
（commit 66258df6175d）：保留「一行一个 block、warp 内归约、重复取 max、
tie-break 取最小下标」的全部结构，删掉非有限值检查与 PDL，并加上逐段注释。

与原文件的差异（都已在本章正文里说明）：
  * 省略 `T.reducer_init/update/finalize` 构成的非有限值检查（原文 :40-46）；
  * 省略 `use_pdl` 与 `T.pdl_sync()` / `T.pdl_trigger()`（原文 :23-24,65-66）；
  * `num_experts` / `num_topk` 由调用方给出（原文件是同名参数，语义相同）。

运行前提：NVIDIA SM90/SM100 + CUDA Toolkit >= 13.1 + TileLang v0.1.15 + torch。
验证级别：仅静态核对（py_compile 通过；API 名与签名在 _vendor/tilelang-v0.1.15/ 中比对）。
"""

from __future__ import annotations

import tilelang
from tilelang import language as T


def align(x: int, y: int) -> int:
    """把 x 向上对齐到 y 的整数倍（与 tile_kernels/utils.py:8 相同）。"""
    return (x + y - 1) // y * y


@tilelang.jit(
    pass_configs={
        # 这个 kernel 只有一个 warp 在做一件事，不需要 producer/consumer 拆分。
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def get_topk_gate_kernel_cuda(num_experts: int, num_topk: int):
    num_tokens = T.dynamic('num_tokens')  # 运行时才知道的行数
    num_threads = 32  # 一个 warp
    num_aligned_experts = align(num_experts, num_threads)  # 向上补齐到 warp 宽度

    @T.prim_func
    def topk_gate_kernel(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
    ):
        # 一行一个 block：所有跨线程通信都退化成 warp 内通信。
        with T.Kernel(num_tokens, threads=num_threads) as pid:
            # fragment = 按线程切分的寄存器数组（第 10 章）。
            scores_fragment = T.alloc_fragment((num_aligned_experts,), T.float32)
            amax_fragment = T.alloc_fragment((1,), T.float32)
            idx_fragment = T.alloc_fragment((num_aligned_experts,), T.int32)
            # reducer = 一等公民的「延迟归约」句柄；物理通信方案由编译器选。
            idx_reducer = T.alloc_reducer((1,), T.int32, op='min')
            idx_result = T.alloc_fragment((1,), T.int32)
            # 只有输出需要跨线程协作，所以放 shared。
            topk_idx_shared = T.alloc_shared((num_topk,), T.int32)

            # 载入本行的分数；越界位置填 -inf，保证它永远不可能是最大值。
            for i in T.Parallel(num_aligned_experts):
                if i < num_experts:
                    scores_fragment[i] = scores[pid, i]
                else:
                    scores_fragment[i] = -T.infinity(T.float32)

            # 记住每个位置对应哪个专家下标，tie-break 时要用。
            for i in T.Parallel(num_aligned_experts):
                idx_fragment[i] = i

            # 重复取 max，每轮输出一个下标。num_topk <= 32 由外层 API 检查。
            for k in T.unroll(num_topk):
                T.reduce_max(scores_fragment, amax_fragment)
                # 在所有「等于最大值」的位置里取最小下标 —— tie-break 契约的落点。
                T.reducer_init(idx_reducer)
                for i in T.Parallel(num_aligned_experts):
                    if scores_fragment[i] == amax_fragment[0]:
                        T.reducer_update(idx_reducer[0], idx_fragment[i])
                T.finalize_reducer(idx_reducer, idx_result)
                topk_idx_shared[k] = idx_result[0]
                # 把中奖位置置 -inf，下一轮不再考虑它。
                for i in T.Parallel(num_aligned_experts):
                    if idx_fragment[i] == idx_result[0]:
                        scores_fragment[i] = -T.infinity(T.float32)

            # int32 shared -> int64 global：T.copy 在两侧 dtype 不同时逐元素转换；
            # 索引非负且 < 2**31，所以转换精确。disable_tma：小拷贝不值得建 TMA 描述符。
            T.copy(topk_idx_shared, topk_idx[pid, 0], disable_tma=True)

    return topk_gate_kernel


def topk_gate(scores, num_topk: int):
    """与 tile_kernels.moe.topk_gate 相同的契约：返回 [T, k] 的 int64 下标。"""
    import torch

    assert scores.dim() == 2 and scores.is_contiguous() and scores.dtype == torch.float32
    num_tokens, num_experts = scores.shape
    assert num_topk <= num_experts and num_topk <= 32

    out = torch.empty((num_tokens, num_topk), dtype=torch.int64, device=scores.device)
    if num_tokens == 0:
        return out
    get_topk_gate_kernel_cuda(num_experts, num_topk)(scores, out)
    return out
