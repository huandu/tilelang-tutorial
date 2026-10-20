#!/usr/bin/env python3
"""第 14 章第 2.5 节的第 ⑤ 级：官方 intrinsic 版（手动 MMA 流水）。

结构来自 `_vendor/tilelang-v0.1.15/examples/gemm/example_gemm_intrinsics.py`，
裁剪掉与本级对照无关的细节，保留核心：

  alloc_local 手动分片 → ldmatrix_a/b → mma → stmatrix

运行前提：NVIDIA GPU + CUDA toolkit + PyTorch(CUDA) + TileLang v0.1.15。
写作机上没有 GPU，本文件只做过静态核对（py_compile + API 比对），**未运行**。
"""

from __future__ import annotations

from tvm import DataType

import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics import get_swizzle_layout
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter


def make_swizzle_layout(shared_buf):
    """按行宽是否正好 512 bit 决定要不要 swizzle（与官方示例同源）。"""
    shape = shared_buf.shape
    can_swizzle = shape[-1] * DataType(shared_buf.dtype).bits == 512
    if not can_swizzle:
        return T.Layout(shape, lambda *args: args)

    def transform_func(i, j):
        return list(get_swizzle_layout(i, j, shape[-1], shared_buf.dtype))

    return T.Layout(shape, transform_func)


@tilelang.jit
def tl_matmul(A, B, in_dtype, out_dtype, accum_dtype, num_stages=2):
    micro_size_x = micro_size_y = 16
    micro_size_k = 32 if out_dtype == T.int32 else 16

    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 64
    warp_col_tiles = 64
    chunk = 32

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), in_dtype)
    B: T.Tensor((N, K), in_dtype)
    C = T.empty((M, N), out_dtype)

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size_a = (micro_size_x * micro_size_k) // warp_size
    local_size_b = (micro_size_y * micro_size_k) // warp_size
    local_size_c = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    mma_emitter = TensorCoreIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=False,
        b_transposed=True,
        block_row_warps=block_row_warps,
        block_col_warps=block_col_warps,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=chunk,
    )

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((block_N, block_K), in_dtype)
        C_shared = T.alloc_shared((block_M // micro_size_x, block_N // micro_size_y,
                                   micro_size_x, micro_size_y), out_dtype)
        A_local = T.alloc_local((warp_rows * local_size_a), in_dtype)
        B_local = T.alloc_local((warp_cols * local_size_b), in_dtype)
        C_local = T.alloc_local((warp_rows * warp_cols * local_size_c), accum_dtype)

        T.annotate_layout({
            A_shared: make_swizzle_layout(A_shared),
            B_shared: make_swizzle_layout(B_shared),
        })
        T.use_swizzle(panel_size=10)

        T.clear(C_local)
        for ko in T.Pipelined(K // block_K, num_stages=num_stages):
            for i, k in T.Parallel(block_M, block_K):
                A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
            for j, k in T.Parallel(block_N, block_K):
                B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

            for ki in T.serial(0, block_K // micro_size_k):
                mma_emitter.ldmatrix_a(A_local, A_shared, ki)
                mma_emitter.ldmatrix_b(B_local, B_shared, ki)
                mma_emitter.mma(A_local, B_local, C_local)

        mma_emitter.stmatrix(C_local, C_shared)
        for i, j in T.Parallel(block_M, block_N):
            C[by * block_M + i, bx * block_N + j] = C_shared[
                i // micro_size_x, j // micro_size_y, i % micro_size_x, j % micro_size_y
            ]

    return C


def main(M: int = 4096, N: int = 4096, K: int = 4096) -> int:
    import torch

    in_dtype, out_dtype, accum_dtype = T.float16, T.float16, T.float32
    kernel = tl_matmul.compile(M=M, N=N, K=K, in_dtype=in_dtype,
                               out_dtype=out_dtype, accum_dtype=accum_dtype)
    src = kernel.get_kernel_source()
    assert src is not None

    profiler = kernel.get_profiler()
    latency = profiler.do_bench(warmup=25, rep=100, return_mode="mean")
    print(f"latency={latency:.4f} ms")

    profiler.assert_allclose(lambda a, b: a @ b.T, atol=1e-2, rtol=1e-2)
    print("正确性通过（对照 A @ B.T）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
