#!/usr/bin/env python3
"""第 14 章第 4 节：GEMM 的四个变体（split-K / stream-K / grouped / batched）。

每个实现都直接对照官方 example：

  gemm_splitk    <- examples/gemm_splitk/example_tilelang_gemm_splitk.py
  gemm_streamk   <- examples/gemm_streamk/example_tilelang_gemm_streamk.py
  gemm_grouped   <- examples/grouped_gemm/example_grouped_gemm_fwd.py
  gemm_batched   <- 一维 grid 展平 batch（思路同昇腾 batch_gemm）

运行前提：NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang v0.1.15。
写作机上没有 GPU，本文件只做过静态核对（py_compile + API 比对），**未运行**。
"""

from __future__ import annotations

import math

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# split-K：输出块数 < 核数时，把 K 切开换并行度
# ---------------------------------------------------------------------------
@tilelang.jit
def gemm_splitk(A, B, C, block_M, block_N, block_K, split_k,
                dtype=T.float16, accum_dtype=T.float32, out_dtype=T.float32):
    M, N, K = T.const("M, N, K")
    splitK = K // split_k

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C: T.Tensor((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=128) as (bx, by, bz):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=0):
            T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
            T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            T.atomic_add(C[by * block_M + i, bx * block_N + j], C_local[i, j])


# ---------------------------------------------------------------------------
# stream-K：把 K 迭代平均分给固定数量的 program，只有被切开的块要原子合并
# ---------------------------------------------------------------------------
@tilelang.jit
def gemm_streamk(A, B, C, streamk_programs, iters_per_tile, full_tiles, partial_tiles,
                 sm_partition_factor, streamk_tiles, block_M, block_N, block_K,
                 dtypeAB=T.float16, dtypeC=T.float32,
                 accum_dtype=T.float32, num_stages=2, threads=64):
    """注意：`streamk_programs` 等规模参数在 host 侧算好后传入，
    与官方 example 用 Python 变量捕获这些常量的做法一致。"""
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtypeAB)
    B: T.Tensor((N, K), dtypeAB)
    C: T.Tensor((M, N), dtypeC)

    with T.Kernel(streamk_programs, threads=threads) as pid:
        A_shared = T.alloc_shared((block_M, block_K), dtypeAB)
        B_shared = T.alloc_shared((block_N, block_K), dtypeAB)
        A_shared_full = T.alloc_shared((block_M, block_K), dtypeAB)
        B_shared_full = T.alloc_shared((block_N, block_K), dtypeAB)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        start_iter = T.alloc_fragment((1,), T.int32, "local")
        end_iter = T.alloc_fragment((1,), T.int32, "local")

        start_iter[0] = pid * full_tiles + T.min(pid, partial_tiles)
        last_iter = (pid + 1) * full_tiles + T.min(pid + 1, partial_tiles)

        while start_iter[0] < last_iter:
            end_iter[0] = T.min(
                start_iter[0] + (iters_per_tile - (start_iter[0] % iters_per_tile)),
                last_iter,
            )
            tile_id = start_iter[0] // iters_per_tile
            remain = start_iter[0] % iters_per_tile
            pid_m = tile_id // T.ceildiv(N, block_N)
            pid_n = tile_id % T.ceildiv(N, block_N)

            T.clear(C_local)
            for k in T.Pipelined(end_iter[0] - start_iter[0], num_stages=num_stages):
                T.copy(A[pid_m * block_M, (k + remain) * block_K], A_shared)
                T.copy(B[pid_n * block_N, (k + remain) * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            if remain == 0 and (end_iter[0] % iters_per_tile == 0):
                T.copy(C_local, C[pid_m * block_M, pid_n * block_N])
            else:
                for i, j in T.Parallel(block_M, block_N):
                    T.atomic_add(C[pid_m * block_M + i, pid_n * block_N + j], C_local[i, j])

            start_iter[0] = end_iter[0]

        # 剩下的整块（blocking tiles）平铺给固定 grid，不需要原子合并
        for p in T.serial(sm_partition_factor):
            tile_id = pid + streamk_tiles + p * streamk_programs
            pid_m = tile_id // T.ceildiv(N, block_N)
            pid_n = tile_id % T.ceildiv(N, block_N)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[pid_m * block_M, k * block_K], A_shared_full)
                T.copy(B[pid_n * block_N, k * block_K], B_shared_full)
                T.gemm(A_shared_full, B_shared_full, C_local, transpose_B=True)
            T.copy(C_local, C[pid_m * block_M, pid_n * block_N])


# ---------------------------------------------------------------------------
# grouped GEMM：每个专家的 M 不同（MoE），用偏移表定位当前 block 属于谁
# ---------------------------------------------------------------------------
@tilelang.jit
def gemm_grouped(A, B, batch_sizes, batch_offsets, batch_padded_offsets, batch_sizes_list,
                 block_M, block_N, block_K, num_stages=2, threads=128, dtype=T.float16):
    K, N = T.const("K, N")
    batch_sum = sum(batch_sizes_list)
    batch_count = len(batch_sizes_list)
    accum_dtype = T.float32
    total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)

    A: T.Tensor([batch_sum, K], dtype)
    B: T.Tensor([batch_count, K, N], dtype)
    C = T.empty([batch_sum, N], dtype)
    batch_sizes: T.Tensor([batch_count], T.int32)
    batch_offsets: T.Tensor([batch_count], T.int32)
    batch_padded_offsets: T.Tensor([batch_count], T.int32)

    with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
        A_shared = T.alloc_shared([block_M, block_K], dtype)
        B_shared = T.alloc_shared([block_K, block_N], dtype)
        C_local = T.alloc_fragment([block_M, block_N], accum_dtype)
        cur_batch_idx = T.alloc_var(dtype=T.int32)

        m_start_padded = bx * block_M
        cur_batch_idx = 0
        for i in range(batch_count):
            in_cur = m_start_padded >= batch_padded_offsets[i]
            cur_batch_idx = T.if_then_else(in_cur, i, cur_batch_idx)

        cur_batch_size = batch_sizes[cur_batch_idx]
        m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
        actual_rows = T.max(0, T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded))

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[m_start : m_start + block_M, k * block_K : (k + 1) * block_K], A_shared)
            T.copy(B[cur_batch_idx, k * block_K : (k + 1) * block_K,
                     by * block_N : (by + 1) * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            if i < actual_rows:
                C[m_start + i, by * block_N + j] = C_local[i, j]

    return C


# ---------------------------------------------------------------------------
# batched GEMM：形状一致的批量，把 (batch, m_block, n_block) 展平成一维 grid
# ---------------------------------------------------------------------------
@tilelang.jit
def gemm_batched(A, B, batch, block_M, block_N, block_K, num_stages=2,
                 threads=128, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")
    m_num = T.ceildiv(M, block_M)
    n_num = T.ceildiv(N, block_N)

    A: T.Tensor((batch, M, K), dtype)
    B: T.Tensor((batch, K, N), dtype)
    C = T.empty((batch, M, N), dtype)

    with T.Kernel(batch * m_num * n_num, threads=threads) as cid:
        bid = cid // (m_num * n_num)
        rem = cid % (m_num * n_num)
        bx = rem // n_num
        by = rem % n_num

        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[bid, bx * block_M, k * block_K], A_shared)
            T.copy(B[bid, k * block_K, by * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[bid, bx * block_M, by * block_N])

    return C


def streamk_plan(m, n, k, block_m, block_n, block_k, total_sm):
    """算出 stream-K 需要的几个规模参数（纯 Python，可在无 GPU 时单独调用）。"""
    num_block_m = math.ceil(m / block_m)
    num_block_n = math.ceil(n / block_n)
    iters_per_tile = math.ceil(k / block_k)
    total_tiles = num_block_m * num_block_n
    streamk_tiles = total_tiles % total_sm
    if total_tiles - streamk_tiles > total_sm:
        streamk_tiles += total_sm
    blocking_tiles = total_tiles - streamk_tiles
    streamk_iters = streamk_tiles * iters_per_tile
    full_tiles = streamk_iters // total_sm
    partial_tiles = streamk_iters % total_sm
    return {
        "iters_per_tile": iters_per_tile,
        "total_tiles": total_tiles,
        "streamk_tiles": streamk_tiles,
        "blocking_tiles": blocking_tiles,
        "full_tiles": full_tiles,
        "partial_tiles": partial_tiles,
        "sm_partition_factor": max(blocking_tiles // total_sm, 1),
    }


def main() -> int:
    import torch

    torch.manual_seed(42)

    # split-K：M=N=256 太小，K=16384 很大
    m, n, k = 256, 256, 16384
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(k, n, dtype=torch.float16, device="cuda")
    c = torch.zeros(m, n, dtype=torch.float32, device="cuda")
    gemm_splitk(a, b, c, 128, 128, 32, 4)
    torch.testing.assert_close(c, (a @ b).float(), rtol=1e-2, atol=1e-2)
    print("split-K 正确性通过")

    # batched
    batch = 8
    ab = torch.randn(batch, 512, 512, dtype=torch.float16, device="cuda")
    bb = torch.randn(batch, 512, 512, dtype=torch.float16, device="cuda")
    cb = gemm_batched(ab, bb, batch, 128, 128, 32)
    torch.testing.assert_close(cb, torch.bmm(ab, bb), rtol=1e-2, atol=1e-2)
    print("batched 正确性通过")

    # stream-K 的规模参数（打印即可，不需要真跑）
    print("stream-K plan:", streamk_plan(256, 1024, 512, 16, 128, 32, total_sm=108))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
