#!/usr/bin/env python3
"""第 14 章第 2 节的阶梯：同一个 GEMM 的 ①–④ 级实现。

每一级只改一件事，方便对照「为什么这一级比上一级快」：

  RUNG_0  朴素分块：直接读 global 做乘加
  RUNG_1  shared memory + T.gemm（不流水）
  RUNG_2  T.Pipelined(num_stages=N)
  RUNG_3  RUNG_2 + swizzle / layout 注解

第 ⑤ 级（intrinsic）在 gemm_intrinsic.py。

运行前提：NVIDIA GPU + CUDA toolkit + PyTorch(CUDA) + TileLang v0.1.15。
写作机上没有 GPU，本文件只做过静态核对（py_compile + API 比对），**未运行**。

用法::

    python3 gemm_ladder.py --rung 2 --m 4096 --n 4096 --k 4096
    python3 gemm_ladder.py --rung 3 --num-stages 3 --bench
"""

from __future__ import annotations

import argparse

import tilelang
import tilelang.language as T

try:  # RUNG_3 需要 swizzle layout 工具；v0.1.15 里它在 cuda.intrinsics 下
    from tilelang.cuda.intrinsics import make_mma_swizzle_layout
except ImportError:  # pragma: no cover - 仅为让文件在无 TileLang 时也能被阅读
    make_mma_swizzle_layout = None


@tilelang.jit
def gemm_rung0(A, B, block_M, block_N, dtype=T.float16, accum_dtype=T.float32):
    """① 朴素分块：每个输出元素沿 K 直接读 global。"""
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)
        for i, j in T.Parallel(block_M, block_N):
            for k in T.serial(K):
                C_local[i, j] = C_local[i, j] + A[by * block_M + i, k] * B[k, bx * block_N + j]
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


@tilelang.jit
def gemm_rung1(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    """② shared memory + T.gemm：先搬上片，再调 tile 级矩阵乘（串行循环）。"""
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for ko in T.serial(T.ceildiv(K, block_K)):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


@tilelang.jit
def gemm_rung2(A, B, block_M, block_N, block_K, num_stages=3, dtype=T.float16, accum_dtype=T.float32):
    """③ 加 T.Pipelined：搬运与计算重叠。"""
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


@tilelang.jit
def gemm_rung3(A, B, block_M, block_N, block_K, num_stages=3, panel_size=10,
               dtype=T.float16, accum_dtype=T.float32):
    """④ 加 swizzle / layout 注解：改物理布局与 block 遍历顺序。"""
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        if make_mma_swizzle_layout is not None:
            T.annotate_layout({
                A_shared: make_mma_swizzle_layout(A_shared),
                B_shared: make_mma_swizzle_layout(B_shared),
            })
        T.use_swizzle(panel_size=panel_size)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def build(rung: int, m: int, n: int, k: int, block_m: int, block_n: int, block_k: int,
          num_stages: int):
    if rung == 0:
        return gemm_rung0.compile(M=m, N=n, K=k, block_M=block_m, block_N=block_n)
    if rung == 1:
        return gemm_rung1.compile(M=m, N=n, K=k, block_M=block_m, block_N=block_n, block_K=block_k)
    if rung == 2:
        return gemm_rung2.compile(M=m, N=n, K=k, block_M=block_m, block_N=block_n,
                                  block_K=block_k, num_stages=num_stages)
    if rung == 3:
        return gemm_rung3.compile(M=m, N=n, K=k, block_M=block_m, block_N=block_n,
                                  block_K=block_k, num_stages=num_stages)
    raise ValueError(f"未知的 rung: {rung}（第 ⑤ 级见 gemm_intrinsic.py）")


def main() -> int:
    parser = argparse.ArgumentParser(description="TileLang GEMM 阶梯 ①–④")
    parser.add_argument("--rung", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--bench", action="store_true", help="跑 do_bench（需要 GPU）")
    parser.add_argument("--show-source", action="store_true", help="打印生成的 CUDA 源码")
    args = parser.parse_args()

    import torch

    kernel = build(args.rung, args.m, args.n, args.k, args.block_m, args.block_n,
                   args.block_k, args.num_stages)
    a = torch.randn(args.m, args.k, device="cuda", dtype=torch.float16)
    b = torch.randn(args.k, args.n, device="cuda", dtype=torch.float16)
    c = kernel(a, b)

    ref = a @ b
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    print(f"rung {args.rung}: 正确性通过（rtol=1e-2, atol=1e-2）")

    if args.show_source:
        print(kernel.get_kernel_source())

    if args.bench:
        profiler = kernel.get_profiler()
        latency = profiler.do_bench(warmup=25, rep=100, return_mode="mean")
        tflops = 2 * args.m * args.n * args.k / (latency / 1e3) / 1e12
        print(f"latency={latency:.4f} ms  ({tflops:.2f} TFLOP/s, 口径 2MNK)")
        print("报数时请一并写明 warmup/rep/backend/return_mode 与是否 flush L2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
