#!/usr/bin/env python3
"""第 19 章 §7：主干（M 树）昇腾方言的 GEMM + ReLU 混核 kernel。

来源：`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:66-108` 的 quickstart。
它同时演示了三件 M 树独有的事：
  1. T.alloc_l1 / T.alloc_l0c 的小写方言 API；
  2. T.Persistent + T.Pipelined 的自动调度；
  3. T.dual_copy 把 L0C 结果劈给两个 AIV，再在 T.SimtVF 里做 ReLU。

运行前提：Ascend 950 NPU + 兼容 CANN（含 bisheng）+ torch_npu + 主干 TileLang v0.1.15。
验证级别：仅静态核对（py_compile + API 比对），未在 NPU 上运行。
"""

from __future__ import annotations

import argparse

import torch
import torch_npu  # noqa: F401 - 注册 torch.npu
import tilelang
import tilelang.ascend.language as T


@tilelang.jit(target="ascend")
def matmul_relu(A, B, block_M: int = 256, block_N: int = 224, block_K: int = 128):
    """C = relu(A @ B^T)，bf16 输入，fp32 累加与输出。

    A:(M, K)，B:(N, K)（对应 transpose_B=True）。
    所有维度都能被 tile 整除；本片段不处理 partial tile。
    """
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.bfloat16)
    B: T.Tensor((N, K), T.bfloat16)
    C = T.empty((M, N), T.float32)
    num_blocks = 32
    n_tiles = N // block_N

    with T.Kernel(num_blocks) as bx:
        A_l1 = T.alloc_l1((block_M, block_K), T.bfloat16)
        B_l1 = T.alloc_l1((block_N, block_K), T.bfloat16)
        C_l0c = T.alloc_l0c((block_M, block_N), T.float32)
        # Cube 与 Vector 不能直接交换数据；这里用 UB 承接 L0C 的输出。
        C_ub = T.alloc_shared((block_M // 2, block_N), T.float32)

        # Persistent：固定 32 个 block，轮转处理所有输出 tile。
        for tile in T.Persistent([M // block_M * n_tiles], num_blocks, bx):
            m = tile // n_tiles * block_M
            n = tile % n_tiles * block_N
            for k in T.Pipelined(K // block_K, num_stages=2):
                T.copy(A[m, k * block_K], A_l1)
                # L2 提示：权重复用度高，保留在 L2（NOTALLOC_KEEP）。
                T.copy(B[n, k * block_K], B_l1, l2_cache_ctrl="NOTALLOC_KEEP")
                T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=(k == 0))
            # L0C -> UB：硬件双目的拷贝，M-split 给两个 AIV。
            T.dual_copy(C_l0c, C_ub)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(block_M // 2, block_N):
                    C_ub[i, j] = T.max(C_ub[i, j], 0)
            # UB -> GM：L2 提示 NOTALLOC_PW（写出后不需要保留）。
            T.dual_copy(C_ub, C[m : m + block_M, n : n + block_N], l2_cache_ctrl="NOTALLOC_PW")

    return C


def ref_program(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """参考实现：relu(A.float() @ B.float().T)。"""
    return torch.relu(a.float() @ b.float().T)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ascend 950 GEMM+ReLU (main tree dialect)")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    m, n, k = args.m, args.n, args.k
    a = torch.randn((m, k), device="npu", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="npu", dtype=torch.bfloat16)

    c = matmul_relu(a, b)
    torch.testing.assert_close(c, ref_program(a, b), rtol=1e-2, atol=1e-2)
    print("GEMM + ReLU passed.")

    if args.bench:
        from tilelang.profiler import do_bench

        # 注意：matmul_relu 已经被 @tilelang.jit 装饰并直接返回输出张量，
        # 所以这里直接把它当 callable 计时，不要再 compile 一次。
        latency = do_bench(lambda: matmul_relu(a, b), backend="msprof", _n_warmup=30, _n_repeat=50)
        tflops = 2.0 * m * n * k / (latency * 1e9)
        print(f"gemm_relu (M={m}, N={n}, K={k}): {latency * 1000:.2f} us | {tflops:.1f} TFLOPS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
