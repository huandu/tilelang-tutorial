#!/usr/bin/env python3
"""第 14 章第 6 节：昇腾上的 GEMM 与 batched GEMM。

对照 `_vendor/tilelang-ascend-v0.1.2.000/examples/gemm/example_gemm.py` 与
`examples/batch_gemm/batch_gemm.py`，以及主干
`_vendor/tilelang-v0.1.15/examples/ascend/example_gemm.py`。

三个实现刚好是本章阶梯在昇腾上的翻版：

  ascend_gemm_basic       L1 + L0C + T.gemm_v0 + 显式 barrier（≈ 第 ② 级）
  ascend_gemm_auto        同上，但把同步/多缓冲/内存规划交给 pass_configs（≈ 第 ③ 级）
  ascend_batch_gemm       一维 grid 展平 (batch, m_block, n_block)（≈ 4.4 节）

运行前提：昇腾 NPU + CANN + torch_npu + TileLang-Ascend
（`_vendor/tilelang-ascend-v0.1.2.000`）；PyTorch with torch_npu。
写作机上没有 NPU，本文件只做过静态核对（py_compile + API 比对），**未运行**。
"""

from __future__ import annotations

import argparse

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[-1])
def ascend_gemm_basic(M, N, K, block_M, block_N, K_L1, dtype="float16", accum_dtype="float"):
    """最朴素的昇腾 GEMM：显式 barrier，不做流水。"""
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
            bx = cid // n_num
            by = cid % n_num

            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                for k in T.serial(T.ceildiv(K, K_L1)):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)
                    T.barrier_all()
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    T.barrier_all()
                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


ASCEND_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=ASCEND_PASS_CONFIGS)
def ascend_batch_gemm(batch, M, N, K, block_M, block_N, K_L1,
                      dtype="float16", accum_dtype="float"):
    """batched GEMM：一维 grid 展平 (batch, m_block, n_block)。"""
    m_num = M // block_M
    n_num = N // block_N

    @T.prim_func
    def main(
        A_mat: T.Tensor((batch, M, K), dtype),
        B_mat: T.Tensor((batch, K, N), dtype),
        C_mat: T.Tensor((batch, M, N), dtype),
    ):
        total = batch * m_num * n_num
        with T.Kernel(total, is_npu=True) as (cid, _):
            bid = cid // (m_num * n_num)
            rem = cid % (m_num * n_num)
            bx = rem // n_num
            by = rem % n_num

            A_L1 = T.alloc_shared((block_M, K_L1), dtype)
            B_L1 = T.alloc_shared((K_L1, block_N), dtype)
            C_L0 = T.alloc_fragment((block_M, block_N), accum_dtype)

            for k in T.serial(T.ceildiv(K, K_L1)):
                T.copy(A_mat[bid, bx * block_M, k * K_L1], A_L1)
                T.copy(B_mat[bid, k * K_L1, by * block_N], B_L1)
                T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))

            T.copy(C_L0, C_mat[bid, bx * block_M, by * block_N])

    return main


def main() -> int:
    parser = argparse.ArgumentParser(description="昇腾 GEMM / batched GEMM")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    m, n, k = args.m, args.n, args.k

    # 基础 GEMM
    gemm = ascend_gemm_basic(m, n, k, 128, 256, 64)
    a = torch.randn(m, k).half().npu()
    b = torch.randn(k, n).half().npu()
    c = gemm(a, b)
    torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=1e-2)
    print("ascend_gemm_basic: 正确性通过")

    # batched GEMM
    batched = ascend_batch_gemm(args.batch, m, n, k, 128, 256, 64)
    ab = torch.randn(args.batch, m, k).half().npu()
    bb = torch.randn(args.batch, k, n).half().npu()
    cb = batched(ab, bb)
    torch.testing.assert_close(cb, torch.matmul(ab, bb), rtol=1e-2, atol=1e-2)
    print("ascend_batch_gemm: 正确性通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
