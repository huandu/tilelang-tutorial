#!/usr/bin/env python3
"""第 19 章 §5：昇腾上的最小 GEMM（A 树）。

对照 `_vendor/tilelang-ascend-v0.1.2.000/examples/gemm/example_gemm.py`。
数据流：GM(A),GM(B) --MTE2--> L1 --(gemm_v0 内部: MTE1)--> L0A/L0B
        --Cube--> L0C --FixPipe--> GM(C)

运行前提：Ascend NPU + CANN + torch_npu + TileLang-Ascend（A 树）；
          M/K/N 能被 128/64/256 整除（示例不处理尾块）。
验证级别：仅静态核对（py_compile + API 比对），未在 NPU 上运行。
"""

from __future__ import annotations

import argparse

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[-1])
def matmul(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    K_L1: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """C = A @ B，朴素分块；A:(M,K), B:(K,N), C:(M,N)。"""
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

            # Cube 的输入先落在 L1（512KB），只放 K_L1 列，沿 K 反复复用。
            A_L1 = T.alloc_L1((block_M, K_L1), dtype)
            B_L1 = T.alloc_L1((K_L1, block_N), dtype)
            # 累加器在 L0C（128KB，fp32）。128x256 的 fp32 恰好等于 L0C 上限。
            C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)

            with T.Scope("C"):
                loop_k = T.ceildiv(K, K_L1)
                for k in T.serial(loop_k):
                    T.copy(A[bx * block_M, k * K_L1], A_L1)
                    T.copy(B[k * K_L1, by * block_N], B_L1)

                    # 屏障 1：等 MTE2 把 L1 填好（MTE2 -> MTE1/M 依赖）。
                    T.barrier_all()
                    # gemm_v0 内部完成 L1->L0A/L0B 搬运与 Cube MAD，写进 L0C。
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
                    # 屏障 2：等 Cube 算完，L1 才能被下一轮覆盖（M -> MTE2 依赖）。
                    T.barrier_all()

                # L0C -> GM（FixPipe）。
                T.copy(C_L0, C[bx * block_M, by * block_N])

    return main


def main() -> int:
    parser = argparse.ArgumentParser(description="Ascend GEMM (adapter tree)")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=256)
    parser.add_argument("--k-l1", type=int, default=64)
    parser.add_argument("--fp32", action="store_true", help="用 fp32 输入（默认 fp16）")
    args = parser.parse_args()

    dtype = "float" if args.fp32 else "float16"
    func = matmul(args.m, args.n, args.k, args.block_m, args.block_n, args.k_l1, dtype=dtype)

    torch.manual_seed(0)
    a = torch.randn(args.m, args.k)
    b = torch.randn(args.k, args.n)
    if dtype == "float16":
        a, b = a.half(), b.half()
    a, b = a.npu(), b.npu()
    print("init successful!")

    c = func(a, b)
    torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
