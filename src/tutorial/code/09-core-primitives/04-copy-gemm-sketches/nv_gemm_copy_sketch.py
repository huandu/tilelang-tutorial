"""第 09 章的 NV 侧 GEMM 骨架：T.copy / T.gemm / T.fill 的最小可读组合。

静态核对来源：
  tilelang/language/copy_op.py:59-130          (T.copy)
  tilelang/language/gemm_op.py:38-122,160-210  (T.gemm)
  tilelang/language/fill_op.py:11-72           (T.fill / T.clear)
  examples/norm/rms_norm.py:14-36              (T.clear + T.copy + T.reduce_sum 的原型)
  examples/gemm/                               (GEMM 原型，见上游 examples)

运行前提：NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang v0.1.15。
作者验证级别：仅静态核对（py_compile + API 逐条比对），未在硬件上运行。
"""

from __future__ import annotations

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=-1)
def matmul(M: int, N: int, K: int, block_M: int = 128, block_N: int = 128,
           block_K: int = 64, num_stages: int = 3, threads: int = 256):
    """C = A @ B，A/B 均为 [*, K]，这里只做 M/N/K 全可整除的教学版本。"""
    assert M % block_M == 0 and N % block_N == 0 and K % block_K == 0

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_K, block_N), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            # T.fill / T.clear：见正文 §3.6。这里等价于 T.fill(C_local, 0.0)。
            T.clear(C_local)

            for ko in T.Pipelined(K // block_K, num_stages=num_stages):
                # T.copy：global -> shared。extent 由目标 buffer 的 shape 决定；
                # src 侧是 BufferLoad（起始地址写法），见正文 §3.1。
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                # T.gemm：M/N/K 从 C 与 A/B 的最后两维推出；
                # clear_accum=False 表示在 C_local 已有值上累加（这里靠上面的 T.clear）。
                T.gemm(A_shared, B_shared, C_local)

            # T.copy：fragment -> global 写回。
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def main() -> None:
    M = N = K = 1024
    kernel = matmul(M, N, K)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = kernel(a, b)
    torch.testing.assert_close(c, (a.float() @ b.float()), rtol=1e-2, atol=1e-2)
    print("GEMM passed.")


if __name__ == "__main__":
    main()
