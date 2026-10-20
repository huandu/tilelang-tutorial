"""昇腾 950 GEMM+ReLU 骨架：T.gemm / T.dual_copy / T.SimtVF。

原型：_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:60-155
静态核对来源：
  tilelang/ascend/language/__init__.py:24,30-40,65   (copy/dual_copy/reduce/frame 导出)
  tilelang/ascend/language/copy_op.py:53-163         (T.dual_copy)
  tilelang/ascend/language/frame.py:107-143          (T.SimtVF)
  tilelang/ascend/language/gemm_op.py:168-210        (T.gemm + unit_flag_ctrl)
  src/ascend/transform/rewrite_dual_copy.cc:97-107   (dual_copy 的路径)

运行前提：昇腾 NPU + CANN + torch_npu + TileLang v0.1.15。
作者验证级别：仅静态核对（py_compile + API 逐条比对），未在 NPU 上运行。
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401  (注册 npu 设备)
import tilelang
import tilelang.ascend.language as T


@tilelang.jit(target="ascend", out_idx=-1)
def matmul_relu(A, B, block_M: int = 256, block_N: int = 224, block_K: int = 128):
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.bfloat16)
    B: T.Tensor((N, K), T.bfloat16)          # transpose_B=True
    C = T.empty((M, N), T.float32)
    num_blocks = 32
    n_tiles = N // block_N

    with T.Kernel(num_blocks) as bx:
        # 注意：C_l0c 是全 tile，C_ub 只有一半 —— dual_copy 会按 N 分裂补齐。
        A_l1 = T.alloc_l1((block_M, block_K), T.bfloat16)
        B_l1 = T.alloc_l1((block_N, block_K), T.bfloat16)
        C_l0c = T.alloc_l0c((block_M, block_N), T.float32)
        C_ub = T.alloc_shared((block_M // 2, block_N), T.float32)

        for tile in T.Persistent([M // block_M * n_tiles], num_blocks, bx):
            m, n = tile // n_tiles * block_M, tile % n_tiles * block_N
            for k in T.Pipelined(K // block_K, num_stages=2):
                T.copy(A[m, k * block_K], A_l1)
                T.copy(B[n, k * block_K], B_l1, l2_cache_ctrl="NOTALLOC_KEEP")
                T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=(k == 0))

            # L0C -> UB：两个 AIV 各拿一半（这里按 M 分裂）。
            T.dual_copy(C_l0c, C_ub)
            with T.SimtVF(threads=128):
                for i, j in T.Parallel(block_M // 2, block_N):
                    C_ub[i, j] = T.max(C_ub[i, j], 0)
            # UB -> GM：同样按 M 分裂写回。
            T.dual_copy(C_ub, C[m:m + block_M, n:n + block_N], l2_cache_ctrl="NOTALLOC_PW")

    return C


def main() -> None:
    M, N, K = 256, 7168, 2048
    kernel = matmul_relu(M, N, K)
    a = torch.randn((M, K), device="npu", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="npu", dtype=torch.bfloat16)
    c = kernel(a, b)
    torch.testing.assert_close(c, torch.relu(a.float() @ b.float().T), rtol=1e-2, atol=1e-2)
    print("GEMM + ReLU passed.")


if __name__ == "__main__":
    main()
