"""TileLang version of the chapter's unified example: C = relu(A @ B).

Adapted from the locked tree's official quickstart
(`_vendor/tilelang-v0.1.15/examples/quickstart.py:1-56`, MIT-licensed upstream).

Verification level: static only. The API names and signatures below were checked
against `_vendor/tilelang-v0.1.15/tilelang/language/{kernel,allocate,copy_op,
gemm_op,loop}.py`; this file has NOT been executed (the writing environment has
no NVIDIA GPU and no Ascend NPU).

What to look for (chapter section 3.4):
  * the user writes dataflow (`T.copy`, `T.gemm`) and tile sizes;
  * the user writes NEITHER barriers NOR buffer-slot rotation;
  * the fused ReLU is just an elementwise step on the accumulator fragment.
"""

import tilelang
import tilelang.language as T


@tilelang.jit
def matmul_relu(A, B, block_M: int, block_N: int, block_K: int):
    M, N, K = T.const("M, N, K")
    dtype = T.float16
    accum_dtype = T.float32
    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    # T.Kernel defines a grid of tile programs; the thread mapping is decided later.
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)

        # T.Pipelined gives the reduction loop a software pipeline. No manual
        # multi-buffering, no explicit async-copy barriers: passes do that.
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            # On NVIDIA/AMD this dispatches to CuTe/HIP MMA (see quickstart.py:38).
            T.gemm(A_shared, B_shared, C_local)

        # Fused epilogue: one elementwise step on the accumulator fragment.
        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


M = N = K = 1024
BLOCK_M = BLOCK_N = 128
BLOCK_K = 32


def main() -> None:
    kernel = matmul_relu.compile(
        M=M, N=N, K=K, block_M=BLOCK_M, block_N=BLOCK_N, block_K=BLOCK_K
    )
    # The lines below require a CUDA device; kept for reference only.
    import torch

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = kernel(a, b)
    torch.testing.assert_close(c, torch.relu(a @ b), rtol=1e-2, atol=1e-2)
    print("GEMM + ReLU matched the PyTorch reference.")


if __name__ == "__main__":
    main()
