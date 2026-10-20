"""Ascend-side GEMM + ReLU sketch for Chapter 06 (STATIC CHECK ONLY).

This mirrors the Quick Start in the locked TileLang Ascend backend README
(``_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:60-100``). It is
syntactically valid Python and is checked with ``python3 -m py_compile``; it is
**not** executed here because the writing environment has no Ascend NPU, no
CANN toolkit and no TileLang installation.

What to read in each line (the point of the sketch):

  * ``T.alloc_l1(...)``      -> the Cube-side staging buffer (GM -> L1, MTE2)
  * ``T.alloc_l0c(...)``     -> the Cube accumulator (L0C)
  * ``T.gemm(A_l1, B_l1, C_l0c, transpose_B=True)`` -> L1 -> L0A/L0B (MTE1) + Cube
  * ``T.dual_copy(C_l0c, C_ub)`` -> L0C -> UB, split across the two AIVs (Fixpipe)
  * ``with T.SimtVF(threads=128):`` -> a SIMT region *on the Vector cores*
  * ``T.dual_copy(C_ub, C[...])``   -> UB -> GM (MTE3)

API provenance (all verified against ``_vendor/tilelang-v0.1.15/``):

  * ``alloc_l1`` / ``alloc_l0a`` / ``alloc_l0b`` / ``alloc_l0c``:
    ``tilelang/ascend/language/allocate.py:26,37,46,55``.
  * ``dual_copy``: ``tilelang/ascend/language/copy_op.py:53``.
  * ``SimtVF`` / ``SimdVF``: ``tilelang/ascend/language/frame.py:107,179``.
  * the full example: ``tilelang/ascend/README.md:60-100``.
"""

from __future__ import annotations

try:  # pragma: no cover - import is expected to fail without an NPU stack
    import tilelang
    import tilelang.ascend.language as T
except ImportError:  # noqa: BLE001 - this file is a static sketch
    tilelang = None
    T = None


if tilelang is not None:

    @tilelang.jit(target="ascend")
    def matmul_relu(A, B, block_M: int = 256, block_N: int = 224, block_K: int = 128):
        """C = relu(A @ B.T) on Ascend, bf16 in, fp32 accumulate/out."""
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), T.bfloat16)
        B: T.Tensor((N, K), T.bfloat16)
        C = T.empty((M, N), T.float32)
        num_blocks = 32
        n_tiles = N // block_N

        with T.Kernel(num_blocks) as bx:
            # Cube side: L1 staging + L0C accumulator.
            A_l1 = T.alloc_l1((block_M, block_K), T.bfloat16)
            B_l1 = T.alloc_l1((block_N, block_K), T.bfloat16)
            C_l0c = T.alloc_l0c((block_M, block_N), T.float32)
            # Vector side: Unified Buffer, half the M extent per AIV.
            C_ub = T.alloc_shared((block_M // 2, block_N), T.float32)

            for tile in T.Persistent([M // block_M * n_tiles], num_blocks, bx):
                m, n = tile // n_tiles * block_M, tile % n_tiles * block_N
                for k in T.Pipelined(K // block_K, num_stages=2):
                    T.copy(A[m, k * block_K], A_l1)  # GM -> L1 (MTE2)
                    T.copy(B[n, k * block_K], B_l1)  # GM -> L1 (MTE2)
                    T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=(k == 0))

                T.dual_copy(C_l0c, C_ub)  # L0C -> UB, M-split across two AIVs
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(block_M // 2, block_N):
                        C_ub[i, j] = T.max(C_ub[i, j], 0)
                T.dual_copy(C_ub, C[m : m + block_M, n : n + block_N])  # UB -> GM (MTE3)
        return C


def explain() -> str:
    return (
        "One T.Kernel mixes Cube work (T.gemm on L0C) with a Vector region "
        "(T.SimtVF), and lets the compiler assign cores, schedule the overlap and "
        "insert the Set/Wait flags. The data paths are the chapter's MTE1/MTE2/MTE3/"
        "Fixpipe routes."
    )


if __name__ == "__main__":
    print(explain())
    if tilelang is None:
        print("(no NPU stack here; nothing was compiled or run)")
