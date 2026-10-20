"""NV-side GEMM sketch for Chapter 06 (STATIC CHECK ONLY).

This file is a *reading aid*, not a runnable benchmark. It is the same shape as
the canonical TileLang quickstart GEMM, annotated with which piece of the
Hopper machine each line is asking for. It is syntactically valid Python and is
checked with ``python3 -m py_compile``, but it is **not** executed here: the
writing environment has no NVIDIA GPU and no TileLang installation.

API provenance (all verified against the locked tree
``_vendor/tilelang-v0.1.15/``):

  * ``tilelang.jit``, ``T.const``, ``T.Tensor``, ``T.empty``, ``T.Kernel``,
    ``T.alloc_shared``, ``T.alloc_fragment``, ``T.clear``, ``T.Pipelined``,
    ``T.copy``, ``T.gemm``, ``T.Parallel`` -- see
    ``_vendor/tilelang-v0.1.15/examples/quickstart.py``.

Optional Hopper-only levers (not used below, but part of the same locked tree):

  * ``T.wgmma_gemm(...)`` / ``T.wait_wgmma(...)``:
    ``tilelang/cuda/language/gemm_op.py:226``.
  * ``T.cluster_arrive()`` / ``T.cluster_wait()`` / ``T.cluster_sync()``:
    ``tilelang/cuda/language/cluster.py:27-46``.
  * ``T.WarpSpecialize(...)``:
    ``tilelang/cuda/language/warpgroup.py:19``.
  * ``T.copy(..., disable_tma=True)`` to forbid the TMA lowering:
    ``tilelang/cuda/language/copy_op.py:42``.
"""

from __future__ import annotations

try:  # pragma: no cover - import is expected to fail without a GPU stack
    import tilelang
    import tilelang.language as T
except ImportError:  # noqa: BLE001 - this file is a static sketch
    tilelang = None
    T = None


if tilelang is not None:

    @tilelang.jit
    def matmul_relu(A, B, block_M: int, block_N: int, block_K: int):
        """C = relu(A @ B) with fp16 inputs and fp32 accumulation."""
        M, N, K = T.const("M, N, K")
        dtype = T.float16
        accum_dtype = T.float32
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((K, N), dtype)
        C = T.empty((M, N), dtype)

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (
            bx,
            by,
        ):
            # grid/block level: one block owns one (block_M x block_N) output tile.
            A_shared = T.alloc_shared((block_M, block_K), dtype)  # -> SMEM
            B_shared = T.alloc_shared((block_K, block_N), dtype)  # -> SMEM
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)  # -> registers

            T.clear(C_local)
            # T.Pipelined asks for multi-buffering; on Hopper the copy is a
            # candidate for the TMA path, and T.gemm a candidate for wgmma.
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])
        return C


def explain() -> str:
    return (
        "This sketch maps to the chapter's four dimensions: grid/block/warp/thread "
        "(execution), shared memory + registers (storage), T.copy -> SMEM (movement), "
        "and T.Pipelined/mbarrier (synchronization)."
    )


if __name__ == "__main__":
    print(explain())
    if tilelang is None:
        print("(no GPU stack here; nothing was compiled or run)")
