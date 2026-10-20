"""Pipelined GEMM in TileLang v0.1.15: the three shapes of T.Pipelined.

This file is written against the locked version in ``_vendor/tilelang-v0.1.15``
and every API it uses was checked against that tree. It needs an NVIDIA GPU to
compile and run; on the authoring machine (macOS, no GPU) only
``python3 -m py_compile`` was run, so the 验证级别 for this example is
「仅静态核对」. See README.md in this directory.

The three kernels share a body and differ only in how the reduction loop is
scheduled:

  * ``matmul_auto``   — ``num_stages=N``: let PipelinePlanning choose stage/order;
  * ``matmul_manual`` — explicit ``stage`` / ``order`` arrays;
  * ``matmul_plain``  — no pipeline at all, for the 「不做流水会怎样」 comparison.

Prototype: ``_vendor/tilelang-v0.1.15/examples/gemm/example_gemm_autotune.py``.
"""

from __future__ import annotations

import argparse

import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[-1])
def matmul_auto(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """Compiler-planned pipeline: only ``num_stages`` is given."""

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            # num_stages is the only schedule hint: PipelinePlanning derives the
            # per-statement stage and order arrays from the body's data flow.
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


@tilelang.jit(out_idx=[-1])
def matmul_manual(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """Hand-written schedule: copies in stage 0, GEMM in stage 1.

    The two arrays are aligned with the *scheduled* statements in source order
    (copy A, copy B, GEMM). The pipeline depth is ``max(stage) + 1``; with
    ``num_stages`` deliberately omitted.
    """

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(
                T.ceildiv(K, block_K),
                stage=[0, 0, 1],
                order=[0, 1, 2],
            ):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


@tilelang.jit(out_idx=[-1])
def matmul_plain(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """The no-pipeline baseline: a plain serial reduction loop.

    Every iteration must finish its copies before the GEMM can start, and the
    GEMM must finish before the next iteration's copies may overwrite the
    shared buffers. This is the shape used in the chapter's contrast.
    """

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return kernel


def shared_bytes(block_M: int, block_N: int, block_K: int, elem_bytes: int, stages: int) -> int:
    """Upper-bound shared memory for a tile pair at a given pipeline depth."""
    return stages * (block_M * block_K + block_N * block_K) * elem_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument(
        "--smem-limit",
        type=int,
        default=228 * 1024,
        help="shared memory budget in bytes; default is the H100 per-SM shared ceiling",
    )
    args = parser.parse_args()

    for stages in (1, args.stages):
        need = shared_bytes(
            args.block_m, args.block_n, args.block_k, 2, stages
        )
        mark = "ok" if need <= args.smem_limit else "OVER"
        print(
            f"stages={stages}: A+B tile x {stages} slots = {need} B "
            f"({need / 1024:.1f} KiB) [{mark}]"
        )

    kernel = matmul_auto(
        args.m,
        args.n,
        args.k,
        block_M=args.block_m,
        block_N=args.block_n,
        block_K=args.block_k,
        num_stages=args.stages,
        threads=args.threads,
    )
    print("\n=== device source (excerpt: the reduction loop) ===")
    source = kernel.get_kernel_source()
    for line in source.splitlines():
        if any(
            token in line
            for token in ("for (", "cp.async", "cp_async", "barrier", "wait", "mma", "__syncthreads")
        ):
            print(line.rstrip())
    print("\nInspect the printed loop: the steady-state body should contain both a")
    print("memory instruction and a tensor-core instruction, and the shared buffer")
    print("index should carry a stage-varying offset (the multi-buffer version).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
