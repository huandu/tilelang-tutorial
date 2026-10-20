"""Triton-style sketch for C = relu(A @ B).  NOT runnable as-is.

This file is a *teaching sketch*, not a verified Triton program.
  * The tile-level idea (block program over a tensor; the compiler owns the
    schedule, shared-memory allocation and synchronization) comes from the
    Triton MAPL 2019 paper, mirrored at
    `../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md` (see
    lines 37, 130-200 and 583-720 for the tile definition and the passes).
  * The Python API surface named here (`triton.language`) is referenced by the
    corpus copy of the Triton docs index
    (`../tilelang-corpus/11-related-compilers/raw/triton-docs.md`), but the exact
    signatures were NOT verified against a pinned Triton version in this corpus.

So: treat this as pseudocode that happens to be syntactically valid Python.
It compiles with `python -m py_compile`; it will not import `triton` here.

Chapter section 3.2 explains the contrast: Triton leaves the schedule to the
compiler, TileLang keeps the dataflow in the language.
"""

from __future__ import annotations

import triton  # noqa: F401  (import guarded only for documentation)
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def gemm_relu_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # Block-level matmul; on modern Triton this lowers to tensor cores.
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Fused epilogue, still at block level.
    acc = tl.maximum(acc, 0.0)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))


def main() -> None:
    # Placeholder: a real launch needs CUDA tensors and a grid lambda.
    # grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
    print("This is a sketch; see the module docstring for what is verified.")


if __name__ == "__main__":
    main()
