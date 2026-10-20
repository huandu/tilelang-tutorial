#!/usr/bin/env python3
"""A GEMM where the user pins layouts instead of letting the compiler choose.

Chapter 10, §3.3. Two independent user interventions are shown:

  * T.annotate_layout({A_shared: make_mma_swizzle_layout(A_shared), ...})
      pin the shared-memory swizzle used when writing/reading A and B;
      normally layout inference picks this, and pinning it is how you take over.
  * T.use_swizzle(panel_size=10, enable=True)
      reorder the *thread blocks* (rasterization) for L2 locality -- this is a
      different concept from a shared-memory swizzle, see §3.3.

The accumulator `C_local` is deliberately left to layout inference: it is a
T.Fragment (one element per (thread, register slot)), and pinning it by hand is
only meaningful when it has to line up with a specific consumer.
It also turns on the compiler's own layout visualization so you can compare
what you pinned with what was inferred for the rest of the kernel.

Needs: NVIDIA GPU + CUDA toolkit + PyTorch(CUDA) + TileLang v0.1.15.

Usage:
    # inside the chapter's code/ directory, with the source available on
    # PYTHONPATH (or installed):
    python3 annotate_swizzle_gemm.py

Verified level: statically checked only. The authoring machine has no GPU and
no TileLang, so this was NOT run. Every API call was checked against the
v0.1.15 sources listed in the README; run it on a GPU box to get real results.
"""

from __future__ import annotations

import tilelang
import tilelang.language as T
from tilelang.cuda.intrinsics import make_mma_swizzle_layout


@tilelang.jit(
    pass_configs={
        # Let the compiler print each inferred fragment layout and dump a PNG/SVG.
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "txt,svg",
    }
)
def gemm_pinned_layouts(
    A,
    B,
    block_M,
    block_N,
    block_K,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # (1) Pin the shared-memory swizzle ourselves. Comment these two out and
        #     the compiler will infer an equivalent swizzle on its own.
        T.annotate_layout(
            {
                A_shared: make_mma_swizzle_layout(A_shared),
                B_shared: make_mma_swizzle_layout(B_shared),
            }
        )

        # (2) Thread-block rasterization order, NOT a shared-memory swizzle.
        T.use_swizzle(panel_size=10, enable=True)

        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def main() -> None:
    M = N = K = 1024
    kernel = gemm_pinned_layouts.compile(
        M=M, N=N, K=K, block_M=128, block_N=128, block_K=32
    )

    # Read what we got without needing a GPU: the generated CUDA source, and
    # the inferred-layout text the visualization pass printed during compile.
    source = kernel.get_kernel_source()
    print(source[:2000])
    print("... generated source printed above; inferred layouts printed during compile.")

    import torch

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = kernel(a, b)
    ref = a @ b
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    print("correctness check passed")


if __name__ == "__main__":
    main()
