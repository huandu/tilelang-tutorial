#!/usr/bin/env python3
"""Blackwell TCGEN05 GEMM with the accumulator living in Tensor Memory (TMEM).

Prototype: _vendor/tilelang-v0.1.15/examples/gemm_tcgen05/gemm_tcgen5mma.py

Requires: NVIDIA Blackwell (sm_100a), CUDA toolkit 12.8+, PyTorch(CUDA),
          tilelang==0.1.15. Statically checked only; see README.md.

NOTE on the API surface: the released example calls `T.tcgen05_gemm(...)`, but
in v0.1.15 `tilelang/cuda/language/gemm_op.py:20-30` defines that name without
`tilelang/cuda/language/__init__.py` re-exporting it (see the chapter, 4.3.5).
This file therefore uses the exported synchronous form
`T.gemm(A, B, C_tmem, mbar=mbar)`, which inserts the matching
`mbarrier_wait_parity` implicitly.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def matmul_tcgen05(A, B, block_M, block_N, block_K,
                   trans_A, trans_B, in_dtype, out_dtype, accum_dtype,
                   num_stages, threads):
    """C = A @ B with the accumulator in TMEM (TCGEN05 MMA).

    The TMA / warp-specialization passes are disabled so the generated code
    shows the plain TMEM + mbarrier protocol.
    """
    M, N, K = T.const("M, N, K")
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    A: T.Tensor(A_shape, in_dtype)
    B: T.Tensor(B_shape, in_dtype)
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                  threads=threads) as (bx, by):
        A_shared = T.alloc_shared(A_shared_shape, in_dtype)
        B_shared = T.alloc_shared(B_shared_shape, in_dtype)
        C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
        mbar = T.alloc_barrier(1)          # TCGEN05 completion: single arrival
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        C_shared = T.alloc_shared((block_M, block_N), out_dtype)

        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K], B_shared)
            # C is in TMEM, so this selects cuda.tcgen05; the compiler inserts
            # the matching mbarrier_wait_parity(mbar, ...) after issue.
            T.gemm(A_shared, B_shared, C_tmem, trans_A, trans_B,
                   clear_accum=(k == 0), mbar=mbar)

        # TMEM -> registers -> shared -> global. Only TCGEN05 and TMEM
        # load/store may touch TMEM, so post-processing happens after this copy.
        T.copy(C_tmem, C_local)
        T.copy(C_local, C_shared)
        T.copy(C_shared, C[by * block_M, bx * block_N])

    return C


def main(M=4096, N=4096, K=8192):
    block_M, block_N, block_K = 128, 128, 128
    trans_A, trans_B = False, True
    in_dtype, out_dtype, accum_dtype = T.bfloat16, T.bfloat16, T.float
    num_stages = 0 if max(block_M, block_N, block_K) >= 256 else 2
    threads = 256

    import torch

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = matmul_tcgen05(a, b, block_M, block_N, block_K, trans_A, trans_B,
                       in_dtype, out_dtype, accum_dtype, num_stages, threads)

    ref_c = (a.to(torch.float) @ b.T.to(torch.float)).to(torch.bfloat16)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

    # Uncomment to read the generated CUDA (look for tcgen05 and TMEM handling).
    # print(matmul_tcgen05.get_kernel_source(
    #     a, b, block_M, block_N, block_K, trans_A, trans_B,
    #     in_dtype, out_dtype, accum_dtype, num_stages, threads))

    from tilelang.profiler import do_bench

    latency = do_bench(
        lambda: matmul_tcgen05(a, b, block_M, block_N, block_K, trans_A,
                               trans_B, in_dtype, out_dtype, accum_dtype,
                               num_stages, threads))
    print(f"TileLang latency on this machine: {latency} ms (machine-specific)")


if __name__ == "__main__":
    main()
