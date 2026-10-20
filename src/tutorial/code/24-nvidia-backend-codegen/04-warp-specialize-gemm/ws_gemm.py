#!/usr/bin/env python3
"""Hand-written warp-specialized GEMM (producer / consumer roles).

Prototype: _vendor/tilelang-v0.1.15/examples/warp_specialize/
           example_warp_specialize_gemm_copy_0_gemm_1.py

Requires: NVIDIA Hopper (sm_90) or newer, CUDA 12.x, PyTorch(CUDA),
          tilelang==0.1.15. This file has only been statically checked; see
          README.md.
"""

import tilelang
import tilelang.language as T


@tilelang.jit
def matmul_warp_specialize(A, B, block_M, block_N, block_K,
                           dtype=T.float16, accum_dtype=T.float32):
    """C = A @ B with a copy/gemm producer-consumer split.

    Role 0 (tx < 128) issues the TMA loads; role 1 (128 <= tx < 256) computes.
    The two roles hand off through a pair of mbarriers.
    """
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # arrive_count = 128: one warp group of threads participates.
        data_is_ready = T.alloc_barrier(arrive_count=128)
        compute_is_done = T.alloc_barrier(arrive_count=128)

        with T.ws(1):
            T.clear(C_local)

        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
            with T.ws(0):
                # Wait until the consumer is done with the previous round.
                T.barrier_wait(compute_is_done, (ko + 1) % 2)
                T.tma_copy(A[by * block_M, ko * block_K], A_shared,
                           barrier=data_is_ready)
                T.tma_copy(B[ko * block_K, bx * block_N], B_shared,
                           barrier=data_is_ready)
                T.barrier_arrive(data_is_ready)
            with T.ws(1):
                T.barrier_wait(data_is_ready, ko % 2)
                T.gemm(A_shared, B_shared, C_local)
                T.barrier_arrive(compute_is_done)

        with T.ws(1):
            T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def main(M=1024, N=1024, K=1024):
    block_M, block_N, block_K = 128, 128, 64

    import torch

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    c = matmul_warp_specialize(a, b, block_M, block_N, block_K)
    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

    # Inspect the generated CUDA: uncomment to look for tl::tma_load,
    # mbarrier arrive/wait pairs and the two `if` role guards.
    # print(matmul_warp_specialize.get_kernel_source(a, b, block_M, block_N, block_K))

    from tilelang.profiler import do_bench

    latency = do_bench(
        lambda: matmul_warp_specialize(a, b, block_M, block_N, block_K))
    print(f"Latency on this machine: {latency} ms (machine-specific)")


if __name__ == "__main__":
    main()
