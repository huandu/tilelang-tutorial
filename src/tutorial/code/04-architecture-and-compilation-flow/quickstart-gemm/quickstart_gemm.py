#!/usr/bin/env python3
"""Walk the compile path of chapter 04 with the smallest official kernel.

The kernel body is taken from `_vendor/tilelang-v0.1.15/examples/quickstart.py`
(lines 8-48). The additions are the ``debug_root_path`` and the step-by-step
prints, which make each stage of figure 2 visible from a real run.

Hardware: this needs a CUDA GPU (or replace the target with a supported one).
On a machine without a GPU the script stops early with exit code 3 and prints
what it could not check. It never fabricates a result.

Usage:
    python3 quickstart_gemm.py                      # dump into ./out
    python3 quickstart_gemm.py --out /tmp/tl-debug
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./out", help="directory for debug_root_path dumps")
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=32)
    return parser.parse_args()


def require_cuda() -> None:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed; this example cannot run.", file=sys.stderr)
        raise SystemExit(3)
    if not torch.cuda.is_available():
        print(
            "No CUDA GPU detected. Chapter 04 marks this example as "
            "'静态核对（py_compile + API 名比对）' on this machine.",
            file=sys.stderr,
        )
        raise SystemExit(3)


def build_kernel():
    import tilelang
    import tilelang.language as T

    # --- stage: decorator -> JITImpl (chapter figure 2 step 1-2) ------------
    @tilelang.jit(out_idx=[-1], debug_root_path=OUT_DIR)
    def matmul_relu(M: int, N: int, K: int, block_M: int, block_N: int, block_K: int):
        dtype = T.float16
        accum_dtype = T.float32

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            # --- stage: language surface -> IRModule (step 3) ---------------
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), dtype)
                B_shared = T.alloc_shared((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[ko * block_K, bx * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)
                for i, j in T.Parallel(block_M, block_N):
                    C_local[i, j] = T.max(C_local[i, j], 0)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return matmul_relu


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    globals()["OUT_DIR"] = str(out_dir)

    require_cuda()

    import torch

    matmul_relu = build_kernel()
    M = N = K = 1024

    print(f"[1-3] elaborating PrimFunc for M=N=K={M} ...")
    prim_func = matmul_relu.get_tir(
        M=M, N=N, K=K, block_M=args.block_m, block_N=args.block_n, block_K=args.block_k
    )
    print(f"      IRModule functions: {list(prim_func.attrs) if prim_func.attrs else '{}'}")

    print("[4] compiling (cache lookup happens inside tilelang.cache.cached) ...")
    kernel = matmul_relu.compile(
        M=M, N=N, K=K, block_M=args.block_m, block_N=args.block_n, block_K=args.block_k
    )

    print("[9] generated device source (first 20 lines):")
    for line in kernel.get_kernel_source().splitlines()[:20]:
        print(f"      {line}")
    print(f"[9] debug dumps written under {out_dir}")

    print("[12-13] calling the compiled kernel through the adapter")
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = kernel(a, b)
    torch.testing.assert_close(c, torch.relu(a @ b), rtol=1e-2, atol=1e-2)
    print("      result matches torch.relu(a @ b)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
