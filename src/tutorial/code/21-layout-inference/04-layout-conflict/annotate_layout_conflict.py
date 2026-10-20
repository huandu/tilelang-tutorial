#!/usr/bin/env python3
"""第 21 章 §3.3 / §7.5：用 T.annotate_layout 制造一次布局冲突。

对照源码：
  src/transform/layout_inference/layout_inference.cc:437-468   冲突三条出路
  src/op/gemm.cc:254-266                                        reuse_existing_shared_layout
  tilelang/language/annotations.py:45-62                        annotate_layout 的契约

用法：
  python3 annotate_layout_conflict.py --mode transpose
  python3 annotate_layout_conflict.py --mode plain
"""

from __future__ import annotations

import argparse

import tilelang
import tilelang.language as T


def build(mode: str, block_M: int = 128, block_N: int = 128,
          block_K: int = 32, dtype=T.float16, accum_dtype=T.float32):
    @tilelang.jit
    def matmul(A, B):
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((K, N), dtype)
        C = T.empty((M, N), dtype)

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            if mode == "transpose":
                # 与 swizzle 不兼容的纯转置布局：用来触发冲突路径。
                bad = T.Layout([block_M, block_K],
                               lambda i, j: (j, i))
                T.annotate_layout({A_shared: bad})

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

        return C

    return matmul


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["plain", "transpose"],
                        default="transpose")
    args = parser.parse_args()

    matmul = build(args.mode)
    print(f"[info] 已构造 mode={args.mode} 的 kernel 包装。")
    print("[info] 本机没有 GPU；要观察冲突，请在有卡机器上做一次带数据的调用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
