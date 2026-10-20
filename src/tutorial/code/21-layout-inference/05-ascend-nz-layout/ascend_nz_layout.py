#!/usr/bin/env python3
"""第 21 章 §5：昇腾分形（fractal / NZ）布局公式与语义标签。

对照源码：
  src/ascend/layout/ascend_layouts.cc:111-115   AscendC0 = 256 / element_bits
  src/ascend/layout/ascend_layouts.h:20-27      物理形状 [.., cols/C0, rows/16, 16, C0]
  src/ascend/layout/ascend_layouts.cc:254-274   L0C 固定 16x16
  tilelang/layout/ascend.py:20-206              Python 包装

用法：
  python3 ascend_nz_layout.py --dry-shape
  python3 ascend_nz_layout.py --compile        # 需要昇腾 NPU
"""

from __future__ import annotations

import argparse

DTYPE_BITS = {"float16": 16, "bfloat16": 16, "float32": 32,
              "int8": 8, "int4": 4}


def ascend_c0(element_bits: int) -> int:
    assert element_bits > 0 and 256 % element_bits == 0, element_bits
    return 256 // element_bits


def fractal_shape(rows: int, cols: int, element_bits: int):
    """MajorK 的物理形状：[cols/C0, rows/16, 16, C0]。"""
    c0 = ascend_c0(element_bits)
    assert cols % c0 == 0, f"cols={cols} must be a multiple of C0={c0}"
    assert rows % 16 == 0, f"rows={rows} must be a multiple of 16"
    return [cols // c0, rows // 16, 16, c0]


def dry_shape() -> None:
    for name, bits in DTYPE_BITS.items():
        print(f"{name:9s}: C0 = 256/{bits} = {ascend_c0(bits)}")
    print("bf16 [64, 48] MajorK ->", fractal_shape(64, 48, 16))
    print("fp32 [64, 48] MajorK ->", fractal_shape(64, 48, 32))


def compile_demo() -> int:
    """最小昇腾 GEMM，编译时观察推断布局（需要 NPU + 昇腾后端）。"""
    import tilelang
    import tilelang.ascend.language as T

    M, N, K = 256, 256, 256
    block_M, block_N, block_K = 64, 64, 64
    M_TILES, N_TILES, K_TILES = M // block_M, N // block_N, K // block_K

    @tilelang.jit
    def gemm(A, B):
        A: T.Tensor((M, K), T.bfloat16)
        B: T.Tensor((K, N), T.bfloat16)
        C = T.empty((M, N), T.bfloat16)
        with T.Kernel(M_TILES * N_TILES) as bx:
            m_tile = bx // N_TILES
            n_tile = bx % N_TILES
            A_l1 = T.alloc_l1((block_M, block_K), T.bfloat16)
            B_l1 = T.alloc_l1((block_K, block_N), T.bfloat16)
            C_l0 = T.alloc_l0c((block_M, block_N), T.float32)
            for k in T.Pipelined(K_TILES, num_stages=2):
                T.copy(A[m_tile * block_M, k * block_K], A_l1)
                T.copy(B[k * block_K, n_tile * block_N], B_l1)
                T.gemm(A_l1, B_l1, C_l0, transpose_B=True, clear_accum=k == 0)
            T.copy(C_l0, C[m_tile * block_M, n_tile * block_N])
        return C

    print("[info] 已构造昇腾 GEMM 包装；在有 NPU 的机器上触发编译即可看到推断布局。")
    _ = gemm
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-shape", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if args.compile:
        return compile_demo()
    dry_shape()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
