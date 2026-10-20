#!/usr/bin/env python3
"""第 21 章 §6.3：打开编译期 layout visualization，观察推断出的 fragment 布局。

参考官方示例：
  _vendor/tilelang-v0.1.15/examples/visual_layout_inference/visual_layout_inference.py
配置键：
  tilelang/transform/pass_config.py:252-256
  TL_LAYOUT_VISUALIZATION_ENABLE / TL_LAYOUT_VISUALIZATION_FORMATS

用法：
  python3 gemm_layout_visualization.py            # txt + svg
  python3 gemm_layout_visualization.py --txt-only # 只打印文本
"""

from __future__ import annotations

import sys

import tilelang
import tilelang.language as T


def make_kernel(formats: str):
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
            tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: formats,
        }
    )
    def matmul(A, B, block_M, block_N, block_K,
               dtype=T.float16, accum_dtype=T.float32):
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((K, N), dtype)
        C = T.empty((M, N), dtype)

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

        return C

    return matmul


def main() -> int:
    txt_only = "--txt-only" in sys.argv
    formats = "txt" if txt_only else "txt,svg"
    matmul = make_kernel(formats)

    # 编译即触发 layout 可视化（打印 + 存图）。
    # 运行 kernel 需要 GPU；这里让用户自己选择是否执行带数据的调用。
    try:
        import torch
    except ImportError:
        print("[info] 未安装 torch，跳过运行；仅完成 JIT 包装。",
              file=sys.stderr)
        return 0

    if not torch.cuda.is_available():
        print("[info] 没有可用 CUDA 设备，跳过运行；布局输出需编译期产生。",
              file=sys.stderr)
        return 0

    a = torch.randn(128, 128).cuda().half()
    b = torch.randn(128, 128).cuda().half()
    c = matmul(a, b, 32, 32, 32)
    ref = a @ b
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    print("correctness check passed; see ./tmp for layout figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
