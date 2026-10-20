"""第七章 · 打印并对照 Python 与生成的 CUDA 源码。

这份脚本**不需要 tilelang、不需要 GPU**：它把本章要讲的「Python ↔ 生成代码」对照
关系固化成一张表，并把锁定版本仓库里官方文档给出的两份真实生成结果原样打印出来，
方便你在真机上拿到 ``kernel.get_kernel_source()`` 的输出来后逐行比对。

它不能替代在真机上调用 ``get_kernel_source()``——那一步必须装好 tilelang 并有一块
GPU。这里只是把「该看什么、该期待什么」变成可 diff 的文本。

运行：
    python inspect_source.py

验证级别：**在本机运行过（纯 Python，无 GPU）**。脚本本身不 import tilelang，
打印的 CUDA 片段逐字取自锁定版本仓库里的官方文档
``_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md``。
"""

from __future__ import annotations

import sys
import textwrap


PYTHON_KERNEL_A = '''\
def elementwise_add(N, threads=256, dtype=T.bfloat16):
    @T.prim_func
    def main(A: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as (b_x):
            for i in T.Parallel(threads):
                C[b_x * threads + i] = A[b_x * threads + i] + B[b_x * threads + i]
    return main
'''

# 官方文档 docs/deeplearning_operators/elementwise.md:155-172
# 编译 elementwise_add(1024, threads=256, dtype=bfloat16) 得到的 CUDA
CUDA_VECTORIZED = r'''
extern "C" __global__ void __launch_bounds__(256) main_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  if (((int)threadIdx.x) < 32) {
    uint4 __1;
      uint4 v_ = *(uint4*)(A + ((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 8)));
      uint4 v__1 = *(uint4*)(B + ((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 8)));
      ((nv_bfloat162*)(&(__1.x)))->x = (((nv_bfloat162*)(&(v_.x)))->x+((nv_bfloat162*)(&(v__1.x)))->x);
      ((nv_bfloat162*)(&(__1.x)))->y = (((nv_bfloat162*)(&(v_.x)))->y+((nv_bfloat162*)(&(v__1.x)))->y);
      ...
    *(uint4*)(C + ((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 8))) = __1;
  }
}
'''

# 官方文档 docs/deeplearning_operators/elementwise.md:122-130
# 编译 N=2047（不能整除）的版本得到的 CUDA：带边界判断，未向量化
CUDA_TAIL_CHECKED = r'''
extern "C" __global__ void __launch_bounds__(256) main_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (((i * 256) + ((int)threadIdx.x)) < 2047) {
      C[((i * 256) + ((int)threadIdx.x))] = (A[((i * 256) + ((int)threadIdx.x))] + B[((i * 256) + ((int)threadIdx.x))]);
    }
  }
}
'''

MAPPING = [
    (
        "with T.Kernel(T.ceildiv(N, threads), threads=threads) as (b_x)",
        'extern "C" __global__ void __launch_bounds__(256) main_kernel(...)',
        "网格大小 = ceil(N/256) 对应 gridDim.x；threads=256 变成 __launch_bounds__(256)；b_x 对应 blockIdx.x。",
    ),
    (
        "for i in T.Parallel(threads)",
        "if (((int)threadIdx.x) < 32) { ... }",
        "T.Parallel 是块级并行域，不指定线程号。LayoutInference 把它映射到线程：这里每线程处理 8 个元素，所以只需要 32 个线程。",
    ),
    (
        "C[b_x * threads + i] = A[...] + B[...]",
        "uint4 v_ = *(uint4*)(A + ...); ((nv_bfloat162*)(&(__1.x)))->x = ...",
        "逐元素代码被向量化成 128-bit 访存（uint4 = 16 字节 = 8 个 bfloat16）与成对加法。",
    ),
    (
        "（Python 里没有 threadIdx.x）",
        "((int)threadIdx.x) * 8",
        "TileLang 自动插入线程索引，并把 stride 设成向量宽度 8。i 这个循环变量在生成代码里消失了。",
    ),
    (
        "（N 不能整除 tile 时）",
        "if (((i * 256) + ((int)threadIdx.x)) < 2047)",
        "编译器自动补边界判断；代价是这一版不做向量化。",
    ),
]


def print_mapping() -> None:
    print("=" * 78)
    print("Python ↔ 生成 CUDA 对照表")
    print("=" * 78)
    for py, cu, why in MAPPING:
        print()
        print("  Python : " + py)
        print("  CUDA   : " + textwrap.shorten(cu, width=100, placeholder=" …"))
        print("  说明   : " + textwrap.fill(why, width=74, subsequent_indent="           "))
    print()


def main() -> int:
    print_mapping()
    print("=" * 78)
    print("生成结果 ①：elementwise_add(1024, threads=256) —— 已向量化")
    print("来源：_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md:155-172")
    print("=" * 78)
    print("Python 输入：")
    print(PYTHON_KERNEL_A)
    print("生成 CUDA（节选，省略号处是重复的成对加法）：")
    print(CUDA_VECTORIZED)

    print("=" * 78)
    print("生成结果 ②：N=2047（无法整除）—— 带边界判断，未向量化")
    print("来源：_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md:122-130")
    print("=" * 78)
    print(CUDA_TAIL_CHECKED)
    print("把你自己机器上 kernel.get_kernel_source() 的输出贴到上面两种形态旁边，")
    print("先判断命中哪一种，再去解释为什么。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
