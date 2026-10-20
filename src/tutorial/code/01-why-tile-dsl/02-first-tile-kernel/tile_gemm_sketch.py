"""第 01 章示例 2：用 tile 视角写一个 GEMM + ReLU kernel（仅静态核对）。

本文件的目的**不是**教语法（那是第 7–9 章的事），而是让读者看到「一个 tile 程序
长什么样」：网格从元素变成块、数据显式声明存储位置、搬运与计算是 tile 级原语、
流水只是一个参数。

来源与原型的说明
----------------
本文件的结构与 API 用法对标锁定版本 v0.1.15 中的两个官方示例：

    _vendor/tilelang-v0.1.15/examples/quickstart.py                 （GEMM + ReLU）
    _vendor/tilelang-v0.1.15/examples/gemm/example_gemm.py          （最小 GEMM）
    _vendor/tilelang-v0.1.15/examples/elementwise/example_elementwise_add.py

用到的每个 API 都已在 v0.1.15 源码中核对过存在性与签名：

    T.Kernel            tilelang/language/kernel.py:431
    T.alloc_shared      tilelang/language/allocate.py:49
    T.alloc_fragment    tilelang/language/allocate.py:81
    T.copy              tilelang/language/copy_op.py:59
    T.gemm              tilelang/language/gemm_op.py:160
    T.Parallel          tilelang/language/loop.py:13
    T.Pipelined         tilelang/language/loop.py:112
    T.clear             tilelang/language/fill_op.py:48
    T.const             tilelang/language/eager/builder.py:1177
    T.Tensor            tilelang/language/proxy.py:256
    T.empty             tilelang/language/allocate.py:400

验证级别：**仅静态核对（static check only）**。

本章的写作环境没有 GPU，因此本文件**没有在本机运行过**，也不给出任何性能数字。
在有 NVIDIA GPU 的机器上运行需要：

  * Python 3.11+、PyTorch（CUDA 版）、可用的 CUDA toolkit；
  * 安装锁定版本的 TileLang：见 _vendor/tilelang-v0.1.15/docs/get_started/Installation.md；
  * 运行：`python3 tile_gemm_sketch.py`。

在没有 GPU 的机器上，仍可做两项离线检查：

    python3 -m py_compile tile_gemm_sketch.py     # 纯语法检查
    python3 tile_gemm_sketch.py --static-only     # 只导入并打印说明，不编译 kernel
"""

from __future__ import annotations

import sys


def build_kernels(tilelang, T):
    """把两个 kernel 定义在函数内部，只是为了把 import 推迟到运行时。

    两个 kernel 的写法与锁定版本 v0.1.15 的官方示例保持一致：

      * gemm_relu  —— 对标 examples/quickstart.py
      * elementwise_add —— 对标 examples/elementwise/example_elementwise_add.py
    """

    @tilelang.jit
    def elementwise_add(x, z, block_M: int, block_N: int, in_dtype: str, threads: int):
        """y = x + z，按 (block_M, block_N) 分块处理。

        这是「tile 视角」的最小形态：网格是 tile 的网格，而不是元素的网格。
        对照 CUDA 的写法，这里没有任何 threadIdx / blockIdx 的运算。
        """
        M, N = T.const("M, N")
        x: T.Tensor((M, N), in_dtype)
        z: T.Tensor((M, N), in_dtype)
        y = T.empty((M, N), in_dtype)

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            x_shared = T.alloc_shared((block_M, block_N), in_dtype)
            z_shared = T.alloc_shared((block_M, block_N), in_dtype)
            y_local = T.alloc_fragment((block_M, block_N), in_dtype)
            y_shared = T.alloc_shared((block_M, block_N), in_dtype)

            # 数据流：把两块输入搬到片上。
            T.copy(x[by * block_M, bx * block_N], x_shared)
            T.copy(z[by * block_M, bx * block_N], z_shared)

            # 计算：逐元素相加，仍然是 tile 级的并行域。
            for i, j in T.Parallel(block_M, block_N):
                y_local[i, j] = x_shared[i, j] + z_shared[i, j]

            # 回写。
            T.copy(y_local, y_shared)
            T.copy(y_shared, y[by * block_M, bx * block_N])
        return y

    @tilelang.jit
    def matmul_relu(A, B, block_M: int, block_N: int, block_K: int):
        """C = relu(A @ B)：本章正文第 3.2 节展示的那个例子。

        三个要点：
          1. with T.Kernel(...) 定义的是「tile 程序的网格」；
          2. T.alloc_shared / T.alloc_fragment 显式声明存储位置；
          3. T.Pipelined(num_stages=3) 用一行表达「多缓冲 + 让搬运与计算重叠」。
        """
        M, N, K = T.const("M, N, K")
        dtype = T.float16
        accum_dtype = T.float32
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((K, N), dtype)
        C = T.empty((M, N), dtype)

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            # 融合的 ReLU：仍然只在片上数据上做，不需要额外的一次全局访存。
            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])
        return C

    return elementwise_add, matmul_relu


def static_only_report() -> int:
    print("第 01 章示例 2：tile 视角的 GEMM + ReLU（仅静态核对）")
    print("-" * 66)
    print("本文件在无 GPU 时只能做语法/API 核对，不能运行 kernel。")
    print()
    print("已核对的 API（v0.1.15）：")
    for line in (
        "  T.Kernel          -> tilelang/language/kernel.py:431",
        "  T.alloc_shared    -> tilelang/language/allocate.py:49",
        "  T.alloc_fragment  -> tilelang/language/allocate.py:81",
        "  T.copy            -> tilelang/language/copy_op.py:59",
        "  T.gemm            -> tilelang/language/gemm_op.py:160",
        "  T.Parallel        -> tilelang/language/loop.py:13",
        "  T.Pipelined       -> tilelang/language/loop.py:112",
        "  T.clear           -> tilelang/language/fill_op.py:48",
        "  T.const           -> tilelang/language/eager/builder.py:1177",
        "  T.Tensor          -> tilelang/language/proxy.py:256",
        "  T.empty           -> tilelang/language/allocate.py:400",
    ):
        print(line)
    print()
    print("在有 GPU 的机器上，把 --static-only 去掉即可编译并运行。")
    return 0


def main(argv: list[str]) -> int:
    if "--static-only" in argv:
        return static_only_report()

    try:
        import torch
        import tilelang
        import tilelang.language as T
    except ImportError as exc:  # pragma: no cover - 只有在目标机器上才会走到
        print(f"缺少依赖：{exc}", file=sys.stderr)
        print("请先按 _vendor/tilelang-v0.1.15/docs/get_started/Installation.md 安装。")
        print("没有 GPU 时可用 --static-only 查看核对信息。")
        return 2

    if not torch.cuda.is_available():
        print("未检测到 CUDA 设备；本示例需要 NVIDIA GPU。", file=sys.stderr)
        print("可改用 --static-only。")
        return 3

    elementwise_add, matmul_relu = build_kernels(tilelang, T)

    # --- GEMM + ReLU ---
    kernel = matmul_relu.compile(
        M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32
    )
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    c = kernel(a, b)
    ref = torch.relu(a @ b)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    print("GEMM + ReLU 正确性通过（对齐 PyTorch 参考实现，rtol=atol=1e-2）。")

    source = kernel.get_kernel_source()
    print("生成的设备源码前 40 行：")
    print("\n".join(source.splitlines()[:40]))

    # --- elementwise ---
    x = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    z = torch.randn(256, 256, device="cuda", dtype=torch.float32)
    y = elementwise_add(x, z, block_M=32, block_N=32, in_dtype="float32", threads=128)
    torch.testing.assert_close(y, x + z, rtol=1e-5, atol=1e-5)
    print("elementwise add 正确性通过。")

    print("本文件不输出性能数字；本章没有在本机运行过任何 kernel。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
