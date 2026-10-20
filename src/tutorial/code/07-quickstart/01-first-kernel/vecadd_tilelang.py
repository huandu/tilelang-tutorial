"""第七章 · 第一个 TileLang kernel：一维向量加法。

这份脚本只有两个目的：

1. 让你把一个 TileLang kernel 从头写完整（版本 A：最朴素的写法；版本 B：显式
   分配片上 buffer 再用 ``T.copy`` 搬运）。
2. 编译它，跑一次正确性校验，并把生成的 CUDA 源码打出来
   （``kernel.get_kernel_source()``）。

运行前提
--------
* Python >= 3.10（见本章 3.1 节）。
* 已经安装 ``tilelang``（wheel 或源码构建，见本章 3.2 / 3.3 节）。
* 一块 NVIDIA GPU。脚本里用 ``torch.cuda`` 造数据，没有 GPU 会在运行时失败。

运行
----
    python vecadd_tilelang.py                 # 默认 N=1<<20（约 100 万元素）
    python vecadd_tilelang.py --n 1048576
    python vecadd_tilelang.py --variant a     # 只跑版本 A

期望输出（在真实机器上）
------------------------
    [variant a] correctness OK (n=1048576, block_N=256, threads=256)
    ----- generated CUDA (variant a) -----
    extern "C" __global__ void __launch_bounds__(256) main_kernel(...) { ... }
    [variant b] correctness OK (n=1048576, block_N=256, threads=256)

验证级别
--------
**仅静态核对**。作者本机是 macOS（Apple Silicon），没有 NVIDIA GPU，也没有安装
tilelang，因此未运行过本脚本。脚本用到的每个 API（``T.prim_func`` / ``T.Kernel`` /
``T.alloc_fragment`` / ``T.copy`` / ``T.Parallel`` / ``T.ceildiv`` / ``tilelang.compile`` /
``JITKernel.get_kernel_source``）都逐个在 ``_vendor/tilelang-v0.1.15/`` 里核对过名称与
签名，详见同目录 README 与本章 references.md。**不要把这段「静态核对」读成「跑过」。**
"""

from __future__ import annotations

import argparse

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# 版本 A：最朴素的写法。没有片上 buffer，直接在全局内存上做逐元素加法。
# 对应官方教程 docs/deeplearning_operators/elementwise.md 的第一段示例。
# ---------------------------------------------------------------------------
def vector_add_direct(N: int, block_N: int = 256, threads: int = 256, dtype: str = "float16"):
    """C = A + B 的一维逐元素版本，不做任何显式搬运。"""

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        # T.Kernel 的每一个实例负责 C 的一段连续 block_N 个元素。
        # 网格大小 = ceil(N / block_N)，每块 threads 个线程。
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            # T.Parallel 是一个「块级并行域」：你只描述「要对 block_N 个元素各做一次」，
            # 由哪个线程做哪几个、要不要向量化，交给编译器。
            for i in T.Parallel(block_N):
                C[bx * block_N + i] = A[bx * block_N + i] + B[bx * block_N + i]

    return main


# ---------------------------------------------------------------------------
# 版本 B：显式分配片上 buffer，用 T.copy 做 tile 级搬运，再在 buffer 上计算。
# 对应官方 docs/deeplearning_operators/elementwise.md 的第三段示例，以及
# examples/elementwise/example_elementwise_add.py 的结构。
# ---------------------------------------------------------------------------
def vector_add_buffered(N: int, block_N: int = 256, threads: int = 256, dtype: str = "float16"):
    """同上，但先把 A/B 的 tile 搬进 fragment，再计算、再写回。"""

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            # alloc_fragment = 每个线程可见的寄存器切片（以整块的比例视角书写）。
            A_local = T.alloc_fragment((block_N,), dtype)
            B_local = T.alloc_fragment((block_N,), dtype)
            C_local = T.alloc_fragment((block_N,), dtype)

            # T.copy 是 tile 级搬运：源给「首地址」，形状从目标推断。
            T.copy(A[bx * block_N], A_local)
            T.copy(B[bx * block_N], B_local)

            for i in T.Parallel(block_N):
                C_local[i] = A_local[i] + B_local[i]

            T.copy(C_local, C[bx * block_N])

    return main


VARIANTS = {"a": vector_add_direct, "b": vector_add_buffered}


def main() -> None:
    parser = argparse.ArgumentParser(description="第七章第一个 kernel：一维向量加法")
    parser.add_argument("--n", type=int, default=1 << 20, help="向量长度（默认 1048576）")
    parser.add_argument("--block-n", type=int, default=256, help="每个 tile 处理的元素数")
    parser.add_argument("--threads", type=int, default=256, help="每个 block 的线程数")
    parser.add_argument("--variant", choices=["a", "b", "all"], default="all")
    args = parser.parse_args()

    # torch 只在真正要执行时才需要。
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("本脚本需要一个可用的 NVIDIA GPU：torch.cuda.is_available() 为 False。")

    dtype = "float16"
    torch_dtype = torch.float16

    a = torch.randn(args.n, dtype=torch_dtype, device="cuda")
    b = torch.randn(args.n, dtype=torch_dtype, device="cuda")
    ref = a + b

    todo = ["a", "b"] if args.variant == "all" else [args.variant]
    for name in todo:
        builder = VARIANTS[name]
        prim_func = builder(args.n, block_N=args.block_n, threads=args.threads, dtype=dtype)
        # tilelang.compile 返回一个 JITKernel；out_idx=-1 表示把最后一个参数
        # （这里是 C）当作输出并作为返回值。
        kernel = tilelang.compile(prim_func, out_idx=-1, target="cuda")
        c = kernel(a, b)
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
        print(f"[variant {name}] correctness OK (n={args.n}, block_N={args.block_n}, threads={args.threads})")

        if name == "a":
            print(f"----- generated CUDA (variant {name}) -----")
            print(kernel.get_kernel_source())


if __name__ == "__main__":
    main()
