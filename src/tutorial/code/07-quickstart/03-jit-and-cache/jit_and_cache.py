"""第七章 · @tilelang.jit 的两种用法与编译缓存。

这个脚本把本章第 6 节讲的三件事落到可执行代码上：

1. **lazy 模式**：``@tilelang.jit`` 修饰「返回 PrimFunc」的函数，调用得到的是
   一个已经编译好的 kernel 对象（``JITKernel``），可以反复调用、可查看生成源码。
2. **eager 模式**：``@tilelang.jit`` 修饰「用 DSL 直接写 body」的函数，调用时
   编译并立即执行，直接返回结果张量（官方 ``examples/elementwise`` 就是这种）。
3. **缓存**：同一份 kernel 用同样的参数再编译一次，第二次应当命中缓存、
   不再走 nvcc；改一个参数就会 miss。

运行前提
--------
* Python >= 3.10；已安装 tilelang；一块 NVIDIA GPU。

运行
----
    python jit_and_cache.py
    TILELANG_VERBOSE=1 python jit_and_cache.py    # 看「cache hit / miss」日志
    TILELANG_CACHE_DIR=/tmp/tl-cache python jit_and_cache.py   # 换个缓存目录，强制冷启动

验证级别
--------
**仅静态核对**（作者无 GPU、未安装 tilelang，未运行）。API 名与签名在
``_vendor/tilelang-v0.1.15/`` 里核对过：``tilelang.jit``（``tilelang/jit/__init__.py:588``）、
``JITImpl.compile``（``:466``）、``JITKernel.get_kernel_source``（``tilelang/jit/kernel.py:532``）、
``tilelang.disable_cache``（``tilelang/env.py:559``）。
"""

from __future__ import annotations

import tilelang
import tilelang.language as T


# --- 用法一：lazy 模式，函数自己返回 PrimFunc --------------------------------
# 调用 build_add(...) 拿到的不是结果，而是一个 JITKernel；想跑它得再调用一次。
@tilelang.jit(out_idx=-1, target="cuda")
def build_add(N: int, block_N: int = 256, threads: int = 256, dtype: str = "float16"):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            for i in T.Parallel(block_N):
                C[bx * block_N + i] = A[bx * block_N + i] + B[bx * block_N + i]

    return main


# --- 用法二：eager 模式，用 DSL 直接写 body ----------------------------------
# 调用 add_eager(a, b) 会就地编译并执行，直接返回结果张量。
# 形状用 T.const 声明成符号，运行时从输入张量推断（与官方
# examples/elementwise/example_elementwise_add.py 同一套写法）。
@tilelang.jit
def add_eager(A, B, block_N: int = 256, threads: int = 256):
    N = T.const("N")
    dtype = "float16"
    A: T.Tensor((N,), dtype)
    B: T.Tensor((N,), dtype)
    C = T.empty((N,), dtype)

    with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
        for i in T.Parallel(block_N):
            C[bx * block_N + i] = A[bx * block_N + i] + B[bx * block_N + i]

    return C


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("需要一个可用的 NVIDIA GPU。")

    n = 1 << 20
    a = torch.randn(n, dtype=torch.float16, device="cuda")
    b = torch.randn(n, dtype=torch.float16, device="cuda")
    ref = a + b

    # --- lazy：先拿 kernel，再决定什么时候跑、跑几次 -----------------------
    kernel = build_add(n)                 # 这里发生「编译 or 命中缓存」
    print("lazy  : type(kernel) =", type(kernel).__name__)
    c1 = kernel(a, b)                     # 这里是「复用」：不再编译
    c2 = kernel(a, b)                     # 同一个 kernel，反复调用
    torch.testing.assert_close(c1, ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(c2, ref, rtol=1e-2, atol=1e-2)
    print("lazy  : correctness OK；同一个 kernel 调用了两次")

    # 换一组参数 = 新 key = 另一次编译（或命中另一条缓存）
    kernel_512 = build_add(n, block_N=512, threads=512)
    torch.testing.assert_close(kernel_512(a, b), ref, rtol=1e-2, atol=1e-2)
    print("lazy  : block_N=512 是另一份 kernel（另一条缓存）")

    # --- eager：直接执行 ---------------------------------------------------
    c3 = add_eager(a, b)
    torch.testing.assert_close(c3, ref, rtol=1e-2, atol=1e-2)
    print("eager : correctness OK；调用即编译 + 执行")

    # --- 生成源码 ---------------------------------------------------------
    print("----- generated CUDA (lazy, block_N=256) -----")
    print(kernel.get_kernel_source())

    # --- 缓存开关 ---------------------------------------------------------
    # 调试编译器时常用：把缓存关掉，避免「改了 pass 却不生效」的假象。
    tilelang.disable_cache()
    try:
        _ = build_add(n, block_N=128, threads=128)
    finally:
        tilelang.enable_cache()
    print("cache : 已演示 disable_cache()/enable_cache() 一轮")


if __name__ == "__main__":
    main()
