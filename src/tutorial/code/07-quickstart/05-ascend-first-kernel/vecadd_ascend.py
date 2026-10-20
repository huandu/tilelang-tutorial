"""第七章 · 昇腾侧的第一个 kernel（对照用）。

同一个 ``C = A + B``，换到昇腾上写，会看到三处结构性差异：

1. 方言换成 ``tilelang.ascend.language``（不是 ``tilelang.language``）；
2. ``T.Kernel`` 是**一维核网格**，没有 ``threads=``；线程域改由
   ``with T.SimtVF(threads=...)`` 单独声明；
3. 片上存储从 ``T.alloc_shared``（UB）开始，``T.gemm`` 的结果走
   ``T.dual_copy`` 从 L0C 搬到 UB。

本文件用 v0.1.15 主干里的 ``tilelang.ascend.language`` 写，是「主干昇腾后端」的写法，
和 ``_vendor/tilelang-ascend-v0.1.2.000`` 那个独立仓库的方言（``T.Kernel(..., is_npu=True)``、
``T.alloc_ub``、``T.Scope("V")``）**不是同一套**。两个方言的区别见本章 7.2 节，
细节留到第 19、25 章。

运行前提
--------
* 一台装好 CANN 的昇腾机器；``torch_npu`` 可用，且 ``torch.npu.is_available()`` 为 True。
* 已经安装 Ascend 版 tilelang（见本章 3.4 节）。

运行
----
    python vecadd_ascend.py

验证级别
--------
**仅静态核对**（作者无昇腾 NPU，未运行）。本文件与
``_vendor/tilelang-v0.1.15/examples/ascend/example_vecadd.py`` 同构，只是把规模压到
最小、并去掉了性能与 SIMD 分支。API 名与签名在
``_vendor/tilelang-v0.1.15/tilelang/ascend/language/`` 里核对过：
``kernel.py:139``（``T.Kernel``）、``frame.py:107``（``T.SimtVF``）、
``allocate.py:15``（``alloc_shared`` → UB）。
"""

from __future__ import annotations

import tilelang
import tilelang.ascend.language as T


def vector_add(N: int, num_blocks: int = 64, tile: int = 8192, threads: int = 2048, dtype: str = "float32"):
    """昇腾上一维向量加法：核网格 × SimtVF 线程域 × 分块搬运。"""

    if N % (tile * num_blocks) != 0:
        raise ValueError(f"N 必须是 tile*num_blocks={tile * num_blocks} 的整数倍")
    iterations = N // (tile * num_blocks)

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        # 一维核网格：每个 core 处理若干 tile；没有 threads=。
        with T.Kernel(num_blocks) as bx:
            a = T.alloc_shared((tile,), dtype)   # 昇腾：alloc_shared = UB
            b = T.alloc_shared((tile,), dtype)
            c = T.alloc_shared((tile,), dtype)

            for step in T.Pipelined(iterations, num_stages=2):
                begin = (step * num_blocks + bx) * tile
                T.copy(A[begin : begin + tile], a)
                T.copy(B[begin : begin + tile], b)
                # 线程域在这里单独声明；T.Parallel 负责把它铺到这些线程上。
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(tile):
                        c[i] = a[i] + b[i]
                T.copy(c, C[begin : begin + tile])

    return main


def main() -> None:
    import torch
    import torch_npu  # noqa: F401  （导入即注册 npu 设备）

    if not torch.npu.is_available():
        raise SystemExit("没有可用的昇腾 NPU：torch.npu.is_available() 为 False。")

    n = 2**20
    kernel = tilelang.compile(vector_add(n), target="ascend", out_idx=-1)

    a = torch.randn(n, device="npu", dtype=torch.float32)
    b = torch.randn(n, device="npu", dtype=torch.float32)
    torch.npu.synchronize()

    c = kernel(a, b)
    torch.testing.assert_close(c, a + b)
    print("ascend vecadd: correctness OK")
    print("----- generated source (ascend) -----")
    print(kernel.get_kernel_source())


if __name__ == "__main__":
    main()
