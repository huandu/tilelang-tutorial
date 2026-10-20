"""第 13 章 · 昇腾侧同款实现（主干 v0.1.15，`tilelang.ascend.language`）。

需要：昇腾 950 NPU + CANN + torch_npu + 从源码构建的 TileLang v0.1.15
（`USE_ASCEND=ON USE_CUDA=OFF`）。

这个方言和 CUDA 方言的三点不同，在本文件里都能看到：

1. `T.Kernel(N)` 只给**一维核网格**，没有 `threads=`：`with T.Kernel(n) as bx`
   拿到的是 AI Core 的下标（对照 `_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:139`）。
2. `T.alloc_shared` 在昇腾侧分配的就是 **UB（Unified Buffer）**
   （`.../tilelang/ascend/language/allocate.py:15-23`），没有单独的 `T.alloc_ub`。
3. 线程级并行要显式套一层 `T.SimtVF(threads=...)`；在它内部 `T.Parallel` 才会被
   向量化成 Vector 单元上的 SIMD 指令。不想用线程抽象时可以用 `T.SimdVF()` 直接
   写向量寄存器（见 `examples/ascend/example_vecadd.py` 的 `mode="simd"` 分支）。

对照的官方示例：
  _vendor/tilelang-v0.1.15/examples/ascend/example_vecadd.py
  _vendor/tilelang-v0.1.15/examples/ascend/example_rmsnorm.py
  _vendor/tilelang-v0.1.15/examples/ascend/example_per_token_cast_to_fp8.py

本文件只做**静态核对**：API 名逐个对照过 `_vendor/`，`python3 -m py_compile` 通过。
**没有在 NPU 上编译或运行过**（本机没有昇腾设备），不声明任何性能数字。
"""

from __future__ import annotations

import tilelang
import tilelang.ascend.language as T


# ---------------------------------------------------------------------------
# ① 逐元素：向量加（SIMT 路径）
# ---------------------------------------------------------------------------


def vector_add(N: int, mode: str = "simt", num_blocks: int = 64, tile: int = 8192, num_stages: int = 2):
    """`C = A + B`，结构简化自 `examples/ascend/example_vecadd.py`。

    与 CUDA 版逐元素 kernel 的三个关键差异：

    * **没有线程块内的 `T.Parallel(tile)`**：数据先 DMA 进 UB，再在 `T.SimtVF`
      里并行计算，最后 DMA 出去。搬运与计算是两条独立的流水线（MTE2 / Vector）。
    * **`T.annotate_buffer_versions` 声明 UB 的多缓冲份数**，配合 `T.Pipelined`
      让「搬下一块」和「算这一块」重叠。
    * **tile 大小由 UB 容量决定**：UB 是有限的一整块片上存储，x/y/z 三个 buffer
      乘以份数不能超过它。这和 CUDA 上「shared memory 用完可以靠 L1 兜底」不同。
    """
    if N <= 0 or N % (tile * num_blocks) != 0:
        raise ValueError(f"N must be a positive multiple of tile*num_blocks, got N={N}")
    if mode not in ("simt", "simd"):
        raise ValueError("mode must be simt or simd")
    iterations = N // (tile * num_blocks)

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),  # type: ignore[valid-type]
        B: T.Tensor((N,), "float32"),  # type: ignore[valid-type]
        C: T.Tensor((N,), "float32"),  # type: ignore[valid-type]
    ):
        with T.Kernel(num_blocks) as bx:
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({a: num_stages, b: num_stages, c: num_stages})

            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(A[begin : begin + tile], a)
                T.copy(B[begin : begin + tile], b)
                if mode == "simt":
                    with T.SimtVF(threads=2048):
                        for i in T.Parallel(tile):
                            c[i] = a[i] + b[i]
                else:
                    with T.SimdVF():
                        mask = T.simd.pset(32)
                        for i in range(tile // 64):
                            a_reg = T.simd.vld(a[i * 64])
                            b_reg = T.simd.vld(b[i * 64])
                            T.simd.vsts(c[i * 64], T.simd.vadd(a_reg, b_reg, mask), mask)
                T.copy(c, C[begin : begin + tile])

    return main


# ---------------------------------------------------------------------------
# ② 归一化：RMSNorm（T.SimtVF + reducer）
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[1, 3])
def rms_norm_ascend(
    batch: int,
    d: int,
    eps: float = 1e-6,
    dtype: str = "float32",
    n_cores: int = 64,
    rows: int = 4,
    threads: int = 256,
):
    """RMSNorm，结构简化自 `examples/ascend/example_rmsnorm.py`（rows 固定、去掉
    `bulk_rstd` 分流）。

    为什么要在 `T.SimtVF` 里用 `T.alloc_reducer` / `T.reducer_update` /
    `T.finalize_reducer`，而不是直接 `T.reduce_sum`：

    * SimtVF 的线程域是可以被编译器**物理复制**的（同一个逻辑输出可能由多个线程
      各自贡献一部分），one-shot 的 reduce 语义在这里没有定义；
    * reducer 是**有 epoch 的**：`reducer_init` 开一轮、`reducer_update` 每次逻辑
      迭代恰好贡献一次、`finalize_reducer` 收口。物理复制多少次都不影响逻辑结果。

    语义与接口见 `_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:358-460`。
    """
    if batch % (n_cores * rows) != 0:
        raise ValueError(f"needs batch % (n_cores*rows) == 0, got batch={batch}")
    tile = tilelang.next_power_of_2(d)
    iterations = batch // (n_cores * rows)

    @T.prim_func
    def main(
        X: T.Tensor((batch * d,), dtype),  # type: ignore[valid-type]
        Y: T.Tensor((batch * d,), dtype),  # type: ignore[valid-type]
        W: T.Tensor((d,), dtype),  # type: ignore[valid-type]
        RSTD: T.Tensor((batch,), "float32"),  # type: ignore[valid-type]
        eps_v: T.float32,
    ):
        with T.Kernel(n_cores) as core_id:
            w_ub = T.alloc_shared((d,), dtype)
            x_ub = T.alloc_shared((rows * d,), "float32")
            y_ub = T.alloc_shared((rows * d,), "float32")
            rstd_ub = T.alloc_shared((rows,), "float32")
            T.annotate_buffer_versions({x_ub: 2, y_ub: 1, rstd_ub: 2})

            T.copy(W[:d], w_ub[:d])
            for t in T.Pipelined(iterations, num_stages=2):
                row_id = (t * n_cores + core_id) * rows
                base = row_id * d
                T.copy(X[base : base + rows * d], x_ub)

                with T.SimtVF(threads=threads):
                    x_frag = T.alloc_fragment((rows, tile), "float32")
                    # d 不一定是 2 的幂：多出来的列读 0，对平方和无影响。
                    for r, i in T.Parallel(rows, tile, coalesced_width=T.int32(1)):
                        x_frag[r, i] = T.if_then_else(i < d, x_ub[r * d + i], T.float32(0))

                    sum_sq = T.alloc_reducer((rows,), "float32", op="sum")
                    T.reducer_init(sum_sq)
                    for r, i in T.Parallel(rows, tile, coalesced_width=T.int32(1)):
                        if i < d:
                            T.reducer_update(sum_sq[r], x_frag[r, i] * x_frag[r, i])
                    sum_result = T.alloc_fragment((rows,), "float32")
                    T.finalize_reducer(sum_sq, sum_result)

                    for r in T.Parallel(rows):
                        sum_result[r] = T.rsqrt(sum_result[r] / d + eps_v)
                        rstd_ub[r] = sum_result[r]

                    w_frag = T.alloc_fragment((tile,), "float32")
                    for i in T.Parallel(tile, coalesced_width=T.int32(1)):
                        w_frag[i] = T.if_then_else(i < d, T.Cast("float32", w_ub[i]), T.float32(0))
                    for r, i in T.Parallel(rows, tile, coalesced_width=T.int32(1)):
                        if i < d:
                            y_ub[r * d + i] = x_frag[r, i] * sum_result[r] * w_frag[i]

                T.copy(y_ub, Y[base : base + rows * d])
                T.copy(rstd_ub, RSTD[row_id : row_id + rows])

    return main


if __name__ == "__main__":
    # 需要昇腾 NPU + torch_npu。本机从未执行过。
    import torch
    import torch_npu  # noqa: F401

    n = 1 << 20
    a = torch.randn(n, device="npu")
    b = torch.randn(n, device="npu")
    kernel = tilelang.compile(vector_add(n, mode="simt"), target="ascend", out_idx=-1)
    torch.testing.assert_close(kernel(a, b), a + b)
    print("vector_add (ascend, simt): correctness passed")
