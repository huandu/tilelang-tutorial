"""第 13 章 · 实战一：逐元素算子的四个版本（需要 NVIDIA GPU + TileLang v0.1.15）。

这四个 kernel 是同一件事的四级实现，用来对照「编译产物里的访存宽度」：

  add_flat          —— 一个 block 处理 threads 个元素，每线程 1 个元素
  add_thread_vec    —— 显式写出「每线程 num_per_thread 个元素」，相邻元素同线程
  add_register_vec  —— 先用 T.copy 把两块数据搬进 fragment，再算，再写回
  scale_cast        —— 乘标量 + dtype 转换（cast 类逐元素算子）
  per_token_cast_fp8—— 「cast」的进阶形态：分组的量化转换（cast 前先做一次归约）

对照的官方示例：
  _vendor/tilelang-v0.1.15/examples/elementwise/example_elementwise_add.py
  _vendor/tilelang-v0.1.15/examples/cast/example_per_token_cast_to_fp8.py

本文件只做**静态核对**：API 名逐个对照过 `_vendor/tilelang-v0.1.15/`，
`python3 -m py_compile` 通过。**没有在 GPU 上运行过**，因此不声明任何性能数字。
运行方式与预期输出见同目录 README.md。
"""

from __future__ import annotations

import tilelang
import tilelang.language as T


def _check_1d(n: int, block: int) -> None:
    if n <= 0 or block <= 0:
        raise ValueError(f"n and block must be positive, got n={n}, block={block}")


# ---------------------------------------------------------------------------
# ① 朴素：一整块交给一个 block，每线程一个元素
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[2])
def add_flat(N: int, threads: int = 256, dtype: str = "float32"):
    """y = x + b，每线程 1 个元素。

    这里**故意不写边界判断**：`T.Kernel` 的网格是 `T.ceildiv(N, threads)`，
    尾块越界时由 TileLang 的 safe-memory-legalize 过程插入谓词。代价是向量化
    被谓词挡住（见正文 §5.1）。
    """
    _check_1d(N, threads)

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),  # type: ignore[valid-type]
        B: T.Tensor((N,), dtype),  # type: ignore[valid-type]
        C: T.Tensor((N,), dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as bx:
            for i in T.Parallel(threads):
                idx = bx * threads + i
                C[idx] = A[idx] + B[idx]

    return main


# ---------------------------------------------------------------------------
# ② 每线程 NUM_PER_THREAD 个元素，并且明确「相邻元素属于同一线程」
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[2])
def add_thread_vec(N: int, threads: int = 256, num_per_thread: int = 8, dtype: str = "float32"):
    """y = x + b，每线程一串连续的 `num_per_thread` 个元素。

    索引写成 `(bx * threads + i) * num_per_thread + j`：最内层 `j` 连续。
    这样最内层循环可以直接被向量化成一次 128-bit 访存（fp32 是 4 个一组，
    fp16/bf16 是 8 个一组）。这是官方 elementwise 文档推荐的那种「显式告诉
    编译器每线程算几个」的写法。
    """
    block = threads * num_per_thread
    _check_1d(N, block)

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),  # type: ignore[valid-type]
        B: T.Tensor((N,), dtype),  # type: ignore[valid-type]
        C: T.Tensor((N,), dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(T.ceildiv(N, block), threads=threads) as bx:
            for i, j in T.Parallel(threads, num_per_thread):
                idx = (bx * threads + i) * num_per_thread + j
                C[idx] = A[idx] + B[idx]

    return main


# ---------------------------------------------------------------------------
# ③ 用 T.copy 显式分两段：global → fragment，算，fragment → global
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[2])
def add_register_vec(N: int, threads: int = 256, num_per_thread: int = 8, dtype: str = "float32"):
    """y = x + b，先把数据搬进寄存器片段再算。

    要求 `N % (threads * num_per_thread) == 0`：`T.copy` 的 region 是整块搬运，
    这里不做尾块处理，这样能把「纯对齐、可整向量化」这一档单独拿出来看。
    与 ② 的差别是**数据流方向**：② 是边读边算边写，③ 是先把整个 tile 读进
    fragment 再算。当一份输入要被用多次时，③ 省下的就是重复的 global 读。
    """
    block = threads * num_per_thread
    _check_1d(N, block)
    if N % block != 0:
        raise ValueError(f"this variant needs N % (threads * num_per_thread) == 0, got N={N}, block={block}")

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),  # type: ignore[valid-type]
        B: T.Tensor((N,), dtype),  # type: ignore[valid-type]
        C: T.Tensor((N,), dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(N // block, threads=threads) as bx:
            a_frag = T.alloc_fragment((block,), dtype)
            b_frag = T.alloc_fragment((block,), dtype)
            c_frag = T.alloc_fragment((block,), dtype)

            start = bx * block
            T.copy(A[start : start + block], a_frag)
            T.copy(B[start : start + block], b_frag)
            for i in T.Parallel(block):
                c_frag[i] = a_frag[i] + b_frag[i]
            T.copy(c_frag, C[start : start + block])

    return main


# ---------------------------------------------------------------------------
# ④ 乘标量 + dtype 转换：y = cast(x * alpha)
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[2])
def scale_cast(
    M: int,
    N: int,
    alpha: float = 0.5,
    in_dtype: str = "float32",
    out_dtype: str = "float16",
    block_M: int = 8,
    block_N: int = 256,
    threads: int = 128,
):
    """y = cast(x * alpha)，二维视角。

    cast 是逐元素的，但**跨 dtype 的 cast 会限制向量宽度**：一次向量化的 cast
    要求源与目标的 lane 数都能被同一个宽度整除。fp32→fp16 时源侧最多 4 lane
    （128-bit / 4 byte），所以这条路径的访存宽度上限比同 dtype 的 add 低。
    """
    if M % block_M != 0 or N % block_N != 0:
        raise ValueError(f"needs M % {block_M} == 0 and N % {block_N} == 0, got M={M}, N={N}")

    @T.prim_func
    def main(
        X: T.Tensor((M, N), in_dtype),  # type: ignore[valid-type]
        Y: T.Tensor((M, N), out_dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                Y[by * block_M + i, bx * block_N + j] = T.Cast(
                    out_dtype, X[by * block_M + i, bx * block_N + j] * alpha
                )

    return main


# ---------------------------------------------------------------------------
# ⑤ per-token / per-group FP8 量化：cast 之前先做一次归约
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[1, 2])
def per_token_cast_to_fp8(
    M: int,
    N: int,
    blk_m: int = 8,
    group_size: int = 128,
    threads: int = 128,
    dtype: str = "float32",
):
    """分组量化：y_fp8 = clamp(x / s), s = max(|x|, 1e-4) / 448。

    这是官方 `examples/cast/example_per_token_cast_to_fp8.py` 的结构（去掉了一些
    工程细节）：**逐元素算子一旦带上「按组求 amax」就变成「归约 + 逐元素」**，
    这正是第 4 节讲的归约模式，也是把它和后面的归一化放在同一章讲的原因。
    """
    if N % group_size != 0:
        raise ValueError(f"needs N % group_size == 0, got N={N}, group_size={group_size}")
    fp8_min, fp8_max = -448.0, 448.0

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),  # type: ignore[valid-type]
        X_fp8: T.Tensor((M, N), "float8_e4m3fn"),  # type: ignore[valid-type]
        X_scale: T.Tensor((M, T.ceildiv(N, group_size)), dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=threads) as (bx, by):
            x_local = T.alloc_fragment((blk_m, group_size), dtype)
            amax_local = T.alloc_fragment((blk_m,), dtype)
            scale_local = T.alloc_fragment((blk_m,), dtype)
            q_local = T.alloc_fragment((blk_m, group_size), dtype)
            q_fp8 = T.alloc_fragment((blk_m, group_size), "float8_e4m3fn")

            T.copy(
                X[bx * blk_m : (bx + 1) * blk_m, by * group_size : (by + 1) * group_size],
                x_local,
            )
            T.reduce_absmax(x_local, amax_local, dim=1)
            for i in T.Parallel(blk_m):
                amax_local[i] = T.max(amax_local[i], 1e-4)
                scale_local[i] = amax_local[i] / fp8_max
            for i, j in T.Parallel(blk_m, group_size):
                q_local[i, j] = T.clamp(x_local[i, j] / scale_local[i], fp8_min, fp8_max)
            T.copy(q_local, q_fp8)

            for i in T.Parallel(blk_m):
                X_scale[bx * blk_m + i, by] = scale_local[i]
            T.copy(
                q_fp8,
                X_fp8[bx * blk_m : (bx + 1) * blk_m, by * group_size : (by + 1) * group_size],
            )

    return main


if __name__ == "__main__":
    # 只有真正有 GPU + TileLang 时才走到这里；本机从未执行过。
    import torch

    n = 1 << 20
    a = torch.randn(n, dtype=torch.float32, device="cuda")
    b = torch.randn(n, dtype=torch.float32, device="cuda")
    for name, kern in (
        ("add_flat", add_flat(n, threads=256, dtype="float32")),
        ("add_thread_vec", add_thread_vec(n, threads=256, num_per_thread=8, dtype="float32")),
        ("add_register_vec", add_register_vec(n, threads=256, num_per_thread=8, dtype="float32")),
    ):
        out = kern(a, b)
        torch.testing.assert_close(out, a + b, rtol=1e-5, atol=1e-5)
        print(f"{name}: correctness passed")
