"""第 13 章 · 实战二：LayerNorm / RMSNorm 的三级实现（需要 NVIDIA GPU + TileLang v0.1.15）。

四个 kernel：

  rms_norm_naive_row_per_block —— 一个 block 一行，先读一遍求平方和，再读一遍归一化
  rms_norm                     —— 一行留在 fragment 里，读一遍、写一遍（对照官方示例）
  rms_norm_splitk              —— 行太长放不进 fragment 时分 K 段：两趟读，第二趟倒序走
  layer_norm_fwd               —— 两遍归约（sum 与 sum of squares）的标准 LayerNorm

对照的官方示例：
  _vendor/tilelang-v0.1.15/examples/norm/rms_norm.py
  _vendor/tilelang-v0.1.15/examples/norm/layernorm.py

本文件只做**静态核对**：API 名逐个对照过 `_vendor/`，`python3 -m py_compile` 通过。
**没有在 GPU 上运行过**，不声明任何性能数字。运行方式见同目录 README.md。
"""

from __future__ import annotations

import tilelang
import tilelang.language as T


def torch_reference_rms_norm(x, weight=None, eps: float = 1e-6):
    """PyTorch 参考实现（小数据量下对照用）。

        rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        y = x * rstd * (weight if weight is not None else 1)
    """
    import torch

    rstd = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    y = x.float() * rstd
    if weight is not None:
        y = y * weight.float().unsqueeze(0)
    return y.to(x.dtype)


def torch_reference_layer_norm(x, gamma, beta, eps: float = 1e-5):
    """PyTorch 参考实现：直接调用 F.layer_norm 最省事。"""
    import torch
    import torch.nn.functional as F

    return F.layer_norm(x, (x.shape[-1],), gamma, beta, eps=eps)


# ---------------------------------------------------------------------------
# ① 朴素：一个 block 一行，读两遍 x
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[1])
def rms_norm_naive_row_per_block(N: int, D: int, eps: float = 1e-6, dtype: str = "float32", threads: int = 128):
    """y = x * rsqrt(mean(x^2) + eps)。

    每一行分给一个 block：先把整行读进 shared，求平方和，再把整行读出来归一化。
    x 在账本上被读了**两遍**——这正是 §3.2 里「先能用」的版本，也是优化要打掉的
    第一笔开销。
    """
    if D % threads != 0:
        raise ValueError(f"needs D % threads == 0, got D={D}, threads={threads}")

    @T.prim_func
    def main(X: T.Tensor((N, D), dtype), Y: T.Tensor((N, D), dtype)):  # type: ignore[valid-type]
        with T.Kernel(N, threads=threads) as bx:
            x_sh = T.alloc_shared((D,), dtype)
            sq = T.alloc_fragment((D,), "float32")
            acc = T.alloc_fragment((1,), "float32")

            T.copy(X[bx, :], x_sh)
            for j in T.Parallel(D):
                sq[j] = T.Cast("float32", x_sh[j]) * T.Cast("float32", x_sh[j])
            T.reduce_sum(sq, acc, dim=0)
            for j in T.Parallel(D):
                Y[bx, j] = T.Cast(dtype, T.Cast("float32", x_sh[j]) * T.rsqrt(acc[0] / D + eps))

    return main


# ---------------------------------------------------------------------------
# ② 一行留在寄存器里：读一遍、写一遍
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[1], pass_configs={"tl.disable_tma_lower": True})
def rms_norm(M: int, N: int, blk_m: int = 1, eps: float = 1e-6, dtype: str = "float", threads: int = 128):
    """对照 `examples/norm/rms_norm.py` 的结构：整行放进 fragment，自己乘回自己。

    数据流：`global → fragment`（一遍）→ 平方 → 归约 → 乘 rstd → `fragment → global`。
    中间结果始终在寄存器里，没有第二遍 global 读。

    注意 `pass_configs`：官方示例用 `tl.disable_tma_lower` 关掉 TMA，只让这一对
    普通的 load/store 走 SIMT 路径；v0.1.15 里这个键已经标记为 deprecated
    （`tilelang/transform/pass_config.py:141,352-355`），推荐改成逐次
    `T.copy(..., disable_tma=True)`。
    """
    if M % blk_m != 0:
        raise ValueError(f"needs M % blk_m == 0, got M={M}, blk_m={blk_m}")

    @T.prim_func
    def main(X: T.Tensor((M, N), dtype), Y: T.Tensor((M, N), dtype)):  # type: ignore[valid-type]
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
            x_local = T.alloc_fragment((blk_m, N), dtype)
            sq_local = T.alloc_fragment((blk_m, N), dtype)
            sumsq = T.alloc_fragment((blk_m,), dtype)

            T.copy(X[bx * blk_m : (bx + 1) * blk_m, :], x_local)
            for i, j in T.Parallel(blk_m, N):
                sq_local[i, j] = x_local[i, j] * x_local[i, j]
            T.reduce_sum(sq_local, sumsq, dim=1)
            for i in T.Parallel(blk_m):
                sumsq[i] = T.rsqrt(sumsq[i] / N + eps)
            for i, j in T.Parallel(blk_m, N):
                x_local[i, j] = x_local[i, j] * sumsq[i]
            T.copy(x_local, Y[bx * blk_m : (bx + 1) * blk_m, :])

    return main


# ---------------------------------------------------------------------------
# ③ 行太长时的分 K 两趟写法
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[1], pass_configs={"tl.disable_tma_lower": True})
def rms_norm_splitk(
    M: int,
    N: int,
    blk_m: int = 1,
    blk_k: int = 512,
    eps: float = 1e-6,
    dtype: str = "float",
    threads: int = 128,
):
    """N 太大、整行放不进 fragment 时：把行切成 `num_k_step` 段。

    结构与 `examples/norm/rms_norm.py` 的 `rms_norm_splitk` 一致：

    * 第一趟只累加平方和（`A_local` 只留当前段的累加结果）；
    * 第二趟倒序从最后一段往前处理，刚读过的数据还在 cache 里。

    两趟仍然读了两遍 x，但每趟的 tile 小得多，占用片上存储少；代价是 x 被读两遍。
    什么时候值得：见正文 §3.2「先能用，再用好」里的取舍讨论。
    """
    if N % blk_k != 0:
        raise ValueError(f"needs N % blk_k == 0, got N={N}, blk_k={blk_k}")

    @T.prim_func
    def main(X: T.Tensor((M, N), dtype), Y: T.Tensor((M, N), dtype)):  # type: ignore[valid-type]
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
            x_sh = T.alloc_shared((blk_m, blk_k), dtype)
            x_local = T.alloc_fragment((blk_m, blk_k), dtype)
            sumsq = T.alloc_fragment((blk_m,), dtype)
            num_k_step = T.ceildiv(N, blk_k)

            T.clear(x_local)
            for k in T.Serial(num_k_step):
                T.copy(X[bx * blk_m : (bx + 1) * blk_m, k * blk_k : (k + 1) * blk_k], x_sh)
                for i, j in T.Parallel(blk_m, blk_k):
                    x_local[i, j] += x_sh[i, j] * x_sh[i, j]
            T.reduce_sum(x_local, sumsq, dim=1)
            for i in T.Parallel(blk_m):
                sumsq[i] = T.rsqrt(sumsq[i] / N + eps)

            for k in T.Serial(num_k_step):
                # 倒序：上一趟刚读过的尾段还在 cache 里
                T.copy(
                    X[bx * blk_m : (bx + 1) * blk_m, (num_k_step - 1 - k) * blk_k : (num_k_step - k) * blk_k],
                    x_sh,
                )
                for i, j in T.Parallel(blk_m, blk_k):
                    x_sh[i, j] *= sumsq[i]
                T.copy(
                    x_sh,
                    Y[bx * blk_m : (bx + 1) * blk_m, (num_k_step - 1 - k) * blk_k : (num_k_step - k) * blk_k],
                )

    return main


# ---------------------------------------------------------------------------
# ④ LayerNorm：两遍归约 + 仿射
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[-3, -2, -1])
def layer_norm_fwd(
    N: int,
    D: int,
    eps: float = 1e-5,
    blk_m: int = 1,
    threads: int = 256,
    in_dtype: str = "bfloat16",
    out_dtype: str = "bfloat16",
):
    """LayerNorm 前向，结构对照 `examples/norm/layernorm.py`。

    三个输出：`Y`、`Mean`、`Rstd`（反传要用）。用 float32 累加，读完一遍就
    把 mean / rstd 都算出来，全程只读一遍 x。
    """
    if N % blk_m != 0:
        raise ValueError(f"needs N % blk_m == 0, got N={N}, blk_m={blk_m}")
    accum_dtype = "float"

    @T.prim_func
    def main(
        X: T.Tensor((N, D), in_dtype),  # type: ignore[valid-type]
        gamma: T.Tensor((D,), in_dtype),  # type: ignore[valid-type]
        beta: T.Tensor((D,), in_dtype),  # type: ignore[valid-type]
        Y: T.Tensor((N, D), out_dtype),  # type: ignore[valid-type]
        Mean: T.Tensor((N,), accum_dtype),  # type: ignore[valid-type]
        Rstd: T.Tensor((N,), accum_dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(T.ceildiv(N, blk_m), threads=threads) as bx:
            x_sh = T.alloc_shared((blk_m, D), in_dtype)
            g_sh = T.alloc_shared((D,), in_dtype)
            b_sh = T.alloc_shared((D,), in_dtype)
            x_local = T.alloc_fragment((blk_m, D), accum_dtype)
            xsq_local = T.alloc_fragment((blk_m, D), accum_dtype)
            sum_row = T.alloc_fragment((blk_m,), accum_dtype)
            sumsq_row = T.alloc_fragment((blk_m,), accum_dtype)
            mean_row = T.alloc_fragment((blk_m,), accum_dtype)
            rstd_row = T.alloc_fragment((blk_m,), accum_dtype)

            T.copy(X[bx * blk_m : (bx + 1) * blk_m, :], x_sh)
            T.copy(gamma, g_sh)
            T.copy(beta, b_sh)
            for i, j in T.Parallel(blk_m, D):
                x_local[i, j] = T.Cast(accum_dtype, x_sh[i, j])
            for i, j in T.Parallel(blk_m, D):
                xsq_local[i, j] = x_local[i, j] * x_local[i, j]
            T.reduce_sum(x_local, sum_row, dim=1)
            T.reduce_sum(xsq_local, sumsq_row, dim=1)

            inv_d = T.float32(1.0) / T.Cast(accum_dtype, D)
            for i in T.Parallel(blk_m):
                mean_row[i] = sum_row[i] * inv_d
                rstd_row[i] = T.rsqrt(sumsq_row[i] * inv_d - mean_row[i] * mean_row[i] + T.Cast(accum_dtype, eps))
                Mean[bx * blk_m + i] = mean_row[i]
                Rstd[bx * blk_m + i] = rstd_row[i]

            for i, j in T.Parallel(blk_m, D):
                x_sh[i, j] = T.Cast(
                    out_dtype,
                    (x_local[i, j] - mean_row[i]) * rstd_row[i] * T.Cast(accum_dtype, g_sh[j])
                    + T.Cast(accum_dtype, b_sh[j]),
                )
            T.copy(x_sh, Y[bx * blk_m : (bx + 1) * blk_m, :])

    return main


# ---------------------------------------------------------------------------
# 可选：把它接进 torch.autograd（只给出接线方式，不是完整实现）
# ---------------------------------------------------------------------------


def make_layer_norm_fn():
    """展示如何把一个 TileLang kernel 包成 `torch.autograd.Function`。

    这里故意**不**给出反传 kernel：本章的四个 kernel 都是前向。真正的反传实现见
    `_vendor/tilelang-v0.1.15/examples/norm/layernorm.py` 的 `_layernorm_bwd`
    （它用 `T.atomic_add` 跨 block 累加 dgamma / dbeta）。
    """
    import torch

    class LayerNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, gamma, beta, eps):  # type: ignore[override]
            n, d = x.shape
            kernel = layer_norm_fwd(n, d, eps=eps, in_dtype="bfloat16", out_dtype="bfloat16")
            y, mean, rstd = kernel(x, gamma, beta)
            ctx.save_for_backward(x, gamma, mean, rstd)
            ctx.eps = eps
            return y

        @staticmethod
        def backward(ctx, dy):  # type: ignore[override]
            raise NotImplementedError("see examples/norm/layernorm.py for the backward kernel")

    return LayerNormFn


if __name__ == "__main__":
    # 需要 cuda。本机从未执行过。
    import torch

    n, d = 1024, 1024
    x = torch.randn(n, d, dtype=torch.float32, device="cuda")
    out = rms_norm(n, d, blk_m=1, dtype="float32")(x)
    torch.testing.assert_close(out, torch_reference_rms_norm(x, eps=1e-6), rtol=1e-2, atol=1e-2)
    print("rms_norm: correctness passed")
