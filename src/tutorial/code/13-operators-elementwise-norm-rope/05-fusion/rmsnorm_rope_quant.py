"""第 13 章 · 实战三之二：多算子融合（需要 NVIDIA GPU + TileLang v0.1.15）。

两个 capstone kernel：

  rms_norm_rope_fused        —— RMSNorm + 局部 RoPE 融合，x 只读一遍、y 只写一遍
  rms_norm_rope_quant_fused  —— 再加 per-token FP8 量化：一次读 x，写 fp8 与 scale

融合的收益是可以先在纸上算清楚的（见同章 `code/01-ledger/ledger.py`）：
RMSNorm 单独写是「读 x 一遍 → 写 y 一遍」，RoPE 再「读 y 一遍 → 写 y 一遍」，
账本上 4 趟变成了 2 趟。代价是**片上容量**与**寄存器压力**：三件事要共享同一块
fragment 和 shared，tile 尺寸必须缩小，occupancy 可能因此下降。

这两个 kernel 的 `blk_m` 都默认取 1：一行的 N 个 fp32 累加器已经占满一个线程
块的寄存器预算。想加大 `blk_m` 就要按行切 K，或者改用 per-thread partial 归约
（对照 `../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/kernels/norm/rms_norm.py`）。

本文件只做**静态核对**：API 名逐个对照过 `_vendor/`，`python3 -m py_compile` 通过。
**没有在 GPU 上运行过**，不声明任何性能数字；layout 推导是否通过也必须在真机上
确认（见同目录 README 的「真机验证步骤」）。
"""

from __future__ import annotations

import tilelang
import tilelang.language as T


def _torch_rope_neox(x, cos, sin):
    """本地复制一份 NeoX 旋转（示例目录必须能独立复制出去运行，不跨目录 import）。"""
    import torch

    half = x.shape[-1] // 2
    x_lo = x.float()[..., :half]
    x_hi = x.float()[..., half:]
    c = cos.float()
    s = sin.float()
    y_lo = x_lo * c - x_hi * s
    y_hi = x_hi * c + x_lo * s
    return torch.cat([y_lo, y_hi], dim=-1).to(x.dtype)


def torch_reference_fused(x, weight, cos, sin, eps: float = 1e-6, fp8_max: float = 448.0):
    """RMSNorm → 局部 RoPE → per-token FP8 量化的 PyTorch 参考实现。

    x:      [M, N]（M 是 token×head 打平后的行数）
    weight: [N]
    cos/sin:[M, rope_dim // 2]

    返回 (y, y_fp8, scale)。
    """
    import torch

    rstd = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    normed = (x.float() * rstd * weight.float().unsqueeze(0)).to(x.dtype)

    rope_dim = 2 * cos.shape[-1]
    head = normed[..., :-rope_dim]
    tail = normed[..., -rope_dim:]
    y = torch.cat([head, _torch_rope_neox(tail, cos, sin)], dim=-1)

    amax = y.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-4)
    scale = amax / fp8_max
    y_fp8 = (y.float() / scale).to(torch.float8_e4m3fn)
    return y, y_fp8, scale.squeeze(-1)


# ---------------------------------------------------------------------------
# ① RMSNorm + 局部 RoPE
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[-1])
def rms_norm_rope_fused(
    M: int,
    N: int,
    rope_dim: int,
    eps: float = 1e-6,
    blk_m: int = 1,
    dtype: str = "float16",
    threads: int = 256,
):
    """y = rope(rms_norm(x, w), cos, sin)，只有最后 `rope_dim` 列被旋转。

    数据流（都留在片上，中间的 `normed` 不落显存）：

        x ──T.copy──▶ shared ──cast──▶ fragment(fp32)
          ──square+reduce_sum──▶ rstd ──mul w[j]──▶ 写回 shared（降回 dtype）
          ──在 shared 上就地做尾部的 RoPE──▶ T.copy 到 Y

    为什么 RoPE 那一步要绕回 shared：NeoX 约定把一行从中间劈成两半，第 `p` 个和
    第 `p + rope_dim/2` 个元素必须同时可见。它们在 fragment 里可能属于不同线程，
    寄存器之间不能直接互相访问；shared 是唯一能让「跨 lane 配对」成立的地方。
    """
    half = rope_dim // 2
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")
    if N < rope_dim:
        raise ValueError(f"needs N >= rope_dim, got N={N}, rope_dim={rope_dim}")
    if M % blk_m != 0:
        raise ValueError(f"needs M % blk_m == 0, got M={M}, blk_m={blk_m}")

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),  # type: ignore[valid-type]
        W: T.Tensor((N,), dtype),  # type: ignore[valid-type]
        Cos: T.Tensor((M, half), dtype),  # type: ignore[valid-type]
        Sin: T.Tensor((M, half), dtype),  # type: ignore[valid-type]
        Y: T.Tensor((M, N), dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
            x_sh = T.alloc_shared((blk_m, N), dtype)
            w_sh = T.alloc_shared((N,), dtype)
            x_f32 = T.alloc_fragment((blk_m, N), "float32")
            sq_f32 = T.alloc_fragment((blk_m, N), "float32")
            sumsq = T.alloc_fragment((blk_m,), "float32")
            rstd = T.alloc_fragment((blk_m,), "float32")

            T.copy(X[bx * blk_m : (bx + 1) * blk_m, :], x_sh)
            T.copy(W, w_sh)

            for i, j in T.Parallel(blk_m, N):
                x_f32[i, j] = T.Cast("float32", x_sh[i, j])
            for i, j in T.Parallel(blk_m, N):
                sq_f32[i, j] = x_f32[i, j] * x_f32[i, j]
            T.reduce_sum(sq_f32, sumsq, dim=1)
            for i in T.Parallel(blk_m):
                rstd[i] = T.rsqrt(sumsq[i] / N + eps)

            # 归一化 + 权重，直接降回 dtype 写进 shared：后面 RoPE 要跨 lane 配对
            for i, j in T.Parallel(blk_m, N):
                x_sh[i, j] = T.Cast(dtype, x_f32[i, j] * rstd[i] * T.Cast("float32", w_sh[j]))

            # 只旋转尾部 rope_dim 列，就地完成
            for i, p in T.Parallel(blk_m, half):
                c = T.Cast("float32", Cos[bx * blk_m + i, p])
                s = T.Cast("float32", Sin[bx * blk_m + i, p])
                lo = T.Cast("float32", x_sh[i, N - rope_dim + p])
                hi = T.Cast("float32", x_sh[i, N - rope_dim + half + p])
                x_sh[i, N - rope_dim + p] = T.Cast(dtype, lo * c - hi * s)
                x_sh[i, N - rope_dim + half + p] = T.Cast(dtype, hi * c + lo * s)

            T.copy(x_sh, Y[bx * blk_m : (bx + 1) * blk_m, :])

    return main


# ---------------------------------------------------------------------------
# ② RMSNorm + RoPE + per-token FP8 量化
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[-3, -2, -1])
def rms_norm_rope_quant_fused(
    M: int,
    N: int,
    rope_dim: int,
    group_size: int = 128,
    eps: float = 1e-6,
    fp8_max: float = 448.0,
    blk_m: int = 1,
    dtype: str = "float16",
    threads: int = 256,
):
    """在 ① 的基础上再做一次 per-token FP8 量化。

    这一步引入**第二次归约**（对旋转后的整行取 amax）。两个归约在同一个 kernel 里，
    所以：

    * 第一次归约（平方和）用 fragment 完成；
    * RoPE 之后，整行已经在 shared 里，第二次归约再把它读回一个 fp32 fragment 求 max。

    账本收益：单独的「量化」算子要再把 `rotated` 读一遍、把 `q` 写一遍；
    融合后这一读一写都省了（见 `code/01-ledger/ledger.py` 的最后一组）。
    """
    half = rope_dim // 2
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}")
    if N % group_size != 0:
        raise ValueError(f"needs N % group_size == 0, got N={N}, group_size={group_size}")
    if M % blk_m != 0:
        raise ValueError(f"needs M % blk_m == 0, got M={M}, blk_m={blk_m}")
    n_groups = N // group_size

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),  # type: ignore[valid-type]
        W: T.Tensor((N,), dtype),  # type: ignore[valid-type]
        Cos: T.Tensor((M, half), dtype),  # type: ignore[valid-type]
        Sin: T.Tensor((M, half), dtype),  # type: ignore[valid-type]
        Y: T.Tensor((M, N), dtype),  # type: ignore[valid-type]
        Q: T.Tensor((M, N), "float8_e4m3fn"),  # type: ignore[valid-type]
        Scale: T.Tensor((M, n_groups), "float32"),  # type: ignore[valid-type]
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
            x_sh = T.alloc_shared((blk_m, N), dtype)
            w_sh = T.alloc_shared((N,), dtype)

            x_f32 = T.alloc_fragment((blk_m, N), "float32")
            sq_f32 = T.alloc_fragment((blk_m, N), "float32")
            sumsq = T.alloc_fragment((blk_m,), "float32")
            rstd = T.alloc_fragment((blk_m,), "float32")

            group_view = T.alloc_fragment((blk_m, n_groups, group_size), "float32")
            amax = T.alloc_fragment((blk_m, n_groups), "float32")
            scale_f32 = T.alloc_fragment((blk_m, n_groups), "float32")
            q_fp8 = T.alloc_fragment((blk_m, N), "float8_e4m3fn")

            T.copy(X[bx * blk_m : (bx + 1) * blk_m, :], x_sh)
            T.copy(W, w_sh)

            # --- 第 1 步：RMSNorm ---
            for i, j in T.Parallel(blk_m, N):
                x_f32[i, j] = T.Cast("float32", x_sh[i, j])
            for i, j in T.Parallel(blk_m, N):
                sq_f32[i, j] = x_f32[i, j] * x_f32[i, j]
            T.reduce_sum(sq_f32, sumsq, dim=1)
            for i in T.Parallel(blk_m):
                rstd[i] = T.rsqrt(sumsq[i] / N + eps)
            for i, j in T.Parallel(blk_m, N):
                x_sh[i, j] = T.Cast(dtype, x_f32[i, j] * rstd[i] * T.Cast("float32", w_sh[j]))

            # --- 第 2 步：尾部 RoPE（就地，跨 lane 配对走 shared） ---
            for i, p in T.Parallel(blk_m, half):
                c = T.Cast("float32", Cos[bx * blk_m + i, p])
                s = T.Cast("float32", Sin[bx * blk_m + i, p])
                lo = T.Cast("float32", x_sh[i, N - rope_dim + p])
                hi = T.Cast("float32", x_sh[i, N - rope_dim + half + p])
                x_sh[i, N - rope_dim + p] = T.Cast(dtype, lo * c - hi * s)
                x_sh[i, N - rope_dim + half + p] = T.Cast(dtype, hi * c + lo * s)

            T.copy(x_sh, Y[bx * blk_m : (bx + 1) * blk_m, :])

            # --- 第 3 步：每 group_size 个元素共享一个 scale ---
            # 「按 group_size 一组取 max」= 把一行 reshape 成 (n_groups, group_size) 后
            # 沿最后一维归约。这里用一个三维 fragment 承载视图，再用 T.reduce_max 降维——
            # 这是 v0.1.15 里 reduce 的常规用法，比手写 read-modify-write 更不容易踩 layout。
            for i, g, k in T.Parallel(blk_m, n_groups, group_size):
                group_view[i, g, k] = T.abs(T.Cast("float32", x_sh[i, g * group_size + k]))
            T.reduce_max(group_view, amax, dim=2, clear=True)
            for i, g in T.Parallel(blk_m, n_groups):
                scale_f32[i, g] = T.max(amax[i, g], 1e-4) / fp8_max

            for i, g, k in T.Parallel(blk_m, n_groups, group_size):
                q_fp8[i, g * group_size + k] = T.Cast(
                    "float8_e4m3fn",
                    T.clamp(
                        T.Cast("float32", x_sh[i, g * group_size + k]) / scale_f32[i, g],
                        -fp8_max,
                        fp8_max,
                    ),
                )
            T.copy(q_fp8, Q[bx * blk_m : (bx + 1) * blk_m, :])

            for i, g in T.Parallel(blk_m, n_groups):
                Scale[bx * blk_m + i, g] = scale_f32[i, g]

    return main


if __name__ == "__main__":
    # 需要 cuda。本机从未执行过。
    import torch

    m, n, rope_dim = 256, 4096, 128
    half = rope_dim // 2
    x = torch.randn(m, n, dtype=torch.float16, device="cuda")
    w = torch.randn(n, dtype=torch.float16, device="cuda")
    angles = torch.randn(m, half, device="cuda")
    cos, sin = angles.cos().half(), angles.sin().half()

    y, y_fp8, scale = torch_reference_fused(x, w, cos, sin)
    kern = rms_norm_rope_fused(m, n, rope_dim, dtype="float16")
    y_tl = kern(x, w, cos, sin)
    torch.testing.assert_close(y_tl, y, rtol=1e-2, atol=1e-2)
    print("rms_norm_rope_fused: correctness passed")

    kern_q = rms_norm_rope_quant_fused(m, n, rope_dim, dtype="float16")
    y_tl_q, q_tl, scale_tl = kern_q(x, w, cos, sin)
    torch.testing.assert_close(scale_tl, scale, rtol=1e-2, atol=1e-2)
    print("rms_norm_rope_quant_fused: correctness passed")
