"""第 13 章 · 实战三之一：RoPE 的两种旋转约定 + 一个面向向量化的版本
（需要 NVIDIA GPU + TileLang v0.1.15）。

RoPE（Rotary Position Embedding）在 kernel 层只有两种旋转约定：

  interleaved（原版 RoFormer / `non-neox`）—— 相邻两个元素是一对：
      y[2p]   = x[2p] * c[p] - x[2p+1] * s[p]
      y[2p+1] = x[2p] * s[p] + x[2p+1] * c[p]

  half-split（NeoX / Llama / Qwen 等）—— 前后两半对应：
      y[p]        = x[p]        * c[p] - x[p + R/2] * s[p]
      y[p + R/2]  = x[p + R/2]  * c[p] + x[p]      * s[p]

两者不是「实现差异」，是**权重训练时就定下来的约定**：写错了不会报错，只会得到
一个数值上完全错误但不会崩溃的模型。所以这里的第一个练习是先把 PyTorch 参考
实现对齐。

cos/sin 表按「一个 pair 一行」存：形状 (seq_len, R/2)。position 的选取、频率基数
（`base`）、以及 YaRN / Llama-3.1 那类分段缩放都属于 **Op 层**的职责，不属于
kernel（对照 `../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/ops/rope.py`）。

本文件只做**静态核对**：API 名逐个对照过 `_vendor/`，`python3 -m py_compile` 通过。
**没有在 GPU 上运行过**，不声明任何性能数字。
"""

from __future__ import annotations

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# PyTorch 参考实现（CPU 上就能跑，用来在真机验证前对齐语义）
# ---------------------------------------------------------------------------


def torch_rope_interleaved(x, cos, sin):
    """x: [..., R]，cos/sin: [..., R//2]。相邻成对旋转。"""
    import torch

    head = x.float().reshape(*x.shape[:-1], -1, 2)
    x0 = head[..., 0]
    x1 = head[..., 1]
    c = cos.float()
    s = sin.float()
    y0 = x0 * c - x1 * s
    y1 = x0 * s + x1 * c
    return torch.stack([y0, y1], dim=-1).flatten(-2).to(x.dtype)


def torch_rope_neox(x, cos, sin):
    """x: [..., R]，cos/sin: [..., R//2]。前后半对应。"""
    import torch

    half = x.shape[-1] // 2
    x_lo = x.float()[..., :half]
    x_hi = x.float()[..., half:]
    c = cos.float()
    s = sin.float()
    y_lo = x_lo * c - x_hi * s
    y_hi = x_hi * c + x_lo * s
    return torch.cat([y_lo, y_hi], dim=-1).to(x.dtype)


def torch_partial_rope(x, cos, sin, mode: str = "neox"):
    """只旋转最后 `2 * cos.shape[-1]` 个通道，前面的通道原样拷过去。

    这是真实模型里的用法（MLA 的 decoupled RoPE 就只旋转 `qk_rope_head_dim`），
    也是昇腾 `examples/pos_embedding/rope.py` 的 `dim_start = hidden_size - rope_dim`
    那几行的意思。
    """
    import torch

    rope_dim = 2 * cos.shape[-1]
    head = x[..., :-rope_dim]
    tail = x[..., -rope_dim:]
    fn = torch_rope_neox if mode == "neox" else torch_rope_interleaved
    return torch.cat([head, fn(tail, cos, sin)], dim=-1)


# ---------------------------------------------------------------------------
# ① 朴素：一行一个 block，两半分别乘、分别写
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[3])
def rope_neox_naive(
    S: int,
    R: int,
    dtype: str = "float16",
    threads: int = 256,
):
    """NeoX 约定的最小实现：单行、直接读写 global。

    `S` 是行数（token × head 打平后的行数），`R` 是 rope 宽度。
    每行读 R 个元素、写 R 个元素，cos/sin 各读 R/2 个。
    """
    if R % 2 != 0:
        raise ValueError(f"rope dim must be even, got R={R}")
    half = R // 2

    @T.prim_func
    def main(
        X: T.Tensor((S, R), dtype),  # type: ignore[valid-type]
        Cos: T.Tensor((S, half), dtype),  # type: ignore[valid-type]
        Sin: T.Tensor((S, half), dtype),  # type: ignore[valid-type]
        Y: T.Tensor((S, R), dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(S, threads=threads) as bx:
            for p in T.Parallel(half):
                c = T.Cast("float32", Cos[bx, p])
                s = T.Cast("float32", Sin[bx, p])
                lo = T.Cast("float32", X[bx, p])
                hi = T.Cast("float32", X[bx, p + half])
                Y[bx, p] = T.Cast(dtype, lo * c - hi * s)
                Y[bx, p + half] = T.Cast(dtype, hi * c + lo * s)

    return main


# ---------------------------------------------------------------------------
# ② 优化：一个 block 吃多行，x 与 cos/sin 先搬进 shared，再算
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[3])
def rope_neox_blocked(
    S: int,
    R: int,
    blk_s: int = 16,
    dtype: str = "float16",
    threads: int = 256,
):
    """把 x 的一个 (blk_s, R) tile 搬进 shared，cos/sin 也搬进来，然后一次算完。

    与 ① 的三个差别：

    1. **每行两个 kernel 变一个 block**：`T.Kernel(S // blk_s)` 而不是 `T.Kernel(S)`，
       块数少了 `blk_s` 倍，调度与尾块开销随之下降。
    2. **cos/sin 在片上复用**：同一行的 R/2 个系数被「读进 shared 一次」，来自不同
       lane 的访问不再各自去 global 取。
    3. **计算统一升到 fp32**，只在写回时 cast 一次；fp16 下 `x*c - y*s` 若全程 fp16
       会多丢一次精度。

    注意 `T.Parallel(blk_s, half)` 里同时访问 `j` 与 `j + half`：两个索引都对
    `j` 是步长 1 的仿射表达，所以两条 store 各自可以被向量化，但它们不是同一次
    向量访存。
    """
    if R % 2 != 0:
        raise ValueError(f"rope dim must be even, got R={R}")
    if S % blk_s != 0:
        raise ValueError(f"needs S % blk_s == 0, got S={S}, blk_s={blk_s}")
    half = R // 2

    @T.prim_func
    def main(
        X: T.Tensor((S, R), dtype),  # type: ignore[valid-type]
        Cos: T.Tensor((S, half), dtype),  # type: ignore[valid-type]
        Sin: T.Tensor((S, half), dtype),  # type: ignore[valid-type]
        Y: T.Tensor((S, R), dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(S // blk_s, threads=threads) as bx:
            x_sh = T.alloc_shared((blk_s, R), dtype)
            c_sh = T.alloc_shared((blk_s, half), "float32")
            s_sh = T.alloc_shared((blk_s, half), "float32")

            base = bx * blk_s
            T.copy(X[base : base + blk_s, :], x_sh)
            for i, p in T.Parallel(blk_s, half):
                c_sh[i, p] = T.Cast("float32", Cos[base + i, p])
                s_sh[i, p] = T.Cast("float32", Sin[base + i, p])
            for i, p in T.Parallel(blk_s, half):
                lo = T.Cast("float32", x_sh[i, p])
                hi = T.Cast("float32", x_sh[i, p + half])
                Y[base + i, p] = T.Cast(dtype, lo * c_sh[i, p] - hi * s_sh[i, p])
                Y[base + i, p + half] = T.Cast(dtype, hi * c_sh[i, p] + lo * s_sh[i, p])

    return main


# ---------------------------------------------------------------------------
# ③ interleaved：相邻成对
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[3])
def rope_interleaved_blocked(
    S: int,
    R: int,
    blk_s: int = 16,
    dtype: str = "float16",
    threads: int = 256,
):
    """原版 RoFormer 约定。与 ② 的唯一差别是「哪两个元素配对」。

    配对方式决定了**访存模式**：NeoX 从行中间劈开，两条 store 的步长都是 1（好）；
    interleaved 把相邻元素配对，如果按 128-bit 向量读整行，每一对的两个元素都落在
    同一个向量里，仍然能满宽访存；但如果按「偶数只读、奇数只读」的方式组织循环，
    就会退化成步长 2 的散访。
    """
    if R % 2 != 0:
        raise ValueError(f"rope dim must be even, got R={R}")
    if S % blk_s != 0:
        raise ValueError(f"needs S % blk_s == 0, got S={S}, blk_s={blk_s}")
    half = R // 2

    @T.prim_func
    def main(
        X: T.Tensor((S, R), dtype),  # type: ignore[valid-type]
        Cos: T.Tensor((S, half), dtype),  # type: ignore[valid-type]
        Sin: T.Tensor((S, half), dtype),  # type: ignore[valid-type]
        Y: T.Tensor((S, R), dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(S // blk_s, threads=threads) as bx:
            x_sh = T.alloc_shared((blk_s, R), dtype)
            c_sh = T.alloc_shared((blk_s, half), "float32")
            s_sh = T.alloc_shared((blk_s, half), "float32")

            base = bx * blk_s
            T.copy(X[base : base + blk_s, :], x_sh)
            for i, p in T.Parallel(blk_s, half):
                c_sh[i, p] = T.Cast("float32", Cos[base + i, p])
                s_sh[i, p] = T.Cast("float32", Sin[base + i, p])
            for i, p in T.Parallel(blk_s, half):
                x0 = T.Cast("float32", x_sh[i, 2 * p])
                x1 = T.Cast("float32", x_sh[i, 2 * p + 1])
                Y[base + i, 2 * p] = T.Cast(dtype, x0 * c_sh[i, p] - x1 * s_sh[i, p])
                Y[base + i, 2 * p + 1] = T.Cast(dtype, x0 * s_sh[i, p] + x1 * c_sh[i, p])

    return main


if __name__ == "__main__":
    # 需要 cuda。本机从未执行过。
    import torch

    s, r = 4096, 128
    x = torch.randn(s, r, dtype=torch.float16, device="cuda")
    half = r // 2
    angles = torch.randn(s, half, device="cuda")
    cos, sin = angles.cos().half(), angles.sin().half()

    y_ref = torch_rope_neox(x, cos, sin)
    y = rope_neox_blocked(s, r, blk_s=16, dtype="float16")(x, cos, sin)
    torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)
    print("rope_neox_blocked: correctness passed")
