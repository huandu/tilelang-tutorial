#!/usr/bin/env python3
"""第 13 章 · 逐元素 / 归一化 / RoPE 的访存账本（纯标准库，CPU 可跑）。

这份脚本把第 02 章定义的 roofline 记账口径落成一段可复算的代码，回答三个问题：

1. 一个逐元素算子在「一个 op 一个 kernel」的写法下，账本上要搬多少字节？
2. 把若干 op 融合成一个 kernel 之后，账本上最少要搬多少字节？比值是多少？
3. 这类算子的算术强度落在什么量级，离机器平衡点有多远？

记账口径（与第 02 章 §2.1 一致）
--------------------------------

* 一次「算子调用」：每个输入张量读一次，每个输出张量写一次。
* 中间张量不免费：它是上一次调用的输出、下一次调用的输入，各计一次。
* 基本运算（add / sub / mul / div / max）记 1 FLOP；超越函数
  （exp / log / sqrt / rsqrt / cos / sin …）在账本上也记 1 FLOP。
* 结果是一个**下界**：不含 cache 命中、kernel 启动、尾部效应与同步开销。

这份脚本是**模型**，不是测量
----------------------------

它不 import TileLang，不碰 GPU/NPU，也不产生任何性能数字。它输出的 bytes / FLOP
是算术事实，可以在任何机器上复算。真实 kernel 一定比这个下界慢。

用法
----

    python3 ledger.py            # 打印全部账本
    python3 ledger.py --json     # 机器可读输出，便于回归比较
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Iterable

# 元素字节数：只覆盖本章用到的 dtype。
ELEM_BYTES: dict[str, int] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
}

@dataclass(frozen=True)
class Buf:
    """一个张量：名字只用于打印，账本只关心元素数与 dtype。"""

    name: str
    numel: int
    dtype: str

    def __post_init__(self) -> None:
        if self.dtype not in ELEM_BYTES:
            raise ValueError(f"unknown dtype {self.dtype!r}")
        if self.numel < 0:
            raise ValueError(f"{self.name}: negative numel")

    @property
    def nbytes(self) -> int:
        return self.numel * ELEM_BYTES[self.dtype]


@dataclass(frozen=True)
class Op:
    """一次算子调用：读哪些张量、写哪些张量、账本上记多少 FLOP。"""

    name: str
    reads: tuple[Buf, ...]
    writes: tuple[Buf, ...]
    flops: int

    @property
    def moved(self) -> int:
        return sum(b.nbytes for b in self.reads) + sum(b.nbytes for b in self.writes)

    @property
    def ai(self) -> float:
        return self.flops / self.moved if self.moved else float("inf")


@dataclass(frozen=True)
class Plan:
    """一组算子调用（一个 kernel 就是一个 Op；一个 plan 是一串 kernel）。"""

    name: str
    ops: tuple[Op, ...]

    @property
    def moved(self) -> int:
        return sum(op.moved for op in self.ops)

    @property
    def flops(self) -> int:
        return sum(op.flops for op in self.ops)

    @property
    def ai(self) -> float:
        return self.flops / self.moved if self.moved else float("inf")

    @property
    def kernels(self) -> int:
        return len(self.ops)


def mb(nbytes: int) -> float:
    return nbytes / 1_000_000


def human(plan: Plan) -> str:
    return (
        f"{plan.name:<34} kernels={plan.kernels}  "
        f"moved={mb(plan.moved):9.2f} MB  flops={plan.flops / 1e6:8.2f} MFLOP  "
        f"AI={plan.ai:6.3f} FLOP/Byte"
    )


# --------------------------------------------------------------------------
# 1) 逐元素：y = x * a + b     （a 是标量，b 是长度 N 的向量）
# --------------------------------------------------------------------------


def elementwise_scale_add(n: int, dtype: str) -> tuple[Plan, Plan]:
    elem = Buf("x", n, dtype)
    bias = Buf("b", n, dtype)
    tmp = Buf("t", n, dtype)  # 中间张量，落一次显存
    out = Buf("y", n, dtype)

    unfused = Plan(
        "scale+add（2 个 kernel）",
        (
            Op("t = x * a", (elem,), (tmp,), flops=n),
            Op("y = t + b", (tmp, bias), (out,), flops=n),
        ),
    )
    fused = Plan(
        "scale+add（融合，1 个 kernel）",
        (Op("y = x * a + b", (elem, bias), (out,), flops=2 * n),),
    )
    return unfused, fused


# --------------------------------------------------------------------------
# 2) RMSNorm： y = x * rsqrt(mean(x^2) + eps) * w
# --------------------------------------------------------------------------


def rms_norm(m: int, n: int, dtype: str, *, eager_steps: bool = True) -> tuple[Plan, Plan]:
    """回到「一个 op 一个 kernel」的写法 vs 融合成一个 kernel。

    eager_steps=True 时把 PyTorch 语义逐步展开成 6 次调用：
    pow → mean → add(eps) → rsqrt → mul → mul(weight)。
    实际 kernel 数取决于 PyTorch 的算子融合与 dispatch；这里只给**下界**。
    """

    x = Buf("x", m * n, dtype)
    w = Buf("w", n, dtype)
    y = Buf("y", m * n, dtype)

    if eager_steps:
        p = Buf("x2", m * n, dtype)
        mean = Buf("mean", m, "float32")
        mean_eps = Buf("mean_eps", m, "float32")
        rstd = Buf("rstd", m, "float32")
        scaled = Buf("scaled", m * n, dtype)
        unfused = Plan(
            "RMSNorm（6 步分解）",
            (
                Op("x2 = x * x", (x,), (p,), flops=m * n),
                Op("mean = x2.sum(-1)", (p,), (mean,), flops=m * n),
                Op("mean_eps = mean + eps", (mean,), (mean_eps,), flops=m),
                Op("rstd = rsqrt(mean_eps)", (mean_eps,), (rstd,), flops=m),
                Op("scaled = x * rstd", (x, rstd), (scaled,), flops=m * n),
                Op("y = scaled * w", (scaled, w), (y,), flops=m * n),
            ),
        )
    else:
        unfused = Plan("RMSNorm（未展开）", ())

    fused = Plan(
        "RMSNorm（融合，1 个 kernel）",
        (
            Op(
                "y = x * rsqrt(mean(x^2)+eps) * w",
                (x, w),
                (y,),
                # 每元素：平方 1 + 归约 1 + 乘 rstd 1 + 乘 w 1；每行：rsqrt 1 + 除 1。
                flops=4 * m * n + 2 * m,
            ),
        ),
    )
    return unfused, fused


# --------------------------------------------------------------------------
# 3) LayerNorm： y = (x - mean) / sqrt(var + eps) * g + b
# --------------------------------------------------------------------------


def layer_norm(m: int, n: int, dtype: str) -> tuple[Plan, Plan]:
    x = Buf("x", m * n, dtype)
    g = Buf("g", n, dtype)
    b = Buf("b", n, dtype)
    y = Buf("y", m * n, dtype)

    sum_x = Buf("sum_x", m, "float32")
    sum_x2 = Buf("sum_x2", m, "float32")
    sq = Buf("x2", m * n, dtype)
    acc = Buf("acc", m * n, dtype)

    unfused = Plan(
        "LayerNorm（多趟分解）",
        (
            Op("x2 = x * x", (x,), (sq,), flops=m * n),
            Op("sum_x = x.sum(-1)", (x,), (sum_x,), flops=m * n),
            Op("sum_x2 = x2.sum(-1)", (sq,), (sum_x2,), flops=m * n),
            Op("acc = x - mean", (x,), (acc,), flops=m * n),
            Op("acc = acc * rstd", (acc,), (acc,), flops=m * n),
            Op("y = acc * g + b", (acc, g, b), (y,), flops=2 * m * n),
        ),
    )
    fused = Plan(
        "LayerNorm（融合，1 个 kernel）",
        (
            Op(
                "y = (x-mean)*rstd*g + b",
                (x, g, b),
                (y,),
                # 每元素：求 mean 1 + 求 mean(x^2) 1 + 减 1 + 乘 rstd 1 + 乘 g 1 + 加 b 1；
                # 每行：sqrt/rsqrt 各 1。
                flops=6 * m * n + 2 * m,
            ),
        ),
    )
    return unfused, fused


# --------------------------------------------------------------------------
# 4) RoPE（NeoX 半维旋转）： y_low = x_low*c - x_high*s ; y_high = x_high*c + x_low*s
# --------------------------------------------------------------------------


def rope(tokens: int, heads: int, rope_dim: int, dtype: str) -> tuple[Plan, Plan]:
    """把 RoPE 写成多个 kernel vs 一个 kernel。

    多 kernel 的分解方式是**本脚本假设的一种常见拆法**（不是对 PyTorch 实际产生的
    kernel 数的追踪）：先把 ``rotate_half`` 的结果物化成一个中间张量，再分两次乘、
    一次加写回。中间张量是整宽 `values` 大小，这正是这种写法的代价所在。

    账本口径下「拷贝 / 拼接」不是基本运算，记 0 FLOP；所以两种写法的 FLOP 总数相同，
    差别全在 bytes 上。
    """

    half = rope_dim // 2
    rows = tokens * heads
    values = rows * rope_dim
    pairs = rows * half
    x = Buf("x", values, dtype)
    x_low = Buf("x_low", pairs, dtype)   # x[..., :half] 的视图，不额外占显存
    x_high = Buf("x_high", pairs, dtype)  # x[..., half:] 的视图
    cos = Buf("cos", pairs, dtype)
    sin = Buf("sin", pairs, dtype)
    cos_b = Buf("cos_broadcast", values, dtype)  # 广播视图，按整宽读
    sin_b = Buf("sin_broadcast", values, dtype)
    y = Buf("y", values, dtype)
    rot = Buf("rot", values, dtype)  # rotate_half 的物化结果
    t = Buf("t", values, dtype)
    u = Buf("u", values, dtype)

    unfused = Plan(
        "RoPE（4 步分解，中间量落显存）",
        (
            Op("rot = cat(-x_high, x_low)", (x,), (rot,), flops=0),
            Op("t = x * cos_b", (x, cos_b), (t,), flops=values),
            Op("u = rot * sin_b", (rot, sin_b), (u,), flops=values),
            Op("y = t + u", (t, u), (y,), flops=values),
        ),
    )
    fused = Plan(
        "RoPE（融合，1 个 kernel）",
        (
            Op(
                "y = rotate_half(x) 混合 cos/sin",
                (x, cos, sin),
                (y,),
                # 每个元素 2 次乘 + 1 次加（账本记 3）。
                flops=3 * values,
            ),
        ),
    )
    # 让两边的输入规模可见：x_low / x_high / cos / sin 都只是 x 的一半或更小。
    assert x_low.numel == x_high.numel == cos.numel == sin.numel == pairs
    return unfused, fused


# --------------------------------------------------------------------------
# 5) 融合三件套：RMSNorm + RoPE + per-token FP8 量化
# --------------------------------------------------------------------------


def fused_norm_rope_quant(m: int, n: int, rope_dim: int, dtype: str, out_dtype: str = "float8_e4m3fn") -> tuple[Plan, Plan]:
    """三个算子分别写 vs 融合成一个 kernel。

    n 是每行的 hidden 宽度，rope_dim 是其中被旋转的尾部宽度。
    """

    x = Buf("x", m * n, dtype)
    w = Buf("w", n, dtype)
    cos = Buf("cos", m * rope_dim // 2, dtype)
    sin = Buf("sin", m * rope_dim // 2, dtype)
    normed = Buf("normed", m * n, dtype)
    rotate = Buf("rotated", m * n, dtype)
    q = Buf("q_fp8", m * n, out_dtype)
    amax = Buf("amax", m, "float32")

    separate = Plan(
        "RMSNorm+RoPE+量化（3+ 个 kernel）",
        (
            Op("RMSNorm", (x, w), (normed,), flops=4 * m * n + 2 * m),
            Op("RoPE", (normed, cos, sin), (rotate,), flops=3 * m * rope_dim),
            Op("absmax + scale + 量化", (rotate,), (q, amax), flops=3 * m * n + m),
        ),
    )
    fused = Plan(
        "RMSNorm+RoPE+量化（融合，1 个 kernel）",
        (
            Op(
                "一次读 x，写 q 与 amax",
                (x, w, cos, sin),
                (q, amax),
                flops=4 * m * n + 3 * m * rope_dim + 3 * m * n + 3 * m,
            ),
        ),
    )
    return separate, fused


def report(pairs: Iterable[tuple[Plan, Plan]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ais: list[float] = []
    for unfused, fused in pairs:
        ratio = unfused.moved / fused.moved
        print("=" * 108)
        print(human(unfused))
        print(human(fused))
        print(f"  → 访存量比值 {ratio:.2f}×")
        assert fused.moved < unfused.moved, f"{fused.name}: 融合没有减少访存？"
        ais.extend([unfused.ai, fused.ai])
        rows.append(
            {
                "unfused": unfused.name,
                "fused": fused.name,
                "unfused_mb": round(mb(unfused.moved), 4),
                "fused_mb": round(mb(fused.moved), 4),
                "bytes_ratio": round(ratio, 4),
                "unfused_ai": round(unfused.ai, 6),
                "fused_ai": round(fused.ai, 6),
            }
        )
    print("=" * 108)
    print(f"全部 {len(ais)} 个方案的算术强度落在 {min(ais):.3f} – {max(ais):.3f} FLOP/Byte。")
    print("只要目标机器的平衡点（第 02 章 §2.2）大于这个区间的上界，这些算子就都是 memory-bound。")
    print("本章不写具体型号的平衡点数字：那取决于精度与峰值，要按第 02 章的方法自己算。")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true", help="输出 JSON（便于回归比较）")
    args = parser.parse_args()

    # ---- 1) 逐元素：y = x * a + b，2^20 个 fp32 ----
    n = 1 << 20
    # ---- 2) RMSNorm / LayerNorm：4096 × 4096 ----
    m_norm = 4096
    # ---- 3) RoPE：4096 个 token、32 个 head、rope_dim = 128 ----
    rope_tokens, rope_heads, rope_dim = 4096, 32, 128
    # ---- 4) 融合：同一组形状 ----
    m_fused, n_fused, r_fused = 4096, 4096, 128

    pairs = [
        elementwise_scale_add(n, "float32"),
        rms_norm(m_norm, m_norm, "float16"),
        layer_norm(m_norm, m_norm, "bfloat16"),
        rope(rope_tokens, rope_heads, rope_dim, "float16"),
        fused_norm_rope_quant(m_fused, n_fused, r_fused, "bfloat16"),
    ]

    if args.json:
        print(json.dumps(report(pairs), ensure_ascii=False, indent=2))
    else:
        print("第 13 章 访存账本 —— 口径见本文件头部；这是模型，不是测量。")
        report(pairs)

    print("结论：融合能把这几个算子的访存量降到原来的 1/1.7 到 1/5；优化空间在访存，不在算力。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
