#!/usr/bin/env python3
"""训练一个 7B 模型要多少显存、多少算力：一台机器上的「账本」估算。

这个脚本只回答两个在动手之前必须先算清楚的问题：

1. **显存账本**：权重、梯度、优化器状态、以及它们被分片（ZeRO/FSDP）之后，
   每张卡各背多少 GiB。
2. **算力账本**：训练 `D` 个 token 需要多少 FLOPs，在给定有效吞吐下要多少
   硬件小时。

它是第 17 章 §2「可行性核算」的可执行版本。所有公式都在正文里推导过，
脚本只做算术，没有任何对本机硬件的测量断言。

运行（纯 Python，不需要 GPU，不需要 torch）：

    python3 train_plan.py
    python3 train_plan.py --tokens 10e9 --eff-tflops 120

依据的资料来源见本章 `references.md`：Qwen2.5-7B 的结构参数来自模型卡，
FineWeb-Edu 的 token 规模来自数据集卡，roofline 方法来自第 2 章。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


# 一个 bf16 数占 2 字节，一个 fp32 数占 4 字节。
BYTES_BF16 = 2
BYTES_FP32 = 4
# GiB = 1024^3 字节。显存账单用 GiB；厂商标称的「80GB」是 10^9 字节，
# 二者相差约 7%，估算时按更大的那个（GiB）保守对待。
BYTES_PER_GIB = float(1024**3)


@dataclass(frozen=True)
class ModelSpec:
    """一个模型在账本里需要的结构参数。

    只有 `params` 是显存账本真正需要的；其余字段用于读者对照第 5 章
    「结构参数如何决定算子选择」。
    """

    name: str
    params: float          # 总参数量（含 embedding）
    non_embedding: float   # 去掉 embedding 的输出层与词表参数
    layers: int
    hidden: int
    q_heads: int
    kv_heads: int
    intermediate: int
    vocab: int
    context: int


# Qwen2.5-7B 的公开结构参数，逐项来自模型卡（见 references.md R2）。
QWEN25_7B = ModelSpec(
    name="Qwen2.5-7B",
    params=7.61e9,
    non_embedding=6.53e9,
    layers=28,
    hidden=3584,
    q_heads=28,
    kv_heads=4,
    intermediate=18944,
    vocab=152064,
    context=131072,
)


@dataclass(frozen=True)
class TrainState:
    """优化器与精度的选择，决定每个参数要存几份。"""

    weight_dtype_bytes: int = BYTES_BF16   # bf16 混合精度：前向/反向用 bf16 权重
    grad_dtype_bytes: int = BYTES_BF16     # 梯度也存 bf16
    adam_dtype_bytes: int = BYTES_FP32     # Adam 的 m、v 用 fp32
    fp32_master: bool = True               # 是否额外保留 fp32 主权重


def memory_bytes(params: float, state: TrainState) -> dict[str, float]:
    """返回训练态下每个参数项占用的字节数。"""

    weights = params * state.weight_dtype_bytes
    grads = params * state.grad_dtype_bytes
    adam_m = params * state.adam_dtype_bytes
    adam_v = params * state.adam_dtype_bytes
    master = params * BYTES_FP32 if state.fp32_master else 0.0
    total = weights + grads + adam_m + adam_v + master
    return {
        "权重 (bf16)": weights,
        "梯度 (bf16)": grads,
        "Adam m (fp32)": adam_m,
        "Adam v (fp32)": adam_v,
        "fp32 主权重": master,
        "合计": total,
    }


def to_gib(num_bytes: float) -> float:
    return num_bytes / BYTES_PER_GIB


def training_flops(params: float, tokens: float) -> float:
    """C ≈ 6 · N · D。

    推导（正文 §2.3）：前向每个参数每个 token 约 2 FLOPs（一次乘一次加），
    反向约为前向的两倍（对权重和输入各算一次梯度），合计每参数每 token
    约 6 FLOPs。这是一个忽略 attention 里序列长度二次项的常用估算，
    仅在「token 数远大于 6·序列长度」时准确——短序列、长上下文时要乘修正系数。
    """

    return 6.0 * params * tokens


def hardware_hours(flops: float, eff_tflops: float, n_accelerators: int) -> float:
    """给定「每张卡的有效吞吐」（TFLOP/s，已含 MFU 折扣），返回卡·小时。"""

    if eff_tflops <= 0 or n_accelerators <= 0:
        raise ValueError("eff_tflops 与 n_accelerators 必须为正")
    per_second = eff_tflops * 1e12 * n_accelerators
    seconds = flops / per_second
    return seconds / 3600.0 * n_accelerators


def print_memory_ledger(spec: ModelSpec, state: TrainState, dp_sizes: tuple[int, ...]) -> float:
    print(f"== 显存账本：{spec.name}（{spec.params / 1e9:.2f}B 参数）==")
    ledger = memory_bytes(spec.params, state)
    for name, value in ledger.items():
        print(f"  {name:<16} {to_gib(value):8.2f} GiB")
    total = ledger["合计"]
    shard = "  ".join(
        f"{dp} 卡→{to_gib(total) / dp:6.2f} GiB" for dp in dp_sizes if dp > 0
    )
    if shard:
        print(f"  按 ZeRO-3/FSDP 均摊（优化器/权重/梯度）：{shard}")
    print("  （不含激活值、通信缓冲、CUDA context 与显存碎片；见正文 §2.2）")
    print()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="7B 训练显存/算力账本")
    parser.add_argument("--tokens", type=float, default=10e9, help="训练 token 数")
    parser.add_argument("--eff-tflops", type=float, default=120.0,
                        help="每张卡的有效吞吐（TFLOP/s，含 MFU 折扣；务必用你机器实测值替换）")
    parser.add_argument("--dp-size", type=int, default=8, help="数据并行的卡数，用于均摊账本")
    args = parser.parse_args()

    state = TrainState()
    dp_sizes = tuple(sorted({1, 8, 16, args.dp_size}))
    print_memory_ledger(QWEN25_7B, state, dp_sizes=dp_sizes)

    print(f"== 算力账本：在 {args.tokens / 1e9:.1f}B token 上训练 ==")
    flops = training_flops(QWEN25_7B.params, args.tokens)
    print(f"  C ≈ 6·N·D = {flops:.3e} FLOPs")
    for cards in (1, 8, 64):
        hours = hardware_hours(flops, args.eff_tflops, cards)
        wall = hours / cards
        print(f"  {cards:>3} 卡 × {args.eff_tflops:>5.0f} TFLOP/s 有效吞吐 → "
              f"{hours:10.1f} 卡·小时，墙钟 {wall:8.2f} 小时")
    print()
    print("提示：`--eff-tflops` 必须替换成你自己机器上实测到的有效吞吐")
    print("（第 12 章讲怎么测），否则上面的时间是假设而不是结论。")


if __name__ == "__main__":
    main()
