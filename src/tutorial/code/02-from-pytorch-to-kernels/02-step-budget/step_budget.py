#!/usr/bin/env python3
"""第 02 章 示例 2：一次训练迭代 / 一次推理请求的算子账本。

示例 1 算的是**一个算子**。这个脚本把一个 decoder-only Transformer 的

* 训练一步：前向 → 反向 → 优化器 → 通信
* 推理一次请求：prefill（算力）+ decode 每 token（带宽）

拆成几张表，用来回答「时间花在哪一段」。它同样**不测量任何性能**，只做算术。

粗模型（本书口径，写死在下面的注释里，便于读者自己反驳）：

* 每层前向的权重访存 = 该层权重读一遍；
* 每层前向的激活访存 = 残差流（读入 + 写出）+ MLP 中间张量（写 + 读 + 写）；
* 反向按前向的 2x FLOPs、2x 激活访存估算，另加权重梯度的写出；
* 优化器按 AdamW 数（见示例 1 的 `adamw`）；
* 通信按 ring all-reduce 的 2(N-1)/N 倍梯度数据量估算；
* 不建 KV cache 的增长模型（decode 只看某个固定长度下的一次调用）。

只用标准库：

    python3 step_budget.py
    python3 step_budget.py --seq-len 32768
    python3 step_budget.py --layers 80 --hidden 8192 --ffn 28672 --params 70e9 --world 64
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


B = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "fp8": 1.0}


@dataclass(frozen=True)
class Config:
    layers: int
    hidden: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int
    seq: int
    batch: int
    params: float
    world: int
    weight_dtype: str
    state_dtype: str
    grad_dtype: str
    compute_dtype: str

    @property
    def q_dim(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_heads * self.head_dim


@dataclass
class Stage:
    name: str
    flops: float
    nbytes: float
    note: str


def per_layer_weights(cfg: Config) -> float:
    """一层 decoder 的权重大小（q/k/v/o 投影 + MLP 三个矩阵 + 两个 norm）。"""
    qkv = cfg.q_dim * cfg.hidden + 2 * cfg.kv_dim * cfg.hidden
    out_proj = cfg.q_dim * cfg.hidden
    mlp = 3 * cfg.hidden * cfg.ffn
    norms = 2 * cfg.hidden
    return float(qkv + out_proj + mlp + norms)


def forward_layer(cfg: Config) -> Stage:
    b, s = cfg.batch, cfg.seq
    h, i = cfg.hidden, cfg.ffn
    q, kv, d, heads = cfg.q_dim, cfg.kv_dim, cfg.head_dim, cfg.heads

    flops = 0.0
    flops += 2 * b * s * q * h  # Q 投影
    flops += 2 * b * s * kv * h * 2  # K、V 投影
    flops += 4 * b * heads * s * s * d  # QK^T + PV（每分数 2 FMAs）
    flops += 5 * b * heads * s * s  # softmax 每分数 5 FLOP（本书口径，见示例 1）
    flops += 2 * b * s * q * h  # 输出投影
    flops += 6 * b * s * h * i  # MLP：gate、up、down

    # 残差流：读入一份、写出两份（attn 输出、MLP 输出前的残差）；MLP 中间张量读写一次
    act_bytes = (3 * b * s * h + 3 * b * s * i) * B[cfg.compute_dtype]
    weight_bytes = per_layer_weights(cfg) * B[cfg.weight_dtype]
    return Stage("前向", flops, act_bytes + weight_bytes,
                 "每层：权重读一遍 + 残差流/MLP 中间张量各读写一次")


def backward_layer(cfg: Config) -> Stage:
    fwd = forward_layer(cfg)
    # 反向：2x 前向 FLOPs；激活重读一遍、写出权重梯度
    act_bytes = (3 * cfg.batch * cfg.seq * cfg.hidden
                 + 3 * cfg.batch * cfg.seq * cfg.ffn) * B[cfg.compute_dtype] * 2
    weight_bytes = per_layer_weights(cfg) * (B[cfg.weight_dtype] + B[cfg.grad_dtype])
    return Stage("反向", 2 * fwd.flops, act_bytes + weight_bytes,
                 "按 2x 前向 FLOPs 估算；激活再读写一遍 + 权重梯度写出")


def optimizer_step(cfg: Config) -> Stage:
    out_el = B[cfg.state_dtype]
    grad_el = B[cfg.grad_dtype]
    nbytes = cfg.params * (grad_el + 6 * out_el)
    return Stage("优化器", 15.0 * cfg.params, nbytes,
                 "AdamW：读 grad/param/m/v，写 param/m/v")


def comm_step(cfg: Config) -> Stage:
    nbytes = 2.0 * cfg.params * B[cfg.grad_dtype] * (cfg.world - 1) / cfg.world
    return Stage("通信", 0.0, nbytes, "ring all-reduce 梯度：搬 2(N-1)/N 倍梯度数据")


def train_stages(cfg: Config) -> list[Stage]:
    fwd = forward_layer(cfg)
    bwd = backward_layer(cfg)
    layers = float(cfg.layers)
    return [
        Stage("前向（全部层）", fwd.flops * layers, fwd.nbytes * layers, fwd.note),
        Stage("反向（全部层）", bwd.flops * layers, bwd.nbytes * layers, bwd.note),
        optimizer_step(cfg),
        comm_step(cfg),
    ]


def prefill_stage(cfg: Config) -> Stage:
    per_layer = forward_layer(cfg)
    return Stage("Prefill（全部层）", per_layer.flops * cfg.layers,
                 per_layer.nbytes * cfg.layers, "见「前向」一行的口径")


def decode_stage(cfg: Config) -> Stage:
    """decode：batch 个 request、每个 1 个 query token，KV cache 长度 seq。"""
    b, s, d = cfg.batch, cfg.seq, cfg.head_dim
    flops = cfg.layers * (
        2 * b * cfg.q_dim * cfg.hidden  # Q 投影
        + 2 * b * 2 * cfg.kv_dim * cfg.hidden  # K、V 投影（只为新 token）
        + 4 * b * cfg.heads * s * d  # 与整段 KV 做 QK^T 和 PV
        + 5 * b * cfg.heads * s  # 在整段 KV 上的 softmax
        + 2 * b * cfg.q_dim * cfg.hidden  # 输出投影
        + 6 * b * cfg.hidden * cfg.ffn  # MLP
    )
    # 访存：读全部权重 + 读整段 KV cache
    nbytes = cfg.layers * per_layer_weights(cfg) * B[cfg.weight_dtype]
    nbytes += cfg.layers * 2 * b * cfg.kv_dim * s * B["fp16"]
    nbytes += cfg.layers * 3 * b * cfg.hidden * B["fp16"]  # 残差流与 MLP 中间量
    return Stage("Decode（每 token）", flops, nbytes,
                 "读全部权重 + 读整段 KV cache；q/o 很小")


def fmt_bytes(value: float) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= scale:
            return f"{value/scale:.2f} {unit}"
    return f"{value:.0f} B"


def fmt_time(value: float) -> str:
    if value >= 1.0:
        return f"{value:.3f} s"
    if value >= 1e-3:
        return f"{value*1e3:.3f} ms"
    return f"{value*1e6:.1f} us"


def print_table(title: str, rows: list[Stage], peak_flops: float, bandwidth: float) -> None:
    print("=" * 104)
    print(title)
    print("-" * 104)
    print(f"{'阶段':<20} {'运算量':>12} {'访存量':>11} {'访存时间':>11} {'计算时间':>11} "
          f"{'下界':>11} {'占比':>7}")
    sols = [max(r.nbytes / bandwidth, r.flops / peak_flops) for r in rows]
    total = sum(sols)
    for row, sol in zip(rows, sols):
        share = sol / total * 100 if total > 0 else 0.0
        print(f"{row.name:<20} {row.flops/1e9:>9.1f} GF {fmt_bytes(row.nbytes):>11} "
              f"{fmt_time(row.nbytes/bandwidth):>11} {fmt_time(row.flops/peak_flops):>11} "
              f"{fmt_time(sol):>11} {share:>6.1f}%")
    for row in rows:
        print(f"    · {row.name}：{row.note}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--ffn", type=int, default=14336)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=4096, help="训练序列长度 / decode 的 KV 长度")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--params", type=float, default=8.03e9)
    parser.add_argument("--world", type=int, default=8)
    parser.add_argument("--peak-tflops", type=float, default=729.3)
    parser.add_argument("--bandwidth-gbps", type=float, default=4502.0)
    args = parser.parse_args()

    cfg = Config(
        layers=args.layers, hidden=args.hidden, ffn=args.ffn, heads=args.heads,
        kv_heads=args.kv_heads, head_dim=args.head_dim, seq=args.seq_len,
        batch=args.batch, params=args.params, world=args.world,
        weight_dtype="bf16", state_dtype="fp32", grad_dtype="bf16", compute_dtype="bf16",
    )
    peak_flops = args.peak_tflops * 1e12
    bandwidth = args.bandwidth_gbps * 1e9

    print("=" * 104)
    print("模型与机器")
    print("-" * 104)
    print(f"  层数 {cfg.layers}，hidden {cfg.hidden}，ffn {cfg.ffn}，"
          f"{cfg.heads} 个 query head / {cfg.kv_heads} 个 KV head，head_dim {cfg.head_dim}")
    print(f"  参数 {cfg.params/1e9:.2f}B，数据并行 {cfg.world} 卡，"
          f"权重 {cfg.weight_dtype}，优化器状态 {cfg.state_dtype}，梯度 {cfg.grad_dtype}")
    print(f"  机器：{peak_flops/1e12:.1f} TFLOPS fp16，{bandwidth/1e9:.0f} GB/s，"
          f"平衡点 {peak_flops/bandwidth:.1f} FLOP/Byte")
    print("  机器参数用示例 1 的 H200 标定值；换成别的机器请用 --peak-tflops/--bandwidth-gbps")

    print_table(f"训练一步（seq_len={cfg.seq}，batch={cfg.batch}）", train_stages(cfg),
                peak_flops, bandwidth)
    print_table("推理", [prefill_stage(cfg), decode_stage(cfg)], peak_flops, bandwidth)

    print("=" * 104)
    print("所有数字都是「理论最少」的下界，按本书口径的粗模型推导，不是实测，也不是性能承诺。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
