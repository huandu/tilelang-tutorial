#!/usr/bin/env python3
"""第 18 章 · 一次 decode step 里各算子的耗时占比（纯标准库，CPU 可跑）。

这是本章第 4 节那张表的**计算过程**。它把第 16 章的「瓶颈类型」结论
（`chapters/16-operators-moe-mla/code/01-cost-ledger/decode_step_ledger.py`
的 ``decode_step_summary()``）落到一个可以在任意机器上复算的 roofline 模型上。

模型
----

对每个算子分别算：

* ``flops``：乘加次数；纯搬运算子记 0。
* ``bytes``：权重流量 + 激活/中间结果流量（理想情况下的下界）。
* ``t_mem = bytes / BW``，``t_compute = flops / peak``，
  ``t = max(t_mem, t_compute)``——**假设访存与计算完美重叠**的下界。
* ``share = t / Σt``。

判类用第 02 / 14 / 16 章同一套口径：``AI = flops / bytes`` 与机器平衡点比较。

这是一个**下界模型**，不是测量：

* 真实 kernel 有启动开销、同步、尾部效应、cache 命中，每一项都会让
  「小算子」的占比比这里更高；
* quantized / fused 实现会改变权重流量与算子个数，占比随之变化；
* 多卡并行时的通信完全不在这里的账里。

所以本脚本的输出只用于回答「哪个算子值得优化」，不能当成 benchmark 结果。

用法
----

    python3 decode_step_share.py
    python3 decode_step_share.py --batch 32 --context 8192
    python3 decode_step_share.py --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 机器与模型
# ---------------------------------------------------------------------------

#: H100 SXM5 公开规格（非实测），与第 02 / 14 / 16 章同源。
PEAK_FLOPS = 1000e12
BANDWIDTH = 3.0e12
BALANCE = PEAK_FLOPS / BANDWIDTH  # 约 333 FLOP/Byte

#: Qwen2.5-7B；字段来源同 `02-kv-ledger`。
CFG = {
    "n_layers": 28,
    "n_q_heads": 28,
    "n_kv_heads": 4,
    "head_dim": 128,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "vocab_size": 152064,
    "dtype_bytes": 2,
}


@dataclass
class Op:
    """一个算子的账。"""

    name: str
    flops: float
    bytes_: float
    note: str = ""

    @property
    def ai(self) -> float:
        return float("inf") if self.bytes_ <= 0 else self.flops / self.bytes_

    @property
    def seconds(self) -> float:
        return max(self.bytes_ / BANDWIDTH, self.flops / PEAK_FLOPS)

    @property
    def bound(self) -> str:
        return "compute" if self.ai >= BALANCE else "memory"


def gemm_ops(name: str, m: int, n: int, k: int, b: int,
             note: str = "", matrices: int = 1) -> Op:
    """``matrices`` 个同形状 GEMM/GEMV：权重各读一次、激活共用、输出各写一次。"""
    flops = 2.0 * m * n * k * matrices
    byts = float(b) * (m * k + matrices * k * n + matrices * m * n)
    return Op(name, flops, byts, note)


def build_ops(batch: int, context: int) -> list[Op]:
    """一层解码 + 最终 lm_head 的算子清单。"""
    b = CFG["dtype_bytes"]
    d = CFG["hidden_size"]
    i = CFG["intermediate_size"]
    h_q, h_kv, hd = CFG["n_q_heads"], CFG["n_kv_heads"], CFG["head_dim"]
    layers = CFG["n_layers"]
    t = batch  # decode：这一步只有 1 个 token / 请求

    per_layer: list[Op] = []

    # QKV 投影：拆成 q / k / v 三块，权重形状不同（GQA）。
    per_layer.append(gemm_ops("QKV 投影 (q)", t, d, d, b))
    per_layer.append(gemm_ops("QKV 投影 (k,v)", t, 2 * h_kv * hd, d, b,
                              "GQA：KV 头只有 4 个"))
    # 输出投影
    per_layer.append(gemm_ops("输出投影 (o)", t, d, d, b))

    # Attention：QK^T 与 PV；访存是读 K 和 V 各一份
    attn_flops = 4.0 * t * h_q * context * hd
    attn_bytes = 2.0 * t * h_kv * context * hd * b
    per_layer.append(Op("decode attention", attn_flops, attn_bytes,
                        "KV 流量随 context 线性增长"))

    # MLP：gate / up / down
    per_layer.append(gemm_ops("MLP gate+up", t, i, d, b, matrices=2))
    per_layer.append(gemm_ops("MLP down", t, d, i, b))

    # Norm + RoPE：纯访存
    act_bytes = float(t) * d * b
    per_layer.append(Op("RMSNorm + RoPE（2 个 norm）", 0.0, 4.0 * act_bytes,
                        "读 + 写各两遍"))

    ops: list[Op] = []
    for op in per_layer:
        ops.append(Op(f"{op.name} × {layers} 层", op.flops * layers,
                      op.bytes_ * layers, op.note))

    # 最终分类头（每步只算一次）
    ops.append(gemm_ops("LM head / logits", t, CFG["vocab_size"], d, b,
                        "每步只算一次，但权重很大"))
    return ops


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def print_weight_check() -> None:
    """自检：账本里的权重字节必须能对上模型卡的总参数量。"""
    b = CFG["dtype_bytes"]
    d = CFG["hidden_size"]
    i = CFG["intermediate_size"]
    h_kv, hd = CFG["n_kv_heads"], CFG["head_dim"]
    v = CFG["vocab_size"]
    layers = CFG["n_layers"]

    per_layer = d * d + 2 * d * (h_kv * hd) + d * d + 2 * d * i + i * d
    embed = 2 * v * d  # token embedding + lm_head
    total_params = per_layer * layers + embed
    print("=" * 96)
    print("自检：账本里出现的矩阵，参数总量必须等于模型卡给的 7.61B")
    print("=" * 96)
    print(f"  每层参数 {per_layer / 1e6:.2f} M × {layers} 层 + "
          f"嵌入/头 {embed / 1e6:.2f} M = {total_params / 1e9:.3f} B"
          f"（模型卡 7.61 B，相对误差 {abs(total_params / 7.61e9 - 1):.4%}）")
    print(f"  其中账本里参与的「权重流量」是除 token embedding 外的部分："
          f"{(total_params - v * d) * b / 1e9:.2f} GB")
    print("  token embedding 只在第一步按 token 查表，decode 每步的流量可忽略，")
    print("  但它照样占显存——这是「显存账本」和「带宽账本」不是一回事的例子。\n")


def report(batch: int, context: int) -> list[dict[str, object]]:
    ops = build_ops(batch, context)
    total = sum(op.seconds for op in ops)
    rows: list[dict[str, object]] = []
    for op in ops:
        rows.append({
            "operator": op.name,
            "t_ms": op.seconds * 1e3,
            "share": op.seconds / total,
            "ai": op.ai,
            "bound": op.bound,
            "bytes": op.bytes_,
            "flops": op.flops,
            "note": op.note,
        })
    rows.sort(key=lambda r: float(r["t_ms"]), reverse=True)
    return rows


def print_table(batch: int, context: int) -> None:
    rows = report(batch, context)
    total = sum(float(r["t_ms"]) for r in rows)
    print("=" * 96)
    print(f"一次 decode step 的算子账本：batch={batch}, context={context}, "
          f"bf16, H100 公开规格")
    print("=" * 96)
    print(f"{'算子':<34} | {'FLOP':>10} | {'字节':>11} | {'AI':>8} | "
          f"{'下界时间':>10} | {'占比':>7} | 受限")
    print("-" * 96)
    for r in rows:
        print(f"{str(r['operator']):<34} | {float(r['flops']):>10.3e} | "
              f"{fmt_bytes(float(r['bytes'])):>11} | {float(r['ai']):>8.1f} | "
              f"{float(r['t_ms']):>8.4f}ms | {float(r['share']):>6.1%} | "
              f"{str(r['bound']):>7}-bound")
    print("-" * 96)
    print(f"{'合计（完美重叠的下界）':<34} |            |             | "
          f"         | {total:>8.4f}ms | 100.0% |")
    print(f"\n  机器平衡点 {BALANCE:.0f} FLOP/Byte；"
          f"表中的 AI 全部远低于它 → 整个 decode step 是 memory-bound。")
    print("  占比最大的两项是权重流量（投影与 MLP）与 KV 流量（attention）；")
    print("  前者与 batch 无关、被 batch 摊薄，后者随 context 线性增长——")
    print("  所以「加 batch」和「加 context」会改变这张表，方向相反。\n")


def print_sensitivity() -> None:
    print("=" * 96)
    print("占比怎么随 batch / context 变（同一套账本）")
    print("=" * 96)
    print(f"{'batch':>6} | {'context':>8} | {'attention 占比':>14} | "
          f"{'权重类算子占比':>16} | {'单步下界(ms)':>13}")
    for batch, ctx in ((1, 8192), (1, 131072), (8, 8192), (32, 8192),
                       (32, 131072), (128, 8192)):
        rows = report(batch, ctx)
        total = sum(float(r["t_ms"]) for r in rows)
        attn = sum(float(r["share"]) for r in rows
                   if "attention" in str(r["operator"]))
        weightish = 1.0 - attn - sum(float(r["share"]) for r in rows
                                     if "Norm" in str(r["operator"]))
        print(f"{batch:>6} | {ctx:>8} | {attn:>13.1%} | {weightish:>15.1%} | "
              f"{total:>11.4f}")
    print("\n  读法：attention 的占比只由 context（和并发序列的总 KV 长度）决定；")
    print("  权重类算子的绝对时间不随 batch 变，但被 batch 摊薄。")
    print("  这就是 continuous batching 能提高吞吐的根本原因（见 `04-serving-schedule-sim`）。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.json:
        payload = {
            "balance": BALANCE,
            "batch": args.batch,
            "context": args.context,
            "rows": report(args.batch, args.context),
            "sensitivity": [
                {"batch": b, "context": c,
                 "rows": report(b, c)}
                for b, c in ((1, 8192), (1, 131072), (32, 8192), (32, 131072))
            ],
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print_weight_check()
    print_table(args.batch, args.context)
    print_sensitivity()
    print("提醒：以上是**下界模型**。真实占比会被 kernel 启动、融合程度、"
          "\n      量化方案和并行策略改变；要拿真实占比请用第 12 章的 profiler。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
