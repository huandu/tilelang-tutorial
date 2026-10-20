#!/usr/bin/env python3
"""Chapter 39, §2.2/§2.8/§4: the per-stage ledger for one MoE layer.

This script recomputes *formulas* that chapter 39 explains; it never measures
hardware. Every printed number carries the source it came from, so a reader can
tell "book recomputation" apart from "vendor-reported measurement".

Sources used here (all quoted in the chapter):
  * comm payload per token per direction = k * d * b
    (ch. 16 §2.6, ch. 38 §1.2)
  * distinct target ranks = R * (1 - (1 - 1/R)^k)
    (ch. 38 §1.3; the table below is the book's own evaluation of it)
  * DeepEP test conditions: T=16384 tokens/rank, hidden=7168, topk=6,
    experts=256, expert_alignment=128, FP8 dispatch + BF16 combine
    (DeepEP-Ascend/README.md:9-21)
  * hardware model constants: HBM 3500 GB/s, bf16 432.5 / fp8 865 TFLOPS
    (DeepGEMM-Ascend/csrc/jit_kernels/heuristics.hpp:46-71; the chapter
    recomputes the peak from flops_per_cycle * clock * cores)

Usage:
    python3 train_step_budget.py
    python3 train_step_budget.py --tokens 16384 --hidden 7168 --topk 6 \
        --experts 256 --ffn 2048 --ep 8 --dispatch-bytes 1 --combine-bytes 2
"""

from __future__ import annotations

import argparse

# --- constants, each with its source -----------------------------------------

# Hardware model constants. These are NOT datasheet peaks: they are the values
# the DeepGEMM-Ascend heuristics use to price a kernel (heuristics.hpp:46-71).
HBM_BW_GBPS = 3500.0
L2_BW_GBPS = 4500.0
BF16_TFLOPS = 432.5  # 8192 flops/cycle * 1650 MHz * 32 cores, book recomputation
FP8_TFLOPS = 865.0

# DeepEP-Ascend README:9-21 test conditions.
DEEPEP_TEST = {
    "tokens_per_rank": 16384,
    "hidden": 7168,
    "topk": 6,
    "experts": 256,
    "expert_alignment": 128,
    "dispatch_bytes": 1,  # FP8
    "combine_bytes": 2,  # BF16
}

EP_SIZES = (8, 16, 32, 64, 128)


def distinct_ranks(rank_count: int, topk: int) -> float:
    """Expected number of distinct target ranks for one token (ch. 38 §1.3)."""
    return rank_count * (1.0 - (1.0 - 1.0 / rank_count) ** topk)


def kb(value_bytes: float) -> str:
    return f"{value_bytes / 1024:.1f} KB"


def gb(value_bytes: float) -> str:
    return f"{value_bytes / 1e9:.3f} GB"


def print_comm_ledger(args: argparse.Namespace) -> None:
    payload_dispatch = args.topk * args.hidden * args.dispatch_bytes
    payload_combine = args.topk * args.hidden * args.combine_bytes + args.topk * 4

    print("=" * 74)
    print("① 通信账：每 token 每方向的逻辑 payload（口径 A）")
    print("=" * 74)
    print("公式      payload = k * hidden * bytes_per_element   (第 16 章 §2.6)")
    print(f"参数      k={args.topk}, hidden={args.hidden}")
    print(f"dispatch  {args.topk}*{args.hidden}*{args.dispatch_bytes} = "
          f"{payload_dispatch} B = {kb(payload_dispatch)}   (FP8，1 字节)")
    print(f"combine   {args.topk}*{args.hidden}*{args.combine_bytes} + k*4 = "
          f"{payload_combine} B = {kb(payload_combine)}   (BF16，2 字节 + 权重)")
    print()

    print("=" * 74)
    print("② 去重修正：一个 token 期望覆盖多少个不同的目标 rank")
    print("=" * 74)
    print("公式      distinct_ranks(R, k) = R * (1 - (1 - 1/R)^k)   (第 38 章 §1.3)")
    print()
    print(f"{'EP 规模 R':>10}  {'distinct_ranks':>15}  {'相对 k 的比例':>14}")
    base = float(args.topk)
    for rank_count in EP_SIZES:
        value = distinct_ranks(rank_count, args.topk)
        print(f"{rank_count:>10}  {value:>15.2f}  {value / base:>13.1%}")
    print()
    first, last = distinct_ranks(EP_SIZES[0], args.topk), distinct_ranks(EP_SIZES[-1], args.topk)
    print(f"EP{EP_SIZES[0]} → EP{EP_SIZES[-1]}：跨设备字节量上升 "
          f"{last / first - 1:.0%}（{first:.2f} → {last:.2f}）")
    print("这是「EP 越大越慢」的第一条原因的定量版本（第 38 章 §7.2）。")
    print()

    print("=" * 74)
    print("③ 全集总量（口径 C）与它相对链路的时间下界")
    print("=" * 74)
    print("口径 A、每 rank 每方向的总 payload，再除以该 EP 规模下的实测带宽。")
    print("带宽取 DeepEP-Ascend/README.md:13-19 的区间上限，条件见该 README。")
    print()
    print(f"{'EP':>5}  {'dispatch':>22}  {'combine':>22}")
    dispatch_bw = {8: 375.0, 16: 352.0, 32: 340.0, 64: 327.0, 128: 320.0}
    combine_bw = {8: 347.0, 16: 341.0, 32: 324.0, 64: 298.0, 128: 278.0}
    for rank_count in EP_SIZES:
        factor = distinct_ranks(rank_count, args.topk) / args.topk
        total_d = payload_dispatch * args.tokens_per_rank * factor
        total_c = payload_combine * args.tokens_per_rank * factor
        t_d = total_d / (dispatch_bw[rank_count] * 1e9) * 1e6
        t_c = total_c / (combine_bw[rank_count] * 1e9) * 1e6
        print(f"{rank_count:>5}  {gb(total_d):>12} / {t_d:7.0f} us  "
              f"{gb(total_c):>12} / {t_c:7.0f} us")
    print()
    print("注意：这两列是「纯搬运时间」的下界估算，不含 kernel 启动、barrier、")
    print("      padding，也没有考虑它能与多少计算重叠。用途只有一个——判类。")
    print()


def print_gemm_ledger(args: argparse.Namespace) -> None:
    # Per rank, one layer: each token goes to k experts; each expert does two
    # GEMMs of 2*M*N*K (gate/up fused as one, then down).
    rows = args.tokens_per_rank * args.topk
    flops_up = 2 * rows * (2 * args.ffn) * args.hidden
    flops_down = 2 * rows * args.hidden * args.ffn
    flops_total = flops_up + flops_down

    bytes_b_up = (2 * args.ffn * args.hidden) * args.dispatch_bytes
    bytes_b_down = (args.hidden * args.ffn) * args.dispatch_bytes
    bytes_a = rows * args.hidden * args.dispatch_bytes * 2
    bytes_total = bytes_b_up + bytes_b_down + bytes_a

    print("=" * 74)
    print("④ 算力账：本 rank 上这一层的专家 GEMM（口径：2*M*N*K）")
    print("=" * 74)
    print(f"行数      M = tokens * topk = {args.tokens_per_rank} * {args.topk} = {rows}")
    print("注意      这是「未去重」的行数上界；真实行数还要乘上 ② 里的比例，")
    print("          并随 expert_alignment 向上取整（口径 B，第 38 章 §2.2）。")
    print(f"gate/up   2 * {rows} * {2 * args.ffn} * {args.hidden} = "
          f"{flops_up:.3e} FLOP")
    print(f"down      2 * {rows} * {args.hidden} * {args.ffn} = "
          f"{flops_down:.3e} FLOP")
    print(f"合计      {flops_total:.3e} FLOP = {flops_total / 1e12:.2f} TFLOP")
    print()
    print(f"权重字节  B(gate/up) {gb(bytes_b_up)} + B(down) {gb(bytes_b_down)}")
    print(f"输入字节  A {gb(bytes_a)}（每行按 gate/up 读一次）")
    print()
    print(f"算术强度  {flops_total / bytes_total:.1f} FLOP/byte（本书复算，只看这一层）")
    print(f"平衡点    fp8: {FP8_TFLOPS * 1e12 / (HBM_BW_GBPS * 1e9):.0f} "
          f"/ bf16: {BF16_TFLOPS * 1e12 / (HBM_BW_GBPS * 1e9):.0f} FLOP/byte")
    print("判类      高于平衡点 → compute-bound；低于 → memory-bound")
    print("          （第 28 章 §2.2；注意 bf16 与 fp8 是两个不同的平衡点）")
    print()


def print_resource_summary() -> None:
    print("=" * 74)
    print("⑤ 三类资源与它们的口径（正文 §4.1）")
    print("=" * 74)
    print(f"{'类别':<10} {'本次引用到的量':<40} {'来源性质'}")
    print(f"{'算力':<10} {'bf16 432.5 / fp8 865 / fp4 1730.1 TFLOPS':<40} 本书复算")
    print(f"{'片内带宽':<8} {f'HBM {HBM_BW_GBPS:.0f} / L2 {L2_BW_GBPS:.0f} GB/s':<40} 源码模型常量")
    print(f"{'卡间带宽':<8} {'dispatch 313-375 / combine 272-347 GB/s':<40} 厂商实测")
    print()
    print(f"唯一的跨类除法：{HBM_BW_GBPS:.0f} / 374 ≈ {HBM_BW_GBPS / 374:.1f}"
          "  → 跨卡搬一次 ≈ 本卡搬十次（量级判类，不可用于预测时间）")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=int, default=DEEPEP_TEST["tokens_per_rank"],
                        dest="tokens_per_rank",
                        help="每 rank 的 token 容量（默认取 DeepEP 测试条件）")
    parser.add_argument("--hidden", type=int, default=DEEPEP_TEST["hidden"])
    parser.add_argument("--topk", type=int, default=DEEPEP_TEST["topk"])
    parser.add_argument("--experts", type=int, default=DEEPEP_TEST["experts"])
    parser.add_argument("--ffn", type=int, default=2048,
                        help="专家的中间维度；默认值只是演示用，请换成你自己的模型参数")
    parser.add_argument("--dispatch-bytes", type=int, default=DEEPEP_TEST["dispatch_bytes"],
                        help="dispatch 每元素字节（FP8=1，BF16=2）")
    parser.add_argument("--combine-bytes", type=int, default=DEEPEP_TEST["combine_bytes"])
    args = parser.parse_args()

    print("第 39 章 · 训练一步的账本（公式复算，无实测）")
    print("每一项都标注来源；不能跨来源相减 / 相乘（正文 §4.4）")
    print()
    print_comm_ledger(args)
    print_gemm_ledger(args)
    print_resource_summary()


if __name__ == "__main__":
    main()
