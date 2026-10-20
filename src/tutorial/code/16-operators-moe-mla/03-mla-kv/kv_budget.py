#!/usr/bin/env python3
"""第 16 章 · MLA 的 KV 预算与「对 kernel 友好/不友好」量化（纯标准库，CPU 可跑）。

三件事：

A. **KV 预算**：MHA / GQA / MLA 每 token、每层、每条序列、整个模型要缓存多少字节。
B. **decode 的算术强度**：``AI = 2*H_q / (H_kv*b)``（MHA/GQA）与
   ``AI = 2*H_q*(2*d_c+d_r) / (b*(d_c+d_r))``（MLA）。前者与序列长度无关，
   后者因为「所有 query head 共享一份潜向量」而高出两个数量级。
C. **为什么不友好**：MLA 的 head 维度是 576/512，``acc_o`` 直接吃掉大量寄存器；
   官方文档给出的解法是按 ``dim`` 把 ``acc_o`` 切开（``policy=FullCol``），
   让两个 warpgroup 各算一半。这里把寄存器账算出来。

数字来源分两类：**本书推导**（公式可复算）与**锁定版本里的第一方配置**
（``_vendor/tilelang-v0.1.15/examples/deepseek_v32/inference/config_671B_v3.2.json``）。

用法
----

    python3 kv_budget.py
    python3 kv_budget.py --seq 65536 --layers 61
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

V32_CONFIG_RELATIVE = (
    "../../../../_vendor/tilelang-v0.1.15/examples/"
    "deepseek_v32/inference/config_671B_v3.2.json"
)


def load_v32_config() -> dict[str, object]:
    path = Path(__file__).resolve().parent / V32_CONFIG_RELATIVE
    if path.is_file():
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "dim": 7168,
        "n_layers": 61,
        "n_heads": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 2048,
        "source": "builtin-fallback",
    }


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


# ---------------------------------------------------------------------------
# A. KV 预算
# ---------------------------------------------------------------------------


def part_a(layers: int, heads: int, head_dim: int, dc: int, dr: int) -> None:
    print("=" * 74)
    print("A. KV 预算（fp16，每个元素 2 字节）")
    print("=" * 74)
    rows = [
        ("MHA  128 heads x 128 dim", 2 * 128 * head_dim * 2),
        ("GQA  8 kv-heads x 128 dim", 2 * 8 * head_dim * 2),
        ("GQA  1 kv-head  x 128 dim", 2 * 1 * head_dim * 2),
        ("MLA  c_KV(512) + k_R(64)", (dc + dr) * 2),
    ]
    print(f"\n{'变体':<28} | {'字节/token/层':>14} | {'字节/token/整模型':>16}")
    for name, per_token_layer in rows:
        print(f"{name:<28} | {per_token_layer:>14} | "
              f"{fmt_bytes(per_token_layer * layers):>16}")
    mha = 2 * 128 * head_dim * 2
    mla = (dc + dr) * 2
    print(f"\nMLA / MHA = {mla / mha:.2%}：每缓存 1 份潜向量，等于省下 "
          f"{(1 - mla / mha):.2%} 的 KV 字节。")
    print("这个比值只由结构决定（潜向量维度 vs 2*H_kv*d），与序列长度、batch 无关。")


# ---------------------------------------------------------------------------
# B. decode 的算术强度
# ---------------------------------------------------------------------------


def mha_like_decode_ai(heads_q: int, heads_kv: int, b: int) -> float:
    return 2.0 * heads_q / (heads_kv * b)


def mla_decode_ai(heads_q: int, dc: int, dr: int, b: int) -> float:
    return 2.0 * heads_q * (2 * dc + dr) / (b * (dc + dr))


def part_b(heads: int, dc: int, dr: int, b: int = 2) -> None:
    print("\n" + "=" * 74)
    print("B. decode 的算术强度（AI = FLOP / Byte）")
    print("=" * 74)
    print("\nMHA / GQA：AI = 2*H_q / (H_kv*b)，与序列长度无关")
    for hkv in (128, 32, 8, 4, 2, 1):
        print(f"  H_q={heads:>3}, H_kv={hkv:>3} : "
              f"{mha_like_decode_ai(heads, hkv, b):>8.1f} FLOP/Byte")
    ai = mla_decode_ai(heads, dc, dr, b)
    print(f"\nMLA：AI = 2*H_q*(2*d_c+d_r) / (b*(d_c+d_r)) = {ai:.1f} FLOP/Byte")

    # 让 MLA decode 跨过机器平衡点需要多少 query head。
    balance = 1000.0e12 / 3.0e12  # H100 SXM5 公开规格
    need_heads = balance * b * (dc + dr) / (2 * (2 * dc + dr))
    print(f"\nH100 公开规格的平衡点约 {balance:.0f} FLOP/Byte；"
          f"MLA decode 要在 H_q ≈ {need_heads:.0f} 才跨过去。")
    print("DeepSeek-V3.2 的 n_heads=128，恰好落在平衡点之下一点——"
          "\n账面判 memory-bound；但离脊很近，所以真实实现是不是吃满张量核"
          "\n还要看 softmax 和 shared memory 流量（见官方例子的说明）。")


# ---------------------------------------------------------------------------
# C. 为什么不友好：寄存器与片上容量
# ---------------------------------------------------------------------------


def part_c(block_h: int, block_n: int, dc: int, dr: int, threads: int,
           stages: int, b: int) -> None:
    print("\n" + "=" * 74)
    print("C. MLA 对 kernel 不友好在哪：寄存器与片上容量")
    print("=" * 74)
    acc_o = block_h * dc
    acc_s = block_h * block_n
    print(f"\nblock_H={block_h} block_N={block_n} dim={dc} "
          f"pe_dim={dr} threads={threads} stages={stages}")
    print(f"  累加器 acc_o = block_H x dim          = {acc_o:>8} 个 fp32")
    print(f"  累加器 acc_s = block_H x block_N      = {acc_s:>8} 个 fp32")
    print(f"  acc_o 折成寄存器/线程                 = "
          f"{acc_o / threads:>8.0f} 个 32 位寄存器")
    print(f"  acc_s 折成寄存器/线程                 = "
          f"{acc_s / threads:>8.0f} 个 32 位寄存器")
    total = (acc_o + acc_s) / threads
    print(f"  两者相加                             = {total:>8.0f} 个（不含 Q/K/V 的 fragment）")
    print("\n  CUDA 常规每线程上限约 255 个寄存器 → "
          f"{'超限，需要切开' if total > 255 else '勉强放得下'}")

    # 按官方 example 里出现的 buffer 逐项相加（Q/Q_pe 只一份，KV/K_pe 乘流水级数）。
    q_smem = block_h * dc * b
    qpe_smem = block_h * dr * b
    s_smem = block_h * block_n * b
    kv_smem = block_n * dc * b * stages
    kpe_smem = block_n * dr * b * stages
    o_smem = block_h * dc * b
    naive = q_smem + qpe_smem + s_smem + kv_smem + kpe_smem + o_smem
    print("\n  片上（shared memory）按官方 example 的 buffer 逐项相加：")
    print(f"    Q      {block_h}x{dc:<4}               = {fmt_bytes(q_smem)}")
    print(f"    Q_pe   {block_h}x{dr:<4}               = {fmt_bytes(qpe_smem)}")
    print(f"    S      {block_h}x{block_n:<4}               = {fmt_bytes(s_smem)}")
    print(f"    KV     {block_n}x{dc:<4} x {stages}       = {fmt_bytes(kv_smem)}")
    print(f"    K_pe   {block_n}x{dr:<4} x {stages}       = {fmt_bytes(kpe_smem)}")
    print(f"    O      {block_h}x{dc:<4}               = {fmt_bytes(o_smem)}")
    print(f"    简单相加                              = {fmt_bytes(naive)}")
    print("    注意：这是「生命周期简单相加」的上界。编译器会做 shared memory")
    print("    规划与复用（第 22 章的 storage rewrite），真实占用比它小；")
    print("    H100 每 SM 可配 228 KB、MI300X 约 64 KB —— 这个上界说明为什么")
    print("    block_H/block_N/num_stages 在这类 kernel 上必须先算容量、再调参。")
    print("    官方文档给 MI300X 的做法正是：把 Q 从 shared 搬到寄存器 fragment，")
    print("    并降低流水级数（见 references.md 的 R2）。")

    print("\n  这就是官方文档说的：")
    print("    * wgmma.mma_async 要求 M >= 64，不能靠缩小 block_H 来减 acc_o；")
    print("    * 只能按 dim 把 acc_o 切成两半，两个 warpgroup 各算一半")
    print("      （T.gemm(..., policy=T.GemmWarpPolicy.FullCol)）；")
    print("    * 代价是两个 warpgroup 都需要完整的 acc_s，于是要走一次 shared。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=61)
    parser.add_argument("--block-h", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--stages", type=int, default=2)
    args = parser.parse_args()

    cfg = load_v32_config()
    heads = int(cfg["n_heads"])
    dc = int(cfg["kv_lora_rank"])
    dr = int(cfg["qk_rope_head_dim"])
    head_dim = int(cfg["qk_nope_head_dim"])
    print(f"结构参数来源：config_671B_v3.2.json（{cfg.get('source', 'vendor-config')}）")
    print(f"n_layers={cfg['n_layers']} n_heads={heads} kv_lora_rank={dc} "
          f"qk_rope_head_dim={dr} qk_nope_head_dim={head_dim}")

    part_a(int(cfg["n_layers"]), heads, head_dim, dc, dr)
    part_b(heads, dc, dr)
    part_c(args.block_h, args.block_n, dc, dr, args.threads, args.stages, 2)

    total = (dc + dr) * 2 * int(cfg["n_layers"]) * args.seq
    print(f"\n参考：序列长度 {args.seq} 时，MLA 潜 KV 共 {fmt_bytes(total)}"
          "（单条序列，fp16）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
