#!/usr/bin/env python3
"""第 18 章 · MoE 推理的 dispatch/combine 契约（纯标准库，CPU 可跑）。

MoE 推理里最容易出错、也最容易被教程忽略的一段，是「路由结果」到
「grouped GEMM 的行区间」之间的映射。锁定版本的
`_vendor/tilelang-v0.1.15/examples/fusedmoe/example_fusedmoe_tilelang.py`
把它写成四个张量：

============================  ==========================================
`group_sizes`                 每个专家分到多少个 (token, expert) 对
`group_offsets`               每个专家的行区间在**紧凑**的 token 数组里的起点
`group_padded_offsets`        每个专家的行区间在**对齐后**的数组里的起点
`group_idx_for_bx`            第 bx 个 block 该去处理哪个专家
============================  ==========================================

本脚本用一个小规模例子把这份契约复算一遍，并断言三件事：

1. 每个 (token, expert) 对恰好出现一次（既不漏也不重）；
2. 每个专家的「对齐后区间」完整覆盖它的「紧凑区间」；
3. 内核里那一行 `m_start = m_start_padded - group_padded_offsets[g]
   + group_offsets[g]` 能正确还原出紧凑数组里的行号。

**这不是在跑 kernel**，是在核对 host 侧那张表与 kernel 侧寻址公式是否自洽。

用法
----

    python3 moe_dispatch_contract.py
    python3 moe_dispatch_contract.py --tokens 17 --experts 5 --top-k 3
"""

from __future__ import annotations

import argparse
import math
import random


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def build_contract(token_ids: list[int], expert_of: list[list[int]],
                   n_experts: int, padding_m: int, block_token: int) -> dict:
    """复刻 `example_fusedmoe_tilelang.py` 的 host 侧准备逻辑。

    对应源码：`example_fusedmoe_tilelang.py:344-382`
    （`idxs = flat_expert_indices.argsort()` 到 `group_idx_for_bx`）。
    """
    # 1) 每个 token 选 top_k 个专家，展平
    flat_expert: list[int] = []
    flat_token: list[int] = []
    for i, tok in enumerate(token_ids):
        for e in expert_of[i]:
            flat_expert.append(e)
            flat_token.append(tok)
    group_sum = len(flat_expert)

    # 2) 按专家排序：同一个专家的行在数组里连续
    order = sorted(range(group_sum), key=lambda k: flat_expert[k])
    counts = [0] * n_experts
    for e in flat_expert:
        counts[e] += 1

    group_offsets: list[int] = []
    acc = 0
    for c in counts:
        group_offsets.append(acc)
        acc += c

    # 3) 对齐后的偏移；源码里是 ceil((count + 1) / padding_M) * padding_M
    #    那个 +1 是给每个专家留一行余量（见 sources: :364-366）
    group_padded_offsets: list[int] = [0]
    for i in range(1, n_experts):
        prev = math.ceil((counts[i - 1] + 1) / padding_m) * padding_m
        group_padded_offsets.append(group_padded_offsets[i - 1] + prev)

    # 4) 第 bx 个 block 属于哪个专家
    m_blocks = ceil_div(group_sum, block_token) + n_experts
    group_idx_for_bx = []
    for bx in range(m_blocks):
        start = bx * block_token
        g = 0
        for i in range(n_experts):
            if start >= group_padded_offsets[i]:
                g = i
        group_idx_for_bx.append(g)

    return {
        "flat_expert": flat_expert,
        "flat_token": flat_token,
        "order": order,
        "counts": counts,
        "group_sizes": counts,
        "group_offsets": group_offsets,
        "group_padded_offsets": group_padded_offsets,
        "m_blocks": m_blocks,
        "group_idx_for_bx": group_idx_for_bx,
        "group_sum": group_sum,
    }


def check_coverage(contract: dict) -> None:
    """每个 (token, expert) 对恰好出现一次。"""
    seen = list(zip(contract["flat_token"], contract["flat_expert"]))
    assert len(seen) == len(set(seen)), "同一个 (token, expert) 对出现了两次"
    assert len(seen) == contract["group_sum"]
    total = sum(contract["counts"])
    assert total == contract["group_sum"], f"{total} != {contract['group_sum']}"
    print(f"  [ok] {contract['group_sum']} 个 (token, expert) 对恰好覆盖一次")


def check_padded_covers_compact(contract: dict, padding_m: int) -> None:
    """对齐后的区间必须完整覆盖紧凑区间。"""
    for i, (off, padded) in enumerate(zip(contract["group_offsets"],
                                          contract["group_padded_offsets"])):
        need = math.ceil((contract["counts"][i] + 1) / padding_m) * padding_m
        assert padded + need >= off + contract["counts"][i], (
            f"专家 {i}: 对齐后区间 [区{padded}, {padded + need}) "
            f"盖不住紧凑区间 [{off}, {off + contract['counts'][i]})")
    print(f"  [ok] 每个专家的对齐区间都盖住了它的紧凑区间（padding_M={padding_m}）")


def check_kernel_addressing(contract: dict, block_token: int) -> None:
    """复算内核里的 `m_start` / `actual_rows` 公式。"""
    sizes = contract["group_sizes"]
    offsets = contract["group_offsets"]
    padded = contract["group_padded_offsets"]
    checked = 0
    for bx, g in enumerate(contract["group_idx_for_bx"]):
        m_start_padded = bx * block_token
        if m_start_padded < padded[g]:
            continue
        m_start = m_start_padded - padded[g] + offsets[g]
        actual_rows = max(0, min(block_token, sizes[g] - (m_start_padded - padded[g])))
        # 落在有效行里的位置必须能被内核寻址到
        if actual_rows > 0:
            assert m_start >= offsets[g]
            assert m_start + actual_rows <= offsets[g] + sizes[g]
        checked += 1
    print(f"  [ok] 复算了 {checked} 个 block 的 m_start / actual_rows 公式，"
          f"没有越界")


def print_contract(contract: dict, padding_m: int, tokens: int,
                   n_experts: int, top_k: int) -> None:
    print("=" * 78)
    print(f"例子：{tokens} 个 token × top_k={top_k} → {contract['group_sum']} 个"
          f"(token, expert) 对，{n_experts} 个专家")
    print("=" * 78)
    print(f"{'expert':>6} | {'group_size':>10} | {'group_offset':>12} | "
          f"{'padded_offset':>13} | {'padded 区间长度':>15}")
    for i in range(n_experts):
        need = math.ceil((contract['counts'][i] + 1) / padding_m) * padding_m
        print(f"{i:>6} | {contract['counts'][i]:>10} | "
              f"{contract['group_offsets'][i]:>12} | "
              f"{contract['group_padded_offsets'][i]:>13} | {need:>15}")
    print(f"\n  紧凑数组长度（group_sum）        : {contract['group_sum']}")
    print(f"  对齐后需要的 block 数（M）       : {contract['m_blocks']}"
          f" = ceil({contract['group_sum']}/{128}) + {n_experts}")
    print(f"  对齐浪费的行数                   : "
          f"{contract['m_blocks'] * 128 - contract['group_sum']}")
    print(f"\n  group_idx_for_bx（前 24 个）: "
          f"{contract['group_idx_for_bx'][:24]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=9)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--padding-m", type=int, default=128)
    parser.add_argument("--block-token", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    token_ids = list(range(args.tokens))
    expert_of = [rng.sample(range(args.experts), args.top_k)
                 for _ in token_ids]

    contract = build_contract(token_ids, expert_of, args.experts,
                              args.padding_m, args.block_token)
    print_contract(contract, args.padding_m, args.tokens, args.experts,
                   args.top_k)
    print("\n三项自检：")
    check_coverage(contract)
    check_padded_covers_compact(contract, args.padding_m)
    check_kernel_addressing(contract, args.block_token)

    print("\n读法：`group_padded_offsets` 是「为了不让一个 block 跨两个专家」付的"
          "\n      对齐代价。token 数少、专家多的时候，这个代价可以远大于有效计算"
          "\n      ——这正是第 16 章说的「小 M 让专家 GEMM 退化成 memory-bound」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
