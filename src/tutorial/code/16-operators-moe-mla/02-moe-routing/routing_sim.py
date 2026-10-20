#!/usr/bin/env python3
"""第 16 章 · MoE 路由与重排的可逆性验证（纯标准库，CPU 可跑）。

这一节把 MoE 一层的「非矩阵乘部分」走一遍完整的数据流，并证明三件事：

1. **重排是可逆的**。token 按专家重排（permute）→ 专家计算 → 按原位置聚合
   （combine），结果等价于「对每个 token 直接调用它选中的那几个专家再按
   gating 权重求和」。这是 `scatter_reduce` / `unpermute` 能成立的前提。
2. **分块表（block metadata）怎么来的**。官方 grouped GEMM / fusedmoe 用
   `batch_sizes` / `batch_offsets` / `batch_padded_offsets`（或等价的
   `block_metadata[bx] = [expert, m_start, valid_m]`）把「每个专家接到的
   token 数不齐」翻译成一个规整的 grid。这里把它复现出来。
3. **不均衡直接变成浪费**。padding 浪费与「最热专家 / 平均专家」的比值，
   都是可以提前算出来的量；它们决定了 grouped GEMM 的实际有效算力。

记账口径与 `../01-cost-ledger/decode_step_ledger.py` 一致。

用法
----

    python3 routing_sim.py
    python3 routing_sim.py --tokens 96 --experts 8 --top-k 2 --seed 7
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# 1. gating：softmax + top-k
# ---------------------------------------------------------------------------


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    s = sum(es)
    return [e / s for e in es]


def select_top_k(scores: list[float], k: int) -> list[tuple[int, float]]:
    """返回 ``[(expert_id, score), ...]``，按分数从大到小、前 k 个。

    ``-idx`` 作为次级键，保证同分时顺序稳定（可复现）。
    """
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return [(i, scores[i]) for i in order[:k]]


# ---------------------------------------------------------------------------
# 2. 重排：assignments → 按专家分组
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteTables:
    """一层 MoE 的重排结果，对应官方 example 里的几张 int32 表。"""

    expert_of: list[int]        # assignments 顺序：第 a 个 (token,expert) 对属于哪个专家
    token_of: list[int]         # assignments 顺序：它来自哪个 token
    weight_of: list[float]      # assignments 顺序：它的 gating 权重
    group_sizes: list[int]      # 每个专家的 token 数（= bincount）
    group_offsets: list[int]    # 每个专家在**紧凑**（未 padding）布局里的起点
    padded_offsets: list[int]   # 每个专家在**padding 对齐**布局里的起点
    tokens_of_expert: list[list[int]]  # 每个专家实际拿到的 token id 列表


def build_route_tables(
    topk: list[list[tuple[int, float]]],
    n_experts: int,
    block_m: int,
) -> RouteTables:
    """复现官方 example 的 ``argsort → bincount → cumsum → padded_offsets`` 流程。"""
    # 官方 fusedmoe：flat_indices.argsort() 把 (token,expert) 对按 expert 排序。
    flat = [(tok, e, w) for tok, row in enumerate(topk) for e, w in row]
    order = sorted(range(len(flat)), key=lambda a: (flat[a][1], flat[a][0]))
    expert_of = [flat[a][1] for a in order]
    token_of = [flat[a][0] for a in order]
    weight_of = [flat[a][2] for a in order]

    group_sizes = [expert_of.count(e) for e in range(n_experts)]
    group_offsets: list[int] = []
    acc = 0
    for s in group_sizes:
        group_offsets.append(acc)
        acc += s

    # padding 对齐：每个专家占用 ceil(size / block_m) * block_m 行。
    padded_offsets: list[int] = []
    acc = 0
    for s in group_sizes:
        padded_offsets.append(acc)
        acc += math.ceil(s / block_m) * block_m

    tokens_of_expert: list[list[int]] = [[] for _ in range(n_experts)]
    for a in range(len(expert_of)):
        tokens_of_expert[expert_of[a]].append(token_of[a])

    return RouteTables(
        expert_of=expert_of,
        token_of=token_of,
        weight_of=weight_of,
        group_sizes=group_sizes,
        group_offsets=group_offsets,
        padded_offsets=padded_offsets,
        tokens_of_expert=tokens_of_expert,
    )


def block_metadata(tables: RouteTables, block_m: int) -> list[tuple[int, int, int]]:
    """生成 ``[expert_id, m_start, valid_m]`` 列表（昇腾 fusedmoe 的 block_metadata）。

    对应 `_vendor/tilelang-ascend-v0.1.2.000/examples/fusedmoe/example_fusedmoe.py`
    里的 ``_build_block_metadata``。
    """
    rows: list[tuple[int, int, int]] = []
    for expert, size in enumerate(tables.group_sizes):
        if size == 0:
            continue
        base = tables.padded_offsets[expert]
        for i in range(math.ceil(size / block_m)):
            m_start = base + i * block_m
            valid_m = min(block_m, size - i * block_m)
            rows.append((expert, m_start, valid_m))
    return rows


# ---------------------------------------------------------------------------
# 3. 专家计算 —— 用一个便宜的、可精确复算的线性变换代替真 GEMM
# ---------------------------------------------------------------------------


def expert_forward(x: list[float], expert: int) -> list[float]:
    """一个「专家」：``y_j = (1 + 0.5*expert) * x_j + 0.01*expert``。

    它刻意是**线性且确定**的，这样 permute/combine 的等价性可以精确断言；
    它不代表真实 MLP 的数值行为，只用来验证数据流。
    """
    scale = 1.0 + 0.5 * expert
    bias = 0.01 * expert
    return [scale * v + bias for v in x]


def reference(xs: list[list[float]], topk: list[list[tuple[int, float]]]) -> list[list[float]]:
    """参考实现：不重排，直接对每个 token 的每个选中专家算完再按权重求和。"""
    out = []
    for x, row in zip(xs, topk):
        acc = [0.0] * len(x)
        for e, w in row:
            y = expert_forward(x, e)
            acc = [a + w * v for a, v in zip(acc, y)]
        out.append(acc)
    return out


def permuted_path(xs: list[list[float]], tables: RouteTables) -> list[list[float]]:
    """重排路径：permute → 逐 assignment 算专家 → combine（scatter-reduce）。"""
    width = len(xs[0])
    # permute：按 assignments 顺序把 token 行搬到「按专家分组」的缓冲里
    permuted = [list(xs[t]) for t in tables.token_of]
    # 专家计算
    computed = [expert_forward(permuted[a], tables.expert_of[a]) for a in range(len(permuted))]
    # combine：按 token 累加（这里显式做加法，模拟 scatter_reduce 的 reduce="sum"）
    acc = [[0.0] * width for _ in range(len(xs))]
    for a in range(len(computed)):
        t = tables.token_of[a]
        w = tables.weight_of[a]
        acc[t] = [a_ + w * c for a_, c in zip(acc[t], computed[a])]
    return acc


# ---------------------------------------------------------------------------
# 4. 主流程与报告
# ---------------------------------------------------------------------------


def run(tokens: int, experts: int, top_k: int, width: int, block_m: int, seed: int) -> int:
    rng = random.Random(seed)
    xs = [[rng.uniform(-1.0, 1.0) for _ in range(width)] for _ in range(tokens)]
    # 让分布不均匀：给前 2 个专家一个偏置，模拟热点专家。
    logits = [
        [rng.gauss(0.0, 1.0) + (2.0 if e < 2 else 0.0) for e in range(experts)]
        for _ in range(tokens)
    ]
    probs = [softmax(row) for row in logits]
    topk = [select_top_k(row, top_k) for row in probs]

    tables = build_route_tables(topk, experts, block_m)
    meta = block_metadata(tables, block_m)

    ref = reference(xs, topk)
    got = permuted_path(xs, tables)
    max_err = max(abs(a - b) for ra, rb in zip(ref, got) for a, b in zip(ra, rb))

    total_assignments = sum(tables.group_sizes)
    padded_rows = len(meta) * block_m
    hot = max(tables.group_sizes)
    mean = total_assignments / experts

    print(f"tokens={tokens} experts={experts} top_k={top_k} "
          f"width={width} block_M={block_m} seed={seed}")
    print(f"assignments (token,expert) 对 : {total_assignments}")
    print(f"每个专家的 token 数           : {tables.group_sizes}")
    print(f"最热专家 / 平均专家           : {hot} / {mean:.2f} "
          f"= {hot / mean:.2f}x  （不均衡）")
    print(f"padding 后总行数              : {padded_rows} "
          f"（浪费 {(padded_rows - total_assignments) / max(total_assignments, 1):.1%}）")
    print(f"grid 的 M 方向块数            : {len(meta)} "
          f"（= 各专家 ceil(size/{block_m}) 之和）")
    print(f"重排 vs 参考的最大误差        : {max_err:.3e}")

    ok = max_err < 1e-12
    print("round-trip 断言              : " + ("通过" if ok else "失败"))
    if not ok:
        raise SystemExit("permute/combine 不等价——实现有 bug")

    print("\nblock_metadata 前若干行（[expert, m_start, valid_m]）：")
    for row in meta[:8]:
        print(f"  {list(row)}")
    if len(meta) > 8:
        print(f"  ... 共 {len(meta)} 行")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--seed", type=int, default=81394)
    args = parser.parse_args()
    if args.top_k > args.experts:
        parser.error("--top-k 不能超过 --experts")
    return run(args.tokens, args.experts, args.top_k, args.width, args.block_m, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
