#!/usr/bin/env python3
"""第 38 章 · 示例 02：一层 MoE 的 dispatch / combine 字节账。

这是第 16 章那套账本在 EP 场景下的直接延伸。口径（全部来自源码）：

* 每 token 的 **dispatch payload**（`tests/ep/test_ep.py:404-406` 的
  ``count_bytes(x, topk_idx, topk_weights) / num_tokens``）= hidden 元素
  的字节 + SF 的字节 + ``topk * 8``（int64 索引）+ ``topk * 4``（float32 权重）。
* 每 token 的 **combine payload** = BF16 hidden 的字节 + ``topk * 4``
  （combine 的输入是 bf16，`csrc/kernels/ep/combine.hpp` 的调用约束）。
* 一个 token 覆盖的**不同目标 rank 数**期望值是 ``R·(1-(1-1/R)^k)``
  （R 个 rank、均匀路由、忽略放回；见正文 §1.3）。
* 每 rank 的接收 token 数 ≈ ``T × E[不同目标 rank 数]``（按对称性）。

作者验证级别：**CPU 逻辑验证**。本脚本只用标准库，不 import torch，
不连接任何设备。**这不是实测数字，是本书复算。**

用法::

    python3 comm_ledger.py
    python3 comm_ledger.py --layers 61 --json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 7168
    num_topk: int = 6
    num_experts: int = 256
    tokens_per_rank: int = 16384


@dataclass(frozen=True)
class ByteConfig:
    """每 token 的 payload 口径（与 DeepEP-Ascend 测试代码一致）。"""

    hidden_bytes_fp8: int
    sf_bytes: int
    indices_bytes: int
    weights_bytes: int
    hidden_bytes_bf16: int

    @property
    def dispatch(self) -> int:
        return (
            self.hidden_bytes_fp8 + self.sf_bytes + self.indices_bytes + self.weights_bytes
        )

    @property
    def combine(self) -> int:
        return self.hidden_bytes_bf16 + self.weights_bytes


def payload_bytes(model: ModelConfig, fp8_dispatch: bool = True) -> ByteConfig:
    """按源码口径算每 token 的 payload 字节数。"""
    # per_token_cast_to_fp8 用 128 元素一个 fp32 scale（tests/ep/test_ep.py:89-93）。
    num_sf = (model.hidden + 127) // 128
    sf_bytes = num_sf * 4  # fp32；view(torch.int16) 后字节数不变，变成 num_sf*2 个 int16
    return ByteConfig(
        hidden_bytes_fp8=model.hidden if fp8_dispatch else model.hidden * 2,
        sf_bytes=sf_bytes if fp8_dispatch else 0,
        indices_bytes=model.num_topk * 8,
        weights_bytes=model.num_topk * 4,
        hidden_bytes_bf16=model.hidden * 2,
    )


def expected_distinct_ranks(num_ranks: int, num_topk: int) -> float:
    """均匀路由下一个 token 覆盖的期望目标 rank 数。"""
    return num_ranks * (1.0 - (1.0 - 1.0 / num_ranks) ** num_topk)


def ledger(model: ModelConfig, num_ranks: int, fp8_dispatch: bool = True):
    by = payload_bytes(model, fp8_dispatch)
    distinct = expected_distinct_ranks(num_ranks, model.num_topk)
    # 每个 rank 收到的 token 数（去重后，按 rank 计）：按对称性 ≈ T × 期望目标 rank 数。
    recv_tokens = model.tokens_per_rank * distinct
    return {
        "num_ranks": num_ranks,
        "experts_per_rank": model.num_experts // num_ranks,
        "expected_distinct_ranks": distinct,
        "recv_tokens_per_rank": recv_tokens,
        "dispatch_bytes_per_rank": recv_tokens * by.dispatch,
        "combine_bytes_per_rank": recv_tokens * by.combine,
        "payload_dispatch": by.dispatch,
        "payload_combine": by.combine,
    }


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:,.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(description="MoE 一层 dispatch / combine 字节账")
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--layers", type=int, default=1, help="按多少层汇总（V3 量级可填 61）")
    parser.add_argument("--bf16-dispatch", action="store_true", help="dispatch 也用 BF16")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    model = ModelConfig(
        hidden=args.hidden,
        num_topk=args.topk,
        num_experts=args.experts,
        tokens_per_rank=args.tokens,
    )
    by = payload_bytes(model, not args.bf16_dispatch)

    rows = [ledger(model, r, not args.bf16_dispatch) for r in (8, 16, 32, 64, 128)]

    if args.json:
        print(json.dumps({"config": model.__dict__, "payload": by.__dict__, "rows": rows}, indent=2))
        return 0

    print("每 token payload（本书复算，口径 = DeepEP-Ascend/tests/ep/test_ep.py:404-406）")
    print(f"  hidden={model.hidden}, topk={model.num_topk}, experts={model.num_experts}, "
          f"T={model.tokens_per_rank}, layers={args.layers}")
    print(f"  dispatch: {by.dispatch:,} B/token "
          f"(hidden {by.hidden_bytes_fp8:,} + sf {by.sf_bytes} + idx {by.indices_bytes} + w {by.weights_bytes})")
    print(f"  combine : {by.combine:,} B/token (bf16 hidden {by.hidden_bytes_bf16:,} + w {by.weights_bytes})")
    print()
    print(f"{'EP':>4} {'专家/rank':>9} {'期望目标rank':>12} {'接收token/rank':>15} "
          f"{'dispatch/rank':>14} {'combine/rank':>14}")
    for row in rows:
        print(
            f"{row['num_ranks']:>4} {row['experts_per_rank']:>9} "
            f"{row['expected_distinct_ranks']:>12.3f} {row['recv_tokens_per_rank']:>15,.0f} "
            f"{human(row['dispatch_bytes_per_rank']):>14} {human(row['combine_bytes_per_rank']):>14}"
        )
    print()
    total = sum(r["dispatch_bytes_per_rank"] + r["combine_bytes_per_rank"] for r in rows) / len(rows)
    print(f"（上表按 EP8..EP128 各一行；若按 {args.layers} 层汇总，"
          f"平均每 rank 的通信量约为 {human(total * args.layers)}，跨 rank 合计约为 "
          f"{human(total * args.layers * len(rows))}。）")
    print()
    print("注意：这是均匀路由假设下的**上界量级**估计，不是实测流量；")
    print("真实值取决于路由分布、去重率与每 rank 的本地专家数（见正文 §1.3）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
