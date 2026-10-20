#!/usr/bin/env python3
"""第 18 章 · KV cache 显存账本与 paged KV 的块账（纯标准库，CPU 可跑）。

这份脚本做四件事：

1. 用 Qwen2.5-7B 的 GQA 配置（28 Q heads / 4 KV heads）算 KV cache 的
   「每 token / 每层 / 每条序列」字节数；
2. 和 MHA（28 KV heads）对照，量化 GQA 省了多少；
3. 把 paged KV 的块账算出来：块大小、每块字节、碎片上界；
4. 算「显存里还能放几条序列」——也就是 concurrency 与 context 的取舍。

口径
----

* KV cache 每个 token 每层缓存 K 和 V 两份，形状都是 ``[n_kv_heads, head_dim]``。
* 每元素字节数取模型 dtype（Qwen2.5-7B 是 bfloat16，2 字节）。
* 这是**账本模型**：不含分配器碎片、对齐 padding、框架的额外元数据
  （block table、序列长度数组等）。真实占用一定比它大。

用法
----

    python3 kv_ledger.py
    python3 kv_ledger.py --json
"""

from __future__ import annotations

import argparse
import json

#: Qwen2.5-7B 的推理配置。前四项来自模型卡（`hf-qwen2-5-7b-card`）；
#: hidden_size / head_dim / intermediate_size 由 `01` 的参数量互校得出，
#: 与 6.53B（非嵌入）和 7.61B（总计）两个卡片数字一致，误差 < 0.07%。
QWEN25_7B = {
    "n_layers": 28,
    "n_q_heads": 28,
    "n_kv_heads": 4,
    "context": 131072,
    "hidden_size": 3584,
    "head_dim": 128,
    "intermediate_size": 18944,
    "vocab_size": 152064,
    "dtype_bytes": 2,          # bfloat16
}

#: H100 SXM5 的显存容量（公开规格 80 GB）。只用于「还能放几条序列」的估算。
H100_MEMORY_BYTES = 80 * (1024 ** 3)


def kv_bytes_per_token_per_layer(n_kv_heads: int, head_dim: int,
                                 dtype_bytes: int) -> int:
    """K 与 V 各一份：``2 * H_kv * d * b``。"""
    return 2 * n_kv_heads * head_dim * dtype_bytes


def kv_bytes_per_token(cfg: dict[str, int], n_kv_heads: int | None = None) -> int:
    h_kv = cfg["n_kv_heads"] if n_kv_heads is None else n_kv_heads
    return kv_bytes_per_token_per_layer(
        h_kv, cfg["head_dim"], cfg["dtype_bytes"]) * cfg["n_layers"]


def fmt_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PiB"


def decode_ai(n_q_heads: int, n_kv_heads: int, dtype_bytes: int) -> float:
    """decode attention 的算术强度（第 16 章）: ``2*H_q / (H_kv*b)``。"""
    return 2.0 * n_q_heads / (n_kv_heads * dtype_bytes)


def print_per_token() -> None:
    cfg = QWEN25_7B
    per_layer = kv_bytes_per_token_per_layer(
        cfg["n_kv_heads"], cfg["head_dim"], cfg["dtype_bytes"])
    per_token = kv_bytes_per_token(cfg)
    print("=" * 78)
    print("KV cache 的单位成本（Qwen2.5-7B, bfloat16）")
    print("=" * 78)
    print(f"  形状：K 和 V 各 [n_kv_heads, head_dim] = "
          f"[{cfg['n_kv_heads']}, {cfg['head_dim']}]")
    print(f"  每 token 每层：2 * {cfg['n_kv_heads']} * {cfg['head_dim']} * "
          f"{cfg['dtype_bytes']} = {per_layer} B = {per_layer / 1024:.1f} KiB")
    print(f"  每 token 全模型：{per_layer} B * {cfg['n_layers']} 层 = "
          f"{per_token} B = {per_token / 1024:.1f} KiB")


def print_context_table() -> None:
    cfg = QWEN25_7B
    per_token = kv_bytes_per_token(cfg)
    per_token_mha = kv_bytes_per_token(cfg, n_kv_heads=cfg["n_q_heads"])
    print("\n" + "=" * 78)
    print("一条序列的 KV cache：上下文越长，账本涨得越快")
    print("=" * 78)
    print(f"{'context':>9} | {'GQA(4 KV heads)':>18} | {'MHA(28 KV heads)':>18} | "
          f"{'GQA 占 MHA':>10}")
    for ctx in (1024, 4096, 8192, 32768, 131072):
        gqa = per_token * ctx
        mha = per_token_mha * ctx
        print(f"{ctx:>9} | {fmt_bytes(gqa):>18} | {fmt_bytes(mha):>18} | "
              f"{gqa / mha:>9.1%}")
    print("\n  GQA 每 token 比 MHA 省 "
          f"{per_token_mha / per_token:.0f} 倍（= n_q_heads / n_kv_heads = "
          f"{cfg['n_q_heads'] // cfg['n_kv_heads']}）。")
    print("  代价：decode 的算术强度也只有 "
          f"{decode_ai(cfg['n_q_heads'], cfg['n_kv_heads'], cfg['dtype_bytes']):.1f} "
          f"FLOP/Byte（MHA 是 "
          f"{decode_ai(cfg['n_q_heads'], cfg['n_q_heads'], cfg['dtype_bytes']):.1f}）。")
    print("  省显存和「每字节能算多少」是同一枚硬币的两面，不能只报一面。")


def print_paged_table() -> None:
    cfg = QWEN25_7B
    per_layer_per_token = kv_bytes_per_token_per_layer(
        cfg["n_kv_heads"], cfg["head_dim"], cfg["dtype_bytes"])
    print("\n" + "=" * 78)
    print("paged KV 的块账")
    print("=" * 78)
    print(f"{'block_size':>10} | {'每层每块':>12} | {'全模型每块':>14} | "
          f"{'块内浪费比例上界':>16}")
    for block in (1, 16, 32, 64, 128):
        per_layer = block * per_layer_per_token
        total = per_layer * cfg["n_layers"]
        waste = (block - 1) / block
        print(f"{block:>10} | {fmt_bytes(per_layer):>12} | {fmt_bytes(total):>14} | "
              f"{waste:>13.1%}")
    print("\n  读法：每条序列的最后一个块最多空 (block_size - 1) 个 token 槽位，"
          "\n  折算成该块的比例就是最后一列。block_size 越大，分配元数据越少、"
          "\n  单条序列可能浪费的绝对量越大。")
    print("  真实框架里 block_table 本身还要占显存，且块必须能被 allocator 整除。")
    print("  连续分配（不分页）的碎片上界是「整条序列跑不满预留长度」，"
          "\n  在长上下文 + 变长输出下这个浪费会远大于块内碎片。")


def print_concurrency() -> None:
    cfg = QWEN25_7B
    per_token = kv_bytes_per_token(cfg)
    weights = 7.61e9 * cfg["dtype_bytes"]
    print("\n" + "=" * 78)
    print("concurrency 与 context 的取舍（H100 80 GB，粗略）")
    print("=" * 78)
    print(f"  权重占用（bf16，按模型卡 7.61B 参数）: {fmt_bytes(weights)}")
    budget = H100_MEMORY_BYTES * 0.90 - weights  # 留 10% 给运行时/激活
    print(f"  留 10% 给运行时与激活后，KV 预算     : {fmt_bytes(budget)}")
    print(f"\n{'context':>9} | {'每条序列 KV':>14} | {'可并发序列数':>12} | "
          f"{'理论 token/s 上限':>18}")
    for ctx in (1024, 4096, 8192, 32768, 131072):
        per_seq = per_token * ctx
        n = int(budget // per_seq)
        print(f"{ctx:>9} | {fmt_bytes(per_seq):>14} | {n:>12} | "
              f"{'—':>18}")
    print("\n  读法：并发数乘上「单步权重成本」就是吞吐上限的来源。"
          "\n  context 翻倍，并发数减半；这就是长上下文服务要专门做"
          "\n  KV 量化、prefix caching、KV 换出的原因。")
    print("  最后两列留空：理论 token/s 还要乘上单步时间，见 `03-decode-step-share`。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="机器可读输出")
    args = parser.parse_args()

    cfg = QWEN25_7B
    if args.json:
        payload = {
            "bytes_per_token_per_layer": kv_bytes_per_token_per_layer(
                cfg["n_kv_heads"], cfg["head_dim"], cfg["dtype_bytes"]),
            "bytes_per_token": kv_bytes_per_token(cfg),
            "bytes_per_token_mha": kv_bytes_per_token(cfg, cfg["n_q_heads"]),
            "decode_ai_gqa": decode_ai(cfg["n_q_heads"], cfg["n_kv_heads"],
                                       cfg["dtype_bytes"]),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print_per_token()
    print_context_table()
    print_paged_table()
    print_concurrency()
    print("\n提醒：以上是**账本模型**，不含分配器碎片、对齐 padding 与框架元数据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
