#!/usr/bin/env python3
"""第 16 章 · 稀疏 / 索引类算子的结构对照与代价模型（纯标准库，CPU 可跑）。

两件事：

A. **结构对照**：把这几个同族算子放一张表里——
   NSA（native sparse attention）、DSA / lightning indexer + top-k selector、
   HISA（两级层级稀疏）、DeepSeek-V4 的 swa / hca / csa——
   说清各自的输入、输出的「索引」、以及它省下的是哪部分 attention。
   同时用「文件 + 行号 + token」核对每个引用。
B. **索引不是免费的**：indexer 要在**整条**序列上算相似度，而被它选中的
   attention 只算 top-k 个位置。给出两者的 FLOP 比与盈亏平衡点。

数字来源：结构参数取锁定版本里 DeepSeek-V3.2 的推理配置
（``index_n_heads=64``、``index_head_dim=128``、``index_topk=2048``、
``n_heads=128``、``kv_lora_rank=512``、``qk_rope_head_dim=64``）；公式是本书推导。

用法
----

    python3 sparse_index_map.py
"""

from __future__ import annotations

import json
from pathlib import Path

V32_CONFIG_RELATIVE = (
    "../../../../_vendor/tilelang-v0.1.15/examples/"
    "deepseek_v32/inference/config_671B_v3.2.json"
)

# (文件, 行号, 该行必须包含的 token, 说明)
MAINLINE: list[tuple[str, int, str, str]] = [
    ("examples/deepseek_nsa/example_tilelang_nsa_fwd.py", 50, "BlockIndices:",
     "NSA 的输入是「每个 query 选中哪些 KV block」"),
    ("examples/deepseek_nsa/reference.py", 9, "def naive_nsa",
     "NSA 的 PyTorch 参考实现（含 g_slc / g_swa）"),
    ("examples/deepseek_v32/fp8_lighting_indexer.py", 90, "def mqa_attn_return_logits",
     "lightning indexer：算 logits 的 kernel 入口"),
    ("examples/deepseek_v32/topk_selector.py", 11, "def convert_to_uint16",
     "top-k selector：先把 float 映射成有序整数"),
    ("examples/deepseek_v32/sparse_mla_fwd.py", 120, "Indices[",
     "sparse MLA：按 Indices gather 被选中的 KV 行"),
    ("examples/dsa_hisa/README.md", 10, "two-stage",
     "HISA：先粗选 pool block、再细选 token 的两级索引"),
    ("examples/dsa_hisa/hisa.py", 1, "", "HISA 的端到端编排（四个 kernel + torch.topk）"),
    ("examples/deepseek_v4/sparse_attn_fwd_sm90.py", 1, "",
     "V4 的 sparse attention：按 top-k 索引做块稀疏"),
]


def fmt_flops(n: float) -> str:
    for unit in ("", "K", "M", "G", "T"):
        if abs(n) < 1000.0:
            return f"{n:.3f}{unit}FLOP"
        n /= 1000.0
    return f"{n:.3f}PFLOP"


def check_mainline(book: Path) -> tuple[int, int]:
    base = book / "_vendor" / "tilelang-v0.1.15"
    if not base.is_dir():
        print("[skip] _vendor/tilelang-v0.1.15 不存在")
        return 0, 0
    ok = bad = 0
    for rel, line, token, why in MAINLINE:
        path = base / rel
        if not path.is_file():
            print(f"  MISSING FILE {rel}")
            bad += 1
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if 1 <= line <= len(lines) and token in lines[line - 1]:
            print(f"  ok    {rel}:{line}   {why}")
            ok += 1
        else:
            got = lines[line - 1].strip() if 1 <= line <= len(lines) else "<越界>"
            print(f"  FAIL  {rel}:{line}  期望含 {token!r}，实际 {got!r}")
            bad += 1
    return ok, bad


def print_structure_table() -> None:
    print("=" * 78)
    print("A. 稀疏 / 索引算子结构对照（本书整理，来源见 examples/ 与 references.md）")
    print("=" * 78)
    rows = [
        ("NSA", "deepseek_nsa/",
         "每个 query → 选中的 KV block 列表",
         "三条分支：压缩块 / 选中块 / 滑窗（g_slc + g_swa）"),
        ("DSA 的 indexer", "deepseek_v32/fp8_lighting_indexer.py",
         "每个 query → 对全序列的 logits",
         "MQA 形状：多 query head 共享 1 个 KV head，ReLU + 按头加权"),
        ("DSA 的 top-k", "deepseek_v32/topk_selector.py",
         "logits → top-k 的**位置**（不是值）",
         "8-bit 基数排序两遍：先直方图定阈值 bin，再细化"),
        ("sparse MLA", "deepseek_v32/sparse_mla_fwd.py",
         "位置列表 → gather 出来的 KV 行",
         "把 dense MLA 的 KV 循环换成「按索引 gather」"),
        ("HISA", "dsa_hisa/",
         "query → 先 pool block、再 block 内 token",
         "两级层级：粗选 block 再细选 token，替换掉平面扫描"),
        ("V4 的 csa/hca/swa", "deepseek_v4/sparse_attn_fwd_sm90.py",
         "query → 三套访问模式共用的 paged KV",
         "同一骨架，差别只在「访问哪些 KV block」"),
    ]
    print(f"\n{'算子':<18} {'代码位置':<46}")
    for name, loc, out, note in rows:
        print(f"\n{name}")
        print(f"  代码    : {loc}")
        print(f"  输出    : {out}")
        print(f"  结构    : {note}")


def print_cost_model(cfg: dict[str, object]) -> None:
    l_values = (8192, 32768, 131072)
    i_heads = int(cfg["index_n_heads"])
    i_dim = int(cfg["index_head_dim"])
    topk = int(cfg["index_topk"])
    heads = int(cfg["n_heads"])
    dc = int(cfg["kv_lora_rank"])
    dr = int(cfg["qk_rope_head_dim"])
    attn_per_kv = 2 * heads * (2 * dc + dr)
    idx_per_kv = 2 * i_heads * i_dim

    print("\n" + "=" * 78)
    print("B. indexer 不是免费的：它扫全序列，attention 只算 top-k")
    print("=" * 78)
    print(f"\nindex_n_heads={i_heads} index_head_dim={i_dim} index_topk={topk} "
          f"n_heads={heads} (dc,dr)=({dc},{dr})")
    print(f"  被选中的 KV 位置，每个位置花在 attention 上的 FLOP: "
          f"{attn_per_kv}（= 2*H_q*(2*d_c+d_r)）")
    print(f"  每个 KV 位置花在 indexer 上的 FLOP            : {idx_per_kv}"
          f"（= 2*I*di，MQA 只算一次 KV）")
    print(f"\n{'L':>8} | {'indexer FLOP':>14} | {'attention(topk) FLOP':>22} | 比值")
    for l in l_values:
        idx = idx_per_kv * l
        attn = attn_per_kv * topk
        print(f"{l:>8} | {fmt_flops(idx):>14} | {fmt_flops(attn):>22} | "
              f"{idx / attn:>6.2f}x")
    break_even = (i_heads * i_dim) / (heads * (2 * dc + dr))
    print(f"\n  盈亏平衡：当 topk = (I*di/(H_q*(2*d_c+d_r))) * L "
          f"= L * {break_even:.4f} 时两者相等。")
    print(f"  L={l_values[-1]} 时要选到约 {int(break_even * l_values[-1])} 个位置，"
          f"indexer 才与 attention 打平；")
    print(f"  而配置里的 index_topk={topk} 远小于它——所以在这条序列长度上，")
    print("  省下的 attention 比 indexer 自己花的还少，收益全在「KV 访存」上。")
    print("  （提醒：这是下界模型，未计 softmax、gather、shared memory 流量。）")

    print("\n  换成访存看就清楚了（fp16，每个 query token 要读多少字节）：")
    print(f"  {'L':>8} | {'dense KV':>12} | {'indexer KV':>12} | "
          f"{'sparse KV':>12} | 稀疏/稠密")
    for l in l_values:
        dense = l * (dc + dr) * 2
        idx_kv = l * i_dim * 2
        spar = topk * (dc + dr) * 2
        print(f"  {l:>8} | {dense / 1e6:>10.1f}MB | {idx_kv / 1e6:>10.1f}MB | "
              f"{spar / 1e6:>10.1f}MB | {(idx_kv + spar) / dense:>8.2f}x")
    print("  这就是 DSA/NSA 的真实卖点：把「读完整 KV」换成「读索引 + 读少量 KV」，")
    print("  代价是 indexer 的算力与一次全序列 top-k。")


def main() -> int:
    book = Path(__file__).resolve().parents[4]
    print(f"教材根目录: {book}")
    print("\n引用核对（v0.1.15）")
    ok, bad = check_mainline(book)
    print(f"  命中 {ok} 处，未命中 {bad} 处。")

    path = Path(__file__).resolve().parent / V32_CONFIG_RELATIVE
    if path.is_file():
        cfg = json.loads(path.read_text(encoding="utf-8"))
    else:
        cfg = {"index_n_heads": 64, "index_head_dim": 128, "index_topk": 2048,
               "n_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
    print_structure_table()
    print_cost_model(cfg)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
