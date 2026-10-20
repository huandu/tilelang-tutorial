#!/usr/bin/env python3
"""第 16 章 · 昇腾侧的 MoE / 稀疏注意力对照（纯标准库，CPU 可跑）。

两件事：

A. **代码对照**：把昇腾侧的四个 example（fusedmoe / moe_token_permute /
   dispatch_combine / deepseek_v4）与本章的概念对齐，并核对每条引用。
B. **专家并行的通信账**：dispatch/combine 是 all-to-all。这里给出每个 token、
   每个 batch 要搬多少字节，并把「通信字节」与「激活字节」放在一起比，
   说明为什么 EP（expert parallel）场景下瓶颈会从矩阵乘挪到通信。

通信带宽**不写死**：脚本只算字节量，时间 = 字节 / 带宽。任何一个具体带宽都是
占位示例，不是任何设备的实测规格。

用法
----

    python3 ascend_map.py
    python3 ascend_map.py --tokens 8192 --top-k 8 --d-model 7168
"""

from __future__ import annotations

import argparse
from pathlib import Path

# (文件, 行号, 该行必须包含的 token, 说明)
ASCEND: list[tuple[str, int, str, str]] = [
    ("examples/fusedmoe/example_fusedmoe.py", 30, "TL_ASCEND_AUTO_CV_COMBINE",
     "combineCV + auto sync：把 Cube/Vector 合流并自动插同步"),
    ("examples/fusedmoe/example_fusedmoe.py", 89, "def dual_gemm_kernel",
     "gate/up 融成一个 kernel，input 只读一次"),
    ("examples/fusedmoe/example_fusedmoe.py", 148, "block_metadata",
     "grouped GEMM 用 [expert, m_start, valid_m] 路由"),
    ("examples/fusedmoe/example_fusedmoe.py", 230, "L0C = 2*(64*256*4) = 128KB",
     "官方注释给出片上容量账（L0C 打满）"),
    ("examples/moe_token_permute/moe_token_permute.py", 79, "T.alloc_ub",
     "token permute 在 UB 上做直方图/前缀和"),
    ("examples/moe_token_permute/moe_token_permute.py", 38, "workspace_idx",
     "permute 需要一块 workspace 做跨核前缀和"),
    ("examples/dispatch_combine/dispatch_combine_shmem.py", 148,
     "T.shmem_ub_put_nbi", "dispatch：跨 rank 的 non-blocking put"),
    ("examples/dispatch_combine/dispatch_combine_shmem.py", 68,
     "ep_receive_count", "每个专家收到多少 token 的前缀和"),
    ("examples/dispatch_combine/dispatch_combine_shmem.py", 254,
     "expert_token_nums_out", "用前缀和差分得到每个专家最终的 token 数"),
    ("examples/deepseek_v4/lightning_indexer.py", 575, "T.alloc_L1",
     "V4 的 indexer 在昇腾上用 L1 缓冲"),
    ("examples/deepseek_v4/sparse_flash_mla/sparse_flash_mla_kernel.py", 1199,
     "cmp_indices", "V4 sparse flash MLA：压缩 KV 由 top-k 索引选中"),
    ("examples/deepseek_v4/sparse_flash_mla/bench_mark.md", 1, "Sparse flash MLA",
     "V4 sparse flash MLA 的官方 benchmark 说明（含 swa/hca/csa 三种访问模式）"),
]


def check(book: Path) -> tuple[int, int]:
    base = book / "_vendor" / "tilelang-ascend-v0.1.2.000"
    if not base.is_dir():
        print("[skip] _vendor/tilelang-ascend-v0.1.2.000 不存在")
        return 0, 0
    ok = bad = 0
    for rel, line, token, why in ASCEND:
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


def comm_model(tokens: int, top_k: int, d_model: int, b: int,
               ranks: int, local_experts: int) -> None:
    print("\n" + "=" * 78)
    print("B. 专家并行（EP）的通信账")
    print("=" * 78)
    print(f"\ntokens={tokens} top_k={top_k} d_model={d_model} "
          f"每元素 {b} 字节  EP ranks={ranks} local_experts={local_experts}")
    per_token_per_way = top_k * d_model * b
    dispatch = tokens * per_token_per_way
    combine = dispatch
    total = dispatch + combine
    print(f"\n  每个 token 每个方向: k*d*b = {per_token_per_way} 字节 "
          f"= {per_token_per_way / 1024:.1f} KB")
    print(f"  dispatch 总量 : {dispatch / 1e6:.1f} MB")
    print(f"  combine  总量 : {combine / 1e6:.1f} MB")
    print(f"  合计          : {total / 1e6:.1f} MB（一层，单方向全量）")

    # 与「激活」比：一层里 token 激活本身只有 tokens*d 个元素。
    activation = tokens * d_model * b
    print(f"\n  对照：这一层的激活（tokens*d*b）只有 {activation / 1e6:.1f} MB；")
    print(f"  也就是说，EP 的 dispatch+combine 要搬 {total / activation:.1f} x 的激活量。")

    # 与专家的权重流量比：本地专家权重 = local_experts * 3*d*de*b。
    # 这里 d_expert 未知，用比值的形式给出，避免臆造参数。
    print("\n  再把互联带宽 BW 代进去：通信时间 = 字节 / BW。")
    print("  （下面带宽只是**占位示例**，不是任何设备的实测规格。）")
    print(f"\n  {'BW (GB/s)':>12} | {'dispatch 时间':>16} | {'combine 时间':>16}")
    for bw_gbps in (50, 100, 200, 400):
        bw = bw_gbps * 1e9
        print(f"  {bw_gbps:>12} | {dispatch / bw * 1e6:>14.1f} us | "
              f"{combine / bw * 1e6:>14.1f} us")
    print("\n  读法：把你们集群的实际互联带宽代进去，就能判断一层 MoE 里")
    print("  「通信」和「专家 GEMM」谁更长。跨机 EP 时前者常常更大。")
    print("  注意：这里的字节量是**单方向全量**的下界，未计协议开销、")
    print("  状态同步、以及 dispatch 里那份 3 元组 (rank, token, topk_id)。")


def pipeline_map() -> None:
    print("\n" + "=" * 78)
    print("A. 昇腾侧 MoE 一层的数据流（对照官方 example）")
    print("=" * 78)
    steps = [
        ("1 gating", "host 或小 kernel", "softmax + topk + 计数（host_preprocess）"),
        ("2 dispatch", "all-to-all", "把 token 发到专家所在 rank（shmem put）"),
        ("3 permute", "UB 直方图/前缀和", "把收到的 token 按专家紧凑排列"),
        ("4 gate/up GEMM", "grouped dual GEMM", "input 读一次，出 gate 与 up 两个结果"),
        ("5 SiLU + mul", "PyTorch 或融合", "up * silu(gate)"),
        ("6 down GEMM", "grouped GEMM", "de -> d_model"),
        ("7 写回 + 加权", "elementwise", "乘 gating 权重，写回 token 布局"),
        ("8 combine", "all-to-all", "把结果送回原 rank 并按 token 求和"),
    ]
    for name, where, what in steps:
        print(f"  {name:<16} | {where:<22} | {what}")
    print("\n  昇腾侧的三个关键点（都能在源码里指到行）：")
    print("   * 累加器在 L0C、输入/权重在 L1，容量要手算（官方注释直接给出 L0C=128KB）；")
    print("   * combineCV + auto sync 把 Cube/Vector 合流并自动插同步，省掉手写 barrier；")
    print("   * 跨 rank 用 shmem 的 put/get，dispatch/combine 在源码里是一对 kernel。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=7168)
    parser.add_argument("--bytes", type=int, default=2)
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--local-experts", type=int, default=3)
    args = parser.parse_args()

    book = Path(__file__).resolve().parents[4]
    print(f"教材根目录: {book}")
    print("\n引用核对（tilelang-ascend v0.1.2.000）")
    ok, bad = check(book)
    print(f"  命中 {ok} 处，未命中 {bad} 处。")
    pipeline_map()
    comm_model(args.tokens, args.top_k, args.d_model, args.bytes,
               args.ranks, args.local_experts)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
