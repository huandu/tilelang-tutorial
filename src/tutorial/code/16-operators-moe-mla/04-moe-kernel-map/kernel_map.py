#!/usr/bin/env python3
"""第 16 章 · MoE 相关引用的「文件 + 行号 + token」核对（纯标准库）。

本章对 MoE 部分的每一条源码引用都写成 ``path:line``。这个脚本把那些引用
落成可执行的检查：打开 ``_vendor/`` 里的文件，确认**第 line 行确实出现该
token**。锁定版本是不动的，所以这个检查可重复运行。

这是一个**引用完整性检查**，不是正确性证明：它只保证「我引用的那一行确实
写着我说的东西」，不保证我对那一行的解释是对的。

用法
----

    python3 kernel_map.py            # 检查并打印命中/未命中
    python3 kernel_map.py --book ../../..

若 ``_vendor/`` 不存在（把本章目录单独复制出去的情形），脚本打印说明并
以 0 退出。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# (相对 _vendor 根的路径, 行号, 该行必须包含的 token, 这个引用想说明什么)
MAINLINE: list[tuple[str, int, str, str]] = [
    ("tilelang/tileop/base.py", 5, "class GemmWarpPolicy",
     "T.gemm 的 warp 划分策略 Square/FullRow/FullCol 在这里定义"),
    ("examples/fusedmoe/example_fusedmoe_tilelang.py", 92,
     "def moe_forward_tilelang_routed",
     "routed expert 的 grouped GEMM kernel 入口"),
    ("examples/fusedmoe/example_fusedmoe_tilelang.py", 120,
     "M = math.ceil(group_sum / block_token) + group_count",
     "grid 的 M 方向块数 = 各专家块数之和"),
    ("examples/fusedmoe/example_fusedmoe_tilelang.py", 157,
     "cur_group_idx = group_idx_for_bx[bx]",
     "kernel 内用查表反解当前 block 属于哪个专家"),
    ("examples/fusedmoe/example_fusedmoe_tilelang.py", 161, "actual_rows",
     "本 block 的有效行数（去掉 padding）"),
    ("examples/fusedmoe/example_fusedmoe_tilelang.py", 406,
     "torch.scatter_reduce",
     "combine：按原 token 位置 scatter-reduce 求和"),
    ("examples/fusedmoe/example_fusedmoe_tilelang.py", 385,
     "torch.cuda.Stream",
     "共享专家与路由专家跑在两条 stream 上"),
    ("examples/grouped_gemm/example_grouped_gemm_fwd.py", 64, "total_m_blocks",
     "「拼接 + 偏移表」版 grouped GEMM 的 grid 大小"),
    ("examples/grouped_gemm/example_grouped_gemm_fwd.py", 74,
     "batch_padded_offsets", "对齐到 block_M 的偏移表"),
    ("examples/grouped_gemm/example_grouped_gemm_fwd_ptr.py", 32,
     "copy_num_stages = 1",
     "指针表版把流水固定为 1 级（文件头注释给了理由）"),
    ("examples/grouped_gemm/example_grouped_gemm_fwd_ptr.py", 63,
     "T.make_tensor", "用指针表在 kernel 内构造张量视图"),
    ("examples/deepseek_mla/example_mla_decode.py", 65,
     "policy=T.GemmWarpPolicy.FullCol",
     "按列切开 acc_o，两个 warpgroup 各算一半 dim"),
    ("examples/deepseek_mla/example_mla_decode.py", 66,
     "T.gemm(Q_pe_shared, K_pe_shared",
     "解耦 RoPE 的第二个 QK 乘积累加到同一个 acc_s"),
    ("examples/deepseek_mla/example_mla_decode.py", 31, "T.alloc_global",
     "split-KV 的中间结果放在 global（跨 block 合并）"),
    ("examples/deepseek_v32/topk_selector.py", 11, "def convert_to_uint16",
     "把 float 映射成有序 uint16，做 8-bit 基数排序"),
    ("examples/deepseek_v32/fp8_lighting_indexer.py", 155, "T.gemm(",
     "lightning indexer 的核心相似度 GEMM"),
    ("examples/deepseek_v32/fp8_lighting_indexer.py", 167,
     "T.reduce_sum", "ReLU + 按头加权后归约成 logits"),
    ("examples/deepseek_v32/sparse_mla_fwd.py", 120, "Indices[",
     "sparse MLA 只按 Indices 加载被选中的 KV 行"),
    ("examples/deepseek_nsa/example_tilelang_nsa_fwd.py", 50, "BlockIndices:",
     "NSA：按 block 索引做块稀疏注意力"),
    ("examples/deepseek_v32/inference/config_671B_v3.2.json", 0, "",
     "（配置数值由 JSON 解析检查，见下）"),
]

ASCEND: list[tuple[str, int, str, str]] = [
    ("examples/fusedmoe/example_fusedmoe.py", 30,
     "TL_ASCEND_AUTO_CV_COMBINE",
     "combineCV：自动把 Cube/Vector 合进一条流水并插同步"),
    ("examples/fusedmoe/example_fusedmoe.py", 163, "block_metadata",
     "[total_m_blocks, 3] 的路由表：[expert, m_start, valid_m]"),
    ("examples/fusedmoe/example_fusedmoe.py", 176, "T.alloc_L0C",
     "累加器放在 L0C"),
    ("examples/fusedmoe/example_fusedmoe.py", 230,
     "L0C = 2*(64*256*4) = 128KB",
     "官方注释直接给出片上容量账"),
    ("examples/moe_token_permute/moe_token_permute.py", 79, "T.alloc_ub",
     "昇腾的向量寄存器缓冲 UB"),
    ("examples/dispatch_combine/dispatch_combine_shmem.py", 148,
     "T.shmem_ub_put_nbi",
     "dispatch：跨 rank 的 non-blocking put"),
    ("examples/dispatch_combine/dispatch_combine_shmem.py", 68,
     "ep_receive_count", "dispatch 输出的前缀和（每个专家收到多少 token）"),
    ("examples/deepseek_v4/lightning_indexer.py", 575, "T.alloc_L1",
     "昇腾 lightning indexer 的 q/k 缓冲在 L1"),
    ("examples/deepseek_v4/sparse_flash_mla/sparse_flash_mla_kernel.py", 1199,
     "cmp_indices", "V4 sparse flash MLA：压缩 KV 由 top-k 索引选中"),
    ("examples/deepseek_v4/hc_split_sinkhorn.py", 27, "def hc_split_sinkhorn",
     "mHC 的 Sinkhorn 归一化（流形约束）"),
]


def check_group(root: Path, rel_root: str, specs) -> tuple[int, int]:
    base = root / rel_root
    if not base.is_dir():
        print(f"[skip] {rel_root} 不存在（把本章单独复制出去时属正常）")
        return 0, 0
    ok = bad = 0
    for rel, line, token, why in specs:
        path = base / rel
        if not path.is_file():
            print(f"  MISSING FILE {rel}   （{why}）")
            bad += 1
            continue
        if line == 0:
            print(f"  ok    {rel:<58} （{why}）")
            ok += 1
            continue
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if 1 <= line <= len(lines) and token in lines[line - 1]:
            print(f"  ok    {rel}:{line}  ({token})")
            ok += 1
        else:
            got = lines[line - 1].strip() if 1 <= line <= len(lines) else "<越界>"
            print(f"  FAIL  {rel}:{line}  期望含 {token!r}，实际是 {got!r}")
            bad += 1
    return ok, bad


def check_v32_config(root: Path) -> tuple[int, int]:
    import json

    path = (root / "tilelang-v0.1.15" / "examples" / "deepseek_v32"
            / "inference" / "config_671B_v3.2.json")
    if not path.is_file():
        print("[skip] DeepSeek-V3.2 配置不存在")
        return 0, 0
    cfg = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "n_layers": 61,
        "n_heads": 128,
        "n_routed_experts": 256,
        "n_activated_experts": 8,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
    }
    ok = bad = 0
    for key, want in expected.items():
        got = cfg.get(key)
        if got == want:
            print(f"  ok    config_671B_v3.2.json {key} = {got}")
            ok += 1
        else:
            print(f"  FAIL  config_671B_v3.2.json {key} = {got}（期望 {want}）")
            bad += 1
    return ok, bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", default=None,
                        help="教材根目录（含 _vendor/），默认从脚本位置推断")
    args = parser.parse_args()
    book = Path(args.book).resolve() if args.book else Path(__file__).resolve().parents[4]
    vendor = book / "_vendor"
    print(f"教材根目录: {book}")
    if not vendor.is_dir():
        print("未找到 _vendor/：跳过检查（结构检查仍可通过）。")
        return 0

    print("\nv0.1.15（主干）")
    ok1, bad1 = check_group(book, "_vendor/tilelang-v0.1.15", MAINLINE)
    ok1b, bad1b = check_v32_config(book / "_vendor")
    print("\nv0.1.2.000（昇腾适配器）")
    ok2, bad2 = check_group(book, "_vendor/tilelang-ascend-v0.1.2.000", ASCEND)

    ok, bad = ok1 + ok1b + ok2, bad1 + bad1b + bad2
    print(f"\n合计：{ok} 处命中，{bad} 处未命中。")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
