#!/usr/bin/env python3
"""Chapter 39, §5: map (chapter, component, typical bottleneck) as a table.

The table is data, not measurement: every row points at the chapter and section
that establishes it. `--verify` checks that the chapter directories this table
references actually exist in the book tree, so the mapping cannot silently rot.

Usage:
    python3 bottleneck_map.py
    python3 bottleneck_map.py --verify
    python3 bottleneck_map.py --verify --book-dir ../../../   # chapters/ 的上一级
"""

from __future__ import annotations

import argparse
from pathlib import Path

# (chapter, slug, component, typical bottleneck, how to check it)
ROWS = [
    ("13", "operators-elementwise-norm-rope", "逐元素 / 归一化 / RoPE",
     "启动开销 + 重复读小张量",
     "核数、kernel 数量；第 28 章 M6"),
    ("14", "operators-gemm", "GEMM 全家族", "大 M 算力 / 小 M 权重流量",
     "先算算术强度，再看 aic_mad；第 35 章 §5.3"),
    ("15", "operators-attention", "Attention 家族", "online softmax 的中间量带宽",
     "L0C / UB 容量与 FixPipe；第 36 章 §3.2"),
    ("16", "operators-moe-mla", "MoE 与 MLA", "两段零 FLOP 的重排与通信",
     "第 16 章 §7 的 regime 表"),
    ("33", "tilekernels-two-backends", "TileKernels（路由 / 量化 / RoPE / Rand）",
     "访存为主 + 跨后端差异", "两个后端的并行映射；第 33 章 §7.1"),
    ("34", "tilekernels-engram-mhc", "Engram / mHC（不是矩阵乘）",
     "kernel 数量 + 迭代的编译期常量",
     "数 launch 次数；第 34 章 §7.1"),
    ("35", "deepgemm-ascend", "DeepGEMM-Ascend（dense / grouped / MQA / MegaMoE）",
     "compute / memory / FixPipe 三种都要判",
     "三个比值；第 35 章 §5.1"),
    ("36", "flashmla-ascend", "FlashMLA 昇腾稀疏注意力",
     "MTE2 与 FixPipe 的赛跑 + icache",
     "报告的四个代价讨论；第 36 章 §3"),
    ("37", "deepselect-topk", "DeepSelect TopK",
     "有效内存带宽（不是 FLOPs）",
     "输入+输出字节 ÷ 时间；第 37 章 §6.1"),
    ("38", "deepep-ascend", "DeepEP-Ascend（EP dispatch / combine）",
     "卡间带宽 + 固定 launch 开销",
     "通算流水图的 Wait Time Ratio；第 38 章 §7"),
]

SLUG_TEMPLATE = "{number}-{slug}"


def print_markdown() -> None:
    print("| 章 | 组件 | 典型瓶颈 | 怎么查 |")
    print("| --- | --- | --- | --- |")
    for number, slug, component, bottleneck, howto in ROWS:
        print(f"| 第 {number} 章 | {component} | {bottleneck} | {howto} |")


def verify(book_dir: Path) -> int:
    print(f"核验章节目录是否存在（chapters/ 的上一级 = {book_dir.resolve()}）")
    missing = 0
    for number, slug, *_rest in ROWS:
        chapter_dir = book_dir / "chapters" / SLUG_TEMPLATE.format(number=number, slug=slug)
        exists = (chapter_dir / "chapter.md").is_file()
        print(f"  {'ok ' if exists else 'MISSING'}  {chapter_dir}")
        if not exists:
            missing += 1
    print()
    if missing == 0:
        print("全部命中：这张映射表指向的章节都存在。")
    else:
        print(f"{missing} 条没命中——请更新本脚本的 slug，而不是忽略它。")
    return 1 if missing else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true",
                        help="核验引用的章节目录存在")
    parser.add_argument("--book-dir", default="../../../..",
                        help=".workspace/research/tilelang-book/（chapters/ 的上一级）")
    args = parser.parse_args()

    print("第 39 章 · 「章节 → 组件 → 典型瓶颈」映射")
    print("每一行都指回一章；瓶颈判据只用于判类，不预测时间。")
    print()
    print_markdown()
    print()
    if args.verify:
        return verify(Path(args.book_dir))
    print("（加 --verify 可以核验这些章节目录都存在。）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
