#!/usr/bin/env python3
"""Static map of the Ascend attention kernels and of the Ascend API surface.

Text-only: it reads the two locked trees and prints what the chapter points
at. No CANN, no NPU, no tilelang import.

  A. The GM/L1/L0A/L0B/L0C/UB copy matrix, parsed straight out of the locked
     programming guide (this is the table that shows why L0C -> UB does not
     exist and why a workspace is needed).
  B. Ascend-specific API definitions with their file:line, in both the
     adapter repo (v0.1.2.000) and the mainline repo (v0.1.15).
  C. The structure of each Ascend attention example: kernel launch, the
     Cube/Vector scopes, the cross-core flags, and the softmax recurrence.
  D. The bottleneck notes, quoted from the locked tuning document.

Run:  python3 ascend_attention_map.py
      python3 ascend_attention_map.py --strict   # non-zero on any drift
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BOOK = Path(__file__).resolve().parents[4]
ASC = BOOK / "_vendor" / "tilelang-ascend-v0.1.2.000"
NV = BOOK / "_vendor" / "tilelang-v0.1.15"
GUIDE = ASC / "docs" / "TileLang-Ascend Programming Guide.md"
OPT_DOC = (ASC / "examples" / "flash_attention" / "fa_opt"
           / "flash_attention_performance_optimization_zh.md")

# (file relative to ASC, symbol, expected 1-based line)
ASC_API: list[tuple[str, str, int]] = [
    ("tilelang/language/allocate.py", "alloc_L1", 140),
    ("tilelang/language/allocate.py", "alloc_L0C", 152),
    ("tilelang/language/allocate.py", "alloc_ub", 156),
    ("tilelang/language/ascend.py", "set_cross_flag", 191),
    ("tilelang/language/ascend.py", "wait_cross_flag", 213),
    ("tilelang/language/ascend.py", "barrier_all", 275),
    ("tilelang/language/ascend.py", "gemm_v0", 418),
    ("tilelang/language/ascend_tile.py", "fill", 343),
    ("tilelang/language/ascend_tile.py", "topk", 550),
    ("tilelang/language/ascend_tile.py", "gather_mask", 608),
    ("tilelang/language/ascend_tile.py", "exp", 1140),
    ("tilelang/language/ascend_tile.py", "cast", 1927),
    ("tilelang/language/ascend_tile.py", "broadcast", 2302),
    ("tilelang/language/warpgroup.py", "Scope", 74),
]

# The same ideas, but on the mainline repo's Ascend dialect.
NV_ASC_API: list[tuple[str, str, int]] = [
    ("tilelang/ascend/language/allocate.py", "alloc_shared", 15),
    ("tilelang/ascend/language/allocate.py", "alloc_l1", 26),
    ("tilelang/ascend/language/allocate.py", "alloc_l0a", 37),
    ("tilelang/ascend/language/allocate.py", "alloc_l0b", 46),
    ("tilelang/ascend/language/allocate.py", "alloc_l0c", 55),
    ("tilelang/ascend/language/copy_op.py", "dual_copy", 53),
    ("tilelang/ascend/language/frame.py", "SimdVF", 179),
    ("tilelang/ascend/language/schedule_hint.py", "Stage", 118),
    ("tilelang/ascend/language/annotations.py", "annotate_buffer_versions", 19),
]

# (file relative to ASC, [(label, regex), ...])
ASC_EXAMPLES: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "examples/flash_attention/flash_attn_bhsd.py",
        [
            ("launch", r"with T\.Kernel\(block_num, is_npu=True\)"),
            ("workspace 1", r"workspace_1: T\.Tensor"),
            ("workspace 2", r"workspace_2: T\.Tensor"),
            ("workspace 3", r"workspace_3: T\.Tensor"),
            ("cube scope", r'with T\.Scope\("C"\)'),
            ("vector scope", r'with T\.Scope\("V"\)'),
            ("cube->vector flag", r'T\.set_cross_flag\("FIX", 0\)'),
            ("vector waits", r"T\.wait_cross_flag\(0\)"),
            ("QK^T on cube", r"T\.gemm_v0\(q_l1, k_l1, acc_s_l0c"),
            ("running max", r"T\.reduce_max\(acc_s_ub, m_i, dim=-1\)"),
            ("rescale", r"T\.tile\.exp\(m_i_prev, m_i_prev\)"),
        ],
    ),
    (
        "examples/flash_attention/paged_flash_attn_bhsd.py",
        [
            ("block table arg", r"block_table: T\.Tensor"),
            ("indirect K address", r"bidx = block_table\[bz, k\]"),
        ],
    ),
    (
        "examples/mha_sink_fwd_bhsd/mha_sink_fwd_bhsd.py",
        [
            ("sink broadcast", r"T\.tile\.broadcast\(sinks_ub"),
            ("m* explanation", r"exp\(sinks - m_i_final\)"),
        ],
    ),
    (
        "examples/gqa_fwd_varlen/gqa_fwd_varlen.py",
        [
            ("gqa groups", r"head_kv"),
            ("cu_seqlens", r"cu_seqlens"),
            ("explicit mask", r"Mask: T\.Tensor"),
        ],
    ),
    (
        "examples/lightning_indexer/example_lightning_indexer.py",
        [
            ("index tensor", r"QK_RES"),
            ("topk", r"T\.tile\.topk\("),
            ("gather mask", r"T\.tile\.gather_mask\("),
        ],
    ),
]


def lines_of(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def section_a() -> int:
    print("=" * 78)
    print("A. the copy matrix from the locked programming guide")
    print(f"   {GUIDE.relative_to(BOOK)}")
    print("=" * 78)
    if not GUIDE.is_file():
        print("MISSING GUIDE")
        return 1
    body = lines_of(GUIDE)
    rows = [f"  L{n:>5}  {line.strip()}" for n, line in enumerate(body, 1)
            if re.match(r"\|\s*(GM|L1|L0[ABC]|UB)\s*\|", line)]
    for row in rows:
        print(row)
    print()
    print("  key point: there is no L0C -> UB row. A Cube-produced score tile")
    print("  reaches the Vector core only by going through GM (the workspace).")
    print()
    return 0


def scan_api(tree: Path, entries: list[tuple[str, str, int]], title: str) -> int:
    print("=" * 78)
    print(title)
    print("=" * 78)
    bad = 0
    for rel, symbol, expected in entries:
        path = tree / rel
        if not path.is_file():
            print(f"  MISSING FILE  {rel}")
            bad += 1
            continue
        body = lines_of(path)
        pattern = re.compile(r"^\s*(?:def |class )?" + re.escape(symbol) + r"\b")
        hit = next((i + 1 for i, line in enumerate(body) if pattern.match(line)), None)
        if hit is None:
            print(f"  {symbol:<24} NOT FOUND in {rel}")
            bad += 1
            continue
        if hit != expected:
            bad += 1
            print(f"  {symbol:<24} {rel}:{hit}   DRIFT (chapter says :{expected})")
        else:
            print(f"  {symbol:<24} {rel}:{hit}")
    print()
    return bad


def scan_examples() -> int:
    print("=" * 78)
    print("C. Ascend attention examples in the locked adapter tree")
    print("=" * 78)
    bad = 0
    for rel, probes in ASC_EXAMPLES:
        path = ASC / rel
        if not path.is_file():
            print(f"MISSING FILE  {rel}")
            bad += 1
            continue
        body = lines_of(path)
        print(f"\n{rel}  ({len(body)} lines)")
        for label, pattern in probes:
            hits = [i + 1 for i, line in enumerate(body) if re.search(pattern, line)]
            if not hits:
                print(f"    {label:<24} not found   <- probe: {pattern}")
                bad += 1
            else:
                shown = ", ".join(f":{h}" for h in hits[:4])
                more = "" if len(hits) <= 4 else f" (+{len(hits) - 4} more)"
                print(f"    {label:<24} {shown}{more}")
    print()
    return bad


def section_d() -> int:
    print("=" * 78)
    print("D. the bottleneck notes, quoted from the locked tuning document")
    print(f"   {OPT_DOC.relative_to(BOOK)}")
    print("=" * 78)
    if not OPT_DOC.is_file():
        print("MISSING TUNING DOC")
        return 1
    body = lines_of(OPT_DOC)
    wanted = [
        r"^## 二、性能优化总体思路",
        r"^### 3\.1 Double Buffer 原理",
        r"^### 4\.1 找到最佳 num_stages",
        r"^### 4\.2 核间同步优化",
        r"^\*\*重要\*\*：`T\.pipelined` 不支持嵌套使用。",
    ]
    for pattern in wanted:
        hit = next((i + 1 for i, line in enumerate(body) if re.search(pattern, line)), None)
        if hit is None:
            print(f"  not found   <- {pattern}")
            return 1
        print(f"  L{hit:>4}  {body[hit - 1].strip()}")
    print()
    print("  the doc's own ordering of the fix list is: Double Buffer / 单一 bound")
    print("  -> best num_stages -> fewer cross-core syncs -> vectorise scalar ops.")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    print(__doc__)
    if not ASC.is_dir() or not NV.is_dir():
        print(f"locked trees not found:\n  {ASC}\n  {NV}")
        print("This example only reads the vendored sources; run it inside the")
        print("book checkout, or point BOOK at a tree that has _vendor/.")
        return 2
    bad = section_a()
    bad += scan_api(ASC, ASC_API, "B1. adapter repo (TileLang-ascend v0.1.2.000)")
    bad += scan_api(NV, NV_ASC_API, "B2. mainline repo Ascend dialect (TileLang v0.1.15)")
    bad += scan_examples()
    bad += section_d()
    print("=" * 78)
    print(f"drift/miss count: {bad}")
    print("Everything above is static reading of the locked trees. Nothing was")
    print("compiled, imported or executed on an NPU.")
    print("=" * 78)
    return 1 if (args.strict and bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
