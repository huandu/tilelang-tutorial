#!/usr/bin/env python3
"""Static map of the official attention examples and of every API the chapter cites.

Two jobs, both done by reading text -- no tilelang, no GPU, no imports of the
vendored packages:

  A. For each attention-family example in the locked tree, print the lines the
     chapter points at (the kernel launch, the pipelined loop, the softmax
     recurrence, the mask, the sink). This is the "where is it implemented"
     table from section 4.6.

  B. For every TileLang API the chapter mentions, locate its definition in the
     locked tree and print `path:line`. Each expected line number is pinned, so
     running this doubles as a check that the chapter's line numbers are still
     accurate.

Run:  python3 map_official_examples.py
      python3 map_official_examples.py --strict   # non-zero on any drift
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BOOK = Path(__file__).resolve().parents[4]          # .workspace/research/tilelang-book
NV = BOOK / "_vendor" / "tilelang-v0.1.15"

# ---------------------------------------------------------------- part A ----

# (relative path, [(label, regex), ...])
EXAMPLES: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "examples/flash_attention/example_mha_fwd_bhsd.py",
        [
            ("launch", r"with T\.Kernel\("),
            ("log2e-folded scale", r"scale = \(1\.0 / dim\)"),
            ("pipelined KV loop", r"for k in T\.Pipelined\("),
            ("causal loop range", r"loop_range = \("),
            ("causal mask", r"q_idx >= k_idx"),
            ("QK^T", r"T\.gemm\(Q_shared, K_shared, acc_s"),
            ("running max merge", r"scores_max\[i\] = T\.max\("),
            ("rescale factor", r"scores_scale\[i\] = T\.exp2\("),
            ("normalise", r"acc_o\[i, j\] /= logsum\[i\]"),
        ],
    ),
    (
        "examples/flash_attention/example_mha_fwd_varlen.py",
        [
            ("launch", r"with T\.Kernel\("),
            ("cu_seqlens read", r"cu_seqlens_q\[batch_idx\]"),
            ("right-align offset", r"offset = kv_current_seqlen - q_current_seqlen"),
        ],
    ),
    (
        "examples/flash_attention/example_gqa_fwd_varlen.py",
        [
            ("kv head index", r"kv_head_idx = head_idx // groups"),
            ("causal mask with -1e9", r"-1e9"),
            ("empty-row guard", r"some tokens can see nothing"),
            ("store guard", r"if bx \* block_M \+ i < q_current_seqlen"),
        ],
    ),
    (
        "examples/flash_decoding/example_mha_inference.py",
        [
            ("split kernel", r"with T\.Kernel\(T\.ceildiv\(seqlen_q, block_M\), heads \* batch, num_split"),
            ("per-split lse store", r"logsum\[i\] = T\.log2\(logsum\[i\]\) \+ scores_max\[i\] \* scale"),
            ("combine kernel", r"# combine"),
            ("combine max", r"T\.reduce_max\(lse_local, lse_max_local, dim=0"),
            ("combine weights", r"scale_local\[i\] = T\.exp2\("),
        ],
    ),
    (
        "examples/attention_sink/example_mha_sink_fwd_bhsd.py",
        [
            ("window lower bound", r"start = T\.max\(0, \(bx \* block_M \+ past_len - window_size\)"),
            ("window mask", r"q_idx >= k_idx and q_idx < k_idx \+ window_size"),
            ("check_inf flag", r"check_inf is necessary"),
            ("sink in denominator", r"logsum\[i\] \+= T\.exp2\(sinks\[i\]"),
        ],
    ),
    (
        "examples/block_causal_attention/block_causal_attention.py",
        [
            ("kernel", r"with T\.Kernel\("),
            ("per-tile mask decision", r"is_noisy_diag = T\.if_then_else\("),
        ],
    ),
    (
        "examples/flash_attention_sm100/mha_fwd_bshd.py",
        [
            ("tmem accumulator", r"S_tmem = T\.alloc_tmem\("),
            ("barrier", r"mbar_s = T\.alloc_barrier\("),
            ("tcgen05 mma", r"T\.tcgen05_gemm\("),
        ],
    ),
    (
        "examples/deepseek_nsa/example_tilelang_nsa_fwd.py",
        [
            ("index tensor", r"block_indices_shape = \[batch, seq_len, head_kv, selected_blocks\]"),
            ("kernel", r"with T\.Kernel\("),
        ],
    ),
    (
        "examples/minference/example_vertical_slash_sparse_attn.py",
        [
            ("vertical/slash index load", r"vertical_slash"),
        ],
    ),
]

# ---------------------------------------------------------------- part B ----

# (file relative to the locked TileLang tree, symbol, expected 1-based line)
NV_API: list[tuple[str, str, int]] = [
    ("tilelang/language/allocate.py", "alloc_shared", 49),
    ("tilelang/language/allocate.py", "alloc_fragment", 81),
    ("tilelang/language/allocate.py", "alloc_global", 171),
    ("tilelang/language/allocate.py", "alloc_barrier", 194),
    ("tilelang/language/allocate.py", "alloc_tmem", 254),
    ("tilelang/language/reduce_op.py", "reduce_max", 176),
    ("tilelang/language/reduce_op.py", "reduce_sum", 230),
    ("tilelang/language/gemm_op.py", "gemm", 160),
    ("tilelang/language/loop.py", "Pipelined", 112),
    ("tilelang/language/copy_op.py", "copy", 59),
    ("tilelang/language/tir/op.py", "exp2", 74),
    ("tilelang/language/tir/op.py", "infinity", 361),
    ("tilelang/cuda/language/gemm_op.py", "tcgen05_gemm", 264),
    ("tilelang/tileop/base.py", "class GemmWarpPolicy", 5),
]


def lines_of(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def scan_examples(strict: bool) -> int:
    bad = 0
    print("=" * 78)
    print("A. attention-family examples in the locked TileLang tree")
    print(f"   root: {NV}")
    print("=" * 78)
    for rel, probes in EXAMPLES:
        path = NV / rel
        if not path.is_file():
            print(f"MISSING FILE  {rel}")
            bad += 1
            continue
        body = lines_of(path)
        print(f"\n{rel}  ({len(body)} lines)")
        for label, pattern in probes:
            hit = next((i + 1 for i, line in enumerate(body) if re.search(pattern, line)), None)
            if hit is None:
                print(f"    {label:<26} not found   <- probe: {pattern}")
                bad += 1
            else:
                print(f"    {label:<26} :{hit}")
    return bad


def scan_api(strict: bool) -> int:
    bad = 0
    print()
    print("=" * 78)
    print("B. API definitions the chapter cites (line numbers are pinned)")
    print("=" * 78)
    for rel, symbol, expected in NV_API:
        path = NV / rel
        if not path.is_file():
            print(f"MISSING FILE  {rel}")
            bad += 1
            continue
        body = lines_of(path)
        pattern = re.compile(r"^\s*(?:def |class )?" + re.escape(symbol) + r"\b")
        hit = next((i + 1 for i, line in enumerate(body) if pattern.match(line)), None)
        if hit is None:
            print(f"  {symbol:<20} NOT FOUND in {rel}")
            bad += 1
            continue
        status = "ok" if hit == expected else "DRIFT"
        if hit != expected:
            bad += 1
        print(f"  {symbol:<20} {rel}:{hit}"
              + ("" if hit == expected else f"   (chapter says :{expected})   {status}"))
    print()
    print(f"drift/miss count: {bad}")
    return bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any probe or line number drifted")
    args = parser.parse_args()
    print(__doc__)
    if not NV.is_dir():
        print(f"locked tree not found: {NV}")
        print("This example only reads the vendored sources; run it inside the")
        print("book checkout, or point BOOK at a tree that has _vendor/.")
        return 2
    bad = scan_examples(args.strict)
    bad += scan_api(args.strict)
    print()
    print("=" * 78)
    print("This checks existence and line numbers only. It does not compile,")
    print("import or run anything, and it needs no GPU/NPU.")
    print("=" * 78)
    return 1 if (args.strict and bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
