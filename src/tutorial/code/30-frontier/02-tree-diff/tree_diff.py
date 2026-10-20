#!/usr/bin/env python3
"""Compare two TileLang source trees and report what one has that the other does not.

The version-discipline question of chapter 30 is "is this capability in
v0.1.15, or only on main?".  This script answers it at the file level: it walks
two source trees, lists the paths that exist in only one of them, and can
optionally check whether a text pattern occurs in each side.

Usage:
    # v0.1.15 (tag) vs the main snapshot shipped in the corpus
    python3 tree_diff.py

    # explicit trees
    python3 tree_diff.py --left /path/to/v0.1.15 --right /path/to/main

    # where does a symbol live?
    python3 tree_diff.py --pattern PersistentKernel

    # assert the exact expected delta for this book's snapshot
    python3 tree_diff.py --selftest

Exit code: 0 on success, 1 when --selftest finds a mismatch.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SKIP_DIRS = {".git", "3rdparty", "build", "dist", "__pycache__", ".mypy_cache", ".pytest_cache"}
# Book-side bookkeeping files that are not part of the upstream source tree.
SKIP_FILES = {".git_commit.txt"}
TEXT_SUFFIXES = {".py", ".pyx", ".pyi", ".cc", ".h", ".cpp", ".hpp", ".md", ".rst", ".txt"}
MAX_FILE_BYTES = 2 * 1024 * 1024

# The expected delta of the locked snapshot: main (commit 994b44e, the merge of
# PR #3310) versus tag v0.1.15 (commit a35f8dd).  Everything else is identical.
EXPECTED_MAIN_ONLY = {
    "src/ascend/codegen/codegen_pto.cc",
    "src/ascend/codegen/codegen_pto.h",
    "src/ascend/codegen/rt_mod_pto.cc",
    "src/ascend/codegen/sfu_precision.h",
    "src/backend/common/codegen/codegen_py.cc",
    "src/backend/common/codegen/codegen_py.h",
    "tilelang/contrib/ptodsl/__init__.py",
    "tilelang/contrib/ptodsl/common.py",
    "tilelang/contrib/ptodsl/dcache_bypass.py",
    "tilelang/contrib/ptodsl/gemm.py",
    "tilelang/contrib/ptodsl/mixed_kernel.py",
    "tilelang/contrib/ptodsl/rng.py",
    "tilelang/contrib/ptodsl/simd_inst.py",
    "tilelang/contrib/ptodsl/simt.py",
    "tilelang/contrib/ptodsl/sync.py",
    "tilelang/jit/adapter/pto/__init__.py",
    "tilelang/jit/adapter/pto/adapter.py",
    "tilelang/jit/adapter/pto/cython_wrapper.pyx",
    "tilelang/jit/adapter/pto/libgen.py",
    "tilelang/jit/adapter/pto/wrapper.py",
}

# PR #3310 also relocated two source generators out of the CUDA tree.
EXPECTED_MAIN_REMOVED = {
    "src/cuda/codegen/codegen_py.cc",
    "src/cuda/codegen/codegen_py.h",
}


def rel_files(root: Path) -> set[str]:
    found: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in filenames:
            rel = str(Path(dirpath, name).relative_to(root))
            if rel in SKIP_FILES:
                continue
            found.add(rel)
    return found


def text_hits(root: Path, pattern: str, scope: str = ".") -> list[str]:
    hits: list[str] = []
    base = root / scope
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath, name)
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if pattern in text:
                hits.append(str(path.relative_to(root)))
    return hits


def group(paths: set[str]) -> list[tuple[str, int]]:
    buckets: dict[str, int] = {}
    for p in paths:
        top = p.split("/")[0] if "/" in p else "(root)"
        buckets[top] = buckets.get(top, 0) + 1
    return sorted(buckets.items(), key=lambda kv: (-kv[1], kv[0]))


def main() -> int:
    here = Path(__file__).resolve()
    book_root = here.parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, default=book_root / "_vendor" / "tilelang-v0.1.15")
    parser.add_argument(
        "--right",
        type=Path,
        default=book_root.parent / "tilelang-corpus" / "02-tilelang-core-repo" / "raw" / "tilelang",
    )
    parser.add_argument("--pattern", help="text to search in both trees")
    parser.add_argument("--max-list", type=int, default=40, help="cap the listing per section")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    for label, root in (("left", args.left), ("right", args.right)):
        if not root.is_dir():
            print(f"{label} tree not found: {root}", file=sys.stderr)
            return 2

    left = rel_files(args.left)
    right = rel_files(args.right)
    only_right = right - left
    only_left = left - right

    print(f"left  : {args.left}  ({len(left)} files)")
    print(f"right : {args.right}  ({len(right)} files)")
    print(f"only in right (new on main) : {len(only_right)}")
    print(f"only in left  (gone on main): {len(only_left)}")

    if only_right:
        print("\n[only in right]")
        for path in sorted(only_right)[: args.max_list]:
            print(f"  + {path}")
        if len(only_right) > args.max_list:
            print(f"  ... {len(only_right) - args.max_list} more")

    if only_left:
        print("\n[only in left]")
        for path in sorted(only_left)[: args.max_list]:
            print(f"  - {path}")
        if len(only_left) > args.max_list:
            print(f"  ... {len(only_left) - args.max_list} more")

    if args.pattern:
        hl = text_hits(args.left, args.pattern)
        hr = text_hits(args.right, args.pattern)
        print(f"\n[pattern {args.pattern!r}]")
        print(f"  left : {len(hl)} files")
        for path in hl[: args.max_list]:
            print(f"    - {path}")
        print(f"  right: {len(hr)} files")
        for path in hr[: args.max_list]:
            print(f"    - {path}")
        if not hl and not hr:
            print("  (no hits on either side)")

    if args.selftest:
        if only_right == EXPECTED_MAIN_ONLY and only_left == EXPECTED_MAIN_REMOVED:
            print("\nselftest: ok (main = v0.1.15 + the PR #3310 PTO files)")
            return 0
        print("\nselftest: MISMATCH", file=sys.stderr)
        missing = EXPECTED_MAIN_ONLY - only_right
        unexpected = only_right - EXPECTED_MAIN_ONLY
        missing_removed = EXPECTED_MAIN_REMOVED - only_left
        unexpected_removed = only_left - EXPECTED_MAIN_REMOVED
        if missing:
            print(f"  missing from right: {sorted(missing)}", file=sys.stderr)
        if unexpected:
            print(f"  unexpected in right: {sorted(unexpected)}", file=sys.stderr)
        if missing_removed:
            print(f"  expected removed but still present: {sorted(missing_removed)}", file=sys.stderr)
        if unexpected_removed:
            print(f"  unexpected removals: {sorted(unexpected_removed)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
