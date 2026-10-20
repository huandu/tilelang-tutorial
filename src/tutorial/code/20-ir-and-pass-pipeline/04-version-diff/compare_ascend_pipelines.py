#!/usr/bin/env python3
"""Compare the Ascend pass vocabulary across the two locked repositories.

The book locks two Ascend sources:

    _vendor/tilelang-v0.1.15/            tilelang/ascend/pipeline.py   (in-tree)
    _vendor/tilelang-ascend-v0.1.2.000/  tilelang/engine/phase.py      (adapter)

They are *not* the same pipeline. This script extracts the pass names from both
and prints the difference, plus a targeted check for the ND->NZ pass name.

Usage:
    python3 compare_ascend_pipelines.py
    python3 compare_ascend_pipelines.py --grep InsertNd

Verification level: CPU logic verified (run on 2026-10-01, macOS, no GPU/NPU).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[4]
MAIN = BOOK_ROOT / "_vendor" / "tilelang-v0.1.15" / "tilelang" / "ascend" / "pipeline.py"
ADAPTER = BOOK_ROOT / "_vendor" / "tilelang-ascend-v0.1.2.000" / "tilelang" / "engine" / "phase.py"
CORPUS_MAIN = (
    BOOK_ROOT.parent
    / "tilelang-corpus"
    / "02-tilelang-core-repo"
    / "raw"
    / "tilelang"
    / "tilelang"
    / "ascend"
    / "pipeline.py"
)

CALL_RE = re.compile(r"\b(?:tilelang|tir|ascend_transform|transform)\.([A-Z][A-Za-z0-9_]*)\s*\(")


def pass_names(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    # Drop line comments so commented-out passes are not counted.
    text = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    names: list[str] = []
    for match in CALL_RE.finditer(text):
        name = match.group(1)
        if name in {"Target", "IRModule"}:
            continue
        if name not in names:
            names.append(name)
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grep", default=None, help="substring to search in both trees")
    parser.add_argument("--compare-main", action="store_true",
                        help="also diff the in-tree pipeline against the corpus main snapshot")
    args = parser.parse_args()

    main_names = pass_names(MAIN)
    adapter_names = pass_names(ADAPTER)

    print(f"# in-tree Ascend pipeline: {MAIN.relative_to(BOOK_ROOT)}  ({len(main_names)} names)")
    print(f"# adapter pipeline:        {ADAPTER.relative_to(BOOK_ROOT)}  ({len(adapter_names)} names)")
    print()
    print("# only in the in-tree v0.1.15 Ascend pipeline")
    for name in main_names:
        if name not in adapter_names:
            print(f"  {name}")
    print()
    print("# only in the adapter v0.1.2.000 pipeline")
    for name in adapter_names:
        if name not in main_names:
            print(f"  {name}")

    if args.grep:
        print()
        print(f"# files under the two trees whose text contains {args.grep!r}")
        for root in (
            BOOK_ROOT / "_vendor" / "tilelang-v0.1.15",
            BOOK_ROOT / "_vendor" / "tilelang-ascend-v0.1.2.000",
        ):
            for path in sorted(root.rglob("*")):
                if not path.is_file() or path.suffix not in {".py", ".cc", ".h", ".md"}:
                    continue
                if any(part in {".git", "3rdparty"} for part in path.parts):
                    continue
                if args.grep in path.read_text(encoding="utf-8", errors="ignore"):
                    print(f"  {path.relative_to(BOOK_ROOT)}")

    if args.compare_main:
        print()
        if not CORPUS_MAIN.is_file():
            print(f"# corpus main snapshot not found: {CORPUS_MAIN}")
            return 0
        corpus_names = pass_names(CORPUS_MAIN)
        print(f"# corpus main snapshot pipeline: {CORPUS_MAIN}  ({len(corpus_names)} names)")
        only_main = [n for n in main_names if n not in corpus_names]
        only_corpus = [n for n in corpus_names if n not in main_names]
        print(f"# only in v0.1.15: {only_main if only_main else '(none)'}")
        print(f"# only in corpus main: {only_corpus if only_corpus else '(none)'}")
        print("# interpretation: an empty pair of diffs means every pass named above is")
        print("# already in the release, i.e. none of them is 'main-only'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
