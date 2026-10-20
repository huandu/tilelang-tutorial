#!/usr/bin/env python3
"""Extract the Ascend scope / DMA-path / pipe tables and cross-check them.

Everything is parsed out of the pinned v0.1.15 tree; nothing is imported and
nothing is executed.  Exit code 1 means one of the four consistency checks
failed, which would invalidate a table in chapter 25.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CODEGEN = "src/ascend/codegen/codegen_ascend.cc"
PIPE_H = "src/ascend/transform/ascend_pipe.h"
OOB_H = "src/ascend/op/oob_padding.h"
OOB_CC = "src/ascend/op/oob_padding.cc"
COPY_CC = "src/ascend/op/copy.cc"
ALLOC_PY = "tilelang/ascend/language/allocate.py"


def find_book_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / "_vendor" / "tilelang-v0.1.15").is_dir():
            return cur
        cur = cur.parent
    raise SystemExit("cannot locate book root")


def read(root: Path, rel: str) -> str:
    return (root / "_vendor" / "tilelang-v0.1.15" / rel).read_text(encoding="utf-8")


def parse_scope_qualifiers(text: str) -> dict[str, str]:
    body = re.search(r"GetAscendScopeQualifier\(const std::string &scope\)\s*\{(.*?)\n\}",
                     text, re.S)
    if body is None:
        raise SystemExit("GetAscendScopeQualifier not found")
    result: dict[str, str] = {}
    # Handles both `scope == "x"` and `scope == "x" || scope == "x.dyn"`.
    for condition, qual in re.findall(r'if \((.*?)\)\s*\n\s*return "([^"]+)"', body.group(1)):
        for scope in re.findall(r'scope == "([^"]+)"', condition):
            result[scope] = qual
    return result


def parse_dma_paths(h_text: str) -> list[str]:
    enum = re.search(r"enum class DMAPath\s*\{(.*?)\};", h_text, re.S)
    if enum is None:
        raise SystemExit("DMAPath enum not found")
    return re.findall(r"\b(k[A-Za-z0-9]+)", enum.group(1))


def parse_alloc_scopes(py_text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for fn, scope in re.findall(r"def (alloc_[a-z0-9_]+)\(.*?scope=\"?([a-z0-9.]+)\"?", py_text, re.S):
        result[fn] = scope
    # alloc_shared's default is a keyword default on one line.
    m = re.search(r"def alloc_shared\([^)]*scope=\"([^\"]+)\"", py_text, re.S)
    if m:
        result["alloc_shared"] = m.group(1)
    return result


def parse_pipes(h_text: str) -> dict[str, list[str]]:
    body = re.search(r"enum class ResourcePipe[^{]*\{(.*?)\n\};", h_text, re.S)
    if body is None:
        raise SystemExit("ResourcePipe enum not found")
    result: dict[str, list[str]] = {}
    for line in body.group(1).splitlines():
        m = re.match(r"\s*(k\w+)\s*=\s*1\s*<<\s*(\d+),\s*//\s*(.*)", line)
        if m is None:
            continue
        name, _, comment = m.groups()
        result[name] = [part.strip() for part in re.split(r"[/<]", comment) if part.strip()]
    return result


# The pipe that owns each DMA path, from the DMAPath enum comment block in
# oob_padding.h plus ResourcePipe's own comments.  Kept as data so a drift in
# either file fails the check instead of silently changing the chapter's table.
EXPECTED_PIPE = {
    "kGMToUB": "kMTE2",
    "kUBToGM": "kMTE3",
    "kGMToL1": "kMTE2",
    "kL1ToL0A": "kMTE1",
    "kL1ToL0B": "kMTE1",
    "kL1ToL0ASF": "kMTE1",
    "kL1ToL0BSF": "kMTE1",
    "kL0CToUB": "kFixpipe",
    "kUBToL1": "kMTE3",
    "kL0CToGM": "kFixpipe",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-root", type=Path, default=None)
    args = parser.parse_args()
    root = args.book_root or find_book_root(Path(__file__).parent)

    qualifiers = parse_scope_qualifiers(read(root, CODEGEN))
    dma_paths = parse_dma_paths(read(root, OOB_H))
    alloc_scopes = parse_alloc_scopes(read(root, ALLOC_PY))
    pipes = parse_pipes(read(root, PIPE_H))
    copy_cc = read(root, COPY_CC)
    print("== scope -> CCE qualifier ==")
    labels = {
        "shared": "UB (dynamic shared)",
        "shared.dyn": "UB (dynamic shared)",
        "shared.l1": "L1 (CBuf)",
        "shared.l0a": "L0A (CA)",
        "shared.l0b": "L0B (CB)",
        "shared.l0c": "L0C (CC)",
    }
    for scope, qual in sorted(qualifiers.items()):
        print(f"  {scope:<24} -> {qual:<10} {labels.get(scope, '')}")
    print()

    print("== alloc_* -> scope ==")
    for fn, scope in sorted(alloc_scopes.items()):
        print(f"  {fn:<16} -> {scope}")
    print()

    print("== DMAPath -> owning pipe ==")
    for path in dma_paths:
        if path == "kNone":
            continue
        pipe = EXPECTED_PIPE.get(path, "?")
        comment = pipes.get(pipe, [])
        print(f"  {path:<14} {pipe:<10} {'/'.join(comment)}")
    print()

    failures: list[str] = []

    # 1. every alloc scope is translatable by codegen
    missing_scopes = sorted(set(alloc_scopes.values()) - set(qualifiers))
    if missing_scopes:
        failures.append(f"alloc scopes without a codegen qualifier: {missing_scopes}")

    # 2. every non-None DMA path is really lowered in copy.cc
    for path in dma_paths:
        if path == "kNone":
            continue
        if f"DMAPath::{path}" not in copy_cc:
            failures.append(f"{path} is never referenced in {COPY_CC}")

    # 3. the expected pipe table covers the enum exactly
    concrete = [p for p in dma_paths if p != "kNone"]
    if sorted(concrete) != sorted(EXPECTED_PIPE):
        failures.append(
            "DMAPath set changed: "
            f"enum={sorted(concrete)} expected={sorted(EXPECTED_PIPE)}"
        )
    for path, pipe in sorted(EXPECTED_PIPE.items()):
        if pipe not in pipes:
            failures.append(f"{path} maps to unknown pipe {pipe}")

    # 4. every scope in the qualifier map is a literal in codegen
    codegen = read(root, CODEGEN)
    for scope, qual in qualifiers.items():
        if f'"{scope}"' not in codegen or f'"{qual}"' not in codegen:
            failures.append(f"scope/qualifier literal missing for {scope} -> {qual}")

    # extra: GetDMAPath body should dispatch on the same scopes
    op_utils = read(root, "src/ascend/op/utils.h")
    for scope in ("shared.l1", "shared.l0a", "shared.l0b", "shared.l0c"):
        if scope not in op_utils:
            failures.append(f"{scope} predicate missing in src/ascend/op/utils.h")

    if failures:
        for f in failures:
            print(f"[FAIL] {f}")
        return 1
    print("[ok] 4 consistency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
