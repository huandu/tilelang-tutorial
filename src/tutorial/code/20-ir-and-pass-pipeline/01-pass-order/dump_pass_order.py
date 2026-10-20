#!/usr/bin/env python3
"""Statically extract the pass order from a TileLang backend pipeline.

This script does NOT import tilelang and does NOT need a GPU/NPU. It only
parses the two locked pipeline modules:

    _vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py
    _vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py

For every ``mod = <pass>(...)(mod)`` statement it reports the 1-based line
number, the pass expression, and the source-level guard (the enclosing
``if``/``elif`` chain) if there is one.

Usage:
    python3 dump_pass_order.py                 # both pipelines, summary + full list
    python3 dump_pass_order.py --pipeline cuda # only one
    python3 dump_pass_order.py --names         # pass expression per line only

Verification level: CPU logic verified (run on 2026-10-01 on macOS, no GPU/NPU).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[4]
VENDOR = BOOK_ROOT / "_vendor" / "tilelang-v0.1.15"

PIPELINES = {
    "cuda": VENDOR / "tilelang" / "cuda" / "pipeline.py",
    "ascend": VENDOR / "tilelang" / "ascend" / "pipeline.py",
}

ASSIGN_RE = re.compile(r"^(?P<indent>\s*)mod\s*=\s*(?P<expr>.+)$")


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
    return depth == 0


def _guards(lines: list[str], index: int) -> str:
    """Return the ``if``/``elif``/``else`` chain guarding line ``index``.

    Walk upwards, collecting guard lines whose indentation is strictly smaller
    than the target line's indentation. Stop at the first line that is not a
    guard and is not a continuation.
    """
    target_indent = len(lines[index]) - len(lines[index].lstrip())
    guards: list[str] = []
    j = index - 1
    while j >= 0:
        raw = lines[j]
        stripped = raw.strip()
        indent = len(raw) - len(raw.lstrip())
        if stripped == "" or stripped.startswith("#"):
            j -= 1
            continue
        if indent >= target_indent:
            j -= 1
            continue
        if stripped.startswith(("if ", "elif ", "else:")):
            guards.append(stripped.rstrip(":"))
            j -= 1
            continue
        break
    return " and ".join(reversed(guards))


def extract(path: Path) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records: list[dict[str, object]] = []
    i = 0
    while i < len(lines):
        match = ASSIGN_RE.match(lines[i])
        if match is None:
            i += 1
            continue
        expr = match.group("expr")
        # Join multi-line calls until the parentheses balance and the statement
        # ends with ``(mod)`` (the pipeline mutates the module in place).
        start = i
        while not _balanced(expr) and i + 1 < len(lines):
            i += 1
            expr += " " + lines[i].strip()
        if _balanced(expr) and expr.rstrip().endswith("(mod)"):
            records.append(
                {
                    "line": start + 1,
                    "expr": re.sub(r"\s+", " ", expr).strip(),
                    "guard": _guards(lines, start),
                }
            )
        i += 1
    return records


def short_name(expr: str) -> str:
    """``tilelang.cuda.transform.Foo()`` -> ``cuda.transform.Foo``."""
    head = expr.split("(", 1)[0]
    parts = head.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else head


def leaf_name(expr: str) -> str:
    """``tilelang.transform.Simplify()`` -> ``Simplify``."""
    head = expr.split("(", 1)[0]
    return head.split(".")[-1]


def qualified_name(expr: str) -> str:
    """``tilelang.transform.Simplify()`` -> ``tilelang.transform.Simplify``.

    Keeping the module prefix distinguishes ``tilelang.transform.Simplify``
    from ``tirx.transform.Simplify``, which is the normalization used by the
    chapter's "shared pass" count.
    """
    head = expr.split("(", 1)[0]
    return ".".join(head.split(".")[-3:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default=None)
    parser.add_argument("--names", action="store_true", help="print pass names only")
    args = parser.parse_args()

    selected = [args.pipeline] if args.pipeline else sorted(PIPELINES)
    extracted: dict[str, list[dict[str, object]]] = {}

    for name in selected:
        path = PIPELINES[name]
        if not path.is_file():
            print(f"!! missing {path}")
            return 1
        records = extract(path)
        extracted[name] = records
        print(f"# {name}: {path.relative_to(BOOK_ROOT)}")
        print(f"# total pass invocations: {len(records)}")
        if args.names:
            for rec in records:
                print(f"{rec['line']:>4}  {short_name(str(rec['expr']))}")
        else:
            for n, rec in enumerate(records, start=1):
                guard = f"   [if {rec['guard']}]" if rec["guard"] else ""
                print(f"{n:>3} | L{rec['line']:>3} | {rec['expr']}{guard}")
        print()

    if len(extracted) == 2 and "cuda" in extracted and "ascend" in extracted:
        cuda_expr = {short_name(str(r["expr"])) for r in extracted["cuda"]}
        ascend_expr = {short_name(str(r["expr"])) for r in extracted["ascend"]}
        cuda_leaf = {leaf_name(str(r["expr"])) for r in extracted["cuda"]}
        ascend_leaf = {leaf_name(str(r["expr"])) for r in extracted["ascend"]}
        cuda_qual = {qualified_name(str(r["expr"])) for r in extracted["cuda"]}
        ascend_qual = {qualified_name(str(r["expr"])) for r in extracted["ascend"]}
        shared_qual = sorted(cuda_qual & ascend_qual)
        print(f"# shared pass expressions: {len(cuda_expr & ascend_expr)}")
        print(f"# expression cuda-only: {len(cuda_expr - ascend_expr)}"
              f"  ascend-only: {len(ascend_expr - cuda_expr)}")
        print(f"# shared passes (module-qualified name): {len(shared_qual)}")
        print(f"# qualified cuda-only: {len(cuda_qual - ascend_qual)}"
              f"  ascend-only: {len(ascend_qual - cuda_qual)}")
        print(f"# (leaf-name-only view, for reference): "
              f"shared {len(cuda_leaf & ascend_leaf)}")
        print()
        print("# shared passes (module-qualified, appears in both pipelines)")
        for name in shared_qual:
            print(f"  {name}")
        print()
        print("# cuda-only (module-qualified)")
        for name in sorted(cuda_qual - ascend_qual):
            print(f"  {name}")
        print()
        print("# ascend-only (module-qualified)")
        for name in sorted(ascend_qual - cuda_qual):
            print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
