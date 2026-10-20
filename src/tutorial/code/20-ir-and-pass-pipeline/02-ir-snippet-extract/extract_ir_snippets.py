#!/usr/bin/env python3
"""Extract real TIR snippets from the locked v0.1.15 tree.

The point of this script is provenance: every IR fragment the chapter shows
is quoted verbatim from a file that ships with the locked release. Running the
script prints those fragments together with their source path and line range,
so a reader can re-derive the chapter's claims without a GPU/NPU and without
installing tilelang.

Sources used (all under ``_vendor/tilelang-v0.1.15/``):

1. ``examples/eager_jit/eagerjit.en.ipynb`` — Jupyter *output* cells that are
   literal ``print(prim_func)`` dumps of the frontend TIR.
2. ``src/transform/materialize_kernel_launch.cc`` — the file header documents
   the pre-materialization launch nest as an IR sketch.
3. ``docs/compiler_internals/inject_fence_proxy.md`` — a before/after pair for
   the ``InjectFenceProxy`` pass.

Usage:
    python3 extract_ir_snippets.py            # print everything
    python3 extract_ir_snippets.py --only notebook
    python3 extract_ir_snippets.py --only header
    python3 extract_ir_snippets.py --only fence

Verification level: CPU logic verified (run on 2026-10-01, macOS, no GPU/NPU).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

BOOK_ROOT = Path(__file__).resolve().parents[4]
VENDOR = BOOK_ROOT / "_vendor" / "tilelang-v0.1.15"


def _line_of(path: Path, needle: str) -> int:
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return n
    return -1


def notebook_dumps() -> list[str]:
    path = VENDOR / "examples" / "eager_jit" / "eagerjit.en.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for i, cell in enumerate(nb["cells"]):
        text = ""
        for o in cell.get("outputs", []):
            if "text" in o:
                text += "".join(o["text"])
            data = o.get("data", {})
            if "text/plain" in data:
                text += "".join(data["text/plain"])
        if "T.launch_thread" in text and "T.block(" in text:
            out.append(f"### notebook cell {i} output  ({path.relative_to(BOOK_ROOT)})\n{text.strip()}")
    return out


def header_sketch() -> str:
    path = VENDOR / "src" / "transform" / "materialize_kernel_launch.cc"
    lines = path.read_text(encoding="utf-8").splitlines()
    # The IR sketch lives inside the ``/*! ... */`` banner comment.
    start = next(i for i, l in enumerate(lines) if "T.Kernel traces into" in l)
    end = next(i for i in range(start, len(lines)) if "is bound." in lines[i])
    body = "\n".join(lines[start - 2 : end + 1])
    return (f"### pre-materialization launch nest "
            f"({path.relative_to(BOOK_ROOT)}:{start - 1}-{end + 1})\n{body}")


def fence_pair() -> str:
    path = VENDOR / "docs" / "compiler_internals" / "inject_fence_proxy.md"
    text = path.read_text(encoding="utf-8")
    section = text.split("## End-to-End Example", 1)[-1]
    blocks = re.findall(r"```python\n(.*?)```", section, flags=re.S)
    if len(blocks) < 2:
        return f"### {path}: could not find two python blocks"
    before_line = _line_of(path, "Before the pass")
    after_line = _line_of(path, "After `tl.cuda.transform.InjectFenceProxy`")
    return (
        f"### InjectFenceProxy before  ({path.relative_to(BOOK_ROOT)}:{before_line})\n"
        f"{blocks[0].strip()}\n\n"
        f"### InjectFenceProxy after   ({path.relative_to(BOOK_ROOT)}:{after_line})\n"
        f"{blocks[1].strip()}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["notebook", "header", "fence"], default=None)
    args = parser.parse_args()

    if args.only in (None, "notebook"):
        for chunk in notebook_dumps():
            print(chunk)
            print()
    if args.only in (None, "header"):
        print(header_sketch())
        print()
    if args.only in (None, "fence"):
        print(fence_pair())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
