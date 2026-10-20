#!/usr/bin/env python3
"""Extract the real pass order of the CUDA and Ascend pipelines.

This script does **not** import tilelang. It reads the two pipeline source files
line by line and reports, in source order:

  * the pass object that is called (``mod = <call>(...)``),
  * the 1-based line number,
  * the enclosing ``if`` guard, when the call is conditional.

Usage:
    python3 dump_pass_order.py                  # use the vendored v0.1.15 tree
    python3 dump_pass_order.py --vendor /path/to/tilelang-v0.1.15
    python3 dump_pass_order.py --format csv

Exit code 0 means both files were found and parsed. The script performs no
network access and needs no GPU.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
# .../chapters/04-architecture-and-compilation-flow/code/pass-order/dump_pass_order.py
BOOK_ROOT = HERE.parents[4]

PIPELINES = (
    ("CUDA", "tilelang/cuda/pipeline.py"),
    ("Ascend", "tilelang/ascend/pipeline.py"),
)

ASSIGN_RE = re.compile(r"^(\s*)mod = ([A-Za-z_][\w.]*)\s*\(")
IF_RE = re.compile(r"^    if (.+):\s*$")
DEF_RE = re.compile(r"^def ([A-Za-z_][\w]*)\(")


def parse_regions(source: str) -> dict[str, list[tuple[int, str, str | None]]]:
    """Split a pipeline module into per-function regions of pass applications.

    Only ``mod = some.pass(...)`` lines count as pass applications; helper calls
    such as ``pass_ctx = get_pass_context()`` are ignored on purpose.
    """
    regions: dict[str, list[tuple[int, str, str | None]]] = {}
    current: str | None = None
    guard: str | None = None

    for lineno, raw in enumerate(source.splitlines(), start=1):
        match_def = DEF_RE.match(raw)
        if match_def is not None:
            current = match_def.group(1)
            guard = None
            regions.setdefault(current, [])
            continue

        if current is None:
            continue

        if not raw.strip() or raw.strip().startswith("#"):
            continue

        match_if = IF_RE.match(raw)
        if match_if is not None:
            guard = match_if.group(1).strip()
            continue

        indent = len(raw) - len(raw.lstrip(" "))
        if indent <= 4:
            # Back at function-body level: any previous guard has closed.
            guard = None

        match_assign = ASSIGN_RE.match(raw)
        if match_assign is not None:
            regions[current].append((lineno, match_assign.group(2), guard))

    return {name: entries for name, entries in regions.items() if entries}


def build_sequences(source: str) -> dict[str, list[tuple[int, str, str | None]]]:
    """Compose the effective pipeline sequence(s) of a pipeline module."""
    regions = parse_regions(source)
    if len(regions) <= 1:
        (name, entries), *_ = list(regions.items())
        return {name: [(line, call, guard) for line, call, guard in entries if call != "CUDAPassPipelineBodyPrologue"]}

    # CUDA: the body calls the prologue first, so the effective order is
    # prologue entries followed by the body entries.
    prologue = next((v for k, v in regions.items() if k.endswith("Prologue")), [])
    body = next(v for k, v in regions.items() if k.endswith("Body") and not k.endswith("Prologue"))
    ordered = list(prologue) + [e for e in body if e[1] != "CUDAPassPipelineBodyPrologue"]
    return {"CUDAPassPipelineBody (via Prologue + Body)": ordered}


def render_markdown(name: str, path: Path, entries: list[tuple[int, str, str | None]]) -> str:
    lines = [f"## {name} — {path.name}（{len(entries)} 个 pass 调用）", ""]
    lines.append("| # | 行号 | pass | 条件 |")
    lines.append("| --- | --- | --- | --- |")
    for index, (line, call, guard) in enumerate(entries, start=1):
        lines.append(f"| {index} | {line} | `{call}` | {guard or '（无条件）'} |")
    return "\n".join(lines)


def render_csv(name: str, entries: list[tuple[int, str, str | None]]) -> str:
    lines = ["pipeline,index,line,call,guard"]
    for index, (line, call, guard) in enumerate(entries, start=1):
        safe_guard = (guard or "").replace(",", ";")
        lines.append(f"{name},{index},{line},{call},{safe_guard}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", default=str(BOOK_ROOT / "_vendor" / "tilelang-v0.1.15"))
    parser.add_argument("--format", choices=("markdown", "csv"), default="markdown")
    args = parser.parse_args()

    vendor = Path(args.vendor)
    if not vendor.is_dir():
        print(f"vendor tree not found: {vendor}", file=sys.stderr)
        return 2

    parsed: dict[str, list[tuple[int, str, str | None]]] = {}
    for name, relative in PIPELINES:
        target = vendor / relative
        if not target.is_file():
            print(f"missing pipeline file: {target}", file=sys.stderr)
            return 2
        sequences = build_sequences(target.read_text(encoding="utf-8"))
        entries = next(iter(sequences.values()))
        parsed[name] = entries
        if args.format == "markdown":
            print(render_markdown(name, target, entries))
            print()
        else:
            print(render_csv(name, entries))

    if args.format != "markdown":
        return 0

    cuda_calls = [call for _, call, _ in parsed["CUDA"]]
    ascend_calls = [call for _, call, _ in parsed["Ascend"]]
    shared = sorted({c for c in cuda_calls} & {c for c in ascend_calls})
    print("## 汇总", "")
    print(f"* CUDA 共 {len(cuda_calls)} 步，Ascend 共 {len(ascend_calls)} 步。")
    print(f"* 两侧都出现的 pass 名共 {len(shared)} 个：{', '.join('`' + c + '`' for c in shared)}")
    conditional_cuda = sum(1 for _, _, guard in parsed["CUDA"] if guard)
    conditional_ascend = sum(1 for _, _, guard in parsed["Ascend"] if guard)
    print(f"* 带条件守卫：CUDA {conditional_cuda} 步，Ascend {conditional_ascend} 步。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
