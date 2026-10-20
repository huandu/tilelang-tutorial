#!/usr/bin/env python3
"""Build the Ascend API presence matrix for the two pinned trees.

Read-only.  The point is that chapter 25 must not say "this API exists in
v0.1.15" unless it really does; the four EXPECTED assertions below are the
ones the chapter leans on most.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Api:
    label: str
    pattern: str
    m_files: tuple[str, ...]
    a_files: tuple[str, ...]


LANG = "tilelang/ascend/language"
ALANG = "tilelang/language"


APIS: tuple[Api, ...] = (
    Api("target 'ascend'/'asc'", r'target\.strip\(\)\s*==\s*"asc"',
        ("tilelang/ascend/target.py",), ()),
    Api("target 'ascendc'", r'ascendc',
        (), (f"{ALANG}/../utils/target.py", "tilelang/engine/lower.py")),
    Api("target 'pto'", r'"pto"',
        ("tilelang/ascend/target.py",), ("tilelang/engine/lower.py",)),
    Api("target kind 'npuir'", r'npuir',
        (), ()),
    Api("alloc_shared", r"def alloc_shared\(",
        (f"{LANG}/allocate.py",), (f"{ALANG}/allocate.py",)),
    Api("alloc_l1 (lowercase)", r"def alloc_l1\(",
        (f"{LANG}/allocate.py",), ()),
    Api("alloc_L1 (uppercase)", r"def alloc_L1\(",
        (), (f"{ALANG}/allocate.py",)),
    Api("alloc_l0a (lowercase)", r"def alloc_l0a\(",
        (f"{LANG}/allocate.py",), ()),
    Api("alloc_L0A (uppercase)", r"def alloc_L0A\(",
        (), (f"{ALANG}/allocate.py",)),
    Api("alloc_l0c (lowercase)", r"def alloc_l0c\(",
        (f"{LANG}/allocate.py",), ()),
    Api("alloc_L0C (uppercase)", r"def alloc_L0C\(",
        (), (f"{ALANG}/allocate.py",)),
    Api("alloc_ub", r"def alloc_ub\(",
        (f"{LANG}/allocate.py",), (f"{ALANG}/allocate.py",)),
    Api("alloc_l0a_sf", r"def alloc_l0a_sf\(",
        (f"{LANG}/allocate.py",), ()),
    Api("copy", r"def copy\(",
        (f"{LANG}/copy_op.py",), (f"{ALANG}/copy_op.py",)),
    Api("dual_copy", r"def dual_copy\(",
        (f"{LANG}/copy_op.py",), (f"{ALANG}/copy.py", f"{LANG}/copy.py")),
    Api("gemm", r"def gemm\(",
        (f"{LANG}/gemm_op.py",), (f"{ALANG}/gemm.py", f"{LANG}/gemm.py")),
    Api("gemm_v0", r"def gemm_v0\(",
        (f"{LANG}/gemm_op.py",), (f"{ALANG}/ascend.py",)),
    Api("gemm_blockscaled", r"def gemm_blockscaled\(",
        (f"{LANG}/gemm_op.py",), ()),
    Api("SimtVF", r"def SimtVF\(",
        (f"{LANG}/frame.py",), (f"{ALANG}/frame.py", f"{LANG}/frame.py")),
    Api("SimdVF", r"def SimdVF\(",
        (f"{LANG}/frame.py",), (f"{ALANG}/frame.py", f"{LANG}/frame.py")),
    Api("Cube / Vector region", r"def Vector\(|def Cube\(",
        (f"{LANG}/frame.py",), (f"{ALANG}/warpgroup.py",)),
    Api("Pipelined", r"def Pipelined\(",
        ("tilelang/language/loop.py",), (f"{ALANG}/pipeline.py",)),
    Api("Persistent", r"def Persistent\(",
        ("tilelang/language/loop.py",), (f"{ALANG}/persistent.py",)),
    Api("annotate_buffer_versions", r"def annotate_buffer_versions\(",
        (f"{LANG}/annotations.py",), ()),
    Api("annotate_manual_multi_buffer", r"def annotate_manual_multi_buffer\(",
        (f"{LANG}/annotations.py",), ()),
    Api("T.tile namespace (ascend_tile alias)", r"import ascend_tile as tile",
        (f"{LANG}/__init__.py",), (f"{ALANG}/__init__.py",)),
    Api("T.tile.* op (ascend_tile body)", r"def row_expand_mul|def sort32",
        (f"{LANG}/tile.py",), (f"{ALANG}/ascend_tile.py",)),
    Api("set_flag / wait_flag (user-facing)", r"def (set|wait)_flag\(",
        (f"{LANG}/sync.py",), (f"{ALANG}/ascend.py",)),
    Api("barrier_all", r"def barrier_all\(",
        (f"{LANG}/sync.py",), (f"{ALANG}/ascend.py",)),
    Api("printf", r"def printf\(",
        (f"{LANG}/print.py",), (f"{ALANG}/ascend.py",)),
    Api("dump_tensor", r"def dump_tensor\(",
        (f"{LANG}/print.py",), (f"{ALANG}/ascend.py",)),
    Api("device_assert", r"def device_assert\(",
        ("tilelang/ascend/debug.py",), ()),
    Api("pass key TL_ENABLE_AUTO_SCHEDULE", r"TL_ENABLE_AUTO_SCHEDULE",
        ("tilelang/transform/pass_config.py",), ()),
    Api("pass key TL_ASCEND_AUTO_SYNC", r"TL_ASCEND_AUTO_SYNC",
        (), ("tilelang/transform/pass_config.py",)),
    Api("artifact: AscendC::TPipe", r"AscendC::TPipe",
        (), ("src/target/codegen_ascend.cc",)),
    Api("artifact: asc_copy_gm2ub_align", r"asc_copy_gm2ub_align",
        ("src/ascend/codegen/codegen_ascend.cc",), ()),
    Api("artifact: pto::Tile", r"pto::Tile",
        (), ("src/tl_templates/pto/common.h",)),
    Api("artifact: asc_mmad (L0 MAD)", r"asc_mmad",
        ("src/ascend/codegen/codegen_ascend.cc",), ()),
)

# Chapter 25 asserts these specific outcomes.
EXPECTED_M_ONLY = {"dual_copy", "SimtVF", "SimdVF", "annotate_buffer_versions",
                   "device_assert", "pass key TL_ENABLE_AUTO_SCHEDULE"}
EXPECTED_A_ONLY = {"alloc_ub", "alloc_L1 (uppercase)", "alloc_L0A (uppercase)",
                   "alloc_L0C (uppercase)", "gemm_v0", "dump_tensor",
                   "pass key TL_ASCEND_AUTO_SYNC", "T.tile namespace (ascend_tile alias)",
                   "T.tile.* op (ascend_tile body)", "artifact: AscendC::TPipe",
                   "artifact: pto::Tile"}


def find_book_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / "_vendor" / "tilelang-v0.1.15").is_dir():
            return cur
        cur = cur.parent
    raise SystemExit("cannot locate book root")


def locate(tree_root: Path, files: tuple[str, ...], pattern: str) -> str | None:
    rx = re.compile(pattern)
    for rel in files:
        path = tree_root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        m = rx.search(text)
        if m:
            line = text[: m.start()].count("\n") + 1
            return f"{rel}:{line}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-root", type=Path, default=None)
    args = parser.parse_args()
    book = args.book_root or find_book_root(Path(__file__).parent)
    m_tree = book / "_vendor" / "tilelang-v0.1.15"
    a_tree = book / "_vendor" / "tilelang-ascend-v0.1.2.000"
    mlir_tree = (book / ".." / "tilelang-corpus" / "07-tilelang-ascend-adapter"
                 / "raw" / "tilelang-mlir-ascend").resolve()

    header = f"{'api':<38} {'M tree (v0.1.15)':<44} {'A tree (v0.1.2.000)':<44}"
    print(header)
    print("-" * len(header))
    problems: list[str] = []
    for api in APIS:
        m_hit = locate(m_tree, api.m_files, api.pattern)
        a_hit = locate(a_tree, api.a_files, api.pattern)
        print(f"{api.label:<38} {m_hit or 'MISSING':<44} {a_hit or 'MISSING':<44}")
        if api.label in EXPECTED_M_ONLY and (m_hit is None or a_hit is not None):
            problems.append(f"{api.label}: expected M-only, got M={m_hit} A={a_hit}")
        if api.label in EXPECTED_A_ONLY and (a_hit is None or m_hit is not None):
            problems.append(f"{api.label}: expected A-only, got M={m_hit} A={a_hit}")

    print()
    npuir_hit = locate(mlir_tree, ("tilelang/engine/lower.py",), r'kind\.name == "npuir"')
    pto_hit = locate(a_tree, ("tilelang/engine/lower.py",), r'target\.model == "pto"')
    m_pto_hit = locate(m_tree, ("tilelang/ascend/pipeline.py", "tilelang/ascend/target.py"),
                       r'ascend_pto|"pto"')
    print(f"npuir route      (MLIR repo) : {npuir_hit or 'MISSING'}")
    print(f"pto route        (A tree)    : {pto_hit or 'MISSING'}")
    print(f"pto route        (M tree)    : {m_pto_hit or 'MISSING'}")
    if npuir_hit is None:
        problems.append("npuir route missing from the MLIR corpus repo")
    if pto_hit is None:
        problems.append("pto route missing from the A tree")
    if m_pto_hit is not None:
        problems.append("v0.1.15 unexpectedly contains a PTO target")

    if problems:
        for p in problems:
            print(f"[FAIL] {p}")
        return 1
    print(f"[ok] 0 unexpected results ({len(APIS)} APIs + 3 route checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
