#!/usr/bin/env python3
"""第 09 章的「原语 -> IR -> 目标指令」映射表，并核对正文引用的源码行号。

只用标准库；不 import tilelang，不编译 kernel。

    python3 primitive_ir_map.py
    python3 primitive_ir_map.py --no-check
    python3 primitive_ir_map.py --vendor /path/to/tilelang-v0.1.15
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cite:
    """一条源码引用：相对某个仓库根的文件路径 + 行号区间（闭区间）。"""

    root: str  # "main" 或 "adapter"
    path: str
    start: int
    end: int


@dataclass(frozen=True)
class Mapping:
    primitive: str
    ir: str
    nv: str
    ascend: str
    cites: tuple[Cite, ...]


MAIN = "main"
ADAPTER = "adapter"

MAPPINGS: tuple[Mapping, ...] = (
    Mapping(
        "T.copy",
        "tl.tileop.copy",
        "SIMT / cp.async / TMA bulk / LDSM / STSM / tcgen05.ld-st",
        "DataCopy / ND2NZ / LoadData / Fixpipe",
        (
            Cite(MAIN, "tilelang/language/copy_op.py", 59, 130),
            Cite(MAIN, "src/op/copy.cc", 481, 545),
            Cite(MAIN, "src/cuda/op/copy.h", 21, 33),
            Cite(MAIN, "src/cuda/op/copy_analysis.cc", 716, 808),
        ),
    ),
    Mapping(
        "T.async_copy",
        "tl.tileop.async_copy",
        "cp.async (+ commit, no wait)",
        "(no equivalent)",
        (
            Cite(MAIN, "tilelang/language/copy_op.py", 188, 229),
            Cite(MAIN, "src/op/copy.cc", 622, 622),
        ),
    ),
    Mapping(
        "T.tma_copy",
        "tl.tileop.tma_copy",
        "expect_tx + tma_load / tma_store (+ arrive, no wait)",
        "(no equivalent)",
        (
            Cite(MAIN, "tilelang/language/copy_op.py", 232, 324),
            Cite(MAIN, "src/op/copy.cc", 640, 640),
        ),
    ),
    Mapping(
        "T.copy_cluster",
        "tl.tileop.copy + dst_block/cluster_mask",
        "TMA multicast / SM-to-SM copy",
        "(no equivalent)",
        (Cite(MAIN, "tilelang/language/copy_op.py", 133, 185),),
    ),
    Mapping(
        "T.gemm",
        "tl.tileop.gemm",
        "cuda.mma / cuda.wgmma / cuda.tcgen05 / cuda.fma",
        "ascend.mad (L0A x L0B -> L0C)",
        (
            Cite(MAIN, "tilelang/language/gemm_op.py", 38, 122),
            Cite(MAIN, "src/cuda/op/gemm.cc", 336, 374),
            Cite(MAIN, "src/ascend/op/gemm.cc", 24, 45),
        ),
    ),
    Mapping(
        "T.gemm_blockscaled",
        "tl.tileop.gemm_blockscaled",
        "tcgen05 block-scaled variant",
        "fused MXFP8 L1 path",
        (
            Cite(MAIN, "src/op/gemm_blockscaled.cc", 105, 105),
            Cite(MAIN, "tilelang/ascend/language/gemm_op.py", 110, 165),
        ),
    ),
    Mapping(
        "T.reduce_sum / max / min",
        "tl.tileop.reduce",
        "thread-local loop + tl::AllReduce<...>::run",
        "ascend_reduce family (direct on UB inside SimdVF)",
        (
            Cite(MAIN, "tilelang/language/reduce_op.py", 25, 173),
            Cite(MAIN, "src/op/reduce.cc", 212, 217),
            Cite(MAIN, "src/tl_templates/cuda/reduce.h", 240, 253),
            Cite(MAIN, "tilelang/ascend/language/reduce_op.py", 43, 88),
        ),
    ),
    Mapping(
        "T.atomic_add",
        "tl.atomic_add_elem_op / tl.tileop.atomicadd",
        "AtomicAdd(x2/x4) / TMA cp.reduce",
        "SetAtomicAdd<T>() + GM store",
        (
            Cite(MAIN, "tilelang/language/atomic.py", 211, 311),
            Cite(MAIN, "src/cuda/op/atomic_add.cc", 36, 62),
            Cite(MAIN, "src/ascend/op/builtin.cc", 446, 451),
        ),
    ),
    Mapping(
        "T.atomic_max / atomic_min",
        "tl.tileop.atomicmax / tl.tileop.atomicmin",
        "AtomicMax / AtomicMin",
        "SetAtomicMax / SetAtomicMin",
        (
            Cite(MAIN, "tilelang/language/atomic.py", 39, 208),
            Cite(MAIN, "src/op/atomic_reduce.cc", 162, 167),
        ),
    ),
    Mapping(
        "T.atomic_load / atomic_store",
        "tl.atomic_load_elem_op / tl.atomic_store_elem_op",
        "memory-ordered load / store",
        "(no public equivalent)",
        (Cite(MAIN, "tilelang/language/atomic.py", 390, 492),),
    ),
    Mapping(
        "T.fill / T.clear",
        "tl.tileop.fill",
        "parallel loop + vectorized store",
        "parallel loop -> Fill/Duplicate",
        (
            Cite(MAIN, "tilelang/language/fill_op.py", 11, 72),
            Cite(MAIN, "src/op/fill.cc", 94, 127),
        ),
    ),
    Mapping(
        "T.transpose",
        "tl.tileop.transpose",
        "SIMT loop (no special layout inference)",
        "T.tile.transpose -> tl.ascend_transpose",
        (
            Cite(MAIN, "tilelang/language/copy_op.py", 525, 562),
            Cite(MAIN, "src/op/transpose.cc", 222, 222),
        ),
    ),
    Mapping(
        "T.tile.* (adapter only)",
        "tl.ascend_<name> intrinsic",
        "(Ascend-only)",
        "AscendC vector intrinsics",
        (
            Cite(ADAPTER, "tilelang/language/ascend_tile.py", 967, 1150),
            Cite(ADAPTER, "src/op/ascend.cc", 975, 1260),
            Cite(ADAPTER, "src/target/codegen_ascend.cc", 522, 630),
        ),
    ),
    Mapping(
        "T.dual_copy",
        "tl.tileop.ascend_copy + dual_dst_ctl",
        "(Ascend-only)",
        "hardware dual-destination (L0C->UB); per-AIV copy otherwise",
        (
            Cite(MAIN, "tilelang/ascend/language/copy_op.py", 53, 163),
            Cite(MAIN, "src/ascend/op/copy.cc", 573, 605),
            Cite(MAIN, "src/ascend/transform/rewrite_dual_copy.cc", 97, 107),
        ),
    ),
    Mapping(
        "T.SimtVF / T.SimdVF",
        "SIMT_VF / SIMD_VF block",
        "(Ascend-only)",
        "asc_vf_call thread body / MicroAPI vector ops",
        (
            Cite(MAIN, "tilelang/ascend/language/frame.py", 107, 195),
            Cite(MAIN, "src/ascend/codegen/codegen_ascend.cc", 904, 940),
        ),
    ),
)


EXTRA_CITES: tuple[Cite, ...] = (
    Cite(MAIN, "tilelang/jit/__init__.py", 507, 509),          # get_kernel_source
    Cite(MAIN, "tilelang/jit/kernel.py", 532, 534),            # get_kernel_source
    Cite(MAIN, "src/transform/lower_tile_op.cc", 207, 224),    # LowerTileOpPass
    Cite(MAIN, "src/transform/lower_tile_op.cc", 1477, 1486),  # tl.LowerTileOp
    Cite(MAIN, "src/op/copy.cc", 530, 545),                    # coalesced_width / loop_layout
    Cite(MAIN, "src/op/copy.cc", 508, 509),                    # dtype cast in SIMT copy
)


def _default_book_root() -> Path:
    # .../chapters/09-core-primitives/code/01-primitive-ir-map/primitive_ir_map.py
    return Path(__file__).resolve().parents[4]


def _column(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "~"


def print_table() -> None:
    widths = (26, 28, 34, 40)
    header = ("primitive", "IR", "NV target", "Ascend target")
    print(" | ".join(_column(h, w).ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for m in MAPPINGS:
        row = (m.primitive, m.ir, m.nv, m.ascend)
        print(" | ".join(_column(c, w).ljust(w) for c, w in zip(row, widths)))


def check_citations(vendor: Path, adapter: Path) -> int:
    total = 0
    bad: list[str] = []
    skipped = not (vendor.is_dir() and adapter.is_dir())
    if skipped:
        print(
            "checked 0 source citations -> skipped "
            f"(vendor={vendor} exists={vendor.is_dir()}, "
            f"adapter={adapter} exists={adapter.is_dir()})"
        )
        return 0

    labelled: list[tuple[str, Cite]] = []
    for m in MAPPINGS:
        labelled.extend((m.primitive, c) for c in m.cites)
    labelled.extend(("(chapter 5.3)", c) for c in EXTRA_CITES)

    for label, c in labelled:
        total += 1
        root = vendor if c.root == MAIN else adapter
        target = root / c.path
        if not target.is_file():
            bad.append(f"{label}: missing file {c.root}:{c.path}")
            continue
        n_lines = sum(1 for _ in target.open("r", encoding="utf-8", errors="ignore"))
        if not (1 <= c.start <= c.end <= n_lines):
            bad.append(
                f"{label}: {c.root}:{c.path}:{c.start}-{c.end} "
                f"out of range (file has {n_lines} lines)"
            )
    if bad:
        print(f"checked {total} source citations -> {len(bad)} problem(s)")
        for item in bad:
            print(f"  - {item}")
        return 1
    print(f"checked {total} source citations -> all in range")
    return 0


def check_tile_namespace_absence(vendor: Path) -> int:
    """复现「v0.1.15 里没有 T.tile 命名空间」这条结论。"""
    lang_dir = vendor / "tilelang" / "ascend" / "language"
    if not lang_dir.is_dir():
        print("claim \"T.tile absent from v0.1.15\": skipped (vendor missing)")
        return 0
    if (vendor / "tilelang" / "language" / "ascend_tile.py").exists():
        print("claim \"T.tile absent from v0.1.15\": CONTRADICTED (ascend_tile.py exists)")
        return 1
    hits = []
    for py in sorted(lang_dir.glob("*.py")):
        text = py.read_text(encoding="utf-8", errors="ignore")
        if "as tile" in text:
            hits.append(py.name)
    if hits:
        print(f"claim \"T.tile absent from v0.1.15\": CONTRADICTED ({hits})")
        return 1
    print(
        "claim \"T.tile absent from v0.1.15\": confirmed "
        "(no ascend_tile.py, no `as tile` import under tilelang/ascend/language)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-check", action="store_true", help="only print the mapping table")
    parser.add_argument("--vendor", type=Path, default=None, help="path to tilelang-v0.1.15")
    parser.add_argument(
        "--adapter", type=Path, default=None, help="path to tilelang-ascend-v0.1.2.000"
    )
    args = parser.parse_args()

    print_table()
    if args.no_check:
        return 0

    book = _default_book_root()
    vendor = args.vendor or (book / "_vendor" / "tilelang-v0.1.15")
    adapter = args.adapter or (book / "_vendor" / "tilelang-ascend-v0.1.2.000")

    problems = check_citations(vendor, adapter)
    problems += check_tile_namespace_absence(vendor)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
