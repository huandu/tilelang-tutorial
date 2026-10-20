#!/usr/bin/env python3
"""Print the four Ascend compilation routes and assert their source anchors.

Reads only. Never imports tilelang, never invokes a compiler.

Routes covered (locked versions):

  M1  tilelang v0.1.15,          target="ascend" / "asc"   -> AscendC (asc_*) CCE source
  A1  tilelang-ascend v0.1.2.000 target="ascendc"          -> AscendC class-library C++
  A2  tilelang-ascend v0.1.2.000 target="pto"              -> PTO-DSL C++ (pto::Tile)
  L1  tilelang-mlir-ascend       target kind "npuir"       -> AscendNPU IR (MLIR text)

For every cell we keep the file and the substring that justifies it.  The
script fails loudly if an anchor disappears, so the chapter's claims cannot
silently drift away from the pinned trees.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Route:
    key: str
    tree: str
    entry: str
    artifact: str
    compiler: str
    anchors: tuple[tuple[str, str], ...]


ROUTES: tuple[Route, ...] = (
    Route(
        key="M1",
        tree="_vendor/tilelang-v0.1.15 (M tree, mainline)",
        entry='target="ascend" | "asc"',
        artifact="AscendC (asc_* intrinsic) CCE source",
        compiler="bisheng --npu-arch=dav-3510 -> linked CCE ELF (aibin)",
        anchors=(
            ("tilelang/ascend/target.py", 'target.strip() != "ascend"'),
            ("tilelang/ascend/target.py", 'target.strip() == "asc"'),
            ("tilelang/ascend/codegen.py", '"target.build.tilelang_ascend"'),
            ("tilelang/ascend/backend.py", "tilelang_callback_ascend_compile"),
            ("src/ascend/codegen/codegen_ascend.cc", "asc_copy_gm2ub_align"),
            ("tilelang/contrib/bisheng.py", 'os.environ.get("ASCEND_NPU_ARCH", "dav-3510")'),
            ("tilelang/contrib/bisheng.py", '"aibin"'),
        ),
    ),
    Route(
        key="A1",
        tree="_vendor/tilelang-ascend-v0.1.2.000 (A tree, adapter)",
        entry='target="ascendc"',
        artifact="AscendC class-library C++ (AscendC::TPipe / TBuf)",
        compiler="bisheng --npu-arch=dav-2201 -xasc --shared -> .so",
        anchors=(
            ("tilelang/utils/target.py", 'target in ["ascendc", "pto"]'),
            ("tilelang/engine/lower.py", 'target.model == "ascendc"'),
            ("tilelang/jit/adapter/libgen.py", '"--npu-arch=dav-2201"'),
            ("tilelang/jit/adapter/libgen.py", '"-xasc"'),
            ("src/target/codegen_ascend.cc", "AscendC::TPipe pipe"),
            ("src/target/codegen_ascend.cc", "AscendC::TBuf<AscendC::TPosition::A1>"),
        ),
    ),
    Route(
        key="A2",
        tree="_vendor/tilelang-ascend-v0.1.2.000 (A tree, adapter)",
        entry='target="pto"',
        artifact="PTO-DSL C++ (pto::Tile<...>, TLOAD/TSTORE/TMATMUL templates)",
        compiler="bisheng --cce-aicore-arch=dav-c220|dav-c310 -xcce -I 3rdparty/pto-isa/include -> .so",
        anchors=(
            ("tilelang/engine/lower.py", 'target.model == "pto"'),
            ("tilelang/engine/lower.py", '"target.build.tilelang_ascend_pto"'),
            ("tilelang/jit/adapter/libgen.py", '"--cce-aicore-arch={ccec}"'),
            ("tilelang/jit/adapter/libgen.py", 'f"-I{TL_ROOT}/3rdparty/pto-isa/include"'),
            ("src/target/codegen_ascend_pto.cc", "#include <pto/pto-inst.hpp>"),
            ("src/tl_templates/pto/common.h", "pto::Tile<"),
        ),
    ),
    Route(
        key="L1",
        tree="../tilelang-corpus/07-.../raw/tilelang-mlir-ascend (AscendNPU IR)",
        entry='TVM target kind "npuir"',
        artifact="AscendNPU IR (npuir / HIVM MLIR text)",
        compiler="bishengir-compile --enable-hivm-compile=true ... -> .o",
        anchors=(
            ("tilelang/engine/lower.py", 'target.kind.name == "npuir"'),
            ("tilelang/engine/lower.py", '"target.build.tilelang_npuir_apis"'),
            ("tilelang/jit/jit_npu.py", "bishengir-compile"),
            ("tilelang/jit/jit_npu.py", '"--enable-hivm-compile=true"'),
            ("tilelang/utils/npu_utils.py", "bishengir-opt"),
        ),
    ),
)


def find_book_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / "_vendor" / "tilelang-v0.1.15").is_dir():
            return cur
        cur = cur.parent
    raise SystemExit("cannot locate book root (no _vendor/tilelang-v0.1.15 upward)")


def anchor_path(book: Path, route: Route, rel: str) -> Path:
    if rel.startswith("../tilelang-corpus/"):
        return (book / rel).resolve()
    if route.key == "L1":
        return (
            book
            / ".."
            / "tilelang-corpus"
            / "07-tilelang-ascend-adapter"
            / "raw"
            / "tilelang-mlir-ascend"
            / rel
        ).resolve()
    if route.key == "M1":
        return book / "_vendor" / "tilelang-v0.1.15" / rel
    return book / "_vendor" / "tilelang-ascend-v0.1.2.000" / rel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book-root", type=Path, default=None)
    args = parser.parse_args()
    book = args.book_root or find_book_root(Path(__file__).parent)

    misses = 0
    header = f"{'route':<6} {'entry':<26} {'artifact':<52} {'compiler':<58}"
    print(header)
    print("-" * len(header))
    for route in ROUTES:
        print(f"{route.key:<6} {route.entry:<26} {route.artifact:<52} {route.compiler:<58}")
    print()

    for route in ROUTES:
        print(f"[{route.key}] {route.tree}")
        for rel, needle in route.anchors:
            path = anchor_path(book, route, rel)
            if not path.is_file():
                print(f"  MISS  {rel}: file not found")
                misses += 1
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if needle in text:
                print(f"  ok    {rel}: {needle!r}")
            else:
                print(f"  MISS  {rel}: missing anchor {needle!r}")
                misses += 1
        print()

    total = sum(len(r.anchors) for r in ROUTES)
    if misses:
        print(f"[FAIL] {len(ROUTES)} routes, {misses}/{total} anchor misses")
        return 1
    print(f"[ok]  {len(ROUTES)} routes, 0 anchor misses ({total} anchors checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
