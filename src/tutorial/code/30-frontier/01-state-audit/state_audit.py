#!/usr/bin/env python3
"""Audit where a TileLang capability lives: v0.1.14 / v0.1.15 / main snapshot.

This script answers one question per capability, using only the filesystem:

    is this capability present in the locked release (v0.1.15),
    only in the newer main snapshot, or nowhere in either?

It never runs a compiler and never needs a GPU.  Every verdict is derived from
file presence plus a plain text search, so the verdict can be reproduced by
hand with the commands the script prints.

Usage:
    python3 state_audit.py                     # auto-detect the trees
    python3 state_audit.py --book-root /path/to/tilelang-book
    python3 state_audit.py --corpus-root /path/to/tilelang-corpus
    python3 state_audit.py --json               # machine-readable output

Exit code: 0 when every probe matches its expected verdict, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Trees
# --------------------------------------------------------------------------

TAG_RELEASE = "tilelang-v0.1.15"
TAG_PREVIOUS = "tilelang-v0.1.14"
MAIN_SNAPSHOT = Path("02-tilelang-core-repo") / "raw" / "tilelang"

# Directories we never descend into: they are third-party checkouts or
# build artifacts, and a hit there would not prove anything about TileLang.
SKIP_DIRS = {
    ".git",
    "3rdparty",
    "build",
    "dist",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
}

# Only these suffixes are searched for text.  Everything else is treated as
# opaque, which keeps the scan fast and deterministic.
TEXT_SUFFIXES = {".py", ".pyx", ".pyi", ".cc", ".h", ".cpp", ".hpp", ".md", ".rst", ".txt", ".cmake", ".sh", ".yaml", ".yml", ".toml", ".json"}

# a few suffix-less files that carry real signal
EXTRA_TEXT_NAMES = {".gitmodules", "VERSION", "CMakeLists.txt", "Makefile"}

MAX_FILE_BYTES = 2 * 1024 * 1024


@dataclass
class Probe:
    """One way to decide whether a capability is present in one tree."""

    kind: str  # "path" | "text"
    value: str
    scope: str = "."  # restrict a text search to this subtree
    note: str = ""


@dataclass
class Capability:
    """A named capability and the probes that decide its state."""

    cid: str
    title: str
    probes: list[Probe]
    expect: dict[str, bool] = field(default_factory=dict)
    state: str | None = None


def walk_text_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() not in TEXT_SUFFIXES and name not in EXTRA_TEXT_NAMES:
                continue
            try:
                if path.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            yield path


def probe_hits(root: Path, probe: Probe) -> list[str]:
    """Return the relative paths that satisfy ``probe`` inside ``root``."""
    if probe.kind == "path":
        target = root / probe.value
        return [probe.value] if target.exists() else []
    if probe.kind == "text":
        needle = probe.value
        search_root = root / probe.scope
        if not search_root.exists():
            return []
        if search_root.is_file():
            try:
                text = search_root.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                return []
            return [probe.scope] if needle in text else []
        hits: list[str] = []
        for path in walk_text_files(search_root):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if needle in text:
                hits.append(str(path.relative_to(root)))
                if len(hits) >= 5:
                    break
        return hits
    raise ValueError(f"unknown probe kind: {probe.kind}")


def capability_state(root: Path, cap: Capability) -> tuple[bool, list[str]]:
    """A capability is present when *all* its probes hit."""
    evidence: list[str] = []
    present = True
    for probe in cap.probes:
        hits = probe_hits(root, probe)
        if not hits:
            present = False
            continue
        evidence.extend(hits[:2])
    return present, evidence


# --------------------------------------------------------------------------
# The capability list (see chapter 30, section 9)
# --------------------------------------------------------------------------

CAPABILITIES = [
    Capability(
        "multi-backend-registry",
        "多后端解耦：backend registry + 每个后端自己的 pipeline",
        [
            Probe("text", "def register_backend", scope="tilelang/backend"),
            Probe("path", "tilelang/cuda/pipeline.py"),
            Probe("path", "tilelang/cuda/cutedsl_backend.py"),
        ],
        expect={TAG_PREVIOUS: True, TAG_RELEASE: True},
    ),
    Capability(
        "tvm-fork-still-pinned",
        "TVM 依赖仍指向 TileLang/tvm 分叉（迁移 #2196 未落地）",
        [Probe("text", "https://github.com/TileLang/tvm", scope=".gitmodules")],
        expect={TAG_PREVIOUS: True, TAG_RELEASE: True},
        state="v0.1.15 已具备（=迁移未落地）",
    ),
    Capability(
        "vendored-z3-prover",
        "RFC #2196 所述的 vendored Z3 prover（下游分支做法）",
        [Probe("path", "src/transform/vendored/z3_prover.cc")],
        expect={TAG_PREVIOUS: False, TAG_RELEASE: False},
    ),
    Capability(
        "cutedsl-backend",
        "CuteDSL 后端（issue #1454）：codegen + adapter + 独立 BackendModule",
        [
            Probe("path", "src/cuda/codegen/codegen_cutedsl.cc", note="device codegen"),
            Probe("path", "tilelang/cuda/cutedsl_backend.py", note="BackendModule 注册"),
            Probe("path", "tilelang/jit/adapter/cutedsl", note="JIT adapter"),
        ],
        expect={TAG_PREVIOUS: True, TAG_RELEASE: True},
    ),
    Capability(
        "ascend-backend-in-mainline",
        "昇腾后端（AscendC 路线）在主干的 tilelang/ascend + src/ascend",
        [
            Probe("path", "tilelang/ascend/language", note="前端方言"),
            Probe("path", "src/ascend/codegen/codegen_ascend.cc", note="device codegen"),
            Probe("text", "dav-3510", scope="tilelang/contrib/bisheng.py"),
        ],
        expect={TAG_PREVIOUS: False, TAG_RELEASE: True},
    ),
    Capability(
        "ascend-pto-backend",
        "昇腾 PTO 后端（PR #3310）",
        [
            Probe("path", "src/ascend/codegen/codegen_pto.cc"),
            Probe("path", "tilelang/jit/adapter/pto"),
        ],
        expect={TAG_PREVIOUS: False, TAG_RELEASE: False},
    ),
    Capability(
        "ascend-auto-schedule",
        "昇腾 AutoSchedule / Z3 调度与 InsertSync",
        [
            Probe("path", "src/ascend/transform/auto_schedule"),
            Probe("path", "src/ascend/transform/insert_sync.cc"),
        ],
        expect={TAG_PREVIOUS: False, TAG_RELEASE: True},
    ),
    Capability(
        "ascend-persistent-kernel",
        "T.PersistentKernel + AutoPersistent（PR #3325）",
        [
            Probe("text", "PersistentKernel"),
            Probe("text", "AutoPersistent"),
        ],
        expect={TAG_PREVIOUS: False, TAG_RELEASE: False},
    ),
    Capability(
        "ascend-auto-simt-vf",
        "AutoSimtVF pass（PR #3320）",
        [Probe("text", "AutoSimtVF")],
        expect={TAG_PREVIOUS: False, TAG_RELEASE: False},
    ),
    Capability(
        "ascend-op-agent",
        "昇腾算子生成 agent：agent/core + tilelang-ascend-op skills（PR #3329）",
        [
            Probe("path", "agent/core"),
            Probe("path", ".agents/skills/tilelang-ascend-op"),
        ],
        expect={TAG_PREVIOUS: False, TAG_RELEASE: False},
    ),
    Capability(
        "reducer-v2",
        "reducer v2（RFC #2897 的主体语义）",
        [
            Probe("text", "def reducer_update", scope="tilelang/language/reduce_op.py"),
            Probe("path", "src/transform/verify_reducer_epoch.cc"),
            Probe("path", "src/transform/canonicalize_legacy_reducer.cc"),
        ],
        expect={TAG_PREVIOUS: True, TAG_RELEASE: True},
    ),
    Capability(
        "copy-cache-policy",
        "T.copy 的 load_cache_policy / store_cache_policy（RFC #3178）",
        [
            Probe("text", "load_cache_policy"),
            Probe("text", "store_cache_policy"),
        ],
        expect={TAG_PREVIOUS: False, TAG_RELEASE: False},
    ),
    Capability(
        "copy-eviction-policy",
        "既有的 TMA eviction_policy（与 RFC #3178 是两回事）",
        [Probe("text", "eviction_policy", scope="tilelang/language/copy_op.py")],
        expect={TAG_PREVIOUS: True, TAG_RELEASE: True},
    ),
]


def state_label(prev: bool, rel: bool, main: bool | None) -> str:
    if rel:
        return "v0.1.15 已具备"
    if main:
        return "仅 main"
    if main is None:
        return "未在 v0.1.15（main 快照不可用）"
    return "仅未合入 PR/RFC"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve()
    parser.add_argument(
        "--book-root",
        type=Path,
        default=here.parents[4],
        help="tilelang-book 根目录（默认从脚本位置推断：../../../../..）",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=None,
        help="tilelang-corpus 根目录（默认取 book-root 的同级目录）",
    )
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    if args.corpus_root is None:
        args.corpus_root = args.book_root.parent / "tilelang-corpus"

    vendor = args.book_root / "_vendor"
    trees: dict[str, Path | None] = {
        TAG_PREVIOUS: vendor / TAG_PREVIOUS,
        TAG_RELEASE: vendor / TAG_RELEASE,
        "main": args.corpus_root / MAIN_SNAPSHOT,
    }
    for name, root in list(trees.items()):
        if root is not None and not root.is_dir():
            if name == "main":
                trees[name] = None
            else:
                print(f"missing tree: {root}", file=sys.stderr)
                return 2

    rows = []
    mismatches = []
    for cap in CAPABILITIES:
        verdicts: dict[str, bool] = {}
        evidence: dict[str, list[str]] = {}
        for name, root in trees.items():
            if root is None:
                continue
            present, ev = capability_state(root, cap)
            verdicts[name] = present
            evidence[name] = ev
        state = cap.state or state_label(
            verdicts.get(TAG_PREVIOUS, False),
            verdicts.get(TAG_RELEASE, False),
            verdicts.get("main"),
        )
        for name, expected in cap.expect.items():
            got = verdicts.get(name)
            if got != expected:
                mismatches.append(
                    f"{cap.cid}: expected {name}={expected}, got {got}"
                )
        rows.append(
            {
                "id": cap.cid,
                "title": cap.title,
                "state": state,
                "in_previous": verdicts.get(TAG_PREVIOUS),
                "in_release": verdicts.get(TAG_RELEASE),
                "in_main": verdicts.get("main"),
                "evidence": evidence,
                "probes": [
                    {"kind": p.kind, "value": p.value} for p in cap.probes
                ],
            }
        )

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print("capability                | v0.1.14 | v0.1.15 | main 快照 | 状态")
        print("--------------------------|---------|---------|-----------|-------------------")
        for row in rows:
            print(
                "{:<25} | {:^7} | {:^7} | {:^9} | {}".format(
                    row["id"],
                    _yn(row["in_previous"]),
                    _yn(row["in_release"]),
                    _yn(row["in_main"]),
                    row["state"],
                )
            )
        print()
        print("证据（每行：能力 -> 命中的文件）：")
        for row in rows:
            hits = sorted({h for ev in row["evidence"].values() for h in ev})
            print(f"  {row['id']}: {', '.join(hits[:3]) if hits else '(无命中)'}")
        print()
        print("手工复现命令：")
        print(
            "  rg -n '<symbol>' _vendor/tilelang-v0.1.15/tilelang "
            "_vendor/tilelang-v0.1.15/src"
        )
        print("  python3 02-tree-diff/tree_diff.py --help")

    if mismatches:
        print("\nself-check: MISMATCH", file=sys.stderr)
        for item in mismatches:
            print(f"  - {item}", file=sys.stderr)
        return 1
    if not args.json:
        print("\nself-check: ok")
    return 0


def _yn(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "Y" if value else "-"


if __name__ == "__main__":
    raise SystemExit(main())
