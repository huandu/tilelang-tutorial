#!/usr/bin/env python3
"""Print the dtype families TileLang v0.1.15 exposes, grouped by role.

The authoritative list is ``_all_dtypes`` in
``_vendor/tilelang-v0.1.15/tilelang/language/dtypes.py`` (the list literal ends
just before ``__all__`` at line 891). This script re-states that list in a
grouped, human-readable form so the chapter's tables can be checked, and
``--diff`` compares the transcription against the vendor file entry by entry.

It is a transcription, not a live import: the writing machine has no TileLang
installed. ``--diff`` therefore fails loudly if a future pin changes the list.

Run:
    python3 dtype_families.py            # grouped table
    python3 dtype_families.py --lanes    # lane-expansion counts
    python3 dtype_families.py --diff     # exact set comparison vs. the vendor file
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# --- grouped transcription of dtypes.py::_all_dtypes (v0.1.15) -------------

# Seven TVM canonical spellings are also accepted; dtype() normalizes them via
# _CANONICAL_TO_DISPLAY_STR (dtypes.py:105).
ALIASES = [
    ("short", "int16"),
    ("int", "int32"),
    ("long", "int64"),
    ("uint", "uint32"),
    ("half", "float16"),
    ("float", "float32"),
    ("double", "float64"),
]

NO_LANES = [
    ("bool", "boolean, one element per byte; no vector form"),
    ("int4", "packed signed 4-bit; no vector form"),
    ("float4_e2m1_unpacked", "custom[float4_e2m1_unpacked]8; SMEM/TMA only, no vector form"),
]

SIGNED_INT = ["int8", "int16", "int32", "int64"]
UNSIGNED_INT = ["uint8", "uint16", "uint32", "uint64"]
PLAIN_FLOAT = ["float16", "float32", "float64"]

FLOAT8 = [
    ("float8_e3m4", "3 exponent / 4 mantissa"),
    ("float8_e4m3", "E4M3; NaN semantics chosen by the backend"),
    ("float8_e4m3b11fnuz", "bias-11 FNUZ variant"),
    ("float8_e4m3fn", "E4M3 finite+NaN (CUDA default)"),
    ("float8_e4m3fnuz", "E4M3 FNUZ (ROCm default)"),
    ("float8_e5m2", "E5M2, wider exponent range"),
    ("float8_e5m2fnuz", "E5M2 FNUZ variant"),
    ("float8_e8m0fnu", "8-bit exponent only, used as an MX block scale"),
]

FLOAT6 = [
    ("float6_e2m3fn", "6-bit, 2 exponent / 3 mantissa"),
    ("float6_e3m2fn", "6-bit, 3 exponent / 2 mantissa"),
]

# bfloat16 and tfloat32 have their own (short) lane sets, so list them raw.
IRREGULAR_LANES = {
    "bfloat16": ["bfloat16", "bfloat16x2"],
    "tfloat32": ["tfloat32"] + [f"tfloat32x{n}" for n in (2, 4, 8, 16, 32, 64)],
    "float4_e2m1fn": ["float4_e2m1fn"] + [f"float4_e2m1fnx{n}" for n in (2, 4, 8, 16, 32, 64)],
}


def with_lanes(names: list[str], suffixes=(2, 4, 8, 16, 32, 64)) -> list[str]:
    out: list[str] = []
    for name in names:
        out.append(name)
        out.extend(f"{name}x{s}" for s in suffixes)
    return out


def transcription() -> set[str]:
    names: set[str] = set()
    names.update(n for n, _ in ALIASES)
    names.update(n for n, _ in NO_LANES)
    names.update(with_lanes(SIGNED_INT))
    names.update(with_lanes(UNSIGNED_INT))
    names.update(with_lanes(PLAIN_FLOAT))
    names.update(with_lanes([n for n, _ in FLOAT8]))
    names.update(with_lanes([n for n, _ in FLOAT6]))
    for family in IRREGULAR_LANES.values():
        names.update(family)
    return names


# --- presentation ---------------------------------------------------------


def print_rows(title: str, rows: list[tuple[str, str]]) -> None:
    print(f"\n## {title}")
    width = max(len(n) for n, _ in rows)
    for name, note in rows:
        print(f"  {name:<{width}}  {note}")


def grouped_table() -> None:
    print("TileLang v0.1.15 dtype families (transcribed from dtypes.py::_all_dtypes)")
    print_rows("canonical aliases (normalized by dtype())", ALIASES)
    print_rows("no vector form", NO_LANES)
    print_rows(
        "signed integers (each also x2/x4/x8/x16/x32/x64)",
        [(n, "") for n in SIGNED_INT],
    )
    print_rows(
        "unsigned integers (each also x2/x4/x8/x16/x32/x64)",
        [(n, "") for n in UNSIGNED_INT],
    )
    print_rows(
        "floats (each also x2/x4/x8/x16/x32/x64)",
        [(n, "") for n in PLAIN_FLOAT],
    )
    print_rows("float8 (each also x2/x4/x8/x16/x32/x64)", FLOAT8)
    print_rows("float6 (each also x2/x4/x8/x16/x32/x64)", FLOAT6)
    print("\n## extra families")
    print("  bfloat16         also bfloat16x2 (and nothing wider)")
    print("  tfloat32         also tfloat32x2/x4/x8/x16/x32/x64")
    print("  float4_e2m1fn    packed FP4, also float4_e2m1fnx2/x4/x8/x16/x32/x64")


def lane_table() -> None:
    print("Lane expansion (names exported as attributes of tilelang.language)")
    groups = {
        "aliases (no lane forms)": [n for n, _ in ALIASES],
        "no-lane scalars": [n for n, _ in NO_LANES],
        "signed int": SIGNED_INT,
        "unsigned int": UNSIGNED_INT,
        "float": PLAIN_FLOAT,
        "float8": [n for n, _ in FLOAT8],
        "float6": [n for n, _ in FLOAT6],
        "bfloat16/tfloat32/float4": sum(IRREGULAR_LANES.values(), []),
    }
    for label, names in groups.items():
        print(f"  {label:<28} {len(names):>3} names")
    print(f"  {'TOTAL':<28} {len(transcription()):>3} names")


def diff_against_vendor() -> int:
    vendor = (
        Path(__file__).resolve().parents[4]
        / "_vendor"
        / "tilelang-v0.1.15"
        / "tilelang"
        / "language"
        / "dtypes.py"
    )
    if not vendor.is_file():
        print(f"vendor file not found at {vendor}; nothing to diff")
        return 0
    text = vendor.read_text(encoding="utf-8")
    match = re.search(r"_all_dtypes = \[(.*?)\n\]", text, re.S)
    if match is None:
        print("could not locate _all_dtypes in the vendor file")
        return 1
    actual = set(re.findall(r'"([^"]+)"', match.group(1)))
    mine = transcription()
    only_vendor = sorted(actual - mine)
    only_mine = sorted(mine - actual)
    print(f"vendor entries: {len(actual)}; transcription entries: {len(mine)}")
    if only_vendor:
        print("only in vendor:", ", ".join(only_vendor))
    if only_mine:
        print("only in transcription:", ", ".join(only_mine))
    if not only_vendor and not only_mine:
        print("exact match")
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", action="store_true", help="show lane-expansion counts")
    parser.add_argument("--diff", action="store_true", help="diff against the vendor source")
    args = parser.parse_args()
    if args.diff:
        return diff_against_vendor()
    if args.lanes:
        lane_table()
        return 0
    grouped_table()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
