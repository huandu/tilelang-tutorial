#!/usr/bin/env python3
"""Layout algebra toy model: bank conflicts, XOR swizzles, fragment→thread.

This script re-derives, with the standard library only, the small models that
chapter 10 uses in figures 2/3/4. It imports nothing from tilelang, so it runs
anywhere (no GPU / NPU needed).

What it reproduces:

  * Shared-memory bank numbering (32 banks, 4-byte words) and the classic
    "read one column of a 32x32 float tile" conflict.
  * The three TileLang bank-swizzle granularities (32B/64B/128B) as XOR maps,
    copied from src/layout/gemm_layouts.cc (xor2x2/xor4x4/xor8x8).
  * MakeGemmFragment8x8's thread/local mapping
    (forward_thread = j//2 + 4*i, index = j%2).

It is a *model*, not a benchmark: it never claims a speedup. It only shows how
many distinct banks a warp accesses and whether a mapping is a bijection.

Usage:
    python3 layout_algebra.py

Verified level: CPU logic verified (runs on macOS / Python 3.11, standard
library only). No result here depends on real hardware.
"""

from __future__ import annotations

NUM_BANKS = 32
WORD_BYTES = 4  # one bank is 32 bits wide on compute capability 5.x and newer


# --------------------------------------------------------------------------- #
# 1. Bank conflicts
# --------------------------------------------------------------------------- #
def bank_of_word(word_offset: int, num_banks: int = NUM_BANKS) -> int:
    """Map a word offset (in 4-byte words) to its bank index."""
    return word_offset % num_banks


def column_access_stats(stride: int, rows: int = 32) -> dict:
    """Warp reads column 0: thread t touches word (t * stride).

    Returns how many distinct banks are hit and the worst per-bank multiplicity,
    which is the serialization factor of that access.
    """
    banks = [bank_of_word(t * stride) for t in range(rows)]
    counts = [banks.count(b) for b in set(banks)]
    return {
        "stride": stride,
        "distinct_banks": len(set(banks)),
        "max_per_bank": max(counts),
        "banks": banks,
    }


def classify(stats: dict) -> str:
    if stats["max_per_bank"] == 1:
        return "conflict-free"
    return f"{stats['max_per_bank']}-way bank conflict"


# --------------------------------------------------------------------------- #
# 2. XOR swizzles (copied from src/layout/gemm_layouts.cc)
# --------------------------------------------------------------------------- #
def xor2x2(a: int, b: int) -> int:
    """1-bit XOR: FloorMod(a + b, 2)."""
    return (a + b) % 2


def xor4x4(a: int, b: int) -> int:
    """2-bit XOR, defined recursively as in the C++ source."""
    a0, b0 = a % 2, b % 2
    a1, b1 = a // 2, b // 2
    return 2 * xor2x2(a1, b1) + xor2x2(a0, b0)


def xor8x8(a: int, b: int) -> int:
    """3-bit XOR, defined recursively as in the C++ source."""
    a0, b0 = a % 2, b % 2
    a1, b1 = a // 2, b // 2
    return 2 * xor4x4(a1, b1) + xor2x2(a0, b0)


SWIZZLE_BITS = {"32B": 1, "64B": 2, "128B": 3}


def unit_permutation(c: int, s: int, bits: int) -> int:
    """Physical unit index for logical unit c in row s (s in a period of 8)."""
    if bits == 1:
        return xor2x2(c, s // 4)
    if bits == 2:
        return xor4x4(c, s // 2)
    if bits == 3:
        return xor8x8(c, s)
    raise ValueError(f"unsupported swizzle width: {bits} bits")


def verify_swizzle_bijection(bits: int) -> bool:
    """A swizzle must be a permutation: no element lost, no element duplicated."""
    width = 1 << bits
    seen = set()
    for s in range(8):
        row = [unit_permutation(c, s, bits) for c in range(width)]
        if sorted(row) != list(range(width)):
            return False
        for c, c_prime in enumerate(row):
            seen.add((s, c, c_prime))
    return len(seen) == 8 * width


def column_spread(bits: int) -> int:
    """Distinct units one logical column uses across a full 8-row period.

    A 1-bit swizzle only has 2 units per group, so 8 rows can never spread over
    more than 2 of them. The invariant is "spread == group width", not always 8.
    """
    width = 1 << bits
    spreads = set()
    for c in range(width):
        col = {unit_permutation(c, s, bits) for s in range(8)}
        spreads.add(len(col))
    assert len(spreads) == 1, f"inconsistent spread for {bits}-bit swizzle"
    return spreads.pop()


def verify_column_spread(bits: int) -> bool:
    """The spread must equal the swizzle group width (why: width == 1 << bits)."""
    return column_spread(bits) == (1 << bits)


# --------------------------------------------------------------------------- #
# 3. MakeGemmFragment8x8 (src/layout/gemm_layouts.cc:47-53)
# --------------------------------------------------------------------------- #
def gemm_fragment_8x8():
    """Yield (i, j, thread, local_index) for the canonical 8x8 GEMM fragment."""
    for i in range(8):
        for j in range(8):
            thread = j // 2 + 4 * i
            local_index = j % 2
            yield i, j, thread, local_index


def fragment_stats():
    cells = list(gemm_fragment_8x8())
    threads: dict[int, int] = {}
    for _, _, t, _ in cells:
        threads[t] = threads.get(t, 0) + 1
    return {
        "num_cells": len(cells),
        "num_threads": len(threads),
        "slots_per_thread": sorted(set(threads.values())),
        "is_bijection": len(threads) == len(cells) // 2 and set(threads.values()) == {2},
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_conflict_section() -> None:
    print("=" * 72)
    print("1) 32x32 float tile, warp reads one column (thread t -> row t, col 0)")
    print("=" * 72)
    for label, stride in (("no padding (row length 32)", 32),
                          ("padded (row length 33)", 33)):
        stats = column_access_stats(stride)
        print(f"  {label:28s} stride={stats['stride']:2d}  "
              f"distinct banks={stats['distinct_banks']:2d}  "
              f"worst per-bank={stats['max_per_bank']:2d}  -> {classify(stats)}")
    print()


def print_swizzle_section() -> None:
    print("=" * 72)
    print("2) XOR swizzle unit permutations (unit = 16B; fp16 -> 8 elements)")
    print("=" * 72)
    for name, bits in SWIZZLE_BITS.items():
        width = 1 << bits
        print(f"  {name}: {bits}-bit XOR, row period 8, {width} units per group")
        print("        c ->", "".join(f"{c:3d}" for c in range(width)))
        for s in range(8):
            row = [unit_permutation(c, s, bits) for c in range(width)]
            print(f"    s={s}  ", "".join(f"{v:3d}" for v in row))
        print(f"        bijection={verify_swizzle_bijection(bits)}  "
              f"column spread={column_spread(bits)} unit(s) over an 8-row period "
              f"(group width={1 << bits})")
        print()


def print_fragment_section() -> None:
    print("=" * 72)
    print("3) MakeGemmFragment8x8: thread = 4*i + j//2, local = j%2")
    print("=" * 72)
    grid = [["" for _ in range(8)] for _ in range(8)]
    for i, j, thread, local_index in gemm_fragment_8x8():
        grid[i][j] = f"T{thread:02d}L{local_index}"
    print("        " + "".join(f"   j={j}    " for j in range(8)))
    for i in range(8):
        print(f"  i={i}  " + "  ".join(grid[i]))
    stats = fragment_stats()
    print()
    print(f"  cells={stats['num_cells']}  threads={stats['num_threads']}  "
          f"slots per thread={stats['slots_per_thread']}  "
          f"valid bijection={stats['is_bijection']}")
    print()


def main() -> int:
    print_conflict_section()
    print_swizzle_section()
    print_fragment_section()
    print("all checks passed" if (
        verify_swizzle_bijection(1)
        and verify_swizzle_bijection(2)
        and verify_swizzle_bijection(3)
        and fragment_stats()["is_bijection"]
    ) else "CHECK FAILED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
