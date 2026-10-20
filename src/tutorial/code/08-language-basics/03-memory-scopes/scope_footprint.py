#!/usr/bin/env python3
"""Chapter 08 / §5: turn a tile shape + dtype + buffer count into bytes.

This is a *pure standard library* bookkeeping helper. It does not model
TileLang's allocator, and it does not know any real device's capacity: it only
answers "how many bytes does this tile occupy" so the scope-choice discussion
in the prose can be checked with numbers instead of adjectives.

The byte count is ``prod(shape) * itemsize * versions``; the ``versions``
factor is the multi-buffer depth (``num_stages`` in a ``T.Pipelined`` loop).

Run:
    python3 scope_footprint.py
    python3 scope_footprint.py --tile 128 256 --dtype float32 --versions 2 --capacity 196608
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# itemsize in bytes for the dtypes chapter 08 discusses.
# int4 / float4_e2m1fn are packed: 2 values per byte.
ITEMSIZE_BITS = {
    "bool": 8,
    "int4": 4,
    "int8": 8,
    "uint8": 8,
    "float8_e4m3fn": 8,
    "float8_e5m2": 8,
    "float6_e2m3fn": 6,
    "float4_e2m1fn": 4,
    "float4_e2m1_unpacked": 8,
    "float16": 16,
    "bfloat16": 16,
    "float32": 32,
    "float64": 64,
}

# role -> scope, as declared by the allocating function's default argument.
SCOPE_OF = {
    "alloc_shared (CUDA)": "shared.dyn",
    "alloc_shared (Ascend)": "shared.dyn  (Unified Buffer)",
    "alloc_fragment": "local.fragment",
    "alloc_local": "local",
    "alloc_var": "local.var",
    "alloc_global": "global",
    "alloc_tmem (Blackwell)": "shared.tmem",
    "alloc_l1 (Ascend)": "shared.l1",
    "alloc_l0a (Ascend)": "shared.l0a",
    "alloc_l0b (Ascend)": "shared.l0b",
    "alloc_l0c (Ascend)": "shared.l0c",
}


@dataclass
class Footprint:
    shape: tuple[int, ...]
    dtype: str
    versions: int

    @property
    def elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def bytes_per_copy(self) -> float:
        return self.elements * ITEMSIZE_BITS[self.dtype] / 8

    @property
    def bytes(self) -> float:
        return self.bytes_per_copy * self.versions


def human(n: float) -> str:
    if n < 1024:
        return f"{n:g} B"
    if n < 1024 * 1024:
        return f"{n / 1024:g} KiB"
    return f"{n / (1024 * 1024):g} MiB"


def scope_table() -> None:
    print("Scope name and allocating API (v0.1.15 defaults)")
    width = max(len(k) for k in SCOPE_OF)
    for role, scope in SCOPE_OF.items():
        print(f"  {role:<{width}}  ->  {scope}")


def case(name: str, shape: tuple[int, ...], dtype: str, versions: int, capacity: int | None) -> None:
    fp = Footprint(shape, dtype, versions)
    line = (
        f"{name:<34} {str(shape):<16} {dtype:<20} x{versions}  "
        f"= {human(fp.bytes):>9}"
    )
    if capacity is not None:
        fits = fp.bytes <= capacity
        line += f"   cap {human(capacity):>9}  {'fits' if fits else 'DOES NOT FIT'}"
    print("  " + line)


def demo() -> None:
    print("Record sizes = elements * bits/8 * versions  (see `--help`)\n")
    scope_table()

    capacity = 196608  # 192 KiB, the UB size quoted in Ascend best-practice docs
    print(f"\nAgainst a 192 KiB on-chip budget ({capacity} B):")
    case("A tile (fp16, no buffering)", (128, 128), "float16", 1, None)
    case("A tile (fp16, 2-stage)", (128, 128), "float16", 2, None)
    case("C fragment (fp32, per block)", (128, 128), "float32", 1, None)
    case("C tile in UB (fp32)", (128, 256), "float32", 1, capacity)
    case("C tile in UB (fp32, 2-stage)", (128, 256), "float32", 2, capacity)
    case("same tile in fp16", (128, 256), "float16", 2, capacity)

    print(
        "\nReading: doubling the dtype width or the version count doubles the\n"
        "footprint; the fragment footprint is *per block*, which is why a\n"
        "128x128 fp32 accumulator costs 64 KiB of register-file budget.\n"
    )

    frag = Footprint((128, 128), "float32", 1)
    per_thread = frag.bytes / 128
    print(
        f"  {human(frag.bytes)} fragment / 128 threads = {human(per_thread)} "
        f"= {frag.elements / 128:g} fp32 registers per thread"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tile", nargs="+", type=int, help="tile shape, e.g. --tile 128 256")
    parser.add_argument("--dtype", default="float32", choices=sorted(ITEMSIZE_BITS))
    parser.add_argument("--versions", type=int, default=1, help="multi-buffer depth")
    parser.add_argument("--capacity", type=int, default=None, help="on-chip budget in bytes")
    args = parser.parse_args()
    if args.tile:
        fp = Footprint(tuple(args.tile), args.dtype, args.versions)
        print(f"shape={fp.shape} dtype={fp.dtype} versions={fp.versions}")
        print(f"elements        = {fp.elements}")
        print(f"bytes / copy    = {fp.bytes_per_copy:g}")
        print(f"bytes total     = {fp.bytes:g}  ({human(fp.bytes)})")
        if args.capacity is not None:
            print(f"fits in {args.capacity} B: {fp.bytes <= args.capacity}")
        return 0
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
