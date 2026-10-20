#!/usr/bin/env python3
"""On-chip footprint estimator for Chapter 06.

This is a *pure arithmetic* model: it multiplies tile shapes by dtype sizes and
compares the result against a small catalogue of on-chip capacity figures taken
from vendor documentation (see ``CAPACITIES`` below). It does **not** run on any
accelerator and makes no claim about what a real kernel achieves.

Run:
    python3 onchip_footprint.py

Everything printed here is reproducible on a CPU.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Capacity catalogue.
#
# Every entry carries an explicit source id and a "scope" note, because these
# numbers are *not* universal:
#   * H100/B200 shared memory is a per-SM vendor figure (compute capability
#     9.0 / 10.0). The per-block limit is 1 KB lower because CUDA reserves it.
#   * The Ascend UB=192K and L1=512KB figures are *examples* from specific
#     documents, not a family-wide specification. Ascend requires querying
#     GetCoreMemSize at runtime instead. We keep them here only to make the
#     arithmetic concrete.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Capacity:
    name: str
    bytes_: int
    source: str
    note: str


KIB = 1024
MIB = 1024 * 1024

CAPACITIES: tuple[Capacity, ...] = (
    Capacity(
        name="NV shared memory per block (H100, cc9.0)",
        bytes_=227 * KIB,
        source="R2 Hopper Tuning Guide 1.4.1.1",
        note="per-SM 228 KB; CUDA reserves 1 KB so a block addresses up to 227 KB",
    ),
    Capacity(
        name="NV shared memory per SM (H100, cc9.0)",
        bytes_=228 * KIB,
        source="R2 Hopper Tuning Guide 1.4.1.1",
        note="carveout may be 0,8,16,32,64,100,132,164,196,228 KB",
    ),
    Capacity(
        name="NV shared memory per block (B200, cc10.0)",
        bytes_=227 * KIB,
        source="R4 Blackwell Tuning Guide 1.4.2.3",
        note="same 228 KB/SM figure as Hopper; 1 KB reserved",
    ),
    Capacity(
        name="Ascend Unified Buffer (documented example)",
        bytes_=192 * KIB,
        source="R10 Ascend C best practices (bank conflict page)",
        note="EXAMPLE for one Atlas A2-class part, not a family-wide spec",
    ),
    Capacity(
        name="Ascend L1 Buffer (documented example)",
        bytes_=512 * KIB,
        source="R11 GetMatmulApiTiling example comment",
        note="EXAMPLE: the comment says 'this example L1 Buffer size is 512KB'",
    ),
)


# fp16/bf16 = 2 bytes, fp32 = 4, fp8 = 1. Intentionally tiny on purpose: this
# table is about the memory hierarchy, not a full dtype system.
DTYPE_BYTES: dict[str, int] = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
}


def tile_bytes(shape: tuple[int, ...], dtype: str) -> int:
    """Bytes for one tile with the given logical shape and dtype."""
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}")
    count = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError(f"tile shape must be positive, got {shape}")
        count *= dim
    return count * DTYPE_BYTES[dtype]


def buffer_bytes(tiles: list[tuple[str, tuple[int, ...], str]], versions: int) -> int:
    """Bytes for a multi-buffered set of tiles.

    ``tiles`` is a list of (label, shape, dtype). Every tile is multiplied by
    ``versions`` because a multi-buffer keeps N physical copies of each slot.
    """
    if versions < 1:
        raise ValueError("versions must be >= 1")
    total = 0
    for _label, shape, dtype in tiles:
        total += tile_bytes(shape, dtype) * versions
    return total


def human(n: int) -> str:
    if n % MIB == 0:
        return f"{n // MIB} MiB"
    if n % KIB == 0:
        return f"{n // KIB} KiB"
    return f"{n} B"


def report(
    tiles: list[tuple[str, tuple[int, ...], str]],
    versions: int,
    against: tuple[str, ...],
) -> None:
    """Print one case.

    ``against`` lists the *names* of the capacities that are actually relevant
    to this case. Comparing an L1 tile set against a UB budget (or vice versa)
    would be meaningless, because they are different physical storages.
    """
    print("tiles:")
    for label, shape, dtype in tiles:
        one = tile_bytes(shape, dtype)
        print(
            f"  {label:<6} {shape!s:<18} {dtype:<5} "
            f"1 copy = {one:>10,} B ({human(one)})"
        )
    total = buffer_bytes(tiles, versions)
    print(f"\nmulti-buffer: {versions} version(s)")
    print(f"single version total : {buffer_bytes(tiles, 1):>12,} B ({human(buffer_bytes(tiles, 1))})")
    print(f"{versions} version(s) total  : {total:>12,} B ({human(total)})")
    print()
    print("fit check against the relevant storages:")
    selected = [cap for cap in CAPACITIES if cap.name in against]
    if not selected:
        raise ValueError(f"no capacity matched {against!r}")
    for cap in CAPACITIES:
        if cap not in selected:
            continue
        verdict = "FITS" if total <= cap.bytes_ else "TOO BIG"
        headroom = cap.bytes_ - total
        sign = "+" if headroom >= 0 else "-"
        print(
            f"  [{verdict:>7}] {cap.name:<44} "
            f"{human(abs(headroom))} {sign}  ({cap.source})"
        )
        print(f"            note: {cap.note}")


def main() -> None:
    print(__doc__.strip().splitlines()[0])
    print("=" * 78)

    # Case 1: a Hopper-style GEMM tile pair in shared memory. This mirrors the
    # kind of A_shared / B_shared pair used by examples/quickstart.py.
    print("\nCase 1: NV GEMM A/B tiles (block_M=block_N=128, block_K=64, fp16)")
    report(
        tiles=[
            ("A_shared", (128, 64), "fp16"),
            ("B_shared", (64, 128), "fp16"),
        ],
        versions=3,
        against=(
            "NV shared memory per block (H100, cc9.0)",
            "NV shared memory per SM (H100, cc9.0)",
            "NV shared memory per block (B200, cc10.0)",
        ),
    )

    # Case 2: an Ascend-style Cube tile set. A_l1/B_l1 are bf16, C_l0c is fp32.
    # NOTE: L0C capacity is NOT in the catalogue (see the chapter) because we
    # could not source a family-wide number, so only L1 is checked here.
    print("\n" + "-" * 78)
    print("Case 2: Ascend GEMM A/B tiles in L1 (block_M=256, block_N=224, block_K=128, bf16)")
    report(
        tiles=[
            ("A_l1", (256, 128), "bf16"),
            ("B_l1", (224, 128), "bf16"),
        ],
        versions=2,
        against=("Ascend L1 Buffer (documented example)",),
    )

    print("\n" + "-" * 78)
    print("Case 3: Ascend vector tiles in UB (a 128x256 fp32 tile, 2 versions)")
    report(
        tiles=[
            ("C_ub", (128, 256), "fp32"),
        ],
        versions=2,
        against=("Ascend Unified Buffer (documented example)",),
    )

    print("\n" + "=" * 78)
    print(
        "Reminder: the Ascend UB/L1 figures are documentation *examples* for one\n"
        "part, and L0A/L0B/L0C capacities are intentionally absent (see chapter\n"
        "section 4.2). Use GetCoreMemSize on a real device instead of trusting a\n"
        "number copied out of an example."
    )


if __name__ == "__main__":
    main()
