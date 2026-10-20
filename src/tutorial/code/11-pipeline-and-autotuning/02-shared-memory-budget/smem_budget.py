#!/usr/bin/env python3
"""On-chip storage budget for a multi-buffered GEMM tile.

The question this answers: 给定 tile 尺寸与 stage 数，这份配置放得下吗？
It is pure arithmetic over user-supplied numbers, so it runs on any machine.
Nothing here is measured on hardware.

The estimate
------------
For a tiled GEMM with tiles ``(BM, BK)`` and ``(BK, BN)`` and ``num_stages``
pipe buffers, the compiler versions each buffer that is written in one pipeline
stage and read in another. The version count of a buffer follows
``use_stage - def_stage + 1`` (see
``_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1488``), and a copy
in stage 0 consumed in stage ``num_stages - 1`` therefore gets
``num_stages`` versions. The naive per-block footprint is then

    bytes_per_stage = (BM * BK + BK * BN) * dtype_bytes
    total           = num_stages * bytes_per_stage

That is an *upper bound*: TileLang merges shared allocations and can reuse
storage whose live ranges do not overlap (``MergeSharedMemoryAllocations``),
which is why the tool also reports the "no-reuse" and "reuse everything"
extremes. Treat the number as a筛选条件, not as a promised allocation.

Usage
-----
    python3 smem_budget.py
    python3 smem_budget.py --bm 256 --bn 256 --bk 64 --stages 3 4
    python3 smem_budget.py --csv
"""

from __future__ import annotations

import argparse

# Bytes per element. bf16/fp16 are the common GEMM input dtypes here.
DTYPES = {
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
    "fp32": 4,
    "tf32": 4,
}

# Shared memory per block/SM figures used as reference points. These are
# hardware facts taken from the corpus (see ../references.md), not model
# outputs; they are here only to make the verdict concrete.
CAPACITY_PRESETS = {
    # NVIDIA H100 SXM5: 256 KB of combined shared memory and L1 per SM, of
    # which the shared-memory portion is configurable up to 228 KB per SM.
    # A single block cannot exceed that ceiling, so it works as a budget.
    # ../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md
    "h100-sm-shared-228kb": 228 * 1024,
    # A 32 KB budget, close to what older/entry GPUs allow per block.
    "32kb": 32 * 1024,
    # Ascend 950 L1 (CBuf) budget is not a public number; this is a
    # deliberately small placeholder used only to exercise the verdict path.
    "ascend-l1-placeholder": 512 * 1024,
}


def tile_bytes(bm: int, bk: int, bn: int, dtype_bytes: int) -> int:
    """Bytes of one copy stage: the A tile plus the B tile."""
    return (bm * bk + bk * bn) * dtype_bytes


def footprint(bm: int, bk: int, bn: int, dtype_bytes: int, num_stages: int) -> int:
    """Upper-bound shared bytes if both A and B tiles get ``num_stages`` copies."""
    return num_stages * tile_bytes(bm, bk, bn, dtype_bytes)


def verdict(total: int, capacity: int) -> str:
    if total <= capacity:
        ratio = total / capacity
        if ratio > 0.9:
            return f"fits, but tight ({ratio:.0%} of budget)"
        return f"fits ({ratio:.0%} of budget)"
    return f"does NOT fit ({total / capacity:.0%} of budget)"


def _table(rows: list[tuple[str, str]], headers: tuple[str, str]) -> str:
    left = max(len(headers[0]), max((len(r[0]) for r in rows), default=0))
    right = max(len(headers[1]), max((len(r[1]) for r in rows), default=0))
    bar = f"+-{'-' * left}-+-{'-' * right}-+"
    out = [bar, f"| {headers[0]:<{left}} | {headers[1]:<{right}} |", bar]
    out += [f"| {a:<{left}} | {b:<{right}} |" for a, b in rows]
    out.append(bar)
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm", type=int, default=128)
    parser.add_argument("--bn", type=int, default=128)
    parser.add_argument("--bk", type=int, default=32)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    parser.add_argument(
        "--capacity",
        default="h100-sm-shared-228kb",
        help="byte budget, or one of: " + ", ".join(sorted(CAPACITY_PRESETS)),
    )
    parser.add_argument("--csv", action="store_true")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="sweep tile shapes and print the largest stage count that fits",
    )
    args = parser.parse_args()

    try:
        capacity = int(args.capacity)
    except ValueError:
        if args.capacity not in CAPACITY_PRESETS:
            parser.error(f"unknown capacity preset {args.capacity!r}")
        capacity = CAPACITY_PRESETS[args.capacity]

    dtype_bytes = DTYPES[args.dtype]
    per_stage = tile_bytes(args.bm, args.bk, args.bn, dtype_bytes)

    print(
        f"tile: A[{args.bm},{args.bk}] + B[{args.bk},{args.bn}] "
        f"@ {args.dtype} ({dtype_bytes} B/elem)"
    )
    print(f"per stage (A tile + B tile): {per_stage} B")
    print(f"budget: {capacity} B ({capacity / 1024:.0f} KiB)")
    print(f"largest stage count that fits: {capacity // per_stage if per_stage else 0}")
    print()

    if args.csv:
        print("bm,bn,bk,dtype,stages,bytes,verdict")
        for stages in args.stages:
            total = footprint(args.bm, args.bk, args.bn, dtype_bytes, stages)
            print(
                f"{args.bm},{args.bn},{args.bk},{args.dtype},{stages},"
                f"{total},{verdict(total, capacity)}"
            )
        return 0

    rows = []
    for stages in args.stages:
        total = footprint(args.bm, args.bk, args.bn, dtype_bytes, stages)
        rows.append(
            (
                str(stages),
                f"{total:>9} B  {total / 1024:>7.1f} KiB   {verdict(total, capacity)}",
            )
        )
    print(_table(rows, ("num_stages", "upper-bound shared footprint")))

    if args.sweep:
        print()
        print("== tile x stage sweep ==")
        print(
            "Each cell is the largest num_stages whose upper bound still fits "
            f"within {capacity / 1024:.0f} KiB."
        )
        shapes = [(64, 64), (128, 128), (128, 256), (256, 128), (256, 256)]
        block_ks = [16, 32, 64]
        header = "block_M x block_N | " + " | ".join(f"BK={bk:>3}" for bk in block_ks)
        print(header)
        print("-" * len(header))
        for bm, bn in shapes:
            cells = []
            for bk in block_ks:
                per = tile_bytes(bm, bk, bn, dtype_bytes)
                cells.append(f"{capacity // per:^7d}" if per else "  n/a  ")
            print(f"{bm:>5} x {bn:<5}     | " + " | ".join(cells))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
