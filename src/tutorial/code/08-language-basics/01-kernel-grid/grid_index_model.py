#!/usr/bin/env python3
"""Model of the grid that ``with T.Kernel(gx, gy, ...) as (bx, by)`` builds.

This is a *pure standard library* re-implementation of the two facts chapter 08
relies on, so they can be checked without a GPU:

  1. The positional arguments of ``T.Kernel`` map, in order, to grid axes
     x, y, z. TileLang builds a thread-binding frame per axis with tags
     "blockIdx.x", "blockIdx.y", "blockIdx.z"
     (``_vendor/tilelang-v0.1.15/src/ir.cc:236-262``).
  2. Blocks are queried with the usual "last axis varies fastest" convention:
     a linear program id ``p`` decomposes as
     ``bx = p % gx``, ``by = (p // gx) % gy``, ``bz = p // (gx*gy)``.

Nothing here talks to TileLang or any accelerator; it is a model.

Run:
    python3 grid_index_model.py
"""

from __future__ import annotations

def ceildiv(a: int, b: int) -> int:
    """Ceiling of a/b for positive integers (what T.ceildiv computes)."""
    return -(-a // b)


def grid_extents(total_m: int, total_n: int, block_m: int, block_n: int) -> tuple[int, int]:
    """Grid extents for ``T.Kernel(ceildiv(N, BN), ceildiv(M, BM))``."""
    return ceildiv(total_n, block_n), ceildiv(total_m, block_m)


def linear_to_block(pid: int, gx: int, gy: int, gz: int = 1) -> tuple[int, int, int]:
    """Decompose a linear program id into (bx, by, bz); x varies fastest."""
    bx = pid % gx
    by = (pid // gx) % gy
    bz = pid // (gx * gy)
    if not 0 <= bz < gz:
        raise ValueError(f"program id {pid} out of range for grid {gx}x{gy}x{gz}")
    return bx, by, bz


def tile_origin(bx: int, by: int, block_m: int, block_n: int) -> tuple[int, int]:
    """The (row, col) origin of the output tile owned by block (bx, by)."""
    return by * block_m, bx * block_n


def out_of_bounds_slots(total: int, tile: int) -> int:
    """How many element slots in the last tile fall past ``total``."""
    return ceildiv(total, tile) * tile - total


def demo_grid(order: str, x_is_n: bool) -> None:
    """Print the launch grid for one axis assignment.

    ``x_is_n=True``  -> ``T.Kernel(ceildiv(N,BN), ceildiv(M,BM)) as (bx, by)``
    ``x_is_n=False`` -> ``T.Kernel(ceildiv(M,BM), ceildiv(N,BN)) as (bx, by)``
    """
    print(f"\n== {order} ==")
    m_total, n_total = 1000, 700
    block_m, block_n = 256, 128
    gm, gn = ceildiv(m_total, block_m), ceildiv(n_total, block_n)
    gx, gy = (gn, gm) if x_is_n else (gm, gn)
    print(f"M,N = {m_total},{n_total};  BM,BN = {block_m},{block_n}")
    print(f"per-axis tiles: M -> {gm}, N -> {gn}")
    print(f"grid = ({gx}, {gy})  ->  {gx * gy} programs; x varies fastest")
    print(f"T.Kernel({gx}, {gy}, threads=128) as (bx, by)")

    print(f"{'pid':>3}  {'bx':>3} {'by':>3}  {'row0':>5} {'col0':>5}  {'m_tile':>6} {'n_tile':>6}")
    for pid in range(gx * gy):
        bx, by, _bz = linear_to_block(pid, gx, gy)
        m_tile, n_tile = (by, bx) if x_is_n else (bx, by)
        row0, col0 = m_tile * block_m, n_tile * block_n
        print(f"{pid:>3}  {bx:>3} {by:>3}  {row0:>5} {col0:>5}  {m_tile:>6} {n_tile:>6}")

    print(
        "tail: M pads "
        f"{out_of_bounds_slots(m_total, block_m)} rows, N pads "
        f"{out_of_bounds_slots(n_total, block_n)} cols "
        "-> LegalizeSafeMemoryAccess inserts the guards"
    )


def demo_three_axis() -> None:
    print("\n== three axes ==")
    gx, gy, gz = 3, 2, 2
    print(f"T.Kernel({gx}, {gy}, {gz}) as (bx, by, bz)")
    for pid in range(gx * gy * gz):
        print(f"  pid {pid:>2} -> bx={pid % gx} by={(pid // gx) % gy} bz={pid // (gx * gy)}")


def demo_threads_default() -> None:
    print("\n== threads default ==")
    # DEFAULT_SIMT_THREADS = 128 in tilelang/transform/__init__.py:201
    print("threads omitted -> the SIMT backend uses DEFAULT_SIMT_THREADS = 128")
    for threads in (None, 64, (64, 2), (32, 4, 2)):
        if threads is None:
            extents = [128, 1, 1]
        elif isinstance(threads, int):
            extents = [threads, 1, 1]
        else:
            extents = list(threads) + [1] * (3 - len(threads))
        total = 1
        for e in extents:
            total *= e
        print(f"  threads={threads!s:>12} -> blockDim={tuple(extents)} total={total}")


def main() -> None:
    demo_grid("x = N axis, y = M axis  (layout used by the quickstart)", x_is_n=True)
    demo_grid("swapped: x = M axis, y = N axis", x_is_n=False)
    demo_three_axis()
    demo_threads_default()
    print("\nAll numbers above are recomputed from the formulas, not measured.")


if __name__ == "__main__":
    main()
