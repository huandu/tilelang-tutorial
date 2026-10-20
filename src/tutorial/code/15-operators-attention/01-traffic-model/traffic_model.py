#!/usr/bin/env python3
"""HBM element-traffic model: naive attention vs. block-fused (flash) attention.

This is a *counting model*, not a measurement. It answers one question:

    For one (batch, head) pair, how many matrix elements are read from and
    written to global memory by the naive three-kernel formulation, and how
    many by the tiled/online-softmax formulation?

Everything here is exact integer arithmetic on shapes. No GPU, no numpy.

Assumptions (stated so the numbers can be audited):

  * One (batch, head) pair: Q is Sq x d, K/V are Skv x d, output O is Sq x d.
  * Naive: three passes that materialise S and P (Sq x Skv each) in global
    memory — the S matrix is written, read by softmax, P is written, read by
    the second matmul.
  * Tiled: the Q axis is covered by ceil(Sq/block_m) blocks of block_m rows.
    Each Q block scans ceil(Skv/block_n) KV blocks once, and nothing of size
    Sq x Skv is ever written to global memory. K and V are therefore re-read
    once per Q block.

Two regimes are printed because they differ by orders of magnitude:

  * prefill-like: Sq = Skv = N (every Q block rescans all of K/V);
  * decode-like:  Sq = 1 (the single Q block reads all of K/V exactly once).

Run:  python3 traffic_model.py
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    sq: int         # query length
    skv: int        # key/value length
    d: int          # head dim
    dtype_bytes: int = 2   # fp16/bf16


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def naive_elements(shape: Shape) -> dict[str, int]:
    """Elements moved to/from HBM by the naive formulation."""
    sq, skv, d = shape.sq, shape.skv, shape.d
    s_elems = sq * skv
    return {
        "Q read": sq * d,                # S = Q K^T
        "K read": skv * d,
        "S write": s_elems,
        "S read": s_elems,               # softmax
        "P write": s_elems,
        "P read": s_elems,               # P V
        "V read": skv * d,
        "O write": sq * d,
    }


def tiled_elements(shape: Shape, block_m: int, block_n: int) -> dict[str, int]:
    """Elements moved to/from HBM by the tiled online-softmax formulation.

    Each Q block reads Q once, and reads all of K and V once. Nothing of size
    N x N leaves the chip. The O accumulator lives in registers; the only
    outputs are O and the per-row final scale (we count the scale as negligible
    and say so explicitly).
    """
    sq, skv, d = shape.sq, shape.skv, shape.d
    n_q_blocks = ceil_div(sq, block_m)
    n_kv_blocks = ceil_div(skv, block_n)  # printed for reference; not used below
    del n_kv_blocks
    return {
        "Q read": sq * d,                         # each Q block reads its own rows
        "K read": n_q_blocks * skv * d,           # every Q block rescans all of K
        "V read": n_q_blocks * skv * d,
        "O write": sq * d,
        "S write": 0,                             # never leaves the chip
        "S read": 0,
        "P write": 0,
        "P read": 0,
    }


def total(parts: dict[str, int]) -> int:
    return sum(parts.values())


def report(shape: Shape, block_m: int = 128, block_n: int = 128, label: str = "") -> None:
    naive = naive_elements(shape)
    tiled = tiled_elements(shape, block_m, block_n)
    tn, tt = total(naive), total(tiled)
    sq, skv, d = shape.sq, shape.skv, shape.d

    if label:
        print(f"[{label}]")
    print(f"shape: Sq={sq}, Skv={skv}, d={d}, dtype={shape.dtype_bytes * 8} bit "
          f"({shape.dtype_bytes} B/element)")
    print(f"tiles: block_m={block_m}, block_n={block_n}, "
          f"Q blocks={ceil_div(sq, block_m)}, KV blocks={ceil_div(skv, block_n)}")
    print()
    print(f"{'item':<12}{'naive':>16}{'tiled':>16}")
    print("-" * 44)
    for key in naive:
        print(f"{key:<12}{naive[key]:>16,}{tiled[key]:>16,}")
    print("-" * 44)
    print(f"{'total':<12}{tn:>16,}{tt:>16,}")
    print(f"{'bytes':<12}{tn * shape.dtype_bytes:>16,}{tt * shape.dtype_bytes:>16,}")
    print()
    print("asymptotic terms:")
    print(f"  naive ~ 4*Sq*Skv + 2*(Sq+Skv)*d = "
          f"{4 * sq * skv + 2 * (sq + skv) * d:,} (model total {tn:,})")
    print(f"  tiled ~ 2*ceil(Sq/block_m)*Skv*d = "
          f"{2 * ceil_div(sq, block_m) * skv * d:,} (model total {tt:,})")
    if tt > 0:
        print(f"ratio naive/tiled = {tn / tt:.2f}x")
    print()

    # Crossover for the prefill regime (sq == skv == N): the tiled version
    # re-reads K/V once per Q block, costing 2*(N/block_m)*N*d, against the
    # 4*N*N the naive version pays for materialising S and P.
    if sq == skv:
        n = sq
        print("prefill crossover (Sq == Skv == N, this model):")
        print("  tiled wins while 2*(N/block_m)*N*d < 4*N*N, i.e. block_m > d/2")
        print(f"  -> with d={d}, need block_m > {d / 2:.0f} "
              f"(in practice a multiple of the MMA tile size)")
        print("  This is a property of *this* model, not of real hardware:")
        print("  the model ignores (a) that N x N does not fit in SRAM at all,")
        print("  and (b) that real kernels fuse more than the element counts show.")
    print()


def sweep() -> None:
    print("=" * 60)
    print("A. prefill-like: Sq = Skv = N (block_m = block_n = 128, d = 128)")
    print("=" * 60)
    print(f"{'N':>8}{'naive':>16}{'tiled':>16}{'ratio':>8}")
    for n in (512, 2048, 8192, 32768, 131072):
        shape = Shape(sq=n, skv=n, d=128)
        tn = total(naive_elements(shape))
        tt = total(tiled_elements(shape, 128, 128))
        print(f"{n:>8}{tn:>16,}{tt:>16,}{tn / tt:>7.1f}x")
    print()
    print("note: the tiled column still grows with N (K/V are re-read per Q")
    print("block), but by a factor proportional to d/block_m rather than to N.")
    print()
    print("=" * 60)
    print("B. decode-like: Sq = 1, Skv = N (block_m = block_n = 128, d = 128)")
    print("=" * 60)
    print(f"{'N':>8}{'naive':>16}{'tiled':>16}{'ratio':>8}")
    for n in (512, 2048, 8192, 32768, 131072):
        shape = Shape(sq=1, skv=n, d=128)
        tn = total(naive_elements(shape))
        tt = total(tiled_elements(shape, 128, 128))
        print(f"{n:>8}{tn:>16,}{tt:>16,}{tn / tt:>7.0f}x")
    print()
    print("note: with one query row the naive S matrix is 1 x Skv, not N x N.")
    print("Both columns are O(Skv*d) and the ratio stays ~1x -- so at decode")
    print("time the 4*Sq*Skv term is NOT the problem at all.")
    print("What is missing is parallelism: there is only one query row, so a")
    print("kernel with one block per query row cannot fill the device. That is")
    print("why split-KV (flash decoding) splits the KV axis instead, and why it")
    print("needs an *extra combine kernel* rather than a cheaper main loop.")


def main() -> None:
    print(__doc__)
    report(Shape(sq=8192, skv=8192, d=128), label="prefill 8K")
    sweep()
    print("=" * 60)
    print("REMINDER: these are element counts from a model, not measurements.")
    print("They say nothing about achieved bandwidth on any real device.")


if __name__ == "__main__":
    main()
