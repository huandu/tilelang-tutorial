#!/usr/bin/env python3
"""Which pipe is binding on Ascend, and why the tile has to be big.

The Ascend side does not use a traffic-times-waves heuristic.  It turns every
statement into a task and charges it against a physical pipe, using the
constants in

    _vendor/tilelang-v0.1.15/src/ascend/transform/estimate_latency.h:27-158

and the rules in

    _vendor/tilelang-v0.1.15/src/ascend/transform/estimate_latency.cc
        CubeOperations                       ~:695-707
        per-path memory latency / II         ~:750-870
        Cube latency and II                  ~:900-965

This script extracts the two halves of that model that decide a *tile size* --
the Cube throughput cost and the GM->L1 MTE2 cost -- and solves for the tile at
which they balance.  Everything is in cycles, so no clock and no TFLOPS table is
needed, and every number can be recomputed by hand.

The constants used, verbatim from `AscendLatencyParams`:

    cube_base_latency          {8}       cube completion overhead
    cube_throughput_fp16       {8192}    ops/cycle, a multiply-add counting as 2
    cube_throughput_bf16       {8192}
    cube_throughput_fp32       {512}
    cube_throughput_hf32       {4096}
    cube_throughput_fp8        {16384}
    cube_unit_m/k/n            {16}      fractal edge, and the K alignment
    aic_mte2_to_l1_bandwidth   {100}     bytes/cycle, GM -> L1
    mte2_gm_to_l1_base_latency {190}     completion overhead
    mte2_gm_to_l1_min_ii       {64}      minimum issue interval per descriptor
    l1_to_l0a_bandwidth        {256}     bytes/cycle, L1 -> L0A
    l1_to_l0b_bandwidth        {256}     bytes/cycle, L1 -> L0B
    mte1_base_latency          {5}
    mte1_issue_overhead        {3}       per L0 load descriptor
    mul_throughput             {51}      VALU ops/cycle
    special_func_throughput    {8}       ops/cycle

What this script does **not** model: register/files, UB capacity, the Z3
scheduler's interaction between tasks, and the Vector path's `lanes` scaling.
Chapter 23 covers the pass; this is the back-of-the-envelope that comes before
it.

Pure standard library.  Exit code 0 on success.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# AscendLatencyParams, transcribed
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AscendParams:
    cube_base_latency: int = 8
    cube_unit_k: int = 16
    cube_throughput: dict[str, int] = None  # type: ignore[assignment]
    cube_min_ii: int = 1
    cube_pipeline_depth: int = 1
    aic_mte2_to_l1_bandwidth: int = 100
    mte2_gm_to_l1_base_latency: int = 190
    mte2_gm_to_l1_min_ii: int = 64
    l1_to_l0a_bandwidth: int = 256
    l1_to_l0b_bandwidth: int = 256
    mte1_base_latency: int = 5
    mte1_issue_overhead: int = 3

    def __post_init__(self) -> None:
        if self.cube_throughput is None:
            object.__setattr__(
                self,
                "cube_throughput",
                {"fp16": 8192, "bf16": 8192, "fp32": 512, "hf32": 4096, "fp8": 16384},
            )


VALU_THROUGHPUT = {
    "add": 51,
    "sub": 51,
    "mul": 51,
    "min_max": 51,
    "cmp": 51,
    "logic": 51,
    "div": 13,
    "mod": 13,
    "special_func": 8,
}

DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4, "fp8": 1}


def align_up(v: int, a: int) -> int:
    """CubeOperations aligns K to the 16-element fractal edge."""
    return (v + a - 1) // a * a


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# --------------------------------------------------------------------------- #
# The two costs that decide a tile
# --------------------------------------------------------------------------- #


def cube_cycles(m: int, n: int, k: int, dtype: str, p: AscendParams) -> int:
    """estimate_latency.cc:692-707 + :920.

    ``CubeOperations`` aligns **M and N to 16** and K to ``k_group``, which is
    ``256 / bits`` for 16-bit inputs (so 16) and 1 or 8 for fp32 depending on the
    HF32 mode.  The op count is ``2 * align(M,16) * align(N,16) * align(K,group)``
    -- the leading 2 is the multiply-add counted as two operations.
    """
    k_group = p.cube_unit_k if DTYPE_BYTES[dtype] * 8 == 16 else (8 if dtype == "hf32" else 1)
    ops = 2 * align_up(m, 16) * align_up(n, 16) * align_up(k, k_group)
    return ceil_div(ops, p.cube_throughput[dtype])


def cube_latency(m: int, n: int, k: int, dtype: str, p: AscendParams) -> int:
    return p.cube_base_latency + cube_cycles(m, n, k, dtype, p)


def gm_to_l1_cycles(m: int, n: int, k: int, dtype: str, p: AscendParams,
                    descriptors: int = 2) -> tuple[int, int]:
    """Returns (steady-state II in cycles, completion latency in cycles).

    Bytes are the two operand tiles; the issue interval is
    `descriptors * mte2_gm_to_l1_min_ii` and the transfer interval is
    `bytes / aic_mte2_to_l1_bandwidth`.  The completion latency adds
    `mte2_gm_to_l1_base_latency` on top of the transfer.
    """
    nbytes = DTYPE_BYTES[dtype] * (m * k + k * n)
    transfer = ceil_div(nbytes, p.aic_mte2_to_l1_bandwidth)
    issue = descriptors * p.mte2_gm_to_l1_min_ii
    ii = max(transfer, issue)
    latency = p.mte2_gm_to_l1_base_latency + transfer
    return ii, latency


def l1_to_l0_cycles(m: int, n: int, k: int, dtype: str, p: AscendParams,
                    descriptors: int = 2) -> int:
    a_bytes = DTYPE_BYTES[dtype] * m * k
    b_bytes = DTYPE_BYTES[dtype] * k * n
    return max(
        ceil_div(a_bytes, p.l1_to_l0a_bandwidth),
        ceil_div(b_bytes, p.l1_to_l0b_bandwidth),
        descriptors * p.mte1_issue_overhead,
    )


@dataclass
class TileReport:
    m: int
    n: int
    k: int
    dtype: str
    cube_ii: int
    cube_lat: int
    mte2_ii: int
    mte2_lat: int
    mte1_ii: int
    binding: str
    stages_needed: int


def analyse(m: int, n: int, k: int, dtype: str, p: AscendParams) -> TileReport:
    c_ii = cube_cycles(m, n, k, dtype, p)
    c_lat = cube_latency(m, n, k, dtype, p)
    t_ii, t_lat = gm_to_l1_cycles(m, n, k, dtype, p)
    l_ii = l1_to_l0_cycles(m, n, k, dtype, p)
    pipes = {"Cube": c_ii, "MTE2 GM->L1": t_ii, "MTE1 L1->L0": l_ii}
    binding = max(pipes, key=lambda kk: pipes[kk])
    period = pipes[binding]
    # chapter 11's rule, applied with this pipe's completion latency
    lat = {"Cube": c_lat, "MTE2 GM->L1": t_lat, "MTE1 L1->L0": l_ii}[binding]
    stages = 1 + ceil_div(lat, max(period, 1))
    return TileReport(m, n, k, dtype, c_ii, c_lat, t_ii, t_lat, l_ii, binding, stages)


def print_report(r: TileReport) -> None:
    period = {"Cube": r.cube_ii, "MTE2 GM->L1": r.mte2_ii, "MTE1 L1->L0": r.mte1_ii}[r.binding]
    lat = {"Cube": r.cube_lat, "MTE2 GM->L1": r.mte2_lat, "MTE1 L1->L0": r.mte1_ii}[r.binding]
    print(
        f"tile M={r.m:>4} N={r.n:>4} K={r.k:>4} {r.dtype:<4}"
        f"  Cube II {r.cube_ii:>7}  MTE2 II {r.mte2_ii:>7}  MTE1 II {r.mte1_ii:>6}"
        f"  -> binding {r.binding:<12} needs >= {r.stages_needed} buffers"
        f"  (latency/period = {lat / period:.3f})"
    )


def coverage_sweep(p: AscendParams, dtype: str) -> None:
    k_group = 16 if DTYPE_BYTES[dtype] * 8 == 16 else (8 if dtype == "hf32" else 1)
    print("Cube/MTE2 ratio for square tiles (M and N already multiples of 16,")
    print(f"K = 16 >= the dtype's K group of {k_group}):")
    print(f"{'M=N':>6}{'cube II':>12}{'mte2 II':>12}{'cube/mte2':>12}  binding")
    for side in (64, 128, 192, 256, 320, 384, 512):
        c = cube_cycles(side, side, 16, dtype, p)
        t, _ = gm_to_l1_cycles(side, side, 16, dtype, p)
        print(
            f"{side:>6}{c:>12}{t:>12}{c / t:>12.2f}  "
            f"{'Cube' if c >= t else 'MTE2'}"
        )
    print()
    threshold = p.cube_throughput[dtype] / p.aic_mte2_to_l1_bandwidth
    print("Closed form: the Cube is the bottleneck iff")
    print(f"    max( 2K(M+N)/mte2_bw , descriptors*mte2_min_ii ) <= 2MNK/cube_throughput")
    print(f"which for large tiles (the issue floor is {2 * p.mte2_gm_to_l1_min_ii} cycles)")
    print(f"reduces to  M*N / (M + N) >= cube_throughput / mte2_bw = {threshold:.2f},")
    print(f"    i.e. for a square tile M = N >= {2 * threshold:.1f}")
    print(f"    (the two 2s -- the multiply-add counted as two ops, and the two operands")
    print(f"     K*(M+N) bytes per K step -- cancel; fp16/bf16 against {p.aic_mte2_to_l1_bandwidth} bytes/cycle)")
    print(f"    below that size the binding pipe is MTE2, and below the issue floor")
    print(f"    it is descriptor issue rather than bandwidth.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtype", default="bf16", choices=sorted(DTYPE_BYTES))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--tile", nargs=3, type=int, metavar=("M", "N", "K"),
                    help="analyse one tile, e.g. --tile 256 224 128")
    args = ap.parse_args()

    p = AscendParams()
    print("Ascend pipe model (constants from src/ascend/transform/estimate_latency.h)")
    print()
    coverage_sweep(p, args.dtype)

    official = [(256, 224, 128, "the shape in tilelang/ascend/README.md's GEMM+ReLU example"),
                (128, 128, 32, "a typical NVIDIA-side tile, for contrast: MTE2-bound here"),
                (128, 128, 512, "same footprint, deep K: still MTE2-bound, because the "
                                 "operand bytes grow with K too")]
    for m, n, k, note in official:
        print_report(analyse(m, n, k, args.dtype, p))
        print(f"      {note}")
    print()

    if args.sweep:
        print("binding pipe over a tile sweep")
        for m in (64, 128, 256, 512):
            for n in (64, 128, 256, 512):
                for k in (32, 128):
                    print_report(analyse(m, n, k, args.dtype, p))
        print()
    if args.tile:
        m, n, k = args.tile
        print("requested tile")
        print_report(analyse(m, n, k, args.dtype, p))
        print()

    print("Vector side (VALU), for the elementwise half of a fused operator:")
    for name, tp in VALU_THROUGHPUT.items():
        print(f"    {name:<14}{tp:>5} ops/cycle")
    print(
        f"    a transcendental costs {VALU_THROUGHPUT['mul'] / VALU_THROUGHPUT['special_func']:.2f}x"
        f" what a multiply costs, per op, on this pipe."
    )
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
