#!/usr/bin/env python3
"""Worked case study: QK-Norm + RoPE + fp8 quantization, budget table -> v1 -> v2.

The operator is an inference-side attention prologue:

    q,k = proj(x)                         # bf16, [tokens, heads, head_dim]
    q,k = rmsnorm(q,k) per (token, head)  # over head_dim
    q,k = rope(q,k)                       # same table for every head at a token
    q8,k8, scale = fp8_quant(q,k)         # one fp32 scale per row

It runs on every token of every layer, and it is three passes of pure elementwise
work over the same data, so the only interesting question is how much DRAM traffic
the design actually causes.

The script prints, for three designs:

    A: one row (one (token, head) pair of `head_dim` elements) per block
    B: `--rows-per-block` consecutive rows per block, aligned to head groups
    C: one whole token (all q and k heads) per block

    * the on-chip budget (bytes the block must hold at once)
    * the grid and the wave count
    * the modelled DRAM traffic, split into payload and lookup

"Lookup" is the part that a first design always gets wrong: the norm weight and
the RoPE cos/sin table are small, so people stop counting them -- and then they
are re-read once per row instead of once per token.

Same convention as the rest of the book: `bytes` is distinct traffic for this
call, `flops` follows the ledger rule, and nothing is measured.  Pure standard
library.  Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field


GB = 1e9


@dataclass
class Shape:
    tokens: int = 4096
    q_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    in_bytes: int = 2  # bf16
    out_bytes: int = 1  # fp8
    table_bytes: int = 4  # cos/sin kept in fp32
    smem_budget: int = 192 * 1024  # UB on Ascend / shared on NVIDIA, chosen per design
    sm_count: int = 132
    blocks_per_sm: int = 8

    @property
    def rows(self) -> int:
        return self.tokens * (self.q_heads + self.kv_heads)

    @property
    def elems(self) -> int:
        return self.rows * self.head_dim


@dataclass
class Design:
    name: str
    rows_per_block: int
    shape: Shape
    lookup_reads_per_token: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    # ---------------------------------------------------------------- ledger

    def flops(self) -> float:
        s = self.shape
        rows = s.rows
        dh = s.head_dim
        return (
            2.0 * rows * dh  # norm: sum of squares
            + 3.0 * rows  # norm: rsqrt + two scales
            + 2.0 * rows * dh  # norm: multiply by weight
            + 6.0 * (dh // 2) * rows  # RoPE: 4 mul + 2 add per pair
            + 1.0 * rows * dh  # quant: absmax
            + 2.0 * rows * dh  # quant: scale + convert
        )

    def payload_bytes(self) -> float:
        s = self.shape
        elems = s.elems
        return (
            elems * s.in_bytes
            + elems * s.out_bytes
            + s.rows * 4  # one fp32 scale per row
        )

    def ideal_lookup_bytes(self) -> float:
        """What a perfect design moves: weight and table once per token."""
        s = self.shape
        weight = 2 * s.head_dim * s.in_bytes  # q weight and k weight
        table = 2 * s.tokens * s.head_dim * s.table_bytes  # cos and sin
        return weight + table

    def modelled_lookup_bytes(self) -> float:
        """What the design actually moves, given how often it re-reads."""
        s = self.shape
        blocks = math.ceil(s.rows / self.rows_per_block)
        weight = blocks * (2 * s.head_dim * s.in_bytes)
        table = self.lookup_reads_per_token * s.tokens * (2 * s.head_dim * s.table_bytes)
        return weight + table

    def total_bytes(self) -> float:
        return self.payload_bytes() + self.modelled_lookup_bytes()

    # -------------------------------------------------------------- on-chip

    def on_chip_bytes(self) -> float:
        s = self.shape
        rows = self.rows_per_block
        # input tile (bf16) + output tile (fp8) + fp32 staging for the norm
        # reduction and the absmax + the cos/sin row
        return (
            rows * s.head_dim * s.in_bytes
            + rows * s.head_dim * s.out_bytes
            + rows * s.head_dim * 4
            + rows * 4
            + 2 * s.head_dim * s.table_bytes
        )

    def grid(self) -> int:
        return math.ceil(self.shape.rows / self.rows_per_block)

    def waves(self, blocks_per_sm: int) -> int:
        resident = max(self.shape.sm_count * blocks_per_sm, 1)
        return math.ceil(self.grid() / resident)


def report(shape: Shape, designs: list[Design], bandwidth_gbs: float) -> None:
    B = bandwidth_gbs * GB  # bytes/s
    ideal = designs[0].payload_bytes() + designs[0].ideal_lookup_bytes()

    print("=" * 96)
    print("part 0 -- the operator ledger (this part is design-independent)")
    print("=" * 96)
    d0 = designs[0]
    flops = d0.flops()
    print(f"tokens={shape.tokens}  q_heads={shape.q_heads}  kv_heads={shape.kv_heads}"
          f"  head_dim={shape.head_dim}  in=bf16  out=fp8")
    print(f"rows (token,head) pairs : {shape.rows}")
    print(f"elements                : {shape.elems}")
    print(f"flops                   : {flops / 1e6:.1f} MFLOP")
    print(f"ideal bytes             : {ideal / 1e6:.2f} MB"
          f"   (payload {d0.payload_bytes() / 1e6:.2f} MB + lookup {d0.ideal_lookup_bytes() / 1e6:.2f} MB)")
    print(f"arithmetic intensity    : {flops / ideal:.3f} FLOP/Byte")
    print(f"memory roof at {bandwidth_gbs:.1f} GB/s : {ideal / B * 1e6:.3f} us")
    print("judgement               : far below the balance point of every roof on the chip,")
    print("                          so this operator is memory-bound and the design question is")
    print("                          'who moves the fewest bytes', not 'who uses the best unit'.")
    print()

    print("=" * 96)
    print("part 1 -- budget table")
    print("=" * 96)
    print(
        f"{'design':<26}{'rows/blk':>9}{'grid':>8}{'waves':>7}{'on-chip KB':>12}"
        f"{'payload MB':>12}{'lookup MB':>11}{'total MB':>10}{'vs ideal':>10}"
    )
    for d in designs:
        print(
            f"{d.name:<26}{d.rows_per_block:>9}{d.grid():>8}{d.waves(shape.blocks_per_sm):>7}"
            f"{d.on_chip_bytes() / 1024:>12.1f}"
            f"{d.payload_bytes() / 1e6:>12.2f}"
            f"{d.modelled_lookup_bytes() / 1e6:>11.2f}"
            f"{d.total_bytes() / 1e6:>10.2f}"
            f"{d.total_bytes() / ideal:>9.2f}x"
        )
    print(f"on-chip budget per block: {shape.smem_budget / 1024:.0f} KB"
          f"   SMs: {shape.sm_count}   assumed resident blocks/SM: {shape.blocks_per_sm}")
    over = [d for d in designs if d.on_chip_bytes() > shape.smem_budget]
    for d in over:
        print(f"  FAIL {d.name}: on-chip need {d.on_chip_bytes() / 1024:.1f} KB exceeds the budget")
    print()

    print("=" * 96)
    print("part 2 -- diagnosis")
    print("=" * 96)
    for d in designs:
        lookup = d.modelled_lookup_bytes()
        total = d.total_bytes()
        excess = total - ideal
        lookup_share = lookup / total
        print(f"{d.name}")
        print(
            f"    total {total / 1e6:7.2f} MB = payload {d.payload_bytes() / 1e6:6.2f} MB"
            f" + lookup {lookup / 1e6:7.2f} MB"
            f"   (lookup is {lookup_share * 100:5.1f}% of all traffic)"
        )
        print(
            f"    over the ideal by {excess / 1e6:7.2f} MB"
            f"  -> the achievable time floor rises from {ideal / B * 1e6:.2f} us"
            f" to {total / B * 1e6:.2f} us"
        )
        print(f"    what to fix: {d.extra.get('fix', '-')}")
        print()

    print("=" * 96)
    print("part 3 -- what each design would look like to a profiler")
    print("=" * 96)
    for d in designs:
        achieved = d.total_bytes() / B * 1e6
        print(
            f"{d.name:<26} SOL(ideal) {ideal / B * 1e6:7.3f} us   "
            f"this design's floor {achieved:7.3f} us   "
            f"ratio {achieved / (ideal / B * 1e6):5.2f}x"
        )
    print()
    print("If a measured time comes back close to the *last* column while the kernel's")
    print("achieved bandwidth looks healthy, the gap is the design, not the tuning:")
    print("the kernel is moving more bytes than the algorithm requires.")
    print()


def default_shape() -> Shape:
    return Shape()


def default_designs(shape: Shape) -> list[Design]:
    per_token_blocks = (shape.q_heads + shape.kv_heads)  # rows per token
    return [
        Design(
            "A: one row per block",
            rows_per_block=1,
            shape=shape,
            lookup_reads_per_token=float(per_token_blocks),
            extra={
                "fix": "the weight and the table are re-read once per row; nothing is amortised",
            },
        ),
        Design(
            "B: 8 rows per block",
            rows_per_block=8,
            shape=shape,
            lookup_reads_per_token=float(math.ceil(per_token_blocks / 8)),
            extra={
                "fix": "amortise the table over a head group; keep groups inside one token",
            },
        ),
        Design(
            "C: one whole token per block",
            rows_per_block=per_token_blocks,
            shape=shape,
            lookup_reads_per_token=1.0,
            extra={
                "fix": "the best traffic; requires the two norm weights and the table to be indexed per row",
            },
        ),
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--q-heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--bandwidth-gbs", type=float, default=4502.4)
    p.add_argument("--sm-count", type=int, default=132)
    p.add_argument("--on-chip-kb", type=int, default=192)
    p.add_argument("--rows-per-block", type=int, default=8, help="rows per block for design B")
    args = p.parse_args()

    shape = Shape(
        tokens=args.tokens,
        q_heads=args.q_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        sm_count=args.sm_count,
        smem_budget=args.on_chip_kb * 1024,
    )
    designs = default_designs(shape)
    designs[1] = Design(
        f"B: {args.rows_per_block} rows per block",
        rows_per_block=args.rows_per_block,
        shape=shape,
        lookup_reads_per_token=float(
            math.ceil((shape.q_heads + shape.kv_heads) / args.rows_per_block)
        ),
        extra={"fix": "amortise the table over a head group; keep groups inside one token"},
    )
    report(shape, designs, args.bandwidth_gbs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
