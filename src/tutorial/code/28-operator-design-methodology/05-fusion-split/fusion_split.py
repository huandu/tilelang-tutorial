#!/usr/bin/env python3
"""Fuse or split?  A four-line ledger that turns a hunch into a judgement.

Every fusion decision in this book reduces to four questions, and each one is a
number:

    1. traffic      how many bytes does the coarser option move?
    2. capacity     what does each option have to keep on chip at once?
    3. parallelism  how many blocks does each option launch?
    4. epilogue     what new work does the finer option introduce?

The working pattern is: 1 argues for the coarse option, 2/3 argue for the fine
one, and the decision is whichever constraint the numbers actually break.  Any
one of capacity or parallelism can veto; a traffic gain of zero means the
finer option is free to win on the other two.

For each decision below the two options are named explicitly, so the ledger is a
comparison rather than a slogan.  The rule the script applies is deliberately
dumb and therefore checkable:

    an option is usable  iff  it fits on chip  and  its grid covers the machine
    among usable options, pick the one that moves fewer bytes
    if neither is usable, the answer is "neither" -- re-tile, do not choose

In two of the decisions the option that moves *more* bytes wins, because the
other one cannot fill the device.  That is the whole content of split-K and
split-KV, and it is why the rule has to be stated rather than felt.

Nothing here is measured.  Pure standard library.  Exit code 0 on success.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


GB = 1e9


@dataclass
class Option:
    label: str
    bytes: float
    grid: int
    on_chip: float


@dataclass
class Decision:
    name: str
    option_a: Option
    option_b: Option
    budget: float
    bytes_per_us: float
    sm_count: int
    note: str = ""

    def verdict(self) -> tuple[str, str]:
        a, b = self.option_a, self.option_b
        fit_a, fit_b = a.on_chip <= self.budget, b.on_chip <= self.budget
        ok_a = fit_a and a.grid >= self.sm_count
        ok_b = fit_b and b.grid >= self.sm_count

        if not fit_a and not fit_b:
            return "NEITHER", (
                f"capacity: {a.on_chip / 1024:.0f} KB and {b.on_chip / 1024:.0f} KB"
                f" both exceed the {self.budget / 1024:.0f} KB budget; re-tile, do not choose"
            )
        if fit_a and not fit_b:
            return a.label, f"capacity veto on the other option ({b.on_chip / 1024:.0f} KB)"
        if fit_b and not fit_a:
            return b.label, f"capacity veto on the other option ({a.on_chip / 1024:.0f} KB)"
        if ok_a and not ok_b:
            return a.label, (
                f"parallelism: the other option launches {b.grid} blocks"
                f" for {self.sm_count} SMs"
            )
        if ok_b and not ok_a:
            return b.label, (
                f"parallelism: the other option launches {a.grid} blocks"
                f" for {self.sm_count} SMs, so it pays"
                f" {(b.bytes - a.bytes) / 1e6:+.1f} MB to reach {b.grid}"
            )
        if not ok_a and not ok_b:
            return "NEITHER", "neither option both fits on chip and fills the machine"
        # both usable: fewest bytes wins
        if a.bytes <= b.bytes:
            return a.label, (
                f"traffic: {a.bytes / 1e6:.1f} MB vs {b.bytes / 1e6:.1f} MB,"
                f" and both fit"
            )
        return b.label, (
            f"traffic: {b.bytes / 1e6:.1f} MB vs {a.bytes / 1e6:.1f} MB"
            f" (saves {(a.bytes - b.bytes) / 1e6:.1f} MB ="
            f" {(a.bytes - b.bytes) / self.bytes_per_us:.2f} us)"
        )


def build(bw_gbs: float, sm_count: int) -> list[Decision]:
    B = bw_gbs * GB / 1e6  # bytes per microsecond
    out: list[Decision] = []

    # ---------------------------------------------------------------- 1
    # y = sin(cos(x)), n = 16.8M elements, fp16.
    n, b = 16 * 1024 * 1024, 2
    out.append(
        Decision(
            "elementwise chain, 2 ops",
            option_a=Option("2 kernels", 4 * n * b, 2 * (n // 1024), 1024 * b),
            option_b=Option("1 fused kernel", 2 * n * b, n // 1024, 1024 * b),
            budget=228 * 1024,
            bytes_per_us=B,
            sm_count=sm_count,
            note="the textbook case: one read + one write instead of two of each",
        )
    )

    # ---------------------------------------------------------------- 2
    # QK-Norm+RoPE+fp8 quant: fuse all three, or stop after RoPE and quantize later.
    tokens, q_heads, kv_heads, dh = 4096, 32, 8, 128
    rows = tokens * (q_heads + kv_heads)
    qk = rows * dh
    table = 2 * tokens * dh * 4
    out.append(
        Decision(
            "QK-Norm+RoPE then fp8 quant",
            option_a=Option(
                "norm+rope | quant as 2 kernels",
                qk * 2 + qk * 2 + table + qk * 2 + qk * 1 + rows * 4,
                rows,
                8192 * 2,
            ),
            option_b=Option(
                "one fused kernel",
                qk * 2 + qk * 1 + rows * 4 + table,
                rows // 8,
                8192 * 2 * 2 + 8192 * 2 * 4 + 8192,
            ),
            budget=192 * 1024,
            bytes_per_us=B,
            sm_count=sm_count,
            note="the case study of section 6; the row of 8192 elements is the fusion's on-chip cost",
        )
    )

    # ---------------------------------------------------------------- 3
    # GEMM + bias + activation epilogue, 4096^3, fp16 out.
    m = n_ = k = 4096
    mn = m * n_ * 2
    grid_gemm = (m // 128) * (n_ // 128)
    out.append(
        Decision(
            "GEMM + bias + GeLU epilogue",
            option_a=Option("GEMM | epilogue, 2 kernels", 2 * mn + mn, grid_gemm, 128 * 128 * 4),
            option_b=Option("GEMM with fused epilogue", mn, grid_gemm, 128 * 128 * 4 + 128 * 2),
            budget=228 * 1024,
            bytes_per_us=B,
            sm_count=sm_count,
            note="C never leaves the accumulator fragment",
        )
    )

    # ---------------------------------------------------------------- 4
    # split-K on a skinny GEMM: M=N=1024, K=16384, grid is only 8*8=64 blocks.
    m2 = n2 = 1024
    c_bytes = m2 * n2 * 4  # fp32 partials
    split = 8
    out.append(
        Decision(
            "skinny GEMM 1024x1024x16384, split-K=8",
            option_a=Option("no split, 64 blocks", c_bytes, (m2 // 128) * (n2 // 128), 128 * 128 * 4),
            option_b=Option(
                f"split-K={split}",
                c_bytes + 2 * split * c_bytes,
                (m2 // 128) * (n2 // 128) * split,
                128 * 128 * 4,
            ),
            budget=228 * 1024,
            bytes_per_us=B,
            sm_count=sm_count,
            note="the fine option buys parallelism with 2*split*M*N*4 bytes of partials",
        )
    )

    # ---------------------------------------------------------------- 5
    # Flash-decoding: split-KV for a single-token query.
    bh, sq, skv, d = 32, 1, 128 * 1024, 128
    kv_bytes = bh * skv * d * 2 * 2
    splits, kv_block = 64, 256
    out.append(
        Decision(
            "decode attention, sq=1, skv=128k, split-KV=64",
            option_a=Option("one block per head", kv_bytes, bh, 2 * kv_block * d * 2),
            option_b=Option(
                f"split-KV={splits}",
                kv_bytes + 2 * bh * splits * d * 4,
                bh * splits,
                2 * kv_block * d * 2,
            ),
            budget=228 * 1024,
            bytes_per_us=B,
            sm_count=sm_count,
            note="splitting does not cut traffic here at all; it buys occupancy",
        )
    )

    # ---------------------------------------------------------------- 6
    # Fusing a norm into the *head* of the Q projection's output.
    rows_attn = 4096 * 32
    dh2 = 128
    norm_bytes = rows_attn * dh2 * 2
    out.append(
        Decision(
            "norm fused into the Q projection's output",
            option_a=Option("project | norm, 2 kernels", 2 * norm_bytes, rows_attn // 8, 8192 * 2),
            option_b=Option(
                "norm fused into the GEMM epilogue",
                0.0,
                (4096 // 128) * (4096 // 128),
                128 * 128 * 4 + 128 * 128 * 2,
            ),
            budget=228 * 1024,
            bytes_per_us=B,
            sm_count=sm_count,
            note="fusion inherits the GEMM's grid; that is usually good, not bad",
        )
    )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bandwidth-gbs", type=float, default=4502.4,
                   help="effective bandwidth, GB/s (default: calibrated H200)")
    p.add_argument("--sm-count", type=int, default=132)
    args = p.parse_args()

    bw = args.bandwidth_gbs
    print(f"machine: {bw:.1f} GB/s effective ({bw * 1e9 / 1e6:.0f} bytes/us), {args.sm_count} SMs")
    print()
    print(
        f"{'decision':<42}{'A MB':>10}{'B MB':>10}{'delta MB':>10}"
        f"{'A grid':>10}{'B grid':>10}  winner"
    )
    for d in build(bw, args.sm_count):
        verdict, _ = d.verdict()
        print(
            f"{d.name:<42}{d.option_a.bytes / 1e6:>10.1f}{d.option_b.bytes / 1e6:>10.1f}"
            f"{(d.option_a.bytes - d.option_b.bytes) / 1e6:>10.1f}"
            f"{d.option_a.grid:>10}{d.option_b.grid:>10}  {verdict}"
        )
    print()
    for d in build(bw, args.sm_count):
        verdict, reason = d.verdict()
        print(d.name)
        print(f"    {d.note}")
        print(
            f"    A: {d.option_a.label:<34} {d.option_a.bytes / 1e6:8.1f} MB"
            f"  grid {d.option_a.grid:>7}  on chip {d.option_a.on_chip / 1024:7.1f} KB"
        )
        print(
            f"    B: {d.option_b.label:<34} {d.option_b.bytes / 1e6:8.1f} MB"
            f"  grid {d.option_b.grid:>7}  on chip {d.option_b.on_chip / 1024:7.1f} KB"
        )
        print(f"    on-chip budget {d.budget / 1024:.0f} KB; machine has {d.sm_count} SMs")
        print(f"    winner: {verdict} -- {reason}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
