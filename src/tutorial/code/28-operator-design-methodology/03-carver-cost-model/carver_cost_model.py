#!/usr/bin/env python3
"""A CPU re-implementation of TileLang carver's tile cost model (v0.1.15).

The real model lives in the locked source tree:

* traffic     `_vendor/tilelang-v0.1.15/tilelang/carver/roller/policy/default.py:398-432`
* shared mem  `.../roller/policy/default.py:452-495` (+ `roller/bestfit.py`)
* hard caps   `.../roller/policy/default.py:537-574`
* ranking     `.../roller/policy/default.py:96-134`   priority = (traffic + 1) * num_wave
* arch model  `.../carver/arch/cuda.py:153-167`, `.../carver/arch/arch_base.py:6-27`

You cannot import carver on a machine without a GPU: `CUDA.__init__` calls
`tvm.runtime.cuda(0)` and raises `RuntimeError("Cannot find cuda device 0.")`
(`carver/arch/cuda.py:144-146`).  This script therefore re-states the arithmetic
in plain Python so that the *model* -- not the toolchain -- can be studied and
re-run anywhere.

What is faithful:

    traffic per output tile   = sum over input tensors of
                                coalesced_tensor_shape(footprint, full_shape,
                                                       read_transaction_elems) * nbytes
                                where `footprint` is the region of that input the
                                output tile needs over the *whole* reduction
                                extent: A is [block_m, K], B is [block_n, K].
                                Summed over the grid this equals the textbook
                                b*M*N*K*(1/block_N + 1/block_M) + b*M*N, which
                                `--self-test` checks.
    coalesced_factor / coalesced_tensor_shape  copied 1:1 from
                                `roller/policy/common.py:32-52`
    shared memory             = best-fit allocation with 32-byte alignment,
                                as `roller/bestfit.py:22-51` performs it
    reg_usage                 = 2 * max(prod(tile) * bits / 32) over nodes
    block_per_SM              = min(max_smem_usage // smem, reg_cap // reg, sm_partition)
    num_wave                  = ceil(grid_size / (block_per_SM * compute_max_core))
    priority                  = (traffic + 1) * num_wave

What is simplified, and why it does not change the conclusions here:

    * tile propagation is written out analytically for a plain GEMM instead of
      going through `PrimFuncNode.propagate` (`roller/node.py:332-399`).  Which
      region `propagate` returns for a reduction block -- one `rstep` slice or
      the full reduction extent -- can only be settled by running carver on a
      device, which was not possible here; the full-extent reading is the one
      that reproduces the textbook traffic formula, and the script states that
      reading instead of hiding it.  See `../README.md` "known limitations";
    * the candidate walk visits every divisor pair once instead of doing
      carver's best-first `dfs_smem_tile` expansion;
    * `smem_cap` / `compute_max_core` are taken from a command-line flag rather
      than queried from the device, because there is no device here.

Pure standard library.  Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# The device model (copied from carver/arch/cuda.py and arch/arch_base.py)
# --------------------------------------------------------------------------- #


@dataclass
class CudaDevice:
    smem_cap: int = 227 * 1024  # bytes per block, queried at runtime in carver
    max_smem_usage: int = 2 * 227 * 1024  # carver/arch/cuda.py:158
    reg_cap: int = 65536  # carver/arch/cuda.py:157
    sm_partition: int = 4  # carver/arch/cuda.py:159
    compute_max_core: int = 132  # H100/H200 SM count; queried at runtime in carver
    transaction_size: tuple[int, int] = (32, 128)  # carver/arch/cuda.py:162
    warp_size: int = 32

    def __post_init__(self) -> None:
        self.max_smem_usage = 2 * self.smem_cap


# --------------------------------------------------------------------------- #
# carver/roller/policy/common.py:32-52, transcribed
# --------------------------------------------------------------------------- #


def coalesced_factor(subtensor: list[int], tensor: list[int]) -> int:
    if subtensor[-1] != tensor[-1] or len(subtensor) == 1:
        return subtensor[-1]
    return subtensor[-1] * coalesced_factor(subtensor[:-1], tensor[:-1])


def coalesced_tensor_shape(subtensor: list[int], tensor: list[int], transaction_size: int) -> float:
    """carver/roller/policy/common.py:41-52."""
    total = 1
    for v in subtensor:
        total *= v
    if total == 0:
        return 0.0
    factor = coalesced_factor(subtensor, tensor)
    return transaction_size * total / min(transaction_size, factor)


# --------------------------------------------------------------------------- #
# carver/roller/bestfit.py:22-61, transcribed
# --------------------------------------------------------------------------- #


class _Block:
    __slots__ = ("start", "end", "is_free")

    def __init__(self, start: int, end: int, is_free: bool) -> None:
        self.start, self.end, self.is_free = start, end, is_free

    @property
    def size(self) -> int:
        return self.end - self.start


class BestFit:
    """best-fit allocator; carver aligns every request up to 32 bytes."""

    def __init__(self, align: int = 32) -> None:
        self.align = align
        self.limit = 0
        self.blocks: list[_Block] = []

    def malloc(self, size: int) -> _Block:
        size = (size + self.align - 1) // self.align * self.align
        found = None
        for b in self.blocks:
            if b.is_free and b.size >= size and (found is None or found.size > b.size):
                found = b
        if found is not None:
            found.is_free = False
            remain = found.size - size
            if remain:
                found.end -= remain
                self.blocks.insert(self.blocks.index(found) + 1, _Block(found.end, found.end + remain, True))
            return found
        if self.blocks and self.blocks[-1].is_free:
            last = self.blocks[-1]
            last.end += size - last.size
            self.limit = last.end
            last.is_free = False
            return last
        blk = _Block(self.limit, self.limit + size, False)
        self.blocks.append(blk)
        self.limit += size
        return blk

    def free(self, blk: _Block) -> None:
        blk.is_free = True
        self.blocks.sort(key=lambda b: b.start)
        merged: list[_Block] = []
        for b in self.blocks:
            if merged and merged[-1].is_free and b.is_free and merged[-1].end == b.start:
                merged[-1].end = b.end
            else:
                merged.append(b)
        self.blocks = merged


# --------------------------------------------------------------------------- #
# The GEMM shape this script models
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Gemm:
    m: int
    n: int
    k: int
    in_bits: int = 16
    accum_bits: int = 32  # fp32 accumulator -> matters for reg_usage

    @property
    def in_bytes(self) -> int:
        return self.in_bits // 8

    @property
    def out_bytes(self) -> int:
        return self.in_bits // 8


@dataclass
class TileDict:
    tile: tuple[int, int]
    rstep: int
    traffic: float = -1.0
    smem_cost: int = -1
    block_per_SM: int = -1
    num_wave: int = -1
    grid_size: int = -1
    reg_usage: int = -1
    valid: bool = True
    binding: str = ""

    @property
    def prio(self) -> float:
        """roller/policy/default.py:111-112."""
        return (self.traffic + 1) * self.num_wave


def read_transaction_elements(dev: CudaDevice, nbytes: int) -> int:
    """roller/policy/default.py:422: transaction_size[1] // nbytes."""
    return dev.transaction_size[1] // nbytes


def compute_tile_dict(g: Gemm, dev: CudaDevice, block_m: int, block_n: int, block_k: int) -> TileDict:
    td = TileDict(tile=(block_m, block_n), rstep=block_k)
    nbytes = g.in_bytes

    # --- traffic (roller/policy/default.py:398-432), propagated by hand -------
    # A is [M, K] and the output tile needs A[block_m, :]; B is [N, K]
    # (transpose_B) and the output tile needs B[block_n, :]; C is [M, N] and the
    # output tile writes C[block_m, block_n].
    read_elems = read_transaction_elements(dev, nbytes)
    write_elems = dev.transaction_size[0] // nbytes
    a_bytes = coalesced_tensor_shape([block_m, g.k], [g.m, g.k], read_elems) * nbytes
    b_bytes = coalesced_tensor_shape([block_n, g.k], [g.n, g.k], read_elems) * nbytes
    c_bytes = coalesced_tensor_shape([block_m, block_n], [g.m, g.n], write_elems) * g.out_bytes
    grid_size = math.ceil(g.m / block_m) * math.ceil(g.n / block_n)
    td.grid_size = grid_size
    td.traffic = (a_bytes + b_bytes + c_bytes) * grid_size

    # --- shared memory (roller/policy/default.py:452-495 + bestfit) ----------
    # Both input tiles are alive at once; carver obtains the same number by
    # allocating each cached tensor through the best-fit allocator.
    alloc = BestFit()
    alloc.malloc(block_m * block_k * nbytes)
    alloc.malloc(block_n * block_k * nbytes)
    td.smem_cost = alloc.limit

    # --- hard caps (roller/policy/default.py:556-574) ------------------------
    if td.smem_cost > dev.smem_cap:
        td.valid = False
        td.binding = "smem"
        return td

    # reg_usage = 2 * max(prod(tile) * bits / 32)  (per-thread accumulator proxy)
    td.reg_usage = int(2 * max(block_m * block_n, block_m * block_k, block_n * block_k) * g.accum_bits / 32)
    if td.reg_usage > dev.reg_cap:
        td.valid = False
        td.binding = "regs"
        return td

    by_smem = dev.max_smem_usage // max(td.smem_cost, 1)
    by_reg = dev.reg_cap // max(td.reg_usage, 1)
    by_partition = dev.sm_partition
    td.block_per_SM = min(by_smem, by_reg, by_partition)
    td.binding = min(
        (("smem", by_smem), ("regs", by_reg), ("partition", by_partition)), key=lambda kv: kv[1]
    )[0]
    td.num_wave = math.ceil(td.grid_size / max(td.block_per_SM * dev.compute_max_core, 1))
    return td


def divisors(n: int, lo: int = 1, hi: int | None = None) -> list[int]:
    hi = hi or n
    out = [d for d in range(1, int(math.isqrt(n)) + 1) if n % d == 0]
    out += [n // d for d in out if d * d != n]
    return sorted(d for d in out if lo <= d <= hi)


def enumerate_tiles(g: Gemm, dev: CudaDevice, topk: int = 12, cap: int = 4000) -> list[TileDict]:
    """A flattened stand-in for roller/policy/default.py:96-134."""
    seen: dict[tuple[int, int, int], TileDict] = {}
    for bm in divisors(g.m, 16):
        for bn in divisors(g.n, 16):
            for bk in divisors(g.k, 16):
                key = (bm, bn, bk)
                if key in seen:
                    continue
                if len(seen) >= cap:
                    break
                seen[key] = compute_tile_dict(g, dev, bm, bn, bk)
    rejected: dict[str, int] = {}
    best_rejected: dict[str, TileDict] = {}
    for td in seen.values():
        if td.valid:
            continue
        rejected[td.binding] = rejected.get(td.binding, 0) + 1
        prev = best_rejected.get(td.binding)
        if prev is None or td.smem_cost > prev.smem_cost:
            best_rejected[td.binding] = td

    valid = [td for td in seen.values() if td.valid]
    valid.sort(key=lambda td: td.prio)
    print(f"candidates visited : {len(seen)}   valid : {len(valid)}")
    for reason, count in sorted(rejected.items()):
        td = best_rejected[reason]
        print(
            f"rejected on {reason:<9}: {count:>4}"
            f"   (largest such tile block_M={td.tile[0]} block_N={td.tile[1]}"
            f" block_K={td.rstep}, smem={td.smem_cost / 1024:.1f} KB,"
            f" reg_usage={td.reg_usage})"
        )
    print(f"model priority values in use : {len({round(td.prio) for td in valid})}")
    print(f"ties at the top     : {sum(1 for td in valid if round(td.prio) == round(valid[0].prio))}")
    print()
    print(
        f"{'#':>3} {'block_M':>8}{'block_N':>8}{'block_K':>8}"
        f"{'traffic MB':>12}{'smem KB':>9}{'blk/SM':>7}{'waves':>6}{'prio':>12}  binding"
    )
    for i, td in enumerate(valid[:topk], 1):
        bm, bn = td.tile
        print(
            f"{i:>3} {bm:>8}{bn:>8}{td.rstep:>8}"
            f"{td.traffic / 1e6:>12.1f}{td.smem_cost / 1024:>9.1f}"
            f"{td.block_per_SM:>7}{td.num_wave:>6}{td.prio:>12.3g}  {td.binding}"
        )
    print()
    return valid


def check_against_closed_form(g: Gemm, dev: CudaDevice) -> None:
    """Self-test: the propagated traffic must equal the textbook formula."""
    print("self-test: carver-style traffic vs the closed form"
          " b*M*N*K*(1/block_N + 1/block_M) + b*M*N")
    print(f"{'block_M':>8}{'block_N':>8}{'block_K':>8}{'model MB':>12}{'closed MB':>12}{'rel err':>10}")
    for bm, bn, bk in ((64, 64, 64), (128, 128, 32), (128, 256, 64), (256, 128, 64)):
        td = compute_tile_dict(g, dev, bm, bn, bk)
        b = g.in_bytes
        closed = b * g.m * g.n * g.k * (1 / bn + 1 / bm) + b * g.m * g.n
        err = abs(td.traffic - closed) / closed
        print(
            f"{bm:>8}{bn:>8}{bk:>8}{td.traffic / 1e6:>12.1f}{closed / 1e6:>12.1f}{err:>10.2e}"
        )
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--smem-cap", type=int, default=227 * 1024, help="bytes per block")
    p.add_argument("--cores", type=int, default=132, help="SM count")
    p.add_argument("--topk", type=int, default=12)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    g = Gemm(args.m, args.n, args.k)
    dev = CudaDevice(smem_cap=args.smem_cap, compute_max_core=args.cores)

    print(f"GEMM {g.m}x{g.n}x{g.k}  fp{g.in_bits} in / fp{g.accum_bits} accum")
    print(
        f"device: smem_cap={dev.smem_cap} B  max_smem_usage={dev.max_smem_usage} B"
        f"  reg_cap={dev.reg_cap}  sm_partition={dev.sm_partition}  SMs={dev.compute_max_core}"
    )
    print(f"self-test flag: {'on' if args.self_test else 'off (pass --self-test)'}")
    print()
    if args.self_test:
        check_against_closed_form(g, dev)
    enumerate_tiles(g, dev, topk=args.topk)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
