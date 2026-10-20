#!/usr/bin/env python3
"""Tile-size decision procedure: coarse arithmetic first, then a short shortlist.

Four kinds of constraint decide a tile size, and they bind in a fixed order of
surprise.  This script checks all four for a GEMM-shaped tile and tells you which
one is binding for each candidate.

    C1 on-chip capacity   stages * (block_M*block_K + block_K*block_N) * elem_bytes
                          <= usable shared memory per block
    C2 register pressure  accumulators per thread, rounded to the 256-registers-
                          per-warp granularity, <= 255 registers per thread
    C3 instruction shape  block_M/block_N/block_K must be multiples of the MMA
                          intrinsic's shape
    C4 parallelism        grid_size / (blocks_per_SM * SMs) -> waves and the fill
                          ratio of the last wave

Sources (all verified against the corpus or the locked tree):

    shared memory per SM / per block, carveout   corpus 05: nvidia-hopper-tuning-guide
    registers per SM 65536, 2048 threads, 32/thread for 100% occupancy,
    256 registers/warp allocation granularity    corpus 05: nvidia-cuda-best-practices:3126
    mma   dense f16 shapes m16n8k8 / m16n8k16    corpus 05: nvidia-ptx-isa:23476-23477
    wgmma dense f16 shapes m64nNk16              corpus 05: nvidia-ptx-isa:28132
    warpgroup = 4 contiguous warps = 128 threads corpus 05: nvidia-ptx-isa:28113-28114

Nothing here measures anything: every number is a formula.  Pass `--diagnose`
to check a single configuration, or let it sweep and print the shortlist.

Pure standard library.  Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field


@dataclass
class Device:
    name: str = "H100/H200 (sm_90)"
    smem_per_sm: int = 228 * 1024  # Hopper tuning guide: 228 KB configurable
    smem_per_block_max: int = 227 * 1024  # 227 KB per block
    regs_per_sm: int = 65536
    max_threads_per_sm: int = 2048
    max_threads_per_block: int = 1024
    max_blocks_per_sm: int = 32
    regs_per_thread_max: int = 255
    reg_granularity_per_warp: int = 256  # registers are allocated per warp
    warp_size: int = 32
    sm_count: int = 132


@dataclass
class Config:
    block_m: int
    block_n: int
    block_k: int
    stages: int = 3
    threads: int = 128
    instr: str = "wgmma"


@dataclass
class Verdict:
    config: Config
    smem_per_stage: int = 0
    smem_total: int = 0
    accum_elems_per_thread: int = 0
    regs_per_thread: int = 0
    regs_per_block: int = 0
    blocks_by_regs: int = 0
    blocks_by_smem: int = 0
    blocks_by_threads: int = 0
    blocks_per_sm: int = 0
    grid: int = 0
    waves: int = 0
    last_wave_fill: float = 0.0
    dram_traffic: float = 0.0
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    binding: str = ""


# --------------------------------------------------------------------------- #
# Instruction-shape constraints
# --------------------------------------------------------------------------- #

INSTR_SHAPES = {
    # name: (M multiple, N multiple, K multiple, threads must be multiple of)
    "mma": (16, 8, 16, 32),
    "wgmma": (64, 8, 16, 128),
    "simt": (1, 1, 1, 32),
}


def evaluate(cfg: Config, dev: Device, elem_bytes: int = 2, accum_bytes: int = 4) -> Verdict:
    v = Verdict(config=cfg)

    # ---- C3: instruction shape (checked first: it is the cheapest to get wrong)
    m_mult, n_mult, k_mult, thread_mult = INSTR_SHAPES[cfg.instr]
    if cfg.block_m % m_mult:
        v.failures.append(f"C3 block_M={cfg.block_m} is not a multiple of {m_mult} ({cfg.instr})")
    if cfg.block_n % n_mult:
        v.failures.append(f"C3 block_N={cfg.block_n} is not a multiple of {n_mult} ({cfg.instr})")
    if cfg.block_k % k_mult:
        v.failures.append(f"C3 block_K={cfg.block_k} is not a multiple of {k_mult} ({cfg.instr})")
    if cfg.threads % thread_mult:
        v.failures.append(f"C3 threads={cfg.threads} is not a multiple of {thread_mult} ({cfg.instr})")
    if cfg.threads > dev.max_threads_per_block:
        v.failures.append(f"C3 threads={cfg.threads} exceeds {dev.max_threads_per_block} per block")

    # ---- C1: on-chip capacity
    v.smem_per_stage = (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n) * elem_bytes
    v.smem_total = v.smem_per_stage * cfg.stages
    if v.smem_total > dev.smem_per_block_max:
        v.failures.append(
            f"C1 shared {v.smem_total} B > {dev.smem_per_block_max} B per block "
            f"at stages={cfg.stages}"
        )

    # ---- C2: register pressure
    v.accum_elems_per_thread = math.ceil(cfg.block_m * cfg.block_n / cfg.threads)
    raw_regs = v.accum_elems_per_thread * accum_bytes // 4
    # registers are allocated per warp, rounded up to 256 per warp
    per_warp = math.ceil(raw_regs * dev.warp_size / dev.reg_granularity_per_warp) * dev.reg_granularity_per_warp
    v.regs_per_thread = per_warp // dev.warp_size
    if v.regs_per_thread > dev.regs_per_thread_max:
        v.failures.append(
            f"C2 accumulators alone need {v.regs_per_thread} registers/thread "
            f"(> {dev.regs_per_thread_max}); expect spilling to local memory"
        )
    v.regs_per_block = v.regs_per_thread * cfg.threads

    # ---- occupancy, then C4: parallelism
    v.blocks_by_regs = dev.regs_per_sm // max(v.regs_per_block, 1)
    v.blocks_by_smem = dev.smem_per_sm // max(v.smem_total, 1)
    v.blocks_by_threads = dev.max_threads_per_sm // max(cfg.threads, 1)
    v.blocks_per_sm = max(
        min(v.blocks_by_regs, v.blocks_by_smem, v.blocks_by_threads, dev.max_blocks_per_sm), 0
    )
    if v.blocks_per_sm == 0:
        v.failures.append("C1/C2 no resident block fits: blocks_per_SM = 0")
    else:
        v.binding = min(
            (("regs", v.blocks_by_regs), ("smem", v.blocks_by_smem), ("threads", v.blocks_by_threads)),
            key=lambda kv: kv[1],
        )[0]
    return v


def finish_parallelism(v: Verdict, m: int, n: int, dev: Device) -> Verdict:
    cfg = v.config
    v.grid = math.ceil(m / cfg.block_m) * math.ceil(n / cfg.block_n)
    resident = max(v.blocks_per_sm * dev.sm_count, 1)
    v.waves = math.ceil(v.grid / resident)
    v.last_wave_fill = (v.grid % resident) / resident if v.grid % resident else 1.0
    if v.waves > 1 and v.last_wave_fill < 0.5:
        v.warnings.append(
            f"C4 last wave only {v.last_wave_fill * 100:.0f}% full "
            f"({v.grid} tiles / {resident} resident): tail effect"
        )
    if v.grid < dev.sm_count:
        v.warnings.append(
            f"C4 only {v.grid} tiles for {dev.sm_count} SMs: more than half the machine is idle"
        )
    return v


def print_verdict(v: Verdict) -> None:
    cfg = v.config
    print(
        f"block=({cfg.block_m:>4},{cfg.block_n:>4},{cfg.block_k:>3})  stages={cfg.stages}"
        f"  threads={cfg.threads:>4}  instr={cfg.instr}"
    )
    print(
        f"  C1 shared/stage {v.smem_per_stage / 1024:8.1f} KB -> total {v.smem_total / 1024:8.1f} KB"
    )
    print(
        f"  C2 accumulators {v.accum_elems_per_thread:5d}/thread"
        f" -> {v.regs_per_thread:4d} regs/thread (warp-rounded),"
        f" {v.regs_per_block:6d} regs/block"
    )
    print(
        f"     occupancy: regs -> {v.blocks_by_regs} blk/SM,"
        f" smem -> {v.blocks_by_smem}, threads -> {v.blocks_by_threads}"
        f"  => {v.blocks_per_sm} blk/SM ({v.blocks_per_sm} of the 32-block hardware limit),"
        f" binding={v.binding}"
    )
    if v.dram_traffic:
        print(f"  C0 DRAM traffic {v.dram_traffic / 1e6:9.1f} MB"
              f"  (model: b*M*N*K*(1/block_N + 1/block_M) + b*M*N)")
    print(
        f"  C4 grid {v.grid:6d} tiles over {v.blocks_per_sm * v.config.threads // 32:3d}-warp blocks"
        f" -> {v.waves} wave(s), last wave {v.last_wave_fill * 100:5.1f}% full"
    )
    for f in v.failures:
        print(f"  FAIL  {f}")
    for w in v.warnings:
        print(f"  warn  {w}")
    if not v.failures:
        print("  OK    all four constraints satisfied")
    print()


def dram_traffic(m: int, n: int, k: int, cfg: Config, elem_bytes: int = 2) -> float:
    return elem_bytes * m * n * k * (1 / cfg.block_n + 1 / cfg.block_m) + elem_bytes * m * n


def sweep(m: int, n: int, k: int, dev: Device, cfg: Config, topk: int) -> list[Verdict]:
    results: list[Verdict] = []
    for bm in (16, 32, 64, 128, 256):
        for bn in (16, 32, 64, 128, 256):
            for bk in (16, 32, 64, 128):
                if bk > k:
                    continue
                for stages in (2, 3, 4):
                    for threads in (128, 256, 512):
                        if threads % INSTR_SHAPES[cfg.instr][3]:
                            continue
                        c = Config(bm, bn, bk, stages, threads, cfg.instr)
                        v = evaluate(c, dev)
                        if v.failures:
                            continue
                        finish_parallelism(v, m, n, dev)
                        v.dram_traffic = dram_traffic(m, n, k, c)
                        results.append(v)
    # The design order is: minimise DRAM traffic subject to the four constraints,
    # then look at occupancy and the tail.  Sorting by traffic first is what makes
    # this a *design* sweep instead of a "smallest tile wins" sweep.
    results.sort(key=lambda x: (round(x.dram_traffic), -x.blocks_per_sm, x.smem_total))
    print(f"feasible configurations found: {len(results)}")
    print(
        f"{'block_M':>8}{'block_N':>8}{'block_K':>8}{'st':>3}{'thr':>5}"
        f"{'trafficMB':>10}{'smemKB':>8}{'regs/thr':>9}{'blk/SM':>7}{'waves':>6}{'tail':>7}  binding"
    )
    for v in results[:topk]:
        c = v.config
        print(
            f"{c.block_m:>8}{c.block_n:>8}{c.block_k:>8}{c.stages:>3}{c.threads:>5}"
            f"{v.dram_traffic / 1e6:>10.1f}{v.smem_total / 1024:>8.1f}{v.regs_per_thread:>9}"
            f"{v.blocks_per_sm:>7}{v.waves:>6}{v.last_wave_fill * 100:>6.0f}%  {v.binding}"
        )
    print()
    best_traffic = round(results[0].dram_traffic) if results else 0
    n_at_best = sum(1 for v in results if round(v.dram_traffic) == best_traffic)
    print(f"configurations tied at the minimum traffic: {n_at_best}")
    best_fill = max(v.last_wave_fill for v in results) if results else 0.0
    print(f"best last-wave fill among feasible configurations: {best_fill * 100:.0f}%")
    print("the traffic-optimal tile and the occupancy-optimal tile are almost never the same;")
    print("that tension is what section 4 of the chapter is about.")
    print()
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--instr", default="wgmma", choices=sorted(INSTR_SHAPES))
    p.add_argument("--sm-count", type=int, default=132)
    p.add_argument("--smem-kb", type=int, default=228, help="shared memory per SM, KB")
    p.add_argument("--block-m", type=int, default=128)
    p.add_argument("--block-n", type=int, default=128)
    p.add_argument("--block-k", type=int, default=32)
    p.add_argument("--stages", type=int, default=3)
    p.add_argument("--threads", type=int, default=256)
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--diagnose", action="store_true")
    args = p.parse_args()

    dev = Device(sm_count=args.sm_count, smem_per_sm=args.smem_kb * 1024,
                 smem_per_block_max=args.smem_kb * 1024 - 1024)
    print(f"device: {dev.name}  SMs={dev.sm_count}  shared/SM={dev.smem_per_sm / 1024:.0f} KB"
          f"  shared/block<={dev.smem_per_block_max / 1024:.0f} KB  regs/SM={dev.regs_per_sm}")
    print(f"target: {args.m}x{args.n}x{args.k} fp16 -> fp32 accum, intrinsic={args.instr}")
    print()

    if args.diagnose:
        cfg = Config(args.block_m, args.block_n, args.block_k, args.stages, args.threads, args.instr)
        v = evaluate(cfg, dev)
        finish_parallelism(v, args.m, args.n, dev)
        print("diagnosis of the requested configuration")
        print("-" * 78)
        print_verdict(v)
        return 0

    sweep(args.m, args.n, args.k, dev, Config(16, 16, 16, 3, 128, args.instr), args.topk)
    print("now diagnose a shortlisted configuration, e.g.")
    print("  python3 tile_decision.py --diagnose --block-m 128 --block-n 256 --block-k 64"
          " --stages 3 --threads 256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
