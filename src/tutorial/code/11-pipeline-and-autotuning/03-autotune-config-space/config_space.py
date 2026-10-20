#!/usr/bin/env python3
"""A miniature stand-in for the TileLang autotuner's search space handling.

The real tuner lives in ``_vendor/tilelang-v0.1.15/tilelang/autotuner/`` and
needs a GPU to be meaningful. This script reimplements *only* the parts that
decide whether a search is worth running, so they can be exercised offline:

  1. build a Cartesian product of tunable parameters (what
     ``examples/gemm/example_gemm_autotune.py`` does with ``itertools.product``);
  2. prune configurations the hardware cannot accept (shared-memory budget,
     tile divisibility, thread-count legality);
  3. rank the survivors with a deliberately simple analytic cost model.

The point of the exercise is the third column of every table: how many of the
configurations you asked for are actually *meaningful*. A search space whose
legal subset is empty, or whose legal subset is one config, or whose cost model
cannot separate the legal subset, wastes tuning time.

Nothing here runs on hardware and no number is a measured latency.

Usage
-----
    python3 config_space.py                  # a well-formed space
    python3 config_space.py --mode too-wide  # a space that is pruned to nothing
    python3 config_space.py --mode degenerate
    python3 config_space.py --mode all
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field

DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1}

# Model constants for the surrogate cost model below. They are illustrative
# numbers chosen so the surrogate behaves like a compute-bound GEMM; they are
# NOT a device specification and are not used to predict any real latency.
SMEM_BYTES_PER_CYCLE = 32.0
"""Model input: sustained copy-engine bytes per cycle."""
MAC_PER_CYCLE = 2048.0
"""Model input: sustained matrix-multiply MACs per cycle."""
COPY_LATENCY = 400.0
"""Model input: cycles between issuing a tile transfer and its data arriving."""
FIXED_PER_TILE = 200.0
"""Model input: per-tile overhead (epilogue, store, launch of the tile)."""
OCCUPANCY_PENALTY = 0.5
"""Model input: how much a tile of 128x128 extra area costs in register pressure.

Larger fragments mean more registers per thread and fewer resident blocks, so
the achieved compute rate falls. The real curve is not linear; this term only
needs to make the surrogate non-degenerate.
"""


@dataclass(frozen=True)
class Problem:
    m: int = 4096
    n: int = 4096
    k: int = 4096
    dtype: str = "fp16"
    smem_budget: int = 228 * 1024
    """Shared-memory budget in bytes; see code/02 for the arithmetic and the
    provenance of the H100 per-SM 228 KB figure."""
    max_threads: int = 1024


@dataclass(frozen=True)
class Config:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    threads: int


@dataclass
class Report:
    total: int = 0
    kept: list[Config] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


def smem_bytes(cfg: Config, dtype: str) -> int:
    elem = DTYPE_BYTES[dtype]
    return cfg.num_stages * (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n) * elem


def check(cfg: Config, problem: Problem) -> str | None:
    """Return a rejection reason, or None when the configuration is legal.

    The rules are the ones that fall out of 第 11 章 and the earlier chapters:
    a tile must divide the problem, a block must not ask for more shared memory
    than the device has, and the thread count must be a legal block size.
    """
    if cfg.num_stages < 1:
        return "num_stages < 1"
    if cfg.threads % 32 != 0 or cfg.threads > problem.max_threads:
        return "illegal thread count"
    if cfg.block_m <= 0 or cfg.block_n <= 0 or cfg.block_k <= 0:
        return "non-positive tile"
    if cfg.block_m % 16 or cfg.block_n % 16 or cfg.block_k % 16:
        return "tile not a multiple of 16"
    if problem.m % cfg.block_m or problem.n % cfg.block_n or problem.k % cfg.block_k:
        return "tile does not divide the problem"
    if smem_bytes(cfg, problem.dtype) > problem.smem_budget:
        return "shared memory over budget"
    return None


def cost(cfg: Config, problem: Problem) -> float:
    """A crude analytic surrogate for total latency (arbitrary units, lower is better).

    It is built from the two ideas of 第 11 章:

    * per reduction step, the loop advances at ``max(copy_issue, compute)``;
    * the copy engine can only keep ``num_stages`` transfers in flight, so by
      Little's law the sustained per-step cost is ``max(copy_issue,
      latency / num_stages)`` until the depth is large enough to hide the
      latency entirely.

    It deliberately ignores register pressure, occupancy and wave quantisation,
    which is why the chapter treats it as a pruning/ranking aid only.
    """
    elem = DTYPE_BYTES[problem.dtype]
    steps = problem.k / cfg.block_k
    tiles = (problem.m / cfg.block_m) * (problem.n / cfg.block_n)

    copy_issue = (cfg.block_m + cfg.block_n) * cfg.block_k * elem / SMEM_BYTES_PER_CYCLE
    compute = cfg.block_m * cfg.block_n * cfg.block_k / MAC_PER_CYCLE
    area_ratio = (cfg.block_m * cfg.block_n) / (128.0 * 128.0)
    compute *= 1.0 + OCCUPANCY_PENALTY * max(0.0, area_ratio - 0.25)
    sustained_copy = max(copy_issue, COPY_LATENCY / cfg.num_stages)
    per_step = max(sustained_copy, compute)

    return tiles * (FIXED_PER_TILE + steps * per_step)


def build_space(mode: str) -> tuple[Problem, list[Config]]:
    problem = Problem()
    if mode == "all-infeasible":
        # The space was written without checking the on-chip budget: every
        # candidate asks for more shared memory than any block on this device
        # can hold, so the tuner would spend the whole budget compiling and
        # benchmarking nothing but failures.
        problem = Problem(smem_budget=16 * 1024)
        block_ms, block_ns = [256, 512], [256, 512]
        block_ks = [64, 128]
        stages = [2, 3, 4]
        threads = [128, 256]
    elif mode == "degenerate":
        # Only one legal configuration: the search cannot learn anything.
        block_ms, block_ns = [128], [128]
        block_ks = [64]
        stages = [2]
        threads = [128]
    else:
        block_ms = [64, 128, 256]
        block_ns = [64, 128, 256]
        block_ks = [16, 32, 64]
        stages = [1, 2, 3, 4]
        threads = [128, 256]

    configs = [
        Config(bm, bn, bk, s, th)
        for bm, bn, bk, s, th in itertools.product(
            block_ms, block_ns, block_ks, stages, threads
        )
    ]
    return problem, configs


def run(mode: str, topk: int) -> None:
    problem, configs = build_space(mode)
    report = Report(total=len(configs))
    for cfg in configs:
        reason = check(cfg, problem)
        if reason is None:
            report.kept.append(cfg)
        else:
            report.reject(reason)

    print(f"mode = {mode}")
    print(
        f"problem: M={problem.m} N={problem.n} K={problem.k} {problem.dtype}, "
        f"shared budget {problem.smem_budget // 1024} KiB"
    )
    print(f"configurations requested: {report.total}")
    print(f"configurations legal:     {len(report.kept)}")
    print(f"distinct cost values:     {len({round(cost(c, problem), 9) for c in report.kept})}")
    for reason, count in sorted(report.rejected.items(), key=lambda kv: -kv[1]):
        print(f"  rejected ({reason}): {count}")
    print()

    if not report.kept:
        print("=> This space is wasted: every candidate is infeasible.")
        return

    ranked = sorted(report.kept, key=lambda c: cost(c, problem))
    print(f"top {topk} by the analytic cost model (lower is better):")
    header = f"{'rank':>4}  {'block_M':>7} {'block_N':>7} {'block_K':>7} {'stages':>6} {'threads':>7} {'smem KiB':>9} {'cost':>9}"
    print(header)
    print("-" * len(header))
    for rank, cfg in enumerate(ranked[:topk], start=1):
        print(
            f"{rank:>4}  {cfg.block_m:>7} {cfg.block_n:>7} {cfg.block_k:>7} "
            f"{cfg.num_stages:>6} {cfg.threads:>7} "
            f"{smem_bytes(cfg, problem.dtype) / 1024:>9.1f} "
            f"{cost(cfg, problem):>9.4f}"
        )

    if len(report.kept) == 1:
        print()
        print("=> Only one legal configuration: tuning cannot rank anything.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["well-formed", "all-infeasible", "degenerate", "all"],
        default="well-formed",
    )
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()

    modes = (
        ["well-formed", "all-infeasible", "degenerate"]
        if args.mode == "all"
        else [args.mode]
    )
    for mode in modes:
        run(mode, args.topk)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
