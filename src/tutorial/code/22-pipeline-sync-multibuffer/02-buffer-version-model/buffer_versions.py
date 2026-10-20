#!/usr/bin/env python3
"""Model how v0.1.15 materializes and packs multi-buffered shared memory.

Pure standard library; no TileLang, no hardware.  It replays three decisions:

1. ``PipelinePlanning`` assigns each schedulable statement a ``(stage, order)``
   pair (``src/transform/pipeline_planning.cc:1226-1300``).  For the plain GEMM
   body the result is ``stage = [0, 0, N-1]`` for ``T.Pipelined(num_stages=N)``,
   which is what the upstream unit test asserts
   (``testing/python/transform/test_tilelang_transform_pipeline_planning.py``,
   ``test_pipeline_planning_before_after_copy_gemm_num_stages_plan``).
2. ``InjectSoftwarePipeline`` turns that into a version count per buffer,
   ``num_versions = use_stage - def_stage + 1``
   (``src/transform/inject_pipeline.cc:1488-1542``) and rewrites the shape to
   ``(num_versions, ...)`` (``:1551-1560``).
3. ``MergeSharedMemoryAllocations`` places every still-live buffer var into one
   arena with a linear scan plus a best-fit free list
   (``src/transform/merge_shared_memory_allocations.cc``: ``FreeList`` and
   ``LinearScanPack``).

Usage:
    python3 buffer_versions.py                       # sweep num_stages = 1..5
    python3 buffer_versions.py --num-stages 4 --budget-kib 228
    python3 buffer_versions.py --scenario carry      # long-lived buffer blocks reuse
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


# --------------------------------------------------------------------------
# Pipeline description
# --------------------------------------------------------------------------


@dataclass
class Access:
    """One scheduled statement of the pipeline body."""

    name: str
    stage: int
    order: int
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    in_body: bool = True  # False: executed once, after the loop


@dataclass
class BufferSpec:
    """A shared-memory buffer and the size of one version, in bytes."""

    name: str
    bytes_per_version: int
    align: int = 128


@dataclass
class Pipeline:
    accesses: list[Access]
    buffers: list[BufferSpec]

    @property
    def depth(self) -> int:
        return max(a.stage for a in self.accesses if a.in_body) + 1


def gemm_pipeline(num_stages: int, block_m: int = 128, block_n: int = 128,
                  block_k: int = 64, elem_bytes: int = 2,
                  scratch_bytes: int = 0, carry: bool = False) -> Pipeline:
    """The canonical whole-pipeline GEMM body.

    * ``scratch_bytes`` adds an epilogue-only shared scratch (``C_shared``):
      written and read once, after the loop, so the arena may reuse tile space.
    * ``carry`` adds a shared buffer written in stage 0 and read in the last
      stage of *every* iteration (a value carried through shared memory), which
      stays live across the whole loop and therefore blocks reuse.
    """

    accesses = [
        Access("copy_a", stage=0, order=0, reads=("A",), writes=("A_shared",)),
        Access("copy_b", stage=0, order=1, reads=("B",), writes=("B_shared",)),
    ]
    buffers = [
        BufferSpec("A_shared", block_m * block_k * elem_bytes),
        BufferSpec("B_shared", block_n * block_k * elem_bytes),
    ]
    if carry:
        accesses.append(Access("produce_carry", stage=0, order=2,
                               reads=("A_shared",), writes=("S_shared",)))
        buffers.append(BufferSpec("S_shared", block_m * block_k * elem_bytes))
    accesses.append(Access("gemm", stage=num_stages - 1, order=3,
                           reads=("A_shared", "B_shared")
                           + (("S_shared",) if carry else ()),
                           writes=("C_local",)))
    if scratch_bytes:
        accesses.append(Access("epilogue_c", stage=0, order=4, in_body=False,
                               reads=("C_local",), writes=("C_shared",)))
        accesses.append(Access("store_c", stage=0, order=5, in_body=False,
                               reads=("C_shared",), writes=("C",)))
        buffers.append(BufferSpec("C_shared", scratch_bytes))
    return Pipeline(accesses, buffers)


# --------------------------------------------------------------------------
# Step 2: version count = use - def + 1
# --------------------------------------------------------------------------


@dataclass
class AccessInfo:
    def_stage: int
    use_stage: int


def access_info(pipeline: Pipeline) -> dict[str, AccessInfo]:
    """Mirror of ``PipelineRewriter::GetBufferAccessInfo`` (inject_pipeline.cc:1421)."""

    infos: dict[str, AccessInfo] = {}
    for access in pipeline.accesses:
        if not access.in_body:
            continue
        for name in access.writes:
            info = infos.setdefault(name, AccessInfo(access.stage, 0))
            info.def_stage = min(info.def_stage, access.stage)
        for name in access.reads:
            info = infos.setdefault(name, AccessInfo(0, access.stage))
            info.use_stage = max(info.use_stage, access.stage)
    return infos


def version_counts(pipeline: Pipeline) -> dict[str, int]:
    """``use - def + 1`` per in-body shared buffer; 1 for outer/epilogue buffers.

    The real pass lowers 2 -> 1 in one special case: when no writer with a
    larger stage overlaps a reader in a smaller order
    (``inject_pipeline.cc:1500-1540``).  That case does not fire for the GEMM
    body modelled here, so this keeps the plain upper bound, which is exactly
    the number the pass computes for it.
    """

    info = access_info(pipeline)
    result: dict[str, int] = {}
    for buf in pipeline.buffers:
        if buf.name in info:
            acc = info[buf.name]
            result[buf.name] = max(1, acc.use_stage - acc.def_stage + 1)
        else:
            result[buf.name] = 1
    return result


# --------------------------------------------------------------------------
# Step 3: liveness + arena packing
# --------------------------------------------------------------------------


@dataclass
class Interval:
    name: str
    start: int
    end: int  # one past the last touch
    size: int
    align: int


def live_intervals(pipeline: Pipeline, iterations: int) -> list[Interval]:
    """First/last touch of every shared buffer var along the linearised trace.

    The real pass gathers the same information with its linear access-pattern
    finder; only the bookkeeping differs.
    """

    known = {buf.name for buf in pipeline.buffers}
    nv = version_counts(pipeline)
    max_stage = pipeline.depth - 1
    first: dict[str, int] = {}
    last: dict[str, int] = {}

    def touch(name: str, pos: int) -> None:
        first.setdefault(name, pos)
        last[name] = pos

    pos = 0
    # A statement in stage s handles logical iteration (physical - s), so the
    # three windows below mirror `BuildPipeline`'s prologue/body/epilogue split
    # (inject_pipeline.cc:1208-1264).
    body = sorted((a for a in pipeline.accesses if a.in_body),
                  key=lambda a: a.order)
    for start, stop in ((-max_stage, 0), (0, iterations - max_stage),
                        (iterations - max_stage, iterations)):
        for logical in range(start, stop):
            for access in body:
                if 0 <= logical < iterations:
                    for name in dict.fromkeys(access.reads + access.writes):
                        if name in known:
                            touch(name, pos)
                pos += 1
    # Statements that run once, after the loop.
    for access in sorted((a for a in pipeline.accesses if not a.in_body),
                         key=lambda a: a.order):
        for name in dict.fromkeys(access.reads + access.writes):
            if name in known:
                touch(name, pos)
        pos += 1

    result: list[Interval] = []
    for buf in pipeline.buffers:
        if buf.name not in first:
            continue
        size = buf.bytes_per_version * nv[buf.name]
        result.append(Interval(buf.name, first[buf.name], last[buf.name] + 1,
                               size, buf.align))
    return result


def align_up(value: int, alignment: int) -> int:
    return value if alignment <= 1 else value + (-value) % alignment


def pack(intervals: list[Interval]) -> tuple[int, dict[str, int]]:
    """Linear-scan arena packing with a coalescing best-fit free list."""

    ordered = sorted(intervals, key=lambda iv: (iv.start, -iv.size, iv.name))
    offsets: dict[str, int] = {}
    free: list[list[int]] = []
    active: list[Interval] = []
    arena_top = 0

    def release(offset: int, size: int) -> None:
        if size <= 0:
            return
        free.append([offset, size])
        free.sort()
        merged: list[list[int]] = []
        for block in free:
            if merged and merged[-1][0] + merged[-1][1] >= block[0]:
                prev = merged[-1]
                prev[1] = max(prev[0] + prev[1], block[0] + block[1]) - prev[0]
            else:
                merged.append(block)
        free[:] = merged

    def retire(pc: int) -> None:
        for iv in list(active):
            if iv.end <= pc:
                release(offsets[iv.name], iv.size)
                active.remove(iv)

    def allocate(need: int, alignment: int) -> int:
        nonlocal arena_top
        best = -1
        best_waste: int | None = None
        for index, (offset, size) in enumerate(free):
            head = align_up(offset, alignment) - offset
            if head > size or size - head < need:
                continue
            waste = size - need
            if best_waste is None or waste < best_waste:
                best, best_waste = index, waste
        if best >= 0:
            offset, size = free.pop(best)
            start = align_up(offset, alignment)
            if start > offset:
                release(offset, start - offset)
            tail = size - (start - offset) - need
            if tail:
                release(start + need, tail)
            return start
        start = align_up(arena_top, alignment)
        arena_top = start + need
        return start

    for iv in ordered:
        retire(iv.start)
        offsets[iv.name] = allocate(iv.size, iv.align)
        active.append(iv)
    return arena_top, offsets


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def kib(value: int) -> str:
    return f"{value / 1024:7.1f}"


def report(num_stages: int, budget_kib: int, iterations: int, scenario: str) -> None:
    pipeline = gemm_pipeline(
        num_stages,
        scratch_bytes=128 * 128 * 2 if scenario == "epilogue" else 0,
        carry=scenario == "carry",
    )
    nv = version_counts(pipeline)
    naive = sum(buf.bytes_per_version * nv[buf.name] for buf in pipeline.buffers)
    intervals = live_intervals(pipeline, iterations)
    arena, offsets = pack(intervals)
    budget = budget_kib * 1024
    verdict = "OK  " if arena <= budget else "OVER"
    stages = [a.stage for a in sorted((a for a in pipeline.accesses if a.in_body),
                                      key=lambda a: a.order)]
    versions = ", ".join(f"{k}={v}" for k, v in nv.items())
    print(f"num_stages={num_stages}  stage={stages}  versions: {versions}")
    print(f"    no-reuse sum {kib(naive)} KiB | arena {kib(arena)} KiB | "
          f"budget {kib(budget)} KiB  [{verdict}]")
    for name, offset in sorted(offsets.items(), key=lambda kv: kv[1]):
        iv = next(i for i in intervals if i.name == name)
        print(f"      {name:10s} offset {kib(offset)} KiB  size {kib(iv.size)} KiB"
              f"  live [{iv.start:6d},{iv.end:6d})")


def main() -> None:
    parser = argparse.ArgumentParser(description="multi-buffer version/packing model")
    parser.add_argument("--num-stages", type=int, default=None)
    parser.add_argument("--budget-kib", type=int, default=228)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--scenario", choices=("plain", "epilogue", "carry"),
                        default="epilogue")
    args = parser.parse_args()

    print(f"scenario: {args.scenario};  128x128x64 fp16 tiles "
          f"({128 * 64 * 2 / 1024:.0f} KiB per A/B version)")
    print("versions from `use - def + 1`; arena from linear scan + best fit")
    print()
    values = range(1, 6) if args.num_stages is None else [args.num_stages]
    for num_stages in values:
        report(num_stages, args.budget_kib, args.iterations, args.scenario)
    print()
    print("The arena is usually smaller than the no-reuse sum: that is")
    print("MergeSharedMemoryAllocations doing its job.  Use --scenario carry to")
    print("see a long-lived buffer destroy the reuse and push the arena back")
    print("towards the no-reuse sum.")


if __name__ == "__main__":
    main()
