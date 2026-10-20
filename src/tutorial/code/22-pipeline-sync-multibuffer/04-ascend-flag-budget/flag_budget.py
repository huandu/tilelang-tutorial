#!/usr/bin/env python3
"""Model the Ascend flag namespace that a pipeline's sync points must fit in.

Pure standard library.  It replays two arithmetic facts of the v0.1.15
Ascend backend:

* every ``set_flag`` / ``wait_flag`` pair is identified by a *hard event*
  (producer PIPE -> consumer PIPE) and owns only **8 event-id slots**
  (``kIntraCoreFlagLimit = 8``, ``src/ascend/transform/insert_sync.cc:2135``);
  cross-core flags get 16 (``kCrossCoreFlagLimit``, ``:2136``).
* the event id of one sync point is ``iteration % num_versions + base_id``
  (``MakeFlagEventId``, ``insert_sync.cc:2299``), so a sync point with
  ``nv`` versions needs ``nv`` *contiguous* slots and cannot be split across
  the 8-slot boundary at run time.

``FlagAllocator::Plan`` first sums ``num_versions`` per hard event
(``insert_sync.cc:1856-1868``).  When that exceeds the capacity,
``RewriteFlagToBuf`` keeps the subset of blocks that fills the capacity best
(a 0/1 knapsack of capacity 8, ``rewrite_flag_to_buf.cc:201-336``) and spills
the rest to the shared ``asc_lock`` / ``asc_unlock`` mutex pool.

Usage:
    python3 flag_budget.py                       # one block per depth
    python3 flag_budget.py --sharing             # some blocks can share a base
    python3 flag_budget.py --cross-core          # 16-slot namespace
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class SyncBlock:
    """One sync point: a logical dependency with ``nv`` physical versions."""

    name: str
    nv: int
    # Two blocks at different physical times cannot be in flight together, so
    # the allocator may give them the same base id (see `reusable_pairs_`).
    shareable: bool = False


def required_ids(blocks: list[SyncBlock]) -> int:
    """What ``FlagAllocator::Plan`` adds up before reuse is considered."""
    return sum(b.nv for b in blocks)


def after_reuse(blocks: list[SyncBlock]) -> int:
    """Upper bound on the ids actually needed once shareable blocks are merged.

    Merging is only sound when the blocks can never be outstanding at the same
    time; the allocator validates that with its reuse analysis.  Here the
    shareable blocks are collapsed into one group whose cost is the widest
    member.
    """

    shared = [b for b in blocks if b.shareable]
    alone = [b for b in blocks if not b.shareable]
    total = sum(b.nv for b in alone)
    if shared:
        total += max(b.nv for b in shared)
    return total


def knapsack_keep(blocks: list[SyncBlock], capacity: int) -> tuple[list[str], list[str]]:
    """0/1 knapsack: maximise the slots used without exceeding ``capacity``.

    Mirrors ``KnapsackKeep`` in ``rewrite_flag_to_buf.cc``: the kept blocks are
    renumbered into the flag namespace, the dropped ones spill to the mutex
    pool.  Returns ``(kept, spilled)`` names.
    """

    n = len(blocks)
    # dp[(i, w)] = best total size using the first i blocks within capacity w
    table = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        size = blocks[i - 1].nv
        for w in range(capacity + 1):
            best = table[i - 1][w]
            if size <= w:
                best = max(best, table[i - 1][w - size] + size)
            table[i][w] = best

    kept: list[str] = []
    w = capacity
    for i in range(n, 0, -1):
        if table[i][w] != table[i - 1][w]:
            kept.append(blocks[i - 1].name)
            w -= blocks[i - 1].nv
    kept.reverse()
    spilled = [b.name for b in blocks if b.name not in kept]
    return kept, spilled


def event_id_formula(block: SyncBlock, base: int) -> str:
    if block.nv == 1:
        return f"id = {base} (constant)"
    return f"id = i % {block.nv} + {base}"


def assign_bases(blocks: list[SyncBlock], sharing: bool) -> dict[str, int]:
    """Hand out base ids the way ``FlagAllocator`` walks the sync points.

    Shareable blocks collapse into one physical channel, so all of them report
    the same base; the channel is as wide as its widest member.
    """

    bases: dict[str, int] = {}
    cursor = 0
    shared_base: int | None = None
    for block in blocks:
        if sharing and block.shareable:
            if shared_base is None:
                shared_base = cursor
                cursor += block.nv
            bases[block.name] = shared_base
            continue
        bases[block.name] = cursor
        cursor += block.nv
    return bases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sharing", action="store_true",
                        help="mark the release sync points as shareable")
    parser.add_argument("--cross-core", action="store_true",
                        help="use the 16-slot cross-core namespace")
    parser.add_argument("--depth", type=int, default=None,
                        help="single pipeline depth instead of a sweep")
    args = parser.parse_args()

    capacity = 16 if args.cross_core else 8
    namespace = "cross-core (16 slots)" if args.cross_core else "intra-core (8 slots)"

    def blocks_for(depth: int) -> list[SyncBlock]:
        # A load/VF/store pipeline on one core: the release edge (Vector ->
        # MTE) is repeated once per buffer version; the acquire edge (MTE ->
        # Vector) likewise.  The release edges of consecutive slots cannot be
        # outstanding together, so they may share a base id.
        return [
            SyncBlock("acquire_A", depth),
            SyncBlock("acquire_B", depth),
            SyncBlock("release_A", depth, shareable=args.sharing),
            SyncBlock("release_B", depth, shareable=args.sharing),
        ]

    print(f"hard-event flag namespace: {namespace}, "
          f"event id = iteration % num_versions + base")
    print()
    for depth in ([args.depth] if args.depth else [1, 2, 3, 4]):
        blocks = blocks_for(depth)
        raw = required_ids(blocks)
        reused = after_reuse(blocks)
        need = reused if args.sharing else raw
        print(f"depth={depth}  num_versions per sync point = {depth}")
        print(f"  blocks: " + ", ".join(f"{b.name}={b.nv}" for b in blocks))
        print(f"  allocator sum = {raw} ids"
              + (f", after reuse = {reused} ids" if args.sharing else ""))
        if need <= capacity:
            bases = assign_bases(blocks, args.sharing)
            print(f"  -> fits in {capacity} slots; no spill"
                  f"  ({capacity - need} slots spare)")
            for block in blocks:
                base = bases[block.name]
                print(f"     {block.name:9s} base {base:2d}  "
                      f"{event_id_formula(block, base)}")
        else:
            kept, spilled = knapsack_keep(blocks, capacity)
            kept_slots = sum(b.nv for b in blocks if b.name in kept)
            print(f"  -> exceeds {capacity} slots: 0/1 knapsack keeps "
                  f"{kept_slots}/{capacity} slots as set/wait flags")
            print(f"     kept    : {', '.join(kept) or '(none)'}")
            print(f"     spilled : {', '.join(spilled)}  -> asc_lock/asc_unlock"
                  f" mutex pool (blocking)")
        print()
    print("The lesson for pipelines: deepening the ring multiplies every")
    print("sync point's version count, so the *flag namespace*, not the")
    print("shared memory, can become the binding constraint.  Spilled blocks")
    print("do not disappear; they turn into blocking lock/unlock pairs.")


if __name__ == "__main__":
    main()
