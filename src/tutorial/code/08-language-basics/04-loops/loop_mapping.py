#!/usr/bin/env python3
"""Chapter 08 / §6: how a nested ``T.Parallel(e0, e1, ...)`` lands on threads.

This is a *pure standard library* re-implementation of the loop partition plan
in ``_vendor/tilelang-v0.1.15/src/transform/loop_partition.cc`` (class
``LoopPartitioner``, ``Partition`` at lines 248-271). For a k-dimensional
parallel nest with extents ``e0..e_{k-1}`` the plan is:

    flattened  = e0*(e1*(...*i0 + i1)...) + i_{k-1}      # row-major flatten
    access_idx = flattened // vector_size                 # group into lanes
    thread     = access_idx % num_thread                  # rotate across threads
    local_idx  = (access_idx // num_thread) * vector_size
                 + (flattened % vector_size)              # per-thread slot

Read it as three nested levels, fastest first: **lane -> thread -> serial
replication**. Consecutive iterations share a lane (so they vectorize), then
spread across threads (so the access coalesces), and only then grow a per-thread
serial loop.

This is the *default* plan; a ``loop_layout=`` annotation on ``T.Parallel``
overrides it. Nothing here is measured on hardware.

Run:
    python3 loop_mapping.py                        # a 2-D nest, 128 threads, lane 8
    python3 loop_mapping.py --extents 128 32 --threads 128 --vector 4
    python3 loop_mapping.py --extents 64 --threads 128 --vector 2
"""

from __future__ import annotations

import argparse
from itertools import product


def flatten(index: tuple[int, ...], extents: tuple[int, ...]) -> int:
    flat = 0
    for i, e in zip(index, extents):
        flat = flat * e + i
    return flat


def unflatten(flat: int, extents: tuple[int, ...]) -> tuple[int, ...]:
    out = [0] * len(extents)
    for k in range(len(extents) - 1, -1, -1):
        out[k] = flat % extents[k]
        flat //= extents[k]
    return tuple(out)


def plan(extents: tuple[int, ...], num_thread: int, vector_size: int) -> dict[tuple[int, ...], tuple[int, int]]:
    """Return {iteration-point: (thread, local_idx)} using the vendor formula."""
    result: dict[tuple[int, ...], tuple[int, int]] = {}
    for index in product(*(range(e) for e in extents)):
        flat = flatten(index, extents)
        access_idx = flat // vector_size
        thread = access_idx % num_thread
        local_idx = (access_idx // num_thread) * vector_size + (flat % vector_size)
        result[index] = (thread, local_idx)
    return result


def show(extents: tuple[int, ...], num_thread: int, vector_size: int) -> None:
    total = 1
    for e in extents:
        total *= e
    print(f"T.Parallel{extents}  threads={num_thread}  vector_size={vector_size}")
    print(f"iteration points = {total};  points per thread = {total / num_thread:g}")
    plan_map = plan(extents, num_thread, vector_size)

    print("\niteration point -> (thread, per-thread slot)")
    print(f"  {'index':<14} {'flat':>4}  {'thread':>6}  {'slot':>4}")
    for index in product(*(range(e) for e in extents)):
        thread, slot = plan_map[index]
        print(f"  {str(index):<14} {flatten(index, extents):>4}  {thread:>6}  {slot:>4}")

    print("\nby thread (which points each thread owns)")
    owned: dict[int, list[tuple[tuple[int, ...], int]]] = {}
    for index, (thread, slot) in plan_map.items():
        owned.setdefault(thread, []).append((index, slot))
    first = min(owned)
    last = max(owned)
    for thread in (first, first + 1, last):
        pts = sorted(owned.get(thread, []), key=lambda p: p[1])
        shown = ", ".join(f"{idx}@{slot}" for idx, slot in pts[:8])
        more = "" if len(pts) <= 8 else f" ... (+{len(pts) - 8})"
        print(f"  thread {thread:>3}: {shown}{more}")

    print("\ninterpretation")
    print("  * the innermost `vector_size` iterations share a lane-set  -> vectorize")
    print("  * the next axis is the thread index                        -> coalesce")
    print("  * whatever is left becomes a per-thread serial loop        -> replicate")


def compare_shapes() -> None:
    print("Same total work, different nest shapes (threads=8, vector=2):\n")
    print("The *count* per thread is identical; what changes is which index")
    print("lands on which thread, because the nest is flattened row-major.\n")
    for extents in [(64,), (32, 2), (8, 8), (2, 32), (4, 4, 4)]:
        total = 1
        for e in extents:
            total *= e
        plan_map = plan(extents, 8, 2)
        threads_used = len({t for t, _ in plan_map.values()})
        owned: dict[int, list[tuple[int, ...]]] = {}
        for index, (thread, _slot) in sorted(plan_map.items(), key=lambda kv: flatten(kv[0], extents)):
            owned.setdefault(thread, []).append(index)
        t0 = ", ".join(str(i) for i in owned[0][:4])
        t1 = ", ".join(str(i) for i in owned[1][:4])
        print(f"  T.Parallel{str(extents):<14} total={total:<4} threads={threads_used:<3} "
              f"per-thread={total // 8}")
        print(f"      thread 0 owns {t0} ...")
        print(f"      thread 1 owns {t1} ...")
    print(
        "\nSo the *shape of the nest* is not cosmetic: it decides which logical\n"
        "index each thread reads, i.e. whether the load coalesces.\n"
    )
    print(
        "When the total iteration count is below `threads`, some threads get no\n"
        "work at all: the plan replicates and the codegen adds a guard.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extents", nargs="+", type=int, default=[128, 32])
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--vector", type=int, default=8)
    parser.add_argument("--compare", action="store_true", help="compare nest shapes")
    args = parser.parse_args()
    if args.compare:
        compare_shapes()
        return 0
    show(tuple(args.extents), args.threads, args.vector)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
