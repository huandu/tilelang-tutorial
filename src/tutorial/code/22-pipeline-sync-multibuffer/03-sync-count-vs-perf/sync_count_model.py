#!/usr/bin/env python3
"""Qualitative model: how many barriers per iteration can a pipeline hide?

Pure standard library.  The model is a `copy -> compute` ring with `depth`
slots, the same shape as chapter 11's timeline model, extended with the one
thing this chapter is about: **the release dependency that makes a slot safe
to rewrite**.

Per iteration k the pipeline has to serialize three things:

* the copy engine issues the copy for iteration k (cost ``issue``);
* the data of iteration k becomes readable ``latency`` cycles after issue;
* the compute of iteration k occupies the compute unit for ``compute`` cycles.

Then the synchronization variants add:

``none``    no barrier at all.  This is *not* a correct kernel: nothing stops
            the copy of iteration ``k + depth`` from overwriting the slot that
            the compute of iteration ``k`` is still reading.
``war``     one release barrier (cost ``barrier``) between the end of
            ``compute[k]`` and the start of ``copy[k + depth]``.  This is the
            minimum the compiler must insert for a ring of ``depth`` slots.
``war+raw`` additionally one acquire barrier between "data of k is ready" and
            ``compute[k]``.  Cheap when the producer is asynchronous, needed
            when visibility itself is in question.
``+cb``     a redundant barrier inserted *inside* the consumer chain, i.e.
            between ``compute[k - 1]`` and ``compute[k]``.  It costs one
            ``barrier`` of steady-state interval per iteration.
``sync-copy`` a pessimistic schedule in which every copy waits for the previous
            copy to land plus a fence, which is what "no asynchronous copy,
            barrier after each statement" degrades into.

Every number printed here is an abstract cycle count from this model.  It is
not a measurement of anything.

Usage:
    python3 sync_count_model.py
    python3 sync_count_model.py --latency 120 --compute 20 --barrier 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class Params:
    iterations: int = 16
    issue: int = 1
    latency: int = 30
    compute: int = 20
    barrier: int = 8
    depth: int = 3


def simulate(p: Params, variant: str) -> dict[str, float]:
    """Fixed-point list schedule over the ring.

    All the constraints are monotone in the start times, so relaxing them to a
    fixed point gives the earliest legal schedule.
    """

    n = p.iterations
    cstart = [0] * n          # copy issue start
    gstart = [0] * n          # compute start
    cend = [p.issue * k for k in range(n)]  # copy engine is serial
    cstart = list(cend)                    # start == when the engine frees
    cend = [cstart[k] + p.issue for k in range(n)]
    ready = [cstart[k] + p.latency for k in range(n)]
    gstart = list(ready)
    gend = [gstart[k] + p.compute for k in range(n)]

    extra_constraints = {
        "none": (),
        "war": ("war",),
        "war+raw": ("war", "raw"),
        "+cb": ("war", "raw", "consumer-barrier"),
        "sync-copy": ("war", "raw", "sync-copy"),
    }[variant]

    for _ in range(4 * n + 16):  # relaxation passes; converges long before
        changed = False

        def raise_to(index: int, value: int, array: list[int]) -> None:
            nonlocal changed
            if array[index] < value:
                array[index] = value
                changed = True

        for k in range(n):
            # copy engine is serial, copies issue in program order
            if k > 0:
                raise_to(k, cend[k - 1], cstart)
            if "sync-copy" in extra_constraints and k > 0:
                # no async overlap: the next copy waits for the previous one to
                # land, plus a fence
                raise_to(k, ready[k - 1] + p.barrier, cstart)
            # war: rewriting the slot read by iteration k - depth must wait for
            # that reader to finish, plus the barrier itself
            if "war" in extra_constraints and k >= p.depth:
                raise_to(k, gend[k - p.depth] + p.barrier, cstart)
            cend[k] = cstart[k] + p.issue
            ready[k] = cstart[k] + p.latency

            # compute unit is serial, computations run in program order
            if k > 0:
                raise_to(k, gend[k - 1], gstart)
                if "consumer-barrier" in extra_constraints:
                    raise_to(k, gend[k - 1] + p.barrier, gstart)
            raise_to(k, ready[k], gstart)
            if "raw" in extra_constraints:
                raise_to(k, ready[k] + p.barrier, gstart)
            gend[k] = gstart[k] + p.compute
        if not changed:
            break

    makespan = gend[-1]
    tail = gstart[max(0, n // 2):]
    if len(tail) > 1:
        intervals = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
        steady = sum(intervals) / len(intervals)
    else:
        steady = float(makespan)
    return {
        "makespan": float(makespan),
        "steady": steady,
        "compute_util": p.iterations * p.compute / makespan,
        "overlap": 1.0 - (p.iterations * (p.issue + p.compute)) / makespan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--issue", type=int, default=1)
    parser.add_argument("--latency", type=int, default=30)
    parser.add_argument("--compute", type=int, default=20)
    parser.add_argument("--barrier", type=int, default=8)
    args = parser.parse_args()

    variants = ("none", "war", "war+raw", "+cb", "sync-copy")
    print(f"copy->compute ring: iterations={args.iterations} issue={args.issue} "
          f"latency={args.latency} compute={args.compute} barrier={args.barrier}")
    print("('none' is not a correct kernel; it is the speed-of-light bound.)")
    print()
    header = f"{'depth':>5} | " + " | ".join(f"{v:>16}" for v in variants)
    print(header)
    print("-" * len(header))
    for depth in (1, 2, 3, 4, 6):
        cells = []
        base = None
        for variant in variants:
            params = Params(args.iterations, args.issue, args.latency,
                            args.compute, args.barrier, depth)
            result = simulate(params, variant)
            if base is None:
                base = result["makespan"]
            cells.append(f"{result['makespan']:5.0f}cyc "
                         f"u={result['compute_util'] * 100:3.0f}%")
        print(f"{depth:5d} | " + " | ".join(f"{c:>16}" for c in cells))
    print()
    print("Steady-state interval per iteration (last half of the loop):")
    for depth in (1, 2, 3, 4, 6):
        row = []
        for variant in variants:
            params = Params(args.iterations, args.issue, args.latency,
                            args.compute, args.barrier, depth)
            row.append(f"{simulate(params, variant)['steady']:6.1f}")
        print(f"  depth={depth}: " + "  ".join(f"{v}={r}" for v, r in zip(variants, row)))
    print()
    print("Read it as: in steady state the release barrier costs nothing while")
    print("    depth >= 1 + (latency + barrier) / compute")
    print("because it hides behind the copy already in flight.  A redundant "
          "barrier that lands")
    print("on the critical path (`+cb`, `sync-copy`) adds its full cost to the "
          "steady interval:")
    print("that is `多一个同步会掉性能`.")


if __name__ == "__main__":
    main()
