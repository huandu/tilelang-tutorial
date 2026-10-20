#!/usr/bin/env python3
"""A tiny discrete-event model of a copy/compute pipeline (Chapter 06).

The chapter's section 1.4 claims that overlapping copy-in, compute and copy-out
raises the utilization of the compute unit. This script makes that claim
checkable on a CPU by *simulating* a timeline, not by measuring a device.

We model a per-core loop over ``iters`` tiles with three stages, each with an
integer cost in "cycles":

    MTE2 (copy in)  -> CUBE/VEC (compute) -> MTE3 (copy out)

Two schedules are compared:

    serial            stage n+1 starts only after stage n finished its own
                      dependency; no overlap across tiles.
    multi_buffer(k)   each buffer has k slots, so copy-in for tile i+k-1 can
                      proceed while compute for tile i is running.

This is a deliberately simple model. It assumes:
  * the three stages can run concurrently on independent units,
  * a stage never blocks for a reason other than its input dependency,
  * stage costs are constants supplied on the command line.

It ignores realistic effects (variable latencies, queue depth, DMA bandwidth,
Set/Wait cost, tail effects). It is a *teaching model*, not a predictor.

Run:
    python3 pipeline_timeline.py
    python3 pipeline_timeline.py --iters 8 --copy-in 10 --compute 4 --copy-out 10 --versions 2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class StageWindow:
    """[start, end) cycle window plus which tile it belongs to."""

    start: int
    end: int

    @property
    def cost(self) -> int:
        return self.end - self.start


def schedule(
    iters: int,
    copy_in: int,
    compute: int,
    copy_out: int,
    versions: int,
) -> tuple[dict[str, list[StageWindow]], int]:
    """Return per-stage windows and the makespan.

    ``versions`` is the number of buffer slots. ``versions=1`` is the serial
    baseline. For a correct multi-buffer schedule, copy-in for tile i may start
    once (a) the previous copy-in has finished and (b) the buffer slot it will
    reuse has been freed by the copy-out of tile i - versions.
    """
    if versions < 1:
        raise ValueError("versions must be >= 1")

    starts: dict[str, list[int]] = {"copy_in": [], "compute": [], "compute_end": [], "copy_out": []}

    for i in range(iters):
        # copy-in: the MTE unit is serial, so it starts after the previous one.
        prev_in_end = starts["copy_in"][-1] + copy_in if starts["copy_in"] else 0
        # buffer reuse: slot i % versions was last used by tile i - versions,
        # and is freed when that tile's copy-out finished.
        reuse_ready = 0
        if i >= versions:
            reuse_ready = starts["copy_out"][i - versions] + copy_out
        cs = max(prev_in_end, reuse_ready)
        starts["copy_in"].append(cs)

        # compute: serial compute unit, waits for this tile's copy-in.
        prev_compute_end = (
            starts["compute"][-1] + compute if starts["compute"] else 0
        )
        ms = max(prev_compute_end, starts["copy_in"][i] + copy_in)
        starts["compute"].append(ms)
        starts["compute_end"].append(ms + compute)

        # copy-out: serial MTE3 unit, waits for this tile's compute.
        prev_out_end = (
            starts["copy_out"][-1] + copy_out if starts["copy_out"] else 0
        )
        os_ = max(prev_out_end, starts["compute_end"][i])
        starts["copy_out"].append(os_)

    windows = {
        "copy_in": [StageWindow(s, s + copy_in) for s in starts["copy_in"]],
        "compute": [StageWindow(s, s + compute) for s in starts["compute"]],
        "copy_out": [StageWindow(s, s + copy_out) for s in starts["copy_out"]],
    }
    makespan = max(w.end for ws in windows.values() for w in ws)
    return windows, makespan


def sum_costs(windows: list[StageWindow]) -> int:
    return sum(w.cost for w in windows)


def busy_cycles(windows: list[StageWindow]) -> int:
    """Union length of the windows (they are disjoint here by construction)."""
    intervals = sorted((w.start, w.end) for w in windows)
    total = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    return total + (cur_end - cur_start)


def render(windows: dict[str, list[StageWindow]], makespan: int, cell: int = 1) -> str:
    """ASCII timeline. One character per ``cell`` cycles."""
    width = max(20, (makespan + cell - 1) // cell)
    rows = []
    order = [("copy_in", "MTE2"), ("compute", "CUBE/VEC"), ("copy_out", "MTE3")]
    for key, label in order:
        row = ["."] * width
        for w in windows[key]:
            for c in range(w.start // cell, (w.end + cell - 1) // cell):
                if 0 <= c < width:
                    row[c] = "#"
        rows.append(f"{label:>9} |{''.join(row)}|")
    rows.append(f"{'cycles':>9} |{'-' * width}|")
    return "\n".join(rows)


def run_case(iters: int, copy_in: int, compute: int, copy_out: int, versions: int) -> None:
    windows, makespan = schedule(iters, copy_in, compute, copy_out, versions)
    compute_busy = busy_cycles(windows["compute"])
    util = 100.0 * compute_busy / makespan
    print(
        f"versions={versions}: makespan={makespan} cycles, "
        f"compute busy={compute_busy}, compute utilization={util:5.1f}%"
    )
    print(render(windows, makespan))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--copy-in", type=int, default=10, dest="copy_in")
    parser.add_argument("--compute", type=int, default=4)
    parser.add_argument("--copy-out", type=int, default=10, dest="copy_out")
    parser.add_argument(
        "--versions",
        type=int,
        default=2,
        help="buffer slots to compare; 1 is the serial baseline",
    )
    args = parser.parse_args()

    print(__doc__.strip().splitlines()[0])
    print("=" * 60)
    print(
        f"iters={args.iters}, copy_in={args.copy_in}, compute={args.compute}, "
        f"copy_out={args.copy_out}\n"
    )

    run_case(args.iters, args.copy_in, args.compute, args.copy_out, 1)
    run_case(args.iters, args.copy_in, args.compute, args.copy_out, args.versions)
    if args.versions != 3:
        run_case(args.iters, args.copy_in, args.compute, args.copy_out, 3)

    print("=" * 60)
    print(
        "Read the timeline: without overlap the compute row has gaps (its\n"
        "utilization is low). With a second buffer slot, copy-in of the next\n"
        "tile runs while compute works on the current one, so the compute row\n"
        "fills in. A third slot can absorb more jitter but costs more on-chip\n"
        "bytes -- see 01-onchip-footprint/. Nothing here is a hardware\n"
        "measurement; it is the chapter's model made executable.\n"
    )
    print("If copy time dominates compute time, even a perfect overlap leaves the")
    print("machine memory-bound: utilization is capped near compute/(copy_in+copy_out).")
    print("Contrast with a compute-bound shape (copy_in=4, compute=20, copy_out=4):\n")
    run_case(args.iters, 4, 20, 4, 1)
    run_case(args.iters, 4, 20, 4, 2)


if __name__ == "__main__":
    main()
