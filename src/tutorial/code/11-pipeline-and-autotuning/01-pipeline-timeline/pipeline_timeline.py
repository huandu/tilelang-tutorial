#!/usr/bin/env python3
"""A tiny event-driven model of a software-pipelined copy -> compute loop.

This module contains no TileLang and no hardware calls: it is a discrete-event
model used in 第 11 章 to make 「不做流水会怎样」 measurable instead of
rhetorical. Every number it prints is a property of the model below, never a
measurement on real hardware.

The model
---------
A loop runs ``n_iters`` iterations over a reduction dimension (think of the
``ko`` loop in a tiled GEMM). Each iteration has exactly two operations:

  * a *copy* (global -> shared, or any producer) that occupies the copy engine
    for ``issue`` cycles and whose data becomes usable ``latency`` cycles after
    the copy is issued;
  * a *compute* (the GEMM / tile op) that occupies the compute unit for
    ``compute`` cycles and may only start once its input data has arrived.

The copy engine and the compute unit are independent resources, so the whole
question of pipelining is: how much overlap can the schedule expose?

``stages`` is the number of buffer slots (``num_stages`` in TileLang). One slot
is the minimum: the consumer must finish reading before the producer may
overwrite it, so nothing overlaps. Two or more slots let the producer write
slot ``i % stages`` while the consumer is still reading an earlier slot.

The scheduler is a deterministic greedy list scheduler: issue the copy as soon
as both the copy engine and the destination slot are free, then start the
compute as soon as both the data has arrived and the compute unit is free.

Usage
-----
    python3 pipeline_timeline.py                 # the default table
    python3 pipeline_timeline.py --sweep         # stage-depth sweep
    python3 pipeline_timeline.py --stages 1 2 3 4 6
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoopModel:
    """Parameters of one copy -> compute reduction loop.

    All times are abstract cycles. They are model inputs, not device specs.
    """

    n_iters: int
    issue: int = 1
    """Cycles the copy engine is occupied per iteration (issue cost)."""
    latency: int = 30
    """Cycles between issuing a copy and being able to read its data."""
    compute: int = 20
    """Cycles the compute unit is occupied per iteration."""


@dataclass
class Timeline:
    """Result of simulating one stage count."""

    stages: int
    model: LoopModel
    copy_issue: list[int] = field(default_factory=list)
    copy_ready: list[int] = field(default_factory=list)
    compute_start: list[int] = field(default_factory=list)
    compute_end: list[int] = field(default_factory=list)

    @property
    def makespan(self) -> int:
        """Wall-clock cycles from the first issue to the last compute end."""
        return self.compute_end[-1] if self.compute_end else 0

    @property
    def throughput(self) -> float:
        """Iterations completed per cycle in steady state (makespan-based)."""
        if self.makespan == 0:
            return 0.0
        return self.model.n_iters / self.makespan

    @property
    def issue_interval(self) -> float:
        """Average cycles between consecutive copy issues (after the first)."""
        if len(self.copy_issue) < 2:
            return 0.0
        return (self.copy_issue[-1] - self.copy_issue[0]) / (len(self.copy_issue) - 1)

    @property
    def compute_interval(self) -> float:
        """Average cycles between consecutive compute starts (after the first)."""
        if len(self.compute_start) < 2:
            return 0.0
        return (self.compute_start[-1] - self.compute_start[0]) / (
            len(self.compute_start) - 1
        )

    @property
    def compute_utilization(self) -> float:
        """Fraction of the accumulated span in which the compute unit is busy.

        Measured from the first compute start to the last compute end, so it
        isolates the steady state from the one-off pipeline fill.
        """
        span = self.compute_end[-1] - self.compute_start[0]
        if span <= 0:
            return 0.0
        return self.model.n_iters * self.model.compute / span

    @property
    def copy_utilization(self) -> float:
        """Fraction of the accumulated span in which the copy engine is busy."""
        span = self.compute_end[-1] - self.compute_start[0]
        if span <= 0:
            return 0.0
        return self.model.n_iters * self.model.issue / span


def simulate(model: LoopModel, stages: int) -> Timeline:
    """Greedy list schedule of the loop for a given number of buffer slots."""
    if stages < 1:
        raise ValueError("stages must be >= 1")

    timeline = Timeline(stages=stages, model=model)
    copy_engine_free = 0
    compute_unit_free = 0
    slot_free = [0] * stages  # when the consumer has finished reading the slot

    for i in range(model.n_iters):
        slot = i % stages
        issue = max(copy_engine_free, slot_free[slot])
        ready = issue + model.latency
        start = max(compute_unit_free, ready)
        end = start + model.compute

        timeline.copy_issue.append(issue)
        timeline.copy_ready.append(ready)
        timeline.compute_start.append(start)
        timeline.compute_end.append(end)

        copy_engine_free = issue + model.issue
        compute_unit_free = end
        slot_free[slot] = end

    return timeline


def required_depth(model: LoopModel) -> int:
    """Smallest stage count that reaches the compute-bound steady state.

    The producer must be able to run ``ceil(latency / compute)`` iterations
    ahead of the consumer to hide the transfer latency, plus one slot for the
    iteration currently being read.
    """
    import math

    return 1 + math.ceil(model.latency / max(model.compute, 1))


def _table(rows: list[tuple[str, str]], headers: tuple[str, str]) -> str:
    left = max(len(headers[0]), max((len(r[0]) for r in rows), default=0))
    right = max(len(headers[1]), max((len(r[1]) for r in rows), default=0))
    bar = f"+-{'-' * left}-+-{'-' * right}-+"
    out = [bar, f"| {headers[0]:<{left}} | {headers[1]:<{right}} |", bar]
    out += [f"| {a:<{left}} | {b:<{right}} |" for a, b in rows]
    out.append(bar)
    return "\n".join(out)


def print_timeline(timeline: Timeline) -> None:
    """Print one simulated schedule as a compact space-time trace."""
    model = timeline.model
    print(f"== stages={timeline.stages} ==")
    rows: list[tuple[str, str]] = []
    for i in range(min(model.n_iters, 8)):
        issue = timeline.copy_issue[i]
        ready = timeline.copy_ready[i]
        start = timeline.compute_start[i]
        end = timeline.compute_end[i]
        rows.append(
            (
                f"iter {i:>2}",
                f"slot {i % timeline.stages}  issue {issue:>3}  ready {ready:>3}  "
                f"gemm [{start:>3},{end:>3})",
            )
        )
    print(_table(rows, ("iteration", "schedule (cycles)")))
    print(
        f"makespan={timeline.makespan}  issue_interval={timeline.issue_interval:.2f}  "
        f"compute_interval={timeline.compute_interval:.2f}  "
        f"compute_util={timeline.compute_utilization:.0%}"
    )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--issue", type=int, default=1)
    parser.add_argument("--latency", type=int, default=30)
    parser.add_argument("--compute", type=int, default=20)
    parser.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="sweep stage depth 1..8 instead of the default table",
    )
    args = parser.parse_args()

    stages = list(range(1, 9)) if args.sweep else args.stages
    model = LoopModel(
        n_iters=args.iterations,
        issue=args.issue,
        latency=args.latency,
        compute=args.compute,
    )

    print("model: copy -> compute, greedy list schedule, abstract cycles")
    print(
        f"  iterations={model.n_iters} issue={model.issue} "
        f"latency={model.latency} compute={model.compute}"
    )
    print(
        f"  latency/compute = {model.latency / model.compute:.2f} "
        f"-> smallest depth that reaches the steady state = {required_depth(model)}"
    )
    print()

    for stages_count in stages:
        timeline = simulate(model, stages_count)
        print_timeline(timeline)

    baseline = simulate(model, 1).makespan
    rows: list[tuple[str, str]] = []
    for stages_count in sorted({1, *stages}):
        timeline = simulate(model, stages_count)
        rows.append(
            (
                f"{stages_count}",
                f"{timeline.makespan:>5}  x{baseline / timeline.makespan:>4.2f}  "
                f"issue_iv={timeline.issue_interval:>5.1f}  "
                f"compute_iv={timeline.compute_interval:>5.1f}  "
                f"compute_util={timeline.compute_utilization:>5.0%}",
            )
        )
    print("== summary ==")
    print(
        _table(
            rows,
            ("stages", "makespan  speedup  issue_iv  compute_iv  compute_util"),
        )
    )
    print()
    print(
        "Reading the table: below the depth computed above the consumer still "
        "stalls on data arrival;\nabove it the makespan stops improving while "
        "every extra slot costs on-chip storage."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
