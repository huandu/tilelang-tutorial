#!/usr/bin/env python3
"""A runnable, CPU-only, pure-Python model of the DeepSelect TopK algorithm.

This is the *algorithm* from the DeepSelect report, not the CUDA/Ascend
kernel. It follows the published pseudo-code literally:

    DeepSelectTopk(x[0:N), k, B, B2):
        topk_candidate  = []
        topk_threshold  = -inf
        p = random_permutation(ceil_div(N, B))      # independent of x
        for block in p:
            for j in [block*B, min((block+1)*B, N)):
                if x[j] > topk_threshold:
                    topk_candidate.append((x[j], j))
            if len(topk_candidate) >= k + B2:
                topk_candidate = RadixSelectTopK(topk_candidate, k)
                topk_threshold = min(v for v, _ in topk_candidate)
        return RadixSelectTopK(topk_candidate, k)

The only substitution is the inner `RadixSelectTopK` compact step: the report's
real kernels use a radix select on device, while this script uses
`torch.topk` when PyTorch is importable and falls back to `heapq.nlargest`
otherwise, so the whole thing runs on any machine with a stock Python.

What it demonstrates (and what it deliberately does not):

  * it reads every element exactly once, in contiguous blocks, in random order;
  * the threshold only ever grows, so the accept rate falls as the scan goes on;
  * the candidate buffer stays bounded by k + B2 + B;
  * it counts `W`, the total number of elements handed to the compact step,
    and compares that against the report's bound (1 + k/B2)·L·H_m.

It says nothing about performance: Python is ~5 orders of magnitude slower than
a kernel. The performance numbers live in the chapter, sourced from the repo's
README and figures.

Run:
    python3 deepselect_python.py               # defaults to torch.topk compact
    python3 deepselect_python.py --no-torch    # force the heapq fallback
    python3 deepselect_python.py --n 1048576 --k 512 --b 1024 --b2 1024
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

try:  # torch is optional: everything still works without it
    import torch

    _HAVE_TORCH = True
except Exception:  # pragma: no cover - environment dependent
    _HAVE_TORCH = False


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# --------------------------------------------------------------------------- #
# The compact step: "RadixSelectTopK" in the report.
# --------------------------------------------------------------------------- #


def _compact_torch(
    values: List[float], indices: List[int], k: int
) -> Tuple[List[float], List[int]]:
    """Top-k by value, keeping the (value, index) pairs aligned."""
    if len(values) <= k:
        return list(values), list(indices)
    v = torch.tensor(values, dtype=torch.float64)
    top_v, top_pos = torch.topk(v, k, largest=True, sorted=False)
    pos = top_pos.tolist()
    return [values[p] for p in pos], [indices[p] for p in pos]


def _compact_heapq(
    values: List[float], indices: List[int], k: int
) -> Tuple[List[float], List[int]]:
    """Same contract, using only the standard library."""
    if len(values) <= k:
        return list(values), list(indices)
    # nlargest with a key over the (value, index) pairs keeps them aligned.
    picked = heapq.nlargest(k, range(len(values)), key=lambda p: values[p])
    return [values[p] for p in picked], [indices[p] for p in picked]


def make_compactor(use_torch: bool) -> Tuple[str, Callable]:
    if use_torch and _HAVE_TORCH:
        return "torch.topk", _compact_torch
    return "heapq.nlargest", _compact_heapq


# --------------------------------------------------------------------------- #
# The algorithm.
# --------------------------------------------------------------------------- #


@dataclass
class Stats:
    """Instrumentation: everything the complexity proof talks about."""

    accepted: int = 0          # A = sum_i A_i  (elements admitted to the buffer)
    compactions: int = 0       # R              (in-loop TopK calls; the final one is not counted)
    compact_work: int = 0      # W = A + k*R    (total elements fed to a compact step)
    max_buffer: int = 0        # observed max len(topk_candidate)
    blocks: int = 0
    scans: int = 0             # number of elements the scan loop looked at (always N)
    elided: int = 0            # elements skipped by the threshold test
    threshold_trace: List[float] = field(default_factory=list)
    buffer_trace: List[int] = field(default_factory=list)


def deepselect_topk(
    x: Sequence[float],
    k: int,
    b: int,
    b2: int,
    *,
    rng: random.Random | None = None,
    order: Sequence[int] | None = None,
    compact: Callable = _compact_torch,
    trace_every: int = 0,
    stats: Stats | None = None,
) -> Tuple[List[float], List[int], Stats]:
    """Return (values, indices, stats) for the top-k of `x`.

    `x` may be any sequence of comparable numbers. `order` (if given) overrides
    the random block permutation, which is what the sequential-vs-random
    experiment in `../02-block-order-experiment/` uses.
    """
    n = len(x)
    if not (1 <= k <= n):
        raise ValueError("need 1 <= k <= N")
    if b < 1 or b2 < 1:
        raise ValueError("need B >= 1 and B2 >= 1")
    if rng is None:
        rng = random.Random(0)

    n_blocks = ceil_div(n, b)
    if order is None:
        order = list(range(n_blocks))
        rng.shuffle(order)
    else:
        order = list(order)

    st = stats if stats is not None else Stats()
    st.blocks = n_blocks

    cand_v: List[float] = []
    cand_i: List[int] = []
    threshold = float("-inf")
    threshold_init = False

    for step, block in enumerate(order):
        lo = block * b
        hi = min(lo + b, n)
        for j in range(lo, hi):
            st.scans += 1
            v = x[j]
            # The threshold test is the whole point: once T has risen, most
            # elements fail it and the buffer stops growing.
            if v > threshold:
                cand_v.append(v)
                cand_i.append(j)
                st.accepted += 1
            else:
                st.elided += 1
        if len(cand_v) > st.max_buffer:
            st.max_buffer = len(cand_v)
        if trace_every and (step % trace_every == 0 or step == len(order) - 1):
            st.threshold_trace.append(threshold)
            st.buffer_trace.append(len(cand_v))
        if len(cand_v) >= k + b2:
            st.compact_work += len(cand_v)
            cand_v, cand_i = compact(cand_v, cand_i, k)
            threshold = min(cand_v)
            threshold_init = True
            st.compactions += 1

    st.compact_work += len(cand_v)
    cand_v, cand_i = compact(cand_v, cand_i, k)
    return cand_v, cand_i, st


# --------------------------------------------------------------------------- #
# Reference / verification helpers.
# --------------------------------------------------------------------------- #


def reference_topk(x: Sequence[float], k: int) -> List[float]:
    """Ground truth top-k values, descending (ties broken arbitrarily)."""
    if _HAVE_TORCH:
        v = torch.tensor(list(x), dtype=torch.float64)
        return torch.topk(v, k, largest=True, sorted=True).values.tolist()
    return heapq.nlargest(k, x)


def topk_is_correct(x: Sequence[float], k: int, idx: List[int]) -> bool:
    """The exact property the repo's test suite checks:
    min(selected) >= max(unselected)."""
    selected = set(idx)
    if len(selected) != len(idx):
        return False
    if not selected:
        return False
    sel_min = min(x[i] for i in selected)
    unsel_max = max(
        (x[i] for i in range(len(x)) if i not in selected), default=float("-inf")
    )
    return sel_min >= unsel_max


def theoretical_bound(n: int, k: int, b: int, b2: int) -> float:
    """(1 + k/B2) * L * H_m, the report's E[W] upper bound."""
    m = ceil_div(n, b)
    L = k + b + b2
    H_m = sum(1.0 / i for i in range(1, m + 1))
    return (1.0 + k / b2) * L * H_m


# --------------------------------------------------------------------------- #
# A few input shapes worth trying.
# --------------------------------------------------------------------------- #


def gen_input(kind: str, n: int, k: int, rng: random.Random, b: int = 1024) -> List[float]:
    if kind == "random":
        return [rng.random() for _ in range(n)]
    if kind == "ascending":
        # Adversarial for sequential block order: the largest values sit in the
        # LAST blocks, so a left-to-right scan admits almost everything.
        return [i / n for i in range(n)]
    if kind == "descending":
        return [(n - i) / n for i in range(n)]
    if kind == "staircase":
        # One constant value per B-sized block, increasing with the block index.
        # A sequential scan therefore keeps admitting whole blocks.
        nb = ceil_div(n, b)
        out = []
        block_id = 0
        while len(out) < n:
            out.extend([(block_id + 1) / nb] * b)
            block_id += 1
        return out[:n]
    raise ValueError(f"unknown input kind: {kind}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=262144, help="row length N")
    ap.add_argument("--k", type=int, default=512, help="topk")
    ap.add_argument("--b", type=int, default=1024, help="scan block size B")
    ap.add_argument("--b2", type=int, default=1024, help="compact trigger slack B2")
    ap.add_argument("--input", default="random",
                    choices=["random", "ascending", "descending", "staircase"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-torch", action="store_true",
                    help="force the heapq fallback compact (no PyTorch import)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--trace", action="store_true",
                    help="print the threshold / buffer size sampled across the scan")
    args = ap.parse_args()

    name, compact = make_compactor(use_torch=not args.no_torch)
    rng = random.Random(args.seed)
    x = gen_input(args.input, args.n, args.k, rng, b=args.b)

    print(f"compact step      : {name}")
    print(f"N={args.n}  k={args.k}  B={args.b}  B2={args.b2}  input={args.input}")
    bound = theoretical_bound(args.n, args.k, args.b, args.b2)
    print(f"theoretical E[W]  : <= (1+k/B2)*L*H_m = {bound:,.0f}")
    print()

    ok_all = True
    w_all = []
    traced = None
    for t in range(args.trials):
        trng = random.Random(1000 + t)
        st = Stats()
        do_trace = args.trace and t == 0
        t0 = time.perf_counter()
        vals, idx, st = deepselect_topk(
            x, args.k, args.b, args.b2, rng=trng, compact=compact,
            trace_every=(max(1, ceil_div(args.n, args.b) // 10) if do_trace else 0),
            stats=st,
        )
        dt = time.perf_counter() - t0

        got = sorted(vals, reverse=True)
        want = sorted(reference_topk(x, args.k), reverse=True)
        same = len(got) == len(want) and all(a == b for a, b in zip(got, want))
        prop = topk_is_correct(x, args.k, idx)
        ok_all = ok_all and same and prop
        w_all.append(st.compact_work)
        print(
            f"trial {t}: W={st.compact_work:>9,}  compactions={st.compactions:>3}"
            f"  max_buffer={st.max_buffer:>6,}  threshold={min(vals):+.4f}"
            f"  values_match={same}  topk_property={prop}  {dt*1000:6.1f} ms"
        )
        if do_trace:
            traced = st

    if traced is not None:
        print()
        print("scan trace (sampled every ~10% of the blocks):")
        print(f"  {'step':>6}{'threshold':>14}{'buffer':>10}")
        for i, (thr, buf) in enumerate(zip(traced.threshold_trace, traced.buffer_trace)):
            shown = "-inf" if thr == float("-inf") else f"{thr:+.6f}"
            print(f"  {i:>6}{shown:>14}{buf:>10,}")

    print()
    print(f"max observed W    : {max(w_all):,}   (bound {bound:,.0f})")
    print(f"W / N             : {max(w_all) / args.n:.4f}   "
          f"(i.e. the extra TopK work is this fraction of one read of the input)")
    print(f"all trials correct: {ok_all}")
    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
