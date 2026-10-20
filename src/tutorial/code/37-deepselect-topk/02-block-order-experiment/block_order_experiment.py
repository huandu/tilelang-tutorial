#!/usr/bin/env python3
"""Why the block order must be random: sequential vs random scan order.

The DeepSelect report proves its complexity bound "under a uniformly random
block permutation that is independent of the input". This script shows, on a
CPU, *why* that hypothesis is load-bearing: on an adversarially ordered input a
left-to-right scan admits almost every element, while the same input under a
random permutation behaves normally.

It reports, for each (input shape x scan order):

    A  = number of elements admitted to the candidate buffer
    R  = number of in-loop compact calls
    W  = A + k*R      (total elements fed to a compact step)

and compares W against N (one full read of the row) and against the report's
bound (1 + k/B2)*L*H_m.

This is a pure-Python model of the algorithm, not the kernel, so the timings are
irrelevant; only A, R and W mean anything here.

Run:
    python3 block_order_experiment.py
    python3 block_order_experiment.py --n 1048576 --k 512
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
from typing import Callable, List, Sequence, Tuple

try:
    import torch

    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def compact_topk(values: List[float], indices: List[int], k: int):
    """Top-k by value, keeping (value, index) aligned."""
    if len(values) <= k:
        return list(values), list(indices)
    if _HAVE_TORCH:
        pos = torch.topk(torch.tensor(values, dtype=torch.float64), k,
                         largest=True, sorted=False).indices.tolist()
    else:
        pos = heapq.nlargest(k, range(len(values)), key=lambda p: values[p])
    return [values[p] for p in pos], [indices[p] for p in pos]


def deepselect(x: Sequence[float], k: int, b: int, b2: int,
               order: Sequence[int]) -> Tuple[List[float], List[int], dict]:
    """Run the algorithm with an explicit block order (no internal shuffle)."""
    n = len(x)
    cand_v: List[float] = []
    cand_i: List[int] = []
    threshold = float("-inf")
    accepted = compactions = work = 0
    max_buffer = 0
    for block in order:
        lo, hi = block * b, min((block + 1) * b, n)
        for j in range(lo, hi):
            v = x[j]
            if v > threshold:
                cand_v.append(v)
                cand_i.append(j)
                accepted += 1
        max_buffer = max(max_buffer, len(cand_v))
        if len(cand_v) >= k + b2:
            work += len(cand_v)
            cand_v, cand_i = compact_topk(cand_v, cand_i, k)
            threshold = min(cand_v)
            compactions += 1
    work += len(cand_v)
    cand_v, cand_i = compact_topk(cand_v, cand_i, k)
    return cand_v, cand_i, dict(
        accepted=accepted, compactions=compactions, work=work, max_buffer=max_buffer
    )


def gen_input(kind: str, n: int, k: int, rng: random.Random, b: int = 1024) -> List[float]:
    if kind == "random":
        return [rng.random() for _ in range(n)]
    if kind == "ascending":
        # Largest values in the LAST blocks: a sequential scan keeps admitting.
        return [i / n for i in range(n)]
    if kind == "descending":
        # Largest values FIRST: a sequential scan tightens the threshold at once.
        return [(n - i) / n for i in range(n)]
    if kind == "staircase":
        # One constant value per B-sized block, rising with the block index:
        # a sequential scan keeps admitting whole blocks.
        nb = ceil_div(n, b)
        out: List[float] = []
        block_id = 0
        while len(out) < n:
            out.extend([(block_id + 1) / nb] * b)
            block_id += 1
        return out[:n]
    raise ValueError(kind)


def bound(n: int, k: int, b: int, b2: int) -> float:
    m = ceil_div(n, b)
    return (1.0 + k / b2) * (k + b + b2) * sum(1.0 / i for i in range(1, m + 1))


def reference(x: Sequence[float], k: int) -> List[float]:
    if _HAVE_TORCH:
        return torch.topk(torch.tensor(list(x), dtype=torch.float64), k).values.tolist()
    return heapq.nlargest(k, x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=262144)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--b", type=int, default=1024)
    ap.add_argument("--b2", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_blocks = ceil_div(args.n, args.b)
    seq = list(range(n_blocks))

    print(f"compact step: {'torch.topk' if _HAVE_TORCH else 'heapq.nlargest'}")
    print(f"N={args.n}  blocks={n_blocks}  k={args.k}  B={args.b}  B2={args.b2}")
    print(f"theoretical E[W] bound = {bound(args.n, args.k, args.b, args.b2):,.0f}")
    print()
    header = f"{'input':<11}{'order':<11}{'A':>10}{'R':>6}{'W':>12}{'W/N':>9}  correct"
    print(header)
    print("-" * len(header))

    for kind in ["random", "ascending", "descending", "staircase"]:
        x = gen_input(kind, args.n, args.k, rng, b=args.b)
        want = sorted(reference(x, args.k), reverse=True)
        for label, order in (
            ("sequential", seq),
            ("random", random.Random(args.seed + 7).sample(seq, len(seq))),
        ):
            rng_trial = random.Random(args.seed + 3)
            vals, _, st = deepselect(x, args.k, args.b, args.b2, order)
            good = len(vals) == len(want) and all(
                abs(a - b) < 1e-12 for a, b in zip(sorted(vals, reverse=True), want)
            )
            print(
                f"{kind:<11}{label:<11}{st['accepted']:>10,}{st['compactions']:>6}"
                f"{st['work']:>12,}{st['work']/args.n:>9.3f}  {good}"
            )
        print()

    print("Read the table like this:")
    print("  * 'random' is stable: W stays a small multiple of k no matter the input.")
    print("  * 'sequential' is fine when the data is already descending, but on")
    print("    'ascending'/'staircase' it admits nearly every element (A ~ N) and")
    print("    does O(N) extra compact work -- that is the degeneration the report's")
    print("    random permutation exists to prevent.")


if __name__ == "__main__":
    main()
