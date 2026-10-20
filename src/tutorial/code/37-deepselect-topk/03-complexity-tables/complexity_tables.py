#!/usr/bin/env python3
"""Small tables for the DeepSelect complexity discussion (pure stdlib).

Everything here is arithmetic on the report's formulas:

    W  <= (1 + k/B2) * L * H_m,        L = k + B + B2,  m = ceil(N/B),
    W   = O( (1 + k/B2) * (k + B + B2) * log(N/B) ),
    W   = O( k * log(N/k) )            when B, B2 = Theta(k).

and on a deliberately crude model of what `torch.topk` has to do:

    torch.topk reader:   read N elements once,
    torch.topk sorter:   ~ N * log2(k) comparisons for a size-k selection.

The numbers are *not* measurements. They are the shapes of the cost curves, so
the reader can see where one term dominates. The performance numbers that do
come from a real machine are in the chapter, sourced from the repo's README.

Run:
    python3 complexity_tables.py
    python3 complexity_tables.py --k 512 --b 1024 --b2 1024
"""

from __future__ import annotations

import argparse
import math


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def harmonic(m: int) -> float:
    return sum(1.0 / i for i in range(1, m + 1))


def w_bound(n: int, k: int, b: int, b2: int) -> float:
    """The report's E[W] upper bound, (1 + k/B2) * L * H_m."""
    m = ceil_div(n, b)
    return (1.0 + k / b2) * (k + b + b2) * harmonic(m)


def w_asymptotic(n: int, k: int, b: int, b2: int) -> float:
    """The O((1 + k/B2)(k+B+B2) log(N/B)) form, with log = natural log."""
    return (1.0 + k / b2) * (k + b + b2) * math.log(max(n / b, 2.0))


def w_klog(n: int, k: int) -> float:
    """The O(k log(N/k)) form you get when B, B2 = Theta(k)."""
    return k * math.log(max(n / k, 2.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--b", type=int, default=1024)
    ap.add_argument("--b2", type=int, default=1024)
    args = ap.parse_args()
    k, b, b2 = args.k, args.b, args.b2

    print(f"k={k}  B={b}  B2={b2}   L = k+B+B2 = {k + b + b2}")
    print()

    print("Table 1 - the extra TopK work shrinks, relative to the input, as N grows")
    head = f"{'N':>10}{'m=N/B':>8}{'H_m':>9}{'E[W] bound':>14}{'bound/N':>11}"
    print(head)
    print("-" * len(head))
    for e in range(12, 25):
        n = 1 << e
        m = ceil_div(n, b)
        wb = w_bound(n, k, b, b2)
        print(f"{n:>10,}{m:>8,}{harmonic(m):>9.2f}{wb:>14,.0f}{wb / n:>11.4f}")
    print()

    print("Table 2 - the three cost forms side by side (all are work in elements)")
    head = (f"{'N':>10}{'bound(N)':>14}{'asymptotic':>14}"
            f"{'k*log(N/k)':>14}{'bound/asym':>12}")
    print(head)
    print("-" * len(head))
    for e in range(12, 25):
        n = 1 << e
        wb = w_bound(n, k, b, b2)
        wa = w_asymptotic(n, k, b, b2)
        wk = w_klog(n, k)
        print(f"{n:>10,}{wb:>14,.0f}{wa:>14,.0f}{wk:>14,.0f}{wb / wa:>12.2f}")
    print("  (bound/asym -> 1 as N grows: H_m ~ ln m + gamma, and ln m ~ log2(m)*ln2)")
    print()

    print("Table 3 - B2 trades compact frequency against compact size")
    print(f"N = {1 << 20:,}, k = {k}")
    head = f"{'B2':>8}{'1 + k/B2':>11}{'L':>8}{'E[W] bound':>14}{'bound/N':>11}"
    print(head)
    print("-" * len(head))
    n = 1 << 20
    for b2v in [128, 256, 512, 1024, 2048, 4096, 8192]:
        wb = w_bound(n, k, b, b2v)
        print(f"{b2v:>8,}{1 + k / b2v:>11.2f}{k + b + b2v:>8,}{wb:>14,.0f}"
              f"{wb / n:>11.4f}")
    print("  (small B2 -> more compactions; large B2 -> bigger buffers. Both raise L or 1+k/B2.)")
    print()

    print("Table 4 - the torch.topk cost we are trying to avoid (crude model)")
    head = f"{'N':>10}{'read elems':>12}{'~sort work':>14}{'total':>14}{'vs DeepSelect':>15}"
    print(head)
    print("-" * len(head))
    for e in range(12, 25):
        n = 1 << e
        sort_work = n * math.log2(k)
        total = n + sort_work
        ds = w_bound(n, k, b, b2)
        print(f"{n:>10,}{n:>12,}{sort_work:>14,.0f}{total:>14,.0f}{total / ds:>15.1f}x")
    print()
    print("Reading: the DeepSelect column is the *extra* work on top of the single")
    print("read of the input. torch.topk's model adds an N*log2(k) sort pass, which")
    print("is why it is the slower baseline -- and why the gap widens with N.")


if __name__ == "__main__":
    main()
