#!/usr/bin/env python3
"""Numerical check of the online-softmax derivation in chapter 2 of this chapter.

The chapter derives, for one query row with raw scores x_j and scale s:

    m_new = max(m_prev, M_blk)
    c     = exp(m_prev - m_new)                            # always <= 1
    l     = c * l                     + sum(exp(s*x_j - m_new))
    O     = c * O                     + sum(exp(s*x_j - m_new) * v_j)
    out   = O / l

and proves by induction that out == sum_j softmax(s*x)_j * v_j for **any**
choice of running maximum.

This program checks that claim numerically, and additionally checks the three
statements the chapter makes about *numbers* rather than algebra:

  (a) the online form is stable where the "no max" form overflows;
  (b) the result is independent of the running-max schedule (we replay the
      same row with different block sizes and with the keys reversed);
  (c) the running denominator l is NOT monotone, even though O/l is exact;
  (d) the specific failure modes named in section 5.1:
      -exp(-inf) - (-inf) style NaN from a running max that is still -inf, and
      what `check_inf` does about it.

Everything runs on the CPU with numpy. No GPU, no tilelang.

Run:  python3 online_softmax.py
"""

from __future__ import annotations

import numpy as np


def _as_matrix(v: np.ndarray) -> np.ndarray:
    """Values are always [n, d]; a 1-D input is treated as d = 1."""
    v = np.asarray(v, dtype=float)
    return v.reshape(v.shape[0], -1)


def exact_softmax_attention(x: np.ndarray, v: np.ndarray, s: float) -> tuple[np.ndarray, float]:
    """Textbook three-pass reference: max, then sum, then weighted sum."""
    v = _as_matrix(v)
    z = s * x
    m = np.max(z)
    e = np.exp(z - m)
    l = np.sum(e)
    p = e / l
    return p @ v, m


def online_softmax_attention(
    x: np.ndarray,
    v: np.ndarray,
    s: float,
    block: int,
    trace: bool = False,
) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """Single-pass blockwise algorithm, written exactly as the chapter states it."""
    n = x.shape[0]
    v = _as_matrix(v)
    m = -np.inf          # running max of s*x, i.e. scores_max in the kernel
    l = 0.0              # running denominator, i.e. logsum in the kernel
    o = np.zeros(v.shape[1:])  # unnormalised numerator, i.e. acc_o in the kernel
    rows: list[tuple[int, float, float, float, float]] = []

    for start in range(0, n, block):
        xb = x[start : start + block]
        vb = v[start : start + block]

        M = float(np.max(s * xb))          # block max, = reduce_max(acc_s)
        m_new = max(m, M)                  # merge with running max
        c = float(np.exp(m - m_new))       # rescale factor, <= 1
        p = np.exp(s * xb - m_new)         # block probabilities (unnormalised)

        l = c * l + float(np.sum(p))
        o = c * o + p @ vb
        m = m_new

        if trace:
            rows.append((start // block, m, c, l, float(np.max(np.abs(o)))))

    if trace:
        return o / l, rows
    return o / l, []


def demo_trace() -> None:
    print("=" * 72)
    print("1. per-step trace of the recurrence (small, exact arithmetic in fp64)")
    print("=" * 72)
    x = np.array([2.0, -1.0, 3.0, 0.5, -4.0, 1.5, 0.0, -2.0, 1.0, 2.5, -3.0, 0.25])
    v = rng_like_v(x.shape[0])       # [n, d] with d = 3
    s = 0.5
    out_online, rows = online_softmax_attention(x, v, s, block=4, trace=True)
    out_exact, m_exact = exact_softmax_attention(x, v, s)

    print(f"{'blk':>4}{'m (running max)':>18}{'c = exp(m_prev-m)':>20}"
          f"{'l (denominator)':>18}{'max|O|':>12}")
    print("-" * 72)
    for blk, m, c, l, onorm in rows:
        print(f"{blk:>4}{m:>18.6f}{c:>20.6f}{l:>18.6f}{onorm:>12.6f}")
    print("-" * 72)
    print(f"exact row max (scaled): {m_exact:.6f}")
    print(f"online output : {np.array2string(out_online, precision=10)}")
    print(f"exact  output : {np.array2string(out_exact, precision=10)}")
    print(f"max abs diff  : {np.max(np.abs(out_online - out_exact)):.3e}")
    assert np.allclose(out_online, out_exact, rtol=0, atol=1e-12)
    print("OK: online == exact")
    print()


def rng_like_v(n: int, d: int = 3) -> np.ndarray:
    """Deterministic [n, d] value matrix, so the trace is reproducible."""
    return np.arange(1.0, 1.0 + n * d).reshape(n, d) / n


def demo_block_invariance() -> None:
    print("=" * 72)
    print("2. result does not depend on the block schedule")
    print("=" * 72)
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 8.0, size=4096)
    v = rng.normal(size=(4096, 4))
    s = 1.0 / 64.0
    ref, _ = exact_softmax_attention(x, v, s)

    print(f"{'block size':>12}{'#blocks':>10}{'max abs diff':>18}")
    print("-" * 42)
    for block in (16, 64, 128, 256, 1024, 4096):
        out, _ = online_softmax_attention(x, v, s, block=block)
        print(f"{block:>12}{len(range(0, 4096, block)):>10}"
              f"{np.max(np.abs(out - ref)):>18.3e}")
    print()

    # Reversing the key order gives a different running-max schedule. The
    # weighted sum must be identical to within fp rounding.
    out_rev, _ = online_softmax_attention(x[::-1].copy(), v[::-1].copy(), s, block=128)
    print(f"reversed key order vs reference: max abs diff = "
          f"{np.max(np.abs(out_rev - ref)):.3e}")
    print("(the running max path is completely different; the ratio is the same)")
    print()


def demo_overflow() -> None:
    print("=" * 72)
    print("3. what the running max buys: overflow of exp without it")
    print("=" * 72)
    x = np.array([700.0, 710.0, 720.0, 690.0])
    v = np.array([[1.0], [2.0], [3.0], [4.0]])
    s = 1.0

    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.exp(s * x)
        naive = raw @ v / np.sum(raw)
    safe, m = exact_softmax_attention(x, v, s)
    online, _ = online_softmax_attention(x, v, s, block=2)

    print(f"exp(s*x) without subtracting the max : {raw}")
    print(f"naive sum_j exp(sx_j) v_j / sum exp    : {naive.ravel()}   <- inf/inf = NaN")
    print(f"row max m = {m:.1f}")
    print(f"exact (subtract max)                   : {safe.ravel()}")
    print(f"online (block = 2)                     : {online.ravel()}")
    assert np.all(np.isfinite(safe)) and np.all(np.isfinite(online))
    assert np.all(np.isnan(naive))
    print("OK: the max is a range device, not a mathematical one")
    print()


def demo_l_not_monotone() -> None:
    print("=" * 72)
    print("4. the running denominator l is NOT monotone")
    print("=" * 72)
    x = np.array([0.0, 0.0, 0.0, 0.0, 100.0, 100.0])
    v = np.ones((6, 1))
    out, rows = online_softmax_attention(x, v, 1.0, block=2, trace=True)
    print(f"{'blk':>4}{'m':>10}{'c':>14}{'l':>16}")
    print("-" * 44)
    for blk, m, c, l, _ in rows:
        print(f"{blk:>4}{m:>10.1f}{c:>14.3e}{l:>16.6f}")
    print("-" * 44)
    print("l goes 2 -> 4 -> 2: when a much larger max shows up, the rescale")
    print("factor c shrinks the old l faster than the new block adds to it.")
    print("what stays exact is the ratio O/l, not l itself:")
    exact, _ = exact_softmax_attention(x, v, 1.0)
    print(f"  online O/l = {out.ravel()}, exact = {exact.ravel()}")
    assert np.allclose(out, exact, rtol=0, atol=1e-12)
    assert rows[1][3] > rows[2][3], "expected l to decrease here"
    print("OK: l is not monotone, but O/l still equals the exact softmax")
    print()


def demo_check_inf() -> None:
    print("=" * 72)
    print("5. the -inf - (-inf) hazard and what check_inf does")
    print("=" * 72)
    neg_inf = -np.inf
    with np.errstate(invalid="ignore"):
        bad = np.exp2(neg_inf - neg_inf)
    print(f"exp2(-inf - -inf) = {bad}   <- NaN, this is the failure mode")
    print()
    print("scenario: sliding window, first query block sees NO key at all.")
    # Emulate the kernel: scores are all -inf for block 0, then real scores.
    x_blocks = [np.array([neg_inf, neg_inf, neg_inf]), np.array([1.0, 2.0, 0.5])]
    v_blocks = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])]

    def run(xs: list[np.ndarray], vs: list[np.ndarray], check_inf: bool):
        m = -np.inf
        l = 0.0
        o = 0.0
        for xb, vb in zip(xs, vs):
            M = float(np.max(xb))
            m_new = max(m, M)
            c = np.exp(m - m_new)
            if check_inf and m_new == -np.inf:
                m_new = 0.0
                c = np.exp(m - m_new) if np.isfinite(m) else 0.0
            p = np.exp(xb - m_new)
            l = c * l + float(np.sum(p))
            o = c * o + float(p @ vb)
            m = m_new
        return o / l if l != 0 else float("nan")

    with np.errstate(all="ignore"):
        without = run(x_blocks, v_blocks, check_inf=False)
        with_fix = run(x_blocks, v_blocks, check_inf=True)
    expected = (np.exp(1.0) * 1 + np.exp(2.0) * 1 + np.exp(0.5) * 1) / (
        np.exp(1.0) + np.exp(2.0) + np.exp(0.5)
    )
    print(f"without check_inf : {without}   <- NaN propagates")
    print(f"with    check_inf : {with_fix:.10f}")
    print(f"only-visible-block: {expected:.10f}")
    assert np.isnan(without)
    assert np.isclose(with_fix, expected)
    print("OK: check_inf replaces a still-infinite running max with 0 so that")
    print("    the next rescale factor is exp2(-inf - 0) = 0 instead of NaN")
    print("    (see example_mha_sink_fwd_bhsd.py:103-109 in the locked tree).")
    print()


def main() -> None:
    print(__doc__)
    demo_trace()
    demo_block_invariance()
    demo_overflow()
    demo_l_not_monotone()
    demo_check_inf()
    print("=" * 72)
    print("All checks passed. This validates the derivation on the CPU; it says")
    print("nothing about any GPU or NPU kernel.")
    print("=" * 72)


if __name__ == "__main__":
    main()
