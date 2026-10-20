#!/usr/bin/env python3
"""Check the split-KV (flash-decoding) combine step from section 5.4.

The chapter claims: partial outputs from KV splits cannot be averaged or
summed; they must be weighted by each split's softmax mass.

    O_s  = (1/l_s) * sum_{j in split s} exp(s*x_j - m_s) * v_j     (normalised)
    lse_s = log2(l_s) + m_s * log2(e) * scale_ratio                (base-2 LSE)
    lse_logsum = log2( sum_s 2**(lse_s - lse_max) ) + lse_max      (logsumexp)
    w_s  = 2**(lse_s - lse_logsum)                                 ( = l_s / sum l )
    O    = sum_s w_s * O_s

This program:

  1. builds a decode-shaped problem (one query row, long KV),
  2. computes the per-split normalised partials, exactly as the split kernel
     does (see examples/flash_decoding/example_mha_inference.py:80-113),
  3. combines them three ways -- correctly (weighted), wrongly (mean), and
     wrongly (sum) -- and shows which one reproduces the exact answer,
  4. repeats with the partials and the LSE rounded to float16, to show what
     the split count does to the error (this is the numerical half of the
     "reduction order" trap; it is a CPU model, not a device measurement).

Run:  python3 split_kv_combine.py
"""

from __future__ import annotations

import numpy as np

LOG2E = 1.4426950408889634


def exact_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float) -> np.ndarray:
    x = k @ q                      # [n_kv]
    z = x * scale
    m = np.max(z)
    e = np.exp(z - m)
    p = e / np.sum(e)
    return p @ v


def split_partials(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    scale: float,
    num_split: int,
    round_to_fp16: bool = False,
) -> tuple[list[np.ndarray], list[float]]:
    """Return per-split normalised outputs O_s and base-2 LSEs lse_s."""
    n = k.shape[0]
    assert n % num_split == 0, "this model requires an even split"
    span = n // num_split
    scale_log2e = scale * LOG2E
    outs: list[np.ndarray] = []
    lses: list[float] = []

    for s_i in range(num_split):
        ks = k[s_i * span : (s_i + 1) * span]
        vs = v[s_i * span : (s_i + 1) * span]
        x = ks @ q                              # raw scores of this split
        m = float(np.max(x))                    # running max inside the split
        p = np.exp2(x * scale_log2e - m * scale_log2e)
        l = float(np.sum(p))
        o = (p @ vs) / l                        # normalised partial output
        lse = float(np.log2(l) + m * scale_log2e)
        if round_to_fp16:
            o = o.astype(np.float16).astype(np.float64)
            lse = float(np.float16(lse).astype(np.float64))
        outs.append(o)
        lses.append(lse)
    return outs, lses


def combine(outs: list[np.ndarray], lses: list[float]) -> np.ndarray:
    """The combine kernel's formula, verbatim."""
    lse_arr = np.asarray(lses, dtype=np.float64)
    lse_max = float(np.max(lse_arr))
    lse_logsum = float(np.log2(np.sum(np.exp2(lse_arr - lse_max))) + lse_max)
    acc = np.zeros_like(outs[0])
    for o, lse in zip(outs, lses):
        w = float(np.exp2(lse - lse_logsum))
        acc += w * o
    return acc


def demo_weighting() -> None:
    print("=" * 74)
    print("1. weighted combine vs. the two naive ways of merging partials")
    print("=" * 74)
    rng = np.random.default_rng(7)
    d, n_kv, num_split = 8, 1024, 8
    q = rng.normal(size=d)
    k = rng.normal(size=(n_kv, d))
    v = rng.normal(size=(n_kv, d))
    scale = 1.0 / np.sqrt(d)

    ref = exact_attention(q, k, v, scale)
    outs, lses = split_partials(q, k, v, scale, num_split)

    weighted = combine(outs, lses)
    mean_of = np.mean(np.stack(outs), axis=0)      # wrong
    sum_of = np.sum(np.stack(outs), axis=0)        # wrong

    print(f"d={d}, Skv={n_kv}, num_split={num_split}")
    print(f"{'merge rule':<34}{'max abs err':>16}")
    print("-" * 50)
    print(f"{'weighted by l_s / sum(l)  (correct)':<34}"
          f"{np.max(np.abs(weighted - ref)):>16.3e}")
    print(f"{'plain mean of partials      (wrong)':<34}"
          f"{np.max(np.abs(mean_of - ref)):>16.3e}")
    print(f"{'plain sum of partials       (wrong)':<34}"
          f"{np.max(np.abs(sum_of - ref)):>16.3e}")
    print()
    print("per-split weights (exp2(lse_s - logsumexp(lse))):")
    lse_arr = np.asarray(lses)
    lse_max = float(np.max(lse_arr))
    lse_logsum = float(np.log2(np.sum(np.exp2(lse_arr - lse_max))) + lse_max)
    weights = np.exp2(lse_arr - lse_logsum)
    print("  " + " ".join(f"{w:.4f}" for w in weights))
    print(f"  sum of weights = {weights.sum():.10f}  (must be 1)")
    print()


def demo_precision_vs_num_split() -> None:
    print("=" * 74)
    print("2. error vs. number of splits, with fp16 partials + fp16 LSE")
    print("=" * 74)
    rng = np.random.default_rng(11)
    d, n_kv = 128, 16384
    q = rng.normal(size=d) * 0.5
    k = rng.normal(size=(n_kv, d))
    v = rng.normal(size=(n_kv, d))
    scale = 1.0 / np.sqrt(d)
    ref = exact_attention(q, k, v, scale)

    print(f"d={d}, Skv={n_kv}")
    print(f"{'num_split':>10}{'fp64 partials':>18}{'fp16 partials':>18}")
    print("-" * 46)
    for num_split in (1, 2, 4, 8, 16, 32):
        if n_kv % num_split:
            continue
        o64, l64 = split_partials(q, k, v, scale, num_split, round_to_fp16=False)
        o16, l16 = split_partials(q, k, v, scale, num_split, round_to_fp16=True)
        e64 = np.max(np.abs(combine(o64, l64) - ref))
        e16 = np.max(np.abs(combine(o16, l16) - ref))
        print(f"{num_split:>10}{e64:>18.3e}{e16:>18.3e}")
    print()
    print("reading: the fp64 column stays near machine precision, so the")
    print("        *algebra* of the combine is exact. The fp16 column grows")
    print("        with num_split because each split contributes one rounded")
    print("        partial output plus one rounded LSE, and the LSE's absolute")
    print("        magnitude (hundreds) eats most of fp16's 10-bit mantissa.")
    print()
    print("this is the same effect the Ascend backward README reports for dQ")
    print("(repeated BF16 rounding while atomically accumulating across 64 KV")
    print("tiles): the reduction split count is a numerical parameter, not just")
    print("a scheduling parameter.")
    print()


def main() -> None:
    print(__doc__)
    demo_weighting()
    demo_precision_vs_num_split()
    print("=" * 74)
    print("REMINDER: a CPU model. It validates the algebra and shows the shape")
    print("of the precision effect; it is not a measurement of any kernel.")
    print("=" * 74)


if __name__ == "__main__":
    main()
