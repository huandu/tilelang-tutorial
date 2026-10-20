#!/usr/bin/env python3
"""A block-by-block CPU reference for the four kernel variants in this chapter.

This file is the chapter's *reference implementation*. It re-implements the
official TileLang forward loop with numpy, keeping the same structure as the
kernel, so that each claim in the chapter can be checked numerically:

    for m_block:                       # bx
        for n_block in pipelined range:  # k
            acc_s = Q_blk @ K_blk^T        ( + mask )
            m_new = max(m_prev, rowmax(acc_s))
            c     = exp2(m_prev*scale - m_new*scale)
            p     = exp2(acc_s*scale - m_new*scale)
            l     = c*l + rowsum(p)
            acc_o = c*acc_o + p @ V_blk
        acc_o /= l

Variants covered, mirroring `_vendor/tilelang-v0.1.15/examples/`:

  * full attention                       (example_mha_fwd_bhsd.py)
  * causal with a KV-cache prefix        (past_len, same file)
  * sliding window                       (example_mha_sink_fwd_bhsd.py)
  * attention sink                       (same file)
  * varlen packing                       (example_mha_fwd_varlen.py)

For each variant the result is compared against an exact masked softmax, and
the `check_inf` line is toggled to show when it is load-bearing.

Run:  python3 block_flash_attention.py
"""

from __future__ import annotations

import numpy as np

LOG2E = 1.4426950408889634
NEG_INF = -np.inf


def exact_masked_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    scale: float,
    causal: bool = False,
    past_len: int = 0,
    window: int | None = None,
    sink: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Textbook softmax attention with the same masking rules as the kernels."""
    sq, d = q.shape
    skv = k.shape[0]
    z = (q @ k.T) * scale                       # [sq, skv]
    mask = np.zeros((sq, skv), dtype=bool)
    if causal or window is not None:
        qi = np.arange(sq)[:, None] + past_len
        kj = np.arange(skv)[None, :]
        mask |= qi < kj
        if window is not None:
            mask |= qi >= kj + window
    z = np.where(mask, NEG_INF, z)
    if sink is not None:
        # the sink is one extra softmax entry with value `sink` and a zero
        # contribution to the output (see example_mha_sink_fwd_bhsd.py:124-126)
        z = np.concatenate([z, sink[:, None]], axis=1)
    m = np.max(z, axis=1, keepdims=True)
    e = np.exp(z - m)
    l = np.sum(e, axis=1)
    p = e / l[:, None]
    if sink is not None:
        p = p[:, :-1]
    # base-2 log-sum-exp of the scaled scores: log2(sum exp(z)) = log2(l) + m*log2(e)
    return p @ v, np.log2(l) + m[:, 0] * LOG2E


def block_flash_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    scale: float,
    block_m: int = 4,
    block_n: int = 4,
    causal: bool = False,
    past_len: int = 0,
    window: int | None = None,
    sink: float | None = None,
    check_inf: bool = True,
    q_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """The chapter's kernel loop, executed block by block on the CPU.

    `q_offset` is the absolute index of `q[0]`; it matters for causal masking
    when the kernel is invoked on a sub-range of the query axis (as the
    per-sample loop in case 4 does).
    """
    sq, d = q.shape
    skv = k.shape[0]
    scale_log2e = scale * LOG2E
    out = np.zeros((sq, d), dtype=np.float64)
    lse = np.zeros(sq, dtype=np.float64)

    n_m_blocks = -(-sq // block_m)
    for mb in range(n_m_blocks):
        rows = slice(mb * block_m, min((mb + 1) * block_m, sq))
        q_blk = q[rows]
        rows_eff = q_blk.shape[0]

        acc_o = np.zeros((rows_eff, d))
        logsum = np.zeros(rows_eff)
        scores_max = np.full(rows_eff, NEG_INF)

        # loop_range, exactly as example_mha_fwd_bhsd.py:59-63
        first_q = q_offset + mb * block_m
        end = -(-skv // block_n)
        if causal or window is not None:
            end = min(end, -(-(first_q + rows_eff + past_len) // block_n))
        start = 0
        if window is not None:
            start = max(0, (first_q + past_len - window) // block_n)

        for nb in range(start, end):
            k_blk = k[nb * block_n : (nb + 1) * block_n]
            v_blk = v[nb * block_n : (nb + 1) * block_n]
            cols = k_blk.shape[0]

            acc_s = q_blk @ k_blk.T                       # [rows_eff, cols]
            q_idx = np.arange(first_q, first_q + rows_eff)[:, None] + past_len
            k_idx = np.arange(nb * block_n, nb * block_n + cols)[None, :]
            masked = np.zeros_like(acc_s, dtype=bool)
            if causal or window is not None:
                masked |= q_idx < k_idx
                if window is not None:
                    masked |= q_idx >= k_idx + window
            acc_s = np.where(masked, NEG_INF, acc_s)

            scores_max_prev = scores_max.copy()
            scores_max = np.max(acc_s, axis=1)
            scores_max = np.maximum(scores_max, scores_max_prev)

            if check_inf:
                # example_mha_sink_fwd_bhsd.py:103-108
                scores_max = np.where(np.isneginf(scores_max), 0.0, scores_max)

            with np.errstate(invalid="ignore"):
                c = np.exp2(scores_max_prev * scale_log2e - scores_max * scale_log2e)
            p = np.exp2(acc_s * scale_log2e - scores_max[:, None] * scale_log2e)
            p = np.where(masked, 0.0, p)

            logsum = logsum * c + np.sum(p, axis=1)
            acc_o = acc_o * c[:, None] + p @ v_blk

        if sink is not None:
            # the ONLY change for attention sink, example_mha_sink_fwd_bhsd.py:126
            logsum += np.exp2(sink * LOG2E - scores_max * scale_log2e)

        with np.errstate(divide="ignore", invalid="ignore"):
            out[rows] = acc_o / logsum[:, None]
        lse[rows] = np.log2(logsum) + scores_max * scale_log2e

    return out, lse


def err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def case_full() -> None:
    print("-" * 74)
    print("case 1: full attention (no mask)")
    print("-" * 74)
    rng = np.random.default_rng(1)
    sq, skv, d = 20, 20, 16
    q, k, v = rng.normal(size=(sq, d)), rng.normal(size=(skv, d)), rng.normal(size=(skv, d))
    scale = 1.0 / np.sqrt(d)
    ref, _ = exact_masked_attention(q, k, v, scale)
    print(f"{'block_m x block_n':>20}{'max abs err':>18}")
    for bm, bn in ((4, 4), (8, 5), (20, 20), (8, 8), (4, 20)):
        got, _ = block_flash_attention(q, k, v, scale, block_m=bm, block_n=bn)
        e = err(got, ref)
        assert e < 1e-12, f"full attention mismatch for {bm}x{bn}: {e}"
        print(f"{f'{bm} x {bn}':>20}{e:>18.3e}")
    print("all block shapes reproduce the exact answer")
    print()


def case_causal() -> None:
    print("-" * 74)
    print("case 2: causal, with and without a KV-cache prefix (past_len)")
    print("-" * 74)
    rng = np.random.default_rng(2)
    d = 16
    for past_len in (0, 5, 12):
        sq = 24
        skv = sq + past_len
        q = rng.normal(size=(sq, d))
        k = rng.normal(size=(skv, d))
        v = rng.normal(size=(skv, d))
        scale = 1.0 / np.sqrt(d)
        ref, _ = exact_masked_attention(q, k, v, scale, causal=True, past_len=past_len)
        got, _ = block_flash_attention(q, k, v, scale, 8, 8, causal=True, past_len=past_len)
        assert err(got, ref) < 1e-12, f"causal mismatch at past_len={past_len}"
        print(f"past_len={past_len:>3} (Sq={sq}, Skv={skv}): "
              f"max abs err = {err(got, ref):.3e}")
    print()
    print("note: past_len shifts the causal diagonal, so a query row may see")
    print("      keys that are 'in the future' relative to the query index.")
    print()


def case_window_and_sink() -> None:
    print("-" * 74)
    print("case 3: sliding window + attention sink, with and without check_inf")
    print("-" * 74)
    rng = np.random.default_rng(3)
    sq = skv = 32
    d = 16
    q, k, v = rng.normal(size=(sq, d)), rng.normal(size=(skv, d)), rng.normal(size=(skv, d))
    scale = 1.0 / np.sqrt(d)
    window = 8
    sink = 3.5

    ref, _ = exact_masked_attention(q, k, v, scale, causal=True, window=window,
                                    sink=np.full(sq, sink))
    ok, _ = block_flash_attention(q, k, v, scale, 8, 8, causal=True, window=window,
                                  sink=sink, check_inf=True)
    bad, _ = block_flash_attention(q, k, v, scale, 8, 8, causal=True, window=window,
                                   sink=sink, check_inf=False)
    print(f"window={window}, sink={sink}")
    print(f"  with    check_inf : max abs err = {err(ok, ref):.3e}")
    print(f"  without check_inf : max abs err = {err(bad, ref):.3e}"
          f"   (NaN in {int(np.sum(~np.isfinite(bad)))} entries)")

    # A sink alone (no window) never leaves a row with an empty prefix under
    # causal masking, so check_inf is not needed there. Show it for contrast.
    ref2, _ = exact_masked_attention(q, k, v, scale, causal=True,
                                     sink=np.full(sq, sink))
    got2, _ = block_flash_attention(q, k, v, scale, 8, 8, causal=True,
                                    sink=sink, check_inf=False)
    print(f"  causal + sink, no window, check_inf off: "
          f"max abs err = {err(got2, ref2):.3e}")
    assert err(ok, ref) < 1e-12, "sliding window + sink mismatch"
    assert not np.all(np.isfinite(bad)), "expected check_inf to be load-bearing here"
    assert err(got2, ref2) < 1e-12, "causal + sink mismatch"
    print()
    print("reading: under causal masking the diagonal is always visible, so the")
    print("running max is finite from the first block. The window is what makes")
    print("early query rows see *nothing*, and that is when check_inf matters")
    print("(example_mha_sink_fwd_bhsd.py:104 says exactly this).")
    print()


def case_varlen() -> None:
    print("-" * 74)
    print("case 4: varlen packing (cu_seqlens) vs. one kernel per sample")
    print("-" * 74)
    rng = np.random.default_rng(4)
    d = 16
    lengths = [7, 19, 4, 13]
    scale = 1.0 / np.sqrt(d)
    cu = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    total = int(cu[-1])
    q_unpad = rng.normal(size=(total, d))
    k_unpad = rng.normal(size=(total, d))
    v_unpad = rng.normal(size=(total, d))

    # reference: run the "kernel" once per sample on its own slice
    pieces = []
    for b, n in enumerate(lengths):
        qs = slice(int(cu[b]), int(cu[b + 1]))
        ref, _ = exact_masked_attention(q_unpad[qs], k_unpad[qs], v_unpad[qs], scale,
                                        causal=True)
        pieces.append(ref)
    ref_packed = np.concatenate(pieces)

    # emulate the packed kernel: grid over ceil(max_seqlen/block_m) blocks per
    # sample, with an explicit guard on the write-back
    out = np.full_like(q_unpad, np.nan)
    max_seq = max(lengths)
    for b, n in enumerate(lengths):
        q0 = int(cu[b])
        qs = q_unpad[q0 : q0 + n]
        ks = k_unpad[q0 : q0 + n]
        vs = v_unpad[q0 : q0 + n]
        n_blocks = -(-max_seq // 8)
        for mb in range(n_blocks):
            lo, hi = mb * 8, min((mb + 1) * 8, n)
            if lo >= n:
                break
            got, _ = block_flash_attention(qs[lo:hi], ks, vs, scale, 8, 8,
                                           causal=True, q_offset=lo)
            # the guard in example_gqa_fwd_varlen.py:122-124
            out[q0 + lo : q0 + hi] = got
    print(f"lengths={lengths}, packed total={total}, max_seqlen={max_seq}")
    assert err(out, ref_packed) < 1e-12, "varlen packing mismatch"
    assert np.all(np.isfinite(out)), "packed output must cover every row"
    print(f"packed kernel vs per-sample reference: max abs err = "
          f"{err(out, ref_packed):.3e}")
    print(f"unwritten padding rows (should be 0): "
          f"{int(np.sum(~np.isfinite(out)))}")
    print()
    print("note: the grid is sized by max_seqlen, so short samples leave whole")
    print("      blocks idle; the guard is what keeps them from writing garbage.")
    print()


def main() -> None:
    print(__doc__)
    case_full()
    case_causal()
    case_window_and_sink()
    case_varlen()
    print("=" * 74)
    print("All variants reproduce their exact reference on the CPU.")
    print("This validates the formulas, not any device kernel.")
    print("=" * 74)


if __name__ == "__main__":
    main()
