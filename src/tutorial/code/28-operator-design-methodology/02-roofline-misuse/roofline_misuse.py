#!/usr/bin/env python3
"""Eight ways to read a roofline wrong, each with a number attached.

Roofline is a bound, and like every bound it is only as good as the two numbers
you put into it.  This script takes one real operator -- QK-Norm + RoPE + fp8
quantization, the case study of chapter 28 -- and the machine profile from
`TileOPs/src/tileops/perf/profiles/h200.yaml`, and shows what each mistake does
to the answer.

Every mistake is stated as a pair: what the wrong reading says, and what the
honest reading says, on the same operator.  Nothing is measured.

Pure standard library.  Exit code 0 on success.
"""

from __future__ import annotations

import argparse


GB = 1e9
T = 1e12

# TileOPs h200.yaml: effective = theoretical * calibration
HBM_THEORETICAL = 4800e9
HBM_CALIBRATION = 0.9380
CUDA_CORE_FP32 = 67.0e12 * 0.8548
TENSOR_FP16 = 989.5e12 * 0.7370
TENSOR_BF16 = 989.5e12 * 0.7680
TENSOR_FP8 = 1979.0e12 * 0.6977
# vendor spec sheet, same file
TENSOR_FP16_SPEC = 989.5e12

# Ascend VALU throughputs (estimate_latency.h): ops/cycle, not FLOP/s
ASCEND_MUL_TP = 51
ASCEND_SPECIAL_FP = 8

# the case study, one token-block: see code/06 for the full ledger
TOKENS, Q_HEADS, KV_HEADS, HEAD_DIM = 4096, 32, 8, 128
ROWS = TOKENS * (Q_HEADS + KV_HEADS)
ELEMS = ROWS * HEAD_DIM
FLOP_PER_ROW = 1155.0  # norm 385 + rope 384 + quant 386
FLOPS = FLOP_PER_ROW * ROWS
PAYLOAD = ELEMS * 2 + ELEMS * 1 + ROWS * 4  # bf16 in, fp8 out, fp32 scales
LOOKUP = 2 * HEAD_DIM * 2 + 2 * TOKENS * HEAD_DIM * 4
BYTES = PAYLOAD + LOOKUP


def line(title: str) -> None:
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)


def m1_spec_instead_of_attainable(bw: float) -> None:
    line("M1  using the datasheet peak instead of the calibrated sustained rate")
    t_spec = FLOPS / TENSOR_FP16_SPEC * 1e6
    t_cal = FLOPS / TENSOR_FP16 * 1e6
    print(f"  datasheet   fp16 tensor {TENSOR_FP16_SPEC / T:8.1f} TFLOP/s -> compute time {t_spec:8.3f} us")
    print(f"  calibrated  fp16 tensor {TENSOR_FP16 / T:8.1f} TFLOP/s -> compute time {t_cal:8.3f} us")
    print(f"  understated by {TENSOR_FP16_SPEC / TENSOR_FP16:.2f}x")
    print("  The calibration is not a fudge factor: on H200 the fp16 tensor rate holds at")
    print("  ~0.67 of peak under a sustained 700 W cap and reaches 0.74 in burst windows")
    print("  (h200.yaml:31-34). Whichever one your kernel sees depends on how long it runs.")
    print(f"  same lesson on the memory side: {HBM_THEORETICAL / GB:.0f} GB/s spec"
          f" -> {HBM_THEORETICAL * HBM_CALIBRATION / GB:.0f} GB/s envelope")


def m2_wrong_roof() -> None:
    line("M2  pricing an elementwise operator against the tensor-core roof")
    ai = FLOPS / BYTES
    print(f"  operator: {FLOPS / 1e6:.1f} MFLOP over {BYTES / 1e6:.2f} MB -> AI = {ai:.2f} FLOP/Byte")
    for name, peak in (("tensor_core.fp16", TENSOR_FP16),
                       ("tensor_core.bf16", TENSOR_BF16),
                       ("tensor_core.fp8", TENSOR_FP8),
                       ("cuda_core.fp32", CUDA_CORE_FP32)):
        bp = peak / (HBM_THEORETICAL * HBM_CALIBRATION)
        print(f"  {name:<18} balance {bp:7.2f} FLOP/Byte   -> "
              f"{'memory-bound' if ai < bp else 'COMPUTE-bound'}")
    print("  The roof key must name the unit an *optimal* implementation would use")
    print("  (TileOPs roofline.md:63-70). A norm/RoPE/quantize kernel runs on CUDA cores,")
    print("  not on tensor cores, so only the last row prices it honestly.")
    print("  Note what the wrong roof does here: it moves the balance point by 12.7x.")
    print("  With a heavier per-element body (a long reduction, an activation chain)")
    print("  that gap is enough to flip the verdict and send you tuning the wrong thing.")


def m3_roof_from_dtype() -> None:
    line("M3  declaring the roof from the input dtype instead of from the algorithm")
    print("  an fp8 attention that consumes this kernel's output would be priced, wrongly,")
    print("  against tensor_core.fp8 -- but its contraction may run in bf16 or fp16,")
    print("  and its softmax runs on CUDA cores either way.")
    print(f"  balance points: fp8 {TENSOR_FP8 / (HBM_THEORETICAL * HBM_CALIBRATION):.2f},"
          f" fp16 {TENSOR_FP16 / (HBM_THEORETICAL * HBM_CALIBRATION):.2f},"
          f" cuda-core {CUDA_CORE_FP32 / (HBM_THEORETICAL * HBM_CALIBRATION):.2f}")
    print("  Same kernel, three different ceilings -- and the roofline tool cannot tell")
    print("  you which one you meant. TileSight's authors name this failure mode directly:")
    print("  existing tooling offers 'coarse roofline bounds' (arXiv 2607.22432).")


def m4_wrong_ai_for_fused_op() -> None:
    line("M4  computing AI on one stage and applying it to the fused kernel")
    eager = 2 * ELEMS * 2 + 2 * ELEMS * 1 + ROWS * 4 + 2 * TOKENS * HEAD_DIM * 4 * 2
    fused = BYTES
    print(f"  as two kernels : {eager / 1e6:8.2f} MB, AI = {FLOPS / eager:.2f}")
    print(f"  as one kernel  : {fused / 1e6:8.2f} MB, AI = {FLOPS / fused:.2f}")
    print(f"  using the split AI for the fused kernel understates its arithmetic intensity"
          f" by {(FLOPS / fused) / (FLOPS / eager):.2f}x,")
    print("  which biases every later verdict toward 'memory-bound'.")
    print("  The ledger is defined per *call*: rules 1-4 of the convention count each")
    print("  input once and each output once, and intermediates never. Change the number")
    print("  of calls and you change the arithmetic intensity, with the same kernel body.")


def m5_transcendental() -> None:
    line("M5  charging a transcendental the same as a multiply")
    print("  the sharpest case is not this operator, it is the softmax in the attention")
    print("  kernel downstream of it. Take rows of 8192 scores:")
    scores = 8192
    exp_ops = scores  # one exp per score
    other_ops = 4 * scores  # scale, running max, subtract, running sum
    ledger_flops = exp_ops + other_ops
    print(f"  ledger says  : {ledger_flops / 1e3:.1f} kFLOP per row at 1 FLOP each")
    print(f"  of which {exp_ops / ledger_flops * 100:.0f}% are exp, charged as 1 FLOP")
    print("  On the Ascend VALU the same op is charged at ops/cycle:")
    print(f"      mul/add/min-max/cmp/logic  {ASCEND_MUL_TP:>3} ops/cycle")
    print(f"      div/mod                    {13:>3} ops/cycle")
    print(f"      special_func (rsqrt, exp)  {ASCEND_SPECIAL_FP:>3} ops/cycle")
    print(f"  a transcendental is {ASCEND_MUL_TP / ASCEND_SPECIAL_FP:.2f}x a multiply per op.")
    print("  Weight each op by its own pipe and the 'compute time' moves:")
    weighted_mul = other_ops / ASCEND_MUL_TP
    weighted_sf = exp_ops / ASCEND_SPECIAL_FP
    unweighted = ledger_flops / ASCEND_MUL_TP
    print(f"      unweighted: {unweighted / 1e3:8.1f} k cycles  ({ledger_flops} ops / {ASCEND_MUL_TP})")
    print(f"      per-pipe  : {(weighted_mul + weighted_sf) / 1e3:8.1f} k cycles"
          f"   ({(weighted_mul + weighted_sf) / unweighted:.3f}x)")
    print("  The comparison that matters is not 'memory vs compute' but 'which pipe':")
    print(f"      the exp pipe alone is {weighted_sf / 1e3:8.1f} k cycles"
          f" against {weighted_mul / 1e3:8.1f} k for everything else.")
    print("  TileOPs states the same limit: the metric 'does not certify an SFU-bound")
    print("  kernel as at its limit' (roofline.md:34). The NVIDIA-side docs in this")
    print("  corpus give no equivalent ops/cycle table; do not invent one.")


def m6_forget_the_lookups() -> None:
    line("M6  counting only the payload")
    print(f"  payload only      : {PAYLOAD / 1e6:8.2f} MB")
    print(f"  payload + lookups : {BYTES / 1e6:8.2f} MB"
          f"   (lookups are {LOOKUP / BYTES * 100:.1f}% of the ideal traffic)")
    print(f"  SOL time          : {PAYLOAD / (HBM_THEORETICAL * HBM_CALIBRATION) * 1e6:8.3f} us"
          f" -> {BYTES / (HBM_THEORETICAL * HBM_CALIBRATION) * 1e6:8.3f} us")
    print("  This is the *ideal* count, i.e. lookups charged once. code/06 shows what")
    print("  happens when a design re-reads them once per row instead of once per token:")
    print("  the same operator moves 4.65x the ideal bytes, and 79.8% of that is lookups.")


def m7_latency_floor() -> None:
    line("M7  reading a numerical result off a row where the model has no traction")
    bw = HBM_THEORETICAL * HBM_CALIBRATION
    cases = [
        ("whole prefill chunk, 4096 tokens", BYTES),
        ("one decode step, 1 token", (40 * HEAD_DIM) * 2 + (40 * HEAD_DIM) * 1 + 40 * 4),
    ]
    for label, byts in cases:
        print(f"  {label:<34}{byts:>10} B -> SOL {byts / bw * 1e6:9.3f} us")
    print("  TileOPs labels such rows 'latency-bound': when sol_time and the measured")
    print("  time are both below the latency floor, launch overhead dominates and")
    print("  'the model has no traction' (roofline.md:40-41).")
    print(f"  The corollary is a design rule, not a measurement: the same kernel body is")
    print(f"  priced sensibly at prefill shapes and has no traction at decode shapes.")
    print(f"  Below the latency floor, efficiency is the wrong metric -- the number of")
    print(f"  launches and the smallest grid that still fills the device are the metrics.")


def m8_anomaly_is_not_a_win() -> None:
    line("M8  reading a >100% efficiency as a great kernel")
    print("  TileOPs' verdict bands (roofline.md:156-163):")
    print("      at ceiling : >= 80%   the tool is satisfied")
    print("      anomaly    : > 105%   above the achievable ceiling -- a formula or")
    print("                            calibration error, excluded from 'at ceiling'")
    print("  and every row's implied rates are checked against the *theoretical* ceilings:")
    print(f"      bytes/time  <= {HBM_THEORETICAL / GB:.0f} GB/s")
    print(f"      flops/time  <= the declared roof's theoretical peak")
    print("  A breach is physically impossible, so it is a bug in your accounting --")
    print("  most often an under-counted byte total, e.g. forgetting the scale tensor")
    print(f"  ({ROWS * 4 / 1e6:.3f} MB here) or an intermediate that does reach DRAM.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", type=int, default=0, help="run a single misuse, 1..8")
    args = p.parse_args()
    bw = HBM_THEORETICAL * HBM_CALIBRATION

    print("operator : QK-Norm + RoPE + fp8 quant")
    print(f"           {ROWS} rows of {HEAD_DIM} elements,"
          f" {FLOPS / 1e6:.1f} MFLOP, {BYTES / 1e6:.2f} MB ideal")
    print(f"machine  : NVIDIA H200, {bw / GB:.1f} GB/s effective")

    demos = [
        lambda: m1_spec_instead_of_attainable(bw),
        m2_wrong_roof,
        m3_roof_from_dtype,
        m4_wrong_ai_for_fused_op,
        m5_transcendental,
        m6_forget_the_lookups,
        m7_latency_floor,
        m8_anomaly_is_not_a_win,
    ]
    if args.only:
        demos[args.only - 1]()
    else:
        for d in demos:
            d()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
