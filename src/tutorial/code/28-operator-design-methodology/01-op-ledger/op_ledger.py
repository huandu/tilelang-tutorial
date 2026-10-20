#!/usr/bin/env python3
"""Operator performance ledger: the ``(flops, bytes)`` pair behind every design decision.

This script is the "step 1" tool of chapter 28.  Given an operator described as a
list of terms (each term = how much arithmetic, how many bytes), it prints

      flops, bytes, arithmetic intensity  ->  memory time, compute time, SOL time

and decides which roof the operator is standing under for a named machine profile.

The accounting convention is the one the book uses everywhere (see chapter 2 and
`../tilelang-corpus/12-operator-libraries/raw/TileOPs/docs/design/roofline.md`):

* every distinct input storage is read once, every output written once;
  intermediates never count;
* a basic arithmetic op is 1 FLOP; a transcendental (exp/log/tanh/sqrt/rsqrt/cos/
  sin...) is also 1 FLOP *on the ledger* -- not a hardware cost;
* one multiply-add inside a contraction is 2 FLOP.

Machine profiles come from two places, and the difference matters:

* `TileOPs/src/tileops/perf/profiles/*.yaml` -- theoretical peak times a
  one-time microbenchmark calibration, i.e. "what you can actually reach";
* the NVIDIA Hopper blog Table 1 -- the vendor's published peak, i.e. "what the
  marketing number says".

Both are printed so the gap is visible.

Exit code 0 on success.  Pure standard library: no GPU, no TileLang, no torch.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field


GB = 1e9
T = 1e12


# --------------------------------------------------------------------------- #
# Machine profiles
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Roof:
    """One compute roof: the unit an optimal implementation would use."""

    key: str
    peak_flops: float
    unit: str
    source: str


@dataclass(frozen=True)
class Machine:
    name: str
    bandwidth: float  # bytes/s, effective
    bandwidth_theoretical: float
    bandwidth_source: str
    roofs: dict[str, Roof]

    def balance_point(self, roof_key: str) -> float:
        return self.roofs[roof_key].peak_flops / self.bandwidth


# TileOPs profiles: effective = theoretical x calibration (docs/design/roofline.md 5.1).
H200_PROFILE = "tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/profiles/h200.yaml"
H100_BLOG = "tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md"

MACHINES: dict[str, Machine] = {
    "h200-calibrated": Machine(
        name="NVIDIA H200",
        bandwidth=4800e9 * 0.9380,
        bandwidth_theoretical=4800e9,
        bandwidth_source=f"{H200_PROFILE}:15-18 (envelope over copy/triad/read/write)",
        roofs={
            "cuda_core.fp32": Roof(
                "cuda_core.fp32", 67.0e12 * 0.8548, "CUDA-core FMA", f"{H200_PROFILE}:20-26"
            ),
            "tensor_core.fp16": Roof(
                "tensor_core.fp16", 989.5e12 * 0.7370, "HMMA fp16", f"{H200_PROFILE}:31-34"
            ),
            "tensor_core.bf16": Roof(
                "tensor_core.bf16", 989.5e12 * 0.7680, "HMMA bf16", f"{H200_PROFILE}:35-38"
            ),
            "tensor_core.fp8": Roof(
                "tensor_core.fp8", 1979.0e12 * 0.6977, "HMMA fp8", f"{H200_PROFILE}:48-50"
            ),
        },
    ),
    "h100-spec": Machine(
        name="NVIDIA H100 SXM5 (vendor spec, preliminary, not measured)",
        bandwidth=3.0e12,
        bandwidth_theoretical=3.0e12,
        bandwidth_source=f"{H100_BLOG}:334 ('over 3 TB/sec', HBM3)",
        roofs={
            "cuda_core.fp32": Roof("cuda_core.fp32", 60.0e12, "CUDA-core FMA", f"{H100_BLOG}:137"),
            "tensor_core.fp16": Roof("tensor_core.fp16", 1000.0e12, "HMMA fp16", f"{H100_BLOG}:141"),
            "tensor_core.bf16": Roof("tensor_core.bf16", 1000.0e12, "HMMA bf16", f"{H100_BLOG}:142"),
            "tensor_core.fp8": Roof("tensor_core.fp8", 2000.0e12, "HMMA fp8", f"{H100_BLOG}:143"),
        },
    ),
}


# --------------------------------------------------------------------------- #
# Operator description
# --------------------------------------------------------------------------- #


@dataclass
class Term:
    """One line of the ledger."""

    name: str
    flops: float = 0.0
    bytes: float = 0.0
    note: str = ""


@dataclass
class Op:
    name: str
    terms: list[Term] = field(default_factory=list)
    roof: str = "tensor_core.fp16"
    shape: dict[str, int] = field(default_factory=dict)

    @property
    def flops(self) -> float:
        return sum(t.flops for t in self.terms)

    @property
    def bytes(self) -> float:
        return sum(t.bytes for t in self.terms)


def nbytes(count: float, dtype: str) -> float:
    return count * DTYPE_BYTES[dtype]


DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
    "int32": 4,
}


# --------------------------------------------------------------------------- #
# A small library of operators, each built from that operator's own algebra
# --------------------------------------------------------------------------- #


def case_elementwise_chain(n: int, dtype: str = "fp16", fused: bool = True) -> Op:
    """y = sin(cos(x)) over n elements."""
    op = Op("elementwise chain (fused)" if fused else "elementwise chain (eager)", roof="cuda_core.fp32")
    rx = nbytes(n, dtype)
    op.shape = {"n": n, "dtype_bytes": DTYPE_BYTES[dtype]}
    if fused:
        op.terms.append(Term("read x", bytes=rx))
        op.terms.append(Term("write y", bytes=rx))
        op.terms.append(Term("cos+sin", flops=2.0 * n))
    else:
        op.terms.append(Term("read x", bytes=rx))
        op.terms.append(Term("write a", bytes=rx))
        op.terms.append(Term("cos", flops=1.0 * n))
        op.terms.append(Term("read a", bytes=rx))
        op.terms.append(Term("write b", bytes=rx))
        op.terms.append(Term("sin", flops=1.0 * n))
    return op


def case_rmsnorm(rows: int, hidden: int, dtype: str = "fp16", weight_in_l2: bool = True) -> Op:
    """RMSNorm: rsqrt(mean(x^2) + eps) * x * w, one pass over the row."""
    op = Op("RMSNorm", roof="cuda_core.fp32")
    op.shape = {"rows": rows, "hidden": hidden}
    op.terms.append(Term("read x", bytes=nbytes(rows * hidden, dtype)))
    op.terms.append(Term("write y", bytes=nbytes(rows * hidden, dtype)))
    op.terms.append(
        Term(
            "read w",
            bytes=0.0 if weight_in_l2 else nbytes(rows * hidden, dtype),
            note="0 when w stays resident; rows*hidden when it is re-read per row",
        )
    )
    op.terms.append(Term("sum of squares (FMA)", flops=2.0 * rows * hidden))
    op.terms.append(Term("rsqrt", flops=1.0 * rows))
    op.terms.append(Term("scale by rsqrt(mean)+eps", flops=2.0 * rows))
    op.terms.append(Term("multiply by w", flops=2.0 * rows * hidden))
    return op


def case_fused_norm_rope_quant(
    tokens: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    in_dtype: str = "bf16",
    out_dtype: str = "fp8",
) -> Op:
    """QK-Norm + RoPE + fp8 quantization, the case study of chapter 28 section 6.

    Rows are (tokens x heads) for q and (tokens x kv_heads) for k.  Norm weight
    and the RoPE table are counted once: they are shared by every head at a
    token, so a design that re-reads them per head is modelled by
    ``--rope-table-per-head`` in the CLI, not here.
    """
    op = Op("QK-Norm + RoPE + fp8 quant", roof="cuda_core.fp32")
    op.shape = {
        "tokens": tokens,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
    }
    rows = tokens * (q_heads + kv_heads)
    in_elems = rows * head_dim

    op.terms.append(Term("read q,k", bytes=nbytes(in_elems, in_dtype)))
    op.terms.append(Term("write q8,k8", bytes=nbytes(in_elems, out_dtype)))
    op.terms.append(
        Term("write per-row scales", bytes=nbytes(rows, "fp32"), note="one fp32 scale per row")
    )
    op.terms.append(
        Term(
            "read norm weights (q,k)",
            bytes=nbytes(2 * head_dim, in_dtype),
            note="resident in cache, counted once",
        )
    )
    op.terms.append(
        Term(
            "read RoPE table",
            bytes=nbytes(2 * tokens * head_dim, "fp32"),
            note="cos/sin at fp32, counted once for the whole layer",
        )
    )

    # Per row of head_dim elements.
    op.terms.append(Term("norm: sum of squares", flops=2.0 * rows * head_dim))
    op.terms.append(Term("norm: rsqrt + 2 scales", flops=3.0 * rows))
    op.terms.append(Term("norm: multiply by weight", flops=2.0 * rows * head_dim))
    op.terms.append(
        Term("RoPE: 4 mul + 2 add per pair", flops=6.0 * (head_dim // 2) * rows)
    )
    op.terms.append(Term("quant: absmax reduction", flops=1.0 * rows * head_dim))
    op.terms.append(Term("quant: scale + convert", flops=2.0 * rows * head_dim))
    return op


def case_gemm(m: int, n: int, k: int, dtype: str = "fp16", accum: str = "fp32") -> Op:
    op = Op(f"GEMM {m}x{n}x{k}", roof="tensor_core.fp16")
    op.shape = {"m": m, "n": n, "k": k}
    op.terms.append(Term("read A", bytes=nbytes(m * k, dtype)))
    op.terms.append(Term("read B", bytes=nbytes(k * n, dtype)))
    op.terms.append(Term("write C", bytes=nbytes(m * n, dtype)))
    op.terms.append(Term("2*M*N*K", flops=2.0 * m * n * k))
    _ = accum
    return op


def case_attention(batch_heads: int, sq: int, skv: int, head_dim: int, dtype: str = "fp16") -> Op:
    """FlashAttention-style fused attention, minimum traffic: read Q,K,V, write O."""
    op = Op(f"attention b*h={batch_heads} sq={sq} skv={skv}", roof="tensor_core.fp16")
    op.shape = {"bh": batch_heads, "sq": sq, "skv": skv, "d": head_dim}
    bh = batch_heads
    op.terms.append(Term("read Q", bytes=nbytes(bh * sq * head_dim, dtype)))
    op.terms.append(Term("read K", bytes=nbytes(bh * skv * head_dim, dtype)))
    op.terms.append(Term("read V", bytes=nbytes(bh * skv * head_dim, dtype)))
    op.terms.append(Term("write O", bytes=nbytes(bh * sq * head_dim, dtype)))
    # QK^T and PV, 2 FLOP per multiply-add.
    op.terms.append(Term("QK^T + PV", flops=4.0 * bh * sq * skv * head_dim))
    # softmax: scale, running max, subtract, exp, sum, and one divide per output.
    op.terms.append(
        Term(
            "softmax",
            flops=bh * sq * skv * 5.0 + bh * sq * head_dim * 1.0,
            note="5 per score + 1 divide per output element",
        )
    )
    return op


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def human(n: float) -> str:
    for unit, scale in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.3f} {unit}"
    return f"{n:.3f} "


def report(op: Op, machine: Machine, roof_key: str | None = None) -> dict:
    roof_key = roof_key or op.roof
    roof = machine.roofs[roof_key]
    flops, byts = op.flops, op.bytes
    ai = flops / byts if byts > 0 else math.inf
    mem_t = byts / machine.bandwidth
    cmp_t = flops / roof.peak_flops
    sol_t = max(mem_t, cmp_t)
    bound = "memory-bound" if mem_t > cmp_t else "compute-bound"
    if abs(mem_t - cmp_t) <= 1e-30:
        bound = "memory-bound (tie)"
    balance = machine.balance_point(roof_key)

    print("=" * 78)
    print(f"operator : {op.name}")
    print(f"machine  : {machine.name}")
    if op.shape:
        print("shape    : " + ", ".join(f"{k}={v}" for k, v in op.shape.items()))
    print("=" * 78)
    if op.terms:
        print(f"{'term':<32}{'flops':>14}{'bytes':>16}  note")
        for t in op.terms:
            print(f"{t.name:<32}{human(t.flops):>14}{human(t.bytes):>16}  {t.note}")
        print("-" * 78)
    print(f"{'TOTAL':<32}{human(flops):>14}{human(byts):>16}")
    print()
    print(f"arithmetic intensity : {ai:10.3f} FLOP/Byte")
    print(f"roof                 : {roof.key:<18} {roof.peak_flops / T:8.2f} TFLOP/s ({roof.unit})")
    print(f"                       source: {roof.source}")
    print(f"bandwidth            : {machine.bandwidth / GB:10.2f} GB/s (effective)")
    print(f"                       source: {machine.bandwidth_source}")
    print(f"balance point        : {balance:10.3f} FLOP/Byte")
    print()
    print(f"memory time          : {mem_t * 1e6:12.3f} us")
    print(f"compute time         : {cmp_t * 1e6:12.3f} us")
    print(f"SOL time (max)       : {sol_t * 1e6:12.3f} us   -> {bound}")
    print()
    print("the other roof, for contrast:")
    for key, r in machine.roofs.items():
        if key == roof_key:
            continue
        bp = machine.balance_point(key)
        verdict = "memory-bound under this roof too" if ai < bp else "COMPUTE-bound under this roof"
        print(f"  {key:<18} {r.peak_flops / T:8.2f} TFLOP/s  balance {bp:7.2f}  -> {verdict}")
    print()

    return {
        "operator": op.name,
        "machine": machine.name,
        "roof": roof_key,
        "flops": flops,
        "bytes": byts,
        "arithmetic_intensity": ai,
        "balance_point": balance,
        "memory_time_us": mem_t * 1e6,
        "compute_time_us": cmp_t * 1e6,
        "sol_time_us": sol_t * 1e6,
        "bound": bound,
    }


def sweep(machine: Machine, roof_key: str) -> None:
    print(f"machine {machine.name}  roof {roof_key}")
    balance = machine.balance_point(roof_key)
    print(f"balance point at this roof: {balance:.2f} FLOP/Byte\n")
    rows = [
        ("gemm 4096^3 fp16", case_gemm(4096, 4096, 4096)),
        ("gemm 512^3 fp16", case_gemm(512, 512, 512)),
        ("gemm 128^3 fp16", case_gemm(128, 128, 128)),
        ("attention 32x4096x4096 d128", case_attention(32, 4096, 4096, 128)),
        ("attention 32x1x8192 d128", case_attention(32, 1, 8192, 128)),
        ("rmsnorm 8192x4096", case_rmsnorm(8192, 4096)),
        ("elementwise chain n=16.8M fused", case_elementwise_chain(16 * 1024 * 1024)),
        ("elementwise chain n=16.8M eager", case_elementwise_chain(16 * 1024 * 1024, fused=False)),
        ("qk-norm+rope+quant 4096 tok", case_fused_norm_rope_quant(4096, 32, 8, 128)),
    ]
    print(f"{'operator':<36}{'AI':>10}{'bound':>16}{'mem us':>12}{'cmp us':>12}{'SOL us':>12}")
    for label, op in rows:
        roof = machine.roofs[roof_key]
        ai = op.flops / op.bytes
        mem_t = op.bytes / machine.bandwidth * 1e6
        cmp_t = op.flops / roof.peak_flops * 1e6
        print(
            f"{label:<36}{ai:>10.2f}"
            f"{('memory' if mem_t > cmp_t else 'compute'):>16}"
            f"{mem_t:>12.3f}{cmp_t:>12.3f}{max(mem_t, cmp_t):>12.3f}"
        )
    print()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", default="h200-calibrated", choices=sorted(MACHINES))
    p.add_argument(
        "--case",
        default="fused-norm-rope-quant",
        choices=[
            "elementwise-fused",
            "elementwise-eager",
            "rmsnorm",
            "fused-norm-rope-quant",
            "gemm",
            "attention",
        ],
    )
    p.add_argument("--roof", default=None, help="override the compute roof key")
    p.add_argument("--sweep", action="store_true", help="print the operator table instead")
    p.add_argument("--json", action="store_true", help="emit the result as JSON")
    # shape knobs
    p.add_argument("--rows", type=int, default=8192)
    p.add_argument("--hidden", type=int, default=4096)
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--q-heads", type=int, default=32)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--k", type=int, default=4096)
    p.add_argument("--sq", type=int, default=4096)
    p.add_argument("--skv", type=int, default=4096)
    p.add_argument("--batch-heads", type=int, default=32)
    p.add_argument("--n-elem", type=int, default=16 * 1024 * 1024)
    args = p.parse_args()

    machine = MACHINES[args.machine]
    if args.sweep:
        sweep(machine, args.roof or "tensor_core.fp16")
        return 0

    if args.case == "elementwise-fused":
        op = case_elementwise_chain(args.n_elem, fused=True)
    elif args.case == "elementwise-eager":
        op = case_elementwise_chain(args.n_elem, fused=False)
    elif args.case == "rmsnorm":
        op = case_rmsnorm(args.rows, args.hidden)
    elif args.case == "fused-norm-rope-quant":
        op = case_fused_norm_rope_quant(args.tokens, args.q_heads, args.kv_heads, args.head_dim)
    elif args.case == "gemm":
        op = case_gemm(args.m, args.n, args.k)
    elif args.case == "attention":
        op = case_attention(args.batch_heads, args.sq, args.skv, args.head_dim)
    else:  # pragma: no cover - argparse already restricts this
        raise SystemExit(f"unknown case {args.case}")

    if args.roof:
        op.roof = args.roof
    result = report(op, machine, args.roof)
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
