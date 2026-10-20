"""Excerpt of a real CuTe DSL kernel (GEMV + RMSNorm prologue).

Source (verified, read-only snapshot):
  ../tilelang-corpus/07-tilelang-ascend-adapter/raw/TileFoundry/
      examples/minicpm3_4b-cutedsl/kernels/cute_gemv.py:29-113

This is an EXCERPT, not a standalone program: helper modules
(`._common`) and the surrounding kernel/launch code are omitted.

Why it is in this chapter (section 3.3): it shows that CuTe DSL is a
thread/instruction-level model, not a tile-level one. You can see `tidx`,
`cute.arch.barrier()`, `cute.autovec_copy()` and `cute.make_fragment()` -- all
of which an author writing TileLang would not write.

Verification level: static only. Not run (no NVIDIA GPU), and dependencies
(`nvidia-cutlass-dsl`, `cuda.bindings`) are not installed here.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.cute.math as cmath

# 8 bf16 = one 128-bit load per lane; 32 lanes = a 512-byte contiguous slab.
VEC = 8
BF16 = cutlass.BFloat16
F32 = cutlass.Float32


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@cute.jit
def _block_sum(
    value: cutlass.Float32,
    tidx: cutlass.Int32,
    warps: cutlass.Constexpr,
    scratch: cute.Tensor,
) -> cutlass.Float32:
    v = cute.arch.warp_reduction_sum(value)
    if tidx % 32 == 0:
        scratch[tidx // 32] = v
    cute.arch.barrier()
    total = F32(0.0)
    for i in cutlass.range_constexpr(warps):
        total += scratch[i]
    return total


@cute.jit
def _load_and_normalise(
    x: cute.Tensor,
    gamma: cute.Tensor,
    smem: cute.Tensor,
    scratch: cute.Tensor,
    tidx: cutlass.Int32,
    k: cutlass.Constexpr,
    eps: cutlass.Constexpr,
    nt: cutlass.Constexpr,
) -> None:
    """Put rms_norm(x, gamma) into *smem* as bf16."""
    nch = k // VEC
    xv = cute.tiled_divide(x, (VEC,))
    gv = cute.tiled_divide(gamma, (VEC,))
    sv = cute.tiled_divide(smem, (VEC,))
    frag = cute.make_fragment(VEC, BF16)
    fg = cute.make_fragment(VEC, BF16)

    acc = F32(0.0)
    for i in cutlass.range_constexpr(cdiv(nch, nt)):
        j = i * nt + tidx
        if j < nch:
            cute.autovec_copy(xv[None, j], frag)
            for e in cutlass.range_constexpr(VEC):
                f = frag[e].to(F32)
                acc += f * f
    total = _block_sum(acc, tidx, nt // 32, scratch)
    inv = F32(cmath.rsqrt(total / F32(float(k)) + F32(eps)))

    for i in cutlass.range_constexpr(cdiv(nch, nt)):
        j = i * nt + tidx
        if j < nch:
            cute.autovec_copy(xv[None, j], frag)
            cute.autovec_copy(gv[None, j], fg)
            for e in cutlass.range_constexpr(VEC):
                landed = (frag[e].to(F32) * inv).to(BF16)
                frag[e] = (landed.to(F32) * fg[e].to(F32)).to(BF16)
            cute.autovec_copy(frag, sv[None, j])
    cute.arch.barrier()
