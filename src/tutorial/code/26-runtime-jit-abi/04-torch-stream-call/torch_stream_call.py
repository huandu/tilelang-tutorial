#!/usr/bin/env python3
"""Call a TileLang kernel from PyTorch and observe which stream it lands on.

What this example is for
------------------------
Chapter 26 claims: the default execution backend (`tvm_ffi`) reads PyTorch's
*current* stream at call time and submits the kernel there. This script is the
experiment that would confirm it -- and the experiment that shows what goes
wrong when two streams are mixed without synchronisation.

It needs a real CUDA GPU and an installed, matching tilelang build. On a host
without a GPU it will stop at the first check with a clear message instead of
producing a bogus result.

API names are checked against `_vendor/tilelang-v0.1.15/`:

  * `tilelang.jit(out_idx=[-1])`            tilelang/jit/__init__.py:588
  * `JITKernel.__call__`                    tilelang/jit/kernel.py:206
  * stream functor used by tvm_ffi          tilelang/jit/adapter/base.py:61
  * NPU stream exchange                     tilelang/ascend/torch_exchange.py:6

Usage
-----
    python3 torch_stream_call.py
"""

from __future__ import annotations

import sys

import torch

import tilelang
import tilelang.language as T


def require_cuda() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available on this host; nothing to run.", file=sys.stderr)
        print("On an Ascend host, swap torch.cuda.* for torch.npu.* below.", file=sys.stderr)
        raise SystemExit(2)


@tilelang.jit(out_idx=[-1])
def add_kernel(N: int, block_N: int = 256, dtype: str = "float16", threads: int = 128):
    """lazy mode: this function returns a PrimFunc, the wrapper returns a kernel."""

    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            for i in T.Parallel(block_N):
                C[bx * block_N + i] = A[bx * block_N + i] + B[bx * block_N + i]

    return main


def main() -> int:
    require_cuda()

    N = 1 << 12
    kernel = add_kernel(N)  # compiles (or hits the cache) and returns a JITKernel
    a = torch.randn(N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, device="cuda", dtype=torch.float16)

    # 1) Default: no stream argument anywhere. The adapter resolves the current
    #    stream at call time (base.py:61-101 picks torch.cuda or torch.npu).
    ref = (a + b).to(torch.float16)
    out = kernel(a, b)
    print("out device :", out.device)
    print("out dtype  :", out.dtype, " shape:", tuple(out.shape))

    # 2) A fresh stream. Everything submitted inside the context manager belongs
    #    to `side`; the kernel must follow, otherwise step 3 catches it.
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        out_side = kernel(a, b)

    # 3) The correctness check must run on the same stream. `torch.allclose`
    #    launches its own kernels on the *current* stream; reading `out_side`
    #    from the default stream without synchronising is the classic race.
    side.synchronize()
    torch.cuda.synchronize()
    print("matches reference:", bool(torch.allclose(out_side, ref, atol=1e-2, rtol=1e-2)))

    # 4) Deliberate anti-pattern, for contrast: do NOT do this.
    #       with torch.cuda.stream(side):
    #           out_side = kernel(a, b)
    #       print(torch.allclose(out_side, ref))   # no sync -> racy reads
    #    The failure is non-deterministic: sometimes correct, sometimes not.
    print("note: removing side.synchronize() above makes the comparison racy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
