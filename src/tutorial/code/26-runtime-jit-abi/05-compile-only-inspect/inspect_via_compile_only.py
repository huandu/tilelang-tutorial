#!/usr/bin/env python3
"""Inspect what TileLang generates for a contract kernel -- without a GPU.

Chapter 26 separates the compile-time ABI decision from the runtime adapter.
This example walks the *compile* half on a CPU-only host using the tool that
ships with TileLang itself: `python -m tilelang.tools.compile_only`
(`_vendor/tilelang-v0.1.15/tilelang/tools/compile_only.py`).

It shows four things, in order:

  1. the TIR the kernel elaborates to (`get_tir`, tilelang/jit/__init__.py:379);
  2. the generated *device* source, produced by lowering with
     `enable_device_compile=False` -- no driver, no GPU, no cubin;
  3. which attributes on the elaborated PrimFunc already exist, and which two
     are decided later (at JITKernel construction) rather than here;
  4. the cache-key fields this exact call would hash
     (tilelang/cache/kernel_cache.py:269-280).

Scope note, to avoid a common misreading: `compile_only` stops before host
code generation, so the text it emits is the device half. The packed host entry
`(self_handle, args, num_args, result)` is produced by the *host* half
(`MakePackedAPI` plus `host_codegen`, tilelang/engine/lower.py:65-89), which
only exists once `enable_host_codegen=True`. On a real device run you read it
with `JITKernel.get_host_source()` (tilelang/jit/kernel.py:545).

Everything here depends on a real `tilelang` import, so it cannot run on this
authoring host (no tilelang/torch installed). It is written to be run on a
normal CPU-only Linux box:

    pip install tilelang==0.1.15
    python3 inspect_via_compile_only.py

Usage
-----
    python3 inspect_via_compile_only.py --target c
    python3 inspect_via_compile_only.py --target cuda   # needs a CUDA-enabled wheel
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_module():
    import tilelang
    import tilelang.language as T

    @tilelang.jit(out_idx=[-1])
    def scale_add(N: int, alpha: float, block_N: int = 256, dtype: str = "float32"):
        """out[i] = a[i] * alpha + b[i]; `out` is the only output."""

        @T.prim_func
        def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N,), dtype),
            Out: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), threads=128) as bx:
                for i in T.Parallel(block_N):
                    Out[bx * block_N + i] = A[bx * block_N + i] * alpha + B[bx * block_N + i]

        return main

    return tilelang, scale_add


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="c", help="compile target (default: c, CPU C)")
    parser.add_argument("--output", default="out.c", help="where to write the generated source")
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    try:
        import tilelang  # noqa: F401
    except ModuleNotFoundError:
        print("tilelang is not installed; this example is static-only on this host.", file=sys.stderr)
        return 2

    tilelang, scale_add = build_module()

    # 1) Elaborate to TIR. `get_tir` is the same entry the JIT wrapper uses
    #    (tilelang/jit/__init__.py:379) but stops before lowering.
    prim_func = scale_add.get_tir(args.N, args.alpha)
    print("=== TIR ===")
    print(prim_func.script())

    # 2) Compile without a device: lower() with enable_device_compile=False,
    #    then write the device source (see the scope note in the docstring).
    from tilelang.tools.compile_only import compile_kernel_source

    source = compile_kernel_source(prim_func, target=args.target)
    Path(args.output).write_text(source)
    print(f"=== generated device source ({args.target}) -> {args.output} ===")
    for line in source.splitlines():
        if "int32_t" in line or "__global__" in line or "void " in line:
            print(f"  {line}")

    # 3) Attributes that exist *now*, versus the two that are decided later.
    attrs = dict(prim_func.attrs) if prim_func.attrs else {}
    print("=== attributes on the elaborated PrimFunc ===")
    print(" ", json.dumps({k: str(v) for k, v in attrs.items()}, indent=2))
    print("  expected here: global_symbol, tir.is_entry_func (+ compiler hints)")
    print("  NOT expected here in lazy mode:")
    print("    tilelang_out_idx                  -- added on a derived copy at")
    print("        JITKernel construction (kernel.py:236 -> abi.py:39-74)")
    print("    tilelang_callee_allocated_outputs -- same place, and only when the")
    print("        resolved execution backend declares the capability")
    print("        (abi.py:63-73; execution_backend.py:29)")
    print("  In eager mode (T.empty outputs) tilelang_out_idx *is* set on the")
    print("  elaborated function itself (language/eager/builder.py:1269).")

    # 4) The fields that go into the on-disk cache key.
    print("=== cache key inputs (kernel_cache.py:269-280) ===")
    print("  func.script(show_meta=True) :  sha256 of the TIR text above")
    print(f"  out_idx                     :  {scale_add.out_idx}")
    print(f"  target                      :  {args.target}")
    print(f"  args (repr)                 :  {(args.N, args.alpha)}")
    print(f"  execution_backend           :  {scale_add.execution_backend or 'auto'}")
    print(f"  pass_configs                :  {scale_add.pass_configs}")
    print(f"  compile_flags               :  {scale_add.compile_flags}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
