#!/usr/bin/env python3
"""Walk a real kernel pass-by-pass with IR Lower Trace.

This is the "do it yourself" script for the chapter: it compiles one small
elementwise kernel, records the IR before/after every pass, and prints a
compact per-pass summary (name + insertions/deletions) so you can see exactly
where each load-bearing pass fires.

It mirrors the official tutorial
``tilelang/tools/lower_trace`` (docs/tools/lower_trace.md) and uses the
programmatic API instead of the environment variable so the output is easy to
filter with grep.

Requirements (this machine does NOT satisfy them):
    * a tilelang install of the locked version
    * an NVIDIA GPU + CUDA toolkit (the kernel is written for target="cuda")
    * the matching torch build

Because the authoring machine is macOS/Apple-Silicon without a GPU, this script
is *not* claimed to have been executed end-to-end. See README.md for the
verification level and the exact commands to run on a GPU host.

Usage on a GPU host:
    python3 lower_trace_walkthrough.py --target cuda
    python3 lower_trace_walkthrough.py --target cuda --only LayoutInference,LowerTileOp
    python3 lower_trace_walkthrough.py --codegen-only
"""

from __future__ import annotations

import argparse


def build_kernel(threads: int, num_per_thread: int, dtype: str):
    import tilelang
    import tilelang.language as T

    def elementwise_add(N):
        @T.prim_func
        def main(
            A: T.Tensor((N,), dtype),
            B: T.Tensor((N,), dtype),
            C: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, threads * num_per_thread), threads=threads) as bx:
                for i, j in T.Parallel(threads, num_per_thread):
                    off = (bx * threads + i) * num_per_thread
                    C[off + j] = A[off + j] + B[off + j]

        return main

    return elementwise_add


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="cuda")
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--num-per-thread", type=int, default=8)
    parser.add_argument("--only", default="", help="comma-separated pass names to keep")
    parser.add_argument("--mode", default="terminal", choices=["terminal", "html", "both"])
    parser.add_argument("--codegen-only", action="store_true",
                        help="also print the generated CUDA source")
    args = parser.parse_args()

    import tilelang
    from tilelang.tools import lower_trace as lt

    dtype = "float16"
    elementwise_add = build_kernel(args.threads, args.num_per_thread, dtype)
    func = elementwise_add(args.n)

    # lower_trace(mod, *passes) records before/after IR for the given passes and
    # returns a list of dicts with keys name / before_script / after_script /
    # diff_lines / insertions / deletions / changed.
    passes = [
        tilelang.transform.PipelinePlanning(),
        tilelang.transform.InjectSoftwarePipeline(),
        tilelang.transform.LayoutInference(),
        tilelang.transform.LowerTileOp(),
        tilelang.transform.StorageRewrite(),
        tilelang.transform.SplitHostDevice(),
        tilelang.transform.ThreadSync("shared"),
        tilelang.transform.MakePackedAPI(),
    ]
    records = lt.lower_trace(func, *passes, mode=args.mode)

    wanted = {name for name in args.only.split(",") if name}
    for rec in records:
        name = str(rec.get("name", ""))
        if wanted and not any(w in name for w in wanted):
            continue
        print(
            f"{name:<44} +{int(rec.get('insertions', 0)):<4} "
            f"-{int(rec.get('deletions', 0)):<4} changed={bool(rec.get('changed'))}"
        )

    if args.codegen_only:
        kernel = tilelang.compile(func, target=args.target)
        print("\n===== generated source =====")
        print(kernel.get_kernel_source())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
