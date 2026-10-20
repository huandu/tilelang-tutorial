#!/usr/bin/env python3
"""Change ``num_stages`` and watch the IR and the generated CUDA change.

This is the hands-on experiment of chapter 22.  The interesting half of it
**does not need a GPU**: ``tilelang.tools.compile_only.compile_kernel_source``
lowers the kernel with ``enable_device_compile=False`` and returns the CUDA
source, so the pipeline, multi-buffer and synchronization decisions are all
visible on a machine with only the TileLang wheel
(``tilelang/tools/compile_only.py:169-200``).

What the script prints, per ``num_stages``:

* the ``(stage, order)`` arrays that ``PipelinePlanning`` wrote onto the loop
  (``src/transform/pipeline_planning.cc:1293-1305``) -- this is where the
  version count comes from;
* the leading dimension of every ``__shared__`` declaration in the generated
  CUDA, i.e. the materialized version dimension
  (``src/transform/inject_pipeline.cc:1551-1560``);
* the number of ``__syncthreads()``, ``cp.async.*`` and ``mbarrier``
  occurrences, i.e. what the synchronization passes decided to insert.

Add ``--dump-dir DIR`` to write each variant's CUDA source out for diffing,
and set ``TL_LOWER_TRACE=terminal`` to see the pass-by-pass IR diff produced by
the real lowering pipeline.

Requires: TileLang v0.1.15 (``pip install tilelang==0.1.15``) and, for the CUDA
source path, a wheel that ships the CUDA codegen FFI.  No GPU and no nvcc are
needed for anything this script does.

Usage:
    python3 lower_num_stages.py
    python3 lower_num_stages.py --num-stages 1 2 3 4 --dump-dir /tmp/tl22
    TL_LOWER_TRACE=terminal python3 lower_num_stages.py --num-stages 3
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path


def build_module(num_stages: int, shape: tuple[int, int, int],
                 tile: tuple[int, int, int], threads: int):
    """Return ``(tir, target)`` for one GEMM variant.

    The body is the canonical whole-pipeline GEMM: two global->shared copies in
    stage 0 and one ``T.gemm`` in the last stage, which is exactly the shape
    ``test_pipeline_planning_before_after_copy_gemm_num_stages_plan`` asserts.
    """

    import tilelang
    import tilelang.language as T
    from tilelang import tvm as tvm
    from tilelang.backend.target import determine_target

    m, n, k = shape
    block_m, block_n, block_k = tile
    dtype = "float16"
    accum_dtype = "float32"

    @tilelang.jit
    def matmul(M: int, N: int, K: int, bm: int, bn: int, bk: int, stages: int,
               thread_num: int):
        @T.prim_func
        def kernel(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm),
                          threads=thread_num) as (bx, by):
                A_shared = T.alloc_shared((bm, bk), dtype)
                B_shared = T.alloc_shared((bn, bk), dtype)
                C_local = T.alloc_fragment((bm, bn), accum_dtype)

                T.clear(C_local)
                for ko in T.Pipelined(T.ceildiv(K, bk), num_stages=stages):
                    T.copy(A[by * bm, ko * bk], A_shared)
                    T.copy(B[bx * bn, ko * bk], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)

                T.copy(C_local, C[by * bm, bx * bn])

        return kernel

    tir = matmul.get_tir(M=m, N=n, K=k, bm=block_m, bn=block_n, bk=block_k,
                         stages=num_stages, thread_num=threads)
    try:
        target = tvm.target.Target(determine_target("auto"))
    except Exception:  # pragma: no cover - depends on the local wheel
        target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    return tir, target


def pipeline_annotations(tir, target) -> list[dict[str, list[int]]]:
    """Run only ``PipelinePlanning`` and read back the loop annotations."""

    import tilelang
    from tilelang import tvm as tvm
    from tvm.tirx.stmt_functor import post_order_visit

    mod = tvm.IRModule.from_expr(tir.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.IfStmtBinding()(mod)
    mod = tilelang.transform.PipelinePlanning()(mod)

    found: list[dict[str, list[int]]] = []

    def visit(node) -> None:
        if not isinstance(node, tvm.tirx.For):
            return
        stage = node.annotations.get("software_pipeline_stage")
        order = node.annotations.get("software_pipeline_order")
        if stage is None:
            return
        entry = {
            "stage": [int(s) for s in stage],
            "order": [int(o) for o in order] if order is not None else [],
        }
        depth = node.annotations.get("tl_pipelined_num_stages")
        if depth is not None:
            entry["tl_pipelined_num_stages"] = [int(depth)]
        found.append(entry)

    post_order_visit(mod["main"].body, visit)
    return found


def lower_to_cuda(tir, target) -> str:
    from tilelang.tools.compile_only import compile_kernel_source

    return compile_kernel_source(tir, target=target)


def static_facts(source: str) -> dict[str, object]:
    """Count the synchronization and allocation facts worth eyeballing."""

    counters = Counter()
    for token in ("__syncthreads", "cp.async.commit_group", "cp.async.wait_group",
                  "cp.async.wait_all", "cp.async", "mbarrier",
                  "ptx_arrive_barrier", "tma", "bar.sync"):
        counters[token] = source.count(token)

    shared: list[str] = []
    for match in re.finditer(r"__shared__\s+([A-Za-z_0-9]+)\s+([A-Za-z_0-9]+)"
                             r"\s*((?:\[[^\]]*\])*)", source):
        dtype, name, dims = match.group(1), match.group(2), match.group(3)
        shared.append(f"{dtype} {name}{dims}".strip())

    return {"counters": counters, "shared": sorted(set(shared))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-stages", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--block-M", type=int, default=128)
    parser.add_argument("--block-N", type=int, default=128)
    parser.add_argument("--block-K", type=int, default=64)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--dump-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.dump_dir is not None:
        args.dump_dir.mkdir(parents=True, exist_ok=True)

    for num_stages in args.num_stages:
        print(f"===== num_stages = {num_stages} =====")
        try:
            tir, target = build_module(
                num_stages, (args.M, args.N, args.K),
                (args.block_M, args.block_N, args.block_K), args.threads)
        except ImportError as exc:  # pragma: no cover - environment dependent
            print(f"  cannot import TileLang ({exc}); install tilelang==0.1.15")
            return

        annotations = pipeline_annotations(tir, target)
        for entry in annotations:
            print(f"  software_pipeline_stage = {entry['stage']}")
            print(f"  software_pipeline_order = {entry['order']}")
            if "tl_pipelined_num_stages" in entry:
                print(f"  tl_pipelined_num_stages = "
                      f"{entry['tl_pipelined_num_stages'][0]}")
        if not annotations:
            print("  (no pipeline annotations -- num_stages=0 disables "
                  "pipelining entirely)")

        try:
            source = lower_to_cuda(tir, target)
        except RuntimeError as exc:
            print(f"  CUDA codegen unavailable in this wheel: {exc}")
            print("  re-run on a wheel that ships the CUDA FFI, or inspect the "
                  "annotations above only")
            continue

        facts = static_facts(source)
        counters = facts["counters"]
        print(f"  generated CUDA: __syncthreads={counters['__syncthreads']}, "
              f"cp.async={counters['cp.async']} "
              f"(commit={counters['cp.async.commit_group']}, "
              f"wait_group={counters['cp.async.wait_group']}), "
              f"mbarrier={counters['mbarrier']}")
        for decl in facts["shared"]:
            print(f"    shared: {decl}")
        if args.dump_dir is not None:
            path = args.dump_dir / f"gemm_num_stages_{num_stages}.cu"
            path.write_text(source, encoding="utf-8")
            print(f"  wrote {path}")

    if args.dump_dir is not None:
        print()
        print("Now diff two variants, e.g.:")
        print(f"  diff -u {args.dump_dir}/gemm_num_stages_2.cu "
              f"{args.dump_dir}/gemm_num_stages_3.cu")
        print("Expect: the leading dimension of A_shared/B_shared grows, the "
              "number of")
        print("cp.async.wait_group values changes, and __syncthreads() moves.")


if __name__ == "__main__":
    main()
