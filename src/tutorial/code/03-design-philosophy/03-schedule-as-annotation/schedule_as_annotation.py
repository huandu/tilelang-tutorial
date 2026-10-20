#!/usr/bin/env python3
"""第 03 章 示例 3：同一段数据流，四种调度（调度是注解，不是重写）。

四个 kernel 的数据流完全相同（搬入分块 -> 乘加 -> ReLU -> 搬出），只在调度空间的
某一个维度上各差一行/三行：

    v0  基线                —— 四个维度全部缺省
    v1  + thread binding    —— T.gemm(..., policy=T.GemmWarpPolicy.FullRow)
    v2  + layout            —— T.annotate_layout({B_s: make_swizzled_layout(B_s)})
    v3  + pipeline          —— T.Pipelined(..., stage=[0,0,1], order=[0,1,2])

依赖：tilelang v0.1.15、torch(CUDA)、一张 NVIDIA GPU。
无 GPU 时用 `--static-only`。作者验证级别：仅静态核对。
"""

from __future__ import annotations

import argparse
import sys

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32
THREADS = 128

# (变体, 改动的维度, 改了什么, 依据)
SCHEDULE_DELTAS: tuple[tuple[str, str, str, str], ...] = (
    ("v0", "（无）", "全部走缺省值", "tilelang/cuda/language/kernel.py:32-35（threads 默认 128）"),
    ("v1", "thread binding", "T.gemm(..., policy=T.GemmWarpPolicy.FullRow)",
     "tilelang/language/gemm_op.py:166 与 tilelang/tileop/base.py:11"),
    ("v2", "layout", "T.annotate_layout({B_s: make_swizzled_layout(B_s)})",
     "tilelang/language/annotations.py:45 与 tilelang/layout/swizzle.py:66"),
    ("v3", "pipeline", "T.Pipelined(..., stage=[0,0,1], order=[0,1,2])",
     "tilelang/language/loop.py:112-118 与 docs/programming_guides/software_pipeline.md:44-60"),
)

API_CHECKLIST: tuple[str, ...] = (
    "T.Kernel            tilelang/language/kernel.py:431",
    "T.alloc_shared      tilelang/language/allocate.py:49",
    "T.alloc_fragment    tilelang/language/allocate.py:81",
    "T.clear             tilelang/language/fill_op.py:48",
    "T.copy              tilelang/language/copy_op.py:59",
    "T.gemm              tilelang/language/gemm_op.py:160",
    "T.GemmWarpPolicy    tilelang/tileop/base.py:5",
    "T.Pipelined         tilelang/language/loop.py:112",
    "T.Parallel          tilelang/language/loop.py:13",
    "T.annotate_layout   tilelang/language/annotations.py:45",
    "make_swizzled_layout tilelang/layout/swizzle.py:66",
)

# 数据流骨架（四个变体共用；只有带 <> 的地方不同）：
#   with T.Kernel(<grid>, threads=THREADS) as (bx, by):
#       A_s = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
#       B_s = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
#       C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
#       <v2 在这里插一句 T.annotate_layout(...)>
#       T.clear(C_frag)
#       <v3 用 stage/order，其它变体用 num_stages>
#       for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), <调度参数>):
#           T.copy(A[by * BLOCK_M, ko * BLOCK_K], A_s)
#           T.copy(B[ko * BLOCK_K, bx * BLOCK_N], B_s)
#           T.gemm(A_s, B_s, C_frag<v1 在这里加 policy=>)
#       for i, j in T.Parallel(BLOCK_M, BLOCK_N):
#           C_frag[i, j] = T.max(C_frag[i, j], 0)
#       T.copy(C_frag, C[by * BLOCK_M, bx * BLOCK_N])


def _make_kernel(*, warp_policy: bool, layout_annot: bool, manual_pipeline: bool):
    """构造一个变体。导入被推迟到调用时，方便无 GPU 环境下只跑 --static-only。"""
    import tilelang
    import tilelang.language as T
    from tilelang.layout import make_swizzled_layout

    @tilelang.jit(out_idx=[2])
    def kernel(M: int, N: int, K: int):
        @T.prim_func
        def main(
            A: T.Tensor((M, K), "float16"),
            B: T.Tensor((K, N), "float16"),
            C: T.Tensor((M, N), "float16"),
        ):
            with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=THREADS) as (bx, by):
                # ---- 数据流：以下四行在四个变体里一模一样 ----
                A_s = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
                B_s = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
                C_frag = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
                T.clear(C_frag)

                # ---- 调度维度 2：layout（v2 才有这一行）----
                if layout_annot:
                    T.annotate_layout({B_s: make_swizzled_layout(B_s)})

                # ---- 调度维度 4：pipeline（v3 手工 stage/order；其它用 num_stages）----
                if manual_pipeline:
                    for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), stage=[0, 0, 1], order=[0, 1, 2]):
                        T.copy(A[by * BLOCK_M, ko * BLOCK_K], A_s)
                        T.copy(B[ko * BLOCK_K, bx * BLOCK_N], B_s)
                        if warp_policy:
                            T.gemm(A_s, B_s, C_frag, policy=T.GemmWarpPolicy.FullRow)
                        else:
                            T.gemm(A_s, B_s, C_frag)
                else:
                    for ko in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=2):
                        T.copy(A[by * BLOCK_M, ko * BLOCK_K], A_s)
                        T.copy(B[ko * BLOCK_K, bx * BLOCK_N], B_s)
                        # ---- 调度维度 1：thread binding（v1 才有 policy=）----
                        if warp_policy:
                            T.gemm(A_s, B_s, C_frag, policy=T.GemmWarpPolicy.FullRow)
                        else:
                            T.gemm(A_s, B_s, C_frag)

                # ---- 数据流 ----
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    C_frag[i, j] = T.max(C_frag[i, j], 0)
                T.copy(C_frag, C[by * BLOCK_M, bx * BLOCK_N])

        return main

    return kernel


VARIANTS = {
    "v0": dict(warp_policy=False, layout_annot=False, manual_pipeline=False),
    "v1": dict(warp_policy=True, layout_annot=False, manual_pipeline=False),
    "v2": dict(warp_policy=False, layout_annot=True, manual_pipeline=False),
    "v3": dict(warp_policy=False, layout_annot=False, manual_pipeline=True),
}


def static_only() -> int:
    print("第 03 章示例 3：同一段数据流，四种调度（仅静态核对）")
    print("-" * 60)
    print("数据流骨架（四个变体共用，只有 <> 处不同）：")
    print("  with T.Kernel(grid, threads=128) as (bx, by):")
    print("      A_s, B_s, C_frag = T.alloc_shared/fragment(...)")
    print("      <v2: T.annotate_layout({B_s: make_swizzled_layout(B_s)})>")
    print("      T.clear(C_frag)")
    print("      for ko in T.Pipelined(..., <v3: stage=[0,0,1], order=[0,1,2] | else num_stages=2>):")
    print("          T.copy(A[...], A_s); T.copy(B[...], B_s)")
    print("          T.gemm(A_s, B_s, C_frag <v1: , policy=T.GemmWarpPolicy.FullRow>)")
    print("      for i, j in T.Parallel(BLOCK_M, BLOCK_N): C_frag[i, j] = T.max(C_frag[i, j], 0)")
    print("      T.copy(C_frag, C[...])")
    print()
    print("调度维度对照：")
    for name, dim, delta, src in SCHEDULE_DELTAS:
        print(f"  {name}  {dim:<15} {delta}")
        print(f"      依据：{src}")
    print()
    print("API 核对清单（v0.1.15）：")
    for row in API_CHECKLIST:
        print(f"  {row}")
    print()
    print("四个变体的数据流完全相同——调度是注解，不是重写。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--K", type=int, default=512)
    args = parser.parse_args()

    if args.static_only:
        return static_only()

    try:
        import torch
        import tilelang  # noqa: F401
    except ImportError as exc:
        print(f"缺少依赖：{exc}", file=sys.stderr)
        print("本示例需要 NVIDIA GPU + CUDA 版 PyTorch + tilelang v0.1.15。", file=sys.stderr)
        print("无 GPU 时请改用：python3 schedule_as_annotation.py --static-only", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("检测不到 CUDA 设备。", file=sys.stderr)
        print("无 GPU 时请改用：python3 schedule_as_annotation.py --static-only", file=sys.stderr)
        return 3

    torch.manual_seed(0)
    a = torch.randn(args.M, args.K, device="cuda", dtype=torch.float16)
    b = torch.randn(args.K, args.N, device="cuda", dtype=torch.float16)
    ref = torch.relu(a @ b).float()

    for name, opts in VARIANTS.items():
        kernel = _make_kernel(**opts)(args.M, args.N, args.K)
        out = kernel(a, b)
        torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
        print(f"{name}: 与 torch.relu(A @ B) 一致。")
    print("四个变体都正确——它们的差别只在调度注解。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
