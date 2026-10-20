#!/usr/bin/env python3
"""第 12 章 示例 4：看一个 pass 前后的变化（IR Lower Trace / Pass Diff）。

「某个 lowering 阶段之后 IR 就错了」是 TileLang 里最常见的生成类问题。三种看法：

  1. `TL_LOWER_TRACE`（推荐）：环境变量一开，捕获**每一个** pass 的前后 IR，
     再带最终 codegen；输出 terminal diff 或自包含 HTML。
  2. `lower_trace()` 程序 API：只对给定的一串 pass 做前后对比。
  3. `pass_diff`（legacy，仅向后兼容）：和 2 同构，`TILELANG_PASS_DIFF` 已不再演进。

关键时序：`TL_LOWER_TRACE` 是在 **import tilelang 时**被读取的，
所以本脚本先解析参数、设置环境变量、再导入 tilelang。

用法：

    python3 gemm_lower_trace.py --static-only          # 任何机器：语法 + 入口核对
    python3 gemm_lower_trace.py --mode trace-env       # 整条 pipeline，写 HTML 报告
    python3 gemm_lower_trace.py --mode trace-api       # 只比指定的一串 pass
    python3 gemm_lower_trace.py --mode pass-diff       # legacy Pass Diff

报告位置：默认 `./tmp/lower_trace_dir/<脚本名>/report.html`（可用 TL_LOWER_TRACE_DIR 改）。

作者实际做到的验证级别：**仅静态核对**（`py_compile` 通过；`lower_trace` / `enable`
/ `pass_diff` / `Simplify` 四个入口已在 v0.1.15 源码中核对；本机未安装 tilelang，
未生成过任何报告）。
"""

from __future__ import annotations

import argparse
import os
import py_compile
import sys
from pathlib import Path


def build_kernel(T, tilelang):
    @tilelang.jit
    def matmul(A, B, block_M: int, block_N: int, block_K: int):
        M, N, K = T.const("M, N, K")
        dtype = T.float16
        accum_dtype = T.float32
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((N, K), dtype)
        C = T.empty((M, N), dtype)

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])
        return C

    return matmul(1024, 1024, 1024, 128, 128, 32)


def static_only() -> int:
    py_compile.compile(str(Path(__file__)), doraise=True)
    print("py_compile: ok")
    print("需要与锁定版本核对的入口（逐个在 _vendor/tilelang-v0.1.15/ 下确认）：")
    for name, where in (
        ("TL_LOWER_TRACE / TL_LOWER_TRACE_DIR", "tilelang/env.py:390-392"),
        ("lower_trace()", "tilelang/tools/lower_trace/__init__.py:49"),
        ("enable() / disable()", "tilelang/tools/lower_trace/core.py:663-705"),
        ("pass_diff()", "tilelang/utils/pass_diff.py:1030"),
        ("transform.Simplify", "tilelang/transform/simplify.py:20"),
    ):
        print(f"  {name:<38} {where}")
    print("（本机未安装 tilelang：这里只核对语法与入口存在性，不生成任何报告。）")
    return 0


def run_trace_env(target: str, mode: str) -> int:
    # 必须在 import tilelang 之前设置：TileLang 在 import 时读取该变量。
    os.environ["TL_LOWER_TRACE"] = mode
    os.environ.setdefault("TL_LOWER_TRACE_DIR", "./tmp/lower_trace_dir")
    import tilelang
    import tilelang.language as T

    func = build_kernel(T, tilelang)
    # tilelang.lower() 默认不做 device 编译（enable_device_compile=False），
    # 因此即使本机没有 nvcc，也能把 lowering 走完并抓取 IR。
    tilelang.lower(func, target=target)
    print(f"TL_LOWER_TRACE={mode}，报告目录：{os.environ['TL_LOWER_TRACE_DIR']}/")
    print("打开 <脚本名>/report.html 即可按 pass 逐步查看 before/after。")
    return 0


def run_trace_api(_target: str) -> int:
    import tilelang
    import tilelang.language as T
    from tilelang.tools import lower_trace

    func = build_kernel(T, tilelang)
    steps = lower_trace.lower_trace(
        func,
        [("Simplify", tilelang.transform.Simplify())],
        mode="terminal",
    )
    for step in steps:
        print(
            f"pass={step['name']:<12} changed={step['changed']} "
            f"+{step['insertions']} -{step['deletions']}"
        )
    print("注意：文本变了不等于语义或性能变了——它只是一个可定位的线索。")
    return 0


def run_pass_diff(_target: str) -> int:
    import tilelang
    import tilelang.language as T
    from tilelang.utils.pass_diff import pass_diff

    func = build_kernel(T, tilelang)
    steps = pass_diff(func, tilelang.transform.ThreadSync("shared"), mode="terminal")
    for step in steps:
        print(
            f"pass={step['name']:<12} changed={step['changed']} "
            f"+{step['insertions']} -{step['deletions']}"
        )
    print("Pass Diff 已被 IR Lower Trace 取代，新代码建议直接用 TL_LOWER_TRACE。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="cuda", help="编译目标（cuda 只需 codegen，不需要 GPU）")
    parser.add_argument(
        "--mode",
        choices=("trace-env", "trace-api", "pass-diff"),
        default="trace-env",
        help="trace-env = TL_LOWER_TRACE 全流程；trace-api = lower_trace()；pass-diff = legacy",
    )
    parser.add_argument("--static-only", action="store_true", help="只做语法与入口核对，不导入 tilelang")
    args = parser.parse_args()

    if args.static_only:
        return static_only()
    if args.mode == "trace-env":
        return run_trace_env(args.target, "html")
    if args.mode == "trace-api":
        return run_trace_api(args.target)
    return run_pass_diff(args.target)


if __name__ == "__main__":
    sys.exit(main())
