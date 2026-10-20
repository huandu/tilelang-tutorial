#!/usr/bin/env python3
"""第 12 章 示例 5：运行期打印与断言（NVIDIA / CUDA 侧）。

编译期工具回答「IR 对不对」，运行期工具回答「数值对不对」。本文件把 `T.print`
的四种用法和 `T.device_assert` 放在一起，并标注每种用法的**单线程门控语义**——
这是最容易踩的坑：`T.print(global_buf)` 默认从**每个线程**都打印一遍。

单线程门控（见 `_vendor/tilelang-v0.1.15/tilelang/cuda/language/print.py:155-194`）：

  local          → condition = True          （所有线程都打印）
  local.fragment → tx == main_lane           （只由选定线程打印；先 copy 到 shared）
  shared / dyn   → tx == main_lane           （只由一个线程打印）
  global         → condition = True          （所有线程都打印，输出会被重复 N 次）

代价与限制：device 侧 `printf` 走的是受限的片上缓冲，打印会串行化并显著改变耗时，
只应出现在调试版本里；调试完必须移除。

用法：

    python3 debug_print_demo.py --static-only    # 任何机器：语法 + 入口核对
    python3 debug_print_demo.py --which all      # 需要 NVIDIA GPU + CUDA toolkit

作者实际做到的验证级别：**仅静态核对**（`py_compile` 通过；`T.print` / `T.device_assert`
两个入口与 v0.1.15 一致；本机没有 GPU，未运行）。
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path


def build_program(T, tilelang):
    M, N = 16, 16
    dtype = T.float16

    @tilelang.jit
    def kernel_factory():
        @T.prim_func
        def program(Q: T.Tensor((M, N), dtype)):
            with T.Kernel(4, 4, threads=256) as (bx, by):
                tx = T.get_thread_binding()
                shared_buf = T.alloc_shared([M, N], dtype)
                frag = T.alloc_fragment([M, N], dtype)

                # 1) 打印一个 shared buffer：内部已门控，只由一个线程输出。
                T.print(shared_buf, msg="shared_buf")

                # 2) 只让一个线程打印标量，避免 N 份重复输出。
                if tx == 0 and bx == 0 and by == 0:
                    T.print(by + bx, msg="block_sum")

                # 3) 只打印一条消息。
                if tx == 0:
                    T.print(msg="hello from debug_print_demo")

                # 4) 打印 fragment 的元素（会被 copy 到 shared 再打印）。
                for i, j in T.Parallel(M, N):
                    T.print(frag[i, j], msg="frag_elem")

                # 5) device 侧断言：条件不成立时让 kernel 报错并给出消息。
                T.device_assert(tx < 256, "thread binding out of range")

        return program

    return kernel_factory()


def run() -> int:
    import torch

    import tilelang
    import tilelang.language as T

    program = build_program(T, tilelang)
    kernel = tilelang.compile(program, target="cuda")
    profiler = kernel.get_profiler()
    profiler.run_once()
    torch.cuda.synchronize()
    print("已运行一次；device 端 printf 会出现在 stdout（顺序取决于实际调度）。")
    return 0


def static_only() -> int:
    py_compile.compile(str(Path(__file__)), doraise=True)
    print("py_compile: ok")
    print("需要与锁定版本核对的入口（逐个在 _vendor/tilelang-v0.1.15/ 下确认）：")
    for name, where in (
        ("T.print（CUDA 方言，含门控）", "tilelang/cuda/language/print.py:128-207"),
        ("T.device_assert（CUDA 方言）", "tilelang/cuda/debug.py:23"),
        ("device 侧 printf 模板", "src/tl_templates/cuda/debug.h:11-55"),
        ("单线程门控常量（warp_group_size=128）", "tilelang/cuda/language/print.py:149-151"),
    ):
        print(f"  {name:<38} {where}")
    print("（本机没有 GPU：这里只核对语法与入口存在性，未运行任何 kernel。）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--which", default="all", help="保留参数：all = 依次演示全部打印用法")
    parser.add_argument("--static-only", action="store_true", help="只做语法与入口核对，不导入 tilelang")
    args = parser.parse_args()
    if args.static_only:
        return static_only()
    return run()


if __name__ == "__main__":
    sys.exit(main())
