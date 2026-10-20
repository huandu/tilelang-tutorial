#!/usr/bin/env python3
"""第 12 章 示例 3：不跑 GPU，先看编译产物（Compile-Only）。

调试「生成错误（generation issue）」的第一原则：**先看编译器到底产出了什么**，
再去猜哪一行 TileLang 写错了。`tilelang.tools.compile_only` 把 kernel 降到源码
就停下，不做 device 编译、不启动，也不需要 GPU/CUDA driver。

这个文件同时演示两条路径：

  A. 程序 API：`compile_kernel_source(func, target)` 直接拿到源码字符串；
  B. 命令行：`python -m tilelang.tools.compile_only --output_file out.c 本文件`。

本机（macOS，无 GPU）只能走 A 的 `target="c"`：默认的 C 目标在任何机器上都能跑。
`target="cuda"` 需要 wheel 里编进了 CUDA codegen FFI（Linux wheel 有，
macOS Metal wheel 没有），而且**依然不需要 GPU**——它把架构钉死在 sm_80。

顶层只 import 标准库，tilelang 在函数内部延迟导入，因此 `--static-only`
在没装 tilelang 的机器上也能正常跑。

用法：

    python3 add_kernel_compile_only.py --static-only   # 任何机器：语法 + 入口核对
    python3 add_kernel_compile_only.py                 # 有 tilelang：打印 CPU C 源码
    python3 add_kernel_compile_only.py --target cuda   # 有 CUDA codegen FFI：打印 CUDA 源码

作者实际做到的验证级别：**仅静态核对**（`py_compile` 通过；`T.empty` / `T.Kernel`
/ `T.Parallel` / `compile_kernel_source` 四个入口已在 v0.1.15 源码中核对；
本机未安装 tilelang，未实际生成源码）。
"""

from __future__ import annotations

import argparse
import py_compile
import sys
from pathlib import Path


def build_kernel():
    """构造并返回 `@tilelang.jit` kernel（延迟导入 tilelang）。"""
    import tilelang
    import tilelang.language as T

    @tilelang.jit
    def add(A, B):
        N = 64
        A: T.Tensor((N,), T.float16)
        B: T.Tensor((N,), T.float16)
        C = T.empty((N,), T.float16)
        with T.Kernel(1, threads=64):
            for i in T.Parallel(N):
                C[i] = A[i] + B[i]
        return C

    return add


def show_source(target: str) -> int:
    from tilelang.tools.compile_only import compile_kernel_source

    add = build_kernel()
    source = compile_kernel_source(add.get_tir(), target)
    print(f"--- target={target!r} 生成的源码（{len(source)} 字节） ---")
    print(source)
    print("--- 提示：把上面的 add_kernel 与你的预期逐行对照，能定位大多数生成错误 ---")
    return 0


def static_only() -> int:
    """不导入 tilelang：只校验本文件语法，并打印需要与 v0.1.15 核对的入口。"""
    py_compile.compile(str(Path(__file__)), doraise=True)
    print("py_compile: ok")
    print("需要与锁定版本核对的入口（逐个在 _vendor/tilelang-v0.1.15/ 下确认）：")
    for name, where in (
        ("T.empty", "tilelang/language/allocate.py:400"),
        ("T.Kernel", "tilelang/language/kernel.py:431"),
        ("T.Parallel", "tilelang/language/loop.py:13"),
        ("tilelang.jit", "tilelang/jit/__init__.py:575"),
        ("compile_kernel_source", "tilelang/tools/compile_only.py:169"),
    ):
        print(f"  {name:<24} {where}")
    print("（本机未安装 tilelang：这里只核对语法与入口存在性，不生成任何源码。）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default="c", help="c（默认，任何 wheel 都能用）或 cuda")
    parser.add_argument("--static-only", action="store_true", help="只做语法与入口核对，不导入 tilelang")
    args = parser.parse_args()

    if args.static_only:
        return static_only()
    return show_source(args.target)


if __name__ == "__main__":
    sys.exit(main())
