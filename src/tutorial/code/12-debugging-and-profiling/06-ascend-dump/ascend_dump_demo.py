#!/usr/bin/env python3
"""第 12 章 示例 6：昇腾侧的运行期调试（T.printf / T.dump_tensor / 异常 dump）。

昇腾没有 CUDA 的 `printf` 语义可用：设备端打印要靠 Bisheng 编译选项打开，而且
**每个核上的 dump/print 空间上限是 1 MB**（见 corpus 的 `print.md`）。因此昇腾侧
有两条互补的路径：

  A. 主动打印：`T.printf(fmt, *args)` 与 `T.dump_tensor(tensor, desc, dump_size, shape_info)`。
     需要在 kernel 作用域打开 `-D_DEBUG --cce-enable-print` 编译选项。
  B. 事后取现场：AI Core 硬件异常时，由 CANN 回调把输入张量落盘，
     再用 `parse_exception_dump()` 读回 numpy 数组。这一条由
     `TL_ASCEND_EXCEPTION_DUMP` 编译配置开关控制，默认关闭。

**环境变量必须在 import torch / ACL 初始化之前设置**，否则 CANN 读不到。

用法：

    python3 ascend_dump_demo.py --static-only    # 任何机器：语法 + 入口核对
    python3 ascend_dump_demo.py --demo print     # 需要昇腾 NPU + CANN >= 9.3.0
    python3 ascend_dump_demo.py --demo dump      # 需要昇腾 NPU + CANN >= 9.3.0

作者实际做到的验证级别：**仅静态核对**（`py_compile` 通过；`T.printf` / `T.dump_tensor`
/ `TL_ASCEND_EXCEPTION_DUMP` / `parse_exception_dump` 四个入口已在
`_vendor/tilelang-ascend-v0.1.2.000/` 中核对；本机没有 NPU，未运行）。
"""

from __future__ import annotations

import argparse
import os
import py_compile
import sys
from pathlib import Path


def build_printf_kernel(T, tilelang):
    @tilelang.jit(
        out_idx=[2],
        target="ascendc",
        # AscendC 后端用 kernel 作用域的编译选项打开设备端打印。
        compile_flags=["-D_DEBUG", "--cce-enable-print"],
    )
    def kernel(M: int, N: int):
        dtype = "float16"

        @T.prim_func
        def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(N, 64), T.ceildiv(M, 64), threads=128) as (bx, by):
                a_ub = T.alloc_shared((64, 64), dtype)
                c_ub = T.alloc_shared((64, 64), dtype)
                T.copy(A[by * 64, bx * 64], a_ub)
                for i, j in T.Parallel(64, 64):
                    c_ub[i, j] = a_ub[i, j] * 2
                # 打印标量（一个核一次）+ 一条消息。
                T.printf("block=(%d, %d) ready\\n", bx, by)
                # 转储一个 tile：desc 用行号，dump_size=64，按 8x8 格式化。
                T.dump_tensor(c_ub, 111, 64, (8, 8))
                T.copy(c_ub, C[by * 64, bx * 64])

        return main

    return kernel


def run_print_demo() -> int:
    import tilelang
    import tilelang.language as T  # A 树（适配器）方言提供 T.printf / T.dump_tensor

    factory = build_printf_kernel(T, tilelang)
    src = tilelang.lower(factory(64, 64), target="ascendc").kernel_source
    assert "printf" in src or "PRINTF" in src, "设备端打印没有出现在生成源码里"
    print("已生成 AscendC 源码，且里面含设备端打印调用。")
    return 0


def run_exception_dump_demo() -> int:
    # 必须在 import torch / ACL 初始化之前设置。
    os.environ.setdefault("ASCEND_DUMP_PATH", "/tmp/exc_dump")
    os.environ.setdefault("ASCEND_DUMP_SCENE", "aic_err_brief_dump")

    import torch  # noqa: F401  （导入顺序本身就是演示的一部分）

    import tilelang
    import tilelang.language as T  # A 树（适配器）方言
    from tilelang.tools.ascend_exception_dump_bin import parse_exception_dump

    @tilelang.jit(
        out_idx=[2],
        target="ascendc",
        pass_configs={tilelang.PassConfigKey.TL_ASCEND_EXCEPTION_DUMP: True},
    )
    def kernel(M: int, N: int):
        dtype = "float16"

        @T.prim_func
        def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(N, 64), T.ceildiv(M, 64), threads=128) as (bx, by):
                a_ub = T.alloc_shared((64, 64), dtype)
                T.copy(A[by * 64, bx * 64], a_ub)
                T.copy(a_ub, C[by * 64, bx * 64])

        return main

    fn = kernel(64, 64)  # noqa: F841  ——只演示编译与开关，不真正制造异常
    print(f"TL_ASCEND_EXCEPTION_DUMP 已开启；dump 根目录 = {os.environ['ASCEND_DUMP_PATH']}")
    print("真实调用时在 except 里用：")
    print("  tensors = parse_exception_dump(dump_path=..., kernel_name='main_kernel')")
    _ = parse_exception_dump  # 保持引用，避免被 lint 当成未使用
    return 0


def static_only() -> int:
    py_compile.compile(str(Path(__file__)), doraise=True)
    print("py_compile: ok")
    print("需要与锁定版本核对的入口：")
    for name, where in (
        ("T.printf（昇腾适配器）", "_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:534"),
        ("T.dump_tensor（昇腾适配器）", "_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:582"),
        ("TL_ASCEND_EXCEPTION_DUMP", "_vendor/tilelang-ascend-v0.1.2.000/tilelang/transform/pass_config.py:62"),
        ("parse_exception_dump()", "_vendor/tilelang-ascend-v0.1.2.000/tilelang/tools/ascend_exception_dump_bin.py:67"),
        ("T.print（主干昇腾方言）", "_vendor/tilelang-v0.1.15/tilelang/ascend/language/print.py:31"),
    ):
        print(f"  {name:<34} {where}")
    print("（本机没有 NPU：这里只核对语法与入口存在性，未运行任何 kernel。）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demo", choices=("print", "dump"), default="print")
    parser.add_argument("--static-only", action="store_true", help="只做语法与入口核对，不导入 tilelang")
    args = parser.parse_args()
    if args.static_only:
        return static_only()
    if args.demo == "print":
        return run_print_demo()
    return run_exception_dump_demo()


if __name__ == "__main__":
    sys.exit(main())
