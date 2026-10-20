#!/usr/bin/env python3
"""第 19 章 §10.3：AICore exception dump（A 树）。

对照 `_vendor/tilelang-ascend-v0.1.2.000/examples/exception_dump_test/example_exception_dump.py`。
演示「硬件异常时自动抓取 kernel 输入张量」：
  * 环境变量 ASCEND_DUMP_PATH / ASCEND_DUMP_SCENE 必须在 import torch 之前设置；
  * 开关是 PassConfigKey.TL_ASCEND_EXCEPTION_DUMP（默认关闭）；
  * 异常后用 tilelang.tools.ascend_exception_dump_bin.parse_exception_dump 读回。

注意：本示例故意让 kernel 越界以触发异常；请不要在生产环境里带这个开关。

运行前提：Ascend NPU + CANN >= 9.3.0（提供 aclrtSetExceptionInfoCallback 等）
          + torch_npu + TileLang-Ascend（A 树）。
验证级别：仅静态核对（py_compile + API 比对），未在 NPU 上运行。
"""

from __future__ import annotations

import argparse
import os


def main() -> int:
    parser = argparse.ArgumentParser(description="Ascend AI Core exception dump demo")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--dump-dir", type=str, default="/tmp/tilelang_exc_dump")
    args = parser.parse_args()

    # 关键：这两个变量必须在 import torch / ACL 初始化之前生效。
    # 本模块顶部刻意没有 import torch，就是为了保证这里的顺序。
    os.environ["ASCEND_DUMP_PATH"] = args.dump_dir
    os.environ["ASCEND_DUMP_SCENE"] = "aic_err_brief_dump"
    os.makedirs(args.dump_dir, exist_ok=True)

    import tilelang
    import tilelang.language as T
    import torch
    from tilelang.tools.ascend_exception_dump_bin import parse_exception_dump

    M, N = args.m, args.n

    @tilelang.jit(
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
            tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
            tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
            tilelang.PassConfigKey.TL_ASCEND_EXCEPTION_DUMP: True,
        },
    )
    def vec_add_oob(M: int, N: int, dtype: str = "float16"):
        """一个**故意越界**的 kernel，用来触发 AICore 异常并产生 dump。"""
        M_, N_ = M + 1, N + 1

        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(1, is_npu=True) as (cid, vid):
                a_ub = T.alloc_ub((M_, N_), dtype)
                b_ub = T.alloc_ub((M_, N_), dtype)
                c_ub = T.alloc_ub((M_, N_), dtype)
                # 读 M_ x N_ 却只有 M x N：越界。
                T.copy(A[0, 0], a_ub)
                T.copy(B[0, 0], b_ub)
                for i, j in T.Parallel(M_, N_):
                    c_ub[i, j] = a_ub[i, j] + b_ub[i, j]
                T.copy(c_ub[0:M, 0:N], C[0, 0])

        return main

    kernel = vec_add_oob(M, N)
    a = torch.randn(M, N, dtype=torch.float16, device="npu")
    b = torch.randn(M, N, dtype=torch.float16, device="npu")

    try:
        result = kernel(a, b)
        print("no exception raised; kernel output shape:", tuple(result.shape))
    except Exception as exc:  # noqa: BLE001 - 演示用，任何异常都进入 dump 解析
        print("AI Core exception:", type(exc).__name__, exc)
        tensors = parse_exception_dump(dump_path=args.dump_dir, kernel_name="main")
        for t in tensors:
            data = t["data"]
            print(
                f"  {t['type']}[{t['index']}] dtype={t['dtype']} "
                f"shape={data.shape} min={float(data.min()):.4f} max={float(data.max()):.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
