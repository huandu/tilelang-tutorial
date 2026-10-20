#!/usr/bin/env python3
"""第 19 章 §10.1/§10.2：T.printf 与 T.dump_tensor（A 树）。

把 A 树 `examples/print/elementwise_print.py` 收窄到最小可读版本：
  1. 用 kernel 级 compile_flags 打开设备端打印（推荐做法）；
  2. 打印标量（cid/vid）、打印 UB buffer、按 shape_info 打印。

注意：设备端打印会明显拖慢算子，且每个核上 dump/printf 的总空间上限是 1 MB。
只在调试期开，上线一定关掉。

运行前提：Ascend NPU + CANN（bisheng 支持 --cce-enable-print）+ torch_npu + A 树。
验证级别：仅静态核对（py_compile + API 比对），未在 NPU 上运行。
"""

from __future__ import annotations

import argparse

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(
    out_idx=[-1],
    target="ascendc",
    # 推荐做法：kernel 级 compile_flags（见 docs/tutorials/print.md）。
    # TL_PTO_DEBUG=1 只对 target="pto" 生效，不要用它来给 ascendc 开后门。
    compile_flags=["-D_DEBUG", "--cce-enable-print"],
)
def debug_add(M: int, N: int, block_M: int, block_N: int, dtype: str = "float16"):
    m_num = M // block_M
    n_num = N // block_N
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # 打印标量：格式串 + 可变参数。%d 十进制、%f 浮点、%x 十六进制、%s 字符串。
        T.printf("===========A (first row):\n")
        T.dump_tensor(A[0, :], 111, N, (1, N))

        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num
            T.printf("-----cid:%d vid:%d-----\n", cid, vid)

            row = bx * block_M + vid * block_M // VEC_NUM
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                T.copy(A[row, by * block_N], a_ub)
                T.copy(B[row, by * block_N], b_ub)
                T.barrier_all()

                T.printf("===========a_ub after copy (desc=222):\n")
                T.dump_tensor(a_ub, 222, block_M // VEC_NUM * block_N, (block_M // VEC_NUM, block_N))

                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()

                T.printf("===========c_ub after add (desc=333):\n")
                T.dump_tensor(c_ub, 333, block_M // VEC_NUM * block_N)

                T.copy(c_ub, C[row, by * block_N])

        T.printf("===========C (first row):\n")
        T.dump_tensor(C[0, :], 444, N, (1, N))

    return main


def main() -> int:
    parser = argparse.ArgumentParser(description="Ascend device printf / dump_tensor demo")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--block-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, default=16)
    args = parser.parse_args()

    func = debug_add(args.m, args.n, args.block_m, args.block_n)
    a = torch.randn(args.m, args.n, dtype=torch.float16, device="npu")
    b = torch.randn(args.m, args.n, dtype=torch.float16, device="npu")
    c = func(a, b)
    torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
