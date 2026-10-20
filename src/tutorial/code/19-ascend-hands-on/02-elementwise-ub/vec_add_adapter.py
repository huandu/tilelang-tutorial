#!/usr/bin/env python3
"""第 19 章 §4：昇腾上的最小逐元素 add（A 树 / Adapter tree）。

对照 `_vendor/tilelang-ascend-v0.1.2.000/examples/elementwise/elementwise_add.py`，
把「注释、CLI、对拍」保留，把无关分支去掉，便于逐行阅读。

数据流：GM(A),GM(B) --MTE2--> UB --Vector--> UB --MTE3--> GM(C)
同步  ：两个 T.barrier_all()，一个保 MTE2->V，一个保 V->MTE3。

运行前提：Ascend NPU + CANN + torch_npu + TileLang-Ascend（A 树）。
验证级别：仅静态核对（py_compile + 逐个 API 比对），未在 NPU 上运行。
"""

from __future__ import annotations

import argparse

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[-1])
def vec_add(M: int, N: int, block_M: int, block_N: int, dtype: str = "float"):
    """C = A + B，逐元素；一个 block 处理一块 (block_M, block_N) 的 tile。"""
    m_num = M // block_M
    n_num = N // block_N

    # 昇腾每个 AI Core 有两个 Vector 子单元，用 vid ∈ {0,1} 把 tile 再劈成两半。
    VEC_NUM = 2

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            # 昇腾只有 1D grid：从 cid 还原二维块坐标。
            bx = cid // n_num
            by = cid % n_num

            # UB（Unified Buffer）是 Vector 单元唯一能直接读写的片上存储。
            a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                # MTE2：GM -> UB。起始行 = 第 bx 块的第 vid 半。
                row = bx * block_M + vid * block_M // VEC_NUM
                T.copy(A[row, by * block_N], a_ub)
                T.copy(B[row, by * block_N], b_ub)

                # 屏障 1：等两次 MTE2 搬运全部完成，Vector 才能安全读 a_ub/b_ub。
                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                # 屏障 2：等 Vector 计算全部完成，MTE3 才能安全读 c_ub 写回 GM。
                T.barrier_all()

                # MTE3：UB -> GM。
                T.copy(c_ub, C[row, by * block_N])

    return main


def main() -> int:
    parser = argparse.ArgumentParser(description="Ascend elementwise add (adapter tree)")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=256)
    args = parser.parse_args()

    m, n = args.m, args.n
    block_m, block_n = args.block_m, args.block_n
    if m % block_m != 0 or n % block_n != 0:
        raise SystemExit(f"M={m},N={n} 必须能被 block=({block_m},{block_n}) 整除（本示例不处理尾块）")

    func = vec_add(m, n, block_m, block_n)

    torch.manual_seed(0)
    a = torch.randn(m, n).npu()
    b = torch.randn(m, n).npu()
    torch.npu.synchronize()
    print("init successful!")

    c = func(a, b)
    torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
