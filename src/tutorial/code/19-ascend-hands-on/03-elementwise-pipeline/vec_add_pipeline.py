#!/usr/bin/env python3
"""第 19 章 §4.4：手写双缓冲的昇腾逐元素 add（A 树）。

对照 `_vendor/tilelang-ascend-v0.1.2.000/examples/elementwise/elementwise_add_pipeline.py`。
它回答的问题是：「搬下一块」和「算这一块」怎么重叠？答案是**两组 flag + 两个缓冲版本**。

三对 flag（都是同一核内 PIPE 之间的锁）：
  v    -> mte3 : 这一槽位算完了，可以写回
  mte3 -> mte2 : 这一槽位已经消费完，可以给下一轮预取复用
  mte2 -> v    : 这一槽位的数据搬完了，可以计算

运行前提：Ascend NPU + CANN + torch_npu + TileLang-Ascend（A 树）。
验证级别：仅静态核对（py_compile + API 比对），未在 NPU 上运行。
"""

from __future__ import annotations

import argparse

import tilelang
import tilelang.language as T
import torch


@tilelang.jit(out_idx=[-1])
def vec_add_pipeline(
    M: int,
    N: int,
    block_M: int,
    block_N: int,
    sub_M: int,
    dtype: str = "float",
):
    """C = A + B，逐元素；块内再按 sub_M 分片做两级软件流水。"""
    m_num = M // block_M
    n_num = N // block_N

    VEC_NUM = 2
    stages = 2

    @T.macro
    def init_flag():
        # 初始状态：两个槽位都「可写」，所以先给 MTE2 两个 token。
        T.set_flag("mte3", "mte2", 0)
        T.set_flag("mte3", "mte2", 1)

    @T.macro
    def clear_flag():
        # 收尾：把两个槽位最后一次归还的 token 消费掉。
        T.wait_flag("mte3", "mte2", 0)
        T.wait_flag("mte3", "mte2", 1)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = cid // n_num
            by = cid % n_num

            vec_proc = block_M // sub_M
            base_row = bx * block_M + vid * sub_M // VEC_NUM

            # 首维是「版本维」：stages 个物理槽位轮转。
            a_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)
            b_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)
            c_ub = T.alloc_ub((stages, sub_M // VEC_NUM, block_N), dtype)

            with T.Scope("V"):
                init_flag()

                # 预取第 0 片
                T.wait_flag("mte3", "mte2", 0)
                T.copy(A[base_row, by * block_N], a_ub[0, :, :])
                T.copy(B[base_row, by * block_N], b_ub[0, :, :])
                T.set_flag("mte2", "v", 0)

                for mm in T.serial(vec_proc):
                    cur = mm % stages
                    nxt = (mm + 1) % stages

                    if mm < vec_proc - 1:
                        # 预取下一片（等它的槽位被上一轮消费完）
                        T.wait_flag("mte3", "mte2", nxt)
                        row = base_row + (mm + 1) * sub_M
                        T.copy(A[row, by * block_N], a_ub[nxt, :, :])
                        T.copy(B[row, by * block_N], b_ub[nxt, :, :])
                        T.set_flag("mte2", "v", nxt)

                    # 计算当前片（等它的数据搬完）
                    T.wait_flag("mte2", "v", cur)
                    for i, j in T.Parallel(sub_M // VEC_NUM, block_N):
                        c_ub[cur, i, j] = a_ub[cur, i, j] + b_ub[cur, i, j]

                    # 写回当前片（等它算完），然后归还槽位
                    T.set_flag("v", "mte3", cur)
                    T.wait_flag("v", "mte3", cur)
                    T.copy(c_ub[cur, :, :], C[base_row + mm * sub_M, by * block_N])
                    T.set_flag("mte3", "mte2", cur)

                clear_flag()

    return main


def main() -> int:
    parser = argparse.ArgumentParser(description="Ascend pipelined elementwise add (adapter tree)")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--sub-m", type=int, default=32)
    args = parser.parse_args()

    m, n = args.m, args.n
    func = vec_add_pipeline(m, n, args.block_m, args.block_n, args.sub_m)

    torch.manual_seed(0)
    a = torch.randn(m, n).npu()
    b = torch.randn(m, n).npu()
    torch.npu.synchronize()
    print("init successful!")

    c = func(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
