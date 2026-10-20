#!/usr/bin/env python3
"""第 19 章 §9.2：把 RMSNorm 用 ACLGraph 入图（A 树）。

对照 `_vendor/tilelang-ascend-v0.1.2.000/examples/aclgraph/rms_rope_aclgraph.py`。
要演示的核心只有三行：

    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        ...  # 只记录，不执行
    g.replay()  # 一次调用重放整张图

注意：入图治的是 Host-bound（下发开销），不是图编译/图优化。

运行前提：Ascend NPU + CANN + torch_npu + TileLang-Ascend（A 树）。
验证级别：仅静态核对（py_compile + API 比对），未在 NPU 上运行。
"""

from __future__ import annotations

import argparse

import tilelang
import tilelang.language as T
import torch


# 四个开关：核内自动同步、内存规划、核间自动同步、核间通信合并。
# 这些 PassConfigKey 只存在于 A 树（tilelang/transform/pass_config.py:38-62）。
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def rms_norm_kernel(M: int, head_dim: int, block_M: int, eps: float, dtype: str = "float16"):
    """out = x / sqrt(mean(x^2) + eps)，逐行做。"""
    m_num = M // block_M
    VEC_NUM = 2
    row_per_vec = block_M // VEC_NUM
    ACC_DTYPE = "float32"

    @T.prim_func
    def main_rms(
        x: T.Tensor((M, head_dim), dtype),
        out: T.Tensor((M, head_dim), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row_x = cid * block_M + vid * row_per_vec

            x_ub = T.alloc_shared([row_per_vec, head_dim], dtype)
            x_ub_fp32 = T.alloc_shared([row_per_vec, head_dim], ACC_DTYPE)
            sum_square_ub = T.alloc_shared([row_per_vec, head_dim], ACC_DTYPE)
            rms_ub = T.alloc_shared([row_per_vec], ACC_DTYPE)

            # 1) 搬入并升精度
            T.copy(x[row_x : row_x + row_per_vec, :], x_ub)
            T.copy(x_ub, x_ub_fp32)
            T.tile.mul(sum_square_ub, x_ub_fp32, x_ub_fp32)
            # 2) 归约出每行的平方和
            T.reduce_sum(sum_square_ub, rms_ub, dim=-1)
            # 3) mean -> +eps -> sqrt
            T.tile.div(rms_ub, rms_ub, head_dim)
            T.tile.add(rms_ub, rms_ub, eps)
            T.tile.sqrt(rms_ub, rms_ub)
            # 4) 归一化并写回
            for i in T.serial(0, row_per_vec):
                T.tile.div(x_ub_fp32[i, :], x_ub_fp32[i, :], rms_ub[i])
            T.copy(x_ub_fp32, x_ub)
            T.copy(x_ub, out[row_x : row_x + row_per_vec, :])

    return main_rms


def tilelang_rms_norm(q: torch.Tensor, epsilon: float) -> torch.Tensor:
    batch_size, head_num, hidden_size = q.shape
    total_batch = batch_size * head_num
    flat = q.view(total_batch, hidden_size)
    func = rms_norm_kernel(total_batch, hidden_size, 32, epsilon)
    return func(flat).view(batch_size, head_num, hidden_size)


def torch_rms_norm(q: torch.Tensor, epsilon: float) -> torch.Tensor:
    x = q.float()
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return (x * rms).to(q.dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description="ACLGraph capture/replay with TileLang ops")
    parser.add_argument("--b", type=int, default=2, help="batch")
    parser.add_argument("--s", type=int, default=512, help="sequence length (heads)")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    if args.hidden % 32 != 0:
        raise SystemExit("hidden_size 必须能被 block_M=32 整除")

    q = torch.randn(args.b, args.s, args.hidden, dtype=torch.float16, device="npu")
    ref = torch_rms_norm(q, args.eps)

    # 先跑一次 eager 版，确认算子本身正确
    eager = tilelang_rms_norm(q, args.eps)
    torch.testing.assert_close(eager, ref, rtol=1e-2, atol=1e-2)
    print("eager correctness passed")

    # 入图：capture 一次，之后反复 replay。
    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        graph_out = tilelang_rms_norm(q, args.eps)
    g.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(graph_out, ref, rtol=1e-2, atol=1e-2)
    print("aclgraph replay correctness passed")

    # 想确认入图是否真的生效：把 ASCEND_GLOBAL_LOG_LEVEL=1 打开，
    # 在 Host 日志里搜 "capture model"（0=关闭，1=开启）。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
