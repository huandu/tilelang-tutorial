#!/usr/bin/env python3
"""第 18 章 · 昇腾侧「算子入图」（ACLGraph）的 TileLang 写法。

验证级别：**仅静态核对**。本机没有昇腾 NPU，本文件从未在 NPU 上运行；
`torch.npu`、`torch_npu`、CANN 都不在本机。

版本归属（很重要）
-----------------

本文件用的是 **TileLang-Ascend 适配器**（`_vendor/tilelang-ascend-v0.1.2.000`）
的语言面：

* ``T.Kernel(m_num, is_npu=True) as (cid, vid)``
  —— `is_npu` 是适配器 `tilelang/language/kernel.py:220,247` 的参数；
* ``tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC`` 等四个开关
  —— 定义在适配器 `tilelang/transform/pass_config.py:38-50`；
* ``T.tile.mul/add/div/sqrt``
  —— 适配器把 `ascend_tile` 模块暴露成 `T.tile`（`tilelang/language/__init__.py:86`）。

**主干 v0.1.15 的昇腾方言不是这套 API**：在
`_vendor/tilelang-v0.1.15/tilelang/ascend/language/` 里，kernel 写成
``with T.Kernel(N_CORES) as core_id``（没有 `is_npu` 参数），auto-schedule 由
pass context 开关控制（`tilelang/ascend/pipeline.py` 里的 `allow_autoschedule`），
而且 `TL_ASCEND_AUTO_SYNC` 这类键在 v0.1.15 里**不存在**（检索过，
`tilelang/transform/pass_config.py` 无 `ASCEND` 项）。

也就是说：**照抄本文件要在适配器仓库上跑，不是主干 v0.1.15。**
这一点在正文第 7 节与 `references.md` 里都写明了。

与官方示例的关系
---------------

结构对照 corpus 里的
`07-tilelang-ascend-adapter/raw/tilelang-ascend/examples/aclgraph/rms_rope_aclgraph.py`
与它的 `README.md`；这里把「RMSNorm → RoPE → 一步 decode」压成一个更小的
例子，重点只剩两件：

1. 两个 TileLang 算子如何在同一条 NPU stream 上依次下发；
2. 如何用 `torch.npu.graph` 捕获一次、重放多次。
"""

from __future__ import annotations

import argparse

import torch
import tilelang
import tilelang.language as T

#: 适配器的四个昇腾 pass 开关（`tilelang/transform/pass_config.py:38-50`）。
#: 它们分别负责：共享 buffer 的读写同步、显存规划、Cube/Vector 核间同步、
#: 以及把跨核通信合并（减少同步次数）。
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
}

DEVICE = torch.device("npu")


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def rms_norm_kernel(m: int, head_dim: int, block_m: int, eps: float,
                    vec_num: int = 2, dtype: str = "float16"):
    """RMSNorm：GM → UB 一次搬运，片上算平方和、求均值、归一化，再写回。"""
    rows_per_vec = block_m // vec_num
    n_blocks = m // block_m

    @T.prim_func
    def main_rms(
        x: T.Tensor((m, head_dim), dtype),
        out: T.Tensor((m, head_dim), dtype),
    ):
        with T.Kernel(n_blocks, is_npu=True) as (cid, vid):
            row_x = cid * block_m + vid * rows_per_vec
            x_ub = T.alloc_shared([rows_per_vec, head_dim], dtype)
            x_ub_fp32 = T.alloc_shared([rows_per_vec, head_dim], "float32")
            sum_square_ub = T.alloc_shared([rows_per_vec, head_dim], "float32")
            rms_ub = T.alloc_shared([rows_per_vec], "float32")

            T.copy(x[row_x:row_x + rows_per_vec, :], x_ub)
            T.copy(x_ub, x_ub_fp32)
            T.tile.mul(sum_square_ub, x_ub_fp32, x_ub_fp32)
            T.reduce_sum(sum_square_ub, rms_ub, dim=-1)
            T.tile.div(rms_ub, rms_ub, head_dim)
            T.tile.add(rms_ub, rms_ub, eps)
            T.tile.sqrt(rms_ub, rms_ub)
            for i in T.serial(0, rows_per_vec):
                T.tile.div(x_ub_fp32[i, :], x_ub_fp32[i, :], rms_ub[i])
            T.copy(x_ub_fp32, x_ub)
            T.copy(x_ub, out[row_x:row_x + rows_per_vec, :])

    return main_rms


def run_rms_norm(q: torch.Tensor, eps: float) -> torch.Tensor:
    """把 [B, H, D] 摊成 [B*H, D] 再调用 kernel。"""
    batch, heads, hidden = q.shape
    rows = batch * heads
    block_m = 32
    kernel = rms_norm_kernel(rows, hidden, block_m, eps)
    return kernel(q.view(rows, hidden)).view(batch, heads, hidden)


def capture_and_replay(q: torch.Tensor, eps: float) -> torch.Tensor:
    """ACLGraph 的三步：capture → replay → 校验。

    对应适配器文档第 7 章与 `examples/aclgraph/README.md`：

    1. `g = torch.npu.NPUGraph()`
    2. `with torch.npu.graph(g):` 里只**记录**算子调用，不执行
    3. `g.replay()` 真正执行；之后可以反复 replay
    """
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        out = run_rms_norm(q, eps)
    graph.replay()
    return out


def reference(q: torch.Tensor, eps: float) -> torch.Tensor:
    q32 = q.float()
    mean_square = (q32 * q32).sum(dim=-1, keepdim=True) / q.shape[-1]
    return (q32 / torch.sqrt(mean_square + eps)).half()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    torch.manual_seed(42)
    q = torch.randn((args.batch, args.heads, args.hidden),
                    device=DEVICE, dtype=torch.float16)
    out = capture_and_replay(q, args.eps)
    torch.testing.assert_close(out, reference(q, args.eps),
                               rtol=1e-2, atol=1e-2)
    print("Kernel Output Match!")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "本文件只做静态核对：请在装有昇腾 NPU + CANN + torch_npu +\n"
        "TileLang-Ascend（_vendor/tilelang-ascend-v0.1.2.000）的机器上运行；\n"
        "参照 code/08-ascend-aclgraph/README.md 中的命令与期望输出。"
    )
