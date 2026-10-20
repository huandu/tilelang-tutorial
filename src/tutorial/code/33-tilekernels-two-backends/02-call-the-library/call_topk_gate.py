#!/usr/bin/env python3
"""第 33 章 §3.4：用 TileKernels 的公开 API 跑一次 top-k gate（对拍参考实现）。

照 `tests/moe/test_topk_gate.py:52-59` 的写法复现一次端到端调用：
只碰外层 API，因此这份脚本在 CUDA 机器上跑 CUDA 实现、
在昇腾机器上跑昇腾实现（后端由 tile_kernels.config.is_ascend() 判定）。

运行前提：
  * 已安装 `tile-kernels`（或把仓库根加进 PYTHONPATH），
  * TileLang v0.1.15，
  * NVIDIA SM90/SM100，或 Ascend 950 + CANN >= 9.2.0。
验证级别：仅静态核对（py_compile 通过；逐条 API 与语料比对）。
"""

from __future__ import annotations

import torch

import tile_kernels
from tile_kernels.config import get_device, get_num_vec_cores, is_ascend, set_num_sms, set_pdl
from tile_kernels.testing.numeric import assert_equal
from tile_kernels.torch import stable_topk as torch_stable_topk


def main() -> None:
    # 1) 运行期后端探测：本章 §3.1 讲的全部判据都在这里。
    print(f'is_ascend() = {is_ascend()}')
    print(f'get_device() = {get_device()}')
    if is_ascend():
        # 纯向量算子用 get_num_vec_cores()（= Cube 核数 * 2）。
        print(f'get_num_vec_cores() = {get_num_vec_cores()}')
    else:
        # 可选：限制使用的 SM 数，或打开 PDL（CUDA-only）。
        # set_num_sms(8)
        set_pdl(False)

    # 2) 造数据。形状取自 tests/moe/test_topk_gate.py:47 的参数网格
    #    （num_experts in {1, 35, 1025}，num_topk in {1, 9, 32}），这里取一个常见值。
    torch.manual_seed(0)
    num_tokens, num_experts, num_topk = 512, 72, 6
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=get_device())

    # 3) 同一份 API：内部按 is_ascend() 分派到 topk_gate_cuda.py 或 topk_gate_asc.py。
    topk_idx = tile_kernels.moe.topk_gate(scores, num_topk)

    # 4) 对拍。参考实现是 torch.sort(stable=True)，三行、慢、但 tie-break 与契约一致。
    topk_idx_ref = torch_stable_topk(scores, num_topk)
    assert_equal(topk_idx, topk_idx_ref)

    print(f'topk_idx.shape = {tuple(topk_idx.shape)}, dtype = {topk_idx.dtype}')
    print('Kernel Output Match!')

    # 5) 边界：空输入在最外层短路，不编译也不启动（§2.3）。
    empty = torch.empty((0, num_experts), dtype=torch.float32, device=get_device())
    assert tuple(tile_kernels.moe.topk_gate(empty, num_topk).shape) == (0, num_topk)
    print('empty input short-circuits ok')


if __name__ == '__main__':
    main()
