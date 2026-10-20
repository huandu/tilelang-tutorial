#!/usr/bin/env python3
"""第 33 章 §4.3：top-k gate 的 Ascend 版最小可读实现。

改写自 corpus 快照
`.workspace/research/tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/moe/topk_gate_asc.py`
（commit 66258df6175d）：保留「UB 缓冲 + 显式缓冲版本 + SIMD 归约 + int64 位视图」
的全部结构，只保留单组（num_groups == 1）快路径，并加上逐段注释。

与原文件的差异（都已在本章正文里说明）：
  * 只保留 num_groups == 1 的快路径（原文 :73-105），省略多组慢路径（原文 :106-151）；
  * 保留非法值打标（原文 :66-71）与循环外断言（原文 :155），因为它是「缓冲版本 = 1」
    这个设计的直接后果；
  * 省略 `T.copy(..., l2_cache_ctrl=...)` 之类的可选提示。

运行前提：Ascend 950 NPU + CANN >= 9.2.0 + torch_npu + TileLang v0.1.15
（用 `tilelang.ascend.language` 方言，即本章所说的 M 树）。
验证级别：仅静态核对（py_compile 通过；API 名与签名在 _vendor/tilelang-v0.1.15/ 中比对）。
"""

from __future__ import annotations

import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


@tilelang.jit
def get_topk_gate_kernel_asc(num_experts: int, num_topk: int, num_vec_cores: int):
    num_stages = 6
    vector_length = 64  # 一条向量寄存器（vreg）装 64 条 fp32 SIMD lane

    # 一组最多用 6 个 vreg。num_experts 不大时只开够用的几个，减少寄存器压力。
    max_vregs_per_group = 6
    num_vregs_per_group = ceil_div(min(num_experts, max_vregs_per_group * vector_length), vector_length)
    experts_per_group = num_vregs_per_group * vector_length
    num_groups = ceil_div(num_experts, experts_per_group)
    num_aligned_experts = num_groups * experts_per_group

    # 对齐到 vector_length：既方便 UB 访存，也顺便给下面的 int64 位视图留出空间
    # （2 * num_topk <= 64 <= num_aligned_topk）。
    num_aligned_topk = align(num_topk, vector_length)
    num_tokens = T.dynamic('num_tokens')

    assert num_groups == 1, '本最小示例只保留单组快路径；多组路径见原生实现'

    @T.prim_func
    def topk_gate_kernel_asc(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
    ):
        # 昇腾的 grid 就是 Vector 核的编号；这一层没有 threads= 参数。
        with T.Kernel(num_vec_cores) as core_id:
            # 在昇腾方言里 T.alloc_shared 指向 UB，不是 CUDA 的 shared memory。
            scores_ub = T.alloc_shared((num_aligned_experts,), T.float32)
            out_ub = T.alloc_shared((num_aligned_topk,), T.int32)
            invalid_logits_ub = T.alloc_shared((vector_length,), T.int32)
            # 把 int64 输出重新解释成 int32：低位写下标，高位保持 0。
            topk_idx_view = T.view(topk_idx, (num_tokens, num_topk * 2), T.int32)
            # 显式声明缓冲版本：前两个是流水数据（num_stages 份），
            # 第三个是跨所有迭代共享的累积器（只有 1 份）。
            T.annotate_buffer_versions({
                scores_ub: num_stages,
                out_ub: num_stages,
                invalid_logits_ub: 1,
            })

            with T.SimdVF():
                T.clear(out_ub)             # 高位必须为 0，int64 视图才正确
                T.clear(invalid_logits_ub)

            # 流水：每核处理 ceil_div(num_tokens, num_vec_cores) 行，按核交错分配。
            for w in T.Pipelined(
                T.ceildiv(num_tokens, num_vec_cores),
                num_stages=num_stages,
                annotations={'multi_buffer_eligible': [scores_ub, out_ub]},
            ):
                row = w * num_vec_cores + core_id
                if row < num_tokens:
                    T.copy(scores[row, :], scores_ub[:num_experts])

                    with T.SimdVF():
                        mask_32b = S.pset(32, 'PAT_ALL')
                        one = S.pset(32, 'PAT_VL1')
                        neg_inf = S.vdup(-T.infinity(T.float32), T.float32, mask_32b)
                        int_max = S.vdup(T.max_value(T.int32), T.int32, mask_32b)

                        scores_vec = S.alloc_local((num_vregs_per_group,), T.float32)
                        index_vec = S.alloc_local((num_vregs_per_group,), T.int32)
                        tmp_idx_vec = S.alloc_local((num_vregs_per_group,), T.int32)

                        lanes = S.vci(0, T.int32)
                        zero = S.vdup(0.0, T.float32)
                        zeros = S.vdup(0, T.int32)

                        # 非有限值检测：x - x != 0 对 NaN 与 +/-inf 都成立。
                        # 结果用 max 累积进一个 64 宽的向量寄存器，循环外统一断言。
                        invalid = T.alloc_var(T.int32x64, init=S.vld(invalid_logits_ub[0]))
                        for j in T.unroll(ceil_div(num_experts, vector_length), explicit=True):
                            valid = S.vcmps(S.vadds(lanes, j * vector_length), num_experts, op='lt')
                            logit = S.vsel(S.vld(scores_ub[j * vector_length]), zero, valid)
                            nonfinite = S.vcmps(S.vsub(logit, logit), 0.0, op='ne')
                            invalid = S.vmax(invalid, S.vsel(S.vdup(1, T.int32), zeros, nonfinite))
                        S.vsts(invalid_logits_ub[0], S.vcmax(invalid), one)

                        # 载入 vreg，并把越界 lane 掩成 -inf（而不是预先补齐 UB）。
                        for i in T.unroll(num_vregs_per_group, explicit=True):
                            scores_vec[i] = S.vld(scores_ub[i * vector_length])
                            index_vec[i] = S.vci(T.int32(i * vector_length), T.int32)
                        num_exp = S.vdup(T.int32(num_experts), T.int32, mask_32b)
                        for i in T.unroll(num_vregs_per_group, explicit=True):
                            scores_vec[i] = S.vsel(
                                scores_vec[i], neg_inf, S.vcmp(index_vec[i], num_exp, mask_32b, 'lt')
                            )

                        acc_max = S.alloc_var(T.float32)
                        acc_min_idx = S.alloc_var(T.int32)
                        for k in T.serial(num_topk):
                            # 1) 组内 vmax 折叠，再 vcmax 做跨 lane 归约，得到全局最大值。
                            acc_max = scores_vec[0]
                            for i in T.unroll(1, num_vregs_per_group, explicit=True):
                                acc_max = S.vmax(acc_max, scores_vec[i], mask_32b)
                            mx = S.vdupv(S.vcmax(acc_max, mask_32b), mask_32b)

                            # 2) 非并列位置换成 INT_MAX，再 vmin + vcmin 取最小下标（tie-break）。
                            for i in T.unroll(num_vregs_per_group, explicit=True):
                                tmp_idx_vec[i] = S.vsel(
                                    index_vec[i], int_max, S.vcmp(scores_vec[i], mx, mask_32b, 'eq')
                                )
                            acc_min_idx = tmp_idx_vec[0]
                            for i in T.unroll(1, num_vregs_per_group, explicit=True):
                                acc_min_idx = S.vmin(acc_min_idx, tmp_idx_vec[i], mask_32b)
                            idx = S.vdupv(S.vcmin(acc_min_idx, mask_32b), mask_32b)

                            # 3) 单 lane 写第 k 个结果；然后把中奖 lane 掩成 -inf。
                            S.vsts(out_ub[2 * k], idx, one, 'ONEPT_B32')
                            for i in T.unroll(num_vregs_per_group, explicit=True):
                                scores_vec[i] = S.vsel(
                                    neg_inf, scores_vec[i], S.vcmp(index_vec[i], idx, mask_32b, 'eq')
                                )

                    # 低位 + 高位（0）一起搬过去，就是 int64 下标。
                    T.copy(out_ub[: num_topk * 2], topk_idx_view[row, :])

            # 因为 invalid_logits_ub 只有 1 份、被所有迭代共享，断言放在循环之外。
            T.device_assert(invalid_logits_ub[0] == 0, msg='topk_gate input contains nan or inf!')

    return topk_gate_kernel_asc


def topk_gate(scores, num_topk: int, num_vec_cores: int):
    """与 tile_kernels.moe.topk_gate 相同的契约；num_vec_cores 由调用方探测后传入。"""
    import torch

    assert scores.dim() == 2 and scores.is_contiguous() and scores.dtype == torch.float32
    num_tokens, num_experts = scores.shape
    assert num_topk <= num_experts and num_topk <= 32

    out = torch.empty((num_tokens, num_topk), dtype=torch.int64, device=scores.device)
    if num_tokens == 0:
        return out
    get_topk_gate_kernel_asc(num_experts, num_topk, num_vec_cores)(scores, out)
    return out
