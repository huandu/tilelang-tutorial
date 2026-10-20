"""hc_prenorm 的 TileLang 骨架（提炼自 DeepGEMM-Ascend，仅静态核对）。

原文件：DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py
          （0.1.0 快照，commit 8491bbb4b8c0）

算法（原文件 :367-377）：
    D       = float(A) @ B.T                 A:[M,K] bf16, B:[N,K] fp32, D:[M,N] fp32
    sqr_sum = sum(float(A) ** 2, dim=-1)     sqr_sum:[M] fp32

本文件把「结构」抽出来加注释，**不是可运行的替代品**；真机请调用
`deep_gemm.tf32_hc_prenorm_gemm`。验证级别：仅静态核对（见同目录 README）。
"""

import math

import torch
import tilelang
from tilelang.ascend import language as T
from tilelang.layout import make_ascend_compact_nz_layout

from deep_gemm import _C  # 只为拿 get_num_sms()，原文件 :26 用的也是这个接口


# ---------------------------------------------------------------------------
# 1) get_config：三个「2 的幂」约束取最小，决定 K 方向切几份
#    原文件 hc_prenorm.py:13-57
# ---------------------------------------------------------------------------
def get_config(mhc_hidden_size: int, mhc_mult3: int, num_tokens: int) -> dict:
    block_num_tokens = 128
    block_hidden = 256
    l0b_size_bytes = 64 * 1024
    l0c_size_bytes = 256 * 1024            # Ascend 950 口径（common.hpp:57）

    assert 0 < mhc_mult3 <= 64 and mhc_mult3 % 8 == 0
    assert mhc_hidden_size > 0 and mhc_hidden_size % block_hidden == 0

    num_aic_cores = _C.get_num_sms()
    block_mhc_mult3 = (mhc_mult3 + 15) // 16 * 16   # 物理对齐到 16、逻辑仍是 mhc_mult3
    num_hidden_blocks = mhc_hidden_size // block_hidden
    num_token_blocks = math.ceil(num_tokens / block_num_tokens)
    num_l0b_stages = l0b_size_bytes // (block_mhc_mult3 * block_hidden * 4)
    num_l0c_stages = l0c_size_bytes // (block_num_tokens * block_mhc_mult3 * 4)

    # 「x & -x」取出一个数里最大的 2 的幂因子，保证切分不产生尾块。
    split_limit = min(
        max(1, num_aic_cores * num_l0c_stages // num_token_blocks),  # token 并行度够
        num_hidden_blocks & -num_hidden_blocks,                      # 整除 K 方向块数
        num_aic_cores & -num_aic_cores,                              # 整除核数
    )
    num_k_splits = 1 << (split_limit.bit_length() - 1)
    active_aic_cores = num_k_splits * min(num_aic_cores // num_k_splits, num_token_blocks)
    cores_per_split = active_aic_cores // num_k_splits
    return {
        'block_mhc_mult3': block_mhc_mult3,
        'num_aic_cores': active_aic_cores,
        'num_k_splits': num_k_splits,
        'cores_per_split': cores_per_split,
        'num_hidden_blocks_per_split': num_hidden_blocks // num_k_splits,
        'num_l0b_stages': num_l0b_stages,
        'num_l0c_stages': num_l0c_stages,
        'num_token_blocks_per_chunk': cores_per_split * num_l0c_stages,
    }


# ---------------------------------------------------------------------------
# 2) kernel 骨架
#    原文件 hc_prenorm.py:59-359
# ---------------------------------------------------------------------------
@tilelang.jit
def hc_prenorm_sketch_kernel(
    x, fn, out_mul, sqrsum,
    block_mhc_mult3: int, num_aic_cores: int, num_k_splits: int,
    cores_per_split: int, num_hidden_blocks_per_split: int,
    num_l0b_stages: int, num_l0c_stages: int, num_token_blocks_per_chunk: int,
    deterministic: bool = False,
):
    block_num_tokens, block_hidden, mad_hidden = 128, 256, 64
    num_aivs = 2
    half_block_num_tokens = block_num_tokens // num_aivs
    num_mads_per_block = block_hidden // mad_hidden
    num_load_stages, num_cast_stages, num_l1_x_stages, num_mad_stages = 3, 2, 3, 2

    num_tokens = T.dynamic('num_tokens')
    mhc_mult3 = T.const('mhc_mult3')
    mhc_hidden_size = T.const('mhc_hidden_size')

    x: T.Tensor[(num_tokens, mhc_hidden_size), T.bfloat16]
    fn: T.Tensor[(mhc_mult3, mhc_hidden_size), T.float32]
    out_mul: T.Tensor[(num_tokens, mhc_mult3), T.float32]
    sqrsum: T.Tensor[(num_tokens,), T.float32]

    # 非确定性模式下：相位 0 让 split 0 普通写，其余 split 再原子加。
    num_store_phases = num_k_splits if deterministic else (2 if num_k_splits > 1 else 1)
    nz_stage_rows = half_block_num_tokens + 1

    with T.MixedKernel(num_aic_cores) as (core_id, aiv_id):
        split_id = core_id // cores_per_split
        core_in_split = core_id % cores_per_split
        num_token_blocks = T.ceildiv(num_tokens, block_num_tokens)
        num_token_chunks = T.ceildiv(num_token_blocks, num_token_blocks_per_chunk)

        # --- 片上缓冲：UB / L1 / L0A / L0B / L0C --------------------------------
        x_ub = T.alloc_shared((half_block_num_tokens, block_hidden), T.bfloat16)
        x_nz_ub = T.alloc_shared((nz_stage_rows, block_hidden), T.float32)
        T.annotate_layout({x_nz_ub: make_ascend_compact_nz_layout(x_nz_ub)})
        sqrsum_ub = T.alloc_shared((num_l0c_stages, half_block_num_tokens), T.float32)
        x_l1 = T.alloc_l1((block_num_tokens, block_hidden), T.float32)
        fn_l1 = T.alloc_l1((block_mhc_mult3, block_hidden), T.float32)
        x_l0 = T.alloc_l0a((block_num_tokens, mad_hidden), T.float32)
        fn_l0 = T.alloc_l0b((num_mads_per_block, block_mhc_mult3, mad_hidden), T.float32)
        out_mul_l0 = T.alloc_l0c((num_l0c_stages, block_num_tokens, block_mhc_mult3),
                                 T.float32)

        T.annotate_buffer_versions({
            x_ub: num_load_stages, x_nz_ub: num_cast_stages, x_l1: num_l1_x_stages,
            fn_l1: num_l0b_stages, x_l0: num_mad_stages, fn_l0: num_l0b_stages,
        })

        # K-split 合并：只有 split 0 不做原子加。
        if split_id != 0:
            T.set_atomic('add', 'float32')
        T.set_hf32_mode('nearest_even')

        for chunk_idx in T.Serial(num_token_chunks):
            token_block_base = chunk_idx * num_token_blocks_per_chunk + core_in_split
            num_valid_token_stages = T.min(
                num_l0c_stages,
                T.max(T.ceildiv(num_token_blocks - token_block_base, cores_per_split), 0),
            )

            # 隐藏维流水：每块 256 个 K 元素；权重沿 GM→L1→L0B 走一次。
            for local_hidden_block in T.Pipelined(num_hidden_blocks_per_split, num_stages=2):
                global_hidden_block = split_id * num_hidden_blocks_per_split + local_hidden_block
                hidden_begin = global_hidden_block * block_hidden

                # 物理 buffer 按 block_mhc_mult3 对齐，逻辑只搬 mhc_mult3 行。
                T.copy(fn[0:mhc_mult3, hidden_begin:hidden_begin + block_hidden],
                       fn_l1[0:mhc_mult3, :])
                for hidden_mad in T.Serial(num_mads_per_block):
                    T.copy(fn_l1[0:mhc_mult3,
                                 hidden_mad * mad_hidden:(hidden_mad + 1) * mad_hidden],
                           fn_l0[hidden_mad, 0:mhc_mult3, :])

                for token_stage in T.Serial(
                    num_valid_token_stages,
                    annotations={'multi_buffer_eligible': [x_l1]},
                ):
                    token_block = token_block_base + token_stage * cores_per_split
                    token_begin = token_block * block_num_tokens
                    aiv_token_begin = token_begin + aiv_id * half_block_num_tokens
                    actual_aiv = T.min(num_tokens - aiv_token_begin, half_block_num_tokens)

                    if aiv_token_begin < num_tokens:
                        T.copy(x[aiv_token_begin:aiv_token_begin + actual_aiv,
                                 hidden_begin:hidden_begin + block_hidden],
                               x_ub[0:actual_aiv, :],
                               l2_cache_ctrl='notalloc_keep', pad_value=0)

                        # 一趟遍历做完三件事：bf16→fp32 展宽、NZ 打包、行平方和。
                        with T.SimdVF(latency=640):
                            f32_mask = T.simd.pset(32)
                            nz_stride = T.int32((nz_stage_rows << 16) | (8 * nz_stage_rows))
                            zero_bf16 = T.simd.vdup(T.bfloat16(0), 'bfloat16', T.simd.pset(16))
                            for row in T.Serial(half_block_num_tokens):
                                row_acc = T.simd.alloc_var('float32')
                                row_acc = T.simd.vdup(T.float32(0), 'float32', f32_mask)
                                nz_ptr = T.simd.make_ubuf_ptr(
                                    T.access_ptr(x_nz_ub[row, 0], 'w', 1, block_hidden),
                                    'float32')
                                for pass_id in T.Serial(block_hidden // 128):
                                    packed = T.simd.vld(x_ub[row, pass_id * 128])
                                    lo_bits, hi_bits = T.simd.vintlv(zero_bf16, packed)
                                    lo = T.reinterpret(lo_bits, 'float32x64')
                                    hi = T.reinterpret(hi_bits, 'float32x64')
                                    row_acc = T.simd.vadd(row_acc,
                                                          T.simd.vmul(lo, lo, f32_mask),
                                                          f32_mask)
                                    row_acc = T.simd.vadd(row_acc,
                                                          T.simd.vmul(hi, hi, f32_mask),
                                                          f32_mask)
                                    nz_ptr = T.simd.vsstb(lo, nz_ptr, nz_stride, f32_mask,
                                                          update=True)
                                    nz_ptr = T.simd.vsstb(hi, nz_ptr, nz_stride, f32_mask,
                                                          update=True)

                        # UB → L1：劈成两个 AIV 的半块（dual_copy 的语义见第 25 章）。
                        T.dual_copy(x_nz_ub[0:half_block_num_tokens, 0:block_hidden],
                                    x_l1[0:block_num_tokens, 0:block_hidden])

                    # L1 → L0A，然后一条 MAD 出 L0C。
                    for hidden_mad in T.Serial(num_mads_per_block):
                        T.copy(x_l1[0:block_num_tokens,
                                    hidden_mad * mad_hidden:(hidden_mad + 1) * mad_hidden],
                               x_l0[0:block_num_tokens, :])
                        T.gemm(x_l0[0:block_num_tokens, :],
                               fn_l0[hidden_mad, 0:mhc_mult3, :],
                               out_mul_l0[token_stage, 0:block_num_tokens, 0:mhc_mult3],
                               transpose_B=True,
                               clear_accum=(local_hidden_block == 0 and hidden_mad == 0),
                               unit_flag_ctrl=T.Select(
                                   local_hidden_block == num_hidden_blocks_per_split - 1
                                   and hidden_mad == num_mads_per_block - 1, 3, 2))

            # store 阶段放在 T.PerCoreTask 里，靠核间同步串起来（原文件 :294-356）。
            with T.PerCoreTask():
                for store_phase in T.Serial(num_store_phases):
                    store_this_phase = T.if_then_else(
                        deterministic, split_id == store_phase,
                        T.if_then_else(store_phase == 0, split_id == 0, split_id != 0))
                    out_mul_flag = (chunk_idx * num_store_phases + store_phase) % 4
                    if store_this_phase:
                        with T.Task():
                            for token_stage in T.Serial(num_valid_token_stages):
                                token_begin = (token_block_base
                                               + token_stage * cores_per_split) * block_num_tokens
                                actual = T.min(num_tokens - token_begin, block_num_tokens)
                                T.copy(out_mul_l0[token_stage, 0:actual, 0:mhc_mult3],
                                       out_mul[token_begin:token_begin + actual, 0:mhc_mult3],
                                       unit_flag_ctrl=3)
                    num_store_syncs = T.if_then_else(
                        store_phase < (num_store_phases if deterministic else 1), 1,
                        T.min(num_token_chunks - chunk_idx - 1,
                              0 if deterministic or num_k_splits == 1 else 1))
                    for _ in T.Serial(num_store_syncs):
                        T.ascend_sync_inter_arrive('PIPE_FIX', out_mul_flag)
                        T.ascend_sync_inter_wait('PIPE_FIX', out_mul_flag)

        if split_id != 0:
            T.set_atomic_none()


if __name__ == '__main__':
    # 本骨架只用于阅读；这里仅演示配置怎么算出来（原文件 :392-397 的调用路径）。
    print('提示：真机上请调用 deep_gemm.tf32_hc_prenorm_gemm(a, b, d, sqr_sum)。')
    print('本机没有 NPU，未编译、未运行。')

