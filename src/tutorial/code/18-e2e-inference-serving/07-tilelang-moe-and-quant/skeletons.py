#!/usr/bin/env python3
"""第 18 章 · fused MoE 与量化 GEMM 的 TileLang 骨架。

验证级别：**仅静态核对**（本机没有 GPU / NPU，从未编译或运行）。

两个骨架：

1. `moe_routed_kernel`：MoE 的路由专家部分，结构对照
   `_vendor/tilelang-v0.1.15/examples/fusedmoe/example_fusedmoe_tilelang.py`
   的 `moe_forward_tilelang_routed`（`:92-226`）。重点是 **dispatch 契约**：
   kernel 不自己做 sort/bincount，而是从 host 拿四张表，按 block 反查自己
   该处理哪个专家的哪一段行。
2. `w4a8_dequant_gemm`：权重 int4、激活 int8 的反量化 GEMM，结构对照
   `examples/dequantize_gemm/example_dequant_gemm_w4a8.py:86-148`。
   重点是 **反量化发生在什么时候**：把打包的 int4 权重搬进片上，
   在 fragment 里解包并转成计算精度，再喂给 `T.gemm`。

量化推理的三种接法（本章第 5 节）在这里都有落点：

* **weight-only**（W4A16/W8A16）：权重解包，激活保持高精度 —— 骨架 2 的
  `in_dtype="int8"` 换成 `"float16"` 就是这一路；
* **weight + activation**（W4A8/W8A8）：两侧都量化，kernel 内各自反量化；
* **block-scaled**（MXFP4/MXFP8）：`T.gemm_blockscaled(..., SFA, SFB, ...)`
  （`tilelang/language/gemm_op.py:275`），scale 的粒度是新的一个维度。
"""

from __future__ import annotations

import tilelang
import tilelang.language as T
from tvm import tirx


def _tir_u8_to_i4_to_i8(nbit: int, val, pos, dtype: str):
    """把 uint8 里的一个 4 bit 字段解成 int8（带符号扩展）。

    与 `example_dequant_gemm_w4a8.py:10-24` 的 `_tir_u8_to_i4_to_i8` 同构。
    """
    assert nbit == 4 and dtype == T.int8 and val.dtype == T.uint8
    mask = tirx.const((1 << nbit) - 1, T.uint8)
    i4 = (val >> (pos.astype(T.uint8) * tirx.const(nbit, T.uint8))) & mask
    i8_shifted = tirx.reinterpret(T.int8, i4 << tirx.const(4, T.uint8))
    return i8_shifted >> tirx.const(4, T.int8)


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def moe_routed_kernel(
    d_hidden: int,
    d_expert: int,
    n_routed_experts: int,
    group_sum: int,
    block_token: int = 128,
    block_dhidden: int = 128,
    block_dexpert: int = 128,
    threads: int = 256,
    num_stages: int = 1,
    dtype: str = "float16",
):
    """路由专家的 grouped GEMM（gate/up → SiLU → down → 加权写回）。

    ``group_sum`` 是 (token, expert) 对的总数，也就是
    ``batch * seq_len * n_experts_per_token``——它是**编译期常量**，
    decode 每步都在变，所以真实框架要按 batch 分桶编译（第 7 章的 JIT 缓存）。
    """
    group_count = n_routed_experts
    m_blocks = (group_sum + block_token - 1) // block_token + group_count
    accum_dtype = "float32"
    scale = 1.44269504  # log2(e)

    input_shape = (group_sum, d_hidden)
    gate_shape = (n_routed_experts, d_expert, d_hidden)
    up_shape = (n_routed_experts, d_expert, d_hidden)
    down_shape = (n_routed_experts, d_hidden, d_expert)

    @T.prim_func
    def moe_routed(
        X: T.Tensor(input_shape, dtype),
        W_gate: T.Tensor(gate_shape, dtype),
        W_up: T.Tensor(up_shape, dtype),
        W_down: T.Tensor(down_shape, dtype),
        RowWeights: T.Tensor((group_sum,), dtype),
        group_sizes: T.Tensor((group_count,), T.int32),
        group_offsets: T.Tensor((group_count,), T.int32),
        group_padded_offsets: T.Tensor((group_count,), T.int32),
        group_idx_for_bx: T.Tensor((m_blocks,), T.int32),
        UpLogits: T.Tensor((group_sum, d_expert), dtype),
        Output: T.Tensor(input_shape, dtype),
    ):
        # ---- gate + up，融合 SiLU 与逐元素乘 ----
        with T.Kernel(m_blocks, T.ceildiv(d_expert, block_dexpert),
                      threads=threads) as (bx, by):
            x_shared = T.alloc_fragment((block_token, block_dhidden), dtype)
            w_gate_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype)
            w_up_shared = T.alloc_shared((block_dexpert, block_dhidden), dtype)
            gate_local = T.alloc_fragment((block_token, block_dexpert), accum_dtype)
            up_local = T.alloc_fragment((block_token, block_dexpert), accum_dtype)

            # 这是 dispatch 契约的 kernel 侧：从三张表还原出「我在哪、处理谁」
            cur_group = group_idx_for_bx[bx]
            cur_size = group_sizes[cur_group]
            m_start_padded = bx * block_token
            m_start = (m_start_padded - group_padded_offsets[cur_group]
                       + group_offsets[cur_group])
            actual_rows = T.max(0, T.min(block_token,
                                         cur_size - (m_start_padded
                                                     - group_padded_offsets[cur_group])))
            T.clear(gate_local)
            T.clear(up_local)
            for k in T.Pipelined(T.ceildiv(d_hidden, block_dhidden),
                                 num_stages=num_stages):
                T.copy(X[m_start, k * block_dhidden], x_shared)
                T.copy(W_gate[cur_group, by * block_dexpert,
                              k * block_dhidden], w_gate_shared)
                T.copy(W_up[cur_group, by * block_dexpert,
                            k * block_dhidden], w_up_shared)
                T.gemm(x_shared, w_gate_shared, gate_local, transpose_B=True)
                T.gemm(x_shared, w_up_shared, up_local, transpose_B=True)
            for i, j in T.Parallel(block_token, block_dexpert):
                # SiLU(gate) * up，同一个 kernel 里做完，省一趟 HBM
                gate_local[i, j] = gate_local[i, j] * (
                    1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                up_local[i, j] = up_local[i, j] * gate_local[i, j]
            for i, j in T.Parallel(block_token, block_dexpert):
                if i < actual_rows:
                    UpLogits[m_start + i, by * block_dexpert + j] = up_local[i, j]

        # ---- down + 按路由权重写回 ----
        with T.Kernel(m_blocks, T.ceildiv(d_hidden, block_dhidden),
                      threads=threads) as (bx, by):
            up_shared = T.alloc_fragment((block_token, block_dexpert), dtype)
            w_down_shared = T.alloc_shared((block_dhidden, block_dexpert), dtype)
            out_local = T.alloc_fragment((block_token, block_dhidden), accum_dtype)

            cur_group = group_idx_for_bx[bx]
            cur_size = group_sizes[cur_group]
            m_start_padded = bx * block_token
            m_start = (m_start_padded - group_padded_offsets[cur_group]
                       + group_offsets[cur_group])
            actual_rows = T.max(0, T.min(block_token,
                                         cur_size - (m_start_padded
                                                     - group_padded_offsets[cur_group])))
            T.clear(out_local)
            for k in T.Pipelined(T.ceildiv(d_expert, block_dexpert),
                                 num_stages=num_stages):
                T.copy(UpLogits[m_start, k * block_dexpert], up_shared)
                T.copy(W_down[cur_group, by * block_dhidden,
                              k * block_dexpert], w_down_shared)
                T.gemm(up_shared, w_down_shared, out_local, transpose_B=True)
            for i, j in T.Parallel(block_token, block_dhidden):
                if i < actual_rows:
                    # combine 的权重乘法放在这里，避免再开一个 kernel
                    Output[m_start + i, by * block_dhidden + j] = (
                        out_local[i, j] * RowWeights[m_start + i])

    return moe_routed


@tilelang.jit(out_idx=[2])
def w4a8_dequant_gemm(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 256,
    num_stages: int = 2,
    threads: int = 128,
    num_bits: int = 4,
    in_dtype: str = "int8",
    out_dtype: str = "int32",
    accum_dtype: str = "int32",
):
    """W4A8：权重 int4（打包进 uint8）、激活 int8、int32 累加。

    ``Ct`` 的形状是 ``(N, M)``（与官方示例一致，转置写回）。
    """
    num_elems_per_byte = 8 // num_bits
    storage_dtype = "uint8"
    B_shape = (N, K // num_elems_per_byte)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K // num_elems_per_byte)
    B_dequant_shape = (block_N, block_K)

    @T.prim_func
    def matmul_int8xint4(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor(B_shape, storage_dtype),
        Ct: T.Tensor((N, M), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, storage_dtype)
            B_local = T.alloc_fragment(B_shared_shape, storage_dtype)
            B_dequant = T.alloc_fragment(B_dequant_shape, in_dtype)
            B_dequant_prev = T.alloc_fragment(B_dequant_shape, in_dtype)
            Ct_local = T.alloc_fragment((block_N, block_M), accum_dtype)
            Ct_shared = T.alloc_shared((block_N, block_M), out_dtype)

            T.annotate_layout(
                {B_shared: tilelang.layout.make_swizzled_layout(B_shared)}
            )
            T.clear(Ct_local)
            for k in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N,
                         k * block_K // num_elems_per_byte], B_shared)
                T.copy(B_shared, B_local)
                # 反量化：解包 + 符号扩展，全部在片上做
                for i, j in T.Parallel(block_N, block_K):
                    B_dequant[i, j] = _tir_u8_to_i4_to_i8(
                        num_bits,
                        B_local[i, j // num_elems_per_byte],
                        j % num_elems_per_byte,
                        dtype=in_dtype,
                    )
                # 再拷一份给 GEMM 用：下轮循环会覆盖 B_dequant
                T.copy(B_dequant, B_dequant_prev)
                T.gemm(B_dequant_prev, A_shared, Ct_local, transpose_B=True)
            T.copy(Ct_local, Ct_shared)
            T.copy(Ct_shared, Ct[bx * block_N:(bx + 1) * block_N,
                                 by * block_M:(by + 1) * block_M])

    return matmul_int8xint4


#: 与官方示例 `example_dequant_gemm_w4a8.py` 的默认形状同量级。
SERVING_SHAPES = {
    # decode 单步：M = batch，N/K 是权重矩阵的一行
    "gemv_like": {"M": 1, "N": 3584, "K": 18944},
    # 小 batch：仍然是 memory-bound（见 03-decode-step-share 的账本）
    "small_batch": {"M": 32, "N": 3584, "K": 18944},
}


if __name__ == "__main__":
    raise SystemExit(
        "本文件只做静态核对：请在装有 NVIDIA GPU + TileLang v0.1.15 的机器上\n"
        "参照 code/07-tilelang-moe-and-quant/README.md 运行。"
    )
