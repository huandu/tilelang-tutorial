#!/usr/bin/env python3
"""第 14 章第 5 节：四类低精度 GEMM 的**结构**与调用方式。

  gemm_fp8          e4m3 / e5m2，fp32 累加
  gemm_int4         int4 打包输入，int32 累加
  gemm_blockscaled  fp8 + SFA/SFB，scale 按 K 块施加
  dequantize 的搬运-反量化模式见 README（它没有独立的 T.gemm 变体）

数值正确性（误差传播、舍入、scale 表示）留给第 27 章；这里只讲结构。

运行前提：NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang v0.1.15。
fp8 需要 sm89+；block-scaled 的公开示例跑在 sm100 上（本文件不依赖它，
只用通用接口，能否编译取决于后端是否支持该 dtype 组合）。
写作机上没有 GPU，本文件只做过静态核对（py_compile + API 比对），**未运行**。
"""

from __future__ import annotations

import tilelang
import tilelang.language as T
from tilelang.language.fp8 import determine_fp8_type


@tilelang.jit
def gemm_fp8(A, B, block_M, block_N, block_K, dtype, accum_dtype=T.float32):
    """fp8：结构上和 fp16 一模一样，只是 dtype 变了、K 对齐更严。"""
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((N, K), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_N, block_K), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def matmul_int4(M, N, K, block_M, block_N, block_K, threads=128):
    """int4：A/B 声明成 T.int4，累加器是 T.int32；输入需按两条 nibble 打包。"""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.int4),
        B: T.Tensor((N, K), T.int4),
        C: T.Tensor((M, N), T.int32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.int4)
            B_shared = T.alloc_shared((block_N, block_K), T.int4)
            C_local = T.alloc_fragment((block_M, block_N), T.int32)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def pack_int4(tensor):
    """把逻辑 int4（用 int8 表示）打包成每字节两个 nibble，与官方示例一致。"""
    import torch  # 只在这个 helper 里用到 torch

    if tensor.dtype != torch.int8:
        raise TypeError("期望 torch.int8 表示的逻辑 int4 张量")
    if tensor.ndim == 0 or tensor.shape[-1] % 2 != 0:
        raise ValueError("逻辑 int4 张量的最后一维必须是偶数")
    t16 = tensor.to(torch.int16)
    packed = (t16[..., ::2] & 0x0F) | ((t16[..., 1::2] & 0x0F) << 4)
    return packed.to(torch.int8).contiguous()


@tilelang.jit
def gemm_blockscaled(A, B, SFA, SFB, block_M, block_N, block_K, sf_granularity_k,
                     in_dtype, out_dtype, accum_dtype=T.float32, num_stages=3):
    """block-scaled：多一对 SFA/SFB，scale 沿 K 以块为单位施加。

    参数名与语义对照 `T.gemm_blockscaled(A, B, C, SFA, SFB, k_start=...,
    sf_a_granularity_k=..., sf_b_granularity_k=...)`。
    这里只保留结构：真实实现的 scale 打包方式与粒度由后端决定
    （sm100 的 MXFP8 用 128，昇腾 MX 用 32）。
    """
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), in_dtype)
    B: T.Tensor((N, K), in_dtype)
    C = T.empty((M, N), out_dtype)
    SFA: T.Tensor((M, T.ceildiv(K, sf_granularity_k)), accum_dtype)
    SFB: T.Tensor((N, T.ceildiv(K, sf_granularity_k)), accum_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), in_dtype)
        B_shared = T.alloc_shared((block_N, block_K), in_dtype)
        SFA_shared = T.alloc_shared((block_M, block_K // sf_granularity_k), accum_dtype)
        SFB_shared = T.alloc_shared((block_N, block_K // sf_granularity_k), accum_dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K], B_shared)
            T.copy(SFA[by * block_M, k * (block_K // sf_granularity_k)], SFA_shared)
            T.copy(SFB[bx * block_N, k * (block_K // sf_granularity_k)], SFB_shared)
            T.gemm_blockscaled(
                A_shared, B_shared, C_local, SFA_shared, SFB_shared,
                transpose_B=True, k_start=k * block_K,
                sf_a_granularity_k=sf_granularity_k,
                sf_b_granularity_k=sf_granularity_k,
            )

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def main() -> int:
    import torch

    m, n, k = 1024, 1024, 1024

    # fp8：e4m3
    dtype = determine_fp8_type()
    torch_dtype = T.dtype(dtype).as_torch()
    a = torch.randn(m, k, dtype=torch.float16, device="cuda").to(torch_dtype)
    b = torch.randn(n, k, dtype=torch.float16, device="cuda").to(torch_dtype)
    c = gemm_fp8(a, b, 128, 128, 64, dtype)
    ref = (a.half() @ b.half().T).to(torch_dtype)
    print("fp8 dtype:", dtype, "| 结构检查完成（未评估数值误差，见第 27 章）")
    _ = ref
    _ = c

    # int4：打包
    logical = torch.randint(-8, 8, (m, k), dtype=torch.int8, device="cuda")
    packed = pack_int4(logical)
    print("int4 打包后形状:", tuple(packed.shape), "（最后一维减半）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
