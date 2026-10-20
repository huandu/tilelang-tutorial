"""T.gemm 的四种参数组合：transpose_A / transpose_B / policy / clear_accum。

静态核对来源：
  tilelang/language/gemm_op.py:38-122,160-210
    - M/N/K 推导与静态性检查 :77-92
    - clear_accum 改 access region :103
    - 签名（无 accum_dtype）:160-169
  tilelang/tileop/base.py:5-12                (GemmWarpPolicy 三个取值)
  src/cuda/op/gemm.cc:336-395                 (指令选择与 warp 划分)

运行前提：NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang v0.1.15。
作者验证级别：仅静态核对（py_compile + API 逐条比对），未在硬件上运行。
"""

from __future__ import annotations

import torch
import tilelang
import tilelang.language as T


def make_kernel(M: int, N: int, K: int, *, transpose_a: bool, transpose_b: bool,
                policy: T.GemmWarpPolicy, clear_by_mask: bool,
                block_M: int = 128, block_N: int = 128, block_K: int = 64,
                threads: int = 256):
    """A 的形状由 transpose_a 决定；B 的形状由 transpose_b 决定。

    transpose_a=False -> A 是 [M, K]；True -> A 是 [K, M]
    transpose_b=False -> B 是 [K, N]；True -> B 是 [N, K]
    """
    a_shape = (K, M) if transpose_a else (M, K)
    b_shape = (N, K) if transpose_b else (K, N)

    @tilelang.jit(out_idx=-1)
    def build():
        @T.prim_func
        def main(
            A: T.Tensor(a_shape, "float16"),
            B: T.Tensor(b_shape, "float16"),
            C: T.Tensor((M, N), "float32"),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_shared = T.alloc_shared(
                    (block_K, block_M) if transpose_a else (block_M, block_K), "float16"
                )
                B_shared = T.alloc_shared(
                    (block_N, block_K) if transpose_b else (block_K, block_N), "float16"
                )
                C_local = T.alloc_fragment((block_M, block_N), "float32")

                # 不想依赖 clear_accum 时，显式清零一次。
                if not clear_by_mask:
                    T.clear(C_local)

                for ko in T.Pipelined(K // block_K, num_stages=2):
                    if transpose_a:
                        T.copy(A[ko * block_K, by * block_M], A_shared)
                    else:
                        T.copy(A[by * block_M, ko * block_K], A_shared)
                    if transpose_b:
                        T.copy(B[bx * block_N, ko * block_K], B_shared)
                    else:
                        T.copy(B[ko * block_K, bx * block_N], B_shared)

                    # clear_accum 是逐次调用的参数：第一次调用时清零，之后累加。
                    # 注意它在 IR 层把 C 的 access region 从 "rw" 改成 "w"
                    # (gemm_op.py:103)，因此不是纯运行期开关。
                    T.gemm(
                        A_shared,
                        B_shared,
                        C_local,
                        transpose_A=transpose_a,
                        transpose_B=transpose_b,
                        policy=policy,
                        clear_accum=clear_by_mask and (ko == 0),
                    )

                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return build


def main() -> None:
    M = N = K = 512
    variants = [
        # (transpose_a, transpose_b, policy, clear_by_mask, 名字)
        (False, False, T.GemmWarpPolicy.Square, False, "nn-square"),
        (False, True, T.GemmWarpPolicy.FullRow, True, "nt-fullrow"),
        (True, False, T.GemmWarpPolicy.FullCol, True, "tn-fullcol"),
        (True, True, T.GemmWarpPolicy.Square, True, "tt-square"),
    ]
    for transpose_a, transpose_b, policy, clear_by_mask, name in variants:
        kernel = make_kernel(
            M, N, K,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
            policy=policy,
            clear_by_mask=clear_by_mask,
        )()
        a = torch.randn((K, M) if transpose_a else (M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((N, K) if transpose_b else (K, N), device="cuda", dtype=torch.float16)
        c = kernel(a, b)
        ref = (a.T if transpose_a else a).float() @ (b.T if transpose_b else b).float()
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
        print(f"{name}: passed")


if __name__ == "__main__":
    main()
