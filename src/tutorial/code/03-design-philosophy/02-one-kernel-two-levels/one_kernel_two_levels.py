#!/usr/bin/env python3
"""第 03 章 示例 2：同一个 kernel 里的两层（composable 的最小实证）。

这个文件只做一件事：把一个 kernel 写成「developer 层的 GEMM + expert 层的线程级部分和」，
并在注释里逐行标出每句属于哪一层。

依赖：tilelang v0.1.15、torch(CUDA)、一张 NVIDIA GPU。
无 GPU 时用 `--static-only`，只做语法与 API 核对。

作者验证级别：仅静态核对（见同目录 README.md）。
"""

from __future__ import annotations

import argparse
import sys

# --- 静态核对清单：每个 API -> (层次, 在锁定版本里的定义位置) ------------------
API_CHECKLIST: tuple[tuple[str, str, str], ...] = (
    ("T.Kernel", "developer", "tilelang/language/kernel.py:431"),
    ("T.alloc_shared", "developer", "tilelang/language/allocate.py:49"),
    ("T.alloc_fragment", "developer", "tilelang/language/allocate.py:81"),
    ("T.alloc_local", "expert", "tilelang/language/allocate.py:67"),
    ("T.clear", "developer", "tilelang/language/fill_op.py:48"),
    ("T.copy", "developer", "tilelang/language/copy_op.py:59"),
    ("T.gemm", "developer", "tilelang/language/gemm_op.py:160"),
    ("T.Pipelined", "developer", "tilelang/language/loop.py:112"),
    ("T.get_thread_binding", "expert", "tilelang/language/kernel.py:587"),
    ("T.serial", "expert", "tilelang/language/loop.py:189"),
)


def build_kernel(block_M: int = 64, block_N: int = 64, block_K: int = 32, num_stages: int = 2):
    """构造 kernel（lazy 模式：返回 PrimFunc）。导入被推迟到调用时。"""
    import tilelang
    import tilelang.language as T

    @tilelang.jit(out_idx=[2, 3])
    def mixed_level_gemm(M: int, N: int, K: int, threads: int = 128):
        n_tiles_n = T.ceildiv(N, block_N)
        n_tiles_m = T.ceildiv(M, block_M)

        @T.prim_func
        def main(
            A: T.Tensor((M, K), "float16"),
            B: T.Tensor((K, N), "float16"),
            C: T.Tensor((M, N), "float16"),
            Partial: T.Tensor((n_tiles_n, n_tiles_m, threads), "float32"),
        ):
            with T.Kernel(n_tiles_n, n_tiles_m, threads=threads) as (bx, by):
                # ================= developer 层：tile 库 =================
                # 全程没有线程索引，也没有同步原语。
                A_s = T.alloc_shared((block_M, block_K), "float16")
                B_s = T.alloc_shared((block_K, block_N), "float16")
                C_frag = T.alloc_fragment((block_M, block_N), "float32")
                C_s = T.alloc_shared((block_M, block_N), "float16")

                T.clear(C_frag)
                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_s)
                    T.copy(B[ko * block_K, bx * block_N], B_s)
                    T.gemm(A_s, B_s, C_frag)
                T.copy(C_frag, C_s)  # fragment -> shared，协作写入

                # ================= expert 层：thread 原语 =================
                # 同一个 kernel 体里直接下探到线程级，没有任何模式切换。
                # 这一段只用来演示「可以混用」，不参与 C 的计算。
                tx = T.get_thread_binding(0)  # tx = threadIdx.x
                part = T.alloc_local((1,), "float32")  # 线程私有
                part[0] = 0
                for jr in T.serial(T.ceildiv(block_N, threads)):
                    j = jr * threads + tx
                    if j < block_N:
                        part[0] += C_s[0, j]
                Partial[bx, by, tx] = part[0]  # 每个线程写自己的槽位，无竞争

                # ================= 回到 developer 层 =================
                T.copy(C_frag, C[by * block_M, bx * block_N])

        return main

    return mixed_level_gemm


def static_only() -> int:
    print("第 03 章示例 2：同一个 kernel 里的两层（仅静态核对）")
    print("-" * 56)
    print("本文件在无 GPU 时只做语法与 API 核对，不能编译/运行 kernel。")
    print()
    print("用到的 API 与层次归属（v0.1.15）：")
    for name, layer, where in API_CHECKLIST:
        print(f"  {name:<22} [{layer:<9}] {where}")
    print()
    print("层次分布：developer", sum(1 for _, l, _ in API_CHECKLIST if l == "developer"),
          "个；expert", sum(1 for _, l, _ in API_CHECKLIST if l == "expert"), "个。")
    print("两者出现在同一个 T.Kernel 体内 —— 这就是 composable。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--static-only", action="store_true", help="只打印 API 核对清单，不编译 kernel")
    parser.add_argument("--M", type=int, default=512)
    parser.add_argument("--N", type=int, default=512)
    parser.add_argument("--K", type=int, default=512)
    args = parser.parse_args()

    if args.static_only:
        return static_only()

    try:
        import torch  # noqa: F401  （仅为提前给出更清楚的错误信息）
        import tilelang  # noqa: F401
    except ImportError as exc:
        print(f"缺少依赖：{exc}", file=sys.stderr)
        print("本示例需要 NVIDIA GPU + CUDA 版 PyTorch + tilelang v0.1.15。", file=sys.stderr)
        print("无 GPU 时请改用：python3 one_kernel_two_levels.py --static-only", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("检测不到 CUDA 设备。", file=sys.stderr)
        print("无 GPU 时请改用：python3 one_kernel_two_levels.py --static-only", file=sys.stderr)
        return 3

    torch.manual_seed(0)
    M, N, K = args.M, args.N, args.K
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)

    kernel = build_kernel()(M, N, K, threads=128)
    c, partial = kernel(a, b)

    torch.testing.assert_close(c.float(), (a @ b).float(), rtol=1e-2, atol=1e-2)
    print("GEMM 输出与 PyTorch 参考一致。")

    # Partial[0, 0, :] 的和应等于 C 第一个 (block_M, block_N) 分块第一行的和。
    tile_row = c[0, :64].float()
    print(f"分块第一行和 = {tile_row.sum().item():.4f}，"
          f"线程部分和之和 = {partial[0, 0, :].sum().item():.4f}"
          "（本示例不把这两者当作断言，仅作为人工核对提示）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
