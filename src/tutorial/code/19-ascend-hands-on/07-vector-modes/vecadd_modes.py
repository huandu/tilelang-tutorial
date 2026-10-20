#!/usr/bin/env python3
"""第 19 章 §7.4 / §7.5：M 树昇腾方言的两种 Vector 写法 + buffer 版本注解。

来源：`_vendor/tilelang-v0.1.15/examples/ascend/example_vecadd.py`（保留了
SIMT / SIMD 两种实现与 T.annotate_buffer_versions 的用法，去掉了回归测试脚手架）。

两种写法：
  - simt：with T.SimtVF(threads=N) + T.Parallel(...)   —— 线程域由编译器映射
  - simd：with T.SimdVF() + 显式 T.simd.* 指令         —— 2048-bit 寄存器级向量

运行前提：Ascend 950 NPU + CANN + torch_npu + 主干 TileLang v0.1.15。
验证级别：仅静态核对（py_compile + API 比对），未在 NPU 上运行。
"""

from __future__ import annotations

import argparse

import torch
import tilelang
import tilelang.ascend.language as T


def vector_add(N: int, mode: str = "simt"):
    """C = A + B（一维），用 SIMT 或 SIMD 实现，并开三级多缓冲。"""
    if mode not in ("simt", "simd"):
        raise ValueError("mode must be 'simt' or 'simd'")
    num_blocks, num_threads, tile, num_stages = 64, 2048, 8192, 2
    if N <= 0 or N % (tile * num_blocks) != 0:
        raise ValueError(f"N must be a positive multiple of {tile * num_blocks}")
    iterations = N // (tile * num_blocks)

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
        C: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(num_blocks) as bx:
            # 注意：M 树的昇腾 T.Kernel 不接受 threads=；线程域在 T.SimtVF 里。
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")
            # 告诉 AutoSchedule：这三个 UB buffer 各要 num_stages 个物理版本。
            T.annotate_buffer_versions({a: num_stages, b: num_stages, c: num_stages})
            for step in T.Pipelined(iterations, num_stages=num_stages):
                begin = (step * num_blocks + bx) * tile
                T.copy(A[begin : begin + tile], a)
                T.copy(B[begin : begin + tile], b)
                if mode == "simt":
                    with T.SimtVF(threads=num_threads):
                        for i in T.Parallel(tile):
                            c[i] = a[i] + b[i]
                else:
                    with T.SimdVF():
                        mask = T.simd.pset(32)  # 32 位元素，全部 lane 有效
                        for i in range(tile // 64):
                            a_reg = T.simd.vld(a[i * 64])
                            b_reg = T.simd.vld(b[i * 64])
                            result = T.simd.vadd(a_reg, b_reg, mask)
                            T.simd.vsts(c[i * 64], result, mask)
                T.copy(c, C[begin : begin + tile])

    return main


def run_regression_perf(n: int = 2**30, mode: str = "simt") -> float:
    """用 msprof 后端测延迟并换算有效带宽（口径：读写各一次）。"""
    from tilelang.profiler import do_bench

    kernel = tilelang.compile(vector_add(n, mode), target="ascend", out_idx=-1)
    a = torch.randn(n, device="npu")
    b = torch.randn(n, device="npu")
    latency = do_bench(lambda: kernel(a, b), backend="msprof", _n_warmup=30, _n_repeat=10)
    num_bytes = 3 * n * a.element_size()  # 两次输入读 + 一次输出写
    bandwidth = num_bytes / (latency * 1e6)
    print(f"vecadd ({mode}, N={n}): {latency * 1000:.2f} us | {bandwidth:.2f} GB/s")
    return latency


def main() -> int:
    parser = argparse.ArgumentParser(description="Ascend vector add: SIMT vs SIMD (main tree)")
    parser.add_argument("--mode", choices=["all", "simt", "simd"], default="all")
    parser.add_argument("--n", type=int, default=2**20)
    parser.add_argument("--bench", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    n = args.n
    a = torch.randn(n, device="npu")
    b = torch.randn(n, device="npu")
    for mode in ("simt", "simd") if args.mode == "all" else (args.mode,):
        kernel = tilelang.compile(vector_add(n, mode), target="ascend", out_idx=-1)
        torch.testing.assert_close(kernel(a, b), a + b)
        print(f"vecadd ({mode}): correctness passed")
        if args.bench:
            run_regression_perf(mode=mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
