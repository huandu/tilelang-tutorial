#!/usr/bin/env python3
"""第 19 章 §8：用 msProf 后端采集性能并读瓶颈。

这个骨架演示三件事：
  1. do_bench(backend="msprof") 拿一个延迟数字；
  2. do_bench(backend="msprof_detail") 拿 KernelProfile（AIC/AIV 各 pipe 占用）；
  3. 按 §8.3 的判据，把比值翻译成「瓶颈在 Cube / 搬运 / Vector」的结论。

注意：这里的 kernel 只是一个占位（逐元素 add），真正要测性能请替换成你的算子。

运行前提：Ascend NPU + CANN + torch_npu + 主干 TileLang v0.1.15 + msProf 可用。
验证级别：仅静态核对（py_compile + API 比对），未在 NPU 上运行。
"""

from __future__ import annotations

import argparse
import statistics

import torch
import tilelang
import tilelang.ascend.language as T


def vec_add(N: int, num_blocks: int = 64, tile: int = 8192):
    """被测对象：最小的一维逐元素 add。返回 prim_func，由调用方 compile。"""
    if N % (num_blocks * tile) != 0:
        raise ValueError(f"N must be a multiple of {num_blocks * tile}")
    iters = N // (num_blocks * tile)

    @T.prim_func
    def main(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32"), C: T.Tensor((N,), "float32")):
        with T.Kernel(num_blocks) as bx:
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({a: 2, b: 2, c: 2})
            for step in T.Pipelined(iters, num_stages=2):
                begin = (step * num_blocks + bx) * tile
                T.copy(A[begin : begin + tile], a)
                T.copy(B[begin : begin + tile], b)
                with T.SimtVF(threads=2048):
                    for i in T.Parallel(tile):
                        c[i] = a[i] + b[i]
                T.copy(c, C[begin : begin + tile])

    return main


def read_bottleneck(profile) -> str:
    """把 KernelProfile 的 pipe 占比翻译成一句瓶颈结论（§8.3）。

    注意：AIC 与 AIV 的比值各自相对自己的 total_cycles，不能相加。
    """
    aic = {
        "mad": profile.aic_mad,
        "mte1": profile.aic_mte1,
        "mte2": profile.aic_mte2,
        "mte3": profile.aic_mte3,
        "fixpipe": profile.aic_fixpipe,
    }
    aiv = {
        "vec": profile.aiv_vec,
        "mte2": profile.aiv_mte2,
        "mte3": profile.aiv_mte3,
    }
    busiest_aic = max(aic, key=lambda k: aic[k])
    busiest_aiv = max(aiv, key=lambda k: aiv[k])
    if aic[busiest_aic] >= aiv[busiest_aiv]:
        return (
            f"AIC 最忙的是 {busiest_aic} ({aic[busiest_aic] * 100:.1f}%)；"
            f"若它就是 mad，说明 Cube 已打满、该改算法；若是 mte1/mte2/fixpipe，"
            f"说明 Cube 在等数据、该改搬运或 L0/L1 切分。"
        )
    return (
        f"AIV 最忙的是 {busiest_aiv} ({aiv[busiest_aiv] * 100:.1f}%)；"
        f"若是 mte2/mte3，说明 Vector 侧受搬运带宽限制、该做融合；若是 vec，"
        f"说明 Vector 已打满、该减少 Vector 工作量或改用 SIMD。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="msProf-based benchmarking on Ascend")
    parser.add_argument("--n", type=int, default=2**20)
    parser.add_argument("--samples", type=int, default=5, help="重复多少次 do_bench 取中位数")
    args = parser.parse_args()

    from tilelang.profiler import do_bench

    n = args.n
    kernel = tilelang.compile(vec_add(n), target="ascend", out_idx=-1)
    a = torch.randn(n, device="npu")
    b = torch.randn(n, device="npu")
    torch.testing.assert_close(kernel(a, b), a + b)
    print("correctness passed")

    # 1) 只取延迟：多取几次自己求中位数（msprof 后端不带 median 选择）。
    samples = [do_bench(lambda: kernel(a, b), backend="msprof", _n_warmup=30, _n_repeat=50) for _ in range(args.samples)]
    latency_ms = statistics.median(samples)
    num_bytes = 3 * n * a.element_size()
    print(f"latency median = {latency_ms * 1000:.2f} us | effective BW = {num_bytes / (latency_ms * 1e6):.2f} GB/s")
    print("raw samples (us):", [round(s * 1000, 2) for s in samples])

    # 2) 取详细画像：AIC/AIV 各 pipe 占比。
    profile = do_bench(lambda: kernel(a, b), backend="msprof_detail", _n_warmup=10, _n_repeat=10)
    print("profile:", profile)
    print("bottleneck:", read_bottleneck(profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
