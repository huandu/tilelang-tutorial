#!/usr/bin/env python3
"""GEMM 的性能账本与 tile 占用估算（纯标准库，不需要 GPU/NPU）。

这个脚本只做两件事：

1. 按第 14 章 1.1 节的公式，算一个 GEMM 的计算量、最小 DRAM 访存、
   算术强度，并判断 compute-bound / memory-bound；
2. 按 1.3 节的公式，估算一份 tile 配置的 shared memory 占用、
   每线程累加器寄存器数，以及只看 shared memory 时能驻留几个 block。

它不是性能预测器：所有输出都是公式推导，没有任何实测成分。

用法::

    python3 gemm_ledger.py                          # 默认 4096^3 fp16, H100 规格
    python3 gemm_ledger.py --m 256 --n 256 --k 16384
    python3 gemm_ledger.py --block-m 128 --block-n 256 --block-k 32 --stages 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Machine:
    """一台机器的 roofline 参数。默认值是 H100 SXM5 的厂商公开规格。"""

    name: str = "H100 SXM5"
    peak_tflops: float = 1000.0      # fp16 张量核，厂商公开 preliminary specs
    bandwidth_tbps: float = 3.0      # HBM3, "over 3 TB/sec"

    @property
    def balance(self) -> float:
        """机器平衡点（FLOP/Byte）= 峰值算力 / 峰值带宽。"""
        return self.peak_tflops * 1e12 / (self.bandwidth_tbps * 1e12)


def gemm_ledger(m: int, n: int, k: int, dtype_bytes: int = 2) -> dict:
    """算出 (m, n, k) GEMM 的账本三行。"""
    flops = 2 * m * n * k
    min_bytes = (m * k + k * n + m * n) * dtype_bytes
    intensity = flops / min_bytes if min_bytes > 0 else float("inf")
    return {
        "flops": flops,
        "min_bytes": min_bytes,
        "intensity": intensity,
    }


def classify(intensity: float, balance: float) -> str:
    return "compute-bound" if intensity >= balance else "memory-bound"


def tile_budget(
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    dtype_bytes: int = 2,
    threads: int = 128,
    accum_bytes: int = 4,
    smem_limit_kb: float = 228.0,
) -> dict:
    """估算一份 tile 配置的片上占用与寄存器压力。

    只统计 A/B tile 的 shared memory 与 C 累加器的寄存器；
    epilogue、pipeline barrier、padding 都未计入（因此是乐观下界）。
    """
    per_stage_bytes = dtype_bytes * (block_m * block_k + block_k * block_n)
    smem_bytes = per_stage_bytes * stages
    accum_regs = block_m * block_n * accum_bytes // 4 / threads
    blocks_by_smem = int(smem_limit_kb * 1024 // smem_bytes) if smem_bytes else 0
    return {
        "per_stage_bytes": per_stage_bytes,
        "smem_bytes": smem_bytes,
        "accum_regs_per_thread": accum_regs,
        "blocks_by_smem": blocks_by_smem,
    }


def human_bytes(value: float) -> str:
    """按 1024 进制格式化，单位写成 KiB/MiB/GiB（避免和十进制 MB 混淆）。"""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.4g} {unit}"
        value /= 1024
    return f"{value:.4g} TiB"


def human_flops(value: float) -> str:
    for unit, scale in (("TFLOP", 1e12), ("GFLOP", 1e9), ("MFLOP", 1e6)):
        if value >= scale:
            return f"{value / scale:.4g} {unit}"
    return f"{value:.4g} FLOP"


def print_ledger(m: int, n: int, k: int, machine: Machine, dtype_bytes: int) -> None:
    row = gemm_ledger(m, n, k, dtype_bytes)
    print(f"GEMM {m} x {n} x {k}  (dtype {dtype_bytes * 8} bit)")
    print(f"  计算量          : {human_flops(row['flops'])}")
    print(f"  最小 DRAM 访存  : {human_bytes(row['min_bytes'])}")
    print(f"  算术强度        : {row['intensity']:.2f} FLOP/Byte")
    print(f"  机器平衡点      : {machine.balance:.2f} FLOP/Byte")
    print(f"  判定            : {classify(row['intensity'], machine.balance)}")
    print()


def print_budget(args: argparse.Namespace) -> None:
    budget = tile_budget(
        args.block_m,
        args.block_n,
        args.block_k,
        args.stages,
        dtype_bytes=args.dtype_bytes,
        threads=args.threads,
    )
    print(
        f"tile 配置 block_M={args.block_m} block_N={args.block_n} "
        f"block_K={args.block_k} stages={args.stages} threads={args.threads}"
    )
    print(f"  单级 shared memory   : {human_bytes(budget['per_stage_bytes'])}")
    print(f"  多级 shared memory   : {human_bytes(budget['smem_bytes'])}")
    print(f"  每线程累加寄存器     : {budget['accum_regs_per_thread']:.0f}")
    print(f"  受 shared 限制的 block 数 : {budget['blocks_by_smem']}")
    if budget["accum_regs_per_thread"] > 255:
        print("  警告：累加器寄存器已超过 CUDA 常规上限 255，几乎必然溢出到 local memory")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="GEMM 性能账本与 tile 占用估算")
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--dtype-bytes", type=int, default=2, help="输入 dtype 的字节数（fp16=2）")
    parser.add_argument("--peak-tflops", type=float, default=1000.0)
    parser.add_argument("--bandwidth-tbps", type=float, default=3.0)
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--sweep", action="store_true", help="对几组常见 shape 打印对比表")
    args = parser.parse_args()

    machine = Machine(peak_tflops=args.peak_tflops, bandwidth_tbps=args.bandwidth_tbps)
    print("=" * 72)
    print(f"机器模型：{machine.name}  peak={machine.peak_tflops} TFLOPS  "
          f"bw={machine.bandwidth_tbps} TB/s  balance={machine.balance:.1f} F/B")
    print("=" * 72)
    print()

    if args.sweep:
        for (m, n, k) in ((4096, 4096, 4096), (512, 512, 512), (128, 128, 128),
                          (256, 256, 16384), (8192, 8192, 8192)):
            print_ledger(m, n, k, machine, args.dtype_bytes)
    else:
        print_ledger(args.m, args.n, args.k, machine, args.dtype_bytes)

    print_budget(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
