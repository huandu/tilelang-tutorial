#!/usr/bin/env python3
"""第 27 章 · 示例 4：对拍时「参照选错」会怎样（CPU 逻辑验证）。

低精度算子的对拍有一个独有的陷阱：**被测的 kernel 和参照实现必须共享同一套量化
规则**。如果参照用的是量化前的 fp32 权重，那么量化误差（正常）会被判成逻辑错误
（不正常）；反过来，如果参照自己也犯同样的错，逻辑错误又会被掩盖。

脚本用一个最小的 int4 per-channel 反量化 GEMM 演示四种参照：

    A. 正确参照：用反量化后的权重（fp64）做 matmul            -> 应该通过
    B. 粒度/位置写错：scale 作用在输入上而不是权重上          -> 应该失败（逻辑错误）
    C. 参照漏了 scale：直接拿整数权重做 matmul                 -> 应该失败（逻辑错误）
    D. 参照是量化前的 fp32 结果（紧容差）                      -> 「失败」，但这不是 bug

每个候选都按锁定版本里 `torch_assert_close` 的缺省判据
（`rtol=1e-2, atol=1e-3, max_mismatched_ratio=1e-3`）给出结论，并同时打印
相对 L2 与逐元素最大误差，说明「只看一个指标」为什么不够。

依赖：Python 3.9+ 与 numpy。不需要 GPU/NPU。

用法：

    python3 reference_matching.py
    python3 reference_matching.py --m 256 --n 256 --k 1024 --group 64

作者实际做到的验证级别：**CPU 逻辑验证**（写作机 macOS / Apple Silicon、
Python 3.11.7 + numpy 1.24.1 上实际运行过，退出码 0；输出见同目录 README.md）。
"""

from __future__ import annotations

import argparse

import numpy as np

RTOL = 1e-2
ATOL = 1e-3
MAX_MISMATCHED_RATIO = 1e-3


def quantize_int4_per_group(weight: np.ndarray, group: int) -> tuple[np.ndarray, np.ndarray]:
    """int4 对称量化（沿 K 每 group 个元素共享一个 scale）。

    返回 (整数权重, 每 (行, 组) 的 float scale)。这是 GPTQ / AWQ 一类方案的
    最简形式；真实实现还会有 zero point 与重排，但误差结构相同。
    """

    rows, cols = weight.shape
    qmax = 7
    blocks = weight.reshape(rows, cols // group, group)
    absmax = np.max(np.abs(blocks), axis=2)
    scale = np.where(absmax > 0, absmax / qmax, 1.0)
    q = np.clip(np.rint(blocks / scale[:, :, None]), -qmax - 1, qmax).astype(np.int32)
    return q.reshape(rows, cols), scale


def dequantize_int4(q: np.ndarray, scale: np.ndarray, group: int) -> np.ndarray:
    rows, cols = q.shape
    blocks = q.reshape(rows, cols // group, group).astype(np.float64)
    return (blocks * scale[:, :, None]).reshape(rows, cols)


def kernel_output(
    x_int: np.ndarray,
    x_scale: float,
    q: np.ndarray,
    w_scale: np.ndarray,
    group: int,
) -> np.ndarray:
    """模拟 TileLang kernel：整数域精确累加 + epilogue 里乘 scale。

    与真实 kernel 的差别只有一处：这里把「激活 scale × 权重 group scale」在
    epilogue 一次乘完，真实实现通常先把整数乘积累加进 int32 L0C / 寄存器，
    再在 epilogue 乘 scale。**整数累加是精确的**，所以两者的结果只差浮点乘法的
    结合顺序，量级在 1e-16 以下。
    """

    rows, cols = q.shape
    groups = cols // group
    xb = x_int.reshape(x_int.shape[0], groups, group).astype(np.float64)
    qb = q.reshape(rows, groups, group).astype(np.float64)
    out = np.zeros((x_int.shape[0], rows), dtype=np.float64)
    for g in range(groups):
        combined = x_scale * w_scale[:, g]  # [N]
        out += (xb[:, g, :] @ qb[:, g, :].T) * combined[None, :]
    return out


def report(label: str, approx: np.ndarray, reference: np.ndarray) -> bool:
    diff = np.abs(approx - reference)
    rel = np.linalg.norm(approx - reference) / np.linalg.norm(reference)
    close = np.isclose(approx, reference, rtol=RTOL, atol=ATOL)
    mismatched = int(np.size(close) - np.count_nonzero(close))
    ratio = mismatched / close.size
    verdict = "通过" if ratio <= MAX_MISMATCHED_RATIO else "失败"
    print(
        f"{label:<42}{rel:<14.3e}{float(diff.max()):<16.3e}"
        f"{mismatched:<12}{ratio:<12.3e}{verdict}"
    )
    return verdict == "通过"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    assert args.k % args.group == 0

    rng = np.random.default_rng(args.seed)
    x_fp = rng.standard_normal((args.m, args.k))
    w_fp = rng.standard_normal((args.n, args.k)) * 0.1

    # 输入侧也做一次 int8 量化（激活量化），让「参照必须共享量化规则」这件事更明显。
    x_absmax = float(np.max(np.abs(x_fp)))
    x_scale = x_absmax / 127.0
    x_int = np.clip(np.rint(x_fp / x_scale), -128, 127).astype(np.int32)

    q, w_scale = quantize_int4_per_group(w_fp, args.group)
    w_deq = dequantize_int4(q, w_scale, args.group)

    got = kernel_output(x_int, x_scale, q, w_scale, args.group)

    print(f"形状：X[{args.m},{args.k}] int8，W[{args.n},{args.k}] int4，group={args.group}")
    print(f"判据：rtol={RTOL:g}, atol={ATOL:g}, max_mismatched_ratio={MAX_MISMATCHED_RATIO:g}")
    print()
    print(f"{'参照':<44}{'相对 L2':<14}{'max abs err':<16}{'不匹配元素数':<12}{'不匹配比例':<12}{'结论'}")

    # A. 正确参照：量化感知（同样的量化规则 + fp64 计算）
    ref_a = (x_int.astype(np.float64) * x_scale) @ w_deq.T
    report("A 量化感知参照（fp64）", got, ref_a)

    # B. 逻辑错误：反量化融合位置错——把 group 循环里的 scale 提到循环外，只用了最后一组
    ref_b = (
        (x_int.astype(np.float64) * x_scale) @ q.T.astype(np.float64) * w_scale[:, -1][None, :]
    )
    report("B 反量化位置错（只乘最后一组 scale）", got, ref_b)

    # C. 逻辑错误：参照忘了乘 scale，直接用整数权重
    ref_c = (x_int.astype(np.float64) * x_scale) @ q.T.astype(np.float64)
    report("C 参照漏乘 scale（逻辑错误）", got, ref_c)

    # D. 参照用错层次：拿量化前的 fp32 权重做参照
    ref_d = (x_int.astype(np.float64) * x_scale) @ w_fp.T
    report("D 用量化前的 fp32 权重当参照", got, ref_d)

    # E. 反例：被测实现犯了 C 里同样的错，对拍就「通过」了——错对错互相掩盖
    rng_noise = np.random.default_rng(args.seed + 7)
    wrong_kernel = ref_c * (1.0 + 1e-12 * rng_noise.standard_normal(ref_c.shape))
    report("E 被测与参照 C 犯同一个错", wrong_kernel, ref_c)
    report("E' 同一个错误的实现 vs 正确的 A", wrong_kernel, ref_a)

    print()
    print("== 怎么读这张表 ==")
    print("1) A 通过：说明 kernel 的整数累加与 epilogue 都对。")
    print("2) B、C 失败且 max abs err 是结果的量级（不是 1e-3 量级）：这类是逻辑错误，")
    print("   与容差无关，改容差也过不去。判据是「误差随某个维度整体偏移」，不是「个别元素超差」。")
    print("3) D 是一条重要的陷阱：它失败，但**不是 bug**——94% 的元素不匹配，误差 1.08e-1，")
    print("   这正是 int4 权重的量化误差。把 D 当判据会逼着你把 atol 放水到没法查错。")
    print("4) E 通过：两个都漏了同一个 scale。对拍只能证明「两边一致」，")
    print("   不能证明「两边都对」——E' 一行把这件事量化了：同一个实现相对正确参照 A")
    print("   的相对 L2 是 2.7e+1，错得离谱，可它偏偏通过了 E 这一关。")
    print("   所以每个量化 kernel 都要有 A 这样的绝对参照，不能只有「两边对拍」。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
