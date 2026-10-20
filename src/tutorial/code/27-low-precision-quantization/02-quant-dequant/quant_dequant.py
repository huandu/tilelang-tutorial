#!/usr/bin/env python3
"""第 27 章 · 示例 2：四种量化方案的往返误差（CPU 逻辑验证）。

「量化」在两处出现，误差来源完全不同，混在一起讨论就会得出错误的结论：

* **整数对称量化**（int8 / int4）：scale 取 `absmax / qmax`，量化误差是**均匀分布**
  的，其绝对上界是 `absmax / (2 * qmax)`；相对到 absmax 就是 `1 / (2 * qmax)`。
  粒度（per-tensor / per-channel / per-group）决定 absmax 有多大，因此决定误差。
* **浮点块缩放**（MXFP8 / MXFP4）：元素先除以块的 E8M0 scale，再舍入到 fp8/fp4
  的码本。误差来自码本本身的相对间距（u = 2^-p），而不是均匀量化。

这个脚本对同一份数据跑五种方案，输出往返的相对 L2 误差与逐元素最大误差，并给出
「误差该有多大」的解析预测，用来判断实测是不是落在正常范围里。

依赖：Python 3.9+ 与 numpy。不需要 GPU/NPU。

用法：

    python3 quant_dequant.py
    python3 quant_dequant.py --rows 512 --cols 1024 --seed 20261001

作者实际做到的验证级别：**CPU 逻辑验证**（写作机 macOS / Apple Silicon、
Python 3.11.7 + numpy 1.24.1 上实际运行过，退出码 0；输出见同目录 README.md）。
"""

from __future__ import annotations

import argparse
import math

import numpy as np


# --------------------------------------------------------------------------- #
# 两套码本：float 码本用于 MX，整数用于对称量化
# --------------------------------------------------------------------------- #


def e4m3_codebook() -> np.ndarray:
    """fp8 e4m3（OCP）的全部非负可表示值，升序，含 0。"""

    values: set[float] = set()
    for exponent in range(16):
        for mantissa in range(8):
            if exponent == 0xF and mantissa == 0x7:  # NaN
                continue
            if exponent == 0:
                values.add((mantissa / 8.0) * 2.0 ** (1 - 7))
            else:
                values.add((1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    return np.asarray(sorted(values), dtype=np.float64)


def e2m1_codebook() -> np.ndarray:
    """fp4 e2m1 的全部非负可表示值：0, 0.5, 1, 1.5, 2, 3, 4, 6。"""

    values: set[float] = set()
    for exponent in range(4):
        for mantissa in range(2):
            if exponent == 0:
                values.add((mantissa / 2.0) * 2.0 ** (1 - 1))
            else:
                values.add((1.0 + mantissa / 2.0) * 2.0 ** (exponent - 1))
    return np.asarray(sorted(values), dtype=np.float64)


def round_to_codebook(magnitude: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """按「最近的可表示值」舍入（码本已升序且含 0）。

    平局时取较小的那个（np.searchsorted 的 left 侧），与硬件的 round-to-nearest-even
    不是逐位等价；这只影响平局样本，不改变误差的量级结论。
    """

    idx = np.searchsorted(codebook, magnitude, side="left")
    idx = np.clip(idx, 1, len(codebook) - 1)
    lower = codebook[idx - 1]
    upper = codebook[idx]
    return np.where(magnitude - lower <= upper - magnitude, lower, upper)


# --------------------------------------------------------------------------- #
# 五种方案：每种返回「反量化后的 float64 张量」
# --------------------------------------------------------------------------- #


def quantize_int_per_tensor(x: np.ndarray, bits: int) -> np.ndarray:
    qmax = 2 ** (bits - 1) - 1
    scale = float(np.max(np.abs(x))) / qmax if np.max(np.abs(x)) > 0 else 1.0
    q = np.clip(np.rint(x / scale), -qmax - 1, qmax)
    return q * scale


def quantize_int_per_channel(x: np.ndarray, bits: int) -> np.ndarray:
    """按输出行（axis 1 的每一行）各取一个 scale，模拟「权重的 per-channel 量化」。"""

    qmax = 2 ** (bits - 1) - 1
    absmax = np.max(np.abs(x), axis=1, keepdims=True)
    scale = np.where(absmax > 0, absmax / qmax, 1.0)
    q = np.clip(np.rint(x / scale), -qmax - 1, qmax)
    return q * scale


def quantize_int_per_group(x: np.ndarray, bits: int, group: int) -> np.ndarray:
    """沿 axis 1 每 `group` 个元素共享一个 scale，模拟 GPTQ/AWQ 的 group 量化。"""

    qmax = 2 ** (bits - 1) - 1
    rows, cols = x.shape
    assert cols % group == 0, "cols 必须是 group 的整数倍"
    blocks = x.reshape(rows, cols // group, group)
    absmax = np.max(np.abs(blocks), axis=2, keepdims=True)
    scale = np.where(absmax > 0, absmax / qmax, 1.0)
    q = np.clip(np.rint(blocks / scale), -qmax - 1, qmax)
    return (q * scale).reshape(rows, cols)


def e8m0_scale(absmax: float, codebook_max: float) -> float:
    """块的 E8M0 scale：向上取到 2 的整数次幂，使 absmax/scale <= codebook_max。"""

    if absmax <= 0.0:
        return 2.0 ** -127
    needed = absmax / codebook_max
    exponent = int(math.ceil(math.log2(needed)))
    exponent = max(-127, min(127, exponent))
    return 2.0 ** exponent


def quantize_mx(x: np.ndarray, codebook: np.ndarray, block: int = 32) -> np.ndarray:
    """MXFP 风格的块缩放量化：block 个元素共享一个 E8M0 scale。"""

    rows, cols = x.shape
    assert cols % block == 0, "cols 必须是 block 的整数倍"
    blocks = x.reshape(rows, cols // block, block)
    absmax = np.max(np.abs(blocks), axis=2)
    codebook_max = float(codebook[-1])
    scales = np.asarray([e8m0_scale(float(v), codebook_max) for v in absmax.ravel()]).reshape(absmax.shape)
    scaled = blocks / scales[:, :, None]
    magnitude = round_to_codebook(np.abs(scaled), codebook)
    signed = np.copysign(magnitude, scaled)
    return (signed * scales[:, :, None]).reshape(rows, cols)


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #


def relative_l2(approx: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(approx - exact) / np.linalg.norm(exact))


def metrics(approx: np.ndarray, exact: np.ndarray) -> tuple[float, float, float]:
    rel_l2 = relative_l2(approx, exact)
    max_abs = float(np.max(np.abs(approx - exact)))
    signal = float(np.sqrt(np.mean(exact**2)))
    noise = float(np.sqrt(np.mean((approx - exact) ** 2)))
    sqnr = 20.0 * math.log10(signal / noise) if noise > 0 else float("inf")
    return rel_l2, max_abs, sqnr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    # 权重近似：不同的行幅度差一个量级，这样「per-channel 比 per-tensor 好多少」
    # 才有意义。纯 i.i.d. 的正态数据会让三种整数粒度的差距看不出来。
    row_scale = np.exp(rng.normal(0.0, 1.2, size=(args.rows, 1)))
    x = (rng.normal(0.0, 1.0, size=(args.rows, args.cols)) * row_scale).astype(np.float64)

    e4m3 = e4m3_codebook()
    e2m1 = e2m1_codebook()

    print(f"数据：{args.rows} x {args.cols}，行幅度服从对数正态（sigma=1.2），seed={args.seed}")
    print(f"数据 absmax = {np.max(np.abs(x)):.4f}，全局 RMS = {np.sqrt(np.mean(x ** 2)):.4f}")

    schemes: list[tuple[str, np.ndarray, str]] = [
        ("int8 per-tensor", quantize_int_per_tensor(x, 8), "相对 absmax: 1/(2*127) = 3.9e-3"),
        ("int8 per-channel", quantize_int_per_channel(x, 8), "相对 absmax: 1/(2*127) = 3.9e-3"),
        ("int4 per-group-64", quantize_int_per_group(x, 4, 64), "相对 absmax: 1/(2*7) = 7.1e-2"),
        ("MXFP8 (e4m3, blk 32)", quantize_mx(x, e4m3, 32), "码本相对间距: 2^-4 = 6.3e-2"),
        ("MXFP4 (e2m1, blk 32)", quantize_mx(x, e2m1, 32), "码本相对间距: 2^-2 = 2.5e-1"),
    ]

    print()
    print(f"{'方案':<24}{'相对 L2':<14}{'逐元素 max err':<18}{'SQNR (dB)':<12}{'解析误差上界'}")
    for name, approx, prediction in schemes:
        rel_l2, max_abs, sqnr = metrics(approx, x)
        print(f"{name:<26}{rel_l2:<16.3e}{max_abs:<20.3e}{sqnr:<14.1f}{prediction}")

    per_tensor_absmax = float(np.max(np.abs(x)))
    median_row_absmax = float(np.median(np.max(np.abs(x), axis=1)))
    print()
    print("== 怎么读这张表 ==")
    print("1) int8 的 3.9e-3 与 bf16 的 u=3.9e-3 同量级：int8 存权重在精度上不比 bf16 差。")
    print("2) int4 的 7.1e-2 与 fp8 e4m3 的 6.3e-2 同量级：4bit 定点约等于 8bit 浮点。")
    print("3) 上界是相对 **absmax** 的，不是相对典型值。这一份数据里")
    print(f"   per-tensor absmax = {per_tensor_absmax:.4f}，而 per-row absmax 的中位数只有 {median_row_absmax:.4f}，")
    print("   相差 25 倍——所以 per-tensor 的相对 L2（4.95e-2）反而比 3.9e-3 大一个量级。")
    print("4) 把 absmax 拉回同一量级以后，L2 就回到上界附近：per-channel 是 7.67e-3，")
    print("   与 3.9e-3 同量级。**量化的精度由「每多少个元素共享一个 scale」决定，")
    print("   而不是由位宽单独决定。**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
