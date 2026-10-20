#!/usr/bin/env python3
"""第 27 章 · 示例 1：把低精度格式的范围与精度算出来（CPU 逻辑验证）。

低精度格式的参数（指数位、尾数位、最大值、最小规格化数、单位舍入误差）网上到处
都有，但抄来的数字一旦错了，后面所有的容差推导都跟着错。这个脚本的做法是
**从位模式枚举出全部可表示值**，再由这些值直接算出表格里的每一项：

* fp16 / bf16 / fp32：fp16 用 numpy 的 finfo 取；bf16 没有原生 numpy 类型，
  按「1 符号 + 8 指数 + 7 尾数、bias 127」自己算边界；
* fp8 e4m3（OCP，PTX 里的 `.e4m3` / `e4m3fn`）：4 指数 + 3 尾数，bias 7，
  没有 inf，NaN 只占 `0x7f` / `0xff`；枚举全部 256 个位模式；
* fp8 e5m2（IEEE 风格）：5 指数 + 2 尾数，bias 15，有 inf 与 NaN；
* fp4 e2m1：2 指数 + 1 尾数，bias 1，无 inf / NaN，共 16 个位模式；
* MX 的 E8M0 scale：8 位指数 + 0 位尾数，**没有零**，值域 2^-127 .. 2^127。

输出的三张表分别回答三个问题：

1. 每个格式的有效位 p 与单位舍入误差 u = 2^-p 是多少（容差模型的输入）；
2. 每个格式能表示多大的数、多小的数（溢出与下溢的边界）；
3. MX 的 E8M0 scale 为什么适合做块缩放（相邻值恒差 2 倍，缩放不引入尾数误差）。

依赖：Python 3.9+ 与 numpy。不需要 GPU/NPU。

用法：

    python3 number_formats.py
    python3 number_formats.py --show-fp4          # 顺带打印 E2M1 的全部取值

作者实际做到的验证级别：**CPU 逻辑验证**（写作机 macOS / Apple Silicon、
Python 3.11.7 + numpy 1.24.1 上实际运行过，退出码 0；输出见同目录 README.md）。
"""

from __future__ import annotations

import argparse
import math

import numpy as np


# --------------------------------------------------------------------------- #
# 位模式枚举：本脚本所有数字的唯一来源
# --------------------------------------------------------------------------- #


def e4m3_values() -> list[float]:
    """枚举 fp8 e4m3（OCP FN 变体）的全部可表示值。

    布局：1 符号 + 4 指数 + 3 尾数，指数 bias = 7。
    与 IEEE 惯例的两点差别：

    * **没有 inf**：指数全 1 也是有限值（1.75 * 2^8 = 448）；
    * **NaN 只有两个编码**：`0x7f` 与 `0xff`，其余位模式都是有限数。
    """

    values: set[float] = set()
    for bits in range(256):
        sign = -1.0 if (bits >> 7) & 1 else 1.0
        exponent = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        if exponent == 0xF and mantissa == 0x7:  # 0x7f / 0xff -> NaN
            continue
        if exponent == 0:
            magnitude = (mantissa / 8.0) * 2.0 ** (1 - 7)  # 次正规数
        else:
            magnitude = (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        values.add(sign * magnitude)
    return sorted(values)


def e5m2_values() -> list[float]:
    """枚举 fp8 e5m2 的全部有限可表示值。

    布局：1 符号 + 5 指数 + 2 尾数，指数 bias = 15。指数全 1 是 inf / NaN，
    与 IEEE-754 一致，所以这里直接跳过。
    """

    values: set[float] = set()
    for bits in range(256):
        sign = -1.0 if (bits >> 7) & 1 else 1.0
        exponent = (bits >> 2) & 0x1F
        mantissa = bits & 0x3
        if exponent == 0x1F:  # inf / NaN
            continue
        if exponent == 0:
            magnitude = (mantissa / 4.0) * 2.0 ** (1 - 15)
        else:
            magnitude = (1.0 + mantissa / 4.0) * 2.0 ** (exponent - 15)
        values.add(sign * magnitude)
    return sorted(values)


def e2m1_values() -> list[float]:
    """枚举 fp4 e2m1 的全部 16 个位模式（无 inf / NaN，共 15 个不同取值）。"""

    values: set[float] = set()
    for bits in range(16):
        sign = -1.0 if (bits >> 3) & 1 else 1.0
        exponent = (bits >> 1) & 0x3
        mantissa = bits & 0x1
        if exponent == 0:
            magnitude = (mantissa / 2.0) * 2.0 ** (1 - 1)
        else:
            magnitude = (1.0 + mantissa / 2.0) * 2.0 ** (exponent - 1)
        values.add(sign * magnitude)
    return sorted(values)


def e8m0_values() -> list[float]:
    """枚举 MX 的 E8M0 scale 值：2^(e-127)，e = 0..254（255 是 NaN）。"""

    return [2.0 ** (code - 127) for code in range(255)]


def bf16_extremes() -> tuple[float, float, float]:
    """按 1+8+7、bias 127 的布局推出 bf16 的最大值、最小规格化数与最小次正规数。"""

    max_finite = (1.0 + 127.0 / 128.0) * 2.0 ** (254 - 127)
    min_normal = 2.0 ** (1 - 127)
    min_subnormal = (1.0 / 128.0) * 2.0 ** (1 - 127)
    return max_finite, min_normal, min_subnormal


# --------------------------------------------------------------------------- #
# 三张表
# --------------------------------------------------------------------------- #


def print_representation_table() -> None:
    print("== 一、表示精度：有效位与单位舍入误差 u = 2^-p ==")
    print("u 是这个格式「能分辨多小差别」的上界，也是第 12 章容差模型的唯一输入。")
    print(f"{'格式':<12}{'符号/指数/尾数':<18}{'有效位 p':<10}{'u = 2^-p':<14}{'atol 起点(10 的幂)'}")
    rows = [
        ("fp32", "1/8/23", 24),
        ("fp16", "1/5/10", 11),
        ("bf16", "1/8/7", 8),
        ("fp8 e4m3", "1/4/3", 4),
        ("fp8 e5m2", "1/5/2", 3),
        ("fp4 e2m1", "1/2/1", 2),
    ]
    for name, layout, p in rows:
        u = math.ldexp(1.0, -p)
        hint = 10.0 ** math.ceil(math.log10(u))
        print(f"{name:<14}{layout:<20}{p:<12}{u:<16.3e}{hint:<.0e}")


def print_range_table() -> None:
    print("\n== 二、表示范围：三个边界 ==")
    print("「最小规格化数」是相对精度开始退化的地方（再往下是次正规数，间距不再随值缩小）。")
    print(f"{'格式':<12}{'max 有限值':<16}{'min 规格化数':<16}{'min 次正规数':<16}{'动态范围(10 的幂)'}")

    fp16 = np.finfo(np.float16)
    bf_max, bf_min_norm, bf_min_sub = bf16_extremes()
    rows: list[tuple[str, float, float, float]] = [
        ("fp16", float(fp16.max), float(fp16.smallest_normal), float(2.0 ** -24)),
        ("bf16", bf_max, bf_min_norm, bf_min_sub),
    ]
    for tag in ("e4m3", "e5m2", "e2m1"):
        values = {"e4m3": e4m3_values(), "e5m2": e5m2_values(), "e2m1": e2m1_values()}[tag]
        exponent_bits = {"e4m3": 4, "e5m2": 5, "e2m1": 2}[tag]
        bias = 2 ** (exponent_bits - 1) - 1
        label = f"fp8 {tag}" if tag != "e2m1" else "fp4 e2m1"
        rows.append((label, max(values), 2.0 ** (1 - bias), min(v for v in values if v > 0.0)))

    for name, max_finite, min_normal, min_sub in rows:
        print(f"{name:<14}{max_finite:<18.4g}{min_normal:<18.4g}{min_sub:<18.4g}{math.log10(max_finite / min_sub):<.1f}")


def print_mx_scale_table() -> None:
    scales = e8m0_values()
    print("\n== 三、MX 的 E8M0 scale ==")
    print(f"可表示值个数：{len(scales)}（8 位指数，编码 255 留给 NaN；没有 0，也没有负数）")
    print(f"最小：2^-127 = {min(scales):.6e}")
    print(f"最大：2^127  = {max(scales):.6e}")
    print("相邻值之比恒为 2：缩放本身不引入尾数误差，只搬运指数。")
    print("注意：E8M0 不能表示 0，所以「整块全零」要选一个约定的最小指数兜底。")


def print_fp4_codebook() -> None:
    print("\n== 四、fp4 e2m1 的全部取值（做量化时可以直接当码本用）=")
    positives = sorted(v for v in e2m1_values() if v >= 0.0)
    print("非负值：" + ", ".join(f"{v:g}" for v in positives))
    print("间距在 [1,2) 上是 0.5 → u = 0.25：靠近 1 的数只有 3 位有效信息。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show-fp4", action="store_true", help="打印 fp4 e2m1 的全部取值")
    args = parser.parse_args()

    print_representation_table()
    print_range_table()
    print_mx_scale_table()
    if args.show_fp4:
        print_fp4_codebook()
    else:
        print("\n（加 --show-fp4 可以打印 fp4 e2m1 的全部取值。）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
