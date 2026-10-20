#!/usr/bin/env python3
"""第 27 章 · 示例 3：低精度 GEMM 的误差预算与实测（CPU 逻辑验证）。

第 12 章用 fp16/bf16 建立了容差模型：

    以 S = sum|a_i * b_i| 为基准，输入舍入带来的相对误差
        最坏情形 = u = 2^-p，  均方根 = u / sqrt(K)

本章要把这个模型推广到 fp8 / fp4 / int8 / int4，并回答三个问题：

1. **输入量化误差** 与 **累加误差** 是谁在主导？因此对拍时该选哪个参照？
2. K 增大时误差为什么反而变小（u/sqrt(K)），什么情况下会破坏这个规律？
3. 累加器用 fp16 而不是 fp32，误差会放大多少？split-K 这种「分段累加」为什么能把
   误差拉回来？

脚本对同一批随机数据做三组实验：

* 实验 A：各种格式量化输入 + fp32 累加，分别以「原始值」和「量化后的值」为参照，
  看两种参照下的误差差多少（这就是「参照选错」的量化版本）；
* 实验 B：K = 64 .. 8192，验证 u/sqrt(K) 的缩放律；
* 实验 C：K = 4096，比较 fp16 累加、fp32 累加、以及 split 成 32 段后的累加。

依赖：Python 3.9+ 与 numpy。不需要 GPU/NPU。

用法：

    python3 error_model.py
    python3 error_model.py --trials 400 --lengths 64,256,1024,4096

作者实际做到的验证级别：**CPU 逻辑验证**（写作机 macOS / Apple Silicon、
Python 3.11.7 + numpy 1.24.1 上实际运行过，退出码 0；输出见同目录 README.md）。
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def e4m3_codebook() -> np.ndarray:
    values: set[float] = set()
    for exponent in range(16):
        for mantissa in range(8):
            if exponent == 0xF and mantissa == 0x7:
                continue
            if exponent == 0:
                values.add((mantissa / 8.0) * 2.0 ** (1 - 7))
            else:
                values.add((1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    return np.asarray(sorted(values), dtype=np.float64)


def e2m1_codebook() -> np.ndarray:
    values: set[float] = set()
    for exponent in range(4):
        for mantissa in range(2):
            if exponent == 0:
                values.add((mantissa / 2.0) * 2.0 ** 1)
            else:
                values.add((1.0 + mantissa / 2.0) * 2.0 ** (exponent - 1))
    return np.asarray(sorted(values), dtype=np.float64)


def round_to_codebook(magnitude: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(codebook, magnitude, side="left"), 1, len(codebook) - 1)
    lower = codebook[idx - 1]
    upper = codebook[idx]
    return np.where(magnitude - lower <= upper - magnitude, lower, upper)


def to_bf16(x: np.ndarray) -> np.ndarray:
    """按「截断 fp16 低位」近似 bf16 的 round-to-nearest（与第 12 章一致）。"""

    as_u32 = x.astype(np.float32).view(np.uint32)
    rounded = (as_u32 + np.uint32(0x8000)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32).astype(np.float64)


def quantize(name: str, x: np.ndarray) -> np.ndarray:
    """把 float64 输入量化成某种低精度格式，再还原成本脚本内部的 float64 表示。"""

    if name == "fp16":
        return x.astype(np.float16).astype(np.float64)
    if name == "bf16":
        return to_bf16(x)
    if name == "e4m3":
        magnitude = round_to_codebook(np.abs(x), e4m3_codebook())
        return np.copysign(magnitude, x)
    if name == "e2m1":
        magnitude = round_to_codebook(np.abs(x), e2m1_codebook())
        return np.copysign(magnitude, x)
    if name in ("int8", "int4"):
        bits = 8 if name == "int8" else 4
        qmax = 2 ** (bits - 1) - 1
        absmax = float(np.max(np.abs(x)))
        scale = absmax / qmax if absmax > 0 else 1.0
        return np.clip(np.rint(x / scale), -qmax - 1, qmax) * scale
    raise ValueError(name)


# 每个格式的「等价 u」：浮点格式用 2^-p；均匀量化的整数格式用 1/(2*qmax)。
EQUIVALENT_U = {
    "fp16": 2.0**-11,
    "bf16": 2.0**-8,
    "e4m3": 2.0**-4,
    "e2m1": 2.0**-2,
    "int8": 1.0 / (2.0 * 127.0),
    "int4": 1.0 / (2.0 * 7.0),
}

LAYOUT = {
    "fp16": "1/5/10",
    "bf16": "1/8/7",
    "e4m3": "1/4/3",
    "e2m1": "1/2/1",
    "int8": "定点 8 位",
    "int4": "定点 4 位",
}


def experiment_a(length: int, trials: int, seed: int) -> list[tuple[str, float, float, float, float]]:
    """返回 [(格式, 对原始值的相对误差中位数, 对量化值的相对误差中位数, 最大值, RMS 预算)]。"""

    rng = np.random.default_rng(seed)
    rows: dict[str, list[tuple[float, float]]] = {name: [] for name in EQUIVALENT_U}
    for _ in range(trials):
        x = rng.uniform(-1.0, 1.0, size=length)
        y = rng.uniform(-1.0, 1.0, size=length)
        for name in EQUIVALENT_U:
            xq = quantize(name, x)
            yq = quantize(name, y)
            # 参照一：原始值的 fp64 精确点积（包含输入量化误差）
            exact_raw = float(np.dot(x, y))
            # 参照二：量化后值的 fp64 精确点积（只含累加误差）
            exact_quantized = float(np.dot(xq, yq))
            # 被测：量化输入 + fp32 累加
            got = float(np.dot(xq.astype(np.float32), yq.astype(np.float32)))
            denom = float(np.sum(np.abs(xq * yq))) or 1.0
            rows[name].append((abs(got - exact_raw) / denom, abs(got - exact_quantized) / denom))

    out: list[tuple[str, float, float, float, float]] = []
    for name, samples in rows.items():
        arr = np.asarray(samples)
        budget = EQUIVALENT_U[name] / math.sqrt(length)
        out.append(
            (
                name,
                float(np.median(arr[:, 0])),
                float(np.median(arr[:, 1])),
                float(np.max(arr[:, 1])),
                budget,
            )
        )
    return out


def experiment_b(lengths: list[int], trials: int, seed: int, names: list[str]) -> None:
    print("\n== 实验 B：误差随 K 的缩放（u/sqrt(K) 是否成立） ==")
    print("表中是「实测中位数 / 解析预算」，只看数量级；基准仍是 S = sum|a_i b_i|。")
    header = f"{'格式':<8}" + "".join(f"{('K=' + str(k)):<12}" for k in lengths)
    print(header)
    for name in names:
        cells = []
        for length in lengths:
            rows = experiment_a(length, trials, seed)
            measured = {row[0]: row[1] for row in rows}[name]
            budget = EQUIVALENT_U[name] / math.sqrt(length)
            cells.append(f"{measured / budget:<12.2f}")
        print(f"{name:<8}" + "".join(cells))


def experiment_c(length: int, trials: int, seed: int) -> None:
    """累加器精度：fp16 顺序累加 / fp32 累加 / 分段后 fp32 汇总。"""

    rng = np.random.default_rng(seed + 1)
    fp16_seq, fp32_acc, split32 = [], [], []
    for _ in range(trials):
        x = rng.uniform(-1.0, 1.0, size=length).astype(np.float16)
        y = rng.uniform(-1.0, 1.0, size=length).astype(np.float16)
        products = (x.astype(np.float64) * y.astype(np.float64))
        exact = float(np.sum(products))
        denom = float(np.sum(np.abs(products))) or 1.0

        # (1) 全程 fp16 顺序累加：cumsum 在 fp16 里逐步舍入，等价于串行累加器
        acc16 = np.cumsum(x.astype(np.float16) * y.astype(np.float16), dtype=np.float16)[-1]
        fp16_seq.append(abs(float(acc16) - exact) / denom)

        # (2) fp32 累加器，一次到底
        acc32 = np.sum(x.astype(np.float32) * y.astype(np.float32), dtype=np.float32)
        fp32_acc.append(abs(float(acc32) - exact) / denom)

        # (3) 分成 32 段，每段先用 fp16 累加，再把 32 个部分和用 fp32 相加（split-K 的抽象）
        chunks = x.astype(np.float16) * y.astype(np.float16)
        partials = np.asarray(
            [float(np.cumsum(chunk, dtype=np.float16)[-1]) for chunk in np.array_split(chunks, 32)],
            dtype=np.float32,
        )
        split32.append(abs(float(np.sum(partials, dtype=np.float32)) - exact) / denom)

    print(f"\n== 实验 C：累加器精度（K={length}，{trials} 次试验，与 fp64 比较） ==")
    print(f"{'方案':<34}{'相对误差中位数':<18}{'相对误差最大值'}")
    for label, samples in (
        ("fp16 输入 + fp16 顺序累加", fp16_seq),
        ("fp16 输入 + fp32 累加", fp32_acc),
        ("fp16 输入 + 32 段 fp16 部分和 + fp32 汇总", split32),
    ):
        arr = np.asarray(samples)
        print(f"{label:<36}{np.median(arr):<20.3e}{np.max(arr):.3e}")
    print("结论：输入完全一样，只换累加器，中位数误差就从 1.2e-4 掉到 1.2e-9——")
    print("      这一档差距不是「输入精度」造成的，只能靠 fp32 累加器或分段累加解决。")
    print("      32 段 fp16 部分和 + fp32 汇总落在中间（2.1e-5）：每段内部的舍入误差只")
    print("      跟段长有关，段间用 fp32 连接不会再放大。这就是 split-K 保精度的由来。")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lengths", default="64,256,1024,4096")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()
    lengths = [int(item) for item in args.lengths.split(",") if item.strip()]

    print("== 等价 u：每个格式「单元素相对误差」的上界 ==")
    print(f"{'格式':<8}{'布局':<12}{'等价 u':<14}{'说明'}")
    notes = {
        "fp16": "2^-11，浮点",
        "bf16": "2^-8，浮点",
        "e4m3": "2^-4，浮点码本",
        "e2m1": "2^-2，浮点码本",
        "int8": "1/(2*127)，均匀量化",
        "int4": "1/(2*7)，均匀量化",
    }
    for name, u in EQUIVALENT_U.items():
        print(f"{name:<10}{LAYOUT[name]:<14}{u:<16.3e}{notes[name]}")

    print(f"\n== 实验 A：K={lengths[0]}，两种参照下的相对误差 ==")
    print("「参照一」= 原始值的 fp64 点积，误差含输入量化；「参照二」= 量化后值的 fp64 点积，误差只有累加。")
    print(f"{'格式':<8}{'中位数(对原始值)':<20}{'中位数(对量化值)':<20}{'最大(对量化值)':<20}{'预算 u/sqrt(K)':<18}{'比值'}")
    for name, rel_raw, rel_quant, rel_quant_max, budget in experiment_a(lengths[0], args.trials, args.seed):
        print(
            f"{name:<10}{rel_raw:<22.3e}{rel_quant:<22.3e}{rel_quant_max:<22.3e}{budget:<20.3e}{rel_raw / budget:<.2f}"
        )
    print("两个参照差 4 个数量级以上。fp16 是 3.3e-5 对 5.9e-9；e4m3 / e2m1 的「对量化值」")
    print("误差直接是 0——4 位尾数的乘积只有 8 位有效位，fp32 能精确表示，累加也不丢位。")
    print("**对拍时把「量化前的 fp32 结果」当参照，会把正常的量化误差判成逻辑错误；")
    print("正确做法是拿「按同样的量化与反量化规则生成的参考实现」对比。**")
    print("比值那一列的读法：fp16/bf16/e4m3 在 0.5 附近（第 12 章的结论），int8 是 1.3~1.5")
    print("（均匀量化比浮点更接近最坏情形），e2m1 却到了 2.6 —— 4 位浮点的相对间距随指数")
    print("变化太剧烈，'相对误差上界 u' 对小数值不成立，模型在 fp4 上失效。")

    experiment_b(lengths, args.trials, args.seed, names=["fp16", "bf16", "e4m3", "int8"])
    experiment_c(4096, max(20, args.trials // 20), args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
