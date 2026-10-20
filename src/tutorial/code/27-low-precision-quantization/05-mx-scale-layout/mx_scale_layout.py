#!/usr/bin/env python3
"""第 27 章 · 示例 5：block-scaled 的 scale 布局（CPU 逻辑验证）。

block-scaled GEMM 最容易错的地方不是 MMA，而是**scale 的打包与索引**：

* 4 个 E8M0 字节打包进 1 个 `uint32`（`w0 | w1<<8 | w2<<16 | w3<<24`）；
* 全局是 group-major 扁平布局（`SFA[g*M + m]`），不是行主序；
* 一个字节覆盖多少个 K 元素，由 `sf_granularity_k` 决定；NVIDIA 硬件原生的 MX
  粒度是 32，而锁定版本里的 SM100 MXFP8 示例用的是 128。

脚本用 numpy 复刻锁定版本示例里的三个函数
（`ceil_to_ue8m0` / `pack_sf_u8_to_u32_1d` / `unpack_sf_u32_1d`），把「正确的
打包 + 正确的解包」当作基准，再看四种典型错误各会造成多大的结果偏差。

对应源码：
    _vendor/tilelang-v0.1.15/examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py
        :518-525  pack_sf_u8_to_u32_1d
        :509-515  unpack_sf_u32_1d
        :543-546  ceil_to_ue8m0
        :557-571  amax -> scale -> fp8 -> 打包
        :575-620  blockscaled_gemm_ref（按 sf_granularity_k 分块的参考实现）

依赖：Python 3.9+ 与 numpy。不需要 GPU/NPU。

用法：

    python3 mx_scale_layout.py
    python3 mx_scale_layout.py --m 4 --n 8 --k 512 --gran 128

作者实际做到的验证级别：**CPU 逻辑验证**（写作机 macOS / Apple Silicon、
Python 3.11.7 + numpy 1.24.1 上实际运行过，退出码 0；输出见同目录 README.md）。
"""

from __future__ import annotations

import argparse
import math

import numpy as np


# --------------------------------------------------------------------------- #
# 与锁定版本示例一一对应的三个函数
# --------------------------------------------------------------------------- #


def ceil_to_ue8m0(x: np.ndarray) -> np.ndarray:
    """向上取到 2 的整数次幂（E8M0 可表示的最小上界）。

    锁定版本直接从 fp32 的位里抠指数：
        exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF != 0)
        return (clamp(exp, 1, 254) << 23).view(float32)
    这里用位运算完全照搬，保证与 `pack_sf_u8_to_u32_1d` 的字节视图一致。
    """

    bits = np.abs(x).astype(np.float32).view(np.uint32)
    exponent = ((bits >> np.uint32(23)) & np.uint32(0xFF)) + ((bits & np.uint32(0x7FFFFF)) != 0).astype(np.uint32)
    exponent = np.clip(exponent, 1, 254).astype(np.uint32)
    return (exponent << np.uint32(23)).view(np.float32)


def fp32_exponent_bytes(values: np.ndarray) -> np.ndarray:
    """把 float32 值转成它的 biased exponent 字节（就是 E8M0 的编码）。"""

    return ((values.astype(np.float32).view(np.uint32) >> 23) & 0xFF).astype(np.uint8)


def pack_sf_u8_to_u32_1d(sf_u8: np.ndarray) -> np.ndarray:
    """[mn, sf_k_padded] 的 E8M0 字节 -> group-major 的 uint32 扁平数组。"""

    mn, sf_k_padded = sf_u8.shape
    assert sf_k_padded % 4 == 0
    words = sf_u8.astype(np.int64)
    packed = (
        words[:, 0::4]
        | (words[:, 1::4] << 8)
        | (words[:, 2::4] << 16)
        | (words[:, 3::4] << 24)
    ).astype(np.uint32)
    return np.ascontiguousarray(packed.T).reshape(-1)


def unpack_sf_u32_1d(
    packed_sf: np.ndarray,
    mn: int,
    sf_k_blocks: int,
    *,
    layout: str = "group-major",
    byte_order: str = "lsb-first",
) -> np.ndarray:
    """uint32 -> [mn, sf_k_blocks] 的 E8M0 字节。

    `layout` 与 `byte_order` 是为了把两种典型的写错方式做成开关，默认值是锁定版本
    的行为（group-major 扁平 + 低字节放第一个 scale）。
    """

    groups = (sf_k_blocks + 3) // 4
    words = packed_sf.view(np.uint32)
    if layout == "group-major":
        packed_2d = words.reshape(groups, mn).T.astype(np.int64)
    elif layout == "row-major":
        packed_2d = words.reshape(mn, groups).astype(np.int64)
    else:
        raise ValueError(layout)
    unpacked = np.empty((mn, groups * 4), dtype=np.uint8)
    shifts = range(4) if byte_order == "lsb-first" else range(3, -1, -1)
    for position, shift in enumerate(shifts):
        unpacked[:, position::4] = ((packed_2d >> (8 * shift)) & 0xFF).astype(np.uint8)
    return unpacked[:, :sf_k_blocks].copy()


def e4m3_codebook() -> np.ndarray:
    values: set[float] = set()
    for exponent in range(16):
        for mantissa in range(8):
            if exponent == 0xF and mantissa == 0x7:
                continue
            values.add((mantissa / 8.0) * 2.0 ** (1 - 7) if exponent == 0 else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    return np.asarray(sorted(values), dtype=np.float64)


def round_to_codebook(magnitude: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(codebook, magnitude, side="left"), 1, len(codebook) - 1)
    lower, upper = codebook[idx - 1], codebook[idx]
    return np.where(magnitude - lower <= upper - magnitude, lower, upper)


# --------------------------------------------------------------------------- #
# 量化 + 打包 + 参考 GEMM
# --------------------------------------------------------------------------- #


def quantize_and_pack(x: np.ndarray, gran_k: int, codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (fp8 值, 解包后的 scale 字节 [mn, blocks], 打包后的 uint32 扁平数组)。"""

    mn, k = x.shape
    padded_k = int(math.ceil(k / gran_k) * gran_k)
    padded = np.zeros((mn, padded_k), dtype=np.float64)
    padded[:, :k] = x
    view = padded.reshape(mn, padded_k // gran_k, gran_k)

    amax = np.maximum(np.max(np.abs(view), axis=2), 1e-4)  # 与源码一致的下界
    scale = ceil_to_ue8m0(amax / float(codebook[-1]))      # 保证 amax/scale <= 448
    scaled = view / scale[:, :, None]
    fp8 = np.copysign(round_to_codebook(np.abs(scaled), codebook), scaled)
    fp8 = fp8.reshape(mn, padded_k)[:, :k].copy()

    sf_u8 = fp32_exponent_bytes(scale)
    blocks = sf_u8.shape[1]
    padded_blocks = int(math.ceil(blocks / 4) * 4)
    if padded_blocks != blocks:
        sf_padded = np.full((mn, padded_blocks), 127, dtype=np.uint8)  # 源码里的填充值
        sf_padded[:, :blocks] = sf_u8
    else:
        sf_padded = sf_u8
    return fp8, sf_u8, pack_sf_u8_to_u32_1d(sf_padded)


def blockscaled_gemm(
    a: np.ndarray,
    b: np.ndarray,
    sfa: np.ndarray,
    sfb: np.ndarray,
    gran_k: int,
) -> np.ndarray:
    """按 sf_granularity_k 分块的参考 GEMM（对应示例里的 blockscaled_gemm_ref）。"""

    k = a.shape[1]
    blocks = sfa.shape[1]
    out = np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    for bi in range(blocks):
        start, end = bi * gran_k, min((bi + 1) * gran_k, k)
        sa = np.ldexp(1.0, sfa[:, bi].astype(np.int32) - 127)[:, None]
        sb = np.ldexp(1.0, sfb[:, bi].astype(np.int32) - 127)[:, None]
        out += (a[:, start:end] * sa) @ (b[:, start:end] * sb).T
    return out


def relative_l2(approx: np.ndarray, exact: np.ndarray) -> float:
    return float(np.linalg.norm(approx - exact) / np.linalg.norm(exact))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--gran", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    codebook = e4m3_codebook()
    # 行级幅度（模拟不同通道的量级差）+ 沿 K 的块级幅度（模拟同一个通道里
    # 「某些 K 段明显更大」）。后者是让 scale 逐块变化的关键：如果每个 K 块的
    # amax 都差不多，四个字节就会一模一样，字节序写反也看不出来。
    def make(rows: int) -> np.ndarray:
        row_gain = np.exp(rng.normal(0.0, 2.0, size=(rows, 1)))
        k_gain = np.exp(np.repeat(rng.normal(0.0, 0.8, size=args.k // 32), 32))[None, :]
        return rng.standard_normal((rows, args.k)) * row_gain * k_gain

    a = make(args.m)
    b = make(args.n)

    a_fp8, sfa_u8, sfa_packed = quantize_and_pack(a, args.gran, codebook)
    b_fp8, sfb_u8, sfb_packed = quantize_and_pack(b, args.gran, codebook)
    blocks = sfa_u8.shape[1]

    print(f"形状：A[{args.m},{args.k}]、B[{args.n},{args.k}]，sf_granularity_k={args.gran}")
    print(f"每行 scale 字节数 = ceil(K/{args.gran}) = {blocks}；每 4 个字节打包成 1 个 uint32")
    print(f"SFA 扁平长度 = ceil({blocks}/4) * M = {sfa_packed.size}（group-major：SFA[g*M + m]）")
    print(f"打包后的第一个 word = 0x{int(sfa_packed[0]):08x} -> 字节 "
          f"{[(int(sfa_packed[0]) >> (8 * i)) & 0xFF for i in range(4)]}")
    print(f"解包回来与原始字节一致：{np.array_equal(unpack_sf_u32_1d(sfa_packed, args.m, blocks), sfa_u8)}")

    # 基准：正确布局
    ref = blockscaled_gemm(a_fp8, b_fp8, sfa_u8, sfb_u8, args.gran)
    print(f"\n正确布局下的输出范数 = {np.linalg.norm(ref):.6f}（作为下面所有偏差的分母）")

    print()
    print(f"{'解码方式':<44}{'相对 L2':<14}{'说明'}")

    # (0) 正确
    got = blockscaled_gemm(
        a_fp8, b_fp8, unpack_sf_u32_1d(sfa_packed, args.m, blocks),
        unpack_sf_u32_1d(sfb_packed, args.n, blocks), args.gran,
    )
    print(f"{'0 正确解包':<46}{relative_l2(got, ref):<16.3e}{'基准'}")

    # (1) 字节序反了
    sfa_bad = unpack_sf_u32_1d(sfa_packed, args.m, blocks, byte_order="msb-first")
    got = blockscaled_gemm(a_fp8, b_fp8, sfa_bad, sfb_u8, args.gran)
    print(f"{'1 字节序写反（w0 放最高字节）':<46}{relative_l2(got, ref):<16.3e}{'scale 索引整体错位'}")

    # (2) 行主序而不是 group-major
    sfa_bad = unpack_sf_u32_1d(sfa_packed, args.m, blocks, layout="row-major")
    got = blockscaled_gemm(a_fp8, b_fp8, sfa_bad, sfb_u8, args.gran)
    print(f"{'2 用行主序索引（SFA[m*G + g]）':<46}{relative_l2(got, ref):<16.3e}{'行与组互换'}")

    # (3) 生产端用 32 粒度、消费端按 128 粒度读
    a_fp8_32, sfa_u8_32, sfa_packed_32 = quantize_and_pack(a, 32, codebook)
    # 消费端以为每 128 个 K 元素一个 scale，于是只按自己的尺寸取前 groups*mn 个 word
    consumer_words = sfa_packed_32[: (blocks // 4) * args.m]
    sfa_bad = unpack_sf_u32_1d(consumer_words, args.m, blocks)
    got = blockscaled_gemm(a_fp8, b_fp8, sfa_bad, sfb_u8, args.gran)
    print(f"{'3 生产端粒度 32、消费端按 128 读':<46}{relative_l2(got, ref):<16.3e}{'scale 覆盖范围差 4 倍'}")

    print()
    print("== 结论 ==")
    print("1) 三种错误里，字节序与粒度错误会把结果整体推到「和真值同一个量级」的偏差，")
    print("   属于逻辑错误；行主序错误在 M 或组数等于 1 时可能恰好不出错（这也是它危险的地方）。")
    print("2) 打包与解包是一对互逆操作：只要有一侧写错，往返自测（pack -> unpack）就能发现；")
    print("   真正难查的是「两侧都错、但错得一致」——必须拿一个独立的参考实现做绝对对拍。")
    print(f"3) 粒度约定必须写进接口：NVIDIA 硬件原生的 MX 粒度是 32，锁定版本 SM100 示例用 {args.gran}，")
    print("   昇腾 CATLASS 的 MX_SCALE_GROUP_NUM 也是 32。跨实现对接时这一步最容易踩坑。")

    # 单组（K == gran）时两种索引恰好等价 —— 这就是行主序 bug 藏得住的原因。
    single_group = np.arange(1, args.m + 1, dtype=np.uint32)
    as_group = unpack_sf_u32_1d(single_group, args.m, 4)
    as_row = unpack_sf_u32_1d(single_group, args.m, 4, layout="row-major")
    print(f"\n补充：当组数 = 1（K 不超过 {4 * args.gran}）时，group-major 与 row-major 的解包结果相同：")
    print(f"      {np.array_equal(as_group, as_row)}。所以行主序这个 bug 只在 K 超过 4 个粒度块时才暴露。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
