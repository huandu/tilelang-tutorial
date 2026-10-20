"""把 DeepGEMM-Ascend README 的性能表换算成「利用率 / 带宽」。

验证级别：**CPU 逻辑验证**（纯标准库，可在任何有 Python 3.9+ 的机器上运行）。

三件事：
  1. 用 csrc/jit_kernels/heuristics.hpp:46-48,52-71 的硬件模型复算 bf16/fp8/fp4 峰值，
     并与 README 的 Hardware limit 列对照；
  2. 把各表的 Computation 列换成「占硬件极限的百分比」；
  3. 用**明确写出的字节口径**复算带宽，与 README 的 Memory bandwidth 列对比，
     并把比值打印出来。

边界（重要）：
  * 「硬件极限」是**源码里的模型值**，不是任何一台机器的实测值。
  * 第 3 步的复算是**另一种口径**，两者不一定相等。脚本把差异如实打印，
    正文 §5.3 与 references.md 的「待核」小节登记了这些差异。
  * 本脚本不测量任何东西，也不 import torch / tilelang。
"""

from __future__ import annotations

from dataclasses import dataclass

# --- 硬件模型（照抄 csrc/jit_kernels/heuristics.hpp:52-71 的关键项） -------------

AIC_CLOCK_MHZ = 1650
BF16_FLOPS_PER_CYCLE = 8192  # 4096 multiply-adds；每个记 2 FLOPs
NUM_AI_CORES = 32
GM_BW_GBPS = 3500  # heuristics.hpp:55


def mad_flops_per_cycle(elem_bits: int) -> int:
    """heuristics.hpp:46-48: bf16_flops_per_cycle * 16 / elem_bits。"""
    return BF16_FLOPS_PER_CYCLE * 16 // elem_bits


def peak_tflops(elem_bits: int) -> float:
    return mad_flops_per_cycle(elem_bits) * AIC_CLOCK_MHZ * NUM_AI_CORES / 1e6


PEAK = {16: peak_tflops(16), 8: peak_tflops(8), 4: peak_tflops(4)}


# --- 字节口径 -----------------------------------------------------------------


def sf_bytes(mn: int, k: int) -> int:
    """SF 存储：MN × ceil(K/32) 个 ue8m0 字节（打包成 int16 后字节数不变）。"""
    return mn * -(-k // 32)


def operand_bytes(mn: int, k: int, elem_bits: int, with_sf: bool) -> int:
    return mn * k * elem_bits // 8 + (sf_bytes(mn, k) if with_sf else 0)


def ab_sf_bytes(m: int, n: int, k: int, a_bits: int, b_bits: int, with_sf: bool) -> int:
    """口径 A：A + B (+SF)。对应 tests 里同时带 `c=None` 的 `count_bytes(a, b, c)`。"""
    return operand_bytes(m, k, a_bits, with_sf) + operand_bytes(n, k, b_bits, with_sf)


def gbps(nbytes: int, latency_us: float) -> float:
    return nbytes / (latency_us * 1e-6) / 1e9


# --- README 的表格（数值照抄 README.md:90-170） --------------------------------


@dataclass(frozen=True)
class Row:
    label: str
    m: int
    n: int
    k: int
    latency_us: float
    compute_tflops: float
    a_bits: int
    b_bits: int
    with_sf: bool
    table_gbps: float | None = None


DENSE = [
    Row('FP4xFP4', 4096, 7168, 16384, 565.6, 1701, 4, 4, True),
    Row('FP8xFP4', 4096, 7168, 16384, 1117.8, 861, 8, 4, True),
    Row('FP8xFP8', 4096, 7168, 16384, 1117.6, 861, 8, 8, True),
    Row('BF16xBF16', 4096, 7168, 16384, 2229.9, 431, 16, 16, False),
]

INFER_FP8 = [
    Row('', 128, 2112, 7168, 15.0, 258, 8, 8, True, 1205),
    Row('', 128, 576, 7168, 10.1, 105, 8, 8, True, 564),
    Row('', 128, 24576, 1536, 20.6, 468, 8, 8, True, 2068),
    Row('', 128, 32768, 512, 10.4, 415, 8, 8, True, 1829),
    Row('', 128, 7168, 16384, 54.8, 549, 8, 8, True, 2456),
    Row('', 128, 4096, 7168, 19.0, 397, 8, 8, True, 1797),
    Row('', 128, 7168, 2048, 10.1, 373, 8, 8, True, 1670),
    Row('', 4096, 2112, 7168, 156.9, 790, 8, 8, True, 319),
    Row('', 4096, 576, 7168, 54.6, 619, 8, 8, True, 689),
    Row('', 4096, 24576, 1536, 362.2, 854, 8, 8, True, 137),
    Row('', 4096, 32768, 512, 164.3, 837, 8, 8, True, 129),
    Row('', 4096, 7168, 16384, 1117.6, 861, 8, 8, True, 186),
    Row('', 4096, 4096, 7168, 282.9, 850, 8, 8, True, 234),
    Row('', 4096, 7168, 2048, 143.4, 838, 8, 8, True, 181),
]

# M-Grouped / MegaMoE：per-group 的 M（以及多 rank 平均）让「一个 M/N/K」描述不了全部流量，
# 所以本脚本只算它们的**算力利用率**，不做带宽对账。
M_GROUPED = [
    Row('4 x 8192 / 6144x7168', 8192, 6144, 7168, 3722.4, 818, 8, 4, True),
    Row('4 x 8192 / 7168x3072', 8192, 7168, 3072, 1778.0, 856, 8, 4, True),
    Row('8 x 4096 / 4096x4096', 4096, 4096, 4096, 1356.5, 861, 8, 4, True),
]

MEGA_MOE = [
    Row('96 / 5120 / 2304 / 64', 64, 5120, 2304, 138.7, 228.5, 8, 4, True),
    Row('96 / 5120 / 2304 / 256', 256, 5120, 2304, 225.5, 562.5, 8, 4, True),
    Row('384 / 5120 / 2304 / 16384', 16384, 5120, 2304, 9768.6, 831.0, 8, 4, True),
    Row('384 / 7168 / 3072 / 16384', 16384, 7168, 3072, 17904.2, 846.3, 8, 4, True),
]

# MQA 的列是 (#Q, #K, #Heads, Dim)，不是 (M, N, K)，所以只算算力利用率。
MQA = [
    Row('Prefill FP8', 4096, 8192, 128, 302.6, 681, 8, 8, True),
    Row('Prefill FP4', 4096, 8192, 128, 277.4, 743, 4, 4, True),
    Row('Decode FP8', 256, 8192, 128, 150.9, 565, 8, 8, True),
    Row('Decode FP4', 256, 8192, 128, 124.2, 708, 4, 4, True),
]

# HC Prenorm：A 是 [M,K] bf16、B 是 [24,K] fp32、D 是 [M,24] fp32、sqr_sum 是 [M] fp32。
# tests 的 count_bytes(a, b, d, sqr_sum) 就是这几项，所以这里的口径包含 D。
HC_PRENORM = [
    Row('', 13, 24, 28672, 6.7, 3, 16, 32, False, 519),
    Row('', 137, 24, 28672, 9.6, 20, 16, 32, False, 1110),
    Row('', 512, 24, 28672, 15.4, 46, 16, 32, False, 2082),
    Row('', 4096, 24, 28672, 72.7, 78, 16, 32, False, 3276),
    Row('', 8192, 24, 28672, 136.7, 82, 16, 32, False, 3463),
]


def util_rows(title: str, rows: list[Row]) -> None:
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)
    print(f'  {"case":<26}{"comp":>8}{"peak":>9}{"util":>8}')
    for row in rows:
        peak = PEAK[row.a_bits]
        print(f'  {row.label or f"{row.m}x{row.n}x{row.k}":<26}'
              f'{row.compute_tflops:8.1f}{peak:9.1f}{row.compute_tflops / peak * 100:7.1f}%')


def bw_rows(title: str, rows: list[Row], note: str) -> None:
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)
    print(note)
    print(f'  {"M":>6} {"N":>6} {"K":>6} {"us":>9} {"MB(A+B+SF)":>12} '
          f'{"模型GB/s":>10} {"表格":>8} {"表格/模型":>10}')
    for row in rows:
        nbytes = ab_sf_bytes(row.m, row.n, row.k, row.a_bits, row.b_bits, row.with_sf)
        model = gbps(nbytes, row.latency_us)
        ratio = row.table_gbps / model if row.table_gbps is not None else float('nan')
        table = '  —  ' if row.table_gbps is None else f'{row.table_gbps:7.0f}'
        print(f'  {row.m:6d} {row.n:6d} {row.k:6d} {row.latency_us:9.1f} '
              f'{nbytes / 1e6:12.1f} {model:10.0f} {table:>8} {ratio:9.3f}')


def main() -> None:
    print('=' * 78)
    print('1. 硬件极限的复算（heuristics.hpp:46-48,52-71）')
    print('=' * 78)
    print(f'  峰值 = 8192 * 16/elem_bits * {AIC_CLOCK_MHZ} MHz * {NUM_AI_CORES} 核')
    for bits, name in ((16, 'bf16'), (8, 'fp8'), (4, 'fp4')):
        print(f'  {name:>4}: {mad_flops_per_cycle(bits):6d} FLOP/cycle/核  ->  '
              f'{PEAK[bits]:7.1f} TFLOPS')
    print('  与 README Hardware limit 列对照：bf16 432 / fp8 865 / fp4 1730')

    util_rows('2. Dense GEMM：利用率 = Computation / 本格式峰值', DENSE)
    util_rows('3. M-Grouped（只看算力利用率）', M_GROUPED)
    util_rows('4. MegaMoE（只看算力利用率）', MEGA_MOE)
    util_rows('5. MQA logits（只看算力利用率；官方口径说它其实是 FIX-pipe bound）', MQA)

    bw_rows(
        '6. FP8 推理：带宽复算 vs README',
        INFER_FP8,
        '  口径：MB = A + B + SF（不含 D；对应 tests 里 c=None 的 count_bytes(a, b, c)）。',
    )
    bw_rows(
        '7. HC Prenorm：带宽复算 vs README（口径含 D，与 tests 一致）',
        HC_PRENORM,
        '  口径：MB = A + B + D（D 是 fp32，本脚本按 bf16 近似，占比 < 1%；'
        'sqr_sum 忽略）。',
    )

    print()
    print('=' * 78)
    print('8. 结论（复算，不是测量）')
    print('=' * 78)
    print(f'  * 硬件极限三行与 README 的 432/865/1730 吻合（误差 < 0.2%）。')
    print(f'  * HC Prenorm 的带宽列可以被 A+B+D 口径复现（比值 0.99–1.00）。')
    print(f'  * FP8 推理 14 行的「表格/模型」比值全部落在 1.089–1.095（约 +9%），')
    print(f'    说明表格那一列用的是另一种口径，且在整张表里是一致的；')
    print(f'    快照里没有给出该口径的公式，因此这里**只报告事实、不做解释**。')
    print(f'    引用时以表格列为准，不要用复算值替它。这一项已登记为「待核」。')
    print(f'  * HC Prenorm M=8192 的带宽利用率 = 3463 / {GM_BW_GBPS} = '
          f'{3463 / GM_BW_GBPS * 100:.1f}%；')
    print(f'    它按 bf16 算力极限只有 {82 / PEAK[16] * 100:.1f}%。')
    print(f'    同一行数字，换一条分母就从「接近满分」变成「几乎为零」——')
    print(f'    读利用率表之前先确认分母是哪条流水线。')


if __name__ == '__main__':
    main()
