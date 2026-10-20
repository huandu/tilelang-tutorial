"""从 heuristics.hpp 提炼的候选 / 容量模型（教学版，纯标准库）。

验证级别：**CPU 逻辑验证**。它不是 select_gemm_config 的移植，只复现其中
「容量约束」这一半——那部分是可以逐行对上的整数运算：

  * L0C 装得下 block_m*block_n（heuristics.hpp:100-102）
  * M-grouped 强制 block_m = 256（heuristics.hpp:106,124-130）
  * mad_m/mad_n/mad_k 的搜索（heuristics.hpp:114-146）
  * L1 里 A/B/SF 能放几级（heuristics.hpp:164-172）

代价模型（get_gemm_cost）与后面的几个 refine 补丁（direct_store / resident_* /
dual_aiv_dequant）**不在**本脚本里——那部分要看带宽、频率与 event 预算，
移植过来只会变成一份没法验证的副本，不如直接读源码。
"""

from __future__ import annotations

# --- 容量与常数（deep_gemm/include/deep_gemm/common.hpp:50-64） -----------------

L1_SIZE_BYTES = 512 * 1024
L0A_SIZE_BYTES = 64 * 1024
L0B_SIZE_BYTES = 64 * 1024
L0C_SIZE_BYTES = 256 * 1024
NUM_PIPE_EVENTS = 8
NUM_INTRA_BLOCK_EVENTS = 16
FRAC_MN = 16
MX_SF_DIVISOR = 64  # 两个 ue8m0 一个 int16 → 64 个 K 元素一对

# heuristics.hpp:57-59、:133-140
BF16_FLOPS_PER_CYCLE = 8192
AIC_CLOCK_MHZ = 1650
L1_TO_L0_BYTES_PER_CYCLE = 256


def elem_bits(dtype: str) -> int:
    return {'bf16': 16, 'fp8': 8, 'fp4': 4}[dtype]


def has_sf(dtype: str) -> bool:
    return dtype != 'bf16'


def frac_k(dtype: str) -> int:
    return MX_SF_DIVISOR if dtype == 'fp4' else 32


def k_alignment(dtype: str) -> int:
    return MX_SF_DIVISOR if has_sf(dtype) else 32


def candidates(dtype: str, block_k: int):
    """列出合法的 (block_m, block_n) 与它们各自的 MAD 形状。

    这是 make_gemm_candidate（heuristics.hpp:92-200）的容量部分。
    """
    bits = elem_bits(dtype)
    l0_stages = 2                      # heuristics.hpp:112
    align_mn = FRAC_MN
    align_k = k_alignment(dtype)
    out = []
    for block_m in (512, 256, 128, 64, 32, 16):
        # L0C：block_m * block_n * 4B 必须放得下（:100-102）
        max_block_n = L0C_SIZE_BYTES // (4 * block_m)
        for block_n in range(max_block_n, 15, -FRAC_MN):
            if block_n * 4 * block_m > L0C_SIZE_BYTES:
                continue
            # MAD_M/N/K 的搜索（:114-146）
            mad_m = mad_n = mad_k = 0
            for split_m, split_n in ((1, 1), (1, 2), (2, 1)):
                if block_m % (split_m * align_mn) or block_n % (split_n * align_mn):
                    continue
                cand_m, cand_n = block_m // split_m, block_n // split_n
                if cand_m * cand_n * 4 > L0C_SIZE_BYTES // 2:
                    continue
                l0a_k = L0A_SIZE_BYTES * 8 // (l0_stages * cand_m * bits)
                l0b_k = L0B_SIZE_BYTES * 8 // (l0_stages * cand_n * bits)
                cand_k = min(l0a_k, l0b_k, max(align_k, block_k // l0_stages))
                cand_k = cand_k // align_k * align_k
                while cand_k and block_k % cand_k:
                    cand_k -= align_k
                mte1_copy_k = L1_TO_L0_BYTES_PER_CYCLE * 8 // bits
                if cand_k * l0_stages < min(block_k, mte1_copy_k):
                    continue          # :140 的「mad_k 太小」判据
                mad_m, mad_n, mad_k = cand_m, cand_n, cand_k
                break
            if not (mad_m and mad_n and mad_k):
                continue
            # L1 里 A/B/SF 能放几级（:164-172）
            sf_k_blocks = min(4, -(-block_k // block_k)) if has_sf(dtype) else 0
            sf_stages = 2 if has_sf(dtype) else 0
            max_l1 = (min(4, NUM_PIPE_EVENTS - sf_stages) if has_sf(dtype)
                      else NUM_PIPE_EVENTS)
            sf_bytes = (block_m + block_n) * (block_k // MX_SF_DIVISOR) * 2 * sf_stages
            ab_bytes = (block_m + block_n) * block_k * bits // 8
            l1_stages = (L1_SIZE_BYTES - sf_bytes) // ab_bytes
            l1_stages = min(max_l1, l1_stages)
            out.append((block_m, block_n, mad_m, mad_n, mad_k, l1_stages, sf_stages))
    return out


def main() -> None:
    print('L0C 容量 =', L0C_SIZE_BYTES // 1024, 'KiB  L1 =', L1_SIZE_BYTES // 1024,
          'KiB  L0A/L0B =', L0A_SIZE_BYTES // 1024, 'KiB')
    print('每级 L1 的 A+B 字节数 = (block_m + block_n) * block_k * elem_bytes')
    for dtype, block_k in (('bf16', 128), ('fp8', 128), ('fp4', 256)):
        print()
        print('=' * 78)
        print(f'{dtype}（block_k={block_k}）：合法候选（block_m=512/256 各列前 3 行）')
        print('=' * 78)
        print(f'  {"blk_m":>6}{"blk_n":>6}{"mad_m":>7}{"mad_n":>7}{"mad_k":>7}'
              f'{"l1_stg":>8}{"sf_stg":>8}   L0C 占用')
        rows = candidates(dtype, block_k)
        for want_m in (512, 256):
            shown = 0
            for row in rows:
                block_m, block_n, mad_m, mad_n, mad_k, l1_stages, sf_stages = row
                if block_m != want_m:
                    continue
                l0c = block_m * block_n * 4
                flag = '  ← 占满 L0C' if l0c == L0C_SIZE_BYTES else ''
                print(f'  {block_m:6d}{block_n:6d}{mad_m:7d}{mad_n:7d}{mad_k:7d}'
                      f'{l1_stages:8d}{sf_stages:8d}   {l0c // 1024:4d} KiB{flag}')
                shown += 1
                if shown >= 3:
                    break

    print()
    print('=' * 78)
    print('结论（复算）')
    print('=' * 78)
    print('  * block_m=256 时 block_n 最多只能是 256：'
          f'256*256*4 = {256 * 256 * 4} B = 恰好 L0C_SIZE_BYTES。')
    print('  * 想让 block_n 更大，只能把 block_m 减半——'
          '这是「tile 尺寸由片上容量决定」最直接的一例。')
    print('  * mad_k 由 L0A/L0B 容量与 num_l0_stages=2 反推，'
          '并向下对齐到 k_alignment（bf16=32，带 SF=64）。')
    print('  * 带 SF 时 num_l1_stages 被 event 预算压到 <= 4'
          '（num_pipe_events=8 减去 SF 的 2 级）。')


if __name__ == '__main__':
    main()
