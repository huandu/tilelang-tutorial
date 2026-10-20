#!/usr/bin/env python3
"""第 38 章 · 示例 03：复算 TokenLayout / BufferLayout 与 get_buffer_size_hint。

把 DeepEP-Ascend 里这几段的整数运算照抄成 Python，便于读者对着源码读：

* ``deep_ep/include/deep_ep/layout/ep/token.hpp:36-43``  ``TokenLayout::get_num_bytes``
* ``deep_ep/include/deep_ep/layout/ep/token.hpp:102-112``  ``BufferLayout::get_num_bytes``
* ``csrc/buffers/ep.hpp:560-605``  ``get_dispatch_buffer_size`` / ``get_combine_buffer_size``
  / ``calculate_buffer_size``

作者验证级别：**CPU 逻辑验证**。纯标准库，不连接设备。
"""

from __future__ import annotations

import argparse

UB_ALIGN = 32            # kNumUbAlignmentBytes（common/compiled.hpp:38）
ALLOC_ALIGN = 2 << 20    # kNumAllocationAlignmentBytes = 2 MB（common/compiled.hpp:48）
SF_PACK_BYTES = 2        # sizeof(sf_pack_t) = sizeof(int16_t)（compiled.hpp:33）


def align(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def token_layout_bytes(hidden_bytes: int, num_sf_packs: int, num_topk: int,
                       with_metadata: bool = True) -> int:
    """对应 TokenLayout（token.hpp:9-43）。"""
    num_sf_bytes = num_sf_packs * SF_PACK_BYTES
    num_metadata_bytes = num_topk * 4 + (
        align(num_topk * 8, UB_ALIGN) + 4 if with_metadata else 0
    )
    if num_sf_bytes > 0:
        return hidden_bytes + align(num_sf_bytes, UB_ALIGN) + align(num_metadata_bytes, UB_ALIGN)
    return hidden_bytes + align(num_metadata_bytes, UB_ALIGN)


def buffer_layout_bytes(hidden_bytes: int, num_sf_packs: int, num_topk: int,
                        num_buffers: int, num_max_tokens: int,
                        with_metadata: bool = True) -> int:
    """对应 BufferLayout（token.hpp:85-130）。"""
    per_token = token_layout_bytes(hidden_bytes, num_sf_packs, num_topk, with_metadata)
    return num_buffers * num_max_tokens * per_token


def dispatch_buffer_bytes(num_ranks, num_max_tokens, hidden, num_sf_packs, num_topk,
                          elem_size) -> int:
    """对应 get_dispatch_buffer_size（ep.hpp:560-569）。"""
    return buffer_layout_bytes(hidden * elem_size, num_sf_packs, num_topk,
                               num_ranks, num_max_tokens, with_metadata=True)


def combine_buffer_bytes(num_ranks, num_max_tokens, hidden, num_topk) -> int:
    """对应 get_combine_buffer_size（ep.hpp:571-579）。"""
    return buffer_layout_bytes(hidden * 2, 0, num_topk,
                               min(num_topk, num_ranks), num_max_tokens, with_metadata=False)


def calculate_buffer_size(num_ranks, num_max_tokens, hidden, num_topk, use_fp8) -> int:
    """对应 calculate_buffer_size（ep.hpp:581-605）。"""
    if num_topk == 0:
        num_topk = 32  # NOTES: dispatch requires kNumTopk <= 32
    # NOTES: hint 用 ceil_div(hidden, 32) 近似 SF 数量（ep.hpp:593-594）
    num_sf_packs = ceil_div(hidden, 32) if use_fp8 else 0
    elem_size = 1 if use_fp8 else 2
    d = dispatch_buffer_bytes(num_ranks, num_max_tokens, hidden, num_sf_packs, num_topk, elem_size)
    c = combine_buffer_bytes(num_ranks, num_max_tokens, hidden, num_topk)
    return align(max(d, c), ALLOC_ALIGN)


def main() -> int:
    ap = argparse.ArgumentParser(description="复算 EPBuffer 的容量公式")
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--experts", type=int, default=256)
    args = ap.parse_args()

    ranks, T, hidden, topk = args.ranks, args.tokens, args.hidden, args.topk

    # 实际运行时的 SF pack 数：128 元素一个 fp32 scale → view(int16) 后每个 scale 变两个 int16
    actual_sf_packs = ceil_div(hidden, 128) * 2
    hint_sf_packs = ceil_div(hidden, 32)

    print("TokenLayout（每 token 字节）")
    print(f"  dispatch(with metadata, {actual_sf_packs} 个 SF pack): "
          f"{token_layout_bytes(hidden, actual_sf_packs, topk, True)} B")
    print(f"  combine (no metadata, 无 SF)           : "
          f"{token_layout_bytes(hidden * 2, 0, topk, False)} B")
    print()
    print(f"get_buffer_size_hint(num_ranks={ranks}, T={T}, hidden={hidden}, "
          f"topk={topk}, use_fp8_dispatch=True)")
    print(f"  hint 用的 SF pack 数 = ceil(hidden/32) = {hint_sf_packs} "
          f"（实际是 {actual_sf_packs}，多预留 {hint_sf_packs - actual_sf_packs} 个）")
    disp = dispatch_buffer_bytes(ranks, T, hidden, hint_sf_packs, topk, 1)
    comb = combine_buffer_bytes(ranks, T, hidden, topk)
    print(f"  dispatch 布局 = {disp:,} B")
    print(f"  combine  布局 = {comb:,} B")
    print(f"  max           = {max(disp, comb):,} B")
    hint = calculate_buffer_size(ranks, T, hidden, topk, True)
    print(f"  对齐到 2 MB   = {hint:,} B = {hint / (1 << 20):.2f} MiB")
    print()
    print("用实际 SF 数量（而非 hint 的 2 倍）重算 dispatch 布局：")
    disp_real = dispatch_buffer_bytes(ranks, T, hidden, actual_sf_packs, topk, 1)
    print(f"  dispatch 布局 = {disp_real:,} B（比 hint 少 {disp - disp_real:,} B）")
    print(f"  真正需要的容量 = {align(max(disp_real, comb), ALLOC_ALIGN):,} B")
    print()
    print("注意：hint 的偏大是**故意**的（ep.hpp:593 的注释），用于安全规划；")
    print("它不会影响实际运行时的字节流量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
