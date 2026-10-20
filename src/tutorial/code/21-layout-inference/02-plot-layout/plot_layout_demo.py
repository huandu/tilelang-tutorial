#!/usr/bin/env python3
"""第 21 章 §6.1：用 plot_layout 观察 Layout 与 Fragment。

仅需 TileLang + matplotlib，不需要 GPU。API 已按 v0.1.15 核对：
  tilelang/layout/layout.py:12-43     T.Layout(shape, forward_fn)
  tilelang/layout/fragment.py:24-186  T.Fragment(shape, forward_thread_fn, forward_index_fn, replicate)
  tilelang/tools/plot_layout.py:6-58  plot_layout(layout, save_directory, name, formats)
"""

from __future__ import annotations

import tilelang.language as T
from tilelang.tools import plot_layout


def xor_swizzle_128b(i: int, j: int):
    """一个简化的 128B（8 行 × 64 个 16-bit 元素）XOR swizzle。

    真实实现见 v0.1.15 的 src/layout/gemm_layouts.cc:442-530；这里只演示
    「行内按 XOR 置换列」这一条直觉，公式对 fp16 的 128B swizzle 成立。
    """
    row = i % 8
    return (i, j ^ ((row % 8) * 8))


def main() -> None:
    # 1) 转置：input 视图与 output 视图各画一张
    transpose = T.Layout([4, 4], lambda i, j: (j, i))
    print("transpose:", transpose)
    plot_layout(transpose, save_directory="./tmp", name="transpose_4x4",
                formats="png", view="input")
    plot_layout(transpose, save_directory="./tmp", name="transpose_4x4_out",
                formats="png", view="output")

    # 2) 手写 XOR swizzle：8 行 x 64 列（fp16 的 128B 行）
    swizzle = T.Layout([8, 64], xor_swizzle_128b)
    print("swizzle:", swizzle)
    plot_layout(swizzle, save_directory="./tmp", name="swizzle_xor_8x32",
                formats="png")

    # 3) 基础 fragment：复刻 mma load layout 的构造方式
    #    一个 8x8 的底座，thread = j//2 + 4*i，slot = j%2
    #    （对照 src/layout/gemm_layouts.cc:45-52 的 MakeGemmFragment8x8）
    def forward_thread(i, j):
        return j // 2 + 4 * i

    def forward_index(i, j):
        return j % 2

    base = T.Fragment([8, 8], forward_thread_fn=forward_thread,
                      forward_index_fn=forward_index)
    print("base fragment:", base)
    plot_layout(base, save_directory="./tmp", name="mma_base_fragment",
                formats="png")

    # 4) 在 fragment 上做一次 repeat：沿 warp 维复制（repeat_on_thread=False）
    warp = base.repeat([1, 1], repeat_on_thread=False, lower_dim_first=True)
    print("warp fragment:", warp)
    plot_layout(warp, save_directory="./tmp", name="warp_fragment",
                formats="png")


if __name__ == "__main__":
    main()
