#!/usr/bin/env python3
"""Observe TileLang layouts with tilelang.tools.plot_layout (chapter 10, §3.4).

This is the "how to look at a layout" example. It plots four things:

  1. a hand-written T.Layout (4x4 transpose);
  2. a hand-written T.Fragment (2D -> (thread, local index));
  3. the 128B full-bank shared-memory swizzle of an 8x64 fp16 buffer;
  4. the same buffer read/viewed from the output side.

It needs TileLang (v0.1.15) and Matplotlib installed; it does NOT need a GPU:
layouts are pure index maps, and plot_layout enumerates them in Python.

Usage:
    pip install "tilelang[vis]"      # [vis] pulls in Matplotlib
    python3 plot_layout_demo.py
    # write-ups land under ./tmp/

Verified level: statically checked only. The authoring machine has no TileLang
installed and no GPU, so this file was NOT run; every API call was checked by
reading the v0.1.15 sources listed in the README. Do not treat its output as
observed until you run it.
"""

from __future__ import annotations

import tilelang.language as T
from tilelang.tools import plot_layout

OUT = "./tmp"


def demo_transpose_layout() -> None:
    """A layout is just a forward function: logical (i, j) -> physical (j, i)."""
    transpose = T.Layout([4, 4], lambda i, j: (j, i))
    print("transpose 4x4:", transpose)
    plot_layout(transpose, save_directory=OUT, name="transpose_4x4", formats="png")
    # view="output" draws the output space and labels cells with their source.
    plot_layout(
        transpose,
        save_directory=OUT,
        name="transpose_4x4_output_view",
        formats="png",
        view="output",
    )


def demo_fragment() -> None:
    """A fragment adds a thread mapping and a per-thread local index.

    Here every element (i, j) goes to thread (4*i + j//2) as that thread's
    register slot (j % 2). plot_layout labels each cell with T<thread> and
    L<local>; the picture is exactly figure 4 of the chapter.
    """
    fragment = T.Fragment([8, 8], forward_fn=lambda i, j: (4 * i + j // 2, j % 2))
    print("fragment 8x8:", fragment)
    plot_layout(fragment, save_directory=OUT, name="fragment_8x8", formats="png")


def demo_shared_swizzle() -> None:
    """Shared-memory bank swizzles are plain Layouts too.

    make_full_bank_swizzled_layout takes a *buffer*, not a shape: the swizzle
    depends on the buffer's dtype (to compute the 16B unit) and its strides.
    """
    from tvm import tirx
    from tilelang.layout import make_full_bank_swizzled_layout

    # 8x64 fp16: 128B swizzle needs continuous % (16B-unit * 8) == 0 -> 64 % 64 == 0.
    buf = tirx.decl_buffer((8, 64), "float16", name="A_shared", scope="shared")
    swizzled = make_full_bank_swizzled_layout(buf)
    print("full-bank swizzle:", swizzled)
    plot_layout(swizzled, save_directory=OUT, name="swizzle_full_8x64", formats="png")
    plot_layout(
        swizzled,
        save_directory=OUT,
        name="swizzle_full_8x64_output_view",
        formats="png",
        view="output",
    )


def main() -> None:
    demo_transpose_layout()
    demo_fragment()
    demo_shared_swizzle()
    print(f"PNG files written under {OUT}/; open them and read each cell's label.")


if __name__ == "__main__":
    main()
