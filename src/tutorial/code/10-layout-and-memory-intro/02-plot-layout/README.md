# 示例 02：用 `plot_layout` 看布局

## 目的

对应正文第 3.4 节“如何观察 layout”。把三种 layout 画成图，学会读它：

1. 手写的 `T.Layout([4,4], lambda i, j: (j, i))` —— 一个转置映射；
2. 手写的 `T.Fragment([8,8], forward_fn=lambda i, j: (4*i + j//2, j % 2))`
   —— 每个格子标 `T<线程>` / `L<槽>`，就是正文图 4；
3. `make_full_bank_swizzled_layout` 作用在 `(8, 64) fp16` 共享 buffer 上的结果
   —— 就是正文图 3 的 128B 行。

## 运行前提

* TileLang `v0.1.15`（`pip install "tilelang[vis]"`；`[vis]` 会装上 Matplotlib）。
* **不需要 GPU**：layout 是纯索引映射，`plot_layout` 在 Python 里逐格枚举。
* 本机**没有安装 TileLang**，所以本示例没有在本机运行过。

## 运行命令

```bash
python3 plot_layout_demo.py
# 产物写到 ./tmp/：
#   transpose_4x4.png、transpose_4x4_output_view.png
#   fragment_8x8.png
#   swizzle_full_8x64.png、swizzle_full_8x64_output_view.png
```

## 期望看到什么

* `view="input"`（默认）：格子代表**输入**坐标，格子里写的是它被映射到的**输出**（展平）
  坐标。转置图里 `(i, j)` 的格子里是 `j*4 + i`。
* `view="output"`：格子代表**输出**空间，格子里写回输入坐标。
* Fragment 图：格子按线程号染色，标签是 `T<线程> L<槽>`；8×8 的 fragment 应显示
  32 种颜色、每种颜色 2 格。
* Swizzle 图：输出侧的格子分布应呈现“按行异或”的错位，而不是逐行相同。

## API 依赖（逐个在 v0.1.15 里核对过）

| 调用 | 定义位置 |
| --- | --- |
| `T.Layout(shape, forward_fn)` | `_vendor/tilelang-v0.1.15/tilelang/layout/layout.py:12-43` |
| `T.Fragment(shape, forward_fn=...)` | `_vendor/tilelang-v0.1.15/tilelang/layout/fragment.py:23-99` |
| `make_full_bank_swizzled_layout(buffer)` | `_vendor/tilelang-v0.1.15/tilelang/layout/swizzle.py:94-102` |
| `plot_layout(layout, save_directory, name, colormap, verbose, formats, view, grid_shape)` | `_vendor/tilelang-v0.1.15/tilelang/tools/plot_layout.py:6-61` |
| `tirx.decl_buffer((8,64), "float16", scope="shared")` 作为 swizzle 输入 | `_vendor/tilelang-v0.1.15/testing/python/layout/test_tilelang_bank_swizzle_expand.py:23-26` |

## 验证级别

**仅静态核对**：

* `python3 -m py_compile plot_layout_demo.py` 通过；
* 每个 API 的存在性与签名逐个在 `_vendor/tilelang-v0.1.15/` 中比对；
* **未在本机运行**（本机无 TileLang、无 GPU）。运行后请以实际产物为准，不要引用本
  README 里的“期望看到什么”当作已观测结果。

## 注意：与仓库示例的一处 API 漂移

`_vendor/tilelang-v0.1.15/examples/plot_layout/layout_swizzle.py` 里调用
`make_full_bank_swizzled_layout(8, 64, element_size)`（三个位置参数），而 v0.1.15 的库
签名是 `make_full_bank_swizzled_layout(buffer)`（只接一个 buffer；见
`tilelang/layout/swizzle.py:94-102`，也见 `testing/python/layout/test_tilelang_bank_swizzle_expand.py`
的用法）。本示例按**库签名**书写。这个不一致已登记在 `../../references.md` 的
“待补资料 / 待确认”小节。
