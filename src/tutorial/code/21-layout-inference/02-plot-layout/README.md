# 02 · 用 plot_layout 画一张 map 与一张 fragment

## 目的

对应第 21 章 §6.1。用官方 `tilelang.tools.plot_layout`（`_vendor/tilelang-v0.1.15/
tilelang/tools/plot_layout.py:6-58`）画三种布局：

1. 一个 `T.Layout` 转置映射（input view 与 output view）；
2. 一个手写的 128B swizzle（XOR）映射，观察行内的置换；
3. 一个 `T.Fragment` 基础片段（`forward_thread_fn` + `forward_index_fn`），
   复刻 `make_mma_load_layout` 的构造方式
   （`_vendor/tilelang-v0.1.15/tilelang/cuda/intrinsics/macro/mma_macro_generator.py:684-810`），
   并用 `repeat` / `replicate` 看看线程与槽位怎么扩散。

## 运行前提

* TileLang `v0.1.15` + `matplotlib`（`pip install "tilelang[vis]"`）。
* **不需要 GPU**（只画布局，不编译 kernel）。但 TileLang 的 import 需要能加载其
  FFI 库。

## 运行命令

```bash
python3 plot_layout_demo.py
# 结果：./tmp/ 下生成 transpose_4x4.png、swizzle_xor_8x32.png、
#       mma_base_fragment.png、（以及 repeat 后的 warp_fragment.png）
```

## 预期输出

控制台打印每个布局的字符串表示（`Layout` 的 `__repr__` 会显示 forward 表达式）。
本脚本**不打印任何性能数字**。

## 验证级别

**仅静态核对**：`python3 -m py_compile` 通过；用到的 API
（`T.Layout`、`T.Fragment`、`Fragment.repeat`、`Fragment.replicate`、`plot_layout`）
逐个在 `_vendor/tilelang-v0.1.15/` 里核对了签名（`layout.py:12-43`、
`fragment.py:24-186`、`plot_layout.py:6-58`）。**本机未安装 TileLang，脚本未运行。**

> 注意：示例目录 `examples/plot_layout/layout_swizzle.py` 里的
> `make_quarter_bank_swizzled_layout(8, 16, element_size)` 调用形式与 v0.1.15 的
> buffer 版签名不一致（见本章「待核」）。本脚本不使用该 API，改用手写的 XOR 映射，
> 以免踩到版本漂移。
