# 03 · 让编译器画出它推断出的布局

## 目的

对应第 21 章 §6.3。打开 `TL_LAYOUT_VISUALIZATION_ENABLE`，编译一个 GEMM，观察
编译器对 `C_local` 等 fragment **实际推断出**的布局
（`_vendor/tilelang-v0.1.15/tilelang/analysis/layout_visual.py:47-112`）。

## 运行前提

* TileLang `v0.1.15`；`matplotlib` 用于出图（`pip install "tilelang[vis]"`）。
* **想真的编到 CUDA 并跑 kernel**：需要 NVIDIA GPU + 驱动 + 工具链 + PyTorch(CUDA)。
* 没有 GPU 时，仍可取到**文本**形式（`formats="txt"`），前提是 TileLang 能完成
  lowering。

## 运行命令

```bash
# 文本 + SVG（需要有能编译的环境）
python3 gemm_layout_visualization.py
# 结果：控制台打印 C_local 的 Shape / Thread / Index / Replicate 四行，
#       ./tmp/ 下生成 <buffer>_layout.svg（如 C_local_layout.svg）

# 只要文本：
python3 gemm_layout_visualization.py --txt-only
```

## 预期输出

脚本本身不硬编码任何布局表达式；它把编译器打印出来的内容原样交给你。
官方示例 `_vendor/tilelang-v0.1.15/examples/visual_layout_inference/
visual_layout_inference.py` 对一个 `32/32/32, threads=128` 的 GEMM 记录过如下期望
输出（**属于该示例文件的内容，不是本脚本的运行结果**）：

```text
C_local inferenced layout:
  Shape: [32, 32] -> [8]
  Thread: _j // 16 * 64 + _i // 16 * 32 + _i % 8 * 4 + _j % 8 // 2
  Index:  [_j % 16 // 8 * 4 + _i % 16 // 8 * 2 + _j % 2]
```

## 验证级别

**仅静态核对**：`python3 -m py_compile` 通过；`PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE`
/ `..._FORMATS` 两个键在 `tilelang/transform/pass_config.py:252-256` 核对过，
`tilelang.jit(pass_configs=...)` 的用法照官方示例写。**本机未安装 TileLang，
脚本未运行，没有任何真实的布局图产物。**
