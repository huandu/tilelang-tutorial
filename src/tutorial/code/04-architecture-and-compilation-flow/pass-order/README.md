# pass-order：从源码里读出真实的 pass 顺序

## 目的

正文 4.5 / 4.6 节的两张 pass 序列表格，就是本脚本的输出。它回答的是
「`tilelang/cuda/pipeline.py` 与 `tilelang/ascend/pipeline.py` 实际按什么顺序跑 pass」，
而不是「文档说应该按什么顺序」。

## 为什么不能靠 import

直接 `import tilelang` 跑 pipeline 需要编译好的 `libtilelang`、TVM 与真实设备。
本脚本改用**逐行解析源码**的办法：只认 `mod = <pass>(...)` 这一种语句，记录它的
1-based 行号和当前所处的 `if` 守卫。这样得到的结果与编译器实际执行顺序一致
（因为 pipeline 就是一段顺序 Python），而且完全离线。

## 运行前提

* Python 3.8+（只用到标准库 `argparse` / `re` / `pathlib`）。
* `_vendor/tilelang-v0.1.15/tilelang/{cuda,ascend}/pipeline.py` 存在。
* **不需要** GPU/NPU，**不需要** `pip install tilelang`。

## 运行命令

```bash
python3 dump_pass_order.py                 # Markdown 表格（正文表格）
python3 dump_pass_order.py --format csv    # 机器可读版本
python3 dump_pass_order.py --vendor /path/to/another/checkout
```

## 期望输出

在 v0.1.15 上应当得到：

```text
## CUDA — pipeline.py（73 个 pass 调用）
  ...
## Ascend — pipeline.py（80 个 pass 调用）
  ...
## 汇总
* CUDA 共 73 步，Ascend 共 80 步。
* 两侧都出现的 pass 名共 39 个：...
* 带条件守卫：CUDA 7 步，Ascend 17 步。
```

**正确性判据**：CUDA 的 73 步 = `CUDAPassPipelineBodyPrologue`（33 步）+ `Body`
中 prologue 之后的 40 步；Ascend 的 80 步全部来自 `AscendPassPipelineBody`。
如果数字对不上，先确认 `--vendor` 指的是不是 v0.1.15（`cat VERSION` 应为 `0.1.15`）。

## 已知的口径细节

* 脚本只统计 `mod = ...` 形式，因此不会把 `pass_ctx = get_pass_context()` 这类辅助调用
  算成 pass。代价是：如果未来某个 pass 结果不赋给 `mod`，脚本会漏掉它。
* `mod = CUDAPassPipelineBodyPrologue(...)` 这一行本身不计入，改为把 Prologue 函数体内的
  33 步展开平铺，这样表格顺序 = 实际执行顺序。
* 守卫是**词法**推断（最近的、尚未闭合的 `if`），不是数据流分析。`if` 的取值由
  `tilelang/backend/pass_pipeline/pipeline_utils.py` 与 `PassConfigKey` 决定，正文第 5 节有例子。

## 验证级别

**在本机运行过**（macOS Apple Silicon，Python 3.11，2026-10-01）。输出与正文表格一致。
