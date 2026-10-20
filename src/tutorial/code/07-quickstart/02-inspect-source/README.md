# 02-inspect-source · 打印并对照 Python 与生成的 CUDA

## 目的

把“怎么读生成代码”变成一份可 diff 的文本：一组 Python → CUDA 的对照条目，
外加官方文档里两份真实的生成结果（已向量化版 / 带边界判断版）。

这不是替你跑 `get_kernel_source()`——那一步要在装了 tilelang 的真机上做。
它是“该看什么、该期待什么”的参照物。

## 运行前提

只要一台能跑 Python 3.10+ 的机器。**不需要 tilelang，不需要 GPU，不需要 torch。**

## 运行命令

```bash
python inspect_source.py
```

## 预期输出

先是“Python ↔ 生成 CUDA 对照表”，然后是两段标了出处的 CUDA 源码：

* `elementwise_add(1024, threads=256)` → `__launch_bounds__(256)` + `uint4` 向量化 +
  `nv_bfloat162` 成对加法；
* `N=2047` 的版本 → `#pragma unroll` 的 8 次循环 + `if (... < 2047)` 边界判断，
  没有向量化。

**正确性判据**：脚本退出码为 0，并且你能指出两份输出“哪一份做了向量化、为什么”。

## 验证级别

**在本机运行过（纯 Python，无 GPU）。** 打印的两段 CUDA 逐字取自锁定版本仓库中的
官方文档，来源写在输出里：

* `_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md:122-130`
* `_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md:155-172`

## 在真实机器上的验证步骤

1. 先按本章 3 节装好 tilelang，确认有 GPU。
2. 跑 `../01-first-kernel/vecadd_tilelang.py`，把它的 `get_kernel_source()` 输出存成文件。
3. 用 `diff` 或肉眼把它和本脚本打印的两段比：命中哪一种形态？有没有出现
   `if (... < N)`？为什么？
