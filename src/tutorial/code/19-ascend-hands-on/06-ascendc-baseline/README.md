# 示例 06：Ascend C 基线（对照用）

## 目的

本章 §6.1。把同一个逐元素 add 的 Ascend C 写法摆出来，用于回答“TileLang 省掉了什么”。
它不是可独立编译的工程，只是一个**结构对照**。

## 文件与来源

`add_ascendc.cpp` 改写自昇腾官方文档
`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_00020_html.md`
（“如何使用 Tensor 原地操作提升算子性能”）里的示例代码，补上中文注释标出行号。

> 该文档是**商用版 CANN** 的页面，与 community 版可能有版本差异；引用时请注明出处。

## 运行前提

无（不需要运行）。真正的 Ascend C 算子需要：CANN 工具包 + Ascend C 算子工程模板
（算子原型定义、tiling、`<<<>>>` 直调或自定义算子包）+ 昇腾 NPU。

## “运行”命令

不适用。这个文件**不能**用 `g++` 直接编译（需要 Ascend C 的头文件、
`__aicore__` 等语言扩展、以及 bisheng 编译器）。

## 期望输出

不适用。

## 作者实际做到的验证级别

**仅静态核对**：只做了“把官方文档里的示例抄下来 + 加注释 + 标出对照关系”，
**未编译、未运行**。

## 与 TileLang 的逐项对照

见本章正文 §6.1 的表。核心结论：

| Ascend C 要做的事 | TileLang 里对应 |
| --- | --- |
| `TPipe` / `TQue` / `InitBuffer` | 不需要（`T.alloc_ub` + 编译器内存规划） |
| `AllocTensor` / `FreeTensor`（缓冲生命周期） | 不需要 |
| `DataCopy`（GM↔UB） | `T.copy(...)` |
| `EnQue` / `DeQue`（隐含同步） | `T.barrier_all()`（A 树）或自动同步（M 树） |
| `Add(...)` | `T.tile.add(...)` |
