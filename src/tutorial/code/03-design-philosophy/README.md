# 第 03 章 代码

本章讲的是设计哲学，所以代码也存在两个使命：**把抽象主张变成可以复算的东西**，
以及**把「可以混用」写成看得见的代码**。三个示例互相独立，可以单独复制出去运行。

## 目录

```
code/
├─ README.md                        本文件
├─ 01-layer-inventory/
│  ├─ README.md
│  └─ layer_inventory.py            纯标准库；可在任意机器离线运行
├─ 02-one-kernel-two-levels/
│  ├─ README.md
│  └─ one_kernel_two_levels.py      TileLang kernel；需要 NVIDIA GPU
└─ 03-schedule-as-annotation/
   ├─ README.md
   └─ schedule_as_annotation.py     四个变体；需要 NVIDIA GPU
```

## 验证级别总表

| 示例 | 它证明什么 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- | --- |
| `01-layer-inventory/layer_inventory.py` | 三层接口 + CUDA 方言扩展的 47 个 API 名字，逐个在 v0.1.15 的语言层里存在；官方文档里「beginner 层未完全实现」的说明仍在 | Python 3.9+（只用标准库） | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过，退出码 0，输出见该目录 README |
| `02-one-kernel-two-levels/one_kernel_two_levels.py` | 同一个 `T.Kernel` 体里 developer 层与 expert 层共存（composable 的最小实证） | NVIDIA GPU（sm70+）+ CUDA toolkit + torch(CUDA) + TileLang v0.1.15 | **仅静态核对**：`py_compile` 通过；10 个 API 逐个在 `_vendor/` 里核对；**未在硬件上运行** |
| `03-schedule-as-annotation/schedule_as_annotation.py` | 同一段数据流，四个变体各只改一个调度维度（thread binding / layout / pipeline） | 同上 | **仅静态核对**：`py_compile` 通过；API 逐个核对；**未运行、未做任何性能比较** |

## 统一说明

* 写作环境（见 `../../VERSIONS.md`）：macOS (Apple Silicon)、Python 3.11（miniconda），
  **没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此本章不存在任何性能数字：凡是需要加速器才能得到的结果（延迟、吞吐、实测利用率）
  本章一律不写。示例 3 故意不比较四种调度的快慢——那是读者在自己机器上做的事。
* 需要 GPU 的两个示例都提供 `--static-only` 路径：在没有加速器的机器上打印 API 核对清单
  后正常退出（退出码 0）；直接运行则给出缺少依赖的提示并以退出码 2 或 3 结束。
* 版本一律为 TileLang `v0.1.15`（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）。
  本章所有结论都锚定在这个 tag 上，不引用随上游漂移的未锁定版本。

## 建议的阅读顺序

1. 先跑 `01-layer-inventory`，拿到本章那张 API 表的机器输出；
2. 再读 `02-one-kernel-two-levels`，对照正文 3.3 节的两个真实例子
   （`examples/dequantize_gemm/...` 与 `examples/gemm/example_gemm_intrinsics.py`）；
3. 最后读 `03-schedule-as-annotation`，对照正文 2.3 节的四个调度维度。
