# 第 09 章 代码

本章的代码分两类，目的不同：

1. **可以在 CPU 上跑的模型**（`01-`、`02-`、`03-`）。它们把本章的几条规则
   做成可复算的东西：原语到指令的映射表、`T.copy` 的越界谓词、`T.reduce_*` 的
   butterfly 归约。`01-` 还会顺手把正文里的源码引用逐条核对一遍。
2. **只做静态核对的 TileLang 骨架**（`04-`、`05-`）。它们需要真实加速器才能运行，
   本章只保证 `python3 -m py_compile` 通过、且每个 API 名都在锁定版本里核对过。

## 目录

```text
code/
├─ README.md                          本文件
├─ 01-primitive-ir-map/
│  ├─ README.md
│  └─ primitive_ir_map.py             纯标准库；打印映射表并核对源码引用行号
├─ 02-copy-predicate-model/
│  ├─ README.md
│  └─ copy_predicate_model.py         纯标准库；模拟 T.copy 的越界谓词与安全值
├─ 03-thread-reduce-model/
│  ├─ README.md
│  └─ thread_reduce_model.py          纯标准库；模拟 tl::AllReduce 的 butterfly
├─ 04-copy-gemm-sketches/
│  ├─ README.md
│  ├─ nv_gemm_copy_sketch.py          需要 NVIDIA GPU + TileLang
│  └─ gemm_primitive_variants.py      需要 NVIDIA GPU + TileLang
└─ 05-ascend-primitives/
   ├─ README.md
   ├─ ascend_dual_copy_simtvf_sketch.py  需要昇腾 NPU + CANN + TileLang
   └─ ascend_tile_library_sketch.py      需要昇腾 NPU + tilelang-ascend 适配器
```

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-primitive-ir-map/primitive_ir_map.py` | Python 3.9+；核对功能需要 `_vendor/` 存在 | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过，输出见该目录 README |
| `02-copy-predicate-model/copy_predicate_model.py` | Python 3.9+ | **CPU 逻辑验证**：实际运行过 |
| `03-thread-reduce-model/thread_reduce_model.py` | Python 3.9+ | **CPU 逻辑验证**：实际运行过 |
| `04-copy-gemm-sketches/nv_gemm_copy_sketch.py` | NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang v0.1.15 | **仅静态核对**：`python3 -m py_compile` 通过；API 在 `_vendor/` 中逐个核对；**未在硬件上运行** |
| `04-copy-gemm-sketches/gemm_primitive_variants.py` | 同上 | **仅静态核对**；**未在硬件上运行** |
| `05-ascend-primitives/ascend_dual_copy_simtvf_sketch.py` | 昇腾 NPU + CANN + torch_npu + TileLang v0.1.15 | **仅静态核对**；**未在硬件上运行** |
| `05-ascend-primitives/ascend_tile_library_sketch.py` | 昇腾 NPU + CANN + tilelang-ascend `v0.1.2.000` | **仅静态核对**；**未在硬件上运行** |

## 统一说明

* 写作环境见 [`../../../VERSIONS.md`](../../../VERSIONS.md)：macOS (Apple Silicon)、
  Python 3.11（miniconda），**没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此本章没有任何本机实测结果。所有需要在加速器上运行的示例，
  预期输出一律写成「正确性判据 + 性能口径」，而不是具体数字。
* 每个示例目录都独立可复制，不依赖其他章节的文件。
* `01-`/`02-`/`03-` 是**模型**，不是硬件测量；它们的价值在于把规则写成可复算的代码。
