# 第 01 章 代码

本章是全书开篇，读者此时还没有任何 kernel 经验，因此这里的代码有两个使命：

1. **`01-op-cost-model/`** —— 一台**不需要 GPU**、纯标准库就能跑的估算器。它把
   “算术强度 / 机器平衡点”这两个本章的核心概念算成一张表。这是唯一在写作机上真正
   运行过的示例。
2. **`02-first-tile-kernel/`** —— 一个 **tile 视角的 GEMM + ReLU** 示例，让读者在第 1 章
   就看到“tile 程序长什么样”。它需要 NVIDIA GPU 才能编译运行，本章只做了静态核对。

两个示例相互独立，可以单独复制出去运行；都不依赖其他章节的文件。

## 目录

```
code/
├─ README.md                     本文件
├─ 01-op-cost-model/
│  ├─ README.md
│  └─ op_cost_model.py           纯标准库；可在任意机器离线运行
└─ 02-first-tile-kernel/
   ├─ README.md
   └─ tile_gemm_sketch.py        TileLang kernel；需要 NVIDIA GPU 才能运行
```

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-op-cost-model/op_cost_model.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：在 macOS（Apple Silicon）、Python 3.11 上实际运行过，输出见该目录 README |
| `02-first-tile-kernel/tile_gemm_sketch.py` | NVIDIA GPU + CUDA toolkit + PyTorch(CUDA) + TileLang v0.1.15 | **仅静态核对**：`python3 -m py_compile` 通过；每个 API 逐个在 `_vendor/tilelang-v0.1.15/` 中核对过存在性与签名；**未在硬件上运行** |

## 统一说明

* 本章写作环境（见 `../../VERSIONS.md`）：macOS (Apple Silicon)、Python 3.11（miniconda），
  **没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此：任何需要加速器才能得到的结果（实测算力利用率、延迟、吞吐）在本章中都不存在。
  凡涉及数字的地方，都给出 corpus 路径或上游链接与测量条件（见 `../references.md`）。
* 版本一律为 TileLang `v0.1.15`；本书不使用“最新版”这类表述。
