# 第 11 章 代码

本章的代码分两类，验证级别写得很清楚，请不要把两者混起来：

* **三个纯标准库的离线工具**（`01`–`03`）：不 import TileLang、不碰硬件，把
  “流水为什么值钱”“这份配置放得下吗”“这个搜索空间值不值得跑”变成可复算的算术。
  它们在本机（macOS Apple Silicon，Python 3.11）**实际运行过**。
* **两个 TileLang kernel**（`04`–`05`）：需要 GPU/NPU 才能编译运行。本机没有
  NVIDIA GPU，也没有昇腾 NPU，因此它们只做到 **`python3 -m py_compile` 通过 + API
  逐个在 `_vendor/tilelang-v0.1.15/` 里核对**，即“仅静态核对”。

五个示例相互独立，可以单独复制出去运行；都不依赖其他章节的文件。

## 目录

```
code/
├─ README.md                          本文件
├─ 01-pipeline-timeline/
│  ├─ README.md
│  └─ pipeline_timeline.py            纯标准库：copy/compute 重叠的离散事件模型
├─ 02-shared-memory-budget/
│  ├─ README.md
│  └─ smem_budget.py                  纯标准库：多缓冲的片上占用估算与可行性判断
├─ 03-autotune-config-space/
│  ├─ README.md
│  └─ config_space.py                 纯标准库：搜索空间的生成、剪枝与排序
├─ 04-tilelang-pipelined-gemm/
│  ├─ README.md
│  └─ pipelined_gemm.py               TileLang：T.Pipelined 的三种写法（自动/手动/不流水）
└─ 05-tilelang-autotune-gemm/
   ├─ README.md
   └─ autotune_gemm.py                TileLang：@tilelang.autotune 与 AutoTuner.from_kernel
```

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-pipeline-timeline/pipeline_timeline.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行，输出见该目录 README |
| `02-shared-memory-budget/smem_budget.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：同上 |
| `03-autotune-config-space/config_space.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：同上 |
| `04-tilelang-pipelined-gemm/pipelined_gemm.py` | NVIDIA GPU + CUDA toolkit + PyTorch(CUDA) + TileLang v0.1.15 | **仅静态核对**：`python3 -m py_compile` 通过；API 名与签名逐个在 `_vendor/tilelang-v0.1.15/` 中核对；**未在硬件上运行** |
| `05-tilelang-autotune-gemm/autotune_gemm.py` | 同上（+ 可选 `z3-solver` 仅昇腾侧需要） | **仅静态核对**：同上；**未在硬件上运行** |

## 统一说明

* 本章写作环境（见 `../../VERSIONS.md`）：macOS (Apple Silicon)、Python 3.11（miniconda），
  **没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此：任何需要加速器才能得到的结果（实测延迟、吞吐、算力利用率）在本章中都不存在。
  `01`–`03` 打印出来的所有数字都是**模型输出**，不是测量值；`04`–`05` 的
  `code/README` 只给出“在有卡机器上怎么验证”，不给出性能承诺。
* 版本一律为 TileLang `v0.1.15`；本书不使用“最新版”这类表述。
