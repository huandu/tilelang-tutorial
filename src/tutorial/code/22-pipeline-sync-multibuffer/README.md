# 第 22 章 代码

四个示例分两类，验证级别差别很大，请不要混起来看：

* `01` 需要 TileLang（且需要 wheel 里带 CUDA codegen FFI），但**不需要 GPU**：
  用 `tilelang.tools.compile_only.compile_kernel_source` 做
  `enable_device_compile=False` 的 lowering，直接把多缓冲与同步的决策摊在 IR 与
  生成的 CUDA 源里。作者本机没有装 TileLang，因此它只做到
  「`python3 -m py_compile` 通过 + API 名逐个在 `_vendor/tilelang-v0.1.15/` 里核对」
  = **仅静态核对**，没有任何输出来自本机运行。
* `02`–`04` 是纯标准库的模型，不 import TileLang、不碰硬件，在 macOS (Apple
  Silicon) + Python 3.11 上**实际运行过** = **CPU 逻辑验证**。它们把
  「版本数怎么算」「同步多一个少一个差多少」「昇腾的 flag 槽位够不够」变成可复算的
  算术，用来解释正文里的结论，而不是预测性能。

四个示例相互独立，可以单独复制出去运行；都不依赖其他章节的文件。

## 目录

```
code/
├─ README.md                                   本文件
├─ 01-num-stages-ir-diff/
│  ├─ README.md
│  └─ lower_num_stages.py                      TileLang：同一 GEMM 换 num_stages，
│                                              打印 stage/order、版本维、同步点计数
├─ 02-buffer-version-model/
│  ├─ README.md
│  └─ buffer_versions.py                       纯标准库：use-def+1 版本数与 arena 打包
├─ 03-sync-count-vs-perf/
│  ├─ README.md
│  └─ sync_count_model.py                      纯标准库：同步点数量 vs 稳态步长
└─ 04-ascend-flag-budget/
   ├─ README.md
   └─ flag_budget.py                           纯标准库：8 槽 flag 命名空间与溢出
```

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-num-stages-ir-diff/lower_num_stages.py` | TileLang v0.1.15（wheel 需带 CUDA codegen FFI）；**不需要 GPU、不需要 nvcc** | **仅静态核对**：`python3 -m py_compile` 通过；`compile_kernel_source` / `PipelinePlanning` / `compile_only` 等 API 逐个在 `_vendor/tilelang-v0.1.15/` 中核对；**未运行** |
| `02-buffer-version-model/buffer_versions.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行，输出见该目录 README |
| `03-sync-count-vs-perf/sync_count_model.py` | 同上 | **CPU 逻辑验证**：同上 |
| `04-ascend-flag-budget/flag_budget.py` | 同上 | **CPU 逻辑验证**：同上 |

## 统一说明

* 本章写作环境（见 `../../VERSIONS.md`）：macOS (Apple Silicon)、Python 3.11，
  **没有 NVIDIA GPU，也没有昇腾 NPU**。
* `02`–`04` 打印的所有数字都是**模型输出**，不是任何硬件上的测量值；`01` 在有卡机器上
  跑出来的计数也只是「这一份 lowering 结果里出现了几次这个 token」，不是性能数字。
* 版本一律为 TileLang `v0.1.15`；任何泛指「更新了一版」的说法都不在本书出现，
  版本结论一律以锁定 tag 为准。
