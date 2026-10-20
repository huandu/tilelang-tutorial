# 第 14 章 代码

本章是算子实战章，代码分三类：

1. **纯标准库模型**（`01-gemm-ledger/`、`06-bench-methodology/`）——不需要 GPU /
   NPU，也不依赖 TileLang 与 PyTorch。它们在写作机上**实际运行过**，输出见各
   自的 README。
2. **TileLang kernel**（`02-tilelang-ladder/`、`03-gemm-variants/`、
   `04-low-precision/`、`05-ascend-gemm/`）——需要真实加速器才能编译运行。
   本章对这些示例只做到**仅静态核对**：用 `python3 -m py_compile` 过语法、
   逐个在 `_vendor/` 里比对 API 名与签名。**没有在硬件上运行过任何一条。**

## 目录

```
code/
├─ README.md                          本文件
├─ 01-gemm-ledger/                    纯标准库；GEMM 性能账本与 tile 占用估算
├─ 02-tilelang-ladder/                TileLang；五级阶梯中的 ①–⑤
├─ 03-gemm-variants/                  TileLang；split-K / stream-K / grouped
├─ 04-low-precision/                  TileLang；fp8 / int4 / block-scaled
├─ 05-ascend-gemm/                    昇腾；基础 GEMM 与 batched GEMM
└─ 06-bench-methodology/              纯标准库；预热/重复次数换算与对比模板
```

每个子目录都能独立复制出去运行，不依赖其他章节或其他示例的文件。

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-gemm-ledger/gemm_ledger.py` | Python 3.9+，仅标准库 | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过 |
| `02-tilelang-ladder/gemm_ladder.py` | NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang v0.1.15 | **仅静态核对**（`py_compile` + API 比对）；未运行 |
| `02-tilelang-ladder/gemm_intrinsic.py` | 同上 | **仅静态核对**；未运行 |
| `03-gemm-variants/gemm_variants.py` | NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang v0.1.15 | **仅静态核对**；未运行 |
| `04-low-precision/low_precision_gemm.py` | NVIDIA GPU（fp8 需要 sm89+，block-scaled 参考示例用 sm100） | **仅静态核对**；未运行 |
| `05-ascend-gemm/ascend_gemm.py` | 昇腾 NPU + CANN + torch_npu + TileLang（`_vendor/tilelang-ascend-v0.1.2.000`） | **仅静态核对**；未在 NPU 上运行 |
| `06-bench-methodology/bench_methodology.py` | Python 3.9+，仅标准库 | **CPU 逻辑验证**：实际运行过 |

## 统一说明

* 写作环境（见 `../../VERSIONS.md`）：macOS (Apple Silicon)、Python 3.11，
  **没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此任何需要加速器才能得到的数字（延迟、吞吐、算力利用率）在 `code/` 里
  都不存在；模型里出现的数字全是公式推导，性能对比的口径与来源见
  `../references.md`。
* 版本一律为 TileLang `v0.1.15` 与 TileLang-Ascend
  `v0.1.2.000-release`；本章不使用“最新版”这类表述。
