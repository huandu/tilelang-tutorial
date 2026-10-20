# 第 13 章 代码

本章的代码分两类，目的不同：

1. **可以在 CPU 上跑的模型与检查**（`01-`、`07-`）。`01-` 把第 02 章的 roofline
   记账口径做成可复算的账本；`07-` 把「示例里用到的每个 API 名在锁定版本里到底
   有没有」做成可跑的静态检查。
2. **只做静态核对的 TileLang 骨架**（`02-`、`03-`、`04-`、`05-`、`06-`）。
   它们需要 NVIDIA GPU 或昇腾 NPU 才能运行；本章只保证 `python3 -m py_compile`
   通过、每个 API 名都在锁定版本里核对过、且代码里给出的 PyTorch 参考实现是自洽的。

## 目录

```text
code/
├─ README.md                       本文件
├─ 01-ledger/
│  ├─ README.md
│  └─ ledger.py                    纯标准库；逐元素/归一化/RoPE 的访存账本
├─ 02-elementwise/
│  ├─ README.md
│  └─ elementwise_variants.py      需要 NVIDIA GPU + TileLang v0.1.15
├─ 03-norm/
│  ├─ README.md
│  └─ norm_kernels.py              需要 NVIDIA GPU + TileLang v0.1.15
├─ 04-rope/
│  ├─ README.md
│  └─ rope_kernels.py              需要 NVIDIA GPU + TileLang v0.1.15
├─ 05-fusion/
│  ├─ README.md
│  └─ rmsnorm_rope_quant.py        需要 NVIDIA GPU + TileLang v0.1.15
├─ 06-ascend/
│  ├─ README.md
│  ├─ ascend_mainline_kernels.py   需要昇腾 950 NPU + CANN + TileLang v0.1.15
│  └─ ascend_adapter_sketch.py     需要昇腾 NPU + CANN + tilelang-ascend v0.1.2.000
└─ 07-api-audit/
   ├─ README.md
   └─ api_audit.py                 纯标准库；核对示例用到的 API 名
```

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-ledger/ledger.py` | Python 3.9+ | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过，输出见该目录 README |
| `02-elementwise/elementwise_variants.py` | NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang v0.1.15 | **仅静态核对**：`py_compile` 通过；API 名经 `07-api-audit` 核对；**未在硬件上运行** |
| `03-norm/norm_kernels.py` | 同上 | **仅静态核对**；**未在硬件上运行** |
| `04-rope/rope_kernels.py` | 同上 | **仅静态核对**；**未在硬件上运行** |
| `05-fusion/rmsnorm_rope_quant.py` | 同上 | **仅静态核对**；**未在硬件上运行**（layout 推导是否通过必须在真机上确认） |
| `06-ascend/ascend_mainline_kernels.py` | 昇腾 950 + CANN + torch_npu + `USE_ASCEND=ON` 构建的 TileLang v0.1.15 | **仅静态核对**；**未在硬件上运行** |
| `06-ascend/ascend_adapter_sketch.py` | 昇腾 NPU + CANN + torch_npu + tilelang-ascend v0.1.2.000-release | **仅静态核对**；**未在硬件上运行** |
| `07-api-audit/api_audit.py` | Python 3.9+；核对功能需要 `_vendor/` 存在 | **CPU 逻辑验证**：实际运行过，输出见该目录 README |

## 统一说明

* 写作环境见 [`../../../VERSIONS.md`](../../../VERSIONS.md)：macOS (Apple Silicon)、
  Python 3.11（miniconda），**没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此本章没有任何本机实测的性能或正确性结果。所有需要在加速器上运行的示例，
  预期输出一律写成「正确性判据 + 性能口径」，而不是具体数字。
* 每个示例目录独立可复制，不依赖其他示例目录或章节的文件。
* `01-` 与 `07-` 是**模型与检查**，不是硬件测量。
