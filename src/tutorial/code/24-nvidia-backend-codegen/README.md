# 第 24 章 代码

本章写作环境没有 NVIDIA GPU 可用，因此这里的示例分成两个层次。

* **可运行**（`01`–`03`）：把 NVIDIA 后端里**三条纯逻辑的判定规则**用标准库重写一遍，
  在没有 GPU 的机器上也能跑出确定结果，用来验证你对规则的理解。
* **仅静态核对**（`04`–`05`）：真实 TileLang kernel，需要 NVIDIA GPU + CUDA toolkit +
  PyTorch(CUDA)；本章只做了 API 与签名的逐个核对。

## 目录

```
code/
├─ README.md                    本文件
├─ 01-copy-inst-selection/
│  ├─ README.md
│  └─ copy_inst_selection.py    纯标准库；复现 SelectCopyInstForLowering 的判定顺序
├─ 02-gemm-inst-selection/
│  ├─ README.md
│  └─ gemm_inst_selection.py    纯标准库；复现 Gemm::SelectInst 的谓词链
├─ 03-tma-descriptor/
│  ├─ README.md
│  └─ tma_descriptor.py         纯标准库；复现 cuTensorMapEncodeTiled 的参数与约束
├─ 04-warp-specialize-gemm/
│  ├─ README.md
│  └─ ws_gemm.py                TileLang kernel；需要 NVIDIA Hopper GPU
└─ 05-tcgen05-gemm/
   ├─ README.md
   └─ tcgen05_gemm.py           TileLang kernel；需要 NVIDIA Blackwell GPU
```

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-copy-inst-selection/copy_inst_selection.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过 |
| `02-gemm-inst-selection/gemm_inst_selection.py` | 同上 | **CPU 逻辑验证**：实际运行过 |
| `03-tma-descriptor/tma_descriptor.py` | 同上 | **CPU 逻辑验证**：实际运行过 |
| `04-warp-specialize-gemm/ws_gemm.py` | NVIDIA Hopper (sm_90) + CUDA 12.x + PyTorch(CUDA) + TileLang v0.1.15 | **仅静态核对**：`python3 -m py_compile` 通过；API 逐个在 `_vendor/tilelang-v0.1.15/` 中核对过；**未在硬件上运行** |
| `05-tcgen05-gemm/tcgen05_gemm.py` | NVIDIA Blackwell (sm_100) + CUDA 12.8+ + PyTorch(CUDA) + TileLang v0.1.15 | **仅静态核对**：同上；**未在硬件上运行** |

## 关于 `01`–`03` 的说明

这三个脚本**不是** TileLang 编译器的一部分，也**不能**替代编译器。它们按
`_vendor/tilelang-v0.1.15` 里的判定逻辑重写为独立模型，用途是：让你反复试不同输入、
观察结果怎么变；让你在改读源码时手里有一个“我预期它会怎么选”的对照物。

它们不保证与编译器逐位一致（真实实现里的 `arith::Analyzer::CanProve` 在某些符号
表达式上比你手算更强或更弱）。每个脚本的 README 都写明了简化了哪些部分。

## 运行

```bash
python3 01-copy-inst-selection/copy_inst_selection.py
python3 02-gemm-inst-selection/gemm_inst_selection.py
python3 03-tma-descriptor/tma_descriptor.py

# 只做语法检查（不需要 GPU）
python3 -m py_compile 04-warp-specialize-gemm/ws_gemm.py 05-tcgen05-gemm/tcgen05_gemm.py
```

写作环境见 [../../VERSIONS.md](../../VERSIONS.md)：macOS (Apple Silicon)、Python 3.11，
**没有 NVIDIA GPU，也没有昇腾 NPU**。因此本章不存在任何来自本机实测的性能数字。
