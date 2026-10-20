# 第 39 章 代码

本章是**收口章**，不引入任何新的 kernel。`code/` 下的三个示例都是**纯 Python、
只用标准库**的账本程序：它们把正文里的公式复算一遍，并把每个数字的出处打印在
旁边。三个脚本都不需要 NPU / GPU，也不需要 TileLang 或 PyTorch。

## 目录

```text
code/
├─ README.md                          本文件
├─ 01-train-step-budget/              训练一步的通信 / 算力账本（§2.2、§2.8、§4）
│  ├─ README.md
│  └─ train_step_budget.py
├─ 02-inference-latency-model/       推理侧的判类账本（§3.4、§3.7）
│  ├─ README.md
│  └─ inference_latency_model.py
└─ 03-bottleneck-map/                「教程章节 → 组件 → 典型瓶颈」映射（§5）
   ├─ README.md
   └─ bottleneck_map.py
```

每个子目录都能独立复制出去运行，不依赖其他章节或其他示例的文件。

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-train-step-budget/train_step_budget.py` | Python 3.9+，仅标准库 | **CPU 逻辑验证**：在写作机上实际运行过，输出见该目录 README |
| `02-inference-latency-model/inference_latency_model.py` | 同上 | **CPU 逻辑验证**：实际运行过 |
| `03-bottleneck-map/bottleneck_map.py` | 同上（`--verify` 需要本书的章节目录） | **CPU 逻辑验证**：实际运行过 |

## 统一说明

* 写作环境（见 `../../VERSIONS.md`）：macOS (Apple Silicon)、Python 3.11，
  **没有 NVIDIA GPU，也没有昇腾 NPU，也没有 CANN**。
* 这三个脚本**复算的是公式，不是硬件**。它们打印出来的每一个数字都必须与正文里
  标注的来源对得上；脚本本身不产生任何实测性能数字。
* 脚本里用到的两个硬件模型常量（HBM 3500 GB/s、fp8 峰值 865 TFLOPS）来自
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/csrc/jit_kernels/heuristics.hpp:46-71`，
  是**源码里的模型常量**，不是实测值。
* 脚本里用到的通信口径（FP8 dispatch、BF16 combine、`expert_alignment=128` 等）
  来自 `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:9-21` 的测量条件。
* DeepSeek 仓库都是 corpus 里的 2026-10-01 快照，commit 见
  `../../../../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md`。

## 这三个脚本「不能」做什么

* 不能预测任何真实运行的端到端时间。它们给的是**下界与判类**，而且是
  在「假设实现已经接近上限」的前提下。
* 不能替代 profiler。第 12 章、第 19 章 §8 讲的 `msprof` 判读才是真机上的依据。
* 不能跨机器套用。换硬件、换 CANN 版本，两个常量都要重新标定
  （第 28 章 §2.2 的第三种误读）。
