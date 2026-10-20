# 第 33 章 代码

本章是“一份算法、两个后端”的结构分析章，代码分三类：

1. **纯标准库脚本**（`03-map-and-diff/`）——不需要 GPU/NPU，也不需要装 TileLang。
   作者在写作机上**实际运行过**，输出见该目录的 README。
2. **两个后端的精简 kernel**（`01-topk-gate-minimal/`）——需要真实硬件 +
   对应版本的 TileLang。本章对它们只做到**仅静态核对**：
   `python3 -m py_compile` 过语法，API 名与用法逐个在
   `_vendor/tilelang-v0.1.15/` 里比对过。**没有在硬件上运行过任何一条。**
3. **调用 TileKernels 的端到端样例**（`02-call-the-library/`）——需要真实硬件 +
   安装好的 TileKernels 包。同样**仅静态核对**。

## 目录

```
code/
├─ README.md                     本文件
├─ 01-topk-gate-minimal/         两个后端的精简 top-k gate（纯静态阅读）
│  ├─ topk_gate_cuda_min.py      CUDA 版：fragment + reducer + warp 内归约
│  ├─ topk_gate_asc_min.py       Ascend 版：UB + SIMD + 显式缓冲版本
│  └─ README.md
├─ 02-call-the-library/          用 TileKernels 的公开 API 跑一次（需要硬件）
│  ├─ call_topk_gate.py
│  ├─ call_quant_and_rope.py
│  └─ README.md
└─ 03-map-and-diff/              纯标准库：扫描三层结构 + 产出结构对照表
   ├─ map_and_diff.py
   └─ README.md
```

每个子目录都能独立复制出去运行，不依赖其他章节或其他示例的文件。

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-topk-gate-minimal/topk_gate_cuda_min.py` | NVIDIA SM90/SM100 + CUDA ≥ 13.1 + TileLang `v0.1.15` | **仅静态核对**；未运行 |
| `01-topk-gate-minimal/topk_gate_asc_min.py` | Ascend 950 + CANN ≥ 9.2.0 + TileLang `v0.1.15`（`tilelang.ascend.language`） | **仅静态核对**；未运行 |
| `02-call-the-library/call_topk_gate.py` | 同上任一 + 已安装 `tile-kernels` 包 | **仅静态核对**；未运行 |
| `02-call-the-library/call_quant_and_rope.py` | 同上 | **仅静态核对**；未运行 |
| `03-map-and-diff/map_and_diff.py` | Python 3.9+，仅标准库（读一个 TileKernels 源码树） | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过 |

## 统一说明

* 写作环境（见 `../../VERSIONS.md`）：macOS (Apple Silicon)、Python 3.11，
  **没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此任何需要加速器才能得到的数字（延迟、吞吐、利用率）在 `code/` 里都不存在；
  性能口径与来源见 `../references.md`。
* 版本一律为 TileLang `v0.1.15`（`_vendor/tilelang-v0.1.15/`）。
  本章的 Ascend 代码用的是**主干 M 树的昇腾方言**（`tilelang.ascend.language`），
  不是第 19 章的适配器 A 树（`tilelang.language` + `target='ascendc'`）。
* `01-*` 里的 `import` 按各自后端写死。**不要**把两个文件在同一个进程里跑。
