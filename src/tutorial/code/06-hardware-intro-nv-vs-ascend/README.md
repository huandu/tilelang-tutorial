# 第 06 章 代码

本章是硬件导论，代码的使命不是「跑出一个算子」，而是把本章的两个核心直觉变成
**可以在 CPU 上复算的东西**：

1. **片上容量是要算的账**——`01-onchip-footprint/` 把 tile 形状 × dtype × 多缓冲
   版本数换算成字节，并和厂商公开的容量对上。
2. **多缓冲为什么能把计算填满**——`02-pipeline-timeline/` 用一个离散事件模型画出
   搬运/计算/回写的时间线，对比 1 份 / 2 份 / 3 份缓冲。
3. **同一件事在两台上怎么表达**——`03-target-sketch/` 给出一段 NV 侧和一段昇腾侧的
   tile 程序骨架，逐行标注它对应本章哪条硬件路径。这两份**只做静态核对**。

三个示例互相独立，可以单独复制出去运行，都不依赖其他章节的文件。

## 目录

```text
code/
├─ README.md                       本文件
├─ 01-onchip-footprint/
│  ├─ README.md
│  └─ onchip_footprint.py          纯标准库；可在任意机器离线运行
├─ 02-pipeline-timeline/
│  ├─ README.md
│  └─ pipeline_timeline.py         纯标准库；可在任意机器离线运行
└─ 03-target-sketch/
   ├─ README.md
   ├─ nv_hopper_sketch.py          需要 NVIDIA GPU + TileLang 才能运行
   └─ ascend_950_sketch.py         需要昇腾 NPU + CANN + TileLang 才能运行
```

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-onchip-footprint/onchip_footprint.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：在 macOS（Apple Silicon）、Python 3.11 上实际运行过，输出见该目录 README |
| `02-pipeline-timeline/pipeline_timeline.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：实际运行过，输出见该目录 README |
| `03-target-sketch/nv_hopper_sketch.py` | NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang v0.1.15 | **仅静态核对**：`python3 -m py_compile` 通过，API 逐个在 `_vendor/tilelang-v0.1.15/` 中核对；**未在硬件上运行** |
| `03-target-sketch/ascend_950_sketch.py` | 昇腾 NPU + CANN + torch_npu + TileLang v0.1.15 | **仅静态核对**：`python3 -m py_compile` 通过，API 逐个在 `_vendor/tilelang-v0.1.15/` 中核对；**未在硬件上运行** |

## 统一说明

* 本章写作环境（见 `../../VERSIONS.md`）：macOS (Apple Silicon)、Python 3.11（miniconda），
  **没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此本章没有任何本机实测的性能数字。所有容量、带宽、延迟数字都来自
  `../references.md` 里的厂商文档或社区一手文章，并标注了条件。
* `01-` 和 `02-` 两个示例是**模型**，不是硬件测量；它们的价值在于「把公式和假设写清楚，
  可以被别人复算和质疑」。请把它们当草稿纸，不要当基线。
* 版本一律为 TileLang `v0.1.15`；本书不使用「最新版」这类表述。
