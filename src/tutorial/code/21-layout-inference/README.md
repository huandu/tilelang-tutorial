# 第 21 章 代码

五个示例，按「能不能在本机跑」排序。本章写作环境（见 `../../VERSIONS.md`）是
macOS (Apple Silicon) + Python 3.11，**没有 NVIDIA GPU，也没有昇腾 NPU**。

```
code/
├─ README.md                        本文件
├─ 01-layout-algebra/
│  ├─ README.md
│  └─ layout_algebra.py             纯标准库，任意机器可跑（本机实跑过）
├─ 02-plot-layout/
│  ├─ README.md
│  └─ plot_layout_demo.py           需要 TileLang + Matplotlib（不需要 GPU）
├─ 03-infer-visualization/
│  ├─ README.md
│  └─ gemm_layout_visualization.py  需要 TileLang；打印/导出布局需要 GPU 才能编译
├─ 04-layout-conflict/
│  ├─ README.md
│  └─ annotate_layout_conflict.py   需要 NVIDIA GPU + TileLang
└─ 05-ascend-nz-layout/
   ├─ README.md
   └─ ascend_nz_layout.py           需要昇腾 NPU + TileLang
```

五个示例相互独立，可单独复制出去运行；都不依赖其他章节的文件。

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-layout-algebra/layout_algebra.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：在 macOS / Apple Silicon、Python 3.11 上实跑，输出见该目录 README |
| `02-plot-layout/plot_layout_demo.py` | TileLang v0.1.15 + Matplotlib（不需要 GPU） | **仅静态核对**：`python3 -m py_compile` 通过；API 逐个在 `_vendor/` 核对；**未运行**（本机未装 TileLang） |
| `03-infer-visualization/gemm_layout_visualization.py` | TileLang v0.1.15；编译到 CUDA 需要驱动/工具链 | **仅静态核对**：同上；**未运行** |
| `04-layout-conflict/annotate_layout_conflict.py` | NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang v0.1.15 | **仅静态核对**：同上；**未在硬件上运行** |
| `05-ascend-nz-layout/ascend_nz_layout.py` | 昇腾 NPU + CANN + TileLang v0.1.15 | **仅静态核对**：同上；**未在硬件上运行** |

## 统一说明

* 版本一律为 TileLang `v0.1.15`（commit 见 `../../VERSIONS.md`），不出现「最新版」
  表述。
* 任何需要真实加速器才能得到的结果（推断出的布局表达式、实测耗时、bank conflict
  次数）在本章中都不存在。凡涉及数字，都给出 `_vendor/` 路径或 corpus 路径。
* `01-layout-algebra` 输出的是**排列模型**的结果（哪些点映射到哪个线程/槽位），
  不是任何机器上的真实耗时；请勿把它读成性能数据。
* 本目录所有 `.py` 都通过 `python3 -m py_compile`。
