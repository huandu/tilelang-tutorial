# 第 37 章 代码

本章只需要一台能跑 Python 3.9+ 的机器：三个示例都是**纯 CPU**的算法与复杂度模型，
不需要 GPU、不需要 NPU、不需要 TileLang。它们验证的是**算法与账本**，不是 kernel
性能——本机也没有任何加速器（见 [`../../VERSIONS.md`](../../VERSIONS.md)）。

```
code/
├─ README.md                          本文件
├─ 01-deepselect-python/              DeepSelect 的纯 Python 复刻（随机块序 + 阈值筛选 + compact）
├─ 02-block-order-experiment/         随机块序 vs 顺序扫描：候选量差异
└─ 03-complexity-tables/              把复杂度公式与 torch.topk 代价模型画成小表
```

每个子目录都能独立复制出去运行，互不依赖，也不依赖其它章节的文件。三个脚本都只用
标准库；`import torch` 是可选的**加速/对照**路径（有就用 `torch.topk` 当 compact 步骤
与真值来源，没有就回退到 `heapq`）。

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-deepselect-python/deepselect_python.py` | Python 3.9+；PyTorch 可选 | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11.7、torch 2.1.1（CPU）上实际运行过（含 `--no-torch` 回退路径） |
| `02-block-order-experiment/block_order_experiment.py` | Python 3.9+；PyTorch 可选 | **CPU 逻辑验证**：实际运行过 |
| `03-complexity-tables/complexity_tables.py` | Python 3.9+，仅标准库 | **CPU 逻辑验证**：实际运行过 |

**没有任何一个示例在 GPU / NPU 上运行过**，也没有任何一条来自本机的性能数字。
性能口径与来源见 [`../references.md`](../references.md)。

## 这些脚本在证明什么

| 脚本 | 回答的问题 | 结论 |
| --- | --- | --- |
| `01` | 报告里的伪代码跑起来到底对不对、候选量到底有多大？ | 与 `torch.topk` 逐值对拍一致；`W` 远小于 `N` |
| `02` | 为什么块顺序必须是随机的？ | 顺序扫描在「大值靠后」的输入上退化成 `A = N`、`W ≈ 1.5N`；随机块序在任何输入上都稳定 |
| `03` | `(1+k/B2)·L·H_m`、`O(k log(N/k))`、`torch.topk` 的代价各是什么形状？ | 额外工作量相对 `N` 越来越小；`B2 = k` 附近是甜点 |

三个脚本的输出都随附在各自 README 的「期望输出」一节，可逐行复算。
