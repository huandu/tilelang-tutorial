# 02 · 多缓冲流水时间线模型

## 目的

把本章 1.4 的直觉做成可复算的时间线：**为什么把缓冲区开成两份，能让搬运和计算重叠，
从而提高计算单元的利用率？**

脚本用一个离散事件模型模拟一个核上循环的三段流水：

```text
MTE2（搬入）  →  CUBE/VEC（计算）  →  MTE3（搬出）
```

对同一个负载分别跑 `versions=1`（串行基线）、`versions=2`、`versions=3`，画出每一段
的占用时间线，并报告 makespan 与计算单元利用率。

## 运行前提

Python 3.9+。只用标准库（`argparse`、`dataclasses`）。**不需要 GPU / NPU / TileLang。**

## 运行命令

```bash
cd 02-pipeline-timeline
python3 pipeline_timeline.py

# 也可以换参数
python3 pipeline_timeline.py --iters 8 --copy-in 10 --compute 4 --copy-out 10 --versions 2
```

## 期望输出（摘要）

在 macOS（Apple Silicon）、Python 3.11 上实际运行得到的摘要。

**访存受限（copy=10+10，compute=4）：**

```text
versions=1: makespan=144 cycles, compute busy=24, compute utilization= 16.7%
versions=2: makespan=82 cycles,  compute busy=24, compute utilization= 29.3%
versions=3: makespan=74 cycles,  compute busy=24, compute utilization= 32.4%
```

**计算受限（copy=4+4，compute=20）：**

```text
versions=1: makespan=168 cycles, compute busy=120, compute utilization= 71.4%
versions=2: makespan=128 cycles, compute busy=120, compute utilization= 93.8%
```

两个对照想说明的事：

1. 多缓冲把 makespan 显著压下来，计算单元的“忙碌占比”上升；
2. 但**利用率的上限由负载本身决定**。当搬运时间远大于计算时间时，即便完美重叠，
   利用率也只能接近 `compute / (copy_in + copy_out)`——这就是“memory-bound”；
   反过来，计算受限时，多缓冲能把利用率从约 71% 推到约 94%。

## 模型的假设与边界

脚本开头列出了全部假设，这里重复关键的几条——**它们都是简化，不是硬件事实**：

* 三段各自运行在独立单元上，可以真正并行；
* 每段的耗时是常数，不随 tile 变化；
* 忽略 Set/Wait 本身的开销、队列深度、DMA 带宽上限、流水填充/排空（tail）效应；
* “buffer 归还”被简化为“上一轮 copy-out 结束”。

因此脚本**不能**用来预测真实 kernel 的耗时或吞吐。它的用途是把“多缓冲到底买到了
什么”写成可被别人质疑和修改的公式。

## 作者验证级别

**CPU 逻辑验证**：在 macOS（Apple Silicon）、Python 3.11 上运行过，上面的数字与时间是
脚本实际输出。**未在任何加速器上运行，也未与 Nsight / msProf 的真实时间线对照。**
