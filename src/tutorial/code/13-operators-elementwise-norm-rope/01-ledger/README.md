# 01 · 访存账本（可复算）

## 目的

把第 02 章 §2.1 的 roofline 记账口径写成一段可执行代码，用来回答：

1. 一个逐元素算子在「一个 op 一个 kernel」的写法下账本上要搬多少字节？
2. 融合成一个 kernel 之后最少要搬多少字节？比值是多少？
3. 这几类算子的算术强度落在什么量级，离机器平衡点有多远？

脚本不 import TileLang、不编译任何 kernel、不碰 GPU/NPU。

## 运行前提

Python 3.9+，只用标准库（`argparse`、`json`、`dataclasses`）。
**不需要 GPU / NPU / PyTorch / TileLang。**

## 运行命令

```bash
cd 01-ledger
python3 ledger.py            # 打印全部账本
python3 ledger.py --json     # 机器可读输出，便于回归比较
```

## 预期输出（实测摘要）

在 macOS（Apple Silicon）、Python 3.11 上实际运行：

```text
scale+add（2 个 kernel）        kernels=2  moved=    20.97 MB  flops=    2.10 MFLOP  AI= 0.100 FLOP/Byte
scale+add（融合，1 个 kernel）   kernels=1  moved=    12.58 MB  flops=    2.10 MFLOP  AI= 0.167 FLOP/Byte
  → 访存量比值 1.67×

RMSNorm（6 步分解）              kernels=6  moved=   234.99 MB  flops=   67.12 MFLOP  AI= 0.286 FLOP/Byte
RMSNorm（融合，1 个 kernel）     kernels=1  moved=    67.12 MB  flops=   67.12 MFLOP  AI= 1.000 FLOP/Byte
  → 访存量比值 3.50×

LayerNorm（多趟分解）            kernels=6  moved=   335.59 MB  flops=  117.44 MFLOP  AI= 0.350 FLOP/Byte
LayerNorm（融合，1 个 kernel）   kernels=1  moved=    67.13 MB  flops=  100.67 MFLOP  AI= 1.500 FLOP/Byte
  → 访存量比值 5.00×

RoPE（4 步分解，中间量落显存）      kernels=4  moved=   369.10 MB  flops=   50.33 MFLOP  AI= 0.136 FLOP/Byte
RoPE（融合，1 个 kernel）         kernels=1  moved=   100.66 MB  flops=   50.33 MFLOP  AI= 0.500 FLOP/Byte
  → 访存量比值 3.67×

RMSNorm+RoPE+量化（3+ 个 kernel） kernels=3  moved=   185.62 MB  flops=  119.03 MFLOP  AI= 0.641 FLOP/Byte
RMSNorm+RoPE+量化（融合）          kernels=1  moved=    51.40 MB  flops=  119.03 MFLOP  AI= 2.315 FLOP/Byte
  → 访存量比值 3.61×

全部 10 个方案的算术强度落在 0.100 – 2.315 FLOP/Byte。
```

形状：逐元素 N = 2^20、fp32；归一化 4096×4096、fp16/bf16；RoPE 4096 token ×
32 head × rope_dim 128、fp16；融合那一组 4096×4096、rope_dim 128、bf16。

## 重要边界

* 这是**下界模型**，不是测量。它不含 cache 命中、kernel 启动、尾部效应与同步开销。
* 「6 步分解」「4 步分解」是**本脚本假设的一种拆法**，不是对 PyTorch 实际产生的
  kernel 数的追踪。真实 kernel 数取决于 PyTorch 版本的融合与 dispatch。
* LayerNorm 的两组数字 FLOP 略有差异（117.44 对 100.67 MFLOP），因为多趟分解每趟
  单独读一遍数据、算法与融合版不完全相同；本脚本报告的是各自的账本，不做归一。
* 脚本不写任何具体机器的平衡点数字——那取决于精度与峰值，要按第 02 章 §2.2 自己算。

## 作者验证级别

**CPU 逻辑验证**：在 macOS（Apple Silicon）、Python 3.11 上运行过，上面的输出是本机
实际输出。**未在任何加速器上运行。**
