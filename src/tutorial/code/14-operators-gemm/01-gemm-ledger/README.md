# 示例 01：GEMM 的性能账本与 tile 占用估算

## 目的

把第 14 章 1.1 与 1.3 节的公式写成可运行的脚本：

```
计算量        = 2·M·N·K                         （FLOP）
最小访存      = (M·K + K·N + M·N) · dtype_bytes （Byte）
算术强度 AI   = 计算量 / 最小访存               （FLOP/Byte）
机器平衡点    = 峰值算力 / 峰值带宽             （FLOP/Byte）

单级 shared   = dtype_bytes · (block_M·block_K + block_K·block_N)
每线程累加 reg = block_M·block_N·(accum_bytes/4) / threads
```

它回答两个问题：**这个 GEMM 是先撞带宽还是先撞算力；这份 tile 配置放不放得下。**

## 运行前提

* Python 3.9 或更高（脚本只用标准库：`argparse`、`dataclasses`）。
* **不需要 GPU、NPU，也不需要 TileLang 与 PyTorch。**

## 运行命令

```bash
# 默认：4096^3，fp16，H100 SXM5 的公开规格
python3 gemm_ledger.py

# 小 M/N 大 K（习题 E10 第 2、3 问）
python3 gemm_ledger.py --m 256 --n 256 --k 16384

# 一次打印多个 shape 的对比表
python3 gemm_ledger.py --sweep

# 换一份 tile 配置（习题 E2、E11）
python3 gemm_ledger.py --block-m 256 --block-n 128 --block-k 64 --stages 2
```

## 期望输出

正确性判据：脚本正常退出（退出码 0）。默认参数下的实际输出（在 macOS
Apple Silicon、Python 3.11 上运行得到）：

```
========================================================================
机器模型：H100 SXM5  peak=1000.0 TFLOPS  bw=3.0 TB/s  balance=333.3 F/B
========================================================================

GEMM 4096 x 4096 x 4096  (dtype 16 bit)
  计算量          : 137.4 GFLOP
  最小 DRAM 访存  : 96 MiB
  算术强度        : 1365.33 FLOP/Byte
  机器平衡点      : 333.33 FLOP/Byte
  判定            : compute-bound

tile 配置 block_M=128 block_N=128 block_K=32 stages=3 threads=128
  单级 shared memory   : 16 KiB
  多级 shared memory   : 48 KiB
  每线程累加寄存器     : 128
  受 shared 限制的 block 数 : 4
```

`--sweep` 下几组 shape 的实际结果：

| shape（fp16） | 计算量 | 最小访存 | 算术强度 | 判定 |
| --- | --- | --- | --- | --- |
| 4096³ | 137.4 GFLOP | 96 MiB | 1365.33 | compute-bound |
| 512³ | 268.4 MFLOP | 1.5 MiB | 170.67 | memory-bound |
| 128³ | 4.194 MFLOP | 96 KiB | 42.67 | memory-bound |
| 256×256×16384 | 2.147 GFLOP | 16.12 MiB | 127.01 | memory-bound |
| 8192³ | 1.1 TFLOP | 384 MiB | 2730.67 | compute-bound |

## 性能口径

**本示例不测量任何性能。** 它输出的数字全部是公式推导：

* 机器参数（1000 TFLOPS、3 TB/s）来自
  `../../../../../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md`
  （H100 SXM5，厂商公开 preliminary specs，未实测）；
* 机器平衡点是这两个数相除，不是厂商给出的数字；
* 所有算子成本按「理论最小访存量」计算，真实 kernel 一定高于这个下界；
* tile 占用只统计 A/B tile 与 C 累加器，不含 epilogue、barrier 与 padding，
  因此也是乐观下界。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过默认参数、
`--sweep`、`--m 256 --n 256 --k 16384` 与多组 tile 参数等调用，输出与本
文档一致。没有在 GPU/NPU 上运行过任何内容。

## 已知局限

* 只统计「读入 + 写出」一行，不考虑中间量回写显存。
* 不考虑 cache 命中、kernel 启动开销、尾部效应（wave quantization）。
* 寄存器估算只算累加器，不含地址、索引、流水所需的其它寄存器；真实占用
  一定更高。CUDA 的寄存器上限 255 与 cc7.0 的 occupancy 结论见
  `../../references.md` 的 R10。
