# 示例 1：单个算子的访存/计算账本

## 目的

把第 02 章 2.1–2.3 节的记账法写成可运行的脚本：

```
算术强度 AI = 运算量(FLOP) / 访存量(Byte)
机器平衡点   = 峰值算力 / 峰值带宽
时间下界     = max(访存量 / 带宽, 运算量 / 峰值算力)
```

对逐元素、逐元素链、LayerNorm、GEMM、attention（prefill 两种写法 / decode）、
AdamW、梯度 all-reduce 各算一次，并给出 memory-bound / compute-bound 判据。

这不是性能预测器。它只回答一个问题：**这个算子先撞带宽，还是先撞算力；理论最快多快。**

## 运行前提

* Python 3.9 或更高（只用标准库：`argparse`、`dataclasses`）。
* **不需要 GPU / NPU / PyTorch / TileLang。**

## 运行命令

```bash
python3 op_traffic.py                                  # 默认：H200 标定参数
python3 op_traffic.py --machine h100-spec               # 换成 H100 的厂商 preliminary 规格
python3 op_traffic.py --seq-len 8192 --heads 32 --head-dim 128
python3 op_traffic.py --params 7e9 --world 64 --op adamw,allreduce
```

## 期望输出（默认参数下的实际输出）

正确性判据：退出码 0，并打印机器模型与一张算子账本表。

```
====================================================================================================
机器模型
----------------------------------------------------------------------------------------------------
  profile      : H200（TileOPs 微基准标定：0.737x 算力 / 0.938x 带宽）
  峰值算力     : 729.3 TFLOPS（fp16 张量核口径）
  峰值带宽     : 4502 GB/s
  机器平衡点   : 162.0 FLOP/Byte
====================================================================================================
算子账本（理论最少访存量；时间下界 = max(访存时间, 计算时间)）
----------------------------------------------------------------------------------------------------
逐元素加 y=x+b  (16.8M 元素, fp16)                       0.0168 GFLOP  100.66 MB      0.17 F/B    22.4 us    memory-bound
逐元素链 2 个 op（eager, 每个 op 一个 kernel, fp16）          0.0336 GFLOP  134.22 MB      0.25 F/B    29.8 us    memory-bound
逐元素链 2 个 op（融合成 1 个 kernel, fp16）                  0.0336 GFLOP   67.11 MB      0.50 F/B    14.9 us    memory-bound
LayerNorm 融合（8192x4096, fp16）                      0.268 GFLOP  134.22 MB      2.00 F/B    29.8 us    memory-bound
LayerNorm 拆成 3 趟（8192x4096, fp16）                  0.268 GFLOP  268.44 MB      1.00 F/B    59.6 us    memory-bound
GEMM 4096x4096x4096 (fp16)                          137 GFLOP  100.66 MB   1365.33 F/B   188.5 us   compute-bound
Attention prefill（先落 SxS 分数矩阵, S=4096）              278 GFLOP    6.58 GB     42.20 F/B   1.461 ms    memory-bound
Attention prefill（flash 风格，不落 SxS, S=4096）          278 GFLOP  134.22 MB   2068.00 F/B   380.6 us   compute-bound
Attention decode（KV cache 长 32768, fp16）           0.542 GFLOP  536.87 MB      1.01 F/B   119.2 us    memory-bound
AdamW step（7.00B 参数, state fp32, grad bf16）         105 GFLOP  182.00 GB      0.58 F/B  40.423 ms    memory-bound
梯度 all-reduce（ring, 8 卡, bf16）                              0   24.50 GB      0.00 F/B    5.442 ms    memory-bound
====================================================================================================
```

（上表为便于阅读省略了机器来源行与页脚说明；完整输出以实际运行为准。）

## 性能口径

**本示例不测量任何性能。** 所有数字都是公式推导：

* 机器参数来自两处，都在 `MACHINES` 表里逐条注明：
  * `h100-spec`：`../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md`
    （H100 SXM5，厂商公开的 preliminary specs：1000 TFLOPS fp16、over 3 TB/s）；
  * `h200-spec` / `h200-measured`：`../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/profiles/h200.yaml`
    （厂商规格 × 仓库自带微基准标定；标定的测量条件写在该 YAML 的注释里：
    2048 MB working set、5 次 ×200 轮、显存 3201 MHz，算力受 700 W 限制）。
* 算子的 (flops, bytes) 按 `../tilelang-corpus/12-operator-libraries/raw/TileOPs/docs/design/roofline.md:44-66`
  的记账口径（每个输入读一遍、每个输出写一遍、中间量不计；基本运算与超越函数各 1 FLOP）。

**换机器只需换两个数。** 如果手上有别的卡，用 `--machine` 之外的参数改不动，
可以照 `MACHINES` 的格式加一条，并写上出处。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过默认参数、
`--machine h100-spec`、`--op` 过滤与多组形状；`python3 -m py_compile op_traffic.py` 通过。
没有在 GPU/NPU 上运行过任何内容。

## 已知局限

* 只统计「算法理论最少」的访存，真实实现的高于这个下界；本脚本给出的是**乐观下界**。
* 不考虑 cache 命中、kernel 启动开销、尾部效应；TileOPs 的 roofline 文档把这类情形
  单独标为「latency-bound」，本脚本不做这个判断。
* 超越函数在账本上按 1 FLOP 记（TileOPs 的口径），真实硬件上 SFU 的吞吐与 FMA 差很多，
  所以 `exp/tanh` 密集的算子（softmax、SiLU）实际会更慢。
* LayerNorm 的 8 FLOP/元素、AdamW 的 15 FLOP/参数都是本书口径的粗估，用来说明量级。
