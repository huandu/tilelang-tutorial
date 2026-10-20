# 示例 03：一次 decode step 的算子耗时占比

## 目的

把本章第 4 节那张表做成可复算的：

1. 对每个算子算 FLOP、字节、算术强度与「完美重叠下的下界时间」；
2. 给出占比排序，并标出每个算子受什么限制（对应第 16 章的结论）；
3. 演示占比如何随 batch 与 context 变化——这是本章「batching 策略」那条
   主线的量化依据；
4. 自检：账本里的矩阵参数总量必须等于模型卡给的 7.61B。

## 运行前提

* Python 3.9+，仅标准库（`argparse`、`json`、`dataclasses`）。
* **不需要 GPU / NPU，也不需要 TileLang 与 PyTorch。**

## 运行命令

```bash
python3 decode_step_share.py
python3 decode_step_share.py --batch 32 --context 8192
python3 decode_step_share.py --json
```

## 期望输出

默认参数（batch=1、context=8192）下的**实际输出**节选：

```text
自检：账本里出现的矩阵，参数总量必须等于模型卡给的 7.61B
  每层参数 233.05 M × 28 层 + 嵌入/头 1089.99 M = 7.615 B
  （模型卡 7.61 B，相对误差 0.0694%）

一次 decode step 的算子账本：batch=1, context=8192, bf16, H100 公开规格
算子                            |       FLOP |      字节 |   AI |   下界时间 |   占比 | 受限
MLP gate+up × 28 层             |  7.604e+09 |  7.08 GiB |  1.0 | 2.5355ms | 52.0% | memory-bound
MLP down × 28 层                |  3.802e+09 |  3.54 GiB |  1.0 | 1.2678ms | 26.0% | memory-bound
LM head / logits                |  1.090e+09 |  1.02 GiB |  1.0 | 0.3634ms |  7.5% | memory-bound
QKV 投影 (q) × 28 层            |  7.193e+08 | 686.38 MiB | 1.0 | 0.2399ms |  4.9% | memory-bound
输出投影 (o) × 28 层            |  7.193e+08 | 686.38 MiB | 1.0 | 0.2399ms |  4.9% | memory-bound
decode attention × 28 层        |  3.288e+09 | 448.00 MiB | 7.0 | 0.1566ms |  3.2% | memory-bound
QKV 投影 (k,v) × 28 层          |  2.055e+08 | 196.25 MiB | 1.0 | 0.0686ms |  1.4% | memory-bound
RMSNorm + RoPE（2 个 norm）× 28 层 |  0.000e+00 | 784.00 KiB | 0.0 | 0.0003ms |  0.0% | memory-bound
合计（完美重叠的下界）          |            |            |     | 4.8720ms | 100.0% |

占比怎么随 batch / context 变（同一套账本）
 batch |  context |  attention 占比 |  权重类算子占比 |  单步下界(ms)
     1 |     8192 |           3.2% |           96.8% |      4.8720
     1 |   131072 |          34.7% |           65.3% |      7.2208
    32 |     8192 |          51.2% |           48.7% |      9.7857
    32 |   131072 |          94.4% |            5.6% |     84.9477
```

（排版中的空格已压缩；真实输出是等宽对齐的表格。）

## 性能口径

**本示例不测量任何性能。** 占比是「完美重叠的下界模型」推出来的，
不是 profiler 结果。它回答的是「哪个算子值得优化」，不是「实际花了多少」。

真实占比会因为这些因素偏离：kernel 启动与同步开销（会让小算子占比更高）、
算子融合（算子个数变少）、量化（权重流量减半）、并行策略（TP 会把权重流量
除以卡数，但加一次通信）。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过默认参数、
`--batch/--context` 与 `--json` 三种调用；输出与上面一致。

## 已知局限

* 只算一层解码加最终分类头，**没有**算 embedding 查表、采样与多卡通信。
* attention 的字节只算 KV 的读取，没算 split-KV 带来的 `Output_partial`
  额外流量；开了 split 之后这一项会变成 `batch × heads × num_splits × dim`。
* 每个 GEMM 的激活读写取的是理想下界；真实实现会多读多写。
