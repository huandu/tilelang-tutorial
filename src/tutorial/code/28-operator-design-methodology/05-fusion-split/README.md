# 示例 05：fusion / split 的四行账本

## 目的

第 28 章第 5 节的判据在这里变成一张表。六个真实取舍，每个都写成两个具名选项
（A / B），然后套同一条规则：

```
一个选项可用  <=>  它在片上放得下 且 它的 grid 装得满设备
在可用选项里，选搬运字节更少的那个
两个都不可用   ->  答案是「都不行」，先去重新分块
```

六个取舍里有两个是**故意**让搬运更多字节的那一方胜出——split-K 和 split-KV 就是
“用流量买并行度”。这就是为什么判据必须写出来，而不能靠感觉。

## 运行前提

* Python 3.9+，只用标准库。
* **不需要 GPU / NPU / TileLang / PyTorch。**

## 运行命令

```bash
python3 fusion_split.py
python3 fusion_split.py --bandwidth-gbs 3350 --sm-count 108   # 换成别的卡
```

## 期望输出

正确性判据：退出码为 0。实际输出的汇总部分：

```text
decision                                        A MB      B MB  delta MB    A grid    B grid  winner
elementwise chain, 2 ops                       134.2      67.1      67.1     32768     16384  1 fused kernel
QK-Norm+RoPE then fp8 quant                    151.7      67.8      83.9    163840     20480  one fused kernel
GEMM + bias + GeLU epilogue                    100.7      33.6      67.1      1024      1024  GEMM with fused epilogue
skinny GEMM 1024x1024x16384, split-K=8           4.2      71.3     -67.1        64       512  split-K=8
decode attention, sq=1, skv=128k, split-KV=64    2147.5    2149.6      -2.1        32      2048  split-KV=64
norm fused into the Q projection's output       67.1       0.0      67.1     16384      1024  norm fused into the GEMM epilogue
```

两条被同一个判据选中的“搬运更多字节”的选项：

```text
winner: split-K=8 -- parallelism: the other option launches 64 blocks for 132 SMs,
        so it pays +67.1 MB to reach 512
winner: split-KV=64 -- parallelism: the other option launches 32 blocks for 132 SMs,
        so it pays +2.1 MB to reach 2048
```

## 性能口径

**不测量任何性能。** `delta MB` 与由它换算的 µs 都是模型值。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过默认与多组
`--bandwidth-gbs / --sm-count`；输出与本文件一致。没有在 GPU/NPU 上运行过任何内容。

## 已知局限

* 六个取舍的字节数都是**手推的**，公式写在构造它们的 `build()` 里（每一项都有一行
  注释说明它数的是什么）。它们不是从真实 kernel 里量出来的。
* “装得满设备”的判据用了简单的 `grid >= SM 数`，没有考虑 `blocks_per_SM`。
  严格的做法见 `../04-tile-size-decision/`。
* `on chip` 只算了这一级 fusion 需要同时活着的 buffer，不含流水多版本。
* 芯片型号不同，结论可能翻转：把带宽或 SM 数改小，元素级融合的收益会一起变小。
