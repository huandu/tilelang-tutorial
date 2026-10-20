# 示例 02：roofline 的八种误读

## 目的

第 28 章 2.3 节列了 roofline 的常见误用。这个脚本把其中八条**各自量化**一遍，
全部作用在同一个算子（QK-Norm + RoPE + fp8 量化）上，这样每条误用带来的偏差
可以直接比较：

| 编号 | 误读 | 它把答案带偏多少 |
| --- | --- | --- |
| M1 | 用 datasheet 峰值而不是标定后可达到的值 | compute time 低估 1.36× |
| M2 | 用张量核 roof 给逐元素算子定价 | 平衡点从 12.72 变成 161.97（12.7×） |
| M3 | 按输入 dtype 声明 roof | 同一算子三个上限：306.67 / 161.97 / 12.72 |
| M4 | 用其中一个 stage 的 AI 代表融合后的整算子 | AI 从 2.79 被读成 1.40（2.0×） |
| M5 | 超越函数按 1 FLOP 记账 | softmax 行：0.8k → 1.7k cycles（2.08×） |
| M6 | 只数 payload、不数查表 | 63.57 MB → 67.76 MB；真实设计里可放大到 4.65× |
| M7 | 在模型没有抓手的尺度上读数字 | 一个 decode step 的 SOL 是 0.003 µs |
| M8 | 把 >100% 的效率当成好 kernel | 那是公式错误，不是性能 |

## 运行前提

* Python 3.9+，只用标准库。
* **不需要 GPU / NPU / TileLang / PyTorch。**

## 运行命令

```bash
python3 roofline_misuse.py            # 全部八条
python3 roofline_misuse.py --only 5   # 只看第 5 条
```

## 期望输出

正确性判据：退出码为 0。`--only 5` 的实际输出：

```text
M5  charging a transcendental the same as a multiply
  the sharpest case is not this operator, it is the softmax in the attention
  kernel downstream of it. Take rows of 8192 scores:
  ledger says  : 41.0 kFLOP per row at 1 FLOP each
  of which 20% are exp, charged as 1 FLOP
  On the Ascend VALU the same op is charged at ops/cycle:
      mul/add/min-max/cmp/logic   51 ops/cycle
      div/mod                     13 ops/cycle
      special_func (rsqrt, exp)    8 ops/cycle
  a transcendental is 6.38x a multiply per op.
  Weight each op by its own pipe and the 'compute time' moves:
      unweighted:      0.8 k cycles  (40960 ops / 51)
      per-pipe  :      1.7 k cycles   (2.075x)
```

## 性能口径

**不测量任何性能。** 每条误读的「正确读法」也仍然是模型值，不是实测值。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过全部八条与
单条模式，输出与本文件一致。没有在 GPU/NPU 上运行过任何内容。

## 已知局限

* 八条误读里，M5 与 M6 的量化只覆盖了该误读的一种形态；同一条误读在别的算子上
  会给出别的倍数（脚本里都写了「为什么」）。
* M5 用的是**昇腾**的 ops/cycle 表（`estimate_latency.h:57-68`）。本 corpus 内
  NVIDIA 侧没有等价的 SFU 吞吐表，脚本按这个事实直说，不编数字。
* 所有倍数都是在默认机器 profile 下的结果；换机器 profile 会变。
