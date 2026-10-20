# 示例 06：完整案例——QK-Norm + RoPE + fp8 量化

## 目的

这是第 28 章第 6 节案例的可复算版本：同一个算子，三种设计，
`预算表 → 第一版 → 性能诊断 → 第二版`。

被建模的算子（推理侧 attention 前处理）：

```
q,k = proj(x)                         # bf16, [tokens, heads, head_dim]
q,k = rmsnorm(q,k) per (token, head)  # 沿 head_dim
q,k = rope(q,k)                       # 每个 token 所有 head 共用同一张表
q8,k8, scale = fp8_quant(q,k)         # 每行一个 fp32 scale
```

三种设计：

| 设计 | 一个 block 处理 | 特点 |
| --- | --- | --- |
| A | 一行（一个 (token, head) 的 128 个元素） | 最直觉的写法：一行一 block |
| B | 连续的 8 行（对齐到 head 组） | 把查表摊到 8 行上 |
| C | 一整个 token（全部 q 和 k 的 40 行） | 把查表摊到整个 token |

模型只数两件事：**payload**（输入、输出、scale）和 **lookup**（norm 权重、RoPE
cos/sin 表）。lookup 的关键不是它有多大，而是**它被重复读了多少次**。

## 运行前提

* Python 3.9+，只用标准库。
* **不需要 GPU / NPU / TileLang / PyTorch。**

## 运行命令

```bash
python3 case_norm_rope_quant.py
python3 case_norm_rope_quant.py --rows-per-block 8
python3 case_norm_rope_quant.py --tokens 8192 --bandwidth-gbs 3350
python3 case_norm_rope_quant.py --q-heads 64 --kv-heads 8   # 别的模型配置
```

## 期望输出

正确性判据：退出码为 0。默认参数下的实际输出（节选）：

```text
part 0 -- the operator ledger
flops                   : 210.2 MFLOP
ideal bytes             : 67.76 MB   (payload 63.57 MB + lookup 4.19 MB)
arithmetic intensity    : 3.102 FLOP/Byte
memory roof at 4502.4 GB/s : 15.051 us

part 1 -- budget table
design                     rows/blk    grid  waves  on-chip KB  payload MB  lookup MB  total MB  vs ideal
A: one row per block              1  163840    156         1.9       63.57     251.66    315.23     4.65x
B: 8 rows per block               8   20480     20         8.0       63.57      31.46     95.03     1.40x
C: one whole token per block     40    4096      4        36.2       63.57       6.29     69.86     1.03x

part 2 -- diagnosis
A: one row per block
    total  315.23 MB = payload  63.57 MB + lookup  251.66 MB   (lookup is  79.8% of all traffic)
    over the ideal by  247.46 MB  -> the achievable time floor rises from 15.05 us to 70.01 us
B: 8 rows per block   -> 95.03 MB, floor 21.11 us
C: one whole token    -> 69.86 MB, floor 15.52 us
```

这组数字就是案例的骨架：**第一版（A）比理想多搬 4.65 倍的字节，其中 79.8% 是查表；
把粒度和 head 组对齐之后（B/C），剩下的差距只有 3%。**

## 性能口径

**不测量任何性能。** 只有 `bytes / bandwidth` 一类的下界。案例里凡是要说
「实测多少」的地方，正文都写成「在有卡机器上应该这样验证」，不给数字。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过默认、
`--rows-per-block`、`--tokens`、`--q-heads/--kv-heads` 多组参数；输出与本文件一致。
没有在 GPU/NPU 上运行过任何内容。

## 已知局限

* **lookup 的重复次数是设计意图，不是实测的 cache 行为。** 真实机器上 L2 完全可能
  把权重和表兜住，那时 A 设计的实际流量会接近 B/C。脚本把两种情况分开说：
  「模型上被重复读了几次」由设计决定，「实际少读了多少」由 cache 决定。
  `../02-roofline-misuse/` 的 `M6` 是同一件事的另一个侧面。
* 片上占用只算「一个 block 同时要活着的 buffer」，不含流水多版本。
* 没有建模 DMA 描述符开销：A 设计有 163840 个 block，每个 block 只搬 256 B 输入，
  这个开销在真实机器上只会让 A 更差，所以模型对 A 是**乐观**的。
* 波数用一个假设的 `blocks_per_SM`（默认 8）算，真实占用要用 `../04-tile-size-decision/`
  去查。
