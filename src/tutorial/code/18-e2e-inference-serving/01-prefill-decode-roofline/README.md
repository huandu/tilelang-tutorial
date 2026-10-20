# 示例 01：prefill 与 decode 的算术强度账本

## 目的

把「prefill 是 compute-bound、decode 是 memory-bound」从一句口号变成可以
复算的东西：

1. 用模型卡的两个参数量反推并互校 Qwen2.5-7B 的形状字段；
2. 看同一个 GEMM 形状在 M 从 1 涨到 prompt 长度时，算术强度怎么翻过平衡点；
3. 算 decode attention 的算术强度（只由 GQA 复用率决定）；
4. 算「读一遍权重」的时间下界，以及 batch 把它摊薄的效果。

## 运行前提

* Python 3.9+，仅标准库（`argparse`、`json`）。
* **不需要 GPU / NPU，也不需要 TileLang 与 PyTorch。**

## 运行命令

```bash
python3 prefill_decode_roofline.py
python3 prefill_decode_roofline.py --json
```

## 期望输出

下面是默认参数下的**实际输出**节选（macOS Apple Silicon、Python 3.11）：

```text
机器平衡点（H100 SXM5 公开规格 fp16）: 333.3 FLOP/Byte

先用模型卡的两个参数量反推/互校形状字段
  非嵌入参数（本书按形状算）: 6.525 B   模型卡给出 6.53 B
  总参数（本书按形状算）    : 7.615 B   模型卡给出 7.61 B
  相对误差                  : 0.0694%

MLP down-proj  [I -> d]  N=18944 K=3584
     M |         FLOP |          bytes |   AI (F/B) | 判定
     1 |    1.358e+08 |      1.358e+08 |        1.0 | memory-bound
     8 |    1.086e+09 |      1.362e+08 |        8.0 | memory-bound
    64 |    8.691e+09 |      1.387e+08 |       62.7 | memory-bound
   512 |    6.952e+10 |      1.589e+08 |      437.7 | compute-bound
  → 翻转点 M* ≈ 375（N、K 固定，只让 M 变）。

decode attention：算术强度与序列长度无关（第 16 章的结论）
  AI = 2 * H_q / (H_kv * b)
  MHA   (H_kv=28, b=2) :      1.0 FLOP/Byte
  GQA   (H_kv=4,  b=2) :      7.0 FLOP/Byte
  MQA   (H_kv=1,  b=2) :     28.0 FLOP/Byte
  机器平衡点             :    333.3 FLOP/Byte

一次 decode step 的「权重底线」
  权重字节（bf16）            : 15.23 GB
  → batch=1 读一遍权重的下界  : 5.08 ms
  → 单请求速度上界（不含 KV） : 197.0 tok/s
```

`--json` 输出同一批数字的机器可读版本，便于做回归比较。

## 性能口径

**本示例不测量任何性能。** 其中：

* H100 SXM5 的 fp16 1000 TFLOPS / HBM3 over 3 TB/s 是**厂商公开规格**
  （见 `../../references.md` 的 R1、R2），不是本机测量；
* 平衡点约 333 FLOP/Byte 是上面两个数的商；
* `AI`、`M*`、权重下界时间、tok/s 上界全部是**本书推导的下界模型**；
* 模型卡的 6.53B / 7.61B 参数量用于反向校验形状字段，误差 0.07%。

## 作者实际做到的验证级别

**CPU 逻辑验证。** 在 macOS (Apple Silicon)、Python 3.11 上运行过默认参数与
`--json` 两种调用；数字与上面的输出一致。没有在 GPU / NPU 上运行过任何内容。

## 已知局限

* 模型是**下界**：不含 cache 命中、kernel 启动、同步、通信、尾部效应；
  真实 kernel 一定更慢。
* 「权重读一遍」假设没有 TP / EP 切分。切成 N 份后每份只读自己的权重，
  但要多付一次通信。
* 算术强度用「A、B 各读一次、C 写一次」的上界口径；epilogue 与中间结果
  落地都会让真实值更低。
