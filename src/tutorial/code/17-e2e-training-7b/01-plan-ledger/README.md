# 01 · 显存与算力账本

## 目的

动手训练之前，先把两个数算清楚：训练态下每个参数要存几份（`train_plan.py`
的 `memory_bytes`），以及训练 `D` 个 token 需要多少 FLOPs（`training_flops`，
`C ≈ 6·N·D`）。它是正文 §2“可行性核算”的可执行版本。

## 运行前提

纯 Python 3，只用标准库。**不需要 GPU、不需要 torch、不需要网络。**

## 命令

```bash
python3 train_plan.py                                   # 默认：10B token、120 TFLOP/s 有效吞吐
python3 train_plan.py --tokens 350e9 --eff-tflops 90    # FineWeb-Edu 350B 子集
python3 train_plan.py --dp-size 16                      # 16 卡均摊账本
```

## 期望输出

以默认参数为例（写作机器实测）：

```
== 显存账本：Qwen2.5-7B（7.61B 参数）==
  权重 (bf16)           14.17 GiB
  梯度 (bf16)           14.17 GiB
  Adam m (fp32)       28.35 GiB
  Adam v (fp32)       28.35 GiB
  fp32 主权重            28.35 GiB
  合计                 113.40 GiB
  按 ZeRO-3/FSDP 均摊（优化器/权重/梯度）：1 卡→113.40 GiB  8 卡→ 14.17 GiB  16 卡→  7.09 GiB
  （不含激活值、通信缓冲、CUDA context 与显存碎片；见正文 §2.2）

== 算力账本：在 10.0B token 上训练 ==
  C ≈ 6·N·D = 4.566e+20 FLOPs
    1 卡 ×   120 TFLOP/s 有效吞吐 →     1056.9 卡·小时，墙钟  1056.94 小时
    8 卡 ×   120 TFLOP/s 有效吞吐 →     1056.9 卡·小时，墙钟   132.12 小时
   64 卡 ×   120 TFLOP/s 有效吞吐 →     1056.9 卡·小时，墙钟    16.51 小时
```

## 正确性判据

* 手算核对 `memory_bytes`：`7.61e9 × 2 × 2 (权重+梯度) + 7.61e9 × 4 × 3 (Adam m/v + 主权重)`
  再除以 `2^30`，应等于 113.40 GiB。
* **`--eff-tflops` 默认值 120 是占位假设，不是测量结论。** 它必须替换成你自己机器
  上实测到的有效吞吐（第 12 章讲怎么测），否则输出的时间是假设。

## 验证级别

**CPU 逻辑验证**：`python3 -m py_compile train_plan.py` 与 `python3 train_plan.py`
都在写作机器上跑过，上面的输出是真实运行结果，不是手写的。
