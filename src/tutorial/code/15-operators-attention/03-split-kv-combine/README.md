# 示例 3：split-KV 的合并公式与它的精度代价

## 目的

正文第 5.4 节说：decode 阶段把 KV 切成 `num_split` 段之后，各段的**输出已经各自除过
自己的 `ℓ`**，所以合并时必须按 `ℓ_s / Σℓ` 加权，既不能平均也不能直接相加。本示例
把这句话变成可执行的检查：

1. 用“一个 query 行 + 长 KV”的 decode 形状，分别用**加权**、**平均**、**直接相加**
   三种方式合并各段输出，看哪一种能复现精确结果；
2. 打印各段权重，确认权重之和为 1；
3. 把各段的部分输出与 LSE 都按 **float16** 圆整后重跑，观察误差随 `num_split` 的增长。

## 运行前提

Python 3.9+ 与 numpy。不需要 GPU/NPU、不 import tilelang。

## 运行命令

```bash
python3 split_kv_combine.py
```

## 期望输出（写作机实际输出，退出码 0）

```text
1. weighted combine vs. the two naive ways of merging partials
d=8, Skv=1024, num_split=8
merge rule                             max abs err
--------------------------------------------------
weighted by l_s / sum(l)  (correct)       5.551e-17
plain mean of partials      (wrong)       4.667e-03
plain sum of partials       (wrong)       6.147e-01

per-split weights (exp2(lse_s - logsumexp(lse))):
  0.1298 0.1168 0.1096 0.1216 0.1295 0.1415 0.1350 0.1162
  sum of weights = 1.0000000000  (must be 1)

2. error vs. number of splits, with fp16 partials + fp16 LSE
d=128, Skv=16384
 num_split     fp64 partials     fp16 partials
----------------------------------------------
         1         2.116e-16         6.445e-06
         2         1.717e-16         1.510e-05
         4         1.475e-16         2.248e-05
         8         1.561e-16         3.719e-05
        16         1.509e-16         3.496e-05
        32         1.422e-16         3.771e-05
```

## 判据

* 退出码 0；
* 加权合并的误差在 `1e-16` 量级（fp64 机器精度）；
* 权重之和为 1；
* `fp64` 那一列与 `num_split` 无关，`fp16` 那一列随 `num_split` 增大而变大。

## 注意事项

* 第 2 段的 fp16 误差绝对量很小（`1e-5`），是因为这里的分数幅度不大。**真实的
  LLM decode 里 LSE 的绝对值可以到几百**，fp16 只有 10 位尾数，`2^(lse_s − lse_max)`
  会被显著扭曲。脚本演示的是**趋势**，不是最坏情况。
* 更极端的例子在昇腾 backward 的 README 里：dQ 的 partial gradient 先转 bf16 再对
  64 个 KV tile 做原子累加，实测相对 L2 误差约 **1.024%**（厂商自报，作者未复现）。
  正文第 5.4 节引用了这个数字。
* 这里的“split 段”是照着
  `_vendor/tilelang-v0.1.15/examples/flash_decoding/example_mha_inference.py:80-111`
  的语义重写的 CPU 版本，**不是**在运行那个 kernel。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 + numpy 1.24.1 上实际运行过，
退出码 0，输出即上表。**没有运行过 flash decoding kernel。**
