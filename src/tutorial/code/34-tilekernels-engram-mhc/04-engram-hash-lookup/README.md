# 示例 04：Engram 的 n-gram 哈希与 O(1) 查表

## 目的

Engram 论文说它的查表是 `O(1)`，而且因为寻址是**确定性的**，把上百亿参数的表
卸载到 host memory 也“开销可忽略”。这个脚本把这两句话落到可复算的代码上：

* 复现 `engram_hash_ref`（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/engram.py:118`）与内核的
  乘法-XOR 哈希（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_hash_cuda.py:54-63`）；
* 证明“同一串 token id → 同一组 index”（确定性）；
* 说明“每个 token 取的行数是架构常量，与表大小无关”；
* 展示 image token 写 `-1` 这个“不查表”标记。

## 运行前提

* Python 3.9+、PyTorch（CPU 即可）。

## 运行命令

```bash
python3 engram_hash_lookup.py
```

## 期望输出

在写作机上的**实际输出**（index 数值是随机种子固定的，可以复现）：

```text
== shapes ==
  ngram_token_ids     (6, 3) int32
  multipliers         (2, 3) int64
  vocab_sizes         (2, 2, 8) int32 (prime, per head)
  offsets             (2, 16) int32 (exclusive prefix sum)
  indices             (2, 6, 16) int32
    = num_layers x num_tokens x ((max_ngram-1) * heads) = 2 x 6 x 16
  index range: min=17329 max=2091973 (table spans 0 .. 4201439)

== deterministic addressing ==
  rerun on the same ids: identical = True
  layer 0, token 0 indices: [29861, 217345, 281072, 397629, ...]

== the number of gathered rows is set by the architecture ==
  table with   258,000 slots (~     3.9 MiB at 4 fp32 dims): rows gathered per token = 16
  table with 10,000,000 slots (~   152.6 MiB at 4 fp32 dims): rows gathered per token = 16
  => a 40x larger table costs the same number of lookups; ...

== image tokens: the kernel writes -1 ==
  mask                 [True, False, False, True, False, False]
  layer 0, token 0     [-1, -1, ...]  (all -1, no lookup)
  layer 0, token 1     [56815, 191622, 297290, ...]  (real indices)

== the actual lookup is a plain gather ==
  table (4201440, 4); gathered (16, 4) -> concat into the memory vector e_t
  e_t per token = 16 rows x 4 dims = 64 values
```

## 怎么读这个结果

* **确定性**：`indices` 只依赖 token id。真实实现里 `multipliers`、`vocab_sizes`
  （都是素数）与 `offsets` 是**权重之外**的固定超参，训练时不更新（见
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_hash_kernel.py:14-27` 的参数说明）。所以**前向还没开始**，
  系统就知道要读哪些行——这是“提前预取、把通信藏在上一层计算后面”的前提。
* **`O(1)` 取行数**：每个 token 每层取 `(max_ngram - 1) × heads` 行。脚本里是
  `2 × 8 = 16` 行。把表从 25.8 万槽放大到 1000 万槽（40 倍），取行数不变。
  论文的 `O(1) lookup` 说的就是这个（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:27,71`）。
* **`-1` 标记**：图像 token 不做 Engram 查表，kernel 直接把整行输出写成 `-1`
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_hash_cuda.py:45-47`），后续的 gather 与 gate 会跳过它。
  这正是正文 §3 里门控 kernel 需要 `image_token_mask` 的原因。
* **真正的查表**就是一次 gather：`table[indices]`。脚本给的表是 fp32 的玩具表；
  真实表的“大”体现在**容量**上（论文里到 130 亿参数），而不是每次访问的行数上。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11、torch 2.1.1 上实际运行过。
哈希与 offsets 的实现与 corpus 的 `engram_hash_ref`、`make_offsets` 一致；
“确定性”“取行数与表大小无关”“`-1` 标记”三条结论都可以在这个脚本里直接看到。
**没有跑过 kernel，也没有跑过论文的模型。**

## 已知局限

* 论文里 `φ` 是“lightweight multiplicative-XOR hash”，本脚本按仓库的实现复现；
  论文的 appendix C 关于 tokenizer 压缩（词表缩小 23%）的细节没有在这里复现
  （那需要真实 tokenizer，见 `../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/engram_demo_v1.py` 的
  `CompressedTokenizer`）。
* “把表放到 host memory、PCIe 预取的开销 < 3%”是**论文自报**的实验结论
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:95-99`），本机无法复现，脚本也不涉及。
