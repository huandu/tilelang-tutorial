# 图 6：两个场景 × 两个后端的约束矩阵

**用途**：正文 §1.2、§5.1。

来源：`.../DeepSelect/README.md:15-37`（两个场景的 dtype / batch / vocab / topk）、
`:37`（「目前只有 CUDA 实现支持 fp32；昇腾实现只支持 bf16」）、
`.../DeepSelect/deep_select/interface.py:53-66`（各参数说明）、
`csrc/api.cpp:57`（`topk <= 4096`）、`:223-225`（昇腾侧只接受 bf16 / int32 / 不排序）、
`csrc/structs.h:23-24,46-49`（`MAX_VOCAB_SIZE` 与 stride 对齐要求）、
`tests/test.py:20`（`MAX_VOCAB_SIZE`）、`:238-253`（性能用例的形状）。

```text
项                       Lightning Indexer (DSA)          Sampler
──────────────────────────────────────────────────────────────────────────
输入 dtype               bfloat16                         float32
topk                     ≤ 4096，主要优化 512             ≤ 4096（benchmark 512）
行长 / vocab_size        1 ~ +∞（benchmark 到 1M）        ≈ 128K（benchmark 129280）
batch_size               1 ~ +∞                           1 ~ +∞
CUDA 支持                是（bf16）                       是（fp32 的唯一路径）
昇腾支持                 是（仅 bf16）                    否（不支持 fp32）
输出 index dtype         int32 / int64                    同左；昇腾仅 int32
sorted_value             仅 CUDA 的 fp32 变体支持         支持（仅 CUDA）
```

## 读图提示

* 两个场景**唯一稳定不变的是「`k` 小、`N` 大」**；dtype、行长、后端能力都在变——
  这就是为什么算法要写成「阈值 + 随机块序」，而不是写死某个 dtype 的实现。
* `topk ≤ 4096` 是**硬约束**（`csrc/api.cpp:57`），不是优化建议；仓库也明确说
  「更大的值不支持」。
* 昇腾目前只覆盖 Lightning Indexer 场景（bf16），采样场景的 fp32 只有 CUDA 有实现。
* 输入张量的行 stride 有对齐要求：CUDA `1024 B`、昇腾 `32 B`（`csrc/structs.h`）；
  不满足要自己 padding。
