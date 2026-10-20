# 图 5：这套生态的能力状态

**用途**：正文 §7.1、§7.2。

**事实来源**：各仓库 README 的 `Features` / `Ongoing` 一节，以及源码里的断言。
逐条出处写在正文 §7.1 的三张表里。

```text
状态        能力                                        依据（一章 / 一文件）
─────────────────────────────────────────────────────────────────────────
可用        EP dispatch / combine（BF16 / FP8）           DeepEP README:60-74
可用        DeepGEMM dense / M-grouped / MQA / MegaMoE    DeepGEMM README:90-158
可用        FlashMLA 昇腾稀疏 prefill / decode             FlashMLA README:13-15
可用        DeepSelect Lightning Indexer（bf16）           DeepSelect README:19-29
可用        TileKernels 全算子族（TileLang 双后端）         TileKernels README:9-19

实验性      PPBuffer / EngramBuffer / Bucket all-gather   DeepEP README:67-69,80-82
实验性      两后端的语义等价（未对拍）                     第 33 章 §8.3

不支持      Bucket reduce-scatter / all-reduce             DeepEP README:80
不支持      lb_prefetch_weights / lb_reduce_grads          DeepEP README:81
不支持      EP 入 graph capture                            DeepEP README:84
不支持      masked grouped GEMM                            gemm.hpp:591,615
不支持      昇腾上的融合 norm-RoPE-attn-RoPE-cast          FlashMLA README:39,53
不支持      DeepSelect 的非 bf16 路径                      DeepSelect README:37
不支持      DeepJIT 的缓存预热 / Python 编译 API            DeepJIT README:18-21
```

**要点**

* 这是 **2026-09-30 这次发布时点**的状态，依据是 corpus 里各仓库的 snapshot commit
  （`PROVENANCE.md`）。仓库一更新，这张表就可能过期。
* “不支持”不等于“做不到”，多数是“昇腾侧 kernel 还没写”（正文 §7.2）。
* 维护这张表的四步方法见正文 §7.3。
