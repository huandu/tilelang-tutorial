# 图 2：prefill 与 decode 的组件链路对照

**用途**：正文 §3.1–§3.4。

**事实来源**：

* 第 32 章 §6.1 的“prefill / decode 各组件位置”表。
* `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:142-149`
  （MQA logits：prefill `Q=4096` 681 TFLOPS / 283 GB/s；decode `Q=256`
  565 TFLOPS / 2006 GB/s；FIX-pipe bound 99%）。
* `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:19-44`
  （Lightning Indexer 场景：bf16、`topk=512`；指标是有效内存带宽）。
* `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:13-15,63,136`
  （昇腾 prefill 410 TFlops / 95%，decode 360 TFlops / 83%；
  decode 只支持 FP8 / FP4 KV cache）。
* `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:9-11`
  （EP 通信的测量条件）。

```mermaid
graph TB
    subgraph P["prefill（大 M）"]
        P1["① 索引打分：MQA logits<br/>FIX-pipe bound 99%"]
        P2["② TopK：DeepSelect<br/>topk=512 · 有效内存带宽"]
        P3["③ 稀疏注意力：FlashMLA<br/>410 TFlops / 95%"]
        P4["④ 专家通信：DeepEP<br/>dispatch 达链路 90–95%"]
        P5["⑤ 常规算子：TileKernels"]
        P1 --> P2 --> P3 --> P4
        P5
    end
    subgraph D["decode（小 M）"]
        D1["① 索引打分：MQA logits<br/>Q=256 · 带宽 2006 GB/s"]
        D2["② TopK：DeepSelect<br/>bf16 only（昇腾）"]
        D3["③ 稀疏注意力：FlashMLA<br/>360 TFlops / 83%"]
        D4["④ 专家通信：DeepEP<br/>小 batch · 启动开销占比高"]
        D5["⑤ 采样 TopK：DeepSelect<br/>（昇腾无 fp32 路径）"]
        D1 --> D2 --> D3 --> D4 --> D5
    end
```

**要点**

* 两段用的是**同一批算子**，但形状不同：prefill 的 `M` 是几千，decode 的是几百
  甚至更小。**瓶颈是形状的函数**（正文 §3.4）。
* prefill 的 ① 是 FIX-pipe bound，既不是算力也不是 HBM（第 35 章 §5.5）。
* decode 的 ③ 有一条硬约束：只支持 FP8 / FP4 KV cache（`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:136`）。
