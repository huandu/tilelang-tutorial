# 图 4：一次推理请求（prefill / decode）里各组件的位置

**用途**：正文 §6.1–§6.3。

```text
[fig-04] 一次推理请求（prefill / decode）里各组件的位置

  prefill：整段 prompt 一次算完
    ① 输入 token
         │
    ② MQA logits         DeepGEMM-Ascend（Lightning Indexer 打分）
         │
    ③ TopK 筛选          DeepSelect（选出 top-k KV）
         │
    ④ 稀疏注意力          FlashMLA（DSA 稀疏 prefill）
         │
    ⑤ 输出 / 首个 token

  decode：每生成一个 token 循环一次（③④ 只对选中的 KV 计算）
    ⑤ ──► ② ──► ③ ──► ④ ──► ⑤

  MoE 推理的专家通信（DeepEP-Ascend）与常规算子（TileKernels）穿插在 ②–⑤ 之间
```

## 要点

* **DSA 改变了推理链的形状**：先打分（②）再筛选（③）再稀疏计算（④），不再是
  「一次稠密注意力」。
* **各段的实现分属不同仓库**：② DeepGEMM-Ascend、③ DeepSelect、④ FlashMLA、
  常规算子 TileKernels、专家通信 DeepEP-Ascend。
* **decode 只支持量化 KV cache**（FP8 / FP4），不支持 bfloat16 KV cache；
  prefill 版本没有 batch 维。

来源：`../../../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:7,13,111-126,136,172`、
`.../FlashMLA/docs/20260930-ascend-prefill-deep-dive-zh.md:18-24`、
`.../DeepSelect/README.md:3`、`.../DeepGEMM-Ascend/README.md:142`。
