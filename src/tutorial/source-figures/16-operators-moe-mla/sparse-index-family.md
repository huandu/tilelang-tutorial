# 图 · 稀疏/索引算子的关系

```mermaid
flowchart TD
    Q["query"] --> IDX["indexer<br/>全序列打分（MQA 形状）"]
    IDX --> LOGITS["logits: S x S_kv"]
    LOGITS --> TOPK["top-k selector<br/>输出位置，不输出值"]
    TOPK --> IDS["indices: S x topk"]
    IDS --> GATHER["按索引 gather KV 行"]
    GATHER --> SMA["sparse MLA<br/>只算被选中的位置"]

    Q --> NSA["NSA<br/>缓存压缩块 + 选中块 + 滑窗"]
    Q --> HISA["HISA<br/>先选 pool block，再选 block 内 token"]
    IDX --> HISA
```

```text
三条路线，一个共同点：把「读完整 KV」换成「读索引 + 读少量 KV」。
  NSA  : 输入是 block 索引，kernel 按块 gather
  DSA  : indexer 出 logits -> top-k 出位置 -> sparse MLA gather
  HISA : 两级层级，粗选 block 再细选 token
代价  : indexer 要扫全序列；L 很长时它的 FLOP 可以超过被它选中的 attention
       （见 code/05-sparse-index-map/sparse_index_map.py 的盈亏平衡表）。
```

依据：`_vendor/tilelang-v0.1.15/examples/{deepseek_nsa,deepseek_v32,dsa_hisa}/`。
