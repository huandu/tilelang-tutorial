# 05-sparse-index-map · 稀疏/索引类算子的结构对照与代价模型

**目的**：(A) 把 NSA / DSA indexer / top-k selector / sparse MLA / HISA / V4 的
swa·hca·csa 放在一张结构对照表里，并核对引用；(B) 用配置里的真实参数算
“indexer 扫全序列 vs attention 只算 top-k”的 FLOP 比与访存比。

**运行前提**：Python 3.9+，纯标准库；核对需要 `_vendor/`。

**运行命令**

```bash
python3 sparse_index_map.py
```

**预期输出（判据）**

* 引用核对：`命中 8 处，未命中 0 处`。
* 代价表：给定 L，比较 indexer 的 FLOP 与 attention(topk) 的 FLOP；给出盈亏平衡
  所需的 topk；再按 fp16 打印“dense KV / indexer KV / sparse KV”的字节与比值。

**作者实际做到的验证级别**：`CPU 逻辑验证`（实际运行过）。
