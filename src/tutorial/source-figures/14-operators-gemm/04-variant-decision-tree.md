# 图 4：GEMM 变体家族的选择判断树

**用途**：正文第 4.5 节。

```mermaid
flowchart TD
    A["要做一个 GEMM"] --> B{"是不是多个 GEMM?"}
    B -- 否 --> C{"输出块数 ≥ 核数?"}
    C -- 是 --> D{"最后一波是否明显不满?"}
    D -- 否 --> E["普通 GEMM<br/>阶梯 ③ / ④"]
    D -- 是 --> F["stream-K"]
    C -- 否 --> G{"形状小且 K 大?"}
    G -- 是 --> H["split-K"]
    G -- 否 --> E
    B -- 是 --> I{"各 GEMM 形状一致?"}
    I -- 是 --> J["batched GEMM"]
    I -- 否 / 动态 --> K{"需要各自独立张量?"}
    K -- 否 --> L["grouped GEMM<br/>拼接 + 偏移表"]
    K -- 是 --> M["grouped GEMM<br/>指针表, 单级流水"]
```

## 四个变体的“账”对照

```text
变体        读入量            写出量              额外开销         解决的问题
普通 GEMM   MNK(1/m+1/n)     MN                ——               并行度足够
split-K     MNK(1/m+1/n)     MNK/k  (×split_k) 原子合并         输出块数 < 核数
stream-K    同普通            同普通 + 被切块    workspace/fixup   尾波量化
grouped     各专家之和        Σ M_g·N           偏移表/间接寻址   形状各异 (MoE)
batched     同普通 × batch    同普通 × batch     ——               形状一致
```

来源：R3（CATLASS 的 MultiCoreSplitK 模板给出读入量不变、写出量
$MNK/k$ 的结论）、S22–S24、S28、A4、R7。
