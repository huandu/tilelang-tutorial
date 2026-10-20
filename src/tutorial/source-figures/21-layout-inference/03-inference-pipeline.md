# 图 3：推断引擎的 step 0–4 与三级约束

来源：`_vendor/tilelang-v0.1.15/src/transform/layout_inference/layout_inference.cc:538-690`
（`Run`）、`:1396-1550`（`InferInFreeMode`）；三级语义见
`src/op/parallel.cc:311-329`。图为**本书绘制**。

## Mermaid

```mermaid
flowchart TD
    S0["step 0：浮动 fragment → 全复制"] --> S1
    S1["step 1：kStrict<br/>只接受已定义布局"] --> S1b["把结果镜像进 strict_layout_map"]
    S1b --> S2
    S2["step 2：kCommon<br/>工作列表 BFS 传播到不动点"] --> S3
    S3["step 3：kFree<br/>连通用并查集分组 → 逐根重推 → 代价择优"] --> S4
    S4["step 4：按 storage Var 把布局补齐到所有 alias"] --> OUT["LayoutInferenceResult<br/>layout_map / for_map / predicate_map"]
```

## ASCII（工作列表的优先级入队）

```text
  q: [op0, op1, op2, ...]
  某个 op 产出新布局 → 写 layout_map
        │
        ├─ 若下游 op 已接触的 buffer 里已有布局 → push_front（先处理）
        └─ 否则                                 → push_back （后处理）
  循环直到 q 空（不动点）
```

关键实现点：`EnqueueWithPriority`（`layout_inference.cc:762-777`）；
结果以 **annotation** 形式挂回 IR（`:1581-1586` Block / `:1605-1633` For），
真正改写发生在后面的 `LowerTileOp`。
