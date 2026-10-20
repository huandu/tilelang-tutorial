# 图 2：从 m16k16 底座组合出 4 warp 的块布局

来源：论文 `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577-fulltext.md:598-608`
（图 6）；实现见 `_vendor/tilelang-v0.1.15/tilelang/layout/fragment.py:117-186`
与 `src/layout/gemm_layouts.cc:45-52`（`MakeGemmFragment8x8`）。

## Mermaid

```mermaid
flowchart LR
    L0["base_layout<br/>一个 warp 消费 m16k16<br/>(16x16)"] -->|repeat([warp_m/16, block_k/16])| L1["warp_layout<br/>一个 warp 消费 m32k16<br/>(32x16)"]
    L1 -->|repeat_on_thread([block_m/warp_m, 1])| L2["block_layout<br/>四个 warp 消费 m128k16<br/>(128x16)"]
    L2 -->|replicate(n_warp)| L3["跨 N 方向复制<br/>覆盖 block_n"]
```

## ASCII（以本书的 8x8 底座为最小单位）

```text
 base: (i, j) -> thread = j//2 + 4*i , slot = j%2        [8x8, 32 线程, 每线程 2 槽]
        │
        │ repeat([1, 1], repeat_on_thread=False)   ← 同一 warp 内重复
        ▼
 warp: 逻辑形状变大，线程号带 warp 维的偏移
        │
        │ repeat([block_s, 1], repeat_on_thread=True)  ← 再来一个 warp
        ▼
 block: 多个 warp 拼成一个 block 布局
```

数值可核对：`../code/01-layout-algebra/layout_algebra.py` 会打印
`MakeGemmFragment8x8: threads=32, slots_per_thread={2}, bijective=True`。
