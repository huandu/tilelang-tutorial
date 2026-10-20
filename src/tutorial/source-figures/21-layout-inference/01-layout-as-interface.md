# 图 1：三类约束汇聚到一个 buffer 的物理布局

来源：`_vendor/tilelang-v0.1.15/tilelang/cuda/op/gemm/gemm_mma.py:42-66`
（mma fragment / swizzle 约束）；`src/layout/gemm_layouts.cc:442-530`（swizzle 档位）；
`src/ascend/layout/ascend_layouts.h:20-27`（分形对齐）。图的结论是**本书推导**。

## Mermaid

```mermaid
flowchart TD
    A["T.copy 写入 A_shared<br/>自然顺序 / 向量化"] --> B{"A_shared 的物理布局<br/>L_A 是未知量"}
    C["T.gemm 读取 A_shared<br/>要求 swizzle（消 bank conflict）"] --> B
    D["T.gemm 累加器 C_local<br/>要求某条 mma 的 fragment"] --> E{"C_local 的物理布局<br/>L_C 是未知量"}
    F["T.copy 回写 C 到 global<br/>希望访存合并"] --> E
    G["昇腾：Cube 输入必须是<br/>16 x C0 的分形排布"] --> B
    B -->|冲突时| H["包含升级 / swizzle 合并<br/>换根重推 / 插入转换"]
    E -->|冲突时| H
    H --> I["一组可行布局<br/>(或编译期硬失败)"]
```

## ASCII

```text
  tile op 1 (T.copy)            tile op 2 (T.gemm)          tile op 3 (T.Parallel)
        │                             │                            │
        │ 约束：搬运顺序              │ 约束：swizzle / fragment   │ 约束：循环可寻址
        ▼                             ▼                            ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │                  求解：给每个 buffer 选一个 layout                    │
   │      冲突时：包含检查 → swizzle 合并 → 换根重推 → 否则硬失败           │
   └──────────────────────────────────────────────────────────────────────┘
```
