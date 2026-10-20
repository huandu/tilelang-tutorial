# 图 4 · 存储作用域与分配函数

正文位置：第 5 节（buffer 与内存作用域）。

## 4.1 通用侧：四个常用作用域

```mermaid
flowchart TB
    subgraph K["一个 T.Kernel 块的作用域"]
        direction TB
        S["T.alloc_shared(shape, dtype)<br/>scope = shared.dyn<br/>块内所有线程可见；放 tile 的暂存区"]
        F["T.alloc_fragment(shape, dtype)<br/>scope = local.fragment<br/>逻辑上是块形状，物理上切到每线程寄存器"]
        L["T.alloc_local(shape, dtype)<br/>scope = local<br/>每线程私有、普通下标空间"]
        V["T.alloc_var(dtype, init=...)<br/>scope = local.var<br/>单元素标量"]
    end
    G["T.alloc_global(shape, dtype)<br/>scope = global<br/>后端直接分配的工作区（cudaMalloc 风格）"]
    T["T.alloc_tmem(shape, dtype)<br/>scope = shared.tmem<br/>Blackwell 第五代张量核累加器"]
    B["T.alloc_barrier(arrive_count)<br/>scope = shared.barrier"]

    S --> Q1{"这份数据谁看得到？"}
    F --> Q1
    L --> Q1
    V --> Q1
    Q1 -->|"整个块共享"| A1["shared"]
    Q1 -->|"每个线程一份"| A2["fragment / local / var"]
    Q1 -->|"一次 launch 的所有块共享"| A3["global"]
```

### 选择依据（一句话版）

```text
要放一个 tile 的暂存副本，块内共享          → alloc_shared
要放一个「块形状」的累加器 / 中间结果        → alloc_fragment
要一个每线程私有、能自己下标的小数组         → alloc_local
要一个标量（计数器、标志、临时值）           → alloc_var
要一块跨块的工作区（少见）                  → alloc_global
要 Blackwell 的 TMEM 累加器                 → alloc_tmem
```

## 4.2 `fragment` 为什么“看起来像 shared，实际在寄存器里”

```text
你写的：  C_f = T.alloc_fragment((128, 128), "float32")     # 64 KiB 的"逻辑"缓冲

编译器：  在 LayoutInference 里给 C_f 推一个 T.Fragment 布局，
          把 128x128 的迭代空间映射到 128 个线程 + 每线程 128 个 slot

实际：    64 KiB / 128 线程 = 512 B/线程 = 128 个 fp32 寄存器/线程
```

数字来自 `code/03-memory-scopes/scope_footprint.py`（现算，不是实测）。

## 4.3 昇腾侧：UB / L1 / L0A / L0B / L0C

```text
主库 v0.1.15 的 ascend 方言                  适配器仓库 tilelang-ascend-v0.1.2.000
──────────────────────────────────         ─────────────────────────────────────
T.alloc_shared(shape, dtype)               T.alloc_shared(shape, dtype)
  → UB，scope "shared.dyn"                   → UB，scope "shared"（bool → "shared.ub"）
                                             T.alloc_ub(shape, dtype)  ← 只在这里有
                                               → UB，scope "shared.ub"

T.alloc_l1(shape, dtype)  → "shared.l1"    T.alloc_L1(shape, dtype)  → "shared.l1"
T.alloc_l0a(shape, dtype) → "shared.l0a"   T.alloc_L0A(shape, dtype) → "wmma.matrix_a"
T.alloc_l0b(shape, dtype) → "shared.l0b"   T.alloc_L0B(shape, dtype) → "wmma.matrix_b"
T.alloc_l0c(shape, dtype) → "shared.l0c"   T.alloc_L0C(shape, dtype) → "wmma.accumulator"
```

**两处容易踩的点。**

1. 主库 v0.1.15 **没有** `T.alloc_ub`；UB 就是用 `T.alloc_shared` 分配
   （`tilelang/ascend/README.md:138`：`Use T.alloc_shared for Unified Buffer
   storage`）。`alloc_ub` 是适配器仓库的拼法。
2. 两边的 scope 名不一样（`shared.dyn` vs `shared`，`shared.l0a` vs
   `wmma.matrix_a`）。把片段在两边搬来搬去不会自动等价。

## 4.4 片上容量是要算的账

```text
tile 形状 x dtype x 多缓冲份数 = 占用

  (128,256) fp32 x 1 = 128 KiB     放得进示例里的 192 KiB UB
  (128,256) fp32 x 2 = 256 KiB     ❌ 放不下
  (128,256) fp16 x 2 = 128 KiB     ✔
  (128,128) fp16 x 2 =  64 KiB     ✔
```

对应 `code/03-memory-scopes/scope_footprint.py` 的输出。注意第 6 章的提醒：
192 KiB、512 KB 这类数字是**特定型号的示例**，不是昇腾通用规格，真实容量要用
`GetCoreMemSize` 查。

## 事实来源

* `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:49-191, 254-289`
* `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:15-80, 103-125`
* `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/allocate.py:140-157`
* `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:131-151`
* `_vendor/tilelang-v0.1.15/docs/programming_guides/language_basics.md:152-172`

自绘，随本书正文。
