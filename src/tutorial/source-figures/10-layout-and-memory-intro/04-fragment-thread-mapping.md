# 图 4：fragment 与线程的对应

对应正文第 2.2 节。一个 `T.Fragment` 描述两件事（
`_vendor/tilelang-v0.1.15/tilelang/layout/fragment.py:12-99`）：

1. **线程映射** `forward_thread(i, j, ...)`：元素 `(i, j)` 由哪个线程持有；
2. **局部索引** `forward_index(i, j, ...)`：该元素是这个线程的第几个寄存器槽。

加上可选的 `replicate`（同一份数据在多个线程上各存一份），就构成了完整的 fragment。

## 例子：`MakeGemmFragment8x8`（8×8 → 32 线程 × 2 槽）

源码（`_vendor/tilelang-v0.1.15/src/layout/gemm_layouts.cc:47-52`）：

```cpp
Fragment MakeGemmFragment8x8() {
  IterVar i = MakeIterVar("i", 8);
  IterVar j = MakeIterVar("j", 8);
  IterVar rep = MakeIterVar("rep", 1);
  PrimExpr forward_thread = FloorDiv(j->var, 2) + 4 * i;   // 线程号 = 4*i + j/2
  PrimExpr index          = FloorMod(j->var, 2);           // 槽号   = j % 2
  return Fragment({i, j}, {index}, forward_thread, rep);
}
```

把它逐格画出来（格内标 `T<线程号> L<槽号>`）：

```text
         j=0        j=1        j=2        j=3        j=4        j=5        j=6        j=7
 i=0  T0 L0      T0 L1      T1 L0      T1 L1      T2 L0      T2 L1      T3 L0      T3 L1
 i=1  T4 L0      T4 L1      T5 L0      T5 L1      T6 L0      T6 L1      T7 L0      T7 L1
 i=2  T8 L0      T8 L1      T9 L0      T9 L1     T10 L0     T10 L1     T11 L0     T11 L1
 i=3  T12 L0     T12 L1     T13 L0     T13 L1     T14 L0     T14 L1     T15 L0     T15 L1
 i=4  T16 L0     T16 L1     T17 L0     T17 L1     T18 L0     T18 L1     T19 L0     T19 L1
 i=5  T20 L0     T20 L1     T21 L0     T21 L1     T22 L0     T22 L1     T23 L0     T23 L1
 i=6  T24 L0     T24 L1     T25 L0     T25 L1     T26 L0     T26 L1     T27 L0     T27 L1
 i=7  T28 L0     T28 L1     T29 L0     T29 L1     T30 L0     T30 L1     T31 L0     T31 L1

线程号的取值范围 = 0..31（32 个线程），每个线程恰好拿到 2 个元素（L0、L1）。
64 个逻辑元素 ÷ 32 个线程 = 每线程 2 个槽 —— 与 T.Fragment 的输出形状 [2] 一致。
```

## 关键点

* **逻辑形状 ≠ 每线程持有量。** `(8, 8)` 是逻辑大小；每线程只持有 **2 个**元素。
  这也是为什么 `T.alloc_fragment((block_M, block_N))` 里 `block_M×block_N` 可以远大于
  寄存器总量：它描述的是**整个线程块**的寄存器切片，不是单个线程。
  （见 `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/overview.md`
  对 layout inference 的说明。）
* **布局是编译器推出来的，不是随便选的。** 上面这个形状由 `T.gemm` 选中的 MMA 指令
  决定；换成别的原子指令（wgmma / tcgen05 / MFMA）就换成另一张表。
* **它是可打印的。** `plot_layout(fragment)` 会把每个格子按线程染色并标上 `T`/`L`，
  这正是把“凭感觉的 fragment”变成“看得见的表”的工具（见正文第 3.4 节）。

逐格内容由 `../code/01-layout-algebra/layout_algebra.py` 按上面的公式枚举生成，
并额外校验了它确实是双射（每个线程恰好 2 个槽）。
