# 图 5：昇腾 fractal 存储、ND→NZ 与 OOB padding（入门示意）

对应正文第 2.5 节。这里只建立直觉，推导与 pass 细节留给第 21 章。

## 一、为什么昇腾需要 fractal

NVIDIA 的张量核从 shared memory 取数据时可以配合 `ldmatrix` 做细粒度分块；昇腾的
Cube 单元则期望数据在 L1 中就是**分形（fractal）**摆放——即以 16 行为一组、
宽度为 C0 的小块，C0 由元素位宽决定（bf16/fp16 为 16，fp8 为 32，fp32 为 8）。

逻辑上一块 `[rows, cols]`，物理上被摆成 `[cols/C0, rows/16, 16, C0]`
（`_vendor/tilelang-v0.1.15/tilelang/layout/ascend.py:20-37`）：

```text
逻辑 [rows=32, cols=32] fp16   →   C0 = 32B / 2B = 16

  逻辑视图（行主序）                 物理视图（NZ / MajorK）
  ┌──────────────────────────┐      ┌────────────┬────────────┐
  │ 一行 32 个 fp16 元素      │      │ 16×16 分形 │ 16×16 分形 │
  │ ... 共 32 行 ...          │  →   ├────────────┼────────────┤
  └──────────────────────────┘      │ 16×16 分形 │ 16×16 分形 │
                                    └────────────┴────────────┘
  物理形如 [cols/C0=2, rows/16=2, 16, C0=16]，前两维在外面，
  后两维 (row%16, col%C0) 是分形内部的行主序。

正映射： (r, c) → [ c/C0, r/16, r%16, c%C0 ]
```

不同角色用不同的分形方向（`tilelang/layout/ascend.py:61-92`）：

```text
make_ascend_major_k_layout   逻辑 [MN, K] → 归约轴 K 在外：  [ k/C0, mn/16, mn%16, k%C0 ]
make_ascend_major_mn_layout  逻辑 [K, MN] → 输出轴 MN 在外： [ mn/C0, k/16, k%16, mn%C0 ]
make_ascend_l0c_layout       逻辑 [M, N]  → 固定累加器：    [ n/16, m/16, m%16, n%16 ]
```

`T.alloc_l0c` 在分配时就会自动挂上 L0C 的固定布局
（`_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:55-70`：
`layout=True` 时执行 `make_ascend_l0c_layout`），因为 L0C 的物理摆法是硬约束。

## 二、ND → NZ：把行主序搬进分形

GM / UB 上的张量通常是 ND（row-major，就是普通二维数组）；Cube 需要 NZ。
两者之间必须做一次**重排搬运**，对应 pass `InsertNd2Nz`
（`_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:91`、
`src/ascend/transform/insert_nd2nz.cc`）。

```text
    GM (ND, 行主序)        ── T.copy / ND→NZ 重排 ──▶   L1 (NZ, 分形)
    [M, K] 元素按行连续                                 [K/C0, M/16, 16, C0]
```

重排不是「再存一份」，而是把同一次搬运的目标地址换成 NZ 公式；代价是搬运指令里
多一层索引计算，收益是 Cube 可以直接吃。

## 三、OOB padding：边界不对齐时的补零

分形要求 `rows` 是 16 的倍数、`cols` 是 C0 的倍数。真实 shape（如 M=33、K=100）
往往不满足，于是需要一个 pass 在**逻辑 region 之外**补零，让物理块填满、且越界读
拿到的是 0（`_vendor/tilelang-v0.1.15/src/ascend/transform/insert_oob_padding.cc`；
大纲里记作 `AscendInsertOOBPadding`）。

```text
  M = 33, 分形行粒度为 16

  逻辑:   [ 0 ................ 31 ][ 32 ]
  物理:   [ 0 ................ 31 ][ 32 ][ pad ][ pad ]...[ pad ]   ← 补 0 到 48 行
                                        └────── 15 行 OOB padding ──────┘
```

补零区在正确的计算里不会被写进输出（对应 region 会裁掉），它只是为了让**物理块完整**，
从而能走固定的分形搬运指令。

## 与本章主线的联系

* 昇腾 fractal 是同一件事（逻辑→物理映射）在另一套硬件上的形态；
* 区别在于：NVIDIA 的 fragment 布局是**推导出来的**，昇腾的 fractal 是**硬件规定的**，
  编译器要做的是「把逻辑 shape 归一化 + 补零 + 重排」去迎合它，而不是自由选择；
* 第 21 章会把 `NormalizeAscendFractalStorage`、`InsertNd2Nz`、
  `AscendInsertOOBPadding` 三个 pass 的顺序、约束与转换代价展开讲。

来源：`_vendor/tilelang-v0.1.15/tilelang/layout/ascend.py:20-92`、
`src/ascend/layout/ascend_layouts.h:1-70`、`src/ascend/transform/{insert_nd2nz,insert_oob_padding,normalize_fractal_storage}.cc`；
昇腾侧的 pass 顺序见 `../tilelang-corpus/90-synthesis/02-compiler-internals-map.md`
（本书 corpus 综合文档）。
