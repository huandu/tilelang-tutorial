# 第 25 章 习题

难度：`★` 入门 / `★★` 需要读一点源码 / `★★★` 需要动手或综合。
“需要硬件”一栏标明是否必须在有昇腾 NPU 的机器上做；本章绝大多数习题可以在
任意机器上完成。所有动手题都只读 `_vendor/` 与 corpus。

---

### E1 概念题 ★

锁定版本里与昇腾代码生成相关的**四条可执行路线**分别是哪四条？对每一条写出：

1. 目标名（`target=` 或 target kind）；
2. 生成物的形状（CCE/Ascend C？AscendC C++？PTO-DSL C++？MLIR？）；
3. 最终编译器与它收到的关键参数。

*需要硬件：否*

---

### E2 概念题 ★★

`CodeGenTileLangAscend::PrintFuncPrefix`（`_vendor/tilelang-v0.1.15/src/ascend/codegen/codegen_ascend.cc:336-351`）
有三种前缀。请回答：

1. 三种前缀分别对应什么 `kernel_mode_`？它们各自意味着这个 kernel 里有
   Cube 还是 Vector 的工作？
2. `__mix__(1, 2)` 为什么是“1, 2”而不是“1, 1”或“2, 2”？请给出两处依据：
   一处源码注释，一处 corpus 里的硬件规格。
3. 什么决定一个函数用哪种前缀？

*需要硬件：否*

---

### E3 代码阅读题 ★★

下面是从 `_vendor/tilelang-v0.1.15/src/ascend/codegen/codegen_ascend.cc:152-164`
抽出的片段（略去行号）：

```cpp
if (scope == "shared" || scope == "shared.dyn")  return "__ubuf__";
if (scope == "shared.l0a" || scope == "shared.l0a.dyn") return "__ca__";
if (scope == "shared.l0b" || scope == "shared.l0b.dyn") return "__cb__";
if (scope == "shared.l0c" || scope == "shared.l0c.dyn") return "__cc__";
if (scope == "shared.l1"  || scope == "shared.l1.dyn")  return "__cbuf__";
return "";
```

请回答：

1. 把这些 scope 一一对到硬件存储（UB / L1 / L0A / L0B / L0C）。
2. `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py` 里
   `alloc_shared` / `alloc_l1` / `alloc_l0a` / `alloc_l0b` / `alloc_l0c` 的默认
   scope 各是什么？是否都被上面这段覆盖？
3. 为什么这段函数**没有**处理 `"local"` 或 `"local.fragment"`？提示：看看
   `alloc_l0c` 在分配时顺手做了什么（`allocate.py:55-70`）。

*需要硬件：否*

---

### E4 代码阅读题 ★★

给定 kernel 里的四条拷贝（都来自 M 树）：

```python
T.copy(A[m, k*block_K], A_l1)          # (a)
T.copy(A_l1, A_l0a)                     # (b)
T.copy(C_l0c, C_ub)                     # (c)
T.copy(C_ub, C[m:m+block_M, n:n+block_N])  # (d)
```

对每一条写出：

1. `GetDMAPath` 会返回哪个枚举值；
2. 它归属哪条硬件流水线（MTE1 / MTE2 / MTE3 / FIX）；
3. `codegen_ascend.cc` 会发出哪个 intrinsic 名字（有多个变体时说明差别）。

依据：`src/ascend/op/oob_padding.cc:46-68`、`src/ascend/codegen/codegen_ascend.cc:1383-1609`。

*需要硬件：否*

---

### E5 概念题 ★★

`_vendor/tilelang-v0.1.15/src/ascend/op/gemm.cc:23-63` 里，`SelectInst` 与
`ComputeWarpPartition` 都几乎不做事。请回答：

1. `SelectInst` 永远返回什么？这个常量的字面值是什么？
2. `ComputeWarpPartition` 把 `policy.m_warp` / `policy.n_warp` 设成多少？
   为什么昇腾侧不需要 warp 划分？
3. 那么 `T.gemm` 在昇腾上“不一样”的地方体现在哪些参数上？至少写出三个
   （提示：累加、精度模式、块缩放）。

*需要硬件：否*

---

### E6 代码阅读题 ★★

阅读 `_vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py:53-163`
（`dual_copy`），回答：

1. 对 rank ≥ 2 的区域，形状推断规则是什么？`[M, N] → [M/2, N]` 与
   `[M, N] → [M, N/2]` 分别对应哪个 `dual_dst_ctl`？
2. 哪些源/目的组合走**硬件双目的**，哪些走**逐 AIV 的软件拷贝**？
3. 为什么在 `src` 的 scope 是 `shared.l0c` 且 dtype 不同时代码会直接 `raise`？
   如果这里不拦，会发生什么（注释怎么说的）？

*需要硬件：否*

---

### E7 概念题 ★★

`T.SimtVF(threads=N)` 与 `T.SimdVF()` 是两种 Vector 编程模型。请回答：

1. 哪一个有真正的线程域？它在 IR 里的标记是什么（提示：
   `src/ascend/transform/vf_regions.h:38-60`）？
2. `SimdVF` 里的向量类型必须满足什么位宽约束？predicate 必须是什么类型？
   出错时代码是报错还是悄悄降级？
3. 为什么官方 `example_vecadd.py` 的 SIMD 分支里步长是 `tile // 64`？

*需要硬件：否*

---

### E8 动手题 ★★

在书根运行：

```bash
python3 chapters/25-ascend-backend-codegen/code/02-scope-dma-pipes/scope_dma_pipes.py
```

然后：

1. 贴出它打印的“DMAPath → owning pipe”表；
2. 在脚本里新增一条检查：`kL0CToUB` 与 `kL0CToGM` 都必须映射到 `kFixpipe`，
   否则报错；
3. 说明为什么 `kGMToUB` 与 `kGMToL1` 可以共用 MTE2，而
   `kL0CToUB` 必须用 FIX 而不是 MTE3。

*需要硬件：否*

---

### E9 动手题 ★★

运行 `code/03-api-presence-matrix/api_matrix.py`，然后回答：

1. 哪一列里出现了 `alloc_ub`？另一列是什么？
2. `T.tile` 命名空间在哪一列？给出它的注册行。
3. 如果一段教程代码同时用了 `T.alloc_l1` 和 `T.alloc_L1`，你能立刻判断它写错了
   吗？为什么？

*需要硬件：否*

---

### E10 概念题 ★★★

比较 M 树（TIR 内做自动调度）与 L1（AscendNPU IR/MLIR）两条路线在下面四件事上的
**责任划分**：CV 融合、多缓冲、同步插入、内存规划。

要求每一格都给出源码或 corpus 依据（M 树用 `tilelang/ascend/pipeline.py` 与
`src/ascend/transform/`，L1 用 corpus 的 `tilelang-mlir-ascend` 与华为技术文章）。

*需要硬件：否*

---

### E11 代码阅读题 ★★★

下面三行分别来自三份源码，请指出各属于哪条路线，并解释每个参数/开关的作用：

```text
bisheng --npu-arch=dav-2201 -std=c++17 -xasc ... --shared
bisheng --cce-aicore-arch=dav-c220 -D MEMORY_BASE -std=gnu++17 -xcce -I .../3rdparty/pto-isa/include
bishengir-compile kernel.npuir --enable-hivm-compile=true --enable-triton-kernel-compile=true -o kernel.o
```

另外回答：为什么第一条里没有 `--cce-aicore-arch`，第三条里没有 `-xasc`？

*需要硬件：否*

---

### E12 动手题 ★★★

你被要求给一个昇腾团队做技术选型建议。已知：

* 他们的现有算子库全部是手写 Ascend C，跑在 A2；
* 他们想先在**没有 NPU 的开发机**上把新算子写出来、验证功能，再上板；
* 他们不接受“生成物不可读”的方案。

请基于本章的四条路线，给出你的推荐与理由，并明确指出：

1. 哪条路线适合/不适合，依据是什么；
2. 你会先用哪几个现成脚本来验证环境与生成物；
3. 你的建议里有哪些是**必须上板才能确认**的（也就是本章不能替你保证的部分）。

*需要硬件：否（建议本身）；若要坚持某一结论，需要硬件*
