# 第 09 章 核心原语详解

## 学习目标

读完本章，你应当能够：

1. 说清 TileLang 的“原语（primitive）”到底是什么：它不是普通的 Python 函数，而是
   **一个写进 TIR 的 tile op 或 extrinsic**，函数体只负责参数校验与 IR 组装，真正的语义
   与硬件选择发生在 lowering/codegen 阶段。
2. 对 `T.copy` 写出“源/目标作用域 → 指令形态”的对照：什么时候是普通 SIMT
   copy、什么时候是 `cp.async`、什么时候是 TMA（`cp.async.bulk.tensor`）、什么时候是
   `ldmatrix`/`stmatrix`/TMEM 搬运，以及在昇腾上落到 L1/L0/UB/GM 的哪条 DMA 路径。
3. 解释 `T.copy` 的**越界与 padding 语义**：谓词从哪里来、默认填什么值、怎么用
   `T.annotate_safe_value` 改掉它、TMA 的 OOB 填充与 SIMT 路径的差别。
4. 解释 `T.gemm` 的 M/N/K 形状约束、`transpose_A`/`transpose_B` 的含义、
   `policy` 如何决定 warp 划分、`clear_accum` 为什么会影响 IR 里的读写标记，以及
   `accum_dtype` 在 v0.1.15 里由谁决定。
5. 解释 `T.reduce_sum` / `T.reduce_max` / `T.reduce_min` 的 `clear` 语义，以及
   “线程内归约 + 跨线程 butterfly 归约”这条分工线，知道
   `tl::AllReduce<...>` 在什么宽度上走 shared memory、什么宽度上走 shuffle。
6. 说明原子操作（`T.atomic_add` 等）的两种形态：标量地址路径与 tile-region 路径，
   各自的限制（`return_prev`、`memory_order`、`use_tma`）。
7. 说明 `T.fill` / `T.clear` 的实现是一条并行循环，而不是特殊的硬件指令。
8. 分清“v0.1.15 里就有的原语”与“只在昇腾适配器仓库里的 `T.tile.*`”，
   并能读懂 `T.dual_copy`、`T.SimtVF`、`T.SimdVF` 的适用条件。

## 前置知识

* 第 03 章：tile / dataflow / schedule 解耦，三层接口（beginner / developer / expert）。
* 第 04 章：一次编译的全流程，Python 前端 → TIR → pass → codegen。
* 第 06 章：NV 与昇腾的存储层级与执行单元对照，本章只补“原语怎么落到这些层级”。
* 第 08 章：buffer 与内存作用域（`global` / `shared` / `local.fragment`）、
  `T.Kernel` / `T.Parallel` / `T.Pipelined` 的基本写法。

本章的代码示例分三类：能在 CPU 上跑的纯 Python 模型、只做静态核对的 TileLang
骨架、以及昇腾骨架。**没有 NVIDIA GPU，也没有昇腾 NPU**（见
[`../../VERSIONS.md`](../../VERSIONS.md)），因此本章不含任何本机实测结果。

---

## 1. 问题的由来：原语是“接口”，不是“实现”

### 1.1 同一个 `T.copy`，可以变成四种东西

先看官方文档对 `T.copy` 说的一句关键话（`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/instructions.md`，第 40–52 行）：

> `T.copy(src, dst, ...)`（同步语义）
> - 编译器可以把它降成不同的机制（同步 SIMT 拷贝 `ld.global`、warp 级 `ldmatrix`、
>   TMA 异步拷贝 `cp.async.bulk`、老的异步拷贝 `cp.async` 等），取决于 target 与
>   提示；但**可观测语义是同步的**：语句之后使用 `dst` 是安全的。

对用过 PyTorch 的人，这句话的第一反应可能是“那就当 `dst.copy_(src)` 用”。这个直觉
在结果上是错的，不是结果值错，而是**成本模型错**。同一行 `T.copy(A[...], A_shared)`：

* 在 Hopper 上、形状对齐、目标 shared buffer 加了 swizzle。它可能是**一条 TMA 指令**，
  由单个线程发起，搬运整块 tile，不占用计算线程的寄存器带宽；
* 同一行，如果目标维度不是 16 字节对齐，或显式写了 `disable_tma=True`。它回退成
  一个 SIMT 并行循环，每个线程用 `ld.global` 搬几个元素；
* 如果反过来是 `shared → fragment`。它可能是 `ldmatrix`；
* 在昇腾上同一行。它是 `DataCopy` / `ND2NZ` / `LoadData` / `Fixpipe` 中的某一条，
  由源和目标的 scope 决定。

所以“原语（primitive）”这个词在这里要有比“库函数”更强的含义：**原语是用户意图与
硬件能力之间的一个契约**。你写的是意图（把这片刻搬过去、把这两个 tile 乘起来、
沿这个轴求和），编译器负责选择实现。你越了解契约的边界，就越能预测它选了哪条实现。

### 1.2 原语的三个层次

第 03 章讲过 TileLang 有 beginner / developer / expert 三层接口。落到原语上：

| 层次 | 典型写法 | 你放弃的控制 | 你换到的东西 |
| --- | --- | --- | --- |
| 高层 tile 原语 | `T.copy`、`T.gemm`、`T.reduce_sum`、`T.fill` | 具体指令、layout、同步 | 可移植性；编译器自动做 layout 推导与同步插入 |
| 带提示的 tile 原语 | `T.copy(..., prefer_instruction="tma", eviction_policy=...)`、`T.gemm(..., policy=...)` | 只剩细节 | 用注解影响选择，语义不变 |
| 线程/硬件原语 | `T.async_copy` + `T.ptx_wait_group`、`T.mbarrier_wait_parity`、昇腾 `T.set_atomic` | 同步正确性要自己负责 | 手工流水、手工同步能达到的极限 |

本章主要讲第一、二层，并在需要解释机制时下探到第三层。**本章不教你怎么调优**，调优
要看第 11、12 章；本章只回答“每个原语到底承诺了什么、编译器把它变成了什么”。

---

## 2. 概念与模型

### 2.1 一块原语的解剖

以 `T.copy` 为例，从你写下它到它变成机器码，中间只有三步（源码位置见下表）：

```text
T.copy(A[...], A_shared)                    # ① 前端：校验 + 组装 IR
   → call_intrin("handle", "tl.tileop.copy", src_region, dst_region, annotations={...})
   → CopyNode{src, dst, src_range, dst_range, annotations}      # ② IR 节点
   → SelectInst(...) 选出 CopyInst                              # ③ 后端：选择指令
   → 一条 SIMT 循环 / cp.async / TMA / ldmatrix / Ascend DMA
```

三条事实值得单独记下来：

1. **前端大部分时候只做“把参数变成 region”**。`T.copy` 的前端函数
   （`_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:59-130`）做完 extent 归一化后，
   只有一行真正发射 IR：`call_intrin("handle", Op.get("tl.tileop.copy"), src, dst, ...)`
   （同文件 `:130`）。
2. **有一个例外：标量 copy 直接退化成 `BufferStore`**。当两边都是没有 extent 的
   `BufferLoad` 时，前端直接生成 `dst[...] = src[...]`（同文件 `:121-128`）。这就是
   “tile op 只在需要成块搬运时出现”。
3. **后端只认 op 名字和注解，不认 Python**。CUDA 侧的指令选择在
   `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:716` 的
   `SelectCopyInstForLowering`；昇腾侧在 `src/ascend/op/copy.cc`。它们读的都是
   `annotations` 里那几个键（`disable_tma`、`prefer_instruction`、`l2_cache_ctrl`…），
   而不是函数签名。

### 2.2 原语 → IR → 目标代码 映射表

这张表是本章的骨架。左三列在 v0.1.15 源码里逐条核对过；右两列是**指令类别**，
不是实测结果。

| 前端原语 | TIR（tile op / intrinsic） | NV 侧目标代码 | 昇腾侧目标代码 | 关键源码位置 |
| --- | --- | --- | --- | --- |
| `T.copy` | `tl.tileop.copy` | SIMT 循环 / `ldmatrix` / `stmatrix` / `cp.async` / TMA `cp.async.bulk.tensor` / `tcgen05.ld`·`st` | `DataCopy`、`ND2NZ`、`LoadData`(L1→L0)、`Fixpipe`(L0C→GM) | `src/op/copy.cc:615`；`src/cuda/op/copy.h:21-33`；`src/ascend/op/copy.cc` |
| `T.async_copy` | `tl.tileop.async_copy` | `ptx_cp_async` + `commit_group`（**不插 wait**） | 不支持（昇腾侧无对应 op） | `tilelang/language/copy_op.py:188-229`；`src/op/copy.cc:622` |
| `T.tma_copy` | `tl.tileop.tma_copy` | `expect_tx` + `tma_load`（load）/ `tma_store`+`arrive`（无 wait） | 不支持 | `tilelang/language/copy_op.py:232-324`；`src/op/copy.cc:640` |
| `T.copy_cluster` | `tl.tileop.copy` + `dst_block`/`cluster_mask` | TMA multicast / SM→SM 共享内存拷贝 | 不支持 | `tilelang/language/copy_op.py:133-185` |
| `T.gemm` | `tl.tileop.gemm` | `cuda.mma` / `cuda.wgmma` / `cuda.tcgen05` / `cuda.fma` | `ascend.mad`（L0A/L0B→L0C） | `src/op/gemm.cc:275`；`src/cuda/op/gemm.cc:34-38,336`；`src/ascend/op/gemm.cc:24` |
| `T.gemm_blockscaled` | `tl.tileop.gemm_blockscaled` | `tcgen05` 块缩放变体 | MXFP8 融合 L1 路径 | `src/op/gemm_blockscaled.cc:105`；`tilelang/ascend/language/gemm_op.py:110-165` |
| `T.reduce_sum` / `max` / `min` | `tl.tileop.reduce` | 线程内串行归约 + `tl::AllReduce<Reducer,threads,scale,Barrier>::run` | `ascend_reduce` 系列（SimdVF 内直接对 UB 归约） | `src/op/reduce.cc:245`；`src/tl_templates/cuda/reduce.h:240`；`tilelang/ascend/language/reduce_op.py:43-88` |
| `T.atomic_add` | 标量：`tl.atomic_add_elem_op`；tile：`tl.tileop.atomicadd` | `AtomicAdd` / `AtomicAddx2` / `AtomicAddx4` / TMA `cp.reduce` | `SetAtomicAdd<T>()` + GM 写回（经 `T.set_atomic` / `T.tile.atomic_add`） | `tilelang/language/atomic.py:211-311`；`src/cuda/op/atomic_add.cc:36-62`；`src/ascend/op/builtin.cc:446` |
| `T.atomic_max` / `T.atomic_min` | `tl.atomic_max_elem_op` / `tl.tileop.atomicmax`（min 同构） | `AtomicMax` / `AtomicMin` | `SetAtomicMax` / `SetAtomicMin` | `tilelang/language/atomic.py:39-208`；`src/op/atomic_reduce.cc:162,167` |
| `T.atomic_load` / `T.atomic_store` | `tl.atomic_load_elem_op` / `tl.atomic_store_elem_op` | 带 memory-order 的 load/store | 无对应公开原语 | `tilelang/language/atomic.py:390-492` |
| `T.fill` / `T.clear` | `tl.tileop.fill` | 并行循环 + 向量化 `BufferStore` | 并行循环 → `Fill`/`Duplicate` 向量指令 | `tilelang/language/fill_op.py:11-72`；`src/op/fill.cc:94-127,214` |
| `T.transpose` | `tl.tileop.transpose` | SIMT 循环（无专用 layout 推导） | `T.tile.transpose` → `ascend_transpose` | `tilelang/language/copy_op.py:525-562`；`src/op/transpose.cc:222` |
| `T.tile.add`／`mul`／`div`／`select`／`gather`／`sort`／`topk`… | `tl.ascend_*` intrinsic | —（昇腾专属） | AscendC 向量内联函数 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend_tile.py`；`_vendor/tilelang-ascend-v0.1.2.000/src/op/ascend.cc:975-1260` |
| `T.dual_copy` | `tl.tileop.ascend_copy` + `dual_dst_ctl` | —（昇腾专属） | L0C→UB 硬件双目标搬运；GM↔UB、UB→L1 退化为按 `cthread` 的一路拷贝 | `tilelang/ascend/language/copy_op.py:53-163`；`src/ascend/op/copy.cc:573-605` |
| `T.SimtVF` / `T.SimdVF` | `SIMT_VF` / `SIMD_VF` block | —（昇腾专属） | `SimtVF` 走 `asc_vf_call` 线程体；`SimdVF` 生成 MicroAPI 向量指令 | `tilelang/ascend/language/frame.py:107-143,179-195`；`src/ascend/codegen/codegen_ascend.cc:401-403,904-940` |

> **一句话读法**：左列是你能写的东西，第二列是编译器内部记得的东西，第三、四列是它
> 最终可能变成的东西。“可能”是重点，精确的映射由 target + layout + 注解共同决定。

### 2.3 存储作用域是原语的坐标系

所有原语的合法性都可以用一个问题筛掉一半：“源和目标各自在哪个 scope？”v0.1.15 里
通用的 scope 名有 `global`、`shared` / `shared.dyn`、`local.fragment`、`local`；昇腾
再细化出 `shared.l1`、`shared.ub`、`shared.l0a`、`shared.l0b`、`shared.l0c`、
`shared.tmem`（NV 侧 TMEM）等。

两条推论：

* `T.gemm` 的合法性与性能**完全由 A/B/C 的 scope 组合决定**。昇腾侧 `_dialect_gemm_call`
  只有在 `A∈L0A, B∈L0B, C∈L0C` 时才走专门的 L0 MAD 组装
  （`tilelang/ascend/language/gemm_op.py:94-95`）；CUDA 侧 `AllowTcgen5Mma` 要求
  A 在 shared 或 TMEM、B 在 shared、**C 在 TMEM**
  （`src/cuda/op/gemm.cc:80-87`）。C 放在 fragment 上就永远不会选到 `tcgen05`。
* `T.copy` 的指令选择本质是“scope 对 + 对齐 + dtype 对”三件事的组合
  （见 §4.1）。昇腾的 `T.copy` 前端甚至先检查两个整 buffer 的**元素总数**是否相等，
  然后才允许形状不同（为了 ND↔NZ 与转置路径）
  （`tilelang/ascend/language/copy_op.py:296-308`）。

### 2.4 三类原语，三种“必须自己盯住的东西”

把本章的原语按职责分三类，会更容易记住各自的坑：

| 类别 | 原语 | 它真正在动的资源 | 必须自己盯住的东西 |
| --- | --- | --- | --- |
| 搬运类 | `T.copy`、`T.async_copy`、`T.tma_copy`、`T.copy_cluster`、`T.dual_copy` | 存储层级之间的带宽与延迟（HBM↔SMEM、GM↔UB/L1/L0） | **边界与对齐**：越界怎么填、行宽够不够 32B / 128bit、swizzle 与指令是否匹配 |
| 计算类 | `T.gemm`、`T.reduce_*`、`T.tile.*`、`T.simd.*` | 计算单元（Tensor Core / Cube / Vector）与寄存器 | **scope 组合与形状约束**：M/N/K 必须静态、归约宽度必须是 2 的幂、L0C 才能接 dual_copy |
| 元操作类 | `T.fill`、`T.clear`、`T.transpose`、`T.atomic_*` | 没有专属硬件，落在通用并行/向量指令上 | **语义是否被误读**：`clear=False` 是“合并”不是“清零”、`set_atomic` 是有状态的、`return_prev` 只在标量路径可用 |

这张分类不是学术划分，而是一条排障顺序：**搬运类的问题通常表现为“数值对但很慢”，
计算类的问题通常表现为编译期报错，元操作类的问题通常表现为“数值悄悄不对”**。
本章 §5.2 的错误表按同一顺序排列。

---

## 3. 用法：每个原语的最小示例

下面每个片段都是**最小可读**版本，不含完整 host 代码。完整骨架在 `code/` 与
`examples/` 里，来源逐条写在 `code/*/README.md`。

### 3.1 `T.copy`

```python
with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_K), "float16")
    A_local  = T.alloc_fragment((block_M, block_K), "float16")

    T.copy(A[by * block_M, 0], A_shared)          # global → shared：整块 tile
    T.copy(A_shared, A_local)                     # shared → fragment
    T.copy(A_local, C[by * block_M, bx * block_N])# fragment → global：写回
```

要点：

* **extent 从哪来**：`Buffer` 用它的 shape，`BufferRegion` 用每个维度的 extent，
  `BufferLoad`（`A[i, j]` 这种“起始地址”写法）用推断出的区域。缺 extent 的一侧按
  另一侧的秩补齐为 1（`tilelang/language/copy_op.py:38-46`）。这只是有限的语法糖，
  **不是通用广播**，同一份 docstring 明确写了“we don't support something like
  broadcast”（同文件 `:99-102`）。
* **两个整 buffer 直接对拷时形状必须相等**（同文件 `:24-25`）。
* **一维切片要写清楚**：`T.copy(A[begin:begin+tile], a)`（`examples/ascend/example_vecadd.py`）。
* **同步语义**：`T.copy` 是同步的。降级成 `cp.async` 时，编译器会自己补
  `commit`/`wait`（官方文档 `instructions.md:46-52`）。
* **dtype 不同会自动插转换**：SIMT 路径里 `if (src->dtype != dst->dtype) value =
  Cast(dst->dtype, value)`（`src/op/copy.cc:508-509`）。这不是免费操作。它会让
  向量化宽度变小，也可能让某些后端直接拒收。写 `fp32 → fp16` 的整块搬运前先想清楚
  为什么需要它。
* **`coalesced_width` 只影响 SIMT 循环的向量化宽度**，它被贴在 `T.copy` 生成的最外层
  并行循环上（`src/op/copy.cc:529-540`）。如果这次 copy 最终被降级成 TMA/LDSM，
  它根本没有作用点。
* **`loop_layout` 是给 SIMT 循环的 loop layout 提示**，与 TMA/LDSM/STSM/TMEM 互斥：
  一旦选到那些指令形态，前端会直接 fatal（`src/cuda/op/copy.cc:549-562`）。
  它的用途是让这次搬运的线程映射与某个 fragment layout 对齐，而不是“调优开关”。
* **`eviction_policy` 只影响缓存行为**：`evict_normal` / `evict_first` / `evict_last`
  被映射成整数 id 挂在注解上（`tilelang/language/copy_op.py:54-56`），能读懂它的只有
  CUDA 的 TMA / load-store 降级路径；其它 target 会忽略它。

### 3.2 `T.async_copy` / `T.tma_copy`

当你需要手工流水线时，用显式异步版本：

```python
T.async_copy(A[by * block_M, ko * block_K], A_shared)   # 发射 cp.async + commit
# ... 做别的事 ...
T.ptx_wait_group(0)                                     # 自己等
```

`T.async_copy` 的 docstring 写得很直白：它**只**发射 `ptx_cp_async(...)` +
`ptx_commit_group()`，不等（`tilelang/language/copy_op.py:196-202`）。如果约束不满足
（不是 global→shared，或 dtype 不同），它**直接编译失败**，不会静默退回同步拷贝
（`src/cuda/op/copy_analysis.cc:560-575`）。

`T.tma_copy` 是 TMA 的“半个序列”：load 只发 `expect_tx + tma_load`，store 只发
`tma_store + arrive`，不收尾（`tilelang/language/copy_op.py:242-263`）。它要求你显式
传 `barrier` 并自己 `T.mbarrier_wait_parity(...)`。

### 3.3 `T.gemm`

```python
with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=256) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_K), "float16")
    B_shared = T.alloc_shared((block_N, block_K), "float16")   # 注意是 [N, K]
    C_local  = T.alloc_fragment((block_M, block_N), "float32")

    T.clear(C_local)
    for ko in T.Pipelined(K // block_K, num_stages=3):
        T.copy(A[by * block_M, ko * block_K], A_shared)
        T.copy(B[bx * block_N, ko * block_K], B_shared)
        T.gemm(A_shared, B_shared, C_local, transpose_B=True,
               policy=T.GemmWarpPolicy.Square, clear_accum=(ko == 0))
    T.copy(C_local, C[by * block_M, bx * block_N])
```

* M/N/K 从 A/B/C 的**最后两维**推出来（`tilelang/language/gemm_op.py:77-88`），并且
  **必须是静态常数**，否则前端直接报错（同文件 `:90-92`）。
* `transpose_A/B` 只改变“哪个维度是 M/K”的解释：`transpose_A=False` 时
  A 是 `[M, K]`，`True` 时是 `[K, M]`；`transpose_B=False` 时 B 是 `[K, N]`，
  `True` 时是 `[N, K]`（同文件 `:78-81`）。
* `policy` 是 warp 划分策略，取值 `Square` / `FullRow` / `FullCol`
  （`tilelang/tileop/base.py:5-12`），默认 `Square`。它对 layout 的影响见 §4.3。
* `clear_accum=True` 除了语义上“先清零再累加”，还会把 C 的 **access region 从 `rw`
  改成 `w`**（`tilelang/language/gemm_op.py:103`）。这会喂给依赖分析与同步插入。
  所以它不是一个纯运行期开关。
* **`accum_dtype` 不是 v0.1.15 `T.gemm` 的参数**。累加精度由 C 的 dtype 决定，而 C 的
  dtype 又是指令选择的一等输入（`src/cuda/op/gemm.cc:44-76` 里 `CheckWgmma` 第一件事就
  是看 `op.c_->dtype`）。写 `T.gemm(..., accum_dtype="float32")` 在 v0.1.15 会直接
  报参数错误。

### 3.4 `T.reduce_sum` / `T.reduce_max` / `T.reduce_min`

```python
A_local  = T.alloc_fragment((block_M, block_N), "float32")
row_sum  = T.alloc_fragment((block_M,), "float32")

T.reduce_sum(A_local, row_sum, dim=1)               # out[M] = sum(A_local, axis=1)
T.reduce_max(A_local, row_max, dim=1, clear=True)   # clear 默认就是 True
T.reduce_min(A_local, row_min, dim=-1, clear=False) # 在 out 已有值之上“合并”
```

* `dim` 支持负数，前端先做 `dim += rank` 归一化
  （`tilelang/language/reduce_op.py:13-16`）。
* 输出 shape 必须严格等于“去掉被归约轴”或“被归约轴变成 1”二者之一
  （同文件 `:63-71`）。
* `clear=True`（默认）表示**先初始化再写结果**：sum 从 0 起、max 从 `-inf` 起、
  min 从 `+inf` 起（`reduce_max`/`reduce_min`/`reduce_sum` 的 docstring，同文件
  `:176-259`）。
* `clear=False` 表示**在已有 `out` 上再合并一次**：`out = out ⊕ reduce(src)`。
  这不是实现细节，前端就会走不同分支（同文件 `:141-158`）。
* 昇腾适配器版本额外支持 `real_shape=` 与 `tmp=` 两个关键字；v0.1.15 的
  `T.reduce_*` **没有**这两个参数（见 §5.1）。

### 3.5 原子操作

```python
# ① 标量地址路径：dst 是一个 [1] 的全局 buffer / 一个地址
counter = T.Tensor([1], "int32", name="counter")
T.atomic_add(counter, 1)                       # 线程级原子加
prev = T.atomic_add(counter, 5, return_prev=True)   # 标量路径才支持 return_prev

# ② tile-region 路径：整块 tensor 原子累加
T.atomic_add(C_global[by * block_M, bx * block_N], C_local)  # 部分和累加回全局

# ③ 别的算子
T.atomic_max(max_buf, value)
T.atomic_min(min_buf, value)
T.atomic_store(flag, 1, memory_order="release")
flag_val = T.atomic_load(flag, memory_order="acquire")
```

关键分叉在 `get_extent()` 能不能推出区域：推不出就是标量路径，推得出就是 tile-region
路径（`tilelang/language/atomic.py:264-311`）。两条路径的能力不一样：

* `return_prev=True` **只**在标量路径可用；tile-region 路径直接
  `NotImplementedError`（同文件 `:304-305`）。
* `memory_order` 在 CUDA 上两条路径都生效；HIP 与 CuTeDSL 目前忽略它（同文件
  `:221`）。取值集合是 `relaxed/consume/acquire/release/acq_rel/seq_cst`，映射成整数
  id 传下去（同文件 `:13-23`）。
* CUDA 侧还有 `use_tma=True`，把 tile-region 的原子加降成 sm90+ 的 TMA `cp.reduce`
  （`tilelang/cuda/language/atomic.py:12-50`）。它只支持
  `float16/bfloat16/float32/int32/uint32/uint64`（`src/cuda/op/atomic_add.cc:48-62`）。

### 3.6 `T.fill` / `T.clear`

```python
C_local = T.alloc_fragment((block_M, block_N), "float32")
T.fill(C_local, 0.0)          # 填一个值
T.clear(C_local)              # 等价于 T.fill(C_local, 0)
T.fill(C_local[:, 0:64], -1.0)  # 也可以只填一个 region
```

`T.clear(buffer)` 的定义就是 `fill(buffer, 0)`（`tilelang/language/fill_op.py:48-72`）。
它接受 `Buffer`、`BufferRegion`、`BufferLoad`；`BufferLoad` 会先被还原成 region
（同文件 `:27-38`）。适配器仓库的语言参考甚至直接写“`T.clear` `T.fill`：nothing
special，它们会被转成 `T.Parallel`”
（`_vendor/tilelang-ascend-v0.1.2.000/docs/language_ref/tilelibrary.md:206-209`）。
这条在 v0.1.15 源码里成立：`FillNode::MakeSIMTLoop` 就是生成嵌套 `ForKind::kParallel`
循环（`src/op/fill.cc:152-172`）。

### 3.7 昇腾 tile library：`T.tile.*`

**先纠正一个常见误解**：`T.tile.*` 不在 TileLang 主干 v0.1.15 里。它是昇腾适配器仓库
`tilelang-ascend` 的 `tilelang/language/ascend_tile.py`，通过
`from . import ascend_tile as tile` 挂到 `T.tile` 命名空间
（`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/__init__.py:86`）。在 v0.1.15 的
`tilelang/language/` 与 `tilelang/ascend/language/` 下**没有** `T.tile` 这个名字空间
（核验方式见 `references.md` 的 S20）。

它提供的是一批**显式向量内联函数**，用途是在 UB 上手写计算，绕开
`T.Parallel` 的自动映射：

```python
with T.SimdVF():                       # 或 with T.SimtVF(threads=...)
    T.tile.add(c_ub, a_ub, b_ub)             # dst = src0 + src1
    T.tile.mul(c_ub, a_ub, 2.0)              # dst = src0 * scalar
    T.tile.select(dst, selMask, a, b, "VSEL_CMPMASK_SPR")
    T.tile.compare(dst, a_ub, 1.0, "GE")     # 逐元素比较，模式 EQ/NE/GT/GE/LT/LE
    T.tile.transpose(dst_ub, src_ub)         # 2D 转置 [H,W] → [W,H]
    T.tile.sort(dst_ub, src_ub, actual_num)  # 降序，输出 (值, 索引) 交错对
    T.tile.topk(dst_ub, src_ub, K, actual_num)
    T.tile.gather(dst, src, src_offset, src_base_addr)  # 按字节偏移收集
    T.tile.broadcast(dst_ub, src_ub, axis=1)
```

分类与语义（源码位置 `tilelang/language/ascend_tile.py`，行号见括号）：

| 类别 | API | 语义要点 |
| --- | --- | --- |
| 双目算术 | `add`(1026)、`sub`(1037)、`mul`(1048)、`div`(1059)、`max`(1070)、`min`(1081)、`bitwise_and`(1092)、`bitwise_or`(1103) | `dst = op(src0, src1)`，`src1` 可以是 buffer 也可以是标量 |
| 单目 | `exp`(1140)、`ln`(1211)、`abs`(1221)、`reciprocal`(1231)、`sqrt`(1241)、`rsqrt`(1251)、`relu`(1261)、`bitwise_not`(1271)、`sin`(1976)、`cos`(2012) | 逐元素 |
| 标量/复合 | `leaky_relu`(1313)、`axpy`(1326)、`bitwise_lshift`(1398)、`bitwise_rshift`(1432)、`clamp`(2228)、`round`(2277) | 多带一个标量参数 |
| 比较/选择 | `compare`(1837)、`select`(707) | `compare` 输出 mask；`select` 按 **bit-packed mask** 选 `src0`/`src1` |
| 排序类 | `sort32`(1572)、`sort`(412)、`merge_sort`(452)、`topk`(550)、`init_sort_buf`(857) | 输出 `[val0, idx0, val1, idx1, …]` 交错对，按降序；不足部分用 `-inf` 填充 |
| 索引类 | `gather`(1661)、`gather_mask`(608)、`gatherb`(669)、`createvecindex`(1594)、`arith_progression`(389) | `gather` 的偏移量单位是**字节**，不是元素下标 |
| 数据整形 | `transpose`(1616)、`broadcast`(2302)、`brcb`(884)、`row_expand_*`(2410 起) | `transpose` 只接受整块 2D buffer |
| 写回 | `atomic_add`(313)、`fill`(343)、`clear`(370) | `atomic_add` 的 V1 模型：目标必须是 GM，源必须是本地 tensor |

三类必须记住的约束（来自 `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/api_docs/`）：

* **`sort`/`topk` 会修改源**。float32 输入时，`actual_num` 到 `aligned_count` 之间的位置
  会被**原地**填 `-inf`；float16 输入则不改源（内部转 float32 到临时 buffer）
  （`T.tile.sort.md`）。
* **`gather` 用字节偏移**：`dst[i] = src[(src_base_addr + src_offset[i]) / sizeof(dtype)]`
  （`T.tile.gather.md`）。
* **`select` 的 mask 是 bit-packed**：`selMask` 是 `uint8` 的位串，第 i 位选
  `src0[i]` 还是 `src1[i]`（`T.tile.select.md`）。

另外，适配器把这些 API 的临时 workspace 统一成一个仅关键字参数 `tmp`：不传就由编译器
分配，传了就是“这次调用完整的 target-specific 字节 arena”，其 dtype 只决定字节几何，
不表示数值类型（`_vendor/tilelang-ascend-v0.1.2.000/docs/language_ref/tilelibrary.md:64-100`）。

### 3.8 `T.dual_copy`（昇腾专有）

```python
with T.Kernel(num_blocks) as bx:
    C_l0c = T.alloc_l0c((block_M, block_N), "float32")
    C_ub  = T.alloc_shared((block_M // 2, block_N), "float32")

    T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=(k == 0))
    T.dual_copy(C_l0c, C_ub)                                   # L0C → UB，两个 AIV 各拿一半
    with T.SimtVF(threads=128):
        for i, j in T.Parallel(block_M // 2, block_N):
            C_ub[i, j] = T.max(C_ub[i, j], 0)
    T.dual_copy(C_ub, C[m:m + block_M, n:n + block_N], l2_cache_ctrl="NOTALLOC_PW")
```

（骨架取自 `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:60-115`。）

它的语义是“**一次搬运，两个 AIV 各处理半边**”：

* 划分方向由 shape 自动推断：`[M,N] ↔ [M/2,N]` 是 M 分裂（`dual_dst_ctl=0b01`），
  `[M,N] ↔ [M,N/2]` 是 N 分裂（`0b10`）。**有且只能有一个**尾部维度满足 2:1
  （`tilelang/ascend/language/copy_op.py:113-134`）。
* 支持的路径只有四条：L0C→UB（硬件双目标）、GM→UB、UB→GM、UB→L1（后三条退化为
  按 `cthread` 索引的普通 per-AIV 拷贝）（同文件 `:74-81`）。
* 硬件 L0C→UB 要求 rank ≥ 2；N 分裂还要求完整 N 是 32 的倍数（同文件 `:86-88`）。
* **它不是一个可以把纯 Vector kernel 变成双 AIV 的技巧**：用户必须同时提供 Cube 侧的
  工作，编译器不会替你合成（同文件 `:62-66`）。
* dav-950 上 `L0C→UB` 的 fixpipe 不支持类型转换；`src`/`dst` dtype 不同会直接报错
  （同文件 `:141-151`）。

### 3.9 `T.SimtVF` / `T.SimdVF`（昇腾专有）

```python
with T.SimtVF(threads=128):                      # SIMT 区域，有自己的线程域
    for i in T.Parallel(tile):
        c[i] = a[i] + b[i]

with T.SimdVF():                                 # SIMD 区域，没有线程域
    mask = T.simd.pset(32)
    for i in range(tile // 64):
        a_reg = T.simd.vld(a[i * 64])
        b_reg = T.simd.vld(b[i * 64])
        T.simd.vsts(c[i * 64], T.simd.vadd(a_reg, b_reg, mask), mask)
```

（改编自 `examples/ascend/example_vecadd.py`。）

* `T.SimtVF(threads=...)` 的 `threads` 可以是 int、list 或 tuple，**最多三维**；缺的维度
  补 1（`tilelang/ascend/language/frame.py:107-143`）。
* `T.SimdVF()` **没有线程维度**，是寄存器级 SIMD 区域，只能用 `T.simd.*`
  显式向量指令（同文件 `:192-206`；`T.simd` 命名空间见
  `tilelang/ascend/language/__init__.py:67`）。
* 两者都可以和 `T.gemm` 出现在同一个 `T.Kernel` 里：`T.gemm` 跑在 Cube/AIC，
  VF 区域跑在 Vector/AIV，编译器负责核间分工与同步
  （`tilelang/ascend/README.md:117-127`）。

---

## 4. 实现机制

### 4.1 `T.copy` 怎么选指令

CUDA 侧的指令选择是一条有优先级的判定链，实现在
`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:716-808` 的
`SelectCopyInstForLowering`。把它翻译成顺序判断：

```text
1. 带 is_gather4 / is_scatter4 标记 → 强制走 gather4/scatter4 TMA（无回退）
2. cluster_mask != 0               → 必须 TMA multicast，否则报错
3. is_tma_copy（T.tma_copy）        → 必须 TMA
4. is_async_copy（T.async_copy）    → 必须 cp.async，否则报错
5. prefer_instruction="tma"        → 与 disable_tma / pass config 冲突则报错
6. prefer_instruction="cp_async"   → 同上；与 enable_async_copy=False 冲突则报错
7. prefer_instruction="sync"       → 只允许同步形态（LDSM/STSM/TMEM/Normal）
8. 默认                            → 先试 TMA *store*；不行再走同步形态
                                      （LDSM → STSM → TMEM → Normal）
```

几个精确条件：

* TMA bulk load 要求 **src 在 global、dst 在 shared/shared.dyn**，且最后一维的
  **位宽是 128 的整数倍**（即 `extent * dtype.bits % 128 == 0`），并且 dtype 对、
  global stride 不超限（`src/cuda/op/copy_analysis.cc:195-225`）。
* `cp.async` 的前提是 **global → shared 且两侧 dtype 相同**，
  不满足就退回同步路径（同文件 `:427-437`）。
* `ldmatrix`/`stmatrix` 只在 `shared → fragment` / `fragment → shared` 时被考虑，
  TMEM 搬运只在 SM100 的 TMEM 与 fragment 之间（同文件 `:402-419`）。
* 默认路径**先试 TMA store，再试同步形态**，注意 load 方向默认不会自动选 TMA，
  除非显式 `prefer_instruction="tma"` 或 `T.tma_copy`（同文件 `:796-807`）。

这套判定在 **layout 推导阶段就会跑一次**（`src/cuda/op/copy.cc:528-547`），因为
选出来的指令形态不同，需要的 layout 也不同。这就是为什么“换个指令”会连带改变
shared memory 的 swizzle。它们不是两件独立的事。

还有一条容易踩的规则：如果给 `T.copy` 传了 `loop_layout=`，但实际选出的指令不是
`Normal` 或 `CPAsync`，编译器会直接报错（`src/cuda/op/copy.cc:549-562`）。因为
TMA/LDSM 路径根本没有那个 SIMT 循环给你贴 layout。

### 4.2 越界与 padding

这是 `T.copy` 最容易被误解的部分。分两条路径讲。

**SIMT 路径**（`src/op/copy.cc:481-545`）：为每个非 1 维度生成边界谓词
`range.min + i < extent && >= 0`（`MakePredicate`，同文件 `:449-481`），然后：

```cpp
PrimExpr safe_value = make_zero(src->dtype);
if (src_oob_safe_value.defined()) safe_value = src_oob_safe_value.value();
value = if_then_else(src_predicate, value, safe_value);   // :520
```

也就是说：**源越界时读到的是“安全值”，默认 0**；目标越界时整条 store 被
`IfThenElse` 包住（同文件 `:525`）。安全值可以用 `T.annotate_safe_value({A: v})`
改掉（`tilelang/language/annotations.py:69-74`），它在 lowering 时被挂到
`CopyNode::src_oob_safe_value`（`src/op/copy.cc:94-116`）。

**TMA 路径**：越界由 descriptor 的 OOB fill 处理（`TMADesc` 里有独立的 `oob_fill`
字段，`src/cuda/op/copy.h:49-60`）。这时你拿不到一个可以随手改的 Python 层安全值，
语义由指令本身定义。

**昇腾路径**另有自己的 padding 概念，而且**不是越界填充**，是 DMA 对齐填充：
`T.copy(..., pad_value=v)` 会把行宽向上取整到下一个 32B 边界，用 `v` 填 pad lane
（`tilelang/ascend/language/copy_op.py:278-287`）。实现上前端先发一条
`T.ascend_set_copy_pad_value(v)`，**必须单独作为一条语句**，这样 AutoSchedule 才能
在“标量写 pad 寄存器”与“读该寄存器的 MTE2 拷贝”之间插入 `PIPE_S → PIPE_MTE2`
同步（同文件 `:332-351`）。

### 4.3 `T.gemm` 的 layout 与指令选择

CUDA 侧的指令选择在 `src/cuda/op/gemm.cc:336-374`，顺序是：

```text
isWgmma_  → 强制 wgmma（不满足条件则 Fatal）
isTcgen05_→ 强制 tcgen05（不满足则 Fatal）
use_2cta  → 只能 tcgen05（没有合法的单 CTA 回退）
能 tcgen05 → cuda.tcgen05
能 wgmma   → cuda.wgmma
Volta/Turing 且 MMA 形态不合法 → cuda.fma
否则 → cuda.mma
```

`wgmma` 的条件是：Hopper + `M >= 64` + warp 数是 4 的倍数 + dtype/transpose/K 对齐
满足逐条检查（同文件 `:89-93` 与 `:40-76`）。`tcgen05` 的条件是：SM100 +
A 在 shared 或 TMEM + B 在 shared + **C 在 TMEM** + 形状有对应的 TCGen5 meta
（同文件 `:80-87`）。

**`policy` 影响的是 warp 划分，进而是每个 warp 拿到的 fragment 形状，也就是 layout**：

```text
tcgen05 → (m_warp, n_warp) = (1, num_warps)
wgmma   → ComputeWgmmaWarpPartition(policy, M, N, num_warps)
fma     → (1, num_warps)
其它     → ComputeDefaultWarpPartition(policy, M, N, num_warps, k_n_per_warp=8)
```

（`src/cuda/op/gemm.cc:375-395`。）另一条相关的规则在
`cuda::Gemm::ReuseExistingSharedLayout`：只有 `cuda.mma` 与块缩放 MMA **复用**已有的
shared layout；`wgmma`/`tcgen05` 不复用（同文件 `:398-400`）。这意味着把一次 `T.gemm`
从 mma 换成 wgmma，shared buffer 的 layout 可能整个变掉。

昇腾侧简单得多：`ascend::Gemm::SelectInst` 永远返回 `ascend.mad`，
warp 划分恒为 `(1,1)`（`src/ascend/op/gemm.cc:24-45`）。差异被放进了“从 L1 到
L0A/L0B 的搬运”和“Cube 的 unit flag”这两处，而不是藏在 GEMM 内部。

### 4.4 reduce 的跨线程归约

`T.reduce_*` 的前端先按 scope 组合选一条路径
（`tilelang/language/reduce_op.py:103-171`）：

| src / out | 前端做什么 |
| --- | --- |
| shared / shared | 各开一个 fragment，`src→frag_in`、`frag_out→out` |
| shared / fragment | `src→frag_in`，归约写进你的 fragment |
| fragment / shared | 开 `frag_out`；`clear=False` 时先把 out 拷进来，算完拷回去 |
| fragment / fragment | 直接发 `tl.tileop.reduce` |
| local / local.var | 直接发 intrinsic（`tilelang/language/reduce_op.py:79-91`） |

真正的归约发生在 lowering 里，分两级：

1. **线程内**：把你负责的那部分元素用串行循环（并做迭代压缩）累起来；
2. **跨线程**：调用 `tl::AllReduce<Reducer, threads, scale, thread_offset, Barrier,
   batch_size, workspace_stride>::run(...)`（`src/tl_templates/cuda/reduce.h:240`）。

`AllReduce` 是一个递归 XOR-butterfly：

```text
offset = threads/2
offset >= 32 → __syncthreads(); red_buf[threadIdx.x] = x; __syncthreads();
               x = Reducer()(x, red_buf[(threadIdx.x) ^ offset]);   // 走 shared
offset <  32 → x = Reducer()(x, shfl_xor_sync(-1, x, offset));      // 走 warp shuffle
```

（同文件 `:280-295`；`offset` 从 `threads/2` 逐级减半，直到等于 `scale`。）

由此得到三条工程结论：

* 归约宽度必须是 **2 的幂**（`(threads/scale)` 必须是 2 的幂，
  `src/tl_templates/cuda/reduce.h:245-247`），否则编译期 `static_assert` 直接失败。
* 当参与归约的线程数 **> 32** 时，第一级要经过 shared memory 并把 workspace 传进来
  （`src/op/reduce.cc:185-188` 的注释与 `add_workspace` 调用）。
* `batch > 1` 时编译器会把 `ceil(N/batch)` 个输出打包成一次 batched AllReduce，
  **共享同一对 barrier**，把 barrier 开销按 batch 倍摊薄
  （`tilelang/language/reduce_op.py:42-46`）。

### 4.5 原子操作的路径

`T.atomic_add` 的 tile-region 路径在 CUDA 侧有两条实现：普通的 per-element 原子
（可能自动向量化成 `AtomicAddx2`/`AtomicAddx4`），以及在
`use_tma=True` 时的 TMA `cp.reduce`（`src/cuda/op/atomic_add.cc:36-62`）。
昇腾侧的对应物不是 `T.atomic_add` 而是两种风格：

* **store-mode**：`T.set_atomic("add", "float32")` 打开硬件标志，之后普通的
  `T.copy`（L0C→GM）或 `T.dual_copy`（UB→GM）写回就变成“读-改-写”的原子规约，
  用完必须 `T.set_atomic_none()` 关掉（`tilelang/ascend/language/mode.py:63-105`）。
  这个标志是**有状态的**，忘记关掉会让后面的普通 store 也变成累加。
* **适配器 API**：`T.tile.atomic_add(dst_gm, src_local)` 直接表达“把本地 tensor 原子
  累加到 GM”，V1 只支持 `UB→GM` 与 `L0C→GM`
  （`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend_tile.py:313-341`；
  `_vendor/tilelang-ascend-v0.1.2.000/src/op/ascend.cc:748-761`）。

### 4.6 昇腾侧的一条完整链路

把上面的零件串起来，昇腾一次 GEMM+ReLU 的数据流是：

```text
GM ──T.copy(GM→L1)──▶ L1 ──T.copy(L1→L0A/L0B)──▶ L0A/L0B
                                   │
                            T.gemm (ascend.mad)
                                   ▼
                                  L0C ──T.dual_copy──▶ UB
                                   │
                          T.SimtVF / T.SimdVF 上的 ReLU
                                   ▼
                          T.dual_copy(UB→GM) ──▶ GM
```

每一步都对应 §2.2 表里的一行。`T.dual_copy` 在这里的作用不只是搬数据，而是把
“L0C 的累加器”按 M 或 N 切成两半交给两个 AIV，否则 Vector 侧的并行度只有一半。

---

## 5. 进阶与陷阱

### 5.1 版本差异清单（两处最容易踩）

1. **`T.tile.*` 不在 v0.1.15**。它在昇腾适配器仓库
   `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend_tile.py`，通过
   `from . import ascend_tile as tile`（同仓库 `tilelang/language/__init__.py:86`）暴露。
   在 v0.1.15 里搜不到这个名字空间（核验方式见 `references.md` S20）。
2. **`real_shape=` / `tmp=` 是适配器为 `T.reduce_*` 加的**。适配器签名是
   `reduce_sum(buffer, out, dim=-1, *args, clear=..., real_shape=..., tmp=...)`
   （`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/reduce_ascend.py`），
   而 v0.1.15 的签名是
   `reduce_sum(buffer, out, dim=-1, clear=True, batch=1, annotations=None)`
   （`_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:230-232`）。用错版本会
   报未知参数。

其余差异：`T.dual_copy`、`T.SimtVF`、`T.SimdVF`、`T.set_atomic` 在 **v0.1.15 里就有**
（分别在 `tilelang/ascend/language/copy_op.py:53`、`frame.py:107`、`frame.py:179`、
`mode.py:63`），不需要适配器仓库。

### 5.2 常见错误

| 症状 | 真实原因 | 检查点 |
| --- | --- | --- |
| `T.gemm` 报 “requires static tile dimensions” | M/N/K 是符号表达式 | 用常数 tile 尺寸；符号只出现在 K 轴上也不行（`gemm_op.py:90-92`） |
| `T.gemm` 报 `M_A`/`K_A` 形状不匹配 | `transpose_A/B` 与 buffer 的实际排布不一致 | 口径：A `[M,K]`、B（`transpose_B=False`）`[K,N]`（`gemm_op.py:77-88`） |
| reduce 报 shape error | 输出既不是压缩 shape 也不是 keepdim shape | 二选一，别写第三种（`reduce_op.py:63-71`） |
| reduce 编译期 `static_assert` 失败 | 参与归约的线程宽度不是 2 的幂 | `threads/scale` 必须是 2 的幂（`reduce.h:245-247`） |
| `T.atomic_add(..., return_prev=True)` 抛 `NotImplementedError` | 用了 tile-region 路径 | `return_prev` 只在标量路径可用（`atomic.py:304-305`） |
| `T.copy(loop_layout=...)` 报 fatal | 实际选出的指令不是 SIMT/CPAsync | 加 `disable_tma=True` / `prefer_instruction="sync"`（`cuda/op/copy.cc:549-562`） |
| 昇腾 `T.copy` 报 element count mismatch | 两个整 buffer 元素数不等 | Ascend 只允许“元素数相等、形状不同”（`ascend/language/copy_op.py:296-304`） |
| 昇腾 `dual_copy` 报 “Cannot infer split direction” | 尾部两个维度都不是 2:1，或都不是 | 有且只能有一个 2:1（`ascend/language/copy_op.py:122-134`） |
| 昇腾寄存器值不对 | `set_atomic` 忘了 `set_atomic_none()` | 它是**有状态**的全局开关 |
| 文档里 `clear=False` 看不懂 | v0.1.15 `reduce_sum` docstring 的 Note 段把取值写成了 `True` | 以代码为准：只有 `clear=False` 才会把已有 `out` 载入 fragment 再累加（`reduce_op.py:141-158`） |

### 5.3 自己确认这些结论：三条路径

本章的结论都是读源码读出来的。你可以在有硬件的机器上用下面三条路径把它变成观测：

1. **看生成代码**：编译后调用 `kernel.get_kernel_source()`，把生成的 CUDA / Ascend C
   打印出来，逐行对着 §2.2 的表数。入口在
   `tilelang/jit/__init__.py:507` 与 `tilelang/jit/kernel.py:532`。
   这是最直接的一条；本章的 `code/04-`、`code/05-` 两个骨架就是为它准备的。
2. **看每一次 pass 前后的 IR**：设 `TL_LOWER_TRACE=1` 跑同一个脚本，会得到每个 pass
   的 before/after diff，以及最终 codegen 产出的源码文件。工具说明见
   `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/lower_trace.md`
   （R10）。这条路径最适合回答“tile op 是在哪一步变成具体指令的”，
   `tl.LowerTileOp` 这个 pass 的注册在
   `src/transform/lower_tile_op.cc:1477-1486`，pass 本体在 `:207-224`。
3. **看 layout**：想知道某个 `T.gemm` 的 policy 到底把 fragment 切成了什么形状，
   用 layout 可视化（`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/layout_visualization.md`，R11）。
   第 10 章会专门讲这件事，这里只需要知道有这条路。

三条路径都需要真实硬件才能走到最后一步；没有硬件的读者可以用本章 `code/01-` 的自检
脚本先确认“引用的源码位置是对的”，再用 `code/02-`、`code/03-` 的模型确认“规则是
自洽的”。这两步不能替代硬件验证，但能把“我读错了源码”这一类错误先排掉。

### 5.4 一个仍需确认的地方

本章把 `T.copy` 的 SIMT 越界语义、TMA 的 OOB fill、昇腾的 32B pad 三件事分开讲，
依据是三条不同的代码路径；`T.gemm` 的指令选择、reduce 的通信层级同理。
这些结论在本机只能靠读源码得到，因此**本章的所有机制描述都是“静态推断”，不是观测**。
真正需要的补齐动作只有一个：在有卡机器上跑上面 5.3 的第 1、2 条路径，把生成代码与本
章的表逐条对照。没有这一步，本章的内容适合当阅读源码的地图，不适合当性能结论。

---

## 6. 本章小结

* 原语是**意图契约**：前端只组装 IR，实现由 lowering 按 target + scope + layout +
  注解选择。同一行 `T.copy` 可能变成 SIMT 循环、`cp.async`、TMA、`ldmatrix` 或
  Ascend DMA。
* `T.copy` 的三件事要分开记：**extent 从哪来**（Buffer shape / region extent）、
  **越界怎么办**（SIMT 用谓词 + 安全值，默认 0，可被 `T.annotate_safe_value` 改；
  TMA 用 descriptor 的 OOB fill）、**昇腾的 pad 是 DMA 对齐填充而非越界填充**。
* `T.gemm` 的输入是三个 buffer：M/N/K 从 C 与 A/B 的最后两维推出且必须静态；
  `clear_accum` 会改变 C 在 IR 里的读写标记；`policy` 决定 warp 划分与 fragment
  layout；**`accum_dtype` 不是参数**，累加类型由 C 的 dtype 决定。
* `T.reduce_*` 分两级：线程内串行归约 + `tl::AllReduce` 的 XOR-butterfly；
  宽度 > 32 走 shared + barrier，< 32 走 `shfl_xor_sync`；`clear` 决定是“先清零”
  还是“与已有 out 合并”。
* 原子操作分标量路径与 tile-region 路径，能力不同；昇腾用有状态的
  `set_atomic` 或适配器的 `T.tile.atomic_add`。
* `T.fill`/`T.clear` 就是并行循环，没有魔法。
* `T.tile.*`、`real_shape`、`tmp` 属于昇腾适配器仓库；`T.dual_copy`、`T.SimtVF`、
  `T.SimdVF`、`T.set_atomic` 属于 v0.1.15 主干。

## 延伸阅读

* 官方 `Instructions` 页：[`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/instructions.md`](../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/instructions.md)（R1）
* 昇腾适配器逐 API 文档：[`../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/api_docs/`](../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/api_docs/)（R6）
* 昇腾《Programming Guide》第 4 章：[`_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md`](../../_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend%20Programming%20Guide.md)（S14）
* 本机最小示例：见本目录 `code/README.md` 与各子目录 README。
* 下一章（第 10 章）会把本章反复提到的 layout 单独展开。

本章引用的每条源码与资料都在 [`references.md`](references.md) 里登记了核验方式与状态。
