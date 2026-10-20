# 第 03 章 TileLang 的设计哲学与编程模型

## 学习目标

读完本章，你应当能够：

1. 用自己的话说清 TileLang 论文的核心主张，把**调度空间（scheduling space）**与
   **数据流（dataflow）**解耦，并解释这两个词在代码里各自对应什么。
2. 说出调度空间被拆成了哪四个维度（thread binding / layout / tensorize / pipeline），
   并为每一个在 v0.1.15 里指出至少一个能操控它的 API 或注解。
3. 说出三层编程接口（beginner / developer / expert）各自的定位、v0.1.15 里对应的具体
   构造，并如实说明 beginner 层的实现状态。
4. 解释 “composable”（可组合）在 TileLang 里的确切含义，并能在真实例子里逐行指出哪一句
   是高层 tile 原语、哪一句是低层 thread 原语。
5. 给出本书对 tile 的定义，并说明它分别落在语言层（`T.Kernel` / `T.alloc_*`）与编译期
   （LayoutInference / `Fragment`）的哪个位置。
6. 对照 Triton“让编译器选调度”这一路线，讲清 TileLang 换回了什么、付出了什么，以及
   这套设计的能力边界在哪里。

## 前置知识

* 第 01 章：tile 的第一版定义、算术强度与机器平衡点、DSL 谱系（Halide → TVM → Triton →
  TileLang）。
* 第 02 章：一个算子在 PyTorch 的软件栈里处于什么位置，以及“什么时候必须自己写 kernel”。
* 会用 Python 函数、装饰器与 `with` 语句。**不需要**写过 CUDA，也不需要懂编译器。
  本章出现的每个硬件名词都会先给一句话解释。

本章是第一部分承上启下的一章：第 01 章说清了“为什么需要 tile DSL”，这一章回答
“TileLang 选择了哪一种 tile DSL”，第 04 章再回答“这套选择在代码里落成了什么架构”。

---

## 1. 问题的由来：调度为什么会变成负担

### 1.1 一个 GEMM 的四种写法

先看一个具体场景。要在 GPU 上算 `C = A @ B`（fp16，fp32 累加），至少可以这样写：

1. 在 PyTorch 里写 `C = A @ B`，让 cuBLAS 去选 tile 尺寸、线程映射与流水线；
2. 用 Triton 写一个 128×128 的块级 kernel，指定 block size，其余交给 Triton 的编译器；
3. 用 TileLang 写一个 tile 程序，显式声明“把 A、B 的哪一块搬进 shared memory、
   累加器放在 fragment 里、沿 K 方向流水几级”；
4. 手写 CUDA，自己算 `threadIdx` 到元素的映射、自己插 `__syncthreads()`、
   自己发 `ldmatrix` / `mma` 指令。

四种写法算的是同一个数学式子，差别全在**“哪些决定由谁来做”**。

### 1.2 决定可以分成两类

把上面四种写法里的决定列出来，会发现它们天然分成两类。

**第一类是数据流（dataflow）**：数据从哪来、经过哪些计算、到哪去。对 GEMM 来说就是
“把 A、B 的分块读进来 → 乘加累加 → 写回去”，这一类决定和硬件无关，换一台机器也成立。

**第二类是调度（schedule）**：为了让这段数据流在具体硬件上跑得快，需要额外做的决定。
它们和硬件强耦合，换一台机器往往要重来一遍。至少有这么四组：

* **thread binding（线程绑定）**：这段计算分给多少个线程？一个 tile 由几个 warp 协作？
  哪个线程负责哪个元素？
* **layout（布局）**：同一个逻辑上的 `[128, 64]` tile，在线程的寄存器里和组织成什么形状？
  在 shared memory 里要不要 swizzle（打散）以避免 bank conflict？
* **tensorize（张量化）**：这一步计算要不要映射到张量核（tensor core / Cube）指令上？
  映射成哪一代指令（`mma` / `wgmma` / `tcgen05`）？
* **pipeline（流水线）**：搬运和计算怎么重叠？开几级多缓冲？

### 1.3 传统做法把两类决定混在一起

手写 CUDA 时，这两类决定是**混在同一段代码里**的：你写 `C_frag[...] += A_frag[...] * B_frag[...]`
的同时，也决定了它最终映射到哪条 `mma` 指令；你写 `__syncthreads()` 的位置，既表达了
数据依赖，也表达了流水阶段的划分。代价是：想换一个 tile 尺寸或换一代张量核指令，
往往要重写一大段代码，而且很容易把“数据流”改坏。

Halide 与 TVM 提出了第一次解耦，**算法（algorithm）与调度（schedule）分离**：先描述
“算什么”，再用 schedule 描述“怎么算”。可它们的 schedule 是一套复杂的循环变换语言，
对只写过 PyTorch 的人来说门槛很高（这一段的谱系与出处见第 01 章；第 5 章会给出三者的
横向对照）。

Triton 走了相反的方向：**把调度尽量交给编译器**。用户写块级程序，编译器决定 intra-tile
的布局、合并访存与向量化。这条路线换来了极低的编写门槛，代价是用户对关键决定的控制力
被削弱（第 5 节会具体展开）。

TileLang 的选择介于两者之间，也正是本章要讲清楚的东西。

---

## 2. 概念与模型

### 2.1 论文的核心主张

TileLang 的论文摘要把主张写得很直接（原文英文，此处逐字引用）：

> “TileLang decouples scheduling space (thread binding, layout, tensorize and pipeline)
> from dataflow, and encapsulated them as a set of customization annotations and primitives.
> This approach allows users to focus on the kernel's data-flow itself, while leaving most
> other optimizations to compilers.”
>
> 来源：`../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577.md`
> （TileLang: A Composable Tiled Programming Model for AI Systems，arXiv:2504.17577v2，
> 2025-04-27 更新；作者列表见该文件头，单位标注见同目录的 `_index.md`）。

这段话里有三个可以直接检验的点，本章剩下的部分就是逐条把它们落到代码上：

1. **解耦的对象是什么**：`scheduling space` 与 `dataflow`；
2. **调度空间被拆成哪几块**：括号里点名的四个词，thread binding、layout、tensorize、pipeline；
3. **解耦之后靠什么表达**：`customization annotations and primitives`，注解与原语。

第 3 条在本章第 4 节展开（注解最终变成 IR 上的属性，被具体的 pass 消费），第 1、2 条在
下面两小节展开。

### 2.2 数据流：写在 kernel 里的那部分

在 TileLang 里，数据流就是你**显式写出来的**那些语句：申请哪些缓冲、把哪些 tile 搬进来、
对它们做什么计算、结果写回哪里。下面这段取自锁定版本的 `examples/quickstart.py`
（TileLang `v0.1.15`），其中 `A`、`B` 是输入，`C = relu(A @ B)`：

```python
# _vendor/tilelang-v0.1.15/examples/quickstart.py:18-46（节选，删去了几行原文注释）
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_K), dtype)
    B_shared = T.alloc_shared((block_K, block_N), dtype)
    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

    T.clear(C_local)

    for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
        T.copy(A[by * block_M, ko * block_K], A_shared)
        T.copy(B[ko * block_K, bx * block_N], B_shared)
        T.gemm(A_shared, B_shared, C_local)

    for i, j in T.Parallel(block_M, block_N):
        C_local[i, j] = T.max(C_local[i, j], 0)

    T.copy(C_local, C[by * block_M, bx * block_N])
```

这十几行里，数据流一目了然：**搬入 → 乘加 → 激活 → 搬出**。注意它的写法特点：

* 全程**没有出现线程索引**。没有 `threadIdx.x`，没有“第 3 号 warp 负责第几行”；
* 全程**没有出现同步原语**。没有 `__syncthreads()`，没有 barrier；
* 缓冲的**存储位置是显式写出来的**（`alloc_shared` / `alloc_fragment`），
  但每个线程具体分到哪些元素，代码里没写。

后两点就是“解耦”在读者眼里的样子。存储位置属于数据流的一部分（它决定了数据要经过
哪一级存储），而“怎么分给线程、怎么同步”属于调度，被推到了编译期。

### 2.3 调度空间：四个维度及其在 v0.1.15 里的落点

把论文点名的四个维度落到实处，每一个在 v0.1.15 里都能找到对应的 API 或注解。

| 维度 | 它决定什么 | v0.1.15 里操控它的入口 | 缺省行为 |
| --- | --- | --- | --- |
| thread binding | 一个 tile 分给多少个线程 / warp，谁负责哪一块 | `T.Kernel(..., threads=...)`（`tilelang/language/kernel.py:431`，CUDA 方言见 `tilelang/cuda/language/kernel.py:12-54`）、`T.get_thread_binding()`（`kernel.py:587`）、GEMM 的 warp 划分 `policy=GemmWarpPolicy.*`（`tilelang/language/gemm_op.py:160`，枚举定义在 `tilelang/tileop/base.py:5-12`） | 不写就由编译器推导（`T.Kernel` 缺省 128 线程，见 `tilelang/cuda/language/kernel.py:32-35`） |
| layout | 一个逻辑 tile 映射到线程寄存器 / shared memory 的形状 | `T.annotate_layout({buf: T.Layout(...)})`（`tilelang/language/annotations.py:45`）、`T.Fragment` / `T.Layout`（`tilelang/layout/__init__.py:4-5`）、`make_swizzled_layout`（`tilelang/layout/swizzle.py:66`）、`T.Parallel(..., loop_layout=...)`（`tilelang/language/loop.py:13-18`） | 不写就由 **LayoutInference** pass 推导（`tilelang/transform/__init__.py:30-38`） |
| tensorize | 一步计算映射到哪条硬件指令 | `T.gemm`（`gemm_op.py:160`）与 `T.gemm_blockscaled` / `T.gemm_sp`；CUDA 方言另加 `T.wgmma_gemm` / `T.tcgen05_gemm` / `T.tcgen05_gemm_blockscaled` / `T.mma_gemm_blockscaled`（`tilelang/cuda/language/gemm_op.py:19-29` 的 `__all__` 列全了） | 不写就由后端按 target 与操作数 scope 选择 |
| pipeline | 搬运与计算怎样重叠、开几级缓冲 | `T.Pipelined(..., num_stages=...)`（`loop.py:112`）、显式 `stage=[...] / order=[...]`（`_vendor/tilelang-v0.1.15/docs/programming_guides/software_pipeline.md:44-60`）、`T.Persistent`（`loop.py:82`） | 软件流水由 `T.Pipelined` 表达；对编译器推导的流水，`num_stages=0` 就是不启用（`loop.py:128-133`） |

这张表就是本章最重要的结论之一：**论文里的四个词不是修辞，它们在锁定版本里都有确切的
API 入口，而且都有“不写则由编译器决定”的缺省路径。** 这正是“解耦”的工程含义，
用户可以在任何一个维度上介入，也可以完全不介入。

### 2.4 tile 是一等概念：定义与落点

第 01 章给过 tile 的第一版定义。到了这一章，可以把它说得更精确。**在本书里，一个 tile
由三件事共同定义：**

1. **一段有形状的数据**（shaped portion of data），不是“一个元素”，也不是“整个张量”，
   而是一个有具体 shape 的切片；
2. **一处明确的归属**。它由谁拥有：一个线程块、一个 warp，还是某个专门的存储单元；
3. **一个显式的存储位置**。它此刻在哪一级存储里（global / shared / fragment / 专用 buffer）。

第 2 条里的“归属”不是一个抽象说法：它在编译期具体落实为 **layout**，即“这块数据的
哪个元素落在哪个线程、哪个寄存器槽位或哪段片上存储”的映射（含 fragment、swizzle、
昇腾的 fractal/ND→NZ 等形式）。第 01 章把这一条称作“物理布局”，与本节的“归属”是
同一件事，本书后续统一写作“归属（物理布局）”。

第 3 条不是我们编的：TileLang 官方文档把它称为 hallmark（标志性特征），
“Rather than leaving it to a compiler's opaque optimization passes, TileLang exposes
user-facing intrinsics that map directly to physical memory spaces or accelerator-specific
constructs”（`_vendor/tilelang-v0.1.15/docs/get_started/overview.md:78-84`）。
同一份文档还给 tile 下了官方定义：

> “At the heart of our approach is the notion of *tiles* as first-class objects in the
> programming model. A tile represents a shaped portion of data, which can be owned and
> manipulated by a warp, thread block, or equivalent parallel unit.”
>
> 来源：`_vendor/tilelang-v0.1.15/docs/get_started/overview.md:74`

“一等概念”在语言里的直接体现，就是**tile 有专门的分配原语**，而不是靠程序员自己
从一大块内存里切：

| 原语 | 默认存储作用域 | 语义 | 定义位置 |
| --- | --- | --- | --- |
| `T.alloc_shared` | `shared.dyn` | 线程块内共享的片上存储 | `tilelang/language/allocate.py:49` |
| `T.alloc_local` | `local` | 线程私有的存储（寄存器或本地内存） | `tilelang/language/allocate.py:67` |
| `T.alloc_fragment` | `local.fragment` | “整块 tile 的逻辑缓冲”，物理上由 LayoutInference 摊到每个线程的寄存器 | `tilelang/language/allocate.py:81` |

注意 `alloc_fragment` 这条：它是“tile 是一等概念”最有说服力的证据。你申请的是
`(block_M, block_N)` 这么大的一块**逻辑**缓冲，但物理上它并不存在，编译期会推导出一个
`Fragment` 布局，把它切成每个线程各自持有的若干寄存器。官方文档把这个过程写得很清楚：
“`TileLang` uses a Layout Inference Pass during compilation to derive a Layout object
`T.Fragment`, which determines how to allocate the corresponding register files for each
thread”（`docs/get_started/overview.md:82`）。也就是说：**tile 在语言层是一个整体，
在编译期才被折叠成线程私有的碎片。**

### 2.5 三层接口的定位、现状与如实说明

官方文档把用户分成三层（`_vendor/tilelang-v0.1.15/docs/get_started/overview.md:5-30`）：

```text
Beginner Level (Hardware-Unaware)
Developer Level (Hardware-Aware with Tile Library)
Expert Level (Hardware-Aware with Thread Primitives)
```

同一份文档接着写道：

> “The **Tile Language** also allows mixing these interfaces within the same kernel,
> enabling users to work at whichever level of abstraction best suits their needs.”
>
> 来源：`_vendor/tilelang-v0.1.15/docs/get_started/overview.md:5`

**这里有一件事必须如实写清楚：beginner 层在 v0.1.15 里还没有完全实现。** 官方文档在
beginner 那条下面直接标注：

> “*Note:* This interface is not yet fully implemented.”
>
> 来源：`_vendor/tilelang-v0.1.15/docs/get_started/overview.md:20`

这句话在 corpus 转存的同一份文档里也一致
（`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/overview.md`）。
所以本章不会说“TileLang 有完整的硬件无关层”，只会说**这个方向存在、有可见的实现
雏形、但尚未完成**。这个雏形是什么，见下面 3.4 节。

另一份更早的官方文档（`docs/deeplearning_operators/matmul.md`）用了 Level 1 / 2 / 3 的
编号，并给出了三个很有用的类比（`:17-21`）：

| 官方文档的编号 | 本章的称呼 | 文档给出的类比 | 文档原话 |
| --- | --- | --- | --- |
| Level 1 | beginner | “conceptually similar to the idea behind TVM” | 用户只写纯计算逻辑，调度与优化由编译器/运行时做 |
| Level 2 | developer | “somewhat comparable to Triton's programming model” | 懂 shared memory / tiling / thread block，但不必下到显式线程控制 |
| Level 3 | expert | “almost as explicit as a hand-written CUDA/HIP kernel” | 完全掌控 thread 级原语，可写 PTX 内联汇编 |

三层在 v0.1.15 里对应的具体构造，可以这样对应：

| 层 | 定位 | v0.1.15 里的可见形态 | 状态 |
| --- | --- | --- | --- |
| beginner | 不知道硬件，也能描述计算 | eager 前端：`@tilelang.jit` + `T.const` / `T.dynamic` + `T.Tensor[[...]]` 注解 + `T.empty`（`tilelang/language/eager/builder.py:1177`、`tilelang/language/allocate.py:400`；示例 `examples/eager_jit/eagerjit.en.ipynb`） | **方向存在，层本身未完成**（见上） |
| developer | 知道存储层级，用 tile 库 | `T.Kernel` + `T.alloc_shared/fragment` + `T.copy` + `T.gemm` + `T.reduce_*` + `T.Pipelined` + `T.Parallel`（示例 `examples/flash_attention/example_mha_fwd_bshd.py`） | v0.1.15 的主体，示例与文档最厚 |
| expert | 掌控线程与指令 | `T.get_thread_binding()`（`kernel.py:587`）+ `T.alloc_local`（`allocate.py:67`）+ `T.serial` / `T.vectorized`；再往下有 CUDA 方言的 `T.tma_load`（`tilelang/language/builtin.py:321`）、`T.mbarrier_*`、`T.ptx_ldmatrix`（`tilelang/cuda/language/tir.py:555`，且被登记为 CUDA-only，见 `tilelang/language/tir/exports.py:7-23`）等（示例 `examples/gemv/example_gemv.py`） | v0.1.15 里存在且被大量示例使用 |

一个容易踩的坑：`T.const` 的 docstring 写得很明确，“Declare constexpr variables for
dynamic tensor dimensions (**eager mode only**)”，并且在非 eager 上下文里调用会直接
抛 `JITNoBuilderError`（`tilelang/language/eager/builder.py:1177-1196`）。它虽然在
`T` 命名空间里可见（`tilelang/language/common.py:17` 那句 `from .eager import *`），
但**只有配合 `@tilelang.jit` 的 eager 模式才有意义**。

### 2.6 “composable” 的确切含义

论文标题里的 composable（可组合）最容易被误读成“原语可以自由拼装”。它在 TileLang 里
有一个更具体、也更强的意思：**同一个 kernel 里，高层 tile 原语与低层 thread 原语可以共存，
中间不需要任何“模式切换”或“边界声明”。**

官方文档把这一点写成了 TileLang 相对 Triton 的三条差异之一：

> “- Enables mixing different levels in a single program—for example, you can write some
> parts of your kernel in Level 3 (thread primitives) for fine-grained PTX/inline-assembly
> and keep the rest in Level 2.”
>
> 来源：`_vendor/tilelang-v0.1.15/docs/deeplearning_operators/matmul.md:214-218`

注意这句话把两种可能性都包含了：

* 高层写在低层**旁边**（同一个 kernel 里一段用 `T.Parallel`、一段用 `T.get_thread_binding`）；
* 高层写在低层**里面**（一个由线程原语手写的 macro，被塞进 `T.Pipelined` 的循环体，
  和 `T.gemm` 相邻）。

3.3 节会给出 v0.1.15 里两个真实例子，一个对应前者，一个对应后者。

---

## 3. 用法：同一段数据流，三种写法

本节的目标不是教会你写这些 kernel（那是第 8–11 章），而是让你**认出**每一层长什么样，
以及它们如何拼进同一个 kernel。

### 3.1 developer 层：tile 库写法

下面这段取自 v0.1.15 的 FlashAttention 前向示例（`examples/flash_attention/example_mha_fwd_bshd.py:29-92`，
为节省篇幅删去了其余缓冲声明、mask 分支与注释，切片表达式里的空格做了压缩，
`...` 处为省略）：

```python
@T.prim_func
def main(Q: T.Tensor(shape, dtype), K: T.Tensor(shape, dtype),
         V: T.Tensor(shape, dtype), Output: T.Tensor(shape, dtype)):
    with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
        Q_shared = T.alloc_shared([block_M, dim], dtype)
        K_shared = T.alloc_shared([block_N, dim], dtype)
        V_shared = T.alloc_shared([block_N, dim], dtype)
        acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
        acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
        scores_max = T.alloc_fragment([block_M], accum_dtype)
        ...
        T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
        T.fill(acc_o, 0)
        for k in T.Pipelined(loop_range, num_stages=num_stages):
            T.copy(K[bz, k * block_N:(k + 1) * block_N, by, :], K_shared)
            ...
            T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                   policy=T.GemmWarpPolicy.FullRow)
            T.reduce_max(acc_s, scores_max, dim=1, clear=False)
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
            T.reduce_sum(acc_s, scores_sum, dim=1)
            ...
            T.copy(V[bz, k * block_N:(k + 1) * block_N, by, :], V_shared)
            T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
```

这一段里没有出现过一个线程索引，却已经把 attention 的整个数据流写完了。**所有“怎么分给
线程”的问题都留给了编译器**：

* `T.copy` 会自己决定用什么机制搬（同步 SIMT 拷贝、`cp.async`、TMA 都可能在列，
  见 `docs/programming_guides/instructions.md:40-47`）；
* `T.gemm` 会自己选指令（NVIDIA 上可以是 CUTLASS/CuTe 或 WMMA，
  见 `docs/deeplearning_operators/matmul.md:202`）；
* `T.reduce_max` / `T.reduce_sum` 会自己决定用什么归约方式（第 9 章会讲 `reduce_*`
  的降级行为）。

唯一“越界”的是 `policy=T.GemmWarpPolicy.FullRow`。这是用户对 thread binding 的一次
显式介入：告诉编译器把 warp 全部分给行方向。它可以选择不写（默认 `Square`，
见 `tilelang/tileop/base.py:10`）。**这正是“可以介入、也可以不介入”的具体样子。**

### 3.2 expert 层：thread 原语写法

同一个仓库里的 GEMV 示例走了另一个极端（`examples/gemv/example_gemv.py:14-42`；
参数列表被压缩成 `...`，其余逐行保留，含原文的 `# tn = threadIdx.x` 注释）：

```python
@tl.jit
def naive_gemv(A, B, BLOCK_N: int, BLOCK_K: int, ...):
    N, K = T.const("N, K")
    A: T.Tensor((K,), dtype)
    B: T.Tensor((N, K), dtype)
    C = T.empty((N,), dtype)

    with T.Kernel(T.ceildiv(N, BLOCK_N)) as bn:
        tn = T.get_thread_binding(0)  # tn = threadIdx.x
        A_shared = T.alloc_shared((BLOCK_K,), dtype)
        B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
        C_reg = T.alloc_local((1,), accum_dtype)
        T.clear(C_reg)
        for bk in T.serial(T.ceildiv(K, BLOCK_K)):
            for tk in T.serial(BLOCK_K):
                A_shared[tk] = A[bk * BLOCK_K + tk]
                B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
            for tk in T.serial(BLOCK_K):
                C_reg[0] += A_shared[tk].astype(accum_dtype) * B_shared[tn, tk].astype(accum_dtype)
        C[bn * BLOCK_N + tn] = C_reg[0]
    return C
```

这一段的写法几乎就是 CUDA：

* `T.get_thread_binding(0)` 拿到 `threadIdx.x`（CUDA 方言的 docstring 原文就写着
  “``T.get_thread_binding()`` exposes ``threadIdx`` for thread-level code”，
  `tilelang/cuda/language/kernel.py:21-25`）；
* 所有循环都是 `T.serial`，也就是“每个线程各跑一遍”；
* 数据要搬进 shared memory 得自己写循环，不再有 `T.copy`；
* 累加器是 `T.alloc_local((1,), ...)`，一个线程一个标量。

同一文件里还写出了更深的一层，直接落到 TIR 的线程归约内建（`example_gemv.py:191-207`）：

```python
with T.attr(
    T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)]),
    "reduce_scope",
    T.reinterpret(T.uint64(0), dtype="handle"),
):
    T.evaluate(
        T.tvm_thread_allreduce(
            T.uint32(1), C_accum[0], True, C_reduced[0], tk, dtype="handle"
        )
    )
```

这是 expert 层再往下的地板：`T.attr` / `T.evaluate` / `T.comm_reducer` 是 TVM TIR
script 的原语，`T.tvm_thread_allreduce` 是 TIR 内建
（定义在 `tilelang/language/tir/ir.py` 的 `__all__`，见 `:506`）。到这一层，
“tile”这个概念基本退场了，用户面对的就是线程、寄存器与内建函数。

### 3.3 混用：composable 的实证

**例子 A（高层写在低层里面）**：`examples/dequantize_gemm/example_dequant_gemm_bf16_mxfp4_hopper.py`
是一个“反量化 + GEMM”的融合 kernel。它定义了**两个功能等价、层次完全不同**的
dequant macro，然后在同一个 `T.Pipelined` 循环里按开关二选一：

```python
# _vendor/tilelang-v0.1.15/examples/dequantize_gemm/example_dequant_gemm_bf16_mxfp4_hopper.py:366-373
for k in T.Pipelined(K // block_K, num_stages=num_stages):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
    if fast_dequant:
        get_fast_dequant_twiddling_func()(B_shared, B_dequantize_shared, Scale, k)
    else:
        get_simple_dequant_func()(B_shared, B_dequantize_shared, Scale, k)
    T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)
```

两个 macro 的层次差异是这样分布的：

| macro | 定义位置 | 用到的构造 | 相当于哪一层 |
| --- | --- | --- | --- |
| `fast_dequant_bf16_fp4_twiddling` | `:196-263` | `T.get_thread_binding()`（`:225`）、`T.alloc_local`（`:228-231`）、`T.serial` / `T.vectorized`、`T.call_extern`（`:247`） | expert（thread 原语 + 外部内联函数） |
| `simple_dequant_bf16_fp4` | `:279-315` | `T.alloc_fragment`（`:298-299`）、`T.Parallel`（`:303`）、`T.copy`（`:302,313`） | developer（tile 库） |

而它们**被同一行 `T.gemm(...)` 消费**，和 `T.copy` 挤在同一个流水循环里。这就是论文说
“encapsulated them as a set of customization annotations and primitives”的实际形态：
不同层次的东西最后都变成同一种东西，IR 里的 tile op / 语句。因此可以并存。

顺带一提：这个 kernel 还在 `:344-348` 用 `T.annotate_layout` 给 `B_shared` 指定了
swizzle 布局（`tilelang.layout.make_swizzled_layout`），这是 layout 维度上的显式介入。
**一个 kernel 里同时动用了 thread binding、layout、tensorize、pipeline 四个维度**，
这正是 TileLang 想要的表达密度。

**例子 B（低层写在高层的旁边）**：`examples/gemm/example_gemm_intrinsics.py` 把
`T.Parallel` / `T.Pipelined` / `T.annotate_layout` 这些高层构造，和
`TensorCoreIntrinEmitter` 的 `ldmatrix_a` / `ldmatrix_b` / `mma` / `stmatrix`
（`:133-142`）放在同一个 kernel 体里：

```python
# _vendor/tilelang-v0.1.15/examples/gemm/example_gemm_intrinsics.py:122-142（节选，删去了原文注释）
for ko in T.Pipelined((K // block_K), num_stages=stage):
    for i, k in T.Parallel(block_M, block_K):
        A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
    for j, k in T.Parallel(block_N, block_K):
        B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

    for ki in T.serial(0, (block_K // micro_size_k)):
        mma_emitter.ldmatrix_a(A_local, A_shared, ki)
        mma_emitter.ldmatrix_b(B_local, B_shared, ki)
        mma_emitter.mma(A_local, B_local, C_local)
```

搬运用高层 `T.Parallel`，矩阵乘用最底层的 `mma` 指令序列，同一段代码里两种层次直接相邻，
没有过渡层。**如果 TileLang 不是 composable 的，这种写法在语法上就无法表达。**

### 3.4 beginner 层的雏形：eager 前端

2.5 节说过 beginner 层尚未完全实现。那么现存的东西是什么？

v0.1.15 的 `@tilelang.jit` 支持两种模式（`tilelang/jit/__init__.py:277-312`）：

* **lazy 模式**：被装饰的函数返回一个 `PrimFunc`，调用它得到编译好的 kernel 对象，
  再单独调用 kernel。上面 3.1 / 3.2 的例子大多是这种。
* **eager 模式**：函数直接用“tensor 注解 + builder”描述 kernel，调用时**编译并立即执行**。

eager 模式的官方示例长这样（`examples/eager_jit/eagerjit.en.ipynb`，`@tilelang.jit` 单元格）：

```python
@tilelang.jit
def gemm(A, B, out_dtype: T.dtype = T.float32,
         block_M: int = 128, block_N: int = 128, block_K: int = 32):
    M, N, K = T.const("M, N, K")
    A: T.Tensor[[M, K], T.float16]
    B: T.Tensor[[K, N], T.float16]
    C = T.empty((M, N), out_dtype)
    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128) as (bx, by):
        ...
```

notebook 里自己的说明是：**“The function signature looks similar to Triton, but we add
many enhancements; the most important one is allowing rich Tensor annotations”。**

这就是为什么说 beginner 层“方向存在、尚未完成”：eager 前端确实把“形状从张量参数里
推出来”这件事自动化了（`T.const` 负责声明符号维度，`T.dynamic` 负责声明动态维度，
见 `tilelang/language/symbolics.py:13`），但 kernel 体内部**仍然要写
`T.alloc_shared` / `T.gemm` / `T.Pipelined`**，也就是说，用户仍然要理解存储层级。
真正“不需要知道 shared memory 存在”的那一层，还没有交付。

> **读者要注意的陷阱**：本节引用的 eager 写法在 corpus 与 `_vendor/` 里都能核对到，
> 但它属于**官方文档明确标注为未完成的 beginner 方向**。本教材后续章节（第 7–15 章）
> 教你写算子时会以 developer 层为主，因为那是 v0.1.15 里唯一被大规模示例验证过的层次。

---

## 4. 实现机制：解耦在编译器里落在哪

“解耦”不是一句设计口号，它必须落到编译期的具体位置上。这一节给出 v0.1.15 里三个
承重点：tile op 的引入、LayoutInference、以及 pass 顺序。

### 4.1 用户写的是 tile op，不是代码

`T.copy` / `T.gemm` 这些原语在 Python 层**不会展开成任何循环或指令**，它们只是往 IR 里
放一个“tile op”节点。以 `T.copy` 为例，它的实现末尾就是一句构造 intrinsic call：

```python
# _vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:130（该函数最后一句）
return tirx.call_intrin("handle", tirx.op.Op.get("tl.tileop.copy"), src, dst, annotations=ann)
```

`T.gemm` 同理，产出一个 `"tl.tileop.gemm"` 调用（`tilelang/language/gemm_op.py:199-208`），
并带上 `policy`、`clear_accum` 等注解。这带来两个后果：

1. 语言层不需要知道硬件。它只负责记录“这里有一次 tile 拷贝 / tile 矩阵乘”；
2. 硬件相关的决定被推迟到 lowering 阶段，可以按 target 分别处理。

### 4.2 LayoutInference：把“逻辑 tile”折成“每个线程的寄存器”

LayoutInference 是这一整套设计里最关键、也最容易被忽视的一环。它的职责在 C++ 源文件
开头只有一行注释：

```cpp
/*!
 * \file layout_inference.cc
 * \brief infer the fragment/shared memory layout
 */
```

`_vendor/tilelang-v0.1.15/src/transform/layout_inference/layout_inference.cc:1-4`。

它就是官方文档所说的那个“Layout Inference Pass”，负责给每个 `alloc_fragment` 缓冲
推导出一个 `Fragment` 布局，也负责给 `T.Parallel` 循环推一个循环布局。用户如果不满意，
有两种介入方式：

* 在缓冲上直接写死布局：`T.annotate_layout({buf: make_swizzled_layout(buf)})`
  （`tilelang/language/annotations.py:45`）；
* 在循环上给提示：`T.Parallel(..., loop_layout=...)`（`tilelang/language/loop.py:13-18`）。

第二种方式有明确的约束，而且这些约束本身就说明了“布局是接口”，用户在循环上给的布局
必须满足：每个并行循环最终都要被布局覆盖、注解只能挂在最外层循环上、嵌套深度为 `k` 时
布局的 `InputDim` 必须等于 `k`；违反任何一条都会编译报错
（`tilelang/language/loop.py:43-60`）。

### 4.3 pass 顺序：解耦后的四个维度按什么次序被落实

CUDA 后端的 pipeline body 把顺序写得很清楚
（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:127-142`）：

```python
# Run pipeline planning and software-pipeline rewriting before layout
# inference so inferred layouts see the final pipelined structure directly.
mod = tilelang.transform.PipelinePlanning()(mod)
mod = tilelang.transform.InjectSoftwarePipeline()(mod)
mod = tilelang.transform.Simplify()(mod)

# Infer memory layouts for fragments and shared memory
mod = tilelang.transform.LayoutInference()(mod)
mod = tilelang.transform.ReducerPlanAndMaterialize()(mod)
LayoutVisual(mod)
# Lower high-level tile operations to low-level operations
mod = tilelang.transform.LowerTileOp()(mod)
```

`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:127-142`。

这个顺序回答了一个很自然的问题：**为什么布局推导放在流水线重写之后？** 源码注释里
给了答案，“so inferred layouts see the final pipelined structure directly”
（让布局推导直接看到最终的流水线结构）。这就是“解耦”在工程上的代价与收益：一旦
分开，就必须把依赖关系明确排成顺序；排好之后，每一层优化只需要面对它自己的问题。

再往后一步，才是 `LowerTileOp`，把 tile op 拆解成具体的循环、线程绑定与指令。
**用户的 kernel 从“数据流”变成“可执行代码”，是分几个阶段完成的，而不是一次翻译。**

### 4.4 后端方言：解耦让“同一门语言、多个后端”成为可能

一个直接后果是：`T.Kernel`、`T.copy`、`T.gemm` 这些名字在不同后端上可以有不同的扩展，
而共享同一套语义。v0.1.15 的组织方式是（`tilelang/language/common.py:1-8` 的模块
docstring 原文）：

> “This module owns the common (backend-agnostic) language definitions. Each per-backend
> dialect (``tilelang.cuda.language`` etc.) re-exports this surface via
> ``from tilelang.language.common import *`` and layers its own extensions on top.
> ``tilelang.language`` itself is a thin facade that re-exports the default (CUDA) dialect.”

也就是说：`import tilelang.language as T` 默认拿到的是 **CUDA 方言**
（`tilelang/language/__init__.py:32` 的 `__tilelang_dialect__ = "cuda"`），
而昇腾等后端有自己的方言模块（例如 `tilelang/ascend/language/__init__.py` 里
`T.alloc_l1` / `alloc_l0a` / `alloc_l0c` / `dual_copy` 等扩展）。

这一点很重要，它解释了为什么“调度空间”必须被显式建模：**如果调度决定被硬编码在语言里，
多后端就无法共享同一份数据流描述。** 关于各后端的代码生成细节，见第 24、25 章。

---

## 5. 与 Triton 的取舍差异

官方文档自己把 developer 层类比成 Triton 的编程模型
（`docs/deeplearning_operators/matmul.md:19`）。所以这一节不是“谁更好”，而是
**同样面对“块级编程”这个问题，两条路线分别把决定权放在哪里**。

### 5.1 Triton 的路线：把调度交给编译器

Triton 2019 年的原始论文（MAPL'19）把自己的定位写得很清楚：语言与编译器**围绕 tile
这一概念**，而 tile 的定义是
“statically shaped multi-dimensional sub-arrays”
（`../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md:36-37`）。
它明确把“自动推导调度”当成卖点，并承认这是要付出代价的：

> “...(3) automatic inference of possible execution schedule that would otherwise have
> to be specified manually to Halide or TVM. The benefits of Triton come at the cost of
> increased programming efforts...”
>
> 来源：`../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md:248-256`

论文还说，语言内建的 tile 语义“simplifies the structure of tensor programs by **hiding**
important performance details pertaining to intra-tile memory coalescing, cache management
and specialized hardware utilization”（同文件 `:325-329`，加粗为本书所加）。

**请注意“hiding”这个词。** 这就是两条路线的分水岭：Triton 把 tile 内的布局、合并访存、
专用硬件使用**藏起来**，让编译器决定；TileLang 把这些决定**留在语言表面**，让用户可以写、
也可以不写。

线程层面的差别同样写在论文里：Triton 的编程模型“each kernel is **single-threaded** –
though automatically parallelized”，因此“CUDA-like concurrency primitives (shared memory
synchronization, inter-thread communication, etc.) are **inexistent**”
（`../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md:361-370`，
加粗为本书所加）。**在 2019 年的这套语言里，用户没有地方写线程级代码**。这正是
TileLang 保留 expert 层的理由之一。

一个值得注意的补充：Triton 自己也意识到了这条路线在极端场景下的边界，并在文档里给出了
更低层的编程模型 **Gluon**（“Learn about Gluon's lower-level programming model in the
Gluon overview”，`../tilelang-corpus/11-related-compilers/raw/triton-docs.md`）。
这与 TileLang 从第一天起就走 expert 层形成了有意思的对照，但两者的时间线不同，
本书不做更多比较。

### 5.2 TileLang 的路线：把调度留在语言里，但给缺省值

官方文档把差异总结成了三条（`docs/deeplearning_operators/matmul.md:214-218`），
译为中文并补上代码位置：

1. **允许显式布局注解**（如 `make_swizzle_layout`），
   `T.annotate_layout` + `make_swizzled_layout`，见 `tilelang/language/annotations.py:45`
   与 `tilelang/layout/swizzle.py:66`；
2. **流水线既可自动推导、也可手工指定**，`T.Pipelined(..., num_stages=...)` 是自动路径，
   `stage=[...] / order=[...]` 是手工路径，见
   `_vendor/tilelang-v0.1.15/docs/programming_guides/software_pipeline.md:44-60`；
3. **允许在同一个程序里混用不同层次**，见本章 3.3 节的两个例子。

### 5.3 取舍对照

| 问题 | Triton（2019 论文的路线） | TileLang（v0.1.15） |
| --- | --- | --- |
| tile 是什么 | 一种**有形状的值**（statically shaped sub-arrays） | 一种**有归属和存储位置的缓冲**（一等对象，`T.alloc_shared/local/fragment`） |
| tile 内部布局 | 被语言藏起来，由编译器决定 | 可显式写（`T.annotate_layout` / `loop_layout`），不写则由 LayoutInference 推导 |
| 存储位置 | 不直接暴露 | 显式暴露（shared / fragment / local，以及各后端的 L1/L0A/L0B/L0C/UB 等） |
| 流水线 | 编译器决定 | `num_stages` 一行写死，或 `stage/order` 精细控制 |
| 线程级控制 | 2019 版语言里“CUDA-like concurrency primitives … are inexistent”（`:361-370`） | 提供（`T.get_thread_binding` + `T.alloc_local` + 内建） |
| 用户上手成本 | 低 | 中（要先理解存储层级） |
| 用户对关键决定的控制力 | 低 | 高 |
| 换来什么 | 代码短、可移植、编译器可以做大范围重写 | 决定可见、可复现、可下探；同一 kernel 可混层次 |
| 付出什么 | 关键决定不可见，遇到编译器覆盖不到的模式时缺少抓手 | 要学更多概念；写错了编译器帮不上忙（例如把 `T.Parallel` 的布局写错会直接报错） |

**一句话总结**：Triton 让编译器替你选调度，TileLang 让你可以选、也可以不选。
前者赌“编译器足够聪明”，后者赌“用户知道自己在做什么，并且需要把这件事写下来”。

### 5.4 这套设计的能力边界

没有免费的午餐。TileLang 的选择带来四项明确的限制，写在这里免得读者产生错误期待。

**（a）beginner 层还没有交付。** 官方文档白纸黑字写了
“This interface is not yet fully implemented”（`docs/get_started/overview.md:20`）。
同一份文档的另一处还提醒：Level 1 虽然对一般用户更舒适，但
“can incur longer auto-tuning times and may not handle some complex kernel fusion patterns
(e.g., Flash Attention) as easily”（`docs/deeplearning_operators/matmul.md:41`）。
所以“不想懂硬件也能写出快 kernel”在当前版本还不成立。

**（b）用户仍然要理解存储层级。** 显式存储分配是 hallmark（`overview.md:78`），
换句话说，不理解 shared memory 与寄存器文件的分工，就写不好 TileLang kernel。
这与 Triton 的“藏起来”正好相反。

**（c）手动布局写错不会静默降级，而是编译报错。** `T.Parallel` 的布局约束是硬性的
（`tilelang/language/loop.py:43-60`）。好处是问题暴露得早，坏处是学习曲线陡。

**（d）不同后端的方言不完全一致。** `tilelang.language` 是 CUDA 方言
（`tilelang/language/__init__.py:32`），昇腾等后端在此之上加自己的构造。
写可移植 kernel 时要留意自己用到的 API 属于 common 还是某个方言的扩展。

### 5.5 为什么这件事和你有关

到这里可以回答本章标题里的最后一个问题：**为什么一个只想写算子的人，需要关心设计哲学？**

因为这三件事直接影响你每天的调试与调优：

1. **当 kernel 不够快时，你知道该往哪个维度调。** 性能问题可以定位到四个维度中的某一个：
   是线程划分不合理（改 `threads` / `policy`）、布局有冲突（加 `annotate_layout`）、
   没走到张量核（检查 `T.gemm` 的操作数 scope 与 dtype）、还是搬运没被流水化
   （加 `num_stages`）。第 28 章会把这个决策流程做成方法论。
2. **当高层原语覆盖不了你的需求时，你知道可以下探，而不是换语言。** 3.3 节的两个例子
   说明：混用在 v0.1.15 里是被官方文档写进差异列表的能力，不是 hack。
3. **当你需要读编译器源码或改它时，你知道该看哪个 pass。** 解耦把“数据流”和“调度”
   分到了不同的 pass 里（4.3 节的顺序），改布局去看 LayoutInference（第 21 章）、
   改流水去看 PipelinePlanning（第 22 章）、改后端去看 codegen（第 24、25 章）。

---

## 6. 本章小结

* **核心主张**：TileLang 把**调度空间**（thread binding / layout / tensorize / pipeline）
  从**数据流**里解耦出来，用注解与原语表达，其余优化交给编译器
  （`../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577.md`）。
* **解耦的落地**：语言层只放 tile op（`tl.tileop.copy` / `tl.tileop.gemm`），
  硬件决定推迟到 pass；CUDA pipeline 的顺序是
  `PipelinePlanning → InjectSoftwarePipeline → LayoutInference → LowerTileOp`
  （`tilelang/cuda/pipeline.py:127-142`）。
* **四个维度都有入口，也都有缺省**：`threads` / `policy`、`annotate_layout`、
  `T.gemm` 及其变体、`Pipelined(num_stages=...)`。
* **三层接口**：beginner（未完成，方向体现在 eager 前端）、developer（tile 库，v0.1.15
  的主体）、expert（thread 原语与内建）。三层可以混用。
* **composable 的确切含义**：高层 tile 原语与低层 thread 原语可以出现在同一个 kernel、
  甚至同一段循环体里，不需要模式切换（`docs/deeplearning_operators/matmul.md:214-218`；
  实例见 `examples/dequantize_gemm/example_dequant_gemm_bf16_mxfp4_hopper.py:366-373`）。
* **tile 的定义**：一段有形状的数据 + 一处明确归属 + 一个显式存储位置；
  语言层体现为 `T.alloc_shared/local/fragment`，编译期体现为 LayoutInference 产出的
  `Fragment`。
* **与 Triton 的取舍**：Triton 藏起 intra-tile 的调度细节并自动推导 schedule；
  TileLang 把调度留在语言表面，可以介入也可以不介入，代价是必须理解存储层级。
* **能力边界**：beginner 层未完成；必须懂存储层级；布局写错会直接编译失败；
  后端方言不完全一致。

---

## 延伸阅读

* 论文摘要（本章 2.1 节的引用来源）：`../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577.md`。
* 三层接口的官方原文（含 beginner 未完成的标注）：
  `_vendor/tilelang-v0.1.15/docs/get_started/overview.md`。
* 三层与 TVM / Triton / CUDA 的类比，以及混用不同层次的官方表述：
  `_vendor/tilelang-v0.1.15/docs/deeplearning_operators/matmul.md`。
* Triton 的原始设计文档（tile 的定义、自动推导 schedule 的取舍）：
  `../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md`。
* 完整引用清单与核验状态：见本章 [references.md](references.md) 的 S1–S34 与 R1–R6。
* 下一章：第 04 章会把本章提到的 pass 串成一次完整的编译流程。
