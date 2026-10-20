# 第 05 章 横向对比：CUDA / Triton / CUTLASS-CuTe / TVM / cuTile / Ascend C / PTO-DSL

## 学习目标

读完本章，你应当能够：

1. 用“抽象层次”这把尺子把本章的七个对象放到同一张图上，并说出每一个层次
   **用户写什么、编译器替你做什么、谁来决定 tiling 与数据搬运**。
2. 拿一个统一的例子（带融合 epilogue 的 GEMM：`C = relu(A @ B)`）说明它在 CUDA、
   Triton、CUTLASS-CuTe、TileLang、Ascend C、PTO-DSL 里各要写多少东西，差别具体在哪。
3. 说清 TVM 在 TileLang 里的角色：它是**编译器底盘**（IR、pass 基础设施、target/
   device 抽象），不是 TileLang 的竞争者，也不是一个可以直接拿来写 AI kernel 的前端。
4. 说清 Ascend C 与 PTO-DSL 作为“昇腾原生路线”的定位，以及它们和 TileLang@Ascend
   的竞争/互补关系：谁提供 ISA，谁提供编译期调度。
5. 面对“我该用哪个”的问题，用本章的决策表给出一个有理由的答案，而不是凭印象站队。
6. 分清哪些结论有本地一手资料支撑，哪些只能标注“资料不足”，以及要补什么资料才能下结论。

## 前置知识

* 第 03 章：TileLang 的设计哲学与编程模型（tile / dataflow / schedule 解耦、三层接口）。
* 第 04 章：架构总览与一次编译的全流程（Python → TIR → pass → codegen → runtime）。
* 会用 Python、写过 PyTorch（能读懂 `A @ B`、`relu(x)`）。
* **不需要** CUDA、Triton 或 Ascend C 的编程经验。本章出现的每个 C/C++/Python 片段都
  会先说明它“属于哪一层”，再解释它在整个对比里的作用。

本章依赖的锁定版本见 [../../VERSIONS.md](../../VERSIONS.md)：TileLang `v0.1.15`、
TileLang-Ascend `TileLang-ascend-v0.1.2.000-release`。写作环境为 macOS (Apple
Silicon)、Python 3.11，**没有 NVIDIA GPU，也没有昇腾 NPU**；因此本章不给出任何
本机实测的性能数字，任何引用的数字都必须带来源与测量条件。

---

## 1. 问题的由来：为什么需要一张坐标系

一个刚会用 PyTorch 的人，第一次想把算子写快时，会同时听到七八个名字：手写 CUDA、
Triton、CUTLASS、CuTe、cutlass-cute、cuTile、TileLang、Ascend C、PTO-DSL、TVM……
它们看起来都能“写 kernel”，于是最常见的两个错误随之出现：

* **把它们当成同一层的竞品**：认为“Triton 是不是要取代 CUDA”“TileLang 就是另一个
  Triton”，于是无法解释为什么 TileLang 还在用 TVM，为什么 CUTLASS 还在用 CuTe，
  为什么昇腾上既有 Ascend C 又有 PTO-DSL。
* **只比较语法长短**：数谁的代码行数少，而不问“少掉的那些行由谁在什么时候补回来”，
  是编译器补、是库补、还是运行时补。代码短不等于事情被消灭，很多时候只是被搬到了编译期
  或运行时里。

本章要做的事，是给出**一把能反复使用的尺子**：用“抽象层次”把每个对象放到它应该在的
位置上，然后用一个统一的例子把它们串起来。这样你在第 14、15 章真正动手写 GEMM /
Attention 时，才知道自己站在哪一层、TileLang 帮你吃掉了哪一段、剩下的哪一段还得自己写。

一句话预告本文的结论：**这七个对象不在同一层**。它们大致分布在“线程/指令层”
（CUDA、Ascend C、PTO-DSL、CUTLASS-CuTe 的 C++ 与 DSL）与“tile 层”
（Triton、cuTile、TileLang）两级，而 TVM 根本不在这条纵轴的正中间。它是一条横穿
两级的**地基**。TileLang 的独特之处，是同时坐在 tile 层（前端）和底盘层（它自己写 pass）。

---

## 2. 概念与模型：三层抽象，七个对象

### 2.1 抽象层次的定义

本书把“抽象层次”定义为：**用户为表达一个算子所必须提供的、与硬件相关的决策数量**。
数量越少，层次越高。三个关键决策是：

1. **谁负责搬数据**：用户显式写 `copy` / `DataCopy` / TMA，还是编译器从数据流反推；
2. **谁负责切 tile**：用户给 tile 尺寸，还是编译器 / 自动调优搜索；
3. **谁负责同步**：用户写 barrier / `EnQue/DeQue` / `set_flag/wait_flag`，还是编译器插入。

按这三个问题，本章的七个对象可以放在一条纵轴上（从低到高）：

```text
高层  ▲  图级 / 模型级 ：   torch.compile（不在本章范围，见第 02 章）
      │
      │  tile 级        ：   Triton      cuTile      TileLang
      │                    （块 + 编译器选调度）   （块 + 用户显式数据流）
      │
      │  线程 / 指令级  ：   CUDA        CUTLASS-CuTe（C++ 与 CuTe DSL）
      │                    Ascend C      PTO-DSL
      ▼
低层     硬件 ISA       ：   PTX / SASS        PTO-ISA / CCE

横穿所有层的底盘：  TVM（IR + pass 基础设施 + target/device 抽象）
```

注意三件事，它们后面会反复用到：

* **CuTe 与 CuTe DSL 是一族**（CUTLASS 4 把 CuTe DSL 定义为“与 CuTe C++ 抽象完全一致的
  低层编程模型”，来源 `../tilelang-corpus/05-nvidia-backend/raw/github-nvidia-cutlass.md:39-51`），
  所以在抽象层次上它们和 CUDA 是同一级，而不是和 Triton 同一级。
* **Ascend C 与 PTO-DSL 也在低层**，但它们暴露的不是 NVIDIA 的 shared/register，而是
  昇腾特有的 L1 / L0A / L0B / L0C / UB 与 MTE 搬运流水。
* **TVM 不在这条纵轴上**。它既可以被 CUDA 一侧的代码使用，也可以被昇腾一侧使用；它是
  横切面。第 4 节专门讲这一点。

### 2.2 七个对象一览

下表是本章的主索引。每一行对应一个对象，列就是本章要求你建立的坐标系要素。
“与 TileLang 的关系”一列里，**竞争**指两者在同一层抢同一件事，**互补**指两者在不同层
可以叠在一起用，**底盘**指它不是同类物。

| 对象 | 抽象层次 | 用户写什么 | 编译器替你做什么 | 适用场景 | 生态与成熟度 | 与 TileLang 的关系 |
| --- | --- | --- | --- | --- | --- | --- |
| **CUDA** | 线程/指令级 | grid/block/warp/thread 映射、`__shared__`、`__syncthreads()`、`cp.async`/TMA | 把 SPMD C++ 编成 PTX/SASS；不做 tile 级推导 | 需要极限控制、非常规算子、教学 | 最成熟，CUDA 生态的底座（NVCC、cuBLAS、cuDNN、NCCL 都在其上） | **互补**：TileLang 的 NVIDIA 后端最终产出 CUDA C |
| **Triton** | tile 级 | Python 里的块级程序（`tl.load`/`tl.dot`/`tl.store`），块尺寸由 autotune 搜索 | tile 级优化 pass：内存合并、共享内存分配与同步、流水 | 通用算子快速出好性能，尤其是内存受限与归约类 | 生态广泛（PyTorch/Inductor 侧大量使用）；文档见 R4 | **竞争**：最直接的对手，都在 tile 层 |
| **CUTLASS / CuTe** | 线程/指令级 | C++ 模板：`Layout`、`Tensor`、MMA/Copy atom；CUTLASS 4 起还有 CuTe DSL | CuTe 做 layout 代数与索引；CUTLASS 提供分层 GEMM 组件与 kernel 模板 | 生产级 GEMM/卷积，需要贴合每一代 tensor core | NVIDIA 官方库，README 自列为 Volta→Blackwell（含 Ada）；CuTe DSL 于 CUTLASS 4 发布、自述公测（R5） | **互补**：TileLang 的 `T.gemm` 在 NVIDIA 侧就落到 CuTe/MMA 上 |
| **TVM** | 编译器底盘（横切） | 一般用户不直接写；研究者写 TE/TIR/TIRx 与 schedule | IR、pass 基础设施、target/device 抽象、自动调优（Ansor/MetaSchedule） | 做编译器、加后端、做自动调度研究 | Apache 顶级项目，长期维护；TIRx 是其 tile 方向的新接口（R10） | **底盘**：TileLang 建在 TVM 之上 |
| **cuTile** | tile 级 | Python 里的 tile 程序（据第三方描述，与 CuTe DSL / Triton 同级） | 自述为“类似 cuTile 但原生 NPU”的反面参照；细节**资料不足** | NVIDIA 新一代 Python tile DSL | **资料不足**：本 corpus 只有指向 `docs.nvidia.com/cuda/cutile-python/` 的链接（R17），PTX ISA 里出现 `python-cutile` 语言码（R9） | **竞争**：在 NVIDIA tile 层与 TileLang 正面重叠 |
| **Ascend C** | 线程/指令级 | `__global__ __aicore__` 核函数；`TPipe`/`TQue`；`CopyIn/Compute/CopyOut`；`EnQue/DeQue` 与 `set_flag/wait_flag` | bisheng 编译器把 CCE C++ 编成 AI Core 指令；不替你做 tile 级推导与跨流水同步计划 | 昇腾算子的原生开发语言，CATLASS 等模板库建其上 | 华为官方主推，CANN 的一部分（R14、R15） | **竞争 + 互补**：TileLang@Ascend 的 Ascend C&PTO 路线最终产出 Ascend C；TileLang 替掉的正是它的同步（R3/R22） |
| **PTO-DSL** | 线程/指令级 | Python：`pto.load`/`tile.matmul`/`pto.store`，显式 L1/L0A/L0B/L0C/UB | PTOAS 的 InsertSync 自动插同步；自动软件流水；编译到 PTO-ISA | 想用 Python 写昇腾、又要贴硬件原语的场景 | 华为实验室开源，自述目标对标硬件 intrinsic 性能；2026/05 起并入 PTOAS 主仓（R17） | **竞争 + 互补**：与 TileLang@Ascend 抢同一批用户；TileLang 也可把 PTO-ISA 当 codegen 后端（R17） |

表里每个“功能”级别的事实都能落到来源：CUDA 的同步与共享内存机制见 R7/R8；Triton 的
tile 语义与编译器 pass 见 R2；CUTLASS 的 CuTe DSL 定位见 R5；TVM 的 TIRx 见 R10；
Ascend C 的 `TPipe`/`TQue` 范式见 R14；PTO-DSL 的定位见 R17。**cuTile 一行的内容最少，
因为本地资料最少。这一点在第 5 节会单独交代。**

---

## 3. 用法：同一个 GEMM，六种写法

统一例子选 **带融合 epilogue 的 GEMM**：`C = relu(A @ B)`，`A`、`B` 为 fp16，
累加与输出为 fp32，形状 `M = N = K = 1024`，tile 取 `128×128×32`。这组参数不是随手取的：
它正是 TileLang 官方 quickstart 示例的默认配置（来源
`_vendor/tilelang-v0.1.15/examples/quickstart.py:51-56`），所以下面每一版都可以和 TileLang
这一版逐行对照。

选“融合 epilogue”而不是纯 GEMM 的原因：它同时考验三件事，(1) 能不能做 tensor core /
Cube 上的矩阵乘；(2) 能不能把 `relu` 合进同一个 kernel，省掉一次 HBM 往返；(3) 能不能
在矩阵乘结束后让计算单元与搬运流水正确衔接。这三件事正好能把不同层次的差异暴露出来。

### 3.1 CUDA：从线程写起

CUDA 是这一族的“汇编之上”的地板。用户自己回答第 2.1 节那三个问题：自己决定 block 的
二维划分，自己用 `__shared__` 开 tile，自己在 `for` 循环里 `__syncthreads()`，
epilogue 由自己写在线程里。NVIDIA 官方《CUDA C++ Best Practices Guide》给过一个用
共享内存做分块矩阵乘的示范，其核心是：

> `__shared__ float aTile[TILE_DIM][TILE_DIM];`
> 来源：`../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md:1745-1830`
> （“Using shared memory to improve the global memory load efficiency in matrix
> multiplication”，CUDA C++ Best Practices Guide）。

也就是说，在 CUDA 这一层，“把 tile 放进 shared memory”是**用户动作**，不是编译器的义务。
本 tilelang-book 的 `code/01-unified-gemm-relu/cuda_gemm_relu_illustrative.cu` 按这个结构
写出了带 ReLU epilogue 的版本，并**明确标注为教学用伪代码**：它演示层次，不是可直接
编译的生产代码。

需要强调：CUDA 本身**没有 tile 概念**。这正是 Triton 论文要解决的问题，见下一节。

### 3.2 Triton：把 tile 写进语言，让编译器选调度

2019 年的 Triton 原始论文把 tile 定义为“**statically shaped multi-dimensional
sub-arrays**”（静态形状的多维子数组），来源
`../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md:37`。用户写的是一个
“单线程、但会被自动并行化”的块级程序，编译器负责机器相关优化。

论文里那段矩阵乘是 Triton-C（一种 CUDA 风格的 C 方言），核心几行是（原文 Listing 1）：

```c
float C[TM, TN] = 0;
float A[TM, TK] = check_a ? *pa : 0;
float B[TN, TK] = check_b ? *pb : 0;
C += dot( A, trans( B ) );
```

来源：`../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md:130-200`
（Listing 1，“C = A×BT in Triton-C”）。Triton-JIT 的机器相关 pass 恰好就是 CUDA 用户要
手写的那几件事，论文列得很清楚：(1) 分层 tiling、(2) 内存合并、(3) 共享内存分配、
(4) 共享内存同步（barrier 插入），来源
`../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md:583-720`。

**必须注明一个版本事实**：2019 年的 Triton 语言比今天小得多。论文自己在结论里写，
未来的工作包括“support for tensor cores”和“implementation of quantized kernels”，
来源 `../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md:943-946`，也就是
说，**当年的 Triton-C 还没有张量核支持**。今天 Triton 的张量核/`tl.dot`、FP8、以及
更低的 Gluon 层都远远超出了这篇论文（今天的文档索引里能看到 `triton.language` 与
Gluon，来源 `../tilelang-corpus/11-related-compilers/raw/triton-docs.md:11-35`）。
所以引用 2019 论文时只能说清“tile 抽象与编译器分工的设计来源”，不能拿它当今天 Triton
的能力清单。

今天的 Triton 用 Python 写，形如（**本书示意，按论文与官方文档的语义重写，非 corpus 内原文**）：

```python
# 示意伪代码：Triton 的 tile 级 GEMM + 融合 ReLU
a = tl.load(A_ptrs)                # 块级 load，编译器决定合并/共享内存
b = tl.load(B_ptrs)
acc = tl.dot(a, b, acc)            # 块级矩阵乘，落到 tensor core
acc = tl.maximum(acc, 0.0)         # 融合 ReLU，仍在块级
tl.store(C_ptrs, acc)
```

关键差异在**决策权**：Triton 里你写“块大小”，但共享内存怎么用、什么时候插 barrier、
要不要多缓冲，基本交给编译器（论文把这个思路讲得很明确：单线程程序 + 自动并行化 +
编译器负责同步）。TileLang 反过来，把这些决策**部分留在语言里**。

### 3.3 CUTLASS / CuTe：在低层把生产 GEMM 拆成可复用组件

CUTLASS 的定位是“在 CUDA 里实现高性能 GEMM 及其相关计算的全套抽象”，把 GEMM 拆成
可在不同并行层次上专门化、调优的组件（来源
`../tilelang-corpus/05-nvidia-backend/raw/github-nvidia-cutlass.md:19-33`）。CUTLASS 4 起
加上了 **CUTLASS DSLs**，第一个就是 **CuTe DSL**：一个“与 CuTe C++ 抽象完全一致的低层
编程模型，暴露 layout、tensor、hardware atom，以及对硬件线程与数据层级（hierarchy）
的完全控制”，来源 `../tilelang-corpus/05-nvidia-backend/raw/github-nvidia-cutlass.md:39-51`。

CuTe 的核心抽象是“hierarchically multidimensional layouts of threads and data”：
用 `Layout` 与 `Tensor` 同时描述数据和线程的多维布局，并做 layout 代数与索引推导，
来源 `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cutlass-media-docs-cute.md:10-12`。

这意味着 CuTe DSL **不是** tile 层。它是把 CuTe C++ 的布局代数搬到了 Python，但用户仍然
要自己写 `tidx`、自己调 `cute.arch.barrier()`、自己用 `cute.autovec_copy` 搬元素。
一个真实的 CuTe DSL 片段（TileFoundry 的 GEMV kernel）长这样：

```python
@cute.jit
def _load_and_normalise(x, gamma, smem, scratch, tidx, k, eps, nt):
    nch = k // VEC
    xv = cute.tiled_divide(x, (VEC,))
    frag = cute.make_fragment(VEC, BF16)
    acc = F32(0.0)
    for i in cutlass.range_constexpr(cdiv(nch, nt)):
        j = i * nt + tidx
        if j < nch:
            cute.autovec_copy(xv[None, j], frag)
            ...
    total = _block_sum(acc, tidx, nt // 32, scratch)
```

来源：
`../tilelang-corpus/07-tilelang-ascend-adapter/raw/TileFoundry/examples/minicpm3_4b-cutedsl/kernels/cute_gemv.py:36-113`。

对照一下：这里出现的是 `tidx`、`warp_reduction_sum`、`barrier`、`autovec_copy`，全是
线程/指令级词汇。而 TileLang 的同一件事只需要 `T.gemm` + 一行 `T.copy`。所以
CUTLASS-CuTe 和 TileLang 在**同一份程序里的不同层**：TileLang 的 `T.gemm` 在 NVIDIA 侧
正是落到 CuTe/MMA 上的（quickstart 注释原文：“Currently we dispatch to the cute/hip on
Nvidia/AMD GPUs”，来源 `_vendor/tilelang-v0.1.15/examples/quickstart.py:38`）。

> **资料不足**：CUTLASS 里“融合 epilogue 的 GEMM”在 C++ 与 CuTe DSL 两侧的完整、
> 可运行写法，本 corpus 没有收录（只有 README 介绍与 CuTe 教程目录链接）。要给出可
> 对照的真实 epilogue 代码，需要 CUTLASS 仓库 `examples/` 下的完整源文件，或
> `media/docs/cpp/cute/0x_gemm_tutorial` 的全文。

### 3.4 TileLang：把数据流留在语言里，把调度交给 pass

回到统一例子。TileLang 的这一版就是官方 quickstart（来源
`_vendor/tilelang-v0.1.15/examples/quickstart.py:1-46`），去掉注释后的骨架是：

```python
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
        C_local[i, j] = T.max(C_local[i, j], 0)   # 融合 ReLU
    T.copy(C_local, C[by * block_M, bx * block_N])
```

逐点对照第 2.1 节那三个问题：

* **谁搬数据**：用户写 `T.copy`（`_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:59`），
  但“搬成什么指令”（cp.async / TMA / 向量访问）由 codegen 决定；
* **谁切 tile**：用户给 `block_M/block_N/block_K` 与 `num_stages`，也就是用户决定 tile
  形状，编译器负责把它变成 thread binding 与多缓冲；
* **谁同步**：**都不写**。TileLang 的 CUDA pipeline 里由 `PipelinePlanning` →
  `InjectSoftwarePipeline` → `LayoutInference` → `LowerTileOp` → …… → `ThreadSync`
  一串 pass 插入同步（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:129-142`、
  `:258-259`）。

这就是 TileLang 相对 Triton 的取舍：**Triton 把 tile 与调度都让编译器定，TileLang 把
数据流（每片数据从哪来、去哪、在哪一级算）写进语言，把剩下的布局/流水/同步交给 pass。**
quickstart 里那行 ReLU 正好说明“融合 epilogue”在 tile 层有多便宜：它只是对
`C_local`（fragment）做一次元素级变换，再 `T.copy` 出去，不需要新开一个 kernel。

### 3.5 Ascend C：昇腾官方原生语言

Ascend C 是华为官方的算子开发语言，编译运行在昇腾 AI 处理器上（CANN 的一部分，
来源 `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/developer/zh_developer_techArticles_20230830-1.md:29`）。
它把算子实现拆成 **CopyIn / Compute / CopyOut** 三段，用 `TPipe` 管内存、用
`TQue`（`EnQue`/`DeQue`）连接三段做同步与多缓冲（来源同文件 `:51-57`）。

一个真实的 Ascend C 片段（官方的 LeakyReLU 样例）：

```cpp
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, ...) {
    xGm.SetGlobalBuffer((__gm__ half*)x + this->blockLength * get_block_idx(), this->blockLength);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
}
__aicore__ inline void CopyIn(int32_t progress) {
    LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    DataCopy(xLocal, xGm[progress * tileLength], tileLength);
    inQueueX.EnQue(xLocal);
}
__aicore__ inline void Compute(int32_t progress) {
    LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
    LeakyRelu(yLocal, xLocal, scalar, tileLength);
    outQueueY.EnQue<half>(yLocal);
}
```

来源：
`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/developer/zh_developer_techArticles_20230830-1.md:350-440`。

注意 `BUFFER_NUM`：多缓冲在这里是**用户显式声明**的东西（`loopCount = tileNum *
BUFFER_NUM`，`:376-386`）。这正是 TileLang@Ascend 想拿走的部分。它把缓冲与同步建模成
编译期调度问题。TileLang 官方 Ascend 文档有一张直接对照表：用户只写 `T.alloc_*`、
`T.copy`、`T.dual_copy`，而“Ascend C code you can skip”一栏列的就是 `TPipe`/`TQue`
与 DMA 调用、Set/Wait flag 与 flag ID 管理，Synchronization 一行的原文是
“None (Just write operations in order!)”（来源
`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:141-151`）。

> **资料不足**：本 corpus 没有 Ascend C 版本的“GEMM + 融合 epilogue”完整样例（上面的
> 是逐元素算子）。要逐行对照 GEMM 的 Cube 指令与 epilogue，需要 CANN 官方 GEMM 样例或
> CATLASS 的具体 kernel 源码（CATLASS README 只给了仓库入口，来源
> `../tilelang-corpus/06-huawei-ascend/raw/gitcode-cann-catlass.md`）。

### 3.6 PTO-DSL：用 Python 写昇腾，但仍然贴硬件原语

PTO-DSL 自述是“类似 cuTile、但原生 NPU”的 Python 接口与 JIT 编译器，目标是提供
“低层、显式、NPU 原生的原语”，性能对标直接写硬件 intrinsic（来源
`../tilelang-corpus/06-huawei-ascend/raw/pto-dsl/README.md:4-7,46-48`）。它和 TileLang@Ascend
的关系，PTO-DSL 自己说得很直接：tilelang 也可以把 PTO-ISA 当 codegen 后端，而 PTO-DSL
故意暴露更低层的控制。例如 L2 swizzling 在 tilelang 里是一行 `T.use_swizzle`，在
PTO-DSL 里要用户自己写（来源同上 `:49`）。

它的 matmul 教程给出了一条从零优化的路径，第一步是一段约 80 行的 Python，结构是
显式的三级循环（core 循环 → `GM→L1` → `L1→L0`），核心 API 是：

```python
for li in pto.range(bid, core_loop, num_blocks):   # persistent：每核迭代取 tile
    ...
    for k_idx in pto.range(c0, k_dtile_num, c1):
        for phase in range(8):                      # L1→L0 静态展开
            tile.extract(a_l1, c0, a_col, a_l0)     # TEXTRACT
            tile.extract(b_l1, b_row, c0, b_l0)
            tile.matmul_acc(c_l0, a_l0, b_l0, c_l0) # TMATMUL_ACC
```

来源：
`../tilelang-corpus/06-huawei-ascend/raw/pto-dsl/examples/aot/matmul_optimization_guide/step1_baseline.py:34-125`
与 `.../matmul_optim_guide_zh.md`（说明 `pto.load`=`TLOAD`、`tile.extract`=`TEXTRACT`、
`tile.matmul`=`TMATMUL`，以及 `GM→L1→L0A/L0B→Cube→L0C→GM` 的搬运路径）。

PTO-DSL 的定位因此很清楚：它**在 TileLang 之下**，用户仍要显式写出 L1/L0 的搬运与
`extract`。它自述的卖点是“自动软件流水、无需手动同步”，由 PTOAS 的 InsertSync pass
实现（来源 `../tilelang-corpus/06-huawei-ascend/raw/pto-dsl/README.md:9-12`）。
一个能说明“昇腾原生 FP 语义”的细节：PTO 规范把 64 个 Tile 寄存器、118 条 Direct Tile
operation、execution mask 都形式化了（来源
`../tilelang-corpus/06-huawei-ascend/raw/pto-spec/README.md:15-40`）。

### 3.7 一张对照表

把“带融合 ReLU 的 GEMM”这件事横着摊开：

| 你要提供的 | CUDA | Triton | CUTLASS-CuTe | TileLang | Ascend C | PTO-DSL |
| --- | --- | --- | --- | --- | --- | --- |
| 任务/块划分 | grid + block | 块尺寸（可 autotune） | 分层 tile（可 policy 化） | `T.Kernel` + tile 尺寸 | 核函数 + `get_block_idx` | persistent `pto.range` |
| 片上存储放置 | `__shared__` 手写 | 编译器决定 | 用户按 CuTe 层级放置 | `T.alloc_shared`（声明，pass 决定布局） | `TPipe.InitBuffer` 手写 | `pto.alloc_tile` 手写 |
| 矩阵乘 | 手写 MMA/tensor core 序列 | `tl.dot` | CuTe atom / CUTLASS 组件 | `T.gemm` | Cube 指令 / 库 | `tile.matmul` |
| 多缓冲 | 手写 | 编译器（流水 pass） | 用户/模板 | `T.Pipelined(num_stages=k)` | `BUFFER_NUM` 手写 | 自动 + 可选手写 |
| 同步 | `__syncthreads`/mbarrier | 编译器插入 | 用户/atom | 编译器插入 | `EnQue/DeQue`、Set/Wait | PTOAS InsertSync 自动 |
| 融合 ReLU | 手写 epilogue | 块上再做一次 | epilogue 组件 | `T.Parallel` 里一行 | 手写 Vector 段 | 手写 Vector 段 |
| 抽象层次 | 低 | 高 | 低 | 高（但数据流显式） | 低 | 低 |

一句话读法：**行越往下的是“用户要提供的东西”，列越往右的越自动化**。但“自动化”不等于
“更短”，自动化只是把决定权从你手里搬到了编译器里，代价是编译期复杂度（见第 4 节）。

---

## 4. 实现机制：TVM 是底盘，不是竞争者

### 4.1 TVM 提供什么

TVM 是一个端到端的深度学习编译器，暴露图级与算子级优化，为不同硬件后端提供性能可移植性，
并用学习式代价模型自动探索低级优化，来源
`../tilelang-corpus/04-language-and-ir/raw/arxiv-1802-04799.md`（TVM，OSDI 2018 摘要）。
它带来的三样东西构成了 TileLang 的地基：

1. **IR**：TensorIR 把“循环嵌套”升级成以张量计算为一等公民的 block/axis 表示，来源
   `../tilelang-corpus/04-language-and-ir/raw/arxiv-2207-04296.md`（TensorIR, ASPLOS 2023）。
   TileLang 的 IR 就是 TVM 的 TIR（v0.1.15 已迁到 TIRX 表示，见第 30 章）。
2. **pass 基础设施**：一个 IRModule 上串起一长串 pass 的能力。这正是 TileLang 的
   `LowerTileOp` / `LayoutInference` / `ThreadSync` 的落点。
3. **target / device 抽象**：把“给哪个后端生成代码”与“用什么运行时加载”分开。
   TileLang 在此基础上做了自己的 backend 化改造(见 4.3)。

TVM 自己也在往 tile 层演化：上游文档里已经出现 **TIRx**，包含“Tensor Layout”“Tile
Primitives”“Tile Primitive Dispatch”等章节，来源
`../tilelang-corpus/11-related-compilers/raw/tvm-docs.md`（TIRx 小节）。但即便如此，
TVM 的自我定位仍是**编译基础设施**。它提供的是一套能装各种前端/后端的骨架，而不是
“推荐你直接用 TVM 写大模型算子”的产品级前端。这就是本章把 TVM 标成“底盘”而不是
“竞争者”的依据。

### 4.2 为什么“TVM 是竞争者”是错的

如果 TVM 是竞争者，那么 TileLang 就应该自己写一套 IR 和 pass 框架。事实相反：TileLang
把几乎全部结构调整都表达成 TIR pass。举例来说，昇腾侧那条自动调度流水是这样的
（来源 `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:108-114`）：

```text
EstimateLatency → AutoSchedule → PrepareMultiBuffer → InsertSync → MaterializeMultiBuffer
```

其中 `AutoSchedule` 背后是 Z3（SMT 求解器），`_vendor/tilelang-v0.1.15/tilelang/ascend/transform/z3_scheduler.py:1-26`
开头就写着“Z3-based scheduler for auto-scheduling”。这些 pass 之所以能存在，是因为
TVM 先把 IR 与 pass 机制准备好了。换句话说：**TileLang 的竞争力，很大程度上是“知道在
TVM 这条底盘上把哪些新 pass 装进去”**。把它和 TVM 对立起来，等于否定它自己的实现方式。

### 4.3 TileLang 的 backend 化改造

v0.1.15 里的 TileLang 已经把自己讲成“multi-backend compiler (TileLang-X)”：每个
target backend 被视为一条“compiler vertical slice”，自己拥有语言方言、pass 流水、
host/device codegen，而 build/load/launch 交给可复用的 execution backend（来源
`_vendor/tilelang-v0.1.15/tilelang/backend/README.md:1-27`）。官方 README 的
Platform/Backend 表把后端分成 Primary / Supported / Experimental / Ecosystem 四档，
其中 NVIDIA CUDA 是 Primary，Huawei Ascend 950 是 Supported，NVIDIA CuTe DSL 是
Experimental（来源 `_vendor/tilelang-v0.1.15/README.md:113-128`）。

这张表恰好回扣本章的坐标系：

* `cutedsl` 后端的存在，说明 TileLang 把 CuTe DSL 当成**下层的可替换 codegen**
  （`_vendor/tilelang-v0.1.15/tilelang/cuda/cutedsl_backend.py:1-20` 把 cutedsl 注册成
  复用 CUDA pipeline 的 backend；`tilelang/contrib/cutedsl/gemm_tcgen05.py:1-13`
  提供 Blackwell 的 tcgen05 MMA 原语）。也就是说 TileLang 与 CUTLASS-CuTe 的关系是
  **互补**，不是二选一。
* `ascend` 后端对应的是昇腾主线（950），而 A2/A3 的 `ascendc`/`pto`/`npuir` 是
  Ecosystem 档（在独立仓库 `tilelang-ascend` / `tilelang-mlir-ascend` 里，来源
  `_vendor/tilelang-v0.1.15/README.md:129-132`）。

### 4.4 cuTile 与 CuTe DSL：NVIDIA 自己的新 tile 层

TileLang 在 NVIDIA 侧最需要警惕的对手，是 NVIDIA 自己在推的 Python tile DSL。
本地能确证的事实只有三条：

1. CUTLASS 4 引入了 CUTLASS DSLs，第一个是 CuTe DSL，自述为“与 CuTe C++ 抽象完全一致
   的低层编程模型”（R5），**它并不是 tile 层**；
2. TileLang 有一个 `[RoadMap] CuteDSL Backend` issue，把 CuTe DSL 称作
   “NVIDIA's newer Python tile DSL (cuTile/CuTe DSL)，which is the most serious
   competitor on the NVIDIA side”（来源
   `../tilelang-corpus/08-frontier-unmerged/raw/github-issue-1454.md`）；
3. PTO-DSL 的 README 反复把 cuTile 当作参照物，并给出链接
   `https://docs.nvidia.com/cuda/cutile-python/`（R17）；PTX ISA 的语言码表里出现了
   `8 | "python-cutile"`（来源
   `../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md:38645-38658`），说明
   “python-cutile”是被 PTX 工具链识别的源语言之一。

> **资料不足（cuTile）**：本 corpus **没有** cuTile 的官方文档快照，也没有任何 cuTile
> 代码样例。上面三条只够说明“cuTile 作为 NVIDIA 新一代 Python tile DSL 存在，且与
> CuTe DSL 被并列讨论”。要给出 cuTile 的“用户写什么 / 编译器替你做什么”，必须补齐
> `docs.nvidia.com/cuda/cutile-python/` 的正文与至少两个官方样例。在此之前，任何关于
> cuTile 语言细节的陈述都只能是猜测，本章不予给出。

---

## 5. 什么时候该选谁：决策表

把前面所有内容压成一张可以照着做的表。使用顺序：先确定**硬件**，再确定**你要写的是
普通算子、还是要在编译器/后端里做事**，最后看**你愿意让出多少控制权**。

| 你的处境 | 首选 | 理由（可回溯到本文哪节/哪条来源） | 不要用它来做 |
| --- | --- | --- | --- |
| 想快速给一个具体算子拿到接近库的性能，规模通用 | **Triton**（GPU）/ **TileLang**（要更细的数据流控制或要跨 NVIDIA+昇腾） | 两者都在 tile 层；Triton 让编译器定调度（R2/R4），TileLang 把数据流留在语言里（S1、`quickstart.py`） | 需要手写非常规同步协议的实验 |
| 要在 NVIDIA 上对齐 cuBLAS/CUTLASS 级别的 GEMM/卷积，且要跟着每一代 tensor core 走 | **CUTLASS / CuTe**（含 CuTe DSL） | CuTe 的 layout 代数与分层组件是官方路线（R5/R6） | 想要“几十行写一个算子” |
| 要写 TileLang 之外、但比 CUDA 高一层的东西，且愿意让编译器插同步 | **TileLang** | `T.gemm`/`T.copy`/`T.Pipelined` + 编译器插同步（S1、`cuda/pipeline.py`） | 需要直接控制每条 PTX/mbarrier 时序 |
| 要改编译器、加 pass、做自动调度研究 | **TVM**（作为底盘）+ TileLang（作为上层） | TVM 提供 TIR/pass/target（R10/R11/R12），TileLang 在其上装 pass（S10/S11/S12） | 直接用 TVM 当“大模型算子前端” |
| 在昇腾上要贴硬件、要极致控制 | **Ascend C** 或 **PTO-DSL** | 两者都暴露 L1/L0/UB 与 MTE（R14/R17/R18） | 想自动调度 |
| 在昇腾上要自动调度、自动同步，愿意用 tile 前端 | **TileLang@Ascend** | 自动 layout/schedule/sync（`tilelang/ascend/README.md:141-151`、`ascend/pipeline.py`） | 需要访问后端暂不支持的非常规指令 |
| 要把 kernel 融进 PyTorch 训练/推理 | TileLang / Triton / CUTLASS 都可能，取决于**图的哪一部分** | 图级融合仍归 torch.compile/Inductor（第 02 章） | 用 kernel DSL 去做图级优化 |

两条使用建议：

* **不要用“代码行数”选工具**。本节引用的社区文章（R22）确实说过 TileLang@Ascend“代码
  行数差了好几倍”、性能“和主线 Ascend C 打得有来有回”，但作者自己也承认没有公开基线
  与测量口径，本书只把它当定性描述。（R22 是社区一手文章，非厂商文档。）
* **不要跨层比较“谁更强”**。比较 CUDA 和 Triton 的“性能”，就像比较汇编和 C 的性能，
  只有在同一目标、同一优化预算、同一测量口径下才有意义。

---

## 6. 进阶与陷阱

1. **“2019 年的 Triton”不是“今天的 Triton”。** 引用原始论文时，注意它的贡献是
   Triton-C + LLVM 式 IR + tile 级 pass + autotuner，而张量核支持是论文里写明
   “未来的工作”（R2:943-946）。用 2019 论文评价今天 Triton 的能力是时代错位。
2. **“没有同步”不等于“没有代价”。** TileLang 的 `T.Pipelined` 在昇腾侧背后是
   EstimateLatency → AutoSchedule(Z3) → PrepareMultiBuffer → InsertSync → Materialize
   一整条流水（S11）。代码里看不到 `set_flag/wait_flag`，是因为它们被生成出来了，而不是
   不存在。
3. **不要把 CuTe DSL 当成“NVIDIA 的 Triton”。** CuTe DSL 自述与 CuTe C++ 抽象一致、
   暴露 hardware atom 与线程/数据层级（R5）；从真实片段看，它是要写 `tidx` 和
   `cute.arch.barrier()` 的（R27）。它和 cuTile 是两回事。
4. **昇腾有两条 950 路线，选哪条要问维护者。** issue #3317 就是用户问“PR3308 和 PR3310
   两个 Ascend 950 后端该用哪个”。但本 corpus 抓到的快照里**只有提问、没有回答**，
   所以本章不臆测官方建议；见“待补资料”。
5. **判据优先级**：先看硬件（NVIDIA / 昇腾 / 其他）→ 再看你要改的是算子还是编译器 →
   最后才看语法。跳过前两步直接比语法，得到的结论通常不可靠。

---

## 7. 本章小结

* 本章用“用户必须提供的硬件相关决策数量”定义抽象层次，把七个对象放到一条纵轴上：
  **线程/指令级**（CUDA、CUTLASS-CuTe、Ascend C、PTO-DSL）↔ **tile 级**（Triton、
  cuTile、TileLang）；**TVM 是横穿两级的底盘**。
* **统一例子（`C = relu(A @ B)`）**：CUDA 要手写 shared memory 与同步；Triton 用
  `tl.dot` 让编译器管调度；CUTLASS-CuTe 仍在低层，但给你可复用的布局与组件；TileLang
  用 `T.gemm` + `T.copy` + `T.Pipelined` 把数据流写出来、把布局/流水/同步交给 pass；
  Ascend C 用 `TPipe`/`TQue` 手写三段流水；PTO-DSL 用 Python 但仍显式写 L1/L0 搬运。
* **TVM 不是竞争者，是底盘**：TIR/pass/target 都来自 TVM，TileLang 的全部魔法都是装进
  这条底盘的 pass（layout inference、autoschedule、sync insertion）。
* **Ascend C 与 PTO-DSL 是昇腾原生路线**：TileLang@Ascend 与它们**竞争用户、互补层次**，
  TileLang 可以产出 Ascend C，也可以把 PTO-ISA 当 codegen 后端。
* **决策表**：先硬件、再“算子还是编译器”、最后“让出多少控制权”。
* **证据边界**：cuTile 与 CUTLASS epilogue 细节本地资料不足（见 references.md 的
  “待补资料”）；社区文章的“性能打平”说法没有公开基线，只作定性引用。

## 延伸阅读

* TileLang 的 tile 抽象与三层接口：第 03 章；论文摘要见 R1。
* TVM/TensorIR 的 IR 与 pass 基础：第 20 章；资料见 R10、R11、R12。
* NVIDIA 后端（CuTe/MMA/TMA）：第 24 章。
* 昇腾后端（Ascend C & PTO vs AscendNPU IR）：第 25 章；资料见 R14、R17、R18、R20。
* 本章所有引用的完整清单与核验状态：见 [references.md](references.md)。
