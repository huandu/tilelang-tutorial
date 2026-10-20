# TileLang 知识体系与学习路径

读者设定：有编程基础，会用 PyTorch，但没写过 CUDA kernel，也没学过编译器。
目标有三个终点，按难度递增：

* **T1 算子工程师**：能用 TileLang 写出达到生产水平的大模型算子（GEMM / attention /
  MoE / 量化），能定位并修复性能问题。
* **T2 编译器工程师**：能读懂并修改 TileLang 的编译期行为（加 pass、改 layout
  inference、加语言原语），理解每个 IR 变换做了什么。
* **T3 后端工程师**：能为一个新硬件（昇腾优先）接后端，或深度迭代现有昇腾后端。

下文所有“资料”都指本 corpus 里的本地文件，路径相对 `.workspace/research/tilelang-corpus/`。

---

## 0. 先建立正确的心智模型

### 0.1 TileLang 在软件栈里的位置

```
PyTorch 模型
   │  (torch.compile / Inductor：图捕获、算子融合、算子选择)
   ▼
图级编译（TileLang 不负责）
   │
   ▼
TileLang kernel（本 corpus 的主角）
   │  Python 前端：T.Kernel / T.copy / T.gemm / T.Parallel / T.Pipelined ...
   ▼
TVM TIR（TensorIR） IRModule
   │  主流：CUDA/ROCm/Metal pipeline   昇腾：ascend pipeline（含 AutoSchedule/Z3）
   ▼
device 源码（CUDA C / PTX / Ascend C / PTO / AscendNPU IR）
   │  nvcc / nvrtc / bisheng / PTO 工具链
   ▼
可加载的 kernel + host 侧 wrapper（cython / torch / tvm_ffi / ...）
```

三个必须记住的判断：

1. TileLang **不是** PyTorch 的替代品，也不是完整的图编译器。它解决的是“一个算子内部
   怎么写快”，位置在 Inductor 之下、CUDA/Ascend C 之上。
2. TileLang **不是** 语法糖。它的价值在编译期：layout 推导、软件流水、同步插入、
   多缓冲、片内存储复用这些都由 pass 完成。
3. TileLang 的 IR 就是 **TVM TIR**。“给 TileLang 加功能”很多时候等于“写一个 TIR
   pass”，所以 TVM 是必修，不是选修。

### 0.2 五个核心概念（后续所有内容的骨架）

| 概念 | 含义 | 在语言里的样子 | 在编译器里的落点 |
| --- | --- | --- | --- |
| **Tile（分块）** | 计算的基本单位是一片数据，而非一个元素或一个线程 | `T.alloc_shared((M,N))`、`T.copy`、`T.gemm` | `src/op/`、`LowerTileOp` |
| **Layout（布局）** | 逻辑 `[M,K]` 到物理线程/寄存器/存储的映射（含 swizzle、fractal、padding） | 用户通常不写，用 `T.annotate_layout` 覆盖 | `tilelang/layout/`、`src/transform/layout_inference/` |
| **Storage scope（存储层级）** | 数据放在哪一级：global / shared / fragment / L1 / L0A-L0C / UB | `T.alloc_shared`、`T.alloc_fragment`、`T.alloc_l1`、`T.alloc_l0c` | buffer scope、`MergeSharedMemoryAllocations`、昇腾 `MergeUBAllocations` |
| **Pipeline（流水）** | 搬运与计算重叠、多缓冲、stage 数 | `T.Pipelined(..., num_stages=k)` | `PipelinePlanning` → `InjectSoftwarePipeline` → 多缓冲物化 |
| **Thread / warp / core（执行映射）** | 谁执行、怎么分工（SIMT 线程、warp 组、Cube/Vector 核） | `T.Parallel`、`T.thread_binding`、`T.warp_specialize`、`T.SimtVF/T.SimdVF` | `MaterializeKernelLaunch`、`LowerThreadAllreduce`、`ThreadSync` |

学会 TileLang 的过程，就是把这五个概念在“用户视角”和“编译器视角”之间来回翻译。

---

## 1. 前置知识体系

每块给出：为什么需要、要掌握到什么程度、判断是否掌握、本地资料。

### 1.1 计算机体系结构与并行计算（必修，1–2 周）

**为什么需要**：TileLang 的每个设计决策都在回答硬件问题。没有存储层级和流水线的直觉，
`T.Pipelined(num_stages=3)` 就只是魔法数字。

知识点：

* 存储层级与延迟/带宽差异（寄存器 → 片上 SRAM → 片外 HBM/DRAM）；“搬数据比算数据
  贵”这件事的量化形式。
* 计算并行度的三种基本形式：指令级并行（ILP）、数据级并行（SIMD/向量）、线程级并行
  （SIMT/多核）。
* 流水线、延迟隐藏、吞吐 vs 延迟；生产者-消费者与双缓冲。
* 同步原语与内存一致性（barrier、fence、acquire/release）的直觉。
* Amdahl 定律与 roofline 模型（后者见 1.2）。

掌握标准：能解释“为什么 attention 的 softmax 要在线做（online softmax）而不是两遍”，
并说出这样省掉了哪一级存储的读写。

本地资料：
* `05-nvidia-backend/raw/nvidia-cuda-c-programming-guide.html`（第 2–5 章是通用体系结构知识）
* `05-nvidia-backend/raw/nvidia-cuda-best-practices.html`
* `06-huawei-ascend/raw/hiascend/llms-content/hardware/zh_hardware_processor.md`（昇腾硬件规格）

需要额外自备的通用教材（本 corpus 未收录）：Hennessy & Patterson《计算机体系结构》，
或任何一份讲 cache/流水/并行体系结构的公开课。这是本 corpus 唯一明确“需要外部教材”
的基础块。

### 1.2 深度学习算子与性能模型（必修，2 周）

**为什么需要**：TileLang 的 examples 全是模型算子。不懂算子的数学结构，读代码只是在
读循环。

知识点：

* GEMM 及其变体（batched、grouped、split-K、stream-K、量化 GEMM）。
* 注意力：MHA/GQA/MQA、FlashAttention 的在线 softmax、varlen、paged KV、decode vs
  prefill 的差别。
* 稀疏注意力与索引类算子（DeepSeek NSA/DSA、indexer、top-k）。
* MoE：dispatch/combine、grouped GEMM、token permute。
* 归一化与逐元素融合（RMSNorm、RoPE、SwiGLU、量化/反量化）。
* roofline 分析：算术强度、memory-bound vs compute-bound、达到峰值百分比的口径。

掌握标准：给定一个算子，能算出它的理论最小数据搬运量，并据此判断它应该是
memory-bound 还是 compute-bound；能说清 TileLang 里哪个 example 对应哪个算子。

本地资料：
* `01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/*.md`
* `02-tilelang-core-repo/raw/examples/`（57 个目录，逐一看）
* `03-tilelang-papers/raw/arxiv-2205-14135.md`、`2307-08691`、`2407-08608`（FlashAttention 三部曲）
* `03-tilelang-papers/raw/arxiv-2502-11089.md`（NSA）、`2605-02568.md`（StreamIndex）
* `12-operator-libraries/raw/TileOPs/`（生产算子库写法）

### 1.3 GPU 执行模型（必修，3–4 周）

**为什么需要**：TileLang 的主干后端是 CUDA，它的绝大多数原语（TMA、wgmma、tcgen05、
warp specialization）都是 CUDA 概念的直接包装。

知识点：

* grid / block / warp / thread 层次，`blockIdx`/`threadIdx` 与 `T.Kernel`/`T.Parallel`
  的对应关系。
* 存储：global、shared memory、register、常量、L2；`shared` 的 bank conflict；
  swizzle 的目的。
* 同步：`__syncthreads`、`mbarrier`、memory fence、`cp.async` 的 commit/wait。
* 异步搬运：`cp.async` → Hopper/Blackwell 的 **TMA**（`cp.async.bulk.tensor`）。
* Tensor Core：`mma`（sm70–sm80）→ `wgmma`（sm90）→ `tcgen05`（sm100/sm120）；
  fragment/accumulator 的 layout 约束、TMEM。
* Warp specialization 与 producer/consumer 划分；thread block cluster 与 distributed
  shared memory。
* 性能工具：occupancy、寄存器压力、occupancy vs ILP 的权衡、Nsight Compute 的 SOL 指标。

掌握标准：能读懂 `examples/gemm/` 里生成的 CUDA 代码；能解释为什么 Hopper 上 `T.copy`
会变成 TMA 而 sm80 上是 `cp.async`。

本地资料：
* `05-nvidia-backend/raw/nvidia-cuda-c-programming-guide.html`（必读第 3、5、6、10 章）
* `05-nvidia-backend/raw/nvidia-ptx-isa.html`（按需查 `mma`/`wgmma`/`tcgen05`/`cp.async.bulk`）
* `05-nvidia-backend/raw/nvidia-hopper-tuning-guide.html`、`nvidia-blackwell-tuning-guide.html`
* `05-nvidia-backend/raw/nvidia-cutlass-docs.html`、`nvidia-cutlass-media-docs-cute.html`
* `05-nvidia-backend/raw/arxiv-2402-13499.md`（Hopper 实测数字）
* 代码对照：`02-tilelang-core-repo/raw/tilelang/cuda/op/gemm/` 与
  `tilelang/cuda/intrinsics/macro/`（`wgmma_macro_generator`、`tcgen05_macro_generator`）

### 1.4 NPU / 昇腾执行模型（必修，3–4 周；本项目的最高优先级）

**为什么需要**：昇腾的机器结构与 CUDA 不是一对一映射；不理解 Cube/Vector/MTE 与
UB/L1/L0A-L0C，就读不懂 `tilelang/ascend/` 里任何一个 pass。

知识点：

* CANN 全栈结构：GE（图引擎）、runtime、编译器（毕昇）、算子库、HCCL。
* Ascend C 编程模型（作为“TileLang 替你隐藏了什么”的对照）。
* AIC（Cube，矩阵）与 AIV（Vector，向量）的分工；SIMD 与 SIMT 区域。
* 存储层级与搬运流水：GM → L1 → L0A/L0B → L0C（Cube 侧）；GM → UB（Vector 侧）；
  MTE1/MTE2/MTE3/Fixpipe 的角色。
* 同步：Set/Wait 语义、流水间完成协议、double buffer；
  *这是昇腾上手最难的部分，也是 TileLang 自动化收益最大的部分。*
* PTO（Programmable Tensor Operations）：tile 粒度的指令集，及其形式化定义。
* 工具链：bisheng 编译器、msProf 性能分析、ACLGraph 入图、torch_npu。
* 硬件代差：910B/A2、A3、950 系列的差异，以及为什么 950 同时有 SIMT 与 SIMD。

掌握标准：能画出 950 上一次 GEMM 的数据流（从 GM 到 L0C 到 UB 到输出），并说出每一步
谁来执行、需要哪些同步；能解释 Ascend C 与 TileLang 在同一算子上的代码量差异来自哪。

本地资料：
* `06-huawei-ascend/raw/hiascend/llms-content/cann/zh_cann.md` 与同目录其余 CANN 页面
* `06-huawei-ascend/raw/hiascend/llms-content/`（算子开发、Ascend C、毕昇、硬件页）
* `06-huawei-ascend/raw/pto-spec/`、`raw/PTO-Gym/`、`raw/pto-kernels/`
* `07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/TileLang-Ascend Programming Guide.md`
* `09-zhihu-da-zhi-ruo-yu/raw/zhihu-dazhiruoyu-deepseek-ascend.md`（架构差异表 + 代码对照）
* `06-huawei-ascend/raw/hiascend/llms-content/developer/zh_developer_techArticles_20260506-1.md`
  （昇腾官方的 TileLang AscendNPU IR 教程）

### 1.5 编译器与 IR（T2/T3 必修，4–6 周；T1 了解即可）

**为什么需要**：TileLang 的差异化全在编译期。不学 IR，就只能当 API 使用者。

知识点：

* 编译流水线概念：前端 → IR → pass → 后端 codegen；AST、类型系统、绑定。
* IR 的形态：结构化 IR、SSA、block/axis（TensorIR）、`Buffer`/`Var`/`PrimExpr`、
  `IRModule` 与 `PrimFunc`。
* 循环变换：split/fuse/reorder/tile/unroll/vectorize 的语义与合法性。
* 数据流分析：读写集（read/write set）、依赖分析、别名分析、活跃区间。
* 布局与内存：storage rewrite、buffer 合并/复用、内存规划。
* 同步插入算法：从读写集推导必要的 barrier（`ThreadSync` 类 pass 的核心）。
* 流水线变换：stage 划分与软件流水（`InjectSoftwarePipeline` / `PipelinePlanning`）。
* 约束求解在编译期调度中的应用（昇腾侧用 Z3/SMT）。
* polyhedral 基础（可选，用于理解 layout/loop 变换的理论背景）。
* 如何读一个 pass：进 pass 前 dump IR，出 pass 后 dump IR，diff 之。

掌握标准：能读懂 `src/transform/layout_inference/` 的大意；能自己写一个 TIR pass 并
在 TileLang pipeline 里挂上；能用 `pass_diff`/`lower_trace` 定位一个 IR 变化发生在哪一步。

本地资料：
* `04-language-and-ir/raw/arxiv-1802-04799.md`、`2207-04296`、`2006-06762`、`2205-13603`
* `11-related-compilers/raw/mlir-docs.html`、`tvm-docs.html`
* `01-tilelang-official-docs/raw/site/_sources/compiler_internals/*.md`（官方内部文档）
* `01-.../tools/lower_trace.md`、`tools/pass_diff.md`、`tools/pass_visualizer` 相关页
* `02-.../raw/tilelang/src/transform/`（60+ 个 pass 的实现）
* 需要外部教材：任何一份讲编译器前端/数据流分析的公开课（本 corpus 未收录）。

### 1.6 数值、低精度与量化（选修但强烈建议，1–2 周）

**为什么需要**：TileLang 的低精度/量化 kernel（fp8、mxfp4、int4、block scaling）是它
最早成名的方向，也是昇腾 950 的主要卖点之一。

知识点：fp16/bf16 的差异；fp8（e4m3/e5m2）与 UE8M0 缩放；MX/block-scaled 格式；
int4/int8 量化与反量化；累加精度与舍入；为什么需要 block scaling 与 scale 布局约束。

本地资料：
* `02-.../raw/tilelang/examples/gemm_fp8/`、`gemm_int4/`、`dequantize_gemm/`、
  `blockscaled_gemm_sm100/`
* `05-.../raw/nvidia-blackwell-tuning-guide.html`（block-scaled MMA）
* `02-.../raw/tilelang/tilelang/cuda/intrinsics/macro/tcgen05_macro_generator`（看 b32 列布局）
* `08-.../raw/github-issue-2748.md`（MXFP4 scale-byte 编码的 RFC）

### 1.7 工程与工具链（贯穿全程）

* 构建与安装：`01-.../get_started/Installation.md`；昇腾侧见 `07-.../README.md` 的
  wheel 安装与 `install_ascend.sh`。
* PyTorch 集成：`tilelang/jit/adapter/torch/`、`07-.../examples/torch_tl_ascend/`。
* 性能测量：Nsight Compute（NVIDIA）、msProf（昇腾），见 `01-.../tools/` 与昇腾教程。
* 版本管理：TileLang 0.1.x 与 nightly 差异大；引用任何结论都要带 commit（见
  `02-.../raw/PROVENANCE.md`）。

---

## 2. TileLang 自身知识体系

### 2.1 用法层：语言与 API

必须掌握（按学习顺序）：

1. kernel 定义与 launch：`T.Kernel`、grid/block 绑定，`01-.../language_basics.md`、
   `compiler_internals/*`。
2. 数据与存储：`T.alloc_shared` / `T.alloc_fragment` / `T.alloc_local`，昇腾的
   `T.alloc_l1` / `alloc_l0a` / `alloc_l0b` / `alloc_l0c` / `alloc_ub`。
3. 搬运：`T.copy`、`T.Parallel` 下的隐式搬运、昇腾 `T.dual_copy`。
4. 计算：`T.gemm`、elementwise、`T.reduce_*`、`T.atomic_add`、`T.fill`、
   `T.tile.*`（昇腾 tile library：add/div/mul/select/gather/sort/topk/transpose…）。
5. 控制与并行：`T.Parallel`、`T.Pipelined`、`T.serial`、`T.vectorized`、条件与循环。
6. 线程/低层：`T.thread_binding`、warp group、`T.ptx_*`、昇腾 `T.SimtVF` / `T.SimdVF`。
7. 注解与覆盖：`T.annotate_layout`、`T.use_swizzle`、`T.annotate_*`、pass config
   （`PassConfigKey`）。
8. JIT 与可组合性：`@tilelang.jit`、宏（`T.macro`）、eager JIT、kernel 缓存。

掌握标准：不看例子能从零写出 fused RMSNorm+RoPE、GEMM、因果 attention。

本地资料：`01-.../raw/site/_sources/programming_guides/` 全部；
`07-.../docs/language_ref/`、`docs/api_docs/`；`10-.../raw/tilelang-puzzles/`。

### 2.2 编程模型：三个层次与“可组合”

官方把接口分三层（`get_started/overview.md`）：

* **Beginner（硬件无关）**：官方标注“尚未完全实现”。现阶段不要依赖。
* **Developer（tile library）**：用 tile 原语描述数据流，`T.gemm`/`T.copy`/`T.Pipelined`。
  日常 90% 的代码在这一层。
* **Expert（thread primitives）**：直接控制线程/搬运/同步，或使用硬件专用存储。

关键点：三层可以**混用**在同一 kernel 里（论文标题里的 “composable” 就是指这个）。
这是 TileLang 与 Triton 的核心分野：Triton 一旦下钻就离开模型，TileLang 允许局部下钻
而不破坏其余部分的优化。

### 2.3 编译期：管线与 pass

一个 TileLang kernel 的一生（CUDA 侧，`tilelang/cuda/pipeline.py`）：

```
BindTarget → MaterializeKernelLaunch → LetInline → AddWrapperForSingleBufStore
→ LegalizeNegativeIndex → VerifyParallelLoop → InjectAssumes → Simplify
→ CanonicalizeLegacyReducer → VerifyReducerEpoch → VerifyBufferInit → IfStmtBinding
→ UnrollLoop → Simplify
→ PipelinePlanning → InjectSoftwarePipeline → Simplify        ← 流水与多缓冲
→ LayoutInference                                            ← 布局推导
→ ReducerPlanAndMaterialize → LowerTileOp                     ← tile op 展开
→ VerifyReducerConsumed → DecoupleTypeCast → LegalizeVectorizedLoop
→ LegalizeSafeMemoryAccess → LowerAccessPtr → Simplify
→ PlanAndUpdateBufferAllocationLocation → HoistGlobalBufferAllocations
→ LowerOpaqueBlock → Simplify → NarrowDataType → FlattenBuffer → ConfigIndexBitwidth
→ Simplify → VectorizeLoop → StorageRewrite
→ LoopUnswitching → UnrollLoop → Simplify → RemoveNoOp → VerifyMemory
→ AnnotateEntryFunc → LowerThreadAllreduce → AnnotateDeviceRegions → SplitHostDevice
→ AnnotateReadOnlyParams → MergeSharedMemoryAllocations
→ ThreadSync ×2                                               ← 同步插入
→ MergeIfStmt → MakePackedAPI → Simplify → LowerDeviceKernelLaunch
```

昇腾侧（`tilelang/ascend/pipeline.py`）在同一结构上换掉了关键几步，多出：

```
RewriteDualCopy → LegalizeSimdMerging → UnrollLoopSkipVF
→ NormalizeAscendFractalStorage → InsertNd → AscendInsertOOBPadding
→ NormalizeControlFlowForSchedule → NormalizeConflictHints
→ MaterializeScheduleUnits → AnnotateMultiBufferEligible → EstimateLatency
→ AutoSchedule           ← 用 SMT/Z3 解流水与多缓冲约束
→ AssignCore → PrepareMultiBuffer → ResolveCore → InsertSync
→ MaterializeMultiBuffer → LowerScheduledTIR → RestoreWhileLoops
→ AscendSimdVFLowerParallel → AscendLowerTileOp → ... → MergeUBAllocations
→ RewriteFlagToBuf → AscendThreadSync ×3
```

要掌握的三件事：

1. **怎么看到中间态**：`tools/lower_trace`、`tools/pass_diff`、`tools/pass_visualizer`、
   `TL_DEBUG`/pass config。没有这个能力，改 pass 就是盲改。
2. **哪些 pass 是“承重墙”**：`LayoutInference`、`InjectSoftwarePipeline`、
   `LowerTileOp`、`ThreadSync`、`StorageRewrite`、昇腾的 `AutoSchedule` + `InsertSync`。
3. **pass 的开关与配置**：`tilelang/transform/pass_config.py` 里的 `PassConfigKey`
   （例如禁用 shared memory 复用、开 race check、强制 let inline）。

### 2.4 核心算法（TileLang 的“为什么快”）

#### 2.4.1 Layout inference

问题：用户写的是逻辑张量 `[M, N]`，但 MMA/wgmma/tcgen05/Cube 需要的物理布局是
fragment、swizzle、fractal（ND→NZ）等带约束的映射。

TileLang 的解法（`tilelang/layout/` + `src/transform/layout_inference/`）：
* 用 layout 变量表达“元素 → (线程, 寄存器槽位)”的映射；
* 在 tile op 图上传播约束，求解出满足所有 op 的 layout；
* 冲突时插入转换（转换代价进入布局选择）。

要点：这是“信息等价的推导”而非“猜”。知乎那篇原文的判断值得记住：*编译器应做等价
推导，而不是在信息不足时补充信息*。所以 TileLang 把会影响合法性/性能的资源信息
（L1/L0C 的使用、SIMT/SIMD 分区）保留在前端。

#### 2.4.2 软件流水与多缓冲

`T.Pipelined(..., num_stages=k)` → `PipelinePlanning` 定 stage 与顺序 →
`InjectSoftwarePipeline` 改写循环 → 多缓冲物化（buffer 版本化）×
`MergeSharedMemoryAllocations`/`MergeUBAllocations` 做空间规划。

必须理解：stage 数与片上容量是硬约束（昇腾侧被显式建模进 Z3 约束里）。

#### 2.4.3 同步插入

从 buffer 的读写集 + 执行体（线程/warp/core）推导必要的 barrier 与 fence。
`ThreadSync`（CUDA）与 `AscendThreadSync` + `InsertSync` + `RewriteFlagToBuf`（昇腾）
是同一问题的两种实现。昇腾侧的难点在 MTE1/MTE2/MTE3/Cube/Vector/Fixpipe 之间的
多流水依赖。

#### 2.4.4 自动调度（昇腾专有）

* 把 kernel 拆成带资源标签的任务（`MaterializeScheduleUnits`）；
* 估计延迟（`EstimateLatency`）与容量（`AnnotateMultiBufferEligible`）；
* 用 Z3 求满足依赖、流水间隔、缓冲区容量的可行解（`AutoSchedule`）；
* 落地为多缓冲与同步（`PrepareMultiBuffer` → `InsertSync` → `MaterializeMultiBuffer`）。

这是昇腾后端最有技术含量的部分，也是“TileLang 比 Ascend C 省掉人肉排期”的核心。

#### 2.4.5 自动调参与代价模型

`tilelang/autotuner/`（参数空间、capture、grouped compile、tuner）+
`tilelang/carver/`（模板、cost model、roller policy）。
NVIDIA 侧写 kernel 时的 block size / num_stages / 线程划分通常在 autotune 空间里；
昇腾侧调参维度不同（core ratio、多缓冲份数等）。

#### 2.4.6 Warp specialization

`T.warp_specialize` + `tilelang/language/warp_specialize.py`：把搬运与计算分成不同 warp
组，配合 TMA 与 mbarrier。Hopper 上的 attention kernel 基本都靠它。可先读
`examples/warp_specialize/`。

### 2.5 代码生成与运行时

| 后端 | 生成什么 | 入口 |
| --- | --- | --- |
| CUDA | CUDA C / PTX（`nvcc` / `nvrtc`） | `tilelang/cuda/codegen/`、`tilelang/contrib/{nvcc,nvrtc}/` |
| CuTe DSL | CUTLASS Python DSL | `tilelang/contrib/cutedsl/`、`tilelang/jit/adapter/cutedsl/` |
| ROCm | HIP | `tilelang/rocm/` |
| Metal / WebGPU / CPU | 各自后端 | `tilelang/{metal,webgpu,cpu}/` |
| **Ascend (PTO)** | PTO / Ascend C | `tilelang/ascend/codegen/`、`tilelang/jit/adapter/pto/` |
| **Ascend (NPU IR)** | AscendNPU IR (MLIR) | `07-.../raw/tilelang-mlir-ascend/` |

运行时侧：host wrapper 生成（`src/transform/make_packed_api.cc`）、kernel 缓存
（`tilelang/cache/`）、torch 适配（`tilelang/jit/adapter/torch/`）、
轻量运行时（`12-.../raw/TileRT/`）。

### 2.6 调试与可视化工具（必须熟练）

* `TL_DEBUG`/日志：`01-.../tutorials/logging.md`
* `lower_trace`：降级全过程追踪
* `pass_diff`：两个 pass 之间 IR 差异
* `layout_visualization` / `plot_layout`：看 layout
* `Analyzer`、`compile_only`、`iket`（kernel 评测相关）
* `T.print`/`T.printf`、昇腾 `T.dump_tensor`
* 昇腾侧：msProf、AICore exception dump、`docs/coverage_guide.md`

掌握标准：任意一个编译错误或性能异常，能用上述工具定位到“哪个 pass 出了什么问题”。

### 2.7 算子实现模式库（读完这些，就有了“模式语言”）

| 模式 | 参考实现 |
| --- | --- |
| 逐元素 + 融合 | `examples/elementwise/`、`cast`、`norm` |
| 标准 GEMM 阶梯 | `examples/gemm/`（从 naive 到 intrinsic，一档档看） |
| Split-K / Stream-K / Grouped | `gemm_splitk/`、`gemm_streamk/`、`grouped_gemm/` |
| Tensor core 变体 | `gemm_tcgen05/`、`blockscaled_gemm_sm100/`、`gemm_sm120/` |
| 量化 | `gemm_fp8/`、`gemm_int4/`、`dequantize_gemm/` |
| Flash attention | `flash_attention/`、`flash_attention_sm100/`、`flash_decoding/` |
| 稀疏 / 索引 | `blocksparse_attention/`、`deepseek_nsa/`、`minference/`、`topk/` |
| MLA / DeepSeek 系列 | `deepseek_mla/`、`deepseek_v32/`、`deepseek_v4/`、`deepseek_mhc/` |
| MoE | `fusedmoe/` |
| 昇腾专有 | `07-.../raw/tilelang-ascend/examples/`（HISA、aclgraph、lightning_indexer、…） |

---

## 3. 分阶段学习路径（从零到专业）

每阶段给出：目标 / 学什么 / 动手产出 / 验收标准 / 主要资料。
阶段 0–3 面向所有人；阶段 4+ 按终点（T1/T2/T3）分流。

### 阶段 0 · 环境与第一次运行（1–3 天）

目标：跑通第一个 kernel，建立“Python 里写 kernel”的手感。

动手产出：在本机安装 TileLang（`01-.../get_started/Installation.md`），跑通
`examples/elementwise/` 与 `examples/gemm/`，把生成源码（`kernel.get_kernel_source()`）
打印出来看。

验收：能解释打印出来的 CUDA 里哪一段对应你写的哪一行 Python。

资料：`01-.../get_started/` 三页 + `02-.../examples/quickstart.py`。

### 阶段 1 · 写出第一个自己的算子（1–2 周）

目标：从“照着抄”到“照着需求写”。

学什么：`T.Kernel` 网格语义、`T.Parallel`、`T.copy`、`T.alloc_shared/fragment`、
类型系统、控制流。

动手产出：不看例子写出 ① y = x*2+b ② 行 softmax ③ 一个朴素 GEMM；
然后做 `10-.../tilelang-puzzles` 的前 5 题。

验收：三个 kernel 结果都对，且能说清每个 buffer 为什么放在那一级存储。

资料：`01-.../programming_guides/language_basics.md`、`type_system.md`、
`control_flow.md`、`instructions.md`；`10-.../raw/tilelang-puzzles/`。

### 阶段 2 · 性能直觉（2–4 周）

目标：能解释一个 kernel 快/慢的原因，而不是靠试。

学什么：roofline、算术强度、occupancy；`T.Pipelined` 与多缓冲；
layout 与 bank conflict；auto-tuning。

动手产出：把阶段 1 的 GEMM 优化到接近该卡的 cuBLAS 的 70% 以上；用 Nsight Compute
（或 msProf）写出 3 条可量化的瓶颈结论；跑一次 `autotune` 并解释最优参数为什么最优。

验收：能对任何新 kernel 先给出性能预测（大致能到峰值多少），再用实测验证。

资料：`01-.../software_pipeline.md`、`autotuning.md`、`tutorials/auto_tuning.md`；
`03-.../FlashAttention 三部曲`；`05-.../best-practices`、`nsight-compute`；
`12-.../raw/tilelang-benchmark/`。

### 阶段 3 · attention 与融合算子（3–6 周）

目标：能实现生产级融合算子。这通常是“会用 TileLang”的分水岭。

学什么：online softmax、attention 变体（MHA/GQA/MLA）、varlen/paged、Flash-Decoding
与 split-KV、稀疏/索引类算子、MoE。

动手产出：自己实现一版 FlashAttention（对齐 `examples/flash_attention/`），在正确性
通过的前提下达到官方实现 90% 以上的性能；再实现一个 RMSNorm+RoPE 融合算子。

验收：能对着 DeepSeek MLA 的论文说出 kernel 的每个 tile 级步骤。

资料：`01-.../deeplearning_operators/`（含 `deepseek_mla.md`）；
`02-.../examples/{flash_attention,flash_decoding,deepseek_mla,deepseek_nsa,fusedmoe}/`；
`03-.../arxiv-2412-19437.md`。

> 走到这里，T1（算子工程师）基本达成；下面按终点分流。

### 阶段 4 · 编译期（T2 起点，4–8 周）

目标：把 TileLang 从“黑盒编译器”变成“可读的编译器”。

学什么：TVM TIR 基础、pass 写法、layout inference 与 pipeline 的实现、
调试工具链。

动手产出：
1. 用 `lower_trace` + `pass_diff` 追踪一个 kernel 从 Python 到 CUDA 的全过程，并写成
   一份逐 pass 笔记（这是本项目里最有价值的自测产出）。
2. 自己写一个 TIR pass（例如统计某个 buffer 的访存次数）并挂进 pipeline。
3. 修改 `LayoutInference` 的一个分支或它的输入注解，观察行为变化。

验收：能回答“`T.gemm` 的 `transpose_B=True` 最终改变了哪些 IR 节点”。

资料：`04-...`；`01-.../compiler_internals/`；`02-.../src/transform/`、
`tilelang/layout/`、`tilelang/transform/`；`03-.../arxiv-2504-17577.md`。

### 阶段 5 · Warp specialization 与 Hopper/Blackwell（T2/T3，4–6 周）

目标：掌握当下最激进的 GPU 编程形态。

动手产出：实现一个使用 TMA + warp specialization 的 attention 或 GEMM；
对比 `examples/warp_specialize/`、`flash_attention_sm100/`；讲清 producer/consumer
各自的 mbarrier 协议。

资料：`05-.../hopper-tuning-guide`、`blackwell-tuning-guide`、`ptx-isa`；
`03-.../arxiv-2407-08608.md`；`02-.../tilelang/{language/warp_specialize.py,cuda/}`。

### 阶段 6 · 昇腾路线（本项目最高优先，6–12 周）

目标：能写、能调、能改昇腾侧的 TileLang。

步骤：
1. 读懂 `09-...` 那篇知乎文章 + `07-.../docs/TileLang-Ascend Programming Guide.md`。
2. 读懂昇腾机器模型（`06-.../raw/hiascend/`：CANN、Ascend C、硬件页）。
3. 跑通 `07-.../examples/`：先从 `elementwise`、`gemm/example_gemm_intrinsic.py`、
   `flash_attention` 起，再到 `deepseek_v4`。
4. 手写对比：同一个 GEMM 用 Ascend C 与 TileLang 各写一版，比较代码量与同步处理。
5. 读 `tilelang/ascend/pipeline.py` 与 `tilelang/ascend/transform/`，特别是
   `AutoSchedule`/`InsertSync`/`MaterializeMultiBuffer`。
6. 理解 PTO：`06-.../raw/pto-spec/`、`raw/PTO-Gym/`，并对照 `T.alloc_l1/l0c`、
   `T.dual_copy` 的 codegen。

验收：能对昇腾上一个 attention 算子自主完成“复现 → 定位瓶颈 → 改 layout/流水 →
提升性能”的闭环，并说清每一步对应哪个 pass。

资料：`06-...`、`07-...`、`08-...`、`09-...`。

### 阶段 7 · 后端与贡献（T3，持续）

目标：能改 TileLang 本身，并让改动被合并。

见下文章节“4. 给 TileLang 做贡献”。

---

## 4. 三个终点的能力验收清单

**T1 算子工程师**

- [ ] 能独立实现 GEMM / attention / MoE / 量化类算子并达到生产性能
- [ ] 能用 profiler 解释性能，而不是靠直觉
- [ ] 能判断一个算子应该用 TileLang、Triton 还是 torch.compile
- [ ] 能把 kernel 接进 PyTorch（autograd、自定义 op、torch.compile 兼容）

**T2 编译器工程师**

- [ ] 能读 TIR，能写 pass，能解释每个 pass 的作用与顺序理由
- [ ] 能修改 layout inference / pipeline / sync 插入中的至少一处
- [ ] 能诊断“编译通过但结果错”的问题（多数是同步或 layout 问题）
- [ ] 能评审别人对 `src/transform/` 的改动

**T3 后端工程师**

- [ ] 能画出目标机器从指令到存储的完整模型
- [ ] 能新增/修改一个 codegen 路径或一个后端 pipeline
- [ ] 能写出该后端上的正确性测试与性能基准（对齐 `tilelang-benchmark` 口径）
- [ ] 能参与 roadmap/RFC（`08-...`）级别的设计讨论

---

## 5. 给 TileLang 做贡献（T2/T3 的实操路径）

1. **先读懂“边界”**：`08-.../raw/github-issue-2115`（多后端解耦）与
   `github-issue-2196`（TVM 迁移）决定你该往哪儿加东西。
2. **从当前在飞的工作切入**：`08-...` 里那些未合入的 Ascend PR 就是最好的“下一步”。
   读它们的 diff，找准一个自己能复现的部分。
3. **必须掌握的工程要求**：
   * 测试：`testing/` 下的 pytest（CUDA 侧）与 `07-.../testing/`（昇腾侧）；
   * 性能回归：`12-.../raw/tilelang-benchmark/`；
   * 代码风格：`docs/developer_guide/cpp_style.md`；昇腾仓库有自己的 `AGENTS.md`。
4. **改 pass 的标准动作**：先写 IR 前后 diff 作为测试基线 → 改 pass → 跑全量测试 →
   跑 benchmark 看回归 → 提 PR，附上 lower_trace 证据。
5. **改后端的标准动作**：先加最小可用（能跑通一个 elementwise）→ 再加 GEMM →
   再对齐 `InsertSync`/多缓冲 → 最后补 layout inference。
6. **昇腾侧特殊之处**：`tilelang-ascend` 是独立仓库、独立发版，很多新能力先进那里，
   再以 `[Ascend]` PR 的方式回主干。两边都要跟。

---

## 6. NVIDIA 专项（最小必要集）

如果只学 NVIDIA 侧的“核心设计思想”，按这个顺序，不要试图覆盖全部 CUDA：

1. 线程与存储模型（1.3 的知识点）→ 2. TMA（`cp.async.bulk.tensor`）→
3. `wgmma`（sm90）→ 4. `tcgen05` + TMEM（sm100）→ 5. warp specialization →
6. cluster / distributed shared memory。

对应到 TileLang：`T.copy`→TMA；`T.gemm`→按架构选 mma/wgmma/tcgen05；
`T.Pipelined`→多缓冲 + mbarrier；`T.warp_specialize`→producer/consumer。

前沿观察点（可能在未合入 PR 中）：CuteDSL 后端（`08-.../github-issue-1454`）、
SM120 block-scaled MMA、persistent kernel、PDL（programmatic dependent launch）。

---

## 7. 昇腾专项（最高优先）

### 7.1 知识顺序（严格按此顺序，跳步会卡）

1. **硬件角色**：AIC/Cube、AIV/Vector、Scalar；UB / L1 / L0A / L0B / L0C 的容量与用途。
2. **搬运流水**：MTE1/MTE2/MTE3、Fixpipe；每种搬运的源与目标。
3. **同步语义**：Set/Wait、流水间依赖、多缓冲覆盖约束。
4. **Ascend C**：能读懂一个 GEMM 的 Ascend C 实现（这是对照物）。
5. **PTO**：指令集与 tile op；`pto-spec` 的形式化定义；
   `pto-dsl` 的 Python 写法。
6. **TileLang@Ascend 的前端**：`T.alloc_l1/l0a/l0b/l0c`、`T.SimtVF/T.SimdVF`、
   `T.dual_copy`、`T.Pipelined`。
7. **TileLang@Ascend 的编译期**：autoschedule（Z3）、InsertSync、多缓冲、
   fractal/NZ layout。
8. **工程化**：msProf、ACLGraph 入图、torch_npu、算子入图、覆盖度工具。

### 7.2 三条必须自己动手的练习

* **同一算子的三版对照**：Ascend C / PTO 手写 / TileLang。比较代码量与同步处理。
* **同步实验**：故意删除/移动一个同步点，观察结果错误或性能下降，理解 InsertSync 的必要性。
* **调度实验**：改 `num_stages` 或 buffer 份数，看 AutoSchedule 的可行解如何变化
  （或直接不可解），理解容量约束。

### 7.3 易被忽略但重要的事

* 昇腾不是“SM 换个名字”：Cube 与 Vector 是不同执行体，不能按 CUDA 的直觉推。
* `T.dual_copy`（L0C→UB 双目标）这类原语是硬件特性穿透到语言层的结果；
  这类“必须显式表达”的信息是昇腾编程的成本来源，也是 TileLang 需要权衡的地方。
* 950 同时支持 SIMT 与 SIMD（`T.SimtVF` / `T.SimdVF`），因此“GPU 是 SIMT、NPU 是
  SIMD”的旧概括已经失效。

---

## 8. 常见陷阱

1. **把 TileLang 当 Triton 用**：TileLang 要求你显式决定 tile 与存储位置，写得太
   “高层”反而拿不到性能。
2. **忽略 layout 约束**：`T.gemm` 的输入 layout 不对时，性能会断崖下跌而不是报错。
3. **多缓冲份数随意调**：stage 数受片上容量硬约束，昇腾上会更严格。
4. **把“编译通过”当“正确”**：同步缺失经常表现为偶发错误（尤其昇腾）。
5. **用 master 的最新行为解释旧版本**：0.1.x 与 nightly 差异很大，引用必须带 commit。
6. **拿 vendor 的宣传性能做基线**：核对测量口径（dtype、shape、是否含 epilogue）。
7. **在 NVIDIA 上验证过的结论直接搬到昇腾**：存储层级、执行体、同步协议全不同。
8. **误以为 beginner 层可用**：官方文档明确说它尚未完成。
9. **把 TileLang 和 Inductor 当竞品**：它们是上下层关系。
10. **忽略 pass config 的副作用**：关掉某个优化（如 buffer 复用）会静默改变性能与
    正确性边界。
11. **在错误的地方找答案**：编译问题是 `src/transform/`，性能问题是 layout/pipeline，
    数值问题是 dtype 与累加顺序。
12. **不看上游 issue/PR**：很多“为什么这样设计”的答案只在 PR 讨论里。

---

## 9. 资料地图（按问题找文件）

| 我想知道 | 去看 |
| --- | --- |
| TileLang 是什么、设计动机 | `03-.../raw/arxiv-2504-17577.md` |
| 怎么装、怎么跑第一个 kernel | `01-.../site/_sources/get_started/*` |
| 语言有哪些原语、什么语义 | `01-.../programming_guides/*`；`07-.../docs/language_ref/` |
| 一个具体算子怎么写 | `01-.../deeplearning_operators/*`；`02-.../tilelang/examples/*` |
| 编译期到底做了哪些变换 | `01-.../compiler_internals/*`；`02-.../tilelang/src/transform/*` |
| layout 是怎么推出来的 | `02-.../tilelang/tilelang/layout/*`；`src/transform/layout_inference/*` |
| 流水与同步怎么插 | `src/transform/{inject_pipeline.cc,thread_storage_sync.cc}`；昇腾 `tilelang/ascend/transform/` |
| 昇腾后端做了什么 | `07-.../raw/tilelang-ascend/`；`02-.../tilelang/tilelang/ascend/` |
| 昇腾为什么这么难、TileLang 怎么解 | `09-.../raw/zhihu-dazhiruoyu-deepseek-ascend.md` |
| PTO 是什么 | `06-.../raw/pto-spec/`、`raw/PTO-Gym/`；`gitcode.com/cann/pto-isa` |
| CUDA 侧指令细节 | `05-.../nvidia-ptx-isa.html`、`nvidia-cutlass-docs.html` |
| 未来会怎么变 | `08-.../raw/github-{issue,pr}-*.md` |
| 生产里怎么用 | `12-.../raw/TileOPs/`、`TileRT/`、`tilelang-benchmark/` |
| 和 Triton/TVM 的关系 | `11-.../` 与 `04-.../` |

---

## 10. 一句话总结

学好 TileLang 需要的不是“更多 CUDA 语法”，而是三层能力叠加：
**硬件直觉**（存储层级、流水、同步）× **算子与性能模型**（roofline、attention/MoE 的结构）
× **编译器思维**（IR、pass、layout 推导）。昇腾优先不是因为政治上更重要，而是因为
它把这三层能力的要求同时推到了极限：谁能在昇腾上把同步与调度问题解干净，谁就真正
理解了 TileLang 想解决的问题。
