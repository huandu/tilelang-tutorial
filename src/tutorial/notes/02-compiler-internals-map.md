# TileLang 编译期内部地图

> **版本提示（2026-10-01 追加）**：本文写于 corpus 的快照 `main`（commit `994b44e`），
> 而教材《从零到一：TileLang 系统教程》锁定的是 release tag `v0.1.15`。两者在个别 pass
> 名字上有差异（例如本文写的 `InsertNd`，在 v0.1.15 里是 `InsertNd2Nz`）。做逐行核对时
> **一律以 `../../tilelang-book/_vendor/tilelang-v0.1.15/` 为准**，本文只作导航。

这份文档回答一个问题：**我写的一行 Python，最后变成了什么，中间经过了谁。**
所有文件路径都相对 `02-tilelang-core-repo/raw/`（主干 clone，commit 见
该目录 `PROVENANCE.md`）。

## 1. 一次编译的全部参与者

```
用户 kernel (Python)
  │  decorator: tilelang.jit            tilelang/jit/__init__.py
  │  AST 解析 / 变量绑定                 tilelang/language/parser/
  ▼
TileLang 前端 frame & 原语
  │  tilelang/language/{kernel,copy_op,gemm_op,loop,allocate,builtin,...}.py
  ▼
TIR IRModule (tvm.tir / s_tir / tirx)
  │  tilelang/engine/lower.py  ── 选择 backend，跑 pass pipeline
  ▼
Backend PassPipeline
  │  tilelang/backend/pass_pipeline/pipeline.py   (框架)
  │  tilelang/cuda/pipeline.py    (NVIDIA)
  │  tilelang/ascend/pipeline.py  (Ascend)
  │  tilelang/{rocm,metal,cpu,webgpu}/pipeline.py
  ▼
Device 源码
  │  tilelang/cuda/codegen/        → CUDA C / PTX
  │  tilelang/ascend/codegen/      → PTO / Ascend C   (另有 MLIR 路线见 07)
  ▼
Host wrapper + 二进制
  │  src/transform/make_packed_api.cc, tvm_ffi_binder.cc
  │  tilelang/jit/adapter/{cython,torch,nvrtc,pto,cutedsl}/
  ▼
可调用 kernel（torch / cython / tvm_ffi）
```

## 2. 前端：语言层做了什么

`tilelang/language/` 里的每个模块对应一组语义，而不是一组字符串模板：

| 模块 | 语义 | 关键点 |
| --- | --- | --- |
| `kernel.py` | 定义网格与 block 级作用域 | 与 `MaterializeKernelLaunch` 配对 |
| `allocate.py` | 声明 buffer 及其 scope | scope 决定后续所有 layout/同步决策 |
| `copy_op.py` | 搬运语义 | 会降级为 TMA / cp.async / PTX / 昇腾搬运 |
| `gemm_op.py` | tile 级矩阵乘 | 决定选 mma/wgmma/tcgen05/Cube |
| `loop.py` | `T.Parallel`/`T.Pipelined`/`T.serial` | 流水与并行语义的来源 |
| `reduce_op.py`/`atomic.py`/`fill_op.py` | 归约与原子操作 | 涉及 reducer 生命周期验证 |
| `warp_specialize.py` | producer/consumer 划分 | 与 mbarrier/TMA 绑定 |
| `tile_schedule.py`、`annotations.py` | 调度提示与注解 | 用户干预编译器的入口 |
| `customize.py`、`builtin.py` | 定制与底层内建 | 下钻到硬件原语的通道 |
| `eager/`、`ast/`、`parser/` | 前端实现细节 | 调试语法问题时看这里 |

设计要点：这些原语**不直接展开成 CUDA**，而是先建立 IR 层的事实（buffer、访问关系、
并行域），把优化空间留给 pass。这就是论文里 “decouple scheduling space from dataflow”
的实际含义。

## 3. 承重 pass：六个必须理解的地方

### 3.1 `MaterializeKernelLaunch`
把 `T.Kernel` 的抽象网格物化成目标机器需要的 launch 结构。CUDA 上产生 `blockIdx`/
`threadIdx` 的绑定；昇腾上只保留 1-D core 网格（`lower_grid_binding=True,
lower_thread_binding=False`），线程域只在 `T.SimtVF` 内部出现。**这一条差异直接决定
了两侧编程模型的形状**。

### 3.2 `PipelinePlanning` + `InjectSoftwarePipeline`
`T.Pipelined(num_stages=k)` 的语义实现。规划阶段决定每个语句属于哪个 stage、顺序如何；
注入阶段改写循环体并引入 buffer 版本（多缓冲）。配套的
`MergeSharedMemoryAllocations`（CUDA）/ `MergeUBAllocations`（Ascend）负责把多份缓冲
装进有限的片上空间。

### 3.3 `LayoutInference`
在 tile op 图上传播并求解 layout 约束。它决定了：
* 每个 fragment/local buffer 的元素落在哪个线程的哪个寄存器槽；
* shared buffer 的 swizzle 形态；
* 是否需要插入 layout 转换（以及代价）。

昇腾额外需要 `NormalizeAscendFractalStorage`、`InsertNd`、
`AscendInsertOOBPadding`。因为 Cube 的物理布局（fractal / ND→NZ / 对齐）是硬约束。

### 3.4 `LowerTileOp`
把 tile 级 op 展开成具体的指令或子图（MMA 序列、搬运序列）。在这之后就不再有
“tile op”这个概念，只剩 TIR 语句。调试 tile 语义问题时，在这一步前后对比 IR 最有效。

### 3.5 `VectorizeLoop` / `UnrollLoop` / `StorageRewrite`
常规但重要：向量化、展开、片上存储复用与分配。`StorageRewrite` 决定 buffer 复用，
也是“关掉某个优化后性能突变”的常见原因。

### 3.6 `ThreadSync`（CUDA）/ `InsertSync` + `AscendThreadSync`（昇腾）
根据读写集推导 barrier/fence 并插入。**多数“结果偶发错误”都在这里。**
昇腾侧还要处理不同流水（MTE/Cube/Vector/Fixpipe）之间的完成协议，以及
`RewriteFlagToBuf`（把 flag 变成 buffer）这类目标机器特有的做法。

## 4. 昇腾独有的前半段

昇腾 pipeline 在 layout/向量化阶段之前插入了整段“调度”：

```
RewriteDualCopy                # L0C→UB 双目标搬运展开
LegalizeSimdMerging            # MODE_MERGING 是读改写，显式暴露目的为 rw
UnrollLoopSkipVF               # 只在 VF 之外展开，避免破坏向量块
NormalizeAscendFractalStorage  # fractal 存储归一化
InsertNd                       # 插入 ND 变换
AscendInsertOOBPadding         # 边界填充
NormalizeControlFlowForSchedule
NormalizeConflictHints
MaterializeScheduleUnits       # 把计算/搬运拆成带资源标签的任务
AnnotateMultiBufferEligible    # 哪些 buffer 可以多份
EstimateLatency                # 任务延迟估计
AutoSchedule                   # Z3/SMT 求解调度
AssignCore / ResolveCore       # 任务分派到 AIC / AIV / 具体 core
PrepareMultiBuffer → InsertSync → MaterializeMultiBuffer
LowerScheduledTIR → RestoreWhileLoops
```

这段是 TileLang@Ascend 的核心竞争力：把 Ascend C 里人肉维护的
“这一轮 MTE2 搬到哪了 / Cube 什么时候能吃 / 何时能覆盖 UB”变成约束求解问题。
细节见 `09-zhihu-da-zhi-ruo-yu/raw/zhihu-dazhiruoyu-deepseek-ascend.md`。

## 5. 怎么自己看一遍（可复现的操作）

1. 选一个 kernel（推荐 `examples/gemm/`）编译，拿到 IRModule。
2. 用 `tilelang/tools/lower_trace` 或 `pass_diff` 打印各 pass 前后的 IR。
3. 按上面的“承重 pass”列表逐条对照：每一步 IR 里多了/少了什么。
4. 用 `plot_layout` / `layout_visualization` 看 layout 的实际形态。
5. 把结论写下来。这份笔记就是“读懂 TileLang 编译器”的证据。

相关工具页面（官方文档源文件）：
`01-tilelang-official-docs/raw/site/_sources/tools/{lower_trace,pass_diff,analyzer,
layout_visualization,compile_only,iket,autodd}.md`。

## 6. 想改编译期，最小改动清单

| 目标 | 落点 | 必须同时做的事 |
| --- | --- | --- |
| 新增语言原语 | `tilelang/language/` + 对应 `src/op/` | 加 lowering、加测试、加文档 |
| 新增 pass | `src/transform/` + 在 pipeline 里注册 | 记录前后 IR diff 作为回归基线 |
| 改 layout 推导 | `src/transform/layout_inference/` | 跑全量正确性 + benchmark |
| 改流水 | `src/transform/inject_pipeline.cc` | 关心多缓冲容量与同步 |
| 新增后端 | `tilelang/backend/` + `tilelang/<chip>/` | 见 `07-.../raw/TileFoundry/` 与 issue #2115 |
