# 昇腾路线：现状、差异与入手点

> **版本提示（2026-10-01 追加）**：本文基于 corpus 的快照 `main`（commit `994b44e`）。
> 教材锁定 `v0.1.15` 与 `tilelang-ascend v0.1.2.000-release`，个别 pass 名与 API 会与本文
> 不一致；逐行核对请以 `../../tilelang-book/_vendor/` 里的两个 checkout 为准。

这份文档把“TileLang + 华为昇腾”这一侧拼成一张完整的图：硬件 → 指令集 → 语言 →
编译期 → 工程落地 → 未合入的前沿。所有结论都有本 corpus 内的出处。

## 1. 三句话概括现状（2026-09-30）

1. 昇腾侧有**两个官方后端路线**：`ascendc_pto`（生成 Ascend C / PTO）与 `npuir`
   （MLIR 方言 AscendNPU IR）。适配器仓库 `tile-ai/tilelang-ascend` 是主战场，主干
   `tile-ai/tilelang` 已经接收了 `[Ascend]` 系列 PR（`tilelang/ascend/`、`src/ascend/`）。
2. 官方测试覆盖 **A2 / A3**（适配器 README），而 950 后端在 2026-09-29/30 才刚并入主干
   （PR #3308、#3310），大量能力仍在未合入的 PR 里。
3. 昇腾侧的技术难点集中在**同步与调度**，TileLang 的解法是**把调度建模成约束问题并用
   Z3 求解**，这是它相对 Ascend C 最实质的价值。

## 2. 机器模型差异（来自知乎原文的对照表，已按 PDF/MD 校对）

| 维度 | NVIDIA GPU（CUDA 习惯） | 昇腾 950 |
| --- | --- | --- |
| 程序组织 | Grid / thread block / warp / thread | kernel 定义核心块网格；SIMT 区域另行定义线程域 |
| 计算分工 | 线程 + 协作线程组使用通用计算与矩阵指令 | **Cube/AIC 承担矩阵，Vector/AIV 承担向量** |
| 片上存储 | 寄存器、shared memory（新架构有专用存储） | **显式区分 UB、L1、L0A、L0B、L0C** |
| 数据传递 | 普通访存 + `cp.async` / TMA 等异步机制 | 不同存储层级通过**指定搬运路径**连接，并与计算流水协作 |
| 同步 | 线程、线程组、异步操作各自的完成协议 | 需要处理**流水线间、Cube/Vector 间、多缓冲复用**的完成协议 |

结论：**维度不同，不能按 CUDA 直觉类推**。TileLang 的选择是“不强行弥合差异”，而是把
昇腾特有的资源信息暴露到前端。

## 3. 语言层：昇腾专属原语

来自 `09-.../raw/zhihu-dazhiruoyu-deepseek-ascend.md` 与
`02-.../tilelang/tilelang/ascend/language/`：

| 原语 | 作用 | 为什么必须显式 |
| --- | --- | --- |
| `T.alloc_l1` / `T.alloc_l0a` / `T.alloc_l0b` / `T.alloc_l0c` | 指定 Cube 侧各级片上存储 | 存储位置直接决定合法性与性能，编译器无法替你猜 |
| `T.alloc_ub` | Vector 侧统一缓冲 | AIV 的工作内存 |
| `T.SimtVF` / `T.SimdVF` | 声明 SIMT / SIMD 计算区域 | 950 同机同时支持两种模式，线程域只在区域内存在 |
| `T.dual_copy` | L0C → UB 的双目标搬运 | 硬件特性（dual destination）穿透到语言 |
| `T.Pipelined` | 软件流水 | 与多缓冲、同步协议绑定 |
| `T.tile.*`（add/div/mul/select/gather/sort/topk/transpose/…） | tile 级向量操作库 | 见 `07-.../docs/api_docs/` |

官方给的 GEMM+ReLU 例子（原文引用）用不到 20 行就表达了完整的 dataflow：
`alloc_l1 → copy → gemm(l0c) → dual_copy → SimtVF(max)`，开发者不需要关心插多少条搬运
指令，也不需要手写 Cube 与 Vector 之间的同步。原文对性能的判断是“与主线 Ascend C
打得有来有回，代码行数差好几倍”，注意这是 community 级证据，需要自行复核。

## 4. 编译期：昇腾后端在解什么问题

见 `02-compiler-internals-map.md` 第 4 节的 pass 列表。四个关键点：

1. **fractal / ND→NZ / 边界填充**（`NormalizeAscendFractalStorage`、`InsertNd`、
   `AscendInsertOOBPadding`）：逻辑 `[M, K]` 与 Cube 需要的物理存储几何之间的差异，
   由编译器推导而不是用户填写。
2. **任务化与资源建模**（`MaterializeScheduleUnits`、`AnnotateMultiBufferEligible`、
   `EstimateLatency`）：把 kernel 拆成带流水标签（MTE1/MTE2/MTE3/Cube/Vector/Fixpipe/
   Scalar）的任务，并记录读写依赖、延迟、buffer 容量。
3. **Z3 求解**（`AutoSchedule`）：把“流水能不能成立”变成整数/布尔约束问题，一次性
   求解任务发射时刻、跨迭代重叠与 buffer 份数。
4. **落地**（`PrepareMultiBuffer` → `InsertSync` → `MaterializeMultiBuffer` →
   `AscendThreadSync` → `RewriteFlagToBuf`）：把解变成真实的同步指令与 buffer 版本。

**要理解的一句话**：Z3 不知道 Cube 是什么，硬件知识完全来自 TileLang 对访问区域、
资源占用、延迟和物理存储大小的建模。所以“给昇腾后端加能力”的大部分工作是**更准确
地建这个模**。

## 5. 工具链与工程落地

| 关注点 | 位置 |
| --- | --- |
| 安装（wheel / 源码） | `07-.../raw/tilelang-ascend/README.md`、`install_ascend.sh`、`set_env.sh` |
| 编译与运行、仿真 | `07-.../docs/TileLang-Ascend Programming Guide.md` 第 2 章（含 A5 Camodel） |
| 调试打印 | `T.printf` / `T.dump_tensor`；`07-.../examples/print/` |
| AICore 异常 dump | `07-.../docs/tutorials/ascend_aicore_exception_dump_{zh,en}.md` |
| 性能分析 | msProf；`07-.../docs/TileLang-Ascend Programming Guide.md` 第 6 章 |
| 入图（ACLGraph） | `07-.../examples/aclgraph/`；指南第 7 章 |
| PyTorch 集成 | `07-.../examples/torch_tl_ascend/`、`torch_npu` 任务队列支持（PR #3324） |
| 覆盖率/测试 | `07-.../docs/coverage_guide.md`、`docs/pytest_marker_guide.md` |

硬件与栈的官方描述见 `06-.../raw/hiascend/llms-content/cann/`（CANN、Ascend C、毕昇、
GE、runtime、HCCL）与 `hardware/`（处理器规格、集群）。

## 6. PTO：昇腾的 tile 指令集

* 规范：`06-.../raw/pto-spec/`（含形式化定义）与
  `https://gitcode.com/cann/pto-isa`。
* 教程/示例：`06-.../raw/PTO-Gym/`；手写 kernel：
  `06-.../raw/pto-kernels/`、`raw/megagdn-pto/`。
* Pythonic 写法：`06-.../raw/pto-dsl/`（华为自己的 tile DSL + JIT）。
* 硬件生成方向：`06-.../raw/pyCircuit/`（Python 前端 + MLIR 生成硬件）。
* TileLang 侧的对接：主干 PR #3310（PTO backend for Ascend950）、
  `tilelang/jit/adapter/pto/`、`tilelang/ascend/codegen/`。

理解 PTO 的意义：它把“tile 粒度”做成指令集层面的概念。TileLang 的 PTO 后端就是
“把 tile 程序降到 tile 指令”，这比降级到 Ascend C 少了中间层，也更容易做资源建模。

## 7. 未合入的前沿（必须跟）

| 内容 | 出处 |
| --- | --- |
| `T.PersistentKernel` + AutoPersistent | `08-.../raw/github-pr-3325.md` |
| `AutoSimtVF` pass（自动 SIMT/向量化分区） | `08-.../raw/github-pr-3320.md` |
| Ascend 算子生成与调优 agent | `08-.../raw/github-pr-3329.md` |
| 950 PTO codegen 改进（DeepGEMM） | PR #3330 |
| L0C fp32→f16/bf16 dual_copy on-path cast | PR #3328 |
| torch_npu 任务队列 | PR #3324 |
| causal sparse attention 例子 | PR #3322 |
| host/device profiling 关联 | PR #3321 |

（PR 编号大于 #3310 的都可以直接在 GitHub 上按号查；本 corpus 收录的是写这份材料时
已抓取的那几个。）

## 8. 入手建议（按投入产出比排序）

1. 先把 `09-...` 那篇原文 + 官方《Programming Guide》读完，建立整体图景（1 周）。
2. 跑通 `07-.../examples/gemm`、`flash_attention`、`deepseek_v4`，并对比 `examples_experiment/`。
3. 用 msProf 做一次端到端性能分析，建立昇腾的性能口径。
4. 读 `tilelang/ascend/pipeline.py` + `tilelang/ascend/transform/`，
   重点看 `AutoSchedule`、`InsertSync`、`MaterializeMultiBuffer`。
5. 挑一个未合入 PR（建议从 `AutoSimtVF` 或 `T.PersistentKernel` 起），复现并尝试改。
6. 读 PTO 规范，理解“tile 指令集”这条路线的终点形态。
