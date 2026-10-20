# 《从零到一：TileLang 系统教程》教材大纲

## 这份教材写给谁

有编程基础、会用 PyTorch、但没有写过 CUDA kernel、也没学过编译器的研发人员。
读完之后应当能做到三件事：

1. 用 TileLang 写出达到生产水平的大模型算子（GEMM / Attention / MoE / 量化）；
2. 用 TileLang 从零训练/推理一个 7B 级别开源模型，并理解每一步为什么这样做；
3. 读懂并修改 TileLang 的编译期与后端，能为昇腾或 NVIDIA 侧迭代功能。

全书基于 **TileLang v0.1.15** 与 **TileLang-Ascend v0.1.2.000-release**（见
[VERSIONS.md](VERSIONS.md)）。

## 组织原则

* **由浅入深**：第一部分宏观（为什么、是什么、和谁比），第二部分实用（怎么用、先跑起来），
  第三部分深入（编译器、后端、芯片、方法论）。
* **由宏观到微观**：每一章内部也按“它解决什么问题 → 用户怎么用 → 编译器怎么实现 →
  代码在哪”推进。
* **由稳定到前沿**：v0.1.15 里稳定的能力先讲透；只在 `main` 上、或只在未合入 PR 里的
  能力集中放在第 30 章，并显式标注状态。
* **理论与工程并重**：每个结论都要有“为什么”和“代码在哪一行”；不写没有出处的性能数字。
* **两条硬件主线**：NVIDIA 与华为昇腾；其他后端（ROCm / Metal / WebGPU / CPU）只在
  解释抽象设计时提及，不展开。

## 章节地图（39 章 / 5 部分）

### 第一部分 · 宏观：背景、设计思想、架构与对比（第 1–6 章）

> 目标：让读者建立正确的心智模型，知道 TileLang 在软件栈中的位置，以及它为什么长成现在这样。

| 章 | 标题 | 讲什么 | 依赖 |
| --- | --- | --- | --- |
| 01 | 为什么需要 Tile DSL | AI 算子的性能困境；硬件演进（SIMT→张量核→异步/专用存储）；CUDA/Triton/TileLang 的问题演进；DSL 谱系与“tile”概念的定义 | — |
| 02 | 从 PyTorch 到 kernel：算子在栈里的位置 | 训练/推理一次迭代里算子的角色；roofline 与算术强度；torch.compile/Inductor 边界；何时必须自己写 kernel | 01 |
| 03 | TileLang 的设计哲学与编程模型 | tile / dataflow / schedule 解耦；三层接口（beginner/developer/expert）；composable 的含义；与 Triton 的取舍 | 01,02 |
| 04 | 架构总览与一次编译的全流程 | 仓库结构与模块地图；Python→TIR→pass→codegen→runtime；各后端 pipeline 一览；版本与发布方式 | 03 |
| 05 | 横向对比：CUDA / Triton / CUTLASS-CuTe / TVM / cuTile / Ascend C / PTO-DSL | 七个同类物的定位、抽象层次、适用场景；用同一段 GEMM 做对照 | 03,04 |
| 06 | 芯片导论：NV GPU 与昇腾 NPU 的抽象对照 | 存储层级、执行单元、同步模型、异步机制；为什么“GPU 是 SIMT、NPU 是 SIMD”已经过时 | 01,05 |

### 第二部分 · 实用：从会用到敢用（第 7–19 章）

> 目标：PyTorch tutorial 风格，先跑起来、再写算子、最后完成端到端训练与推理。
> 每章都有可运行代码（无法在本机运行的，给出静态检查版本与验证步骤）。

| 章 | 标题 | 讲什么 | 依赖 |
| --- | --- | --- | --- |
| 07 | 快速上手：安装与第一个 kernel | 安装（含昇腾 wheel）；第一个 elementwise；打印生成代码并逐行读懂；JIT 与缓存 | 04,06 |
| 08 | 语言基础 | `T.Kernel` 网格语义、类型系统、buffer 与内存作用域、`T.Parallel`/循环/控制流、Python 兼容性边界 | 07 |
| 09 | 核心原语详解 | `T.copy`/`T.gemm`/`T.reduce_*`/`T.atomic_add`/`T.fill`/`T.tile.*`；每个原语的语义、参数、降级行为 | 08 |
| 10 | 布局与存储层级入门 | 为什么 layout 是接口；fragment/shared/register；swizzle 直觉；用 `plot_layout` 观察 | 09 |
| 11 | 流水线与自动调优 | `T.Pipelined` 与多缓冲；autotuner 参数空间；如何阅读调优结果 | 09,10 |
| 12 | 调试与性能分析 | 编译错误定位；`lower_trace`/`pass_diff`；Nsight Compute（NV）与 msProf（昇腾）；常见错误模式 | 07–11 |
| 13 | 算子实战 I：逐元素、归一化与 RoPE | 融合的意义；把多个 op 合成一个 kernel；昇腾与 NV 的差异点 | 09,12 |
| 14 | 算子实战 II：GEMM 全家族 | 从朴素到 intrinsic 的阶梯；split-K/stream-K/grouped；fp8/int4/blockscaled；如何对齐 cuBLAS/CATLASS | 11–13 |
| 15 | 算子实战 III：Attention 家族 | FA1/2/3 的演进；decode 与 varlen；paged KV；稀疏与索引类算子 | 14 |
| 16 | 算子实战 IV：MoE 与 MLA | DeepSeek 系列算子（MLA/NSA/DSA/MoE）；超长 context 的工程约束 | 15 |
| 17 | 端到端 I：用 HuggingFace 资产从零训练一个 7B 模型 | 数据（HF datasets）、tokenizer、训练循环、把热点算子换成 TileLang kernel、混合精度、checkpoint 与恢复、评测 | 13–16 |
| 18 | 端到端 II：推理与部署 | KV cache 管理、continuous batching、量化推理、与 vLLM/TileRT 风格集成、延迟与吞吐基准 | 17 |
| 19 | 昇腾实战：在昇腾上开发与调优算子 | Ascend C 对照；昇腾专属原语；msProf 调优；ACLGraph 入图；从 Ascend C 迁移到 TileLang 的实操 | 12,13 |

### 第三部分 · 深入：编译期、后端与芯片（第 20–29 章）

> 目标：从“会用”到“能改”。每章都要落到 v0.1.15 的具体文件与行号。

| 章 | 标题 | 讲什么 | 依赖 |
| --- | --- | --- | --- |
| 20 | IR 与编译期管线精读 | TIR 结构；CUDA 与 Ascend 两条 pipeline 的完整 pass 列表；每个承重 pass 做什么；如何自己 dump IR | 04,12 |
| 21 | 布局推导（Layout Inference）原理与实现 | layout 代数；约束传播与求解；昇腾 fractal/NZ/OOB padding；冲突与转换代价 | 20 |
| 22 | 流水线、多缓冲与同步插入 | PipelinePlanning/InjectSoftwarePipeline；多缓冲容量约束；ThreadSync 与昇腾 InsertSync 的算法 | 20 |
| 23 | 自动调度与约束求解（昇腾） | 任务化与资源建模；延迟估计；Z3/SMT 求解；多缓冲物化；为什么 Ascend C 难而这里可自动化 | 22 |
| 24 | NVIDIA 后端：代码生成与硬件特性 | CUDA codegen；TMA；`mma`→`wgmma`→`tcgen05`；warp specialization；cluster/DSM；CuTe DSL 后端 | 20–22 |
| 25 | 昇腾后端：代码生成与硬件特性 | Ascend C & PTO 路线；AscendNPU IR 路线；Cube/Vector/MTE/UB/L1/L0A-L0C；`dual_copy`；两条路线怎么选 | 20–23 |
| 26 | 运行时、JIT、ABI 与框架集成 | kernel 缓存；host wrapper 与 packed API；torch/cython/tvm_ffi 适配；autograd 集成；启动开销与持久化 kernel | 20 |
| 27 | 低精度、量化与数值正确性 | fp16/bf16/fp8/MXFP4/int4；block scaling；累加与舍入；数值验证方法 | 09,14 |
| 28 | 算子设计方法论与性能模型 | roofline 与 TileSight 类解析模型；tile 尺寸选择；memory-bound vs compute-bound 的决策；从性能模型反推实现 | 14,15,24,25 |
| 29 | 编译器工程：写 pass、测试与回归 | 写一个 TIR pass 并挂进 pipeline；pass 的测试与 IR diff 基线；性能回归检测；代码风格与评审要求 | 20–23 |

### 第四部分 · 前沿与参与（第 30–31 章）

| 章 | 标题 | 讲什么 | 依赖 |
| --- | --- | --- | --- |
| 30 | 前沿探索：只有 main / 未合入的方向 | 多后端解耦、TVM→apache/tvm+tvm-ffi、CuteDSL 后端、`T.PersistentKernel`、`AutoSimtVF`、Ascend agent、PTO 的演进；如何跟踪上游 | 23–25 |
| 31 | 贡献指南 | 从 issue 到 PR 的完整流程；改编译器/加后端/加算子三种贡献的检查清单；昇腾双仓库协作模式；社区沟通 | 29,30 |

### 第五部分 · DeepSeek 昇腾全栈：一次真实开源的完整拆解（第 32–39 章）

> 目标：把 2026-09-30 DeepSeek 面向昇腾开源的整套基础设施（官方发布说明见 corpus
> `14-deepseek-ascend-stack/raw/deepseek-article-20260930.md`）逐层拆开讲透。
> **TileLang 是主线**：每个组件都要回答“它的算子用 TileLang 怎么写 / 为什么这里
> 没有用 TileLang / 两者边界在哪”。其他工具链（Ascend C、PTO、HCCL/URMA、
> DeepJIT、CUDA/PTX）的用法与原理作为必要的辅助展开。
>
> 与前四部分的关系：第一、二部分给出 TileLang 的用法，第三部分给出编译期与后端原理，
> 本部分把这些能力放到一个真实的生产栈里检验。所以本部分是**综合应用**，
> 也是全书“华为优先”路线的落点。

| 章 | 标题 | 讲什么 | 依赖 |
| --- | --- | --- | --- |
| 32 | DeepSeek 昇腾全栈：这次开源了什么 | 官方文章与六个仓库的清单；与 NVIDIA 侧组件的一一对应关系；训练/推理全流程里各自的位置；版本与可用性如何核验；本部分的学习路线与验证边界 | 19,25 |
| 33 | TileKernels（上）：一份算法，两个后端 | 库结构（`*_kernel.py` 共享层 / `*_cuda.py` / `*_asc.py` + 运行期选择）；以 top-k gate 为例逐行对照 CUDA 与 Ascend 两种 TileLang 写法；量化算子族（per-token/per-block cast、SwiGLU 融合）；RoPE 与 Rand | 09,11,25 |
| 34 | TileKernels（下）：Engram 与流形超连接 | Engram 条件记忆与 O(1) 查表；门控 kernel（融合 RMSNorm、前后向、梯度归约、Sinkhorn）；mHC 的 Sinkhorn 归一化与 mix 拆分/应用、多层重计算；“不是矩阵乘的算子”怎么写 TileLang | 33 |
| 35 | DeepGEMM-Ascend：把 GEMM 压到硬件极限 | MAD 原语抽象与 fractal/对齐/地址计算；稀疏加载与协程流水；scale factor 的 int16 打包与 MN-major；MQA logits（FixPipe bound）；MegaMoE 融合；HC prenorm 为什么用 TileLang 写 | 14,25,27 |
| 36 | FlashMLA 昇腾稀疏注意力：算法与流水 | DSA 与 online softmax；skip-scale；1C2V 的推导；rescale 放在哪；ND2NZ 的时机；gather2、KV cache layout、高速 ND2NZ、icache、L1 bank 配平；哪些结论能回到 TileLang 表述 | 15,23,25 |
| 37 | DeepSelect：算法先行的 TopK | 阈值筛选 + RadixSelect 的算法与期望复杂度证明；CUDA 侧用 PTX 与位运算压低常数的技巧；昇腾实现的差异；“算法决定上限、硬件决定常数” | 13,28 |
| 38 | DeepEP-Ascend：专家并行通信与计算重叠 | EP dispatch/combine、FP8 dispatch、延迟 epilogue；HCCL/HCOMM、UBMEM、URMA；DeepJIT 运行时编译；PP/Bucket/Engram 现状；128 卡超节点带宽读法；DualPipe 与 EPLB 的位置 | 16,26,35 |
| 39 | 端到端：128 卡超节点上的训练与推理 | 把六个组件拼回一条流水线；一次 decode step 的账；训练侧的并行与通信重叠；MFU/带宽检查清单；从“能跑”到“跑满”；如何为这套生态贡献 | 32–38 |

## 每章的必备结构

每章由一组相互对应的 Markdown、图表和代码文件组成（见
[仓库内容维护说明](../docs/content.md)），必须包含：

1. **学习目标**（读完能做什么，3–6 条，可检验）
2. **前置知识**（指向前面章节或 corpus 资料）
3. **正文**（概念 → 用法 → 实现 → 代码位置，逐层深入）
4. **代码**（`code/` 目录，可离线静态检查；标注是否需要真实硬件）
5. **小结**（一页以内）
6. **习题**（`exercises.md`，8–15 题，含概念题、代码阅读题、动手题）
7. **习题答案**（`solutions.md`，与习题一一对应，含解题思路而非只有答案）
8. **参考资料**（`references.md`，corpus 路径 + 上游链接 + 版本号）

## 章节之间的依赖与并行说明

* 第 1–6 章是全书的公共前提，写作时可以互相引用，但内容不能互相依赖实现细节。
* 第 7–19 章依赖第 1–6 章；同部分内部只有弱依赖（如 15 依赖 14 的 GEMM 概念）。
* 第 20–29 章依赖第 4 章与第 7–12 章的工具方法。
* 第 30–31 章依赖第三部分。
* 第 32–39 章依赖第二、三部分；部分内部只有弱依赖（33 前置，35–38 可并行，39 收口）。
* 所有章节都可以并行写作；跨章引用一律用“见第 N 章”的相对表述，不写死路径。

## 明确不做的事（范围边界）

* 不深入 AMD ROCm / Metal / WebGPU / CPU 后端，只在第 5、6 章作为抽象设计对照提及。
* 第五部分只覆盖 2026-09-30 这次开源清单里的组件，以及理解它们所必需的架构论文与
  配套仓库（Engram、DSA、DualPipe、EPLB、DeepJIT 等）；不展开 DeepSeek 的其他
  开源项目（3FS、smallpond、Harness、DeepSpec 等）。
* 不写 CUDA 语言的完整教程（只覆盖读懂 TileLang 产物所需的部分）。
* 不写昇腾硬件的完整手册（只覆盖算子开发所需部分）。
* 不给没有出处的性能承诺。
