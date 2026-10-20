# 第 06 章 芯片导论：NV GPU 与昇腾 NPU 的抽象对照

## 学习目标

读完本章，你应当能够：

1. 用“容量—带宽—延迟”三者的不对称，解释为什么加速器必须有片上存储（on-chip
   storage），以及为什么算子必须先“分块搬上来、算完再搬回去”。
2. 画出 NVIDIA 侧的 `grid → thread block → warp → thread` 执行层次与
   `寄存器 → shared memory → L2 → HBM` 存储层次，并说清每层的角色与已知容量。
3. 按“为什么需要下一代”讲出 Tensor Core 三代指令（`mma` → `wgmma` → `tcgen05`）
   各自把哪一类资源从寄存器里拿了出来，以及 TMA、warp specialization、thread block
   cluster 与分布式共享内存（DSM）分别解决什么问题。
4. 画出昇腾侧的 `Cube/AIC` 与 `Vector/AIV` 分工，以及 `GM → L1 → L0A/L0B → Cube →
   L0C → Fixpipe`、`GM → UB → Vector → UB → GM` 两条数据流，并说清 UB/L1/L0A/L0B/L0C
   的角色。
5. 用 `MTE1/MTE2/MTE3/Fixpipe` 与成对的 Set/Wait 解释昇腾的多缓冲流水，并说明
   `T.Pipelined` 的软件流水为什么离不开“什么时候可以覆盖同一块 buffer”这件事。
6. 用“存储层级 / 执行单元 / 数据搬运 / 同步 / 编程粒度”五个维度列出两台机器的对照
   表，并解释为什么“GPU 是 SIMT、NPU 是 SIMD”这个概括在昇腾 950 上已经过时。
7. 指出本章引用的每一条硬件数字哪些来自厂商文档、哪些来自社区文章，以及哪些是
   **未确认**。

## 前置知识

* 第 1 章：算术强度与机器平衡点、存储层级与“搬数据比算数据贵”、tile 的三要素。
* 第 5 章：CUDA / Triton / TileLang 等七种同类物的定位（本章不重复横向对比）。
* 会用 Python、写过 PyTorch；**不需要**任何 CUDA、Ascend C 或芯片背景。本章每个硬件
  名词第一次出现时都会先用一两句话说明它解决什么问题。

本章是全书硬件认知的起点。后面所有章节（第 10 章布局、第 11 章流水、第 21–25 章
后端）都会回到本章建立的这张机器图上。请把“数字”和“结构”分开读：结构是本章的
重点，数字只用来把结构变得可感。

> **本章的验证边界。** 写作环境是 macOS（Apple Silicon），**没有 NVIDIA GPU，也没有
> 昇腾 NPU**（见 `../../VERSIONS.md`）。本章不含任何本机实测的性能数字；所有容量、带宽、
> 延迟数字都来自厂商公开文档或 corpus 里的社区一手文章，并逐条标注来源与测量/发布条件。
> 凡找不到可靠出处的，一律写“未确认”，不填估计值。

---

## 1. 问题的由来：为什么必须有“片上存储”和“分块搬运”

### 1.1 数据离计算单元越远，就越贵

在 PyTorch 里，`y = A @ B` 只写了一行。但硬件要完成它，必须先把 `A`、`B` 从外部显存
（NVIDIA 叫 HBM，昇腾叫 GM / Global Memory）搬到计算单元旁边，算完再把结果搬回去。

这两件事的成本完全不对称。第 1 章引用过 CUDA 官方文档的一个对比：

> “When accessing uncached local or global memory, there are hundreds of clock cycles of
> memory latency.”
> “The latency of most arithmetic instructions is typically 4 cycles on devices of compute
> capability 7.0.”
> 来源：`../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md`
> （CUDA C++ Best Practices Guide，抓取日期 2026-09-30）。条件：未缓存的 local/global
> 访存延迟；算术延迟以 compute capability 7.0 设备为例。

也就是说，**一次访存要等上百个时钟周期，一次浮点运算只要几个时钟周期**。如果每次
计算都直接去 HBM 取数，计算单元大部分时间都在等，而不是在算。

### 1.2 解法一：把数据放到离计算单元更近的地方

硬件给出的第一个解法是**存储层级（memory hierarchy）**：离计算单元越近，容量越小、
延迟越低、带宽越高。加速器在计算单元旁边放了几块小得多的片上存储（on-chip memory），
把“马上要算的数据”先搬到这里。

以 NVIDIA H100（Hopper）为例，厂商公开资料给出的层级是：

| 层级 | 角色 | H100 SXM5 的公开数字 | 来源 |
| --- | --- | --- | --- |
| 寄存器文件（register file） | 计算指令直接读写的操作数 | 每 SM 256 KB（65536 个 32-bit 寄存器） | R1，R2 |
| shared memory / L1 | block 内共享、可编程的片上 SRAM | 每 SM 合并 256 KB，shared 可配置到 228 KB；单 block 最大 227 KB | R2 |
| L2 cache | 所有 SM 共享的片上缓存 | 50 MB | R1，R2 |
| HBM / 显存 | 容量最大的外部存储 | 80 GB HBM3，带宽超过 3 TB/s | R1 |

（厂商公开资料，**非本机实测**；H100 的算力/带宽在官方博客里标注为“preliminary”。
本表只用于建立层级关系，不作为性能基线。）

昇腾侧是同一张图、不同的名字。官方 Ascend C 文档给出的 AI Core 内部存储与数据流是：

| 层级 | 角色 | 数据流 |
| --- | --- | --- |
| Unified Buffer（UB） | 向量/标量计算的输入和输出 | `GM → UB → Vector → UB → GM` |
| L1 Buffer | Cube 计算的中转区，暂存反复使用的数据 | `GM → L1 → L0A/L0B` |
| L0A / L0B Buffer | Cube 指令的左/右矩阵输入 | `L1 → L0A/L0B → Cube` |
| L0C Buffer | Cube 指令的输出；累加时也作为输入 | `Cube → L0C → Fixpipe` |
| BT Buffer | 存放矩阵计算的 Bias | `L1 → BT Buffer` |
| FP Buffer | 存放量化参数、ReLU 参数等 | `L1 → FP Buffer` |

来源：`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_0008_html.md`
（“基本架构”，CANN 商用版 Ascend C 算子开发指南；以下称“Ascend C 基本架构”）。

### 1.3 解法二：搬一次，算很多次

光有片上存储还不够。数据显示：一个 $M=N=K=4096$ 的 fp16 矩阵乘，理论最小访存量是
$96$ MiB，而计算量约 $1.37\times10^{11}$ FLOP（见第 1 章）；如果朴素地让每个输出元素
都去 HBM 读一遍 `A` 的一行和 `B` 的一列，访存量会放大到不可接受。

解法是**分块（tiling）**：把输出矩阵切成小块，每次只把一块计算需要的 `A`、`B` 子矩阵
搬到片上，在片上反复复用，算完再搬回下一块。**同一份片上数据被复用的次数，就是分块
把 HBM 流量压下来多少倍**。这就是 tile 视角的物理动机，也是第 1 章“tile 三要素”的
硬件对应物。

示例 `code/01-onchip-footprint/onchip_footprint.py` 会替你算：给定 tile 形状和 dtype，
这块 tile 占多少片上空间、开 N 份多缓冲后还放不放得下。这是本章唯一在写作机上真正
运行过的代码。

### 1.4 解法三：让搬运和计算同时进行

分块之后，程序的时间线变成“搬一块 → 算一块 → 搬一块 → 算一块”。如果搬运和计算
严格串行，计算单元在每个搬运阶段都在闲置。

解法是**多缓冲（multi-buffer / double buffer）**：把片上缓冲区开成两份或多份，第 0
份在算的时候，第 1 份正在搬。理想情况下，搬运的延迟被完全藏在计算的背后。这带来两个
必须管理的资源：**片上容量**（多开一份就多占一份空间）和**同步**（“搬完了”与
“可以覆盖了”必须被正确表达）。

昇腾官方 Ascend C 流水优化文档把这个动机讲得很直白：假设 CopyIn、Compute、CopyOut
三阶段耗时相同均为 $t$，不使用 double buffer 时 Vector 单元利用率只有 $1/3$；把数据
一分为二以后，搬运和计算可以并行，Vector 闲置被显著缓解。

> 来源：`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/developer/zh_developer_techArticles_20240819-1.md`
> （“Ascend C 算子性能优化实用技巧 01，流水优化”，昇腾社区官方技术文章，2024-08-19）。
> 注意这是**厂商的示意模型**（假设三阶段耗时相等），不是对某一真实算子的实测。

### 1.5 本章要回答的问题

两台机器都想解决同一个问题：**怎么让数据在正确的时间出现在正确的位置，并且让计算
单元尽量不空转。** 但它们的执行模型、可编程存储、搬运路径和同步协议都不一样。
本章接下来分四步：

1. 先建共同的四个维度（存储层级 / 执行单元 / 数据搬运 / 同步），第 2 节；
2. 讲 NV 侧从 SIMT 到 Tensor Core 三代、TMA、warp specialization、cluster 的演进
   第 3 节；
3. 讲昇腾侧 Cube/Vector 分工、UB/L1/L0、MTE 流水、Set/Wait、SIMT+SIMD，第 4 节；
4. 把两边压成一张对照表，并解释“GPU 是 SIMT、NPU 是 SIMD”为什么已经过时
   第 5 节。

---

## 2. 概念与模型：描述任何加速器的四个维度

在分别看两台机器之前，先把“描述一台加速器”的四组词汇定义清楚。后面无论 NV 还是
昇腾，都只是在这四个维度上填不同的值。

### 2.1 维度一：存储层级，数据住在哪

**存储层级（memory hierarchy）** 回答“数据住在哪、多大、多快”。描述任何一层需要
四个量：

* **容量（capacity）**：能放多少字节。容量决定了“一块 tile 最大能开多大、能开几份
  多缓冲”。
* **带宽（bandwidth）**：每秒能搬多少字节。带宽决定了搬运一批数据要多久。
* **延迟（latency）**：从发出请求到数据可用要等多久。延迟决定了“要多少在飞并行的
  请求才能把流水填满”。
* **可编程性（programmability）**：程序员能不能显式地用它、以什么粒度用它。

最后一条最容易被忽略，但它决定了编程模型的形状：寄存器通常由编译器分配、程序员不能
直接寻址；shared memory / UB / L1 是可以显式声明和读写的“可编程片上存储”；而
Tensor Core 的累加器在 Blackwell 上干脆搬进了第三种存储（Tensor Memory）。**每一代
硬件新增一种可编程存储，语言就多一类必须显式表达的对象。**

### 2.2 维度二：执行单元，谁来算

**执行单元（execution unit）** 回答“谁在算、以什么粒度并行”。这里要区分两件事：

1. **并行层级（parallel hierarchy）**：程序怎么被切分给一群执行者。NV 的是
   `grid → block → warp → thread`；昇腾 950 的是一个一维的 AI Core 网格，核内再分
   Cube / Vector / Scalar 三种单元。
2. **执行风格（execution style）**：这一群执行者怎么共享同一条指令。

   * **SIMT（Single Instruction, Multiple Threads）**：一条指令驱动多个**线程**，
     每个线程有自己的寄存器状态和地址。GPU 的 warp（32 个线程）就是 SIMT 的单位：
     同一 warp 的 32 个线程执行同一条指令，但可以访问不同地址。
   * **SIMD（Single Instruction, Multiple Data）**：一条指令驱动一个**固定宽度的
     向量寄存器**，所有 lane 一起做同一个运算。昇腾的 Vector Core 传统上就是 SIMD。

   SIMT 强调“每个线程独立寻址”，SIMD 强调“一条指令处理一整个向量”。这个区别
   在本章第 5 节会成为“为什么旧概括过时”的关键。

3. **专用单元（specialized unit）**：除了通用计算，硬件往往还有专用单元。NV 的是
   Tensor Core（矩阵），昇腾的是 Cube（矩阵）；昇腾还有独立的 Vector 单元负责
   向量/逐元素，Scalar 负责流程控制和地址计算。

### 2.3 维度三：数据搬运，数据怎么流动

**数据搬运（data movement）** 回答“谁把数据从 A 层搬到 B 层”。描述它要看三点：

* **路径（path）**：哪些层之间能直接搬，哪些必须经过中转。昇腾把每条路径都固定给了
  一个搬运单元（MTE1/MTE2/MTE3/Fixpipe）；NV 侧普通拷贝走 `LDG/STS` 指令，异步拷贝走
  `cp.async`，批量张量搬运走 TMA。
* **是否异步（asynchronous）**：发出搬运指令的线程要不要等数据到达。“已经发出去了”
  和“已经做完了”是两件事。这是理解现代加速器同步协议的前提。
* **随路变换（in-flight transform）**：搬运过程中能不能顺便做类型转换、格式转换、
  量化或 reduction。昇腾的 MTE 在搬运时可执行随路数据格式/类型转换，Fixpipe 在
  `L0C → GM/L1` 时也能做转换（Ascend C 基本架构）；Hopper 的 TMA 在写回时可以指定
  element-wise reduction。

### 2.4 维度四：同步，谁等谁

**同步（synchronization）** 回答“什么时候必须等”。现代加速器里至少有三类等待：

1. **数据就绪（producer → consumer）**：计算开始前，搬运必须真的完成。
2. **缓冲归还（consumer → producer）**：下一轮搬运覆盖同一块 buffer 前，上一轮读取
   必须真的结束。
3. **跨核/跨单元依赖（cross-core）**：NV 的 cluster barrier、昇腾的 Cube/Vector 之间
   的生产—消费关系。

两台机器用的是同一类机制，只是名字不同：

| 机器 | “就绪/归还”的机制 | 最小单位 |
| --- | --- | --- |
| NV | `__syncthreads()`、`cp.async` 的 commit/wait、`mbarrier`（含 async transaction barrier）、cluster barrier | 线程 / warp / block / cluster |
| 昇腾 | 成对的 SetFlag / WaitFlag，绑定在一对流水（如 `MTE2 → MTE1`）之间 | 流水单元（MTE/Cube/Vector/Scalar） |

关键直觉：**同步不是“让大家对齐”，而是“表达依赖”。** 少一处同步，往往不是编译
报错，而是偶发的数值错误；多一处同步，则直接损失性能。第 11 章讲流水、第 22 章讲
同步插入算法，都会回到这一点。

### 2.5 一条贯穿全章的主线

把四个维度串起来，就是本章的叙事主线：

```text
存储层级越来越深  →  必须显式声明 tile 放在哪（维度一）
执行单元越来越专用 →  必须显式说明谁负责搬、谁负责算（维度二）
搬运越来越异步     →  必须显式管理「在飞」的数据（维度三）
流水越来越深       →  必须显式表达就绪与归还（维度四）
```

NV 和昇腾在这条主线上走的是同一个方向，只是每一步的**具体资源单位**不同。下面分别展开。

---

## 3. NV 侧：从 SIMT 线程模型到异步张量流水

### 3.1 执行层次：grid / block / warp / thread

CUDA 的执行层次是：

```text
Grid（一次 launch 的所有 block）
 └── Thread Block / CTA（一个 block，最多 1024 个线程）
      └── Warp（32 个线程，SIMT 的最小调度单位）
           └── Thread（程序员写的那一条「我做什么」）
```

H100 的公开规格里，这个层次的约束是：每个 warp 32 个线程，每个 SM 最多 64 个 warp、
2048 个线程、32 个 block，每个 block 最多 1024 个线程，每个 cluster 最多 16 个 block。
来源：R1（Hopper 架构博客，Table“Compute capability: V100 vs. A100 vs. H100”）。

**SIMT 的代价与收益**：程序员只写“第 `threadIdx.x` 个线程算什么”，硬件负责把一个 warp
的 32 条线程指令合并发射。收益是灵活（每个线程可以独立寻址、独立分支）；代价是当
同一 warp 的线程走不同分支时，硬件只能串行执行不同路径（warp divergence），并且
访存必须“合并（coalesce）”才能高效。CUDA 官方 Best Practices Guide 的原话是：

> “Global memory loads and stores by threads of a warp are coalesced by the device into
> as few as possible transactions.”
> 来源：`../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md`，
> “Coalesced Access to Global Memory”一节。条件：compute capability 6.0 及以上的
> 一般性说明。

官方文档对两种机器的定性对比也很清楚：CPU 用少量重线程追求低延迟，GPU 用大量轻线程
追求高吞吐（来源同 R3，第 94–98 行附近）。这就是为什么 GPU 需要成百上千个在飞线程
来“隐藏延迟”，某个 warp 在等访存时，调度器切到另一个 warp。

图见 `figures/03-nv-execution-hierarchy.md`。

### 3.2 存储层级：寄存器 / shared memory / L2 / HBM

NV 侧的存储层级与上一节的通用模型完全对应：

| 层级 | 容量（H100 SXM5，厂商公开资料） | 谁可见 | 典型用途 |
| --- | --- | --- | --- |
| 寄存器文件 | 256 KB / SM（64K × 32-bit） | 单线程私有，编译器分配 | 累加器、MMA 操作数 fragment |
| shared memory | 可配置最多 228 KB / SM；单 block 最多 227 KB | block 内所有线程共享 | `T.alloc_shared` 的 tile、swizzle 布局 |
| L1 / texture | 与 shared 合并为 256 KB / SM | 自动缓存 | 访存合并缓冲 |
| L2 | 50 MB（H100） | 全 GPU 共享 | 跨 block 复用的数据 |
| HBM | 80 GB HBM3，> 3 TB/s（H100 SXM5） | 全局 | 模型权重、激活、KV cache |

来源：R1（Hopper 架构博客）、R2（Hopper Tuning Guide 的
shared memory carveout 列表：`0, 8, 16, 32, 64, 100, 132, 164, 196, 228 KB`）。
均为**厂商公开资料，非实测**；H100 博客里的算力数字明确标为“preliminary”。

Blackwell（compute capability 10.0，如 B200）的公开数字：shared memory 每 SM 228 KB、
单 block 最多 227 KB，与 Hopper 相同；GB200 的 L2 提升到 126 MB；B200 支持 HBM3/HBM3e，
容量最高 180 GB。来源：R4（Blackwell Tuning Guide）。

> **容量不是全部。** 官方文档特别指出：CUDA 会为每个 block 保留 1 KB，静态 shared
> 分配仍限制在 48 KB，超过需要显式 opt-in（R2、R4）。写 kernel 时这些边界会直接决定
> “这个 tile 开几份多缓冲”。

### 3.3 Tensor Core 三代：`mma` → `wgmma` → `tcgen05`

上一节讲的通用 SIMT 单元负责逐元素计算，但矩阵乘的吞吐来自**专用矩阵单元**：
Tensor Core。它从 Volta 起出现，至今经历过三代**编程接口**的演进。理解这三代的关键，
不是背指令名，而是问：**这一代把哪一类资源从寄存器里拿了出来，从而允许更大的 tile、
更深的流水？**

#### 第一代：warp 级 `mma`（Volta / Turing / Ampere）

`mma.sync` 是 **warp 级**指令：一个 warp 的 32 个线程协作完成一个小矩阵乘
（如 `m16n8k16`），操作数 `A`、`B` 和累加器 `C`/`D` 都以 **寄存器 fragment** 的形式
分布在 32 个线程的寄存器里。PTX 文档在 `mma` 一节里明确写：

> “Similar to `wmma`, `mma` also requires computation to be performed collectively by all
> threads in the warp however distribution of matrix elements across different threads in
> warp needs to be done explicitly before invoking the `mma` operation.”
> 来源：`../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md`，9.7.16 节。

**它解决了什么**：第一次让 GPU 有了远超通用 FP32 的矩阵乘吞吐。

**它留下了什么**：累加器和操作数都住在寄存器里，而寄存器是每线程最稀缺的资源
（最多 255 个 32-bit 寄存器 / 线程）。想要更大的 `M×N` tile，就要么拆成很多条 `mma`
手工拼、要么牺牲 occupancy。布局也必须由程序员显式安排成 fragment 分布。这正是
第 1 章说的“数据怎么摆第一次成了必须显式表达的约束”。

#### 第二代：warpgroup 级异步 `wgmma`（Hopper，sm_90a）

Hopper 把粒度从 warp 提升到 **warpgroup**（4 个连续 warp、共 128 个线程），并且把
矩阵指令变成**异步**的：

> “A warpgroup is a set of four contiguous warps... The `wgmma` instructions perform
> warpgroup level matrix multiply-and-accumulate operation...”
> 来源：PTX ISA 9.7.17 节（R5）。

PTX 文档给出的执行步骤是：`wgmma.fence` → 发射 `wgmma.mma_async` → `wgmma.commit_group`
→ `wgmma.wait_group`（R5，9.7.17 节开头）。支持的形状是 `m64nNk16` 这类更大的
`M=64` 形状。

**它把什么拿出来了**：

* **操作数可以放在共享内存里**。PTX 步骤第一步写的是“Load matrices A, B and D into
  registers **or into shared memory**”（R5）；`wgmma` 用描述符（descriptor）引用
  共享内存里的 `B`。这样就不必先把整个 `B` 塞进寄存器。
* **发射与完成分离**。异步语义让“发射一条矩阵指令”和“等它算完”成为两件事，
  于是可以一边等上一组 `wgmma` 完成、一边发射下一组。这就是 Hopper 深流水的来源。

**它留下了什么**：累加器 `D` 仍然在寄存器里，而且 `wgmma.group` 的 commit/wait 又是
一套新的同步协议，程序员要手工维护。

#### 第三代：CTA 级 `tcgen05` 与 Tensor Memory（Blackwell，sm_100a/sm_100f）

Blackwell 的第五代 Tensor Core 做了一件结构性的事：**给累加器专门造了一块片上存储
Tensor Memory（TMEM）**。

> “The 5th generation TensorCore has dedicated on-chip memory that is specialized for
> use by TensorCore operations. This Tensor Memory is organized as a two-dimensional
> matrix where the horizontal rows are called lanes and the vertical columns are called
> columns. On architecture `sm_100a`/`sm_100f`, the 5th generation TensorCore's Tensor
> Memory has a two-dimensional structure of 512 columns and 128 rows per CTA, with each
> cell being 32-bits in size.”
> 来源：PTX ISA 9.7.18.1 节（R5）。

`tcgen05.mma` 的操作数位置也随之改变：

> “the `A` matrix has shape MxK, in either Tensor Memory or Shared Memory；the `B` matrix
> has shape KxN, in Shared Memory of the current CTA and optionally in peer CTA；the `D`
> matrix is of the shape MxN, in Tensor Memory.”
> 来源：PTX ISA 9.7.18.10 节（R5）。

**它把什么拿出来了**：**累加器从寄存器搬到了 TMEM**。这意味着矩阵乘的“在飞”状态
不再占用宝贵的寄存器，`M×N` tile 可以做得更大，寄存器可以留给其他工作。付出的代价
是：又多了一种必须显式管理的存储（TMEM 要 `tcgen05.alloc` 动态分配），以及一套基于
`mbarrier` 的完成协议。

三代放在一起看：

| 代际 | 指令 | 粒度 | 累加器住哪 | 操作数 A 住哪 | 同步方式 |
| --- | --- | --- | --- | --- | --- |
| 1 | `mma.sync` | warp（32 线程） | 寄存器 fragment | 寄存器 fragment | 同步（指令返回即完成） |
| 2 | `wgmma.mma_async` | warpgroup（128 线程） | 寄存器 fragment | 寄存器**或** shared memory | `commit_group` / `wait_group` |
| 3 | `tcgen05.mma` | CTA（可选 2-CTA） | **Tensor Memory** | Tensor Memory 或 shared memory | `mbarrier`（完成即 post） |

来源：R5（PTX ISA 9.7.16 / 9.7.17 / 9.7.18 各节）。图见
`figures/04-tensor-core-generations.md`。

**为什么要讲这代际演进？** 因为 TileLang 的 `T.gemm` 会根据目标架构选择这三条路径
之一自动生成。锁定版本的 `T.gemm` docstring 明确写着：

> “On Hopper, if the compiler selects WGMMA lowering, TileLang inserts the corresponding
> wait implicitly. On Blackwell TCGEN5MMA, TileLang inserts the corresponding
> `mbarrier_wait_parity(...)` implicitly after issue.”
> 来源：S5，`_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:177-181`。

也就是说，同一段 tile 程序在 A100、H100、B200 上会落到 `mma`、`wgmma`、`tcgen05` 三套
不同的指令与同步协议上。这正是“编译器接管调度空间”的一个具体含义。

### 3.4 TMA：把“批量张量搬运”交给硬件

异步搬运的起点是 Ampere 的 `cp.async`：线程可以直接把 global memory 的数据拷到
shared memory，不经过寄存器。但它仍然需要线程自己生成所有地址、循环整个拷贝区域。

Hopper 引入了 **TMA（Tensor Memory Accelerator）**，把地址生成和数据搬运整个交给
硬件：

> “TMA allows applications to transfer 1D and up to 5D tensors between global memory and
> shared memory, in both directions, as well as between the shared memory regions of
> different SMs in the same cluster... a single thread can issue large data movement
> instructions to the TMA unit.”
> 来源：R2（Hopper Tuning Guide，1.4.1.2 节）。

Hopper 架构博客补了一个关键点：TMA 是一个描述符（descriptor）驱动的单元，“A single
thread creates a copy descriptor before launching the TMA, and from then on address
generation and data movement are handled in hardware”（R1）。它还会自动处理 stride、
offset 和边界（boundary）计算。这对处理非规则形状的 tile 特别重要。

**为什么这件事改变了编程模型**：既然一个线程就能驱动整块张量搬运，那么“专门派几个
warp 只负责搬数据、其余 warp 只负责算”就变得自然了。这就是下一节的 warp
specialization。

### 3.5 Warp specialization：让不同的 warp 干不同的事

传统 CUDA 里，一个 block 的所有线程执行同一份代码，靠 `if (threadIdx.x < ...)` 分工，
然后 `__syncthreads()`。当搬运是异步的、计算是异步的，这种“大家一起同步”的模型
就浪费了重叠机会：搬数据的 warp 在搬完后无事可做，算的 warp 在等数据时也闲置。

**warp specialization（warp 专门化）** 把 block 内的 warp 分成角色：一部分 warp 专做
生产者（搬数据、发 TMA、驱动 mbarrier），另一部分专做消费者（等 mbarrier、跑 MMA）。
Hopper Tuning Guide 把它和 TMA 直接联系在一起：

> “Enables users to write warp specialized codes, where specific warps specialize on data
> movement between the different memory spaces while other warps only work on local data
> within the SM.”
> 来源：R2，1.4.1.2 节。

生产者—消费者之间用 **mbarrier**（异步 barrier）交接：生产者 `Arrive` 后可以继续做别的
事，消费者 `Wait` 到所有生产者都到达才继续。Hopper 还新增了 **asynchronous transaction
barrier**。它不只数“多少个线程到达”，还数“多少个字节到达”，于是 TMA 搬运完成本身
就可以直接触发 barrier（R1，Figure 17 一节）。

这是“同步维度”的一次升级：从“等所有线程到齐”变成“等一批字节到齐”。TileLang 在
CUDA 侧把它实现为一组 pass（`AutoWarpSpecialization` →
`MaterializeWSSchedule` → `ProducerConsumerWarpSpecialized`，见
`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:103-108`），语言层则暴露
`T.WarpSpecialize(...)`（`tilelang/cuda/language/warpgroup.py:19`）。

### 3.6 Cluster 与分布式共享内存（DSM）

Hopper 在执行层次里加了一层：**thread block cluster**。

> “A cluster is a group of thread blocks that are guaranteed to be concurrently scheduled,
> and enable efficient cooperation and data sharing for threads across multiple SMs.”
> “Distributed shared memory allows direct SM-to-SM communications for loads, stores, and
> atomics across multiple SM shared memory blocks.”
> 来源：R1（Hopper 架构博客）。

cluster 的动机是：有些算子需要的片上数据超过单个 block 的 shared memory 容量，但又
不值得绕回 L2/HBM。DSM 让 cluster 内的多个 block 可以直接读写彼此的 shared memory，
相当于把“block 内共享”扩展成“cluster 内共享”。Tuning Guide 也提醒，DSM 访问应当
像 global memory 一样合并、按 32 字节对齐（R2、R4）。

公开的规模上限：portable cluster size 最大 8；H100 和 B200 通过非 portable opt-in
可以到 16（R2、R4）。启用更大 cluster 可能降低同时活跃的 block 数（R2）。

cluster 也带来了新的同步：`barrier.cluster.arrive` / `barrier.cluster.wait`。TileLang
在 CUDA 侧暴露了 `T.cluster_arrive()`、`T.cluster_wait()`、`T.cluster_sync()`、
`T.block_rank_in_cluster()`（S9，`tilelang/cuda/language/cluster.py:27-46`）。

---

## 4. 昇腾侧：Cube / Vector 分工与 MTE 搬运流水

### 4.1 执行单元：AIC/Cube、AIV/Vector 与 Scalar

昇腾的 AI Core 从设计上就把计算分成三类单元。官方 Ascend C 文档的定义是：

> “计算单元：包括 Cube（矩阵）计算单元、Vector（矢量）计算单元和 Scalar（标量）计算
> 单元。”
> 来源：R6（Ascend C 基本架构）。

三者的分工：

* **Cube（矩阵计算单元）**：负责矩阵运算。“Cube 每次执行可完成两个 float16 类型的
  16x16 矩阵的乘法操作”（R6）。它访问 L0A/L0B（输入）和 L0C（输出/累加）。
* **Vector（矢量计算单元）**：负责向量运算。“向量计算单元执行向量指令，类似于传统的
  单指令多数据（SIMD）指令，每个向量指令可以完成多个操作数的同一类型运算”（R6）。
  Vector 的源数据和目标数据都要求存储在 Unified Buffer 中，首地址和操作长度通常要求
  32B 对齐（R6）。
* **Scalar（标量计算单元）**：负责标量运算、循环控制、分支判断，以及“Cube/Vector 等
  指令的地址和参数计算”，“功能上可以看做一个小 CPU”。官方特别提醒 Scalar 计算能力较弱，
  应尽量减少 `if/else` 和变量运算（R6）。

在“分离模式”下，AI Core 被拆成 **Cube Core（AIC）** 和 **Vector Core（AIV）** 两个
独立核，“每个核都有自己的 Scalar 单元，能独立加载自己的代码段，从而实现矩阵计算与
向量计算的解耦”（R6）。官方给出的产品对应关系是：Atlas A2 / A3 及 Atlas 350 加速卡
为分离模式；Atlas 推理/训练系列产品为耦合模式（R6）。

> **AIC/AIV 与 CUDA 的对应不是一对一的。** 在 CUDA 里，“谁搬数据、谁算矩阵”是同一个
> block 内的线程分工；在昇腾里，矩阵计算和向量计算从一开始就是两个不同的核，这可能
> 带来跨核的生产—消费依赖。这是 TileLang 昇腾后端必须处理的第一个结构差异
> （见第 24、25 章）。

### 4.2 存储层级：UB / L1 / L0A / L0B / L0C 的角色

官方 Ascend C 文档给出的内部存储与用途（R6，表 1）：

| 存储 | 角色 | 官方原文要点 |
| --- | --- | --- |
| L1 Buffer | 通用内部存储、Cube 的中转区 | “AI Core 内比较大的一块数据中转区，可暂存 Cube 计算单元需要反复使用的一些数据从而减少从总线读写的次数” |
| L0A / L0B Buffer | Cube 的左/右矩阵输入 | “Cube 指令的输入” |
| L0C Buffer | Cube 的输出 | “Cube 指令的输出，但进行累加计算的时候，也是输入的一部分” |
| Unified Buffer（UB） | 向量/标量的输入输出 | “向量和标量计算的输入和输出” |
| BT Buffer | Bias | “存放矩阵计算中的 Bias” |
| FP Buffer | 量化/ReLU 参数 | “Fixpipe Buffer，存放量化参数、Relu 参数等” |

两条典型数据流（R6，原文“典型的数据流”）：

```text
Vector 计算：  GM → UB → Vector → UB → GM
Cube 计算：    GM → L1 → L0A/L0B → Cube → L0C → Fixpipe → GM
                                                    └────→ L1
```

**容量定位**：官方文档按产品型号给出不同大小，并通过
`GetCoreMemSize` 接口在运行时查询（R6）。本 corpus 里能核到的**具体数字**只有两个，
且都带示例条件，不能当成所有昇腾产品的通用规格：

* Unified Buffer：官方最佳实践文章在讲 bank 冲突时给出“Unified Buffer 总大小为 192K，
  划分为 48 个 bank，每个 bank 由 128 行组成，每行长度为 32B，48 个 bank 又组织为 16 个
  bank group，每个 group 3 个 bank”。来源：
  `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_best_practices_10_00020_html.md`
  （Atlas A2 上下文的最佳实践）。**这是该文档举例的某一型号，不是“昇腾 UB 都是 192K”。**
* L1 Buffer：API 文档的示例注释写“该示例 L1 Buffer 可用大小为 512KB”，条件是该
  `GetMatmulApiTiling` 示例。来源：
  `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_API_ascendcopapi_atlasascendc_api_07_0666_html.md`。

**未确认**：L0A、L0B、L0C 与 BT/FP Buffer 的具体容量，在本 corpus 内没有找到一个通用、
可跨型号引用的数字；官方要求用 `GetCoreMemSize` 运行时查询（R6）。因此本章不给出
L0A/L0B/L0C 的容量数值。

对齐约束则有明确出处（对齐表在
`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_0011_html.md:43-51`，
即 R6 的姊妹页）：UB 32 Byte、L1 32 Byte、L0A/L0B 512 Byte、
L0C 64 Byte、BT 64 Byte、Fixpipe 64 Byte。这些数字会直接影响 tile 的合法形状。

### 4.3 搬运流水：MTE1 / MTE2 / MTE3 / Fixpipe

昇腾把每条搬运路径固定给一个搬运单元（MTE = Memory Transfer Engine）。官方定义
（R6，表 2）：

| 搬运单元 | 负责的通路 |
| --- | --- |
| MTE1 | `L1 → L0A/L0B`、`L1 → BT Buffer` |
| MTE2 | `GM → {L1, L0A/L0B}`、`GM → UB` |
| MTE3 | `UB → GM` |
| Fixpipe | `L0C → {GM, L1}`、`L1 → FP Buffer`，搬运过程中可完成格式/类型转换 |

官方还给出对齐建议：MTE2 的 `GM → {L1, L0A/L0B}` 通路“基于分形大小搬运，搬运时满足
Cache Line 大小对齐，性能更优”；`GM → UB` 也“基于 Cache Line 大小搬运性能更优”（R6）。

**这里的结构与 NV 最大的不同**：NV 的搬运是“线程执行访存指令”，路径由指令决定；
昇腾的搬运是“流水单元执行既定路径”，路径是硬件结构的一部分。这解释了为什么昇腾
程序员脑子里要维护一张明确的“数据从哪来、经过哪条流水、到哪去”的时间表。

另一条重要区别：**Cube/Vector/MTE 是彼此独立的流水**。上面引用的官方流水优化文章
把它们列为不同的指令队列，“Vector 指令队列（V）、Cube 指令队列（M）、Scalar 指令
队列（S）和搬运指令队列（MTE1/MTE2/MTE3）”，并指出“不同指令队列间的相互独立性
和可并行执行的特点，是 double buffer 优化机制的基石”（R7）。

### 4.4 同步：Set / Wait 与多缓冲

既然 MTE 和计算是独立流水，就必须显式表达“谁的数据好了”。官方给出两种机制
（R6，“典型的指令流”一节）：

* **PipeBarrier**：序列内部的执行顺序约束，“保证前序指令中所有数据读写全部完成，
  后序指令才开始执行”。
* **SetFlag / WaitFlag**：在两个指令序列之间建立一对“锁”：

  > “SetFlag：当前序指令的所有读写操作都完成之后，当前指令开始执行，并将硬件中的
  > 对应标志位设置为 1。”
  > “WaitFlag：当执行到该指令时，如果发现对应标志位为 0，该序列的后续指令将一直被
  > 阻塞；如果发现对应标志位为 1，则将对应标志位设置为 0，同时后续指令开始执行。”
  > 来源：R6。

注意 `WaitFlag` 会自动把标志位清 0。所以每一个 Set 只对应一次 Wait，这是一对
“信号量”。官方特别提醒：“通常情况下，开发者基于编程模型……不需要关注同步”，
自行控制会带来复杂度（R6）。在 TileLang 里，这套 Set/Wait 由编译器插入。

**多缓冲为什么必须和同步一起算。** 考虑 double buffer：buffer 有 slot 0 和 slot 1，
第 0 轮用 slot 0，第 1 轮用 slot 1，第 2 轮又回到 slot 0。第 2 轮的 MTE2 要覆盖 slot 0
之前，必须确认第 0 轮的消费者（Cube 或 Vector）真的读完了。这需要一对**反向**的
Set/Wait：`MTE2 → Cube` 表达“数据就绪”，`Cube → MTE2` 表达“缓冲可以归还”。

TileLang-Ascend 的编程指南把完整生命周期写成了四步（来源：
`_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md`，
“同步原语”一节）：

1. consumer 在作用域入口预先归还 token；
2. producer 获取 token、写缓冲并通知 consumer；
3. consumer 获取 token、完成所有读取并归还给下一轮 producer；
4. 作用域退出前消费最后一次归还的 token。

同一份文档给出的例子正是本章 1.4 节讲的那条路径：“MTE2 写 L1、MTE1 读取后，正向使用
`MTE2 → MTE1`，反向归还使用 `MTE1 → MTE2`”；并强调如果 MTE1 在多个内层循环中持续
读取该 L1 缓冲，必须在**最后一次**读取之后再归还。

图见 `figures/06-ascend-pipeline.md`。

### 4.5 昇腾 950：同时支持 SIMT 与 SIMD

这是本章最需要小心处理的一点，因为**厂商资料本身存在文档范围差异**，必须分开陈述。

**事实 A（厂商硬件页）**：昇腾 950PR 的产品特性写“支持 SIMD+SIMT 混合编程架构，
性能与易用性同步提升”；产品描述写“更多精度格式、SIMT+SIMD 混合编程、灵衢高速互联”。
来源：
`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/hardware/zh_hardware_processor.md`
（昇腾 NPU 处理器页，厂商公开资料）。

**事实 B（厂商 Atlas 加速卡页）**：Atlas 350 加速卡“采用 Ascend 950PR”，“支持 SIMT、
SIMD 编程”，“支持 SIMT，逻辑类、离散访存类算子性能和开发易用性提升”。来源：
`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/hardware/zh_hardware_accelerator-card.md`。

**事实 C（厂商 CANN 文档）**：AI Core 的 SIMT 编程“目前仅支持 Atlas 350 加速卡”，
并且给出了 SIMT 的抽象硬件模型，“每个 Vector Core 包含计算单元、Share Memory 即
Unified Buffer、寄存器和堆栈空间”；SIMT 读 GM 时通过 Data Cache 中转，而 Data Cache
“是 Unified Buffer 中预留的一部分空间，其最大容量为 128KB，最小容量为 32KB”。来源：
`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_10064_html.md`。

**事实 D（社区一手文章）**：昇腾 950 后端“本身同时支持 SIMT 和 SIMD”，并且“SIMT
路径的访问 UB 的带宽只有 SIMD 的一半”。来源：
`../tilelang-corpus/09-zhihu-da-zhi-ruo-yu/raw/zhihu-dazhiruoyu-deepseek-ascend.md`
（知乎“乱序摸鱼”，2026；**社区文章，非厂商官方文档，数字未经官方复核**）。

**怎么合起来读**：厂商硬件页与 Atlas 卡页都明确把 SIMT 列为 950PR/Atlas 350 的能力，
因此“昇腾 950 支持 SIMT”有厂商出处。CANN 文档那句“目前仅支持 Atlas 350 加速卡”
是在说明**当前 SIMT 编程支持的产品范围**，Atlas 350 正是 950PR 的加速卡形态，
两者并不矛盾，而是“芯片能力”与“CANN 编程支持范围”两个口径。至于“SIMT 路径 UB
带宽只有 SIMD 的一半”，只有社区来源，本章把它标为**社区说法、未确认**，不作为结论。

锁定版本的 TileLang 昇腾语言方言直接暴露了这两种执行方式：`T.SimtVF(threads=...)`
与 `T.SimdVF()`（S8，`tilelang/ascend/language/frame.py:107`、`:179`），后端 README 也写明
“Ascend supports mixing SIMT and SIMD code within a single kernel”
（S1，`tilelang/ascend/README.md:133-136`）。这是本章第 5 节破除旧概括的实证。

---

## 5. 对照：把两台机器压成一张表

### 5.1 五个维度的对照

下表把第 2 节的四个维度加上“编程粒度”，做成 NV 与昇腾 950 的对照。每一行都给出
来源；**标“社区”的行来自知乎一手文章，非厂商官方文档。**

| 维度 | NVIDIA GPU（Hopper / Blackwell） | 昇腾 950（Atlas 350 加速卡） | 来源 |
| --- | --- | --- | --- |
| 程序组织 | grid → thread block（CTA）→ warp → thread | 一维 AI Core 网格；核内 Cube/Vector/Scalar；SIMT 区域另行定义线程域 | R1、R6、R13 |
| 矩阵计算 | Tensor Core（`mma`/`wgmma`/`tcgen05`） | Cube（AIC），每次 16×16 fp16 矩阵乘 | R5、R6 |
| 向量计算 | 通用 FP32/INT 核心 + SIMT 线程 | Vector（AIV），SIMD 指令；950 另支持 SIMT | R6、R8 |
| 显式片上存储 | 寄存器、shared memory（+ Blackwell 的 Tensor Memory） | UB、L1、L0A、L0B、L0C、BT、FP | R1、R5、R6 |
| 数据搬运 | 普通访存、`cp.async`、TMA；TMA 负责地址/stride/边界 | MTE1/MTE2/MTE3/Fixpipe，每条路径固定到一个单元 | R1、R2、R6 |
| 同步 | `__syncthreads`、`cp.async` commit/wait、`mbarrier`、cluster barrier | 成对的 SetFlag/WaitFlag，绑定在一对流水之间；PipeBarrier | R1、R6 |
| 多缓冲单位 | shared memory buffer（字节数 / block） | UB、L1 的 buffer 版本数（与流水号绑定） | R2、R6、R13 |
| 编程粒度 | thread / warp / warpgroup / CTA / cluster | AI Core 网格；Cube/Vector 独立核；TileLang 里以 tile 为单位 | R1、R6 |

补充几条有明确出处的量化对照：

| 项 | NV（H100 SXM5） | 昇腾 950PR | 来源 |
| --- | --- | --- | --- |
| 外部存储带宽 | HBM3 > 3 TB/s | 1.6 TB/s（Atlas 350 卡为 1.4 TB/s） | R1、R9 |
| L2 | 50 MB | 128 MB | R1、R9 |
| 外部存储容量 | 80 GB HBM3 | 128 GB | R1、R9 |
| 矩阵单元数 | 4 Tensor Core / SM（132 SM） | Cube Core 32（另有 28 的型号档位） | R1、R9 |
| 向量单元数 | （以 FP32 核心计）128 / SM | Vector Core 64（另有 56 档位） | R1、R9 |

来源说明：R1 = Hopper 架构博客；R9 =
`../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/hardware/zh_hardware_processor.md`
（昇腾 NPU 处理器页）。**两侧都是厂商公开资料，非实测**；昇腾页把两档数字写成
“1784/1561”这样的“满配/另一档”格式，本表只取其中一档并标注。跨厂商的算力口径
（例如 MXFP4 与 FP16 累加的对应关系）不完全可比，本章不做算力横比。

### 5.2 为什么“GPU 是 SIMT、NPU 是 SIMD”已经过时

这句话在过去是对“默认执行风格”的粗略概括，但它今天至少有三处不成立：

**第一，昇腾 950 本身同时支持 SIMT 和 SIMD。** 厂商硬件页写“支持 SIMD+SIMT 混合编程
架构”（R9）；CANN 文档给出了 SIMT 的线程/线程块/Data Cache 模型（R8）；锁定版本
TileLang 的昇腾方言在同一 kernel 里提供 `T.SimtVF` 和 `T.SimdVF` 两个区域
（S8、S1）。所以“NPU = SIMD”在 950 上已经不成立。

**第二，NVIDIA GPU 也早就不只是 SIMT。** 现代 GPU 把异步搬运（`cp.async`、TMA）、
专用矩阵单元（Tensor Core）、warp specialization 和生产者—消费者流水都塞进了同一套
执行体系（R1、R2）。同一份 CUDA 程序里，通用线程走 SIMT，Tensor Core 走 warpgroup/CTA
级的矩阵指令，TMA 由单个线程驱动硬件单元。用一个 SIMT 概括 NV 也只覆盖了一半。

**第三，真正的区别不在“SIMT vs SIMD”，而在四个维度上。** 更准确的问法是：

* 数据必须住在哪一级存储、以什么粒度？（NV：寄存器/shared；昇腾：UB/L1/L0）
* 谁负责搬、走哪条路径？（NV：线程发指令 + TMA；昇腾：固定到 MTE1/2/3/Fixpipe）
* 同步的默认单位是什么？（NV：warp/block/cluster + mbarrier；昇腾：流水单元对 + Set/Wait）
* 程序员表达的最小并行单位是什么？（NV：thread 或 warpgroup；昇腾：AI Core 上的 tile）

这也是 TileLang 存在的意义之一：它允许程序员继续用同一套 tile 语言，而把“落到哪种
执行风格”交给后端。锁定版本的昇腾 README 把这一点写得很直接，同一个 `T.Kernel` 里
可以同时出现 `T.gemm`（Cube）、`T.SimtVF(...)`（SIMT 线程）和 `T.SimdVF()`（SIMD）
（S1，`tilelang/ascend/README.md:113-136`）。

---

## 6. 实现机制：TileLang 如何把两台机器收进同一套语言

本章讲的是硬件，但你最终要写的是 TileLang。这一节把“硬件抽象”映射到“语言里能看到
的东西”，并给出锁定版本的文件位置。**本节的结论全部在 `_vendor/` 里逐个核对过 API
存在性与签名，但没有任何一条在真实硬件上运行过。**

### 6.1 存储层级 → 存储作用域（storage scope）

TileLang 里每种片上存储都对应一个分配函数：

| 硬件层级 | TileLang API | scope | 来源 |
| --- | --- | --- | --- |
| NV shared memory | `T.alloc_shared(shape, dtype)` | `shared.dyn` | S2 |
| NV 寄存器累加器 | `T.alloc_fragment(shape, dtype)` | `local.fragment` | S2 |
| 昇腾 UB | `T.alloc_shared(shape, dtype)` | `shared.dyn` | S3 |
| 昇腾 L1 | `T.alloc_l1(shape, dtype)` | `shared.l1` | S3 |
| 昇腾 L0A / L0B | `T.alloc_l0a` / `T.alloc_l0b` | `shared.l0a` / `shared.l0b` | S3 |
| 昇腾 L0C | `T.alloc_l0c(shape, dtype)` | `shared.l0c` | S3 |

注意昇腾方言的 `alloc_shared` **故意覆盖**了通用版本：它的注释写着“Ascend UB handles bool
buffers in the requested scope directly”（S3，`allocate.py:15-25`）。这说明“同一个高层
概念在两种硬件上落点不同”不是抽象泄漏，而是被显式承认的事实。

`alloc_l0a` / `alloc_l0b` 的布局是“延迟到消费它的 gemm 操作，由 gemm 根据 transpose
标志推出 K-major 还是 MN-major”（S3）；`alloc_l0c` 则会标注昇腾固定的 L0C 累加器布局
（S3）。这正是第 21 章“布局推导”要处理的核心问题。

### 6.2 数据搬运 → `T.copy` 与 `T.dual_copy`

通用侧 `T.copy(src, dst)` 描述“把一块 tile 从 A 层搬到 B 层”。在 CUDA 侧，同一个
`T.copy` 可能降级为普通 SIMT 拷贝、`cp.async`、TMA，或用 `disable_tma=True` 强制
不降级到 TMA（S4，`tilelang/cuda/language/copy_op.py:42`）；TMA 路径还会消费 L2 的
eviction hint（S4，`:91`）。

昇腾侧多了一个 `T.dual_copy`，它把一次拷贝按 M 或 N 方向**对半拆给两个 AIV**
（S5，`tilelang/ascend/language/copy_op.py:53`）。支持的路径是：`L0C→UB`（硬件双目标
拷贝）、`GM→UB`、`UB→GM`、`UB→L1`（后者还会顺带做 ND→NZ 格式转换）。硬约束包括：
`L0C→UB` 至少是二维，N-split 时完整 N 必须是 32 的倍数；被拆维度的完整 extent 必须
是偶数（S5）。

这个 API 的存在本身就是“两台机器不同”的证据：在 NV 上，“一个 block 的线程怎么分工”
通常被编译器藏起来；在昇腾上，Cube 的结果必须显式搬给两个 Vector 核，于是这个动作
需要一个名字。

### 6.3 执行单元 → `T.gemm` 与 `T.SimtVF` / `T.SimdVF`

* `T.gemm(A, B, C, ...)` 是默认的同步 GEMM。它会按目标选择 `mma`/`wgmma`/`tcgen05`，
  并在 Hopper/Blackwell 上隐式插入对应的 wait（S6，`tilelang/language/gemm_op.py:160`、
  `:177-181`）。需要手工异步调度时，CUDA 方言额外提供 `T.wgmma_gemm(...)`、
  `T.tcgen05_gemm(...)`（S7，`tilelang/cuda/language/gemm_op.py:226,264`）。
* 昇腾方言用 `T.SimtVF(threads=...)` 定义一个 SIMT 区域（可一到三维线程域），
  用 `T.SimdVF()` 定义一个**无线程维度**的寄存器级 SIMD 区域（S8，`frame.py:107`、
  `:179`）。后端 README 说明：`T.gemm` 跑在 Cube 上，`T.SimtVF`/`T.SimdVF` 跑在
  Vector 上，编译器默认处理核分配、调度和同步（S1，`README.md:113-121`）。

### 6.4 同步 → 编译器插入，而不是手写

NV 侧的 warp specialization 由 `AutoWarpSpecialization` → `MaterializeWSSchedule` →
`ProducerConsumerWarpSpecialized` 三个 pass 处理（S10，
`tilelang/cuda/pipeline.py:103-108`）。

昇腾侧的流水线更完整地体现了“硬件事实 → 约束求解 → 落地同步”的链条：自动调度启用时，
pass 顺序是 `EstimateLatency → AutoSchedule → AssignCore → PrepareMultiBuffer →
ResolveCore → InsertSync → MaterializeMultiBuffer → LowerScheduledTIR`
（S11，`tilelang/ascend/pipeline.py:107-115`）。其中：

* `EstimateLatency` 给每个任务估计延迟；
* `AutoSchedule` 决定任务顺序、重叠和多缓冲版本数；
* `InsertSync` 把依赖关系兑现成真实的 Set/Wait；
* `MaterializeMultiBuffer` 把一个 buffer 真正扩成多个物理槽位。

对读者来说，这里最重要的结论是：**你只需要按顺序写“搬、算、回写”，编译器负责生成
Set/Wait。** 后端 README 在“Synchronization”一行写的是“None（Just write operations in
order!）”，“What the compiler does”是“Infers pipeline and Cube/Vector dependencies;
inserts paired flags”（S1，`README.md:150`）。

### 6.5 一个同题对照：GEMM + ReLU

昇腾后端 README 的 Quick Start 用一段代码同时展示了 Cube、Vector、UB/L1/L0C 和
dual copy（**下面是按锁定版本 README 复述的骨架，不是运行结果**）：

```python
with T.Kernel(num_blocks) as bx:
    A_l1 = T.alloc_l1((block_M, block_K), T.bfloat16)
    B_l1 = T.alloc_l1((block_N, block_K), T.bfloat16)
    C_l0c = T.alloc_l0c((block_M, block_N), T.float32)
    C_ub = T.alloc_shared((block_M // 2, block_N), T.float32)

    for tile in T.Persistent([M // block_M * n_tiles], num_blocks, bx):
        m, n = tile // n_tiles * block_M, tile % n_tiles * block_N
        for k in T.Pipelined(K // block_K, num_stages=2):
            T.copy(A[m, k * block_K], A_l1)
            T.copy(B[n, k * block_K], B_l1)
            T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=(k == 0))

        T.dual_copy(C_l0c, C_ub)          # L0C → 两个 AIV 的 UB
        with T.SimtVF(threads=128):       # Vector 核上的 SIMT 区域
            for i, j in T.Parallel(block_M // 2, block_N):
                C_ub[i, j] = T.max(C_ub[i, j], 0)
        T.dual_copy(C_ub, C[m : m + block_M, n : n + block_N])
```

来源：S1，`tilelang/ascend/README.md:80-100`（为节省篇幅略去了原例中两处
`l2_cache_ctrl` 参数，其余与原例一致；完整原文见该文件）。同一份 README 也强调：
自动调度处理任务顺序和依赖，但**不**意味着任意 `T.Parallel` 都会自动张量化成 SIMD
指令（S1，`:153`）。

请把这段代码与本章第 4 节的数据流对照着读：

* `T.copy(A..., A_l1)` 走 MTE2（`GM → L1`）；
* `T.gemm(A_l1, B_l1, C_l0c)` 隐含 `L1 → L0A/L0B`（MTE1）和 Cube 计算；
* `T.dual_copy(C_l0c, C_ub)` 走 Fixpipe 的 `L0C → UB`，并按 M 拆给两个 AIV；
* `T.SimtVF` + `T.Parallel` 在 UB 上做 ReLU；
* `T.dual_copy(C_ub, C[...])` 走 MTE3（`UB → GM`）。

你在语言层只写了“数据先去哪、计算在哪发生、结果流向哪”，MTE 路径、Set/Wait、
多缓冲版本都交给后端。这就是第 3 章讲的“数据流留给人、调度空间交给编译器”在本章
硬件层面的落点。

---

## 7. 进阶与陷阱

1. **把“片上”当成“免费”。** 片上存储是分块、多缓冲、occupancy 三者相互挤压的结果。
   官方文档里 H100 的 shared memory 上限（228 KB/SM、227 KB/block）是硬约束（R2）；
   昇腾的 UB/L1 容量按型号不同，需要 `GetCoreMemSize` 查询（R6）。开多缓冲之前先算
   账。这正是 `code/01-onchip-footprint/` 的用途。

2. **把“发出去了”当成“做完了”。** 异步搬运、异步 MMA、TMA、MTE 都是“发射”与
   “完成”分离的。少了 wait，得到的是偶发错误，而不是编译报错。NV 用 `mbarrier` /
   `wgmma.wait_group`，昇腾用 `WaitFlag`，本质相同。

3. **把同步当成“越少越好”或“越多越好”。** 少一处会错，多一处会慢。昇腾的
   `WaitFlag` 一次 Set 对应一次 Wait（R6）；TileLang 的 `InsertSync` 是为了在保证
   正确的前提下把同步放到最需要的位置（S11）。

4. **把容量数字当成通用规格。** 本章列出的 UB=192K、L1=512KB 都是**特定示例**的条件
   数字，不是“昇腾通用规格”；H100 的 228 KB 是 SXM5 的公开上限，不同型号会不同。
   报数字时一定要带型号和出处。

5. **用“SIMT / SIMD”给两台机器贴标签。** 见 5.2 节：950 同时支持两者，NV 也早就
   不只是 SIMT。正确的做法是逐项问“存储 / 执行单元 / 搬运 / 同步 / 粒度”。

6. **相信厂商宣传口径的算力横比。** 昇腾页给的是 MXFP4 满配算力，NV 页给的是按
   dtype 分列的 preliminary 峰值；两者的累加精度、稀疏口径、测量条件都不同。本书
   只把这类数字当“量级参考”，不做“谁更快”的结论。

7. **把社区文章的定性结论当事实。** 本章引用的知乎文章是社区一手文章，其中的
   “性能与 Ascend C 打得有来有回”“SIMT 路径 UB 带宽只有 SIMD 一半”等说法没有公开
   基线与测量口径，本章只作为**定性描述**引用，并明确标注为社区来源；具体数字一律
   标为未确认。

---

## 8. 本章小结

* **为什么需要片上存储和分块搬运**：访存延迟（上百周期）远大于算术延迟（几个周期），
  所以必须把数据搬到离计算单元更近的地方；而片上容量有限，所以必须分块、复用，才能
  把 HBM 流量压下来；再靠多缓冲让搬运和计算重叠。
* **描述任何加速器的四个维度**：存储层级（容量/带宽/延迟/可编程性）、执行单元
  （并行层级/执行风格/专用单元）、数据搬运（路径/是否异步/随路变换）、同步
  （就绪/归还/跨核）。
* **NV 侧**：`grid → block → warp → thread`；存储是“寄存器 → shared/L1 → L2 → HBM”
  （H100：256 KB 寄存器/SM、共享 256 KB/SM、shared 可配到 228 KB、L2 50 MB、HBM3
  > 3 TB/s）。Tensor Core 三代把不同资源从寄存器里拿了出来：`mma`（warp 级、全寄存器）
  → `wgmma`（warpgroup 级、操作数可进 shared、异步）→ `tcgen05`（CTA 级、累加器进
  Tensor Memory）。TMA 把批量张量搬运交给硬件；warp specialization 让生产者/消费者
  warp 分工；cluster + DSM 把“block 内共享”扩展到“跨 SM 共享”。
* **昇腾侧**：Cube（AIC，矩阵）与 Vector（AIV，向量/标量）是不同核，Scalar 负责调度；
  存储是“UB / L1 / L0A / L0B / L0C / BT / FP”，路径由 MTE1（L1→L0）、MTE2（GM→L1/UB）、
  MTE3（UB→GM）、Fixpipe（L0C→GM/L1）固定承担；同步是成对的 SetFlag/WaitFlag，多缓冲
  必须和调度、同步一起算。**950 同时支持 SIMT 与 SIMD**。
* **对照表**（见 5.1 节）把五个维度并列后可以看到：两台机器解决同一个问题，但
  “可编程存储的单位、搬运路径的归属、同步的默认粒度”都不同。
* **“GPU 是 SIMT、NPU 是 SIMD”已经过时**：昇腾 950 同时支持 SIMT 与 SIMD（厂商硬件页
  + TileLang 的 `T.SimtVF`/`T.SimdVF`），NV 也早已是“SIMT + 异步 + 专用矩阵单元”的
  复合体系。正确的问法是逐维度比较。
* **TileLang 的位置**：它把存储层级变成 storage scope、把搬运变成 `T.copy`/`T.dual_copy`、
  把执行单元变成 `T.gemm`/`T.SimtVF`/`T.SimdVF`、把同步交给编译器插入。你在语言层写
  数据流，硬件差异由后端吸收。

下一章（第 7 章）会把这些概念变成可运行的第一步：安装 TileLang，写出第一个 kernel，
并把生成的代码逐行读回本章的机器图。

## 延伸阅读

* 想回到“为什么慢”的性能模型：第 1 章、第 2 章。
* 想看七种同类方案的横向对比：第 5 章。
* 想按存储作用域写第一个程序：第 7–9 章。
* 想看布局如何在两台上分别推导：第 10 章、第 21 章。
* 想看流水与同步插入算法：第 11 章、第 22 章。
* 想看两台的代码生成与硬件特性：第 24 章（NV）、第 25 章（昇腾）。
* 本章引用的全部原始资料与核验状态：见 `references.md`。
* 想亲手算一遍 tile 占用与流水重叠：`code/` 下三个示例。
