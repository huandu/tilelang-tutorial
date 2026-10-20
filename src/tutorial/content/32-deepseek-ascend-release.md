# 第 32 章 DeepSeek 昇腾全栈：这次开源了什么

## 学习目标

读完本章，你应当能够：

1. **背出这次开源的清单**：2026-09-30 的官方文章列出了哪六个组件，每个组件用一句话
   说清它解决什么问题，以及官方为什么要用“一一对应”来组织这次发布。
2. **画出组件地图**：说清 TileLang（语言/编译）→ TileKernels（用 TileLang 写的算子库）
   → DeepGEMM-Ascend / FlashMLA / DeepSelect（极限性能计算算子）→ DeepEP-Ascend（通信）
   这条主线，以及 DeepJIT 作为共用运行时的位置；每个组件落在哪一层、依赖谁、被谁依赖。
3. **做出一一对应表**：把 NVIDIA 侧与昇腾侧的组件逐个对上，并说清“一一对应”在工程上
   意味着什么（同一套 API 与开发流程、已有知识可迁移），以及**哪里并不对应**，至少
   举出 DeepGEMM-Ascend 的 scaling factor 打包格式这个反例。
4. **把组件放回流水线**：在一次训练 step（MoE 路由 → EP dispatch → grouped GEMM →
   SwiGLU → combine → mHC → 优化器）和一次推理请求（prefill / decode、DSA 稀疏注意力）
   里指出每个组件的位置。
5. **看懂“昇腾 950 的 128 卡超节点”这句话**：结合第 6 章的加速器抽象与第 19 章的
   Cube/Vector/UB/L1 模型，说明它指的是什么，以及为什么在这个方案里通信库是主角之一。
6. **自己核验**：会读每个仓库 README 的 `Requirements` / `News` / `Ongoing` 三节，
   判断一个组件在什么硬件与 CANN 版本上可用，会区分“已发布 / 仍在开发 / 未实现”，
   并说清为什么本部分的每个结论都要绑定到一次 commit 快照。

## 前置知识

* **第 6 章**：NV GPU 与昇腾 NPU 的抽象对照。本章反复用到“存储层级 / 执行单元 /
  数据搬运 / 同步”这四个维度，以及“昇腾 950 同时支持 SIMT 与 SIMD”这条结论。
* **第 19 章**：昇腾实战。本章提到的 AIC（Cube）/ AIV（Vector）、UB / L1 / L0A / L0B /
  L0C、MTE1/2/3、以及“两棵树”（适配器仓库与主干昇腾方言）都出自那里。
* **第 14 章（GEMM）、第 15 章（Attention）、第 16 章（MoE 与 MLA）**：本章只讲这些
  算子在整条流水线里的位置，不复述算法；读不懂的算子细节可以回查这三章。
* **第 25 章（昇腾后端）与第 28 章（算子设计方法论）**：想读懂后面各章的实现时用得到，
  本章只要求知道“TileLang 在昇腾上生成 Ascend C 或 PTO”这件事。

你**不需要**有昇腾 NPU，也**不需要**有 NVIDIA GPU：本章是导览章，所有内容都能
静态核对。

### 术语速查（读本章时随时回查）

本章出现的名词很多，先集中解释一遍，后面各章不再重复：

| 词 | 一句话解释 | 在哪深入 |
| --- | --- | --- |
| **TileLang** | 面向 tile 的 DSL 与编译器，把 tile 程序降成 CUDA / Ascend C / PTO | 第 1–4 章 |
| **昇腾方言 / 昇腾后端** | TileLang 里负责生成昇腾代码的那一部分（`tilelang/ascend/`） | 第 25 章 |
| **TileKernels** | 用 TileLang 写的生产算子库，CUDA 与 Ascend 双后端 | 第 33、34 章 |
| **DeepGEMM-Ascend** | 昇腾版 GEMM 库，主体 Ascend C，mHC prenorm 用 TileLang | 第 35 章 |
| **FlashMLA** | 注意力算子库，昇腾侧提供 DSA 稀疏 prefill / decode | 第 36 章 |
| **DeepSelect** | TopK 算子，供 DSA 的索引与 sampler 用 | 第 37 章 |
| **DeepEP-Ascend** | 专家并行通信库（EP dispatch / combine 等） | 第 38 章 |
| **DeepJIT** | 头文件式 C++20 JIT 运行时，CUDA / 昇腾双后端 | 第 38 章（与第 26 章对照） |
| **EP / EPBuffer** | 专家并行；`EPBuffer` 是 DeepEP 系列的公开接口 | 第 16、38 章 |
| **DSA** | DeepSeek Sparse Attention，先筛选 KV 再做注意力 | 第 16、36 章 |
| **MQA logits** | 索引器（indexer）给每个 (q, kv) 打分的算子 | 第 35、36 章 |
| **mHC** | 流形超连接（Manifold HyperConnection） | 第 34、35 章 |
| **超节点** | 把几十到上百块 NPU 连成一个整体的大规模互联域 | 第 6、39 章 |
| **UBMEM / URMA / HCCL / HCOMM** | 昇腾侧的共享内存与通信底座 | 第 38 章 |

---

## 本章的验证边界（务必先读）

写作环境是 macOS (Apple Silicon)、Python 3.11，**没有 NVIDIA GPU，也没有昇腾 NPU**
（见 [`../../VERSIONS.md`](../../VERSIONS.md)）。因此本章遵守三条硬规矩：

1. **不给出任何本机实测数字。** 本章出现的全部性能数字都来自组件 README 的公开口径，
   每一条都带来源与测量条件（硬件型号、CANN 版本、形状、口径）。凡是本机跑不出来的
   （延迟、吞吐、算力利用率、带宽），本章只写“README 原文怎么说”，不写“我们测到多少”。
2. **所有示例的最高验证级别是“仅静态核对”。** `code/` 里的脚本只用 Python 标准库，
   在写作机上跑过；涉及 NPU 的步骤一律写成“前提 / 命令 / 期望输出 / 验证级别”。
3. **一切结论绑定一次快照。** 本章引用的是 2026-10-01 抓取的仓库快照，commit 见
   `../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md`。这些仓库都还在
   快速演进（例如 DeepEP-Ascend README 里的 `Ongoing` 一节，见 §8.3），所以**任何
   “这个功能有没有”的判断都必须先说清是哪一次快照**。离开这份快照，本章的结论就
   可能过时。

本章把六个组件的 README 当作一手材料；官方文章的措辞是**发布会口径**，不等于
性能承诺。凡是官方文章与仓库 README 的口径不一致（例如“接近硬件上限”这种概括性
表述 vs. README 里带形状的表格），本章以**带测量条件的 README 表格**为准。

---

## 1. 问题的由来：为什么需要先读一张地图

### 1.1 一个“会用 PyTorch”的人看到的是什么

如果你平时的工作是写 `nn.Module`、跑 `torch.compile`、调 dataloader，那么
2026-09-30 这次开源在你眼里大概长这样：一夜之间，仓库列表里多了一串名字，

```
TileKernels   DeepGEMM-Ascend   DeepEP-Ascend
FlashMLA      DeepSelect       TileLang（昇腾版）
```

它们每一个都有自己的 README、自己的性能表、自己的安装说明。问题是：**我该从哪个
读起？它们之间是什么关系？我要用哪一个？**

这正是本章要解决的问题。后面七章会逐个拆开这些组件（第 33–39 章），而这一章的任务
是先把地图画出来，让每一章都有落脚点。

### 1.2 这次开源为什么叫“全栈”

官方文章的措辞很明确：

> 今天，我们正式开源面向华为昇腾算力平台的基础设施组件，涵盖 TileLang 高级语言编译
> 工具、计算库、分布式通信库。所有组件与此前面向英伟达平台的开源组件一一对应。
> （`../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:9`）

这句话里有三个关键词，正好是本章的三条主线：

* **三层**：高级语言编译工具（TileLang）、计算库（DeepGEMM-Ascend / FlashMLA /
  DeepSelect / TileKernels）、分布式通信库（DeepEP-Ascend）。只开源一个算子库算不上
  “全栈”；这次是从语言到算子再到通信一起给的。
* **昇腾**：全部面向华为昇腾算力平台。这与本教材“昇腾优先”的路线一致（见第 19 章
  §1.2“为什么昇腾线是本教材的第一优先级”）。
* **一一对应**：与 NVIDIA 侧此前开源的组件逐个对应。这是本章 §4 的主题，也是理解
  “已有的知识能不能迁移”的关键。

### 1.3 本章不做的事

作为导览章，本章**不**深入任何单个组件的实现。具体地说：

* 不复述 GEMM / Attention / MoE 的算法。那是第 14–16 章。
  **所有算法细节都在后面的章节里**，本章只回答“谁在哪儿、干什么、怎么和邻居接上”。
  例如 FlashMLA 的 online softmax 为什么要把 rescale 放在那个位置、DeepSelect 的
  阈值筛选为什么有期望复杂度保证，本章都只提一句、指向第 36、37 章。
* 不讲 TileLang 编译器内部怎么把 tile 降成 Ascend C 。那是第 20–25 章。
* 不覆盖 DeepSeek 的其它开源项目（3FS、smallpond、Harness、DeepSpec 等）。它们不在
  这次昇腾发布清单里，见 §10.2。

### 1.4 本章要交给你的是“一张可信的地图”

导览章最容易写成两种失败品：一种是“仓库清单 + 一句吹捧”，读完你还是不知道该看哪；
另一种是“把七章的内容压缩成七段摘要”，读完你以为懂了，实际上每一条都记不住。

本章的目标是第三种：**一张能被检验的地图**。它要达到三条标准：

1. **每个位置都有坐标。** 说“TileKernels 在算子层”不够，要能指出它用 TileLang 写、
   运行期选后端、被哪一步调用（§3.2、§5.1）。
2. **每条对应都有反例。** 说“与 NVIDIA 一一对应”不够，要能指出至少三处不对应，
   并给出出处（§4.3）。
3. **每个结论都能自己复核。** 说“某组件支持昇腾 950”不够，要给出一条任何人拿着
   README 和 commit 就能重复的核验流程（§8）。

后面七章是这张地图上的一次次“实地勘察”；读完它们，你应该能回头把这张地图重新画
一遍，并且画得更细。

---

## 2. 官方文章的原始表述

本章的第一手来源是官方发布文章的正文转录。它的价值在于**给出了 DeepSeek 自己的定位
措辞**，后面各章会展开实现细节，但“官方说这个东西是干什么用的”只有这里有。

### 2.1 六个组件的一句话定位

文章用一句话把五个计算/通信组件排开，加上 TileLang 一共六项：

> 本次开源还包括昇腾平台核心计算与通信组件，为不同使用场景提供可复用的基础能力：
> DeepGEMM 加速通用矩阵运算；DeepEP 提供高效的大规模跨设备通信；TileKernels 提供
> 数据处理所需的常规向量计算和访存算子；FlashMLA 提供稀疏注意力算子，提升长上下文
> 处理效率；DeepSelect 则实现了高效的数据筛选。在多项关键测试用例中，上述组件的
> 计算与通信性能已接近硬件上限。
> （`../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:12`）

把它拆成表，就是本章的全景清单：

| 组件 | 官方文章的一句话 | 对应仓库 |
| --- | --- | --- |
| TileLang（昇腾版） | 高级语言编译工具 | `tile-ai/tilelang` |
| DeepGEMM-Ascend | 加速通用矩阵运算 | `deepseek-ai/DeepGEMM-Ascend` |
| DeepEP-Ascend | 高效的大规模跨设备通信 | `deepseek-ai/DeepEP-Ascend` |
| TileKernels | 常规向量计算和访存算子 | `deepseek-ai/TileKernels` |
| FlashMLA | 稀疏注意力算子 | `deepseek-ai/FlashMLA` |
| DeepSelect | 高效的数据筛选 | `deepseek-ai/DeepSelect` |

（仓库地址见文章末尾的开源项目链接，
`../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:16-27`；
快照 commit 见 `PROVENANCE.md`。）

注意“**常规**向量计算和访存算子”这个定语：TileKernels 的定位不是去抢 GEMM 与
Attention 的极限性能，而是把训练/推理里那些“不写 kernel 就慢、写了又不值得写”的
琐碎算子（路由、量化、归一化、Engram、mHC）凑齐。极限性能是 DeepGEMM-Ascend、
FlashMLA、DeepSelect 的活。

### 2.2 TileLang 的措辞：封装、高级、不损失性能

文章对 TileLang 昇腾版的定义是本章最需要逐字读的一句：

> 本次开源的 TileLang 昇腾版本，对昇腾 Ascend C 底层指令进行封装、提供高级语言的
> 编程方式、同时不损失硬件性能。这正是 TileLang 的设计初衷。
> （`../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:11`）

这句话把“为什么需要 TileLang”拆成三个互相牵制的目标，缺一个都不成立：

1. **封装 Ascend C 底层指令**，写算子的人不该直接去管理学过的那些东西：
   `TPipe` / `TQue` 的建队、L1→L0 的 `LoadData`、`SetFlag`/`WaitFlag` 的配对。
   （这些在 Ascend C 里要手写，对照见第 19 章 §6.3，那一章的代码列表里有 `code/06-ascendc-baseline/`。）
2. **提供高级语言的编程方式**，用 tile 级的数据流描述算法，而不是用搬运指令拼装。
   这正是本书第 1–5 章讲的 `T.copy` / `T.gemm` / `T.Pipelined` 那套语言。
3. **不损失硬件性能**，封装不能以性能为代价。这一点是全文最容易被误读的地方：
   官方说的是**设计目标**，不是“所有算子都达到峰值”。真正能核验的性能数字在
   各仓库 README 的表格里（见 §8）。

### 2.3 “每一个 TileLang 算子，昇腾上都有对应实现”

紧接着的一句是本次发布最有分量的承诺：

> 目前，DeepSeek 训练中用到的每一个 TileLang 算子，在昇腾上都有对应的高性能实现。
> （`../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:11`）

读这句话要注意两点，否则很容易过度推论：

* 它说的是“**训练中用到的、且本来就用 TileLang 写的**算子”。它不是“PyTorch 里所有
  算子都有昇腾实现”，也不是“DeepSeek 全部算子都用 TileLang 写”。
* 它说的是“有对应的**高性能实现**”，没说是“同一份代码”。事实上后面 §4.3 会看到，
  昇腾侧有相当一部分是用 Ascend C / PTO 手写的，TileLang 覆盖的是其中一部分。

把这两点合起来，就是本教材第五部分要检验的问题：**TileLang 覆盖到哪里、Ascend C
覆盖到哪里、边界为什么在那里**。

### 2.4 昇腾 950 的 128 卡超节点

文章里“超节点”只出现了一次：

> 在面向昇腾平台的研发过程中，华为团队给予了毫无保留的大力支持。双方紧密合作，
> 共同推进基于昇腾 950 的 128 卡超节点方案、共同对计算与通信进行了深度优化。
> （`../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:13`）

这一句话信息量很大，但**没有给出硬件型号与拓扑细节**。所以本章对它只做三件事：
说清“超节点”指什么（§7.1）、说明为什么它把通信库推成主角（§7.2）、以及指出
README 里能核到的实测部署形态（§7.3）。**不要**从这一句里推出具体的互联带宽数字。

### 2.5 官方为什么要做这件事

文章还有一段常被忽略、但决定了整个第五部分基调的话：

> 工欲善其事，必先利其器。建立新一代自主可控的 GPU 软件生态，首要的事情是先建立一个
> 通用的、编程简单的、同时能达到硬件性能上限的高级语言。TileLang 在这样的历史背景下
> 诞生。一方面，相对于英伟达的 CUDA 语言，TileLang 的编程更简单，能显著提高开发效率、
> 简化代码逻辑。另一方面，相对于其它同类高级语言，TileLang 的编程模型可以充分发挥
> 芯片的特性，达到硬件的性能上限。TileLang 路线首先在英伟达成熟平台上得到了验证，
> 如今已承载了 DeepSeek V4 系列模型训练中大部分算子的实现，是我们探索 AGI 新范式、
> 开发高性能算子的核心工具。
> （`../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:10`）

这段话给了三个可以检验的信息：

1. **优先级是“先有语言，再有生态”。** 官方把语言放在第一位，算子库与通信库放在后面，
   这与 §3 的分层图是一致的。
2. **TileLang 先在 NVIDIA 上验证，再上昇腾。** “TileLang 路线首先在英伟达成熟平台上
   得到了验证”。这就是“一一对应”能成立的原因：算法与编程模型先在一条硬件线上跑通，
   再移植到另一条。
3. **它已经在生产里用了。** “已承载了 DeepSeek V4 系列模型训练中大部分算子的实现”
   说明 TileKernels / DeepGEMM 这些库不是 demo。当然，这句话本身是发布口径，不能当作
   性能证据；能核验的数字仍然在 README 里（§6.2、§8）。

### 2.6 三条主线，选一条开始读

把 §2.1–§2.5 收一收。官方文章的表述可以压成三句话，正好对应本部分的三种读法：

| 主线 | 官方原文 | 对应章节 |
| --- | --- | --- |
| **语言** | “对昇腾 Ascend C 底层指令进行封装、提供高级语言的编程方式” | 第 33–35 章的 TileLang 用法、第 25 章的编译期 |
| **组件** | “DeepGEMM 加速通用矩阵运算；DeepEP 提供高效的大规模跨设备通信；TileKernels 提供……常规向量计算和访存算子；FlashMLA 提供稀疏注意力算子；DeepSelect 则实现了高效的数据筛选” | 第 33–38 章，一章一个组件 |
| **对应** | “所有组件与此前面向英伟达平台的开源组件一一对应” | 本章 §4，以及每一章的“NV 侧对照”小节 |

---

## 3. 组件地图：六件东西怎么拼在一起

### 3.1 先看一张分层图

```text
[fig-01] DeepSeek 昇腾全栈：组件的分层位置
```

这张图（`figures/01-stack-map.md`，渲染见本章图 1）按“离硬件多远”分四层：

1. **语言与编译层**：TileLang（昇腾版）。它把 tile 程序降到 Ascend C / PTO。
2. **运行时层**：DeepJIT。它负责运行时编译、缓存、装载与启动。
3. **计算与算子层**：DeepGEMM-Ascend、FlashMLA、DeepSelect、TileKernels。
4. **通信层**：DeepEP-Ascend。

### 3.2 每个组件是谁、干什么、用什么写

| 组件 | 是什么 | 算子用什么写 | 一句话职责 |
| --- | --- | --- | --- |
| TileLang（昇腾版） | 面向 tile 的 DSL 与编译器 | 它是语言本身 | 把 tile 程序编译成昇腾可执行的 kernel |
| DeepJIT | 头文件式 C++20 JIT 运行时 | C++ | 运行时编译 / 缓存 / 装载 / 启动 kernel |
| TileKernels | 生产算子库 | **TileLang** | MoE 路由、量化、RoPE、Rand、Engram、mHC |
| DeepGEMM-Ascend | BLAS 级计算库 | **Ascend C 为主 + 少量 TileLang** | GEMM（BF16/FP8/FP4）、MQA logits、MegaMoE、mHC prenorm |
| FlashMLA | 注意力算子库 | Ascend C | DSA 稀疏 prefill / decode，密集 MHA，融合 norm-RoPE-attn-RoPE-cast |
| DeepSelect | TopK 算子 | Ascend C / CUDA | DSA 与 sampler 用的数据筛选 |
| DeepEP-Ascend | 专家并行通信库 | Ascend C（走 HCCL/HCOMM/UBMEM/URMA） | EP dispatch / combine，PP / Bucket / Engram |

“用什么写”这一列是本章和后面几章的主线，**每一章都要回答“这里为什么用/不用
TileLang”**。先给结论：

* **TileKernels 是纯 TileLang 的**：同一份算法配 CUDA 与 Ascend 两个后端。
  官方 README 写得很清楚：“kernels now ship a second backend that is selected
  automatically at runtime, so the same Python APIs run on both NVIDIA GPUs and
  Huawei NPUs”（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:9`）。
  具体实现见第 33、34 章。
* **DeepGEMM-Ascend 是 Ascend C 为主、TileLang 补位**：它自称对昇腾 MAD（矩阵乘加）
  原语做了一层轻量抽象，“hiding much of the complexity associated with fractal
  layouts, alignment constraints, address calculations, and verbose low-level
  parameters”（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:5`）。
  而它唯一的 TileLang 部件写在依赖里：“`tilelang`, used by the HC prenorm kernel
  (declared as a package dependency)”
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:28`）。
  为什么 mHC prenorm 用 TileLang、GEMM 主体不用，是第 35 章的核心问题。
* **FlashMLA / DeepSelect / DeepEP-Ascend 没有用 TileLang**（至少在这份快照里没有）：
  它们用 Ascend C / CUDA / PTX 手写。它们为什么不用、以及在什么条件下值得用 TileLang
  重写，分别见第 36、37、38 章。

### 3.3 谁依赖谁：三条边

组件之间只有三条“硬”依赖边，记住它们就不会迷路：

1. **TileKernels → TileLang**：算子库是 TileLang 的第一个大用户。它同时提供 CUDA 与
   Ascend 两个后端，运行期自动选择（`tile_kernels/config.py:24-31` 用
   `os.path.exists('/dev/davinci_manager')` 判断是不是昇腾）。
2. **DeepGEMM-Ascend → DeepJIT +（仅 HC prenorm）TileLang**：GEMM 主体的 kernel 由
   DeepJIT 在运行时编译；mHC prenorm 这一支用 `from tilelang.ascend import language as T`
   写（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py:7`）。
3. **DeepEP-Ascend → DeepJIT**：通信 kernel 也是运行时编译的，README 明确说
   “Ascend C kernels use HCCL/HCOMM, UBMEM and URMA, and are compiled at runtime
   through DeepJIT”
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:5`）。

**DeepJIT 是这条链上唯一的共享件**。它同时支持 CUDA 与昇腾两个后端，用同一套
compile/load/launch 流程（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepJIT/README.md:11`），
所以“NVIDIA 与昇腾共用一套运行时”这件事在 DeepJIT 这一层是成立的。

值得对照的一点：TileLang 自己也有一层 JIT（第 26 章讲的 `tilelang.jit` 与 kernel cache）。
DeepJIT 与它**不是同一个东西**，一个是给手写 Ascend C/CUDA kernel 用的通用 JIT，
一个是 TileLang 编译管线的一部分。第 38 章会把两者放在一起比较。

### 3.4 它们怎么拼成一条流水线

把上面三层串起来，就是一条完整的“训练/推理流水线”：

```text
TileLang（语言）
   └── 编译出 → TileKernels /（部分）DeepGEMM-Ascend 的算子
                        │
   训练：路由 → 通信 → GEMM → 激活 → GEMM → 通信 → 超连接 → 优化器
   推理：索引打分 → TopK 筛选 → 稀疏注意力 → 通信（EP）→ 输出
                        │
              DeepJIT 负责所有这些 kernel 的运行时编译与装载
```

这张骨架会在 §5、§6 各展开一次（训练一次、推理一次），并在第 39 章重新拼回一条
端到端的流水线。

### 3.5 “TileLang 昇腾版本”具体指什么

这里必须先澄清一个容易误读的地方，否则后面七章都会跟着跑偏。

官方文章给的链接是主干仓库 `https://github.com/tile-ai/tilelang`
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:16-17`），
**不是**一个单独的适配器仓库。而在锁定版本里，昇腾侧的代码确实存在两条路线：

| 路线 | 在哪 | 形态 | 前端 |
| --- | --- | --- | --- |
| 主干昇腾后端 | `_vendor/tilelang-v0.1.15/tilelang/ascend/` | 随 release tag 一起发布的昇腾方言 | `import tilelang.ascend.language as T` |
| 适配器仓库 | `_vendor/tilelang-ascend-v0.1.2.000/` | 独立仓库 `tile-ai/tilelang-ascend` | `import tilelang.language as T` |

主干里的这一支是一个完整的后端目录，包含 `language/`、`codegen.py`、
`execution_backend.py`、`kernel_cache.py`、`analysis/`、`pipeline.py`、`target.py` 等
（`ls _vendor/tilelang-v0.1.15/tilelang/ascend/`）；它的自述说的是“Ascend 950 Backend”，
默认 target 为 `dav-3510`（`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:1-30`）。
适配器仓库则是另一种形态的昇腾实现，走 Ascend C 与 PTO 两条 codegen 路线。

**本章不展开这两条路线的取舍**，第 19 章 §7 给了完整对照表与选择规则，第 25 章讲
各自的 codegen 与硬件特性。本章只要求你记住：**“TileLang 昇腾版本”不是单一的一个
包，看到 `T.alloc_l1` 与 `T.alloc_L1` 这种大小写差异时先回去确认手里是哪棵树。**

### 3.6 六个仓库的速览卡片

下面是每章的“出发页”。看一张卡片，就知道进那一章之前该准备什么。

**① TileLang（语言与编译）** — 不是一次发布里的新仓库，而是这次发布的地基。
主干仓库已经有了昇腾后端（`_vendor/tilelang-v0.1.15/tilelang/ascend/`），另有适配器
仓库 `tile-ai/tilelang-ascend`。读它之前先读本书第 1–4 章；想改它读第 20–25、29 章。

**② TileKernels（算子库，纯 TileLang）** — 仓库里每个算子都是
`*_kernel.py`（共享）+ `*_cuda.py` + `*_asc.py` 三件套，运行期选后端
（例如 `tile_kernels/moe/topk_gate_kernel.py:35-42`）。覆盖 MoE 路由、量化、RoPE、
Rand、Engram、mHC。第 33、34 章的正文全部基于这个仓库。

**③ DeepGEMM-Ascend（BLAS，Ascend C 为主）** — 对外声称“fully API-compatible with
DeepGEMM”，覆盖 BF16/FP8/FP4 GEMM、MQA logits、MegaMoE；唯一的 TileLang 部件是
mHC prenorm（`deep_gemm/tilelang_ops/hc_prenorm.py`）。第 35 章会逐段拆它的 GEMM
实现与那条 TileLang 支线。

**④ FlashMLA（注意力）** — 同一个仓库同时服务 NV 与昇腾；昇腾侧这一版只放出了稀疏
prefill 与 decode，融合算子仍是 CUDA-only。它附带一篇昇腾实现的技术报告
（`FlashMLA/docs/20260930-ascend-prefill-deep-dive-zh.md`），是第 36 章的主要依据。

**⑤ DeepSelect（TopK）** — 支持 CUDA 与昇腾，对外是一份 Python API
（`deep_select/interface.py:10-23` 用 `/dev/davinci_manager` 选后端）；仓库里还带一篇
算法与实现分析（`docs/DeepSelect-deep-dive.zh.md`）。第 37 章的重点是“算法决定上限、
硬件决定常数”。

**⑥ DeepEP-Ascend（通信）** — 提供 `EPBuffer` / `PPBuffer` / `EngramBuffer` /
`BucketBuffer` 四类 buffer，公开 API 与 NV 版 DeepEP 对齐；实现在 Ascend C 里走
HCCL/HCOMM、UBMEM、URMA。第 38 章讲它的 EP 流水与重叠。

**⑦ DeepJIT（共用运行时）** — 严格说它不在这六个仓库的清单里，但它是 DeepGEMM-Ascend
与 DeepEP-Ascend 共同的依赖，所以卡片里单列一条。第 26 章讲过 TileLang 自己的 JIT，
第 38 章会把两者放在一起对照。

---

## 4. 与 NVIDIA 侧的一一对应

### 4.1 对照表

“一一对应”是官方文章的组织方式，也是这次发布最实用的卖点。把两侧摆在一起：

```text
[fig-02] NVIDIA 侧与昇腾侧的一一对应
```

表里的每一行都可以在仓库 README 里核到出处：

| 能力 | NVIDIA 侧 | 昇腾侧 | 对应方式与出处 |
| --- | --- | --- | --- |
| GEMM / MQA logits / MegaMoE | DeepGEMM | DeepGEMM-Ascend | “fully API-compatible with DeepGEMM”；同包名，装了就能沿用同一套 API 与开发流程（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:3,9`） |
| 专家并行通信 | DeepEP | DeepEP-Ascend | “Its public buffer APIs are aligned with the NVIDIA version of DeepEP”，训练/推理/prefill/decode 共用同一个 `EPBuffer`（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:3,5,62`） |
| 注意力 | FlashMLA（NV 侧在同一仓库） | FlashMLA（昇腾侧在同一仓库） | 同一个仓库、同一套 Python API，平台在构建期自动探测（`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:7,101`） |
| 算子库 | TileKernels（CUDA 后端） | TileKernels（Ascend 后端） | “the same Python APIs run on both NVIDIA GPUs and Huawei NPUs”，运行期自动选后端（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:9`） |
| TopK 筛选 | DeepSelect（CUDA） | DeepSelect（Ascend） | 同一套 API，`deep_select.topk(...)` 两个平台通用（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:3,85`） |
| 语言/编译 | TileLang 主干 | TileLang 昇腾后端（+ 适配器仓库） | 见 §3.5；主干 `tilelang/ascend/` 与 `tile-ai/tilelang-ascend` 两条路线 |
| 运行时 JIT | DeepJIT（CUDA 后端） | DeepJIT（Ascend 后端） | 同一个 header-only 运行时，`Runtime<CUDA>` 与 `Runtime<Ascend>` 共用 compile/load/launch 流程（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepJIT/README.md:3,11`） |

### 4.2 “一一对应”在工程上意味着什么

“对应”不等于“同一份代码”。它的实际含义是三层可迁移性，从强到弱：

1. **API 可迁移（最强）**：DeepGEMM → DeepGEMM-Ascend 是“同一个包名、同一套 API”，
   DeepEP → DeepEP-Ascend 是“公开 buffer API 对齐”。你在 NV 侧写的调用代码，
   换个设备基本不用改。TileKernels 与 DeepSelect 更进一步：**一份 Python API，运行期
   按设备选后端**（判断逻辑在
   `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/config.py:24-31`，
   用 `/dev/davinci_manager` 是否存在来决定返回 `'npu'` 还是 `'cuda'`）。
2. **心智模型可迁移（中等）**：你在第 14–16 章学的 GEMM / Attention / MoE 算法不变，
   变的只是硬件约束。例如同样一个 grouped GEMM，NV 侧的瓶颈可能是 tensor core 占用，
   昇腾侧则要先看 L0C 容量与 FixPipe（见第 35 章）。
3. **实现技巧不可迁移（最弱）**：具体的流水排布、地址计算、同步插入完全不同。
   “稀疏数据加载、基于协程的流水线”这类技巧是昇腾特有的
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:5,11`），
   NV 侧的经验帮不上忙。

所以正确的用法是：**用“对应”来定位该读哪一章、该复用哪段调用代码；但不要用
“对应”去假设实现、性能或边界行为一致。**

### 4.3 哪里并不一一对应

这一节是本章最重要的一节。README 里明写的差异至少有三类：

**(1) 数据格式不同。** DeepGEMM-Ascend 在接口一节专门加了一条 NOTE：

> The scaling factor format on Ascend differs from NVIDIA's: each pair of UE8M0
> scaling factors along the K dimension is packed into an `int16`, and the packed
> values are stored in MN-major order for optimal hardware efficiency.
> （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:55-56`）

对照 NV 侧 DeepGEMM 的口径：SM90 的 scaling factor 是 FP32，SM100 是把 4 个 UE8M0
打包进一个 `torch.int`（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM/README.md:64-67`）。
也就是说，**同样的 FP8 GEMM，三套硬件的 scale factor 布局互不相同**，跨平台搬权重
时必须转换。昇腾侧为此提供了 `deep_gemm.transform_sf_into_required_layout` 等工具函数
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:67-68`）。

**(2) 功能子集不同。** 同一个仓库在昇腾上只实现了部分算子：

* FlashMLA 的融合算子 `fused-norm-rope-attn-rope-cast` 明确写着
  “This kernel currently supports CUDA only; Ascend is not supported.”
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:43`）；
  昇腾侧这一版给的是稀疏 prefill 与 decode 两个算子（同文件 `:13-15`）。
* DeepSelect 的 Ascend 实现只支持 `torch.bfloat16`，`torch.float32` 只有 CUDA 实现
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:37`），
  且 `indices_type` 只支持 `torch.int32`（同文件 `:90`）。
* DeepEP-Ascend 把 PP / Engram / Bucket 标为 experimental，reduce-scatter /
  all-reduce 与专家负载均衡的通信 kernel 还没实现（同文件 `:76-84`）。

**(3) API 相同不等于语义相同。** DeepEP-Ascend 自己的 README 提醒：“supported modes
and stream behavior are specific to Ascend”（`:62`），并且 EP 模式有额外约束，
`do_expand=True`、`allow_multiple_reduction=True`、`do_handle_copy` 必须保持 `False`
（`:72`）。同样是 `EPBuffer`，NV 侧能跑的模式昇腾侧不一定能跑。

> **一句话总结**：一一对应是“接口层”的对应，不是“行为层”的等价。读后面每一章时，
> 都要问一句：这里的差异是格式差异、子集差异，还是语义差异？

### 4.4 回到主线：三种“为什么不用 TileLang”

有了 §4.3 的三类差异，就能理解后面各章要反复回答的那个问题。把六个组件按 TileLang
的使用程度排开，正好是三种典型情况：

| 情况 | 例子 | 本部分哪一章讲 |
| --- | --- | --- |
| **完全用 TileLang** | TileKernels 的全部算子 | 第 33、34 章 |
| **Ascend C 为主，TileLang 补位** | DeepGEMM-Ascend（GEMM 用 Ascend C，mHC prenorm 用 TileLang） | 第 35 章 |
| **完全不用 TileLang** | FlashMLA、DeepSelect、DeepEP-Ascend | 第 36、37、38 章 |

这不是“谁先进谁落后”的排序，而是**算子形态决定的**：TileKernels 里的算子大多是
“一份算法、两个后端”都能表达的数据处理，TileLang 的 tile 抽象刚好合适；DeepGEMM
要把 GEMM 压到 99% 硬件利用率，需要直接控制 MAD 原语、稀疏加载与协程流水，落到了
TileLang 抽象之下；FlashMLA/DeepSelect 的瓶颈在索引与流水编排上，不是 tile 计算。
第 33–38 章会一章一章把这条判断讲实。

### 4.5 动手看一眼“同一个 API，两个后端”

抽象讨论不如看一段真代码。TileKernels 的 `topk_gate` 是 §5.1 里 ① 那一步的实现，
它的结构可以当作“一份算法、两个后端”的模板：

```python
# tile_kernels/moe/topk_gate_kernel.py:7-43（节选）
def topk_gate(scores: torch.Tensor, num_topk: int) -> torch.Tensor:
    ...
    topk_idx = torch.empty((num_tokens, num_topk), dtype=torch.int64, device=scores.device)
    if num_tokens == 0:
        return topk_idx

    if is_ascend():
        from tile_kernels.moe.topk_gate_asc import get_topk_gate_kernel_asc

        kernel = get_topk_gate_kernel_asc(num_experts, num_topk, get_num_vec_cores())
    else:
        kernel = get_topk_gate_kernel_cuda(num_experts, num_topk, use_pdl=get_pdl())

    kernel(scores, topk_idx)
    return topk_idx
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/moe/topk_gate_kernel.py:7-43`。）

三个文件的分工是固定的：

| 文件 | 职责 | 是否依赖硬件 |
| --- | --- | --- |
| `topk_gate_kernel.py` | 参数校验、形状推导、**选后端**、启动 | 否（纯 Python；`is_ascend()` 靠探测设备文件） |
| `topk_gate_cuda.py` | CUDA 后端的 TileLang kernel | 是（`T.Kernel(..., threads=32)`、`T.alloc_fragment`、`T.reduce_max`、`T.reducer_*`） |
| `topk_gate_asc.py` | Ascend 后端的 TileLang kernel | 是（用 `get_num_vec_cores()` 这类昇腾概念） |

（CUDA 后端全文见
`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/moe/topk_gate_cuda.py:1-68`。）

这就是 §4.2 说的“API 可迁移”在代码里的样子：**调用方只看见 `topk_gate(scores, k)`，
看不见设备**。而 §4.3 说的“实现不可迁移”也在这里：两个后端文件里的写法是两份，
不是一份。第 33 章会把这两个文件逐行对照着读。那是本部分第一个“动手”章节。

---

## 5. 一次训练 step 里，各组件在哪

### 5.1 从 MoE 路由到优化器

```text
[fig-03] 一次训练 step 里各组件的位置
```

把一次 MoE 层的训练 step 展开，大致是这七步（对照 `figures/03-train-step.md`）：

| 步骤 | 做什么 | 组件 | 本部分哪一章 |
| --- | --- | --- | --- |
| ① MoE 路由 | 对每个 token 算专家分数、选 top-k | TileKernels（`topk_gate` 等） | 第 33 章 |
| ② EP dispatch | 把 token 按专家分布发到不同卡 | DeepEP-Ascend（`EPBuffer.dispatch`） | 第 38 章 |
| ③ Grouped GEMM（升维） | 每个专家做 `up` 投影 | DeepGEMM-Ascend（`m_grouped_*` / MegaMoE） | 第 35 章 |
| ④ SwiGLU | 激活与门控 | TileKernels（融合量化） | 第 33 章 |
| ⑤ Grouped GEMM（降维） | 每个专家做 `down` 投影 | DeepGEMM-Ascend | 第 35 章 |
| ⑥ combine | 把专家输出按权重收回原卡 | DeepEP-Ascend（`EPBuffer.combine`） | 第 38 章 |
| ⑦ mHC + 优化器 | 流形超连接、梯度归约、参数更新 | TileKernels（mHC）、DeepGEMM-Ascend（HC prenorm） | 第 34、35 章 |

这条链在 README 里有一句**几乎逐字的对应**：MegaMoE“fuses EP dispatch, two grouped
GEMMs, SwiGLU, and combine”（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:153`）。
也就是说，③④⑤ 加上 ②⑥ 被融合成了一个“mega kernel”。这正好说明为什么这些组件
必须一起看：它们在同一条链上，边界可以移动。

### 5.2 训练侧的重点是“重叠”

训练 step 里最贵的两件事是**专家 GEMM**和**跨卡通信**。理想情况是两者重叠：通信在
后面跑，计算在前面跑。DeepEP-Ascend 的用法小节就是在教这件事：

```python
pending = dispatch_forward(buffer, x, topk_idx, topk_weights, num_experts, ...)

# Overlap application computation with communication.
shared_output = shared_experts(x)
recv_x, _, recv_weights, handle = pending.wait()
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:175-193`；
完整示例见同文件 `:122-172`。）先发起通信，再算不相干的部分，最后 `.wait()` 取结果。

训练侧还有两个**没有出现在这次发布清单里、但会被用到**的组件：

* **DualPipe**（`deepseek-ai/DualPipe`）：双向流水并行，目标是“full overlap of forward
  and backward computation-communication phases”并减少流水气泡
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DualPipe/README.md:3`）。
* **EPLB**（`deepseek-ai/EPLB`）：专家并行负载均衡，用“redundant experts”策略复制
  热点专家再打包到各卡（`../tilelang-corpus/14-deepseek-ascend-stack/raw/EPLB/README.md:3-8`）。

它们不在 2026-09-30 的昇腾清单里，但 DeepEP-Ascend 为 EPLB 预留了对齐的接口，
`EPBuffer.lb_prefetch_weights` 与 `lb_reduce_grads`“expose the aligned APIs, but their
Ascend communication kernels are not implemented yet”
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:81`）。
所以第 38 章会把它们放在“位置”里讲，而不是当成这次开源的一部分。

### 5.3 训练里 TileLang 覆盖了多少

文章说“训练中用到的每一个 TileLang 算子，在昇腾上都有对应的高性能实现”
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:11`）。
把它落到 §5.1 的表格上，TileLang 直接覆盖的是 ①④⑦ 的一部分（TileKernels 那三块），
而 ②③⑤⑥ 是 Ascend C 手写的。

这不是缺陷，而是分工：**TileLang 保证“常用但零碎”的算子能快速写、快速改；Ascend C
保证“少但关键”的算子能压到硬件极限。** 两条路线的边界放在哪里、什么时候该跨过它，
是本部分最重要的一条工程结论。

### 5.4 一次训练 step 的调用骨架

把 §5.1 的七步写成调用骨架，可以看清“哪一步是谁的活”。下面是**伪代码**（不是可运行
程序，也不对应某个仓库里的文件），每一行后面标了来源章节：

```python
# —— ① MoE 路由（TileKernels，第 33 章）
topk_idx = tile_kernels.moe.topk_gate(scores, num_topk)

# —— ② EP dispatch（DeepEP-Ascend，第 38 章）；异步发起，先不算结果
pending = buffer.dispatch(
    x, topk_idx=topk_idx, topk_weights=topk_weights, num_experts=num_experts,
    do_expand=True, defer_epilogue=True, async_with_compute_stream=True,
)

# —— 与通信重叠：算共享专家（第 38 章 §Overlap）
shared_output = shared_experts(x)

# —— ③ 取回 dispatch 结果，做两个 grouped GEMM（DeepGEMM-Ascend，第 35 章）
recv_x, _, recv_weights, handle = pending.wait()
up = deep_gemm.m_grouped_bf16_gemm_nt_contiguous(recv_x, l1_weights, ...)  # 升维

# —— ④ SwiGLU（TileKernels，第 33 章）
act = tile_kernels.quant.swiglu_forward(up)

# —— ⑤ 降维 grouped GEMM（DeepGEMM-Ascend）
down = deep_gemm.m_grouped_bf16_gemm_nt_contiguous(act, l2_weights, ...)

# —— ⑥ combine（DeepEP-Ascend）；同样异步发起
pending = buffer.combine(down, handle=handle, bias=shared_output, defer_epilogue=True)

# —— ⑦ mHC 与优化器（TileKernels / DeepGEMM-Ascend，第 34、35 章）
mixed = tile_kernels.mhc.mhc_pre_apply_mix_fwd(mix, down, ...)
optimizer.step()
```

对照 §5.1 的表格可以看到两件事：

* **通信是异步的、可重叠的**：② 与 ⑥ 都是“发起 → 干别的 → `wait()`”的形式。这正是
  §5.2 说的训练侧重点。
* **TileLang 覆盖的是 ①④⑦ 的一部分**，而 ②⑤⑥ 这类“要压到硬件极限”的算子交给了
  Ascend C。第 33–38 章会把每一行替换成真实的 API。

> **注意**：上面的函数名（`tile_kernels.moe.topk_gate`、`buffer.dispatch`、
> `deep_gemm.m_grouped_bf16_gemm_nt_contiguous`、`tile_kernels.quant.swiglu_forward`、
> `tile_kernels.mhc.mhc_pre_apply_mix_fwd` 等）都能在各仓库里核到
> （`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/__init__.py:1-17`、
> `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/deep_gemm/__init__.py:44-79`），
> 但**参数组合不是**，每一章的正文会给出与实际 README 一致的调用形式。这里只用来理解
> “谁在哪一步”。

---

## 6. 一次推理请求里，各组件在哪

### 6.1 prefill 与 decode 是两条不同的路

推理分两个阶段：**prefill**（一次处理整段 prompt）和 **decode**（每次生成一个 token）。
两者的瓶颈完全不同，用到的算子也不一样。

```text
[fig-04] 一次推理请求（prefill / decode）里各组件的位置
```

DeepSeek 这一代模型的推理走的是 **DSA（DeepSeek Sparse Attention）**：不是让每个 query
关注所有 KV，而是先给 KV 打分（Lightning Indexer），选出 top-k 个“最重要的”KV，
再只对这些 KV 做注意力（见
`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/docs/20260930-ascend-prefill-deep-dive-zh.md:18-24`；
DSA 的首次公开实现是 `deepseek-ai/DeepSeek-V3.2-Exp`，见
`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSeek-V3.2-Exp/README.md:42`）。
于是推理链上多了一段“先筛选、再计算”：

| 步骤 | 做什么 | 组件 | 哪一章 |
| --- | --- | --- | --- |
| ① 索引打分 | 算 q 与 KV 的 MQA logits | DeepGEMM-Ascend（MQA logits kernel） | 第 35 章 |
| ② TopK 筛选 | 从打分里选出 top-k KV / 专家 | DeepSelect | 第 37 章 |
| ③ 稀疏注意力 | 只对选中的 KV 做 prefill / decode | FlashMLA（昇腾稀疏算子） | 第 36 章 |
| ④ 专家并行通信 | MoE 推理的 dispatch / combine | DeepEP-Ascend | 第 38 章 |
| ⑤ 常规算子 | 归一化、RoPE、量化、采样 | TileKernels | 第 33、34 章 |

### 6.2 各组件在推理里的实测位置（附口径）

README 给出的性能表其实是“这张图上的哪一段”的最好注解：

| 组件 | README 里的用例 | 测量条件（原文口径） |
| --- | --- | --- |
| DeepGEMM-Ascend MQA logits | prefill / decode 的 indexer 打分 | Ascend 950DT、CANN 9.20、`bench_msprof`、cold L2；“The kernel is FIX-pipe bound rather than compute bound, saturating the FIX pipe at 99% utilization”（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:93,142`） |
| DeepSelect | Lightning Indexer 场景的 TopK | bfloat16、`topk=512`；指标是“effective memory bandwidth”，因为 TopK 不做浮点运算（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:41-44,48`） |
| FlashMLA | DSA 的稀疏 prefill / decode | Ascend 950；prefill 410 TFlops（95% 硬件极限）、decode 360 TFlops（83% 硬件极限）（`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:13-15,53,63`） |
| DeepEP-Ascend | MoE 推理的 EP dispatch / combine | Ascend 950DT、CANN 9.2.0、PoC HDK；16384 tokens/rank、hidden 7168、top-6 / 256 experts（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:9-11`） |

**注意 FlashMLA 那两句话的落点**：410 TFlops 是“up to”，95% 是相对于 Ascend 950 的
理论峰值，且来自 README，不是本章的实测。第 36 章会给出这个数字是在什么 kernel、
什么形状下取得的。

### 6.3 推理的“读哪个 README”捷径

如果你只想知道“某个算子在昇腾上到底有没有、性能如何”，按这个顺序查：

1. **先看它属于哪一段**（上表 ①–⑤），确定组件；
2. **打开该组件 README 的 `Performance` / `News` 一节**，确认有没有昇腾条目；
3. **再看 `Requirements`**，确认它要求的硬件与 CANN 版本；
4. **最后看 `Ongoing`**（如果有），确认哪些还只是 API 预留。

第 8 节会把这条流程做成一张可执行的清单。

### 6.4 一次 decode step 的调用骨架

同样用伪代码看一眼。下面的函数名都能在 README 里核到，参数被省略：

```python
# 每个 decode step 之前（或第一次调用时）准备一次调度元数据
tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata()

for step in decode_loop:
    # —— ① 索引打分：算 q 与 KV 的 MQA logits（DeepGEMM-Ascend，第 35 章）
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(q, kv_cache, ...)

    # —— ② 选出 top-k KV（DeepSelect，第 37 章）
    _, indices = deep_select.topk(logits, topk, indices_type=torch.int32, return_value=False)

    # —— ③ 稀疏注意力：只对选中的 KV 做 decode（FlashMLA，第 36 章）
    out, lse = flash_mla.flash_mla_with_kvcache(
        q, kvcache, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits,
        indices=indices,
    )

    # —— ④ MoE 推理的专家通信（DeepEP-Ascend，第 38 章）
    #     与 ⑤ 常规算子（TileKernels：norm / RoPE / 量化 / 采样）穿插其间
```

（`tile_scheduler_metadata` 与 `flash_mla_with_kvcache` 的用法见
`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:111-126`；
`indices` 的形状与语义见同文件 `:145-151`；`deep_select.topk` 的签名见
`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:85-92`。）

几个容易踩的点，先在这里提一句，细节留给第 36、37 章：

* **decode 只支持量化 KV cache**：“For decoding, this kernel currently supports only
  the FP8 and FP4 KV cache formats. Unquantized (bfloat16) KV cache format is not
  supported.”（`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:136`）
* **prefill 版不批处理**：`flash_mla_sparse_fwd` 没有 batch 维，多 batch 要自己重排
  （同文件 `:172`）。
* **TopK 的输出可能不是连续的**：`deep_select.topk` 的输出 stride 按
  `get_stride_requirement()` 对齐，可能非连续（同文件 `:97-99`）。

---

## 7. 昇腾 950 的 128 卡超节点：为什么通信是主角

### 7.1 “超节点”指的是什么

第 6 章讲过：现代加速器框图里有四个维度，存储层级、执行单元、数据搬运、同步。把
这四个维度从“一块芯片内部”放大到“一个机柜里的多块芯片”，得到的就是**超节点
（supernode）**：在一个比传统以太网/IB 网络更近的域内，把几十到上百块 NPU 连成
一个整体。

第 19 章讲的 UB / L1 / L0A / L0B / L0C 是**单卡内部**的存储层级；超节点把“共享”
这个层级向外扩了一层，多张卡之间有一块可以被直接访问的共享内存域（昇腾侧叫
**UBMEM**），跨卡访问不必每次都走主机侧网络栈。

官方文章的原话是“基于昇腾 950 的 128 卡超节点方案”
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:13`）。
**文章没有给出拓扑细节**（多少交换机、什么组网、多少跳），所以能引用的只有这句。
能核到的部署形态来自 DeepEP-Ascend 的 README，见 §7.3。

### 7.2 为什么 128 卡会把通信推到台前

先做一个不需要任何实测的算术。MoE 训练里，一次专家并行 dispatch 要搬运的数据量是
（近似）：

$$
\text{通信量} \approx \text{tokens} \times \text{topk} \times \text{hidden} \times b
$$

其中 $b$ 是每个元素的字节数。这个量随**卡数**增长：卡越多，每个 token 越可能被路由到
别的卡上；同时每张卡本地要算的 token 越少，通信/计算的比值就越大。128 卡意味着这个
比值已经高到“通信不是配角”的程度。这正是为什么这次开源里通信库（DeepEP-Ascend）
占了整整一章（第 38 章）。

第 39 章会给出更精确的账：一次 decode step 里通信、计算、访存各自占多少。本章只需要
记住结论：**在超节点规模上，算子的极限性能不再是唯一瓶颈，通信的带宽与重叠程度
同样是第一类指标。**

### 7.3 README 里能核到的部署形态

DeepEP-Ascend 的 README 把条件写得很细，这也是本章 §8“怎么核验”的样板：

* **硬件**：“Ascend 950 (A5) with UBMEM connectivity and UBC_CTP/URMA channels
  between the participating ranks”，且参与方必须是**偶数个 rank**
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:28-29`）。
* **网络口径**：“All measurements use netlayer 1, the supernode's external Clos
  network.”（`:9`），即测的是超节点**对外**的 Clos 网络一层。
* **HDK 状态**：结果表明他们用的是**给 DeepSeek 的 PoC HDK 加手工配置**，不是公开
  发行版；README 明确区分了这一点，并建议等华为的 Q3 商用 HDK（预计 2026-10-15
  前后公开）（`:40-42`）。

这一段值得反复读，因为它示范了一种**诚实的性能表述**：把“我们测的是 PoC 配置”
“公开版本还没有”这两件事分开写。第 8 节会把它变成一份可抄的核验清单。

### 7.4 通信不只是 DeepEP

超节点里跑的通信有三类，DeepEP-Ascend 只覆盖其中一类：

1. **EP all-to-all**（DeepEP-Ascend 的主业）：MoE 的 dispatch / combine。
2. **流水并行 / 数据并行的集合通信**：DeepEP-Ascend 用 `BucketBuffer` 提供
   all-gather（reduce-scatter / all-reduce 还在做，
   `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:80`），
   上层由 DualPipe 这类流水方案组织。
3. **专家负载均衡带来的权重搬运**：EPLB 的复制专家策略需要把权重搬来搬去，
   DeepEP-Ascend 为它预留了 `lb_prefetch_weights` / `lb_reduce_grads` 接口（同上 `:81`）。

这三类通信都建立在同一套底座上：HCCL/HCOMM、UBMEM、URMA（`:288`）。所以第 38 章讲
DeepEP-Ascend 时，会先把这套底座讲清楚，再讲 EP 的流水。

### 7.5 判断“这个方案有多大”的三条线

超节点这类概念最容易被数字带着跑。给三条可操作的判断线，读完第 39 章再回来看：

**第一条线：通信域有多大。** 一张卡内部的通信是寄存器/shared memory 级别；超节点把
这个域扩展到“一次集合通信不需要走主机网络栈”。DeepEP-Ascend 的 `EPBuffer` 支持
EP8–EP128（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:13-19`），
这个范围就是它的通信域下限。

**第二条线：谁和谁通信。** 训练里 MoE 的 all-to-all 是“每个 rank 发给所有可能持有
目标专家的 rank”；流水并行是“只和相邻 rank 通信”；数据/上下文并行是集合通信。
三类通信的带宽需求与拓扑敏感度完全不同，DeepEP-Ascend 为此提供了 `BucketBuffer`
（all-gather）与 `PPBuffer`（相邻 rank 收发）两种不同的接口（`:64-70`）。

**第三条线：通信能不能被藏起来。** 这就是 §5.2 讲的“重叠”。判断一个方案成熟不成熟，
不看峰值带宽，而看**它有没有把通信放到计算后面**。DeepEP-Ascend 的 `.wait()` 模式与
`defer_epilogue` 参数（`:120`）就是为这件事服务的；第 38 章会给出完整的重叠时序。

---

## 8. 如何自己核验

这一节是本章的“操作手册”。后面每一章都会引用它。

### 8.1 第一步：读 `Requirements`，确定硬件与工具链

每个 README 都有一节 `Requirements`（DeepGEMM-Ascend 是 `### Requirements`）。它回答的
问题只有一个：**这个东西要跑起来，最低需要什么？** 把六个仓库的答案摆在一起：

```text
[fig-05] 昇腾侧组件的硬件与版本要求
```

逐条读出来是这样的（全部是 README 原文，不是本章的推断）：

| 组件 | 昇腾硬件 | 工具链 | 出处 |
| --- | --- | --- | --- |
| TileLang（主干昇腾后端） | Ascend 950 | CANN + `bisheng` + CCE `ld.lld` + `torch_npu` | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:14-30` |
| TileKernels | Ascend 950 NPU | CANN 9.2.0 或更高 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:29-31` |
| DeepGEMM-Ascend | Ascend 950 系列 | CANN 9.20（`bin/bisheng`、`bin/ld.lld`）+ `torch_npu` | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:23-26` |
| FlashMLA（昇腾侧） | Ascend 950 NPU | CANN 9.2.0 或更高 + `torch_npu` | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:83-88` |
| DeepSelect（昇腾侧） | Ascend NPU | 见仓库 README 的 `Installation` | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:64-71` |
| DeepEP-Ascend | Ascend 950 (A5) + UBMEM 连接 + UBC_CTP/URMA 通道 | CANN（含 Ascend C、Bisheng、HCCL、HCOMM）+ `torch_npu` | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:25-34` |
| DeepJIT | 随使用方（CUDA 或昇腾） | CUDA：NVCC；昇腾：Bisheng + `ld.lld` + ACL + `torch_npu` | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepJIT/README.md:25-28` |

三个读 Requirements 的要点：

1. **硬件是“950”级别，不是所有昇腾**。多个 README 都写 Ascend 950 / 950DT。
   DeepEP-Ascend 甚至明说“Kernel support on other Ascend generations or CANN versions
   has not been established by these measurements”（`:34`）。**不要**假设 A2/A3 也能跑。
2. **CANN 版本要分别记**：TileKernels 与 FlashMLA 写 9.2.0，DeepGEMM-Ascend 写 9.20。
   这两个写法指的是同一代工具链，但**以各 README 为准**，不要互相套用。
3. **工具链要在运行时可用**。DeepJIT 与 DeepEP-Ascend 都强调 kernel 是**首次使用时
   编译**的，所以 `bisheng` 与 CANN 头文件必须一直可用，不是装完就能删
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:36`）。

### 8.2 第二步：读 `News`，判断“什么时候发布的”

`News` 给的是时间线。按时间读一遍，你能看出每个组件的成熟度：

| 日期 | 事件 | 出处 |
| --- | --- | --- |
| 2026-09-30 | TileKernels 加华为昇腾支持，运行时自动选后端 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:9` |
| 2026-09-30 | DeepGEMM-Ascend 首次发布，支持 Ascend 950 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:17` |
| 2026-09-29 | DeepEP-Ascend 首次公开发布 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md`（快照 HEAD 摘要见 `PROVENANCE.md`） |
| 2026-09-30 | FlashMLA 发布昇腾 950 的稀疏注意力 prefill/decode | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:13` |
| 2026-09-30 | DeepSelect 发布昇腾 TopK | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:9` |
| 2026-09-10 | DeepSelect 发布算法与实现简析、v1.0.0 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:10-11` |
| 2026-09-30 | DeepGEMM（NV 侧）在 News 里指向 DeepGEMM-Ascend | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM/README.md:14-16` |

从这张时间线能读出两件事：

* **时间集中**：六个仓库的昇腾版本几乎都在 2026-09-29/30 落地。这说明它们是一次
  协调发布，而不是各自独立演进出来的。
* **成熟度不齐**：DeepSelect 有 9-10 日的算法报告与 v1.0.0，FlashMLA 有 2025 年就
  在 NV 侧验证过的稀疏 kernel；而 DeepEP-Ascend 是 9-29 的首次公开版本。**同一个
  “已发布”标签下，成熟度可以差很多**。这直接影响你能不能在生产里用。

### 8.3 第三种状态：仍在开发 / 未实现

README 里最有价值、也最容易漏掉的是 **`Ongoing`** 一节。DeepEP-Ascend 是最好的例子：

> The Ascend backend is under active development:
>
> - Bucket collectives: batched all-gather is available; Ascend reduce-scatter and
>   all-reduce kernels are still being built.
> - Expert load balancing: `EPBuffer.lb_prefetch_weights` and `EPBuffer.lb_reduce_grads`
>   expose the aligned APIs, but their Ascend communication kernels are not
>   implemented yet.
> - PP and Engram: these interfaces remain experimental, with further correctness
>   and performance validation ongoing.
>
> Hybrid communication, CPU-backed Engram storage and graph capture remain unsupported.
> （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:76-84`）

这段话里出现了**三种不同的状态**，一定要分清：

| 状态 | 判据 | 例子 | 你该怎么做 |
| --- | --- | --- | --- |
| **已发布** | README 有性能表；接口在 API 列表里且标了支持 | EP dispatch / combine | 可以直接用，但注意 `Requirements` |
| **实验性（experimental）** | 接口存在，但被显式标为 experimental / 需要额外验证 | PP、Engram、Bucket | 可以试，但不要假设正确性与性能已定 |
| **未实现（not implemented yet）** | 接口存在只是“API 预留”，或根本不在表里 | `lb_*` 的 Ascend kernel、reduce-scatter / all-reduce、CPU-backed Engram、graph capture | **不要用**；用之前先回来看这一节有没有更新 |

第 38 章会逐条核对这些状态的现状。这里先把判据记住：**“有 API”不等于“有实现”，
“有实现”不等于“已验证”。**

### 8.4 为什么要绑定 commit，而不是绑定 main

上面所有结论都来自 2026-10-01 的一次快照，commit 记在
`../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md` 的表里（13 个仓库，
每个都有 commit 与 commit 日期）。例如：

* `DeepEP-Ascend` `3b25377d04b2`（2026-09-29，Initial public release）
* `FlashMLA` `2e5429fc5653`（2026-09-30，Open-source release for Ascend 950）
* `TileKernels` `66258df6175d`（2026-09-30，Merge pull request #34 from deepseek-ai/v2.0.0）
* `DeepSelect` `bfa4507d935f`（2026-09-30，Add Kernels for Huawei Ascend NPU）
* `DeepGEMM-Ascend` `8491bbb4b8c0`（2026-09-30，OSS release）

**为什么不写“main 上已经有 X”？** 因为这些仓库都还在动。同一个 README 里的
`Ongoing` 一节可能下个月就被改掉，性能表也可能换一组形状。绑定 commit 的好处是
**任何一条结论都可以被复现**：拿到同一个 commit，README 里的每个数字、每一句状态
描述都冻结不变。

本教材的通用规则（见 [`../../VERSIONS.md`](../../VERSIONS.md)）是：**所有结论必须
绑定到锁定版本**；只有当某个能力明显只在更新分支上时，才显式写“仅 main / 未进
锁定版本”，并放到第 30 章。本章对第五部分的所有仓库都用这条规则。

### 8.5 一份可抄的核验清单

把 §8.1–§8.4 合成一个流程。想确认“组件 X 能不能在昇腾 Y 上用、状态如何”时，按
顺序回答这六个问题：

1. **它属于哪一段？**（用 §5、§6 的表格定位，确定是哪个仓库。）
2. **`Requirements` 里的硬件是 950 吗？CANN 版本是多少？**（§8.1）
3. **`News` 里最后一次昇腾相关更新是什么时候？**（§8.2）
4. **`Ongoing` / 性能表备注里有“experimental”或“not implemented”吗？**（§8.3）
5. **我拿到的 commit 是这个快照吗？**如果不是，回 `PROVENANCE.md` 对照差异。（§8.4）
6. **README 给的性能数字，测量条件是什么？**（硬件、CANN、形状、口径，缺一不可。）
   本章 §6.2 的表就是范例。

第 39 章会把这份清单升级成一张“MFU / 带宽检查表”，用于判断一条真实流水线跑没跑满。

### 8.6 本部分的验证边界

最后必须把边界说清楚，因为后面七章都会引用它：

* **本机没有昇腾 NPU，也没有 NVIDIA GPU。** 所以本部分所有涉及加速器的示例都是
  **仅静态核对**：代码能通过语法/API 核对，但**没有任何一条来自本机的性能数字**。
* **所有性能数字都带测量条件**。本部分不会出现“昇腾上 TileKernels 比 X 快 N 倍”
  这种没有出处的比较；只会出现“README 在什么条件下测到了什么”。
* **“未运行”也是一条结论**。如果后面某章说“这个 kernel 我们没能运行”，那意味着
  它按要求写了前提与命令，但没有真机验证。这本身是读者必须知道的信息。

### 8.7 走一遍清单：以 DeepSelect 为例

拿 §8.5 的六个问题，对 DeepSelect 走一遍，看看一次核验长什么样。

**Q1 它属于哪一段？** DeepSelect 是 §6.1 的 ②（TopK 筛选），服务于 DSA 的索引器与
sampler。它不在训练主链上，但推理链绕不开。

**Q2 硬件与工具链？** README 的 `News` 明确说 2026-09-30 发布了昇腾 TopK
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:9`）；支持场景一节
说明输入 dtype 是 `torch.bfloat16`、`topk` 主要优化到 512
（同文件 `:19-23`）；安装一节给了编译方式（`:64-71`）。硬件细节以仓库 README 为准，
**这份快照里没有像其它仓库那样标出“Ascend 950 / CANN 版本”**。这是一个真实的
资料缺口，第 37 章会再核对一遍。

**Q3 最近一次昇腾更新？** 2026-09-30（`News` 第一条）。

**Q4 有没有 experimental / not implemented？** 这个仓库没有 `Ongoing` 一节，平台差异
写在 API 文档字符串里
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/deep_select/interface.py:52-68`）：

* `sorted` 只支持 fp32，且“Ascend only supports False”（`:54`）；
* `indices_type`“Ascend only supports torch.int32”（`:59`）；
* NaN 处理的回退行为在两个后端不同（`:66-68`）；
* `begin` / `hint` 参数“CURRENTLY NOT SUPPORTED”（`:55,61`）。

**Q5 commit 是这个快照吗？** 快照是 `bfa4507d935f`（2026-09-30，“Add Kernels for Huawei
Ascend NPU”），见 `PROVENANCE.md`。

**Q6 性能数字的测量条件？** README 说指标是“effective memory bandwidth”，因为 TopK
不做浮点运算；对比基准是同一输入下的 `torch.topk`，通过 `tests/test.py --perf-only`
得到（`:41-44`）。官方给的总体结论是“2 ~ 20x speedup compared to vanilla `torch.topk`”
（`:3`）。**注意**：这是 README 的自述区间，图里的分档取决于 batch size 与 vocab size，
`code/README.md` 里说的“仅静态核对”同样适用于它。

走完这一遍，你会得到一个**可以直接写进结论**的句子：“DeepSelect 在 2026-09-30 的快照上
支持昇腾的 bfloat16 TopK，但昇腾实现不支持 fp32 输入、不支持排序输出、indices 只支持
int32；性能口径是相对 torch.topk 的有效带宽，出处在 README。”。这正是本章希望你在
读后面每一章时都能产出的那种句子。

---

## 9. 本部分的学习路线

第 32–39 章是一条从“导览”到“端到端”的链，依赖关系是弱依赖（可以跳读，但按顺序
读最省力）：

| 章 | 主题 | 读它之前最好先读 | 一句话 |
| --- | --- | --- | --- |
| **32** | 本章：开源了什么 | — | 地图、对照表、核验方法 |
| 33 | TileKernels（上）：一份算法，两个后端 | 32、9、11、25 | 用 top-k gate 为例逐行对照 CUDA / Ascend 两种 TileLang 写法 |
| 34 | TileKernels（下）：Engram 与 mHC | 33 | “不是矩阵乘”的算子怎么写 TileLang |
| 35 | DeepGEMM-Ascend：把 GEMM 压到硬件极限 | 14、25、27 | MAD 原语、稀疏加载、协程流水、scale factor 打包 |
| 36 | FlashMLA 昇腾稀疏注意力：算法与流水 | 15、23、25 | DSA、online softmax、1C2V、ND2NZ |
| 37 | DeepSelect：算法先行的 TopK | 13、28 | 阈值筛选 + RadixSelect；算法决定上限，硬件决定常数 |
| 38 | DeepEP-Ascend：专家并行通信与计算重叠 | 16、26、35 | EP dispatch/combine、HCCL/UBMEM/URMA、DeepJIT |
| 39 | 端到端：128 卡超节点上的训练与推理 | 32–38 | 把六件东西拼回一条流水线，算一次 decode step 的账 |

依赖关系可以用一张小图记住（箭头表示“先读前面，再读后面”）：

```text
            32（本章：地图与核验方法）
             │
   ┌─────────┴─────────┐
   ▼                   ▼
 33 TileKernels（上）  35 DeepGEMM-Ascend
   │                   │
   ▼                   ├──────────┐
 34 TileKernels（下）   │          │
                       ▼          ▼
                  36 FlashMLA  37 DeepSelect
                       └────┬─────┘
                            ▼
                    38 DeepEP-Ascend
                            │
                            ▼
              39 端到端：128 卡上的训练与推理
```

这张图只是“推荐顺序”，不是硬依赖：第 35–38 章互相之间只有弱依赖，可以按兴趣跳读；
但第 34 章依赖第 33 章的 `*_kernel.py` / `*_cuda.py` / `*_asc.py` 三件套写法，
第 38 章依赖第 35 章对 MegaMoE 的理解。

三条可选路线：

* **只想看懂“这次开源了什么”**：读本章 + 第 39 章，中间的章节当字典查。
* **想学怎么用 TileLang 写算子**：本章 → 第 33、34 章 → 回到第 9–11 章补语言基础。
* **想学怎么把算子压到硬件极限**：本章 → 第 35、36、37 章 → 第 28 章的性能模型。

---

## 10. 常见坑（本章特有）

### 10.1 六个坑

* **坑 1：把“一一对应”当成“同一份代码”。** 最典型的后果是直接搬 scale factor 的
  打包格式。昇腾把**一对** UE8M0 打包进一个 `int16` 且按 MN-major 存
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:55-56`），
  NV 侧 SM100 是**四个** UE8M0 打包进一个 `torch.int`
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM/README.md:64-67`）。
  两者不能互换。**先读 §4.3。**
* **坑 2：拿 NV 侧的数字套昇腾。** DeepGEMM NV 侧有 1550 TFLOPS（H800）的旧数字
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM/README.md:25`），
  昇腾侧是 99.8%（BF16，4096×7168×16384，Ascend 950DT/CANN 9.20）
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:93,104`）。
  **两个数字的分母、硬件、形状都不同，不能放在一起比。**
* **坑 3：把“有接口”当成“有实现”。** DeepEP-Ascend 的 `lb_prefetch_weights` 有
  对齐的 API，但通信 kernel 还没实现（`:81`）。写代码前先看 `Ongoing`。
* **坑 4：把“实验性”当成“稳定”。** PP / Engram / Bucket 被标为 experimental
  （`:72,82`），且“Hybrid communication, CPU-backed Engram storage and graph capture
  remain unsupported”（`:84`）。
* **坑 5：版本对不上。** 六个仓库都要求 950 级别的硬件，CANN 版本要求分散在 9.2.0 与
  9.20 之间。**以各自 README 为准**，不要用一个组件的环境去套另一个。
* **坑 6：忘记 kernel 是运行时编译的。** DeepJIT 与 DeepEP-Ascend 的 kernel 在首次
  使用时编译（`:36`），所以运行环境里必须一直有 CANN / Bisheng；只留一个 `.so` 或
  wheel 是不够的。缓存目录与调试开关见 DeepJIT 的 `Environment variables` 一节
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepJIT/README.md:336`；共享缓存见
  同文件 `:34-45`）与 DeepEP-Ascend 的 `EP_JIT_*`（`:238-250`）。

### 10.2 明确不覆盖的范围

本部分只覆盖 2026-09-30 这次发布清单里的组件，加上理解它们所必需的配套仓库
（DeepJIT、Engram、DeepSeek-V3.2-Exp、DualPipe、EPLB、clangd-ascend）。**以下
DeepSeek 的其他开源项目不在范围内**，本书不展开：

* 3FS（分布式文件系统）
* smallpond（数据处理）
* Harness（评测/训练框架）
* DeepSpec（投机解码相关）

它们与本次昇腾发布没有直接关系。如果你在别处看到它们，请不要以为本书漏写。
同理，本部分也不展开 AMD ROCm / Metal / WebGPU / CPU 后端，只在第 5、6 章作为抽象
对照出现。

另外提醒一处**未核实**的地方：FlashMLA 的 README 提到它支撑 NVIDIA GPU 与昇腾 NPU 上
某个 DeepSeek V4.1 系列模型（Flash 版）的推理，并引用了 HuggingFace 上的模型页
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:7`）。本仓库在
写作环境下无法访问 HuggingFace，因此该模型页**未能核实**：本书只把它当作 README 的
自述转述，不给出可点击链接，也不据此推断模型细节。

### 10.3 读后面每一章前，先做这三件事

第五部分的其余七章会各讲一个组件。进每一章之前，建议固定做三件事，成本很低，
但能避免大多数误读：

1. **打开那一章的组件 README，找到它的 `Requirements` 与 `Ongoing`。** 先知道“要什么
   硬件、缺什么功能”，再读实现细节。缺功能的地方往往是那一章的重点（例如第 35 章的
   mHC prenorm 支线、第 38 章的 `Ongoing`）。
2. **在 `PROVENANCE.md` 里确认 commit。** 如果那一章引用的快照与手头的代码不是同一次，
   行号就会漂移；这时以“代码里实际的函数名”为准，而不是以行号为准。
3. **先问一遍“这里为什么用/不用 TileLang”。** 这是本部分每章的主线问题（§4.4）。
   带着这个问题读，你会发现同一个技术点的解释会自然分层：算法层用 TileLang 表达，
   性能层落回 Ascend C。

---

## 本章小结

一页纸复习：

1. **这次开源是一套全栈**：语言/编译（TileLang 昇腾版）、计算库（DeepGEMM-Ascend、
   FlashMLA、DeepSelect、TileKernels）、通信库（DeepEP-Ascend），外加共用的运行时
   DeepJIT。官方文章的清单见 §2.1。
2. **官方对 TileLang 的定位**是“对昇腾 Ascend C 底层指令进行封装、提供高级语言的
   编程方式、同时不损失硬件性能”，并且“训练中用到的每一个 TileLang 算子，在昇腾上
   都有对应的高性能实现”（§2.2、§2.3）。
3. **组件地图**：TileLang 在最底层（编译），DeepJIT 做运行时，TileKernels / DeepGEMM-Ascend
   / FlashMLA / DeepSelect 是计算与算子层，DeepEP-Ascend 是通信层（§3）。
4. **“TileLang 昇腾版”不是一个包**：主干 `tilelang/ascend/` 与独立适配器仓库
   `tile-ai/tilelang-ascend` 两条路线并存，见 §3.5 与第 19、25 章。
5. **一一对应是接口层的**：API 与开发流程可迁移（DeepGEMM ↔ DeepGEMM-Ascend、
   DeepEP ↔ DeepEP-Ascend、TileKernels/DeepSelect 同一 API 双后端、FlashMLA 同仓库
   双平台），但格式（scale factor 打包）、子集（CUDA-only 的融合 kernel）、语义
   （stream 行为）都可能不同（§4）。
6. **训练 step 的位置**：路由（TileKernels）→ dispatch（DeepEP-Ascend）→ grouped GEMM
   （DeepGEMM-Ascend）→ SwiGLU（TileKernels）→ grouped GEMM → combine（DeepEP-Ascend）
   → mHC + 优化器（TileKernels / DeepGEMM-Ascend）。TileLang 直接覆盖的是路由、激活、
   超连接这些“常用但零碎”的算子（§5）。
7. **推理的位置**：MQA logits（DeepGEMM-Ascend）→ TopK 筛选（DeepSelect）→ 稀疏注意力
   （FlashMLA）→ 专家通信（DeepEP-Ascend），常规算子（TileKernels）穿插其间（§6）。
8. **超节点把通信推成主角**：128 卡的规模让通信/计算的比值显著上升，所以 DeepEP-Ascend
   单独成章；它建立在 HCCL/HCOMM、UBMEM、URMA 上，实测用的是超节点对外的 Clos 网络
   （§7）。
9. **核验三件事**：读 `Requirements`（硬件 + CANN）、读 `News`（发布时间与成熟度）、
   读 `Ongoing`（区分已发布 / 实验性 / 未实现）。所有结论绑定 `PROVENANCE.md` 里的
   commit，不绑 main（§8）。
10. **验证边界**：本机没有 NPU / GPU，本部分全部示例只做静态核对；所有性能数字都带
    测量条件，来自 README（§8.6）。

---

## 延伸阅读

* **[R1] 官方发布文章**：`../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md`
  本次开源的清单与官方定位措辞（§2）。
* **[R2] 快照来源**：`../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md`
  13 个仓库的 upstream、commit 与日期（§8.4）。
* **[R3] TileKernels**：`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md`
  “一份算法、两个后端”的原话与 `Requirements`（§3.2、§8.1）。正文见第 33、34 章。
* **[R4] DeepGEMM-Ascend**：`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md`
  API 兼容声明、MAD 抽象、scale factor 的 NOTE、性能表。正文见第 35 章。
* **[R5] DeepEP-Ascend**：`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md`
  EP/PP/Bucket/Engram 接口、带宽表、`Ongoing` 与 `Requirements`（§7、§8.3）。
* **[R6] FlashMLA**：`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md`
  与其 `docs/20260930-ascend-prefill-deep-dive-zh.md` ，昇腾稀疏 prefill/decode 的
  性能口径与技术报告（§6.2）。正文见第 36 章。
* **[R7] DeepSelect**：`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md`
  TopK 的支持场景、性能口径与平台差异（§4.3、§6.2）。正文见第 37 章。
* **[R8] DeepJIT**：`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepJIT/README.md`
  双后端 JIT 运行时、缓存与环境变量（§3.3、§10.1）。
* **[R9] NVIDIA 侧对照物**：`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM/README.md`
  与 `.../DeepEP/README.md` ，讲“一一对应”时的另一半（§4）。
* **[R10] 配套仓库**：`.../Engram/README.md`、`.../DeepSeek-V3.2-Exp/README.md`、
  `.../DualPipe/README.md`、`.../EPLB/README.md`、`.../clangd-ascend/README.md`
  Engram 论文与参考实现、DSA 的首次公开实现、流水并行与负载均衡、昇腾开发工具链。

完整的引用清单（含行号与核验状态）见 [`references.md`](references.md)；
每张图的源描述见 [`figures/`](figures/)；示例代码与验证级别见 [`code/README.md`](code/README.md)。
