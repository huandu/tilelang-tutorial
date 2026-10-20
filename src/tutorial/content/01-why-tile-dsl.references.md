# 第 01 章 参考资料

本章所有结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现「最新版」这类表述。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章所有代码示例的最高验证级别是「仅静态核对」（API 存在性与签名逐个在 `_vendor/`
中比对过），没有任何一条性能数字来自本机实测。

---

## 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:431-455` | `T.Kernel` 的定义与 docstring（「a grid of tile programs」「target-neutral launch」） | 已核验（2026-10-01）：`grep -n "^def Kernel" tilelang/language/kernel.py` → 431；docstring 逐行阅读 |
| S2 | `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:49` | `T.alloc_shared(shape, dtype, scope="shared.dyn")` 签名与默认 scope | 已核验（2026-10-01） |
| S3 | `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:81` | `T.alloc_fragment(shape, dtype, scope="local.fragment")` 签名 | 已核验（2026-10-01） |
| S4 | `_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:59` | `T.copy` 签名与「tile op 不直接展成 CUDA」的说明 | 已核验（2026-10-01） |
| S5 | `_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:160` | `T.gemm(A, B, C, transpose_A, transpose_B, policy, clear_accum, annotations)` 签名 | 已核验（2026-10-01） |
| S6 | `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:13`、`:82`、`:112` | `T.Parallel`、`T.Persistent`、`T.Pipelined(num_stages=...)` 签名 | 已核验（2026-10-01） |
| S7 | `_vendor/tilelang-v0.1.15/examples/quickstart.py` | 第 3 章正文引用的 tile 风格 GEMM + ReLU 示例 | 已核验（2026-10-01）：文件存在，代码逐行比对；**未运行**（无 GPU） |
| S8 | `_vendor/tilelang-v0.1.15/examples/gemm/example_gemm.py` | 习题 E6/E13 引用的最小 GEMM 示例 | 已核验（2026-10-01）；未运行 |
| S9 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:104-196` | NV 侧 pass 顺序中 `PipelinePlanning → InjectSoftwarePipeline → LayoutInference → LowerTileOp` 的位置 | 已核验（2026-10-01） |
| S10 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:1-152` | 昇腾 950 后端定位、Quick Start（GEMM+ReLU）、与 Ascend C 的分工对照表 | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S11 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:26,37,46,55` | `T.alloc_l1` / `alloc_l0a` / `alloc_l0b` / `alloc_l0c` 的 scope 默认值 | 已核验（2026-10-01） |
| S12 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py:53-100` | `T.dual_copy` 的语义（L0C→UB 硬件事务、M/N 拆分、对齐约束） | 已核验（2026-10-01） |
| S13 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py:107,179` | `T.SimtVF(threads=128, latency=0)`、`T.SimdVF(latency=0)` | 已核验（2026-10-01） |
| S14 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:108-114` | 昇腾侧的 `EstimateLatency → AutoSchedule → PrepareMultiBuffer → InsertSync → MaterializeMultiBuffer` | 已核验（2026-10-01） |
| S15 | `_vendor/tilelang-v0.1.15/tilelang/language/eager/builder.py:1177` | `T.const(name, dtype)` | 已核验（2026-10-01） |
| S16 | `_vendor/tilelang-v0.1.15/tilelang/language/proxy.py:256` | `T.Tensor`（TensorProxy） | 已核验（2026-10-01） |

「已核验」的含义：在锁定版本的工作树上用 `grep -n` 定位到定义行，并阅读了完整的签名与
docstring；行号区间与文件内容一致。**这不是运行验证。**

---

## corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577.md` | https://arxiv.org/abs/2504.17577 | TileLang 论文摘要：调度空间与数据流解耦、三层接口、composable | 已核验（2026-10-01）：本地文件为摘要全文，链接可达 |
| R2 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/overview.md` | https://tilelang.com/get_started/overview.html | tile 的官方定义、三层接口（含 beginner 层未完成的说明）、显式存储分配 | 已核验（2026-10-01） |
| R3 | `../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md` | https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf ；DOI 10.1145/3315508.3329973 | Triton 原始论文全文：tile 的定义、"statically shaped multi-dimensional sub-arrays"、三类编译器路线与 Triton 的取舍 | 已核验（2026-10-01）：本地为逐页文本提取 |
| R4 | `../tilelang-corpus/11-related-compilers/raw/halide-docs.md` | https://halide-lang.org/ | Halide 的「算法 / 调度分离」原则 | 已核验（2026-10-01） |
| R5 | `../tilelang-corpus/04-language-and-ir/raw/arxiv-1802-04799.md` | https://arxiv.org/abs/1802.04799 | TVM（OSDI 2018）：Halide 路线在深度学习编译器中的延续 | 已核验（2026-10-01）：本地为摘要 |
| R6 | `../tilelang-corpus/04-language-and-ir/raw/arxiv-2207-04296.md` | https://arxiv.org/abs/2207.04296 | TensorIR（ASPLOS 2023）：block/axis IR，TileLang 的 IR 基础 | 已核验（2026-10-01）：本地为摘要 |
| R7 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md` | https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | H100：HBM3「over 3 TB/sec」、FP16 张量核 preliminary specs、TMA、thread block cluster、DSM、「first truly asynchronous GPU」、256KB shared+L1 / 最多 228KB shared | 已核验（2026-10-01）：本地为全文；厂商公开资料，非实测 |
| R8 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md` | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html | 「hundreds of clock cycles of memory latency」、算术指令延迟约 4 cycles（cc 7.0） | 已核验（2026-10-01）：本地为全文；引用处为第 3629、3145 行附近 |
| R9 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-c-programming-guide.md` | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html | CUDA 的 grid/block/warp/thread 模型与存储层级 | 已核验（2026-10-01）：本地为节选 |
| R10 | `../tilelang-corpus/09-zhihu-da-zhi-ruo-yu/raw/zhihu-dazhiruoyu-deepseek-ascend.md` | https://www.zhihu.com/question/2088578683623814216/answer/2088624284100999121 | 昇腾 950 与 NV 的存储/执行/同步对照、Cube/AIC 与 Vector/AIV、Z3/SMT 调度、latency 与 issue interval 的区分 | **部分核验**：本地为登录态抓取的全文；链接需登录才能访问；内容属**社区一手文章，非厂商官方文档**，其中的性能说法（如「性能与 Ascend C 打得有来有回」「代码行数差好几倍」）无公开基线与测量口径，本章只作为定性描述引用，未当作数字 |
| R11 | `../tilelang-corpus/06-huawei-ascend/raw/github-huawei-csl-pto-dsl.md` | https://github.com/huawei-csl/pto-dsl | PTO-DSL 自述：对标 cuTile、暴露 L0/L1/UB、与 tilelang-ascend 的关系 | 已核验（2026-10-01） |
| R12 | `../tilelang-corpus/06-huawei-ascend/raw/pto-spec/README.md` | https://github.com/PTO-ISA/pto-spec | PTO 的形式化定义：Direct Tile operations 118 条、64 个 Tile 寄存器、execution mask | 已核验（2026-10-01）：本地为仓库工作树快照，版本为 `0.58.7`（开发分支），**不是发布版**；本章只引用其结构性描述 |
| R13 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2607-22432.md` | https://arxiv.org/abs/2607.22432 | TileSight：以 tile 为中心的性能模型；MAPE 12.35%（A100/H200/B200/B6000）等 | 已核验（2026-10-01）：本地为摘要；论文自述「will be open-sourced upon publication」，本章未使用其数字 |
| R14 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2205-14135.md` | https://arxiv.org/abs/2205.14135 | FlashAttention：IO-aware tiling 的经典论证（BERT-large 15%、GPT-2 3×、LRA 2.4×） | 已核验（2026-10-01）：本地为摘要；数字来自论文自述，测量条件见其正文，本章未直接引用为 TileLang 的性能 |
| R15 | `../tilelang-corpus/90-synthesis/01-knowledge-system.md` | （本书 corpus 内的综合文档，无独立上游链接） | 知识体系与「TileLang 在软件栈里的位置」的框架 | 已核验（2026-10-01） |

---

## 上游链接核验说明

* 上表中标注「已核验」的链接，核验方式是：确认对应 corpus 文件在本地存在，且其头部记录了
  抓取日期与原始 URL；本章未在写作时重新发起网络请求（corpus 抓取日期为 2026-09-30，
  早于本章写作日期 2026-10-01，差异在 1 天内）。
* `R10`（知乎）需要登录态才能访问，属 corpus 中唯一依赖登录的来源；本章只把其中的
  架构性描述（Cube/AIC 与 Vector/AIV 的分工、存储角色、Z3 的作用）作为定性材料引用，
  **没有引用其中任何未经公开基线的性能结论**。
* `R12`（PTO 规范）引用的是开发分支工作树快照（version `0.58.7`），不是某个已发布版本；
  本章只使用了「tile 级指令集包含 Direct Tile operations 与 Tile 寄存器」这类结构描述。

---

## 关于性能数字的说明

本章出现的每一个数字都带来源与条件，汇总如下：

| 数字 | 来源 | 测量/口径条件 |
| --- | --- | --- |
| HBM3「over 3 TB/sec」、50 MB L2 | R7 | H100 SXM5，厂商公开资料 |
| FP16 张量核密集 1000 TFLOPS | R7 | H100，厂商公开「preliminary performance specs」表，未实测 |
| 机器平衡点 ≈ 333 FLOP/Byte | **本书推导**，输入为 R7 的两个数字 | 峰值/带宽之比，未实测 |
| 未缓存 local/global 访存「数百个时钟周期」 | R8 | CUDA 编程模型，未缓存访存 |
| 多数算术指令延迟约 4 cycles | R8 | compute capability 7.0 设备 |
| H100 每 SM 256 KB shared+L1，shared 可配到 228 KB | R7 | 厂商公开资料，configurable |
| `softmax`/`MatMul` 的算术强度表 | **本书推导**，公式见 1.3 节 | 理论最小访存量，fp16；表由 `code/01-op-cost-model/op_cost_model.py` 复算 |

**没有来源的、或来自厂商宣传口径的数字一律未采用。** 所有需要真实加速器才能得到的数字
（实测算力利用率、延迟、吞吐）在本章中均未出现。

---

## 待补资料

1. **cuTile 的一手资料缺失。** 本章谱系图（图 5）里的 cuTile 节点只有 PTO-DSL 的间接
   提及（R11），corpus 内没有 NVIDIA cuTile 的官方文档或仓库快照。建议后续版本补一条
   `https://docs.nvidia.com/cuda/cutile-python/` 的抓取。本章因此对 cuTile 只做了定位性
   描述，没有展开。
2. **TileLang 论文的正文缺失。** corpus 里的 `arxiv-2504-17577.md` 只有摘要，正文
   （尤其是评测章节的具体数字与图）不在本地。本章因此只引用了摘要中可直接核对的设计
   主张。第 3 章若需要展开论文的评测数据，建议补抓正文 PDF 或 HTML 全量版本。
3. **TileLang 官方文档中「layout inference」的专页缺失。** `_sources/compiler_internals/`
   下只有 4 篇（`letstmt_inline`、`inject_fence_proxy`、`tensor_checks`、
   `metal_tilelang_development`），没有 layout inference 的原理页。本章的 layout 描述来自
   `get_started/overview.md` 与源码，第 21 章若需要更细的推导细节，可能需要补抓。
