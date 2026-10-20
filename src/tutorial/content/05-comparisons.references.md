# 第 05 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现「最新版」这类表述；讨论 `main`/未合入的能力时一律显式标注。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章所有代码示例的最高验证级别是「仅静态核对」（API 存在性与签名逐个在 `_vendor/`
中比对过），没有任何一条性能数字来自本机实测。

引用路径约定：源码相对书籍根目录写 `_vendor/...`；corpus 资料写 `../tilelang-corpus/...`。

---

## 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/examples/quickstart.py:1-46`、`:51-56`、`:38`、`:41-43` | 统一例子（GEMM+ReLU）的 TileLang 版；默认 tile 参数；`T.gemm` 落到 cute/hip 的注释；融合 ReLU | 已核验（2026-10-01）：`grep -n`/`sed -n` 定位逐行比对；**未运行**（无 GPU） |
| S2 | `_vendor/tilelang-v0.1.15/README.md:113-132` | Platform/Backend 支持表：CUDA=Primary、Ascend 950=Supported、CuTe DSL=Experimental、A2/A3=Ecosystem | 已核验（2026-10-01）：表格逐行阅读 |
| S3 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:74-124`、`:141-151` | 昇腾 GEMM+ReLU quickstart（含 `:124`「single kernel, without a separate launch for the epilogue」）；「Compared with writing Ascend C directly」对照表（用户写什么 / 可跳过什么） | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S4 | `_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:160` | `T.gemm(...)` 定义行 | 已核验（2026-10-01）：`grep -n "^def gemm"` → 160 |
| S5 | `_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:59` | `T.copy(...)` 定义行 | 已核验（2026-10-01） |
| S6 | `_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:431` | `T.Kernel(...)` 定义行 | 已核验（2026-10-01） |
| S7 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:26,37,46,55` | `alloc_l1` / `alloc_l0a` / `alloc_l0b` / `alloc_l0c` 的 scope 默认值 | 已核验（2026-10-01） |
| S8 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py:53` | `T.dual_copy` 定义行 | 已核验（2026-10-01） |
| S9 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py:107,179` | `T.SimtVF(threads=128, latency=0)`、`T.SimdVF(latency=0)` | 已核验（2026-10-01） |
| S10 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:129-142,258-259` | NVIDIA 后端 pass 顺序：`PipelinePlanning→InjectSoftwarePipeline→LayoutInference→LowerTileOp`；`ThreadSync("shared"/"shared.dyn")` | 已核验（2026-10-01）：逐行阅读 |
| S11 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:108-114` | 昇腾侧 `EstimateLatency→AutoSchedule→PrepareMultiBuffer→InsertSync→MaterializeMultiBuffer` | 已核验（2026-10-01） |
| S12 | `_vendor/tilelang-v0.1.15/tilelang/ascend/transform/z3_scheduler.py:1-26,67` | 昇腾自动调度使用 Z3（SMT）；文件头注明 Z3 非线程安全 | 已核验（2026-10-01）：文件头与函数定义阅读 |
| S13 | `_vendor/tilelang-v0.1.15/tilelang/backend/README.md:1-27` | backend 抽象：target backend = compiler vertical slice；execution backend 可复用 | 已核验（2026-10-01） |
| S14 | `_vendor/tilelang-v0.1.15/tilelang/cuda/cutedsl_backend.py:1-20` | `cutedsl` 后端注册（复用 CUDA pipeline） | 已核验（2026-10-01） |
| S15 | `_vendor/tilelang-v0.1.15/tilelang/contrib/cutedsl/gemm_tcgen05.py:1-13` | CuTe DSL 侧的 tcgen05（Blackwell）MMA 原语清单 | 已核验（2026-10-01） |
| S16 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:58-98` | 昇腾版示例使用 `T.Persistent`（`:88`）、`T.dual_copy`（`:94`）、`T.SimtVF`+`T.Parallel`（`:95-97`） | 已核验（2026-10-01） |

「已核验」的含义：在锁定版本工作树上用 `grep -n` / `sed -n` 定位到定义行，并阅读了完整
签名与 docstring；行号区间与文件内容一致。**这不是运行验证。**

---

## corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577.md` | https://arxiv.org/abs/2504.17577 | TileLang 论文摘要：tile 数据流、调度空间与数据流解耦、composable | 已核验（2026-10-01）：本地为摘要全文 |
| R2 | `../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md`（`:37` tile 定义；`:130-200` Listing 1；`:583-720` 机器相关 pass；`:943-946` 未来工作含 tensor cores） | https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf ；DOI 10.1145/3315508.3329973 | Triton 原始论文全文（PDF 同目录 `triton-mapl-2019.pdf`） | 已核验（2026-10-01）：本地为逐页文本提取，逐行阅读 |
| R3 | `../tilelang-corpus/11-related-compilers/raw/github-triton-lang-triton.md` | https://github.com/triton-lang/triton | Triton 当前实现仓库自述；确认 MAPL2019 为其奠基论文 | 已核验（2026-10-01） |
| R4 | `../tilelang-corpus/11-related-compilers/raw/triton-docs.md` | https://triton-lang.org/main/index.html | 当前 Triton 文档索引：`triton.language`、Gluon 层 | 已核验（2026-10-01）：本地为索引页 |
| R5 | `../tilelang-corpus/05-nvidia-backend/raw/github-nvidia-cutlass.md`（`:19-51` 概览与 DSL；`:28-37` 支持的数据类型与架构） | https://github.com/NVIDIA/cutlass | CUTLASS 4.x 定位、CuTe DSL 自述（低层、与 CuTe C++ 一致、公测） | 已核验（2026-10-01）：本地为 README 快照 |
| R6 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cutlass-media-docs-cute.md:10-12` | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/00_quickstart.html | CuTe 核心抽象：hierarchically multidimensional layouts、Layout/Tensor | 已核验（2026-10-01）：本地为 quickstart 页 |
| R7 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-c-programming-guide.md` | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html | CUDA 编程模型总览（本地为节选，用于层次定位） | 已核验（2026-10-01）；本地为节选，**未用于任何具体数字** |
| R8 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md:1745-1830` | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html | 共享内存分块矩阵乘示例（`__shared__ float aTile[TILE_DIM][TILE_DIM]`） | 已核验（2026-10-01）：本地为全文；引用处为示例段 |
| R9 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md:38645-38659` | https://docs.nvidia.com/cuda/parallel-thread-execution/ | PTX 语言码表含 `8 | "python-cutile"`（PTX ISA 9.3 引入） | 已核验（2026-10-01） |
| R10 | `../tilelang-corpus/11-related-compilers/raw/tvm-docs.md` | https://tvm.apache.org/docs/ | TVM 文档索引；TIRx（Tensor Layout / Tile Primitives）小节 | 已核验（2026-10-01）：本地为索引页 |
| R11 | `../tilelang-corpus/04-language-and-ir/raw/arxiv-1802-04799.md` | https://arxiv.org/abs/1802.04799 | TVM（OSDI 2018）摘要：图级+算子级优化、自动代价模型 | 已核验（2026-10-01）：本地为摘要 |
| R12 | `../tilelang-corpus/04-language-and-ir/raw/arxiv-2207-04296.md` | https://arxiv.org/abs/2207.04296 | TensorIR（ASPLOS 2023）摘要：张量为 first-class 的 block/axis IR | 已核验（2026-10-01）：本地为摘要 |
| R13 | `../tilelang-corpus/11-related-compilers/raw/mlir-docs.md` | https://mlir.llvm.org/docs/ | MLIR dialect/pass 基础设施（昇腾 AscendNPU IR 路线的共同语言） | 已核验（2026-10-01）：本地为索引页 |
| R14 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/developer/zh_developer_techArticles_20230830-1.md`（`:29` 定位；`:51-57` 编程范式与队列；`:350-440` 代码） | https://www.hiascend.com/developer/techArticles | Ascend C 官方技术文章：CopyIn/Compute/CopyOut、`TPipe`/`TQue`/`EnQue`/`DeQue`、LeakyReLU 样例 | 已核验（2026-10-01）：本地为官方文章镜像 |
| R15 | `../tilelang-corpus/06-huawei-ascend/raw/huawei-hiascend-cann.md` | https://www.hiascend.com/cann | CANN 定位（Ascend C 属 CANN 软件栈） | **部分核验**：本地为站点抓取，正文含较多导航噪声；本章仅用于「Ascend C 属 CANN」这一定位，并改用 R14 支撑细节 |
| R16 | `../tilelang-corpus/06-huawei-ascend/raw/gitcode-cann-catlass.md` | https://gitcode.com/cann/catlass | CATLASS：昇腾算子模板库（GEMM 及融合类算子模板），提供仓库入口与 v2.1.0 概述 | 已核验（2026-10-01）：本地为仓库页快照；**未取得具体 kernel 源码** |
| R17 | `../tilelang-corpus/06-huawei-ascend/raw/pto-dsl/README.md`（`:4-7` 类比 cuTile；`:9-12` 自动软件流水；`:46-52` 框架对比，含 `:49` 与 tilelang 的关系） | https://github.com/huawei-csl/pto-dsl | PTO-DSL 定位、与 tilelang-ascend 的关系、目标对标硬件 intrinsic | 已核验（2026-10-01）：本地为仓库 README 快照 |
| R18 | `../tilelang-corpus/06-huawei-ascend/raw/pto-dsl/examples/aot/matmul_optimization_guide/matmul_optim_guide_zh.md` | 同仓库文件 | 昇腾 matmul 逐步优化指南：persistent kernel 行为、GM→L1→L0→Cube→L0C→GM 路径、自动同步 | 已核验（2026-10-01）：本地为仓库快照 |
| R19 | `../tilelang-corpus/06-huawei-ascend/raw/pto-dsl/examples/aot/matmul_optimization_guide/step1_baseline.py:27-125` | 同仓库文件 | PTO-DSL matmul 基线实现（`pto.load`/`tile.extract`/`tile.matmul(_acc)`/`pto.store`） | 已核验（2026-10-01）：本地为仓库快照；**未在 NPU 上运行** |
| R20 | `../tilelang-corpus/06-huawei-ascend/raw/pto-spec/README.md:15-40` | https://github.com/PTO-ISA/pto-spec | PTO ISA 形式规范概览：64 个 T/U/M/N Tile 寄存器、118 条 Direct Tile operation、execution mask | 已核验（2026-10-01）：本地为工作树快照，架构标识 `0.58.7`（开发分支，非发布版） |
| R21 | `../tilelang-corpus/06-huawei-ascend/raw/PTO-Gym/README.md` | https://github.com/PTO-ISA/pto-gym | PTO 教学资源：`kernel.pto`/`launch.cpp`/`golden.py` 结构与 Tile 指令 SPEC | 已核验（2026-10-01）：本地为仓库快照 |
| R22 | `../tilelang-corpus/09-zhihu-da-zhi-ruo-yu/raw/zhihu-dazhiruoyu-deepseek-ascend.md` | https://www.zhihu.com/question/2088578683623814216/answer/2088624284100999121 | 昇腾 950 后端的中文一手叙述：NV/昇腾逐项对照、为什么同步最难、Z3 调度、latency 与 issue interval | **部分核验**：本地为登录态抓取的全文；链接需登录；属**社区一手文章，非厂商官方文档**；其性能说法（「打得有来有回」「代码行数差好几倍」）无公开基线与测量口径，本章只作定性引用 |
| R23 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/README.md` | https://github.com/tile-ai/tilelang-ascend | TileLang-Ascend 适配器定位：两条技术路线（Ascend C & PTO / AscendNPU IR） | 已核验（2026-10-01）：本地为仓库 README 快照 |
| R24 | `../tilelang-corpus/08-frontier-unmerged/raw/github-issue-1454.md` | https://github.com/tile-ai/tilelang/issues/1454 | CuteDSL Backend roadmap；把 CuTe DSL 称为 NVIDIA tile DSL 侧最有力的竞争者 | 已核验（2026-10-01）：本地为 issue 快照 |
| R25 | `../tilelang-corpus/08-frontier-unmerged/raw/github-issue-3317.md` | https://github.com/tile-ai/tilelang/issues/3317 | 用户提问「Ascend 950 的两个后端该用哪个」 | **快照不完整**：抓到的正文只有提问，**没有维护者回答**；本章据此不给出「官方推荐哪条路线」的结论 |
| R26 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3310.md` | https://github.com/tile-ai/tilelang/pull/3310 | 给 Ascend950 增加 PTO 后端：注册 `pto` target、生成 PTO 设备码、`ptoas`+Bisheng 编译 | 已核验（2026-10-01）：本地为 PR 快照（bot 生成的摘要） |
| R27 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/TileFoundry/examples/minicpm3_4b-cutedsl/kernels/cute_gemv.py:29-113` | https://github.com/tile-ai/TileFoundry | 真实 CuTe DSL kernel（GEMV + RMSNorm）：`@cute.jit`、`tidx`、`cute.arch.barrier`、`cute.autovec_copy` | 已核验（2026-10-01）：本地为仓库快照；**未运行** |

---

## 上游链接核验状态

* `https://docs.nvidia.com/cuda/cutile-python/`：**未核验**。本 corpus 只有 PTO-DSL README
  指向它的链接，没有正文快照；未联网核验其内容（写作环境不依赖实时网络）。这是本章
  cuTile 一节标注「资料不足」的直接原因。
* `https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html`：**未核验**。
  同上，只在 PTO-DSL / CUTLASS README 中作为链接出现。
* `https://github.com/tile-ai/tilelang/issues/3317`：本地快照**只有提问**，未包含回答。
* 其余上游链接均以 corpus 本地快照为准，`已核验` 指「本地文件存在且内容与引用一致」，
  不代表已联网复核上游当日页面。

---

## 待补资料

以下缺口会直接影响本章（以及第 24、25 章）能否给出可复核的结论，按优先级列出。

1. **cuTile 官方资料**（高优先级）。缺：`docs.nvidia.com/cuda/cutile-python/` 的正文，
   以及至少两个官方样例（一个 GEMM、一个带 epilogue 的算子）。需要它才能写出 cuTile 的
   「用户写什么 / 编译器替你做什么」，并判断它与 TileLang 在 NVIDIA 侧的重叠边界。
2. **CUTLASS 融合 epilogue 的完整源文件**（高优先级）。缺：CUTLASS `examples/` 下与
   `media/docs/cpp/cute/0x_gemm_tutorial` 的全文。需要它才能把本章的统一例子在
   CUTLASS-CuTe 一侧写到「可对照」的粒度。
3. **Ascend C 的 GEMM + 融合 epilogue 样例**（中优先级）。缺：CANN 官方 GEMM 样例或
   CATLASS 的具体 kernel 源码。现在只有逐元素（LeakyReLU）样例，无法逐行对照 Cube 侧。
4. **Ascend 950 两条后端路线的官方取舍建议**（中优先级）。缺：issue #3317 的维护者回答
   或官方文档中的选型说明。现在只有提问，不能给出「该用 `ascendc`/`pto` 还是 `npuir`」。
5. **带条件的性能对比报告**（中优先级）。缺：明确写出昇腾型号、CANN 版本、矩阵形状、
   基准口径的 benchmark。现在社区文章的「打平 / 差几倍」不可审计，只能定性引用。
6. **PTO-ISA 发布版规范**（低优先级）。本 corpus 的 `pto-spec` 为开发分支快照
   （架构标识 `0.58.7`），不是发布版；若要引用具体指令语义，应换用其 GitHub Releases
   中某个已发布的 release 身份。
