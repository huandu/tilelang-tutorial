# 第 06 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现“最新版”这类表述。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章没有任何本机实测的性能数字；所有容量、带宽、延迟都来自厂商公开文档或 corpus
里的社区一手文章，并逐条标注条件。

本章把来源分成三类，引用时的标记也就有三类：

* **厂商文档**（NVIDIA / 华为昇腾官方）：作为“厂商宣传或公开规格”，不一定经过独立
  复核，凡性能/算力数字都注明其 preliminary / 理论值性质。
* **社区一手文章**：明确标注“社区，非厂商官方文档”，其数字一律视为未确认。
* **锁定版本源码**：只用于“这个 API/流程在 v0.1.15 里存不存在、签名是什么”，
  在 `_vendor/` 工作树上逐条 `grep`/阅读核对过，**不代表运行验证**。

---

## 锁定的源码引用（`_vendor/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:60-153` | 昇腾 950 后端 Quick Start（GEMM+ReLU）、`T.alloc_l1`/`alloc_l0c`/`dual_copy`/`SimtVF` 用法、与 Ascend C 的分工对照表、“Synchronization：None（Just write operations in order!）” | 已核验（2026-10-01）：文件存在，逐行阅读；**未在 NPU 上运行** |
| S2 | `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:49,81` | 通用 `T.alloc_shared`（`shared.dyn`）与 `T.alloc_fragment`（`local.fragment`）签名 | 已核验（2026-10-01） |
| S3 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:15,26,37,46,55` | 昇腾 `alloc_shared`（UB）、`alloc_l1`、`alloc_l0a`、`alloc_l0b`、`alloc_l0c` 的 scope 与布局说明 | 已核验（2026-10-01） |
| S4 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/copy_op.py:42,91` | `disable_tma=True` 可禁止 TMA 降级；TMA im2col 的 L2 eviction hint | 已核验（2026-10-01） |
| S5 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py:53-140` | `T.dual_copy` 的语义、支持的存储路径、M/N 拆分与对齐约束 | 已核验（2026-10-01） |
| S6 | `_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:160,177-181` | `T.gemm` 签名；Hopper 隐式插入 wgmma wait、Blackwell 隐式插入 `mbarrier_wait_parity` | 已核验（2026-10-01） |
| S7 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py:226,264` | `T.wgmma_gemm`、`T.tcgen05_gemm`（手工异步调度入口） | 已核验（2026-10-01） |
| S8 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py:107,179` | `T.SimtVF(threads=..., latency=0)`（可 1–3 维线程域）与 `T.SimdVF(latency=0)`（无线程维度） | 已核验（2026-10-01） |
| S9 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/cluster.py:27-46` | `cluster_arrive_relaxed/arrive/wait/sync`、`block_rank_in_cluster` | 已核验（2026-10-01） |
| S10 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:103-108` | CUDA 侧 warp specialization 三个 pass：`AutoWarpSpecialization` → `MaterializeWSSchedule` → `ProducerConsumerWarpSpecialized` | 已核验（2026-10-01） |
| S11 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:107-115` | 昇腾侧自动调度 pass 顺序：`EstimateLatency → AutoSchedule → AssignCore → PrepareMultiBuffer → ResolveCore → InsertSync → MaterializeMultiBuffer → LowerScheduledTIR` | 已核验（2026-10-01） |
| S12 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/warpgroup.py:19` | `T.WarpSpecialize(*warp_group_idx)` | 已核验（2026-10-01） |
| S13 | `_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md`（“同步原语”一节） | Set/Wait 手动同步的完整生命周期、`T.set_flag`/`T.wait_flag`/`T.barrier_all`/`T.pipe_barrier`/`T.sync_all`、MTE2↔MTE1 的正反向归还示例 | 已核验（2026-10-01）：本地为仓库文档快照；**未在 NPU 上运行** |
| S14 | `_vendor/tilelang-v0.1.15/examples/quickstart.py` | NV 侧 GEMM+ReLU 骨架（`03-target-sketch/nv_hopper_sketch.py` 的原型） | 已核验（2026-10-01）：文件存在，API 逐个比对；**未运行** |

“已核验”的含义：在锁定版本的工作树上用 `rg`/`grep -n` 定位到定义行，并阅读了完整
签名与 docstring；行号区间与文件内容一致。**这不是运行验证。**

---

## corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

### NVIDIA 侧

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md` | https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | H100：HBM3“over 3 TB/sec”、50 MB L2、130+ SM、每 SM 256 KB 寄存器、thread block cluster、DSM、TMA、async transaction barrier、“first truly asynchronous GPU”、单线程驱动 TMA | 已核验（2026-10-01）：本地为全文；**厂商公开资料，非实测**；算力为 preliminary |
| R2 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-hopper-tuning-guide.md` | https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html | shared per block 227 KB、carveout 列表 `0..228 KB`、TMA 1D–5D 与写回 reduction、warp specialization、portable cluster 8 / opt-in 16 | 已核验（2026-10-01）：本地为全文；厂商文档 |
| R3 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md` | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html | 未缓存访存“hundreds of clock cycles”（第 3629 行）、算术指令约 4 cycles（第 3145 行）、访存合并（第 934 行）、CPU 重线程 vs GPU 轻线程（第 94–98 行） | 已核验（2026-10-01）：本地为全文；厂商文档 |
| R4 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blackwell-tuning-guide.md` | https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html | cc10.0 shared 228 KB/SM、227 KB/block、CUDA 保留 1 KB、静态 48 KB、B200 HBM3e 180 GB、GB200 L2 126 MB、cluster 8/16 | 已核验（2026-10-01）：本地为全文；厂商文档 |
| R5 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md` | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html | `mma`（9.7.16，第 23418 行起，“distribution ... explicitly”在第 23449 行）、`wgmma`（9.7.17，warpgroup 定义在 28113 行、执行步骤在 28088 行起）、`tcgen05`（9.7.18；TMEM 定义 29726–29745 行、矩阵位置 31639–31690 行） | 已核验（2026-10-01）：本地为全文；厂商文档 |

### 昇腾侧

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R6 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_0008_html.md` | https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/Ascendcopdevg/atlas_ascendc_10_0008.html | AI Core 计算/存储/搬运单元（第 17、111–126 行）、Cube 16×16、Vector=SIMD、常见数据流（第 141–142 行）、SetFlag/WaitFlag 语义（第 157–158 行）、`GetCoreMemSize`。**本页没有对齐表**；对齐表在第 4.2 节另引的姊妹页 `..._atlas_ascendc_10_0011_html.md:43-51` | 已核验（2026-10-01）：厂商文档全文；**非实测** |
| R7 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/developer/zh_developer_techArticles_20240819-1.md` | https://www.hiascend.com/developer/techArticles/20240819-1 | 指令队列 V/M/S/MTE1/2/3（第 92 行）、double buffer 动机（第 90–107 行）、CopyIn/Compute/CopyOut 各 $t$ 时 Vector 利用率 1/3（第 99 行） | 已核验（2026-10-01）：厂商官方技术文章；其中的利用率是**示意模型**，非实测 |
| R8 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_10064_html.md` | https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/Ascendcopdevg/atlas_ascendc_10_10064.html | AI Core SIMT 抽象模型、“目前仅支持 Atlas 350 加速卡”（第 3、10 行）、每个 Vector Core 含 UB/寄存器/栈、Data Cache 是 UB 的一部分（最大 128 KB、最小 32 KB，第 20–21 行） | 已核验（2026-10-01）：厂商文档全文 |
| R9 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/hardware/zh_hardware_processor.md` | https://www.hiascend.com/hardware/processor | 昇腾 950PR：SIMD+SIMT 混合编程（第 23、47 行）、Cube Core 32/28、Vector Core 64/56、Memory 128/112 GB、带宽 1.6/1.4 TB/s、L2 128/112 MB（第 57–79 行） | 已核验（2026-10-01）：厂商公开产品页；**规格为“满配/另一档”两值，非实测** |
| R10 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_best_practices_10_00020_html.md` | https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00020.html | Unified Buffer 192 K / 48 bank / 16 bank group 的示例（第 18 行） | 已核验（2026-10-01）：厂商文档；**是某型号示例，不是通用规格** |
| R11 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_API_ascendcopapi_atlasascendc_api_07_0666_html.md` | https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlas_ascendc_api_07_0666.html | “该示例 L1 Buffer 可用大小为 512KB”的示例注释（第 85 行） | 已核验（2026-10-01）：厂商文档；**是示例条件，不是通用规格** |
| R12 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/hardware/zh_hardware_accelerator-card.md` | https://www.hiascend.com/hardware/accelerator-card | Atlas 350 加速卡采用 Ascend 950PR、支持 SIMT/SIMD、1561 TFLOPS mxFP4、1.4 TB/s（第 41、57、59、76 行） | 已核验（2026-10-01）：厂商公开产品页 |

### 社区一手文章（非厂商官方文档）

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R13 | `../tilelang-corpus/09-zhihu-da-zhi-ruo-yu/raw/zhihu-dazhiruoyu-deepseek-ascend.md` | https://www.zhihu.com/question/2088578683623814216/answer/2088624284100999121 | NV vs 昇腾 950 的架构差异表（程序组织/计算分工/片上存储/数据传递/同步）、Cube/AIC 与 Vector/AIV、Z3/SMT 自动调度、latency 与 issue interval、`T.alloc_l1`/`T.SimtVF`/`T.SimdVF`/`T.dual_copy` | **部分核验**：本地为登录态抓取的全文；链接需登录；内容属**社区一手文章，非厂商官方文档**。文中“性能与 Ascend C 打得有来有回”“SIMT 路径 UB 带宽只有 SIMD 的一半”等说法**无公开基线与测量口径**，本章只作定性引用并标注为未确认 |

---

## 本章未采用 / 待补资料

本章刻意**没有**给出以下数字，因为在本 corpus 内找不到可跨型号引用、且经得起复核的
出处。这是“不知道就写不知道”的落实，不是遗漏：

| 缺什么 | 为什么需要 | 建议来源 |
| --- | --- | --- |
| 昇腾 L0A / L0B / L0C / BT / FP Buffer 的容量 | 第 4.2 节要回答“一块 Cube tile 开几份多缓冲”；没有容量就无法算 | 各型号的《Ascend C 算子开发指南》架构规格章节，或真实设备上的 `GetCoreMemSize` 实测输出 |
| 昇腾各级存储的带宽与延迟 | 第 1.4、2.1 节只给了结构，没给数字 | 华为官方架构白皮书或独立微基准论文 |
| 昇腾 950 SIMT 路径与 SIMD 路径的带宽比 | 只有社区文章提到“一半”，未经官方复核 | 华为官方文档或本文作者在 Atlas 350 上的微基准 |
| NV 各代 Tensor Core 的每 SM 吞吐随 tile 形状的变化 | 第 3.3 节只讲结构演进，不比较性能 | NVIDIA 官方 tuning guide 或 corpus 里的 Hopper 微基准论文（`arxiv-2402-13499`，本 corpus 仅存摘要） |
| H100 / B200 的纯 TMEM 容量换算成字节的官方表述 | 本 corpus 只有“512 列 × 128 行 × 32-bit / CTA”的 PTX 表述 | PTX ISA 后续版本或 NVIDIA 白皮书 |

## 引用规则提醒

* 源码引用一律写成 `_vendor/tilelang-v0.1.15/<相对路径>` 或
  `_vendor/tilelang-ascend-v0.1.2.000/<相对路径>`，并给出行号。
* 资料引用一律写成 `../tilelang-corpus/<类别>/raw/<文件>`。
* 厂商宣传（“算力”“极致”“领先”）与可验证事实分开陈述；数字必须带测量/发布条件。
* 社区文章的结论不得升级为事实；本章 R13 的每条数字都按未确认处理。
