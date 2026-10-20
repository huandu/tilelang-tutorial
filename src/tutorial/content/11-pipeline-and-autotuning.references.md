# 第 11 章 参考资料

本章所有结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现“最新版”这类表述。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章所有代码示例的最高验证级别是“仅静态核对”（`code/04`、`code/05`），
三个纯标准库工具（`code/01`–`code/03`）是“CPU 逻辑验证”。**没有任何一条性能数字
来自本机实测。**

“已核验（2026-10-01）”的含义：在锁定版本的工作树上用 `grep -n` 定位到定义行，
并阅读了完整的签名与注释；行号区间与文件内容一致。**这不是运行验证。**

---

## 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

### 流水线语言层与 IR 层

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:112-186` | `T.Pipelined(start, stop, num_stages, order, stage, annotations)` 的签名与完整 docstring：深度两种给法、`num_stages=0` 不启用流水、replayable scalar `Bind` 的规则 | 已核验（2026-10-01）：`grep -n "def Pipelined"` → 112；逐行读完 docstring 与函数体 |
| S2 | `_vendor/tilelang-v0.1.15/src/ir.cc:123-148`、`:280-287` | `PipelinedFor` 如何把 `order`/`stage`/`num_stages` 变成循环注解 `tl_pipeline_order` / `tl_pipeline_stage` / `num_stages`；以及 `if (num_stages > 0)` 的写入条件 | 已核验（2026-10-01）：逐行阅读 `PipelinedFor` 与 `TVM_FFI_STATIC_INIT_BLOCK` 的注册 |
| S3 | `_vendor/tilelang-v0.1.15/src/transform/pipeline_planning.cc:1002-1310` | `PipelinePlanner` 的自动 stage/order 分配：摊平循环体、剔除可重放 `Bind`、`PropagateBufferProducersForCopy`、`AnalyzeCopyLastUse`、收尾整形、写注解 | 已核验（2026-10-01）：逐段阅读；`:1287-1293` 的“不保留 `num_stages`”注释已逐字核对 |
| S4 | `_vendor/tilelang-v0.1.15/src/transform/pipeline_planning.cc:1356-1365` | `tl.PipelinePlanning` 的 pass 注册与 `use_async_copy` 配置读取 | 已核验（2026-10-01） |
| S5 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1208-1265` | `PipelineRewriter::BuildPipeline`：版本数计算入口、按 order 排序、prologue / body / epilogue 三段生成、等待收紧、以及“pipeline 不拥有本地分配时不额外造 block”的分支 | 已核验（2026-10-01） |
| S6 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1474-1544`、`:1546-1560`、`:3628-3638` | `ComputeBufferVersions`（`use - def + 1` 与双缓冲特例）、`RewriteAllocBuffer`（把版本维插到 `shape` 最前面），以及“`num_stages` 是缓冲区份数、`max(stage)+1` 是 SW 流水级数”的注释 | 已核验（2026-10-01） |
| S7 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:3097-3138` | `ValidatePipelineBody`：`order` 唯一、生产者不得晚于消费者的两条硬校验，以及它们的报错文案 | 已核验（2026-10-01） |
| S8 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:4033-4048` | `tl.InjectSoftwarePipeline` 的 pass 注册、`InjectPipeline` + `ConvertSSA` 的调用顺序 | 已核验（2026-10-01） |
| S9 | `_vendor/tilelang-v0.1.15/src/transform/common/pipeline_utils.h:74-110` | `GetPipelineNumStages` 的三级读取顺序（`num_stages` → `tl_pipelined_num_stages` → `max(stage)+1`） | 已核验（2026-10-01）：完整读入该头文件 |
| S10 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:126-142` | CUDA 侧 pass 顺序：`PipelinePlanning → InjectSoftwarePipeline → Simplify → LayoutInference → LowerTileOp`，以及“流水先于 layout 推导”的注释 | 已核验（2026-10-01） |
| S11 | `_vendor/tilelang-v0.1.15/src/transform/merge_shared_memory_allocations.cc:1576-1668` | shared 分配的合并与复用，用来说明 4.5 节的乘积是**上界** | 已核验（2026-10-01）：读入 `MergeSharedMemoryAllocations` 及 `merge_static_smem` 配置读取 |
| S12 | `_vendor/tilelang-v0.1.15/tilelang/analysis/nested_loop_checker.py:12-15`、`:65-100` | 流水注解键清单（`num_stages` / `tl_pipeline_order` / `tl_pipeline_stage`）与 `NestedLoopChecker` 的嵌套规则（`T.Pipelined` 不得嵌在 `T.Parallel` 内；嵌套循环“corner cases and undefined behaviours”） | 已核验（2026-10-01）：读入该文件全文 |

### 自动调优与 carver

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S13 | `_vendor/tilelang-v0.1.15/tilelang/autotuner/tuner.py:1318-1400` | `tilelang.autotune` 装饰器的完整签名与 docstring（`configs` 的两种形式、`warmup`/`rep`/`timeout`、`ref_prog`、`skip_check` 等） | 已核验（2026-10-01） |
| S14 | `_vendor/tilelang-v0.1.15/tilelang/autotuner/tuner.py:105-160`、`:809-1000`、`:904-913` | `AutoTuner` 类、`from_kernel`、`set_compile_args` / `set_profile_args` / `run`；`configs` 为 `Callable` 时的调用分支；`Unused keys in config` 校验 | 已核验（2026-10-01） |
| S15 | `_vendor/tilelang-v0.1.15/tilelang/autotuner/param.py:155-177`、`:26-31` | `AutotuneResult` 的字段（`latency`/`config`/`ref_latency`/`libcode`/`func`/`kernel`）与缓存文件名常量（`best_config.json` 等） | 已核验（2026-10-01） |
| S16 | `_vendor/tilelang-v0.1.15/tilelang/carver/README.md` | carver 的定位、模板清单、hint 的字段（`block`/`warp`/`rstep`/`use_tc`）、支持的架构 | 已核验（2026-10-01）：逐节阅读 |
| S17 | `_vendor/tilelang-v0.1.15/tilelang/carver/arch/arch_base.py:1-12` | `TileDevice.smem_cap` / `reg_cap` / `warp_size` 等硬件建模字段 | 已核验（2026-10-01） |
| S18 | `_vendor/tilelang-v0.1.15/tilelang/carver/roller/policy/tensorcore.py:320-335`、`_vendor/tilelang-v0.1.15/tilelang/carver/roller/policy/default.py:553-560` | cost model 里的容量约束：超 `smem_cap` 时改用动态 shared，或直接标记候选无效 | 已核验（2026-10-01） |
| S19 | `_vendor/tilelang-v0.1.15/examples/gemm/example_gemm_autotune.py` | 本章代码的骨架来源：手写笛卡尔积、roller hint → kernel 参数的映射、`get_heuristic_config()` 固定最优配置 | 已核验（2026-10-01）：文件存在，逐行比对；**未运行**（无 GPU） |
| S20 | `_vendor/tilelang-v0.1.15/examples/gemm/example_gemm.py` | 最小 `T.Pipelined` GEMM 参考 | 已核验（2026-10-01）；未运行 |

### 昇腾侧对照

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S21 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:107-115` | 昇腾侧的 `EstimateLatency → AutoSchedule → AssignCore → PrepareMultiBuffer → ResolveCore → InsertSync → MaterializeMultiBuffer` 序列 | 已核验（2026-10-01） |
| S22 | `_vendor/tilelang-v0.1.15/tilelang/ascend/transform/z3_scheduler.py:67-140`、`:228-256` | Z3 求解器的输入（`latencies`/`iis`/`resource_flags`/`data_deps`/`resource_deps`/`owner_exclusion_deps`/`pipe_order_deps`）与 FFI 包装 `tl.transform.z3_schedule_python` | 已核验（2026-10-01）：逐行阅读参数文档与求解器构造 |
| S23 | `_vendor/tilelang-v0.1.15/src/ascend/transform/auto_schedule.cc:174-190`、`:452-480` | C++ 侧调用 Python Z3 调度器、求解失败时的回退路径 | 已核验（2026-10-01）：`grep -n "z3_schedule_python"` 定位后阅读上下文 |
| S24 | `_vendor/tilelang-v0.1.15/src/ascend/transform/estimate_latency.cc:1-40` | 昇腾任务延迟与 initiation interval 的估计 pass（第 23 章的入口） | 已核验（2026-10-01）：读入文件头与 include 列表，确认其定位 |
| S25 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:60-155` | 昇腾 GEMM+ReLU 例子（`T.Pipelined(..., num_stages=2)`）、`T.alloc_l1` / `alloc_l0c` 等存储作用域、以及“与 Ascend C 的分工”对照表 | 已核验（2026-10-01）；**未在 NPU 上运行** |

---

## corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/software_pipeline.md` | https://tilelang.com/programming_guides/software_pipeline.html | 本章 3.2–3.4 节的权威来源：`stage`/`order` 的对齐规则、Reordered Pipeline、Replayable Scalar Bind、Legacy Annotation Form、手动 `T.serial` 注解、Checklist | 已核验（2026-10-01）：本地为官方文档源文件全量；逐节阅读 |
| R2 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tutorials/auto_tuning.md` | https://tilelang.com/tutorials/auto_tuning.html | 本章 3.5、3.9 节的入门版：三步流程、`itertools.product` 生成候选、Carver `MatmulTemplate` → `recommend_hints(topk)`、hint 字段到 kernel 参数的映射 | 已核验（2026-10-01）：本地为全量源文件 |
| R3 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/autotuning.md` | https://tilelang.com/programming_guides/autotuning.html | 本章 3.5–3.8 节的主要来源：两条接口、输入供给的优先级、正确性检查与容差、调参建议、并行编译与超时、缓存与环境变量、手动 sweep 的 `par_compile`、`AutotuneResult.load_from_disk` | 已核验（2026-10-01）：本地为全量源文件 |
| R4 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md` | https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | 第 1.2 节的异步执行引文（第 79、241 行）；第 4.5 节 228 KB 的来源（第 381、415 行的“Shared Memory Size / SM \| Configurable up to 228 KB”，以及第 162 行的“256 KB of combined shared memory and L1 data cache”） | 已核验（2026-10-01）：本地为全文，行号逐条比对；厂商公开资料，非实测 |
| R5 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md` | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html | 第 1.2 节的延迟量级：第 3629 行“hundreds of clock cycles”（未缓存 local/global 访存）、第 3145 行“typically 4 cycles”（compute capability 7.0 的多数算术指令） | 已核验（2026-10-01）：`grep -n` 定位到两行并阅读上下文 |
| R6 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/README.md` | https://github.com/tile-ai/tilelang-ascend | 第 511–535 行的 `T.Pipelined` 一节：核内/核间流水的职责、**不支持嵌套**的明确约束、以及核内与核间两种写法的正确/错误示例。第 1.4、2.6、4.2、4.6 节引用 | 已核验（2026-10-01）：读入 500–560 行；本地快照的 `VERSION` 为 `0.1.4`，与 `_vendor/tilelang-ascend-v0.1.2.000/VERSION` 一致。**注意**：本地快照与 `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md` 不是同一份文本，本章中“uses a two-stage pipeline for the reduction loop”那句来自 **`_vendor` 的版本**（见 S25），其余引文来自本行指向的 corpus 快照 |

---

## 上游链接核验说明

* 上表中标注“已核验”的链接，核验方式是：确认对应 corpus 文件在本地存在，且其头部
  记录了抓取日期与原始 URL。本章写作时**未重新发起网络请求**（corpus 抓取日期为
  2026-09-30，早于本章写作日期 2026-10-01，差异在 1 天内）。
* `R4`、`R5` 是 NVIDIA 官方资料，本章只引用其中的**定性描述**与**厂商公开规格**，
  未把它们当作本机实测结果。
* `R2`、`R3` 里的代码片段包含上游文档的原始字符（例如 `â` 这类编码残留），
  本章引用时只取语义，不照抄乱码。

---

## 关于性能数字的说明

本章出现的每一个数字都带来源与口径，汇总如下：

| 数字 | 来源 | 测量/口径条件 |
| --- | --- | --- |
| 400 / 220 / 190（抽象周期），1.00 / 1.82 / 2.11× | **本章模型**：`code/01-pipeline-timeline/pipeline_timeline.py` | `issue=1, latency=30, compute=20, iterations=8`；贪心 list schedule |
| E10 的两张扫描表（最小够用深度 2/2/3/5/7 等） | **本章模型**：同一脚本 | `--iterations 16`，`latency ∈ {10,20,40,80,120}` 与 `compute ∈ {5,10,20,30,40}` |
| 最大 `num_stages`：7（128×128×64）、28（128×128×16）、3（256×256×64） | **本章模型**：`code/02-shared-memory-budget/smem_budget.py` | `num_stages × (A + B)` 的上界，预算 228 KiB，fp16 |
| 216 个候选 / 214 个合法 | **本章模型**：`code/03-autotune-config-space/config_space.py`、`code/05` 的 `grid_configs` | 4096³、fp16、228 KiB 预算 |
| 228 KB（每 SM 可配置 shared）、256 KB（shared+L1） | R4 | H100，厂商公开资料 |
| 访存“hundreds of clock cycles”、算术“typically 4 cycles” | R5 | 未缓存 local/global 访存；compute capability 7.0 |

**没有来源的、或来自厂商宣传口径的数字一律未采用。** 所有需要真实加速器才能得到的
数字（实测算力利用率、延迟、吞吐）在本章中均未出现。

---

## 待补资料

1. **缺少可在无卡环境执行的“流水 IR 对照基线”。** 本章 4.2–4.4 节讲的是
   `PipelinePlanning` 与 `InjectSoftwarePipeline` 的行为，但作者无法在本机编译一个
   真实 kernel 去 dump 前后 IR，因此正文没有给出“编译前后 IR 的逐行 diff”。
   corpus 里也没有这类基线文件。建议后续版本补一份“同一段 GEMM 代码在
   `num_stages=2/3` 下编译前后的 IR dump”，放到 corpus 的官方文档或综合类别下，
   第 12 章与第 20 章同样能用。
2. **`TileSight` / 解析性能模型的可用材料有限。**
   本章 3.9 节关于 cost model 的讨论只用到 carver 自身的实现。若要在第 28 章展开
   “tile 与深度怎么选”的方法论，需要 `../tilelang-corpus/03-tilelang-papers/raw/`
   下更完整的模型正文（目前只有摘要）。建议补抓。
3. **昇腾侧 L1 / UB / L0 的公开容量数字缺失。**
   本章 4.5 节与 5.1 节只能说明“约束存在且资源不同”，无法给出昇腾的对应预算数字。
   `code/02` 里那个 `ascend-l1-placeholder` 明确标注为占位值，只用于走通判断分支。
   建议在 corpus 的 `06-huawei-ascend/` 下补一条官方文档来源，第 25 章会直接受益。
