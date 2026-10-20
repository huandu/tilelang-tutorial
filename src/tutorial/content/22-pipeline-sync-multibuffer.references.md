# 第 22 章 参考资料

本章所有结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现泛指“更新了一版”的说法，所有版本结论一律以本文件的锁定 tag 为准。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU，
并且没有安装 TileLang**。因此本章代码示例的最高验证级别是：
`code/01` = “仅静态核对”（`py_compile` 通过 + API 名逐个核对，**未运行**）；
`code/02`–`code/04` = “CPU 逻辑验证”（本机实际运行）。**没有任何一条性能数字
来自本机实测。**

“已核验（2026-10-01）”的含义：在锁定版本的工作树上用 `grep -n` 定位到定义行，
并阅读了完整签名、注释与周边上下文；行号区间与文件内容一致，且用脚本对本章所有
`path:line-line` 引用做过“区间落在文件行数内”的批量校验（见文末“引用校验方式”）。
**这不是运行验证。**

---

## 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

### 流水规划：stage / order / 深度

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/src/transform/pipeline_planning.cc:1002-1350` | `PipelinePlanner`：自动 stage/order 的分配、“copy 挪到 order 开头并整体减一”的收尾整形 | 已核验（2026-10-01）：逐段阅读 `VisitStmt_(const ForNode*)` |
| S2 | `_vendor/tilelang-v0.1.15/src/transform/pipeline_planning.cc:1266-1277` | §7.3 里“`num_stages=1` 也给出 `[0,0,1]`”的依据：收尾整形要求 `num_stages >= 2` | 已核验（2026-10-01）：`copy_stage_at_end` 分支逐行阅读 |
| S3 | `_vendor/tilelang-v0.1.15/src/transform/pipeline_planning.cc:1287-1293` | “删掉 `num_stages`、只留 `tl_pipelined_num_stages`”的原因原文（下游把它当 active pipeline marker、不支持嵌套流水） | 已核验（2026-10-01）：逐字校对注释 |
| S4 | `_vendor/tilelang-v0.1.15/src/transform/pipeline_planning.cc:1046-1053`、`:1150-1170` | 可重放 `Bind` 的过滤；以及非 `gfx950` ROCm 目标上删掉 `num_stages` 使流水静默退化 | 已核验（2026-10-01） |
| S5 | `_vendor/tilelang-v0.1.15/src/transform/common/pipeline_utils.h:16-40`、`:87-110` | 流水注解键（`software_pipeline_tma_copies` / `_async_producers` / `_groups` / `_replayable_scalar_binds`）与 `GetPipelineNumStages` 的三级读取 | 已核验（2026-10-01）：读入该头文件全文 |
| S6 | `_vendor/tilelang-v0.1.15/src/ir.cc:123-148` | `T.Pipelined` 如何把 `order`/`stage`/`num_stages` 变成循环注解；`if (num_stages > 0)` 的写入条件 | 已核验（2026-10-01） |
| S7 | `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:112-186` | `T.Pipelined` 的签名与 docstring（深度两种给法、`num_stages=0` 不启用、replayable `Bind` 的规则、“避免同时给 `num_stages` 与手写 `order`/`stage`”） | 已核验（2026-10-01）：逐行阅读 |
| S8 | `_vendor/tilelang-v0.1.15/tilelang/analysis/nested_loop_checker.py:12-15`、`:65-100` | 流水注解键清单与嵌套流水“corner cases and undefined behaviours”的约束 | 已核验（2026-10-01） |

### 流水物化：版本维、区域放宽、barrier 展开

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S9 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:77-90`、`:1421-1457` | `BufferAccessInfo{def, use}` 的定义与 `GetBufferAccessInfo`（谁写、谁读、取 min/max） | 已核验（2026-10-01） |
| S10 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1459-1542` | `MayConflict`（区间相交）与 `ComputeBufferVersions`（`use − def + 1` 上界 + `num_versions >= 2` 时的收紧分支；注释称这是 `== 2` 的特例） | 已核验（2026-10-01）：逐行阅读并把“注释与守卫不一致”写进正文 §2.4 |
| S11 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1551-1560` | `RewriteAllocBuffer`：`shape.insert(begin, num_versions)` 与 strides 同步前插 | 已核验（2026-10-01） |
| S12 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1038-1056`、`:1059-1093` | `RewritePipelineBufferRegion`（`access_all_versions` 决定首维区间）与 `RewriteBufferAccess`（指针访问的字节偏移换算） | 已核验（2026-10-01） |
| S13 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1115-1145` | Store / Load 的版本下标前置 `floormod(k − min, nv)` | 已核验（2026-10-01） |
| S14 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:785-886` | `ExpandPipelineBarriers`：`stage_expr = (k−min) mod ns`、`parity_cycle = ((k−min)/ns) mod 2`、`shape = {ns × 原槽数}`，以及“只展开显式 arrive / ISP 自建 barrier”的判据 | 已核验（2026-10-01）：逐段阅读 |
| S15 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:903-1000` | `RewritePipelineTmaBarriers`：把 TMA copy 改写为 `tl.tileop.tma_copy` 并插入 `mbarrier_wait_parity` | 已核验（2026-10-01） |
| S16 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1208-1284` | `BuildPipeline`：版本数计算入口、按 order 排序、prologue / body / epilogue 三段生成 | 已核验（2026-10-01） |
| S17 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:2825-2905` | `EmitImpl` 的 `(start, end, unroll_loop, need_bound_check)` 参数与 `skewed_loop_var = new_loop_var - stage` 的偏斜 | 已核验（2026-10-01）：`EmitImpl` 与调用点逐行阅读 |
| S18 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:2493-2540` | `RelaxTrailingConsumerWaits`：尾段 drain 的 wait 渐进放宽 | 已核验（2026-10-01） |
| S19 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:3097-3138` | `ValidatePipelineBody` 的两条硬校验与报错文案 | 已核验（2026-10-01）：逐字校对 |
| S20 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:3616-3648` | “用 pipeline depth 而不是 SW 流水级数为 barrier 定尺寸”的注释；`RewritePipelineTmaBarriers` 只在**没有 WS** 时生效 | 已核验（2026-10-01）：逐字校对 |
| S21 | `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:4033-4049` | `tl.InjectSoftwarePipeline` 的 pass 注册与 `InjectPipeline` + `ConvertSSA` | 已核验（2026-10-01） |

### 分配合并与同步

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S22 | `_vendor/tilelang-v0.1.15/src/transform/merge_shared_memory_allocations.cc:1-6`、`:1576-1668` | “每个 kernel 只允许一次 shared 分配”的文件头说明与 `MergeSharedMemoryAllocations` 的入口（`merge_static_smem` 配置、显式对齐、`disable_reuse`） | 已核验（2026-10-01）：读入该文件头与 pass 注册 |
| S23 | `_vendor/tilelang-v0.1.15/src/transform/merge_shared_memory_allocations.cc:111-370`、`:453-600` | 线性访问模式扫描（活跃区间）、`SharedMemoryAlignmentPlanner`、`PlanReuse` | 已核验（2026-10-01）：逐段阅读 |
| S24 | `_vendor/tilelang-v0.1.15/src/transform/merge_shared_memory_allocations.cc:862-880`、`:980-1010` | `FreeList::Allocate` 的 best-fit 搜索与 `LinearScanPack` 的排序键 `(start, -size, name)`、retire/coalesce 逻辑 | 已核验（2026-10-01）：逐行阅读；`code/02` 的模型即按此实现 |
| S25 | `_vendor/tilelang-v0.1.15/src/transform/thread_storage_sync.cc:60-120` | `ThreadSyncAfterWaitGroupInserter`：为什么 `__syncthreads()` 不能替代 `wait_group`（注释逐字引用在正文 §4.3） | 已核验（2026-10-01）：读入文件头注释与类定义 |
| S26 | `_vendor/tilelang-v0.1.15/src/transform/thread_storage_sync.cc:212-260`、`:263-430` | `ThreadSyncInserter`（插 `tvm_storage_sync`）与 `ThreadPartialSyncRewriter`（barrier id / 参与线程数的收窄） | 已核验（2026-10-01） |
| S27 | `_vendor/tilelang-v0.1.15/src/transform/thread_storage_sync.cc:603-700` | `TileLangThreadSyncPlanner` 与 `AccessEntry` 的字段（`threads` / `buffer_indices` / `cset` / `touched` / `is_async_copy` / `is_atomic`） | 已核验（2026-10-01）：逐行阅读结构定义 |
| S28 | `_vendor/tilelang-v0.1.15/src/transform/thread_storage_sync.cc:1759-2043` | `FindConflict`：异步写互免、atomic 互免、Z3 的 `prev ⇒ curr` 同参与者判定、字节区间不相交证明、循环携带的 `loop_var + step` 符号移位 | 已核验（2026-10-01）：逐段阅读 |
| S29 | `_vendor/tilelang-v0.1.15/src/transform/thread_storage_sync.cc:2058-2100` | `TileLangThreadSync` 的三段调用顺序与 `tl.ThreadSync` 的 pass 注册 | 已核验（2026-10-01） |
| S30 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:105-110`、`:129-142`、`:247-263` | CUDA 侧 pass 顺序：WS 三趟在流水之前；`PipelinePlanning → InjectSoftwarePipeline → Simplify → LayoutInference → LowerTileOp`；`MergeSharedMemoryAllocations` 在 `SplitHostDevice` 之后；两处 `ThreadSync` | 已核验（2026-10-01）：逐行阅读该文件 |
| S31 | `_vendor/tilelang-v0.1.15/tilelang/cuda/transform/__init__.py:35-47`、`:158-171` | `ProducerConsumerWarpSpecialized` 与 `MaterializeWSSchedule` 的 docstring（“runs before LayoutInference and LowerTileOp”“rewrites eligible pipelined tile-op loops into warp-specialized producer and consumer branches with explicit barrier synchronization”） | 已核验（2026-10-01） |
| S32 | `_vendor/tilelang-v0.1.15/src/cuda/transform/multi_version_buffer_rewriter.cc:700-712` | 另一个消费 `GetPipelineNumStages` 的下游 pass（正文 §8.3 提到的“内部 buffer 版本化”） | 已核验（2026-10-01）：定位到 `VisitStmt_(const ForNode*)` 的读取点 |
| S33 | `_vendor/tilelang-v0.1.15/tilelang/transform/pass_config.py:210` | `TL_DISABLE_SHARED_MEMORY_REUSE` 配置键（§8.2 的诊断开关） | 已核验（2026-10-01） |
| S34 | `_vendor/tilelang-v0.1.15/src/transform/common/mbarrier.h:1-40` | `shared.barrier` 作用域与 `CreateMBarrierBuffer` / `MakeBarrierRef` | 已核验（2026-10-01） |

### 昇腾侧

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S35 | `_vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc:78-260` | 文件头分节、`SyncInsertionSite`、`DepEdge`、`SyncPoint`、`CounterSyncSignature` 的数据模型 | 已核验（2026-10-01）：逐行阅读 |
| S36 | `_vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc:625-780` | `DepClosure`（距离闭包饱和）、`GetHardEvent` 的四种规范化、`HasImplicitSamePipeOrder` 的 S/V/M 假设 | 已核验（2026-10-01）：逐字校对注释 |
| S37 | `_vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc:1226-1400` | `CollectSyncPoints`：owner 内的版本数/距离解析、`-1` 的两种归宿、按 task_pairs 展开同步点 | 已核验（2026-10-01）：逐段阅读 |
| S38 | `_vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc:1729`、`:1742`、`:1772-1800`、`:1839-1870` | `reusable_pairs_`（可复用同步点配对）、`FlagAllocation`、`FlagAllocator` 的 `required_ids[domain] += num_versions` | 已核验（2026-10-01） |
| S39 | `_vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc:2083-2088`、`:2135-2136` | `FlagDomain` 用 `GetHardEvent` 做 key；`kIntraCoreFlagLimit = 8`、`kCrossCoreFlagLimit = 16` | 已核验（2026-10-01）：逐行阅读常量定义 |
| S40 | `_vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc:2299-2307`、`:2344-2480` | `MakeFlagEventId`（`iteration % num_versions + base_id`）与 `SyncMaterializer::InsertSyncPoints` 的三条落地规则 | 已核验（2026-10-01）：逐段阅读 |
| S41 | `_vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc:2485-2545`、`:2551-2568` | `InsertKernelSync` 的完整流程（收集 task、EpochDomain、FlagAllocator、SyncMaterializer、prefix/suffix 同步任务）与 pass 注册 | 已核验（2026-10-01） |
| S42 | `_vendor/tilelang-v0.1.15/src/ascend/transform/rewrite_flag_to_buf.cc:1-31`、`:63-66`、`:201-210`、`:325-340`、`:379-392` | flag 命名空间的设计说明（每个 hard_event 8 槽、溢出到 32 槽 `asc_lock/asc_unlock` 池）、`kFlagSlotsPerPair = 8`、`KnapsackKeep` 的 0/1 背包、pass 注册 | 已核验（2026-10-01）：读入文件头与关键函数 |
| S43 | `_vendor/tilelang-v0.1.15/src/ascend/transform/thread_storage_sync.cc:2054-2124`、`:2126-2146` | `AscendThreadSync` 在昇腾上只处理 `SIMT_VF` 块；“ThreadSyncAfterWaitQueueInserter 与 ThreadPartialSyncRewriter 是 CUDA 专属”的注释 | 已核验（2026-10-01）：逐行阅读 |
| S44 | `_vendor/tilelang-v0.1.15/src/ascend/transform/auto_schedule/dependency_analysis.h:44-97` | `DependencyKind`（`kData` / `kOwnerExclusion`）与 `DepInfo`（含 `distance` 的三档语义） | 已核验（2026-10-01）：逐行阅读 |
| S45 | `_vendor/tilelang-v0.1.15/src/ascend/transform/auto_schedule/multi_buffer.h:159-200` | `MultiBufferOwnerInfo` / `MultiBufferInfo` 的定义与注释（owner、counter 组、`NeedsVersionDimension`） | 已核验（2026-10-01） |
| S46 | `_vendor/tilelang-v0.1.15/src/ascend/transform/annotate_multi_buffer_eligible.cc:1-30` | “最深且互不嵌套的 write-first owner frontier”与 owner-external fill 的规则 | 已核验（2026-10-01）：读入文件头注释 |
| S47 | `_vendor/tilelang-v0.1.15/src/ascend/transform/auto_schedule.cc:705-736` | 昇腾侧读取 `num_stages` 注解（缺省 1）、没有注解时用 `max_manual_stage + 1` | 已核验（2026-10-01）：逐行阅读 |
| S48 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:100-120`、`:160-185` | 昇腾 pass 顺序：`EstimateLatency → AutoSchedule → AssignCore → PrepareMultiBuffer → ResolveCore → InsertSync → MaterializeMultiBuffer → LowerScheduledTIR`；以及 `AscendThreadSync("global")` 与 `RewriteFlagToBuf()` 的位置 | 已核验（2026-10-01）：逐行阅读 |
| S49 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/sync.py:8-215` | 前端同步原语：`ascend_pipe_barrier` / `ascend_set_flag` / `ascend_wait_flag` / `ascend_cross_core_set_flag` / `ascend_cross_core_wait_flag` / `ascend_get_buf` / `ascend_rls_buf` | 已核验（2026-10-01）：逐个读 docstring |
| S50 | `_vendor/tilelang-v0.1.15/src/ascend/transform/buffer_version.h:19-40` | 版本注解契约：`tl.buffer_versions_map` / `tl.manual_multi_buffer` / `multi_buffer_eligible` | 已核验（2026-10-01） |

### TileLang-Ascend 适配器仓库（`_vendor/tilelang-ascend-v0.1.2.000/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S51 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/pipeline.py:14-52` | 该仓库的 `T.Pipelined` 比主干多一个 `cross_interval` 参数（跨核同步间隔），签名与 docstring | 已核验（2026-10-01）：`grep -n "cross_interval"` 定位到 `:19`，逐行阅读；文件共 52 行。**注意**：这是适配器仓库的能力，未在 `_vendor/tilelang-v0.1.15` 中找到同名参数（正文 §6.2 标为“待核”） |
| S52 | `_vendor/tilelang-ascend-v0.1.2.000/docs/tutorials/t_pipelined.md:1-110` | 核内 / 核间流水的逐拍时间表、`cross_interval` 的语义与生成代码示例 | 已核验（2026-10-01）：`sed -n 1,110p` 读入；**未在 NPU 上运行** |
| S53 | `_vendor/tilelang-ascend-v0.1.2.000/VERSION` | 仓库内 `VERSION` 文件（内容与 tag 的关系见 `../../VERSIONS.md`） | 已核验（2026-10-01）：文件存在 |

---

## corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/software_pipeline.md` | https://tilelang.com/programming_guides/software_pipeline.html | §2.2 的 `stage`/`order` 对齐规则、Replayable Scalar Bind、Legacy Annotation Form（第 11 章也引用同一份） | 已核验（2026-10-01）：本地为官方文档源文件全量；逐节阅读 |
| R2 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tutorials/debug_tools_for_tilelang.md` | https://tilelang.com/tutorials/debug_tools_for_tilelang.html | §7.1 的实验入口：`TL_LOWER_TRACE` 的三种模式、`lower_trace` 的编程接口、以及“IR Lower Trace supersedes the older Pass Diff” | 已核验（2026-10-01）：逐节阅读 |
| R3 | `../tilelang-corpus/02-tilelang-core-repo/raw/testing/python/transform/test_tilelang_transform_pipeline_planning.py` | https://github.com/tile-ai/tilelang/blob/main/testing/python/transform/test_tilelang_transform_pipeline_planning.py | **本章最重要的外部读数**：`test_pipeline_planning_before_after_copy_gemm_num_stages_plan` 对 `num_stages=3` 给出 `software_pipeline_stage: [0, 0, 2]`、`order: [0, 1, 2]`、`tl_pipelined_num_stages: 3`。图 2、§7.3、E14 都引用它 | 已核验（2026-10-01）：读入该测试的 `before`/`after` 两个 `prim_func` 与断言；**未运行** |
| R4 | `../tilelang-corpus/02-tilelang-core-repo/raw/testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py` | 同上目录 | §4.3 的 parity 佐证：`test_pipeline_explicit_gemm_mbar_uses_logical_epoch` 断言 gemm 的 mbar phase 等于 `(loop_var − loop.min) % 2` | 已核验（2026-10-01）：读入该测试；**未运行** |
| R5 | `../tilelang-corpus/02-tilelang-core-repo/raw/testing/python/transform/test_tilelang_transform_disable_memory_reuse.py` | 同上目录 | §8.2 的诊断开关：`PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE` 的行为说明（“still merged into a single buffer, but each buffer gets its own dedicated region without lifetime-based reuse”） | 已核验（2026-10-01）：读入文件头 docstring 与测试体；**未运行** |
| R6 | `../tilelang-corpus/02-tilelang-core-repo/raw/.agents/skills/tilelang-ascend/references/synchronization.md` | https://github.com/tile-ai/tilelang（仓库内文档） | §3.5、§4.4、§5.1 的昇腾语义：三种调度模式、buffer 版本与 owner、iteration/counter 两种时钟、条件执行下的槽位分叉、`T.assume_no_conflict` 的提示机制、以及“Trace the complete storage protocol”这张检查清单 | 已核验（2026-10-01）：读入全文（约 300 行） |
| R7 | `../tilelang-corpus/02-tilelang-core-repo/raw/.agents/skills/tilelang-semantic/references/memory-concurrency.md` | 同上 | §4.3、§5.1 的语义规则：mbarrier 生命周期六条、异步操作的五条要求（“配对 commit/wait 组”“防止在完成前覆盖目标 buffer 或复用版本”）、以及“shared memory is cross-thread state; source order alone does not prove initialization under warp specialization” | 已核验（2026-10-01）：读入该节全文 |
| R8 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md` | https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | §6.1 的厂商原话：第 303–320 行把 mbarrier 描述为 asynchronous transaction barrier（“instead of counting just thread arrivals, it also counts transactions”） | 已核验（2026-10-01）：`grep -n` 定位到第 314、316 行并阅读上下文；厂商公开资料，非实测 |
| R9 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/tutorials/t_pipelined.md` | https://github.com/tile-ai/tilelang-ascend | §6.2 的核间流水逐拍时间表与 `cross_interval` 的生成代码示例（与 S52 同一份文本，条目分开是为了标明“本地快照”与“锁定版本”两种引用口径） | 已核验（2026-10-01）：与 `_vendor/tilelang-ascend-v0.1.2.000/docs/tutorials/t_pipelined.md` 逐段比对，内容一致 |
| R10 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/.agents/skills/tilelang-pass-analyzer/T.pipelined.md` | 同上 | **反例资料（见“待补资料”第 2 条）**：这份文档描述的昇腾 pass 名为 `CrossCorePipeline` / `CombineCV` / `AscendSyncInsert`，与 v0.1.15 主干的 `EstimateLatency → AutoSchedule → … → InsertSync` 不是同一套 | **未核验（不一致）**：2026-10-01 读取该文件全文；无法在 `_vendor/tilelang-v0.1.15` 里找到这三个 pass 名，因此**本章不采用它作为机制依据**，只把它记录为资料缺口 |

---

## 上游链接核验说明

* 上表中标注“已核验”的链接，核验方式是：确认对应 corpus 文件在本地存在，
  且其头部记录了抓取日期（2026-09-30）与原始 URL。本章写作时**未重新发起网络请求**。
* `R8` 是 NVIDIA 官方资料，本章只引用其中的**定性描述**（事务屏障的计数语义），
  未把它当作本机实测结果。
* `R3`/`R4`/`R5` 是上游测试文件，本章只引用其中的**断言文本**作为“同一形态的权威
  读数”，并明确标注“未运行”。
* 所有指向 `_vendor/` 的行号都在锁定 tag 上核验；如果上游 tag 本身发生漂移
  （不可能，但为完备起见），以 `../../VERSIONS.md` 记录的两个 commit 为准。

---

## 关于性能数字的说明

本章没有引用任何厂商性能数字，也没有本机实测。文中出现的每一个数字都能落到下面
三类之一：

| 数字 | 来源 | 口径 |
| --- | --- | --- |
| `stage = [0, 0, 2]` / `order = [0, 1, 2]` / `tl_pipelined_num_stages = 3` | **上游单测断言**（R3），非本机运行 | `num_stages=3`、`A(128,32)+B(32,128)` fp32 的 copy/copy/gemm 形态 |
| `mbar phase = (loop_var − loop.min) % 2` | **上游单测断言**（R4），非本机运行 | 同上形态的显式 `T.serial` + annotated stage |
| arena 96/64/128/160/192/240 KiB、`[OVER]`、offset 0/32/48/64 | **本章模型**（`code/02`，CPU 逻辑验证） | 128×128×64 fp16、单版本 16 KiB、预算 228 KiB、`iterations=16` |
| 稳态 20.0 / 22.6 / 28.0 / 38.0 / 128.0 周期，makespan 350/388/478/628 | **本章模型**（`code/03`，CPU 逻辑验证） | `issue=1, latency=30, compute=20, barrier=8, iterations=16`（另注明 `latency=120` 与 `compute=60` 两组） |
| flag 需求 4/8/12/16、`kept/spilled` 划分 | **本章模型**（`code/04`，CPU 逻辑验证） | 每 hard event 8 槽（跨核 16）、4 个同步点、`num_versions` 取自深度 |
| “每个 hard_event 8 个 event id / 跨核 16 / 32 槽互斥池” | **锁定源码**（S39、S42） | `kIntraCoreFlagLimit = 8`、`kCrossCoreFlagLimit = 16`、文件头注释 |
| “asynchronous transaction barrier 数事务而非线程” | **厂商公开资料**（R8，H100） | 原文第 314、316 行 |

**没有来源的、或来自厂商宣传口径的数字一律未采用。**

---

## 待补资料

1. **缺少可在无卡环境运行的“流水 IR 基线”dump。** 本章 §3、§7 讲的是
   `InjectSoftwarePipeline` 的行为，`code/01` 提供了脚本，但作者本机没有 TileLang
   也没有 GPU，因此正文没有给出“`num_stages=2` 与 `num_stages=3` 的真实 IR
   diff”。建议后续版本补一份“同一段 GEMM 在 `num_stages=2/3` 下
   `InjectSoftwarePipeline` 前后的 IR dump（含 `__shared__` 首维与 wait 参数）”，
   放到 corpus 的官方文档或综合类别下，第 11、12、20 章同样能用。
2. **corpus 里昇腾流水的两份文档与 v0.1.15 主干不一致（已在本章标为待核）。**
   `R10`（`…/tilelang-pass-analyzer/T.pipelined.md`）描述的 pass 名是
   `CrossCorePipeline` / `CombineCV` / `AscendSyncInsert`，这在
   `_vendor/tilelang-v0.1.15` 与 `_vendor/tilelang-ascend-v0.1.2.000` 里都找不到；
   `_vendor/tilelang-ascend-v0.1.2.000` 的 `tilelang/language/pipeline.py` 里的
   `cross_interval` 也不在主干。建议后续版本补一条“昇腾流水 pass 结构在本版本到底是
   哪一套”的权威来源，并在 corpus 里给旧文档打上版本标注；第 19、23、25 章都会
   直接受益。
3. **昇腾 UB / L1 / L0 的公开容量数字缺失。** 本章 §3.5 只能说明“版本数由容量与
   生命周期共同决定”，无法给出昇腾侧的对应预算数字（`code/02` 里用的是 NVIDIA
   的 228 KiB）。建议在 corpus 的 `06-huawei-ascend/` 下补一条官方文档来源。
4. **`RewriteFlagToBuf` 的溢出对性能的影响缺少可引用依据。** 本章只能从源码语义
   推出“事件标志 → 阻塞锁”这一转变的方向，无法量化。若有昇腾侧的分析或实测材料
   （即使是社区口径），应在第 25 章补上。
5. **`num_stages=1` 的三语句 GEMM 形态缺少上游测试覆盖（已标为待核）。**
   `PipelinePlanning` 的注释与守卫都指向“`num_stages=1` 仍生成深度 2 的
   stage/order”，但我找到的上游测试只覆盖了 `num_stages=2`（两语句形态）与
   `num_stages=3`（三语句 GEMM 形态）。建议后续版本补一条覆盖 `num_stages=1` 的
   断言或 IR dump，正文 §7.3 的表与 E14 都会因此升级为“有直接证据”。

---

## 引用校验方式

写完后对本章四个 Markdown 文件做了一类机械校验：

2. **行号区间校验**：用脚本抽取正文中所有 `path:line` 与 `path:line-line` 形式
   的引用，逐个确认目标文件在 `_vendor/` 下存在、且区间落在该文件行数之内。
   这一步能抓“行号写反”“引用到不存在的文件”，但**不能**判断“这段代码是不是真的
   在讲这件事”，后者由上面两张引用表的最后两列逐条人工阅读完成。
