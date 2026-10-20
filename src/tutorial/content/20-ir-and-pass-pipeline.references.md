# 第 20 章 参考资料

本章所有结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`，`VERSION` = `0.1.15`）与
TileLang-Ascend `TileLang-ascend-v0.1.2.000-release`
（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU，
没有安装 tilelang/tvm**。因此本章代码示例的最高验证级别是“仅静态核对”，
三个只读文件/解析源码的脚本是“CPU 逻辑验证”（见 `code/` 下各 README）。
**本章不包含任何自测性能数字。**

核验日期均为 2026-10-01。核验方式：在锁定版本的工作树上用 `grep -n` / `sed -n`
定位定义行与行号，阅读完整签名、文件头注释与断言；正文引用的每一个 IR 片段都由
`code/02-ir-snippet-extract/extract_ir_snippets.py` 原样抽出。**这不是运行验证。**

---

## 一、pass 顺序与框架（`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S1 | `VERSION` | 版本事实 = `0.1.15` | 已核验（2026-10-01） |
| S2 | `tilelang/cuda/pipeline.py:68`、`:168`、`:284` | `CUDAPassPipelineBodyPrologue` / `CUDAPassPipelineBody` / `CUDA_PIPELINE` | 已核验（2026-10-01）：逐行读，`grep -n "mod = "` 得 73 次 pass 调用 |
| S3 | `tilelang/ascend/pipeline.py:22`、`:36`、`:190` | `allow_autoschedule` / `AscendPassPipelineBody` / `ascend_pipeline` | 已核验（2026-10-01）：`grep -n "mod = "` 得 80 次 pass 调用 |
| S4 | `tilelang/backend/pass_pipeline/pipeline.py:11`、`:22` | `PassPipeline` 类与 `lower()`（instrumentation 作用域） | 已核验（2026-10-01） |
| S5 | `tilelang/backend/pass_pipeline/pipeline_utils.py:25`、`:46`、`:58`、`:71`、`:105` | `allow_vectorize` / `should_force_let_inline` / `should_enable_race_check` / `should_disable_shared_memory_reuse` / `LayoutVisual` | 已核验（2026-10-01） |
| S6 | `tilelang/cuda/backend.py:128`、`tilelang/ascend/backend.py:52-63` | 两侧 `BackendModule` 注册与 `pipelines={"cuda": ...}` / `{"ascend": ...}` | 已核验（2026-10-01） |
| S7 | `tilelang/transform/__init__.py:19`、`:30`、`:41`、`:52`、`:108`、`:190`、`:204`、`:285`、`:311`、`:334`、`:382`、`:408` | pass 的 Python 包装入口行号（PipelinePlanning / LayoutInference / LowerTileOp / InjectSoftwarePipeline / ThreadSync / MakePackedAPI / MaterializeKernelLaunch / SplitHostDevice / VectorizeLoop / FlattenBuffer / StorageRewrite / LowerDeviceKernelLaunch） | 已核验（2026-10-01）：`grep -n "^def ..."` |
| S8 | `tilelang/engine/lower.py:136`、`:138-139` | `context.lower(mod)` 唯一分岔点；`Filter` 分离 host/device | 已核验（2026-10-01） |

## 二、通用承重 pass 的 C++ 本体

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S9 | `src/transform/pipeline_planning.cc:447`、`:1002`、`:1356` | `PipelineStageAnalyzer` / `PipelinePlanner` / `Pass PipelinePlanning()` | 已核验（2026-10-01） |
| S10 | `src/transform/pipeline_planning.cc:663`、`:753`、`:1358` | 两处 `LOG(FATAL)` 的原文；`tirx.use_async_copy` 默认值 | 已核验（2026-10-01）：逐行读原文 |
| S11 | `src/transform/inject_pipeline.cc:1017`、`:1179`、`:1205`、`:1340`、`:4033` | `PipelineBodyRewriter` / `PipelineRewriter` / `BuildPipeline()` / 循环标量绑定断言 / `Pass InjectSoftwarePipeline()` | 已核验（2026-10-01） |
| S12 | `src/transform/layout_inference/layout_inference.cc:237`、`:243`、`:279`、`:290`、`:1551`、`:1553`、`:1647` | `BufferUseDefCollector` / `RunInferStep` / `op->InferLayout` / 求解失败 `LOG(FATAL)` / `LayoutInferencer` / `Pass LayoutInference()` | 已核验（2026-10-01）：逐行读 `RunInferStep` 与 `VisitStmt_` |
| S13 | `src/transform/lower_tile_op.cc:43`、`:170`、`:207`、`:209`、`:224`、`:233-235`、`:1477` | `makeBufferWithLayout` / padding 断言 / `LowerTileOpPass` / `Substitute` / target 断言 / `kHasTMA` 记录 / `Pass LowerTileOp()` | 已核验（2026-10-01） |
| S14 | `src/transform/storage_rewrite.cc:100`、`:430`、`:456`、`:1928`、`:1931`、`:1932`、`:1933-1935` | `LinearAccessPatternFinder` / `StoragePlanRewriter` / `VisitBufferAccess` / `PassStorageRewrite()` / `kStorageRewriteDetectInplace` / `enable_reuse=false` / “Always disable reuse ...”注释 | 已核验（2026-10-01）：`sed -n '1932,1940p'` 原文核对 |
| S15 | `src/transform/thread_storage_sync.cc:212`、`:327`、`:603`、`:1068`、`:1085`、`:2091` | `ThreadSyncInserter` / warp-multiple 断言 / `TileLangThreadSyncPlanner` / warp-uniform 断言 / hoist 警告 / `Pass ThreadSync()` | 已核验（2026-10-01） |
| S16 | `src/transform/split_host_device.cc:114`、`:583`、`:617-620`、`:631` | `ForceSplit` 定义 / `::tvm::tl::SplitHostDevice` / ForceSplit 调用点 / `Pass SplitHostDevice()` | 已核验（2026-10-01） |
| S17 | `src/transform/make_packed_api.cc:38`、`:317`、`:386`、`:401`、`:1072` | `kTileLangOutIdx` / `RequiresPackedAPI` / `MakePackedAPI(PrimFunc,...)` / target 断言 / `Pass MakePackedAPI()` | 已核验（2026-10-01） |
| S18 | `src/transform/common/attr.h:17`、`:24` | `DeviceMainBlockName = "tilelang_root"` 与主块判定辅助函数 | 已核验（2026-10-01） |
| S19 | `src/transform/materialize_kernel_launch.cc`（文件头注释，约 :1-52） | T.Kernel 物化**之前**的 launch nest IR 草图（3.3 节引用） | 已核验（2026-10-01）：由 `extract_ir_snippets.py --only header` 抽出 |
| S20 | `src/cuda/op/builtin.h:32`、`src/op/builtin.h:60` | `kHasTMA = "tl.has_tma"`、`kSmemAlignmentMap = "tl.smem_alignment_map"` | 已核验（2026-10-01） |

## 三、昇腾特有承重 pass 的 C++ 本体

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S21 | `src/ascend/transform/auto_schedule.cc:158`、`:449`、`:452`、`:474`、`:680`、`:740`、`:1371` | `ScheduleBuilder` / 未知节点 `LOG(FATAL)` / `Z3SchedulePython` / FFI `z3_schedule_python` / `Z3SchedulePythonLoop` / FFI `z3_schedule_loop_python` / `Pass AutoSchedule()` | 已核验（2026-10-01）：逐行读 `Z3SchedulePython*` 与其回退分支 |
| S22 | `tilelang/ascend/transform/z3_scheduler.py:67`、`:228`、`:268`、`:784` | `z3_schedule_python` 实现与注册、`z3_schedule_loop_python` 实现与注册 | 已核验（2026-10-01）：`grep -n "register_global_func"` |
| S23 | `src/ascend/transform/insert_sync.cc:19-20`、`:132`、`:155`、`:2551` | 文件头“Synchronization analysis, flag allocation, and insertion”/ 两处同步锚点断言 / `Pass InsertSync()` | 已核验（2026-10-01） |
| S24 | `src/ascend/transform/materialize_multi_buffer.cc:19-20`、`:123`、`:538`、`:544` | 文件头“Lower prepared logical multi-buffer clocks ... after synchronization insertion”/ pitch 断言 / `tilelang_root` 保持断言 / `Pass MaterializeMultiBuffer()` | 已核验（2026-10-01） |
| S25 | `src/ascend/transform/insert_nd2nz.cc`（文件头） | “Rewrite layout-driven Ascend ND-to-NZ copies before AutoSchedule”；pass 真名 `InsertNd2Nz` | 已核验（2026-10-01）：`--grep InsertNd2Nz` 命中本文件 |
| S26 | `src/ascend/transform/layout_inference.cc:1684`、`:1686` | 昇腾 fork 的 `LayoutInferencer` 与 `Substitute`（`AscendLayoutInference`） | 已核验（2026-10-01） |
| S27 | `src/ascend/transform/lower_tile_op.cc:200`、`:202` | 昇腾 fork 的 `LowerTileOpPass`（`AscendLowerTileOp`） | 已核验（2026-10-01） |
| S28 | `tilelang/ascend/transform/__init__.py` | 昇腾 pass 的 Python 包装与 docstring（含 `RewriteFlagToBuf` 的 8 槽/32 槽说明、`AscendInsertOOBPadding` 的调度顺序说明） | 已核验（2026-10-01）：全文阅读 |

## 四、IR 片段的一手来源

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S29 | `examples/eager_jit/eagerjit.en.ipynb`（cell 49、cell 57 的输出） | 3.1 节的最小 kernel IR dump（`T.match_buffer` / `T.launch_thread` / `T.block("tilelang_root")` / `scope="local.var"`） | 已核验（2026-10-01）：由 `extract_ir_snippets.py --only notebook` 原样抽出 |
| S30 | `testing/python/transform/test_tilelang_transform_pipeline_planning.py`（`before` / `after`） | 3.4 节 `PipelinePlanning` 的输入/输出规格 | 已核验（2026-10-01） |
| S31 | `testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py`、`test_tilelang_transform_thread_sync.py`、`test_tilelang_transform_lower_tile_op.py`、`test_tilelang_transform_materialize_kernel_launch.py`、`test_tilelang_transform_split_host_device.py` | 各 pass 的行为断言与 before/after（正文 4.3 的“输入/输出差异”依据） | 已核验（2026-10-01） |
| S32 | `docs/compiler_internals/inject_fence_proxy.md:48`、`:78` | `InjectFenceProxy` 的 before/after IR 对（E8 与 4.1 ⑦ 依据） | 已核验（2026-10-01）：由 `extract_ir_snippets.py --only fence` 原样抽出 |
| S33 | `docs/deeplearning_operators/elementwise.md` | 逐元素 kernel 的生成 CUDA 代码（向量化前后对照） | 已核验（2026-10-01）：只作旁证，未在正文引用具体行号 |
| S34 | `docs/tools/lower_trace.md` | 逐 pass 观察工具的官方说明（5.1 节依据） | 已核验（2026-10-01） |
| S35 | `tilelang/tools/lower_trace/core.py:663` | `lower_trace.enable()`；配合 `tilelang/tools/lower_trace/__init__.py` 的一次性 API 使用 | 已核验（2026-10-01） |

## 五、昇腾适配器仓库（`_vendor/tilelang-ascend-v0.1.2.000/`）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| A1 | `tilelang/engine/phase.py` | 适配器路线的 pass 序列（与进主干的昇腾 pipeline 词汇不同） | 已核验（2026-10-01）：全文阅读；`code/04-version-diff/` 做差集 |
| A2 | `src/transform/` 目录清单（`ascend_storage_rewrite.cc`、`ascend_sync_insert.cc`、`ascend_sync_insert_vs.cc`、`ascend_memory_planning.cc`、`ascend_lower_parallel_to_vector.cc`、`cross_core_pipeline.cc`、`ascend_combinecv.cc` 等） | “适配器仓库用的是另一套昇腾 pass 词汇”这一结论的直接证据 | 已核验（2026-10-01）：`ls src/transform/` |
| A3 | `VERSION` | 仓库内写 `0.1.4`，**以 tag `TileLang-ascend-v0.1.2.000-release` 为准** | 已核验（2026-10-01）：与 [../../VERSIONS.md](../../VERSIONS.md) 一致 |

## 六、corpus 资料引用

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| R1 | `../tilelang-corpus/90-synthesis/02-compiler-internals-map.md` | 编译期内部地图（**二手导航材料**）；顶部有“本文写 `InsertNd`，v0.1.15 是 `InsertNd2Nz`，核对以 `_vendor/` 为准”的版本提示 | 已核验（2026-10-01）：全文阅读；其版本提示与 `--grep` 结果一致 |
| R2 | `../tilelang-corpus/90-synthesis/03-ascend-route.md` | 昇腾路线（fractal / ND→NZ / 边界填充）的编者综合 | 已核验（2026-10-01）：仅作背景，正文未据此下源码结论 |
| R3 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/lower_trace.md`、`.../pass_diff.md` | 官方工具文档（5.1 节的可复现流程） | 已核验（2026-10-01） |

---

## 待补资料

本章目前没有阻塞性缺口，但有两处值得在后续版本核对时补强：

1. **缺少 GPU 机器上的 lower_trace 实测快照。** 本机无卡，`code/03-lower-trace-walkthrough/`
   只有静态核对版本。若后续在有卡机器上跑通，应把
   `tmp/lower_trace_dir/` 下的 `.tir` 快照存成回归基线，并在此处登记路径与 commit。
2. **缺少更早 release（如 v0.1.13 / v0.1.14）的 checkout，用于确认“哪些 pass 是
   v0.1.15 才新增的”。** 当前 `_vendor/` 只有 v0.1.15 与适配器 v0.1.2.000，无法对
   “相对上一个小版本新增了什么”做直接 diff。建议由后续版本补充一个更早 tag 的
   checkout，或在 corpus 里登记一份 pass 名单的时间序列。**在此之前，正文不写任何
   “v0.1.15 才新增”的具体断言，只写“在 v0.1.15 里存在”并可 grep 复核。**

已刻意**未**采用的来源（避免与锁定版本冲突）：

* corpus 里 `02-tilelang-core-repo/raw/` 的 **main 快照**。它不是 tag
  `v0.1.15`，只在需要确认“某个说法是否是文档漂移”时作为旁证，正文对它的使用均已
  显式标注。
* 任何未在本 corpus 内、也未现场核验可达的上游 URL。
