# 第 04 章 参考资料

本章所有结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不使用任何模糊的版本措辞；凡涉及版本，一律写出具体 tag 或 commit。

写作环境：macOS（Apple Silicon）、Python 3.11；**没有 NVIDIA GPU，没有昇腾 NPU**，
也没有安装 `tilelang` / `torch`。因此：

* 标题为“已核验”的源码引用，指的是**在锁定版本工作树上用 `grep -n` / `sed -n` /
  `cat -n` 定位并逐行阅读**，不是运行验证；
* 本章唯一涉及真实硬件的示例是 `code/quickstart-gemm/`，其验证级别为
  **仅静态核对**（`python3 -m py_compile` 通过 + API 名与签名逐个比对）。

---

## 一、本章自带的离线工具（可复现全部数字）

| id | 位置 | 用途 | 状态 |
| --- | --- | --- | --- |
| T1 | `code/pass-order/dump_pass_order.py` | 从 `tilelang/cuda/pipeline.py` 与 `tilelang/ascend/pipeline.py` 逐行提取真实 pass 顺序与条件守卫；生成正文 4.5 / 4.6 / 4.7 节的数字 | **在本机运行过（2026-10-01）**：输出 CUDA 73 步、Ascend 80 步、同名 39 个、条件 CUDA 7 / Ascend 17 |
| T2 | `code/repo-map/inventory.py` | 统计模块清单与规模，生成正文 2.3 节表格与图 1 的数字 | **在本机运行过（2026-10-01）** |
| T3 | `code/compile-path-check/check_compile_path.py` | 逐跳校验图 2 与 4.1–4.4 节声称的 30 个“文件:行号” | **在本机运行过（2026-10-01）**：`30/30 verified`，退出码 0 |
| T4 | `code/quickstart-gemm/quickstart_gemm.py` | 用官方 GEMM+ReLU 示例走一遍完整编译流程（kernel 主体取自 `examples/quickstart.py:8-48`） | **仅静态核对**：`py_compile` 通过；本机无 GPU、无 torch/tilelang，**未运行** |
| F1 | `figures/fig-01-repo-map.md` | 仓库地图（Mermaid + ASCII）与数量口径 | 自绘；数据由 T2 生成 |
| F2 | `figures/fig-02-compile-flow.md` | 一次编译的全流程图 | 自绘；每跳行号由 T3 校验 |
| F3 | `figures/fig-03-backend-abstraction.md` | 多后端抽象对象图 | 自绘 |
| F4 | `figures/fig-04-pipeline-compare.md` | 两条 pipeline 的阶段对照 | 自绘；阶段划分由 T1 输出归纳 |

所有图都是本章作者自绘的 Mermaid / ASCII 文本，没有从外部复制图片，因此不需要
`figures/SOURCES.md`。

---

## 二、锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

> 表中“行号”均已用 `sed -n '<行号>p' <文件>` 或 `cat -n` 现场核对。
> “已核验（2026-10-01）”= 定位到该行并阅读了上下文，**不代表运行过**。

### 2.1 仓库结构与模块职责

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/VERSION` | 版本号 `0.1.15`；`git describe --tags` → `v0.1.15`；`git log -1` → `a35f8dd` | 已核验（2026-10-01） |
| S2 | `_vendor/tilelang-v0.1.15/tilelang/__init__.py:211-230` | import 时加载 `analysis/transform/language/engine/tools/autotuner` 与五个后端包（`cpu/ascend/cuda/rocm/metal/webgpu`） | 已核验（2026-10-01） |
| S3 | `_vendor/tilelang-v0.1.15/tilelang/language/__init__.py:1-32` | `tilelang.language` 复合 CUDA 方言；`__tilelang_dialect__ = "cuda"`（:32） | 已核验（2026-10-01） |
| S4 | `_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:431-455` | target-neutral `T.Kernel`（“a grid of tile programs”） | 已核验（2026-10-01） |
| S5 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/kernel.py:13-55` | CUDA 方言 `T.Kernel(..., threads, prelude, cluster_dims)` | 已核验（2026-10-01） |
| S6 | `_vendor/tilelang-v0.1.15/tilelang/engine/lower.py:26-44, :65-106, :109-141, :144-213, :216-244` | engine 的四个入口与 host/device 判定 | 已核验（2026-10-01） |
| S7 | `_vendor/tilelang-v0.1.15/tilelang/engine/semantic_check.py:9-31` | `PreLowerSemanticCheck` 与它的开关、四个检查器 | 已核验（2026-10-01） |
| S8 | `_vendor/tilelang-v0.1.15/tilelang/engine/param.py:153-166` | `CompiledArtifact` 字段定义 | 已核验（2026-10-01） |
| S9 | `_vendor/tilelang-v0.1.15/tilelang/transform/__init__.py:19-471` | 通用 pass 的 Python 包装（`PipelinePlanning:19`、`LayoutInference:30`、`LowerTileOp:41`、`LowerDeviceKernelLaunch:408`、`MakePackedAPI:190`、`MaterializeKernelLaunch:204`、`MergeSharedMemoryAllocations:345`、`SplitHostDevice:285`、`ThreadSync:108`、`UnrollLoop:471` 等） | 已核验（2026-10-01） |
| S10 | `_vendor/tilelang-v0.1.15/src/transform/` | pass 的 C++ 本体；42 个 `.cc`、共 66 个文件、含 `layout_inference/` 子目录 | 已核验（2026-10-01）：`ls src/transform/*.cc \| wc -l` → 42 |
| S11 | `_vendor/tilelang-v0.1.15/tilelang/layout/__init__.py:1-36` | `Layout` / `Fragment` / `PartialFragment` / swizzle 家族 / 昇腾 NZ 家族 | 已核验（2026-10-01） |
| S12 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/__init__.py:1-6` | 五个 adapter 类 | 已核验（2026-10-01） |
| S13 | `_vendor/tilelang-v0.1.15/tilelang/cache/__init__.py:24-35, :38-72, :75-98` | 七条缓存派发路径（5 通用 + 2 昇腾）；按 module 名而非 target kind 派发（:68-69） | 已核验（2026-10-01） |
| S14 | `_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py:35-58` | `KernelCache` 的缓存文件布局（`device_kernel.cu` / `host_kernel.cu` / `kernel_lib.so` / `params.json`） | 已核验（2026-10-01） |
| S15 | `_vendor/tilelang-v0.1.15/tilelang/autotuner/__init__.py:1-8` | `autotune` / `AutoTuner` / `set_autotune_inputs` 等导出 | 已核验（2026-10-01） |
| S16 | `_vendor/tilelang-v0.1.15/tilelang/carver/README.md:1-10` | Carver 的定位：tile 配置推荐，可对接 TVM / Triton / TileLang | 已核验（2026-10-01） |
| S17 | `_vendor/tilelang-v0.1.15/tilelang/tools/__init__.py:1-20` | 工具包导出与 `lower_trace` 的惰性加载 | 已核验（2026-10-01） |
| S18 | `_vendor/tilelang-v0.1.15/examples/quickstart.py:8-48, :59, :79, :82-87` | 官方 GEMM+ReLU 示例、`compile`、`get_kernel_source`、`get_profiler` | 已核验（2026-10-01）；**未运行** |
| S19 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:1-60` | Ascend 950 后端的定位、安装前提（CANN / bisheng / torch_npu）、Quick Start | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S20 | `_vendor/tilelang-v0.1.15/tilelang/ascend/target.py:57-59` | 注册 `ascend` / `asc` 的 detector 与 normalizer | 已核验（2026-10-01） |

### 2.2 多后端抽象

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S21 | `_vendor/tilelang-v0.1.15/tilelang/backend/README.md:14-27, :56-78, :196-220, :240-242, :244-282, :284-293, :325-345, :374-392` | Design Model、术语（target backend vs execution backend）、Pass Pipeline、codegen 分派禁令、execution backend 契约、已注册后端表、包布局、Ownership Rules | 已核验（2026-10-01） |
| S22 | `_vendor/tilelang-v0.1.15/tilelang/backend/pass_pipeline/pipeline.py:11-39` | `PassPipeline` 的全部实现（含 instrumentation 与 `enrich_error`） | 已核验（2026-10-01） |
| S23 | `_vendor/tilelang-v0.1.15/tilelang/backend/pass_pipeline/pipeline_utils.py:12-109` | `allow_vectorize` / `should_force_let_inline` / `should_enable_race_check` / `should_disable_shared_memory_reuse` / `LayoutVisual` | 已核验（2026-10-01） |
| S24 | `_vendor/tilelang-v0.1.15/tilelang/backend/module.py:28-99, :188-231, :234-272, :291-307, :310-335` | `BackendModule` 校验、`BackendContext`、注册表、按 target kind 解析、上下文工厂 | 已核验（2026-10-01） |
| S25 | `_vendor/tilelang-v0.1.15/tilelang/backend/device_codegen.py:14-44` | `DeviceCodegen` 的 `build` / `build_without_compile` 两个入口 | 已核验（2026-10-01） |
| S26 | `_vendor/tilelang-v0.1.15/tilelang/backend/host_codegen.py:33-59` | `HostCodegen` / `HostCodegenHook` / `STANDARD_HOST_CODEGENS`（只有 `c` 与 `llvm`） | 已核验（2026-10-01） |
| S27 | `_vendor/tilelang-v0.1.15/tilelang/backend/execution_backend.py:14-32` | `ExecutionBackendSpec` 字段（含 `supports_callee_allocated_outputs`） | 已核验（2026-10-01） |
| S28 | `_vendor/tilelang-v0.1.15/tilelang/backend/target.py:28-134` | target detector / normalizer 注册（`:32`、`:45`）与 `determine_target`（`:122`） | 已核验（2026-10-01） |
| S29 | `_vendor/tilelang-v0.1.15/tilelang/cuda/backend.py:59-120, :123-143` | CUDA manifest 与 `tilelang_callback_cuda_compile`（把 arch / code / options 塞进二进制缓存 key） | 已核验（2026-10-01） |
| S30 | `_vendor/tilelang-v0.1.15/tilelang/cuda/cutedsl_backend.py:8-23` | CuTeDSL 单独 manifest 却显式复用 CUDA pipeline | 已核验（2026-10-01） |
| S31 | `_vendor/tilelang-v0.1.15/tilelang/cuda/codegen.py:8-19` | `is_plain_cuda_target` / `is_cutedsl_target` 与两个 device codegen 入口 | 已核验（2026-10-01） |
| S32 | `_vendor/tilelang-v0.1.15/tilelang/cuda/execution_backend.py:24-37` | CUDA 的 `tvm_ffi → nvrtc → cython` 优先级；CuTeDSL 单独一条 | 已核验（2026-10-01） |
| S33 | `_vendor/tilelang-v0.1.15/tilelang/ascend/backend.py:15-63` | Ascend manifest 与 `tilelang_callback_ascend_compile`（bisheng 编译 + 二进制缓存） | 已核验（2026-10-01） |
| S34 | `_vendor/tilelang-v0.1.15/tilelang/ascend/codegen.py:5-9` | Ascend 的两个 native codegen 入口 | 已核验（2026-10-01） |
| S35 | `_vendor/tilelang-v0.1.15/tilelang/ascend/execution_backend.py:5-21` | Ascend 的 `tvm_ffi → cython` 与 `supports_callee_allocated_outputs=True` | 已核验（2026-10-01） |

### 2.3 两条 pipeline（正文表格的一手来源）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S36 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:68-165` | `CUDAPassPipelineBodyPrologue`（33 步）、warp specialization 门控、mbarrier 前置检查（:186-195 的报错在 Body 中） | 已核验（2026-10-01）；顺序由 T1 提取 |
| S37 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:168-281` | `CUDAPassPipelineBody`（调 Prologue + 40 步） | 已核验（2026-10-01）；顺序由 T1 提取 |
| S38 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:284` | `CUDA_PIPELINE = PassPipeline("cuda", CUDAPassPipelineBody)` | 已核验（2026-10-01） |
| S39 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:27-46` | `allow_warp_specialized` 与 `module_has_tma` 两个门控函数 | 已核验（2026-10-01） |
| S40 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:22-33, :36-51` | `allow_autoschedule` 与昇腾版 `MaterializeKernelLaunch` 参数 | 已核验（2026-10-01） |
| S41 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:52-187` | `AscendPassPipelineBody`（80 步，含 13 步自动调度段与 `not auto_schedule_enabled` 分支） | 已核验（2026-10-01）；顺序由 T1 提取 |
| S42 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:190` | `ascend_pipeline = PassPipeline("ascend", AscendPassPipelineBody)` | 已核验（2026-10-01） |
| S43 | `_vendor/tilelang-v0.1.15/tilelang/cuda/transform/__init__.py:1-196` | 16 个 CUDA 专属 pass 的 Python 前端 | 已核验（2026-10-01） |
| S44 | `_vendor/tilelang-v0.1.15/tilelang/ascend/transform/__init__.py:1-256` | 31 个昇腾专属 pass 的 Python 前端（含 `z3_scheduler.py` 同目录） | 已核验（2026-10-01） |

### 2.4 JIT、adapter 与 host wrapper

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S45 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:94-173, :269-415, :417-497, :519-564, :588-652` | `compile()`、`JITImpl`、`get_tir` / `_infer_jit_mode`、`JITImpl.compile`、`__call__`、`jit()` 装饰器 | 已核验（2026-10-01） |
| S46 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:283-313` | lazy / eager 两种模式的官方说明 | 已核验（2026-10-01） |
| S47 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:41-149, :206-222, :224-286, :288-322, :324-424` | `JITKernel` 构造、`__call__`、`_compile_and_create_adapter`、`_compile_artifact`（含 `TL_ENABLE_DUMP_IR` 分支）、adapter 选择 | 已核验（2026-10-01） |
| S48 | `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:1-70` | host wrapper 源码模板（含 shared memory 属性设置与错误缓冲） | 已核验（2026-10-01） |
| S49 | `_vendor/tilelang-v0.1.15/src/transform/make_packed_api.cc:1-42` | `MakePackedAPI` 的 C++ 本体与 `tilelang_callee_allocated_outputs` 等属性名 | 已核验（2026-10-01） |
| S50 | `_vendor/tilelang-v0.1.15/src/transform/lower_device_kernel_launch.cc:26-31` | `LaunchGridOnly(target)`：grid-only 目标不把线程域当 launch 维 | 已核验（2026-10-01） |
| S51 | `_vendor/tilelang-v0.1.15/tilelang/engine/param.py:26-68` | `KernelParam.from_buffer` / `from_var`：参数元数据的来源 | 已核验（2026-10-01） |
| S52 | `_vendor/tilelang-v0.1.15/tilelang/language/eager/builder.py:1760-1800` | eager 前端的 `prim_func`（内部 `mutate(func)` 改写 AST） | 已核验（2026-10-01） |
| S53 | `_vendor/tilelang-v0.1.15/tilelang/language/parser/entry.py:37` | parser 前端的 `prim_func` | 已核验（2026-10-01） |
| S58 | `_vendor/tilelang-v0.1.15/tilelang/env.py:368` | `TILELANG_CACHE_DIR`（默认 `~/.tilelang/cache`）；正文 5.2 节陷阱 3 引用 | 已核验（2026-10-01） |
| S59 | `_vendor/tilelang-v0.1.15/tilelang/env.py:435-445` | `env.is_cache_enabled` / `enable_cache` / `disable_cache`（经 `tilelang/__init__.py:161-167` 导出为 `tilelang.disable_cache`） | 已核验（2026-10-01） |
| S60 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:532-543` | `JITKernel.get_kernel_source`（正文 3.3 与 E9 引用） | 已核验（2026-10-01） |
| S61 | `_vendor/tilelang-v0.1.15/tilelang/ascend/transform/__init__.py:9-11, :240` | 昇腾侧 pass 的正确名字是 **`InsertNd2Nz`**（不是 `InsertNd`）；正文 4.6 段 B 与 R10 的口径说明引用 | 已核验（2026-10-01） |
| S62 | `_vendor/tilelang-v0.1.15/src/transform/make_packed_api.cc:38-42, :342-355, :421-475` | 打包入口 `(args, num_args)`、`tilelang_out_idx` / `tilelang_callee_allocated_outputs` 两个属性、参数断言；正文 4.8 节引用 | 已核验（2026-10-01） |
| S63 | `_vendor/tilelang-v0.1.15/src/cuda/transform/lower_shared_tmem.cc:3` | `LowerSharedTmem` 的职责（`shared.tmem` → plain shared + PTX init）；正文 4.9 节引用 | 已核验（2026-10-01） |
| S64 | `_vendor/tilelang-v0.1.15/README.md:113-135` | 官方后端支持表：CUDA 为 Primary、Ascend 950 为 Supported、“TMA, WGMMA, and TMEM features require the corresponding GPU architecture”（:123）；正文 4.9 节与 references 4.3 节引用 | 已核验（2026-10-01） |

### 2.5 版本边界（v0.1.15 vs main）

| id | 引用位置 / 命令 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S54 | `ls _vendor/tilelang-v0.1.15/tilelang/jit/adapter/` | v0.1.15 **没有** `pto/` 目录 | 已核验（2026-10-01）：输出为 `base.py cutedsl cython kernel_cache.py libgen.py nvrtc torch tvm_ffi.py utils.py wrapper.py` |
| S55 | `grep -rn "AutoSimtVF" _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src` | v0.1.15 没有 `AutoSimtVF` | 已核验（2026-10-01）：无输出 |
| S56 | `grep -rn "PersistentKernel\|AutoPersistent" _vendor/tilelang-v0.1.15/tilelang _vendor/tilelang-v0.1.15/src` | v0.1.15 没有 `T.PersistentKernel` / `AutoPersistent` | 已核验（2026-10-01）：无输出 |
| S57 | `_vendor/tilelang-v0.1.15/.gitmodules` | 依赖 `3rdparty/tvm`（fork）与 `3rdparty/cutlass` | 已核验（2026-10-01） |

---

## 三、corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/overview.md:32-50` | https://tilelang.com/get_started/overview.html | 官方给出的六步 Compilation Flow（Tile Program → Tile Program with Tile Library → … → Hardware-Specific Executable/Runtime） | 已核验（2026-10-01）：本地为站点源文件 `_sources`；链接可达 |
| R2 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/targets.md:10-22` | https://tilelang.com/get_started/targets.html | 常见 target 列表与 `auto` 探测顺序（CUDA → HIP → Metal） | 已核验（2026-10-01） |
| R3 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/targets.md:37-51` | https://tilelang.com/get_started/targets.html | target 的四种输入形式（字符串 / dict / `Target` / `auto`） | 已核验（2026-10-01） |
| R4 | `../tilelang-corpus/02-tilelang-core-repo/raw/PROVENANCE.md` | https://github.com/tile-ai/tilelang | **main 快照的 commit 是 `994b44eca1a8`（2026-09-30）**，与 v0.1.15（`a35f8dd`）不是同一个提交；本表用于版本对照，不作为 v0.1.15 的证据 | 已核验（2026-10-01）：`git log -1` → `994b44e [Ascend] Add PTO backend for Ascend950 (#3310)` |
| R5 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3308.md` | https://github.com/tile-ai/tilelang/pull/3308 | “[Public Release 9/30] Introduce Ascend 950 backend”，昇腾后端进入主线的那个 PR | 已核验（2026-10-01）：本地为抓取的 PR 页面正文；**PR 标题仅作线索，判据以 S54–S56 的 grep 为准** |
| R6 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3310.md` | https://github.com/tile-ai/tilelang/pull/3310 | “[Ascend] Add PTO backend for Ascend950”，正文 5.3 节用它解释“main 有、v0.1.15 没有”的 PTO 路径 | 已核验（2026-10-01）：同上 |
| R7 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3320.md` | https://github.com/tile-ai/tilelang/pull/3320 | `AutoSimtVF` pass（未合入） | 已核验（2026-10-01）：本地文件存在；**未合入状态以 S55 的 grep 为准** |
| R8 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3325.md` | https://github.com/tile-ai/tilelang/pull/3325 | `T.PersistentKernel` + `AutoPersistent`（未合入） | 已核验（2026-10-01）：同上；**判据见 S56** |
| R9 | `../tilelang-corpus/08-frontier-unmerged/raw/github-issue-2115.md` | https://github.com/tile-ai/tilelang/issues/2115 | “[RoadMap] Multi-Backend Refactoring/Decoupling”，本章 4.4 节所讲抽象的设计动机 | 已核验（2026-10-01）：本地为抓取的 issue 正文；本章只引用其结构性描述 |
| R10 | `../tilelang-corpus/90-synthesis/02-compiler-internals-map.md:7-35, :94-118` | （编者综合稿，非上游） | 与本章 4.1–4.6 节互为对照：同一份“编译参与者”与“昇腾调度段”的另一版叙述 | **部分核验**：这是 corpus 编者的综合文档，**属于二手材料**；本章只把它当对照，所有行号仍以 `_vendor/` 为准。另注：该文档正文所依据的是 corpus 的 main 快照（`994b44e`），其中 `InsertNd` 的表述与 v0.1.15 里实际存在的 `InsertNd2Nz` 略有出入，本章采用 `InsertNd2Nz` |
| R11 | `../tilelang-corpus/90-synthesis/03-ascend-route.md:1-14, :48-65, :97-111` | （编者综合稿，非上游） | 昇腾两条路线（AscendC/PTO 与 AscendNPU IR）的定位，以及未合入前沿清单 | **部分核验**：同样属于编者综合材料；本章 5.3 节的结论已用 S54–S56 独立复核 |

---

## 四、口径与待补

### 4.1 已登记的口径差异

| 项 | `VERSIONS.md` 的说法 | 本章实测 | 处理方式 |
| --- | --- | --- | --- |
| `examples/` 规模 | “有 57 个目录” | 54 个目录 + 3 个顶层文件（`conftest.py` / `pytest.ini` / `quickstart.py`），共 57 项 | 本章采用实测口径，并在正文 2.3 节与图 1 中说明；建议后续版本把 `VERSIONS.md` 的措辞改为“57 项（54 目录 + 3 文件）” |

### 4.2 尚需在真实硬件上验证的事项（本章不宣称）

本章没有任何运行期结论，但下面几条只有在真实设备上才能闭环，留给第 12、20–25 章：

1. `code/quickstart-gemm/` 实际编译出的 CUDA C 形态（是否走 TMA / cp.async）、
   `__shared__` 大小与 `MergeSharedMemoryAllocations` 的关系。
2. Ascend 侧 13 步自动调度在真实 NPU 上解出的多缓冲份数与同步指令形态。
3. `unsupported_annotations=["cluster_dims"]` 在昇腾目标上的具体报错信息（本章只说明该
   注解不会参与物化）。

### 4.3 待补资料

| 缺什么 | 为什么需要 | 建议来源 |
| --- | --- | --- |
| TileLang 官方对“backend 抽象”的设计文档（RFC 级别） | 本章 4.4 节目前只能引用 `tilelang/backend/README.md` 与 issue #2115；缺少一份跨版本的正式设计说明，难判断哪些字段是稳定契约 | `tile-ai/tilelang` 的 `docs/developer_guide/` 或 RFC 目录 |
| v0.1.15 官方 release note 的完整文本 | 本章用 `git log -1` 确认了 tag 的 commit，但缺少 release note 来核对“本版新增/移除”的官方表述 | https://github.com/tile-ai/tilelang/releases/tag/v0.1.15 |
| 昇腾 950 后端在 v0.1.15 中的官方支持等级说明 | `_vendor/tilelang-v0.1.15/README.md:113-135` 的表格标注 Ascend 950 为 `Supported`，但缺少与 CI 覆盖范围对应的说明 | `tile-ai/tilelang` 的 CI 配置与 `tilelang/ascend/README.md` 更新 |
