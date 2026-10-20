# 第 25 章 参考资料

本章所有结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`，`VERSION` = `0.1.15`）与
TileLang-Ascend `TileLang-ascend-v0.1.2.000-release`
（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`，仓库内 `VERSION` 写 `0.1.4`）。**

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，没有昇腾 NPU，
没有安装 CANN / bisheng，也没有安装 tilelang/tvm**。因此本章代码示例的最高验证
级别是“仅静态核对”；`code/` 下的五个脚本是“CPU 逻辑验证”（见各目录 README）。
**本章不包含任何自测性能数字。**

核验日期均为 2026-10-01。核验方式：在锁定版本的工作树上用 `rg -n` / `sed -n` 定位、
阅读完整签名与注释；正文每个 `path:line` 都由
`code/04-cite-check/check_citations.py` 重新解析校验；两棵树的 API 差异由
`code/03-api-presence-matrix/api_matrix.py` 现场生成。**这不是运行验证。**

---

## 一、版本与总则

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| V1 | `../../VERSIONS.md` | 版本锁定表：两个 commit、日期、路径 | 已核验（2026-10-01） |
| V2 | `_vendor/tilelang-v0.1.15/VERSION` | 主干版本 = `0.1.15` | 已核验（2026-10-01） |
| V3 | `_vendor/tilelang-ascend-v0.1.2.000/VERSION` | 适配器仓库内版本号 = `0.1.4`（以 tag `TileLang-ascend-v0.1.2.000-release` 为准） | 已核验（2026-10-01） |
| V4 | `_vendor/tilelang-v0.1.14/`（对照） | 只用于回答“某能力是不是 v0.1.15 新增”；本章未做 v0.1.14 对照，正文不含相关结论 | 未使用（本章不需要） |

---

## 二、M 树（主干 `_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| R1 | `tilelang/ascend/target.py:40-58` | `normalize_ascend_target` / `normalize_asc_target`；`"ascend"` 与 `"asc"` 归一化 | 已核验（2026-10-01）：逐行读 |
| R2 | `tilelang/ascend/backend.py:15-62` | `tilelang_callback_ascend_compile`（bisheng 编译 + aibin 缓存）与 `BackendModule` 注册 | 已核验（2026-10-01）：逐行读 |
| R3 | `tilelang/ascend/codegen.py:5-9` | FFI 名 `target.build.tilelang_ascend` / `..._without_compile` | 已核验（2026-10-01） |
| R4 | `tilelang/ascend/pipeline.py:19-190` | 昇腾 pass 管线全文：`MaterializeKernelLaunch` 参数、`RewriteDualCopy`、`AutoSchedule` 九步、`RewriteFlagToBuf`、`MergeUBAllocations(align_bytes=32)` | 已核验（2026-10-01）：全文阅读 |
| R5 | `tilelang/ascend/pipeline.py:22-33,103-116` | `allow_autoschedule` 与 `TL_ENABLE_AUTO_SCHEDULE`（默认 True） | 已核验（2026-10-01） |
| R6 | `tilelang/ascend/README.md:29-30,55-154,156-173` | 默认 `dav-3510`；GEMM+ReLU 快速开始；“You write / Ascend C you can skip”分工表 | 已核验（2026-10-01）：全文阅读 |
| R7 | `tilelang/ascend/language/__init__.py:8-124` | 方言导出清单；确认**没有** `tile` 别名 | 已核验（2026-10-01）：全文阅读 |
| R8 | `tilelang/ascend/language/allocate.py:15-70` | `alloc_shared`（UB）/`alloc_l1`/`alloc_l0a`/`alloc_l0b`/`alloc_l0c`（含 L0C 布局） | 已核验（2026-10-01）：逐行读 |
| R9 | `tilelang/ascend/language/allocate.py:73-128` | MX 尺度因子槽位 `alloc_l0a_sf` / `alloc_l0b_sf` | 已核验（2026-10-01） |
| R10 | `tilelang/ascend/language/copy_op.py:53-163` | `dual_copy` 契约：混合内核约束、M/N split、`dual_dst_ctl`、L0C dtype 拦截 | 已核验（2026-10-01）：逐行读 docstring 与校验 |
| R11 | `tilelang/ascend/language/copy_op.py:230-372` | `copy` 的 Ascend 参数归一化与 `tl.tileop.ascend_copy` 重拼 | 已核验（2026-10-01） |
| R12 | `tilelang/ascend/language/gemm_op.py:18-100,168` | L0 MAD 组装、静态 M/N/K 断言、`T.gemm` 签名 | 已核验（2026-10-01） |
| R13 | `tilelang/ascend/language/frame.py:29-31,50-63,81-196` | `Cube` / `Vector` / `SimtVF` / `SimdVF` 的区域定义与 `latency` 参数 | 已核验（2026-10-01）：逐行读 |
| R14 | `tilelang/ascend/language/kernel.py:150-228` | `T.Kernel(N)` 一维核网格、`T.MixedKernel(N, sids=2) as (bx, sid)` | 已核验（2026-10-01） |
| R15 | `tilelang/ascend/language/annotations.py:19-188` | `annotate_buffer_versions` / `annotate_manual_multi_buffer` / `annotate_unlimit_memory` | 已核验（2026-10-01）：逐行读 docstring |
| R16 | `tilelang/ascend/language/mode.py:8-105` | `set_mmad_direction` / `set_hf32_mode` / `set_atomic` | 已核验（2026-10-01） |
| R17 | `tilelang/ascend/analysis/vf_checker.py:1-60` | `VFChecker` 与 `SIMD_VF`/`SIMT_VF` 的分配归属 | 已核验（2026-10-01） |
| R18 | `tilelang/ascend/debug.py:35` | `device_assert`（M 树有、A 树无） | 已核验（2026-10-01）：`rg -n "def device_assert"` |
| R19 | `tilelang/transform/pass_config.py:76` | `TL_ENABLE_AUTO_SCHEDULE = "tl.enable_auto_schedule"` | 已核验（2026-10-01） |
| R20 | `src/ascend/codegen/codegen_ascend.cc:152-164` | `GetAscendScopeQualifier`：scope → `__ubuf__/__ca__/__cb__/__cc__/__cbuf__` | 已核验（2026-10-01）：逐行读 |
| R21 | `src/ascend/codegen/codegen_ascend.cc:262-292` | `GetPipePair`（合法 pipe 集合）与 `GetCrossCoreSyncScope` | 已核验（2026-10-01） |
| R22 | `src/ascend/codegen/codegen_ascend.cc:327-351` | 头文件包含与 `PrintFuncPrefix`（`__mix__(1, 2)` / `__vector__` / `__cube__`） | 已核验（2026-10-01）：逐行读 |
| R23 | `src/ascend/codegen/codegen_ascend.cc:408-434` | `ValidateSimdVectorType`：2048-bit 寄存器与 `boolx256` predicate | 已核验（2026-10-01） |
| R24 | `src/ascend/codegen/codegen_ascend.cc:1349-1381` | `EmitHardEventSync_`（`asc_sync_notify/wait`）与 `EmitCrossCoreSync_` | 已核验（2026-10-01） |
| R25 | `src/ascend/codegen/codegen_ascend.cc:1383-1609` | 七个 `Emit*Copy_`：`asc_copy_gm2ub_align` / `_ub2gm_align` / `_gm2l1_nd2nz|dn2nz` / `_l0c2ub` / `_l0c2gm` / `_l12l0a|b` / `_ub2l1` | 已核验（2026-10-01）：逐行读 |
| R26 | `src/ascend/codegen/codegen_ascend.cc:1652-1690` | `EmitMad_`（`asc_mmad` / `asc_mmad_mx`） | 已核验（2026-10-01） |
| R27 | `src/ascend/codegen/codegen_ascend.cc:3554-3557` | `PrintStorageSync` → `asc_syncthreads()` | 已核验（2026-10-01） |
| R28 | `src/ascend/op/oob_padding.h:32-44` | `enum class DMAPath`（7 条搬运 + 两条 MX SF） | 已核验（2026-10-01） |
| R29 | `src/ascend/op/oob_padding.cc:46-68` | `GetDMAPath`：scope 组合 → 通路 | 已核验（2026-10-01）：逐行读 |
| R30 | `src/ascend/op/utils.h:30-69` | `IsL1Buffer` / `IsL0ABuffer` / `IsL0BBuffer` / `IsL0CBuffer` 等 scope 谓词 | 已核验（2026-10-01） |
| R31 | `src/ascend/op/gemm.cc:23-63` | `SelectInst` 恒返回 `ascend.mad`；`ComputeWarpPartition` 恒 `{1,1}` | 已核验（2026-10-01）：逐行读 |
| R32 | `src/ascend/transform/ascend_pipe.h:52-117` | `ResourcePipe` 位掩码与注释（MTE1/MTE2/MTE3/M/V/FIX/S） | 已核验（2026-10-01）：逐行读 |
| R33 | `src/ascend/transform/vf_regions.h:19-60` | `SIMT_VF` / `SIMD_VF` 为不透明 SBlock；`SIMT_VF` 自绑 `threadIdx.x` | 已核验（2026-10-01） |
| R34 | `src/ascend/transform/assign_core.cc:64-75`、`src/ascend/transform/core_mask.h:26-37` | `CoreMask` 与跨核 pipe 的核归属 | 已核验（2026-10-01） |
| R35 | `src/ascend/transform/rewrite_dual_copy.cc`（文件头 :1-20） | `RewriteDualCopy` 的职责与执行时机 | 已核验（2026-10-01） |
| R36 | `src/ascend/transform/auto_schedule.cc:705-740` | 读循环 `num_stages` 注解决定多缓冲 | 已核验（2026-10-01）：`rg -n "num_stages"` |
| R37 | `src/ascend/op/ascend_mte_plan.h:1-8,198-275,296-409` | MTE 二维拷贝描述符的规划（`MTECopy2D`、`PlanMTECopy`） | 已核验（2026-10-01）：全文阅读 |
| R38 | `src/tl_templates/ascend/common.h:1-13` | 生成源包含的 `c_api/asc_simd.h`、`simt_api/*` | 已核验（2026-10-01） |
| R39 | `tilelang/contrib/bisheng.py:88-228` | `dav-3510` 默认值、编译选项、`aibin` 两段式（bisheng + `ld.lld`） | 已核验（2026-10-01）：逐行读 |
| R40 | `src/ascend/runtime/ascend_module.cc:400-430,480-500` | `aclrtBinaryLoadFromData` / `aclrtBinaryGetFunction` / `aclrtLaunchKernelWithHostArgs` | 已核验（2026-10-01）：`rg -n "aclrt"` |
| R41 | `examples/ascend/example_vecadd.py:14-42` | SIMT / SIMD 两种写法的官方对照（`tile // 64`） | 已核验（2026-10-01）：逐行读 |
| R42 | `examples/ascend/README.md` | 官方示例清单与“每个 example 一个 correctness test”的约定 | 已核验（2026-10-01） |
| R43 | `testing/ascend/README.md` | 昇腾测试分层（analysis / transform / auto_schedule / layout / runtime / target / language）与无卡检查命令 | 已核验（2026-10-01）：全文阅读 |

---

## 三、A 树（适配器 `_vendor/tilelang-ascend-v0.1.2.000/`）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| A1 | `README.md:12` | 自述里并列的“两条技术路线：Ascend C & PTO / AscendNPU IR” | 已核验（2026-10-01） |
| A2 | `VERSION` | 仓库内版本号 `0.1.4`（tag 为准，见 V3） | 已核验（2026-10-01） |
| A3 | `tilelang/utils/target.py:99-100,111-152` | `target in ["ascendc", "pto"]` → `llvm --keys=ascend`；平台 A2/A3/A5 判定 | 已核验（2026-10-01）：逐行读 |
| A4 | `tilelang/engine/lower.py:156-170` | `device_codegen` 按 `target.model` 分派 `tilelang_ascend` / `tilelang_ascend_pto` | 已核验（2026-10-01） |
| A5 | `tilelang/engine/phase.py:49-124` | 两条阶段：`LowerAndLegalize` 与 `OptimizeForTarget`（含 `PipelinePlanning`+`InjectSoftwarePipeline`、`AscendSyncInsert`） | 已核验（2026-10-01）：全文阅读 |
| A6 | `tilelang/jit/adapter/libgen.py:60-290` | `resolve_compile_flags` 与两条 `bisheng` 命令行（`-xasc` 与 `-xcce`）、camodel 分支 | 已核验（2026-10-01）：逐行读 |
| A7 | `tilelang/transform/pass_config.py:38-62` | `TL_ASCEND_AUTO_SYNC` / `TL_ASCEND_MEMORY_PLANNING` / `TL_ASCEND_AUTO_CV_SYNC` 等 8 个键 | 已核验（2026-10-01） |
| A8 | `tilelang/language/allocate.py:130-156` | `alloc_L1` / `alloc_L0A` / `alloc_L0B` / `alloc_L0C` / `alloc_ub` 与 scope 对应表 | 已核验（2026-10-01）：逐行读 |
| A9 | `tilelang/language/__init__.py:83-86` | `from .ascend import *`、`from . import ascend_tile as tile`（`T.tile` 的来源） | 已核验（2026-10-01） |
| A10 | `tilelang/language/ascend.py:191-317` | `set_cross_flag` / `wait_cross_flag` / `set_flag` / `wait_flag` / `barrier_all` / `pipe_barrier` / `sync_all` | 已核验（2026-10-01）：逐行读 |
| A11 | `tilelang/language/ascend.py:418-533` | `gemm_v0(A, B, C, transpose_A, transpose_B, init, kL0Size, n_actual)` | 已核验（2026-10-01） |
| A12 | `tilelang/language/ascend.py:534-616` | `printf` 与 `dump_tensor`（M 树没有 `dump_tensor`） | 已核验（2026-10-01） |
| A13 | `tilelang/language/ascend_tile.py`（3,112 行；示例锚点 :1572） | `T.tile.*` 的实现体（`sort32`、`row_expand_mul` 等） | 已核验（2026-10-01）：文件存在 + 抽样读取；未逐行审阅全部 3,112 行 |
| A14 | `tilelang/language/warpgroup.py:74` | `T.Scope(name)`（A 树用 `T.Scope("C")`/`T.Scope("V")` 指定执行域） | 已核验（2026-10-01） |
| A15 | `tilelang/language/pipeline.py:11`、`tilelang/language/persistent.py:11` | A 树的 `T.Pipelined` / `T.Persistent` | 已核验（2026-10-01） |
| A16 | `src/target/codegen_ascend.cc:100-109,808-860,986,1028-1032` | `__global__ __aicore__` 前缀、scope→`ascend_l1/l0a/l0b/l0c/ub` 的 `GetWithOffset`、`AscendC::TPipe` / `TBuf` | 已核验（2026-10-01）：逐行读 |
| A17 | `src/target/codegen_ascend_pto.cc:495-512,4249-4250` | PTO codegen 的包含（`<pto/pto-inst.hpp>`）与 `__global__ AICORE` 前缀 | 已核验（2026-10-01） |
| A18 | `src/tl_templates/pto/common.h:1-60` | `pto::Tile<...>` 系列别名（L1 / L0A / L0B / UB）与 `PTO_PLATFORM_A5` | 已核验（2026-10-01）：全文阅读 |
| A19 | `3rdparty/pto-isa/` | PTO-ISA 子模块目录（`.gitmodules` 指向 `gitcode.com/cann/pto-isa`） | **未核验内容**：子模块目录为空，未取得头文件 |
| A20 | `examples/gemm/example_gemm.py`、`examples/elementwise/elementwise_add.py` | 正文 §3.2 的 A 树示例原型 | 已核验（2026-10-01）：逐行读 |
| A21 | `docs/TileLang-Ascend Programming Guide.md`（corpus 副本） | 调试（§5）、生成代码查看（§5.2）、msProf（§6）、aclgraph（§7）、camodel（§2.4） | 已核验（2026-10-01，见 R44 同样路径） |

---

## 四、corpus 资料

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| R44 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/TileLang-Ascend Programming Guide.md:2342-2560` | T.printf/T.dump_tensor、`get_kernel_source()`、msProf、aclgraph | 已核验（2026-10-01）：逐行读该区间 |
| R45 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-mlir-ascend/README.md:11` | “两条技术路线”的原文 | 已核验（2026-10-01） |
| R46 | `.../tilelang-mlir-ascend/tilelang/engine/lower.py:167-181,269-294` | `target kind == "npuir"` 的分派与 `TILELANG_DUMP_IR` 输出 | 已核验（2026-10-01）：逐行读 |
| R47 | `.../tilelang-mlir-ascend/tilelang/jit/jit_npu.py:1862-1916` | `bishengir-compile` 调用与 A5/非 A5 两套开关 | 已核验（2026-10-01）：逐行读 |
| R48 | `.../tilelang-mlir-ascend/tilelang/utils/npu_utils.py:355-380` | `bishengir-compile` / `bishengir-opt` 的查找 | 已核验（2026-10-01） |
| R49 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/huawei-techarticle-tilelang-npuir.md:54-146` | AscendNPU IR 定义、MLIR 方言（HFusion/HIVM/HACC/Annotation、Scope）与关键编译特性 | 已核验（2026-10-01）：逐行读 |
| R50 | 同上 `:402-455` | JIT 编译流程与 `bishengir-compile` 开关 | 已核验（2026-10-01） |
| R51 | `../tilelang-corpus/06-huawei-ascend/raw/PTO-Gym/docs/PTO-tile-Instruction-SPEC.md:1-140` | PTO tile 模型（tile buffer、物理形状 vs 有效区、pipeline 归属、`pto.tload`/`tstore`） | 已核验（2026-10-01）：逐行读前 140 行 |
| R52 | `../tilelang-corpus/06-huawei-ascend/raw/pto-spec/docs/tile/matrix-and-matrix-vector/matrix-matrix/TMATMUL.md:1-40` | `TMATMUL` 的语义与执行引擎（CUBE） | 已核验（2026-10-01） |
| R53 | `../tilelang-corpus/06-huawei-ascend/raw/pto-spec/README.md:1-40` | PTO ISA 的规模（466 scalar forms、96 bundle/command、118 direct Tile ops、905 ASL units） | 已核验（2026-10-01） |
| R54 | `../tilelang-corpus/06-huawei-ascend/raw/pto-dsl/README.md:1-52` | PTO-DSL 与 tilelang-ascend 的关系（“tilelang 用 PTO-ISA 作为 codegen 后端”）、与 triton-ascend/CATLASS/PyPTO 的定位对照 | 已核验（2026-10-01）：逐行读 |
| R55 | `../tilelang-corpus/06-huawei-ascend/raw/PTO-Gym/README.md:1-40` | `ptoas` 与 PTO 微指令/tile 指令的验证工作流 | 已核验（2026-10-01） |
| R56 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/cann/zh_cann_bisheng.md:1-60` | 毕昇编译器的异构编译、微架构优化、亲和调度描述 | 已核验（2026-10-01） |
| R57 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/cann/zh_cann_ascend-c.md:1-40` | Ascend C 的定位（语言扩展层 + C++ 类库，`.asc` 后缀，异构编译） | 已核验（2026-10-01） |
| R58 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/hardware/zh_hardware_processor.md:57-58` | Ascend 950PR 的 Cube Core 32/28、Vector Core 64/56（1:2 比例的依据） | 已核验（2026-10-01） |
| R59 | `../tilelang-corpus/08-frontier-unmerged/raw/github-issue-3317.md` | “两个昇腾 950 后端选哪个”的提问；**无维护者结论** | 已核验（2026-10-01）：全文读；页面 Activity 区无可见评论 |
| R60 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3308.md` | 主干昇腾后端 PR 的正文：目标硬件 `dav-3510`、A2/A3 走社区仓库、功能清单 | 已核验（2026-10-01）：逐行读 |
| R61 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3310.md` | PTO 后端 PR：注册 `pto` target、用 `ptoas`/bisheng 编译 PTODSL。**corpus 的 main 源码快照（commit `994b44e`）即其合并提交** | 已核验（2026-10-01）：全文读；PR 页面快照被标为 draft（`marked this pull request as draft`），但 main 源码快照里有 `src/ascend/codegen/codegen_pto.cc` 与 `tilelang/jit/adapter/pto/`，以代码为准 |
| R62 | `../tilelang-corpus/11-related-compilers/raw/mlir-docs.html` | MLIR 官方文档 | **未核验（内容不足）**：抓取到的是导航页，正文未落盘；正文未引用其具体内容 |

---

## 五、NV 侧对照（仅用于 §5 的结构对照表）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| N1 | `_vendor/tilelang-v0.1.15/tilelang/cuda/backend.py:69-120` | `tilelang_callback_cuda_compile`（nvcc、gencode、cubin/fatbin） | 已核验（2026-10-01）：逐行读 |
| N2 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:69-275` | CUDA pipeline：`ThreadSync("shared")`、mbarrier、warp specialization 等 | 已核验（2026-10-01）：`rg -n "^    mod = "` |
| N3 | `_vendor/tilelang-v0.1.15/src/cuda/codegen/codegen_cuda.cc`（6,631 行） | CUDA codegen 的规模对照 | 已核验（2026-10-01）：文件存在与行数；未逐行审阅（第 24 章的职责） |

---

## 六、上游链接

| id | 链接 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| U1 | https://github.com/tile-ai/tilelang/pull/3308 | 主干昇腾 950 后端 PR | 未核验（本机未联网；corpus 快照 `08/raw/github-pr-3308.md` 已核验） |
| U2 | https://github.com/tile-ai/tilelang/pull/3310 | PTO 后端 PR（draft） | 未核验（同上，见 `08/raw/github-pr-3310.md`） |
| U3 | https://github.com/tile-ai/tilelang/issues/3317 | 两条路线选型提问 | 未核验（同上，见 `08/raw/github-issue-3317.md`） |
| U4 | https://github.com/tile-ai/tilelang-ascend | 适配器仓库 | 未核验（本机未联网；锁定快照见 `_vendor/tilelang-ascend-v0.1.2.000/`） |
| U5 | https://github.com/tile-ai/tilelang-ascend/tree/npuir | AscendNPU IR 分支 | 未核验（本机未联网；corpus 快照 `07/raw/tilelang-mlir-ascend/`） |
| U6 | https://github.com/tile-ai/tilelang-ascend/tree/ascendc_pto | Ascend C & PTO 分支 | 未核验（同上） |
| U7 | https://gitcode.com/cann/pto-isa | PTO-ISA（适配器 `3rdparty/pto-isa` 的上游） | 未核验（本机未联网；corpus 里 `PTO-Isa` 相关文档来自 PTO-Gym，见 R51/R55） |
| U8 | https://www.hiascend.com/cann/bisheng | 毕昇编译器官方页 | 未核验（本机未联网；镜像见 R56） |

---

## 七、本章代码

| 目录 | 作用 | 验证级别 |
| --- | --- | --- |
| `code/01-route-map/` | 抽取四条路线的入口/生成物/编译器并做锚点断言（24 个锚点） | CPU 逻辑验证 |
| `code/02-scope-dma-pipes/` | 抽取 scope→限定符、`DMAPath`、`ResourcePipe` 三张表并做 4 条一致性检查 | CPU 逻辑验证 |
| `code/03-api-presence-matrix/` | 两棵树的 API 存在性矩阵（38 个 API + 3 条路线检查），含 4 条断言 | CPU 逻辑验证 |
| `code/04-cite-check/` | 校验本章所有 `path:line` 引用可解析、不越界 | CPU 逻辑验证 |
| `code/05-emit-ascend-c/` | 真实 kernel + 生成物检查清单；`--audit` 离线，`--emit` 需要昇腾环境 | `--audit` = CPU 逻辑验证；`--emit` = **仅静态核对（未运行）** |

---

## 八、待补资料

| # | 缺什么 | 为什么需要 | 建议来源 |
| --- | --- | --- | --- |
| G1 | AscendNPU IR 的方言定义（`.td` 文件、`tilelangir` 的 op/type 定义） | 正文 §4.9 只能引用二手技术文章描述 HFusion/HIVM/HACC；若拿到 `.td`，可把“TIR → npuir”的映射逐条落到 op 级别 | `tilelang-mlir-ascend/3rdparty/AscendNPU-IR` 与 `AscendNPU-IR-Dev` 子模块（corpus 快照中目录为空）；需在上游 clone 后补进 corpus |
| G2 | `3rdparty/pto-isa` 的头文件（`pto/pto-inst.hpp` 与 `pto::Tile` 的定义） | 正文 §4.10 与 A18 只能看到别名用法，看不到 `Tile` 模板的完整语义（padding、valid region 如何落成指令） | `gitcode.com/cann/pto-isa`；或 PTO-Gym 的 `PTO-tile-Instruction-SPEC.md` 已覆盖一部分（R51） |
| G3 | 昇腾 A2/A3 上 `T.dual_copy` 的等价写法 | 正文 §4.5 只覆盖 M 树（950）的 `dual_copy`；A 树没有该原语，读者若在 A2 上需要双 AIV 分裂只能手写 | A 树 `examples/`（如 `gemv`、`flash_attention`）里的 `vid` 手写切分示例；需要一次专门的例子梳理 |
| G4 | `T.tile.*` 的语义清单 | 正文只指出它存在于 A 树，没有逐个讲清 `sort32`/`brcb`/`row_expand_mul` 等 op 的语义 | A19/A13 已定位文件；需要按 `docs/language_ref/` 逐条整理（本任务范围外） |
| G5 | mlir-docs 的可读正文 | R62 只抓到导航页，导致 §4.9 无法直接引用 MLIR 官方对 dialect/progressive lowering 的定义 | 在可联网环境重新抓 `mlir.llvm.org/docs/`（或改用 Huawei 技术文章的等价描述，正文当前采用后者） |
