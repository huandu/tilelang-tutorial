# 第 29 章 参考资料

本章所有技术结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`，`VERSION` = `0.1.15`）与
TileLang-Ascend `TileLang-ascend-v0.1.2.000-release`
（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU、没有昇腾 NPU、
没有安装 tilelang/tvm**（`import tilelang` → `ModuleNotFoundError`，
`command -v nvcc` 无输出）。因此本章代码示例的最高验证级别是“仅静态核对”；
可离线运行的三个脚本是“CPU 逻辑验证”（见 `code/` 下各 README）。
**本章不包含任何自测性能数字**，也不包含任何编译/运行结果。

核验日期均为 2026-10-01。核验方式：在锁定版本的工作树上用 `grep -n` / `sed -n`
定位定义行与行号，阅读完整签名、文件头注释与断言；计数类结论用脚本统计
（例如 `ls src/transform/*.cc | wc -l`、`python3 -c` 统计 `__all__`）。

---

## 一、双语结构与 FFI

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S1 | `VERSION` | 版本事实 = `0.1.15` | 已核验（2026-10-01） |
| S2 | `tilelang/transform/_ffi_api.py:1-6` | 全文件 6 行；`:3` `import tvm_ffi`、`:6` `init_ffi_api("tl.transform", __name__)` | 已核验（2026-10-01）：逐行读 |
| S3 | `tilelang/transform/__init__.py:9-11`（Python pass 的 import）、`:19-26`（`PipelinePlanning`）、`:86-94`（`VerifyParallelLoop`）、`:97-108`（`VerifyBufferInit`、`ThreadSync`） | 薄包装与门面 | 已核验（2026-10-01） |
| S4 | `src/transform/verify_buffer_init.cc:400-422` | `Pass VerifyBufferInit()`（`:400-417`）与 `TVM_FFI_STATIC_INIT_BLOCK`（`:419-422`） | 已核验（2026-10-01）：逐行读 |
| S5 | `src/transform/thread_storage_sync.cc:2091-2110` | **带参数**的 pass：`ThreadSync(const String &storage_scope)` 与注册 | 已核验（2026-10-01） |
| S6 | `src/ascend/transform/insert_nd2nz.cc:485-491` | 昇腾 pass 的 `CreatePrimFuncPass`（`:485`）与注册（`:488-490`） | 已核验（2026-10-01） |
| S7 | `tilelang/ascend/transform/__init__.py:1-11`（`InsertNd2Nz`）、`:154-163`（`AscendInsertOOBPadding` 的顺序约束）、`:225-256`（`__all__`，共 **30** 个名字） | 昇腾 pass 的 Python 门面与 pass 数量 | 已核验（2026-10-01）：`__all__` 名字数由脚本统计 |
| S8 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/transform/_ffi_api.py:1-8` | 适配器用旧接口 `tvm._ffi._init_api`（`:8`） | 已核验（2026-10-01）：逐行读 |
| S9 | `src/transform/verify_buffer_init.cc:1-16` | 文件头注释：报什么、不报什么、精度边界 | 已核验（2026-10-01） |
| S10 | `tilelang/ir.py:105`、`:123` | `get_stmt_span` / `stamp_stmt_spans` | 已核验（2026-10-01） |
| S11 | `tilelang/ascend/transform/z3_scheduler.py:9`、`:228-229`、`:784-785` | `import tvm_ffi` 与两个 `register_global_func` 的名字 | 已核验（2026-10-01） |

## 二、Python pass 与 mutator（本章示例 A 的原型）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S12 | `tilelang/transform/hoist_broadcast_values.py:16-72`（mutator）、`:75-99`（pass 工厂，`:99` 无 `name=`） | 示例 A 的简化原型 | 已核验（2026-10-01）：逐行读 |
| S13 | `tilelang/transform/add_bufstore_wrapper.py:1-159`（`:104-121` pre/post visit、`:155-159` 收尾与 `prim_func_pass`） | 双访客 + 嵌套深度计数器的范例 | 已核验（2026-10-01） |
| S14 | `tilelang/transform/decouple_type_cast.py:1-44`（文件头 before/after）、`:107-135`（`_CastFinder`）、`:349-410`（`visit_for_`）、`:642-663`（pass 工厂与顺序约束注释 `:650-652`） | 完整 Python mutator 的范例 | 已核验（2026-10-01） |
| S15 | `tilelang/ascend/analysis/vf_checker.py:5`、`:173`、`:186` | `VFChecker` 是 Python 写的 `prim_func_pass` | 已核验（2026-10-01） |

## 三、pipeline 与挂载

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S16 | `tilelang/cuda/pipeline.py:68`、`:79`、`:83-84`、`:93-98`、`:129-130`、`:134`、`:142`、`:150`、`:168`、`:284` | 顺序、守卫、`PassPipeline` 构造 | 已核验（2026-10-01）：`grep -n` 逐点核对 |
| S17 | `tilelang/ascend/pipeline.py:22-33`、`:36`、`:58`、`:61`、`:76`、`:95`、`:103-116`、`:125`、`:190` | 昇腾顺序、`VFChecker` 调用点、自动调度段 | 已核验（2026-10-01） |
| S18 | `tilelang/cpu/pipeline.py:35,50`、`tilelang/metal/pipeline.py:40,59`、`tilelang/rocm/pipeline.py:39,52`、`tilelang/webgpu/pipeline.py:41,54` | `VerifyBufferInit` / `DecoupleTypeCast` 出现在 **6** 条 pipeline 中 | 已核验（2026-10-01）：`grep -rn` |
| S19 | `tilelang/backend/pass_pipeline/pipeline.py:11-40` | `PassPipeline` 与 `pass_pipeline(name)` 作用域 | 已核验（2026-10-01） |
| S20 | `tilelang/backend/pass_pipeline/pipeline_utils.py:25-76`、`:105` | `allow_*` / `should_*` 守卫与 `LayoutVisual` | 已核验（2026-10-01） |
| S21 | `tilelang/engine/lower.py:92-96`、`:117`、`:136`、`:162`、`:234` | `HoistBroadcastValues` 的挂载点；三处 `compile_pass_instrumentation` | 已核验（2026-10-01） |

## 四、instrumentation、调试工具与 pass config

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S22 | `tilelang/instrumentation/session.py:130`、`:163`、`:213` | 工具注册、编译作用域、把工具装进 `PassContext` | 已核验（2026-10-01） |
| S23 | `tilelang/instrumentation/events.py:1-50` | `pass_phase` / `PassEvent` 的定义 | 已核验（2026-10-01） |
| S24 | `tilelang/tools/pass_timing.py:25`（logger）、`:64-73`（`PassTimingRecord` 的 `duration_s` 与 `self_duration_s`）、`:334`（`create_pass_timing_tool`） | 编译期耗时工具 | 已核验（2026-10-01） |
| S25 | `tilelang/jit/kernel.py:243` | JIT 编译时装配 pass timing 工具 | 已核验（2026-10-01） |
| S26 | `tilelang/tools/lower_trace/__init__.py:49`（`lower_trace`）、`tilelang/tools/lower_trace/core.py:648`、`:663`（`create_pass_instrument` / `enable`） | IR Lower Trace 的程序 API | 已核验（2026-10-01） |
| S27 | `tilelang/utils/pass_diff.py:1030`（`pass_diff`） | 旧 IR diff 工具的程序 API | 已核验（2026-10-01） |
| S28 | `docs/tools/lower_trace.md`、`docs/tools/pass_diff.md`（顶部“Superseded — use IR Lower Trace”admonition） | 工具的官方说明与被取代关系 | 已核验（2026-10-01）：逐行读顶部段落 |
| S29 | `src/op/builtin.h:115-130`（`kDisableDataRaceCheck` `:115`、`kDisableBufferInitCheck` `:122`、`kEnableDumpIR` `:129`、`kDumpIRDir` `:130`）、`src/op/builtin.cc:28-42` | pass config 常量的定义与注册 | 已核验（2026-10-01） |
| S30 | `tilelang/transform/pass_config.py:10`、`:52-68`、`:344-347` | `PassConfigKey` 枚举、两个检查开关、`TL_PASS_PROFILE*` | 已核验（2026-10-01） |

## 五、测试

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S31 | `testing/python/transform/test_tilelang_transform_verify_buffer_init.py:54-58`（`_verify`）、`:106-141`（extern / global / 参数 buffer）、`:183-201`（自读自写）、`:218-237`（跨线程作用域）、`:358-371`（只报一次）、`:377-388`（`no_kernel_cache`）、`:447-483`（`requires_cuda` 端到端） | verify pass 的测试范式与边界清单 | 已核验（2026-10-01）：逐段读 |
| S32 | `testing/python/transform/test_tilelang_transform_verify_parallel_loop.py:7-70` | 手搓 `IRModule` + `capfd` 断言 + span 断言 | 已核验（2026-10-01） |
| S33 | `testing/python/transform/test_tilelang_transform_hoist_broadcast_values.py:56-64`（`_check`）、`:67-133`（四个 before/after） | IR diff 基线的范式 | 已核验（2026-10-01） |
| S34 | `testing/ascend/_ir.py:1-58` | 昇腾侧 IR 构造助手（`seq` / `region` / `copy` / `kernel` / `nodes` / `calls`） | 已核验（2026-10-01） |
| S35 | `testing/ascend/transform/test_ascend_insert_nd2nz.py:11-68` | 昇腾 pass 测试的范式（含 `pytest.raises` 的存储约束用例） | 已核验（2026-10-01） |
| S36 | `testing/ascend/` 目录清单（`transform/` 15 个测试文件 + `language` / `layout` / `analysis` / `auto_schedule` / `runtime` / `cache` / `target`） | 昇腾测试分区与数量 | 已核验（2026-10-01）：`ls` |

## 六、性能回归

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S37 | `maint/scripts/run_perf_regression.sh:1-32`（用法与环境变量）、`:150-168`（baseline worktree）、`:240-369`（wheel 缓存）、`:412-493`（跑两遍 + 对比 + 汇总） | “改动前后回归判定”的主入口 | 已核验（2026-10-01）：逐段落读 |
| S38 | `maint/scripts/test_perf_regression.py:24-32`（`THRESHOLD` / `MIN_COUNT` 与标定注释）、`:243-257`（判定与措辞） | 判定规则的唯一出处 | 已核验（2026-10-01） |
| S39 | `maint/scripts/compare_perf_regression.py:20-90`（payload 解析与 speedup）、`:293`、`:303`、`:324`（`--fail-on-regression` 与阈值） | 对比器与退出码 | 已核验（2026-10-01） |
| S40 | `maint/scripts/run_current_regression.py:44-56`（`git` 状态字段）、`:104-146`（缓存 payload 结构：`results` / `failures` / `git`） | `--old` / `--new` 输入 JSON 的 schema | 已核验（2026-10-01）：`grep -n` 核对字段名 |
| S41 | `tilelang/testing/perf_regression.py:16-23`（`PerfResult`）、`:89-106`（`process_func`）、`:109-159`（`regression()` 扫描 `regression_*` 入口） | 用例侧的写法规约 | 已核验（2026-10-01） |
| S42 | `maint/scripts/run_perf_regression_ascend.sh:1-20`、`maint/scripts/run_local_ci_test_ascend.sh:1-20` | 昇腾侧基准与单测入口（与 CUDA 侧的两点差异） | 已核验（2026-10-01） |
| S43 | `_vendor/tilelang-v0.1.15/benchmark/`（`matmul/`、`matmul_fp8/`、`mamba2/`、`blocksparse_attention/`、`compile_speed/`、`matmul_metal/`） | 单算子 benchmark 目录清单 | 已核验（2026-10-01）：`ls` |

## 七、代码风格、构建与评审

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S44 | `docs/developer_guide/cpp_style.md`（corpus 副本：`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/developer_guide/cpp_style.md`） | 命名、TVM object/handle 用法、`*Node` 指针边界、反射字段兼容性 | 已核验（2026-10-01）：读“Formatting / Naming / TVM Object And FFI Types / ObjectNode Fields”四节 |
| S45 | `CONTRIBUTING.md:40-62`（风格与审计）、`:64-93`（开发环境）、`:111-127`（lint 与测试命令） | 贡献流程的官方说明 | 已核验（2026-10-01）：逐行读 |
| S46 | `CMakeLists.txt:374-389`（`file(GLOB TILE_LANG_SRCS ...)`）、`:469`（`add_library(tilelang_objs ...)`） | “新增 `.cc` 要重新 configure”的依据 | 已核验（2026-10-01） |
| S47 | `src/ascend/CMakeLists.txt:1-42`（`:15-20` always 源、`:22` `if(NOT USE_ASCEND) return()`、`:32-38` active 源含 `src/ascend/transform/*.cc`） | 昇腾条件编译与“缺对象类型会 `Cannot find object type index`” | 已核验（2026-10-01） |
| S48 | `maint/layout_inference/README.md` | golden layout harness：为什么 cost model 的改动需要金标准 | 已核验（2026-10-01）：读全文 |
| S49 | `maint/host_checks/README.md`、`maint/precision/README.md` | host 侧 10 个故意报错的复现脚本；精度对照表 | 已核验（2026-10-01）：读开头段落 |
| S50 | `maint/scripts/audit_cpp_api_style.py`、`maint/scripts/apply_mit_license.sh`、`maint/scripts/check_mit_license.sh` | `maint/scripts/` 里的自检脚本清单 | 已核验（2026-10-01）：`ls`；`audit_cpp_api_style.py:601` 是 `main()` |

## 八、昇腾双树与适配器仓库（`_vendor/tilelang-ascend-v0.1.2.000/`）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| A1 | `AGENTS.md:59-75`（四条通用原则）、`:79-113`（核心算子开发原则，含“不要凭记忆猜 API”`:83-87`）、`:117-122`（命名与行宽） | 适配器仓库的硬约束 | 已核验（2026-10-01）：逐行读 |
| A2 | `AGENTS.md:24-28`（Pass 分析与设计 skills：`tilelang-pass-analyzer` / `-pass-workflow-analyzer` / `-pass-design`）、`.agents/skills/`（含 `tilelang-pass-generate`） | “新 Pass 定位”被显式写进工作约定 | 已核验（2026-10-01）：`ls .agents/skills` |
| A3 | `VERSION`（仓库内写 `0.1.4`）、tag `TileLang-ascend-v0.1.2.000-release` | 版本以 tag 为准 | 已核验（2026-10-01）：与 `../../VERSIONS.md` 一致 |
| A4 | `src/transform/` 清单（`ascend_sync_insert.cc`、`ascend_storage_rewrite.cc`、`ascend_memory_planning.cc`、`ascend_lower_parallel_to_vector.cc`、`cross_core_pipeline.cc`、`ascend_combinecv.cc` …） | 适配器是另一套昇腾 pass 词汇 | 已核验（2026-10-01）：`ls` |
| A5 | `maint/scripts/`（`ci_performance.py`、`performance.py`、…）、`ci/operator_test_manifest.yaml`、`testing/`（仅 `cpp/` 与 `python/`） | 适配器自己的测试/性能入口，与主干不同 | 已核验（2026-10-01）：`ls` |

## 九、corpus 资料

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| R1 | `../tilelang-corpus/12-operator-libraries/raw/tilelang-benchmark/README.md` | 公共算子集的形状表（GEMM / FlashAttention / Linear Attention）与标题里的 Archived 声明 | 已核验（2026-10-01）：读全文 |
| R2 | `../tilelang-corpus/12-operator-libraries/raw/tilelang-benchmark/debug.py` | 上游用来给结果加 ±5% 抖动的辅助脚本 | 已核验（2026-10-01）：读全文；**正文未引用其数值** |
| R3 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/developer_guide/cpp_style.md` | 官方 C++ 风格指南的 corpus 副本（与 `_vendor/` 内文档一致） | 已核验（2026-10-01）：与 `_vendor/tilelang-v0.1.15/docs/developer_guide/cpp_style.md` 对照 |
| R4 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/lower_trace.md`、`.../tools/pass_diff.md` | 两个调试工具的官方文档副本 | 已核验（2026-10-01） |
| R5 | `../tilelang-corpus/02-tilelang-core-repo/raw/`（`testing/python/debug/test_pass_timing.py`、`test_pass_diff.py`、`test_pass_events.py`、`tilelang/tools/pass_timing.py` …） | corpus 里的源码镜像；**版本可能与 v0.1.15 不同，正文一律以 `_vendor/` 为准** | 已核验（2026-10-01）：仅用于确认文件存在 |
| R6 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/`（`github-tile-ai-tilelang-ascend.md` 等） | 适配器仓库的一手镜像 | 已核验（2026-10-01）：仅作背景 |

## 十、本章自产材料

| id | 位置 | 内容 | 验证级别 |
| --- | --- | --- | --- |
| C1 | `code/01-python-pass/` | 最小 Python pass `HoistConstantBroadcast` + 三个 IR 对照用例 | **仅静态核对**（`py_compile` 通过；API 名逐个在 v0.1.15 核对） |
| C2 | `code/02-cpp-pass-template/` | C++ pass 模板 `CountParallelLoops` + 包装片段 + pytest 骨架 + 构建步骤 | **仅静态核对**（未编译） |
| C3 | `code/03-verify-pass-template/` | verify pass 模板 `VerifyLocalAllocLimit` + 边界用例 + 骨架抽取脚本 | 抽取脚本：**CPU 逻辑验证**（已运行）；C++ 与 pytest：**仅静态核对** |
| C4 | `code/04-regression-judge/` | 按 v0.1.15 规则判定回归的小工具（`--self-test` 已运行） | **CPU 逻辑验证**（已运行） |
| C5 | `figures/fig-01` … `fig-04` | 四张自绘图（双语结构与 FFI / pass 骨架解剖 / 四层测试 / 回归定位流程） | 自绘；行号取自 `_vendor/`，2026-10-01 核对 |

## 十一、上游链接（本机以 `curl -o /dev/null -w '%{http_code}' -L` 核验）

| id | URL | 用途 | 核验状态 |
| --- | --- | --- | --- |
| U1 | https://github.com/tile-ai/tilelang | 主干仓库 | 已核验（2026-10-01，HTTP 200） |
| U2 | https://github.com/tile-ai/tilelang-ascend | 昇腾适配器仓库 | 已核验（2026-10-01，HTTP 200） |
| U3 | https://github.com/tile-ai/TileOPs | `tilelang-benchmark` README 指向的继任仓库 | 已核验（2026-10-01，HTTP 200） |
| U4 | https://github.com/tile-ai/tilelang-benchmark | 公共算子集 benchmark 仓库（已归档） | 已核验（2026-10-01，HTTP 200） |
| U5 | https://github.com/tile-ai/tilelang/blob/v0.1.15/docs/developer_guide/cpp_style.md | C++ 风格指南（v0.1.15 tag） | 已核验（2026-10-01，HTTP 200） |
| U6 | https://github.com/tile-ai/tilelang/issues?q=is%3Aissue+is%3Aopen+label%3A%22%E2%99%A5+help%22 | `CONTRIBUTING.md:33` 提到的 `♥ help` 标签 | 已核验（2026-10-01，HTTP 200） |
| U7 | https://tilelang.com/get_started/Installation.html | `CONTRIBUTING.md:109` 引用的安装文档 | 已核验（2026-10-01，HTTP 200） |
| U8 | https://tilelang.com/tools/lower_trace.html | IR Lower Trace 的在线文档 | 已核验（2026-10-01，HTTP 200） |
| U9 | https://tilelang.com/tools/pass_diff.html | Pass Diff 的在线文档 | 已核验（2026-10-01 终审复验 `curl -L` 返回 HTTP 200；写作时曾超时，结论仍以本地 `docs/tools/pass_diff.md` 为准） |

## 十二、待补资料

| 缺什么 | 为什么需要 | 建议来源 |
| --- | --- | --- |
| 官方“如何写一个 pass”的教程 | v0.1.15 的 `docs/` 里没有这样一篇：`docs/developer_guide/` 只有 `cpp_style.md`，`docs/compiler_internals/` 只有 4 篇（`inject_fence_proxy` / `letstmt_inline` / `metal_tilelang_development` / `tensor_checks`）。本章 3.2–3.4 的“五步法”是从源码归纳的，不是引用官方教程 | 上游若新增 `docs/contributing/writing-a-pass.md` 之类的文档；或维护者的一份 issue 说明 |
| 一次真实编译 + 测试的完整日志 | 本章所有“能编译 / 测试通过”的说法都停在**待验证**。需要一台有 CUDA 的机器，把 `code/02-cpp-pass-template/README.md` 的六步跑通并留档 | 任一有 sm_80+ GPU 与 nvcc 的开发机 |
| `maint/scripts/run_perf_regression.sh` 的实测输出样例 | 本章只能引用判定规则与脚本结构，给不出真实报告样例 | 上游 CI 产出的 PR 机器人报告（`maint/scripts/run_perf_regression.sh` 的运行结果） |
| 适配器 `.agents/skills/` 下 4 个 pass 相关 skill 的正文 | `AGENTS.md:24-28` 只列了名字与一句话职责；要判断它们与本章方法论是否一致，需要读 `SKILL.md` 正文 | 适配器仓库 `.agents/skills/tilelang-pass-*/` |
