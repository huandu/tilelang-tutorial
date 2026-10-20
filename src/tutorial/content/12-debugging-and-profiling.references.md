# 第 12 章 参考资料

本章所有结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`，`VERSION` 文件 = `0.1.15`）、
TileLang-Ascend `TileLang-ascend-v0.1.2.000-release`
（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。** 正文与示例中不再重复版本号，
也不对「上游现在是什么状态」作任何未核验的断言。

写作环境：macOS (Apple Silicon)、Python 3.11（miniconda）；**没有 NVIDIA GPU，也没有
昇腾 NPU**。因此本章代码示例的最高验证级别是「仅静态核对」，两个纯计算脚本是
「CPU 逻辑验证」（见 `code/README.md` 的验证级别总表）。**本章不包含任何自测性能数字。**

核验日期均为 2026-10-01。核验方式：在锁定版本的工作树上用 `grep -n` 定位定义行，
阅读完整签名与 docstring，并逐行核对正文引用的代码片段。**这不是运行验证。**

---

## 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| S1 | `VERSION` | 版本事实 = `0.1.15` | 已核验（2026-10-01） |
| S2 | `src/runtime/logging.cc:17-23`、`:51-64`、`:65-70`、`:73-135` | 日志等级表、`LogMessageImpl`、`LogFatalImpl`（抛 `tvm::ffi::Error`）、`TvmLogDebugSettings::ParseSpec` | 已核验（2026-10-01）：逐行阅读，行号与内容一致 |
| S3 | `CMakeLists.txt:487-490` | Debug 构建通过 `target_compile_definitions(... "TVM_LOG_DEBUG")` 打开 DLOG | 已核验（2026-10-01） |
| S4 | `tilelang/tools/compile_only.py:169`、`:216-235` | `compile_kernel_source()` 与 CLI（`--target` / `--output_file`） | 已核验（2026-10-01） |
| S5 | `tilelang/jit/kernel.py:532` | `get_kernel_source()` | 已核验（2026-10-01） |
| S6 | `tilelang/engine/callback.py:55`、`tilelang/ascend/callback.py:32` | `register_cuda_postproc_callback` / `register_ascend_postproc_callback` | 已核验（2026-10-01） |
| S7 | `tilelang/env.py:390-392`、`:496-510` | `TL_LOWER_TRACE` / `TL_LOWER_TRACE_DIR` 的定义与取值解析 | 已核验（2026-10-01） |
| S8 | `tilelang/tools/lower_trace/__init__.py:49`、`tilelang/tools/lower_trace/core.py:663-705` | `lower_trace()` 一次性 API；`enable()` / `disable()` / `reset()` | 已核验（2026-10-01） |
| S9 | `tilelang/utils/pass_diff.py:1030` | legacy `pass_diff()` 入口 | 已核验（2026-10-01） |
| S10 | `tilelang/tools/pass_visualizer/viewer.py:151`、`:424`、`:460` | `build_pass_data()` / `emit_html()` / CLI `main()` | 已核验（2026-10-01） |
| S11 | `tilelang/tools/pass_visualizer/README.md` | Pass Visualizer 的定位与用法（结构树视角） | 已核验（2026-10-01） |
| S12 | `tilelang/tools/pass_timing.py:148` | `TileLangPassTimingInstrument`（编译期耗时） | 已核验（2026-10-01） |
| S13 | `tilelang/tools/plot_layout.py:6`；`tilelang/transform/pass_config.py:252`、`:255` | `plot_layout()`；`TL_LAYOUT_VISUALIZATION_ENABLE` / `_FORMATS` | 已核验（2026-10-01） |
| S14 | `tilelang/transform/pass_config.py:107` | `TL_EMIT_LINE_DIRECTIVES`（`#line` 映射） | 已核验（2026-10-01） |
| S15 | `tilelang/cuda/language/print.py:128-207` | `T.print`（CUDA 方言）的四种分支与单线程门控；`main_lane` 在 `:149-151` | 已核验（2026-10-01）：全文逐行阅读 |
| S16 | `tilelang/cuda/debug.py:23` | `T.device_assert` | 已核验（2026-10-01） |
| S17 | `tilelang/ascend/language/print.py:31-70` | `T.print`（昇腾方言，去掉 CUDA 式门控） | 已核验（2026-10-01） |
| S18 | `src/tl_templates/cuda/debug.h`、`src/tl_templates/ascend/debug.h` | 设备端 `printf` 模板与输出格式 | 已核验（2026-10-01） |
| S19 | `tilelang/utils/tensor.py:205` | `torch_assert_close`（默认 `rtol=1e-2, atol=1e-3, max_mismatched_ratio=0.001`） | 已核验（2026-10-01） |
| S20 | `tilelang/profiler/__init__.py:104` | `Profiler.assert_allclose`（默认 `atol=1e-2, rtol=1e-2, max_mismatched_ratio=0.01`） | 已核验（2026-10-01） |
| S21 | `tilelang/autodd.py:1399`；`examples/autodd/` | AutoDD 的 `--err-msg` CLI 与示例 | 已核验（2026-10-01） |
| S22 | `tilelang/tools/Analyzer.py:16-33`、`:196-216` | `AnalysisResult` 字段与 `Analyzer.analysis()` 的结算公式 | 已核验（2026-10-01） |
| S23 | `tilelang/carver/arch/cuda.py:138` | `CUDA` device model（`Analyzer` 的入参；构造时会查询设备） | 已核验（2026-10-01） |
| S24 | `tilelang/profiler/bench.py:39`、`tilelang/profiler/msprof.py:40`、`:59`、`:184-266` | `do_bench(backend="msprof"/"msprof_detail")`、`bench_with_msprof`、`KernelProfile`、`_parse_ffts_profile` | 已核验（2026-10-01） |
| S25 | `tilelang/engine/lower.py:216-235` | `lower()` 的入口签名（`enable_device_compile` 缺省为 False） | 已核验（2026-10-01） |
| S26 | `tilelang/transform/__init__.py:108` | `ThreadSync(storage_scope)` 的 Python 入口 | 已核验（2026-10-01） |
| S27 | `tilelang/ascend/transform/__init__.py`、`tilelang/ascend/pipeline.py` | 昇腾侧 InsertSync 相关逻辑所在模块 | 已核验（2026-10-01）：`grep -rln "InsertSync"` 命中这两个文件 |
| S28 | `tilelang/language/allocate.py:400`、`tilelang/language/kernel.py:431`、`tilelang/language/loop.py:13`、`tilelang/jit/__init__.py:575` | `T.empty` / `T.Kernel` / `T.Parallel` / `tilelang.jit` | 已核验（2026-10-01） |

### 昇腾适配器仓库（`_vendor/tilelang-ascend-v0.1.2.000/`）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| A1 | `tilelang/language/ascend.py:534` | `T.printf`（含 `%p` → `0x%x` 改写） | 已核验（2026-10-01）：逐行阅读函数体 |
| A2 | `tilelang/language/ascend.py:582` | `T.dump_tensor`（`desc` 的 uint32 校验、`shape_info` 传递） | 已核验（2026-10-01） |
| A3 | `tilelang/transform/pass_config.py:62` | `TL_ASCEND_EXCEPTION_DUMP` 配置键（**只在昇腾仓库里**，主干 v0.1.15 的 `pass_config.py` 中不存在） | 已核验（2026-10-01）：分别在两个仓库 grep |
| A4 | `tilelang/tools/ascend_exception_dump_bin.py:27`、`:67` | `read_msaicerr_bin()` / `parse_exception_dump()` | 已核验（2026-10-01） |
| A5 | `src/tl_templates/ascend/exception_dump.h` | 异常回调模板；MAGIC = `0x474e414c454c4954`（ASCII `TILELANG`） | 已核验（2026-10-01）：在 corpus 白皮书中核对 MAGIC 数值与实现文件 |
| A6 | `docs/tutorials/ascend_aicore_exception_dump_zh.md`、`docs/tutorials/print.md` | 异常 dump 白皮书与 print/dump_tensor 教程 | 已核验（2026-10-01） |
| A7 | `VERSION` | 仓库内写 `0.1.4`，**以 tag `TileLang-ascend-v0.1.2.000-release` 为准** | 已核验（2026-10-01）：与 [../../VERSIONS.md](../../VERSIONS.md) 的说明一致 |

---

## corpus 资料引用

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| R1 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tutorials/debug_tools_for_tilelang.md` | 官方调试总览：生成/正确性两类问题；`T.print` 用法与并发提醒；postproc 回调；Pass Visualizer；AutoDD | 已核验（2026-10-01）：全文阅读 |
| R2 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tutorials/logging.md` | `LOG` / `DLOG` / `VLOG` / `CHECK` 的关系；`TVM_LOG_LEVEL` 与 `TVM_LOG_DEBUG` 的定义与陷阱 | 已核验（2026-10-01） |
| R3 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/lower_trace.md` | `TL_LOWER_TRACE` 的取值与输出目录结构；程序 API；edit-and-recompile 三文件工作流；三层实现（How It Works） | 已核验（2026-10-01）：全文阅读 |
| R4 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/pass_diff.md` | legacy Pass Diff 的 `TILELANG_PASS_DIFF` 与 `pass_diff()` API、返回字段 | 已核验（2026-10-01） |
| R5 | `_vendor/tilelang-v0.1.15/tilelang/tools/pass_visualizer/README.md` | Pass Visualizer 的结构树视角与「与 Pass Diff 的区别」 | 已核验（2026-10-01） |
| R6 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/compile_only.md` | Compile-Only 的 CLI、target 规则、C/CUDA 目标差异、`CUDA codegen FFI missing` | 已核验（2026-10-01）：全文阅读 |
| R7 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/layout_visualization.md` | `plot_layout` 与 `TL_LAYOUT_VISUALIZATION_*` 的用法、依赖（Matplotlib）、限制 | 已核验（2026-10-01） |
| R8 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/analyzer.md` | `Analyzer` 的 API、结果字段、Limitations（只认 `T.gemm`、只认跨 global 的 `T.copy`、blockIdx.z 不计、峰值表只覆盖 cc 8.0/8.6/8.9） | 已核验（2026-10-01） |
| R8b | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/autodd.md` | AutoDD 的用法与前提（失败必须确定化） | 已核验（2026-10-01） |
| R9 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-nsight-compute-docs.md:1081-1140`、`:1240-1260` | Details 页的 section 组织（Memory Workload Analysis / Occupancy / Roofline / Source Counters）；rules 的 `Est. Speedup`；warp stall 采样与 long/short scoreboard | 已核验（2026-10-01）：定位并阅读对应行段 |
| R10 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/CANNCommunityEdition/CANNCommunityEdition_920beta1_devaids_Profiling_atlasprofiling_16_0005_html.md` | `msprof --output=<path> <app>` 的一键采集；`PROF_XXX` 目录内默认产物清单；Ascend RC 的 `msprof --export=on` | 已核验（2026-10-01） |
| R11 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/tutorials/print.md` | `T.printf` / `T.dump_tensor` 的完整教程（编译选项、每个核 1 MB 上限、格式说明符、shape_info 语义） | 已核验（2026-10-01）：全文阅读 |
| R12 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/tutorials/ascend_aicore_exception_dump_zh.md` | AICore 异常 dump 白皮书：工作流程、`ParamSizeInfo` 与 MAGIC 搜索、环境变量、路径规则、API 速查 | 已核验（2026-10-01）：全文阅读 |
| R13 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/docs/design/roofline.md` | roofline 性能模型（`sol_time` / `efficiency`、算法最小搬运量的五条界定）——**TileOPs 是另一个库，此处只借方法学** | 已核验（2026-10-01）：阅读 §1 与 §2 |
| R14 | `../tilelang-corpus/02-tilelang-core-repo/raw/tilelang/.agents/skills/tilelang-ascend/references/profiling.md` | 昇腾 profiling 配方：`do_bench` 单位与返回类型、`msprof` 后端口径、pipe 比例不可相加、`npusim record` 周期级测量注意事项 | 已核验（2026-10-01）：全文阅读 |
| R15 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577-fulltext.md` | TileLang 论文全文（76 KB，逐页提取）；本章只用于确认「tile op / 依赖分析 / 自动调度」这套设计语境，**未从中取任何性能数字** | 已核验（2026-10-01）：grep 关键词（debug/profiling/roofline/autotune）后确认与本章无直接可引用结论 |

---

## 上游链接

| 链接 | 用途 | 核验状态 |
| --- | --- | --- |
| <https://docs.nvidia.com/nsight-compute/NsightCompute/index.html> | Nsight Compute 官方用户指南（corpus 镜像的原始出处） | 已核验（2026-10-01）：corpus 镜像页首标注了该 URL 与抓取日期 |
| <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/Profiling/atlasprofiling_16_0005.html> | 昇腾 msprof 离线采集快速入门（corpus 镜像的原始出处） | 已核验（2026-10-01）：corpus 镜像 front-matter 标注了该 URL |
| <https://github.com/tile-ai/tilelang/tree/main/examples/plot_layout> | `plot_layout` 的官方示例 | 未核验（原因：本地无外网访问的书面记录；路径来自 corpus 的 layout_visualization 文档正文，未单独打开） |
| <https://godbolt.org> | Compiler Explorer（compile-only 工具的后端集成目标） | 未核验（原因：同上；来自 corpus 的 compile_only 文档正文） |

---

## 与本章代码示例的对应关系

| 示例 | 覆盖的引用 |
| --- | --- |
| `code/01-tool-inventory/` | S4–S24、A1–A4 的存在性复核 |
| `code/02-tolerance-model/` | S19、S20（两个对拍工具的缺省容差） |
| `code/03-compile-only/` | S4、S5、S25、S28 |
| `code/04-lower-trace/` | S7、S8、S9、S26 |
| `code/05-print-and-assert/` | S15、S16、S18 |
| `code/06-ascend-dump/` | A1–A5、S17 |

---

## 待补资料

* **本机无法验证的部分**：所有需要 NVIDIA GPU 或昇腾 NPU 的实测（SOL 图、msProf 报告、
  `T.print` 实际输出、对拍实测误差）。本章的处理方式是给出「怎么测 + 判据」，不编造数字。
* **`pass_visualizer` 对昇腾目标的 CLI 支持**：corpus 的 debug 文档明确说 CLI「currently
  supports CUDA targets only」。如果之后上游补齐昇腾目标，本章 3.3 节需要相应更新。
* **`Analyzer` 的峰值表覆盖范围**：目前只到 compute capability 8.6 / 8.9；若上游加入
  Blackwell（10.x）条目，3.8 节的 Limitations 需要同步修改。
* **IKET**（CUDA 侧 timeline 标记工具）：corpus 的 `tools/iket.md` 与
  `_vendor/tilelang-v0.1.15/tilelang/tools/cuda/iket` 都存在，本章未展开，因为它需要外部
  IKET runtime；若后续要补「kernel 内 timeline」小节，可从这两处入手。
