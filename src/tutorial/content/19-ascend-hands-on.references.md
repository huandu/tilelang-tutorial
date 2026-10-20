# 第 19 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[`../../VERSIONS.md`](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`，
仓库内 `VERSION` 文件写 `0.1.4`，**以 tag 为准**）。**
文中所有版本结论都以锁定 tag 为准，不引用未锁定的开发分支；讨论 `main` 上更新的东西时
会显式标注“仅 main / 未进 tag”。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章所有 kernel 示例的最高验证级别是“仅静态核对”（API 存在性与签名逐个在
`_vendor/` 里比对过，语法用 `py_compile` 过了），**没有任何一条性能数字来自本机实测**。

本章用“**A 树**”指适配器仓库 `_vendor/tilelang-ascend-v0.1.2.000/`，
用“**M 树**”指主干 `_vendor/tilelang-v0.1.15/`。凡引用源码都标了树别。

---

## 锁定的源码引用（M 树：`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:44-52` | Ascend 950 后端的安装前提与 `USE_ASCEND=ON USE_CUDA=OFF pip install` 命令 | 已核验（2026-10-01）：逐行读 |
| S2 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:20-24` | 安装完成的判据（`torch.npu.is_available()` 为 True） | 已核验（2026-10-01） |
| S3 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:55-108` | GEMM+ReLU quickstart（`T.Persistent` / `T.Pipelined` / `T.gemm` / `T.dual_copy` / `T.SimtVF`） | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S4 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:130-155` | 与 Ascend C 的分工对照表；“自动调度不保证把任意 `T.Parallel` 张量化” | 已核验（2026-10-01） |
| S5 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:175-180` | 与 Torch NPU 的 Benchmark 口径（BF16 GEMM / FP8 casting / GQA backward） | 已核验（2026-10-01）；**无具体数值** |
| S6 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:26,37,46,55` | `alloc_l1` / `alloc_l0a` / `alloc_l0b` / `alloc_l0c` 的默认 scope（`shared.l1`/`shared.l0a`/`shared.l0b`/`shared.l0c`）与 L0C 自动布局 | 已核验（2026-10-01）：逐行读 |
| S7 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py:53` | `T.dual_copy(src, dst, unit_flag_ctrl, l2_cache_ctrl)` 的形状判据、支持通路、mixed-kernel 契约 | 已核验（2026-10-01） |
| S8 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py:107,179` | `T.SimtVF(threads, latency)` / `T.SimdVF(latency)` 的签名 | 已核验（2026-10-01） |
| S9 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/annotations.py:19` | `T.annotate_buffer_versions` 的语义与取值 | 已核验（2026-10-01） |
| S10 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/gemm_op.py:168,110` | M 树 `T.gemm` / `T.gemm_blockscaled` 的签名与 `unit_flag_ctrl` | 已核验（2026-10-01） |
| S11 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/sync.py` | `T.ascend_pipe_barrier` / `ascend_set_flag` / `ascend_wait_flag` 与 PIPE 名 | 已核验（2026-10-01） |
| S12 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/print.py:31-70` | M 树只有 `T.print`，**没有** `T.printf` / `T.dump_tensor` | 已核验（2026-10-01）：`grep -rn "def printf\|def dump_tensor"` 在 M 树无命中 |
| S13 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/mode.py` | `T.set_hf32_mode` / `T.set_atomic` / `T.set_atomic_none` / `T.set_mmad_direction` | 已核验（2026-10-01） |
| S14 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:106-115` | Ascend pass 顺序：`MaterializeScheduleUnits → AnnotateMultiBufferEligible → EstimateLatency → AutoSchedule → AssignCore → PrepareMultiBuffer → ResolveCore → InsertSync → MaterializeMultiBuffer → LowerScheduledTIR` | 已核验（2026-10-01）：逐行读 |
| S15 | `_vendor/tilelang-v0.1.15/tilelang/transform/pass_config.py:76` | M 树的开关是 `TL_ENABLE_AUTO_SCHEDULE`（默认 True）；**该文件里没有 `TL_ASCEND_*`** | 已核验（2026-10-01）：`grep -n "TL_ASCEND"` 无命中 |
| S16 | `_vendor/tilelang-v0.1.15/tilelang/profiler/bench.py:31-66,199-205` | `do_bench` 的 backend 列表；NPU 上只允许 `event`/`msprof`/`wall` | 已核验（2026-10-01） |
| S17 | `_vendor/tilelang-v0.1.15/tilelang/profiler/msprof.py:60-134` | `KernelProfile` 字段：`aic_mad/scalar/mte1/mte2/mte3/fixpipe`、`aiv_vec/scalar/mte2/mte3` | 已核验（2026-10-01） |
| S18 | `_vendor/tilelang-v0.1.15/maint/scripts/ascend_perf_regression.py:56-110` | Ascend 性能回归的 case 清单（`_ENTRIES`）：GEMM/DeepGEMM/FA/RMSNorm/vecadd 等 | 已核验（2026-10-01） |
| S19 | `_vendor/tilelang-v0.1.15/.agents/skills/tilelang-ascend/SKILL.md` | 950 使用手册：AIC/AIV 分工、`T.SimtVF`/`T.SimdVF`、`T.Task`/`T.PerCoreTask`、内存层级、GEMM 约束、`T.dual_copy`、同步三模式、`T.simd.*`、14 类常见坑、性能测量方法 | 已核验（2026-10-01）：整份读完 |
| S20 | `_vendor/tilelang-v0.1.15/.agents/skills/tilelang-ascend/references/synchronization.md` | 三种调度模式；buffer 版本/owner/索引模式；conditional epoch；manual multibuffer 的“有限支持”警告；`T.Stage` 调优；冲突提示；UnitFlag/Tail 生命周期 | 已核验（2026-10-01）：整份读完 |
| S21 | `_vendor/tilelang-v0.1.15/.agents/skills/tilelang-ascend/references/profiling.md` | `do_bench` 单位与返回类型；回归入口；生成源码实验；`npusim record` 周期测量；`T.Task(latency=, ii=)` 覆盖估计 | 已核验（2026-10-01）：整份读完 |
| S22 | `_vendor/tilelang-v0.1.15/examples/ascend/example_vecadd.py` | SIMT/SIMD 向量加 + `T.annotate_buffer_versions` + `do_bench(backend="msprof")` | 已核验（2026-10-01）：逐行比对；**未运行** |
| S23 | `_vendor/tilelang-v0.1.15/examples/ascend/example_gemm.py` | persistent GEMM：`T.Persistent` + `T.Pipelined` + `aswt_swizzle` + mixed epilogue + `unit_flag_ctrl` | 已核验（2026-10-01）；未运行 |
| S24 | `_vendor/tilelang-v0.1.15/examples/ascend/README.md` | 昇腾示例清单与 benchmark 运行方式；`--no-bench` 只跑正确性 | 已核验（2026-10-01） |
| S25 | `_vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc`（存在性） | “同步由编译器自动插入”的实现位置 | 已核验（2026-10-01）：文件存在；未精读算法 |
| S26 | `_vendor/tilelang-v0.1.15/src/ascend/codegen/`、`src/ascend/runtime/`（存在性） | 昇腾 codegen 与 runtime 目录 | 已核验（2026-10-01）：`find` 确认存在 |

---

## 锁定的源码引用（A 树：`_vendor/tilelang-ascend-v0.1.2.000/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| A1 | `_vendor/tilelang-ascend-v0.1.2.000/README.md:74-133` | 安装前提（CANN ≥ 8.3.RC1、torch-npu ≥ 2.6.0.RC1）与三种安装方式 | 已核验（2026-10-01）：逐行读 |
| A2 | `_vendor/tilelang-ascend-v0.1.2.000/README.md:134-...` | `T.Kernel(is_npu=True)`、`alloc_L1/alloc_L0C/gemm_v0/barrier_all` 的入门示例；高性能力版本（flag 管理 + `T.mma`） | 已核验（2026-10-01） |
| A3 | `_vendor/tilelang-ascend-v0.1.2.000/README.md`（Vid reduction & Auto CV Ratio / Workspace reduction 两节） | `threads=1|2` 的 vid 消除；workspace 消除把 `L0C→GM→UB` 变成一条 `T.copy` | 已核验（2026-10-01）：逐段读 |
| A4 | `_vendor/tilelang-ascend-v0.1.2.000/install_ascend.sh` | Python ≥ 3.10 检查、`git submodule`、TVM 补丁、`cmake`（`USE_ASCEND ON`）、`make -j 50% 核数`、`--enable-shmem` 流程 | 已核验（2026-10-01）：逐段读 |
| A5 | `_vendor/tilelang-ascend-v0.1.2.000/set_env.sh` | `TL_ROOT`、`PYTHONPATH` 前置、`ACL_OP_INIT_MODE=1` | 已核验（2026-10-01）：逐行读 |
| A6 | `_vendor/tilelang-ascend-v0.1.2.000/examples/elementwise/elementwise_add.py` | 逐元素 add（GM→UB→V→GM，两个 `barrier_all`） | 已核验（2026-10-01）：逐行比对；**未运行** |
| A7 | `_vendor/tilelang-ascend-v0.1.2.000/examples/elementwise/elementwise_add_pipeline.py` | 手写双缓冲 + 三对 flag（`v→mte3` / `mte3→mte2` / `mte2→v`） | 已核验（2026-10-01）；未运行 |
| A8 | `_vendor/tilelang-ascend-v0.1.2.000/examples/gemm/example_gemm.py` | 最小 GEMM（GM→L1→L0C，两次 `barrier_all`） | 已核验（2026-10-01）；未运行 |
| A9 | `_vendor/tilelang-ascend-v0.1.2.000/examples/gemm/example_gemm_intrinsic.py` | 带完整 flag 管理（`set_flag`/`wait_flag`）、多级流水与 `T.mma` 的手写版；`kL0Size` 说明 | 已核验（2026-10-01）；未运行 |
| A10 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/allocate.py:140-156` | `alloc_L1`（`shared.l1`）/`alloc_L0A`（`wmma.matrix_a`）/`alloc_L0B`（`wmma.matrix_b`）/`alloc_L0C`（`wmma.accumulator`）/`alloc_ub`（`shared.ub`） | 已核验（2026-10-01） |
| A11 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:418` | `gemm_v0(A, B, C, transpose_A, transpose_B, init, kL0Size, n_actual)`：docstring 明确 A/B 在 L1、C 在 L0C，`kL0Size` 建议 | 已核验（2026-10-01） |
| A12 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:191,213,233,251,275` | `set_cross_flag` / `wait_cross_flag` / `set_flag` / `wait_flag` / `barrier_all` | 已核验（2026-10-01） |
| A13 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:534,582` | `T.printf(format_str, *args)` / `T.dump_tensor(tensor, desc, dump_size, shape_info)` | 已核验（2026-10-01） |
| A14 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/transform/pass_config.py:38-62` | `TL_ASCEND_AUTO_SYNC` / `_AUTO_SYNC_VS` / `_MEMORY_PLANNING` / `_AUTO_CV_COMBINE` / `_AUTO_CV_SYNC` / `_TAIL_MASK` / `_EXCEPTION_DUMP`；`_TARGET_PASS_DEFAULTS` 区分 `ascendc` 与 `pto` | 已核验（2026-10-01）：逐行读 |
| A15 | `_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md:25-276` | 开发环境（`source .../set_env.sh`、`install_ascend.sh`、`source set_env.sh`）、compile/run 三阶段、A5 camodel 仿真与限制 | 已核验（2026-10-01）：逐段读 |
| A16 | `_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md:277-980` | 语法基础：`T.Kernel`（Expert `(cid, vid)` / Developer `threads=1|2` → `cid`）、`T.serial/unroll/Parallel/Pipelined/Persistent`、`T.copy` 支持的通路表 | 已核验（2026-10-01） |
| A17 | `_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md:1156-1263` | `T.Pipelined` 核内/核间两种 case 的时间表；“核间与核内不能同时开” | 已核验（2026-10-01） |
| A18 | `_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md:2292-2387` | 同步原语：`set_cross_flag`/`wait_cross_flag`/`set_flag`/`wait_flag`/`barrier_all`/`pipe_barrier`/`sync_all`；flag 严格配对要求（950 与 A2 的差异） | 已核验（2026-10-01） |
| A19 | `_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md:2388-2526` | 调试诊断：`T.printf`/`T.dump_tensor`（`compile_flags=["-D_DEBUG","--cce-enable-print"]`）、1 MB 上限、查看生成的 Ascend C | 已核验（2026-10-01） |
| A20 | `_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md:2527-2587` | 性能调优：`msprof op` 与 `msprof op simulator` 的用法与图；入图：aclGraph capture/replay 与“不承载图编译”的边界 | 已核验（2026-10-01） |
| A21 | `_vendor/tilelang-ascend-v0.1.2.000/docs/tutorials/t_pipelined.md` | `T.Pipelined(loop_iterations, num_stages, cross_interval)`；intra/inter-core 时间表；嵌套不支持；flat pattern | 已核验（2026-10-01）：整份读 |
| A22 | `_vendor/tilelang-ascend-v0.1.2.000/docs/tutorials/print.md` | `T.printf` / `T.dump_tensor` 接口与示例；“优先用 kernel-scoped `compile_flags`，`TL_PTO_DEBUG` 只对 pto” | 已核验（2026-10-01）：整份读 |
| A23 | `_vendor/tilelang-ascend-v0.1.2.000/docs/tutorials/ascend_aicore_exception_dump_zh.md` | exception dump 原理（MAGIC `0x474e414c454c4954`、`ParamSizeInfo`、`acldumpSaveExceptionInfo`）、CANN ≥ 9.3.0、环境变量、`parse_exception_dump` | 已核验（2026-10-01）：整份读 |
| A24 | `_vendor/tilelang-ascend-v0.1.2.000/docs/tutorials/debug_tools_for_tilelang.md` | 三类问题（生成/正确性/性能）；`T.print`；postproc 回调 | 已核验（2026-10-01）：整份读。**注意**：该文档对“性能”部分说的是“未来材料”，与 A20 的 msProf 一节不一致，以 A20 为准 |
| A25 | `_vendor/tilelang-ascend-v0.1.2.000/docs/tutorials/auto_tuning.md` | `AutoTuner.from_kernel` 与 carver 的 `MatmulTemplate.recommend_hints` | 已核验（2026-10-01）：整份读 |
| A26 | `_vendor/tilelang-ascend-v0.1.2.000/examples/aclgraph/README.md`、`rms_rope_aclgraph.py` | `torch.npu.NPUGraph()` / `with torch.npu.graph(g)` / `g.replay()`；四个 pass config；`capture model` 日志 | 已核验（2026-10-01）；未运行 |
| A27 | `_vendor/tilelang-ascend-v0.1.2.000/examples/torch_tl_ascend/README.md` | Python C 扩展集成：`torch.ops.tl_ascend.flash_attention`、`torch_tl_ascend.op_source.*`、libtorch 示例 | 已核验（2026-10-01）；未运行 |
| A28 | `_vendor/tilelang-ascend-v0.1.2.000/examples/print/elementwise_print.py` | `T.printf`/`T.dump_tensor` 的实际用法；**文件里仍用 `os.environ["TL_PTO_DEBUG"]="1"`** | 已核验（2026-10-01）：与 A22 存在口径不一致，见“待补资料”第 3 条 |
| A29 | `_vendor/tilelang-ascend-v0.1.2.000/examples/exception_dump_test/example_exception_dump.py` | 环境变量在 `import torch` 之前设置；`parse_exception_dump` 用法 | 已核验（2026-10-01）；未运行 |
| A30 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/tools/ascend_exception_dump_bin.py`（存在性） | `parse_exception_dump` / `read_msaicerr_bin` 的实现位置 | 已核验（2026-10-01）：文件存在；未精读实现 |
| A31 | `_vendor/tilelang-ascend-v0.1.2.000/VERSION` | 值为 `0.1.4`；与 tag `TileLang-ascend-v0.1.2.000-release` 不一致，**以 tag 为准**（见 VERSIONS.md） | 已核验（2026-10-01） |

“已核验”的含义：在锁定版本的工作树上用 `grep -n` / `sed` 定位到定义行并读完
签名与 docstring；行号区间与文件内容一致。**这不是运行验证。**

---

## corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md` 与各目录的
`PROVENANCE.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/cann/zh_cann.md` | https://www.hiascend.com/cann | CANN 的定位（对上支持 AI 框架、对下服务 AI 处理器与编程） | 已核验（2026-10-01）：本地文件存在，逐句比对 |
| R2 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/cann/zh_cann_driver.md` | https://www.hiascend.com/cann/driver | 驱动职责：设备管理、资源调度、Host/Device 通信、`npu-smi`/`hccn-tool` | 已核验（2026-10-01） |
| R3 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/cann/zh_cann_bisheng.md` | https://www.hiascend.com/cann/bisheng | 毕昇编译器：Host-Device 异构编译、设备侧代码编译成设备二进制；三方语言生态里列出 TileLang-Ascend | 已核验（2026-10-01） |
| R4 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/cann/zh_cann_ascend-c.md` | https://www.hiascend.com/cann/ascend-c | Ascend C 的定位：语言扩展层 + C++ 类库；CPU/NPU 孪生调试；PyAsc | 已核验（2026-10-01） |
| R5 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_0008_html.md` | https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/Ascendcopdevg/atlas_ascendc_10_0008.html | **基本架构**：AI Core 组成（Cube/Vector/Scalar）、AIC/AIV 分离模式、L1/L0A/L0B/L0C/UB/BT/FB 的作用、MTE1/MTE2/MTE3/FixPipe 的通路、两种典型数据流、`PipeBarrier` 与 `SetFlag`/`WaitFlag` 的语义 | 已核验（2026-10-01）：逐段读 |
| R6 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_00020_html.md` | https://www.hiascend.com/document/detail/zh/canncommercial/latest/programug/Ascendcopdevg/atlas_ascendc_10_00020.html | Ascend C 向量 add 的 inplace 写法：`TPipe`/`TQue`/`AllocTensor`/`EnQue`/`DeQue`/`DataCopy`/`Add`/`FreeTensor`；`EnQue`/`DeQue` 承担内存读写同步 | 已核验（2026-10-01）：逐段读；**未编译** |
| R7 | `../tilelang-corpus/06-huawei-ascend/raw/catlass/examples/00_basic_matmul/basic_matmul.cpp` | https://gitcode.com/cann/catlass | Ascend C GEMM 基线：`MmadAtlasA2Pingpong<true>`、`L1TileShape<128,256,256>`、`L0TileShape<128,256,64>`、`GemmIdentityBlockSwizzle<3,0>`、`GetCoreNumAic()`、host 侧 ACL 样板 | 已核验（2026-10-01）：本地为仓库快照，逐段读；**未编译** |
| R8 | `../tilelang-corpus/06-huawei-ascend/raw/catlass/examples/00_basic_matmul/README.md` | https://gitcode.com/cann/catlass | BasicMatmul 的参数、类型映射（fp16→fp32 累加等）、编译与运行方式 | 已核验（2026-10-01） |
| R9 | `../tilelang-corpus/06-huawei-ascend/raw/catlass/docs/zh/2_Design/01_kernel_design/00_basics/atlasA2_hardware_info.md` | https://gitcode.com/cann/catlass | **Atlas A2 容量**：L1 512KB、L0A/L0B 各 64KB、L0C 128KB、UB 192KB、BT 1KB、FB 7KB；各层级的搬入指令与支持格式 | 已核验（2026-10-01） |
| R10 | `../tilelang-corpus/06-huawei-ascend/raw/catlass/docs/zh/2_Design/01_kernel_design/04_matmul_summary.md` | https://gitcode.com/cann/catlass | CATLASS 的 Common / MultiCoreSplitK / SingleCoreSplitK 模板、L1/L0 约束、swizzle 判据（`<3,0>`） | 已核验（2026-10-01） |
| R11 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/CANNCommunityEdition/CANNCommunityEdition_920beta1_devaids_Profiling_atlasprofiling_16_2260_html.md` | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/Profiling/atlasprofiling_16_2260.html | msProf 的“性能调优建议”：通信瓶颈的 Wait Time Ratio / Synchronization Time Ratio 判据、慢链路判定 | 已核验（2026-10-01）：逐段读。**注意**：官方注明该功能“已不再演进”，更多分析见 `msprof-analyze` |
| R12 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/CANNCommunityEdition/`（`Profiling` 系列，如 `_0006`、`_0039`、`_0068`、`_0075`、`_0077`、`_1145`） | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/devaids/Profiling/ | Profiling 概述、PyTorch 训练场景快速入门、`op_statistic`（算子耗时）、`npu_mem`/`memory_record`（内存）等 | 已核验（2026-10-01）：文件存在、读过标题与关键段落；**未逐篇精读** |
| R13 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/huawei-techarticle-tilelang-npuir.md` | https://www.hiascend.com/developer/techArticles/20260506-1 | 华为自己的 TileLang+昇腾培训材料：TileLang 与 MLIR 基础、Pythonic 编程、手写 FlashAttention | 已核验（2026-10-01）：文件存在；**未逐段引用** |
| R14 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/PROVENANCE.md` | — | corpus 里 `tilelang-ascend` 快照的来源与 commit（`cbc67f2697e8`，2026-09-30） | 已核验（2026-10-01） |

---

## 关于性能数字的说明

**本章不提供任何昇腾上的实测数字。** 原因见开头的验证边界。本章出现的所有
“数量级/比例”都只有两类：

| 数字 | 来源 | 条件 |
| --- | --- | --- |
| 容量：L1 512KB / L0A 64KB / L0B 64KB / L0C 128KB / UB 192KB / BT 1KB / FB 7KB | R9 | **Atlas A2** 的公开文档数值，非实测 |
| 每核 dump/printf 上限 1 MB | A19 | 官方文档自述 |
| MSGIC / dump 目录结构 | A23 | 官方白皮书自述 |
| `128×256×4 = 128 KB`（fp32 累加） | **本书推导** | 由 R9 的 L0C 容量与 A11 的 `block_M/block_N` 约定推得 |
| CANN ≥ 8.3.RC1 / torch-npu ≥ 2.6.0.RC1（A 树）；CANN ≥ 9.3.0（exception dump）；CANN ≥ 9.0.0（A5 camodel） | A1 / A23 / A15 | 上游自述的**最低版本要求**，非实测 |
| `capture model=0/1` 的日志语义 | A26 | 官方示例自述 |
| msProf 各图名称（计算内存热力图 / Roofline / Cache 热力图 / 通算流水图 / 指令流水图 等） | A20 | 官方文档自述；**图形名称随 CANN 版本可能变化** |
| `KernelProfile` 的字段集合与“比值相对各自 total_cycles” | S17 / S21 | 源码与 skill 文档，非实测 |

**没有来源的、或只有宣传口径的数字一律未采用。** 任何需要真实加速器才能得到的数字
（实测算力利用率、延迟、吞吐）在本章中均未出现。

---

## 待补资料

1. **A 树与 corpus 快照不是同一个 commit，文档存在版本漂移。**
   `_vendor/tilelang-ascend-v0.1.2.000` 是 tag（commit `89a9364`），而
   `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend` 是
   `cbc67f2697e8`（2026-09-30，见 R14）。两者 `docs/TileLang-Ascend Programming Guide.md`
   行数不同（2596 vs 2560），内容也有差异。本章**所有行号都取锁定 vendor 树**；
   若你照 corpus 路径去查，定位可能偏移。建议后续版本在 corpus 的
   `PROVENANCE.md` 里显式标注“与锁定 tag 不一致”，或补一份 tag 的快照。
2. **corpus 缺 A 树的 tag 快照（AscendC/PTO 两条路线的 codegen 细节）。**
   本章只需要 A 树的**前端 API 与示例**，所以没有阻碍；但第 25 章（昇腾后端）
   若要讲 codegen，需要 A 树 `src/` 的对应文件。建议补抓
   `tile-ai/tilelang-ascend` 的 tag 快照，或至少 `ascendc_pto` / `npuir` 两个分支的
   `src/` 目录清单。
3. **`TL_PTO_DEBUG` 的口径在 A 树内部不一致。** `docs/tutorials/print.md` 与
   编程指南 §5.1 都说“优先用 kernel-scoped `compile_flags`，`TL_PTO_DEBUG` 只对
   `target="pto"`”，但 `examples/print/elementwise_print.py` 里仍然写着
   `os.environ["TL_PTO_DEBUG"] = "1"`（A28）。本章按文档口径写，并在正文 §10.1
   显式提示这处不一致。建议后续版本向上游提 issue 或补一条勘误。
4. **昇腾侧的实测基线完全缺失。**
   `tilelang/ascend/README.md` 提到一张 `images/ascend-950-throughput.png`
   （TileLang vs Torch NPU），但锁定 vendor 树与 corpus 里都没有这张图的**数值表**，
   也没有 Ascend 950 的硬件规格页（容量、算力、带宽）。因此本章 §8 只能给
   **判读方法**，给不出“TileLang 在昇腾上达到基线的百分之多少”。
   建议补抓 Ascend 950 的产品规格页与 `ascend_perf_regression.py` 的实测输出。
5. **`msprof op` / `msprof op simulator` 的真实输出样例缺失。**
   本章 §8.1 只引用了官方文档列出的图名（A20），没有一份真实的 msProf 输出目录树
   或截图，因此无法核对“输出的目录结构、文件名、图形名称”在不同 CANN 版本下的差异。
   建议在有设备的机器上采一次 `msprof op` 的完整输出目录清单。
6. **`_vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc` 的算法未精读。**
   本章只依赖“同步自动插入”这一事实（S14/S25），没有展开它怎么推导依赖。
   这是第 22 章（流水线、多缓冲与同步插入）的范围，本章不越界。
7. **A5 camodel 仿真脚本的路径未经复核。**
   §3.6 引用的 `.agents/skills/tilelang-a5-sim-convert/scripts/run_a5_sim_template.py`
   来自 A 树编程指南 §2.4（A15），但本章没有核对锁定 vendor 树里该路径是否存在。
   建议后续版本在 corpus 或 vendor 树里确认，或改引更稳定的官方入口。
