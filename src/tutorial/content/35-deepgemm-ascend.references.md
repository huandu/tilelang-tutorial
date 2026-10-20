# 第 35 章 参考资料

本章的结论来自两类材料：

1. **DeepSeek 2026-09-30 昇腾开源清单里的仓库快照**（corpus
   `14-deepseek-ascend-stack/raw/`）：`DeepGEMM-Ascend`（本章主角）、
   `DeepGEMM`（NVIDIA 对照物）、`DeepJIT`（JIT 运行时）；
2. **锁定版本的 TileLang 源码**：`_vendor/tilelang-v0.1.15/`
   （第 6 节读 `hc_prenorm.py` 时要对照它的 M 树 API）。

版本纪律：

* DeepGEMM-Ascend / DeepGEMM / DeepJIT 都不是 `_vendor/` 里的锁定树，而是 corpus
  快照；commit 见 corpus 的 `PROVENANCE.md`，本章引用时一律写
  `DeepGEMM-Ascend/…`，站点会把它换成上游 GitHub 链接（含 commit）。
  * `DeepGEMM-Ascend` → `8491bbb4b8c0`（2026-09-30）
  * `DeepGEMM` → `057ca5964aae`（2026-09-30）
  * `DeepJIT` → `3732a3b9a0a2`（2026-09-30）
* TileLang 侧一律以 `_vendor/tilelang-v0.1.15/`（`v0.1.15`，
  commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）为准。
* 不引用未锁定的开发分支；讨论 `main` 上更新的东西时会显式标注。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章所有 kernel/C++ 结论的最高验证级别是「**仅静态阅读**」
（逐行读过、行号与文件内容一致、语法用 `py_compile` 过），
**没有任何一条性能数字来自本机实测**。

---

## 一、DeepGEMM-Ascend（本章主角，corpus 快照）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| D1 | `DeepGEMM-Ascend/README.md:1-11` | 定位：完全兼容 DeepGEMM API、支持 BF16/FP8/FP4/MQA/MegaMoE；「轻量抽象隐藏 fractal 布局、对齐、地址、参数转换」；稀疏加载与协程流水 | 已核验（2026-10-01）：逐段读中英原文 |
| D2 | `DeepGEMM-Ascend/README.md:19-31` | Requirements：Ascend 950 系列、CANN 9.20（`bin/bisheng`、`bin/ld.lld`）、`torch_npu`、Python ≥ 3.10、C++20 `<format>`、`tilelang`（HC prenorm 依赖）、`tree-sitter` | 已核验（2026-10-01）：逐行读 |
| D3 | `DeepGEMM-Ascend/README.md:48-50` | **唯一被标注的不兼容点**：每两个 UE8M0 打包成 `int16`、MN-major | 已核验（2026-10-01）：原文引用 |
| D4 | `DeepGEMM-Ascend/README.md:52-53` | 兼容承诺只覆盖 kernel 接口（「refer to DeepGEMM's interfaces」） | 已核验（2026-10-01） |
| D5 | `DeepGEMM-Ascend/README.md:54-60` | 工具函数清单（`set_num_sms`、`set_npu_arch`、`set_mk_alignment...`、`transform_sf_...`、`get_paged_mqa_logits_metadata`、`aclnn_*` 参考实现） | 已核验（2026-10-01）：逐条读 |
| D6 | `DeepGEMM-Ascend/README.md:62-78` | 环境变量清单：`DG_JIT_DEBUG`、`DG_PRINT_CONFIGS`、`DG_JIT_CACHE_DIR`（默认 `$HOME/.dj`）、`DG_JIT_PRINT_COMPILER_COMMAND`、`DG_JIT_PRINT_LOAD_TIME`、`DG_JIT_KERNEL_DEBUG_INFO`、`DG_JIT_LAUNCH_TIMEOUT`、`ASCEND_LAUNCH_BLOCKING`、`ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` | 已核验（2026-10-01）：逐条读 |
| D7 | `DeepGEMM-Ascend/README.md:84-87` | 性能表口径：Ascend 950DT、CANN 9.20、`bench_msprof`、**冷 L2**、形状取自 DeepGEMM test suite | 已核验（2026-10-01） |
| D8 | `DeepGEMM-Ascend/README.md:90-170` | 六张性能表（Dense / FP8 推理 / M-Grouped / MQA / MegaMoE / HC Prenorm） | 已核验（2026-10-01）：数值逐个抄入正文与 `code/02-perf-ledger/`，并用复算脚本交叉核对 |
| D9 | `DeepGEMM-Ascend/README.md:186-190` | Acknowledgements：JIT 由 DeepJIT 提供；**mHC kernel backed by TileLang** | 已核验（2026-10-01）：原文引用 |
| D10 | `DeepGEMM-Ascend/csrc/python_api.cpp:25-42` | 扩展模块名 `_C`；8 个命名空间的注册入口 | 已核验（2026-10-01）：逐行读 |
| D11 | `DeepGEMM-Ascend/deep_gemm/__init__.py:15-17,25-79,100-101` | `ASCEND_HOME_PATH`/`ASCEND_TOOLKIT_HOME` 的 import 时探测；Python 侧导出清单；`tf32_hc_prenorm_gemm` 的导出 | 已核验（2026-10-01）：逐行读 |
| D12 | `DeepGEMM-Ascend/csrc/apis/gemm.hpp:45-58` | grouped 的形状约定（A `[M,K]`、B `[G,N,K]`、D `[M,N]`；k-grouped 的 `transpose(0,1)` 说明） | 已核验（2026-10-01） |
| D13 | `DeepGEMM-Ascend/csrc/apis/gemm.hpp:60-182` | `get_gemm_desc`：从 torch 张量推出 `GemmDesc`（major、stride、gemm_type、compiled_dims） | 已核验（2026-10-01）：逐段读 |
| D14 | `DeepGEMM-Ascend/csrc/apis/gemm.hpp:184-241` | `fp8_fp4_gemm` 的参数文档：`(A, sfA)` 元组、`recipe*`、`compiled_dims`、`disable_ue8m0_cast`、`alpha`；支持的三种 dtype 组合 | 已核验（2026-10-01） |
| D15 | `DeepGEMM-Ascend/csrc/apis/gemm.hpp:537-662` | `register_apis`：所有 kernel 名与默认参数；`:591`、`:615` 两行注释说明 masked 变体未实现 | 已核验（2026-10-01）：逐行读 |
| D16 | `DeepGEMM-Ascend/csrc/apis/layout.hpp:19-36,37-65,68-92,113-128` | SF 的输出形状/主序说明、`check_sf_layout`（`gran_k==32`、短整型 packing）、`transform_sf_into_required_layout`、`get_recipe` 默认 `(1,32)` | 已核验（2026-10-01）：逐行读 |
| D17 | `DeepGEMM-Ascend/csrc/jit_kernels/transform_sf.hpp:15-105` | SF 转换的 host 侧：物理形状 `[K/64, MN]` 再转置、分块参数、L2 hint 环境变量、JIT 源码字符串与 launch | 已核验（2026-10-01）：逐段读 |
| D18 | `DeepGEMM-Ascend/deep_gemm/include/deep_gemm/transform_sf.hpp:13-21,216-360,455-641` | device 侧：fp32 尾数校验、`pack_validate_*_simd_impl`（shift 23/15 + or）、`transform_sf` kernel 的 UB 分阶段与 flag | 已核验（2026-10-01）：读了函数签名与关键段，未逐行读完 641 行 |
| D19 | `DeepGEMM-Ascend/deep_gemm/include/deep_gemm/common.hpp:50-69,133-182` | 硬件常数（`kNumAICores=32`、L1/L0A/L0B/L0C/UB 容量、`FRAC_MN=16`、`MX_SF_DIVISOR=64`）；`gm_ptr<T, Major>` | 已核验（2026-10-01）：逐行读 |
| D20 | `DeepGEMM-Ascend/deep_gemm/include/deep_gemm/ascend.hpp:1-12` | MAD 抽象层的入口与三文件分工注释 | 已核验（2026-10-01） |
| D21 | `DeepGEMM-Ascend/deep_gemm/include/deep_gemm/ascend/tile_ptr.hpp:19-57,93-97` | `operand_ptr` / `result_ptr` / `as_mad_aligned`；`l1_ptr`/`l0a_ptr`/`l0b_ptr`/`l0c_ptr`/`ub_ptr` 别名 | 已核验（2026-10-01）：逐行读 |
| D22 | `DeepGEMM-Ascend/deep_gemm/include/deep_gemm/ascend/copy.hpp:9-20,102-145,160-198,232-264,266-310,371-380` | `set_l0c_nz2dn`/`set_mte2_nz`、`copy_gm_to_l1`（ND2NZ）、`copy_l1_to_l0a/b`、`copy_gm_to_l1_mx`、`fill_tail_block`、`infer_quant_mode`、`copy_l0c_to_ub/gm`、`mad()` | 已核验（2026-10-01）：逐段读 |
| D23 | `DeepGEMM-Ascend/deep_gemm/include/deep_gemm/ascend/sync.hpp:11-27,29-57` | `PipeEvent` + `notify`/`wait`；`set_intra_block`/`wait_intra_block`；`IntraBlockEvent` | 已核验（2026-10-01）：逐行读 |
| D24 | `DeepGEMM-Ascend/csrc/jit/jit.hpp:14-91` | `AscendJIT`、`Runtime`（`num_sms`/`arch`/`dry_run`/`deterministic`）、`kMkAlignment=256`、`init_jit`（include 目录、bisheng 额外 flag、`DG_JIT_WITH_LINEINFO`） | 已核验（2026-10-01）：逐行读 |
| D25 | `DeepGEMM-Ascend/csrc/jit_kernels/gemm_config.hpp:16-21,39-127,131-172` | `get_compiled_dim`、`GemmDesc`、`GemmConfig`、`GemmArgs`，以及两个 formatter（`set_num_sms`/`arch` 的行为见 D24） | 已核验（2026-10-01）：逐行读 |
| D26 | `DeepGEMM-Ascend/csrc/jit_kernels/heuristics.hpp:22-71,92-250,252-465,504-573,575-719` | `HWDesc` 与 `get_hw_desc`、`make_gemm_candidate`、`GemmComputeCost`、`get_gemm_cost`、`can_direct_store`、`refine_gemm_pipeline`、`get_gemm_k_cost`、`select_gemm_config`（含 `DG_JIT_DEBUG`/`DG_PRINT_CONFIGS` 打印点 `:714-718`） | 已核验（2026-10-01）：逐段读完；数值被 `code/03-config-search/` 复算 |
| D27 | `DeepGEMM-Ascend/csrc/jit_kernels/bf16_gemm.hpp:13-80` | 源码字符串模板 → `jit->compile` → `jit->launch` 的完整例子；`dry_run` 分支 | 已核验（2026-10-01）：逐行读 |
| D28 | `DeepGEMM-Ascend/csrc/jit_kernels/mqa_logits.hpp:11-103` | `MQALogitsDesc`/`Config`、`launch_mqa_logits`、paged 版与 metadata 的 launch | 已核验（2026-10-01）：逐段读 |
| D29 | `DeepGEMM-Ascend/csrc/apis/attention.hpp:17-105,107-140` | MQA 的 host 侧：`logits_stride` 按 512B 对齐、SF 形状校验、config（`split_kv=512`、`mad_m=64/heads*heads`、`num_cd_stages=2`、`num_logits_stages=3`） | 已核验（2026-10-01）：逐段读 |
| D30 | `DeepGEMM-Ascend/csrc/jit_kernels/mega_moe.hpp:11-51` | MegaMoE 的 launch：模板参数即 workspace 维度、每次 launch `aclrtMemsetAsync` 清 progress | 已核验（2026-10-01）：逐行读 |
| D31 | `DeepGEMM-Ascend/csrc/apis/mega_moe.hpp:25-62,72-110,112-126,128-225` | `get_workspace` 的断言、`SymmBuffer`（HCCL 对称缓冲）、切片视图、权重布局校验、`fp8_fp4_mega_moe`（`:159` 的 `num_sms == kNumAICores` 断言、`:145-150` 的能力裁剪） | 已核验（2026-10-01）：逐段读 |
| D32 | `DeepGEMM-Ascend/deep_gemm/include/deep_gemm/mega_moe.hpp:20-107` | MegaMoE 的流水线选择、L1/L0 布局表、AIC/AIV0/AIV1 的 pipe event 常量 | 已核验（2026-10-01）：读了 1-130 行与节选 |
| D33 | `DeepGEMM-Ascend/deep_gemm/include/deep_gemm/layout/mega_moe.hpp:12,26-33,85-118` | `BLOCK_M=BLOCK_N=256`、四个 `ProgressStage`、`Workspace` 的 GM 布局与 `FP8Buffer` | 已核验（2026-10-01）：逐段读 |
| D34 | `DeepGEMM-Ascend/deep_gemm/mega/__init__.py:9-60,84-156,159-184` | Python 侧 `SymmBuffer`、权重布局变换（`_interleave_weights`/`_reshape_weights`/`_scale_fp4_weight_sf`）、`fp8_fp4_mega_moe` | 已核验（2026-10-01）：逐段读 |
| D35 | `DeepGEMM-Ascend/deep_gemm/include/deep_gemm/fp8_gemm.hpp:9-77` | dense FP8 GEMM 的模板参数与主序约定（K/MN-major、SF 恒为 MN-major、D 恒为 N-major） | 已核验（2026-10-01）：读了头部 80 行 |
| D36 | `DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py` | **本章的 TileLang 落点**；正文 §6 逐段读：`:13-57` `get_config`、`:59-104` 签名与阶段常数、`:105-135` 混核与缓冲、`:136-292` 主循环（`:154` 流水、`:170` token 阶段、`:195-267` SIMD、`:268` dual_copy、`:273-292` gemm）、`:294-356` store 与核间同步、`:362-397` 对外包装 | 已核验（2026-10-01）：逐行读完（文件共 397 行） |
| D37 | `DeepGEMM-Ascend/deep_gemm/testing/bench.py:28-66,263-266,284-360` | `KernelProfile` 的字段（含 `aic_fixpipe = L0C -> out`）、`_make_l2_flush_buffer`、`bench_msprof` 的预热/迭代/冷 L2 实现 | 已核验（2026-10-01）：逐段读 |
| D38 | `DeepGEMM-Ascend/tests/common.py:82-195` | ACLNN 对拍的四条提前返回与原因；`run_gemm_test` 的流程（`par_compile` → 对拍 → `bench_msprof` → `print_perf`） | 已核验（2026-10-01）：逐段读 |
| D39 | `DeepGEMM-Ascend/tests/utils.py:8-51` | `calc_diff`（余弦相似度）与 `assert_equal` 的两档判据（threshold 0 = 逐位相等）；`count_bytes` 只统计非 None 张量 | 已核验（2026-10-01）：逐行读 |
| D40 | `DeepGEMM-Ascend/tests/test_hyperconnection.py:14-27,59-107` | HC prenorm 的形状表（`n=24`、`k=16384/20480/28672/7680/7168`）、数值判据、确定性检查、`count_bytes(a,b,d,sqr_sum)` 的口径 | 已核验（2026-10-01）：逐段读 |
| D41 | `DeepGEMM-Ascend/tests/test_mega_moe.py:22-36,60-94` | SF 的造法（uint8 → `.view(torch.int16)`）、TileLang SwiGLU 基线（`third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py`）的加载方式 | 已核验（2026-10-01）：逐段读 |
| D42 | `DeepGEMM-Ascend/third-party/tilelang_ops/swiglu_apply_weight_to_fp8.py:25-105` | MegaMoE 的 TileLang 基线实现（SwiGLU + 量化 + packed ue8m0 SF），`CastOutputConfig` 与 `sf_block=(1,32)` | 已核验（2026-10-01）：读了头部与关键段；**未逐行读完 449 行** |
| D43 | `DeepGEMM-Ascend/csrc/apis/aclnn.hpp:19-104,122-260,262-294` | `aclnn_*` 参考实现：packed fp4 的逻辑视图、`convert_sf_for_aclnn`、支持矩阵表、`cublaslt_gemm_*` 别名 | 已核验（2026-10-01）：逐段读 |
| D44 | `DeepGEMM-Ascend/csrc/apis/runtime.hpp:12-80` | `set_num_sms`/`set_npu_arch`/`set_mk_alignment...`/`set_dry_run`/`use_deterministic_algorithms`/`npu_sleep` 的注册 | 已核验（2026-10-01）：逐行读 |
| D45 | `DeepGEMM-Ascend/csrc/utils/layout.hpp:37-86` | `major_check`、`get_major_type_ab`、`check_major_type_cd`、`gemm_early_return`、`get_shape_by_spec` | 已核验（2026-10-01）：逐行读 |
| D46 | `DeepGEMM-Ascend/PROVENANCE` 所在的 corpus 文件（`14-deepseek-ascend-stack/raw/PROVENANCE.md`） | 各仓库快照的 commit 与日期 | 已核验（2026-10-01）：逐行读 |

---

## 二、NVIDIA 对照物与 JIT 运行时（corpus 快照）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| N1 | `DeepGEMM/README.md:59-70` | NVIDIA 侧的接口约定：命名 `D = C + A @ B`、NT 布局、SM90 用 FP32 scale、SM100 用 packed UE8M0（**4 个打进一个 `torch.int`**） | 已核验（2026-10-01）：逐行读 |
| N2 | `DeepGEMM/README.md:72-88` | Normal / grouped（contiguous）/ grouped（masked）/ K-grouped 的接口划分 | 已核验（2026-10-01） |
| N3 | `DeepGEMM/README.md:88-112` | MQA 两个版本（paged / non-paged）、输入输出形状与公式 | 已核验（2026-10-01） |
| N4 | `DeepGEMM/README.md:112-146` | Mega MoE 的定义与用法（对称内存、权重变换、`fp8_fp4_mega_moe`） | 已核验（2026-10-01） |
| N5 | `DeepGEMM/README.md:146-160` | NVIDIA 侧工具函数清单（`set_tc_util`、`set_pdl`、`set_ignore_compile_dims`、`get_mn_major_tma_aligned_tensor` 等） | 已核验（2026-10-01）：逐条读 |
| N6 | `DeepJIT/README.md:3-52` | DeepJIT 的双后端定位、缓存 key 组成、共享缓存与原子发布 | 已核验（2026-10-01）：逐段读 |
| N7 | `DeepJIT/README.md:54-80` | 仓库布局与集成方式；Ascend 后端需要的工具链（bisheng / ld.lld / adv_api） | 已核验（2026-10-01） |
| N8 | `DeepJIT/include/deep_jit/utils/env.hpp:28-82` | `get_env` 的解析规则与 `Env` 的 `DG_* → DJ_*` 回退 | 已核验（2026-10-01）：逐行读 |
| N9 | `DeepJIT/include/deep_jit/backend/ascend/device.hpp:22-56` | 架构号来自 `ACL_DEV_ATTR_NPU_ARCH`；`get_npu_arch()` / `get_num_sms()` | 已核验（2026-10-01）：逐行读 |
| N10 | `DeepJIT/include/deep_jit/backend/ascend/kernel.hpp:219` | `ASCEND_LAUNCH_BLOCKING` 的实现位置 | 已核验（2026-10-01）：读到该行 |

---

## 三、TileLang 侧（锁定树 `_vendor/tilelang-v0.1.15/`）

本章第 6 节与第 7 节用到以下 M 树接口，全部逐个核对过存在性与签名。

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| T1 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:125-155`（对照表在 `:141-153`） | 与 Ascend C 的分工对照表；「自动调度不保证把任意 `T.Parallel` 张量化」 | 已核验（2026-10-01）：逐段读（第 19 章 S4 用 130-155 的宽范围核验过同一份文档） |
| T2 | `.../tilelang/ascend/language/kernel.py:176` | `T.MixedKernel` | 已核验（2026-10-01）：`grep -n "def MixedKernel"` |
| T3 | `.../tilelang/ascend/language/frame.py:107,179` | `T.SimtVF(threads, latency)` / `T.SimdVF(latency)` | 已核验（2026-10-01） |
| T4 | `.../tilelang/ascend/language/copy_op.py:53` | `T.dual_copy` | 已核验（2026-10-01） |
| T5 | `.../tilelang/ascend/language/annotations.py:19` | `T.annotate_buffer_versions` | 已核验（2026-10-01） |
| T6 | `.../tilelang/ascend/language/mode.py:63` | `T.set_atomic`（另有 `set_atomic_none`、`set_hf32_mode`） | 已核验（2026-10-01） |
| T7 | `.../tilelang/ascend/language/schedule_hint.py:13` | `T.PerCoreTask` | 已核验（2026-10-01） |
| T8 | `.../tilelang/ascend/language/allocate.py:15,26,37,46,55` | `alloc_shared`（UB）/`alloc_l1`/`alloc_l0a`/`alloc_l0b`/`alloc_l0c` | 已核验（2026-10-01） |
| T9 | `.../tilelang/ascend/language/gemm_op.py:168` | M 树 `T.gemm` | 已核验（2026-10-01） |
| T10 | `.../tilelang/ascend/language/sync.py:74,93` | `T.ascend_sync_inter_arrive` / `ascend_sync_inter_wait` | 已核验（2026-10-01） |
| T11 | `.../tilelang/layout/ascend.py:39` | `make_ascend_compact_nz_layout` | 已核验（2026-10-01） |

「已核验」的含义：在锁定工作树上用 `grep -n` / `sed` 定位到定义并读完签名；
行号与文件内容一致。**这不是运行验证。**

---

## 四、本章示例代码

| id | 路径 | 它交付什么 | 验证级别 |
| --- | --- | --- | --- |
| C1 | `code/01-python-api/deepgemm_ascend_call.py` | BF16 dense / FP8 dense / M-grouped 三类最小调用，含 SF 的 `.view(torch.int16)` 造法 | **仅静态核对**：`py_compile` 通过；函数名与参数对照 D15；未运行 |
| C2 | `code/02-perf-ledger/perf_ledger.py` | 硬件极限复算、六张表的利用率、FP8 推理与 HC Prenorm 的带宽复算 | **CPU 逻辑验证**：实跑过，退出码 0（输出见该目录 README） |
| C3 | `code/03-config-search/config_search.py` | 从 D26 提炼的容量模型：L0C/L0A/L0B/L1 约束下的候选枚举 | **CPU 逻辑验证**：实跑过，退出码 0 |
| C4 | `code/04-hc-prenorm-tilelang/hc_prenorm_sketch.py` | D36 的可读骨架，逐段标出原文件行号 | **仅静态核对**：`py_compile` 通过；原语对照 T2–T11；未运行 |

---

## 五、关于性能数字的说明

**本章所有性能数字都来自 D8（README 的六张表）或 C2/C3 的复算。**
没有任何一条来自本机实测（本机无 GPU/NPU）。

| 数字 | 来源 | 条件 / 口径 |
| --- | --- | --- |
| Dense / FP8 推理 / M-Grouped / MQA / MegaMoE / HC Prenorm 的全部 latency、TFLOPS、GB/s | D8 | Ascend 950DT、CANN 9.20、`bench_msprof`、冷 L2（D7、D37）；MegaMoE 另有 EP8 + top-k=6 + 1 个 shared expert、8 rank 平均 |
| bf16 432 / fp8 865 / fp4 1730 TFLOPS（「Hardware limit」） | D8；**本书复算** | 由 `get_mad_flops_per_cycle × 1650 MHz × 32` 复算（D26），误差 < 0.2%；见 C2 |
| 3.5 TB/s、4.5 TB/s、1650 MHz、8192 FLOP/cycle/核、128 MB L2、128 B 扇区、SF pair = 64 | D26（`get_hw_desc`） | **源码里的硬件模型**，不是厂商规格页；corpus 里没有 Ascend 950 的规格文档可供对照 |
| 「MQA 饱和 FIX pipe 到 99%」 | D8 | 官方定性描述；`aic_fixpipe` 字段的定义见 D37（`L0C -> out`），但本章**没有**任何一份真实 profile 输出可以核对这 99% |
| 「HC Prenorm nearly saturates the HBM write bandwidth」 | D8 | 官方措辞；本书按 `count_bytes(a,b,d,sqr_sum)` 复算得到 0.994–1.002 的比值，**但复算出来的是「总流量/时间」，其中 99% 是 A 的读**——与「写带宽」的措辞不一致，见「待核」第 3 条 |
| FP8 推理表带宽列与复算的比值 1.089–1.095 | **本书复算** | C2；口径差异，见「待核」第 2 条 |
| L0C 256 KB / UB 256 KB / L1 512 KB / L0A·L0B 64 KB | D19 | **Ascend 950 口径**（源码常数）；与第 19 章引用的 Atlas A2 规格（L0C 128 KB / UB 192 KB）不同，不是矛盾 |

**没有来源的、或只有宣传口径的数字一律未采用。**

---

## 六、待核 / 资料不足

1. **`DG_NPU_ARCH` 不存在。** 本章任务描述里提到的这个环境变量，在锁定快照里
   grep 不到任何命中；架构只有两个来源：设备上报（`ACL_DEV_ATTR_NPU_ARCH`，N9）
   或 `deep_gemm.set_npu_arch(int)`（D24、D44）。正文 §3.2 已显式更正，
   并说明「若官方后续加了同名变量，请以源码为准」。
2. **FP8 推理表的带宽列口径未给出。** 14 行「表格 / 复算」比值稳定在
   1.089–1.095（同一张表内高度一致），但 corpus 快照里没有该列的计算公式。
   我**没有**把它解释成「profiler 实测 GM 流量」之类的猜测，只在正文 §5.3 与
   `code/02-perf-ledger/README.md` 里登记了事实。建议后续版本补一条官方口径说明，
   或采一次同形状的 `count_bytes` 实测值。
3. **「HC Prenorm nearly saturates the HBM write bandwidth」这句措辞与示例代码
   的计数口径不一致。** 按 `tests/test_hyperconnection.py:103-107` 的
   `count_bytes(a, b, d, sqr_sum)`，`M=8192` 那一行的 473 MB 里 99.4% 是 A 的读、
   写不到 1 MB。建议后续版本确认官方这句是「读+写总带宽」的简写，
   还是另有写路径（例如 mHC 后续步骤的写）。
4. **corpus 里没有 Ascend 950（DT）的硬件规格页。** 因此
   `HWDesc` 的 3500/4500 GB/s、1650 MHz、8192 FLOP/cycle/核这些数字
   **无法与厂商文档交叉验证**；正文只能标注它们是「源码里的模型值」。
   建议补抓 Ascend 950 的产品规格与 CANN 的 per-core 带宽文档。
5. **没有任何真实的 msProf / `KernelProfile` 输出样本。** 正文 §5 的瓶颈判断依赖
   README 的定性描述（FIX 99%）与源码里字段的定义（D37），
   但没有一份真实 profile 可以核对「prefill 真的是 FIX 主导」。
   建议在有设备的机器上采一份 `bench_msprof(..., return_all_kernels=True)`
   的完整输出。
6. **`third-party/tilelang_ops/` 的来源与使用范围需要确认。** 它不在
   `DeepGEMM-Ascend/.gitmodules` 里，仓库源码里除了
   `tests/test_mega_moe.py:74-94` 之外没有别的引用点；
   它的 `swiglu_apply_weight_to_fp8.py:25` 自称「Adapted from TileKernels (MIT)」。
   本章据此把它描述为「MegaMoE 的 TileLang 基线实现」，没有把它说成量产路径。
7. **`deep_gemm/include/deep_gemm/transform_sf.hpp` 与 `mega_moe.hpp` 未逐行读完。**
   D18（641 行）只读了校验、打包与 kernel 骨架；D32（1221 行）只读了 1-130 行
   与若干节选。凡引用处都给了行号，未读部分不作结论。
8. **mHC 的算法背景不在本章范围。** `tests/test_hyperconnection.py:14` 注释说
   `n = hc_mult * (hc_mult + 2)`、`k = hc_mult * hidden`，与第 34 章讲的
   流形超连接相关；本章只按「瘦长 GEMM + 行平方和」处理，不展开算法。
9. **本章的 kernel 名硬编码。** `code/02-perf-ledger/` 里 `aic_clock_mhz=1650`
   等常数是照抄 D26 的；如果上游改了硬件模型，脚本的复算值会失效，
   但 README 的表格不会——两者要对上，仍需人工核对（`code/03` 的结论同理）。

---

## 七、关于 `scripts/tutorial/check-content.mjs` 的输出说明

本章写完后运行了

```bash
node scripts/tutorial/import-book.mjs --only 35-deepgemm-ascend
node scripts/tutorial/check-content.mjs 35-deepgemm-ascend
```

该工具校验的是**结构与可渲染性**：图表 JSON 的合法性与 `id` 一致性、
每张源图是否有对应规格、自测题的题数与选项结构。
它**不检查**内容正确性、引用可达性或行号准确性——那部分由上面几张表逐条人工核验。
