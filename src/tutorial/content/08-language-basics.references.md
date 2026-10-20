# 第 08 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现任何含糊的版本表述（一律写具体 tag）。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章没有任何本机实测的运行结果或性能数字。

本章把来源分成三类：

* **锁定版本源码 / 官方文档快照**（`_vendor/tilelang-v0.1.15/`）：只用于
  「这个 API / 语义在 v0.1.15 里存不存在、签名是什么、文档怎么说」。
  在工作树上用 `rg` / `sed` 定位到定义处并完整读过；**不代表运行验证**。
* **适配器仓库**（`_vendor/tilelang-ascend-v0.1.2.000/`）：只在讨论
  `alloc_ub` / `alloc_L1` 等拼法差异时引用，与主库**分开标注**。
* **corpus 资料**（`../tilelang-corpus/`）：官方文档的镜像快照。本章引用的
  那几份与 `_vendor/` 里同名文件**内容一致**，差异只在部分非 ASCII 字符的
  编码（已 `diff` 验过：只有连字符、emoji 一类的 mojibake 行不同）。

---

## 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `tilelang/cuda/language/kernel.py:13-55` | CUDA 方言 `T.Kernel(*blocks, threads=, prelude=, cluster_dims=)` 的签名与 docstring | 已核验（2026-10-01）：逐行阅读 |
| S2 | `tilelang/language/kernel.py:88-151, 200-268, 392-427, 587-611` | `_normalize_bindings` / `_normalize_threads` / `_normalize_cluster_dims`、`launch_kernel`、`T.Kernel`/`T.ClusterKernel`、`get_block_binding` 与 `get_thread_binding` | 已核验（2026-10-01） |
| S3 | `src/ir.cc:236-262` | 位置参数 → `blockIdx.x/y/z` 的绑定帧；变量名 `bx`/`by`/`bz` | 已核验（2026-10-01） |
| S4 | `tilelang/transform/__init__.py:201-232` | `DEFAULT_SIMT_THREADS = 128` 与 `MaterializeKernelLaunch` 的参数语义（含「Ascend 不降线程域」的说明） | 已核验（2026-10-01） |
| S5 | `tilelang/ascend/language/kernel.py:139-172` | 昇腾 `T.Kernel` 是一维核网格、拒绝多维、无 `threads=` | 已核验（2026-10-01） |
| S6 | `tilelang/ascend/language/frame.py:107, 179` | `T.SimtVF(threads=..., latency=0)` 与 `T.SimdVF(latency=0)` | 已核验（2026-10-01） |
| S7 | `tilelang/language/dtypes.py:105, 120-142, 187-253, 262-317, 319-347, 633-711, 891` | `_CANONICAL_TO_DISPLAY_STR`、`_STR_TO_TVM_DTYPE_CALL`、`as_torch()`、fp4 家族判定、`validate_gemm_ab_dtypes`、全部 dtype 对象、`__all__` | 已核验（2026-10-01）：并用 `code/02-dtypes/dtype_families.py --diff` 与 `_all_dtypes` 做过集合比对，结果 `exact match`（173 条） |
| S8 | `tilelang/language/fp8.py:1-49` | `determine_fp8_type` / `determine_torch_fp8_type` 的平台选择逻辑 | 已核验（2026-10-01） |
| S9 | `tilelang/language/allocate.py:49-193, 254-289` | `alloc_shared`（含 bool workaround `:60-62`）、`alloc_local`、`alloc_fragment`、`alloc_var`、`alloc_global`、`alloc_barrier`、`alloc_tmem` | 已核验（2026-10-01） |
| S10 | `tilelang/ascend/language/allocate.py:15-80, 103-125` | 昇腾 `alloc_shared`（UB）、`alloc_l1`、`alloc_l0a`、`alloc_l0b`、`alloc_l0c`、`alloc_l0a_sf`、`alloc_l0b_sf` | 已核验（2026-10-01） |
| S11 | `tilelang/language/loop.py:13-357` | `Parallel` / `Persistent` / `Pipelined` / `serial` / `unroll` / `vectorized` 与三个大写别名；`order`/`stage` 与可重放 `Bind` 的规则 | 已核验（2026-10-01） |
| S12 | `tilelang/cuda/language/loop.py:16-79` | CUDA 方言的 `Parallel(prefer_async=)` 与 `unroll(unroll_factor=)` | 已核验（2026-10-01） |
| S13 | `src/transform/loop_partition.cc:112-350` | `PartitionLoop`、`LoopPartitioner::Partition`（`:248-271` 的映射公式）、`PlanLoopPartition`、`LowerParallelLoop`（`:324-350`） | 已核验（2026-10-01） |
| S14 | `tilelang/language/tir/ir.py:172-186, 228` | `Select` / `min` / `max` 的 docstring 与 `if_then_else` 的导出 | 已核验（2026-10-01） |
| S15 | `tilelang/language/logical.py:12-87` | `any_of(buffer)` / `all_of(buffer)` 的真实签名（与 `control_flow.md` 的例子不一致） | 已核验（2026-10-01） |
| S16 | `testing/python/language/test_tilelang_language_all_of.py:56, 105` | `T.all_of(<buffer slice>)` 的正确用法 | 已核验（2026-10-01） |
| S17 | `tilelang/language/eager/builder.py:115-128, 812-855, 1075` | `unwrap_cond`、`boolop` / `ifexp` 对 PrimExpr 的处理、eager 侧 `macro` | 已核验（2026-10-01） |
| S18 | `tilelang/language/parser/entry.py:105` | 解析器侧 `T.macro` | 已核验（2026-10-01） |
| S19 | `tilelang/language/customize.py:163-169` | `T.loop_break()` | 已核验（2026-10-01） |
| S20 | `tilelang/ascend/README.md:60-153` | 昇腾 Quick Start、`T.alloc_l1/l0c/alloc_shared`、`T.SimtVF`、`T.dual_copy`、「Use `T.alloc_shared` for Unified Buffer storage」（`:138`）、「Synchronization: None」对照表（`:151`） | 已核验（2026-10-01）：**未在 NPU 上运行** |
| S21 | `docs/programming_guides/type_system.md:1-41` | dtype 三种写法、常见标量类型、fp8/6/4 家族、lane 打包、累加 dtype 建议 | 已核验（2026-10-01） |
| S22 | `docs/programming_guides/python_compatibility.md:1-101` | Python 兼容性总表（控制流 / 数据访问 / 赋值 / 函数与类 / 语句与内建） | 已核验（2026-10-01） |
| S23 | `docs/programming_guides/control_flow.md:1-145` | 条件、循环、`while`、`break`/`continue`、`LegalizeSafeMemoryAccess` 的说明；**`T.all_of(i<M, j<N)` 的过时例子在 `:33-38`（`:141` 又用了一次）** | 已核验（2026-10-01）：已与 `logical.py` / 测试对照，确认不一致 |
| S24 | `docs/programming_guides/language_basics.md:26-172` | `@T.prim_func`、`T.Tensor`、`T.Kernel` 的注解规则、`T.dyn` / `T.dynamic`、内存作用域概述、向量加与 tiled GEMM 骨架 | 已核验（2026-10-01） |
| S25 | `docs/get_started/overview.md:76-90` | 「tile 是一等公民」、`T.alloc_shared` / `T.alloc_fragment` 的物理含义与 Layout Inference 的作用 | 已核验（2026-10-01） |
| S26 | `docs/deeplearning_operators/elementwise.md:28-135` | 「块级而非线程级」的解释；`N=2047` 时守卫阻止向量化的生成代码示例 | 已核验（2026-10-01） |
| S27 | `docs/tutorials/debug_tools_for_tilelang.md:52` | 「`T.copy` 先展开成 `T.Parallel`，再由 `LowerTileOP` 继续下降」 | 已核验（2026-10-01） |
| S28 | `examples/quickstart.py:1-60` | GEMM+ReLU 骨架、`T.Kernel(ceildiv(N,BN), ceildiv(M,BM))`、`T.max(C_local[i,j], 0)` 的整数字面量用法 | 已核验（2026-10-01）：API 逐个比对；**未运行** |

「已核验」的含义：在锁定版本的工作树上用 `rg` / `sed` 定位到定义处，完整读过
签名与 docstring；表中的行号区间与文件内容一致。**这不是运行验证。**

---

## 适配器仓库引用（`_vendor/tilelang-ascend-v0.1.2.000/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| A1 | `tilelang/language/allocate.py:126-157` | scope 对照注释（`shared` / `shared.l1` / `shared.ub` / `wmma.matrix_a` / `wmma.matrix_b` / `wmma.accumulator`）与 `alloc_L1` / `alloc_L0A` / `alloc_L0B` / `alloc_L0C` / `alloc_ub` 的定义 | 已核验（2026-10-01）：文件存在，逐行阅读；**未在 NPU 上运行** |
| A2 | `tilelang/language/__init__.py:47-51, 212-213` | 适配器仓库把上述五个函数导出到 `T.`，并在报错信息里提示可用名字 | 已核验（2026-10-01） |

**注意**：主库 `v0.1.15` 的 `tilelang/ascend/language/` 里**没有** `alloc_ub`。
本章在正文与图上都把两套拼法分开列出，避免读者混用。

---

## corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。
下表几份都是「官网 `_sources` 的原始 Markdown 快照」，与 `_vendor/` 同名文件
内容一致（差异仅在部分非 ASCII 字符的编码）。

| id | 本地路径 | 与 `_vendor/` 对应 | 核验状态 |
| --- | --- | --- | --- |
| R1 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/type_system.md` | `docs/programming_guides/type_system.md` | 已核验（2026-10-01）：`diff` 仅编码差异 |
| R2 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/python_compatibility.md` | `docs/programming_guides/python_compatibility.md` | 已核验（2026-10-01）：同上 |
| R3 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/control_flow.md` | `docs/programming_guides/control_flow.md` | 已核验（2026-10-01）：同上 |
| R4 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/language_basics.md` | `docs/programming_guides/language_basics.md` | 已核验（2026-10-01）：同上 |
| R5 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/elementwise.md` | `docs/deeplearning_operators/elementwise.md` | 已核验（2026-10-01） |
| R6 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tutorials/debug_tools_for_tilelang.md` | `docs/tutorials/debug_tools_for_tilelang.md` | 已核验（2026-10-01） |
| R7 | `../tilelang-corpus/02-tilelang-core-repo/` | `tile-ai/tilelang` 的完整 clone | 已核验（2026-10-01）：作为 `_vendor/tilelang-v0.1.15/` 之外的旁证路径；本章结论一律以 `_vendor/` 为准 |

---

## 上游链接（可达性未核验）

本章**没有**引用任何上游 URL 作为唯一依据：所有结论都落在 `_vendor/` 或
corpus 的本地文件里。这里只列出与本章内容对应、**未核验**的上游页面，
供读者按需自行确认。

| 主题 | 上游 URL | 核验状态 |
| --- | --- | --- |
| Python 兼容性 | https://tilelang.com/programming_guides/python_compatibility.html | 未核验（未抓取） |
| 类型系统 | https://tilelang.com/programming_guides/type_system.html | 未核验（未抓取） |
| 控制流 | https://tilelang.com/programming_guides/control_flow.html | 未核验（未抓取） |
| 快速上手（language basics） | https://tilelang.com/programming_guides/language_basics.html | 未核验（未抓取） |

---

## 本章的代码示例与验证级别

| 示例 | 验证级别 | 说明 |
| --- | --- | --- |
| `code/01-kernel-grid/grid_index_model.py` | CPU 逻辑验证 | 在写作机上实际运行过，输出抄进正文与 README |
| `code/01-kernel-grid/kernel_grid.py` | 仅静态核对 | `py_compile` 通过 + API 逐个核对；未运行 |
| `code/02-dtypes/dtype_families.py` | CPU 逻辑验证 | 实际运行过；`--diff` 与 vendor 的 `_all_dtypes` 完全一致（173 条） |
| `code/02-dtypes/fp8_fp4_exprs.py` | 仅静态核对 | `py_compile` 通过 + dtype 符号与 helper 逐个核对；未运行 |
| `code/03-memory-scopes/scope_footprint.py` | CPU 逻辑验证 | 实际运行过，输出抄进正文与 README |
| `code/03-memory-scopes/alloc_scopes.py` | 仅静态核对 | `py_compile` 通过 + `alloc_*` 签名逐个核对；未运行 |
| `code/04-loops/loop_mapping.py` | CPU 逻辑验证 | 实际运行过；复现 `loop_partition.cc` 的映射公式 |
| `code/04-loops/loops_demo.py` | 仅静态核对 | `py_compile` 通过 + 循环构造子签名逐个核对；未运行 |
| `code/05-control-flow/control_flow_demo.py` | 仅静态核对 | `py_compile` 通过 + 控制流 API 逐个核对；未运行 |
| `code/06-python-boundary/py_compat_scan.py` | CPU 逻辑验证 | 实际运行过；`--self-test` 5/5 通过 |

---

## 本章发现的文档 / 源码不一致

以下两条是写作过程中发现的**上游文档与锁定版本源码不一致**之处。本章按
「以源码为准 + 显式提示」处理，建议在 `audit/` 里登记，必要时向上游反馈。

| 位置 | 文档怎么说 | v0.1.15 源码实际是什么 | 本章处理 |
| --- | --- | --- | --- |
| `docs/programming_guides/control_flow.md:33-38`（又见 `:141`） | `if T.all_of(i < M, j < N):` | `logical.py:12,51` 的 `any_of` / `all_of` 接受 **buffer / buffer 切片**；测试也按缓冲区调用（`test_tilelang_language_all_of.py:56`） | 正文 §7.3 显式指出，并给出 `and` 的替代写法 |
| `docs/programming_guides/control_flow.md:33` | 「Short-circuit boolean ops are supported.」 | `eager/builder.py:820-846` 把 `and`/`or`/`not` 降级为 `tir.And` / `tir.Or` / `tir.Not`，**不是**求值短路 | 正文只写「`and`/`or` 可用并降级成 And/Or/Not」，不声称短路语义 |

---

## 未采用 / 待补资料

下面这些在 corpus 与 `_vendor/` 内都没有可靠出处，本章因此**没有**给出具体
数字或结论。这是「不知道就写不知道」的落实，不是遗漏。

| 缺什么 | 为什么需要 | 建议来源 |
| --- | --- | --- |
| 各型号昇腾 UB / L1 / L0A / L0B / L0C 的**官方容量** | 正文 §5 与 `scope_footprint.py` 只能拿「192 KiB」当示例值 | 各型号《Ascend C 算子开发指南》架构规格章节，或真实设备上的 `GetCoreMemSize` 实测 |
| NV 各代 shared memory / 寄存器文件在**具体 tile 形状下的 occupancy 曲线** | 正文 §5.3 只能给出「128 寄存器/线程」这一个算术结论 | NVIDIA Occupancy Calculator / tuning guide，或真实设备上的 occupancy API |
| `T.Parallel` 的 `vector_size` **由谁、按什么规则选定**的第一手说明 | 正文 §6.3 只给出了映射公式，`vector_size` 的来源要靠 `VectorizeLoop` 的访存分析推断 | 上游 `src/transform/loop_partition.cc` 的 `VectorizeLoop` 注释，或维护者的 issue 答复 |
| `float8_e4m3` 与 `float8_e4m3fn` 在**昇腾后端**上如何映射 | 正文 §4.3 只覆盖了 CUDA / ROCm 两条路线 | 昇腾后端文档或真实设备上的 dtype round-trip 实验 |
| NVIDIA 与 Ascend 两棵树上 **`T.gemm` 参数集合**的完整差异 | 本章只用到 `transpose_B` / `clear_accum` 两个参数 | 第 9 章（核心原语）会展开，本章不重复 |

---

## 引用规则提醒

* 源码引用一律写成 `_vendor/tilelang-v0.1.15/<相对路径>` 或
  `_vendor/tilelang-ascend-v0.1.2.000/<相对路径>`，并给出行号。
* 资料引用一律写成 `../tilelang-corpus/<类别>/raw/<文件>`。
* 「这个 API 在不在 v0.1.15 里」这类判断，一律以 `_vendor/tilelang-v0.1.15/`
  的实际文件为准；本章的 `alloc_ub` 结论就是这样得出的
  （`ls` 与 `rg` 都确认主库 `tilelang/ascend/language/` 下没有该名字）。
