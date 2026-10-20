# 第 09 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[`../../VERSIONS.md`](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现模糊的版本指代，一律写具体 tag 与 commit。

写作环境：macOS (Apple Silicon)、Python 3.11（miniconda）；**没有 NVIDIA GPU，
也没有昇腾 NPU**。因此本章没有任何本机实测的性能或正确性结论。

引用分三类，核验状态只有三种口径：

* **已核验（静态）**：在锁定版本工作树上 `rg`/`sed` 定位到该行并逐行阅读，行号与文件
  内容一致。**这不是运行验证。**
* **已核验（本地全文）**：corpus 里保存了该来源的完整本地副本，内容与链接一致。
* **未核验**：明确写出原因。

---

## 一、锁定的源码引用（`_vendor/`）

### TileLang 主干 `v0.1.15`（`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:59-130` | `T.copy` 前端：extent 归一化、标量快速路径、`tl.tileop.copy` 发射 | 已核验（2026-10-01）：`sed -n '59,130p'` 逐行阅读 |
| S2 | `_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:133-185,188-229,232-324` | `T.copy_cluster` / `T.async_copy`（cp.async 语义）/ `T.tma_copy`（半序列 TMA） | 已核验（2026-10-01） |
| S3 | `_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:525-562` | `T.transpose` 前端 | 已核验（2026-10-01） |
| S4 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/copy_op.py:19-73` | CUDA 侧 `disable_tma` / `eviction_policy` / `prefer_instruction` 三个提示 | 已核验（2026-10-01） |
| S5 | `_vendor/tilelang-v0.1.15/src/op/copy.cc:33-90,449-481,481-545,615,622,640` | 通用 copy lowering、`MakePredicate` 越界谓词、`MakeSIMTLoop` 安全值、三个 op 注册 | 已核验（2026-10-01）：含 `:514-520`（safe value）与 `:523-525`（目标侧 `IfThenElse`） |
| S6 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy.h:21-33,44-56` | `CopyInst` 枚举（kNormal/kLDSM/kSTSM/kBulkLoad/kCPAsync/kTMem…）与 `TMADesc.oob_fill` | 已核验（2026-10-01） |
| S7 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:195-225,387-400,407-425,427-445,716-808` | TMA bulk load 约束、1D bulk、LDSM/STSM/TMEM 判据、cp.async 前置条件、`SelectCopyInstForLowering` 判定链 | 已核验（2026-10-01） |
| S8 | `_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:149-155,528-547,549-562,778-807` | `GetDisableTMA`/`GetIsTmaCopy`、layout 推导期先选指令、`loop_layout` 与指令的互斥检查、`Copy::Lower` 分派 | 已核验（2026-10-01） |
| S9 | `_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:38-122,160-210` | `T.gemm` 前端：13 槽位协议、M/N/K 推导与静态性检查、`clear_accum` 改 access region、签名 | 已核验（2026-10-01） |
| S10 | `_vendor/tilelang-v0.1.15/tilelang/tileop/base.py:5-12` | `GemmWarpPolicy` 三个取值 | 已核验（2026-10-01） |
| S11 | `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:34-38,40-76,80-93,336-374,375-395` | 指令名常量、`CheckWgmma`、`AllowTcgen5Mma`/`AllowWgmma`、`SelectInst`、warp 划分 | 已核验（2026-10-01） |
| S12 | `_vendor/tilelang-v0.1.15/src/ascend/op/gemm.cc:24-45` | 昇腾侧 `SelectInst` 恒为 `ascend.mad`、warp 划分恒为 `(1,1)` | 已核验（2026-10-01） |
| S13 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/gemm_op.py:89-107,168-210` | 昇腾 L0A/L0B/L0C → 专用 L0 MAD 组装；`unit_flag_ctrl` | 已核验（2026-10-01） |
| S14 | `_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:13-16,25-173,176-259,479-524` | `T.reduce_*` 前端、scope 分支矩阵、`clear` 语义、`warp_reduce_*` | 已核验（2026-10-01） |
| S15 | `_vendor/tilelang-v0.1.15/src/op/reduce.cc:78-110,185-188,212-217,224-247,245` | `ReduceOp` 构造、workspace 注释、lowering 入口、layout 冲突检查、op 注册 | 已核验（2026-10-01） |
| S16 | `_vendor/tilelang-v0.1.15/src/tl_templates/cuda/reduce.h:211-238,240-253,279-295` | `tl::AllReduce` 文档、模板与 4 条 `static_assert`、butterfly 实现（≥32 shared / <32 shuffle） | 已核验（2026-10-01） |
| S17 | `_vendor/tilelang-v0.1.15/tilelang/language/atomic.py:13-23,39-122,125-208,211-311,314-387,390-492,495-506` | 原子操作的标量/tile-region 两条路径、`memory_order` id 映射、`return_prev` 限制、`atomic_addx2/x4`、load/store/or | 已核验（2026-10-01） |
| S18 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/atomic.py:12-50` | CUDA 侧 `use_tma`（sm90+ TMA `cp.reduce`） | 已核验（2026-10-01） |
| S19 | `_vendor/tilelang-v0.1.15/src/op/atomic_add.cc:94-129,138`；`src/op/atomic_reduce.cc:162,167`；`src/cuda/op/atomic_add.cc:36-62` | `AtomicAdd` 节点与注册、`AtomicMax/Min` 注册、TMA 原子加的 dtype 白名单 | 已核验（2026-10-01） |
| S20 | `_vendor/tilelang-v0.1.15/tilelang/language/fill_op.py:11-72`；`src/op/fill.cc:94-127,152-172,214` | `T.fill`/`T.clear` 前端与 lowering（并行循环） | 已核验（2026-10-01） |
| S21 | `_vendor/tilelang-v0.1.15/tilelang/language/annotations.py:69-74` | `T.annotate_safe_value`（越界安全值） | 已核验（2026-10-01） |
| S22 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py:53-163,230-372` | `T.dual_copy` 语义与四条路径；昇腾 `T.copy` 的 `pad_value` 与 `P2 + PIPE_MTE2` 同步 | 已核验（2026-10-01） |
| S23 | `_vendor/tilelang-v0.1.15/src/ascend/op/copy.cc:573-605,1182`；`src/ascend/transform/rewrite_dual_copy.cc:97-107,368-377` | L0C→UB 双目标搬运的 lowering、`ascend_copy` 注册、`RewriteDualCopy` pass 与支持的路径 | 已核验（2026-10-01） |
| S24 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py:29-62,107-143,179-195` | `T.Cube`/`T.Vector`、`T.SimtVF`（≤3 维线程域）、`T.SimdVF`（无线程域） | 已核验（2026-10-01）：文件共 206 行 |
| S25 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/__init__.py:24,30-40,65,67,84-129` | 昇腾语言面：`copy`/`dual_copy`、reduce 家族、frame 家族、`T.simd`；**没有 `tile` 命名空间** | 已核验（2026-10-01）：`grep -n "as tile" tilelang/ascend/language/__init__.py` 无输出；`find . -name "*ascend_tile*"` 无输出 |
| S26 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/mode.py:63-105` | `T.set_atomic(op, dtype)` / `T.set_atomic_none()`：有状态的 store-mode 原子；支持 dtype 集合 | 已核验（2026-10-01） |
| S27 | `_vendor/tilelang-v0.1.15/src/ascend/op/builtin.cc:446,451`；`src/ascend/codegen/codegen_ascend.cc:401-403,904-940,1893-1910` | `ascend_set_atomic`/`ascend_set_atomic_none` 的注册与 codegen（`SetAtomic{Add,Max,Min}`）、`SIMT_VF`/`SIMD_VF` block | 已核验（2026-10-01） |
| S28 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:60-155` | 昇腾 950 Quick Start（`T.Persistent` + `T.Pipelined` + `T.gemm` + `T.dual_copy` + `T.SimtVF`）、写法约定 | 已核验（2026-10-01）：文件存在，逐行阅读；**未在 NPU 上运行** |
| S29 | `_vendor/tilelang-v0.1.15/examples/norm/rms_norm.py:14-36`；`examples/ascend/example_vecadd.py`；`examples/ascend/example_rmsnorm.py:36-86` | 本章代码示例的原型（`T.clear`/`T.copy`/`T.reduce_sum`、`T.SimtVF`/`T.SimdVF`） | 已核验（2026-10-01）：API 逐个比对；**未运行** |
| S30 | `_vendor/tilelang-v0.1.15/tilelang/language/builtin.py:321,334,358,370,457,624,900,963,1386` | `tma_load`/`tma_store_arrive`/`tma_store_wait`/`mbarrier_wait_parity`/`barrier_arrive`/`sync_threads`/`get_warp_idx`/`cp_async_barrier_noinc`；`ptx_wait_group` 经 `tilelang/cuda/language/tir.py:19` 导出 | 已核验（2026-10-01） |

### TileLang-Ascend 适配器 `v0.1.2.000`（`_vendor/tilelang-ascend-v0.1.2.000/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S31 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend_tile.py:313-341,343-386,389-410,412-450,452-606,608-705,707-855,884-910,967-1150,1150-1470,1572-1620,1616-1660,1661-1711,1837-1926,1976-2310,2302-2410` | `T.tile.*` 全量：`atomic_add`/`fill`/`clear`/`arith_progression`/`sort`/`merge_sort`/`topk`/`gather*`/`select`/`brcb`/`binary_op` 与一元、标量、compare、transpose、broadcast | 已核验（2026-10-01）：`grep -n "^def "` 得到全部定义行，逐段阅读；**未运行** |
| S32 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/__init__.py:86` | `from . import ascend_tile as tile`，`T.tile` 命名空间的来源 | 已核验（2026-10-01） |
| S33 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/reduce_ascend.py`（`def reduce_sum` 起） | 适配器版 `T.reduce_*` 的 `real_shape` / `tmp` 扩展 | 已核验（2026-10-01）：`grep -n "def reduce_sum"` 与签名阅读 |
| S34 | `_vendor/tilelang-ascend-v0.1.2.000/src/op/ascend.cc:95-110,740-780,975-1260` | `tl.ascend_atomic_add` 的约束（dst 必须 global、src 必须 UB/L0C）、`tl.ascend_*` 内建 op 的定义 | 已核验（2026-10-01） |
| S35 | `_vendor/tilelang-ascend-v0.1.2.000/src/target/codegen_ascend.cc:522-630` | `ascend_add`/`compare`/`gather`/`transpose`/`sort`/`topk`/`select` 的 C++ 代码生成分支 | 已核验（2026-10-01） |
| S36 | `_vendor/tilelang-ascend-v0.1.2.000/src/tl_templates/ascend/common.h:255-290` | `atomic_add_ub_to_gm` / `atomic_add_l0c_to_gm`：`AscendC::SetAtomicAdd<T>()` 包裹的写回 | 已核验（2026-10-01） |
| S37 | `_vendor/tilelang-ascend-v0.1.2.000/docs/language_ref/tilelibrary.md:40-63,64-100,133-175,176-187,206-209` | 适配器语言参考：`T.copy`（含“越界补零”）、`T.gemm` 约束、reduce、`tmp` arena 规则、`T.clear T.fill`“转成 `T.Parallel`” | 已核验（2026-10-01）：本地全文 217 行 |
| S38 | `_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md:481-520,601-700,740-935,989-1160,1397-1520,1502-1540` | Developer / Expert 两种模式、数据搬运与计算原语、reduce 的 `clear=False` merge 语义、`T.tile.*` 数学计算汇总表 | 已核验（2026-10-01）：本地全文 2596 行；**未在 NPU 上运行** |
| S39 | `_vendor/tilelang-ascend-v0.1.2.000/.agents/skills/tilelang-ascend-tile-api/SKILL.md:1-60` | `T.tile.xxx` 的开发流程与“优先新增 Ascend 专属 API、不要改全局 API”的设计约定 | 已核验（2026-10-01） |
| S40 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend_tile.py` 全文 3112 行 | 章节内所有 `T.tile.*` 行号的分母 | 已核验（2026-10-01）：`wc -l` |

> 说明：S37 里该 tag 自带的 `docs/api_docs/` **只有** `T.copy.md` 与 `T.gemm_v0.md`
> 两个文件；本章引用的其余 `T.tile.*` 逐 API 文档来自 corpus 的同一仓库快照（见 R6），
> 其分支/commit 与锁定 tag 不同。两个快照的差异已在“三、资料之间的版本差异”中登记。

---

## 二、corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/instructions.md` | https://tilelang.com/ | 原语分类；`T.copy` 的同步语义与“可降级为 SIMT/LDSM/TMA/cp.async”；`T.async_copy` 不自动插 wait；`T.gemm`/`T.gemm_blockscaled` 后端归属 | 已核验（2026-10-01）：本地为站点源文件全文 |
| R2 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/language_basics.md:164-182` | https://tilelang.com/ | `T.copy` 用法与“编译期做合并与 scope 相关 lowering” | 已核验（2026-10-01） |
| R3 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/matmul.md:60,95-105,198-207,245-248` | https://tilelang.com/ | `T.copy`/`T.gemm`/`T.Pipelined` 组合的官方 GEMM 骨架 | 已核验（2026-10-01） |
| R4 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/elementwise.md:100-115,214-254` | https://tilelang.com/ | 边界检查的代价；显式 `T.copy` 到寄存器的写法 | 已核验（2026-10-01）；**文档自带 experimental 警告** |
| R5 | `../tilelang-corpus/07-tilelang-ascend-adapter/SUMMARY.md` | — | 昇腾两条后端路线、适配器仓库定位、编程指南骨架（Developer = tile library / Expert = thread+搬运原语） | 已核验（2026-10-01）：本地全文 |
| R6 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/api_docs/`（20 个 `.md`：`T.copy`、`T.gemm_v0`、`T.reduce_{sum,max,min}`、`T.tile.{add,sub,mul,div,select,compare,gather,sort,merge_sort,topk,transpose,broadcast,atomic_add,arith_progression,createvecindex}`） | https://github.com/tile-ai/tilelang-ascend（`ascendc_pto` 分支） | `T.tile.*` 的逐 API 语义、dtype/rank 支持、约束与示例 | 已核验（2026-10-01）：本地为该快照的仓库工作树；快照分支 `ascendc_pto @ cbc67f2`，**与 VERSIONS 锁定的 `v0.1.2.000-release` tag 不是同一个 commit** |
| R7 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/language_ref/tilelibrary.md` | 同上 | `T.copy` 的“Zero will be padded if we detect the load is out of boundary”、`T.gemm` 的 K 轴 32 倍数约束、workspace arena 规则 | 已核验（2026-10-01）：本地 188 行；**与 R6 不同分支** |
| R8 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/TileLang-Ascend Programming Guide.md` | 同上 | §4.1 Developer 模式（内存/搬运/计算/调度原语）、§4.2 Expert 模式、reduce 的 `clear=False` merge 规则 | 已核验（2026-10-01）：本地全文 |
| R9 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/tilelang/language/ascend_tile.py` | 同上 | R6 文档对应的前端实现（3263 行，与 S31 的 3112 行不同版本） | 已核验（2026-10-01）：`diff` 确认与锁定 tag 版本不同 |
| R10 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/lower_trace.md` | https://tilelang.com/ | `TL_LOWER_TRACE`：每个 pass 的前后 IR diff + 最终 codegen 源码落盘（本章 §5.3 第 2 条路径） | 已核验（2026-10-01）：本地为站点源文件全文 |
| R11 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tools/layout_visualization.md` | https://tilelang.com/ | `tilelang.tools.plot_layout` 与 layout 推导可视化（本章 §5.3 第 3 条路径） | 已核验（2026-10-01） |

---

## 三、资料之间的版本差异（本章特别标注）

本章有三处必须让读者知道的“同名不同版本”，全部已用 `diff`/`grep` 核验：

| 对象 | 锁定版本（本章引用行号以此为准） | corpus 快照 | 影响 |
| --- | --- | --- | --- |
| `tilelang/language/ascend_tile.py` | `_vendor/tilelang-ascend-v0.1.2.000/`：3112 行 | corpus `ascendc_pto @ cbc67f2`：3263 行 | `T.tile.*` 的函数集合与行号可能不同；本章行号一律取锁定版本 |
| `docs/api_docs/` | 只有 `T.copy.md`、`T.gemm_v0.md` | 20 个 `.md`（含全部 `T.tile.*`） | `T.tile.*` 的逐 API 语义只能引 corpus 路径（R6） |
| `T.reduce_*` 签名 | v0.1.15：`(buffer, out, dim, clear, batch, annotations)`；适配器：多 `real_shape`、`tmp` | — | 写 `real_shape=`/`tmp=` 在主干 v0.1.15 上会报未知参数 |

---

## 四、本章未采用的资料与建议来源

本章刻意**没有**给出下列内容，因为在本 corpus 与锁定源码里找不到可复核的依据。
这是“不知道就写不知道”的落实，不是遗漏：

| 缺什么 | 为什么需要 | 建议来源 |
| --- | --- | --- |
| `T.copy` 各降级路径在真实硬件上的实测吞吐/延迟 | 本章只能给“指令类别”，不能给性能结论 | 作者在 H100/B200 与 Atlas 350 上跑成对的 micro-benchmark，并公开代码与测量口径 |
| 昇腾 `T.tile.sort`/`topk` 的临时 workspace 实际容量 | 只拿到“保守启发式”公式（`library.md:64-100`），没有官方最小值 | 华为 CANN 对应版本的 `LocalTensor` 排序接口文档，或真实设备上的容量实测 |
| `T.dual_copy` 在 L0C→UB 双目标模式下的有效带宽增益 | README 只说“两个 AIV 各处理一半”，没有数字 | 昇腾 950 上的 N 分裂/M 分裂对照实验 |
| v0.1.15 里 `T.tile.*` 是否已有移植计划 | 决定读者该以哪个仓库为准 | `tile-ai/tilelang` 与 `tile-ai/tilelang-ascend` 的 issue/PR；本章不做推测 |

## 五、引用规则提醒

* 源码引用一律写成 `_vendor/tilelang-v0.1.15/<相对路径>` 或
  `_vendor/tilelang-ascend-v0.1.2.000/<相对路径>`，并给出行号区间。
* 资料引用一律写成 `../tilelang-corpus/<类别>/raw/<文件>`。
* 本章所有“v0.1.15 里有没有某某”的结论都在锁定工作树上核验过；核验命令写在对应行。
* “已核验（静态）”≠“运行验证”。本章没有任何运行验证。
