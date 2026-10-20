# 第 13 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[`../../VERSIONS.md`](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
正文只引用本文档锁定的这两个 tag，不使用任何含糊的版本指称。

写作环境：macOS (Apple Silicon)、Python 3.11（miniconda）；**没有 NVIDIA GPU，
也没有昇腾 NPU**。因此本章没有任何本机实测的性能或正确性结论。

两项本机跑过的验证（都可复现）：

* `code/01-ledger/ledger.py` —— 访存账本模型，输出见该目录 README；
* `code/07-api-audit/api_audit.py` —— API 名核对，两轮检查的输出见该目录 README。

核验状态只有三种口径：

* **已核验（静态）**：在锁定版本工作树上定位到该行并逐行阅读，行号与文件内容一致。
  **这不是运行验证。**
* **已核验（本地全文）**：corpus 里保存了该来源的完整本地副本，内容与链接一致。
* **未核验**：明确写出原因。

---

## 一、锁定的源码引用（`_vendor/`）

### 1.1 TileLang 主干 `v0.1.15` —— 官方示例

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/examples/elementwise/example_elementwise_add.py:11-32` | 官方逐元素 add：eager 风格（`T.const` + `T.empty` + 共享内存 + fragment） | 已核验（2026-10-01）：`sed -n` 逐行阅读；**未运行** |
| S2 | `_vendor/tilelang-v0.1.15/examples/norm/rms_norm.py:6-36` | `rms_norm_splitk`：两趟 + 倒序遍历的 cache 技巧 | 已核验（2026-10-01）；**未运行** |
| S3 | `_vendor/tilelang-v0.1.15/examples/norm/rms_norm.py:39-62` | `rms_norm`：整行放 fragment、读一遍写一遍 | 已核验（2026-10-01）；**未运行** |
| S4 | `_vendor/tilelang-v0.1.15/examples/norm/layernorm.py:6-58` | LayerNorm 前向：mean / rstd 两遍归约 + 仿射 | 已核验（2026-10-01）；**未运行** |
| S5 | `_vendor/tilelang-v0.1.15/examples/norm/layernorm.py:126-128` | 反传里用 `T.atomic_add` 跨 block 累加 dgamma/dbeta | 已核验（2026-10-01）；**未运行** |
| S6 | `_vendor/tilelang-v0.1.15/examples/cast/example_per_token_cast_to_fp8.py:8-41` | 分组量化：`T.reduce_absmax` + `T.clamp` + fp8 cast | 已核验（2026-10-01）；**未运行** |
| S7 | `_vendor/tilelang-v0.1.15/examples/ascend/example_vecadd.py:11-44` | 昇腾向量加：`T.Pipelined` + `T.annotate_buffer_versions` + SIMT/SIMD 两条路径 | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S8 | `_vendor/tilelang-v0.1.15/examples/ascend/example_rmsnorm.py:11-86`（重点 `:17-33`、`:36-84`） | 昇腾 RMSNorm：UB 预算 → `rows` 选择 → `T.SimtVF` + `alloc_reducer` | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S9 | `_vendor/tilelang-v0.1.15/examples/ascend/example_per_token_cast_to_fp8.py:17-86` | 昇腾 per-token FP8 量化：SIMT 与显式 SIMD 两种写法对照 | 已核验（2026-10-01）；**未在 NPU 上运行** |

### 1.2 TileLang 主干 `v0.1.15` —— 语言面（Python）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S10 | `_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:19,25-92` | `tl.tileop.reduce` 的 op key、`reduce()` 签名（含 `clear`/`batch`/`annotations`）与形状校验 | 已核验（2026-10-01） |
| S11 | `_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:94-171` | `reduce_macro` 的四个 scope 分支、`clear=False` 时先拷 `out` | 已核验（2026-10-01） |
| S12 | `_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:230-259` | `reduce_sum` 的 `clear` 语义说明（临时 buffer 那一段） | 已核验（2026-10-01） |
| S13 | `_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:358-460` | `reducer_init` / `reducer_update` / `finalize_reducer` 的 epoch 语义 | 已核验（2026-10-01） |
| S14 | `_vendor/tilelang-v0.1.15/tilelang/language/fill_op.py:11-72` | `T.fill` / `T.clear` 只是发射 `tl.tileop.fill` | 已核验（2026-10-01） |
| S15 | `_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:59-130` | `T.copy` 前端：region 归一化与 tile op 发射 | 已核验（2026-10-01） |
| S16 | `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:13-33,73-74` | `T.Parallel` 签名与 `coalesced_width` 注解 | 已核验（2026-10-01） |
| S17 | `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:49,67,81,296` | `alloc_shared` / `alloc_local` / `alloc_fragment` / `alloc_reducer` | 已核验（2026-10-01） |
| S18 | `_vendor/tilelang-v0.1.15/tilelang/language/eager/builder.py:1177-1200` | `T.const` —— **只在 eager（`@tilelang.jit`）模式可用** | 已核验（2026-10-01） |
| S19 | `_vendor/tilelang-v0.1.15/tilelang/language/symbolics.py:13-31` | `T.dynamic`（经典 `@T.prim_func` 风格里声明符号维度） | 已核验（2026-10-01） |
| S20 | `_vendor/tilelang-v0.1.15/tilelang/language/common.py:14-20`；`_vendor/tilelang-v0.1.15/tilelang/language/tir/ir.py:251` | `T.rsqrt` 等数学函数来自 `tir.ir` 的星号导入 | 已核验（2026-10-01） |
| S21 | `_vendor/tilelang-v0.1.15/tilelang/language/math_intrinsics.py:309-323` | `ieee_frsqrt` —— 与 `T.rsqrt` 的区别（快速倒数平方根 v.s. IEEE 版本） | 已核验（2026-10-01） |
| S22 | `_vendor/tilelang-v0.1.15/tilelang/transform/pass_config.py:141-146,352-356` | `TL_DISABLE_TMA_LOWER` 及其 deprecated 提示 | 已核验（2026-10-01）：`grep -n "TL_DISABLE_TMA_LOWER\|disable_tma_lower"` 定位 |

### 1.3 TileLang 主干 `v0.1.15` —— 编译期（C++）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S23 | `_vendor/tilelang-v0.1.15/src/op/reduce.cc:168-198` | ReduceOp 的 lowering 行为说明（线程内 + 跨线程 + workspace） | 已核验（2026-10-01） |
| S24 | `_vendor/tilelang-v0.1.15/src/op/reduce.cc:200-250` | `ReduceOpNode::Lower` / `InferLayout`（layout 冲突时报什么错）与 op 注册 | 已核验（2026-10-01） |
| S25 | `_vendor/tilelang-v0.1.15/src/tl_templates/cuda/reduce.h:211-253,279-296` | `tl::AllReduce` 的文档、四条 `static_assert`、butterfly 实现 | 已核验（2026-10-01） |
| S26 | `_vendor/tilelang-v0.1.15/src/transform/loop_vectorize.cc:1016-1066` | `ComputeBufferVectorSize`：越界谓词不变性、`min_vec_size`、store 不推进时标量化 | 已核验（2026-10-01） |
| S27 | `_vendor/tilelang-v0.1.15/src/transform/loop_vectorize.cc:966-1000` | 跨 dtype cast 的向量宽度 = 源/目标 lane 上限的 GCD | 已核验（2026-10-01） |
| S28 | `_vendor/tilelang-v0.1.15/src/transform/loop_vectorize.cc:1248-1281` | `IsExprInvariantInVectorBoundary` 与 `max_vector_bits` | 已核验（2026-10-01） |
| S29 | `_vendor/tilelang-v0.1.15/src/transform/legalize_safe_memory_access.cc:419-477` | 越界谓词插入的位置与「手工写 `if_then_else` 就跳过检查」的分支 | 已核验（2026-10-01） |
| S30 | `_vendor/tilelang-v0.1.15/src/transform/loop_partition.cc:112-130` | `PartitionLoop`：把一个并行循环按 layout 切到线程上 | 已核验（2026-10-01） |

### 1.4 TileLang 主干 `v0.1.15` —— 昇腾方言

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S31 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:15-23` | `alloc_shared` 在昇腾侧就是 **UB**；主干**没有** `alloc_ub` | 已核验（2026-10-01）：`rg -n "def alloc_ub" _vendor/tilelang-v0.1.15/` 无输出 |
| S32 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:139-160` | 昇腾 `T.Kernel`：一维 AI Core 网格，无 `threads=` | 已核验（2026-10-01） |
| S33 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py:107-140,146-190` | `T.SimtVF`（≤3 维线程域）与 `T.SimdVF`（无线程域） | 已核验（2026-10-01） |
| S34 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:10-40` | 昇腾安装前提（CANN / bisheng / `USE_ASCEND=ON`） | 已核验（2026-10-01）：本地全文 195 行；**未在 NPU 上运行** |
| S35 | `_vendor/tilelang-v0.1.15/tilelang/transform/pass_config.py`（全文） | **负向结论**：主干 v0.1.15 里没有 `TL_ASCEND_*` 这些键 | 已核验（2026-10-01）：`grep -n "ASCEND" tilelang/transform/pass_config.py` 无输出 |

### 1.5 TileLang-Ascend 适配器 `v0.1.2.000`

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S36 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/allocate.py:156-157` | `T.alloc_ub` 的定义（`scope="shared.ub"`） | 已核验（2026-10-01） |
| S37 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/__init__.py:86` | `from . import ascend_tile as tile` —— `T.tile` 命名空间的来源 | 已核验（2026-10-01） |
| S38 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/transform/pass_config.py:34-48` | `TL_ASCEND_AUTO_CV_COMBINE` / `TL_ASCEND_AUTO_CV_SYNC`（适配器专属 pass config） | 已核验（2026-10-01） |
| S39 | `_vendor/tilelang-ascend-v0.1.2.000/examples/elementwise/elementwise_add.py:18-48` | 适配器版向量加：`is_npu=True` → `(cid, vid)`、`T.Scope("V")`、`T.tile.add` | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S40 | `_vendor/tilelang-ascend-v0.1.2.000/examples/normalization/layer_norm.py:15-112` | 适配器版 LayerNorm：UB 上的 `T.tile.*` 全流程 + fp16↔fp32 cast | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S41 | `_vendor/tilelang-ascend-v0.1.2.000/examples/normalization/rms_norm.py:34-110` | 适配器版 RMSNorm：分块累加 + ping-pong + `T.tile.broadcast` | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S42 | `_vendor/tilelang-ascend-v0.1.2.000/examples/pos_embedding/rope.py:16-102,132-150` | 昇腾 partial RoPE：`dim_start`、`T.tile.gather` + mask 配对、interleaved 的 PyTorch 参考 | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S43 | `_vendor/tilelang-ascend-v0.1.2.000/examples/pos_embedding/rms_rope_fused.py:16-40` | 「RMSNorm + RoPE」融合的昇腾写法（本章 §3.3.4 的对照实现） | 已核验（2026-10-01）；**未在 NPU 上运行** |

---

## 二、corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/elementwise.md:26-131,153-176,214-252` | https://tilelang.com/ | 逐元素算子的四种写法；`N=2047` 时边界谓词导致标量化；显式 `T.copy` 到寄存器的写法 | 已核验（2026-10-01）：本地为站点源文件全文；**文档自带 experimental 警告** |
| R2 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/instructions.md:110-122,175-195` | https://tilelang.com/ | 归约与 scan 的清单；`T.alloc_reducer` / `T.finalize_reducer` / 快数学函数的语言面 | 已核验（2026-10-01） |
| R3 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/control_flow.md:73-90` | https://tilelang.com/ | `T.Parallel` 的语义与 `coalesced_width` / `loop_layout` 提示 | 已核验（2026-10-01） |
| R4 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/language_basics.md:164-182` | https://tilelang.com/ | `T.copy` 的编译期合并与 scope 相关 lowering；`T.async_copy` 不自动插 wait | 已核验（2026-10-01） |
| R5 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/kernels/norm/rms_norm.py:1-140`（重点 `:30-52`、`:72-74`） | https://github.com/tile-ai/TileOPs | 生产级 RMSNorm：per-thread partial、padding 列填 0 且分母用 N | 已核验（2026-10-01）：corpus 内为仓库工作树快照 |
| R6 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/kernels/rope.py:1-20` | 同上 | RoPE 的两种约定（NeoX / non-NeoX）与「频率表在 Op 层算」的分工 | 已核验（2026-10-01） |
| R7 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/formulas.py:910-923` | 同上 | `fused_qk_norm_rope_roofline`：Q/K RMSNorm + RoPE 的 (flops, bytes) 记账口径 | 已核验（2026-10-01） |
| R8 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/docs/design/roofline.md:16-66` | 同上 | 第 02 章采用的 roofline 记账口径原文（本章账本沿用同一口径） | 已核验（2026-10-01）：本地全文 269 行 |

---

## 三、资料之间的版本差异（本章特别标注）

本章有两处「同名不同版本 / 同一个东西两个名字」，都用命令核验过：

| 对象 | 主干 v0.1.15 | 适配器 v0.1.2.000 | 影响 |
| --- | --- | --- | --- |
| UB 的分配 API | `T.alloc_shared`（`_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:15-23`） | `T.alloc_ub`（`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/allocate.py:156`） | 把适配器的代码搬到主干会直接 `AttributeError`；本章代码里已用 `07-api-audit` 的负向断言锁住这条 |
| `T.tile.*` 命名空间 | **不存在**（`grep -n "as tile" tilelang/{language,ascend/language,cuda/language}/__init__.py` 无输出） | 存在（`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/__init__.py:86`） | 第 09 章也登记过同一条结论；本章的昇腾示例按方言分成两个文件 |
| 昇腾的自动同步配置 | 没有 `TL_ASCEND_*` 键 | `TL_ASCEND_AUTO_SYNC` 等必须显式打开（`pass_config.py:34-48`） | 示例里 `pass_configs` 不能照抄 |
| `T.Kernel` 签名 | 昇腾方言：`T.Kernel(*blocks) → 一维核网格` | `T.Kernel(blocks, threads=…, is_npu=…) → (cid, vid)` | 不是同一个函数 |
| `T.reduce_*` 签名 | `(buffer, out, dim, clear, batch, annotations)` | 多 `real_shape` / `tmp` 参数（见第 09 章的登记） | 写混了会报未知参数 |

复现方式（两条负向断言已经做成脚本参数）：

```bash
cd code/07-api-audit
python3 api_audit.py --root ../06-ascend \
    --vendor ../../../../_vendor/tilelang-v0.1.15 --label mainline \
    --only ascend_mainline_kernels.py --absent alloc_ub --absent-namespace tile
```

---

## 四、本章未采用的资料与建议来源（待补资料）

本章刻意**没有**给出下列内容，因为在本 corpus 与锁定源码里找不到可复核的依据。
这是「不知道就写不知道」的落实，不是遗漏：

| 缺什么 | 为什么需要 | 建议来源 |
| --- | --- | --- |
| 逐元素 / 归一化 / RoPE 在真实卡上的实测带宽与占用率 | 本章只能给账本下界与优化方向，给不出「融合后快了几倍」 | 作者在 H100/B200 与 Atlas 950 上跑本目录的示例，公开两种实现与 `do_bench` 口径 |
| `N=2047` 那类非对齐长度在不同线程数下的实测差异 | 官方文档只给了一个 N=2047 的生成代码，没有配套测量 | 对 `N` 取 2047/2048/4096、`num_per_thread` 取 1/2/4/8 做网格实验 |
| TileLang 在 `T.SimtVF` 里把 `T.Parallel` 向量化的具体规则（宽度、对齐要求） | 本章只能说明「套了 SimtVF 才会向量化」，给不出宽度判定条件 | 昇腾后端 codegen 源码 + `npusim`/真机的生成代码对照 |
| RMSNorm+RoPE 融合在真实工作负载里的端到端收益 | 单算子收益不等于端到端收益（还受 attention kernel 形状影响） | 在真实推理栈里做 A/B（第 18 章会涉及集成） |
| 主干把 `T.tile.*` 移植过来的计划 | 决定读者该以哪个仓库为准 | `tile-ai/tilelang` 与 `tile-ai/tilelang-ascend` 的 issue/PR；本章不做推测 |

---

## 五、引用规则提醒

* 源码引用一律写成 `_vendor/tilelang-v0.1.15/<相对路径>` 或
  `_vendor/tilelang-ascend-v0.1.2.000/<相对路径>`，并给出行号区间。
* 资料引用一律写成 `../tilelang-corpus/<类别>/raw/<文件>`。
* 本章所有「v0.1.15 里有没有某某」的结论都在锁定工作树上核验过；
  其中两条负向结论已做成 `code/07-api-audit/api_audit.py` 的可执行断言。
* 「已核验（静态）」≠「运行验证」。本章没有任何运行验证。
* 本章所有字节数与比值来自 `code/01-ledger/ledger.py` 这个**模型**，
  不是测量结果；引用时请连同「形状与 dtype」一起引用。
