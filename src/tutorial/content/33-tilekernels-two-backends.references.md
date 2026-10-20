# 第 33 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[`../../VERSIONS.md`](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
本章正文没有引用 A 树（适配器），只把第 19 章的 A 树结论当作对照背景。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章**没有任何本机实测的性能或正确性结论**。所有 kernel 示例的最高验证级别是
**仅静态核对**（`python3 -m py_compile` 通过 + 逐个在源码里比对 API 名与签名）。

本章的引用分三类：

| 类别 | 记号前缀 | 核验状态口径 |
| --- | --- | --- |
| TileLang 锁定源码 | `S…` | **已核验（2026-10-01）**：在 `_vendor/tilelang-v0.1.15/` 工作树上用 `rg -n` / `sed` 定位并逐行阅读，行号与文件内容一致 |
| TileKernels 语料快照 | `T…` | **已核验（2026-10-01）**：corpus 完整快照，逐文件阅读 |
| 官方发布说明与 corpus 元数据 | `R…` | **已核验（2026-10-01）**：本地全文 |

「已核验」**不是**运行验证。任何「跑起来对不对 / 快不快」的结论本章都没有做。
`code/03-map-and-diff/` 的第 3 步对 21 个 API 名做了机械存在性核对
（输出 `21/21 命中；缺：无`），但它只证明「名字存在」。

---

## 一、本章示例代码（`code/`）

| id | 文件 | 用途 | 验证级别 |
| --- | --- | --- | --- |
| C1 | `code/01-topk-gate-minimal/topk_gate_cuda_min.py` | §4.2 的最小可读 CUDA 实现（改写自 T3） | 仅静态核对（py_compile + API 比对）；**未运行** |
| C2 | `code/01-topk-gate-minimal/topk_gate_asc_min.py` | §4.3 的最小可读 Ascend 实现（改写自 T4） | 仅静态核对；**未运行** |
| C3 | `code/02-call-the-library/call_topk_gate.py` | §3.4 的端到端调用 + 对拍（仿 T20） | 仅静态核对；**未运行** |
| C4 | `code/02-call-the-library/call_quant_and_rope.py` | §5、§6 的端到端调用 | 仅静态核对；**未运行** |
| C5 | `code/03-map-and-diff/map_and_diff.py` | 三层结构清点 + 结构化对照 + API 存在性核对 | **CPU 逻辑验证**：在 macOS / Python 3.11 上**实际运行过**，输出见该目录 README |

---

## 二、TileKernels 语料快照（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/`）

快照信息：`git clone --depth 1`，commit `66258df6175d`（2026-09-30，
HEAD 摘要 `Merge pull request #34 from deepseek-ai/v2.0.0`），见
`../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md`（R2）。
**这是一个 commit 快照，不是 tag**；如果上游发布新版本，行号可能漂移。

### 2.1 库定位与结构

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| T0 | `README.md:1-31`（定位、News、Features、Requirements） | §1.1：库的定位、2026-09-30 昇腾支持、TileLang ≥ 0.1.15、Ascend 950 + CANN ≥ 9.2.0 | 已核验（2026-10-01）：逐行读 |
| T1 | `README.md:47-62`（Testing）、`:73-89`（Project Structure） | §2.1：测试方式与目录结构 | 已核验（2026-10-01） |
| T2 | `tile_kernels/__init__.py:1-39`、`tile_kernels/moe/__init__.py:1-4`、`tile_kernels/quant/__init__.py:1-11`、`tile_kernels/transform/__init__.py:1`、`tile_kernels/rand/__init__.py:1-4` | §2.1：包对外暴露的算子清单 | 已核验（2026-10-01） |

### 2.2 运行期配置与后端分派

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| T3 | `tile_kernels/config.py:1-134` | §3.1：`is_ascend()`（`:24-26`）、`get_device()`（`:29-31`）、`get_device_num_sms()`（`:39-46`）、核数访问器（`:62-71`）、`get_max_smem_per_sm`（`:74-77`）、`get_max_ub_per_vector_core`（`:80-83`）、PDL（`:7,105-112`）、clamp 默认值（`:10-15`） | 已核验（2026-10-01）：逐行读 |
| T4 | `tile_kernels/utils.py:1-38` | `align` / `ceil_div` / `dtype_to_str` 等共享小工具 | 已核验（2026-10-01） |

### 2.3 top-k gate 家族（本章核心对照样本）

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| T5 | `tile_kernels/moe/topk_gate_kernel.py:1-43`（全文件） | §3.2：外层分派逐行读；tie-break 契约在 `:22-24` | 已核验（2026-10-01）：逐行读 |
| T6 | `tile_kernels/moe/topk_gate_cuda.py:1-68`（全文件） | §4.2：CUDA 版逐段（`T.Kernel` `:22`、fragment `:25-27`、reducer `:28,41-46,54-58`、`T.copy` `:64`、PDL `:23-24,65-66`、`pass_configs` `:7-11`） | 已核验（2026-10-01）：逐行读 |
| T7 | `tile_kernels/moe/topk_gate_asc.py:1-157`（全文件） | §4.3：Ascend 版逐段（配置 `:9-23`、网格与缓冲 `:30-39`、流水 `:41-48`、SIMD 主体 `:50-105`、慢路径 `:106-151`、`mem_bar` `:151`、收尾 `:153-155`） | 已核验（2026-10-01）：逐行读 |
| T8 | `tile_kernels/moe/moe_topk_gate_forward_kernel.py:116-149`；`..._cuda.py:8-13,15-21,78-91,205-250`；`..._asc.py:9-13,69-99,123-124,179-278` | §3.2、§4.4 的第二个样本：同一套分派模板、`warp_reduce_sum` 宏、`T.SimtVF`/`T.Persistent` 组合 | 已核验（2026-10-01）：分派段与关键段逐行读；未逐行读完全部 293/281 行 |
| T9 | `tile_kernels/moe/moe_topk_gate_backward_kernel.py:9-40`；`..._cuda.py:1-5`；`..._asc.py:1-7` | §3.3：反向算子沿用同一套三层与两棵树约定 | 已核验（2026-10-01）：头部与文件头读 |
| T10 | `tile_kernels/moe/normalize_weight_kernel.py:8-37`；`normalize_weight_asc.py:1-48` | §4.5 补充样本：Ascend 侧也可以在 `T.SimtVF` 里用 fragment + `T.reduce_sum` | 已核验（2026-10-01）：全文件读 |
| T11 | `tile_kernels/moe/sqrt_softplus.py:1-40` | §2.2、§7.3：同一数学近似、两个后端各一份实现（CUDA 用 `T.log1p/T.exp`，Ascend 用 9 阶多项式 + Horner） | 已核验（2026-10-01）：前 40 行；多项式系数表在 `:10-20` |

### 2.4 量化算子族

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| T12 | `tile_kernels/quant/common.py:19-61`（配置 dataclass）、`:64-75`（sf 类型与 UE8M0 打包因子）、`:78-123`（输入配置）、`:126-171`（输出配置与 clamp）、`:174-203`（逻辑/物理宽度、sf 形状）、`:206-248`（sf 分配与 epilogue）、`:251-316`（`get_sf_and_inv` / `load_sf` / `store_sf`） | §5.1、§5.2、§5.6：粒度配置、打包因子后端差异、列主序对齐差异、尺度与倒数 | 已核验（2026-10-01）：逐段读 |
| T13 | `tile_kernels/quant/per_token_cast_kernel.py:24-96,99-142,144-219`；`per_token_cast_cuda.py:28-33,96-105`；`per_token_cast_asc.py`（头部与 `T.Kernel(num_vec_cores)` / `Persistent` / `S.*` 段） | §5.3：per-token 的约束（`num_per_channels ∈ {16,32,64,128}`、`hidden % 64 == 0`）与分派 | 已核验（2026-10-01）：外层与 CUDA 头部逐行读；Ascend 侧读了 `T.Kernel`/`Persistent`/`S.*` 的关键段，**未逐行读完全部 370 行** |
| T14 | `tile_kernels/quant/per_block_cast_kernel.py:30-98,101-134`；`per_block_cast_asc.py:12-42` | §5.3：per-block 的对齐/整除约束（`:33-34`、`:67`）与昇腾只支持 `(32,32)`（`:22`）、`block_k` 取 `align(gcd(512或1024, hidden), 128)`（`:26-27`） | 已核验（2026-10-01）：外层逐行读；Ascend 前 80 行逐行读 |
| T15 | `tile_kernels/quant/per_channel_cast_kernel.py:10-67`；`per_channel_cast_asc.py:13-90` | §5.3：per-channel 的 `num_per_tokens ∈ {32,128}`、昇腾只支持 32 | 已核验（2026-10-01）：外层逐行读；Ascend 前 90 行逐行读 |
| T16 | `tile_kernels/quant/cast_kernel.py:8-59`；`cast_back_kernel.py:9-64` | §5.4、§6：逐元素 cast 与反量化；`use_e4m3_sf` 仅 CUDA（`common.py:118`、`cast_back_kernel.py:41`）；`x.device.type == 'npu'` 与 `is_ascend()` 并存（`cast_back_kernel.py:46` vs `cast_kernel.py:40`） | 已核验（2026-10-01）：两个文件逐行读 |
| T17 | `tile_kernels/quant/swiglu_forward_kernel.py:11-141`；`swiglu_forward_cuda.py:11-24,80-90,91`；`swiglu_forward_asc.py:12,42,45,260-311` | §5.5：融合语义（`hidden = hidden // 2` `:26-27`）、允许的 `(fmt, num_per_channels)` 组合（`:41`）、CUDA 的 `PREFETCH_SOURCE` + `T.import_source`（`:11-16`、`:91`）、Ascend 的多套 UB 缓冲（`:285-311`） | 已核验（2026-10-01）：外层逐行读；CUDA/Ascend 读关键段，**未逐行读完 323/395 行** |
| T18 | `tile_kernels/quant/types.py:1-3`；`tile_kernels/quant/rstd.py:1-98` | `QuantTensor = tuple[Tensor, Tensor]`；Ascend 的补偿式 rsqrt | 已核验（2026-10-01）：`types.py` 全文；`rstd.py` 全文 |

### 2.5 RoPE 与 Rand

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| T19 | `tile_kernels/transform/rope_kernel.py:9-115`（全文件） | §6.1：统一的 3D/4D 处理、`rot_dim ∈ {64,128}`、原地修改、后端分派 | 已核验（2026-10-01）：逐行读 |
| T20 | `tile_kernels/transform/rope_cuda.py:40-45,48-132`（全文件） | §6.1、§7：线程数由 `nheads`/`half_dim` 反推（`:40-45`）、寄存器里的两半（`:62-65`）、两种配对（`:89-100`）、`T.ieee_fmaf`（`:110-114`） | 已核验（2026-10-01）：逐行读 |
| T21 | `tile_kernels/transform/rope_asc.py:9,30-37,45-110,112-200`（全文件） | §6.1、§7、习题 E14：`SIMD_LANES = 64`（`:9`）、由 UB 容量反推 tile（`:30-37`）、`S.vintlv`/`S.vdintlv` 重排（`:168,176`，`rotary_dim = 128` 分支在 `:182-192`） | 已核验（2026-10-01）：逐行读 |
| T22 | `tile_kernels/rand/randn_kernel.py:1-55`（全文件） | §6.2：CUDA 让路给 `torch.randn`（`:31-32`）、昇腾专用的 dtype 与图捕获断言（`:34-39`）、block/stage 配置（`:11-21`） | 已核验（2026-10-01）：逐行读 |
| T23 | `tile_kernels/rand/randn_asc.py:7-42`（全文件） | §6.2：`T.Kernel(num_vec_cores)`、`T.annotate_buffer_versions`、`T.SimtVF(threads=512)`、`T.Persistent`、`T.rng_init` / `T.rng_rand_float(dist='normal')` | 已核验（2026-10-01）：逐行读 |
| T24 | `tile_kernels/rand/seed.py:13-20` | §6.2：`reserve_seed` 在 host 上执行，因此不能进图 | 已核验（2026-10-01）：逐行读 |

### 2.6 参考实现与测试工具

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| T25 | `tile_kernels/torch/__init__.py:1-25`；`tile_kernels/torch/topk.py:9-21` | §2.1、§3.4：参考实现清单；`stable_topk` 的 stride 处理 | 已核验（2026-10-01）：逐行读 |
| T26 | `tile_kernels/testing/numeric.py:5-38`（`assert_equal` / `calc_diff`）、`:41-74`（`check_bias` / `count_bytes`） | §3.4、§4.8：**逐字节**对拍与相对误差口径 | 已核验（2026-10-01）：逐行读 |
| T27 | `tile_kernels/testing/generator.py:12-56`（测试档位与形状生成） | §3.3：`TK_TEST_LEVEL` 三档；测试档位决定形状覆盖 | 已核验（2026-10-01）：前 60 行 |
| T28 | `tests/moe/test_topk_gate.py:18-29,32-59,62-88` | §3.4：参数网格（`num_experts ∈ {1,35,1025}`、`num_topk ∈ {1,9,32}`）、只调外层 API、`assert_equal` 对拍 | 已核验（2026-10-01）：逐行读 |
| T29 | `tests/transform/test_rope.py:15-46,49-95,98-140` | §6.1：RoPE 的覆盖范围与 `calc_diff <= 1e-8` 门限 | 已核验（2026-10-01）：逐行读 |
| T30 | `tests/rand/test_randn.py:17-40,43-69` | §6.2：`set_seed` → 同种子同结果；形状与分布判据 | 已核验（2026-10-01）：逐行读 |
| T31 | `tests/conftest.py:1-30` | §3：xdist worker 的设备绑定、benchmark 目录约定 | 已核验（2026-10-01）：前 30 行 |

---

## 三、TileLang 锁定源码引用（`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `tilelang/language/__init__.py:1-32` | §3.1：`tilelang.language` 是 CUDA 方言的再导出（`:11`），所以 PDL 只在默认命名空间里 | 已核验（2026-10-01）：逐行读 |
| S2 | `tilelang/language/allocate.py:49,67,81,96-103` | §2、§4：`alloc_shared` / `alloc_local` / `alloc_fragment` / `alloc_var` 的 scope | 已核验（2026-10-01）：逐行读 |
| S3 | `tilelang/language/allocate.py:292-330` | §4.2：`alloc_reducer` 的语义与限制（「物理存储与跨线程通信方案由编译器选择」） | 已核验（2026-10-01）：逐行读 docstring |
| S4 | `tilelang/language/reduce_op.py:358,398,434` | §4.2：`reducer_init` / `reducer_update` / `finalize_reducer` 三步 | 已核验（2026-10-01） |
| S5 | `tilelang/language/reduce_op.py:479,495,511` | §4.4：`warp_reduce_sum/max/min` | 已核验（2026-10-01） |
| S6 | `tilelang/language/customize.py:101-140` | §4.7：`T.view` 的契约（总位数不变、`elem_offset == 0`、紧凑或显式 strides） | 已核验（2026-10-01）：逐行读 |
| S7 | `tilelang/language/copy_op.py:59-72,77,122-127` | §4.2：`T.copy` 的签名；dtype 不同时的逐元素转换；`disable_tma` 属于 CUDA 方言 | 已核验（2026-10-01） |
| S8 | `tilelang/language/loop.py:82,112,189,230` | §4.3：`T.Persistent` / `T.Pipelined`（含 `annotations` 参数）/ `T.serial` / `T.unroll` | 已核验（2026-10-01）：读 `Pipelined` 全 docstring |
| S9 | `tilelang/language/kernel.py:431`；`tilelang/language/proxy.py:256-257`；`tilelang/language/symbolics.py:13` | §2.5、§4：通用 `T.Kernel`、`T.Tensor` / `T.StridedTensor` 代理、`T.dynamic` | 已核验（2026-10-01） |
| S10 | `tilelang/language/common.py:178` | §7.3：`T.import_source` 的定义位置（CUDA 侧内联设备函数的入口） | 已核验（2026-10-01） |
| S11 | `tilelang/cuda/language/pdl.py:4-21` | §3.1：PDL 的 `pdl_trigger` / `pdl_sync` 只在 CUDA 方言里 | 已核验（2026-10-01）：全文 |
| S12 | `tilelang/cuda/language/random.py:4-70`；`tilelang/ascend/language/random.py:15-38` | §6.2：两个方言都有 `rng_init` / `rng_rand` / `rng_rand_float`；昇腾版只改了默认序列号 | 已核验（2026-10-01） |
| S13 | `tilelang/cuda/language/print.py:8`；`tilelang/ascend/debug.py`（`device_assert`） | §4.2：两个方言都导出 `device_assert` | 已核验（2026-10-01） |
| S14 | `tilelang/ascend/language/__init__.py:5,8-18,49-56,69-75` | §3.3、§4.3：昇腾方言建立在 `language.common` 之上；`alloc_shared` 故意遮蔽为 UB；`Kernel`/`MixedKernel`/`get_thread_binding` 来自 `ascend/language/kernel.py`；`device_assert` 与 `rng_*` 的导出 | 已核验（2026-10-01）：逐行读 |
| S15 | `tilelang/ascend/language/kernel.py:139-172`（`Kernel`）、`:176-...`（`MixedKernel`） | §4.5：昇腾 `T.Kernel(N)` 只接受一维网格、**没有** `threads=`；线程要用 `T.SimtVF` | 已核验（2026-10-01）：逐行读 docstring |
| S16 | `tilelang/ascend/language/frame.py:107,179` | §4.3、§6.2：`T.SimtVF(threads=128)` 与 `T.SimdVF(latency=0)` 的签名 | 已核验（2026-10-01） |
| S17 | `tilelang/ascend/language/annotations.py:19-65`（`annotate_buffer_versions`）、`:107-162`（`annotate_manual_multi_buffer`）、`:165-188`（`annotate_unlimit_memory`） | §4.6：版本数语义（`1` 退出多缓冲 / 固定份数 / 只给模式），以及「Z3 在 `T.Persistent` + `num_stages > 1` 时会自己算」 | 已核验（2026-10-01）：逐行读 docstring |
| S18 | `tilelang/ascend/language/simd.py:457`（`vsts`）、`:922`（`vdup`）、`:933`（`vdupv`）、`:969`（`vcmax`）、`:976`（`vcmin`）、`:1019`（`vci`）、`:1052`（`vintlv`）、`:1067`（`vdintlv`）、`:1247-1249`（`vcvt` 的 `part=PART_EVEN/PART_ODD`）、`:1429`（`vsel`）、`:1486`（`mem_bar`） | §4.3、§6.1、习题 E14：本章用到的全部 SIMD 原语 | 已核验（2026-10-01）：逐个 `rg -n` 定位并读签名/docstring |
| S19 | `tilelang/transform/pass_config.py:73,83,189,239,335` | §4.2、§5.3：`TL_DISABLE_WARP_SPECIALIZED`（`:73`）、`TL_ENABLE_FAST_MATH`（`:83`）、`TL_ENABLE_LOWER_LDGSTG_PREDICATED`（`:189`）、`TL_DISABLE_THREAD_STORAGE_SYNC`（`:239`）、`TL_DISABLE_OUT_OF_BOUND_WARNING`（`:335`） | 已核验（2026-10-01）：逐个定位 |
| S20 | `tilelang/transform/pass_config.py:76`（`TL_ENABLE_AUTO_SCHEDULE`） | §3.3：昇腾自动调度默认开启（与第 19 章 S15 同一条） | 已核验（2026-10-01） |

---

## 四、官方发布说明与 corpus 元数据

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:12` | 见该文件头部转录说明 | §1.1：官方对 TileKernels 的定位（「数据处理所需的常规向量计算和访存算子」）与「性能接近硬件上限」的自述 | 已核验（2026-10-01）：本地全文（29 行） |
| R2 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md` | — | 本章快照的 commit 与日期（TileKernels `66258df6175d`，2026-09-30） | 已核验（2026-10-01） |
| R3 | `../tilelang-corpus/sources/sources.yaml`（条目 `github-deepseek-tilekernels`，约 `:1495-1505`） | https://github.com/deepseek-ai/TileKernels | 语料的 `trust: primary` 标注与用途说明（「本教程第 33、34 章的主线」） | 已核验（2026-10-01） |
| R4 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md`（GitHub 渲染版） | https://github.com/deepseek-ai/TileKernels | T0/T1 的上游出处 | **未核验**：本机未访问网络核对 GitHub 页面与快照是否一致 |

---

## 关于性能数字的说明

**本章不提供任何实测数字。** 原因见开头的验证边界。正文里出现的「数量级」只有三类：

| 数字 | 来源 | 条件 |
| --- | --- | --- |
| `num_stages = 6`、`vector_length = 64`、`max_vregs_per_group = 6`、`SIMD_LANES = 64`、`num_threads = 32`、`num_topk ≤ 32` | T6 / T7 / T21 等源码常量 | **源码里的常量**，不是调优结果、不是实测最优 |
| 每核 UB 预算 216 KB（`use_simt=True`）/ 248 KB（`use_simt=False`） | T3（`config.py:81-83`） | **库自述的预算口径**；我们没有在官方文档里找到逐字对应的条目（见「待补资料」第 5 条） |
| 「方案 A 4.5H vs 方案 B 2.5H」（以及习题里 fp4 的 5H : 3H） | **本书推导** | 纯整数记账，忽略 sf、缓存与 TMA 一次搬多块；**不是任何机器上的测量** |
| `rot_dim ∈ {64,128}`、`num_per_channels ∈ {16,32,64,128}`、`num_per_tokens ∈ {32,128}`、`(32,32)`、`hidden % 64 == 0`、`num_tokens % 128 == 0` | T13–T17 的断言 | **实现里写死的常量**，不是性能建议 |

「性能接近硬件上限」「已用于内部训练与推理」（T0）是**厂商自述**。
本章引用它只是为了说明「这不是玩具库」，**不作为可复现的结论**。

---

## 待补资料

1. **Ascend 950 的硬件规格页缺失。** 本章多处提到「Vector 核」「SIMD 宽度 64」
   「UB 容量」，但 corpus 里没有 Ascend 950 的官方规格页（容量、算力、带宽、
   每核 lane 宽度）。`SIMD_LANES = 64` 是从 `rope_asc.py:9` 的常量与
   `simd.py` 里 `T.float32x64` / `T.int32x64` 的类型宽度**推出来的**，
   没有官方文档逐字确认「dav-3510 的向量宽度就是 64 条 fp32 lane」。
   建议补抓昇腾 950 的产品规格与 CANN 的 Ascend C API 文档。
2. **两后端的实测等价性完全未验证。** 本章多次断言「两个后端实现同一个算法」，
   依据是源码逐行对照与共用的参考实现（T25/T26）。
   真正的验证需要一台 CUDA 机器 + 一台昇腾机器，跑同一组
   `tests/**` 用例并与 `tile_kernels/torch/**` 对拍。本章**没有**这个条件。
3. **`T.view` 在昇腾上的实际行为只核对了契约。** §4.7 的结论来自
   `customize.py:101-140` 的 docstring 与调用方式；没有在设备上验证
   「int64 → int32 视图 + 单 lane 写高位」这条路真的成立。
4. **corpus 快照不是 tag。** TileKernels 快照是 commit `66258df6175d`
   （HEAD 摘要 `Merge pull request #34 from deepseek-ai/v2.0.0`），
   而 `pip install tile-kernels` 装到的是发布版本，两者可能不同。
   本章所有行号都对应快照；引用具体行号时请注意这一点。
   建议后续在 corpus 的 `PROVENANCE.md` 里为 TileKernels 补一个 tag 或 release 快照。
5. **`get_max_ub_per_vector_core()` 的 216/248 KB 缺少官方对应条目。**
   第 19 章引的 Atlas A2 口径是 UB 192 KB（不同产品、不同预留策略），
   与这里的 216/248 KB 不是一回事。需要一份 Ascend 950 的 UB 规格
   或 CANN 里对应的查询接口文档，才能判断这两个数是怎么来的。
6. **`T.annotate_buffer_versions` 缺省时的实际行为未验证。**
   习题 E4 第 5 问的答案（「大概率仍能编译，但可能丢掉累积器语义」）
   来自 `annotations.py:19-45` 的文档，没有做过「删掉这一行再编译」的实验。
   建议在有设备的机器上做一次对照实验（编译 + 数值对拍），把结论补成实测。
7. **`tilelang/ascend/language/simd.py` 只读了本章用到的原语。**
   该文件 2000+ 行，本章逐个核对了 `vld/vsts/vdup/vdupv/vcmax/vcmin/vci/
   vintlv/vdintlv/vcvt/vsel/pset/vadds/vcmps/vsub/vmax/vmin/mem_bar` 等，
   **没有**系统梳理它的全部原语与语义分类。第 25 章若要展开，需要完整过一遍。

---

## 关于 `check-content.mjs` 的说明

本章写完后运行了 `node scripts/tutorial/check-content.mjs 33-tilekernels-two-backends`，
输出见交付回报。它校验的是**结构与格式**：图 JSON 的合法性与 `id` 一致性、
`source-figures/<slug>/` 里的每个原图都有对应 spec、自测题的数量（5–8）
与字段完整性（每题恰好 4 个选项、`answer` 在 0–3、`explain` 非空）。
它**不检查**内容正确性、引用可达性或行号准确性。
本章的行号准确性由上面各表逐条人工核验，其中「API 名字是否存在」
由 `code/03-map-and-diff/map_and_diff.py` 第 3 步做了机械复核（21/21 命中）。

---

## 附：一条被外部校验工具误报的引用，以及它带来的额外结论

并行写作的另一章在跑 `tools/check_book.py` 时，把本章的一处引用报成
「`_vendor/tilelang-v0.1.15/tilelang/ascend/language/simd.py` 不存在」。
**这是一次误报，但排查过程本身给本章补了一条有价值的结论**，所以记在这里。

### 误报的真实原因：行号后缀写错，不是文件不存在

`tools/check_book.py` 用 `LINE_SUFFIX_RE = re.compile(r":\d+(?:[-,]\d+)*$")`
剥掉行号后缀再去查文件。本章 `solutions.md` 的习题 E14 里写的是
`` `...simd.py:1052,1067+` ``——末尾多了一个 `+`，正则匹配不到，
于是整串（含 `:1052,1067+`）被当成路径去查，自然「不存在」。
**这是行号写法的错误，不是引用目标的问题。** 该处已改为分别标注 `:1052` 与 `:1067`。
修复后 `python3 tools/check_book.py` 的输出是
`checked 2858 citations, 0 broken`。

### 顺带核实：这个文件在锁定版本里确实存在

在该 commit（`a35f8ddf45eba16c21211ec8822d56ce5363036f`，即 `VERSIONS.md` 锁定的
v0.1.15）的工作树上：

```
$ ls -l tilelang/ascend/language/simd.py
-rw-r--r--  1 huandu  staff  57909 Oct  1 00:28 tilelang/ascend/language/simd.py
$ shasum -a 256 tilelang/ascend/language/simd.py
89483579ddde5e36a352a853720fc02ba2b06badeab9eca74b31bcde6a04b57f
$ sed -n '1052p;1067p' tilelang/ascend/language/simd.py
def vintlv(src0, src1):
def vdintlv(src0, src1):
$ git log -1 --format='%H %s'
a35f8ddf45eba16c21211ec8822d56ce5363036f [Release] Bump version into 0.1.15 (#3309)
```

也就是说，TileKernels 的 Ascend kernel 里那句
`from tilelang.ascend.language import simd as S`（例如 `topk_gate_asc.py:3`）
**在锁定的 v0.1.15 主干里是有对应的**——它正是本章 §3.3 所说「TileKernels 选了 M 树」
的直接证据：**如果只有 A 树可用，这些 `S.*` 原语根本无从谈起**。

### 这条对照值得单独记下来：`S.*` 是 M 树专属

同一份检查也确认了 A 树（`_vendor/tilelang-ascend-v0.1.2.000/`）**没有**这个表面：

```
$ ls tilelang/                      # A 树：没有 ascend/ 这个子包
__init__.py  _ffi_api.py  autotuner  cache  carver  common  contrib  engine  env.py
intrinsics  jit  language  layout  libinfo.py  math  primitives  profiler  quantize
testing  tools  transform  utils  version.py

$ find . -name 'simd*.py'           # A 树：没有任何 simd 模块
（无输出）

$ rg -c "vcmax|vintlv" tilelang/     # A 树：跨 lane 归约/重排原语一个都没有
（无输出；A 树的昇腾向量表面在 tilelang/language/ascend.py 与 ascend_tile.py）
```

所以下面的判断是**有据可查**的，而不是推测：

| 结论 | 证据 |
| --- | --- |
| `from tilelang.ascend.language import simd as S` 只在 M 树成立 | M 树有 `tilelang/ascend/language/simd.py`；A 树连 `tilelang/ascend/` 都没有 |
| `S.vcmax` / `S.vcmin` / `S.vintlv` / `S.vdintlv` 是 M 树方言的一等公民 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/simd.py:969,976,1052,1067` |
| A 树的昇腾 API 与 M 树不通用（第 19 章 §2.6 的结论）在本章同样成立 | A 树用 `T.alloc_ub`（`tilelang/language/allocate.py:156`）、无 `simd` 表面；M 树用 `T.alloc_shared` + `S.*` |

**这条对照对读者的实际意义**：如果你想照抄 TileKernels 的 Ascend kernel，
你需要的不是 A 树，而是**主干 ≥ v0.1.15 的昇腾方言**（`tilelang.ascend.language`）。
这也是为什么本章 §1.1 特意指出「TileKernels 要求的 TileLang ≥ 0.1.15」
与本书锁定的主干版本**恰好一致**。

### 关于「本地快照 vs 上游发布」的一句话提醒

另一条容易被误判的情况是：**TileKernels 的语料快照是一个 commit**
（`66258df6175d`，HEAD 摘要 `Merge pull request #34 from deepseek-ai/v2.0.0`），
不是 tag。若上游之后改动文件行数，本章的行号会漂移；届时按
`code/03-map-and-diff/map_and_diff.py` 的机械核对重新标定即可。
