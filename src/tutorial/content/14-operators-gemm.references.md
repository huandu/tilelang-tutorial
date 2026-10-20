# 第 14 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现「最新版」这类表述；讨论 `main` 上更新的东西时会显式标注。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有
昇腾 NPU**。因此本章所有 kernel 示例的最高验证级别是「仅静态核对」
（API 存在性与签名逐个在 `_vendor/` 里比对过，语法用 `py_compile` 过了），
**没有任何一条性能数字来自本机实测**。

---

## 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:160` | `T.gemm(A, B, C, transpose_A, transpose_B, policy, clear_accum, annotations)` 签名与 docstring（同步语义、Hopper/Blackwell 的隐式等待） | 已核验（2026-10-01）：`grep -n "^def gemm"` → 160；逐行读签名 |
| S2 | `_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:275` | `T.gemm_blockscaled(..., SFA, SFB, k_start, sf_a_granularity_k, sf_b_granularity_k)` 签名与语义 | 已核验（2026-10-01） |
| S3 | `_vendor/tilelang-v0.1.15/tilelang/language/experimental/gemm_sp_op.py:106` | `T.gemm_sp` 的签名（2:4 结构化稀疏） | 已核验（2026-10-01） |
| S4 | `_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:59` | `T.copy` 签名 | 已核验（2026-10-01） |
| S5 | `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:13,82,112,189` | `T.Parallel` / `T.Persistent` / `T.Pipelined` / `T.serial`；`num_stages=0` = 不启用流水 | 已核验（2026-10-01）：`Pipelined` docstring 明确写 “if `num_stages` is 0, pipeline will not be enabled” |
| S6 | `_vendor/tilelang-v0.1.15/tilelang/language/fill_op.py:11,48` | `T.fill` / `T.clear` | 已核验（2026-10-01） |
| S7 | `_vendor/tilelang-v0.1.15/tilelang/language/annotations.py:30,45` | `T.use_swizzle(panel_size, order, enable)`（threadblock rasterization）与 `T.annotate_layout(layout_map)` | 已核验（2026-10-01） |
| S8 | `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:49,81,194,254` | `T.alloc_shared` / `T.alloc_fragment` / `T.alloc_barrier` / `T.alloc_tmem` | 已核验（2026-10-01） |
| S9 | `_vendor/tilelang-v0.1.15/tilelang/language/atomic.py:211` | `T.atomic_add`：标量与 tile-region 两条路径、`memory_order`、`return_prev` | 已核验（2026-10-01） |
| S10 | `_vendor/tilelang-v0.1.15/tilelang/tileop/base.py:5` | `GemmWarpPolicy`（`Square` / `FullRow` / `FullCol`） | 已核验（2026-10-01） |
| S11 | `_vendor/tilelang-v0.1.15/tilelang/cuda/intrinsics/layout/mma_layout.py:295,336` | `get_swizzle_layout` / `make_mma_swizzle_layout`（含 `can_swizzle` 的 512-bit 判据） | 已核验（2026-10-01）；`__init__.py:13-14` 确认对外导出 |
| S12 | `_vendor/tilelang-v0.1.15/tilelang/cuda/intrinsics/macro/mma_macro_generator.py:46` | `TensorCoreIntrinEmitter`（`ldmatrix_a/b`、`mma`、`stmatrix`） | 已核验（2026-10-01） |
| S13 | `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:127-142` | CUDA pass 顺序：`PipelinePlanning → InjectSoftwarePipeline → LayoutInference → LowerTileOp` | 已核验（2026-10-01）：逐行读注释与调用 |
| S14 | `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:108-114` | 昇腾 pass 顺序：`EstimateLatency → AutoSchedule → PrepareMultiBuffer → InsertSync → MaterializeMultiBuffer` | 已核验（2026-10-01） |
| S15 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/gemm_op.py:110,168` | 昇腾 `T.gemm` 的 `unit_flag_ctrl` 与 `T.gemm_blockscaled` 方言重载（MX scale 覆盖 32 个 K 元素） | 已核验（2026-10-01） |
| S16 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:26,37,46,55` | `T.alloc_l1` / `alloc_l0a` / `alloc_l0b` / `alloc_l0c` 的 scope 默认值 | 已核验（2026-10-01） |
| S17 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:74-152,175` | 昇腾 950 的 GEMM+ReLU quickstart（`M,N,K=256,7168,2048`、`num_blocks=32`）、与 Ascend C 的分工对照表、Benchmark 口径（对比 Torch NPU，BF16 GEMM / FP8 casting / GQA backward） | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S18 | `_vendor/tilelang-v0.1.15/tilelang/profiler/bench.py:31` | `do_bench(fn, warmup=25, rep=100, ..., backend="event", return_mode="mean")` 与 L2 flush 说明 | 已核验（2026-10-01） |
| S19 | `_vendor/tilelang-v0.1.15/examples/gemm/example_gemm.py` | 本章第 ②–③ 级的最小 GEMM 原型 | 已核验（2026-10-01）：文件存在、逐行比对；**未运行** |
| S20 | `_vendor/tilelang-v0.1.15/examples/gemm/README.md:170-230` | 第 ④ 级的 “Enhanced GEMM Example with Annotations”（`make_mma_swizzle_layout` + `T.use_swizzle`） | 已核验（2026-10-01）；未运行 |
| S21 | `_vendor/tilelang-v0.1.15/examples/gemm/example_gemm_intrinsics.py` | 第 ⑤ 级（`TensorCoreIntrinEmitter` + `ldmatrix/mma/stmatrix`）；文件内自述 “This is a debug config” | 已核验（2026-10-01）；未运行 |
| S22 | `_vendor/tilelang-v0.1.15/examples/gemm_splitk/example_tilelang_gemm_splitk.py`、`..._vectorize_atomicadd.py` | split-K 的两种收尾写法；`num_stages=0` | 已核验（2026-10-01）；未运行 |
| S23 | `_vendor/tilelang-v0.1.15/examples/gemm_streamk/example_tilelang_gemm_streamk.py` | stream-K：`total_sm`、`streamk_tiles` / `blocking_tiles` 的划分，只有被切开的块用 `T.atomic_add` | 已核验（2026-10-01）；未运行 |
| S24 | `_vendor/tilelang-v0.1.15/examples/grouped_gemm/example_grouped_gemm_fwd.py`、`example_grouped_gemm_fwd_ptr.py` | grouped GEMM：偏移表写法与指针表写法；ptr 路径的两条限制（运行时变形状不稳定、多级流水尚不正确） | 已核验（2026-10-01）；未运行 |
| S25 | `_vendor/tilelang-v0.1.15/examples/gemm_fp8/example_tilelang_gemm_fp8.py`、`examples/gemm_int4/example_tilelang_gemm_int4.py` | fp8（`determine_fp8_type`）与 int4（`T.int4` + 打包）的结构 | 已核验（2026-10-01）；未运行 |
| S26 | `_vendor/tilelang-v0.1.15/examples/dequantize_gemm/`、`examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py` | dequantize GEMM 的粒度分类；MXFP8 block-scaled 的 `(1,1,128)` 粒度与 `uint32` 打包 scale | 已核验（2026-10-01）；未运行 |
| S27 | `_vendor/tilelang-v0.1.15/examples/gemm_tcgen05/README.md` | Blackwell TCGEN05 的 `T.alloc_tmem` / `T.tcgen05_gemm` / `mbarrier` 协议；文件标注 “preview version” | 已核验（2026-10-01）；未运行 |
| S28 | `_vendor/tilelang-v0.1.15/examples/ascend/example_gemm.py`、`example_gemm_splitk.py` | 主干昇腾 GEMM（persistent + `T.Pipelined` + swizzle + 混合核 epilogue）与 split-K | 已核验（2026-10-01）；未运行 |

---

## 锁定的源码引用（`_vendor/tilelang-ascend-v0.1.2.000/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| A1 | `_vendor/tilelang-ascend-v0.1.2.000/examples/gemm/example_gemm.py` | 最朴素的昇腾 GEMM：`T.alloc_L1` / `alloc_L0C`、`T.gemm_v0`、`T.barrier_all()` | 已核验（2026-10-01）：文件存在、逐行比对；**未运行** |
| A2 | `_vendor/tilelang-ascend-v0.1.2.000/examples/gemm/example_gemm_intrinsic.py` | 带完整 flag 管理（`T.set_flag`/`T.wait_flag`）、多级流水与 `T.mma` 的手写版本 | 已核验（2026-10-01）；未运行 |
| A3 | `_vendor/tilelang-ascend-v0.1.2.000/examples/gemm/example_gemm_persistent.py`、`example_gemm_transpose_l1.py` | persistent 写法与 L1 转置对照 | 已核验（2026-10-01）；未运行 |
| A4 | `_vendor/tilelang-ascend-v0.1.2.000/examples/batch_gemm/batch_gemm.py` | batched GEMM：一维 grid 展平 `B*m_num*n_num`，`T.gemm_v0(init=(k==0))` | 已核验（2026-10-01）；未运行 |
| A5 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/gemm.py:11`、`tilelang/language/ascend.py:418` | `T.gemm` 与 `T.gemm_v0(A, B, C, transpose_A, transpose_B, init, kL0Size, n_actual)` 的签名 | 已核验（2026-10-01） |

「已核验」的含义：在锁定版本的工作树上用 `grep -n` / `sed` 定位到定义行并读完
签名与 docstring；行号区间与文件内容一致。**这不是运行验证。**

---

## corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/matmul.md` | https://tilelang.com/deeplearning_operators/matmul.html | Level 2 GEMM 教程：`T.Kernel`/`alloc_shared`/`alloc_fragment`/`Pipelined`/`gemm`；可选 `make_mma_swizzle_layout` 与 `T.use_swizzle`；平台性能口径（RTX 4090 ~1.1×、A100 ~0.97×、H100 ~1.0×、MI300X ~1.04× vs cuBLAS，末尾注明 “These measurements will vary …”） | 已核验（2026-10-01）：本地文档存在，逐句比对 |
| R2 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/matmul_sparse.md` | https://tilelang.com/deeplearning_operators/matmul_sparse.html | 2:4 结构化稀疏：`T.gemm_sp`、E-factor（fp16/bf16 默认 int16/16，int8/fp8 默认 int32/32）、`compress()` 工具 | 已核验（2026-10-01） |
| R3 | `../tilelang-corpus/06-huawei-ascend/raw/catlass/docs/zh/2_Design/01_kernel_design/04_matmul_summary.md` | https://gitcode.com/cann/catlass | CATLASS 的 Common / MultiCoreSplitK / SingleCoreSplitK 三套模板：L1/L0A/L0B/L0C 约束、读入量 $2\,\mathrm{Byte}\cdot MNK(1/m_1+1/n_1)$、split-K 写出量 $2\,\mathrm{Byte}\cdot MNK/k$、Multi Buffer / Preload / Padding 三类工程优化 | 已核验（2026-10-01）：本地为仓库工作树快照，610 行 |
| R4 | `../tilelang-corpus/06-huawei-ascend/raw/catlass/docs/zh/2_Design/01_kernel_design/02_swizzle.md` | https://gitcode.com/cann/catlass | Swizzle 策略与判据（M≥N 用 SwizzleOffset=3/Direction=0；M<N 用 Direction=1） | 已核验（2026-10-01） |
| R5 | `../tilelang-corpus/06-huawei-ascend/raw/catlass/docs/zh/2_Design/01_kernel_design/00_basics/atlasA2_hardware_info.md` | https://gitcode.com/cann/catlass | Atlas A2 的 L1=512KB、L0A/L0B 各 64KB、L0C=128KB、UB=192KB；GM/L1/L0A/L0B/L0C/UB 的搬运指令与支持格式 | 已核验（2026-10-01） |
| R6 | `../tilelang-corpus/06-huawei-ascend/raw/catlass/README.md` | https://gitcode.com/cann/catlass | CATLASS 定位、Kernel/Block/Tile 分层、dispatchPolicy 概念、目录结构、性能口径「在定制 shape 下的性能能达到相应算子标杆性能的 0.98~1.2 倍」 | 已核验（2026-10-01） |
| R7 | `../tilelang-corpus/06-huawei-ascend/raw/catlass/examples/`（`00_basic_matmul`、`01_batched_matmul`、`02_grouped_matmul_slice_m`、`09_splitk_matmul`、`37_streamk_matmul`、`41_sparse_matmul_tla`、`53_ascend950_fp8_mx_matmul` 等） | https://gitcode.com/cann/catlass | 变体家族在昇腾模板库里的对应实现与命令行 shape 约定 | 已核验（2026-10-01）：目录与 README 存在；**未编译、未运行** |
| R8 | `../tilelang-corpus/12-operator-libraries/raw/tilelang-benchmark/hopper_benchmark/dense_matmul/0.cublas-benchmark/cublas_benchmark.cu` | https://github.com/tile-ai/tilelang-benchmark | cuBLAS 对比脚本的计时口径：`minimal_repeat_ms = 100`、先跑一轮估算单次耗时、`numRepeats = max(5, 100ms/单次耗时)` | 已核验（2026-10-01）：本地为仓库工作树快照 |
| R9 | `../tilelang-corpus/12-operator-libraries/raw/tilelang-benchmark/hopper_benchmark/dense_matmul/3.tilelang-benchmark/benchmark_tilelang_matmul.py` | https://github.com/tile-ai/tilelang-benchmark | TileLang 侧 benchmark 的搜索空间（block_M/N/K、num_stages、thread_num、policy、rasterization）与 autotune 流程 | 已核验（2026-10-01） |
| R10 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md:3086,3126` | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html | 寄存器压力（register pressure）与 occupancy：cc7.0 每 SM 65536 个 32 位寄存器、最多 2048 线程、每线程 32 个寄存器才 100% occupancy；寄存器分配以每 warp 256 个为单位取整 | 已核验（2026-10-01）：本地为全文，行号已定位 |
| R11 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md` | https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | H100 的 HBM3「over 3 TB/sec」、FP16 张量核 preliminary specs、每 SM 256KB shared+L1（最多 228KB shared） | 已核验（2026-10-01）：厂商公开资料，非实测 |
| R12 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577.md` | https://arxiv.org/abs/2504.17577 | TileLang 论文摘要：调度空间（thread binding/layout/tensorize/pipeline）与 dataflow 解耦 | 已核验（2026-10-01）：本地只有摘要，无性能数字 |

---

## 关于性能数字的说明

本章出现的每一个数字都带来源与条件，汇总如下：

| 数字 | 来源 | 测量/口径条件 |
| --- | --- | --- |
| H100 fp16 张量核 1000 TFLOPS、HBM3 over 3 TB/s | R11 | H100 SXM5，厂商公开 preliminary specs，未实测 |
| 机器平衡点 ≈ 333 FLOP/Byte | **本书推导**，输入为 R11 的两个数字 | 峰值/带宽之比 |
| 4096³ / 512³ / 128³ 的 FLOPs、访存、算术强度 | **本书推导**，由 `code/01-gemm-ledger/` 复算 | fp16 输入输出、理论最小访存量 |
| `M≈10³` 的 compute/memory 翻转点 | **本书推导** | 由 `AI=2M/3b=333` 解出 |
| 单级/多级 shared memory、每线程累加寄存器 | **本书推导**，由 `code/01` 的公式复算 | fp16 输入、fp32 累加、`threads=128` |
| cc7.0 每 SM 65536 个 32 位寄存器、每线程 32 寄存器=100% occupancy | R10 | NVIDIA 最佳实践文档，compute capability 7.0 |
| cuBLAS benchmark 的 `minimal_repeat_ms=100` | R8 | 该脚本的默认配置 |
| `do_bench` 默认 `warmup=25ms`、`rep=100ms`、`mean`、flush 256MB L2 | S18 | v0.1.15 的默认值 |
| CATLASS「0.98~1.2 倍标杆性能」 | R6 | CATLASS README 自述，条件为「定制 shape」，未给具体 shape 与测量方法 |
| TileLang GEMM 相对 cuBLAS 的 ~0.97–1.1× | R1 | 官方文档自述（4090/A100/H100/MI300X），文档末尾注明会随配置与硬件变化；**不是本书实测** |
| TileLang 昇腾对比 Torch NPU 的口径（BF16 GEMM / FP8 casting / GQA backward） | S17 | 官方 README 自述，未给具体数值 |

**没有来源的、或只有宣传口径的数字一律未采用。** 任何需要真实加速器才能得到的
数字（实测算力利用率、延迟、吞吐）在本章中均未出现。

---

## 待补资料

1. **CATLASS 的逐例性能数据缺失。** corpus 里的 CATLASS 快照包含 `examples/` 与
   设计文档，但 `docs/zh/figures/homepage_perf_mm.png`、`homepage_perf_gmm.png`
   两张性能对比图只是图片引用，没有对应的数值表，也没有 shape/口径说明。因此本章
   只能引用 README 的「0.98~1.2 倍」这一区间描述，无法给出逐 shape 的对照。
   建议后续版本补抓 CATLASS 的 `docs/zh/1_Practice/` 性能章节或 `tools/tuner` 的
   实测输出。
2. **CATLASS StreamK 的设计文档缺失。** `04_matmul_summary.md` 明确写「新增
   StreamK 等 Matmul 样例待更新」，`66_ascend950_streamk_matmul` 只有代码与
   README，没有理论模板文档。本章 4.2 节的昇腾侧描述因此只做到「存在对应实现」，
   没有展开其切分算法。
3. **tilelang-benchmark 的实测 CSV 缺失。** corpus 里只有 benchmark 脚本
   （`0.cublas-benchmark/`、`3.tilelang-benchmark/`）和图片，没有跑出来的
   JSON/CSV 原始结果，因此无法引用任何一条「TileLang vs cuBLAS」的独立数字
   （只能引用官方文档 R1 的自述值）。
4. **昇腾 950 的 GEMM 实测基线缺失。** `tilelang/ascend/README.md` 提到有一张
   `images/ascend-950-throughput.png`（TileLang vs Torch NPU），但 corpus 里
   没有这张图对应的数值表，也没有 Ascend 950 的硬件规格页。本章 6.3 节因此只
   能给出「对齐方法论」，给不出「TileLang 在昇腾上达到基线的百分之多少」。
