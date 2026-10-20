# 第 15 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[`../../VERSIONS.md`](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现模糊的版本指代，一律写具体 tag 与 commit。

写作环境：macOS (Apple Silicon)、Python 3.11（miniconda）；**没有 NVIDIA GPU，
也没有昇腾 NPU**。因此本章**没有任何本机实测的性能或正确性结论**。本章的数值结论有两类，
来源都写清楚：

* **本机 CPU 逻辑验证**（`code/01`–`code/04`）：数学性质与访存记账，可复算；
* **厂商自报**（`A5`、`A6`、`A8`）：锁定版本仓库自带的 benchmark 文档/README，
  带原始测量条件，**作者未复现**。

引用分三类，核验状态只有三种口径：

* **已核验（静态）**：在锁定版本工作树上 `rg`/`sed` 定位到该行并逐行阅读，行号与文件
  内容一致；对本章最重要的部分，`code/05` 与 `code/06` 两个脚本会**自动比对行号**
  （`--strict`），输出 `drift/miss count: 0`。**这不是运行验证。**
* **已核验（本地全文）**：corpus 里保存了该来源的完整本地副本。
* **未核验**：明确写出原因。

---

## 一、NVIDIA 侧锁定源码引用（`_vendor/tilelang-v0.1.15/`）

### 1.1 官方 attention 例子（本章逐行对照的对象）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/examples/flash_attention/example_mha_fwd_bhsd.py:24,30-31,40,45-52,54,59-63,65,67-75,77-95,97-100,167` | 本章的主对照对象：`log2e` 折进 `scale`、`past_len`、causal 循环上界、非 causal 的越界 mask、online softmax 五步、`O_shared` epilogue、回归配置 | 已核验（2026-10-01）：`sed -n` 逐段阅读；`code/05` `--strict` 全部命中 |
| S2 | `_vendor/tilelang-v0.1.15/examples/flash_attention/README.md` | 官方把整个 kernel 贴出来的讲解版（与 `.py` 的行文略有差异，正文以 `.py` 为准） | 已核验（2026-10-01）：本地全文 |
| S3 | `_vendor/tilelang-v0.1.15/examples/flash_attention/example_mha_fwd_varlen.py`；`example_gqa_fwd_varlen.py:36,50-60,62,68-74,79-91,93,117-119,122-124` | varlen 打包（`cu_seqlens`、右对齐 `offset`、越界 guard）、GQA 的 `kv_head_idx`、`-1e9` mask | 已核验（2026-10-01）；`code/04` 的用例 4 在 CPU 上复现了结构 |
| S4 | `_vendor/tilelang-v0.1.15/examples/flash_decoding/example_mha_inference.py:30-113,115-152` | split-KV（flash decoding）：split kernel 输出部分结果与逐段 LSE，combine kernel 用 logsumexp 求权重 | 已核验（2026-10-01）；`code/03` 用 CPU 复现了合并公式 |
| S5 | `_vendor/tilelang-v0.1.15/examples/attention_sink/example_mha_sink_fwd_bhsd.py:84,91-92,103-109,125-128` | sliding window 的循环起点与双端裁剪、`check_inf`、attention sink 的 epilogue | 已核验（2026-10-01）；`code/02`/`code/04` 在 CPU 上验证了 `check_inf` 的必要性 |
| S6 | `_vendor/tilelang-v0.1.15/examples/block_causal_attention/block_causal_attention.py:43-147` | block-causal（噪声/干净分区）的逐 tile mask 决策 | 已核验（2026-10-01）：`T.Kernel` 在 `:74`、`is_noisy_diag` 在 `:98`（`code/05` 探针） |
| S7 | `_vendor/tilelang-v0.1.15/examples/flash_attention_sm100/mha_fwd_bshd.py:1-8,54-56,89-93,145-148` | Blackwell 变体：TMEM 累加器、`mbarrier`、`T.tcgen05_gemm`；文件头注释说明它取代 Hopper 的 WGMMA 版本 | 已核验（2026-10-01）；**未运行**（无 sm100 设备） |
| S13 | `_vendor/tilelang-v0.1.15/examples/deepseek_nsa/example_tilelang_nsa_fwd.py:18-29,53` | 稀疏 attention 的结构性变化：`BlockIndices` 索引张量取代连续区间 | 已核验（2026-10-01） |
| S14 | `_vendor/tilelang-v0.1.15/examples/minference/example_vertical_slash_sparse_attn.py` | mInference 的 vertical/slash 稀疏索引（第 16 章入口） | 已核验（2026-10-01）：文件存在且可读；**只做了结构引用，未逐行核对算法** |
| S15 | `_vendor/tilelang-v0.1.15/examples/dsa_hisa/README.md`（+ `hisa.py`） | HISA 的两级索引（先选 pool block，再选 block 内 token） | 已核验（2026-10-01）：README 全文；`hisa.py` 未逐行阅读，正文只引用 README 描述 |
| S16 | `_vendor/tilelang-v0.1.15/examples/seer_attention/block_sparse_attn_tilelang.py`；`examples/blocksparse_attention/` | 学习型块稀疏与块稀疏的另两个实现（变体矩阵里列出） | 已核验（2026-10-01）：文件存在；**未逐行阅读**，正文只作条目列出 |
| S17 | `_vendor/tilelang-v0.1.15/examples/flash_attention/example_mha_fwd_bshd.py`；`example_gqa_fwd_bshd.py` | `BSHD` 布局与 GQA 的固定长度版本（变体矩阵里列出） | 已核验（2026-10-01）：文件存在；行号为索引顺序差异 |

### 1.2 本章引用的 API 定义

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S8 | `_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:176-204,230-256` | `T.reduce_max` 的 `clear` 语义；`reduce_sum(clear=True)` 的 temp buffer 说明 | 已核验（2026-10-01）；`code/05` 固定行号 `:176`/`:230` |
| S9 | `_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:160-210` | `T.gemm` 的签名、`policy`、`clear_accum`，以及「手工异步请用 `T.wgmma_gemm`/`T.tcgen05_gemm`」的说明（`:177-179`） | 已核验（2026-10-01） |
| S10 | `_vendor/tilelang-v0.1.15/tilelang/tileop/base.py:5-12` | `GemmWarpPolicy` 三个取值（`Square`/`FullRow`/`FullCol`） | 已核验（2026-10-01）；`code/05` 固定行号 `:5` |
| S11 | `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:49,81,171,194,254-270` | `alloc_shared`/`alloc_fragment`/`alloc_global`/`alloc_barrier`/`alloc_tmem`（TMEM 的列数约束） | 已核验（2026-10-01） |
| S12 | `_vendor/tilelang-v0.1.15/src/op/math.cc:80-108`；`tilelang/language/tir/op.py:361-377` | `T.infinity` 落到 IEEE `+∞` 而不是 dtype 最大有限值 | 已核验（2026-10-01）：读 `infinity_op` 全部分支 |
| S18 | `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:112`；`tilelang/language/copy_op.py:59-131` | `T.Pipelined` 与 `T.copy` 的签名（正文引用其行为，机制见第 11 章） | 已核验（2026-10-01） |
| S19 | `_vendor/tilelang-v0.1.15/tilelang/cuda/language/gemm_op.py:264` | `T.tcgen05_gemm` 的定义位置 | 已核验（2026-10-01） |

---

## 二、昇腾侧锁定源码引用

### 2.1 主干仓库 `v0.1.15` 的 Ascend 方言与例子

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| A8 | `_vendor/tilelang-v0.1.15/examples/ascend/flash_attention/README.md`；`core.py:1-25,66-99,102-199,200-256,336-420`；`example_mha.py` | 主干仓库的昇腾 FA：Cube+Vector 分工、SIMD softmax 打包、`T.Stage` 自动调度、性能表 | 已核验（2026-10-01）：README 全文 + `core.py` 分块阅读；**未在 NPU 上运行**。性能数字为**厂商自报，作者未复现** |
| A9 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:15,26,37,46,55`；`copy_op.py:53`；`frame.py:179`；`schedule_hint.py:118`；`annotations.py:19` | `alloc_shared/alloc_l1/alloc_l0a/alloc_l0b/alloc_l0c`、`dual_copy`、`SimdVF`、`Stage`、`annotate_buffer_versions` | 已核验（2026-10-01）；`code/06` 固定以上行号，`--strict` 无漂移 |

### 2.2 适配器仓库 `v0.1.2.000`

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| A1 | `_vendor/tilelang-ascend-v0.1.2.000/examples/flash_attention/flash_attn_bhsd.py:26-34,86-131` | 昇腾 FA 的显式 workspace 与跨核 flag 写法（`T.Scope("C")`/`T.Scope("V")`） | 已核验（2026-10-01）：逐行阅读；**未运行** |
| A2 | `_vendor/tilelang-ascend-v0.1.2.000/examples/mha_sink_fwd_bhsd/mha_sink_fwd_bhsd.py:34-101,176-240,378-577` | sink + sliding window 的昇腾实现；`:565-577` 是关于 `m* = max(sink, m_i_final)` 的完整推理 | 已核验（2026-10-01）：重点段逐行阅读；**未运行** |
| A3 | `_vendor/tilelang-ascend-v0.1.2.000/examples/gqa_fwd_varlen/gqa_fwd_varlen.py:31-63,148-172,210-260` | 昇腾 GQA + varlen：显式 `Mask` 输入、`T.mma` + L0A/L0B/L0C 双缓冲 | 已核验（2026-10-01）；**未运行** |
| A4 | `_vendor/tilelang-ascend-v0.1.2.000/examples/lightning_indexer/example_lightning_indexer.py:19-125` | DSA/lightning indexer：Cube 算 `Q·Kᵀ`、Vector 做加权 reduce + `T.tile.topk`/`gather_mask`/`cast` | 已核验（2026-10-01）：逐行阅读；**未运行** |
| A5 | `_vendor/tilelang-ascend-v0.1.2.000/examples/flash_attention/fa_opt/flash_attention_performance_optimization_zh.md`（330 行） | 昇腾 FA 的调优方法论：Double Buffer、单一 bound、最佳 `num_stages`、减少核间同步、scalar 向量化、`T.pipelined` 不支持嵌套 | 已核验（2026-10-01）：本地全文；`code/06` 会定位并打印其中 5 处标题/句子 |
| A6 | `_vendor/tilelang-ascend-v0.1.2.000/examples/flash_attention/fa_opt/bench_mark_zh.md` | 优化收益清单（L1 常驻 / 向量化 / 多 Buffer / 去冗余同步 / CV pipelined / 降同步次数 / axpy）与「达到原生 AscendC 80%」的结论 | 已核验（2026-10-01）：本地全文。**性能数字为厂商自报，作者未复现**；条件：`B=1, Q_N=12, KV_N=1, D=128, S=32K/64K/128K, block_size=128` |
| A7 | `_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md:597,627-636,1345-1390,1422-1500` | ①`shared` 层级 = L1/UB、`fragment` 层级 = L0A/L0B/L0C；②合法搬运对表（**没有 `L0C→UB`**）；③4.1.5.2 节 workspace 消除；④Expert 模式的内存分配原语 | 已核验（2026-10-01）：逐段阅读（文件共 2596 行）；`code/06` 会解析并打印那张搬运表 |
| A10 | `_vendor/tilelang-ascend-v0.1.2.000/src/transform/ascend_workspace_reduction.cc:882-920,999-1004,1455-1481` | workspace 消除 pass 的实现：`copy_l0c_to_ub` → `copy_l0c_to_gm` 的显式改写表，以及 pass 注册入口 | 已核验（2026-10-01）：`grep -n` 定位 + 逐段阅读；**未编译该 pass** |
| A11 | `_vendor/tilelang-ascend-v0.1.2.000/examples/flash_attention/paged_flash_attn_bhsd.py:16-140` | paged KV：`block_table` 间接寻址 | 已核验（2026-10-01）：结构与 A1 同构，`:94` 的 `bidx = block_table[bz, k]` 是唯一关键差异；**未运行** |
| A12 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/allocate.py:140,152,156`；`ascend.py:191,213,275,418`；`ascend_tile.py:343,550,608,1140,1927,2302`；`warpgroup.py:74` | `alloc_L1/alloc_L0C/alloc_ub`、`set_cross_flag/wait_cross_flag/barrier_all/gemm_v0`、`T.tile.{fill,topk,gather_mask,exp,cast,broadcast}`、`T.Scope` | 已核验（2026-10-01）；`code/06` 固定以上行号，`--strict` 无漂移 |

---

## 三、corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2205-14135.md` | https://arxiv.org/abs/2205.14135 | FA1（IO-aware exact attention）：**只用到摘要**里「uses tiling to reduce the number of memory reads/writes between HBM and on-chip SRAM」与「optimal for a range of SRAM sizes」两句定性表述 | 已核验（2026-10-01）：本地为标题+摘要（12 行）；**全文不在 corpus 里**，见「五、待补资料」 |
| R2 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2307-08691.md` | https://arxiv.org/abs/2307.08691 | FA2（更好的并行与工作划分）：论文摘要提到「suboptimal work partitioning … low-occupancy or unnecessary shared memory reads/writes」，这正是本章 3.3 节 `FullRow` 与 4.2 节的动机 | 已核验（2026-10-01）：本地为标题+摘要（12 行）；**全文不在 corpus 里** |
| R3 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2407-08608.md` | https://arxiv.org/abs/2407.08608 | FA3（Hopper 上的 warp specialization 与低精度）：本章只在「`check_inf` 这个名字来自 FA3」这一点上引用它，而那一点实际来自官方例子的注释 | 已核验（2026-10-01）：本地为标题+摘要（12 行）；**全文不在 corpus 里** |
| R4 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md`（相关段落约在第 3145、3629 行附近） | https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html | 「访存 thousands of cycles vs 算术指令 4 cycles」的量级对照（本章 1.2 节） | 已核验（2026-10-01）：与第 1、11 章引用同一段落；条件：未缓存 local/global 访存与 compute capability 7.0 设备 |
| R5 | `../tilelang-corpus/02-tilelang-core-repo/raw/tilelang/examples/`、`.../tilelang/benchmark/` | https://github.com/tile-ai/tilelang | 与锁定版本 `examples/` 的**对照快照**：用于确认「某个例子是不是只存在于快照而不在锁定 tag 里」（本章未发现此类差异） | 已核验（2026-10-01）：按目录清单比对；**未做逐文件 diff** |

> 说明：`R5` 的 corpus 快照 commit 与 `VERSIONS.md` 锁定的 `v0.1.15` 不同。本章所有
> 行号与结论一律取锁定版本 `_vendor/tilelang-v0.1.15/`；R5 只用于「这个文件是否长期
> 存在」这种存在性判断。

---

## 四、本章的数值结论分别来自哪里

为了杜绝「来源不明的数字」，这里把本章出现的每一类数字与它的出处列全：

| 数字 | 出现在 | 出处 | 性质 |
| --- | --- | --- | --- |
| `4N² + 2(Sq+Skv)d`、`2·ceil(Sq/block_m)·Skv·d` 与那张 2.00x 的表 | 1.2、2.5 节、图 1 | `code/01-traffic-model/traffic_model.py` | **本机 CPU 记账模型**（整数运算），不是测量 |
| online softmax 的轨迹与 1e-16 级误差 | 2.3–2.4 节、图 2 | `code/02-online-softmax/online_softmax.py` | **本机 CPU 逻辑验证**（fp64） |
| split-KV 的 `5.6e-17 / 4.7e-3 / 6.1e-01` 三个误差 | 5.4 节 | `code/03-split-kv-combine/split_kv_combine.py` | **本机 CPU 逻辑验证**（fp64 / fp16 模型） |
| 五个变体的 `<1e-12` 一致性、`check_inf` 关闭时的 NaN | 3、5.1、5.2 节 | `code/04-block-flash-attention/block_flash_attention.py` | **本机 CPU 逻辑验证** |
| 昇腾 FA「达到原生 AscendC 80%」、各项优化的 36%/50%/60%/72% | 5.6 节 | `A6`（`fa_opt/bench_mark_zh.md`） | **厂商自报，作者未复现**；条件见 A6 |
| 昇腾 MHA `26.8 us / 320.1 TFLOPS`、GQA `3030 us` vs Torch `2864 us` | 5.6 节 | `A8`（`examples/ascend/flash_attention/README.md`） | **厂商自报，作者未复现**；Ascend 950 / dav-3510 / bf16 / `msprof_detail` |
| dQ 相对 L2 误差 1.024%（dK 0.316%、dV 0.037%） | 5.4 节 | `A8` | **厂商自报，作者未复现**；Ascend950DT、CANN/Bisheng 9.2.0、TileLang `8121f415` |
| attention-sink 的 Triton/TileLang TFLOPS 对比表 | —— | `examples/attention_sink/README.md` | **本章刻意未引用**：它是厂商自报的 H800 性能表，与本章要讲的「结构」无关，引入只会稀释主题 |

---

## 五、待补资料

本章有三处「想要但拿不到」的材料，全部登记在此，由后续版本处理：

| 缺什么 | 为什么需要 | 建议来源 |
| --- | --- | --- |
| **FA1/FA2/FA3 的论文全文** | 现在 corpus 里只有标题+摘要（三个文件各 12 行）。于是：① FA1 的 IO 复杂度上界（`Θ(N²d²M⁻¹)`）无法引用；② FA2 的「减少非矩阵 FLOP」与 warp 划分细节无法引用；③ FA3 的 warp specialization / TMA ping-pong / block quantization 细节无法引用；④ `check_inf` 这个名字只能引用官方例子里的注释，不能引用论文原文 | arXiv 全文（`https://arxiv.org/abs/2205.14135`、`/2307.08691`、`/2407.08608`）的 HTML/PDF 快照，放到 `../tilelang-corpus/03-tilelang-papers/raw/`。核验方式：确认文件包含算法伪码（FA1 Algorithm 1、FA2 Algorithm 1、FA3 Algorithm 1） |
| **`block_causal_attention` 的论文/设计说明** | 该例子实现了「噪声/干净分区 + block causal」的 mask，但锁定版本的例子目录里没有 README 说明它的建模背景，本章只能描述代码行为 | 上游 `tile-ai/tilelang` 的对应 PR 描述，或该算子背后的扩散语言模型论文 |
| **昇腾 `T.pipelined` 的教程页** | `A5` 里引用的 `tutorials/t_pipelined.md` 在锁定 tag 的 docs 目录下**未找到**（该链接指向 `ascendc_pto` 分支） | 上游 `tile-ai/tilelang-ascend` 对应版本的 `docs/tutorials/t_pipelined.md`；核验方式：`find _vendor/tilelang-ascend-v0.1.2.000/docs -name 't_pipelined*'` 为空即可确认缺口 |

本章未采用、也**不建议**补的材料：

* 任何**没有测量条件**的 attention 性能数字（例如某些博客里的「x 倍加速」）；
* 昇腾 attention 的实测带宽/延迟——需要 NPU 与 msProf，本章的环境做不到，
  读者要自己测（见习题 E14 的报告要求）。

## 六、引用规则提醒

* 源码引用一律写成 `_vendor/tilelang-v0.1.15/<相对路径>` 或
  `_vendor/tilelang-ascend-v0.1.2.000/<相对路径>`，并给出行号区间。
* 资料引用一律写成 `../tilelang-corpus/<类别>/raw/<文件>`。
* 本章所有「v0.1.15 里有没有某某」的结论都在锁定工作树上核验过；`code/05` 与
  `code/06` 两个脚本用 `--strict` 把这件事变成了可重复的检查。
* 「已核验（静态）」≠「运行验证」。本章没有任何运行验证。
