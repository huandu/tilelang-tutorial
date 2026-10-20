# 第 37 章 参考资料

本章所有结论都基于下表的本地资料或锁定源码。**版本以
[`../../VERSIONS.md`](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
DeepSelect 本身不在 `_vendor/` 里，它是 corpus 里的仓库快照
（`14-deepseek-ascend-stack/raw/DeepSelect`，commit `bfa4507d935f`，2026-09-30，
见 `.../14-deepseek-ascend-stack/raw/PROVENANCE.md:13`）。

写作环境：macOS (Apple Silicon)、Python 3.11.7、torch 2.1.1（CPU）；
**没有 NVIDIA GPU，也没有昇腾 NPU**。因此：

* 本章的 CUDA / Ascend C 源码**只做了静态阅读**（逐行核对函数、指令、常量），
  **没有编译、没有运行**；
* 本章所有性能数字**只有两个来源**：DeepSelect 仓库自述与它自带的性能图；
* **没有任何一条性能数字来自本机实测**；本章可运行的三个示例都是 CPU 上的算法模型
  （验证级别「CPU 逻辑验证」，见 [`code/README.md`](code/README.md)）。

---

## DeepSelect 一手资料（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/`）

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `.../DeepSelect/docs/DeepSelect-deep-dive.zh.md` | https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.zh.md | **本章主要依据**：算法三步骤与伪代码（`:9-39`）、三条 Notes（`:45-49`）、期望复杂度证明（`:51-92`）、性能综述与渐近结论（`:94-102`）、CUDA 实现技巧（`:104-113`） | 已核验（2026-10-01）：全文 113 行逐行读 |
| R2 | `.../DeepSelect/docs/DeepSelect-deep-dive.md` | https://github.com/deepseek-ai/DeepSelect/blob/main/docs/DeepSelect-deep-dive.md | 同一份报告的英文版（114 行）；用于交叉核对措辞 | 已核验（2026-10-01）：与中文版对照关键段 |
| R3 | `.../DeepSelect/README.md` | https://github.com/deepseek-ai/DeepSelect | 两个使用场景（`:13-37`）、约束（`topk ≤ 4096`、`topk=512`、昇腾只支持 bf16）、性能口径（`:39-62`）、`2 ~ 20x` 的说法（`:3-5`） | 已核验（2026-10-01）：逐行读 |
| R4 | `.../DeepSelect/assets/perf_bf16_cuda.png`、`perf_bf16_ascend.png`、`perf_fp32_cuda.png` | 同上 | 三张性能图；纵轴 effective bandwidth、横轴 vocab/batch；本章 §6.2 的读图依据 | 已核验（2026-10-01）：查看图片内容；**数值只作为「量级」引用，未做精确读数** |
| R5 | `.../14-deepseek-ascend-stack/raw/PROVENANCE.md:13` | — | DeepSelect 快照的 commit（`bfa4507d935f`）与日期（2026-09-30） | 已核验（2026-10-01） |

**说明**：R1 里的三段代码链接（`.../blob/8e70df7.../csrc/cuda_kernels/common_parts.cuh#L1466` 等）
指向上游的另一个 commit（`8e70df7`），**与本地快照 `bfa4507` 的行号不同**。
本章的源码引用一律改用**本地快照的行号**，并另外给出该行所在函数的名称，
便于在两个 commit 之间对照。映射关系见下表。

### R1 的链接 → 本地快照行号

| R1 引用的上游位置 | 指向的东西 | 本地快照（`bfa4507`） |
| --- | --- | --- |
| `common_parts.cuh#L1466` | `__ffs` 枚举命中位 | `csrc/cuda_kernels/common_parts.cuh:1476-1484` |
| `common_parts.cuh#L1431` | mask 生成 | `csrc/cuda_kernels/common_parts.cuh:1413-1428` |
| `common_parts.cuh#L1051` | 浮点加法替整数加法 | `csrc/cuda_kernels/common_parts.cuh:1041`（非 cluster 分支）与 `:1086` |
| `common_parts.cuh#L1478` | warp 级前缀和 | `csrc/cuda_kernels/common_parts.cuh:1457-1465`（ballot 版）；另有 `utils.cuh:6-25`（shfl 版） |
| `common_parts.cuh#L955` | 字节提取 | `csrc/cuda_kernels/common_parts.cuh:944-968`（`prmt`/`lop3`） |
| `common_parts.cuh#L940` | 直方图构建 | `csrc/cuda_kernels/common_parts.cuh:902-925`（`bfe`/`mad`/`red.shared.add`） |

---

## DeepSelect CUDA 源码（本地快照）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `.../DeepSelect/csrc/cuda_kernels/common_parts.cuh:902-925` | `histogram_radix_msb`：用 `bfe`/`mad`/`red.shared.add` 建 256 桶直方图 | 已核验（2026-10-01）：逐行读；**未编译** |
| S2 | `.../DeepSelect/csrc/cuda_kernels/common_parts.cuh:933-970` | `histogram_radix_lsb_for_pivot_msb`：`prmt.b32` / `lop3.b32` 取字节并统计 `pivot` 桶 | 已核验（2026-10-01） |
| S3 | `.../DeepSelect/csrc/cuda_kernels/common_parts.cuh:971-999` | `get_census_counts`：`set.gt.bf16x2` + `add.rn.bf16x2` 统计 `>pivot` / `=pivot` / NaN（含 `static_assert(2*N <= 256)`） | 已核验（2026-10-01） |
| S4 | `.../DeepSelect/csrc/cuda_kernels/common_parts.cuh:1041,1086` | 用 `add.f32` 替代 `add.u32` 做地址自增（§4.4） | 已核验（2026-10-01） |
| S5 | `.../DeepSelect/csrc/cuda_kernels/common_parts.cuh:1106-1128` | `reconstruct`：两趟直方图找 pivot、distort/un-distort、census、算 `eq_quota` | 已核验（2026-10-01） |
| S6 | `.../DeepSelect/csrc/cuda_kernels/common_parts.cuh:1402-1428` | 主筛选循环：`set.gtu.s32.bf16x2` + `prmt.b32` + `dp4a.s32.s32` 生成 `hit_mask` | 已核验（2026-10-01） |
| S7 | `.../DeepSelect/csrc/cuda_kernels/common_parts.cuh:1430-1484` | `__popc`/`__ballot_sync` 算写回偏移；`__ffs` + `mask &= mask-1u` 枚举命中 | 已核验（2026-10-01） |
| S8 | `.../DeepSelect/csrc/cuda_kernels/common_parts.cuh:483-508` | `compute_equal_quota_and_prefix`：并列元素的名额分配（§4.6） | 已核验（2026-10-01） |
| S9 | `.../DeepSelect/csrc/cuda_kernels/bit_utils.cuh:22-45` | `distort` / `un_distort`：把浮点全序映射成无符号整数序（`shr.s32` + `lop3.b32`） | 已核验（2026-10-01） |
| S10 | `.../DeepSelect/csrc/cuda_kernels/bit_utils.cuh:53-63` | `distort_x2`：bf16x2 的 SIMD 版（`prmt.b32` 取符号位） | 已核验（2026-10-01） |
| S11 | `.../DeepSelect/csrc/cuda_kernels/utils.cuh:6-25` | `warp_level_inclusive_prefix_sum`：`shfl.sync.up.b32` 的 log 步扫描 | 已核验（2026-10-01） |
| S12 | `.../DeepSelect/csrc/cuda_kernels/config.h:9-25` | `TopkSelectConfig` 的编译期参数（`value/out_idx`、`sorted_*`、`max_topk`、`num_threads`、`elements_per_round`、`reconstruct_threshold`、`tma_buffer_depth`、`elements_per_segment`、`cluster_size`） | 已核验（2026-10-01） |
| S13 | `.../DeepSelect/csrc/cuda_kernels/v3/topk_select.cuh:46-60` | 一个 CTA 负责一行（`batch_idx = blockIdx.x`）；`end` 与短行处理 | 已核验（2026-10-01） |
| S14 | `.../DeepSelect/csrc/cuda_kernels/v3_cluster/instantiations/`（8 个 `.cu`） | Cluster（`cluster_16`）版本实例清单 | 已核验（2026-10-01）：目录列举 |
| S15 | `.../DeepSelect/csrc/api.cpp:57` | `TORCH_CHECK(topk <= 4096, ...)` 硬约束 | 已核验（2026-10-01） |
| S16 | `.../DeepSelect/csrc/api.cpp:151-190` | 派发逻辑：`num_waves`、cluster 触发条件（`batch_size <= 6 && vocab_size >= 512K && topk <= 1024`，含 `TODO Tune`）、`B/B2/TMA 深度` 的取值 | 已核验（2026-10-01） |
| S17 | `.../DeepSelect/csrc/api.cpp:223-225` | 昇腾分支的三条硬断言（bf16 / int32 / 不支持 `sorted_value`） | 已核验（2026-10-01） |
| S18 | `.../DeepSelect/csrc/structs.h:17-22,39-42` | 输入/输出 stride 对齐要求（CUDA 1024B/32B、昇腾 32B/32B）、`MAX_VOCAB_SIZE`（CUDA `1<<23`、昇腾 `2*1024*1024`）、`MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION` | 已核验（2026-10-01） |

---

## DeepSelect 昇腾源码（本地快照）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| A1 | `.../DeepSelect/csrc/ascend_kernels/kernel.asc:18-74` | `topk_accumulate_histogram`：256 桶 bf16 字节直方图，劈成两个 128 通道向量（`asc_cumulative_histogram_bin0/1`） | 已核验（2026-10-01）：逐行读；**未编译** |
| A2 | `.../DeepSelect/csrc/ascend_kernels/kernel.asc:76-108` | `topk_accumulate_nan_count`：用 `asc_eq`/`asc_reduce_sum` 统计 NaN | 已核验（2026-10-01） |
| A3 | `.../DeepSelect/csrc/ascend_kernels/kernel.asc:110-128` | `topk_find_bucket`：`asc_ge` + `asc_arange` + `asc_reduce_min` 找第一个累计计数达到 rank 的桶 | 已核验（2026-10-01） |
| A4 | `.../DeepSelect/csrc/ascend_kernels/kernel.asc:282-300` | `topk_select_candidates` 的签名与语义（把低 16 位索引先追加、再补高位） | 已核验（2026-10-01） |
| A5 | `.../DeepSelect/csrc/ascend_kernels/kernel.asc:653-780` | `topk_select_run` 主流程：UB 布局别名、`kNumStages` 组 flag、warmup 页建直方图、`kSafeTrigger = kTopkCapacity + kFilteredCapacity` | 已核验（2026-10-01） |
| A6 | `.../DeepSelect/csrc/ascend_kernels/kernel.asc:737-740` | 随机排列的种子来自输入前 8 字节（与报告「independent of x」的偏差，见「待补资料」第 1 条） | 已核验（2026-10-01） |
| A7 | `.../DeepSelect/csrc/ascend_kernels/kernel.asc:844-846` | 只有 `sorted_index` 时才调用 `AscendC::Sort`，且只排 `topk` 个元素 | 已核验（2026-10-01） |
| A8 | `.../DeepSelect/csrc/ascend_kernels/kernel.asc:891-931` | kernel 入口：`__global__ __vector__ __aicore__`，一行一个 block（`get_block_idx()`），按运行期行长选择 Lower/Upper 配置 | 已核验（2026-10-01） |
| A9 | `.../DeepSelect/csrc/ascend_kernels/config.h:17-31` | `TOPK_COMPACT_PADDING=256`、`TOPK_UB_LIMIT_BYTES=248*1024`、`VST_VLD_ALIGN_BYTES=32`、`aligned/ceil_div` | 已核验（2026-10-01） |
| A10 | `.../DeepSelect/csrc/ascend_kernels/config.h:33-88` | `TopkPage` 与 `TopkPermGenerator`：分页、reflect 置换、warmup 页与尾部页的特判 | 已核验（2026-10-01） |
| A11 | `.../DeepSelect/csrc/ascend_kernels/config.h:93-135` | 算法流程的中文注释；`TopkSelectConfig` 参数（`kK/kFilteredCapacity/kWarmupPages/kNumStages/kBlockSize/kShufflePageSize`）；`Topk512/1024/4096Config` 与 pivot 策略 | 已核验（2026-10-01） |
| A12 | `.../DeepSelect/csrc/ascend_kernels/config.h:168-243` | `UnifiedBufferMemoryPlan`：所有偏移、`kCandidateCapacity = k + B + B2`、三份 part、`static_assert(... <= TOPK_UB_LIMIT_BYTES)`、排序区复用输入暂存区 | 已核验（2026-10-01） |
| A13 | `.../DeepSelect/csrc/ascend_kernels/kernel.h:1-19` | 昇腾侧 `Config`（`MAX_TOPK` tier、`ABORT_WHEN_NAN_FOUND`、`SORTED_INDEX`）与入口声明 | 已核验（2026-10-01） |

---

## 接口与测试

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| P1 | `.../DeepSelect/deep_select/interface.py:20-109` | `deep_select.topk(...)` 的签名与每个参数语义；`get_stride_requirement()`；输出张量可能非连续 | 已核验（2026-10-01）：逐行读 |
| P2 | `.../DeepSelect/tests/test.py:20` | `MAX_VOCAB_SIZE`（昇腾 `2*1024*1024-1`、CUDA `(1<<23)-1`） | 已核验（2026-10-01） |
| P3 | `.../DeepSelect/tests/test.py:86-146` | 正确性判据：NaN guard、index 范围、index 唯一、`value = input[index]`、`min(selected) >= max(unselected)`、sorted_index/sorted_value | 已核验（2026-10-01） |
| P4 | `.../DeepSelect/tests/test.py:150-170` | 性能口径：`total_size = 输入字节 + 输出字节`，打印 `TB/s` 与相对 `torch.topk` 的 `speedup` | 已核验（2026-10-01） |
| P5 | `.../DeepSelect/tests/test.py:238-253` | 性能用例形状：Indexer `topk ∈ {512,1024}`、`batch ∈ {6,256,512,768,4096}`、`seqlen ∈ {256…1048576}`；Sampler `vocab=129280, topk=512, fp32, sorted_value=True` | 已核验（2026-10-01） |

---

## TileLang 侧对照（锁定源码 `_vendor/`）

本章只在 §5.4、§5.5 用到 TileLang 侧材料，用于说明「为什么 DeepSelect 不是
TileLang 实现」以及判据：

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| V1 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend_tile.py:550-600` | `T.tile.topk` 的文档：「by sorting the source data」；要求静态形状（`:587` 抛错） | 已核验（2026-10-01）：读函数与 docstring |
| V2 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:132-154` | 昇腾方言的写法（`T.SimtVF`/`T.SimdVF`/`T.alloc_*`/`T.Pipelined`）；与 Ascend C 的分工对照表；「自动调度不保证把任意 `T.Parallel` 张量化为 SIMD」 | 已核验（2026-10-01）：逐行读 |
| V3 | `_vendor/tilelang-v0.1.15/tilelang/language/`、`tilelang/ascend/language/`（不存在性） | 确认 M 树这两个目录下**没有** `sort` / `topk` 原语（`grep -rn "def sort"` 与 `grep -rn "def topk"` 均无命中） | 已核验（2026-10-01） |

---

## 关于性能数字的说明

**本章不提供任何本机实测数字。** 本章出现的性能相关数字只有三类：

| 数字 | 来源 | 条件 |
| --- | --- | --- |
| `2 ~ 20x speedup` vs `torch.topk` | R3（README `:3-5`） | 仓库自述；硬件/版本/口径以仓库为准，**作者未复现** |
| 三张性能图的柱高（≈5 TB/s / ≈3.5 TB/s / ≈2.8 TB/s 等） | R4 | **从图上读的近似量级**；测试条件见 README（`topk=512`、`B` 为 batch、横轴为 vocab/batch） |
| `E[W]` 界、`O(k log(N/k))`、`B2` 的 trade-off 表 | R1（`:51-102`）+ 本书 `code/03` 的复算 | **公式算出来的渐近量级**，不是带宽也不是加速比 |

没有任何「本机实测的延迟 / 吞吐 / 带宽利用率」出现在本章。

---

## 待补资料

1. **昇腾侧的随机种子取自输入，与报告的假设不一致。**
   报告要求块排列「independent of x」（R1 `:23,53`），而昇腾实现把种子设为
   `hash(输入前 8 字节) ^ valid_len`（A6）；`TopkPermGenerator` 本身也是模运算伪随机
   （A10）。这既影响「均匀性」，也影响「独立性」，因此 §3 的期望界**不能原样套用**。
   本章按事实写出并标注为偏差。建议向上游确认：这是有意的工程取舍，还是文档与实现的
   不一致？
2. **报告的推荐参数与发布实例不一致。**
   R1 `:15` 推荐 `B=B2=1024`（`k=512`），而 CUDA 实例用 `B=4096/8192`、`B2=4096`
   （S16）。按 §3.6 的公式，后者的界更松（`N=1M` 时 ≈78k vs ≈28.8k）。
   本章把它作为「算法建议值 vs 硬件调优值」的例子，但**没有实测证据**说明哪一组更快。
   建议补一份在真机上扫 `B`/`B2` 的数据（至少 `(1024,1024)` 与 `(8192,4096)` 两组）。
3. **`2^22` 与 `1<<23` 的关系未核清。**
   R1 `:109` 给的浮点模拟条件是 `0 ≤ x,y ≤ 2^22`；代码里有
   `MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION = 1u << 23`，并用
   `static_assert(MAX_VOCAB_SIZE <= MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION)`
   把 `vocab_size` 卡在 `< 2^23`（S18）。两者是否指同一个量、`2^23` 是否也有严格的
   位模式论证，本章未找到直接依据。建议向上游确认或补一段推导。
4. **昇腾侧的实测性能与硬件规格缺失。**
   本章只有一张昇腾性能图（R4），没有测试机器型号、CANN 版本、编译选项等条件说明，
   也没有 Ascend 950 的带宽/UB 容量规格页。因此 §6.2 只能给出「量级」，
   给不出「达到硬件带宽的百分之多少」。建议补抓测试环境说明与硬件规格。
5. **小 batch 的 cluster 版本缺少独立材料。**
   `csrc/cuda_kernels/v3_cluster/` 只有 8 个 instantiation 文件（S14），
   DSM/cluster 归并的具体实现细节在 `common_parts.cuh` 里（`USE_CLUSTER_ADDRESSING`
   分支），本章只讲到「切分 + CTA0 归并」这一层。建议补一份 cluster 路径的
   内存/同步说明，或直接引用 Hopper cluster 的官方文档。
6. **`TORCH_CHECK(topk <= 4096)` 的边界并未在报告里论证。**
   `topk ≤ 4096` 是硬约束（S15），但报告只说「更大的值不支持」。
   为什么恰好是 4096，本章无法从材料里推出（可能与 UB/shared memory 预算有关）。
   建议向上游确认并补一条说明。
