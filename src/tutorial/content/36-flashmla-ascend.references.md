# 第 36 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[`../../VERSIONS.md`](../../VERSIONS.md) 为准**。

写作环境：macOS (Apple Silicon)、Python 3.11，**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章对 `csrc/ascend_kernels/**` 的引用一律是**静态阅读**，
不存在“运行验证”；`code/` 下的两个算术脚本与一个 NumPy 脚本在本机运行过，
验证级别写在各自的 `README.md`。

本章用“**报告**”指下面 R1（中文版）；“报告 `:N`”指该文件的第 N 行。

---

## 一手材料：FlashMLA 昇腾技术报告与代码

| id | 路径 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| R1 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/docs/20260930-ascend-prefill-deep-dive-zh.md` | 本章**主要依据**：算法回顾（`:11-73`）、四个讨论（`:77-164`）、流水（`:166-210`）、优化技巧（`:212-252`） | 已核验（2026-10-01）：全 255 行逐行读；本章“报告 `:N`”均指此文件 |
| R2 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/docs/20260930-ascend-prefill-deep-dive.md` | 英文版，用于核对术语（`burst_count`/`src_stride`、`skip scale`、`bank`、`icache` 等） | 已核验（2026-10-01）：对照中文版抽查关键段 |
| R3 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md` | 性能口径（`:13,53,63`：410 / 360 TFlops）；模型与仓库定位（`:7`）；KV cache 格式（`:138-143`）；`indices` 语义（`:157`）；运行前提 CANN ≥ 9.2.0（`:85-88`） | 已核验（2026-10-01）：相关段落逐行读；**未运行** |
| R4 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/csrc/ascend_kernels/prefill/sparse/config.h` | `MMA_M`（`:28`）、`RESCALE_THRESHOLD`（`:43`）、`NUM_VEC_CORE_PER_AI_CORE`（`:53`）、`B_TOPK`（`:55`）、`OUTPUT_FRAG_D_VO`（`:59`）、各 buffer 档数（`:67-69`）、`GRID_SHAPE`（`:74`）、padding（`:81`）、SS buffer（`:110-112`） | 已核验（2026-10-01）：全 206 行逐行读；**未编译** |
| R5 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/csrc/ascend_kernels/prefill/sparse/kernel_body.h` | VECTOR 侧 7 个 `run_*`（`:954,983,1018,1078,1114,1186,1195`）、`systolic_arr_activate`（`:1289-1322`）、`SYSTOLIC_ARR_DEPTH`（`:1273`）、ND2NZ VF（`:277-331`）、dequant+ND2NZ VF（`:334-462`）、softmax VF（`:464-685`）、`add_and_scale_o_vf`（`:686-726`）、CUBE 侧 `run_*`（`:1367,1403,1493`）、L1 bank 拆分（`:907-912`）、SS buffer 读写（`:1189,1585-1588`）、gather2（`:1043-1067`）、icache 注释（`:595,1331`） | 已核验（2026-10-01）：全 1701 行逐行读；**未编译、未运行** |
| R6 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/csrc/ascend_kernels/prefill/sparse/kernel.h`、`instantiations/*.asc` | `Config` 结构（`kernel.h:5-11`）、实例化参数 `{Prefill, 64, 640, false, V41, V41}`（`instantiations/v41_h64_prefill_nosink.asc:7`） | 已核验（2026-10-01）：逐行读 |
| R7 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/csrc/cuda_kernels/kv_cache_format.h` | FP8 / FP4 每 token 字节数（`:5-19`：528 / 288）、量化粒度（32 / 16） | 已核验（2026-10-01）：逐行读 |
| R8 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/csrc/params.h`、`csrc/api/sparse_prefill.cpp` | `SparseAttnFwdParams` / `SparseAttnDecodeParams` 的字段与语义；昇腾 dispatch 只支持 `HEAD_64`（`api/sparse_prefill.cpp:63-83`） | 已核验（2026-10-01）：逐行读 |
| R9 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/tests/test-sparse-prefill.py` | TFlops 与带宽的**测量口径**（`:46-51`：理论 FLOPs ÷ kernel 时间）；形状用例（`:70-90`） | 已核验（2026-10-01）：相关段落逐行读；**未运行** |
| R10 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/csrc/ascend_kernels/prefill/sparse/fast_div_mod.h` | decode 侧“index → (block_idx, offset_in_block)”用的 `FastDivMod` | 已核验（2026-10-01）：逐行读 |

## 背景材料：DSA 与 NV 侧对照

| id | 路径 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| R11 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSeek-V3.2-Exp/README.md` | DSA 的定位与“几乎不掉点”的官方表述（`:47-49`）；DSA 首次公开的语境 | 已核验（2026-10-01）：相关段落逐行读 |
| R12 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/docs/20250929-hopper-fp8-sparse-deep-dive.md` | NV 侧对照：dequantization-bound 的分析（每个 token 约 50 周期 vs MMA 34 周期）、crossover、DSM | 已核验（2026-10-01）：全 52 行读；**未运行** |
| R13 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/docs/20250422-new-kernel-deep-dive.md` | NV 侧对照：compute-bound 判据的另一种算法（FLOPs/byte 与 660 TFlops 口径） | 已核验（2026-10-01）：全 77 行读；**未运行** |
| R14 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md` | 2026-09-30 开源清单的官方正文（六个仓库的定位） | 已核验（2026-10-01）：全文读 |
| R15 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md` | FlashMLA 等仓库快照的 commit 与日期（`FlashMLA` = `2e5429fc5653`，2026-09-30） | 已核验（2026-10-01） |

## 锁定源码：本章引用的 `T.*` 原语

本章 §6 的三分类表要给“TileLang 能表达”找证据，逐个在主干 `v0.1.15`（记作 **M 树**）里确认：

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py:107,179` | `T.SimtVF(threads, latency)` / `T.SimdVF(latency)` 的签名 | 已核验（2026-10-01）：`grep -n "def SimtVF\|def SimdVF"` |
| S2 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/simd.py` | `T.simd.*`：100 个 `def`，含 `vld`（`:324`，带 `post_inc`）、`vld2`（`:417`）、`vsts`（`:457`）、`vsstb`（`:482`，带 `update`）、`make_ubuf_ptr`（`:524`）、`pset`/`pge`/`update_mask`（`:111-128`）、`vmull`/`vintlv` 等 | 已核验（2026-10-01）：`grep -n "^def "` 与关键函数 docstring 逐段读 |
| S3 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/sync.py:28,51,124,151` | `T.ascend_set_flag` / `T.ascend_wait_flag` / `T.ascend_cross_core_set_flag` / `T.ascend_cross_core_wait_flag`；`ascend_pipe_barrier`（`:8`） | 已核验（2026-10-01）：`grep -n "def "` |
| S4 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/annotations.py:19` | `T.annotate_buffer_versions(buffer_versions_map)` | 已核验（2026-10-01） |
| S5 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:26,37,46,55` | `alloc_l1/l0a/l0b/l0c` 的 scope | 已核验（2026-10-01） |
| S6 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py:53` | `T.dual_copy(src, dst, unit_flag_ctrl, l2_cache_ctrl)` | 已核验（2026-10-01） |
| S7 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/gemm_op.py:110,168` | `T.gemm` / `T.gemm_blockscaled` 的签名与 `unit_flag_ctrl` | 已核验（2026-10-01） |
| S8 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/mode.py` | `T.set_hf32_mode` / `T.set_mmad_direction` / `T.set_atomic` | 已核验（2026-10-01）：`grep -n "def "` |

## 本章示例代码

| id | 路径 | 用途 | 验证级别 |
| --- | --- | --- | --- |
| C1 | `code/01-head-roofline/head_roofline.py` | 复现报告 `:90-111` 的算存比表（`cube_cycles` / `mem_cycles`） | **本机运行过**（macOS + Python 3.11）；输出与报告逐行一致 |
| C2 | `code/02-fixpipe-bound/fixpipe_bound.py` | 复现报告 `:142` 的 FixPipe 推导，并解出 `B_TOPK > 73.14`；附带 MTE2 队列深度的算术 | **本机运行过** |
| C3 | `code/03-online-softmax-skip-scale/online_softmax_skip_scale.py` | NumPy 版“带 skip-scale 的 online softmax”，与一次算完的参考实现对拍 | **本机运行过**；只验证算法等价性 |
| C4 | `code/04-tilelang-expressibility/expressibility_map.py` | 把本章技术点映射成“A/B/C 三类”的可查询清单 + 源码路径存在性检查 | **本机运行过**（静态核对工具） |

## 教材内部交叉引用

按本书约定，跨章一律写“见第 N 章”，不写死路径。本章与以下章节有直接关系：

| 章 | 关系 |
| --- | --- |
| 第 15 章 | online softmax 的递推与等价性证明（本章 §2.2 直接引用） |
| 第 19 章 §2 | 昇腾机器模型（AIC/AIV、GM/L1/L0/UB、MTE1/2/3、FixPipe） |
| 第 22 章 | 流水线与同步插入（本章 §4 是它的真实样本） |
| 第 23 章 | 昇腾自动调度（本章 §6 讨论“哪些能交给编译器”） |
| 第 25 章 | 昇腾后端、`T.*` 原语清单、两条 codegen 路线（本章 §6 的字典） |
| 第 27 章 | 低精度与量化（FP8/FP4 KV cache 的数值问题） |
| 第 28 章 | 算子设计方法论（本章的四个讨论是它的实例） |
| 第 35 章 | DeepGEMM-Ascend：FixPipe bound、scale factor 打包、协程流水 |

## 待补资料

按本书写作规范登记“核不到或报告未写清”的条目，**不自行发明**：

1. **`RESCALE_THRES = 6` 的选值依据。** 报告 `:57` 只说“一般取 6”。
   §2.3 给的“`exp(6)` 量级 + rescale 次数”是本文的解释，非原文结论。
2. **流水空窗“大约 1024 周期”的推导。** 报告 `:184` 只给数字。
3. **Ascend 950 的 L0C / L1 / UB 精确容量。** 报告说 L0C 被 256 KB 撑满、
   L1 是两个 256 KB bank（`:116,242`）；第 19 章的容量表是 Atlas A2 口径，
   二者不能直接互换。建议来源：昇腾 950 平台的官方硬件规格文档。
4. **410 / 360 TFlops 对应的精确形状与批大小。** README 与报告都只给硬件与 CANN
   版本（`:13,53,63`；`README.md:85-88`）。建议来源：`tests/` 的 benchmark 命令
   加上官方发布说明里的复现步骤。
5. **`cuda_kernels/` 与 `ascend_kernels/` 是否有共享的 head 划分约定。**
   本章只按昇腾侧的 `config.h` 讨论，未展开 NV 侧的 `head64`/`head128` 实例化差异。
