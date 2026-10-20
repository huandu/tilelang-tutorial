# 第 39 章 参考资料

本章是收口章，因此它的引用分成三类：**前面七章的结论**（用“见第 N 章”引用，
不重复路径）、**本章新增的一手材料**（mHC 论文与官方文章）、以及
**第五部分各仓库的 README / 源码**。

版本以 [`../../VERSIONS.md`](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。
第五部分引用的 DeepSeek 仓库是 2026-10-01 抓取的 **corpus 快照**，
每条 commit 记在 `../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md`；
凡引用这些仓库的文件，行号都是**那次快照**里的行号，仓库一更新就会漂移。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章所有 kernel 示例的最高验证级别是“仅静态核对”；`code/` 下的三个账本脚本
是纯标准库程序，在写作机上**实际运行过**（验证级别：CPU 逻辑验证）。

---

## 一手材料：mHC 论文与官方发布文章

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| M1 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md`（全文） | 第 39 章 §2.3、§2.5、§2.9 的论文依据；§4.3.1/§4.3.2/§4.3.3 三节 | 已核验（2026-10-01）：逐段读 §4.3（`:408-563`）、§5.1、§5.4 与附录配置表 |
| M2 | 同文件 `:410-411` 与 `:219` | “marginal training overhead of only 6.7%”“introduces only a 6.7% additional time overhead when expansion rate n=4” | 已核验（2026-10-01）：逐字读；**论文自报，非本书实测** |
| M3 | 同文件 `:413-503`（§4.3.1 Kernel Fusion） | “把除以范数重排到矩阵乘之后”；三个专用 kernel（`ℋ_pre`/`ℋ_post`/`ℋ_res`）；读 `(3n+1)C→(n+1)C`、写 `3nC→nC`；“除 Eq.(14)–(15) 以外大部分 kernel 用 TileLang 实现” | 已核验（2026-10-01）：逐段读 |
| M4 | 同文件 `:504-539`（§4.3.2 Recomputing） | 丢弃 mHC 中间激活、反向重执行；`L_r* ≈ sqrt(nL/(n+2))`；重算块不得跨流水线 stage 边界 | 已核验（2026-10-01）：逐段读 |
| M5 | 同文件 `:541-563`（§4.3.3 Overlapping Communication in DualPipe） | MLP 的 `ℱ_post,res` 放专用高优先级 compute stream；attention 长算子不用 persistent kernel；重算与流水线通信解耦；配图长度“illustrative only” | 已核验（2026-10-01）：逐段读 |
| M6 | 同文件 `:296-345`（§3.2 System Overhead） | “HC increases the memory access cost by a factor approximately proportional to n”，§4.3 优化要解决的问题 | 已核验（2026-10-01）：读关键段落 |
| M7 | 同文件 `paper-deepseek-mhc.pdf`（同目录） | 公式与图的原始排版（文本抽取未保留公式与图） | 存在；**未逐页比对** |
| O1 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:12` | 五个计算/通信组件的一句话定位 | 已核验（2026-10-01）：逐字读 |
| O2 | 同文件 `:13` | “基于昇腾 950 的 128 卡超节点方案”（唯一一处） | 已核验（2026-10-01）：逐字读 |
| O3 | 同文件 `:11` | “对昇腾 Ascend C 底层指令进行封装、提供高级语言的编程方式、同时不损失硬件性能”；“训练中用到的每一个 TileLang 算子，在昇腾上都有对应的高性能实现” | 已核验（2026-10-01）：逐字读 |
| O4 | 同文件 `:16-27` | 六个开源项目的仓库地址 | 已核验（2026-10-01）：逐字读 |
| P1 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md` | 各仓库的 upstream、commit 与抓取日期 | 已核验（2026-10-01） |
| S1 | `../tilelang-corpus/sources/sources.yaml`（`paper-deepseek-mhc` 条目） | mHC 论文的可信度标注（`trust: primary`） | 已核验（2026-10-01） |

---

## corpus 引用：性能表与行为约定

抓取日期 2026-10-01（Asia/Shanghai）。行号为快照内行号；
在 `chapter.md` 里为了读数方便，有些引用写成 `仓库名/README.md:行号`，完整前缀都是
`../tilelang-corpus/14-deepseek-ascend-stack/raw/`。

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| C1 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:9-21` | **卡间带宽表**与全部测量条件（Ascend 950DT、CANN 9.2.0、PoC HDK、netlayer 1、16384 tokens/rank、hidden 7168、top-6/256 experts、10 warmup + 50 samples、刷 cache）；“EP≤32 时 dispatch 达物理 payload 带宽上限的 90–95%” | 已核验（2026-10-01）：逐行读 |
| C2 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:13-19` | dispatch / combine 的 EP8–EP128 带宽区间 | 已核验（2026-10-01） |
| C3 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:15-19` | 带宽表的五行数据 | 已核验（2026-10-01） |
| C4 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:40-42` | HDK 状态：Q3 商用版“planned for mid-October 2026, around October 15”；PoC 配置“not a publicly distributed release”；“these measurements are not results from that unreleased commercial version” | 已核验（2026-10-01）：逐字读；**保留 planned 措辞** |
| C5 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:60-74` | Features 与 API 兼容表；`do_expand=True`、`do_handle_copy=False` 两条硬约束 | 已核验（2026-10-01）：逐行读 |
| C6 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:80-84` | `Ongoing`：Bucket all-gather 可用 / reduce-scatter 与 all-reduce 仍在做；LB 两个入口的昇腾 kernel 未实现；PP 与 Engram experimental；hybrid / CPU Engram / graph capture 不支持 | 已核验（2026-10-01）：逐字读 |
| C7 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:131` | `expert_alignment` 的注释“Match the grouped GEMM's token alignment” | 已核验（2026-10-01） |
| C8 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:175-196` | “先发起、后等待”的 dispatch / combine 骨架 | 已核验（2026-10-01） |
| C9 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:29,72,84`；`:234` | 奇数 rank 不支持；`do_handle_copy` 必须为 False；hybrid / CPU Engram / graph capture 不支持；`EP_AVOID_RECORD_STREAM` 是 Bucket all-gather 的前置条件 | 已核验（2026-10-01） |
| C10 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/csrc/buffers/ep.hpp:398` | `EP_HOST_ASSERT(async_with_compute_stream)`，`defer_epilogue` 必须搭配它 | 已核验（2026-10-01）：`awk` 定位到该行 |
| C11 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/csrc/buffers/ep.hpp:79,85` | `lb_prefetch_weights` / `lb_reduce_grads` 在昇腾上 `EP_HOST_UNREACHABLE` | 已核验（2026-10-01） |
| C12 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/csrc/buffers/ep.hpp:116` | `EP_HOST_ASSERT(do_expand)`，非展开布局不支持 | 已核验（2026-10-01） |
| C13 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:95-104` | Dense GEMM 表：FP8×FP8 在 `4096×7168×16384` 上 1117.6 µs / 861 TFLOPS / 99.5%；“up to 99.8%” | 已核验（2026-10-01）：逐行读 |
| C14 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:108-122` | FP8 推理 GEMM 的 14 行；`M=128,N=7168,K=16384` 是 54.8 µs / 549 TFLOPS / 2456 GB/s；`M=4096` 同形状是 861 TFLOPS / 186 GB/s | 已核验（2026-10-01）：逐行读 |
| C15 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:129-140` | M-Grouped GEMM 表；`#G=8, M/group=4096, N=4096, K=4096` 是 1356.5 µs / 861 TFLOPS / 180 GB/s | 已核验（2026-10-01）：逐行读 |
| C16 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:142` | “The kernel is FIX-pipe bound rather than compute bound, saturating the FIX pipe at 99% utilization” | 已核验（2026-10-01）：逐字读 |
| C17 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:144-149` | MQA logits 表：Prefill FP8 `Q=4096` 302.6 µs / 681 TFLOPS / 283 GB/s；Decode FP8 `Q=256` 150.9 µs / 565 TFLOPS / 2006 GB/s | 已核验（2026-10-01）：逐行读 |
| C18 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:151-164` | MegaMoE 表（含 `Communication bandwidth` 一列与 EP8 / top-k=6 / 一个共享专家的条件） | 已核验（2026-10-01）：逐行读 |
| C19 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:166-176` | HC Prenorm GEMM 表与“nearly saturates the HBM write bandwidth”；`M=8192` 那行 136.7 µs / 82 TFLOPS / 3463 GB/s | 已核验（2026-10-01）：逐行读 |
| C20 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:190` | “The mHC kernel is backed by Tilelang” | 已核验（2026-10-01）：逐字读 |
| C21 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:21-29` | Requirements：Ascend 950 系、CANN 9.20（`bin/bisheng`、`bin/ld.lld`）、Python 3.10+、C++20 `<format>`、`tilelang` 用于 HC prenorm | 已核验（2026-10-01）：逐行读 |
| C22 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/csrc/jit_kernels/heuristics.hpp:46-71` | `get_mad_flops_per_cycle` 与 `HWDesc`：`gm_bw_gbps=3500`、`l2_bw_gbps=4500`、`aic_clock_mhz=1650`、`bf16_flops_per_cycle=8192`、`num_ai_cores=32` | 已核验（2026-10-01）：逐行读；**这是本章唯一的“硬件模型常量”来源** |
| C23 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/csrc/apis/gemm.hpp:591,615` | masked grouped GEMM 两处“not implemented in this backend” | 已核验（2026-10-01）：定位到该行 |
| C24 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/csrc/apis/mega_moe.hpp:145-150,159` | MegaMoE 的配置断言（`rm/rn/rk = 1/1/32`、`swiglu` + `fast_math`）与 `runtime->get_num_sms() == kNumAICores` | 已核验（2026-10-01）：逐行读 |
| C25 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py:1-8` | TileLang 实现的文件头与 import（`tilelang.ascend.language`、`make_ascend_compact_nz_layout`） | 已核验（2026-10-01）：读前 10 行；细节见第 35 章 |
| C26 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:17-28` | Lightning Indexer 场景：bf16、`topk ≤ 4096`、主要优化 `topk=512`；两条使用建议 | 已核验（2026-10-01）：逐行读 |
| C27 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:29-38` | 采样场景：fp32、`vocab ≈ 128K`；“Currently, only the CUDA implementation supports `torch.float32` … The Ascend implementation supports `torch.bfloat16` only” | 已核验（2026-10-01）：逐字读 |
| C28 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:41-44` | 性能指标是“effective memory bandwidth”以及为什么不用 FLOPs | 已核验（2026-10-01）：逐字读 |
| C29 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:85-99` | `deep_select.topk` 的签名、`indices_type` 的昇腾限制、输出 stride 的非连续可能 | 已核验（2026-10-01）：逐行读 |
| C30 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:13` | 昇腾稀疏 prefill / decode 的发布说明与 410 / 360 TFlops | 已核验（2026-10-01）：逐字读 |
| C31 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:53,63` | “up to 410 TFlops, which is 95% of the theoretical hardware peak”“up to 360 TFlops … 83%” | 已核验（2026-10-01）：逐字读 |
| C32 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:39,43,53` | 融合 kernel 需要 TileLang / Tile-Kernels / DeepGEMM，且“currently supports CUDA only; Ascend is not supported” | 已核验（2026-10-01）：逐行读 |
| C33 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:111-151` | decode 的 `get_mla_metadata` + `flash_mla_with_kvcache` 用法与 `indices` 语义 | 已核验（2026-10-01）：逐行读 |
| C34 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:136,172` | decode 只支持 FP8 / FP4 KV cache；prefill 版没有 batch 维 | 已核验（2026-10-01）：逐字读 |
| C35 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:9-19` | “同一份 Python API 跑在 NVIDIA 与昇腾上”与功能清单 | 已核验（2026-10-01）：逐行读 |
| C36 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/moe/topk_gate_kernel.py:7-43` | 路由算子入口：校验 → 选后端 → 启动；tie-break 与连续输出的契约 | 已核验（2026-10-01）：逐行读；**未运行** |
| C37 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/quant/swiglu_forward_kernel.py:40-50` | SwiGLU 前向的入口与后端分派 | 已核验（2026-10-01）：逐行读；**未运行** |
| C38 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/multilayer_recompute_cuda.py:111-112` | 重算时残差先转 bf16 再转回 fp32 | 已核验（2026-10-01）：逐行读；**未运行** |
| C39 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepJIT/README.md:11-16,18-21` | 双后端 compile/load/launch；WIP 中的缓存预热与 Python 编译 API | 已核验（2026-10-01）：逐行读 |
| C40 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DualPipe/README.md:3` 与同文件的“Pipeline Bubbles and Memory Usage Comparison”表 | 双向流水与全重叠的定位；1F1B / ZB1P / DualPipe / DualPipeV 的气泡与显存对照 | 已核验（2026-10-01）：逐行读 |
| C41 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/EPLB/README.md:1-40` | 冗余专家策略、分组受限路由、层级 / 全局两种策略、`rebalance_experts` 的返回 | 已核验（2026-10-01）：逐行读 |
| C42 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM/README.md:25` | NVIDIA 侧“1550 TFLOPS on H800”（2025-04-18 的旧数字），**只用来举例说明“不能跨机器比较”** | 已核验（2026-10-01）：逐字读 |

---

## 前面章节的引用

本章不重复路径，正文里一律用“见第 N 章”；下面是这张对应关系一览。

| 用在哪 | 引用的章节与小节 |
| --- | --- |
| §1.1 的收口表 | 第 32–38 章各自的小结 |
| §2.1–§2.4 训练链路 | 第 16 章 §2.3–§2.6、§7；第 32 章 §5.4；第 33 章 §4、§5.5、§7.1；第 34 章 §1.2、§3.1、§4.3、§6.2–§6.5、§7.1；第 35 章 §5.4；第 38 章 §1.2、§1.3、§2.3、§3.4、§8.3 |
| §2.6 训练侧调度件 | 第 38 章 §5.4；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DualPipe/README.md`、`../tilelang-corpus/14-deepseek-ascend-stack/raw/EPLB/README.md` |
| §2.9 三层重叠 | 第 38 章 §9.3；mHC 论文 §4.3.3 |
| §3.1–§3.5 推理链路 | 第 16 章 §4.3、§7；第 18 章 §2.1–§2.6、§7；第 32 章 §6.1、§6.4；第 35 章 §5.3、§5.5；第 36 章 §1.1、§3.3、§5.2、§5.7；第 37 章 §5.1、§6.1、§6.2 |
| §3.7、§4 预算与判类 | 第 28 章 §2.2–§2.4；第 35 章 §5.1–§5.7；第 38 章 §7.1–§7.4；第 14 章 §1.2、§7.1 |
| §5 瓶颈检查清单 | 第 12 章；第 19 章 §8；第 21 章；第 26 章；第 28 章 §2.3、§7；第 35 章 §3.4、§4.4；第 36 章 §5.3–§5.5 |
| §6 路线图 | 第 13–18 章；第 17 章 §2.3、§7.3、§9；第 18 章 §7；第 28 章 §2–§5；第 33 章 §5.5；第 34 章 §3.1；第 38 章 §3.4、§9.3 |
| §7 现状清单 | 第 32 章 §8.3；第 33 章 §8.3；第 35 章 §5.6、§8；第 36 章 §6.4；第 37 章 §5.1；第 38 章 §8.1–§8.4 |
| §8 收口到贡献 | 第 30 章；第 31 章 §2.3、§7、§9.3、§10、§11 |
| §9 全书回顾 | 第 1–6、7–19、20–29、30–31 章各自的学习目标 |

---

## 关于性能数字的说明

**本章不提供任何昇腾上的实测数字。** 正文里出现的每一个数字都属于下面四类之一：

| 数字 | 来源 | 本章怎么写 |
| --- | --- | --- |
| bf16 432.5 / fp8 865 / fp4 1730.1 TFLOPS；HBM 3500 / L2 4500 GB/s | C22（`HWDesc`），**本书复算** | 写“本书复算”＋公式 |
| `distinct_ranks` 表；§3.7 的两个比值；`2MNK` 的口径标定 | **本书复算**（公式来自第 38 章 §1.3、第 16 章 §2.6） | 写“本书复算”＋公式 |
| Dense / M-grouped / MQA / MegaMoE / HC prenorm 五张表（C13–C19） | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md`（**厂商自报**） | 连条件一起写：Ascend 950DT、CANN 9.20、`bench_msprof`、冷 L2 |
| EP8–EP128 带宽（C1–C3） | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md`（**厂商自报**） | 连条件一起写：PoC HDK、netlayer 1、16384 tokens/rank… |
| 410 / 360 TFlops（C30、C31） | `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md`（**厂商自报**） | 写“up to”＋“相对理论峰值” |
| 2–20×（相对 `torch.topk`） | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md`（**厂商自报**） | 写清“相对加速，不是绝对带宽” |
| 6.7% 额外开销（M2） | mHC 论文（**论文自报**） | 写清 `n=4`、in-house 大规模训练、经过 §4.3 优化 |

唯一的跨类“除法”是正文 §4.4 结尾那一次（3500 ÷ 374 ≈ 9.4），
它在正文里被明确标注为“量级比值、只用于判类”。

---

## 待核事项

本章继承第 32 章的待核清单，没有新增：

1. **DeepSelect 的昇腾硬件要求**：这份快照的 README 没有写明 Ascend 950 / CANN 版本
   （第 32 章“待核事项”第 1 条）。
2. **FlashMLA README 引用的模型页**：写作环境无法访问 HuggingFace，未能核实
   （同第 2 条）。
3. **128 卡超节点的硬件细节**：官方文章只有一句话（O2），corpus 里没有对应白皮书；
   本章 §4.1 的“卡间带宽”一行因此只写实测口径，不写互联规格（同第 3 条）。
4. **DeepEP-Ascend 的 HDK 计划日期**：C4 写的是厂商**发布计划**
   （2026-10-15 前后），不是既成事实；引用时必须保留 planned 措辞。

另外登记一条**本章特有的口径不确定性**：

5. **MQA logits 表的 `Computation` 列口径不明。** C17 的两行是本章 §3.7 唯一
   “复算不上”的两行：按 `2·Q·K·heads·dim` 算出的值比表格列高约三分之一，
   而 README 没有给出该列公式。本章的处理方式是**只报事实、以表格列为准**，
   并把这个不一致登记在此，供后续版本处理。

---

## 本地工具与机械校验

本章的三个 `code/` 示例都是纯标准库脚本，写作时**实际运行过**，
输出抄在各目录的 README 里（验证级别：CPU 逻辑验证）：

* `code/01-train-step-budget/train_step_budget.py`
* `code/02-inference-latency-model/inference_latency_model.py`
* `code/03-bottleneck-map/bottleneck_map.py`（含 `--verify`，对本书 `chapters/` 下
  10 个章节逐一核验目录存在）

图的 SVG 规格与章节自测题的结构由站点的构建脚本校验
（见本章目录同级的图与自测题约定），它检查的是
“JSON 合法、id 与文件名一致、kind/tone 合法、能渲染出 SVG”，
**不检查**内容正确性或引用可达性。
