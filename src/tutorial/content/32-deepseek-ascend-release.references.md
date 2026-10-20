# 第 32 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[`../../VERSIONS.md`](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**

第五部分引用的 DeepSeek 仓库**不是 `_vendor/` 里的锁定源码**，而是 corpus 里 2026-10-01
抓取的快照；每个仓库的 upstream 与 commit 记在
`../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md`。凡引用这些仓库的
README，行号都是**那次快照**里的行号；仓库一更新，行号就可能漂移。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章（以及整个第五部分）的示例最高验证级别是“仅静态核对”，
**没有任何一条性能数字来自本机实测**。

---

## 一手材料：官方文章与快照来源

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| O1 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md:9` | 发布总述与“一一对应” | 已核验（2026-10-01）：逐字读 |
| O2 | `.../deepseek-article-20260930.md:10` | TileLang 的由来与“先 NV 后昇腾” | 已核验（2026-10-01）：逐字读 |
| O3 | `.../deepseek-article-20260930.md:11` | “封装 Ascend C / 高级语言 / 不损失性能”与“每个 TileLang 算子都有昇腾实现” | 已核验（2026-10-01）：逐字读 |
| O4 | `.../deepseek-article-20260930.md:12` | 五个计算/通信组件的一句话定位 | 已核验（2026-10-01）：逐字读 |
| O5 | `.../deepseek-article-20260930.md:13` | “昇腾 950 的 128 卡超节点” | 已核验（2026-10-01）：逐字读 |
| O6 | `.../deepseek-article-20260930.md:16-27` | 六个开源项目的仓库地址 | 已核验（2026-10-01）；六个仓库均已 `git clone --depth 1` 快照 |
| P1 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md` | 13 个仓库的 upstream、commit、commit 日期、HEAD 摘要 | 已核验（2026-10-01）：逐行读 |
| S1 | `../tilelang-corpus/sources/sources.yaml`（`14-deepseek-ascend-stack` 一节） | 本节所有来源的可信度标注（`trust: primary`） | 已核验（2026-10-01） |

---

## corpus 资料引用：六个组件与配套仓库

抓取日期 2026-10-01（Asia/Shanghai），commit 见 P1。行号为快照内行号。

| id | 引用位置 | 用途 | 核验状态 |
| --- | --- | --- | --- |
| C1 | `.../raw/TileKernels/README.md:3,5` | TileKernels 的定位与“数十个算子” | 已核验（2026-10-01）：逐行读 |
| C2 | `.../raw/TileKernels/README.md:9` | 2026-09-30 加昇腾支持、运行期选后端、同一套 Python API | 已核验（2026-10-01） |
| C3 | `.../raw/TileKernels/README.md:13-19` | 功能清单（MoE 路由 / 量化 / Engram / mHC / RoPE / Rand / modeling） | 已核验（2026-10-01） |
| C4 | `.../raw/TileKernels/README.md:23-31` | Requirements：Python 3.12+、PyTorch 2.13+、TileLang 0.1.15+；Ascend 950 + CANN 9.2.0+ | 已核验（2026-10-01）：逐行读 |
| C5 | `.../raw/TileKernels/tile_kernels/config.py:24-31` | `is_ascend()` / `get_device()` 用 `/dev/davinci_manager` 判断平台 | 已核验（2026-10-01）：逐行读 |
| C6 | `.../raw/TileKernels/tile_kernels/config.py:70-71` | `get_num_vec_cores()` = `get_num_sms() * 2` | 已核验（2026-10-01） |
| C7 | `.../raw/TileKernels/tile_kernels/moe/topk_gate_kernel.py:7-43` | 路由算子入口：校验、选后端、启动 | 已核验（2026-10-01）：逐行读；**未运行** |
| C8 | `.../raw/TileKernels/tile_kernels/moe/topk_gate_cuda.py:1-68` | CUDA 后端的 TileLang kernel（`T.reduce_max`、`T.reducer_*`、`T.device_assert`） | 已核验（2026-10-01）；**未运行** |
| C9 | `.../raw/TileKernels/tile_kernels/mhc/__init__.py:1-17` | mHC 算子族与函数名（`mhc_pre_apply_mix_fwd` 等） | 已核验（2026-10-01） |
| C10 | `.../raw/DeepGEMM-Ascend/README.md:3,9` | “port of DeepGEMM”“fully API-compatible”“same package name” | 已核验（2026-10-01）：逐行读 |
| C11 | `.../raw/DeepGEMM-Ascend/README.md:5,7,11,13` | MAD 抽象、稀疏加载与协程流水、接近硬件极限 | 已核验（2026-10-01） |
| C12 | `.../raw/DeepGEMM-Ascend/README.md:17` | 2026-09-30 首次发布，支持 Ascend 950 | 已核验（2026-10-01） |
| C13 | `.../raw/DeepGEMM-Ascend/README.md:23-29` | Requirements：CANN 9.20、Ascend 950、`tilelang` 用于 HC prenorm | 已核验（2026-10-01）：逐行读 |
| C14 | `.../raw/DeepGEMM-Ascend/README.md:55-56` | **NOTE**：昇腾 scaling factor 打包格式（一对 UE8M0 → int16，MN-major） | 已核验（2026-10-01）：逐字读 |
| C15 | `.../raw/DeepGEMM-Ascend/README.md:67-68` | `transform_sf_into_required_layout` 等 SF 布局工具 | 已核验（2026-10-01） |
| C16 | `.../raw/DeepGEMM-Ascend/README.md:93,97,104` | 性能口径与 dense GEMM 99.8%（BF16 4096×7168×16384） | 已核验（2026-10-01）；**未运行** |
| C17 | `.../raw/DeepGEMM-Ascend/README.md:142` | MQA logits 是 FIX-pipe bound，FIX pipe 利用率 99% | 已核验（2026-10-01） |
| C18 | `.../raw/DeepGEMM-Ascend/README.md:153` | MegaMoE 融合 EP dispatch + 两个 grouped GEMM + SwiGLU + combine | 已核验（2026-10-01） |
| C19 | `.../raw/DeepGEMM-Ascend/README.md:166-176` | HC prenorm GEMM 的性能表与“接近 HBM 写带宽” | 已核验（2026-10-01） |
| C20 | `.../raw/DeepGEMM-Ascend/README.md:190` | “The mHC kernel is backed by Tilelang” | 已核验（2026-10-01） |
| C21 | `.../raw/DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py:1,7-8` | TileLang 实现的开头与 import（`tilelang.ascend.language`、`make_ascend_compact_nz_layout`） | 已核验（2026-10-01）：逐行读；**未运行** |
| C22 | `.../raw/DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py:96-131` | AutoSchedule 限制的注释、`T.MixedKernel`、`alloc_l1/l0a/l0b/l0c`、`annotate_buffer_versions` | 已核验（2026-10-01）；未运行 |
| C23 | `.../raw/DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py:195-271` | `T.SimdVF` / `T.simd.*` / `T.dual_copy` | 已核验（2026-10-01）；未运行 |
| C24 | `.../raw/DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py:281-356` | `T.gemm(transpose_B=..., clear_accum=..., unit_flag_ctrl=...)` 与 `T.PerCoreTask`/`T.Task` | 已核验（2026-10-01）；未运行 |
| C25 | `.../raw/DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py:362-397` | `tf32_hc_prenorm_gemm` 的公开签名与 K-split 语义 | 已核验（2026-10-01）；未运行 |
| C26 | `.../raw/DeepGEMM-Ascend/deep_gemm/__init__.py:44-79,81-101` | 导出的 kernel 名（`m_grouped_bf16_gemm_nt_contiguous`、`fp8_fp4_paged_mqa_logits`…）与 `tf32_hc_prenorm_gemm` | 已核验（2026-10-01）：逐行读 |
| C27 | `.../raw/DeepGEMM/README.md:3` | NV 侧 DeepGEMM 的定位 | 已核验（2026-10-01） |
| C28 | `.../raw/DeepGEMM/README.md:14-16` | News：2026-09-30 指向 DeepGEMM-Ascend | 已核验（2026-10-01） |
| C29 | `.../raw/DeepGEMM/README.md:24` | 1550 TFLOPS on H800（2025-04-18 的旧数字） | 已核验（2026-10-01） |
| C30 | `.../raw/DeepGEMM/README.md:64-67` | SM90 用 FP32 SF；SM100 把 4 个 UE8M0 打包进一个 `torch.int` | 已核验（2026-10-01）：逐字读 |
| C31 | `.../raw/DeepEP-Ascend/README.md:3,5` | 中文/英文介绍：EP all-to-all、FP8 dispatch、延迟 epilogue、PP/Bucket/Engram；Ascend C + HCCL/HCOMM/UBMEM/URMA + DeepJIT | 已核验（2026-10-01）：逐字读 |
| C32 | `.../raw/DeepEP-Ascend/README.md:9-11` | 性能口径（Ascend 950DT、CANN 9.2.0、netlayer 1、16384 tokens/rank、hidden 7168、top-6/256 experts、10 warmups/50 samples） | 已核验（2026-10-01） |
| C33 | `.../raw/DeepEP-Ascend/README.md:13-21` | EP8–EP128 的 dispatch/combine 带宽表与 90–95% 的说法 | 已核验（2026-10-01）；**未运行** |
| C34 | `.../raw/DeepEP-Ascend/README.md:25-36` | Requirements：Ascend 950 (A5) + UBMEM + UBC_CTP/URMA；验证组合；kernel 运行时编译 | 已核验（2026-10-01）：逐行读 |
| C35 | `.../raw/DeepEP-Ascend/README.md:40-42` | HDK 状态：PoC 配置、Q3 商用版预计 2026-10-15 前后公开 | 已核验（2026-10-01） |
| C36 | `.../raw/DeepEP-Ascend/README.md:62,64-72` | `EPBuffer` 等接口表；experimental 标注；EP 的额外约束 | 已核验（2026-10-01） |
| C37 | `.../raw/DeepEP-Ascend/README.md:76-84` | `Ongoing` 一节：Bucket / EPLB / PP+Engram 三种状态；不支持的项 | 已核验（2026-10-01）：逐字读 |
| C38 | `.../raw/DeepEP-Ascend/README.md:120,122-172,175-193` | dispatch/combine 的用法与重叠示例 | 已核验（2026-10-01）；未运行 |
| C39 | `.../raw/DeepEP-Ascend/README.md:238-250` | `EP_JIT_*` 环境变量（缓存目录、调试、汇编 dump） | 已核验（2026-10-01） |
| C40 | `.../raw/DeepEP-Ascend/README.md:288` | 基于华为 HCCL/HCOMM、UBMEM、URMA | 已核验（2026-10-01） |
| C41 | `.../raw/DeepEP-Ascend/deep_ep/__init__.py:20-36` | 公开 API：`EPBuffer`/`PPBuffer`/`EngramBuffer`/`BucketBuffer`、`BufferAllocator`、`bench_msprof` | 已核验（2026-10-01）：逐行读 |
| C42 | `.../raw/DeepEP/README.md:3,7-11` | NV 侧 DeepEP 的定位与“Ascend version release: same API and full performance” | 已核验（2026-10-01） |
| C43 | `.../raw/FlashMLA/README.md:3` | Breaking change：移除 Hopper 与旧模型支持、KV cache 格式变化 | 已核验（2026-10-01） |
| C44 | `.../raw/FlashMLA/README.md:7,9` | FlashMLA 的定位：NV GPU 与昇腾 NPU、稀疏 prefill/decode、融合 kernel、DSA | 已核验（2026-10-01）；**模型页未能核实**，见“待核事项” |
| C45 | `.../raw/FlashMLA/README.md:13-15` | 2026-09-30 昇腾发布：prefill 410 TFlops（95%）、decode 360 TFlops（83%） | 已核验（2026-10-01） |
| C46 | `.../raw/FlashMLA/README.md:39,43,53,63` | 融合 kernel 需要 TileLang/Tile-Kernels/DeepGEMM；仅 CUDA；稀疏 prefill/decode 的 Ascend 数字 | 已核验（2026-10-01） |
| C47 | `.../raw/FlashMLA/README.md:83-88,101` | Ascend 侧 Requirements（950 / CANN 9.2.0+ / torch_npu）与构建期平台探测 | 已核验（2026-10-01）：逐行读 |
| C48 | `.../raw/FlashMLA/README.md:111-126,136-151,172` | decode 用法、KV cache 格式、`indices` 语义、prefill 无 batch 维 | 已核验（2026-10-01）；未运行 |
| C49 | `.../raw/FlashMLA/docs/20260930-ascend-prefill-deep-dive-zh.md:1-30` | 昇腾稀疏 prefill 的算法回顾与 online softmax 流程 | 已核验（2026-10-01）：读前 30 行；**全文留给第 36 章** |
| C50 | `.../raw/DeepSelect/README.md:3,5` | 定位：DSA 与 sampler 的 TopK，CUDA + Ascend，2–20× vs `torch.topk` | 已核验（2026-10-01）：逐字读 |
| C51 | `.../raw/DeepSelect/README.md:9-11` | News：2026-09-30 昇腾 TopK；2026-09-10 算法简析与 v1.0.0 | 已核验（2026-10-01） |
| C52 | `.../raw/DeepSelect/README.md:19-23,37` | Lightning Indexer 场景、`topk=512`、昇腾只支持 bf16 | 已核验（2026-10-01） |
| C53 | `.../raw/DeepSelect/README.md:41-44,48,56` | 性能指标是有效内存带宽；CUDA / Ascend 两张图 | 已核验（2026-10-01）；**未运行** |
| C54 | `.../raw/DeepSelect/README.md:64-71,85-99` | 安装、`deep_select.topk` 用法、stride 要求 | 已核验（2026-10-01） |
| C55 | `.../raw/DeepSelect/deep_select/interface.py:10-23` | 平台探测与后端模块选择 | 已核验（2026-10-01）：逐行读 |
| C56 | `.../raw/DeepSelect/deep_select/interface.py:52-68` | 文档字符串里的平台差异（dtype / sorted / indices_type / NaN / begin / hint） | 已核验（2026-10-01）：逐字读 |
| C57 | `.../raw/DeepJIT/README.md:3,5` | DeepJIT 的定位：CUDA 与昇腾共用的 header-only C++20 JIT 运行时 | 已核验（2026-10-01）：逐字读 |
| C58 | `.../raw/DeepJIT/README.md:11-16` | 特性：双后端、缓存、共享缓存、惰性初始化、PyTorch 集成、汇编 dump | 已核验（2026-10-01） |
| C59 | `.../raw/DeepJIT/README.md:18-21` | WIP：缓存预热与 Python 编译 API 尚未可用 | 已核验（2026-10-01） |
| C60 | `.../raw/DeepJIT/README.md:25-28` | 两个后端的工具链（NVCC / Bisheng + ld.lld + ACL） | 已核验（2026-10-01） |
| C61 | `.../raw/DeepJIT/README.md:34-45,336-386` | 共享缓存与 `DJ_JIT_CACHE_DIR`；`Environment variables` 一节 | 已核验（2026-10-01） |
| C62 | `.../raw/DualPipe/README.md:3` | 双向流水并行与计算-通信全重叠 | 已核验（2026-10-01） |
| C63 | `.../raw/EPLB/README.md:3-8` | 冗余专家策略与分组受限路由 | 已核验（2026-10-01） |
| C64 | `.../raw/Engram/README.md:30-40` | Engram 论文定位：条件记忆、O(1) 查表、确定性寻址可卸载到 host memory | 已核验（2026-10-01）：逐行读 |
| C65 | `.../raw/DeepSeek-V3.2-Exp/README.md:42,50` | DSA 的首次公开介绍与“几乎不影响模型质量” | 已核验（2026-10-01）：逐行读 |
| C66 | `.../raw/DeepSeek-V3.2-Exp/README.md:79-83` | 开源 kernel 指引：TileLang 可读版在 `examples/deepseek_v32`，高性能版在 DeepGEMM / FlashMLA | 已核验（2026-10-01） |
| C67 | `.../raw/clangd-ascend/README.md:3-9,11-37` | 昇腾 C/C++ 与 AscendC 的 clangd（补全、诊断、跳转）与安装 | 已核验（2026-10-01）：逐行读 |

---

## 锁定源码引用（`_vendor/`）

本章只在“TileLang 昇腾版本指什么”这一处引用 `_vendor/`，其余行号留给第 25 章。

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| V1 | `ls _vendor/tilelang-v0.1.15/tilelang/ascend/` | 主干昇腾后端目录内容（`language/`、`codegen.py`、`execution_backend.py`、`kernel_cache.py`、`analysis/`、`pipeline.py`、`target.py` 等） | 已核验（2026-10-01）：列出目录 |
| V2 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:1-30` | “Ascend 950 Backend”、安装前提、默认 target `dav-3510` | 已核验（2026-10-01）：逐行读 |
| V3 | `_vendor/tilelang-v0.1.15/VERSION` | 值为 `0.1.15` | 已核验（2026-10-01） |
| V4 | `ls _vendor/tilelang-ascend-v0.1.2.000/` | 适配器仓库快照存在（前端 `tilelang.language`，Ascend C & PTO 两条 codegen 路线） | 已核验（2026-10-01）；细节见第 19、25 章 |

---

## 关于性能数字的说明

**本章不提供任何昇腾上的实测数字。** 正文出现的全部数字都来自下列 README 自述，且
必须连同测量条件一起引用：

| 数字 | 来源 | 条件 |
| --- | --- | --- |
| dense GEMM up to 99.8%（BF16 4096×7168×16384） | C16 | Ascend 950DT、CANN 9.20、`bench_msprof`、cold L2 |
| MQA logits：FIX pipe 99% 利用率 | C17 | 同上；“FIX-pipe bound rather than compute bound” |
| HC prenorm“接近 HBM 写带宽” | C19 | 同上；表格给出 M/N/K 与带宽 |
| EP8 dispatch 373–375 GB/s、combine 345–347 GB/s（EP128 降到 313–320 / 272–278） | C32, C33 | Ascend 950DT、CANN 9.2.0、PoC HDK、netlayer 1（超节点对外 Clos 网络）、16384 tokens/rank、hidden 7168、top-6/256 experts、10 warmups/50 samples、含缓存刷新 |
| EP ≤ 32 的 dispatch 达到物理有效载荷带宽的 90–95% | C33 | 同上 |
| 昇腾稀疏注意力 prefill up to 410 TFlops（95%）、decode up to 360 TFlops（83%） | C45 | Ascend 950；相对“theoretical hardware peak” |
| DeepSelect 2–20× vs `torch.topk` | C50 | 指标是有效内存带宽；分档取决于 batch / vocab；benchmark 在 `tests/test.py` |
| DeepGEMM（NV 侧）up to 1550 TFLOPS | C29 | H800；2025-04-18 的旧数字，**不能与昇腾侧数字直接比较** |

---

## 待核事项

1. **DeepSelect 的昇腾硬件要求**：这份快照的 README 没有像其它仓库那样写明
   “Ascend 950 / CANN 版本”，只给了安装方式与支持场景（C51–C54）。第 37 章写作时
   需要再核对一次（例如看 `setup.py` / `csrc/ascend_kernels/` 的构建说明）。
2. **FlashMLA README 引用的模型页**：README 提到它支撑 NVIDIA GPU 与昇腾 NPU 上某个
   DeepSeek V4.1 系列模型（Flash 版）的推理，并给出一个 HuggingFace 模型页链接
   （C44）。写作环境无法访问 HuggingFace（连接超时），**该页面未能核实**。本章只把
   它当作 README 自述转述，不给可点击链接，也不据此推断模型细节。
3. **“128 卡超节点”的硬件细节**：官方文章只有一句话（O5），corpus 里没有对应的
   硬件白皮书。本章只引用这一句，并把可核到的部署形态交给 DeepEP-Ascend 的 README
   （C32, C35）。如果后续补充到厂商资料，第 39 章应当更新。
4. **DeepEP-Ascend 的 HDK 计划日期**：README 写“Public availability is currently
   planned for mid-October 2026, around October 15”（C35）。这是一个**厂商发布计划**，
   不是既成事实；引用时必须保留“planned”的措辞。
