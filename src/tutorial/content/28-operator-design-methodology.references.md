# 第 28 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`
（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不使用任何“版本模糊”的表述（每一次能力判断都指向具体 tag / commit）；
若涉及 `main` 上更新的东西，会显式标注“仅 main / 未进 v0.1.15”。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章的所有数字都只有三类来源：

* **模型输出**，`code/` 下七个纯标准库脚本，可在任意机器复算；
* **厂商公开资料**，NVIDIA / 华为昇腾官方文档，标注为“厂商公开规格，非实测”；
* **锁定版本源码里的常量**，`_vendor/` 下逐行核对过的常量表与公式。

**没有任何一条性能数字来自本机实测。** “已核验”的含义是：在锁定版本的工作树上
用 `rg`/`grep -n` 定位到定义行，并阅读了完整签名、注释或实现；
**这不是运行验证，也不代表在硬件上验证过。**

---

## 一、锁定源码引用（`_vendor/tilelang-v0.1.15/`）

### 1.1 carver / roller：tile cost model

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/carver/README.md` | carver 的自我定位、支持的模板清单（Matmul / GeneralReduction / Elementwise / GEMV / FlashAttention / Conv）、`recommend_hints` 用法、CUDA 设备模型的示例片段 | 已核验（2026-10-01）：逐行读；**未运行** |
| S2 | `.../tilelang/carver/arch/arch_base.py:6-27` | `TileDevice` 的能力字段：`reg_cap` / `smem_cap` / `compute_max_core` / `warp_size` / `sm_partition` / `transaction_size` / `max_smem_usage` / `bandwidth` | 已核验（2026-10-01）：逐行读 |
| S3 | `.../tilelang/carver/arch/cuda.py:139-167` | CUDA 设备模型：`smem_cap` 与 `compute_max_core` 运行时查询（:153-155）、`reg_cap=65536`（:157）、`max_smem_usage=2*smem_cap`（:158）、`sm_partition=4`（:159）、`transaction_size=[32,128]`（:162）、`bandwidth=[750,12080]` 及其“只用于推荐 basic tile size、只有跨设备比例有意义”的注释（:163-167） | 已核验（2026-10-01）：逐行读 |
| S4 | `.../tilelang/carver/arch/cuda.py:144-146` | `CUDA.__init__` 调用 `tvm.runtime.cuda(0)` 并在无设备时抛 `RuntimeError("Cannot find cuda device 0.")` ，这是本章“无卡机器上不能用 carver”结论的直接依据 | 已核验（2026-10-01）：逐行读 |
| S5 | `.../tilelang/carver/arch/__init__.py:27-56` | `get_arch` 与 `auto_infer_current_arch`：无 CUDA 且无 MPS 时退到 `llvm`（CPU）模型 | 已核验（2026-10-01）：逐行读 |
| S6 | `.../tilelang/carver/arch/cpu.py:12-19` | `CPU` 设备模型只填 `platform`，**没有** `smem_cap` / `reg_cap` 等字段 | 已核验（2026-10-01）：逐行读 |
| S7 | `.../tilelang/carver/roller/policy/default.py:398-432` | `_compute_memory_traffic`：按传播后的区域与读/写事务粒度计算 traffic | 已核验（2026-10-01）：逐行读 |
| S8 | `.../tilelang/carver/roller/policy/common.py:32-52` | `coalesced_factor` 与 `coalesced_tensor_shape` 的完整实现 | 已核验（2026-10-01）：逐行读，并在 `code/03` 中等价重写 |
| S9 | `.../tilelang/carver/roller/policy/default.py:452-495` | `_compute_shared_memory_usage`：best-fit 分配器把所有缓存张量的 footprint 拼成 `smem_cost` | 已核验（2026-10-01）：逐行读 |
| S10 | `.../tilelang/carver/roller/bestfit.py:22-61` | `BestFit` 分配器，32 字节对齐；`code/03` 按它等价重写 | 已核验（2026-10-01）：逐行读 |
| S11 | `.../tilelang/carver/roller/node.py:447-495` | `PrimFuncNode.footprint`：每张要缓存的张量的字节数，32 字节向上取整 | 已核验（2026-10-01）：逐行读 |
| S12 | `.../tilelang/carver/roller/node.py:332-399` | `propagate` / `propagate_inputs` / `propagate_outputs`：区域推导的入口（本章在 `code/03` 中把它对普通 GEMM 解析手写，并在 README 里写明这是假设） | 已核验（2026-10-01）：逐行读 |
| S13 | `.../tilelang/carver/roller/policy/default.py:537-574` | `compute_tile_dict`：`reg_usage = 2 * max(prod(tile) * bits / 32)`（:564）、smem/reg 超限即判 `valid=False`（:558-567）、`block_per_SM` 的三人 `min`（:568-572）、`num_wave`（:573） | 已核验（2026-10-01）：逐行读，公式在 `code/03` 中复算 |
| S14 | `.../tilelang/carver/roller/policy/default.py:96-134` | `dfs_smem_tile` 的候选枚举与最优先搜索；排序键 `priority = (traffic + 1) * num_wave`（:111-112）；已访问 tile 上限 2000（:123） | 已核验（2026-10-01）：逐行读 |
| S15 | `.../tilelang/carver/roller/policy/tensorcore.py:120-140, 320-335` | tensor-core policy 的 `_check_small_tile`（阈值 32）与“smem 超静态容量就退到 `shared.dyn`” | 已核验（2026-10-01）：逐行读 |
| S16 | `.../tilelang/carver/roller/hint.py:154-192` | `Hint` 的字段：`block` / `thread` / `warp` / `rstep` / `reduce_thread` / `vectorize` / `pipeline_stage` / `split_k_factor` | 已核验（2026-10-01）：逐行读 |
| S17 | `.../tilelang/carver/template/matmul.py:40-52` | `MatmulTemplate.get_hardware_aware_configs` -> `get_roller_hints_from_func(..., allow_gemv=True)` | 已核验（2026-10-01）：逐行读 |
| S18 | `.../tilelang/carver/utils.py:29-65` | `get_roller_hints_from_func`：TensorCorePolicy 优先、失败或空则回退 DefaultPolicy | 已核验（2026-10-01）：逐行读 |
| S19 | `.../tilelang/jit/adapter/wrapper.py:403`（`nvrtc/wrapper.py:464` 同） | carver 的 `arch.driver.get_persisting_l2_cache_max_size` 被 JIT wrapper 用于设置 L2 persistence ，carver 不只在离线使用 | 已核验（2026-10-01）：`grep -rn "carver" --include="*.py" tilelang/ \| grep -v tilelang/carver/` 命中三处 |
| S20 | `.../tilelang/carver/arch/driver/cuda_driver.py:22,39,106` | `get_cuda_device_properties` / `get_shared_memory_per_block` / `get_persisting_l2_cache_max_size` ，设备参数来自这里，因此必须有可用的 CUDA 运行时 | 已核验（2026-10-01）：`ls carver/arch/driver/` 与逐函数阅读 |

### 1.2 `tilelang.tools.Analyzer`

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S21 | `.../tilelang/tools/Analyzer.py:10` | `ARCH_CONFIGS = {"80": ..., "86": ..., "89": ...}`：算力表只有三个 compute capability | 已核验（2026-10-01）：逐行读 |
| S22 | `.../tilelang/tools/Analyzer.py:58-88` | `_analyze_copy`：只统计跨 global 边界的 `tl.copy`；按循环与 `blockIdx.x/y` 的数量放大 | 已核验（2026-10-01）：逐行读 |
| S23 | `.../tilelang/tools/Analyzer.py:90-106, 148-151` | `_analyze_gemm`：**只**统计 `tl.gemm` / `tl.tileop.gemm`；FLOPs = `2*M*N*K` | 已核验（2026-10-01）：逐行读 |
| S24 | `.../tilelang/tools/Analyzer.py:172-213` | `calculate`：`get_peak_tflops`（:187-195，查不到返回 `None`）、`bandwidth_GBps = device.bandwidth[1] / 1000`（:198）、`estimated_time` 在无算力表时退化为访存时间（:201-204） | 已核验（2026-10-01）：逐行读；**未运行** |
| S25 | `.../docs/tools/analyzer.md:93-112` | 官方自述的限制：只识别 `T.gemm` 与跨 global 的 `T.copy`、`blockIdx.z` 不计、算力表只有 8.0/8.6/8.9、`expected_tflops` 与 `estimated_time` 只作比较用 | 已核验（2026-10-01）：逐行读 |
| S26 | `.../docs/tools/index.md:11,23` | 工具定位：“Estimate compute and memory cost from TIR”、“Use Analyzer before benchmarking when you need a rough roofline-style estimate” | 已核验（2026-10-01）：逐行读 |
| S27 | `.../examples/analyze/example_gemm_analyze.py`、`examples/analyze/README.md:21-23` | 官方示例：构造 `CUDA("cuda")` 需要可见 GPU，即使分析器本身不执行 kernel | 已核验（2026-10-01）：逐行读；**未运行** |

### 1.3 昇腾侧的延迟模型与调度

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S28 | `.../src/ascend/transform/estimate_latency.h:27-158` | `AscendLatencyParams` 常量表：Cube 基础延迟 8 与各 dtype 吞吐（:32-55）、VALU 各操作吞吐 51/13/8（:57-68）、ND2NZ 参数（:73-85）、MTE1/Fixpipe 参数（:88-101）、各物理路径带宽（:104-113）、AIV MTE 参数（:115-125）、N-strided 与 dual-destination 参数（:127-157） | 已核验（2026-10-01）：逐行读；分量在 `code/07` 中复算 |
| S29 | `.../src/ascend/transform/estimate_latency.cc:326-340` | 模型结构的官方注释：四步（memory latency / compute latency / II by pipe / total = memory + compute） | 已核验（2026-10-01）：逐行读 |
| S30 | `.../src/ascend/transform/estimate_latency.cc:692-707` | `CubeOperations`：`2 * align(M,16) * align(N,16) * align(K, k_group)`，`k_group = 256/bits`（fp32 在 HF32 下为 8、否则 1）；MX 路径的最小 M/N 为 48 | 已核验（2026-10-01）：逐行读 |
| S31 | `.../src/ascend/transform/estimate_latency.cc:910-965` | Cube 的 latency 与 II；`total_latency = memory_latency + compute_latency`（:941）；按 pipe mask 合成 II（:945-965） | 已核验（2026-10-01）：逐行读 |
| S32 | `.../src/ascend/transform/estimate_latency.cc:750-870` | 逐物理路径的 memory latency 与 II；`PathIICost` 与 `add_memory_access` | 已核验（2026-10-01）：阅读结构，未逐项复算 |
| S33 | `.../tilelang/ascend/transform/z3_scheduler.py:67-140` | `z3_schedule_python` 的输入（latencies / iis / resource_flags / data_deps / resource_deps / owner_exclusion_deps / pipe_order_deps）与输出（start_times） | 已核验（2026-10-01）：逐行读 |
| S34 | `.../tilelang/ascend/pipeline.py:103-116` | 昇腾自动调度 pass 链：`EstimateLatency → AutoSchedule → AssignCore → PrepareMultiBuffer → ResolveCore → InsertSync → MaterializeMultiBuffer → LowerScheduledTIR` | 已核验（2026-10-01）：逐行读 |
| S35 | `.../tilelang/ascend/README.md:62-152` | 官方 950 GEMM+ReLU 例子：`block_M=256, block_N=224, block_K=128`（:74）、`num_blocks=32`（:81）、`num_stages=2`（:90）、`T.dual_copy` 与 `T.SimtVF(threads=128)` 的 epilogue（:92-98）、与 Ascend C 的分工表（:132-152）、`tilelang/ascend/language` 的方言入口 | 已核验（2026-10-01）：逐行读；**未在 NPU 上运行** |
| S36 | `.../tilelang/language/gemm_op.py:160-195` | `T.gemm` 签名与 docstring（同步语义；Hopper 隐式 wgmma wait、Blackwell 隐式 `mbarrier_wait_parity`；各后端方言扩展 `mbar` / `k_pack` / `unit_flag_ctrl`） | 已核验（2026-10-01）：逐行读 |

---

## 二、corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `00-methodology.md`。
corpus 自己定义了三级可信度（`primary` / `vendor` / `community`），
本章的数字只取前三类来源中的 `primary` 与 `vendor`。

### 2.1 性能模型与方法规范

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/docs/design/roofline.md` | https://github.com/tile-ai/TileOPs | 本章 2.2–2.3 节的全部判据来源：记数与 FLOP 约定（:43-61）、compute roof 必须由作者声明（:63-70）、不可诚实定价的行的三种处理与 latency-bound 标签（:37-41）、SFU 不保证到顶（:34）、判读分档 80% / 105% 与 physics check（:156-163）、profile 的 `effective = theoretical × calibration` 与包络语义（§5.1，:257-261） | 已核验（2026-10-01）：本地为全文，逐节读 |
| R2 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/profiles/h200.yaml` | https://github.com/tile-ai/TileOPs | 本章所有 H200 数值：HBM `theoretical 4800e9` / `calibration 0.9380` 与四种访问混合（:15-18）、`cuda_core.fp32` 67.0e12 × 0.8548（:20-26）、`tensor_core.fp16` 989.5e12 × 0.7370（含 sustained 0.6684 / burst 0.7370 与 700 W 上限的说明，:31-34）、bf16（:35-38）、tf32（:39-42）、fp8（:48-50） | 已核验（2026-10-01）：本地为全文，逐行读 |
| R3 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/profiles/h20_3e.yaml` | https://github.com/tile-ai/TileOPs | 反例：H20-3e 的 profile **没有** `cuda_core` 段，因此“用哪个 roof”这件事在真实 profile 里确实会缺项 | 已核验（2026-10-01）：本地为全文，逐行读 |
| R4 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2607-22432.md` | https://arxiv.org/abs/2607.22432 | TileSight 的摘要全文（12 行）：三层结构（intra-tile / inter-tile / cross-device）、12.35% pooled MAPE、L2 命中率约 1 个百分点、32 卡 16.18% wMAPE、vLLM 端到端 13.52% wMAPE、“coarse roofline bounds”的问题表述 | 已核验（2026-10-01）：本地为摘要；**全文缺失，见“待补资料”** |
| R5 | `../tilelang-corpus/12-operator-libraries/raw/github-tile-ai-tilesight.md` | https://github.com/tile-ai/TileSight | TileSight 仓库 README 的转录（73 行）：MIT 许可、CPU 上建模 + CUDA/HIP 探针测量、`tilesight/arch` 与 `tilesight/modeling` 的目录分工、`sight.analyze(...)` 与 `result.launches["main"].kernel_body_s` 的示例 | 已核验（2026-10-01）：本地为 README 快照；**源码未纳入 corpus** |
| R6 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/.claude/domain-rules/ops-design.md`、`docs/design/ops-design.md` | https://github.com/tile-ai/TileOPs | 工程化落地：算子库如何把 roofline、kernel 选择与调参参数写成规范（本章只作为“方法论的工业化形态”提及，不展开） | 已核验（2026-10-01）：本地为全文，读小标题与规则条目 |

### 2.2 NVIDIA 硬件与指令形状

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R7 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-hopper-tuning-guide.md` | https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html | cc9.0 每 SM shared 228 KB（:50）、每 block 上限 227 KB（:51）、carveout 档位列表（:226）、CUDA 保留 1 KB 与静态 48 KB 限制（:228） | 已核验（2026-10-01）：本地为全文；厂商文档 |
| R8 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md:3126` | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html | cc7.0：每 SM 65536 个 32 位寄存器、最多 2048 个驻留线程、100% occupancy 时每线程 32 个寄存器、寄存器按每 warp 256 个向上取整 | 已核验（2026-10-01）：本地为全文；厂商文档 |
| R9 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md` | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html | `mma` 密集 f16 形状 `m16n8k8` / `m16n8k16`（:23476-23477）；`wgmma` 的 warpgroup = 4 个连续 warp（:28113-28114）；`wgmma` 密集 f16 形状 `m64nNk16`、N 从 8 到 256（:28132） | 已核验（2026-10-01）：本地为全文；厂商文档 |
| R10 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md` | https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | H100 SXM5 的 datasheet 表：FP32 60 TFLOPS、FP16 张量核 1000 TFLOPS、FP8 2000 TFLOPS（:135-143，Table 1 自述 preliminary）；HBM3“over 3 TB/sec”（:334） | 已核验（2026-10-01）：本地为全文；**厂商公开 preliminary，非实测** |

### 2.3 昇腾硬件、对齐与容量

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R11 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_0011_html.md:39-51` | https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_10_0011.html | **对齐表**：AIV Unified Buffer 32 Byte、AIC L1 Buffer 32 Byte、L0A/L0B 512 Byte、L0C 64 Byte、BiasTable 64 Byte、Fixpipe 64 Byte；以及 L0A/L0B/L0C 的分形格式建议（FRACTAL_ZZ / FRACTAL_ZN / FRACTAL_NZ） | 已核验（2026-10-01）：本地为厂商文档全文，逐行读；**非实测** |
| R12 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_0008_html.md` | https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_10_0008.html | AI Core 的计算/存储/搬运单元、Cube 16×16、Vector=SIMD、`GetCoreMemSize` 运行时查询各存储单元容量、MTE1/2/3 与 Fixpipe 的通路、SetFlag/WaitFlag 语义 | 已核验（2026-10-01）：厂商文档全文；**非实测** |
| R13 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_best_practices_10_00020_html.md:18` | https://www.hiascend.com/document/detail/zh/canncommercial/900/programug/Ascendcopdevg/atlas_ascendc_best_practices_10_00020.html | “Unified Buffer 总大小为 192K，划分为 48 个 bank，每个 bank 由 128 行组成，每行长度为 32B”。**这是该文档举例的某一型号，不是“昇腾 UB 都是 192K”** | 已核验（2026-10-01）：厂商文档全文；**条件性数字** |
| R14 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_API_ascendcopapi_atlasascendc_api_07_0666_html.md` | https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0666.html | “该示例 L1 Buffer 可用大小为 512KB”，**同样是示例条件，不是通用规格** | 已核验（2026-10-01）：厂商文档全文；**条件性数字** |

---

## 三、本章代码示例与它们对应的引用

| 示例 | 用到的引用 |
| --- | --- |
| `code/01-op-ledger/op_ledger.py` | 记账约定（R1:43-61）、机器参数（R2、R10） |
| `code/02-roofline-misuse/roofline_misuse.py` | R1:34,40-41,63-70,156-163；R2；S28（VALU 吞吐）；R4 |
| `code/03-carver-cost-model/carver_cost_model.py` | S2–S14 |
| `code/04-tile-size-decision/tile_decision.py` | R7（shared 容量）、R8（寄存器）、R9（指令形状与 warpgroup） |
| `code/05-fusion-split/fusion_split.py` | 本书推导；判据由正文 5.1 节给出 |
| `code/06-case-norm-rope-quant/case_norm_rope_quant.py` | R2（带宽）；算子结构参考本书第 13、27 章 |
| `code/07-ascend-latency-model/ascend_pipe_model.py` | S28、S30、S31、S35 |

---

## 四、待补资料

以下是本章**明确缺料**的地方。写者不在 corpus 内发明事实，只登记需求：

1. **TileSight 的论文全文与源码。**（优先级：高）
   * 缺什么：`arxiv-2607-22432` 的正文与 PDF；`github.com/tile-ai/TileSight` 的源码树，
     尤其是 `tilesight/modeling/` 下的 cost / cache 模型与 `tilesight/arch/` 下的
     profile 定义。
   * 为什么需要：本章 3.1 节只能引用摘要里的三层结构自报误差，
     **无法给出任何一条公式**。任务是“TileSight 这类 tile 级解析性能模型在做什么”，
     目前只做到“它宣称做什么”。
   * 建议来源：arXiv PDF（`https://arxiv.org/pdf/2607.22432`）、
     仓库 `git clone --depth 1 https://github.com/tile-ai/TileSight`。
   * 当前状态：本地只有 `arxiv-2607-22432.md`（12 行摘要）与
     `github-tile-ai-tilesight.md`（73 行 README 转录）。
2. **carver `propagate` 的确切语义。**（优先级：中）
   * 缺什么：`PrimFuncNode.propagate` 对一个 reduction block 返回的区域，
     是“一个 `rstep` 切片”还是“整条 reduction 维”。
   * 为什么需要：这决定了 `_compute_memory_traffic` 的 traffic 是教科书公式
     （每输出 tile 读 A 的整条 K）还是它的 1/(K/block_K) 倍。
   * 怎么补：在一台有 NVIDIA GPU 的机器上实际跑一次
     `MatmulTemplate(...).with_arch(CUDA("nvidia/...")).recommend_hints(topk=5)`，
     打印 `hint` 并对手算的 traffic 做交叉验证。
   * 当前状态：`code/03` 里把它写成**显式假设**（并在 README 的
     “known limitations”里登记），因为另一种读法无法复现教科书公式。
3. **NVIDIA 侧的 SFU / MUFU 吞吐数字。**（优先级：中）
   * 缺什么：CUDA core 与 SFU 的 ops/cycle 比值（或等价的公开吞吐表）。
   * 为什么需要：本章 2.3 节 M5 只能给出**昇腾侧**的 6.38 倍，
     NV 侧只能引用“不保证 SFU-bound 到顶”这条定性说明。
   * 建议来源：CUDA C Programming Guide 的算术指令吞吐表，
     或 Nsight Compute 的 SM 吞吐 handbook。
4. **Ascend 950 的时钟频率。**（优先级：低）
   * 缺什么：一条可引用的 A5 主频。
   * 为什么需要：有了它，`estimate_latency.h` 的 cycle 数才能换算成时间，
     与 NV 侧 TFLOPS 口径的 roofline 对齐；缺它就只能做 cycle 级比较。
   * 当前状态：本章所有昇腾数字都保持 cycle 单位，不换算成秒。
5. **昇腾 L0A/L0B/L0C 的容量。**（优先级：低）
   * 缺什么：可跨型号引用的容量数字（官方要求用 `GetCoreMemSize` 运行时查询）。
   * 当前状态：本章沿用第 6 章 4.2 节的处理，**不给数值**，只给对齐约束与查询接口。

---

## 五、与其它章节的一致性说明

* 本章第 4.2 节与图 4 引用的昇腾对齐表来自 **R11**
  （`..._atlas_ascendc_10_0011_html.md:39-51`）。第 6 章 4.2 节在讲同一组约束时
  引用的是 `..._atlas_ascendc_10_0008_html.md`，并且把 Fixpipe Buffer 的对齐写成
  “128 Byte”。**R11 的表里写的是 Fixpipe Buffer 64 Byte。** 本章按 R11 的原文
  取值（64 Byte），并在这里登记这处差异，供终审核对；两处不影响本章的任何结论。
* 本章第 2.2 节使用的 H200 机器参数与第 2 章 2.2 节同源（同一份 `h200.yaml`），
  因此两章关于 LayerNorm / 元素链的访存下界可以互相复算：
  `code/01-op-ledger --sweep` 给出的 rmsnorm 8192×4096 = 29.810 µs、
  元素链融合 = 14.905 µs，与第 2 章手算的 29.8 µs / 14.9 µs 一致。
* 本章 3.2 节与第 11 章 3.9 节都讲 carver。第 11 章给的是“它有 cost model、
  cost model 只排序不定案”的定位；本章把它拆到公式与行号一级，并补上了
  “无卡机器上跑不起来”这一条操作性结论。两章的结论不冲突。
