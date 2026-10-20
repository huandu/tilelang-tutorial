# 第 18 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现无法复核的模糊版本说法（“跟随上游最新”一类）；
讨论 `main` 上更新的东西时会显式标注。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有
昇腾 NPU**。因此本章所有 kernel 示例的最高验证级别是“仅静态核对”
（API 存在性与签名逐个在 `_vendor/` 里比对过，语法用 `py_compile` 过了），
**没有任何一条性能数字来自本机实测**。

---

## 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `examples/flash_decoding/example_mha_inference.py:8,27,28,30,111,116,137,149,151` | MHA decode 的 split + combine：`num_split = 4`、两块 `alloc_global`、两个 `T.Kernel` 的 grid、lse 的写出与加权合并 | 已核验（2026-10-01）：`grep -n` 定位后逐行读；**未运行** |
| S2 | `examples/flash_decoding/example_gqa_decode.py:16-23,53,58,61,93,123` | GQA decode：autotune 空间（`num_split ∈ {1,2,4,8}`）、KV 形状 `[batch, seqlen_kv, groups, dim]`、`kv_group_num`、`valid_block_H = min(block_H, kv_group_num)`、`cur_kv_head`、mask 写法 | 已核验（2026-10-01）；**未运行** |
| S3 | `examples/flash_decoding/example_gqa_decode_varlen_logits.py:35,58,82-84,128` | varlen decode：`out_idx=[-2, -1]`、`cu_seqlens_k`、由前缀和算 `cur_start_k/cur_end_k/cur_seqlen_k`、写回分数矩阵 `S` | 已核验（2026-10-01）；**未运行** |
| S4 | `examples/fusedmoe/example_fusedmoe_tilelang.py:12-89,92-226,157-161,202-206,225,344-382` | fused MoE：共享专家与路由专家两个 kernel、dispatch 契约的四张表、`m_start`/`actual_rows` 公式、combine 权重乘法 | 已核验（2026-10-01）；**未运行** |
| S5 | `examples/dequantize_gemm/example_dequant_gemm_w4a8.py:10,86,135-143,145` | W4A8 反量化 GEMM：`_tir_u8_to_i4_to_i8` 的位运算、kernel 内解包、`T.gemm(B_dequantize_prev_local, A_shared, ...)` | 已核验（2026-10-01）；**未运行** |
| S6 | `examples/dequantize_gemm/README.md:2-36` | Dequantization GEMM 的最小写法与“magic layout 见 BitBLAS”的指路 | 已核验（2026-10-01） |
| S7 | `examples/gemv/example_gemv.py:330-337`（`main()` 的 `--n/--k` 默认 1024）与 `examples/gemv/` 全目录 | GEMV 阶梯的官方实现；第 4.2 节那张耗时表的 shape 归属 | 已核验（2026-10-01）；**未运行** |
| S8 | `example_gqa_fwd_varlen.py:68,123`（`examples/flash_attention/`） | varlen 的“右对齐”与越界 guard：`offset = kv_current_seqlen - q_current_seqlen`、`if bx*block_M+i < q_current_seqlen` | 已核验（2026-10-01）；**未运行** |
| S9 | `tilelang/language/gemm_op.py:160,275` | `T.gemm` 与 `T.gemm_blockscaled(..., SFA, SFB, k_start, sf_a_granularity_k, sf_b_granularity_k)` 的签名 | 已核验（2026-10-01） |
| S10 | `tilelang/language/allocate.py:49,67,81,171` | `alloc_shared` / `alloc_local` / `alloc_fragment` / `alloc_global`（split-KV 的 `Output_partial` 与 `Lse` 靠它） | 已核验（2026-10-01） |
| S11 | `tilelang/profiler/bench.py:31-34,39,42,260-263` | `do_bench(fn, warmup=25, rep=100, ..., backend=..., return_mode="mean")` 的默认值、`backend` 的取值、`msprof` 分支、`cache_size=256`（L2 flush） | 已核验（2026-10-01）：`grep -n` 逐条定位 |
| S12 | `tilelang/profiler/msprof.py:1-38` | 昇腾的 msprof 计时路径与专用 cache-flush kernel（`T.SimtVF` + `l2_cache_ctrl="WTS_FV"`） | 已核验（2026-10-01） |
| S13 | `tilelang/ascend/torch_npu_exchange.pyx:1-6,44-56` | TileLang 昇腾后端 patch Torch 的 DLPack 交换表，把 kernel 提交到 torch 当前的 NPU stream | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S14 | `tilelang/ascend/pipeline.py:22,53`（`allow_autoschedule`）与 `examples/ascend/example_rmsnorm.py:40-45` | 主干 v0.1.15 的昇腾方言写法：`with T.Kernel(N_CORES) as core_id`、`T.alloc_shared`、`T.annotate_buffer_versions`；与适配器 API 的差别 | 已核验（2026-10-01）；**未在 NPU 上运行** |
| S15 | `tilelang/language/common.py:78`（`GemmWarpPolicy` 的导出）与 `tilelang/language/tir/op.py:361`（`infinity`） | 本章骨架里用到的两个符号的出处 | 已核验（2026-10-01） |
| S16 | `examples/ascend/flash_attention/README.md`（性能表与 Limitations） | 昇腾 FA 的厂商自报数字与限制（`head_dim` 固定 128、bf16 输入、无 causal mask、GQA `num_blocks` 必须整除 M 块数） | 已核验（2026-10-01）；**厂商自报，作者未复现** |

---

## 锁定的源码引用（`_vendor/tilelang-ascend-v0.1.2.000/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| A1 | `tilelang/language/kernel.py:220,247,250,262` | `T.Kernel(..., is_npu=True)` 的参数与它写入的 `tilelang.is_npu_kernel_frame` 注解 | 已核验（2026-10-01）：`grep -n "is_npu"` 定位 |
| A2 | `tilelang/transform/pass_config.py:38,41,44,47,50` | `TL_ASCEND_AUTO_SYNC` / `TL_ASCEND_MEMORY_PLANNING` / `TL_ASCEND_AUTO_CV_SYNC` / `TL_ASCEND_AUTO_CV_COMBINE` 四个键的定义 | 已核验（2026-10-01） |
| A3 | `tilelang/language/ascend_tile.py:1026,1048,1059,1241` | `T.tile.add/mul/div/sqrt(dst, src0, src1)` 的签名 | 已核验（2026-10-01） |
| A4 | `tilelang/language/__init__.py:86` | `from . import ascend_tile as tile`，`T.tile.*` 的来源 | 已核验（2026-10-01） |
| A5 | `tilelang/language/reduce.py:79` | `T.reduce_sum(buffer, out, dim=-1, clear=True)` 的签名 | 已核验（2026-10-01） |
| A6 | `tilelang/language/allocate.py:140,144,148,152,156,31,49` | `alloc_L1` / `alloc_L0A` / `alloc_L0B` / `alloc_L0C` / `alloc_ub` / `alloc_shared` / `alloc_fragment` | 已核验（2026-10-01） |

### 一条必须显式记下来的版本差异

`TL_ASCEND_*` 这四个 pass 配置键**只在适配器仓库里存在**。在主干
`_vendor/tilelang-v0.1.15/` 里执行 `rg -n "TL_ASCEND" tilelang/transform/pass_config.py`
**没有任何命中**；主干用的是 pass context 开关
（`tilelang/ascend/pipeline.py` 的 `allow_autoschedule`）与
`tilelang.ascend.language` 的 `T.Kernel(N_CORES)`。
因此本章第 6 节与 `code/08-ascend-aclgraph/` 明确写了“本示例属于适配器 API”，
并且给出了主干写法的对照文件（`examples/ascend/example_rmsnorm.py`）。

“已核验”的含义：在锁定版本的工作树上用 `grep -n` / `sed` 定位到定义行并读完
签名与 docstring；行号区间与文件内容一致。**这不是运行验证。**

---

## corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30 / 2026-10-01（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `13-e2e-training-and-serving/raw/hf-qwen2-5-7b-card.md`、`hf-qwen2-5-7b-config.md` | https://huggingface.co/Qwen/Qwen2.5-7B（镜像 `https://hf-mirror.com/Qwen/Qwen2.5-7B`） | Qwen2.5-7B 的结构与参数量：28 层、GQA 28 Q / 4 KV、上下文 131072、7.61B（非嵌入 6.53B）；`config.json` 摘要给出 hidden 3584 / intermediate 18944 / vocab 152064 / rope_theta 1e6 | 已核验（2026-10-01）：模型卡与 config 两份快照都在本地；形状字段直接引用 config，不再反推 |
| R2 | `05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md` | https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | H100 的 HBM3“over 3 TB/sec”与 fp16 张量核公开规格；用于算平衡点与权重底线 | 已核验（2026-10-01）：**厂商公开资料，非实测** |
| R3 | `05-nvidia-backend/raw/nvidia-cuda-best-practices.md` | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html | 访存延迟与算术指令周期的量级（“hundreds of clock cycles”/“typically 4 cycles”）；本章只在第 2 节引用其定性结论 | 已核验（2026-10-01）；条件为 compute capability 7.0 设备 |
| R4 | `13-e2e-training-and-serving/raw/vllm-docs.md` | https://docs.vllm.ai/en/latest/ | vLLM 的能力清单：PagedAttention、continuous batching / chunked prefill / prefix caching、piecewise 与 full CUDA/HIP graphs、量化格式、可替换的 attention 与 GEMM/MoE kernel、disaggregated prefill/decode、hardware plugins | 已核验（2026-10-01）：本地只有**文档首页**（71 行）；接口级结论无法引用，见“待补资料” |
| R5 | `12-operator-libraries/raw/github-tile-ai-tilert.md` | https://github.com/tile-ai/TileRT | TileRT 的定位（超低延迟）、tile 级运行时、编译器技术将逐步进入 TileLang/TileScale、prebuilt wheel 一进程一 backend、vLLM prefill + TileRT decode 的 PD 分离与 KVConnector、两端 KV dtype 必须一致、MTP、性能自报数字 | 已核验（2026-10-01）：本地为仓库 README 快照；性能数字**厂商自报，未复现** |
| R6 | `13-e2e-training-and-serving/raw/github-vllm-ascend.md` | https://github.com/vllm-project/vllm-ascend | 昇腾上的 vLLM 硬件插件：hardware-pluggable RFC、硬件前提、CANN 9.1.0 / PyTorch 2.10.0 / TorchNPU 2.10.0.post4、分支策略、快照时的 release 状态 | 已核验（2026-10-01）：本地为仓库 README 快照（抓取于 2026-09-30） |
| R7 | `06-huawei-ascend/raw/megagdn-pto/blog/mega_gdn_en.md` | https://github.com/huawei-csl/megagdn-pto | 昇腾侧的工程事实：ACL Graph（NPU 版 CUDA Graph）在 vLLM-Ascend 里只覆盖 `decode_only`、prefill 仍走 eager、动态形状要靠 padding/bucketing；`cu_seqlens` 的定义与 NPU 与 Triton 的 grid 差别；TTFT 用 vLLM 的 `first_token_latency`（`vllm/v1/metrics/stats.py`）并报中位数；各家 kernel 的加速比 | 已核验（2026-10-01）：本地为博客全文；其中的性能数字**第三方自报，未复现** |
| R8 | `01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/gemv.md` | https://tilelang.com/deeplearning_operators/gemv.html | GEMV 五级阶梯与 Hopper 上的耗时表；文件自标“experimental and may be incomplete” | 已核验（2026-10-01）：本地为官方文档源文件（470 行）；**表格未写 shape**，见“待补资料” |
| R9 | `07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/TileLang-Ascend Programming Guide.md:2491-2552` | https://github.com/tile-ai/tilelang-ascend（适配器仓库） | msProf 的两种用法与可视化产物；aclGraph 的 capture/replay 模型与示例代码 | 已核验（2026-10-01）：逐节比对行号（第 6 章 2491-2527、第 7 章 2528-2552） |
| R10 | `07-tilelang-ascend-adapter/raw/tilelang-ascend/examples/aclgraph/README.md` 与 `rms_rope_aclgraph.py` | 同上 | ACLGraph 的完整示例：`pass_configs` 四个开关、`torch.npu.graph` 的用法、`torch.testing.assert_close(rtol=1e-2, atol=1e-2)` 的收尾断言、如何用 host 日志确认入图生效 | 已核验（2026-10-01）：逐行比对；**未在 NPU 上运行** |

---

## 关于性能数字的说明

本章出现的每一个数字都带来源与条件，汇总如下：

| 数字 | 来源 | 测量/口径条件 |
| --- | --- | --- |
| H100 fp16 1000 TFLOPS、HBM3 over 3 TB/s | R2 | 厂商公开规格，未实测 |
| 机器平衡点约 333 FLOP/Byte | **本书推导**，输入为 R2 的两个数字 | 峰值/带宽之比 |
| Qwen2.5-7B 的形状与参数量（28 层 / GQA 28:4 / 131072 / 7.61B / 6.53B） | R1 | 模型卡原文；`hidden_size`/`head_dim`/`intermediate_size` 为本书反推 |
| KV 字节、块账、并发估算、翻转点、权重下界、占比表 | **本书模型输出** | `code/01`、`code/02`、`code/03`；bf16、H100 公开规格、完美重叠下界 |
| static 与 continuous batching 的倍数（2.74× 等） | **本书仿真输出** | `code/04`，参数人为设定（64 请求 / 8 槽位 / 8 ms 步长 / 12 us per token） |
| TTFT/TPOT 的三种算法差值（0.20% / 4.90%）与 92.43× 的污染倍数 | **本书模型输出** | `code/05`，输入为演示用假设值 |
| GEMV 五级阶梯的耗时（0.16607 / 0.02419 / 0.00809 / 0.00675 ms vs cuBLAS 0.00784 / Triton 0.00773） | R8 | **官方文档自报**，Hopper GPU，**表格未写 shape**；官方 example 默认 `N=K=1024`（S7），作者未复现 |
| 昇腾 FA 的 26.8 µs / 320.1 TFLOPS（MHA, `SEQ_LEN=4096, D=128`）与 GQA 的 3030 µs vs Torch 2864 µs | S16 | **厂商自报**，Ascend 950 / dav-3510 / bf16；作者未复现；两组数字形状与硬件不同，互相不可比 |
| 昇腾 backward dQ 的 bf16 原子累加相对 L2 误差约 1.024% | 第 15 章引用的适配器 README（在本章只作定性引用） | **厂商自报**，Ascend 950DT、CANN/Bisheng 9.2.0、fast-math 打开 |
| TileRT 的 469 / 1000+ / 500 / 600 tok/s | R5 | **厂商自报**，条件见 README 的 News 条目；与本章账本无可比性，只用于说明“超低延迟路线存在” |
| GDN 内核 3×、prefill TTFT 降低 15% | R7 | **第三方自报**，910B2 / Atlas A2；本章只引用其关于 ACLGraph 与 TTFT 口径的方法论部分 |

**没有来源的、或只有宣传口径的数字一律未采用。** 任何需要真实加速器才能得到的
数字（实测延迟、吞吐、算力利用率）在本章中均未出现。

---

## 待补资料

1. **vLLM 的自定义算子注册接口在 corpus 里没有原文。** `vllm-docs.md` 只有
   文档首页（71 行），只列了能力清单，没有“怎么注册一个自定义 attention /
   GEMM 后端”“KV cache 的 layout 与 dtype 约束”这类接口级内容。因此本章
   5.1 节只给了能力清单与“怎么自己去找”的办法，没有写接口名。
   建议后续版本补抓 vLLM 文档的
   `contributing/`、`design/kv_cache/`、`features/quantization/` 与
   `models/supported_models/` 下的页面。
2. **Qwen2.5-7B 的 `config.json` 缺失。** 模型卡给了参数量、层数与 GQA 配置，
   但没有 `hidden_size` / `head_dim` / `intermediate_size` / `vocab_size`。
   本章用参数量反推并互校（误差 0.07%），结论自洽；但严格说这三个字段应当
   以 HF 仓库的 `config.json` 为准。建议后续版本补抓该文件。
3. **GEMV 教程的 benchmark shape 未写明。** R8 的五级阶梯表给了延迟但没有
   shape，官方 example 的默认是 `N=K=1024`。因此本章引用这组数字时明确标注了
   不确定性；要把它当成可复现的对照，需要补 `N`、`K`、dtype 与卡型。
4. **paged KV 的块表内存开销与真实碎片没有实测数据。** 本章 2.6 节只给了
   块账与上界推导，没有 allocator 的真实碎片率，也没有 `block_table` 的
   实际字节数。`vllm-docs.md` 首页提到 PagedAttention 但没有给出实现细节。
5. **vLLM-Ascend 的图模式与自定义算子接入没有原文。** R7 是第三方博客，
   R6 是仓库 README；两者都没有给出“一个 TileLang kernel 怎么在
   vllm-ascend 里被调用”的具体路径。本章 5.3 节因此只写到环境与形态层面。
6. **昇腾 950 上本章三类 kernel 的实测数据缺失。** 本章没有 NPU，无法运行；
   `examples/ascend/` 下有官方 README 的性能表（S16），但没有覆盖
   decode attention / MoE / 量化 GEMM 这三类推理算子在超低 batch 下的表现。
