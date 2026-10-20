# 第 02 章 参考资料

本章所有结论都基于下表中的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现「最新版」这类表述。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
因此本章没有任何性能数字来自本机实测；`code/` 下两个示例都能在 CPU 上离线运行，
验证级别见 `code/README.md`。

---

## 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:94-140` | `tilelang.compile(...)` 的签名：`out_idx`、`execution_backend` 的六种取值、`target` 支持字符串/字典/`Target` | 已核验（2026-10-01）：`rg -n "^def compile" tilelang/jit/__init__.py` → 94；逐行读签名与 docstring |
| S2 | `_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py:571-600` | `@tilelang.jit` 装饰器（用户侧入口） | 已核验（2026-10-01）：`rg -n "^def jit"` → 571/575/588（三个重载） |
| S3 | `_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:41,206,516,532` | `class JITKernel`、`__call__`、`get_profiler()`、`get_kernel_source()` | 已核验（2026-10-01）：`rg -n "class JITKernel\|def __call__\|def get_profiler\|def get_kernel_source"` 逐条命中 |
| S4 | `_vendor/tilelang-v0.1.15/tilelang/profiler/bench.py:31-60` | `do_bench(...)`：多种计时后端、两次运行之间清 L2、按运行时长自动定 warmup/repeat | 已核验（2026-10-01）：`rg -n "def do_bench"` → 31；「Clearing L2 cache between runs」在 :48 |
| S5 | `_vendor/tilelang-v0.1.15/examples/norm/rms_norm.py:22-30` | RMSNorm 的融合写法：`T.reduce_sum` 与 `T.rsqrt` 在同一个 `T.Kernel` 作用域内 | 已核验（2026-10-01）：`rg -n "reduce_sum\|rsqrt"` → 25、27；文件可被 `python3 -m py_compile` 解析 |
| S6 | `_vendor/tilelang-v0.1.15/examples/elementwise/example_elementwise_add.py:11-33` | 逐元素融合的最小例子（`@tilelang.jit` + `T.Parallel`） | 已核验（2026-10-01）：`rg -n "@tilelang.jit\|T.Parallel"` → 11、27 |
| S7 | `_vendor/tilelang-v0.1.15/examples/deepseek_mla/example_mla_decode_paged.py:30-38` | paged decode 的真实接口形态：`block_table` 与 `cache_seqlens` | 已核验（2026-10-01）：`sed -n '30,38p'` 逐行阅读；文件可被 `python3 -m py_compile` 解析 |
| S8 | `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:141-152` | 「你写 TileLang，Ascend C 里可以省掉什么」对照表：缓冲与数据搬运、tiling、SIMT/SIMD、调度与流水、同步 | 已核验（2026-10-01）：`rg -n "Compared with writing Ascend C directly"` → 141，表格到 152 |

「已核验」的含义：在锁定版本的工作树上用 `rg -n` 定位到定义行，并阅读了完整签名或
docstring；行号区间与文件内容一致。**这不是运行验证**——本章没有任何 API 在真实硬件上运行过。

---

## corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md` | https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | H100 SXM5：HBM3「over 3 TB/sec」（:334）、「Peak FP16 Tensor Core 1000 TFLOPS」（:141，Table 1，dense）、shared memory 可配到 228 KB（:381） | 已核验（2026-10-01）：本地为全文；厂商公开资料，Table 1 自述「preliminary performance specs」，**非实测** |
| R2 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/profiles/h200.yaml` | https://github.com/tile-ai/TileOPs（快照 commit `50dae81fadaf`） | H200 机器参数：fp16 989.5 TFLOPS / HBM 4800 GB/s，标定 0.7370（burst）、0.6684（sustained）、0.9380（带宽最大值）、0.8183（copy 混合）；标定条件写在文件注释里 | 已核验（2026-10-01）：本地为仓库快照，文件头注释给出测量条件（2048 MB working set、5 次 ×200 轮、显存 3201 MHz、700 W 限制） |
| R3 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/docs/design/roofline.md` | https://github.com/tile-ai/TileOPs | 记账法（:31 中间量不计）、判据（:14-27）、latency-bound 定义（:40）、FLOP 计数惯例（:47-58）、SFU 提醒（:34） | 已核验（2026-10-01）：逐行阅读上述行号 |
| R4 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/ops/compile_boundary.py` | https://github.com/tile-ai/TileOPs | 与 `torch.compile` 的接缝约束：被追踪的 `forward` 不能构造 kernel（:15-17）；key 必须是 `str`（:19-22）；Inductor 会把 fake 输出形状烤进磁盘缓存（:24-25） | 已核验（2026-10-01）：读取 :1-30 |
| R5 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/docs/design/layer-boundaries.md` | https://github.com/tile-ai/TileOPs | 「PyTorch is the spec oracle, not a runtime path」（:47）：算子层不允许在 forward 里直接调用高层 PyTorch 算子 | 已核验（2026-10-01） |
| R6 | `../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/formulas.py` | https://github.com/tile-ai/TileOPs | 逐元素与复合算子的 FLOP 计数惯例（如 `sigmoid = 4`、`silu = 5`）、attention 每分数 5 FLOP | 已核验（2026-10-01）：读取模块头部常量与注释 |
| R7 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_API_aolapi_context_ops-nn_aclnnApplyAdamW_md.md` | https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/aolapi/context/ops-nn/aclnnApplyAdamW.md | 优化器阶段：`varRef`/`mRef`/`vRef` 三输入输出 + `grad` + 超参；AdamW 公式（:25-60） | 已核验（2026-10-01）：CANN 官方算子文档镜像 |
| R8 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_API_aolapi_context_ops-transformer_aclnnKvRmsNormRopeCache_md.md` | https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/aolapi/context/ops-transformer/aclnnKvRmsNormRopeCache.md | 融合算子：切分尾轴做 RMSNorm + RoPE，再 scatter 写入两块 KV cache（:25） | 已核验（2026-10-01） |
| R9 | `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_API_aolapi_context_ops-nn_aclnnAddRmsNormDynamicQuant_md.md` | https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/aolapi/context/ops-nn/aclnnAddRmsNormDynamicQuant.md | 融合的动机原文：「…融合起来，减少搬入搬出操作」（:25） | 已核验（2026-10-01） |
| R10 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2205-14135.md` | https://arxiv.org/abs/2205.14135 | FlashAttention：IO-aware 的论证；BERT-large 15%、GPT-2 3×、LRA 2.4× | 已核验（2026-10-01）：本地为摘要全文；数字来自论文自述，条件见论文正文，本章未把它当作 TileLang 的性能 |
| R11 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2407-08608.md` | https://arxiv.org/abs/2407.08608 | FlashAttention-3：FA2 在 H100 上 35% 利用率；FA3 FP16 达 740 TFLOPs/s（75%）、FP8 接近 1.2 PFLOPs/s | 已核验（2026-10-01）：本地为摘要；论文自述数字 |
| R12 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2607-04454.md` | https://arxiv.org/abs/2607.04454 | 「Correct but Slow」：22 个 Triton/TileLang kernel 在 A100 与 GH200 上；一个 LayerNorm 比 PyTorch 基线慢 300 倍以上；库相对效率 + roofline 利用率作为筛选判据 | 已核验（2026-10-01）：本地为摘要全文 |
| R13 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2502-11089.md` | https://arxiv.org/abs/2502.11089 | NSA：「arithmetic intensity-balanced algorithm design」的原文表述 | 已核验（2026-10-01） |
| R14 | `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2506-08889.md` | https://arxiv.org/abs/2506.08889 | SeerAttention-R：「Using TileLang, we develop a highly optimized sparse decoding kernel…」（TileLang 作为研究算子的实现载体） | 已核验（2026-10-01） |
| R15 | `../tilelang-corpus/90-synthesis/01-knowledge-system.md:44-48` | （本书 corpus 内的合成文档，无独立上游链接） | TileLang 的定位：「位置在 Inductor 之下、CUDA/Ascend C 之上」 | 已核验（2026-10-01） |

---

## 上游链接核验说明

corpus 里的 `11-related-compilers/raw/pytorch-torch-compile-docs.md` 与
`pytorch-inductor-kernels.md` **只有 11 行 / 306 字节**：`docs.pytorch.org` 的 stable 索引页
返回的是一张 302 跳转页（指向文档改版后的新路径），抓取工具把跳转页存了下来，
所以 corpus 里没有可用正文。本章需要 TorchDynamo / TorchInductor / AOTAutograd /
kernel 命名这些内容，因此在写作时（2026-10-01）直接核验了下面四个上游页面，
并逐句引用了其中的原文：

| id | URL | 用途 | 核验状态 |
| --- | --- | --- | --- |
| U1 | https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html | TorchDynamo（Frame Evaluation API 捕获图）、TorchInductor（默认后端，在 NVIDIA/AMD/Intel GPU 上以 Triton 为构建块）、AOTAutograd（把反向也提前捕获） | 已核验（2026-10-01）：抓取全文并逐句核对原文 |
| U2 | https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_get_started.html | eager 的逐元素算子「read → modify → write back」；「The single most important optimization that inductor performs is fusion」；`cos`→`sin` 从 2 读 2 写变成 1 读 1 写；`TORCH_COMPILE_DEBUG=1` | 已核验（2026-10-01） |
| U3 | https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_inductor_profiling.html | kernel 命名 `triton_poi_fused_<aten算子>_<序号>`（`poi` = pointwise）；前向图与反向图各一个编译产物；densenet121 小 batch 时「GPU busy」只有 32.69% | 已核验（2026-10-01） |
| U4 | https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler_faq.html | 加速手段（垂直融合、水平融合、乱序执行、自动工作放置）；「Inductor first takes care of fusing whatever it can and only then generates Triton kernels」；graph break 与「We cannot propagate gradients through a graph break」 | 已核验（2026-10-01） |

**注意**：这些页面是会被上游改版的在线文档，且本机没有 GPU、没有在该版本 PyTorch 上
复现过任何行为。它们在本章只用于说明「`torch.compile` 的设计意图与接口边界」，
不用于任何性能结论。若将来 corpus 重新抓取到可用正文，应把上表替换成 corpus 路径。

---

## 本章出现的每一个数字（来源与条件）

| 数字 | 出现在 | 来源 | 测量/推导条件 |
| --- | --- | --- | --- |
| H100：over 3 TB/s、FP16 1000 TFLOPS | 2.2 | R1 | 厂商公开资料，H100 SXM5，Table 1 自述 preliminary，未实测 |
| H100 平衡点 ≈ 333 FLOP/Byte | 2.2 | **本书推导**（1000/3.0） | 用 R1 的两个数字做比值 |
| H200：989.5 TFLOPS、4800 GB/s、标定 0.7370 / 0.6684 / 0.9380 / 0.8183 | 2.2 | R2 | 仓库自带微基准标定，条件见该文件注释；**不是本章实测** |
| H200 平衡点 162（标定后） | 2.2、2.3 | **本书推导** | 729.3 TFLOPS / 4502 GB/s |
| 逐元素链 eager 134.2 MB / 融合 67.1 MB | 2.3、4.2 | **本书推导**（`op_traffic.py`） | 16.8M 个 fp16 元素，2 算子链 |
| `cos`→`sin` 的 2 读 2 写 → 1 读 1 写 | 4.2、图 3、图 5 | U2 | PyTorch 官方文档举例 |
| LayerNorm 134.2 MB / 268.4 MB | 2.3、4.2 | **本书推导**（`op_traffic.py`） | 8192×4096 fp16，每元素 8 FLOP（本书口径） |
| GEMM 4096³：137 GFLOP / 100.7 MB / AI 1365 | 2.3 | **本书推导**（`op_traffic.py`） | fp16，A/B/C 各读或写一遍 |
| attention prefill（S=4096）：42.2 与 2068 FLOP/Byte | 2.3 | **本书推导**（`op_traffic.py`） | 32 head、d=128、fp16；分数矩阵 6 趟访存是假设 |
| attention decode（KV 32768）：AI 1.01 | 2.3 | **本书推导**（`op_traffic.py`） | 每 token 读整段 KV |
| FA2 在 H100 上 35% 利用率；FA3 FP16 740 TFLOPs/s（75%）、FP8 接近 1.2 PFLOPs/s | 2.3 | R11 | 论文自述，测量条件见论文正文 |
| FlashAttention 训练加速 15% / 3× / 2.4× | 2.3 | R10 | 论文自述（BERT-large / GPT-2 / LRA），条件见论文正文 |
| 优化器 26 字节/参数、7B 参数 182 GB、8B 参数 208.78 GB | 1.1、2.4 | **本书推导**（`op_traffic.py`、`step_budget.py`） | bf16 梯度 + fp32 状态；张量与公式来自 R7 |
| all-reduce 搬 $2(N-1)/N$ 倍梯度、8 卡 8B 模型 28.11 GB | 1.1、2.4 | **本书推导** | ring all-reduce 的标准数据量；无实测 |
| 训练一步 / 推理的占比表（27.9% / 55.9% / 14.3% / 1.9% 等） | 2.4、习题 E12 | **本书推导**（`step_budget.py`） | 粗模型，假设写在脚本注释与 `code/02-step-budget/README.md` |
| densenet121 小 batch「GPU busy」32.69% | 3.1 | U3 | 该页所引的 benchmark 脚本，非本章实测 |
| 「正确但慢 300 倍」的 LayerNorm | 5 | R12 | 22 个 kernel、A100 与 GH200，论文自述 |

**没有来源的、或来自厂商宣传口径的数字一律未采用。** 所有需要真实加速器才能得到的数字
（实测算力利用率、延迟、吞吐）在本章中均未出现，或明确标注为论文自述。

---

## 待补资料

1. **corpus 里的 torch.compile 资料不可用（优先级：高）。**
   `11-related-compilers/raw/pytorch-torch-compile-docs.md` 与 `pytorch-inductor-kernels.md`
   抓到的都是 `docs.pytorch.org` 的 302 跳转页（分别 11 行 / 306 字节），没有正文。
   本章 4.3、4.4 节因此只能引用在线官方文档（见「上游链接核验说明」）。
   建议后续版本把 `sources.yaml` 里的 URL 改成不跳转的新路径并重抓，例如
   `https://docs.pytorch.org/docs/2.14/user_guide/torch_compiler/torch.compiler.html`、
   `.../torch.compiler_get_started.html`、`.../torch.compiler_inductor_profiling.html`、
   `.../torch.compiler_faq.html`。
2. **roofline 模型的原始论文缺失（优先级：中）。** 本章用的是 TileLang 生态里
   TileOPs 的记账口径（R3），原始出处是 Williams、Waterman、Patterson 2009 年的
   "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
   （CACM；DOI 10.1145/1498765.1498785）。corpus 里没有这条来源，
   第 28 章若要讲性能模型史，建议补抓。
3. **训练侧并行与显存的一手资料缺失（优先级：中）。** 本章 1.1 节关于优化器状态占多少
   字节、ZeRO 如何分片，只有算术推导（本书口径）；corpus 里没有 ZeRO（arXiv:1910.02054）
   或 Megatron-LM 的一手材料。第 17 章（端到端训练）大概率需要它们。
4. **「反向 ≈ 2 倍前向」缺少官方表述（优先级：低）。** 本章的做法是对单个 GEMM 逐项
   推导（$\mathrm{d}A$、$\mathrm{d}B$ 各 $2MNK$），结论精确；但没有引用官方文档的同类
   表述。若第 17 章要写训练成本预算，建议补一条 PyTorch 或 Megatron 的说明。
5. **Triton 文档快照过短（优先级：低）。** `11-related-compilers/raw/triton-docs.md`
   只有 41 行（首页快照），不足以支撑「Triton 支持哪些后端目标」这类结论。
   本章 4.4(c) 因此写成了「本书基于官方文档措辞的判断」而不是事实断言。
