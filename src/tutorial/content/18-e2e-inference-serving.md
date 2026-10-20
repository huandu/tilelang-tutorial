# 第 18 章 端到端 II：推理与部署

## 学习目标

读完本章，你应当能够：

1. 说清 prefill 与 decode 在**形状、算术强度、瓶颈**上的差别，并解释为什么
   同一个模型、同一块卡，prefill 要优化张量核利用率、decode 要优化访存与批量。
2. 用 Qwen2.5-7B 的 GQA 配置（28 层、28 个 query head、4 个 KV head、head_dim 128、
   bf16）手算出 KV cache 的每 token / 每层 / 每序列字节数，并说出 GQA 相对 MHA
   省了多少、代价是什么。
3. 说明 paged KV 解决的是什么问题、它的块账怎么算、它对 attention kernel 提出了
   什么新要求（间接寻址）。
4. 解释 continuous batching 为什么决定吞吐，以及「batch 内长度不齐」为什么把
   varlen 与 paged KV 变成 kernel 的硬要求而不是可选项。
5. 读懂并在锁定版本的 `examples/` 里指出三个推理算子的实现位置：decode
   attention（`flash_decoding/`）、fused MoE（`fusedmoe/`）、量化 GEMM
   （`dequantize_gemm/`），并说清各自的接入方式与约束。
6. 说出把一个自定义 kernel 放进 vLLM / TileRT / vllm-ascend 这类服务框架时，
   需要对齐的四类东西（形状与元数据契约、dtype 与 layout、stream 与图模式、
   调度接口）。
7. 在昇腾上把算子跑起来并验证：`torch_npu` 的设备与 stream、ACLGraph 的
   capture/replay、msProf 的采集命令，以及入图对形状的要求。
8. 用 TTFT / TPOT / 吞吐的口径报告推理性能，并说清哪些做法会让结论被
   缓存、预热或首次编译污染。

## 前置知识

* 第 02 章：roofline、算术强度、机器平衡点——本章全部账本都用这一套口径。
* 第 14 章：`T.gemm` 的 M/N/K 协议、split-K 与 grouped GEMM——decode 的
  attention 与 MoE 都是它们的变体。
* 第 15 章：online softmax 的推导、`varlen` 的 `cu_seqlens`、split-KV 的
  `ℓ_s / Σℓ` 加权、paged KV 的 `block_table`、昇腾上 `QKᵀ` 必须绕 workspace。
* 第 16 章：一个 decode step 的 MoE / MLA 成本账本与「哪些算子受什么限制」的
  结论表——本章第 4 节的表直接建立在那张表上。
* 第 17 章：训练循环、checkpoint、把热点算子换成 TileLang kernel 的方法。
  本章假设你已经有一个训练好的 7B 模型权重。

**本章的验证边界（很重要）**：写作环境是 macOS (Apple Silicon)，
**没有 NVIDIA GPU，也没有昇腾 NPU**。因此本章出现的数字只有四类，来源都写清：

* **本书模型的输出**：`code/01`–`code/05`、`code/07` 里的纯 CPU 程序，
  可在任意机器上复算，标为「CPU 逻辑验证」；
* **仿真输出**：`code/04` 的调度仿真，参数人为设定，标为「仿真，非测量」；
* **厂商/上游资料的既有数字**：带 corpus 路径与测量条件，标为
  「厂商自报/文档自报，作者未复现」；
* **静态核对结果**：`code/06`–`code/08` 的 kernel 骨架只做了语法与 API 核对，
  标为「仅静态核对」。

本章**没有任何一条本机实测的 kernel 性能数字**，也**不声称任何 kernel 被运行过**。

---

## 1. 问题的由来：训练完的模型怎么变成服务

第 17 章结束时你手上有一组权重和一个训练循环。把它们变成服务，要补上四件
训练里不存在的事：

1. **自回归生成**：模型每次只产出 1 个 token，然后把它接回输入再算一次。
   训练时一次前向处理整段序列；推理时被拆成了「先处理 prompt」和「再一个
   一个吐 token」两个性质完全不同的阶段。
2. **缓存**：每个 token 的 K、V 一旦算出来就不会再变，于是可以缓存起来复用。
   这份 KV cache 的体积会随并发数与上下文长度线性增长，成为显存的主要消费者。
3. **调度**：请求来自四面八方、长度各不相同、随时到达随时结束。一次只服务
   一个请求会浪费机器，把请求攒成固定 batch 又会让早完成的请求空等。
4. **算子之外的接口**：kernel 再快，也要先满足框架对形状、dtype、stream 与
   图模式的契约，才会真正被调用。

第 4 个问题把本章和第 14–16 章区分开：那三章回答「kernel 怎么写才快」，
本章回答「服务为什么慢、慢在哪、以及 kernel 该写成什么样才能被用上」。

一句话概括全章：

> **推理的性能不是由「哪个 kernel 最快」决定的，而是由「这个算子在这一步里
> 占多少时间、以及调度把它放在什么形状下执行」决定的。**

---

## 2. 概念与模型

### 2.1 两个阶段：prefill 与 decode

一次请求被分成两个阶段（图 `figures/01-prefill-vs-decode.md`）：

| | prefill | decode |
| --- | --- | --- |
| 一次处理多少 token | prompt 的全部 N 个 | 每个请求 1 个 |
| 矩阵乘的 M | N（几百到几十万） | batch（1 到几百） |
| attention 的 query 长度 | N（要算 `N×N` 的分数矩阵） | 1（只有一行分数） |
| KV 访问 | 读入并与当前 prompt 一起写入 cache | 从 cache 读已有的全部 KV |
| 主要瓶颈 | 张量核的利用率、`N²` 中间结果的搬运 | HBM 带宽：权重 + KV |
| 优化手段 | 分块 + online softmax + 流水（第 15 章） | 批量、KV 分块并行、量化 |

注意 prefill 也不是「纯 compute」：第 15 章算过，朴素 attention 的瓶颈是
`O(N²)` 的中间矩阵在 HBM 上来回。prefill 与 decode 的区别不是 compute 与
memory 的二分，而是**算术强度高低**的连续谱：prefill 的 M 大，`AI` 高于机器
平衡点；decode 的 M 只有 batch 大小，`AI` 远低于平衡点。

### 2.2 算术强度决定了 kernel 的选择

沿用第 14 章的口径。GEMM `C[M,N] = A[M,K] @ B[K,N]` 的算术强度是

```text
FLOP  = 2·M·N·K
Byte  = b·(M·K + K·N + M·N)          # A、B 各读一次，C 写一次
AI    = FLOP / Byte
```

当 `M ≪ N, K` 时，`K·N`（权重）项主导分母，于是

```text
AI ≈ 2·M·N·K / (b·K·N) = 2M / b
```

也就是**与矩阵的形状无关，只与 M 和元素字节数有关**。代入 bf16（`b = 2`）
与 H100 的公开平衡点（fp16 1000 TFLOPS / HBM3 over 3 TB/s，约 333 FLOP/Byte），
翻转点是

```text
M* ≈ balance · b / 2 = 333 · 2 / 2 = 333
```

把被省略的 `M·K` 与 `M·N` 两项放回去再二分求解，
`code/01-prefill-decode-roofline/` 得到的实际值是 375
（N=18944, K=3584）与 400（N=3584, K=4608）——比简化式略大。
这个量级解释了一切：

* prompt 长度是几千，prefill 稳稳落在 compute-bound 一侧；
* decode 的 M 是 batch 大小。`batch = 1` 时 `AI ≈ 1`，比平衡点低两个半
  数量级——这是「decode 必须靠批量才能跑快」的算术依据。

顺带一个容易混淆的点：第 14 章给的方形 GEMM 翻转点约 `10³`，本章给的是 375。
两个都对，前提不同——方形 GEMM 里 M、N、K 一起涨，字节数约 `3b·M²`、
`AI = 2M/(3b)`；本章固定 N、K 只让 M 涨。**报翻转点时必须一起报形状。**

### 2.3 decode 的权重底线

decode 的第一个硬约束来自权重：每生成一个 token，都要把模型的全部权重读一遍。
Qwen2.5-7B 有 7.61B 参数（模型卡给出的数字），bf16 下是 15.23 GB；
在 3 TB/s 上读一遍是

```text
15.23e9 B / 3.0e12 B/s = 5.08 ms
```

换算成单请求的生成速度上界约 197 tok/s（**下界模型，不含 KV 流量，也不是实测**）。
这个数字有两个直接推论：

* **不可能靠优化 kernel 突破它**。它是「读一遍权重」的物理成本，只能靠
  （a）把 batch 做大，让同一份权重服务多个请求；（b）把权重读得更少
  （量化）；（c）把权重切到多张卡上（TP/EP，代价是通信）。
* **batch 的直接收益是「摊薄权重流量」**：batch 从 1 到 8，单请求分摊的权重
  流量从 15.23 GB 降到 1.90 GB，理想上界从 197 tok/s 涨到 1576 tok/s。
  但 KV 流量也在随 batch 增长，所以这个线性关系只在小 batch 段成立。

这也是为什么第 16 章那张表里「MLP gate+up」「MLP down」的占比最高：
它们的权重最大。在一个 batch=1、context=8192 的 decode step 里，账本给出的
占比是 52.0% 与 26.0%（见 4.1 节的表）。

### 2.4 KV cache：结构

对每一层、每个 KV head，把每个 token 的 K、V 向量存下来：

```text
K_cache[layer][token][kv_head][head_dim]
V_cache[layer][token][kv_head][head_dim]
```

第 15 章已经把这套结构用在 attention kernel 里：decode 时 `Q` 只有 1 行，
`K`、`V` 却是整段历史。本章关心的是它的**显存形状**：

```text
每 token 每层字节 = 2 · n_kv_heads · head_dim · b
                  └ 2 = K 与 V 各一份
```

GQA（grouped-query attention）的要点是 `n_kv_heads < n_q_heads`：多个 query
head 共享同一份 KV。第 16 章给过一个只用形状就能写出来的结论——decode
attention 的算术强度是

```text
AI = 2 · H_q / (H_kv · b)
```

与序列长度无关，只取决于「每字节 KV 被几个 query head 复用」。Qwen2.5-7B 的
`H_q = 28`、`H_kv = 4`，所以复用率是 7，`AI = 7 FLOP/Byte`（bf16）——
依然远低于平衡点 333，这就是 decode attention 天生是 memory-bound 的原因。

### 2.5 KV cache：Qwen2.5-7B 的显存账本

用模型卡的配置（28 层、28 个 query head、4 个 KV head、上下文 131072、
参数量 7.61B / 非嵌入 6.53B）算下去：

```text
每 token 每层：2 · 4 · 128 · 2 B = 2048 B = 2.0 KiB
每 token 全模型：2048 B · 28        = 57344 B = 56.0 KiB
```

| context | GQA（4 KV heads） | MHA（28 KV heads） | GQA 占 MHA |
| --- | --- | --- | --- |
| 1,024 | 56.00 MiB | 392.00 MiB | 14.3% |
| 4,096 | 224.00 MiB | 1.53 GiB | 14.3% |
| 8,192 | 448.00 MiB | 3.06 GiB | 14.3% |
| 32,768 | 1.75 GiB | 12.25 GiB | 14.3% |
| 131,072 | 7.00 GiB | 49.00 GiB | 14.3% |

（`code/02-kv-ledger/`，可复算。）

三条要记住的推论：

1. **模型卡没给的字段有官方来源。** 本书资料库已存档 Qwen2.5-7B 的 `config.json`
   （`../tilelang-corpus/13-e2e-training-and-serving/raw/hf-qwen2-5-7b-config.md`），
   直接给出 `hidden_size = 3584`、`intermediate_size = 18944`、`vocab_size = 152064`；
   本章早期草稿曾用参数量反推同一组数值（误差 0.07%），现在是引用而非推导。
   `intermediate_size = 18944` 这三个字段模型卡没有直接写出来。本章用
   「按形状算出的参数量必须等于卡片上的 6.53B / 7.61B」反推并互校，误差
   0.07%。这是做显存估算时非常实用的一招（`code/01` 与 `code/03` 都做了这个自检）。
2. **GQA 省 7 倍显存，代价是算术强度只有 MHA 的 1/7。** 省显存和「每字节能
   算多少」是同一枚硬币的两面，报结论时不能只报一面。
3. **并发数与 context 成反比。** 80 GB 卡、权重占 14.17 GiB、留 10% 给运行时，
   KV 预算约 57.83 GiB：context=8192 时能放约 132 条序列，context=131072 时
   只剩 8 条。这正是长上下文服务要做 KV 量化、prefix caching、KV 换出的原因。

### 2.6 paged KV：为什么需要分页

KV 的长度事先不知道：输出长度未知，请求随时到达。连续分配只能按「最坏情况」
预留，浪费来自「整条序列跑不满预留长度」，在变长输出下这个浪费远大于块内碎片。

paged KV 把「逻辑位置」与「物理位置」解耦：

```text
K_cache[layer][physical_block][block_size][kv_head][head_dim]
block_table[sequence][logical_block] = physical_block
```

好处是显存按块分配，碎片上界变成「每条序列的最后一块最多空 block_size-1 个
槽位」；代价有三条：

1. **attention kernel 里 KV 的地址要查表**，从 `k · block_N` 变成
   「读 `block_table` + 偏移」。第 15 章 4.6 节的
   `_vendor/tilelang-ascend-v0.1.2.000/examples/flash_attention/paged_flash_attn_bhsd.py`
   就是这种写法。
2. **间接寻址破坏了 TMA 的静态描述符**：描述符要在编译期确定，而查表结果
   是运行期的。
3. **块表本身要占显存**，而且块大小是「元数据开销」与「碎片」之间的折中：
   `block_size = 16` 时，Qwen2.5-7B 全模型每块 896 KiB。

### 2.7 continuous batching 与 varlen

如果每个 decode step 都只服务「同一批、且这批要一起跑完」的请求，
早完成的序列会一直占着槽位。continuous batching（也叫 iteration-level
scheduling）改成：**每个 decode step 结束时，已完成的序列立刻退出，
新请求立刻补进来**。vLLM 的文档把它和自己的其他特性并列成一句话：

> 「Continuous batching of incoming requests, chunked prefill, prefix caching」
> ——`../tilelang-corpus/13-e2e-training-and-serving/raw/vllm-docs.md`

代价是 kernel 必须接受**长度不齐的 batch**：

* 同一批里，各请求的 KV 长度不同；
* 同一批里，可能既有「正在 decode 的请求」也有「刚进来要 prefill 的请求」
  （chunked prefill 把长 prompt 切成几段，和 decode 混在同一步里跑）；
* 每个请求的 KV 在显存里不连续（paged）。

varlen 的表示方式是业界惯例：把整个 batch 的 token 拼成一条扁平序列，
用前缀和数组 `cu_seqlens` 记录每个样本的起止。

```text
cu_seqlens = [0, 5, 8, 15]
→ batch = 3，各序列长度 [5, 3, 7]
→ 第 i 个样本在扁平数组里的区间是 [cu_seqlens[i], cu_seqlens[i+1])
```

这个约定在 corpus 里有独立佐证：昇腾侧的 GDN 内核文档用完全一样的定义
（`../tilelang-corpus/06-huawei-ascend/raw/megagdn-pto/blog/mega_gdn_en.md`
「Dynamic axes」一节），并指出 NPU 与 CUDA/Triton 的一个差别——NPU 通常把
`block_dim` 固定成核数、在核内循环分配工作量，所以不需要 Triton 那套
`chunk_indices` / `chunk_offset` 的旁路元数据。

第 15 章 5.3 节列过 varlen 的四个坑，本章只需要记住一条最重要的：

> **`max_seqlen` 是编译期常量。** grid 大小按它开，短样本靠越界 guard 跳过。

这意味着「换一个更长的 prompt」会触发一次重新编译（第 7 章的 JIT 缓存），
也是后面 6.2 节「入图要求静态形状」的根源。

---

## 3. 用法：从一次 decode 到一次服务

### 3.1 一次 decode step 的算子清单与耗时占比

先把一次 decode step 拆开。`code/03-decode-step-share/` 把第 16 章的成本账本
扩展成模型全量（28 层 + 最终分类头），并用同一套 roofline 口径算出每个算子的
**下界时间**与**占比**。下面是 batch=1、context=8192、bf16、H100 公开规格下的
模型输出（**下界模型，不是测量**）：

| 算子 | 这一步的规模 | 瓶颈类型（第 16 章的结论） | 模型推断的耗时占比 | 决定占比的因素 |
| --- | --- | --- | --- | --- |
| MLP gate+up × 28 层 | `6·T·d·d_expert` FLOP，权重 7.08 GiB | memory-bound（小 batch 下被权重流量压死） | 52.0% | `d_model·d_expert` 的权重字节；batch |
| MLP down × 28 层 | `2·T·d_expert·d` FLOP，权重 3.54 GiB | memory-bound | 26.0% | 同上 |
| LM head / logits | `2·T·d·vocab` FLOP，权重 1.02 GiB | memory-bound（每步只算一次但权重很大） | 7.5% | 词表大小；只算一次 |
| QKV 投影（q） × 28 层 | 权重 686 MiB | memory-bound | 4.9% | `d²` 权重；GQA 不影响 q |
| 输出投影（o） × 28 层 | 权重 686 MiB | memory-bound | 4.9% | `d²` 权重 |
| decode attention × 28 层 | `4·B·H_q·L·d` FLOP，KV 448 MiB | memory-bound（KV 带宽，`AI = 7`） | 3.2% | context 总长（`B×L`） |
| QKV 投影（k,v） × 28 层 | 权重 196 MiB | memory-bound | 1.4% | GQA 让这一项只有 q 的 1/3.5 |
| RMSNorm + RoPE × 28 层 | 784 KiB 流量 | memory-bound（纯访存） | 0.0% | 层数 × `d`；绝对值极小 |
| **合计** | — | 全部 memory-bound | 100%（4.87 ms 下界） | — |

这张表有四个必须一起说清的读法：

1. **每一步都在读权重。** 占比前 5 名的算子全是「权重流量」类的，加起来
   95.3%。这就是 2.3 节的权重底线在算子层面的样子。
2. **attention 在这个配置下只占 3.2%**，因为 batch=1、context=8192 时 KV 流量
   （448 MiB）比权重流量（约 13 GiB）小得多。**别急着下结论**——
3. **占比随 batch 与 context 剧烈变化**（同一套账本，`code/03`）：

   | batch | context | attention 占比 | 权重类算子占比 | 单步下界 |
   | --- | --- | --- | --- | --- |
   | 1 | 8,192 | 3.2% | 96.8% | 4.87 ms |
   | 1 | 131,072 | 34.7% | 65.3% | 7.22 ms |
   | 32 | 8,192 | 51.2% | 48.7% | 9.79 ms |
   | 32 | 131,072 | 94.4% | 5.6% | 84.95 ms |

4. **所以「该优化什么」取决于你的服务形态**：短上下文、高并发做的是权重
   （量化、大 batch）；长上下文、低并发做的是 KV 带宽（KV 量化、更好的
   attention kernel）。报一张占比表而不报 batch / context，等于没报。

**运行步骤**（本节的表就是它的输出）：

* 前提：Python 3.9+，仅标准库；无需 GPU / NPU。
* 命令：`python3 code/03-decode-step-share/decode_step_share.py`
  与 `--batch 32 --context 8192`。
* 期望输出：与上表一致的数字，外加一段自检
  「每层参数 233.05 M × 28 层 + 嵌入/头 1089.99 M = 7.615 B
  （模型卡 7.61 B，相对误差 0.0694%）」。
* 验证级别：**CPU 逻辑验证**（写作机上实际运行过；见 `code/README.md`）。

### 3.2 调度：static 与 continuous

把「调度」单独拿出来做仿真，才看得清它的分量。`code/04-serving-schedule-sim/`
用一组人为参数（64 个请求、8 个槽位、decode step 固定 8 ms、
prefill 12 us/token、输出长度离散）对比两种策略：

| | static batching | continuous batching |
| --- | --- | --- |
| 总耗时（makespan） | 23689.5 ms | 8661.5 ms |
| 吞吐 | 314.7 token/s | 860.8 token/s |
| 平均 E2E 延迟 | 12084.3 ms | 3952.8 ms |
| p95 E2E 延迟 | 22636.6 ms | 7486.9 ms |

**这是仿真输出，不是测量**，参数换了倍数就变；它证明的是方向。方向背后的
机理很朴素：

* 一个 decode step 的耗时**不随 batch 里有多少条序列而变化**（权重流量
  被摊薄，见 2.3 节），所以空槽位是纯浪费；
* 输出长度离散时，「整批等最长的那个」会让多数槽位长时间空转；
* 只有让完成的序列立刻退出、新请求立刻补进来，槽位利用率才接近 1。

**运行步骤**：

* 前提：Python 3.9+，仅标准库。
* 命令：`python3 code/04-serving-schedule-sim/serving_schedule_sim.py`。
* 期望输出：与上表一致的对比段；改动 `--slots/--step-ms/--seed` 后数字变、
  方向不变。
* 验证级别：**CPU 逻辑验证**；输出是**仿真**，不得当作基准数字引用。

### 3.3 varlen 对 kernel 的要求

把 2.7 节的结论落到 kernel 接口上，varlen 要求四件事：

1. **输入是扁平序列**：`Q_unpad: [UQ, heads, dim]`、`K/V: [UKV, kv_heads, dim]`，
   而不是带 padding 的 `[batch, seq, ...]`。
2. **带上前缀和**：`cu_seqlens_q` / `cu_seqlens_k`（int32），每个 block 自己算
   `cur_start` / `cur_end` / `cur_seqlen`。
3. **grid 按编译期常量开**，短样本靠 guard：`if bx * block_M + i < cur_seqlen`。
   官方 varlen 例子的写法是
   `_vendor/tilelang-v0.1.15/examples/flash_attention/example_gqa_fwd_varlen.py:117-123`。
4. **causal 要右对齐**：`offset = kv_current_seqlen - q_current_seqlen`
   （同文件 `:68`，注释写着 `# always align on the right`）。

**运行步骤**（读代码，不需要设备）：

* 前提：能读 Python；无需 GPU。
* 命令：`sed -n '60,130p' _vendor/tilelang-v0.1.15/examples/flash_decoding/example_gqa_decode_varlen_logits.py`。
* 期望输出：能看到 `cu_seqlens_k` 被用来算 `cur_start_k` / `cur_end_k` /
  `cur_seqlen_k`，以及 `loop_range = T.ceildiv(cur_seqlen_k // num_split, block_N)`
  这样的「按本序列长度决定的循环上界」。
* 验证级别：**仅静态核对**（作者读过源码并逐行比对，未运行）。

---

## 4. 用 TileLang 写推理 kernel

本节给三个算子的写法。原则同第 13–16 章：先官方示例、再改动的部分、
最后说约束。完整骨架在 `code/06`–`code/07`，都标注了**仅静态核对**。

### 4.1 decode attention：为什么要 split-KV

decode 时 `seqlen_q = 1`。用第 15 章的 flash attention 骨架直接写，
grid 是 `(ceildiv(seq_q, block_M), heads, batch)`；`seq_q = 1` 时第一维是 1，
并行度只剩 `heads × batch`。Qwen2.5-7B 的 `heads = 28`，batch=1 时
只有 28 个 block，机器远未被喂饱。

配套的问题是：一个 block 要沿 KV 方向循环几十万次，**串行长度就是延迟**。

flash decoding（split-KV）用两步解决（图 `figures/04-decode-split-kv.md`）：

* **split 段**：把 KV 切成 `num_splits` 段，第 `sid` 段只扫
  `[split_start(sid), split_end(sid))`，写出两样东西——本段**已归一化**的
  输出 `Output_partial[..., sid, :]` 与本段的 lse；
* **combine 段**：按 `ℓ_s / Σ_s' ℓ_s'` 加权合并。

官方实现在 `_vendor/tilelang-v0.1.15/examples/flash_decoding/`：

| 文件 | 内容 | 关键行 |
| --- | --- | --- |
| `example_mha_inference.py` | MHA decode 的 split + combine | `:8` `num_split = 4`；`:27-28` 两块 `alloc_global`；`:30` split 的 grid；`:111` 写 lse；`:116` combine 的 grid；`:137,149,151` 先减最大值再算权重 |
| `example_gqa_decode.py` | GQA decode，含 `mask` 与 `cur_kv_head` | `:53` KV 形状 `[batch, seqlen_kv, groups, dim]`；`:58` `kv_group_num = heads // groups`；`:61` `valid_block_H = min(block_H, kv_group_num)`；`:93` `cur_kv_head`；`:123` mask 写法 |
| `example_gqa_decode_varlen_logits.py` | varlen + 输出分数矩阵 `S` | `:35` `out_idx=[-2, -1]`；`:58` `cu_seqlens_k`；`:82-84` 由前缀和算长度；`:128` 写回 `S` |

**为什么必须加权合并。** 每一段输出已经各自除过自己的 `ℓ_s`：

```text
O_s = (1/ℓ_s) · Σ_{j∈段 s} exp(s·x_j − m_s) · v_j
O   = Σ_j P_j v_j = ( Σ_s ℓ_s · O_s ) / ( Σ_s ℓ_s )
```

所以权重是 `ℓ_s / Σ_{s'} ℓ_{s'}`，不是 1。第 15 章 5.4 节推导过这一点，
并给了量级：昇腾 backward 里 dQ 用 bf16 跨 64 个 KV tile 原子累加，
相对 L2 误差约 1.024%（**厂商自报，作者未复现**）。归约顺序真的会影响
正确性，不只是「理论上的舍入」。

`code/06-tilelang-decode-attention/decode_attention_splitkv.py` 是本章的骨架，
相比官方示例只有三处简化（写在文件头）：只做 decode、KV 用 varlen、
split 边界按比例切（`split_start = cur_seqlen_k * sid // num_splits`），
避免短序列被切出空段。

**运行步骤**：

* 前提：NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang v0.1.15；
  `dim = 128`、`heads % kv_heads == 0`、`num_splits <= min_kv`。
* 命令：先在官方示例上确认环境
  `python3 _vendor/tilelang-v0.1.15/examples/flash_decoding/example_mha_inference.py`，
  再编译骨架（见 `code/06-tilelang-decode-attention/README.md`）。
* 期望输出：与参考实现 `torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2)`；
  性能口径是 `2·batch·heads·kv_seqlen·dim` FLOP 除以延迟，并报出
  `num_splits`、`block_N`、`num_stages`、卡型。
* 验证级别：**仅静态核对**（`py_compile` + API 逐项比对）；本机未运行，
  也没有任何实测数字可引用。

### 4.2 M=1 时 GEMM 会退化成什么：GEMV 的阶梯

decode 的每个投影在 `batch = 1` 时就是 GEMV：`(1, K) × (K, N)`。
corpus 里有一篇专门的官方教程
（`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/gemv.md`，
文件开头自标「still experimental and may be incomplete」），
它把 GEMV 从慢到快做了五级，并给出在 **Hopper GPU** 上的一组耗时：

| 实现 | 延迟 |
| --- | --- |
| `torch` / cuBLAS | 0.00784 ms |
| Triton | 0.00773 ms |
| `naive_gemv`（朴素的 GEMM 分块） | 0.16607 ms |
| `splitk_gemv`（沿 K 分线程 + `atomicAdd`） | 0.02419 ms |
| `splitk_gemv_vectorized`（128-bit 向量化读写） | 0.00809 ms |
| `splitk_gemv_vectorized_tvm`（换成 `tvm_thread_allreduce`） | 0.00675 ms |

（**文档自报，作者未复现**；文档的表格没有写 shape，而
`_vendor/tilelang-v0.1.15/examples/gemv/example_gemv.py:330-337` 的默认是
`N = K = 1024`。引用这组数字时必须连同这个不确定性一起说。）

这五级里每一次改动都对应本章的一条结论：

* 第 3 行（0.166 ms）说明**把 GEMM 的 tile 策略直接搬到 M=1 上会灾难性地差**：
  权重只读一遍，但线程没有并行度可言；
* 第 4 行（0.024 ms）说明 M 不够就**沿 K 并行**——这正是 4.1 节 split-KV
  的同一种思路（并行维不够，就切归约维）；
* 第 5、6 行说明 GEMV 是**纯带宽问题**：把每次读写做成 128 bit 事务、
  再用更好的 all-reduce 收尾，就能追平甚至略超 cuBLAS。

写 decode 投影时的直接结论：M 小时不要指望 `T.gemm` 的 tile 划分能救你，
先确认它是不是被权重流量限制住了；如果是，考虑量化（4.4 节）或把 batch 做大。

**运行步骤**：

* 前提：NVIDIA GPU + TileLang v0.1.15。
* 命令：`cd _vendor/tilelang-v0.1.15/examples/gemv && python3 example_gemv.py`。
* 期望输出：官方示例会先 `profiler.assert_allclose(...)` 做正确性校验，
  再依次打印 Torch 与 TileLang 各实现的延迟；最后打印 `Test passed!`。
* 验证级别：**仅静态核对**（作者只核对了 API 与默认参数，未运行）。

### 4.3 fused MoE：把 dispatch 的代价算进来

MoE 推理的一条专家链是「路由 → permute（dispatch）→ grouped GEMM →
combine」。第 16 章的账本已经指出：permute 与 combine 是**零 FLOP 的纯搬运**，
却最容易在账本里被忽略。

锁定版本的实现把它拆成两个 kernel：
`_vendor/tilelang-v0.1.15/examples/fusedmoe/example_fusedmoe_tilelang.py`

| 部分 | 内容 | 关键行 |
| --- | --- | --- |
| 共享专家 | `moe_forward_tilelang_shared`：一个普通的两段 MLP | `:12-89` |
| 路由专家 | `moe_forward_tilelang_routed`：grouped GEMM | `:92-226` |
| host 侧准备 | `argsort` → `bincount` → 四张表 | `:344-382` |
| 内核算行号 | `m_start = m_start_padded - group_padded_offsets[g] + group_offsets[g]` | `:157-161`、`:202-206` |
| combine 权重 | 写回时乘 `routed_expert_weights[m_start + i]` | `:225` |

四张表的含义是这一段的核心：

| 张量 | 含义 |
| --- | --- |
| `group_sizes` | 每个专家分到多少个 `(token, expert)` 对 |
| `group_offsets` | 每个专家的行区间在**紧凑** token 数组里的起点 |
| `group_padded_offsets` | 每个专家的行区间在**对齐后**数组里的起点 |
| `group_idx_for_bx` | 第 `bx` 个 block 该去处理哪个专家 |

`code/07-tilelang-moe-and-quant/moe_dispatch_contract.py` 用一个小规模例子
把这份契约复算了一遍，并断言三件事：每个 `(token, expert)` 对恰好覆盖一次、
对齐区间盖住紧凑区间、内核里的行号公式不越界。默认例子（9 个 token、
top_k=2、4 个专家）的输出是：

```text
  紧凑数组长度（group_sum）        : 18
  对齐后需要的 block 数（M）       : 5 = ceil(18/128) + 4
  对齐浪费的行数                   : 622
  [ok] 18 个 (token, expert) 对恰好覆盖一次
  [ok] 每个专家的对齐区间都盖住了它的紧凑区间（padding_M=128）
  [ok] 复算了 5 个 block 的 m_start / actual_rows 公式，没有越界
```

注意源码里对齐用的是 `ceil((count + 1) / padding_M) * padding_M`，比
`ceil(count / padding_M)` 多留了一行——读别人的实现时这类 `+1` 要单独确认。

**运行步骤**：

* 前提：Python 3.9+，仅标准库（复算脚本）；官方示例需要 NVIDIA GPU。
* 命令：`python3 code/07-tilelang-moe-and-quant/moe_dispatch_contract.py`；
  有卡时再跑
  `python3 _vendor/tilelang-v0.1.15/examples/fusedmoe/example_fusedmoe_tilelang.py`。
* 期望输出：复算脚本打印四张表与三项 `[ok]`；官方示例打印
  `Tilelang and Torch match`，并用 `do_bench(..., backend="cupti")` 报
  shared 与 routed 两个 kernel 的延迟。
* 验证级别：复算脚本是 **CPU 逻辑验证**（实际运行过）；
  官方示例是 **仅静态核对**（未运行）。

### 4.4 量化 GEMM 的接入方式

decode 是 memory-bound，所以「让权重少占字节」直接换来速度。三条常见路线：

| 路线 | 量化对象 | 典型格式 | 在 TileLang 里的落点 |
| --- | --- | --- | --- |
| weight-only | 只量化权重 | W4A16 / W8A16 | 解包权重后 `T.gemm` 用高精度激活 |
| weight + activation | 权重与激活都量化 | W8A8 / W4A8 | 两侧各自反量化，int32 累加 |
| block-scaled | 带块的 scale | MXFP8 / MXFP4 | `T.gemm_blockscaled(..., SFA, SFB, k_start, sf_a_granularity_k, sf_b_granularity_k)` |

第二种的完整写法在
`_vendor/tilelang-v0.1.15/examples/dequantize_gemm/example_dequant_gemm_w4a8.py`，
核心只有几行（`:135-143`，变量名与源文件一致）：

```python
T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
T.copy(B_shared, B_local)
for i, j in T.Parallel(block_N, block_K):
    B_dequantize_local[i, j] = _tir_u8_to_i4_to_i8(
        num_bits, B_local[i, j // num_elems_per_byte], j % num_elems_per_byte,
        dtype=in_dtype)
```

也就是：**把打包的 int4 当 uint8 搬进片上，在 fragment 里解包并做符号扩展，
再喂给 `T.gemm`**。官方目录
`examples/dequantize_gemm/README.md` 把这一类统称 Dequantization GEMM，
并指向 BitBLAS 做更激进的 layout 变换。

接入推理框架时，量化 kernel 的真正约束不在 kernel 里，而在**元数据**：

* 权重必须是**框架约定好的打包顺序**（哪个半字节是哪个元素、scale 的粒度）；
* scale / zero-point 的形状与 dtype 必须一致（block-scaled 的 scale 在
  v0.1.15 里可以是 `uint32` 打包的，见第 14 章 S26）；
* 量化后的权重需要重新反量化成推理框架能识别的 checkpoint 格式，
  或者反过来——推理框架要能接受「未反量化」的权重。

**运行步骤**：

* 前提：NVIDIA GPU（W4A8）或 sm100（block-scaled）+ TileLang v0.1.15。
* 命令：`cd _vendor/tilelang-v0.1.15/examples/dequantize_gemm && python3 example_dequant_gemm_w4a8.py`。
* 期望输出：`assert_allclose` 通过；性能口径是 `2·M·N·K / latency`，
  epilogue 不计入分子。
* 验证级别：**仅静态核对**（本机未运行）。本章骨架
  `code/07-tilelang-moe-and-quant/skeletons.py` 同样是仅静态核对。

---

## 5. 与推理框架集成

### 5.1 vLLM：它承诺了哪些东西是可替换的

vLLM 的文档首页把它的能力列成了一张清单
（`../tilelang-corpus/13-e2e-training-and-serving/raw/vllm-docs.md`）。
对本章有用的只有六条，逐条抄下来：

* 用 PagedAttention 管理 attention 的 key/value 显存；
* continuous batching + chunked prefill + prefix caching；
* 「Fast and flexible model execution with piecewise and full CUDA/HIP graphs」；
* 量化支持 FP8、MXFP8/MXFP4、NVFP4、INT8、INT4、GPTQ/AWQ、GGUF、
  compressed-tensors、ModelOpt、TorchAO 等；
* 「Optimized attention kernels including FlashAttention, FlashInfer,
  TRTLLM-GEN, FlashMLA, and Triton」；
* 「Optimized GEMM/MoE kernels for various precisions using CUTLASS,
  TRTLLM-GEN, CuTeDSL」。

把 TileLang kernel 塞进去的**机会点**就在这两条 kernel 清单里：attention
（decode）与 GEMM/MoE（量化）各有多个可替换实现，而框架已经把「用哪个
实现」做成了后端选择；KV 的分配与 paging 则已经被 PagedAttention 接管，
自定义 kernel 必须按它的块布局与 `block_table` 语义来写。

corpus 里的 vLLM 资料**只有文档首页**（见 `references.md` 的「待补资料」），
所以本章不给「怎么注册一个自定义算子」的具体接口名——那是版本相关的细节，
写出来就要随 vLLM 版本漂移。这里只给可以自查的办法：

* 在装好 vLLM 的环境里用 `python -c "import vllm; print(vllm.__file__)"`
  找到安装目录，再去读它选择 attention backend / GEMM backend 的那几个模块；
* 看框架对自定义算子的要求：张量形状、`cu_seqlens` / `block_table`、
  KV cache 的 dtype 和 layout、以及是否要求在图模式（CUDA graph）下可捕获。

### 5.2 TileRT：把延迟做到极致的另一种形态

TileRT 的定位和 vLLM 明显不同：它公开宣称优先优化**单用户延迟**而不是
吞吐（`../tilelang-corpus/12-operator-libraries/raw/github-tile-ai-tilert.md`）。
它对本章的价值在于给出了「TileLang 系技术进入生产服务」的一条完整形态：

| 事实 | 出处 |
| --- | --- |
| 定位是「Tile-Based Runtime for Ultra-Low-Latency LLM Inference」，目标是把 TPOT 压到毫秒级 | README 概述段 |
| 做法是「把 LLM 算子拆成细粒度 tile 级任务，运行时在多个设备间动态重排计算、I/O、通信」 | 同段 |
| 「the underlying compiler techniques will be gradually shared with the community as they are integrated into TileLang and TileScale」 | 同段 |
| v0.1.6 的 wheel 里**一个进程只能加载一个 backend 库**（`libtilert_dsv32.so` / `libtilert_glm5.so` / `libtilert_glm52_rocm.so`），`tilert.load_backend()` 还会校验 PyTorch 的版本与 flavor | README「Installation」与 `[!NOTE]` |
| 支持以 **vLLM prefill + TileRT decode** 的 PD 分离：connector 通过 vLLM V1 的 `kv_connector_module_path` 加载，不需要 fork vLLM | README「Disaggregated Serving」 |
| **两端 KV cache dtype 必须一致**（示例里 vLLM 侧 `fp8_ds_mla`、TileRT 侧 `fp8`），不一致会在握手时被拒 | 同节 Prerequisites |
| decode 引擎一次只服务一条序列，router 为每个请求独占一个节点，忙时返回 429；`--queue-timeout` 可以让请求排队 | 同节末段 |
| 支持 MTP（multi-token prediction），会打印 `Accepted length: mean=..., min=..., max=...` | README「MTP」 |

性能数字（**厂商自报，作者未复现，条件是各自博客/README 里写的硬件与形状**）：
GLM-5.3 在 8×AMD MI355X 上单用户 469 tok/s（AgentX 榜首位）、
MiMo-V2.5-Pro-UltraSpeed 在单节点 8 卡上超过 1000 tok/s（1T 参数模型）、
v0.1.3 在 GLM-5-FP8 上最高 500 tok/s、DeepSeek-V3.2 上最高 600 tok/s。
**这些数字与本章的模型账本没有可比性**，引用它们的唯一目的是说明
「超低延迟这条路线在工程上确实存在，而且它的公开实现建立在同一套 tile 抽象上」。

对本章最有指导意义的是最后一条结构信息：**PD 分离时，prefill 与 decode 可以
在两套完全不同的引擎上跑，只要 KV cache 的 dtype 与布局对齐**。
这正好对应本章 2.1 节的结论——两个阶段的瓶颈不同，所以在系统层面被拆开。

### 5.3 vllm-ascend：昇腾上的服务形态

昇腾侧的官方路线是 **vllm-ascend 硬件插件**
（`../tilelang-corpus/13-e2e-training-and-serving/raw/github-vllm-ascend.md`）：

| 事实 | 出处 |
| --- | --- |
| 定位是 vLLM 的**社区维护的硬件插件**，遵循 vLLM 的 hardware-pluggable RFC | README「Overview」 |
| 硬件前提：Atlas 800I A2 / A2 训练系列 / 800I A3 / A3 训练系列（300I Duo 为实验性） | README「Prerequisites」 |
| 软件前提（快照日 2026-09-30）：Python ≥3.10 且 <3.13、**CANN == 9.1.0**、**PyTorch == 2.10.0**、**TorchNPU == 2.10.0.post4**、vLLM 与 vllm-ascend 同版本 | 同节 |
| 分支策略：`main` 跟随 vLLM main；`releases/vX.Y.Z` 跟随对应 vLLM 版本 | README「Branch」 |
| 快照里标注的 latest stable 是 v0.23.0，另有 v0.26.0rc1 作为发布候选 | README 的 News 与 Getting Started 表 |

对「用 TileLang 写算子」的人来说，这个仓库的意义是：**它定义了昇腾上
算子必须满足的接口与环境**。你的 kernel 要能跑在服务里，就得和它声明的
CANN / torch_npu / vLLM 版本对齐——版本不匹配是最常见的「本地跑得通、
进服务就崩」的原因。

### 5.4 自定义 kernel 进入服务框架的接口与约束

把 5.1–5.3 合并，四类必须对齐的东西（图
`figures/05-kernel-to-framework.md`）：

| 层次 | 要对齐什么 | 对不齐的后果 |
| --- | --- | --- |
| 形状与元数据契约 | `cu_seqlens`（varlen）、`block_table`（paged）、`max_seqlen` 这类编译期常量、scale 的粒度 | kernel 编不出来，或每步都要重编译 |
| dtype 与 layout | 权重打包顺序、KV cache dtype（fp8/bf16/int8）、NZ/ND 布局、padding 规则 | 数值错、或性能断崖（多一次转换） |
| stream 与图模式 | kernel 必须能落在框架当前的 stream 上；能进 CUDA graph / ACLGraph 的 kernel 必须没有 host 侧同步 | 每步一次隐式同步；入图失败只能退回 eager |
| 调度接口 | batch 是怎么组的、KV 块怎么给、prefill 与 decode 是否混步 | 只在小 batch 下成立的优化，进服务后反向变慢 |

**运行步骤**（检查环境与版本，不需要 GPU 也能做前两条）：

* 前提：已安装对应框架的 Python 环境。
* 命令：`python -c "import vllm, torch; print(vllm.__version__, torch.__version__)"`；
  昇腾侧再加 `python -c "import torch, torch_npu; print(torch.npu.is_available())"`。
* 期望输出：版本号与 5.3 节表格里的要求一致；昇腾侧必须打印 `True`。
* 验证级别：**未核验**（作者本机没有安装 vLLM / torch_npu，命令与判据来自
  上述 corpus 文档；读者需自行核对版本号）。

---

## 6. 昇腾部署：torch_npu、ACLGraph 与 msProf

### 6.1 torch_npu：让 TileLang kernel 和 torch 算子同流

在昇腾上，PyTorch 通过 `torch_npu` 访问 NPU。TileLang 的昇腾后端用**一条
DLPack 交换层的补丁**把自己接进这条流：

> Patches Torch's DLPack Exchange API table so TVM-FFI submits Ascend kernels
> to Torch's current NPU stream.
> —— `_vendor/tilelang-v0.1.15/tilelang/ascend/torch_npu_exchange.pyx:1-6`

实现方式是替换 `current_work_stream` 回调：当设备类型是 NPU 时返回
`torch_npu` 当前 stream 的句柄，其它设备类型转回原来的实现
（同文件 `:44-56`）。

**这件事为什么重要。** 如果 TileLang 在自己的 stream 上跑，那么「torch 前置
算子 → TileLang kernel → torch 后置算子」之间就要显式同步；decode 每步几十个
算子，每次同步都是一次 host 往返。`.pyx:1-6` 那两行注释，是本章「stream 与
图模式」那条约束在昇腾上的具体形态。

### 6.2 ACLGraph：capture 一次，replay 多次

ACLGraph 是昇腾上的图模式（NPU 侧的 CUDA Graph）。适配器文档第 7 章的说法是：

> aclGraph 通过将 eager 模式任务下发即执行区分为下发 capture + 执行 replay
> 两阶段，通过采用 1 次 capture + 多次 replay 的执行模式减小交互开销，
> 对 HostBound 场景做运行时加速，不承载图编译和图优化功能。
> —— `../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/docs/TileLang-Ascend Programming Guide.md` 第 7.1.1 节

用法只有三行（`examples/aclgraph/README.md` 与文档第 7.1.2 节）：

```python
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    q = tilelang_rms_norm(q, variance_epsilon)
    q = tilelang_apply_rope(q, sin, cos)
g.replay()
```

三个必须知道的约束：

1. **capture 阶段只是记录，不执行**；真正的执行在 `g.replay()`。
2. **入图要求形状静态。** 这与 2.7 节的「`max_seqlen` 是编译期常量」是同一件
   事的两面。昇腾侧的工程实践也印证了这一点：vLLM-Ascend 的图路径只覆盖
   `decode_only`，prefill 仍走 eager，因为「prefill tensor shapes change often,
   but capturing and replaying graphs prefer fixed static shapes」，动态形状
   要靠 padding / bucketing 之类的变通
   （`../tilelang-corpus/06-huawei-ascend/raw/megagdn-pto/blog/mega_gdn_en.md`
   「Host overhead in eager PyTorch」一节）。
3. **图模式不改变语义**，只把 host 侧下发摊掉。所以验证方式是「eager 与
   graph replay 的结果一致」。

`code/08-ascend-aclgraph/aclgraph_decode_block.py` 是本章的骨架，它用的是
**适配器仓库**的 API（`T.Kernel(..., is_npu=True)`、
`tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC` 等）。这一点必须说清楚：

> 主干 v0.1.15 的昇腾方言**不是**这套 API。v0.1.15 里 kernel 写成
> `with T.Kernel(N_CORES) as core_id`（没有 `is_npu` 参数），auto-schedule 由
> pass context 开关控制（`_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:22`
> 的 `allow_autoschedule`），而 `TL_ASCEND_AUTO_SYNC` 这类键在 v0.1.15 的
> `tilelang/transform/pass_config.py` 里检索不到。照抄本章骨架要在适配器仓库
> 上跑；想在 v0.1.15 的昇腾方言上写，请看
> `_vendor/tilelang-v0.1.15/examples/ascend/example_rmsnorm.py:40`。

（这条差异是逐个文件核对过的：适配器的
`_vendor/tilelang-ascend-v0.1.2.000/tilelang/transform/pass_config.py:38-50`
有这四个键，主干的同名文件里没有。）

### 6.3 msProf：在昇腾上验证

适配器文档第 6 章给了两条采集路径：

| 方式 | 适用 | 命令 |
| --- | --- | --- |
| `msprof op` | 上板，看真实运行时的内存与性能瓶颈 | `msprof op --kernel-name="your_kernel_func_name" python your_kernel_script.py` |
| `msprof op simulator` | 仿真，看指令流水与代码热点 | `msprof op simulator --soc-version=<ascend_version> --kernel-name="your_kernel_func_name" python your_kernel_script.py` |

文档列出的可视化工具有：计算内存热力图、Roofline 瓶颈分析图、Cache 热力图、
通算流水图、算子代码热点图；仿真模式还有指令流水图与内存通路吞吐率波形图。

锁定版本里还内置了一条昇腾计时路径：`tilelang.profiler.do_bench` 支持
`backend="msprof"` 与 `"msprof_detail"`
（`_vendor/tilelang-v0.1.15/tilelang/profiler/bench.py:39,260-263`），
并且它用一个昇腾专用的 cache-flush kernel 把 L2 冲掉
（`tilelang/profiler/msprof.py:1-38`，那里用了 `T.SimtVF` 与
`l2_cache_ctrl="WTS_FV"`）。**L2 flush 是判断带宽类结论是否可信的前提**，
见 7.2 节。

**运行步骤**（三件事，缺一不可）：

* 前提：昇腾 NPU + CANN + `torch_npu`，`torch.npu.is_available()` 为 `True`；
  适配器版本见 `_vendor/tilelang-ascend-v0.1.2.000/`。
* 命令：
  1. 正确性：`python3 code/08-ascend-aclgraph/aclgraph_decode_block.py`；
  2. 图模式确认：设 `ASCEND_HOST_LOG_FILE_NUM=1000`、
     `ASCEND_PROCESS_LOG_PATH=logs`、`ASCEND_GLOBAL_LOG_LEVEL=1`，
     再跑一次，然后在 host 日志里搜 `capture model`
     （值为 1 表示入图生效，0 为默认即未入图）；
  3. 性能：`msprof op --kernel-name="main_rms" python3 aclgraph_decode_block.py`。
* 期望输出：第 1 步打印 `Kernel Output Match!`（`torch.testing.assert_close`
  通过，`rtol=1e-2, atol=1e-2`，与 corpus 那份 ACLGraph 示例的收尾断言一致）；
  第 2 步日志里出现 `capture model`；第 3 步在 MindStudio Insight 里能看到
  该 kernel 的热力图与流水图。
* 验证级别：**仅静态核对 + 待读者验证**。作者本机没有 NPU，上述命令与判据
  全部来自锁定版本的文档与示例，**未在硬件上执行过**。

---

## 7. 基准方法：TTFT、TPOT 与吞吐

### 7.1 三个指标的定义

推理性能的争议多数源于口径不同。先把定义钉死（`code/05-bench-methodology/`）：

| 指标 | 全称 | 定义 | 受什么影响 |
| --- | --- | --- | --- |
| TTFT | time to first token | 从请求到达到第一个输出 token 生成 | prefill + 排队 + 调度 |
| TPOT | time per output token | `(E2E − TTFT) / (输出 token 数 − 1)` | decode 的每一步 |
| ITL | inter-token latency | 相邻两个输出 token 的间隔分布 | decode 抖动、抢占 |
| E2E | end-to-end latency | 从请求到达到最后一个 token | `TTFT + TPOT × (n−1)` |
| 吞吐 | tokens/s | 总输出 token 数 / 墙钟时间 | 算上所有并发请求 |
| 有效吞吐 | goodput | 只有满足 SLO 的请求计入 | 服务质量 |

同一组数（TTFT 420 ms、E2E 9000 ms、输出 512 token）用三种算法算 TPOT：

```text
(E2E − TTFT) / (n − 1) = 16.79 ms     ← 推荐
(E2E − TTFT) / n       = 16.76 ms
E2E / n                = 17.58 ms     ← 把 TTFT 摊进每 token，不能叫 TPOT
```

差异 0.2% 与 4.9%。**报 TPOT 必须写出公式、n 是否含 EOS、以及 TTFT 的起点
（请求到达 / 入队完成 / prefill 开始）。**

一条可以借鉴的实践：vLLM 内部用 `first_token_latency` 记录 TTFT
（`vllm/v1/metrics/stats.py` 的 `RequestStateStats`），并且报告的是
**中位数**而不是平均数
（`../tilelang-corpus/06-huawei-ascend/raw/megagdn-pto/blog/mega_gdn_en.md`
「Full-model results」一节，原文注明对应 vLLM v0.19.1）。

### 7.2 避免结论被缓存与预热污染

五类污染，每一类都能把数字改掉一个数量级：

**（1）首次编译 / autotune 落进测量窗口。** `code/05` 的演示：稳态 0.350 ms、
首次 3200 ms、测 100 次，均值从 0.350 ms 变成 32.350 ms——抬高 92.43 倍。
这就是 `do_bench` 默认要有 25 ms 预热的原因
（`_vendor/tilelang-v0.1.15/tilelang/profiler/bench.py:31-34`）。
但预热只能吃掉「一次性的」开销：autotune 的搜索、显存池首次增长、
graph capture 都必须先做一遍再开始计时。

**（2）KV / prefix cache 命中。** 第二次跑同一批 prompt 时 TTFT 会显著变低。
做法：换新 prompt、清 cache，或在报告里写明命中率。

**（3）L2 cache 命中。** 小 kernel 的数据整块留在 L2 里，带宽会虚高。
**做法：两次采样之间 flush L2。** TileLang 的 `do_bench` 默认 flush 256 MB
（`bench.py:42` 的 `cache_size: int = 256`），昇腾侧还有专用 flush kernel
（`tilelang/profiler/msprof.py`）。第 14 章的 benchmark 口径也要求这一条。

**（4）eager 与图模式混测。** 同一个算子，eager 有 host 下发开销、graph 没有。
两种模式必须分别计时并写清用的是哪种；对 host-bound 的小算子，这个差别
可以盖过 kernel 本身的差异。

**（5）样本太少。** 20 次采样报 p99 没有意义。要报 p99 就采够几百次；
否则只报 mean / median / min，并写清取数方式。

### 7.3 报告前必须写下来的清单

`code/05-bench-methodology/` 打印 12 条，这里压缩成一句话的版本：
**硬件、软件版本、模型与精度、并行策略、输入与输出长度分布、并发与到达率、
指标公式、聚合方式、预热与缓存状态、图模式、对照组的条件、可复现的脚本与随机种子。**

**运行步骤**：

* 前提：Python 3.9+，仅标准库。
* 命令：`python3 code/05-bench-methodology/bench_methodology.py`。
* 期望输出：三节内容——定义表、TPOT 三种算法的换算（16.79/16.76/17.58 ms）、
  污染演示（均值被抬高 92.43 倍）、四类假数字、12 条清单。
* 验证级别：**CPU 逻辑验证**（写作机上实际运行过）。

---

## 8. 进阶与陷阱

**（1）把 batch=1 的结论当成通用结论。** 3.1 节那张占比表在 batch=1 时
attention 只占 3.2%，但 batch=32、context=131072 时占 94.4%。
「decode 的瓶颈是 attention」和「decode 的瓶颈是权重」都对，取决于配置。
写结论时必须带配置。

**（2）忘了 KV 的写入成本。** attention 的账本里通常只算「读 K/V」，
但每个 decode step 还要往 KV cache 里**写**一个新 token 的 K、V。
在 batch 大、层数多时这是一笔可观的写流量；只有在 KV 长度很长时它才
相对可忽略。

**（3）把 `max_seqlen` / `group_sum` 这类编译期常量当成运行时参数。**
它们进 JIT 缓存键（第 7 章）。服务里换一个更长的 prompt 就触发重编译，
表现为「第一次请求特别慢」。真实框架的做法是按桶（bucket）编译，
或者干脆把形状固定住再 padding。

**（4）量化后忘记核对元数据的打包顺序。** 4.4 节的 `j // num_elems_per_byte`
与 `j % num_elems_per_byte` 决定了哪个半字节是哪个元素。打包方与解包方
不一致时，数值会「大致对但精度异常」，比崩溃更难查。

**（5）split-KV 的并行度与 combine 成本。** `num_splits` 太小喂不饱机器，
太大则 `Output_partial`（`batch × heads × num_splits × dim`）的写读流量
与多一次 kernel 启动开始占主导。官方把它放进 autotune 空间
（`example_gqa_decode.py:16-23` 的 `[1, 2, 4, 8]`）正是因为它没有解析最优解。

**（6）block-scaled 量化的 scale 是一等公民。** MXFP8/MXFP4 的 scale 不是
标量而是按块张量，粒度写错会让结果「量级对、细节错」。v0.1.15 的
`T.gemm_blockscaled(..., SFA, SFB, k_start, sf_a_granularity_k,
sf_b_granularity_k)` 把这几个参数显式曝光出来
（`_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:275`），
就是为了让这个契约写在代码里。

**（7）昇腾的两套方言不能混用。** 6.2 节已经写明：适配器用
`T.Kernel(..., is_npu=True)` + `TL_ASCEND_*` pass configs，
主干 v0.1.15 用 `tilelang.ascend.language` 的 `T.Kernel(N_CORES)` +
pass context。把两边的代码片段拼在一起是最常见的「照抄跑不通」。

**（8）PD 分离只解决了阶段不匹配，不解决一致性。** 5.2 节的 TileRT 例子
里，prefill 与 decode 两端 KV cache 的 dtype 必须一致（`fp8_ds_mla` ↔ `fp8`），
否则连接器握手直接失败。把 prefill/decode 拆到两套引擎上，
KV 的 dtype、布局、块大小就变成了**跨进程的接口**。

---

## 9. 本章小结

把这一页当作复习卡片。

**两个阶段的差别是一根算术强度的轴。** prefill 的 M 是 prompt 长度（几千），
decode 的 M 是 batch（个位数到几百）；`M ≪ N, K` 时 `AI ≈ 2M/b`，
翻转点 `M* ≈ balance·b/2 ≈ 375`（bf16、H100 公开规格）。所以 prefill 优化
张量核利用率与中间结果搬运，decode 优化访存与批量。

**decode 的第一个硬约束是权重。** Qwen2.5-7B 的 7.61B 参数在 bf16 下是
15.23 GB，3 TB/s 上读一遍 5.08 ms，单请求上界约 197 tok/s。突破它的办法只有
三个：加大 batch、量化、多卡切分。

**KV cache 的账本可以手算。** 每 token 每层 `2·H_kv·d·b`；Qwen2.5-7B 是
2.0 KiB / 层、56.0 KiB / token；context=131072 时一条序列 7.00 GiB，
而 MHA 版本要 49.00 GiB。GQA 省 7 倍显存，代价是 decode attention 的
`AI = 2H_q/(H_kv·b) = 7`，只有 MHA 的 1/7。

**paged KV 换来的是显存效率，付出的是间接寻址。** 地址从 `k·block_N` 变成
查 `block_table`；TMA 的静态描述符不再直接适用。

**continuous batching 决定吞吐。** 因为一个 decode step 的耗时基本不随 batch
变化，空槽位就是纯浪费。代价是 kernel 必须接受 varlen（`cu_seqlens`）、
paged KV、以及 prefill/decode 混步。

**三个推理 kernel 都有官方落点。** decode attention 用 split + combine
（`examples/flash_decoding/`）；GEMV 有五级阶梯，从 0.166 ms 到 0.00675 ms
（官方文档自报，Hopper，shape 未写明）；fused MoE 的关键是 host 侧四张表与
kernel 侧那一行 `m_start` 公式（`examples/fusedmoe/`）；量化 GEMM 的关键是
「在 fragment 里解包再 `T.gemm`」（`examples/dequantize_gemm/`）。

**进服务要过四层接口。** 形状与元数据契约、dtype 与 layout、stream 与图模式、
调度接口。vLLM 提供可替换的 attention / GEMM / MoE 后端与 PagedAttention；
TileRT 走超低延迟 + PD 分离（KV dtype 两端必须一致）；
vllm-ascend 定义了昇腾上的环境与插件形态。

**昇腾上先解决 host-bound。** `torch_npu` 的 DLPack 补丁让 kernel 落在 torch
当前的 NPU stream 上；ACLGraph 用 capture + replay 摊掉下发开销，
代价是**形状必须静态**——这正是 vLLM-Ascend 只对 decode 入图的原因。
验证用 msProf（`msprof op` / `msprof op simulator`）。

**报数字前先固定口径。** TTFT、TPOT、吞吐的定义、预热、L2 flush、
prefix cache 状态、图模式、样本量，缺一条数字就不可比。

## 延伸阅读

下面每条都指向本章 `references.md` 里的条目 id。

* **模型与规格**：`R1`（Qwen2.5-7B 模型卡，本章所有模型形状的来源）、
  `R2`（H100 的公开算力与带宽，用于平衡点与权重底线）。
* **服务框架**：`R4`（vLLM 文档首页）、`R5`（TileRT README，含 PD 分离与
  MTP 的命令）、`R6`（vllm-ascend README，环境与分支策略）。
* **昇腾实践**：`R7`（GDN 内核博客，含 ACLGraph 限制与 TTFT 口径）、
  `R9`（适配器编程指南第 6、7 章：msProf 与 aclgraph）、
  `R10`（`examples/aclgraph/` 的完整示例与入图确认方法）。
* **官方示例（本章的主角）**：`S1`（`flash_decoding/example_mha_inference.py`）、
  `S2`（`example_gqa_decode.py`）、`S3`（`example_gqa_decode_varlen_logits.py`）、
  `S4`（`fusedmoe/example_fusedmoe_tilelang.py`）、
  `S5`/`S6`（`dequantize_gemm/`，含 README 的指路）、
  `S7`（`gemv/example_gemv.py`）、`S8`（varlen 的右对齐与 guard）。
* **API 与工具出处**：`S9`（`T.gemm_blockscaled` 的签名）、
  `S11`（`do_bench` 的默认值与 L2 flush）、`S12`（昇腾 msprof 计时后端）、
  `S13`（昇腾的 DLPack 交换补丁）、`S14`（主干昇腾方言的写法对照）。
* **文档**：`R8`（官方 GEMV 教程，含那张五级阶梯的耗时表；注意它的 shape
  没有写明，官方 example 默认 `N = K = 1024`）。
* **代码**：`code/README.md` 里有九个示例的验证级别总表；
  `code/01`–`code/05`、`code/07` 的复算脚本是本机可运行的
  （CPU 逻辑验证），`code/06`/`code/08` 是仅静态核对的骨架。
