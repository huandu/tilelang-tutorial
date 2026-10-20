# 第 17 章 端到端 I：用 HuggingFace 资产从零训练一个 7B 模型

## 学习目标

读完本章，你应当能够：

1. 用一套“账本”方法估算：训练一个 7B 模型在给定硬件上需要多少显存、多少算力，
   并能说清每一步估算的假设与它会在哪里失真。
2. 分辨“从零训练”在实际语境下的两条路线，**预训练小规模复现**与**从开源权重
   继续训练**，并说出它们在成本、风险与可验证性上的取舍。
3. 搭出一条可用的数据管线：HF datasets 流式读取、去重、带缓冲的 shuffle、
   tokenize 与 packing，并解释它如何影响训练吞吐。
4. 读懂 Qwen2.5-7B 的结构参数（层数、hidden、GQA、SwiGLU、RMSNorm），并由此判断
   哪些算子是热点、哪些值得换成 TileLang。
5. 写清一次训练循环里每一个部件（optimizer、混合精度、梯度累积、checkpoint 与恢复、
   评测）的职责，并知道哪些部分**不属于** TileLang。
6. 把 TileLang kernel 接进 PyTorch：用 `tilelang.jit` 的 torch 适配、保证数值一致、
   并解释 autograd 该如何补上。
7. 把同一套流程迁到昇腾（`torch_npu`），并说出昇腾上训练的现实约束。
8. 给出一条从 0.1M 到 7B 的**逐级放大**路径，每一步都有可检查的验收标准。

## 前置知识

* 第 2 章的 roofline 与算术强度：本章的算力账本直接复用它的口径。
* 第 13–16 章的算子实战：本章只讲“换哪个、怎么接、怎么验”，不重复 kernel 写法。
* 会写 PyTorch 训练循环（`forward` / `loss` / `backward` / `step`）。
* 需要 `HuggingFace` 生态的基本概念：`datasets`、`transformers`、`accelerate`、
  `peft`；不熟悉也没关系，本章会指出其中实际用到的部分。

本章的定位是全书两个“大目标”章中的第一个（第二个是第 18 章的推理与部署）。它**不是**
一篇 7B 预训练配方，而是一份**可退可进的工程计划**：先告诉你成本有多高、哪些假设必须
实测，再给你一条从小到大的路径，让每一步都被验证过，再往上放大。

本章有一条贯穿始终的边界，请先记住：

> **TileLang 的职责只有一件事，把模型前向里那几个热点算子写成 tile kernel。
> 数据、分布式、优化器、checkpoint、评测都不归它管。**

任何把 TileLang 说成“训练框架”的说法都是误导。它在训练里扮演的角色，等价于你把
某几行 `torch.nn.functional.*` 换成一个自研算子；剩下的训练骨架原封不动。

---

## 1. 问题的由来：“从零训练一个 7B 模型”这句话哪里是真的

先拆掉两个常见的误解。

**误解一：能在一台机器上训练 7B。** 这是错的。下一节的账本会给出硬数字：Qwen2.5-7B
的权重、梯度、Adam 状态、fp32 主权重加起来约 **113 GiB**，这还不算激活值。单张
80GB 卡连“摆下参数和优化器状态”都做不到，更不用说激活值、通信缓冲和显存碎片。

**误解二：从零训练 7B 就等于复现一个 Qwen。** 也不是。公开的 7B 模型是在数万亿 token
上、用成千上万张卡训练的。本教材的“从零训练”指的是下面这件事：

> 用一套你能完全看懂、完全掌控的代码，在一个**按比例缩小**的规模上，把预训练这条
> 链路完整地跑一遍，数据、tokenize、模型、训练循环、checkpoint、评测、以及把热点
> 算子换成自己的 kernel，并且每一步都验证过它是对的。

这件事的价值不在于训出一个能打的模型，而在于**你不再是黑盒的使用者**。当你在别的
地方遇到一个训练不动、loss 不下降、显存爆掉的问题时，你知道应该从哪里查起。

所以本章的目标分两层：**理解层**（账本、路线、数据、循环）与**动手层**（把算子换掉、
把路径跑通）。两层共用同一套压缩到单机的验证阶梯。

---

## 2. 可行性核算：显存与算力两本账

这一节复用第 2 章的思路：**先算清楚，再动手**。所有数字都可以用本章的
`code/01-plan-ledger/train_plan.py` 复现。它只做算术，不做任何对硬件的断言。

### 2.1 显存账本：一个参数要存几份

在 Adam + 混合精度的标准配置下，**每一个参数在训练态下要占好几份存储**：

| 分量 | 精度 | 每个参数的字节数 | 作用 |
| --- | --- | --- | --- |
| 权重 | bf16 | 2 | 前向/反向实际参与计算的那份 |
| 梯度 | bf16 | 2 | 反向算出来的梯度 |
| Adam 一阶矩 m | fp32 | 4 | 优化器状态 |
| Adam 二阶矩 v | fp32 | 4 | 优化器状态 |
| fp32 主权重 | fp32 | 4（可选） | 参数更新的高精度副本 |

把 Qwen2.5-7B 的 **7.61B** 参数（模型卡给出的含嵌入总参数量）代进去，得到
`train_plan.py` 的真实输出：

```
== 显存账本：Qwen2.5-7B（7.61B 参数）==
  权重 (bf16)           14.17 GiB
  梯度 (bf16)           14.17 GiB
  Adam m (fp32)       28.35 GiB
  Adam v (fp32)       28.35 GiB
  fp32 主权重            28.35 GiB
  合计                 113.40 GiB
  按 ZeRO-3/FSDP 均摊（优化器/权重/梯度）：1 卡→113.40 GiB  8 卡→ 14.17 GiB  16 卡→  7.09 GiB
  （不含激活值、通信缓冲、CUDA context 与显存碎片；见正文 §2.2）
```

两个立刻能得出的结论：

* **优化器状态才是大头。** 权重和梯度加起来 28 GiB，只占四分之一；Adam 的 m、v 加上
  fp32 主权重合计 85 GiB。所以“训练比推理吃显存”的根源在优化器，不在模型本身。
* **分片是唯一出路。** 8 路 ZeRO-3/FSDP 之后，每卡只剩约 14 GiB 的参数相关开销，
  batch 才真正放得下。这也是 torchtitan 把 FSDP2、张量并行、流水并行放在第一位的原因
  （见 `../tilelang-corpus/13-e2e-training-and-serving/raw/github-pytorch-torchtitan.md`）。

### 2.2 激活值：那个必须实测、不能只靠估算的项

上表刻意没有算激活值。原因很实在：激活值的大小取决于 `batch × sequence_length`、
是否做 activation checkpointing（用重算换显存）、以及每一层里哪些中间结果被保留。
一个**教学用的简化模型**是：

```
激活值字节数 ≈ 系数 × micro_batch × seq_len × hidden × 层数 × dtype_bytes
```

其中“系数”随实现变化一个数量级，开了 activation checkpointing 可以小到个位数，
不开则可能是几十。**所以正确做法不是背这个公式，而是量它**：把 `micro_batch` 从 1
开始往上加，每加一档记一次 `torch.cuda.max_memory_allocated()`，直到接近上限。这条
“从 1 开始试探”的纪律，比任何估算都可靠。

### 2.3 算力账本：`C ≈ 6·N·D` 与 MFU

训练一个参数量为 `N` 的模型、处理 `D` 个 token，需要的浮点运算量常用
**`C ≈ 6·N·D`** 估算。这个 6 从哪来：

* **前向**：每个参数、每个 token 约 2 次浮点运算（一次乘、一次加，矩阵乘的基本单位
  是“乘加”）。
* **反向**：需要分别对权重和对输入求梯度，约为前向的 2 倍，即 4 次。
* 合计：`2 + 4 = 6`。

它的**适用边界**必须记住：这个式子忽略了 attention 里随序列长度二次增长的项，只有在
“序列长度远小于训练 token 数”时才准。上下文越长、越需要单独核算 attention 的开销
（见第 15 章）。

把 Qwen2.5-7B 的 `N = 7.61e9` 代入，训练 `D = 10e9` 个 token：

```
  C ≈ 6·N·D = 4.566e+20 FLOPs
    1 卡 ×   120 TFLOP/s 有效吞吐 →     1056.9 卡·小时，墙钟  1056.94 小时
    8 卡 ×   120 TFLOP/s 有效吞吐 →     1056.9 卡·小时，墙钟   132.12 小时
   64 卡 ×   120 TFLOP/s 有效吞吐 →     1056.9 卡·小时，墙钟    16.51 小时
```

**这张表里的 `120 TFLOP/s` 是占位假设，不是测量结论。** 它代表“每张卡的有效吞吐”，
等于硬件峰值乘以 **MFU（Model FLOPs Utilization，模型浮点利用率）**。你要做的第一件事，
就是在自己的机器上用第 12 章的方法实测这个值，然后重新跑这条命令：

```bash
python3 code/01-plan-ledger/train_plan.py --tokens 10e9 --eff-tflops <你的实测值>
```

对照 nanoGPT 的公开参照：GPT-2（124M）在 8×A100 40GB 上训练 4 天、val loss 降到
约 2.85（见 `.../raw/github-nanogpt.md`）。把它放大到 7B，参数多了 60 倍、数据需求还要
再翻数倍。这就是“单机做不了 7B”的直观来源。

### 2.4 单机与集群的分界线

把两本账合起来看，分界线很清楚：

| 规模 | 权重+梯度+优化器（未分片） | 单卡可行性 |
| --- | --- | --- |
| 0.1M（冒烟） | < 1 MiB | 随便跑，CPU 都行 |
| 124M（GPT-2 级） | ≈ 1.8 GiB | 单卡轻松 |
| 0.5B | ≈ 7.4 GiB | 单卡可行（配合 checkpointing） |
| 1.5B | ≈ 22 GiB | 单卡勉强，视激活值而定 |
| 7B | ≈ 113 GiB | **单卡不可行**，需 8 卡以上分片 |

（表中数值可用 `train_plan.py` 把参数量换掉后复现。）

---

## 3. 两条路线：小规模预训练复现 vs 从开源权重继续训练

知道成本之后，路线选择就变成一个明确的工程决策。本章的图 1（`figures/01-two-routes.md`）
画了两条路，这里说清取舍。

**路线 A · 小规模预训练复现（random init → 训练到底）**

* 做法：从随机初始化开始，用公开语料（如 FineWeb-Edu 的子集）训练到 loss 收敛。
* 优点：整条链路你都能解释；能真正观察 loss 从高到低的过程；适合验证数据管线与
  kernel 的正确性。
* 代价：想看到下游指标变化，token 数至少要到百亿量级（见 §2.3 的账本），成本高；
  而且你训出来的模型几乎没有实用能力。
* 适合：教学验证、kernel 数值对齐、研究小规模现象。

**路线 B · 从开源权重继续训练（continue pretraining / 领域适配）**

* 做法：加载 Qwen2.5-7B 的权重，在领域语料上继续训练，通常只跑十亿级 token。
* 优点：成本低得多（初始化已经很好）；能较快看到领域效果；配合 PEFT/LoRA 还能把显存
  需求再压一个数量级（见 `.../raw/hf-peft-docs.md`）。
* 代价：初始化不是你自己训的，前向里有些行为你无法完全解释；领域外能力可能退化
  （灾难性遗忘）。
* 适合：快速见效、把训练流程接到自己的数据上。

两条路线在代码上**共用同一个训练循环**（§6）与同一组算子替换点（§7）。所以本章的做法
是：先用路线 A 在小模型上把链路跑通并验证，再用路线 B 在 7B 上做一次真实（但小规模）的
继续训练。

---

## 4. 数据管线：从 HF datasets 流式到 packing

数据管线不是“训练前的准备”，它就是训练循环的一部分。它供给得够快，GPU 才不空转。

### 4.1 流式读取：`streaming=True`

完整版 FineWeb-Edu 的文件大小是 **5.84 TB**、约 **35 亿行**（见
`.../raw/hf-fineweb-edu-card.md`）。你不可能先把它下载到本地再训练。`datasets` 的流式
模式（`load_dataset(..., streaming=True)`）让迭代器每次只拉取一个分片，内存占用与语料
总量无关（见 `.../raw/hf-datasets-docs.md`）。教学上更小的选择是 **TinyStories**：
约 **2.1M 行、7.62 GB** 的合成短故事，可以在单卡上快速跑完
（见 `.../raw/hf-tinystories-card.md`）。样本版本还有 `sample-10BT` / `100BT` / `350BT`。

### 4.2 去重与 shuffle

* **去重**：网页语料里有大量重复。本章示例用内容哈希做**精确去重**，成本低、无副作用。（近似去重属于数据工程更大的一块，本章不展开。）
* **shuffle**：流式数据不能做全局 shuffle。那要求全部读进内存。标准做法是维护一个
  固定容量的**缓冲**：每次随机取一篇输出、再补一篇进来。固定 `seed` 保证可复现，
  也是断点续训能重放位置的前提。

### 4.3 tokenize 与 packing

tokenize 之后，文档长度参差不齐。两种处理方式：

* **padding**：补齐到统一长度。简单，但填充的 token 不产生有效梯度，白烧算力，还会
  污染 attention 的统计。
* **packing**：把多篇文档首尾相连（文档之间插一个 EOS），再切成定长块。**没有填充**，
  算力全用在真实 token 上。代价是同一块里会跨文档，需要 EOS 告诉模型“这里断开了”。

本章 `code/02-data-pipeline/pack_tokens.py` 实现了 packing，并且能在没有网络、没有
GPU 的机器上跑通（内置语料 + 字节级 tokenizer）。它的真实输出、以及如何验证 packing
的边界行为（尾块被丢弃、EOS 出现在块中间），都写在 `code/02-data-pipeline/README.md`
里。图 3（`figures/03-data-pipeline.md`）画出了整条链路与它对吞吐的影响。

### 4.4 数据与吞吐的关系

回到 §2.3 的算力账本：那个估算假设 GPU 一直在算。如果数据管线供给不够快，GPU 会周期性
空转，实际吞吐就达不到账本上的数字。典型症状是**训练日志里 GPU 利用率周期性掉到接近
0**。这时先查三件事：`DataLoader` 的 `num_workers`、tokenizer 是否单线程成了瓶颈、
以及 packing 是否在训练循环里同步做（应该提前做或并行做）。**不要**去调 kernel，
kernel 再快也救不了喂不进去的数据。

---

## 5. 模型与 tokenizer：Qwen2.5-7B 的结构如何决定算子选择

模型卡给出的关键结构参数（`.../raw/hf-qwen2-5-7b-card.md`）：

| 参数 | 值 | 对算子选择的影响 |
| --- | --- | --- |
| 总参数量 | 7.61B（非嵌入 6.53B） | 决定 GEMM 的规模与显存账本 |
| 层数 | 28 | 每一层都会重复一遍 RMSNorm/RoPE/attention/MLP，融合收益被放大 28 倍 |
| 注意力头 | GQA：Q 头 28、KV 头 4 | KV 投影比 Q 投影小得多；attention 的 KV 侧访存是瓶颈 |
| context | 131,072 tokens | 长上下文下 attention 的二次项不可忽略（§2.3 的修正项） |
| 结构 | RoPE + SwiGLU + RMSNorm + QKV bias | 这四类就是本节的“候选热点算子” |

> **口径说明（2026-10-01 更新）**：`hidden_size`、`intermediate_size` 与词表大小未在
> 模型卡的散文里给出，但本书资料库已存档官方 `config.json` 的原始文件与摘要
> （`../tilelang-corpus/13-e2e-training-and-serving/raw/hf-qwen2-5-7b-config.md`），
> 其中 `hidden=3584`、`intermediate=18944`、`vocab_size=152064`、`rope_theta=1e6`
> 与本章取值一致；本章账本可直接引用该文件，不再需要“待核”。
> **需以官方 `config.json` / `tokenizer_config.json` 为准**（见 `references.md` 的
> “待补资料 E1”）。这几个值只影响账本里的中间量，不影响本章的结论。

**为什么结构决定算子选择？**

* **RMSNorm 出现 2×28 = 56 次，RoPE 出现在每一层的 attention 内部。** 它们单个都不重，
  但调用次数多、访存密集，是“大批小算子融合”的经典目标。这正是第 13 章的主题。
* **GQA 把 KV 头压到 4 个。** 训练时 attention 的算术强度主要由 Q·Kᵀ 的 `seq_len²`
  决定；长上下文下它是 compute-bound，短上下文下更接近 memory-bound，具体判据见第 2 章
  的 roofline。
* **SwiGLU 是三个 GEMM。** Qwen2.5 的前馈网络里，`gate`/`up`/`down` 三个矩阵乘占了
  非嵌入参数的大头，属于第 14 章的 GEMM 家族。

tokenizer 方面：训练时用与权重**同一个** tokenizer，否则 token id 与 embedding 对不上，
loss 会明显偏高且训不动。tokenizer 的选择还会反过来影响 `D`，同样一份语料，词表越大，
切出来的 token 数越少。而 `D` 直接进 §2.3 的算力账本，所以**换 tokenizer 会改变训练
成本**，这不是细节。（Qwen2.5 的词表构成与精确大小以模型仓库的 `tokenizer_config.json`
为准，见上面的待核项。）

---

## 6. 训练循环

训练循环的骨架在所有参考实现里都一样（nanoGPT / litgpt / torchtitan 三者可以互相印证，
见 `.../raw/github-nanogpt.md`、`.../raw/github-litgpt.md`、
`.../raw/github-pytorch-torchtitan.md`）。本章的可运行版本是
`code/06-train-loop/train_step.py`，它把每个部件的一句话职责列在下面。

| 部件 | 职责 | 常见错法 |
| --- | --- | --- |
| 数据 | 产出 `input_ids`；`labels` 是它右移一位 | `logits[:, :-1]` 与 `labels[:, 1:]` 没对齐 |
| 前向 | 模型算 logits | 在 fp32 下跑整个大模型，白白吃显存 |
| loss | 交叉熵（语言建模） | 忘了 `reshape(-1, vocab)` |
| 反向 | `loss.backward()` | 累积期间多次 `step` 或提前清零 |
| 混合精度 | bf16 权重/激活 + fp32 累加 | 对 bf16 用了 fp16 才需要的 `GradScaler` |
| 梯度累积 | 小 batch 拼出等效大 batch | 裁剪放在累积完成之前 |
| 优化器 | AdamW + 学习率调度（warmup-stable-decay） | 没有 warmup，早期 loss 抖动大 |
| checkpoint | 定期保存权重+优化器+步数 | 只存权重，恢复后 loss 跳变 |
| 评测 | 固定验证集算 loss/perplexity | 评测集与训练集有重叠 |

关于**混合精度**：本章统一用 bf16 权重与激活、fp32 累加和优化器状态。bf16 的表示精度
比 fp16 低，但动态范围大，通常不需要梯度缩放（`GradScaler`）。torchtitan 还提供
bf16 优化器状态来省显存（见 R8），代价是更新精度下降。这是显存与精度之间的取舍，
不是死规矩。

关于**梯度累积**：它把小 `micro_batch` 拼成等效的大 `global_batch`，代价是每步多做
几次前向反向。正确性判据很简单，**用累积和用真大 batch 得到的梯度应当一致**（见习题
E10）。torchtitan 用 `--training.num_tokens_per_train_step` 表达这个等效关系，是同一
件事的另一种记法。

关于**checkpoint 与恢复**：至少保存权重、优化器状态、以及已完成的步数/数据位置。
只存权重的 checkpoint 在恢复后会让 loss 出现一次跳变，因为 Adam 的动量没了。本章示例
在 CPU 上验证过保存与恢复都能正常工作（输出见 `code/06-train-loop/README.md`）。

**里程碑 M1/M2 本项目实测**：在写作机器的 CPU 上跑
`python3 train_step.py --steps 100 --micro-batch 8 --grad-accum 2`，loss 从
前 10 步均值 **5.4314** 降到后 10 步均值 **3.8365**，且全程是有限值。这证明“能跑通
一步”和“loss 在下降”两个阶段目标在这套骨架上成立（规模是 0.09M 的冒烟模型）。

---

## 7. 把热点算子换成 TileLang

这是本章最接近书本主题的一节。图 4（`figures/04-hotspot-replacement.md`）先划清边界。

### 7.1 哪些算子值得换

按“换完之后能省多少、风险多大”排序：

| 算子 | 值不值得换 | 理由 |
| --- | --- | --- |
| **Attention（含 softmax）** | 最值得 | 融合 softmax 与两个矩阵乘，能显著减少 HBM 往返；是第 15 章的主战场 |
| **GEMM 家族（QKV/O/SwiGLU）** | 很值得 | 训练算力的主体；可对接厂商库（cuBLAS/CATLASS）或自研（第 14 章） |
| **RMSNorm** | 值得 | 调用次数多、访存密集，融合收益大（第 13 章） |
| **RoPE** | 值得 | 每层都跑，且是逐元素 + 少量三角运算，天然适合融合 |
| **MoE（dispatch/combine）** | 视模型而定 | 只有模型本身是 MoE 时才需要（第 16 章） |
| Embedding lookup | 不值得 | 纯查表，没有计算可优化 |
| Optimizer 步骤 | 不值得 | 逐元素、状态多，收益低、出错代价高 |
| 通信原语 | 不值得 | DDP/FSDP 的 all-gather/reduce-scatter 由框架负责 |

一个务实的顺序：**先换 attention 和 GEMM，再换 norm/RoPE，最后才考虑 fused
cross-entropy**。每换一个都要单独验收（§7.3）。

### 7.2 怎么接进 PyTorch

TileLang 在 v0.1.15 里对接 PyTorch 的方式，需要先把两个事实说清楚，因为很多二手资料
把它们搞混了：

**事实一：编译产物直接吃 torch 张量。** 用 `@tilelang.jit(out_idx=[...])` 编译出的
kernel，调用时可以直接传 `torch.Tensor`，返回的也是 `torch.Tensor`。机制在两处：

* `tilelang/jit/kernel.py:206-222`：`JITKernel.__call__` 转发到 `self.torch_function`；
* `tilelang/jit/adapter/cython/adapter.py:403-420`：`_process_buffer_device` 按 target
  把张量映射到 `torch.device("cuda")`（CUDA/ROCm）、`torch.device("npu")`（昇腾）、
  `torch.device("mps")`（Metal）。

`out_idx` 指定哪些参数是输出（支持负数下标，见
`tilelang/jit/adapter/base.py:27-53`），adapter 会替你在正确的设备上分配输出张量并返回。

**事实二：`tilelang/jit/adapter/torch/` 这个目录并不是通用的“torch 适配层”。**
在 v0.1.15 里它只有 `__init__.py`、`kernel_cache.py`、`metal.py` 三个文件，
`__init__.py` 只导出 `MetalKernelAdapter`（见
`tilelang/jit/adapter/torch/__init__.py:1-3`）。而且 `tilelang/jit/kernel.py:390-393`
明确写着：`execution_backend == "torch"` 时 `assert is_metal_target(target)`，也就是
说，名为 `"torch"` 的执行后端**只服务苹果 Metal**。

CUDA 与昇腾的默认执行后端是 `tvm_ffi`，可选 `cython`；顺序定义在
`tilelang/cuda/execution_backend.py:24-33` 与
`tilelang/ascend/execution_backend.py:1-21`。所以“TileLang 怎么接进 torch”的准确答案
是：**通过 tvm_ffi / cython 适配器直接收发 torch 张量**，而不是通过那个 `torch/` 目录。

一个最小的接入形态（完整版见 `code/03-tilelang-norm-torch/rmsnorm_tl.py`）：

```python
import tilelang
import tilelang.language as T

@tilelang.jit(out_idx=[2])          # 第 3 个参数 Y 是输出，由 adapter 分配
def _rmsnorm_fwd(M, N, eps, dtype="bfloat16"):
    @T.prim_func
    def main(X: T.Tensor((M, N), dtype),
             W: T.Tensor((N,), dtype),
             Y: T.Tensor((M, N), dtype)):
        ...
    return main

y = _rmsnorm_fwd(M=4096, N=3584, eps=1e-6)(x, w)   # x, w 是 torch.Tensor
```

### 7.3 数值一致：把“对不对”变成可执行的检查

换完算子必须有门限、可复现的检查。三条纪律：

1. **逐元素对齐**：同一随机输入，比较 kernel 输出与 PyTorch 参考，报最大绝对/相对误差。
2. **门限按 dtype 选**：bf16 只有 8 位尾数，本来就带千分级相对误差，**不能用 `1e-6`
   当门限**；fp32 才敢收到 `1e-5`。本章示例用 bf16 `rtol=atol=2e-2`、fp32 `1e-5`。
3. **端到端对齐**：在**同一批 token、同一份权重**下，比较“全 PyTorch 模型”与“替换了
   算子的模型”的 loss，相对差要小于千分之一量级。

`code/04-verify-numerics/compare_ref.py` 把这三条实现成一个脚本（写作机器无 GPU，
只做过静态核对，未运行）。

### 7.4 autograd：前向 kernel + 反向从哪里来

这是最关键、也最容易被忽略的一点：

> **TileLang 产出的是前向的 device 函数，它不产生 autograd 图。**

你可以自己验证：在 v0.1.15 的源码树里搜 `autograd` / `requires_grad` /
`no_grad`，`tilelang/` 目录下没有任何命中。也就是说，`loss.backward()` 穿不过一个
裸 TileLang kernel。要让它穿过去，只有一条路：把 kernel 包进
`torch.autograd.Function`。

反向有两种来源，按成本从低到高：

1. **用 PyTorch 算子写反向**（本章示例的做法）。前向用 TileLang，反向用几行 torch
   表达式。优点是快、稳；缺点是反向没享受到 kernel 的加速，而反向的计算量本来就比
   前向大（§2.3 的 4 : 2）。这适合**先验证正确性**。
2. **再写一个 TileLang 反向 kernel**。前向、反向都用自己的 kernel，训练时全程留在
   加速器上。这是生产做法，也是第 15 章 attention 反向示例的思路；代价是你要把
   `dx`/`dw` 的推导写对，且单独验收反向。

示例里 `RMSNormTileLang.forward` 调 kernel、`backward` 用 torch 公式算
`dx`/`dw`，这样 `loss.backward()` 就能穿过这个算子。你可以在习题 E7 里推一遍这个
反向公式，确认它是怎么来的。

---

## 8. 昇腾路线

昇腾上的目标是同一个，硬件与工具链不同。要点如下（细节见
`code/05-ascend-integration/`）。

**安装与前提**：CANN ≥ 8.3.RC1、torch-npu ≥ 2.6.0.RC1；官方在 **A2 / A3** 两款 NPU 上
验证过（`_vendor/tilelang-ascend-v0.1.2.000/README.md:52,79`）。

**两条接入路径**（来自官方示例 `examples/torch_tl_ascend/`）：

* **路径 A**：`tilelang.compile(..., target="ascend", out_idx=[...])` 得到的 kernel
  直接吃 `.npu()` 张量。adapter 会把 ascend target 映射到 `torch.device("npu")`
  （`tilelang/jit/adapter/cython/adapter.py:416`）。实测形态见
  `_vendor/tilelang-v0.1.15/examples/ascend/test_rmsnorm.py:15-16`。
* **路径 B**：按官方示例把算子编译成 `.so`，用一个 Python C 模块注册成
  `torch.ops.tl_ascend.flash_attention`，随后它就能出现在任何 `nn.Module` 里
  （`.../torch_tl_ascend/README.md`、`test_torch.py`）。

**昇腾上训练的四个现实约束**（务必先接受，再动手）：

1. **没有 autograd**。和 NV 侧一样，TileLang-Ascend 只给前向。反向要么自己写 kernel，
   要么用 PyTorch 补。这决定了你能不能真的在 NPU 上端到端训练。
2. **算子覆盖面**。官方已验证的算子族（GEMM / attention / normalization / FP8 等）
   覆盖了训练前向的大部分；但 optimizer 里那些零碎算子通常还是走 torch-npu 自带实现。
3. **同步模型不同**。昇腾后端会自动插入核内同步（`InsertSync`，见
   `_vendor/tilelang-ascend-v0.1.2.000/README.md:32`），不要在 kernel 里手写时序假设。
4. **写法差异**。昇腾 RMSNorm 用 `T.alloc_reducer` / `T.reducer_init` /
   `T.finalize_reducer`，与 NV 侧的 `T.reduce_sum` 写法不同
   （对比 `_vendor/tilelang-v0.1.15/examples/ascend/example_rmsnorm.py` 与
   `.../examples/norm/rms_norm.py`）。跨硬件复用 kernel 需要改这几处。

**结论**：昇腾路线适合**先把推理和算子跑通**，训练则要评估反向链路的完整性。在昇腾上
做完整 7B 训练，目前更像一条需要逐算子验证的工程路径，而不是“装完就能跑”。

---

## 9. 验收标准：可检查的阶段目标

把上面的内容变成一组**可执行、可判定**的里程碑。每一项都写清“前提 / 命令 / 期望输出 /
验证级别”。

**M0 · 计划与账本能复现**

* 前提：Python 3，无需 GPU。
* 命令：`python3 code/01-plan-ledger/train_plan.py --tokens 10e9 --eff-tflops <实测>`
* 期望：显存合计 113.40 GiB；算力账本输出列出的墙钟时间随卡数线性下降。
* 验证级别：**CPU 逻辑验证**（本章已跑）。

**M1 · 能跑通一步**

* 前提：`torch`、`transformers`（CPU 即可）。
* 命令：`python3 code/06-train-loop/train_step.py --steps 1`
* 期望：打印 `step 1 loss=<有限值>`，不出现 `nan`/`inf`。
* 验证级别：**CPU 逻辑验证**（本章已跑）。

**M2 · 100 步 loss 下降**

* 前提：同上。
* 命令：`python3 code/06-train-loop/train_step.py --steps 100 --micro-batch 8 --grad-accum 2`
* 期望：末 10 步平均 loss < 首 10 步平均 loss（本章实测 5.4314 → 3.8365）。
* 验证级别：**CPU 逻辑验证**（本章已跑）。

**M3 · 小规模收敛**

* 前提：一块 GPU；TinyStories 或同量级语料。
* 命令：`python3 code/06-train-loop/train_step.py --from-hf ... --dataset roneneldan/TinyStories --tokenizer ... --steps 2000 --bf16 --device cuda`
* 期望：验证集 loss 明显低于随机基线，并出现“生成的文本开始像故事”的定性信号。
* 验证级别：**未运行**（写作机器无 GPU）；需在真机上取得证据。

**M4 · 与 HF 参考实现数值对齐**

* 前提：一块 GPU；可用 `tilelang`。
* 命令：`python3 code/04-verify-numerics/compare_ref.py --n 3584 --e2e`
* 期望：前向/反向/端到端三项全 `PASS`（门限见 §7.3）。
* 验证级别：**未运行**；脚本仅静态核对。

**M5 · 热点算子换成 TileLang 后训练曲线一致**

* 前提：M4 已通过。
* 做法：把 attention 或 RMSNorm 换成 TileLang 实现，重跑 M3，比较 loss 曲线。
* 期望：两条曲线在统计误差内重合；吞吐不下降（否则说明 kernel 反而更慢）。
* 验证级别：**未运行**。

**M6 · 放大到 7B**

* 前提：8 卡以上、FSDP/ZeRO-3、足够的显存与时间预算。
* 做法：按 §10 的阶梯逐级放大；每升一级前先通过上一级的 M4。
* 期望：loss 曲线稳定、吞吐达到账本预测的量级。
* 验证级别：**未运行**；属于多机/多卡范畴。

---

## 10. 按比例缩小的降级路径

7B 全量训练单机做不了，所以本章不要求你一步到位，而是给一条**可以逐级验证**的阶梯
（图 5，`figures/05-scale-down-ladder.md`）：

| 级 | 模型 | 数据 | 硬件 | 目标 | 显存/时间估法 |
| --- | --- | --- | --- | --- | --- |
| L0 | ~0.1M | 内置语料 | CPU | 一步跑通、loss 有限 | 直接跑，秒级 |
| L1 | ~10M–125M | TinyStories / 字符级 | 1×GPU | 100 步 loss 下降、开始收敛 | 参数量代入 `train_plan.py` |
| L2 | ~0.5B | 领域语料 | 1×GPU | 与 HF 参考数值对齐 | 同上 + 实测激活值 |
| L3 | ~1.5B–3B | 更大语料 | 2–8×GPU | 跑通多卡分片与恢复 | 分片数代入 `train_plan.py` |
| L4 | 7B | 小规模领域语料 | 8–64×GPU | 吞吐/显存达标、loss 稳定 | 全量账本（§2） |

**每一级怎么估显存与时间**：

1. 把该级的参数量与训练 token 数代入 `code/01-plan-ledger/train_plan.py`
   （`--tokens`、`--dp-size`、`--eff-tflops` 都用该级的真实值）。
2. 激活值单独实测：`micro_batch` 从 1 开始往上加，每次记录峰值显存。
3. 升阶条件：上一级的 M4（数值对齐）先通过，否则放大后无法定位问题来自数据、模型还是 kernel。

**为什么必须这样降级？** 因为 7B 上暴露的问题（OOM、通信瓶颈、loss 不收敛、kernel
数值偏差）在 0.1M 上都很便宜、很快能定位；等到 7B 才第一次遇到，你会同时面对一堆
互相纠缠的现象。把变量一个个固定下来，是唯一能有效调试的办法。

---

## 本章小结

* **两本账先行**：Qwen2.5-7B 训练态下权重+梯度+优化器约 **113 GiB**（未分片），
  单卡不可能；算力用 `C ≈ 6·N·D` 估算，`--eff-tflops` 必须实测而非假设。
* **两条路线**：小规模预训练复现（可解释、成本高）与从开源权重继续训练（见效快、
  初始化不是自己的）；二者共用同一个训练循环与算子替换点。
* **数据管线是训练的一部分**：流式读取、精确去重、带缓冲的 shuffle、packing（无 padding）；
  它成为瓶颈时表现为 GPU 利用率周期性掉零。
* **结构决定热点**：Qwen2.5-7B 的 RoPE/SwiGLU/RMSNorm/GQA 决定了这几类算子是候选替换点。
* **TileLang 只管前向算子**：数据、分布式、优化器、checkpoint、评测都不归它管；
  反向必须自己接（`torch.autograd.Function` + 自写反向 kernel）。
* **接入方式是 tvm_ffi / cython 直吃 torch 张量**，不是 `tilelang/jit/adapter/torch/`
  （那个目录在 v0.1.15 里只服务 Metal）。
* **昇腾可行但有约束**：没有 autograd、算子覆盖面有限、写法与 NV 侧不同。
* **一切都要降级验证**：L0 → L1 → … → L4，每一级通过 M0–M4 再放大。

## 延伸阅读

* 数据管线与语料：`references.md` R1（datasets 文档）、R3（FineWeb-Edu）、R4（TinyStories）。
* 训练参考资料：R8（torchtitan）、R9（litgpt）、R10（nanoGPT）。它们互相印证了 §6 的
  每个部件的职责。
* 模型与 tokenizer：R2（Qwen2.5-7B 模型卡）。
* 算子替换与数值验证：本章 `code/03`、`code/04`，以及第 13–15 章的算子实战。
* 昇腾：`code/05` 与 A1–A8 的全部引用。
* 想继续往下走：第 18 章讲把训好的模型推上生产（KV cache、continuous batching、量化推理）。
