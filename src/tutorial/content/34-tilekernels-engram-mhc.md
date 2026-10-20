# 第 34 章 TileKernels（下）：Engram 与流形超连接

## 学习目标

读完本章，你应当能够：

1. 说清 **Engram** 解决什么问题：为什么论文把“条件记忆”称为与 MoE 并列的
   **第二条稀疏轴**，`O(1)` 查表与“确定性寻址”各自指什么，以及为什么确定性
   寻址能把上百亿参数的表卸载到 host memory；能区分**论文结论**与**仓库示例能
   证明的事**。
2. 逐段对照 `engram_gate_fwd_cuda.py` 与 `engram_gate_fwd_asc.py`，指出同一份
   门控算法在两条后端上分别怎么表达（task 划分、片上缓冲、多阶段流水、约简），
   并说得出“把 RMSNorm 的权重先乘成 `weight_fused`”为什么是合法的融合。
3. 说出 `engram_gate_bwd`、`engram_grad_w_reduce`、`engram_fused_weight` 三个
   kernel 在一次训练 step 里各自做什么，并解释**为什么参数梯度必须再有一个
   单独的归约 kernel**。
4. 讲清两处 Sinkhorn 的区别：mHC 把每个 token 的 `4 × 4` mix 矩阵推成
   **双随机矩阵**（论文 §4.1 的 Birkhoff 多胞形约束）；Engram 对一张大矩阵做
   **行列 L2 缩放**。两者为什么都要在 kernel 内迭代，以及迭代结构怎么用
   TileLang 表达。
5. 画出 mHC 的 `pre_big_fuse` / `pre_split_mixes` / `pre_apply_mix` / `post` /
   `head_compute_mix` 拼起来的数据流，并把论文 §4.3 的三条基础设施设计
   （kernel fusion / recomputing / DualPipe 上的重叠）与
   `pre_big_fuse` / `multilayer_recompute` / DualPipe 一一对上。
6. 归纳“访存为主、算力需求低、shape 小”这一整类算子的共同特征，并判断一个
   新算子该用 TileLang 写，还是像 DeepGEMM 那样下钻到 Ascend C / PTX。

## 前置知识

* **第 33 章（TileKernels 上）**：`*_kernel.py`（共享层）/ `*_cuda.py` / `*_asc.py`
  的三份文件结构，以及“运行期按设备选择后端”的写法。本章不再重复这套骨架，
  直接讲这一章特有的两类算子。
* **第 7、9 章**：`T.Kernel`、`T.copy`、`T.Parallel`、`T.alloc_shared` /
  `alloc_fragment` / `alloc_local` 的语义。
* **第 11 章**：`T.Pipelined` 与多缓冲；本章会反复看到 `num_stages`。
* **第 6、19、25 章**：昇腾的存储层级（GM / L1 / L0 / UB）与执行单元（AIC / AIV），
  以及 `T.Persistent`、`T.SimdVF`、`annotate_buffer_versions` 这些昇腾方言原语
  的含义。本章会把它们用在真实的库代码上。
* **第 16 章（MoE）**：本章的对照物。MoE 是“条件计算”，Engram 是“条件记忆”；
  理解前者才能看清后者补的是什么。
* **mHC 论文**（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md`）：
  本章 §6 以它的 §3（动机）与 §4（方法 + 基础设施设计）为**定义**，以
  TileKernels 的 `mhc/` 算子族为**实现**，逐条对照。你不必先读完论文，
  但 §6 里每处“论文式 (N)”都会给出该文件的行号。

不需要你有 GPU 或 NPU。本章所有 kernel 都只做静态阅读；`code/` 下的四个脚本
都能在只有 CPU 的机器上跑。

---

## 本章的验证边界（务必先读）

写作环境是 macOS (Apple Silicon)、Python 3.11、torch 2.1.1，
**没有 NVIDIA GPU，也没有昇腾 NPU**（见 [`../../VERSIONS.md`](../../VERSIONS.md)）。
因此本章遵守三条硬规矩：

1. **每一个 kernel 结论的最高验证级别是“仅静态核对”**：源码逐行读过，行号准确，
   但**没有任何一条性能数字或正确性数字来自本机运行 kernel**。本章引用的源码都在
   corpus 的 `14-deepseek-ascend-stack` 分类下（TileKernels / Engram /
   DeepGEMM-Ascend 三个仓库的快照）。
2. **本章的数字只有三类来源**：
   * **论文自报**（Engram 论文的 scaling law 与 benchmark 表；mHC 论文的
     训练开销、loss gap 与 benchmark 表）；
   * **仓库自报**（DeepGEMM-Ascend README 的 HC prenorm 性能表）；
   * **本书的 CPU 复算**（`code/01`–`code/04`，纯 PyTorch、可复现）。
   每一条都在出现处标注来源；`references.md` 里有完整清单。
3. **mHC 既有论文，也有实现。** 本章以 mHC 论文
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md`，arXiv:2512.24880 v1，2025-12-31）的 §3–§4
   为**定义**，以 TileKernels 的 `mhc/` 算子族为**参考实现**。两者**不一致**的
   地方（迭代次数、`post` 的系数、`eps`、Sinkhorn 的起点与第一步顺序）
   在 §5.2 与 §6.1 逐条列出。**论文的实验条件在本机无法复现**，所以本章
   不给任何 mHC 的实测数字，也**不会把论文在（未具名的）加速器上测到的收益
   说成昇腾上的收益**。

本章涉及三个 corpus 子树加一篇论文，请先认准（仓库快照取自 2026-09-30；
mHC 论文的抓取日期为 2026-10-01）：

| 记号 | corpus 路径 | 是什么 |
| --- | --- | --- |
| **TK** | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/` | 用 TileLang 写的双后端算子库，本章主线 |
| **EN** | `../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/` | Engram 论文仓库：论文 PDF + 一个教学 demo |
| **DG** | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/` | 昇腾 GEMM 库；本章只取它的 HC prenorm GEMM 与“mHC kernel 由 TileLang 支撑”这两点 |
| **MHC** | `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md` | mHC 论文（arXiv:2512.24880 v1）的文本抽取；§6 的定义来源，同目录有 `paper-deepseek-mhc.pdf` |

> **一句话说清 TileKernels 与 TileLang 的关系**：TileKernels 不是 TileLang 的
> 另一个版本，而是**用 TileLang 写的一个库**。`engram_gate_fwd_kernel.py` 里的
> 共享层构造 `kernel` 对象，`*_cuda.py` / `*_asc.py` 里的
> `@tilelang.jit` 函数才是 kernel 本体。本章读的是后者。

---

## 1. 问题的由来：为什么需要“不是矩阵乘”的算子

### 1.1 Transformer 缺一个“查表”原语

大模型的参数里，有相当一部分承担的是**查表**的职责：常见实体、固定搭配、
公式模板。但在标准 Transformer 里，这类静态知识没有专门的存放位置，只能被
**重建成计算**，用前几层的 attention 与 FFN 一步步把它“推”出来。

Engram 论文把这件事说得很直白：语言建模里有两类性质不同的子任务，
**组合推理**需要深而动态的计算，而**知识检索**面对的大量内容（命名实体、
公式化表达）是局部的、静态的、高度刻板的；经典 N-gram 模型之所以有效，
正因为这些规律天然适合用**查表**表达
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:53-71`）。

于是论文提出：既然 MoE 用“条件计算”增加容量，为什么不能有一条
**条件记忆**的稀疏轴，用查表来承载静态容量？

```text
   两种稀疏，两种代价
   ┌──────────────────────────────────────────────────────────────┐
   │ MoE（conditional computation）                                │
   │   路由依赖运行期的 hidden state  →  不可预知、必须等计算      │
   │   容量来自「更多专家」          →  与显存、通信强耦合         │
   ├──────────────────────────────────────────────────────────────┤
   │ Engram（conditional memory）                                  │
   │   索引只依赖输入 token id       →  前向开始前就知道要读哪几行 │
   │   容量来自「更大的表」          →  与算力解耦，可放 host/NVMe │
   └──────────────────────────────────────────────────────────────┘
```

### 1.2 Engram 是什么：一条挂在残差流上的旁路

把论文 §2 的流程压成一句话：**用 token 序列算哈希、用哈希查静态表、用当前
hidden state 给查回来的记忆打一个闸门，再通过残差连接加回主干。**

四个组成部分（术语第一次出现，都给中英对照）：

1. **词表压缩（tokenizer compression）**：先做一次 `P : V → V′` 的满射，
   把“语义等价但 subword 不同”的 token（`Apple` 与 `␣apple`）折成同一个
   canonical id。论文报告在 128k 词表上有效词表缩小 **23%**
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:158-166`）。
2. **多头哈希（multi-head hashing）**：直接把所有 N-gram 参数化是不可行的，
   所以对每个 N-gram order 用 **K 个 head**，每个 head 用“轻量乘法-XOR 哈希”
   `φ_{n,k}` 映射到一张素数大小的表 `E_{n,k}`
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:167-176`）。多个 head 是为了压低碰撞概率。
3. **上下文感知门控（context-aware gating）**：取回的 `e_t` 是静态的、
   可能带碰撞噪声。于是把当前 `h_t` 当 Query、把 `W_K e_t` 当 Key，
   各自 RMSNorm 后做点积、乘 `1/√d`、再套一层激活得到标量闸门
   `α_t = σ( RMSNorm(h_t)ᵀ RMSNorm(k_t) / √d )`，输出 `α_t · v_t`
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:177-208`）。
4. **短卷积与残差**：`Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ`，然后
   `H^(ℓ) ← H^(ℓ) + Y`。卷积核大小 `w = 4`、膨胀率取最大 N-gram order
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:209-217`）。

本章要写的 kernel，就是第 3 步（门控）与第 2 步（寻址）；第 4 步的卷积不在
TileKernels 这次开源的 Engram 算子族里。

### 1.3 论文说了什么，仓库给了什么，证据要分开

这是本章最容易写错的地方，所以单独列一节。

**论文自报的结论（作者未复现）**：

| 结论 | 出处 |
| --- | --- |
| 把总容量在 MoE 与 Engram 之间分配，验证 loss 呈 **U 型**；纯 MoE（ρ=100%）不是最优 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:364-430` |
| 10B 这一档，验证 loss 从 `1.7248`（ρ=100%）降到 `1.7109`（ρ≈80%） | `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:418-426` |
| 最优分配比例在 `ρ ≈ 75%–80%` 附近，且跨两个算力档位稳定 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:426-428` |
| 无限内存 regime 下，“槽位数 ↔ 验证 loss”在对数空间里是**严格幂律** | `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:431-460` |
| Engram-27B 相对严格 iso-parameter/iso-FLOPs 的 MoE-27B 全面更好（MMLU +3.4、BBH +5.0、HumanEval +3.0、MATH +2.4 等） | `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:26-40,470-520` |
| 把 100B 参数的表卸载到 host memory，开销 < 3% | `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:39,95-99` |

**仓库（EN）给的是什么**：一个**教学 demo** 与一个 transformer 骨架的 mock。
`Engram/README.md` 自己写着：

> 注意： The provided code is a demonstration version intended to illustrate the data
> flow. It mocks standard components (like Attention/MoE/mHC)…
> （`../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/README.md:89`）

也就是说：**仓库里的 `engram_demo_v1.py` 不能用来验证论文的任何 scaling 结论**，
它只演示“token id → 哈希 → 查表 → 门控 → 残差”这条数据流；真正可在生产里用的
算子实现，在这次开源的清单里是 **TileKernels 的 `tile_kernels/engram/`**
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:3-5,15`）。

```text
   本章的引用纪律
   ┌──────────────────────────┬────────────────────────────────────────────┐
   │ 想说的话                  │ 应当引哪里                                  │
   ├──────────────────────────┼────────────────────────────────────────────┤
   │ 「加记忆能继续降 loss」    │ 论文（自报），不能引 demo                   │
   │ 「哈希是乘法-XOR」         │ 论文 + TK 的 engram_hash_cuda.py（一致）    │
   │ 「门控用什么激活」         │ 论文式 (4) + TK 的 gate kernel（一致）      │
   │ 「kernel 怎么切任务」      │ 只能引 TK 源码，论文里没有                  │
   │ 「收益几个百分点」         │ 只能引论文，本机无法复现                    │
   └──────────────────────────┴────────────────────────────────────────────┘
```

### 1.4 mHC 是另一半：为什么它也不是矩阵乘

TileKernels 里还有一族名字很长的算子：**manifold hyper-connections（mHC，
流形超连接）**。它把残差流从“一条”扩成 **M 条并列分支**，用可学习的连接权重
决定信息在分支之间怎么流动；而这些权重矩阵被**约束在一个流形上**，具体做法就是
§5 要讲的 Sinkhorn 归一化。

它为什么不是矩阵乘？

* 规模小：`M = 4`，所以 `comb` 矩阵是 `4 × 4`；
* 逐 token：每个 token 一个 `4 × 4`，batch 维度才是并行的来源；
* 访存为主：`pre_apply_mix` 是“4 个分支的加权和”，`post` 是“4×4 乘 4 条向量”，
  算术量远小于搬进来的字节数；
* 关键路径短：这些算子夹在 GEMM 与 Attention / MoE 之间，**延迟**比吞吐重要。

`hyper-connections` 这个方向本身有公开文献（Engram 论文的参考文献里出现了
hyper-connections, 2025，见 `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:1127-1130`）；
而 **mHC 自己的论文**就在 corpus 里：
`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md`
（arXiv:2512.24880 v1）。§6 用它把“代码在做什么”升级成
“论文定义了什么、代码怎么实现、哪里不一致”。§6.1 会把证据等级写清楚。

### 1.5 本章路线图

```text
   §2 Engram 的数据流与算子清单 ──► §3 门控 kernel 的融合设计（CUDA ↔ Ascend 逐段对照）
                                     │
                                     ▼
                                  §4 前后向与梯度归约

   §5 Sinkhorn 归一化（mHC 版 / Engram 版）──► §6 mHC 的 mix 拆分与应用
                                                     │
                                                     ▼
                                              §7 这类算子的共同特征与写法
                                                     │
                                                     ▼
                                              §8 常见坑 · §9 小结
```

---

## 2. Engram 的数据流与算子清单

### 2.1 一次前向的五个步骤

先不看 kernel，只看**逻辑上**要做的事：

```text
  输入：hidden_states [T, hc, d]      kv [T, hc+1, d]      （T = token 数）
        weight_hidden [hc, d]         weight_embed [hc, d]  （两个 RMSNorm 的权重）

  ①  fused_weight :  weight_fused = weight_hidden ⊙ weight_embed      （fp32）
  ②  gate_fwd     :  α, dot, rstd_x, rstd_k = f(hidden_states, kv, weight_fused)
                     其中 out = x + α · v
  ③  （可选）保存 dot / α / rstd_x / rstd_k 供反向

  反向（训练时）：
  ④  gate_bwd        :  给定 grad_out，算出 grad_x、grad_kv 与 grad_w_partial
  ⑤  grad_w_reduce   :  把 grad_w_partial 归约成 grad_weight_hidden /
                        grad_weight_embed 的累积量（走链式法则穿过 ⊙）
```

注意几个约定（都能在源码里核对）：

* `hc`（`hc_mult`）是超连接的分支数，**本库的实现固定要求 `hc_mult = 4`**
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_asc.py:23`、
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_bwd_cuda.py:30`），这正对应论文里 mHC 的 `M = 4`。
* `kv` 把“每个分支的 Key”与“共享的 Value”打成一个张量：前 `hc` 行是 Key，
  最后一行是 Value（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_kernel.py:20-21`、`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/engram.py:191-192`）。
* `alpha` 的公式是 `sigmoid(sign(z)·sqrt(clamp(|z|, clamp_value)))`，其中
  `z = dot · rstd_x · rstd_k · d^{-1/2}`、`dot = Σ x_d · w_h,d · w_e,d · k_d`。

### 2.2 张量与 shape 走查

`code/01-engram-gate-shapes/` 把这套 shape 跑了一遍（CPU、torch 2.1.1）：

```text
== shapes ==
  hidden_states                (5, 4, 256)  torch.bfloat16     # [T, hc, d]
  kv                           (5, 5, 256)  torch.bfloat16     # [T, hc+1, d]
  key k  = kv[:, :hc_mult]     (5, 4, 256)
  value v = kv[:, hc_mult]     (5, 256)
  weight_hidden                (4, 256)  torch.bfloat16        # [hc, d]
  weight_embed                 (4, 256)  torch.bfloat16
  weight_fused (fp32)          (4, 256)  torch.float32
```

最值得记的是**哪些张量小、哪些张量大**：

| 张量 | 形状 | 量级（`T` 大时） |
| --- | --- | --- |
| `hidden_states` / `output` | `[T, 4, d]` | **大**（正比于 T） |
| `kv` | `[T, 5, d]` | **大** |
| `weight_fused` | `[4, d]` | 常驻，与 T 无关 |
| `dot` / `gate_score` / `rstd_x` / `rstd_k` | `[T, 4]` | **小**（比 hidden 小 `d` 倍） |

这就是为什么 `save_for_backward` 值得单独做一个开关：四个 `[T, 4]` 的 fp32 张量在
`T = 8192` 时只有 512 KB，而 `[T, 4, d]` 的激活是 GB 级；推理时不需要反向，
这四块直接不分配（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_kernel.py:63-70`）。

### 2.3 寻址：为什么是 `O(1)`、为什么能卸载

`code/04-engram-hash-lookup/` 把哈希与查表在 CPU 上跑了一遍：

```text
== shapes ==
  ngram_token_ids     (6, 3) int32
  multipliers         (2, 3) int64
  vocab_sizes         (2, 2, 8) int32 (prime, per head)
  offsets             (2, 16) int32 (exclusive prefix sum)
  indices             (2, 6, 16) int32
    = num_layers x num_tokens x ((max_ngram-1) * heads) = 2 x 6 x 16

== the number of gathered rows is set by the architecture ==
  table with   258,000 slots: rows gathered per token = 16
  table with 10,000,000 slots: rows gathered per token = 16
```

三句话总结：

1. **每个 token 取的行数是架构常量**：`(max_ngram - 1) × heads`。
   表从 25.8 万槽放大到 1000 万槽（40 倍），取行数不变。这就是论文说的
   `O(1)` 查表（相对表容量而言）。
2. **索引只依赖 token id**：`code/04` 里第二次运行得到完全相同的 index，
   因为哈希的乘子与素数表都是固定超参，不参与训练
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_hash_kernel.py:14-27`）。
3. **确定性 ⇒ 可预取 ⇒ 可卸载**：既然前向开始前就知道要读哪些行，
   系统就能在上一层还在算的时候把行从 host memory 搬到 HBM。
   论文报告 100B 参数的表卸载开销 < 3%（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:39,95-99`）。
   注意这是**论文自报**的实验结论，本章无法复现。

一个容易混的点：`O(1)` 说的是“取几行”不随表增长，**不是**说访存延迟是常数。
真正的延迟取决于表放在哪一级存储、以及预取有没有被计算盖住。这正是论文 §2.5
讨论“Engram 放在第几层”的原因（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:341-351`）。

### 2.4 Engram 算子清单

| kernel | 文件 | 做什么 | 本章小节 |
| --- | --- | --- | --- |
| `engram_hash` | `engram_hash_kernel.py` / `_cuda.py` / `_asc.py` | N-gram 乘法-XOR 哈希 → 每个 head 的 embedding index（image token 写 `-1`） | §2.3 |
| `fused_weight` | `engram_fused_weight_kernel.py` | `weight_hidden ⊙ weight_embed` → fp32 | §3.1、§4.1 |
| `engram_gate_fwd` | `engram_gate_fwd_kernel.py` | RMSNorm + 门控 + 残差融合的前向 | §3 |
| `engram_gate_bwd` | `engram_gate_bwd_kernel.py` | 反向：`grad_x` / `grad_kv` / `grad_w_partial` | §4.2 |
| `grad_w_reduce` | `engram_grad_w_reduce_kernel.py` | 归约 `grad_w_partial` 并穿过 `⊙` 拆成两个权重梯度 | §4.3 |
| `engram_sinkhorn_*` | `engram_sinkhorn_kernel.py` | 大矩阵的行列 L2 缩放（四个 kernel 接力） | §5.3 |
| autograd 包装 | `modeling/engram/engram_gate.py` | 把上面几个 kernel 串成 `torch.autograd.Function` | §4.4 |

---

## 3. 门控 kernel 的融合设计

### 3.1 融合了什么，为什么可以这样融合

未融合的写法是六个算子：两次 RMSNorm、一次点积、一次 signed-sqrt、一次 sigmoid、
一次残差加。把它们融进一个 kernel 有两个动机：

* **启动开销**：kernel 数与 launch 次数成正比，而这几个算子每个都只干很少的活；
* **访存往返**：`normed_x`、`normed_k` 都是 `[T, hc, d]` 级别的张量，
  写出去再读回来，正是“访存为主”算子的主要浪费。

融合的合法性来自一个恒等式。“带权重的 RMSNorm”的输出就是“归一化后逐元素乘
权重”：`RMSNorm(x, w_h) = x · rstd_x · w_h`，其中
`rstd_x = rsqrt(mean(x²) + eps)`。于是

$$
\begin{aligned}
&\operatorname{RMSNorm}(x,w_h)^\top
 \operatorname{RMSNorm}(k,w_e)\\
&=r_xr_k\sum_d x_dw_{h,d}w_{e,d}k_d\\
&=r_xr_k\sum_d x_dw_{\mathrm{fused},d}k_d.
\end{aligned}
$$

其中 \(r_x\) 和 \(r_k\) 分别是两份输入的 `rstd`。

只要把 `w_fused = w_h ⊙ w_e` 提前算好，一次扫描就能同时得到
`Σx²`、`Σk²`、`Σ x·w_fused·k` 三个标量，而**不需要**把两个归一化后的张量写出来。
`code/01` 验证了这个恒等式在 fp32 下误差 < 1.3e-7：

```text
== fusion identity: unfused vs fused ==
  max|gate_naive - gate_fused| = 1.192e-07
  max|dot_naive  - dot_fused | = 2.384e-07
```

`fused_weight` 因此是一个**独立的 kernel**：它只在权重更新后需要重算一次，
而门控 kernel 每次前向都要跑。把“权重乘积”和“数据相关的点积”分开，
两者被调用的频率差了若干个数量级（每 step 一次 vs 每 micro-batch 一次）。

### 3.2 CUDA 版逐段读

打开 `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_cuda.py`。骨架如下：

```python
@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
})
def get_engram_gate_fwd_kernel_cuda(hidden_size, eps, scalar, num_sms, clamp_value=1e-6,
                                    hc_mult=4, save_for_backward=True,
                                    has_image_token_mask=False, inplace=False, use_pdl=False):
    num_tokens = T.dynamic('num_tokens')
    threads = 32
    vec_size = 8
    ...
    with T.Kernel(hc_mult, num_persistent_blocks, threads=threads) as (pid_h, pid_b):
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_cuda.py:7-12,27-29,68`）

逐段拆：

**① 任务划分：一个“head × 持久块”一个 CTA，32 个线程。**
`T.Kernel(hc_mult, num_persistent_blocks)` 的网格是二维的：`pid_h ∈ [0, 4)` 是分支，
`pid_b` 是 token 上的持久块。每个 CTA 分到
`per_block = ceil(num_tokens / num_persistent_blocks)` 个 token（`:86-88`），
块数由 shared memory 占用反推：

```python
def _choose_num_persistent_blocks(hidden_size, blk_d, num_sms, hc_mult):
    smem_bytes = hidden_size * 2 + blk_d * 4      # x_smem (bf16) + kv_smem 双缓冲
    blocks_per_sm = min(get_max_smem_per_sm() // smem_bytes, 16)   # 16 是寄存器压力上限
    return num_sms * blocks_per_sm // hc_mult
```

（`:38-43`）注意末尾的 `// hc_mult`：网格第一维已经是 4 个分支了，
每个分支能分到的块数是总块数除以 4。**这类“网格维度 × 持久块数”的换算，
是持久化 kernel 里最常写错的地方。**

**② 片上缓冲：`x_smem` 常驻，`kv_smem` 双缓冲。**

```python
x_smem  = T.alloc_shared((hidden_size,), T.bfloat16)     # 整行 x 常驻
kv_smem = T.alloc_shared((2, blk_d), T.bfloat16)         # k 与 v 共用这一个缓冲的两半
```

（`:83-84`）`x` 需要被两遍（求和 + 写输出）都用到，所以整行留在片上；
`k`/`v` 各自只用一次，于是按 tile 双缓冲流式读。

**③ 第一遍：`cp.async` 流水 + 三个累加器。**

```python
for i_b in T.Serial(1, num_blk):
    phase = i_b % 2
    prev_phase = (i_b - 1) % 2
    T.async_copy(hidden_states[i_s, pid_h, i_b*blk_d:(i_b+1)*blk_d], x_smem[...])
    T.async_copy(kv[i_s, pid_h, i_b*blk_d:(i_b+1)*blk_d], kv_smem[phase, :])
    T.ptx_wait_group(2)
    for i_sub in T.Serial(sub_blks):
        ...
        for i_k in T.serial(vec_size):
            rstd_x_local[0] += x_local[i_k] * x_local[i_k]
            rstd_k_local[0] += k_local[i_k] * k_local[i_k]
            gate_score_local[0] += x_local[i_k] * w_local[i_k] * k_local[i_k]
```

（`:110-126`）三个累加器全在寄存器里。注意它加载的是 `w_local = weight_fused[...]`
（`:122`）。**这里读的是融合后的权重**，而不是两个 RMSNorm 权重。

**④ 归约与激活。**

```python
rstd_k_reducer[0] = T.warp_reduce_sum(rstd_k_local[0])
rstd_x_reducer[0] = T.warp_reduce_sum(rstd_x_local[0])
gate_score_reducer[0] = T.warp_reduce_sum(gate_score_local[0])
rstd_x_reducer[0] = T.rsqrt(rstd_x_reducer[0] / hidden_size + eps)
rstd_k_reducer[0] = T.rsqrt(rstd_k_reducer[0] / hidden_size + eps)
...
gate_score_reducer[0] = gate_score_reducer[0] * rstd_x_reducer[0] * rstd_k_reducer[0] * scalar
gate_score_reducer[0] = T.sigmoid(
    T.copysign(T.sqrt(T.clamp(T.abs(gate_score_reducer[0]), clamp_value, float('inf'))),
               gate_score_reducer[0]))
```

（`:149-166`）因为 `threads = 32` 恰好是一个 warp，约简只需要一次
`warp_reduce_sum`（warp shuffle），不需要 shared memory 中转。
那行 `* rstd_x * rstd_k * scalar` 就是式 (4) 的“乘 `1/√d`”，
`T.copysign` 实现 `sign(z)·sqrt(|z|)`，`T.clamp` 是 `clamp_value` 的落点。

**⑤ 第二遍：写输出，同时把下个 token 的 `k`/`x` 预取进来。**

```python
for i_b in T.Serial(num_blk):
    tile_phase = (v_start_phase + i_b) % 2
    ...
    output_target[i_s, pid_h, sub_base + thread_idx*vec_size + i_k] = (
        x_local[i_k] + gate_score_reducer[0] * v_local[i_k])
    if i_b + 2 < num_blk:
        T.async_copy(kv[i_s, hc_mult, (i_b+2)*blk_d:(i_b+3)*blk_d], kv_smem[tile_phase, :])
```

（`:173-195`）`v` 在 `kv` 的最后一行（`kv[i_s, hc_mult, …]`），所以它和 `k` 可以
共用 `kv_smem`：第二遍开始时 `k` 已经不用了，腾出来的 bank 正好放 `v`。
这种“生命周期不重叠的两个张量共用一个缓冲”是手写 pipeline 里常见的省存储手法；
注释里的 `v_start_phase = num_blk % 2`（`:53`）负责让相位对齐。

**⑥ `inplace` 与 `save_for_backward` 的互斥。**

```python
assert not (inplace and save_for_backward), 'inplace destroys hidden_states, which backward reads'
```

（`:26`）推理时可以把结果直接写回 `hidden_states`（省一块 `[T,4,d]` 的显存），
但训练时反向要读原始的 `x`，所以两者不能同时开。

### 3.3 昇腾版逐段读

打开 `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_asc.py`。它的开头就不同：

```python
import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

@tilelang.jit
def get_engram_gate_fwd_kernel_asc(hidden_size, eps, scalar, clamp_value, hc_mult,
                                   save_for_backward, has_image_token_mask, inplace,
                                   num_vec_cores):
    vec_size = 64
    num_accs = 2
    heads_per_task = 2
    num_stages = 2
    ...
    with T.Kernel(num_vec_cores) as core_id:
        head_group = core_id % num_head_groups
        head_base = head_group * heads_per_task
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_asc.py:1-3,18-21,45-47`）

逐段对照 CUDA 版的六点：

**① 任务划分：核数固定，用 `T.Persistent` 把“(token, head 组)”逐个分配给核心。**

```python
for token_id, task_head_group in T.Persistent(
        [num_tokens, num_head_groups], num_vec_cores, core_id,
        group_size=num_head_groups, num_stages=num_stages):
```

（`:76-82`）CUDA 版把“分支”放在网格第一维、把 token 分段；昇腾版把
“分支组”也塞进 `T.Persistent` 的迭代空间，核数由 `num_vec_cores` 决定。
`heads_per_task = 2`、`num_head_groups = 4 / 2 = 2`（`:28-29`），
即一个任务处理 2 个分支。**这样每个任务有 2 个独立的归约目标，
能把向量寄存器的 64 lane 用满**（每 32 lane 一个分支）。

**② 片上缓冲：UB 里的多版本缓冲，用一次注解声明。**

```python
weight_ub   = T.alloc_shared((heads_per_task, hidden_size), T.float32)
x_ub        = T.alloc_shared((heads_per_task, hidden_size), T.bfloat16)
k_output_ub = T.alloc_shared((heads_per_task, hidden_size), T.bfloat16)
v_ub        = T.alloc_shared((hidden_size,), T.bfloat16)
buffer_versions = {x_ub: num_stages, k_output_ub: num_stages,
                   v_ub: num_stages, gate_ub: num_stages}
if save_for_backward:
    buffer_versions.update({dot_ub: num_stages, rstd_x_ub: num_stages, rstd_k_ub: num_stages})
T.annotate_buffer_versions(buffer_versions)
```

（`:49-72`）昇腾的多缓冲不是手写 `phase = i_b % 2`，而是**声明每个缓冲要几个版本**
（`num_stages`），由后端在 pass 里自动展开成多份并插入同步
（`annotate_buffer_versions` 的语义见第 19 章）。这里还有一个细节：
只有 `save_for_backward` 时才给 `dot_ub` / `rstd_*_ub` 多版本，
**注解本身也是按运行期路径条件拼出来的**。

另外注意 `weight_fused` 被一次性搬进 UB 并**常驻**：

```python
T.copy(weight_fused[head_base:head_base+heads_per_task, :], weight_ub)
```

（`:74`）因为每个核心负责的 head 组在整轮迭代里不变，权重只读一次。

**③ 第一遍：显式 SIMD 内建的累加。**

```python
with T.SimdVF():
    one = S.vdup(1.0, T.float32); zero = S.vdup(0.0, T.float32)
    accum = S.alloc_local((3 * num_accs,), T.float32)
    for head_id in T.unroll(heads_per_task, explicit=True):
        ...
        for chunk_group in T.serial(num_chunks // num_accs):
            for acc_id in T.unroll(num_accs, explicit=True):
                offset = (chunk_group * num_accs + acc_id) * vec_size
                x_f32 = S.vcvt(S.vld(x_ub[head_id, offset], dist='UNPK_B16'), T.float32, part=0)
                k_f32 = S.vcvt(S.vld(k_output_ub[head_id, offset], dist='UNPK_B16'), T.float32, part=0)
                weight_f32 = S.vld(weight_ub[head_id, offset])
                S.vmula(accum[acc_id], x_f32, x_f32)
                S.vmula(accum[num_accs + acc_id], k_f32, k_f32)
                weighted_x = S.vmul(x_f32, weight_f32)
                S.vmula(accum[2*num_accs + acc_id], weighted_x, k_f32)
```

（`:122-151`）同样是三个累加量（`Σx²`、`Σk²`、`Σx·w·k`），但表达方式是显式的
向量内建：`S.vld` 取值、`S.vcvt(..., dist='UNPK_B16')` 做 bf16→fp32 的解包、
`S.vmula` 累乘加。CUDA 版靠编译器把 `T.vectorized(8)` 的循环向量化；
昇腾版把向量操作**写死成内建**，控制更精确，也更容易看出用了哪些寄存器。

**④ 归约与激活：**

```python
sum_x2  = S.vcadd(accum[0])
sum_k2  = S.vcadd(accum[num_accs])
raw_dot = S.vcadd(accum[2*num_accs])
rms_x = S.vsqrt(S.vadds(S.vmuls(sum_x2, 1.0 / hidden_size), eps))
inv_rms_x = S.vdiv(one, rms_x)
normalized_dot = S.vmuls(S.vmul(S.vmul(raw_dot, inv_rms_x), inv_rms_k), scalar)
signed_root_abs = S.vsqrt(S.vmaxs(S.vabs(normalized_dot), clamp_value))
signed_root = S.vsel(signed_root_abs, S.vneg(signed_root_abs),
                     S.vcmps(normalized_dot, 0.0, op='ge'))
gate = S.vdiv(one, S.vadd(S.vexpdif(zero, signed_root), one))
```

（`:157-171`）**一模一样的数学式**，逐项对应：
`vcadd` ↔ `warp_reduce_sum`；`vmuls(1/d)` + `vadds(eps)` + `vsqrt` + `vdiv`
↔ `rsqrt(Σ/d + eps)`；`vmaxs` + `vmuls` ↔ `T.clamp` 与 `* scalar`；
`vsel` + `vcmps` ↔ `T.copysign`；`vexpdif` + `vdiv(1, 1+e^x)` ↔ `T.sigmoid`。

**⑤ 第二遍：复用同一个 `k_output_ub` 写输出。**

```python
with T.SimdVF():
    gates_f32 = S.alloc_local((heads_per_task,), T.float32)
    for head_id in T.unroll(heads_per_task, explicit=True):
        gates_f32[head_id] = S.vld(gate_ub[head_id], dist='BRC_B32')   # 广播成向量
    out_f32 = S.alloc_local((heads_per_task,), T.float32)
    for chunk_id in T.serial(num_chunks):
        off = chunk_id * vec_size
        v_f32 = S.vcvt(S.vld(v_ub[off], dist='UNPK_B16'), T.float32, part=0)
        ...
        S.vmula(out_f32[head_id], v_f32, gates_f32[head_id])
        ...
        S.vsts(k_output_ub[head_id, off], S.vcvt(out_f32[head_id], T.bfloat16), dist='PK_B32')
T.copy(k_output_ub, output_target[token_id, task_head_base:task_head_base+heads_per_task, :])
```

（`:188-222`）`k_output_ub` 里的 `k` 用完就腾出来放 `out`，和 CUDA 版
“`kv_smem` 的两半交替放 `k` 与 `v`”是同一个思路。
`BRC_B32` 把标量闸门广播成整条向量，`PK_B32` 把 fp32 打包回 bf16。

**⑥ 写回训练要用的中间量：**

```python
if save_for_backward:
    T.copy(dot_ub[:heads_per_task], dot_out[token_id, task_head_base:task_head_base+heads_per_task])
    T.copy(gate_ub[:heads_per_task], gate_score[token_id, task_head_base:task_head_base+heads_per_task])
    T.copy(rstd_x_ub[:heads_per_task], rstd_x[token_id, task_head_base:task_head_base+heads_per_task])
    T.copy(rstd_k_ub[:heads_per_task], rstd_k[token_id, task_head_base:task_head_base+heads_per_task])
```

（`:223-251`）与 CUDA 版一样，四个 `[T, 4]` 的小张量按需写出。

### 3.4 两份实现的对照表

“一份算法、两个后端”的一般结构（三层文件、运行期后端选择、哪些必须重写）
见**第 33 章 §2–§4**；那里已经用 top-k gate 把“并行映射差异 ≠ 算法差异”
讲过一遍。本节只做一件事：把同一套维度用在门控这个例子上。

| 关注点（与第 33 章 §4.4 同一套维度） | CUDA（`engram_gate_fwd_cuda.py`） | 昇腾（`engram_gate_fwd_asc.py`） |
| --- | --- | --- |
| 网格与并行单位 | `T.Kernel(4, num_persistent_blocks)`，32 线程/CTA | `T.Kernel(num_vec_cores)`；`T.Persistent([num_tokens, head_groups])` |
| 一个任务处理几个 head | 1（网格第一维） | `heads_per_task = 2` |
| 向量宽度 | 由编译器从 `T.vectorized(8)` 决定 | **写死**：向量寄存器 64 lane |
| 数据放在哪 | `alloc_shared`（shared memory） | `alloc_shared`（UB） |
| 归约的表达 | `T.warp_reduce_sum`（warp shuffle） | `S.vcadd` / `S.vreduce` 系列 |
| 激活的表达 | `T.copysign` / `T.clamp` / `T.sqrt` / `T.sigmoid` | `S.vsel` / `S.vmaxs` / `S.vsqrt` / `S.vexpdif` |
| 输出怎么写回 | 直接 `output_target[...] = ...` | 先写 UB，再 `T.copy` 回 GM |
| 缓冲版本 / 流水 | 手写 `phase` 双缓冲 + `async_copy` + `ptx_wait_group` | 声明 `annotate_buffer_versions(num_stages)`，由 pass 展开 |
| 权重驻留 | 每个 tile 从 global 读 `weight_fused` | 整个 head 组的权重常驻 UB |
| 启动/结束提示 | 没有用 PDL（本 kernel 未开 `use_pdl`） | PDL 是 CUDA-only，这里不适用 |

**两条实现算的是同一个式子**：这是本章最想让你记住的一点。
“一份算法、两个后端”不是“两份代码碰巧结果一样”，而是**同一个数学式在两种
机器模型上的忠实表达**；差异全部落在“谁负责同步、谁负责切任务、向量宽度是多少”
这三件事上。

### 3.5 融合的收益从哪来

用 `code/01` 的记账（元素个数 × 字节数，**不是实测**）：

```text
== naive path: what the fusion removes ==
  normed_x + normed_k intermediates: 40960 B (fp32), written then read
  fused path keeps only 3 per-thread accumulators (x^2, k^2, x*w*k)

== how many bytes does the fused kernel touch (bf16 in, bf16 out)? ==
  read  x + kv :   23040 B
  write output :   10240 B
  weight_fused : 4096 B (fp32, resident)
```

把 `normed_x` / `normed_k` 消掉，省下的就是“写一遍 + 读一遍”两个
`[T, hc, d]` 张量的流量；融合后的 kernel 只需要读 `x`、`kv` 与常驻的
`weight_fused`。**这也是为什么本章把它们统称为“访存为主”的算子**：
算力需求很低（每个元素几次乘加），瓶颈在搬了多少字节、启动了多少次 kernel。

---

## 4. 前后向与梯度归约

### 4.1 三个 kernel 的分工

一次训练 step 里，Engram 门控这条链上有三个 kernel：

```text
   forward                                   backward（逆序）
   ───────                                   ────────────────
   ① fused_weight   : w_f = w_h ⊙ w_e
   ② gate_fwd       : out, dot, α, rstd_x, rstd_k
                                              ③ gate_bwd
                                                 in : grad_out, x, kv, w_f,
                                                      dot, α, rstd_x, rstd_k
                                                 out: grad_x, grad_kv,
                                                      grad_w_partial  ← 未归约！
                                              ④ grad_w_reduce
                                                 in : grad_w_partial, w_h, w_e
                                                 out: grad_w_h += , grad_w_e +=
```

分工的原则很清楚：

* **`fused_weight`** 把“两个权重的乘积”算成 fp32，供前向与反向共用；
* **`gate_bwd`** 是“逐 token 的链式法则”，每个 CTA 只负责一部分 token，
  所以它只能产出**部分和** `grad_w_partial`；
* **`grad_w_reduce`** 是“跨 CTA 的归约 + 穿过 ⊙ 的链式法则”，这部分无法在
  并行写里完成。

### 4.2 反向逐段读

`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_bwd_cuda.py` 的 docstring 已经把并行划分
写在第一行：

```python
"""Backward kernel: 8 warps per CTA, 2 warps per head.

grad_out and v cached in shared memory. grad_w accumulates in registers.
Cross-warp dldg reduction via shared memory.
"""
```

（`:25-29`）逐段拆：

**① 配置：`hc_mult = 4` 个 head，每个 head 两个 warp，所以 8 个 warp / CTA。**

```python
warp_size = 32; warps_per_head = 2
num_warps = hc_mult * warps_per_head          # 8
threads = warp_size * num_warps               # 256
threads_per_head = warp_size * warps_per_head  # 64
...
head_id = warp_id // warps_per_head
sub_warp_id = warp_id % warps_per_head
```

（`:32-36, 122-123`）256 个线程一个 CTA，`num_persistent_blocks` 个 CTA 覆盖全部
token。每个 head 被两个 warp 瓜分，**这些几何量决定了后面所有 tile 的尺寸**
（`_choose_go_config`、`_choose_x_blk_d` 都在反推 hidden 应该怎么切）。

**② 第一遍：先算 `dldg`，“输出对闸门的梯度”。**

```python
for i_b in T.serial(1, num_go_tiles):
    T.async_copy(grad_out[i_s, :, i_b*go_blk_d:(i_b+1)*go_blk_d], go_smem[...])
    T.ptx_wait_group(1)
    for i_sub in T.serial(go_sub_blks):
        ...
        for i_k in T.serial(go_vec_size):
            dldg_local[0] += go_local[i_k] * v_local[i_k]

dldg_local[0] = T.warp_reduce_sum(dldg_local[0])
if lane_id == 0:
    dldg_smem[head_id, sub_warp_id] = dldg_local[0]
T.sync_threads()
```

（`:206-236`）`dldg = Σ_d grad_out_d · v_d`（因为 `out = x + α·v`）。
一个 head 的两个 warp 各算一半，写进 shared memory，再读出来相加，
这就是 docstring 里说的 `Cross-warp dldg reduction via shared memory`。

**③ 穿过 signed-sqrt 与 sigmoid，得到对 `dot` 的梯度。**

```python
dldg_r[0] = dldg_smem[head_id, 0] + dldg_smem[head_id, 1]
dldg_r[0] = T.Select(
    T.abs(dot_in_local) * scalar * rstd_x_local * rstd_k_local < clamp_value,
    0.0,
    dldg_r[0] * gate_local_hc * (1.0 - gate_local_hc) * 0.5
        * T.sqrt(scalar * rstd_x_local * rstd_k_local / T.abs(dot_in_local)),
)
```

（`:248-253`）式子本身是“sigmoid 的导数 × signed-sqrt 的导数”；
`T.Select` 那个分支对应前向里 `clamp(|z|, clamp_value)` **在边界上把梯度置零**，
这正是前向写 `T.clamp` 时必须想到的反向行为。

**④ 三个输入张量的梯度，以及 `grad_w` 的部分和。**

```python
dot_x_local = dot_in_local * rstd_x_local * rstd_x_local / hidden_size
dot_k_local = dot_in_local * rstd_k_local * rstd_k_local / hidden_size
...
grad_x[i_s, head_id, ...]  = go_x_local[i_k] + dldg_r[0] * (k_local[i_k] * w_fused_local[i_k] - x_local[i_k] * dot_x_local)
grad_kv[i_s, head_id, ...] = dldg_r[0] * (x_local[i_k] * w_fused_local[i_k] - k_local[i_k] * dot_k_local)
grad_w_local[...] += dldg_r[0] * x_local[i_k] * k_local[i_k]
```

（`:266-267, 291-299`）前两个是 RMSNorm 的链式法则（归一化会“减去自身分量”，
所以有 `- x·dot_x` 这一项）；第三个只是 `∂dot/∂w_f = x·k`，**还没有乘任何东西**，
因为 `w_f` 不是参数。

最后把寄存器里的 `grad_w_local` 写成 global：

```python
grad_w_partial[pid_p, head_id, global_off + i_k] = grad_w_local[...]
```

（`:332-336`）形状是 `[num_persistent_blocks, hc_mult, hidden_size]`，
**第一维就是 CTA 数**，这就是“部分和”的含义。

### 4.3 为什么梯度归约要单独一个 kernel

`grad_w_partial` 的第一维是 CTA 数（CUDA 上是 `num_sms`，昇腾上是
`get_num_vec_cores() // hc_mult`，见 `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_bwd_kernel.py:58`）。
要把它们变成参数梯度，必须做两件事：

1. **跨 CTA 求和**：`Σ_blocks grad_w_partial[b]`；
2. **穿过 `w_f = w_h ⊙ w_e` 的链式法则**：
   `∂L/∂w_h = grad_w · w_e`、`∂L/∂w_e = grad_w · w_h`。

corpus 的 CPU 基线把这两件事写得很清楚（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/engram/test_engram_grad_w_reduce.py:19-22`）：

```python
def grad_w_reduce_ref(grad_w_partial, weight_hidden, weight_embed,
                      grad_weight_hidden, grad_weight_embed):
    grad_w_sum = grad_w_partial.sum(0)
    grad_weight_hidden += grad_w_sum * weight_embed.float()
    grad_weight_embed  += grad_w_sum * weight_hidden.float()
```

为什么不能在前一个 kernel 里顺手做完？

* **归约维度跨 CTA。** 每个 CTA 只看到自己那部分 token，它算出的
  `grad_w_partial` 是**局部的**；跨 CTA 求和需要第二个 kernel（或者原子加）。
   TileKernels 选了“先写部分和，再启动一个 kernel 归约”，因为这样
  **结果确定**（不依赖原子加的浮点顺序）。
* **两个参数共享同一份中间量。** `grad_w` 的求和只需要做一次，然后分别乘
  `w_e` / `w_h` 得到两份参数梯度。放在一个 kernel 里做，正好顺路。
* **`main_grad` 支持。** autograd 包装层优先把梯度累加到参数的 `main_grad`
  缓冲上（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/modeling/engram/engram_gate.py:78-90`），要求“累加”而不是“赋值”，
  这也是 `grad_w_reduce` 的设计前提。

实现上它和 Sinkhorn 的 `step_reduce` 是同一种模式：

```python
T.copy(grad_w_partial[i_r*num_rows:(i_r+1)*num_rows, pid_h, pid_b*blk_d:(pid_b+1)*blk_d], grad_w_shared)
for i in T.Serial(num_rows):
    for j in T.Parallel(blk_d):
        grad_w_fragment[j] += grad_w_shared[i, j]
for j in T.Parallel(blk_d):
    grad_wh_fragment[j] += grad_w_fragment[j] * we_fragment[j]
    grad_we_fragment[j] += grad_w_fragment[j] * wh_fragment[j]
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_grad_w_reduce_cuda.py:45-60`）先 `T.Pipelined` 分批把部分和搬进
shared memory，再在寄存器里逐行累加，**用“分批 + 片上累加”代替一次读完
全部部分和**，因为部分和的总量是 `num_ctas × hidden_size` 级别。

### 4.4 autograd 包装与 `main_grad`

`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/modeling/engram/engram_gate.py` 把上面几个 kernel 串成一个
`torch.autograd.Function`：

* `forward`：`fused_weight` → `engram_gate_fwd` → 把
  `x, kv, w_h, w_e, w_f, dot, α, rstd_x, rstd_k, mask` 存进 `ctx`
  （`:45-58`）；
* `backward`：`engram_gate_bwd` → 处理 `main_grad` → `grad_w_reduce`
  （`:72-90`）。

一个值得学的细节：

```python
main_grad_wh = getattr(weight_hidden, 'main_grad', None)
grad_wh = main_grad_wh if main_grad_wh is not None else torch.zeros_like(weight_hidden, dtype=torch.float32)
grad_we = main_grad_we if main_grad_we is not None else torch.zeros_like(weight_embed, dtype=torch.float32)
grad_w_reduce(grad_w_partial, weight_hidden, weight_embed, grad_wh, grad_we)
...
return (grad_x..., grad_kv..., None if main_grad_wh is not None else grad_wh, ...)
```

（`:80-99`）当参数带 `main_grad`（fp32 梯度缓冲）时，`grad_w_reduce` 往那个缓冲里
**原地累加**，函数本身返回 `None`；否则分配 fp32 缓冲并返回。
这解释了两件事：

1. 为什么 `grad_w_reduce` 的语义是“累加”而不是“写入”；
2. 为什么它必须接受 fp32 的输出缓冲，参数本身是 bf16，但梯度累加要在 fp32 里做。

---

## 5. Sinkhorn 归一化

### 5.1 Sinkhorn-Knopp 在做什么

给定一个非负矩阵，**交替把每一行、每一列归一化到和（或范数）为 1**，
反复若干轮。Sinkhorn-Knopp 定理保证（在一定条件下）迭代收敛到一个
**双随机矩阵**（doubly stochastic：每行每列的和都是 1）。

mHC 论文把这件事写成“把残差映射投影到 Birkhoff 多胞形上”：

```text
   论文 §4.1 式 (6)：P_Mres(H) = { H ∈ R^{n×n} | H·1 = 1, 1ᵀ·H = 1ᵀ, H ≥ 0 }
   论文 §4.1 给出双随机的三条性质（以及为什么值得要它）：
     ① 谱范数 ≤ 1（非扩张，抑制梯度爆炸）
     ② 双随机矩阵对乘法封闭（多层复合之后仍然双随机）
     ③ 几何解释：Birkhoff 多胞形是置换矩阵的凸包 → H·x 是特征的一种凸组合
   论文 §4.2 式 (8)(9)：H^res = Sinkhorn-Knopp(H̃^res)
     M^(0) = exp(H̃^res)，M^(t) = T_r( T_c( M^(t-1) ) )，t_max = 20
```

（行号：`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:349-376`
是 §4.1 的约束与三条性质，`:378-406` 是 §4.2 的参数化、式 (8) 与式 (9) 的
Sinkhorn 迭代，`t_max = 20` 出现在 `:405-406`，Table 5 里也是 `20`（`:1375-1378`）。）

这三条性质也解释了 §3.1 的动机：未加约束的 HC 复合映射
`∏ H^res` 会偏离单位映射，论文测到的 Amax Gain Magnitude 峰值接近 **3000**
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:282-292`，具体数值在 `:288`）；加了双随机约束后
单层近似双随机、复合后的偏差也被限制在约 **1.6**（`:703-715`，数值在 `:711`）。

它在本章里出现两次，但**归一化的目标、范数、状态、迭代粒度都不同**：

```text
                        mHC 的 Sinkhorn              Engram 的 Sinkhorn
   输入形状             [T, 4, 4] 每 token 一个小矩阵  [num_rows, head_dim] 一张大矩阵
   第一步               行 softmax（减行最大值）       无 softmax
   归一化目标           行和 = 列和 = 1                 行 L2 与列 L2 的尺度
   用的范数             和（L1）                       L2 范数
   迭代次数             repeat - 1 轮行列交替           行列交替（由调用方驱动）
   状态                 全部在片上寄存器                  row_scale / col_scale 两个小张量
   拆成几个 kernel       1（或融进 pre_big_fuse）        4（momentum/step/step_reduce/finalize）
   用途                 约束 mix 矩阵的流形             对大矩阵做尺度归一化
```

### 5.2 mHC 版：把 mix 矩阵推成双随机

基线只有六行（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/mhc.py:8-14`）：

```python
def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x
```

kernel 版本把它整段塞进一个 `T.Kernel(...)`：

```python
comb_frag = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)
row_max = T.alloc_fragment((token_block_size, hidden_size), T.float32)
T.reduce_max(comb_frag, row_max, dim=2)
for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
    comb_frag[i, j, k] = T.exp(comb_frag[i, j, k] - row_max[i, j])
T.reduce_sum(comb_frag, row_sum, dim=2)
...
T.reduce_sum(comb_frag, col_sum, dim=1)
...
for _ in T.serial(repeat - 1):
    T.reduce_sum(comb_frag, row_sum, dim=2)
    ...
    T.reduce_sum(comb_frag, col_sum, dim=1)
    ...
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/sinkhorn_cuda.py:24-56`）注意三点：

* 这里的 `hidden_size` 实际是 mHC 的 `M`（`= 4`，见 `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/sinkhorn_kernel.py:6-13`
  传进来的 `comb_res_mix` 形状），**变量名复用了 `hidden_size`**，读代码时别被它骗了；
* 迭代次数是 `T.serial(repeat - 1)`，即**编译期常量**，所以循环会被完全展开；
* `dim=2` 是行内归约、`dim=1` 是列内归约，两个方向正好相反，
  这就是“交替”在代码里的样子。

**论文与源码在这里有四处值得注意的差异**（这正是本章想让你学会做的核对）：

| 论文（`paper-deepseek-mhc.md`） | 源码（`TileKernels/tile_kernels/mhc/`） | 后果 |
| --- | --- | --- |
| 起点 `M^(0) = exp(H̃)`，第一步是 `T_r(T_c(·))`（`:400-406`） | 起点是 `softmax(-1)`，等价于先做了一次**行**归一化，第一步是列归一化（`sinkhorn_cuda.py:35-43`） | `T_r` 与 `T_c` 不可交换，**迭代中间点不同**；两者的不动点（双随机解）相同 |
| 迭代次数 `t_max = 20`（`:405-406`、Table 5 `:1375-1378`） | 测试与融合 kernel 默认 `repeat = 10`（`tests/mhc/test_sinkhorn.py:19`、`tests/mhc/test_pre_big_fuse.py:24`） | 达到同样精度所需轮数不同；论文的“一次迭代”是行+列各一次，源码的 `repeat` 计数把起点那一步也算进去了 |
| 论文只做归一化，没有数值 eps | 每一步都 `+ eps` 并除以 `(sum + eps)`（`sinkhorn_cuda.py:36-52`），测试用 `eps = 1e-6` | 源码的固定点带 `O(eps)` 偏移（本章 §8 的 P7 与 `code/02` 的数字） |
| `H^post = 2σ(·)`（式 (8)，`:393-399`） | 倍数被抽成参数 `mhc_post_mult_value`（`pre_split_mixes_cuda.py:62`），仓库测试默认 `1.0`（`tests/mhc/test_pre_big_fuse.py:23`） | 论文的 `2` 需要调用方显式传入；用默认值跑出来的是“半强度”版本 |

再加一处不算矛盾的差异：论文式 (8) 是 `H^pre = σ(·)`，而源码加了
`+ mhc_pre_eps`（`pre_split_mixes_cuda.py:55-60`，默认 `1e-6`），
目的是让 `pre` 严格大于 0。这是实现侧的数值保护，不改变数学含义。

反向也值得看一眼。它**在片上重算前向的每一轮迭代**（论文 §4.3.1 的说法是
“a custom backward kernel that recomputes the intermediate results on-chip and
traverses the entire iteration”，见 `paper-deepseek-mhc.md:429`），
把每一步的中间矩阵与行列和存进 shared memory：

```python
xs   = T.alloc_shared((repeat * 2, token_block_size, hidden_size, hidden_size), T.float32)
sums = T.alloc_shared((repeat * 2, token_block_size, hidden_size), T.float32)
...
for inv_step in T.unroll(2 * repeat - 1):
    T.copy(xs[2*repeat - 1 - inv_step, 0, 0, 0], x_inter)
    if inv_step % 2 == 0:   # 从「列归一化」逆着走
        ...
    else:                   # 从「行归一化」逆着走
        ...
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/sinkhorn_cuda.py:91-92,124-144`）**用 shared memory 存重算出来的中间量**：
反向不必回到 HBM 去读前向的中间结果，但 shared memory 要放 `2 × repeat` 份
`[token_block, M, M]`，
这也是为什么 `token_block_size` 被压到 8（昇腾端更小，是 4：
`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/sinkhorn_kernel.py:10-12`）。

### 5.3 Engram 版：行列 L2 缩放的四个 kernel

它的 API 分四步（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_sinkhorn_kernel.py`）：

```text
  ① engram_sinkhorn_momentum_update(grad, momentum, beta1, nesterov)
       → 更新 momentum（原地），返回 fp32 的 sinkhorn_input
  ② engram_sinkhorn_step(matrix, row_scale, col_scale, eps)
       → 返回 row_norm[num_rows] 与 col_sumsq_partial[num_partials, head_dim]
         （row_scale 原地更新）
  ③ engram_sinkhorn_step_reduce(col_sumsq_partial, col_sumsq)
       → 按固定顺序归约各 worker 的列部分和
  ④ engram_sinkhorn_finalize(matrix, row_scale, col_scale, eps, alpha)
       → 输出 alpha · (row_scale / (‖iterate‖₂ + eps)) ⊗ matrix ⊗ col_scale
```

核心的一步是 `②`，它的 CPU 基线只有三行
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/torch/engram.py:39-63`）：

```python
iterate = row_scale.unsqueeze(1) * matrix * col_scale
row_norm = torch.norm(iterate, p=2, dim=1)
row_scale.div_(row_norm + eps)
```

kernel 把它拆成“逐行算 L2 → 更新 `row_scale` → 累加列平方和”：

```python
for row in T.serial(...):
    row_scale_var = row_scale[row]
    for col in T.Parallel(head_dim):
        x_frag[col] = row_scale_var * matrix[row, col] * col_scale_frag[col]
        x_sq_frag[col] = x_frag[col] * x_frag[col]
    T.reduce_sum(x_sq_frag, norm_frag)
    norm_var = T.sqrt(norm_frag[0])
    if thread_idx == 0:
        row_norm[row] = norm_var
    norm_var = 1.0 / (norm_var + eps)
    if thread_idx == 0:
        row_scale[row] = row_scale_var * norm_var
    for col in T.Parallel(head_dim):
        x_frag[col] *= norm_var
        col_acc_frag[col] += x_frag[col] * x_frag[col]
T.copy(col_acc_frag, col_sumsq_partial[pid, :], disable_tma=True)
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_sinkhorn_cuda.py:115-137`）两件事值得注意：

* **row 是串行的，col 是并行的**：每一行都被整个 CTA 一起处理
  （`T.Parallel(head_dim)`），但行与行之间是 `T.serial`。因为每个 CTA 要算的
  `head_dim` 很长（测试里到 5120），一次放不进寄存器，所以用“一次一行、
  行内并行”的粒度。
* **列部分和谁写谁不管归约**：`col_acc_frag` 只累加自己 CTA 看到的行，
  写进 `col_sumsq_partial[pid, :]`。归约交给 `③`，
  与 §4.3 的 `grad_w_reduce` 是同一个模式：
  “先写部分和，再用一个 kernel 确定性归约”。

`code/02-sinkhorn-iterations/` 把这条流水在 CPU 上跑了一遍，输出里能直接看到
“行 L2 被拉回 1、列 L2 变成另一个常数、最后由 `finalize` 用一个 alpha 定尺度”：

```text
  step   max|row L2 - 1| after row step        col L2 before col step
     0                      5.960e-08   [1.02401, 0.48524, 1.10937, 0.6966]
     7                      1.192e-07   [0.86589, 0.86586, 0.86646, 0.8659]
    row L2 before finalize: [1.15488, 1.15442, 1.1548]
    row L2 after  finalize (alpha = sqrt(head_dim) = 2.0): [2.0, 2.0, 2.0]
```

### 5.4 为什么它值得单独写 kernel

回到“为什么要为这么小的矩阵写 kernel”：

* **迭代在片上完成，只进出 HBM 一次。** `mhc_sinkhorn_fwd` 的输入输出都是
  `[T, M, M]`（fp32），整个 `repeat` 轮迭代不落地；如果每轮一个 kernel，
  `repeat = 10` 就是 20 次 launch + 20 次读写。
* **它真的在关键路径上。** 每层都要跑（`pre_big_fuse` 里就内嵌了它），
  `repeat` 直接乘进延迟。
* **它的状态小到可以全部留在寄存器 / shared memory。** `M = 4` 且 `repeat` 是
  **编译期常量**，展开后就是“两次 reduce + 两次逐元素除法 × repeat”，
  这正是编译器擅长的循环，也正是 TileLang 的 `T.Parallel` + `T.reduce_*` 想表达的层次。

`code/02` 里那些数字还有一个直接用途：**它告诉你 `repeat` 是精度旋钮。**
`repeat = 10` 时行和偏差约 `5.7e-3`，`repeat = 30` 时降到 `1.4e-6`。
TileKernels 的测试与融合 kernel 默认用 10（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/mhc/test_sinkhorn.py:19`、
`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/mhc/test_pre_big_fuse.py:24`）。

---

## 6. mHC：把 mix 拆开再拼起来

### 6.1 证据等级：论文 §4 是定义，TileKernels 是实现

本节有两份一手材料，各自的角色不同：

| 材料 | 角色 | 引用方式 |
| --- | --- | --- |
| `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md`（arXiv:2512.24880 v1） | **定义与动机**：约束是什么、为什么这么约束、基础设施怎么设计 | 论文式 (5)–(20)、§3/§4 的行号 |
| `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/` | **参考实现**：这些映射在生产里被拆成哪些 kernel、每个 kernel 怎么切任务 | 源码路径 + 行号 |
| `code/03-mhc-mix-flow/` | 本书用 CPU 复算的公式与形状核对 | 脚本输出 |

两份材料**不完全一致**的地方，本章逐条列出：§5.2 列了 Sinkhorn 的四处差异；
§6.2 指出一处实现侧改写（论文把“除以范数”放在矩阵乘之后，代码照做并把
`1/r` 折进 reduce+scale）；§6.4 指出 `expand` 与 `head_compute_mix` 是论文没有、
但接入单流骨干时必须补的两个接口。因为“论文怎么写、代码怎么写”的差，
本身就是最有价值的阅读内容。

另外两条必须记住的边界：

* 论文 §5 的实验全部在**未具名**的加速器上完成（该文本抽取里没有出现任何硬件型号），
  所以论文的收益数字**不能**被当作昇腾上的收益；本章引用时一律带上实验条件（§6.7）。
* 论文 §4.3 里“用 TileLang 实现”的边界说得很明确：
  **除了式 (14)–(15) 那部分（也就是 prenorm GEMM）之外，其余 kernel 都用 TileLang**
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:502`）；这与本仓库把 prenorm GEMM 放进
  DeepGEMM-Ascend 的做法一致，也是本章 §7.3 判断法的最好例证。

**论文第几节 ↔ 哪个 kernel ↔ 仓库哪一行**（本节的总索引）：

| 论文 | 内容 | 对应实现 | 本书小节 |
| --- | --- | --- | --- |
| §3 式 (1)–(5)、`:198-255` | 残差流扩成 `n` 条（`x_l ∈ R^{n×C}`），三个映射 `H^pre/H^post/H^res` | `residual [n, 4, h]`、`pre/post/comb` 三段 | §6.2 |
| §3.2 Table 2、`:294-345` | 逐 token 访存记账（HC 约 `n` 倍），需要 kernel 融合 | 融合后的 `pre_apply_mix` / `post` | §6.3 |
| §4.1 式 (6)、`:349-376` | 把 `H^res` 投影到双随机流形（Birkhoff 多胞形） | `Sinkhorn` + `mhc_sinkhorn_fwd/bwd` | §5.1–5.2 |
| §4.2 式 (7)、`:378-389` | `x⃗=vec(x)`、RMSNorm、动态 `φ` + 静态 `b` | prenorm GEMM + `reduce_partials_and_rmsnorm` + `pre_split_mixes` | §6.2 |
| §4.2 式 (8)、`:391-399` | `H^pre=σ`、`H^post=2σ`、`H^res=Sinkhorn-Knopp` | `pre_split_mixes_cuda.py:54-64` | §6.2 |
| §4.2 式 (9)、`:400-406` | `M^(t)=T_r(T_c(M^(t-1)))`，`t_max=20` | `mhc_sinkhorn_fwd_cuda` | §5.2 |
| §4.3.1 式 (14)–(15)、`:419-423` | 两次扫描（前向）+ 两个矩阵乘（反向）各融一个 kernel | prenorm GEMM（DeepGEMM-Ascend，**唯一不用 TileLang 的部分**） | §6.6、§7.3 |
| §4.3.1 式 (16)–(18)、`:425` | 小系数上的轻量运算融合成一个 kernel | `pre_split_mixes`（+ `head_compute_mix`） | §6.2、§6.4 |
| §4.3.1 式 (19)、`:429` | Sinkhorn 迭代放一个 kernel；反向在片上重算并整轮回放 | `mhc_sinkhorn_fwd/bwd` | §5.2 |
| §4.3.1 `:501-502` | 两个“应用映射”的 kernel + 融合把读从 `(3n+1)C` 降到 `(n+1)C`、写从 `3nC` 降到 `nC` | `pre_apply_mix`、`post` | §6.3 |
| §4.3.2 Table 3 与式 (20)、`:504-539` | 丢掉中间激活、按 `L_r` 层块重算；`L*_r ≈ sqrt(nL/(n+2))` | `multilayer_recompute` | §6.5 |
| §4.3.3 `:541-554` | 在 DualPipe 上错开通信与重算；`F_post,res` 走高优先级流 | 属于 DualPipe，不在 TileKernels 里 | §6.7（详见第 38 章） |
| §4.3 `:411`、§1 `:219` | 大规模训练只增加 **6.7%** 时间开销（`n=4`） | 上述全部融合/重算/重叠的总效果 | §6.6 |

> **一段历史说明**：本章初稿时 corpus 里还没有 mHC 论文，只有一份第三方
> （华为 PTO 教程，`../tilelang-corpus/06-huawei-ascend/raw/pto-dsl/examples/aot/sinkhorn_demo/pto_sinkhorn_demo_zh.md:1-20`）
> 转述“DeepSeek-V4 使用了 mHC”并指向一个 arXiv 编号。现在一手论文已经入库，
> 该教程从“唯一线索”降级为“旁证”；本章所有 mHC 结论都以论文为准。

### 6.2 pre 侧：从 4 条残差分支到这一层的输入

前向的入口是 `mhc_pre_split_mixes_fwd`。它的输入是每个 token 的 `mixes`
（`mhc_mult3 = 2*M + M² = 24` 个数），输出三块：

```python
input_mixes: T.Tensor[(num_tokens, mhc_mult3), dtype]
mhc_scale:   T.Tensor[(3,), dtype]
mhc_base:    T.Tensor[mhc_mult3, dtype]
pre_layer_mix:  T.Tensor[(num_tokens, mhc_mult), dtype]
post_layer_mix: T.Tensor[(num_tokens, mhc_mult), dtype]
comb_res_mix:   T.Tensor[(num_tokens, mhc_mult2), dtype]
...
pre_layer_mix_frag[i, j]  = T.sigmoid(input_mixes_frag[i, j]*mhc_scale[0] + mhc_base[j]) + mhc_pre_eps
post_layer_mix_frag[i, j] = T.sigmoid(input_mixes_frag[i, j + mhc_mult]*mhc_scale[1] + mhc_base[j + mhc_mult]) * mhc_post_mult_value
comb_res_mix_frag[i, j]   = input_mixes_frag[i, j + mhc_mult*2]*mhc_scale[2] + mhc_base[j + mhc_mult*2]
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_split_mixes_cuda.py:28-64`）也就是说：

```text
   mixes[n, 24]
   ├─ [ 0 :  4)  → sigmoid(x·s0 + b) + eps  =  pre_layer_mix    （每分支一个门）
   ├─ [ 4 :  8)  → sigmoid(x·s1 + b) × mult =  post_layer_mix   （每分支一个门）
   └─ [ 8 : 24)  → x·s2 + b ，reshape 成 4×4 =  comb_res_mix     （分支之间的连接）
```

`scale`（3 个数）与 `base`（24 个数）都是可学习参数，所以反向里它们也要梯度，
这就是为什么 `pre_split_mixes_bwd` 会输出两份 `*_grad_partial`
（`:100-104,168-171`），再由 `pre_split_mixes_bwd_reduce_cuda` 归约
（`:176-209`）。**和 Engram 的 `grad_w_reduce` 一模一样的拆分理由。**

把这段实现逐项对回论文，就得到一张完整的“公式 ↔ 张量”表：

| 论文（`paper-deepseek-mhc.md`） | 代码里的对应物 |
| --- | --- |
| `x_l ∈ R^{n×C}` 的 `n` 条残差流（式 (2)(3)，`:198-207`） | `residual [num_tokens, mhc, hidden]`，`mhc = n = 4` |
| `x⃗_l = vec(x_l) ∈ R^{1×nC}`，`x⃗′_l = RMSNorm(x⃗_l)`（式 (7)，`:380-389`） | prenorm GEMM 的输入；RMSNorm 权重被吸进 `φ_l`（论文 `:417`） |
| `H̃̃ = x⃗_l φ_l`（式 (14)，`:462-464`） | `gemm_out_mul`／`gemm_out_sqrsum` 两个 split-K 部分和 |
| `r = ‖x⃗_l‖₂/√(nC)`，`[H̃] = (1/r)[α H̃̃] + b`（式 (15)(16)，`:469-478`） | `mhc_reduce_partials_and_rmsnorm_fwd` 先算 `1/r` 并折进 `mhc_scale`，`pre_split_mixes` 再乘 `scale` 加 `base` |
| `H^pre = σ(H̃^pre)`、`H^post = 2σ(H̃^post)`、`H^res = Sinkhorn-Knopp(H̃^res)`（式 (17)(18)(19)，`:483-499`） | 同一个 kernel 里的三段：`pre`（`+ eps`）、`post`（`× mhc_post_mult_value`）、`comb`（进 Sinkhorn） |
| `n²+2n = 24` 个系数（式 (10)(13)，`:430-457`） | `mhc_mult3 = 2*M + M² = 24` |

**一处实现侧的改写值得单独指出**：论文把“除以范数”放在矩阵乘**之后**
（§4.3.1 第一句：“we reorder the dividing-by-norm operation to follow the matrix
multiplication”，`:415-417`），因此 `φ_l` 里可以直接吸收 RMSNorm 的权重
（`:417`）。代码里对应的就是 `mhc_reduce_partials_and_rmsnorm_fwd` 把 GEMM 的
split-K 部分和与 `Σx²` 一起归约，再乘 `rsqrt(Σx²/(nC) + eps)`
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/norm_fn_kernel.py:64-99`）。
论文声称这一步“maintains mathematical equivalence”，本章只能静态核对，
**没有在数值上验证过这两种顺序的等价性**（`code/03` 用的是已经归约好的 `mixes`）。

pre 侧最后一步是 `pre_apply_mix`：

```python
for i_mhc in T.serial(mhc):
    for i1_h in T.Parallel(h_blk):
        ol[i1_h] += mixl[i_mhc] * xl[i_mhc, i1_h]
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_apply_mix_cuda.py:46-48`）即 `o[n, h] = Σ_m pre[n, m] · residual[n, m, h]`。
这正是论文 §4.3.1 里的 `F_pre := H^pre_l x_l`
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:501`）。
它的反向同时要算 `x_grad` 与 `mix_grad`：

```python
T.reducer_update(mgl_reducer[i_mhc], ogl[i1_h] * xl[i_mhc, i1_h])
xgl[i_mhc, i1_h] += mixl[i_mhc] * ogl[i1_h]
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_apply_mix_cuda.py:103-105`）前者是对 `mix` 的归约，后者是对 `x` 的逐元素累加。
注意这里用的是 `T.alloc_reducer` / `T.reducer_update` / `T.finalize_reducer`
（`:83-85,109`），**Tree-reduction 的写法在 TileKernels 里比手写 `warp_reduce_sum`
更常见**，因为它能跨 warp 自动归约。

### 6.3 post 侧：写出新的 4 条残差分支

`mhc_post_fwd` 的公式（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/post_cuda.py:27-60`）：

```python
a: T.Tensor[(n, mhc, mhc), T.float32]   # comb_res_mix
b: T.Tensor[(n, mhc, h), T.bfloat16]    # residual
c: T.Tensor[(n, mhc), T.float32]        # post_layer_mix
d: T.Tensor[(n, h), T.bfloat16]         # 层的输出 layer_output
x: T.Tensor[(n, mhc, h), T.bfloat16]    # 输出（新的 residual）
...
x_local = c_frag[i_mhco] * d_shared[i1_h]
for i_mhci in T.unroll(mhc):
    x_local += a_frag[i_mhci, i_mhco] * b_shared[i_mhci, i1_h]
x_frag[i_mhco, i1_h] = x_local
```

即

$$
\mathrm{new\_res}[n,o,h]
=\mathrm{post}[n,o]\,\mathrm{layer\_output}[n,h]
+\sum_m \mathrm{comb}[m,o]\,\mathrm{residual}[n,m,h]
$$

这与普通残差 `h ← h + f(h)` 的差别在于：**新残差的第 `o` 条分支由所有旧分支
线性混合而来**，混合系数就是 `comb`。`comb` 被 Sinkhorn 约束成双随机，
于是这种混合是“守恒的”，信息在分支之间流动，但总权重受控。

这条式子在论文里就是 §4.3.1 的 `F_post,res := H^res_l x_l + H^{post⊤}_l F(·,·)`
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:501`），同时它对应 §3 式 (4) 里
`Σ_i (∏ H^res) H^{post⊤} F(H^pre x_i)` 那一项的第二半
（`:205-208`，式 (4) 的完整展开）。论文特别说明：把 `H^post` 与 `H^res`
的“应用”和残差合并融进同一个 kernel 之后，这个 kernel 的**读从 `(3n+1)C` 降到
`(n+1)C`、写从 `3nC` 降到 `nC`**（`:501`）。这正是 `post_cuda.py` 一个 kernel
里同时干“`post ⊙ out`”和“`combᵀ · residual`”两件事的理由。

`code/03-mhc-mix-flow/` 在 CPU 上验证了这条递推，也验证了 `comb` 经过 Sinkhorn
之后行列和都是 1：

```text
== split + sinkhorn ==
  pre  (3, 4, 1)  post (3, 4, 1)  comb (3, 4, 4) -> normalized
  row sums of comb: [0.999999, 0.999999, 0.999999, 0.999999]
  col sums of comb: [0.999999, 0.999999, 0.999999, 0.999999]

  layer_input = sum_m pre_m * residual_m : (3, 64) torch.bfloat16
  new residual = post * layer_output + comb^T @ residual : (3, 4, 64)
```

### 6.4 边界算子：`expand` 与 `head_compute_mix`

两个小算子负责把数据流的两端接上：

* **`expand_to_mhc_fwd`**：把单流 `[n, h]` 复制成 `[n, M, h]`，让模型从
  普通残差进入多分支；反向是沿分支求和。

```python
xl = T.alloc_fragment((blk_n, blk_h), T.bfloat16)
T.copy(x[pid_i*blk_n, pid_j*blk_h], xl)
for m in T.serial(mhc):
    for ti, tj in T.Parallel(blk_n, blk_h):
        o[i, m, j] = xl[ti, tj]          # fwd：复制 M 份
...
xgl[ti, tj] += o_grad[i, m, j]           # bwd：对 M 条分支求和
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/expand_cuda.py:18-25,43-51`）

* **`head_compute_mix`**：只有 `sigmoid(x·scale + base) + eps`
  （`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/head_compute_mix_cuda.py:30-35`）。它是一个“纯逐元素”算子，
  价值在于**和 expand 一起把模型的头尾补齐**，让多分支结构可以从第 0 层
  一直用到最后一层。

论文对这两个算子没有专门定义。它们是把 mHC 接进一个**普通**（单流）骨干时
必须补的接口：

* `expand` 实现的是 §3 开头那句“the input … is expanded by a factor of n to
  construct a hidden matrix `x_l ∈ R^{n×C}`”（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:241`），
  反向自然就是“回到单流”的求和；
* `head_compute_mix` 复用式 (8) 里 `H^pre = σ(·)` 的形式
  （`:393-399`），但只算**模型最后一层**需要的那 `n` 个系数
  （`head_compute_mix_cuda.py:22-25` 的 `[num_tokens, mhc_mult]`），
  所以没有 `comb`，也不需要 Sinkhorn。

换句话说：**论文定义了“层内”三个映射，`expand` / `head_compute_mix` 定义了
“层外”两个接口**。后者是工程约定，不是论文内容。这也是本章 §7.3 说的
“哪些东西是论文的、哪些是实现的”的一个小例子。

### 6.5 `multilayer_recompute`：用重算换显存

反向需要每层的 `residual`、`pre_mix`、`post_mix`、`comb_mix` 与 `layer_output`。
如果全存下来，显存开销是
`L × (4·n·h + 小张量)`，`residual` 比 `layer_input` 大 `M = 4` 倍，
这就是要省的地方。

`mhc_multilayer_recompute` 的做法是：**只保留 `initial_residual` 与各层
“算得出来的东西”（`pre_mix` / `post_mix` / `comb_mix` / `layer_output`），
在反向时按层重算 `layer_input` 与 `residual`。**

它的接口因此接受一串列表，并把它们变成 GPU 上的指针表：

```python
pre_mix_ptrs, layer_input_ptrs, residual_ptrs, layer_output_ptrs, post_mix_ptrs, comb_mix_ptrs = \
    _make_ptr_tables_batched([pre_mix_list, layer_input_list, residual_list,
                              layer_output_list, post_mix_list, comb_mix_list],
                             device=initial_residual.device)
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/multilayer_recompute_kernel.py:57-67`）kernel 里用
`T.make_tensor(layer_input_ptrs[i_layer], ...)` 把“一个地址”还原成张量
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/multilayer_recompute_cuda.py:72-73` 以及 62-69、115-116），
然后逐层跑同一个递推：

```python
for i_layer in T.serial(L_post):
    ...
    T.clear(layer_input_local)
    for i_mhc in T.serial(mhc):
        for i1_h in T.Parallel(h_blk):
            layer_input_local[i1_h] += pre_mix_local[i_mhc] * res_local[i_mhc, i1_h]
    T.copy(layer_input_local, layer_input_tensor[i_n, i0_h])
    ...
    for i_mhco, i1_h in T.Parallel(mhc, h_blk):
        new_res_local[i_mhco, i1_h] = post_mix_local[i_mhco] * layer_output_local[i1_h]
        for i_mhci in T.serial(mhc):
            new_res_local[i_mhco, i1_h] += comb_mix_local[i_mhci, i_mhco] * res_local[i_mhci, i1_h]
    T.copy(new_res_local, output_residual_tensor[i_n, 0, i0_h])
```

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/multilayer_recompute_cuda.py:71-113`）注意最后那一步：

```python
for i_mhc, i1_h in T.Parallel(mhc, h_blk):
    res_local[i_mhc, i1_h] = T.cast(T.cast(new_res_local[i_mhc, i1_h], T.bfloat16), T.float32)
```

（`:111-112`）残差在层间要**先回 bf16 再转回 fp32**。因为前向保存下来的下一层
输入就是 bf16，重算必须复现这个精度的往返，否则数值对不上。
这是一个“重计算（recompute）”实现里非常典型、也非常容易漏的细节。

**这一段在论文里是 §4.3.2（Recomputing）**，而且论文把“存什么、重算什么”
列成了 Table 3：

```text
   论文 Table 3（paper-deepseek-mhc.md:510-531）——每个 token 的激活
     x_l0（l0 是 L_r 层块的第一层）   大小 nC   存储方式：每 L_r 层存一次
     F(H^pre x_l, W_l)               大小 C    存储方式：每层都存
     x_l                             大小 nC   重算（块内瞬态）
     H^pre x_l                       大小 C    重算
     RMSNorm(H^pre x_l)              大小 C    重算
   论文说「excluding lightweight coefficients」——也就是 pre/post/comb
   这些轻量系数不在上面这张表里（论文原话在 `:509`：“Excluding lightweight
   coefficients …”）
```

这与 kernel 的接口完全对得上：`initial_residual` 是 `x_{l0}`、
`layer_output_list` 是每层都存的 `F(·)`、`pre_mix/post_mix/comb_mix` 是被排除在
表外的轻量系数，而 kernel 的**输出** `residual_list` / `layer_input_list`
正是那些“块内瞬态”的重算结果。

论文还给了块长的选择依据：令 `L_r` 为块内层数，总显存 ≈
`nC·⌈L/L_r⌉ + (n+2)C·L_r`，最小化后得到

$$
L_r^*\approx\sqrt{\frac{nL}{n+2}}
$$

这是论文式 (20)，见 `:534-537`。

并说明“重算块不能跨流水级边界”，于是实际做法是**让重算边界与流水级对齐**
（`:539`）。本章对这个公式只做到“读懂”：`code/03` 是单层的 CPU 复算，
没有实现多层重算，也没有验证 `L*_r` 的最优性。

为什么这样做值得？`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/mhc/test_multilayer_recompute.py:99-108` 把两条路径的
IO 字节数摆在一起：分开调用要反复读写每层的 `residual`（`M` 倍于 `layer_input`），
融合重算只保留一份 `initial_residual` 加上按层的小张量。**用一次额外的前向计算
换掉 `L` 份 `[n, M, h]` 的读写**，在显存紧张的训练里，这笔交易通常划算。

### 6.6 kernel 与训练 step 的对应

把 §6.2–6.5 拼起来（图 5 有同一张图）：

```text
   前向（每个含 mHC 的层一次）
     expand（首层） ─► prenorm GEMM（DeepGEMM） ─► reduce_partials_and_rmsnorm
        ─► pre_big_fuse = split_mixes + Sinkhorn + pre_apply_mix  ─► layer_input
        ─► Attention + MoE ─► layer_output
        ─► post ─► 新的 residual
   反向（逆序）
     pre_apply_mix bwd ─► sinkhorn bwd ─► pre_split_mixes bwd（写部分和）
        ─► *_bwd_reduce ─► head_compute_mix bwd ─► expand bwd
   重算（替代保存中间激活）
     multilayer_recompute
```

顺带把“谁用什么写”说清楚，这是本章 §7 判断法的实例：

| 组件 | 实现语言 | 出处 |
| --- | --- | --- |
| mHC 的 mix / Sinkhorn / pre_apply_mix / post / expand / recompute | **TileLang** | `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/`（本节全部） |
| prenorm GEMM | 昇腾侧是 **Ascend C 主体 + 一段 TileLang** | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:27-28,190`；TileLang 实现见 `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/deep_gemm/tilelang_ops/hc_prenorm.py` |

“谁用什么写”这件事，论文自己给出了更权威的版本：**除了式 (14)–(15) 的 prenorm
GEMM 之外，其余 kernel 都用 TileLang 实现**
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:502`）。昇腾侧的落地就是本仓库的做法，
GEMM 交给 DeepGEMM-Ascend（Ascend C 主体 + 一段 TileLang），其余算子在
TileKernels 里。**这条边界是“论文 + 实现”双重印证的**，也是 §7.3 判断法的样板。

论文对这套工程设计给出的总账是：`n = 4` 时，大规模训练只增加
**6.7% 的时间开销**（§4.3 `:411`，§1 也提到 `:219`）。
**引用这条时必须带上条件**：它是论文自报的**内部大规模训练**结果，
文本抽取里没有说明加速器型号、并行规模与序列长度，也不是昇腾上的数字；
本章无法复现，只把它当作“融合 + 重算 + 重叠三件事合起来值多少”的量级参考。

DeepGEMM-Ascend 的 README 给了一张 prenorm GEMM 的性能表（**厂商自报，作者未复现**）：

| M | N | K | Latency (us) | Computation (TFLOPS) | Memory bandwidth (GB/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 13 | 24 | 28672 | 6.7 | 3 | 519 |
| 137 | 24 | 28672 | 9.6 | 20 | 1110 |
| 512 | 24 | 28672 | 15.4 | 46 | 2082 |
| 4096 | 24 | 28672 | 72.7 | 78 | 3276 |
| 8192 | 24 | 28672 | 136.7 | 82 | 3463 |

（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:166-176`）注意 `N = 24`，正是
`mhc_mult3`。这张表的读法是：**算力利用率只有 3–82 TFLOPS，而带宽到 519–3463 GB/s**，
而且 README 自己说它“nearly saturates the HBM write bandwidth”
（`:166-168`）。换句话说，这个 GEMM 也在往**带宽**而不是算力上顶。
本章引用它只为说明“这条链是访存主导”，具体数值属于第 35 章的范围。

### 6.7 论文 §4.3.3 与 §5：DualPipe 上的重叠，以及那些数字的实验条件

论文的第三条基础设施设计是**在 DualPipe 调度里错开通信与计算**：

```text
   论文 §4.3.3（paper-deepseek-mhc.md:541-554）的三个动作：
   ① 把 MLP 层的 F_post,res kernel 放在专用高优先级计算流上，避免堵住通信流（:552）
   ② 长耗时的 attention 算子不用 persistent kernel，以便被抢占（:552）
   ③ 重算与流水通信解耦——每一级的初始激活 x_l0 已经缓存在本机（:554）
```

这条设计落在 **DualPipe** 这个仓库里，而不是 TileKernels：DualPipe 的双向流水
调度本身就是为“前向/反向计算与通信完全重叠”设计的
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DualPipe/README.md:1-9`）。
本章只把它点到为止，**DualPipe 与 EP 通信重叠是第 38 章的内容**，
这里只需要记住：`multilayer_recompute` 的重算块边界必须与流水级对齐（§6.5），
正是为了让 §4.3.3 说的这种重叠能成立。

最后把论文 §5 的收益数字集中列一次，**每一条都带条件**：

| 论文报告的数字 | 出处 | 实验条件（论文自述） |
| --- | --- | --- |
| `n = 4` 时大规模训练只增加 **6.7%** 时间开销 | `:411`、`:219` | 内部大规模训练；论文未在文本里给出加速器型号 |
| 27B 模型上相对 baseline 的最终 **loss 降低 0.021** | `:683` | DeepSeek-V3 风格 MoE；27B；数据量与参数量成比例（`:557-565`） |
| 相对 HC 进一步提升：**BBH +2.1%**、**DROP +2.3%** | `:685` | 同上；benchmark 与 shot 数见 Table 4（`:571-681`） |
| 单层/复合映射的 Amax Gain：HC 峰值 **≈3000** → mHC 复合后 **≈1.6** | `:288`（HC）、`:711`（mHC） | 27B 模型；Sinkhorn 迭代 20 次 |
| Sinkhorn 迭代次数 **t_max = 20**；`n = 4`；`α` 初始化 **0.01**；Layer Norm ε **1e-20** | 式 (9) `:405-406`；Table 5 `:1355-1380`（各行分别在 `:1360,1365,1370,1375`） | 论文的实验超参 |

**本章不引用**论文里任何未给出条件的加速/吞吐数字，也不把上述 27B 的结果
当作昇腾上的预期。那是两次不同的实验。

---

## 7. 这类算子的共同特征与写法

### 7.1 共同特征

把 Engram 门控、两处 Sinkhorn、mHC 的 mix 算子放在一起，会看到同一组特征：

| 特征 | 在代码里的表现 |
| --- | --- |
| **访存为主** | 每个元素只做几次乘加；瓶颈是搬进来的字节数与 kernel 启动次数 |
| **算力需求低** | 没有 GEMM 里的 `M×N×K` 三级循环；用不上 Cube / Tensor Core 的主力 |
| **shape 小** | `4 × 4`、`[T, 4]`、`24` 维，连 `hidden_size` 都是要被切开的维度 |
| **latency-bound** | 夹在 GEMM / Attention 之间，必须尽快交出结果，吞吐不是首要目标 |
| **状态小但迭代多** | Sinkhorn 的迭代状态全在片上；`repeat` 是编译期常量 |
| **逐 token 并行** | 并行维度是 token（和 head），归约维度几乎总是“很小”的那一维 |
| **参数梯度要二次归约** | 逐 token 的 partial gradient 必须再有一个 reduce kernel |

判断一个算子属不属于这一类，最快的办法是问三个问题：

1. **它的 FLOPs 和 bytes 哪个大？** 如果算术强度远低于机器的 `算力/带宽`，
   它就是访存主导的。
2. **它的归约维度有多长？** 如果归约维度小到能放进一个 CTA 的片上存储，
   那么“一个 kernel + 片上迭代”几乎总是对的。
3. **它是不是在关键路径上被反复调用？** 是的话，kernel 的**数量**和**启动开销**
   往往比单次 kernel 的效率更重要。

### 7.2 TileLang 里对应的写法

这一章出现的原语不多，但每一个都对应一类需求。下表都核对过 `v0.1.15`：

| 需求 | TileLang 写法 | 本章例子 | 定义位置（`_vendor/tilelang-v0.1.15`） |
| --- | --- | --- | --- |
| 固定核数、循环取任务 | `T.Persistent(extent, num_cores, core_id, group_size=…, num_stages=…)` | 昇腾版门控、mHC 的 mix 算子 | `tilelang/language/loop.py:82` |
| 多阶段缓冲 | 手写 `phase` + `T.async_copy` + `T.ptx_wait_group`（CUDA）；`T.annotate_buffer_versions({buf: n})`（昇腾） | 门控前向 / 反向的两遍扫描 | `tilelang/language/copy_op.py:188`；`tilelang/ascend/language/annotations.py:19` |
| warp 内归约 | `T.warp_reduce_sum` | 门控前向的 `Σx² / Σk² / Σx·w·k` | 第 9 章 |
| 跨 warp / 跨阶段归约 | `T.alloc_reducer` / `T.reducer_init` / `T.reducer_update` / `T.finalize_reducer` | `pre_apply_mix` 反向后、`pre_split_mixes` 反向 | `tilelang/language/allocate.py:296`、`tilelang/language/reduce_op.py:358,398,434` |
| 向量内建（昇腾） | `with T.SimdVF():` / `T.SimtVF(threads=…)` + `tilelang.ascend.language.simd as S` | 门控前向的 `S.vld/vmula/vcadd/vsel/vsts` | `tilelang/ascend/language/frame.py:107,179` |
| 动态 shape | `T.dynamic('num_tokens')` 与 `T.const('hidden_size')` | 所有 kernel 的签名 | `tilelang/language/symbolics.py:13` |
| 运行时才知道的地址 | `T.ptr` + `T.make_tensor(ptr, shape, dtype)` | `multilayer_recompute` 的指针表 | `tilelang/language/proxy.py:265,325` |

其中最后一项值得单独说：**把“一串张量的地址”做成指针表传进 kernel**，
是“层数维度的循环”唯一的表达方式。因为不同层的张量是各自分配的，
只有地址能整齐地排成一个 `T.Tensor[(L,), T.ptr]`。代价是
**kernel 内的类型信息全靠 `T.make_tensor` 手写**，写错不会在编译期被抓住。

### 7.3 什么时候应该下钻到 Ascend C / PTX

TileKernels 自己给了答案：**同一个仓库里，mix/Sinkhorn 这类算子是纯 TileLang，
而 prenorm GEMM 走 DeepGEMM-Ascend，主体是 Ascend C + MAD 原语抽象。**
mHC 论文在这一点上和实现完全一致：它是**先用一句“excluding Eq. (14) to (15)”
把 prenorm GEMM 划出去、再说其余都用 TileLang**（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:502`），
也就是“论文作者自己也认为只有那一段需要下钻”。

DeepGEMM-Ascend 的 README 把理由写成了它的卖点：

> DeepGEMM Ascend provides a lightweight abstraction over the Ascend MAD
> (matrix multiply-add) primitives, hiding much of the complexity associated with
> fractal layouts, alignment constraints, address calculations, and verbose
> low-level parameters. … makes extensive use of Ascend-specific optimization
> techniques, such as sparse data loading and coroutine-based pipelining, to
> approach the performance limits of the Ascend hardware.
> （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:5`）

翻译成判断法：**当“离硬件上限还差多少”成为主要矛盾，而这件事取决于
分形布局、512 字节对齐、手工地址计算、指令级流水（协程、稀疏加载）时，
就该下钻。** 这些不是 TileLang 表达不了，而是它们需要比“声明式 tile”
更细的控制；“接近硬件极限”的收益来自把这些细节都抠到位。

反过来，本章这些算子的瓶颈在**搬多少**、**启动几次**、**迭代怎么展开**，
而这些恰好是 TileLang 的强项（`T.Persistent` + 多缓冲 + `T.Parallel` + `T.reduce_*`）。
所以它们用 TileLang 写，既够快又能被两条后端共享同一份数学。

至于 PTX / 内联汇编，本章只有一个间接的接触面：CUDA 侧的
`T.ptx_wait_group` 是 TileLang 暴露出的 `cp.async` 等待原语；更深的 PTX
（例如第 37 章 DeepSelect 用来压低常数的位运算技巧）不属于 TileLang 主线，
只在“TileLang 编出来的东西还不够好”时才有必要碰。

### 7.4 一张决策表

```text
   新算子该用什么写？

   ① 它是不是矩阵乘 / 卷积这类「算力主导」的算子？
        是 ─► 先把 TileLang 的 T.gemm 写出来，量它的算力利用率
                ├─ 已经接近上限 ─► 收工
                └─ 差得远     ─► 再考虑像 DeepGEMM 那样下钻（MAD / 分形 / 协程）
        否 ─► ② 它的并行维度和归约维度分别是什么？
                ├─ 归约维度能放进一个 CTA 的片上存储 ─► TileLang，一个 kernel 搞定
                └─ 归约跨 CTA                        ─► TileLang + 一个 reduce kernel

   ③ 它是不是「每个 token 一个小矩阵、还要迭代」这类结构？
        是 ─► 把迭代写成编译期常量（T.serial / T.unroll），
              buffer 用 annotate_buffer_versions 或手写 phase 多缓冲

   ④ 它的性能瓶颈是不是「指令级细节」（对齐、分形、行长、bank 冲突、
      手工地址计算）？
        是 ─► 下钻到 Ascend C / PTX；否则留在 TileLang
```

一句话总结本章的判断法：**先问“瓶颈是字节还是指令”，再问“归约跨不跨 CTA”。**
字节 + 跨 CTA ⇒ TileLang 加一个 reduce kernel；指令 ⇒ 下钻。

---

## 8. 常见坑

这一节把本章读到的坑集中列出来。**这些都是阅读源码时的推断，不是本机踩出来的**
（本机没有加速器）。

**P1 把 `sinkhorn_cuda.py` 里的 `hidden_size` 当成模型的 hidden 维度。**
它实际是 mHC 的 `M`（`= 4`）。同一族代码里 `pre_split_mixes` 又用 `mhc_mult3`
表示 24。**读 TileKernels 时先看构造函数签名，再相信变量名。**

**P2 忘了 `hc_mult = 4` 是硬约束。** 门控前向/反向、`head_compute_mix`、
`pre_big_fuse` 的昇腾版都直接 `assert hc_mult == 4`。
给 `M ≠ 4` 的模型套这些 kernel 会直接断言失败，而不是“慢一点”。

**P3 以为 `O(1)` 查表等于“通信成本可忽略”。** `O(1)` 只是取行数不随表增长；
这些行从 host memory 到 HBM 的搬运仍然要走 PCIe，能不能盖住取决于**放在第几层**
（论文 §2.5）。论文的 < 3% 是在特定放置下测出来的。

**P4 把 `grad_w_partial` 当成最终梯度。** 它的第一维是 CTA 数；
必须过 `grad_w_reduce` 才会变成参数梯度，而且那个 kernel 是**累加**语义
（配合 `main_grad`）。

**P5 只写 `T.clamp` 不管反向。** 前向里的
`clamp(|z|, clamp_value)` 在边界上会截断梯度；反向必须显式写
`T.Select(|z| < clamp_value, 0, …)`（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_bwd_cuda.py:249-253`）。
漏掉这一步，梯度会在边界附近悄悄变大。

**P6 在 `inplace=True` 的同时开 `save_for_backward`。** 前向把结果写回
`hidden_states`，反向再拿它当输入就算错了；源码用一句 `assert` 挡住
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_cuda.py:26`）。

**P7 忘了 `eps` 会留下固定偏差。** mHC 的 Sinkhorn 每步都是
“除以 `(sum + eps)`、再加 `eps`”，所以行列和永远不会精确等于 1。
`code/02` 里的 `max|col_sum − 1| ≈ 1.1e-6` 就是 `eps` 的下限，
再迭代也不会更好。

**P8 在反向里重算前向而忽略精度往返。** `multilayer_recompute` 必须把残差
先 cast 回 bf16 再转回 fp32（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/multilayer_recompute_cuda.py:111-112`），
否则重算出的 `residual` 与真实前向的 bf16 版本不一致，梯度就会偏。

**P9 以为 `T.Persistent` 的 `group_size` 随便给。** 昇腾门控前向里
`group_size=num_head_groups`（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_fwd_asc.py:80`）是为了让同一个 token
的不同 head 组分到不同核、且调度顺序稳定；写错会让 `T.copy` 的
tile 假设失效（它在 `:107-121` 一次性搬 2 个 head）。

**P10 把 `repeat` 当成越小越好。** `repeat` 直接决定 mix 矩阵离双随机有多远，
而 forward 反向都要重跑它。默认 10 是“精度 vs 延迟”的折中，
不是“随便调大调小都行”。

**P11 用 `T.ptr` 指针表时把 shape 写错。** 指针表把类型信息交给
`T.make_tensor`，编译器不会帮你检查；`multilayer_recompute` 里
`(n, h)` 与 `(n, mhc)` 的混用一旦写反，症状是**静默的错值**而不是编译错误。

**P12 Sinkhorn 反向的中间量会吃掉 shared memory。** 它要存 `2 × repeat` 份
`[token_block, M, M]`；`token_block_size` 因此被压到 8（CUDA）/ 4（昇腾）。
调大 block 之前先算一下 shared memory 放不放得下。

**P13 拿代码里的 `repeat = 10` 当论文的 `t_max = 20`。** 两个数不只是大小不同：
论文式 (9) 的一次迭代 = 行归一化 + 列归一化（`:402-406`），而代码把
起点那次列归一化也算进 `repeat`（`mhc/sinkhorn_cuda.py:35-52`）。
要对论文口径，要么把 `repeat` 调到 20 以上并在意那 `O(eps)` 偏移，
要么明确说清“这是仓库默认值”。见 §5.2 的对照表。

**P14 忘了 `H^post = 2σ(·)` 里的那个 2。** 论文式 (8) 是 `2σ`
（`:393-399`），代码把它做成参数 `mhc_post_mult_value`，而仓库测试的默认值是
`1.0`（`tests/mhc/test_pre_big_fuse.py:23`、`pre_split_mixes_cuda.py:62`）。
生产调用必须显式传 2.0；拿默认值跑，`post` 项只有论文强度的一半。

**P15 以为“先 RMSNorm 再乘”和“先乘再除范数”可以随便换。** 论文 §4.3.1 明确
把这一步换序并声明数学等价（`:415-417`），因为换序之后 RMSNorm 的权重可以被吸进
投影矩阵 `φ_l`、并且能和 GEMM 的 split-K 部分和一起归约。代码确实这么做了
（`mhc/norm_fn_kernel.py:64-99`），但**本章只在静态层面核对，没有数值验证**
两种顺序的等价性。如果你要改动这条链，先补一个和 `torch/mhc.py` 基线的对拍。

---

## 9. 本章小结

把这一章压成七句话：

1. **Engram 是第二条稀疏轴。** MoE 用条件计算扩容量，Engram 用条件记忆扩容量；
   两者的容量分配存在 U 型的最优比例（论文自报：约 75%–80% 给 MoE）。
2. **确定性寻址是它的工程命门。** index 只依赖 token id ⇒ 可以提前预取 ⇒
   可以把大表放到 host memory 甚至更慢的介质上。`O(1)` 指的是“每次取几行”，
   不是“延迟可忽略”。
3. **门控 kernel 的融合来自一个恒等式。** `RMSNorm(x,w_h)ᵀRMSNorm(k,w_e)
   = rstd_x·rstd_k·Σx·(w_h⊙w_e)·k`，所以 `w_h⊙w_e` 可以提前算成
   `weight_fused`，kernel 里一次扫描就够。
4. **CUDA 与 Ascend 版本算的是同一个式子。** 差别只在“谁切任务、谁管同步、
   向量多宽”：`T.Kernel(4, blocks)` + 手写相位 vs `T.Persistent` +
   `annotate_buffer_versions`。
5. **参数梯度要二次归约。** 逐 token 的 kernel 只能产出
   `grad_w_partial[num_ctas, hc, d]`；跨 CTA 的求和 + 穿过 ⊙ 的链式法则由
   `grad_w_reduce` 完成。mHC 的 `pre_split_mixes_bwd_reduce` 是同一种模式。
6. **Sinkhorn 有两处，别混。** mHC 是“softmax + 行/列求和归一化 → 双随机”，
   每个 token 一个 4×4、全部在片上；Engram 是“行/列 L2 缩放”，
   一张大矩阵、拆成四个 kernel。
7. **mHC 是“论文定义 + 工程实现”的两层结构。** 论文 §4.1 把 `H^res` 约束到
   Birkhoff 多胞形、§4.2 给出参数化与 Sinkhorn 迭代、§4.3 给出三条基础设施设计
   （kernel fusion / recomputing / DualPipe 重叠）；TileKernels 的 `mhc/` 是它们在
   昇腾上的落地。两者**不一致**的地方有四处（Sinkhorn 起点与首步顺序、迭代次数
   `20` vs `10`、`eps`、`post = 2σ` vs `mhc_post_mult_value = 1.0`），本章逐条列出。
8. **论文自己划出了 TileLang 的边界**：除式 (14)–(15) 的 prenorm GEMM 之外，
   其余 kernel 都用 TileLang（`paper-deepseek-mhc.md:502`）。这与“GEMM 下钻、
   结构性算子留在 TileLang”的判断法互为印证。
9. **论文的收益数字必须带条件读。** `n = 4` 时 6.7% 的训练开销、27B 上 0.021 的
   loss 降低、BBH +2.1% / DROP +2.3% 都是论文自报，硬件型号未在文本中给出，
   **不是昇腾上的数字**；本章一条也没复现。

当你能看着一个新算子，先算出它的访存与算力比、再判断它的归约跨不跨 CTA，
你就能回答本章最重要的问题：**这个算子该不该用 TileLang 写，以及要拆成几个 kernel。**

## 延伸阅读

* `references.md` 的 **TK** 组：本章逐行读过的所有 kernel 与它们的确切行号。
* `references.md` 的 **EN** 组：Engram 论文文本抽取的段落索引，
  以及 `Engram/README.md` 里那句“只是 demo”的原文。
* `references.md` 的 **MHC** 组：mHC 论文（`paper-deepseek-mhc.md`）的 §3–§5 段落
  索引，含 §4.3 三条基础设施设计与 §5 的实验条件。
* `references.md` 的 **DG** 组：HC prenorm GEMM 的性能表与“mHC kernel 由 TileLang
  支撑”的出处；它在第 35 章会被展开。
* `code/01`–`code/04`：四个可以在 CPU 上跑的复算脚本，用来核对本章的代数与形状。
* 第 33 章：`*_kernel` / `*_cuda` / `*_asc` 三份文件的结构与运行期后端选择。
* 第 38 章：DualPipe 与 EP 通信重叠，本章 §6.7 点到为止的部分在那里展开。
* 第 35 章：DeepGEMM-Ascend 的 MAD 抽象、稀疏加载与协程流水，
  也就是本章 §7.3“什么时候下钻”的另一半。
