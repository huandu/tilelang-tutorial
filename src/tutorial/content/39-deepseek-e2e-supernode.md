# 第 39 章 端到端：128 卡超节点上的训练与推理

## 学习目标

读完本章，你应当能够：

1. 把第 32–38 章讲过的六个组件**拼回两条完整链路**，训练一步（前向到反向）与
   推理一次请求（prefill 与 decode），并说清每一步“在什么单元上、搬多少字节、
   算什么”。
2. 把“算力 / 片内带宽 / 卡间带宽”三类资源列成一张**预算表**，说清每一项在
   什么 EP 规模、什么形状下会先成为瓶颈；同时说得出这张表里**哪两个数字不能
   相减、不能相乘**，以及为什么。
3. 拿到一个“慢的 kernel / 慢的一步”时，能按一个**可执行的检查顺序**定位瓶颈
   （先判类 → 再看哪条 PIPE 满 → 再看通信有没有重叠 → 再看 layout/对齐 → 最后看
   启动开销），并指回具体章节去找工具与判据。
4. 用第 17、18、28 章的方法论说清“从能跑到跑满”的五个阶段各自的目标与判据。
5. 背出这套生态的**现状清单**：哪些能力已可用、哪些是 experimental、哪些没实现，
   并知道每一条的依据在哪一章。
6. 把第 32–38 章暴露出的缺口映射回第 30、31 章的贡献路径，说清“读完本教程之后，
   第一步该做什么”。

## 前置知识

本章是第五部分、也是全书的收口章。它**不引入新的技术细节**：所有结论都必须能在
前面章节或 corpus 的一手材料里找到依据。因此前置知识是下面这三组：

* **前面七章（第 32–38 章）**：这是本章的主要依据。本章每个小节开头都会写明
  “这一段收的是第 N 章的哪几条结论”，你不需要重新读一遍，但需要能对上号。
* **第 16、17、18 章**：7B 模型走过的端到端（训练、推理与部署）。本章复用它们的
  账本口径（`6·N·D`、算术强度、TTFT/TPOT），只是把规模换成 128 卡超节点。
* **第 28 章**：算子设计方法论与性能模型（roofline 的三种正确用法、tile 级解析模型、
  反模式清单）。本章第 5 节的检查顺序就是它在“一整套流水线”上的用法。

不需要你事先读过 mHC 论文或 DualPipe 的调度表：第 2 节会用到的部分会在这里
讲清，并给出 corpus 路径与行号。

---

## 本章的验证边界（务必先读）

本章沿用第五部分的统一边界，并且因为它是收口章，边界更严：

1. **本机没有昇腾 NPU、没有 NVIDIA GPU、没有 CANN**（见
   [`../../VERSIONS.md`](../../VERSIONS.md)）。`code/` 下的三个示例都是**纯 Python、
   只用标准库**的账本程序，在写作机上**实际运行过**；它们复算的是**公式**，
   不是硬件。凡是需要真机才能得到的数字，本章一律不写。
2. **本章出现的每一个性能数字都来自指名道姓的一手来源，并连带它的测量条件**：
   要么是某个 README / 技术报告里的表格（附路径与行号），要么是第五部分里
   **本书自己复算过**的量（会写明“本书复算”）。
3. **不同来源、不同机器、不同口径的数字，本章不做相减、不做相乘、不做除。**
   本章只在一处给出“比值”，并且明确标注它是**量级比值、只用于判类**，
   两个数字的出处与口径都写在旁边（见 §4.4）。
4. 本章的**结论**（哪一类资源会先成为瓶颈、瓶颈该按什么顺序查）都标注了来源
   章节；凡是本章自己没有验证能力、只能转述上游“计划 / 进行中”措辞的地方，
   一律保留原文措辞，并在 §7 的现状清单里单独列出。

一句话：**这是一章“把已知的东西拼起来”的章，不是一章“带来新数字”的章。**

---

## 1. 收口：这一章要回答两个问题

### 1.1 前面七章各自留下了什么

第 32 章给了一张地图，后面六章一章拆一个组件。收口之前，先把“每一章交出来的
那句结论”并排摆一次，本章后面每用到一条，都要回到这张表里的出处。

| 章 | 组件 | 它交出来的结论 | 本章用在哪 |
| --- | --- | --- | --- |
| 32 | 全栈地图 | 六个组件分四层：语言/编译、运行时、计算算子、通信；“一一对应”是接口层的 | §2.1、§3.1 |
| 33 | TileKernels（上） | 一份算法、两个后端；共享的是算法与契约，重写的是数据怎么在芯片上流动 | §2.2 |
| 34 | TileKernels（下） | Engram 与 mHC 是“不是矩阵乘”的算子：访存为主、逐 token 并行、迭代是编译期常量 | §2.3、§2.4 |
| 35 | DeepGEMM-Ascend | 算力主导的算子用 Ascend C + MAD 下钻到 98%–99%；访存主导的那一块（HC prenorm）留给 TileLang | §2.2、§4.2 |
| 36 | FlashMLA 昇腾 | 稀疏 attention 的每个结论背后都有一段代价计算；TileLang 能表达 A 类、B 类要下钻、C 类必须手写 | §3.2、§5.2 |
| 37 | DeepSelect | 算法决定上限（`N + O(k log(N/k))`），硬件决定常数；指标是有效内存带宽 | §3.2、§3.3 |
| 38 | DeepEP-Ascend | 通信库自己不用 TileLang；它把 `topk_idx` 与 `fp8 x + sf` 交给两端；重叠分三层 | §2.5、§4.2 |

### 1.2 本章的纪律

正因为前面七章已经把细节讲完，本章的写法必须比它们更克制。三条纪律：

* **只收口、不展开。** 遇到需要新细节的地方，宁可少写一句，也不在本章现场推导。
* **每次引用都指回章节或 corpus。** 本章大量使用“见第 N 章”与
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/...` 两种引用方式。
* **数字必须带条件。** 尤其是 §4 的预算表：每一项都写清“哪台机器、什么版本、
  什么形状、什么口径”。

### 1.3 路线图

```text
§2 训练链路 ──┐
              ├──▶ §4 预算账 ──▶ §5 瓶颈检查清单 ──▶ §6 从能跑到跑满
§3 推理链路 ──┘                                             │
                                                            ▼
                                        §7 现状清单 ──▶ §8 收口到贡献
```

前两节（§2、§3）回答“**这套东西怎么协同工作**”，后四节（§4–§8）回答
“**拿到一个真实任务时，怎么判断瓶颈在哪、下一步做什么**”。这正是本章要交的
两份东西。

---

## 2. 训练侧的一条完整链路

### 2.1 一个含 mHC 的 MoE 层的 forward

第 16 章把 MoE 拆成四段（router/gating → dispatch/permute → grouped GEMM →
combine），第 32 章又把这段放进一次训练 step 的调用骨架里。把两者合起来，
一个**含 mHC 的 MoE 层**的 forward 是八步（图 1）：

1. **mHC pre**（TileLang，第 34 章）：把 `n` 条残差分支混合成本层的输入 `x`；
2. **路由 / TopK**（TileLang，第 33 章）：算出 `topk_idx` 与 `topk_weights`；
3. **dispatch**（Ascend C，第 38 章）：把 token 送到专家所在的 rank；
4. **grouped GEMM ①**（Ascend C + MAD，第 35 章）：gate/up 投影；
5. **SwiGLU + 量化**（TileLang，第 33 章）：激活并转成下一步要的精度；
6. **grouped GEMM ②**（Ascend C + MAD，第 35 章）：down 投影；
7. **combine**（Ascend C，第 38 章）：把专家输出按 `k` 路归约回原 rank；
8. **attention / MLP 本体** `ℱ` 之后，**mHC post**（TileLang）把结果写回 `n` 条残差分支。

![图 1：训练一步的全栈流水](figures/01-train-step-timeline.md)

（源描述见 [`figures/01-train-step-timeline.md`](figures/01-train-step-timeline.md)；
SVG 规格 `src/tutorial/figures/39-deepseek-e2e-supernode/01-train-step-timeline.json`。）

### 2.2 每一步“在哪张卡上、搬多少字节、算什么”

把上一步展开成一张表。**这张表同时也是本章的一份索引**：每一行都必须能对回
第 32–38 章里的某一节。

| 段 | 谁在执行 | 实现语言 | 算什么 | 搬什么（量级） | 出处 |
| --- | --- | --- | --- | --- | --- |
| mHC pre | 全部 rank，逐 token | **TileLang** | 两次扫描 + RMSNorm + mix | 读 `(3n+1)C` → 融合后 `(n+1)C` | 第 34 章 §6.2；mHC 论文 §4.3.1 |
| 路由 / TopK | 全部 rank，逐 token | **TileLang** | 打分 + 选 `k` 个专家 + 权重 | 读 router 权重 `d_model × E`，写 `[T, k]` | 第 33 章 §4；`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/moe/topk_gate_kernel.py:7-43` |
| dispatch | 跨 rank，all-to-all | Ascend C（DeepEP） | 零 FLOP：把 token 搬到专家所在 rank | 每 token 每方向 `k·d·b`；FP8 时 `6×7168×1 ≈ 42 KB` | 第 38 章 §1.2；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:9-21` |
| grouped GEMM ① | 专家所在 rank | Ascend C + MAD | `2·M·N·K` | 每组读一次 B（专家权重） | 第 35 章 §5.4 |
| SwiGLU + 量化 | 专家所在 rank | **TileLang** | 逐元素激活 + 转精度 | 读 hidden，写 hidden/2 | 第 33 章 §5.5；`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/quant/swiglu_forward_kernel.py:40-50` |
| grouped GEMM ② | 专家所在 rank | Ascend C + MAD | `2·M·N·K` | 同 ① | 第 35 章 §5.4 |
| combine | 跨 rank，all-to-all | Ascend C（DeepEP） | 零 FLOP + `k` 路 FP32 归约 | 每 token 每方向 `k·d·b`；BF16 时 `6×7168×2 ≈ 84 KB` | 第 38 章 §1.2、§7.3 |
| mHC post | 全部 rank，逐 token | **TileLang** | 超连接写回 + 残差合并 | 写从 `3nC` 降到 `nC` | 第 34 章 §6.3；mHC 论文 §4.3.1 |

三个可以直接读出来的事实：

1. **TileLang 覆盖的是“两端 + 中间的一小段”**：路由、激活/量化、超连接是纯
   TileLang；最贵的两段 GEMM 与两段通信**不在** TileLang 里（第 35、38 章都给了
   理由）。所以官方那句“每个 TileLang 算子昇腾上都有对应实现”讲的是**覆盖率**，
   不是“所有算子都用 TileLang 写”（第 32 章 §2.3）。
2. **两段零 FLOP 的搬运，字节量与 GEMM 同量级**。第 38 章 §1.2 已经算过：
   一层 dispatch + combine 到 GB 量级，和这一层 GEMM 的访存是同一量级。
   这是 128 卡超节点上“通信是主角”的直接来源。
3. **四段算子会互相等待**。dispatch 要等所有 rank 的 token 到齐，而“最慢的 rank
   决定整步时间”（第 16 章 §2.6）。所以重叠不是锦上添花，是训练能不能跑满的前提。

### 2.3 backward：为什么是“重算 + 梯度归约”

反向不是把前向倒过来写一遍那么简单。原因是 mHC 把残差流从 1 条变成 `n` 条
（论文用 `n = 4`），于是需要保存的中间激活按 `n` 倍增长（mHC 论文 §3.2 的
Table 2 与正文，`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:296-345`）。论文给出的两条对策，
正好对应仓库里的两类 kernel。

**（1）激活存不下，所以重算。** 论文的做法是：前向结束后**丢掉** mHC kernel 的
中间激活，反向时**重新执行一遍 mHC kernel**（不含很重的层函数 `ℱ`）。
重算块的最优长度由式 (20) 给出（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:504-539`）：

$$
L_r^*\approx\sqrt{\frac{nL}{n+2}}
$$

这里 \(L_r\) 是重算块包含的层数，\(L\) 是总层数。

论文还给了一条工程约束：**重算块不能跨流水线的 stage 边界**，所以它把重算边界与
流水线 stage 对齐（同上一段，原文 “we choose to synchronize the recomputation
boundaries with the pipeline stages”）。

在仓库里，这一步对应 TileKernels 的 `multilayer_recompute` kernel（第 34 章 §6.5）。
它接受一串“按层的张量指针表”，在反向时逐层重跑 `layer_input` 与 `residual` 的
递推。一个容易忽略的细节：残差在层间要**先回 bf16 再转回 fp32**，否则重算出来的
数值与前向保存的对不上（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/multilayer_recompute_cuda.py:111-112`）。

**（2）参数梯度要二次归约，所以要多一个 kernel。** Engram 门控与 mHC 的 mix 系数
都是“逐 token 算出来的小矩阵”；反向往回求梯度时，每个 token 只贡献一个**部分和**，
必须再有一个 kernel 把它们归约成真正的参数梯度（第 34 章 §4.3）。这一步没有数学
捷径，归约维度是 token 数，跨 CTA，躲不掉。

把这两条合起来，训练一步的 backward 形状是：

```text
前向保存：每 L_r 层一份 x_{l0}，加上每层的小张量
反向：combine 的反向（= dispatch）→ GEMM 反向 → SwiGLU 反向 → mHC 反向
      → 梯度归约 kernel → 重算 kernel（补上被丢掉的中间激活）
```

### 2.4 mHC 与 Engram 在这条链上的位置

两个“不是矩阵乘”的算子在链路里各占一个位置，都很容易被漏掉：

* **mHC** 是一层的外壳：pre 在本层输入之前，post 在本层输出之后，`ℋ_res` 决定
  `n` 条残差分支怎么混合。它替换的是**残差连接本身**，所以每一层都要执行一次。
* **Engram** 是一条挂在残差流上的旁路（第 34 章 §1.2）：按确定的 hash 地址
  **O(1) 查表**，把查到的记忆向量门控后加回残差。它的价值在于“确定性寻址可以把
  大表卸载到 host memory”，代价是每层一次查表 + 一次门控。

这两个算子都是**访存为主、逐 token 并行、迭代次数是编译期常量**的形态
（第 34 章 §7.1）。所以它们用 TileLang 写，既够快，又能被两个后端共享同一份数学。
反过来，它们也是“kernel 数量”比“单次 kernel 效率”更重要的典型：第 34 章 §3.1
说得很直接，门控前向把三个累加器的两遍扫描融成一个 kernel，省下的主要是
**启动开销**与**重复读 `x` 的带宽**。

### 2.5 论文 §4.3 的三条基础设施结论

mHC 论文的 §4.3“Efficient Infrastructure Design”是一节工程实现说明，它给出的
三条结论正好对应本章要用的三个机制。三条都来自
`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:408-563`：

| 小节 | 论文做法 | 收益（论文口径） | 条件 |
| --- | --- | --- | --- |
| §4.3.1 Kernel Fusion | 把“除以范数”重排到矩阵乘之后；按 Eq.(10)–(13) 实现三个专用 kernel（`ℋ_pre`/`ℋ_post`/`ℋ_res`）；把共享同一份内存访问的多个操作融进一个 kernel；Sinkhorn 的迭代放在**一个 kernel 内**，反向用一个自定义 kernel **在片上重算中间结果** | 读的元素从 `(3n+1)C` 降到 `(n+1)C`，写的从 `3nC` 降到 `nC`（`...:485-500`） | `n = 4`；论文明确说**除 Eq.(14)–(15) 以外的大部分 kernel 用 TileLang 实现**（`...:502-503`） |
| §4.3.2 Recomputing | 前向后丢弃 mHC 中间激活，反向重执行 mHC kernel（不含 `ℱ`）；每 `L_r` 层只需常驻保存一份 `x_{l0}`；`L_r* ≈ sqrt(nL/(n+2))`；重算边界与流水线 stage 对齐 | 峰值显存由 `nC·⌈L/L_r⌉ + (n+2)C·L_r` 决定，按上式取最优（`...:504-539`） | 重算块不得跨 stage 边界（`...:539`） |
| §4.3.3 Overlapping Communication in DualPipe | 把 MLP 层的 `ℱ_post,res` kernel 放在**专用高优先级 compute stream** 上执行，以免阻塞通信流；attention 层的长算子**不使用 persistent kernel**，以便被抢占；重算与流水线通信解耦（因为 `x_{l0}` 已在本地缓存） | 论文配图（Fig. 4）是对 DualPipe 调度的扩展 | 论文明确标注图里各段长度“illustrative only and do not represent actual duration”（`...:547-551`） |

**收益数字的读法。** 论文给的一个综合数字是：`n = 4` 的 mHC 在大规模模型上
“introduces only a 6.7% additional time overhead”（摘要，
`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:219`；§4.3 开头复述为“a marginal training
overhead of only 6.7%”，`...:410-411`）。引用它必须带上四个条件：

1. 这是**论文自报的 in-house 大规模训练**结果，既不是本章的实测，也不在
   TileKernels 仓库里；
2. 它是 `n = 4` 的 mHC 相对**同规模基线**的额外时间开销，不是绝对吞吐；
3. 它是**经过 §4.3 三条优化之后**的数字，论文 §3.2 先说明“HC increases the
   memory access cost by a factor approximately proportional to `n`”，然后才说
   §4.3 的优化把它压回来（`...:345`、`...:411`）；
4. 论文的对比模型族是 3B / 9B / 27B 与 27B-1T（§5.1），本章不把它外推到别的规模。

### 2.6 DualPipe 与 EPLB：训练侧的两个调度件

前面几节讲的是“一层怎么跑”。整步怎么跑，靠两个**调度件**。它们不是算子，
所以本章只为它们各留一段，细节仍见第 38 章 §5.4。

**DualPipe** 是双向流水并行算法，出处是 DeepSeek-V3 技术报告。仓库
`../tilelang-corpus/14-deepseek-ascend-stack/raw/DualPipe/README.md:3` 的自我定位是“achieves full overlap of forward and
backward computation-communication phases, also reducing pipeline bubbles”。
收益写在同文件的气泡对照表里：同样 PP stage 数下，气泡从 1F1B 的 `(PP-1)(F+B)`
降到 DualPipe 的 `(PP/2-1)(F&B+B-3W)`，代价是**每卡参数 2×、激活 `PP+1`**。
注意表里的 `F&B` 表示“前向与反向两块互相重叠后的耗时”，气泡变小与
“重叠成立”是同一件事。

**EPLB** 是专家负载均衡器，`../tilelang-corpus/14-deepseek-ascend-stack/raw/EPLB/README.md:3-8` 说明它采用**冗余专家**策略复制
热点专家，再按“分组受限路由”把同组专家尽量放到同一个节点上，以减少跨节点流量。
公开接口是 `eplb.rebalance_experts(...)`，返回 `phy2log / log2phy / logcnt`
三张映射表。

两者在本章的链路里各管一段：**DualPipe 管“跨 stage 的重叠”，EPLB 管“跨 rank
的负载”**。它们都需要真实的多机环境才能验证，本章只登记位置，不做性能断言。

### 2.7 训练一步的调用骨架

把 §2.1–§2.6 写成伪代码，就是第 32 章 §5.4 那张骨架的完整版。这里逐行标注它
调用了哪个组件、来自哪一章：

```python
# —— ① mHC pre：n 条残差分支 → 本层输入（TileLang，第 34 章）
x = mhc_pre_apply_mix(residual, pre_mix)               # tile_kernels/mhc/pre_apply_mix_*

# —— ② 路由：算出 topk_idx / topk_weights（TileLang，第 33 章）
topk_idx, topk_weights = moe_topk_gate(x, gate_weight)  # moe/topk_gate_kernel.py

# —— ③ dispatch：异步发起，先不取结果（DeepEP-Ascend，第 38 章）
recv = ep_buffer.dispatch(x_fp8, topk_idx, topk_weights, defer_epilogue=True)

# —— 与通信重叠：算共享专家或其它不依赖 recv 的东西（第 38 章 §3.4）
shared_out = shared_expert(x)

# —— ④ ⑤ 两个 grouped GEMM + SwiGLU（DeepGEMM-Ascend + TileKernels）
h = m_grouped_gemm(recv.x, recv.weights)                # 第 35 章 §5.4
h = swiglu_quant(h)                                     # 第 33 章 §5.5
y = m_grouped_gemm(h, down_weight)                      # 第 35 章 §5.4

# —— ⑥ combine：同样异步发起（DeepEP-Ascend，第 38 章）
combined = ep_buffer.combine(y, handle, async_with_compute_stream=True)

# —— ⑦ attention / MLP 本体之后，再由 mHC post 写回（TileLang，第 34 章）
layer_out = attention_or_mlp(combined + shared_out)
residual = mhc_post(layer_out, residual, post_mix, comb_mix)

# —— ⑧ 反向：combine 的反向是 dispatch、dispatch 的反向是 combine（第 38 章 §2.3）
#      再加 mHC 的梯度归约 kernel 与 multilayer_recompute（第 34 章 §4.3、§6.5）
```

这份骨架里有两行“看起来多余、其实不能删”：`defer_epilogue=True`
（把发起与收尾拆开）和 `async_with_compute_stream=True`（让通信与计算真并行）。
第 38 章 §3.4 有一条硬约束：**`defer_epilogue=True` 必须同时设
`async_with_compute_stream=True`，否则 C++ 侧直接断言失败**
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/csrc/buffers/ep.hpp:398`）。

### 2.8 字节账的三种口径，比较之前先对齐

上一节表里的字节量有**三种互不相等的口径**，混用会让结论直接错掉。
这是第 38 章 §7.4 讲的“physical payload”的推广，值得在这里正式说一次：

| 口径 | 定义 | 典型用途 | 陷阱 |
| --- | --- | --- | --- |
| **A · 逻辑 payload** | 每 token 每方向 `k·d·b` | 与链路带宽上限比，算“吃到上限的百分之几” | 忽略了去重：不是每个 token 真的会去 `k` 个不同的 rank |
| **B · 展开后的行数** | `Σ_e ceil(n_e / align) × align` | 决定 grouped GEMM 的 grid 与显存 | 含 padding；`do_zero_padding=False` 时尾部是脏数据 |
| **C · 全集总量** | 所有 rank 的 payload 求和 | 估算一次 dispatch 的整体规模 | 随 EP 规模变化的量，不能当成“每卡”的量 |

**口径 A 的一个修正项（本书复算）。** 第 38 章 §1.3 给出了“一个 token 实际
覆盖的不同目标 rank 数”的期望：

$$
\operatorname{distinct\_ranks}(R,k)
=R\left(1-\left(1-\frac{1}{R}\right)^k\right)
$$

这是 \(R\) 个 rank、均匀路由时的期望。

把测试条件（`k=6`，`R` 取 8 / 128）代进去：

| EP 规模 R | `distinct_ranks` | 相对 `k=6` 的比例 |
| ---: | ---: | ---: |
| 8 | ≈ 4.41 | 73.5% |
| 16 | ≈ 5.14 | 85.6% |
| 32 | ≈ 5.55 | 92.5% |
| 64 | ≈ 5.77 | 96.2% |
| 128 | ≈ 5.88 | 98.1% |

（`code/01-train-step-budget/` 会用同一公式打印这张表。**这是本书的公式复算**，
它解释的是“为什么 EP 越大越慢”的第一条原因，不是任何实测值。）

**为什么这个修正项重要？** 因为它正是第 38 章 §7.2 第一条原因的定量版本：
EP 从 8 涨到 128，同一个 token 跨卡的次数从 4.41 涨到 5.88，
**跨设备字节量上升 33%**（`5.88 / 4.41 ≈ 1.33`），而有线带宽固定。
再叠加“每 rank 专家变少、固定开销占比上升”与“barrier 成本随 rank 上升”，
就是那张带宽表里 dispatch 单调下降的全部原因。

所以：**引用任何一个通信数字之前，先问它是 A / B / C 哪一种口径。**
第 38 章 §7.4 拆的那句“90–95% of the physical payload bandwidth limit”
用的是口径 A，并且明确不含 padding、不含 barrier。这就是它必须拆开读的原因。

### 2.9 训练侧的三层重叠

第 38 章 §9.3 把重叠分成三层，讲的是 DeepEP 的视角。站在整步的角度再看一次，
三层各自的“谁负责”与“怎么验证”是这样的：

| 层次 | 重叠的是什么 | 谁负责 | 怎么验证 |
| --- | --- | --- | --- |
| **kernel 内** | dispatch 的 issue 与本地拷贝；combine 的发送与归约 | DeepEP-Ascend 的 Ascend C kernel | 看 kernel 自身的流水图；第 38 章 §4.3–§4.5 |
| **调用级** | “这一次 dispatch”与调用方后续的独立计算 | 使用者的代码：`defer_epilogue=True` + `async_with_compute_stream=True` | 通算流水图上两段是否真的并行；第 38 章 §3.4 |
| **流水线级** | 前向 chunk 与反向 chunk 的计算-通信 | DualPipe 的调度骨架 | 气泡公式与重叠区；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DualPipe/README.md` 的对照表 |

mHC 论文 §4.3.3 做的正是**在第三层上再加两条规则**（`...:541-563`）：

1. **把 MLP 层的 `ℱ_post,res` kernel 放到专用高优先级 compute stream 上**，
   目的是“prevent blocking the communication stream”；
2. **attention 层的长算子不使用 persistent kernel**，目的是让它可被抢占
   （“preventing extended stalls ... allowing for flexible scheduling”）。

论文还说明了为什么这两条能成立：**重算与流水线通信解耦**，因为每个 stage 的
初始激活 `x_{l0}` 已经缓存在本地（同上一段）。这一句把 §2.3 的“重算”与
§2.6 的“DualPipe”连了起来，**重算不是额外负担，它恰好绕开了通信依赖**。

三层可以叠加，互相不冲突（第 38 章 §9.3）。但要注意一件事实：
**第一、二层在本章是“有依据但未验证”的**（需要真机与多卡环境），
第三层还需要一台真的 PP 规模的机器。**本章不为任何一层给实测数字。**

---

## 3. 推理侧的一条完整链路

### 3.1 一次请求的两段

推理分两段，第 18 章 §2.1 已经定义过：**prefill** 一次处理整段 prompt，
**decode** 每次生成一个 token。第 32 章 §6.1 把这两段各自要用到的组件排成了
一张表。本节做的是**收口**：把它压成两条链路，然后回答“为什么两段的瓶颈类型
不同”。

DeepSeek 这一代模型走 **DSA（DeepSeek Sparse Attention）**：不是让每个 query
关注所有 KV，而是先给 KV 打分（Lightning Indexer）、选出 top-k，再只对这些 KV
做注意力。于是推理链上多了一段“先筛选、再计算”（第 16 章 §4.3、第 36 章 §1.1）。

![图 2：prefill 与 decode 的组件链路对照](figures/02-inference-chains.md)

（源描述见 [`figures/02-inference-chains.md`](figures/02-inference-chains.md)；
SVG 规格 `src/tutorial/figures/39-deepseek-e2e-supernode/02-inference-chains.json`。）

### 3.2 prefill 链

prefill 的五段，以及每段的“谁写、卡在哪”：

| 段 | 组件 | 实现语言 | 卡在哪 | 出处 |
| --- | --- | --- | --- | --- |
| ① 索引打分（MQA logits） | DeepGEMM-Ascend | Ascend C | **FixPipe**：官方原话“FIX-pipe bound rather than compute bound, saturating the FIX pipe at 99% utilization” | 第 35 章 §5.5；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:142-149` |
| ② TopK 筛选 | DeepSelect | Ascend C | **有效内存带宽**（TopK 不做浮点运算，所以不用 FLOPs 做指标） | 第 37 章 §6.1；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:41-44` |
| ③ 稀疏注意力 | FlashMLA | Ascend C | **L0C / UB / MTE / FixPipe 的赛跑**；官方口径 prefill 达 410 TFlops（硬件峰值 95%） | 第 36 章 §1.1、§5.7；`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:13,53` |
| ④ 专家并行通信 | DeepEP-Ascend | Ascend C | **卡间带宽**；dispatch 在 EP≤32 时达物理 payload 带宽上限的 90–95% | 第 38 章 §7.1、§7.4 |
| ⑤ 常规算子 | TileKernels | **TileLang** | 启动开销与重复读；融合的收益主要来自这两项 | 第 33、34 章 |

prefill 的形状是**大 M**：一次进来几千个 token，算术强度高，所以②③是
compute/FixPipe 主导，④是唯一可能把它拖住的通信段。第 35 章 §5.5 给了一组
能直接对照的数字：prefill 的 MQA logits 是 `Q=4096, K=8192, heads=32, dim=128`，
FP8 下 302.6 µs、681 TFLOPS、283 GB/s。

### 3.3 decode 链

decode 只有 5 段里的 4 段（索引打分要复用/增量，注意力换成 decode 版本、KV cache
换成分页量化格式），但每一段的形状都掉到了另一个区间：

| 段 | 组件 | decode 与 prefill 的差别 | 出处 |
| --- | --- | --- | --- |
| ① 索引打分 | DeepGEMM-Ascend | 同一条 kernel、`Q` 从 4096 掉到 256：算力从 681 掉到 565 TFLOPS，**带宽从 283 涨到 2006 GB/s** | 第 35 章 §5.5；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:142-149` |
| ② TopK 筛选 | DeepSelect | 同一场景（Lightning Indexer，bf16，`topk=512`）；昇腾只支持 bf16 | 第 37 章 §6.2；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:17-28,37` |
| ③ 稀疏注意力 | FlashMLA | 官方口径 decode 达 360 TFlops（硬件峰值 83%）；**只支持 FP8 / FP4 KV cache，不支持 bf16** | 第 36 章 §5.7；`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:63,136` |
| ④ 专家并行通信 | DeepEP-Ascend | 同样的 dispatch/combine，但 batch 小、**每次 launch 的固定开销占比显著上升** | 第 38 章 §7.2、§8.4 |
| ⑤ 常规算子 | TileKernels | 逐元素算子变成 memory-bound；采样 TopK 走 DeepSelect 的采样场景（`vocab ≈ 128K`） | 第 18 章 §2.2；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:31-38` |

decode 的形状是**小 M**：第 35 章 §5.3 用同一张表证明了“M=128 与 M=4096 是两种
问题”，M 小时权重矩阵只用一次，算术强度掉到机器平衡点附近（251 vs 247
FLOP/byte），于是瓶颈从 Cube 变成“把 B 搬进来”。

### 3.4 为什么两段的瓶颈类型不同

把上面两节收成一句话：**瓶颈是形状的函数，不是算子的属性。**

第 28 章 §2.2 给的判据是 `AI = flops / bytes` 与平衡点 `峰值算力 / 峰值带宽`
两个数比大小。同一个算子，只改变量 `M`，AI 就跟着变：

| 形状（第 35 章 §5.3 的两行） | 算术强度（本书复算） | 在 865 TFLOPS / 3500 GB/s 的平衡点（约 247）下 | 该优化什么 |
| --- | --- | --- | --- |
| FP8 推理 GEMM，`M=4096, N=7168, K=16384` | `2·4096·7168/11264 ≈ 5213` | 远高于平衡点 → compute-bound | Cube 已到 99.5%，别动 tile |
| FP8 推理 GEMM，`M=128`（同 N、K） | `2·128·7168/7296 ≈ 251` | **就在平衡点上**（251 vs 247） | 减少 B 的搬运（权重常驻、L2 hint） |
| MQA logits（prefill 与 decode 都是） | 用这两个 roof 读不出来 | 两个比值都不饱和 | **第三条流水线：FixPipe**（见下） |

这里的“平衡点”用的是第 35 章 §5.1 里从 `HWDesc` 复算出来的两个数
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/csrc/jit_kernels/heuristics.hpp:46-71`）：`gm_bw_gbps = 3500`，fp8 峰值 865 TFLOPS。
**这张表是本书的复算，不是任何一次实测**；它的用途只有一个，
说明“同一个算子在两个形状下会被两条不同的流水线卡住”。

第三行还有一个必须讲清的例外：**算术强度高不等于一定卡在 Cube 上。**
MQA logits 每个 MAD 产出的 L0C 数据能换来的额外计算很少（`head_dim` 只有
64/128，点积结果要立刻搬出去），而输出 logits 是个稠密大矩阵，于是
`L0C → out` 成了上界。所以它的两个比值（§3.7）谁都不饱和。
第 35 章 §5.5 与第 36 章 §7.2 都指出这是**同一个单元、同一类等式**：
FixPipe 的 128 B/cycle 在这个平台上反复出现。判读方法是看 profiler 里的
`aic_fixpipe` 高不高、`aic_mad` 高不高（第 19 章 §8）。

### 3.5 一次 decode step 的调用骨架

第 32 章 §6.4 已经给过一份骨架，这里把它与“哪一段归谁”对齐：

```python
# 每个 decode step 之前（或第一次调用时）准备一次调度元数据
tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata()

for step in decode_loop:
    # —— ① 索引打分：算 q 与 KV 的 MQA logits（Ascend C，第 35 章）
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(q, kv_cache, ...)

    # —— ② 选出 top-k KV（Ascend C，第 37 章）
    _, indices = deep_select.topk(logits, topk, indices_type=torch.int32, return_value=False)

    # —— ③ 稀疏注意力：只对选中的 KV 做 decode（Ascend C，第 36 章）
    out, lse = flash_mla.flash_mla_with_kvcache(
        q, kvcache, block_table, cache_seqlens, dv,
        tile_scheduler_metadata, num_splits, indices=indices,
    )

    # —— ④ MoE 推理的专家通信（Ascend C，第 38 章）
    # —— ⑤ 常规算子穿插其间（TileLang：norm / RoPE / 量化 / 采样，第 33、34 章）
```

（`flash_mla_with_kvcache` 与 `indices` 的用法见 `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:111-151`；
`deep_select.topk` 的签名见 `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:85-99`。）

这一段骨架里有三个“只能记住、不能推理出来”的约束，全部来自上游 README：

* **decode 只支持量化 KV cache**（FP8 / FP4），bf16 KV cache
  “is not supported”（`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:136`）。
* **prefill 版没有 batch 维**，多 batch 要自己重排（`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:172`）。
* **TopK 的输出可能不是连续的**：`deep_select.topk` 的输出 stride 按
  `get_stride_requirement()` 对齐，可能非连续（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:97-99`）。

第 18 章 §7 讲的三条指标（TTFT / TPOT / 吞吐）在这里各对应一段：**TTFT 由
prefill 链决定，TPOT 由 decode 链决定**，而吞吐是两个阶段的混合。所以
“哪一段慢”这件事，必须先说清问的是哪个指标。这是第 18 章 §7.3 那份
“报告前必须写下来的清单”里排第一位的原因。

### 3.6 推理链上最容易被漏掉的一环：KV cache

上面两条链里没有一格写着“KV cache”，但它决定了 decode 能跑多快。
把第 16、18、36 章的三段结论拼起来：

* **缓存什么，由模型结构决定。** MLA 缓存的是压缩后的潜向量
  `c_KV(512) + k_R(64)`，而不是每个 head 的 K/V。第 16 章 §3.3 给的结论是
  KV 降到 MHA 的 1.76%，decode 的算术强度从 1.0 抬到 241.8 FLOP/Byte。
* **缓存多少，由序列长度决定。** 第 16 章 §7 的表里，MLA decode 每 token 要读
  `L·(d_c+d_r)·b` 字节；`L` 是十万级时，这是每步都要付的固定成本。
* **缓存放在哪，由 kernel 决定。** 第 18 章 §2.6 讲了为什么要分页（paged KV）；
  第 36 章 §5.2 讲了 FlashMLA 昇腾版的 **Decoding KV Cache Layout**，
  把 data 与 scale 相邻放置，让一次搬运能同时拿到两者。

对本章的链路而言，KV cache 有三个直接后果：

1. **decode 是 memory-bound 的根因在这里**：权重每次只用一次、KV 每次都要读
   （第 18 章 §2.3“decode 的权重底线”）。
2. **量化 KV 不是可选项**：FlashMLA 的 decode 内核**只支持 FP8 / FP4 KV cache**，
   bf16 “is not supported”（`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:136`）。所以“昇腾上能跑
   decode”这条结论，隐含了“你接受了量化 KV”这个前提。
3. **它是 §5 检查清单里第 ④ 步的常客**：KV 的 layout（NZ 与否、
   data/scale 是否相邻、page 大小）会直接决定第 36 章 §3.4 讲的那句
   “源和目标都连续才会触发 BIU 合并”成立与否。

顺带说清一个常见误解：**prefill 不需要 KV cache**（它一次算出整段 KV），
所以 prefill 与 decode 的成本模型从一开始就不同。这也是第 18 章 §2.1
把它们分成两个阶段的原因。

### 3.7 把两条链各算成一个数：两个比值的用法

判类只需要两个比值：`Computation / 算力上限` 与 `Memory bandwidth / 带宽上限`。
下面是**用第 35 章 §5 表里的两列**（官方 README 给出的 Computation 与
Memory BW），除以第 35 章 §5.1 从 `HWDesc` 复算出的两个常量
（fp8 峰值 865 TFLOPS、HBM 3500 GB/s）。**除法是本书做的，两个分子来自同一张
官方表格，两个分母来自同一份源码**：

| 形状（来自第 35 章的六张表） | Computation | Memory BW | 算力占比 = ÷865 | 带宽占比 = ÷3500 | 判读 |
| --- | ---: | ---: | ---: | ---: | --- |
| Dense FP8 GEMM，`M=4096` | 861 TFLOPS | 186 GB/s | 99.5% | 5.3% | **算力**（差 0.5%） |
| M-Grouped FP8×FP4，每组 `M=4096` | 861 TFLOPS | 180 GB/s | 99.5% | 5.1% | **算力** |
| FP8 推理 GEMM，`M=128` | 549 TFLOPS | 2456 GB/s | 63.5% | 70.2% | **两列都在一半以上** |
| MQA logits prefill，`Q=4096` | 681 TFLOPS | 283 GB/s | 78.7% | 8.1% | **都不是** → 第三条路（FixPipe） |
| MQA logits decode，`Q=256` | 565 TFLOPS | 2006 GB/s | 65.3% | 57.3% | **两列都在一半以上** |
| HC prenorm，`M=8192` | 82 TFLOPS | 3463 GB/s | 9.5% | 98.9% | **带宽** |

这张表是本章 §3.4 那句“瓶颈是形状的函数”的可操作版本。读法有三条：

1. **一个比值 78.7%、另一个只有 8.1%** 的那一行（MQA prefill），说明瓶颈在
   **第三类单元**上。官方对此的说明是 FIX-pipe（第 35 章 §5.5）。这正是
   “只看算力和带宽两个 roof 是不够的”的实证。
2. **两列都在一半以上**的那两行（`M=128` 的 GEMM、decode 的 MQA logits），
   说明这个形状**处在大 M 与小 M 之间**：谁也不能吃掉谁，优化要同时动两边。
   这正是第 35 章 §5.3 说的“处在这个点上，读 B 的时间和算的时间同量级”。
3. **占满一列、另一列很低**的两行（大 M GEMM、HC prenorm）是最容易判的，
   也最容易优化。

**顺带做一个口径检查。** 上表第一行的 861 TFLOPS 可以用 `2·M·N·K` 复算出来：

$$
2\cdot4096\cdot7168\cdot16384=9.62\times10^{11}\ \mathrm{FLOPs}
$$

$$
\frac{9.62\times10^{11}}{1117.6\ \mu\mathrm{s}}
\approx860.8\ \mathrm{TFLOPS}
$$

结果与 README 的 Computation 列（861）一致。

所以 dense GEMM 那一列的口径是“按 `2MNK` 计的理论 FLOPs ÷ 实测 kernel 时间”。
但**同样的复算套到 MQA logits 上对不上**：按 `2·Q·K·heads·dim` 算出的值
比表里的 681 TFLOPS 高约三分之一，而 README 没有给出这一列的计算公式。
按第 35 章 §5.3 的处理方式，本章**只报这个事实、不做解释**，并以表格列为准。

这段的意义不在数字本身，而在那个动作：**拿到任何一张表，先用一个能自己复算的
行去标定它的口径。** 第 14 章要求 benchmark 写清条件，这条就是它的实操版。
`code/02-inference-latency-model/` 会把这六个形状的判类过程打成一张可运行的账。

---

## 4. 128 卡超节点的预算账

### 4.1 三类资源，三套口径

先说清楚本章的“预算表”是什么：它把这套方案要花的资源分成三类，
**每一类只和自己比**，绝不跨类运算。

| 资源 | 它决定什么 | 本章用的量 | 一手出处 | 口径 |
| --- | --- | --- | --- | --- |
| **算力** | 大 M 形状下的时间下界 | bf16 432.5 / fp8 865 / fp4 1730.1 TFLOPS | 第 35 章 §5.1；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/csrc/jit_kernels/heuristics.hpp:46-71` | **本书复算**：`flops_per_cycle × 1650 MHz × 32 核`，与 README 的 `Hardware limit` 列吻合 |
| **片内带宽** | 访存主导的算子（逐元素、超连接、小 M GEMM）的时间下界 | HBM 3500 GB/s、L2 4500 GB/s | 第 35 章 §5.1；同文件 `HWDesc` | 源码里的**硬件模型常量**，不是实测；单卡口径 |
| **卡间带宽** | dispatch / combine 的时间下界，也是 EP 的规模上限 | dispatch 313–375 GB/s、combine 272–347 GB/s | 第 38 章 §7.1；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:9-21` | **厂商自报实测**：Ascend 950DT + CANN 9.2.0 + PoC HDK，netlayer 1，16384 tokens/rank、hidden 7168、top-6/256 experts，10 warmup + 50 samples、刷 cache |

![图 3：三类资源的预算与口径](figures/03-resource-budget.md)

（源描述见 [`figures/03-resource-budget.md`](figures/03-resource-budget.md)；
SVG 规格 `src/tutorial/figures/39-deepseek-e2e-supernode/03-resource-budget.json`。）

三行分别来自三个不同性质的来源，这正是本章要读者建立的习惯：
**“算力”是一个模型常量，“片内带宽”也是模型常量，“卡间带宽”才是实测。**
把三者当成同一种“芯片规格”来比较，是这一节最容易犯的错。

### 4.2 把这套方案的每一段对号入座

把 §2.2 的八段按“它主要吃哪一类资源”重排一次，就得到这张分档表。
**分档依据是第 32–38 章各自的结论，不是本章的测量**：

| 段 | 主资源 | 依据（哪一章说了它是这类） |
| --- | --- | --- |
| grouped GEMM ①②（大 M） | **算力** | 第 35 章 §5.4：M-grouped 在每组 `M=4096/8192` 时到 94.6%–99.7% 算力利用率，带宽只有 98–180 GB/s |
| grouped GEMM ①②（小 M，decode） | **片内带宽** | 第 35 章 §5.3：`M=128` 时算力只有 48%–63%，带宽到 1829–2456 GB/s |
| MQA logits | **FixPipe**（既不是算力也不是 HBM） | 第 35 章 §5.5：`aic_fixpipe` 99% 利用率 |
| 稀疏 attention（prefill） | **算力 + FixPipe** | 第 36 章 §3.3：三条 bound 里 FixPipe 先到；prefill 410 TFlops / 95% |
| 稀疏 attention（decode） | **片内带宽 + 启动开销** | 第 36 章 §1.3、§3.4：KV 的 ND2NZ 与 gather 是赛跑；`B_TOPK` 小、并行度低 |
| dispatch / combine | **卡间带宽** | 第 38 章 §7：EP≤32 时 dispatch 已达链路物理上限的 90–95% |
| 路由 / TopK、量化 / 激活、超连接、Engram | **片内带宽 + 启动开销** | 第 33 章 §7.1、第 34 章 §7.1：访存为主、逐 token 并行、kernel 数量比单次效率更重要 |
| mHC prenorm GEMM | **片内带宽（HBM 写出）** | 第 35 章 §5.7：形状 `[M, 24, 28672]`，带宽 519–3463 GB/s，官方说它“nearly saturates the HBM write bandwidth” |

### 4.3 什么规模下，哪一项先成为瓶颈

把上表按规模收成三条判据。三条都只用到“同一来源内部”的比较，并且都指回章节：

**（1）EP 规模变大时，卡间带宽先成为瓶颈。** 这不是猜测：第 38 章 §7.1 的表
里，dispatch 从 EP8 的 373–375 GB/s 单调降到 EP128 的 313–320 GB/s，combine
从 345–347 降到 272–278 GB/s；而第 38 章 §7.2 给出的三条原因（去重率下降、
每 rank 专家数变少导致固定开销占比上升、barrier 成本随 rank 数上升）
每一条都随 EP 规模恶化。所以“通信成为主角”是**规模推出来的**，
不是一句定性描述。

**（2）M 变小时，片内带宽先成为瓶颈。** 第 35 章 §5.3 用同一张表的两行证明了
这件事：`M=4096` 的两行算力 854–861 TFLOPS（98.7%–99.5%）、带宽只有 137–186
GB/s；`M=128` 的两行算力降到 415–549 TFLOPS、带宽升到 1829–2456 GB/s。
**同一个 kernel，量级直接掉一半、带宽涨十倍**。这是 decode 与 prefill
要用不同策略的直接原因。

**（3）FixPipe 是一条独立的第三条路。** 第 35 章 §5.5 的 MQA logits 与第 36 章
§3.3 的 1C2V bound 都指向同一个单元。它的判据很简单：**看 `aic_fixpipe` 与
`aic_mad` 谁高**。两个都高才轮到算力；只有 fixpipe 高，就去想“能不能少搬、
或者搬到别处再算”（第 19 章 §8.3 的决策树）。

### 4.4 三条不许做的运算

这一节是本章的“安全规则”。前面三节的所有数字，如果被错误地组合，都会得出
看起来很对、其实不成立的结论。三条禁令：

1. **不许把不同来源的数字相减。** 例如“3500 GB/s 的 HBM 带宽减掉 374 GB/s
   的链路带宽”没有任何物理含义：前者是源码里的硬件模型常量，后者是
   特定条件下的实测 payload 带宽，单位相同不代表可以相减。
2. **不许把不同来源的数字相乘。** 例如“用 99.8% 的 dense GEMM 利用率
   去乘另一个形状的 TFLOPS”，利用率的分母是那个形状自己的硬件模型
   （第 35 章 §5.1），换形状就失效。
3. **不许跨机器比较。** 第 32 章 §4.3 已经列过一次反例：NVIDIA 侧 DeepGEMM 的
   “up to 1550 TFLOPS on H800”（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM/README.md:25`，2025-04-18 的旧数字）
   与昇腾侧的数字**不能放在一起比**。

**唯一允许的一次除法**（本章只在这里做，并且标注为“量级比值、只用于判类”）：

```text
同一台机器（Ascend 950DT）上，单卡 HBM 带宽 3500 GB/s（模型常量）
                         vs  EP8 dispatch 373–375 GB/s（实测）
量级比 ≈ 3500 / 374 ≈ 9.4 倍

用途：回答「这张卡上，片内搬一次和跨卡搬一次差几个量级」——答案是约一个数量级。
不可用于：预测任何具体算子的端到端时间。
```

这个比值能说出的结论只有一句：**在 128 卡这个规模上，让数据跨卡走一趟的代价，
和让它在本卡内存里走十趟相当**。所以第 38 章 §9.4 的那句总结，
“TileLang 负责算、DeepEP-Ascend 负责搬”，在预算上是有据可依的。

`code/01-train-step-budget/` 会把 §2.2 的八段按这张表算成一份可运行的账本，
并把每个数字的出处打印在旁边；它**只做公式复算，不产生任何实测结论**。

### 4.5 一张“规模 → 瓶颈”的对照表

把 §4.2 的分档与 §4.3 的三条判据交叉一次，得到一张可以贴在工位上的表。
它的每一行都只引用了同一来源内部的比较：

| 你在什么规模 | 先到瓶颈的是 | 数字依据（同源） | 该做什么 |
| --- | --- | --- | --- |
| 单卡、大 M（训练 / prefill） | **算力** | dense GEMM 98.3%–99.8%、M-grouped 94.6%–99.7%（第 35 章 §5.2、§5.4） | 调 L1/L0 流水深度、别动 tile 尺寸 |
| 单卡、小 M（decode） | **片内带宽** | `M=128` 时算力 48%–63%、带宽 1829–2456 GB/s（第 35 章 §5.3） | 减少 B 的搬运：权重常驻、L2 hint |
| MQA logits（两端都有） | **FixPipe** | `aic_fixpipe` 99%（第 35 章 §5.5） | 少搬、或换个地方算 |
| 超长上下文 decode | **KV 带宽** | 每 token 读 `L·(d_c+d_r)·b`（第 16 章 §7） | 量化 KV、分页、split-KV |
| EP ≤ 32 | 通信**还能藏住** | dispatch 达物理 payload 上限的 90–95%（第 38 章 §7.4） | 把它重叠掉，别再优化它 |
| EP ≥ 64 | 通信**变成主角** | EP64 combine 294–298、EP128 combine 272–278 GB/s（第 38 章 §7.1） | 换并行策略，或走 MegaMoE 式的融合（第 35 章 §5.6） |
| 一层里有很多小算子 | **启动开销** | 逐元素/超连接类算子的共同特征（第 34 章 §7.1） | 融合，先数 kernel 数量 |

读这张表的方法是**先定位自己在哪一行，再去看那一列的数字**。
反过来，先看数字再找行，是最容易犯的错，因为同一个数字在不同行里
代表的东西完全不同（第 28 章 §2.3 的 M7）。

### 4.6 哪些数字是本书复算的，哪些不是

第五部分的数字分四类，引用时的措辞必须跟着变。本章把它们集中列一次，
方便你在写自己的报告时对号入座：

| 类别 | 例子 | 本章怎么写 |
| --- | --- | --- |
| **本书复算的公式** | 三档硬件峰值（432.5 / 865 / 1730.1 TFLOPS）、`distinct_ranks` 表、§3.7 的两个比值 | 写“**本书复算**”+ 公式 + 参数从哪来 |
| **同源内的比较** | §4.5 那张表的每一行 | 写清“同一来源内部”＋两侧的出处 |
| **上游自报的实测** | EP8–EP128 带宽表、`410 / 360 TFlops`、`2–20×`、`6.7%` | 写“**厂商/论文自报**”+ 硬件、版本、形状、口径 |
| **不在本章范围内的** | MQA logits 的 `Computation` 列公式、`Memory BW` 列的口径 | 明确写“**口径不明，只报事实**” |

前三类的处理方式在本章各有先例（§2.5、§3.7、§4.1）；第四类只有一条处理原则：
**不要用“我觉得应该是”去补上游没写的东西**，
第 35 章 §5.3 与本章 §3.7 都演示了这条原则该怎么落笔。

顺带登记三条**本章核不到、只能写“待核”**的内容，它们都继承自第 32 章：

1. **DeepSelect 的昇腾硬件要求**：这份快照的 README 没有像其它仓库那样写明
   “Ascend 950 / CANN 版本”（第 32 章“待核事项”第 1 条）。
2. **FlashMLA README 引用的模型页**：写作环境无法访问 HuggingFace，
   该页面未能核实（同第 2 条）。
3. **128 卡超节点的硬件细节**：官方文章只有一句话，corpus 里没有对应白皮书
   （同第 3 条）。本章 §4.1 里“卡间带宽”那一行因此只写了实测口径，
   没有写任何互联规格。

---

## 5. 瓶颈定位的检查清单

### 5.1 六步检查顺序

拿到一个“慢的 kernel / 慢的一步”，最忌讳的是直接去改 tile 尺寸。第 28 章 §7
的反模式清单里，排在最前面的两条就是“不判类就调参”和“拿错 roof 定价”。
所以这里给一个**从上到下、每一步都能独立否掉上一步**的顺序（图 4）：

```text
① 先判类：算 flops/bytes，和平衡点比大小（第 28 章 §2.2）
      ↓ 判类结果决定下面四条里先查哪一条
② 是 compute-bound？看哪条 PIPE 满：aic_mad / aic_fixpipe / aiv_vec / mte2
      ↓ 如果 PIPE 都不满，说明它在等
③ 在看通信有没有重叠：defer_epilogue / DualPipe / 三层重叠（第 38 章 §9.3）
      ↓ 如果重叠了还是慢
④ 看 layout 与对齐：ND2NZ、512 B 对齐、bank 冲突、fractal（第 21、36 章）
      ↓ 如果这些都对
⑤ 看启动开销与小 kernel：融合、入图、persistent kernel（第 26、35、38 章）
      ↓ 如果还慢
⑥ 回到 ①，用第 14 章的对拍方法复核：是不是算错了一个字节（第 28 章 M8）
```

![图 4：瓶颈定位的检查顺序](figures/04-bottleneck-decision.md)

（源描述见 [`figures/04-bottleneck-decision.md`](figures/04-bottleneck-decision.md)；
SVG 规格 `src/tutorial/figures/39-deepseek-e2e-supernode/04-bottleneck-decision.json`。）

### 5.2 每一步的仪器与判据

顺序只有配上去哪拿数据才有用。下表把六步各自对应的工具与判据指回章节：

| 步 | 要回答的问题 | 在昇腾上用什么看 | 判据 | 章节 |
| --- | --- | --- | --- | --- |
| ① | 算力还是访存？ | 不用 profiler：先算 `flops`/`bytes` | AI 与平衡点比大小 | 第 28 章 §2.2 |
| ② | 哪条流水线满？ | `do_bench(backend="msprof")` 的 `KernelProfile` 字段 | `aic_mad` / `aic_fixpipe` / `mte2` / `aiv_vec` 里谁接近饱和 | 第 12 章、第 19 章 §8；`tilelang/profiler/msprof.py:60-134` |
| ③ | 通信有没有重叠？ | 通算流水图（msProf 的“communication-computation pipeline”） | Wait Time Ratio / Synchronization Time Ratio 高 = 在等 | 第 19 章 §8；第 38 章 §7.2、§9.3 |
| ④ | layout / 对齐有没有拖后腿？ | 生成源码 + 地址计算；NZ/ND 的转换次数 | 对齐字节、bank 冲突、ND2NZ 是否走了 BIU 合并 | 第 21 章、第 36 章 §5.3、§5.5 |
| ⑤ | 是不是启动开销占主导？ | kernel 数量、每次 launch 的固定开销；graph 能不能捕获 | SOL 只有几百纳秒 = “模型没有抓手” | 第 26 章、第 28 章 M7；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/csrc/buffers/ep.hpp`（`1.9 us`、`4 us` 两条注释） |
| ⑥ | 是不是模型算错了？ | 用 `aclnn_*` 或参考实现对拍 | 效率 >100% 就是公式错，不是性能 | 第 14 章、第 28 章 M8；第 35 章 §3.4 |

这张表是**推理顺序**，不是“按顺序都做一遍”。大多数情况下前两步就能定位；
能走到第 ⑤ 步，通常意味着这个算子已经接近它的形状能给的极限，
剩下的是“融合”或“换形状”级别的改动。

### 5.3 三个走查

**走查 A：一个慢的 grouped GEMM（训练侧）。**

1. 第 ① 步：它的 `M` 是每组 4096 还是几十？第 35 章 §5.3/§5.4 给了两个截然不同的
   答案，前者 compute-bound，后者 memory-bound。**先确定自己在哪个 regime**。
2. 如果 `M` 大（compute-bound），进第 ② 步：看 `aic_mad`。
   - `aic_mad` 接近饱和 → 已经到位，别再调 tile；第 35 章 §5.2 的 dense GEMM
     在 98.3%–99.8%。
   - `aic_mad` 不满 → 大概率是 L1/L0 的流水不够深，去看 `num_l1_stages` 与
     `num_l0_stages`（第 35 章 §4.4）。
3. 如果 `M` 小（memory-bound），第 ② 步要看的就不是 `aic_mad`，而是 `mte2`：
   B（专家权重）每次只用一次，优化方向是第 35 章 §5.3 列的两条，
   权重常驻（`resident_b`）与 L2 hint。

**走查 B：一个慢的 MoE 层（端到端）。**

1. 第 ① 步：先把这一层拆成 §2.2 的八段，逐段算字节与 FLOPs。
   第 16 章 §7 的表可以直接用来判断“这个 regime 里谁最贵”。
2. 第 ② 步：如果单算子都不慢，进第 ③ 步。第 38 章 §9.3 说重叠分三层
   （kernel 内 / 调用级 / 流水线级），要逐层问：
   - kernel 内：dispatch 的 issue 与本地拷贝有没有并行？（DeepEP 自己负责）
   - 调用级：有没有用 `defer_epilogue=True` + `async_with_compute_stream=True`？
   - 流水线级：有没有用 DualPipe 把前后向错开？
3. 第 ⑤ 步：如果这一层有很多小算子（超连接、量化、Engram 门控），
   先数 kernel 数量。第 34 章 §3.1 的融合收益主要来自启动开销与重复读。
4. 第 ③ 步的补充：**不要指望 graph capture**。第 38 章 §8.3 明确写了
   DeepEP-Ascend 目前**不支持把 EP 入图**，所以小 batch 的固定开销目前只能靠
   融合与批处理摊薄。

**走查 C：一个访存为主的 TileLang 算子（超连接 / 量化 / Engram 门控）。**

这一类算子在第五部分里占了 TileLang 覆盖面的绝大多数，它们的检查顺序与前两个
走查不同，**第一步就该跳过“哪条计算流水线满”**：

1. 第 ① 步：算 AI。第 34 章 §7.1 的三个判据可以直接用：
   FLOPs 远小于 bytes、归约维度能放进一个 CTA、且在关键路径上被反复调用。
   三条都成立时，它就是访存/延迟主导的，**不要去看 `aic_mad`**。
2. 第 ② 步：这时要看的是 `mte2` / `mte3` / `aiv_vec` 三者的配比。
   - `mte2` 满、`aiv_vec` 空 → 数据没搬够，先看多缓冲与预取；
   - `aiv_vec` 满、`mte2` 不满 → 计算本身重，去看第 34 章 §7.2 的向量内建
     （`T.SimdVF` + `T.simd.*`）；
   - 两者都不满 → 进第 ⑤ 步，去数 kernel 数量。
3. 第 ⑤ 步：把“这一层里有多少个小算子”数清楚。第 34 章 §3.1 的结论是
   融合的收益主要来自**启动开销**与**重复读 `x` 的带宽**这两项，
   而不是算术本身。
4. 第 ④ 步的一个昇腾专属检查：**缓冲版本数**。第 33 章 §4.6 讲得很清楚，
   `T.Pipelined` 引入“多个迭代同时在飞”的状态，份数是**性能意图**，
   调度器只能保证下限。所以 `T.annotate_buffer_versions({buf: n})` 里的 `n`
   该不该加，要按“同时有几个迭代在飞”来定，而不是照抄别的算子。
5. 最后一个检查：**迭代次数是不是编译期常量**。第 34 章 §7.1 把
   “迭代是编译期常量（`T.serial` / `T.unroll`）”列为这类算子的共同特征；
   如果你的实现把迭代放进了运行时循环，那它多半跑不过预期。

### 5.4 把三个走查串成一次

下面是一个**构造出来的**完整走查（不是任何一次实测）：设你手上有一个 128 卡的
MoE 训练任务，某一步的耗时比按 §4 预算表估出来的高一倍。按六步走：

```text
【第 ① 步】先算账。把这一步拆成 §2.2 的八段，得到每一段的 bytes 与 flops。
           → 发现「dispatch + combine」两段加起来占预算的很大一块。
           → 结论：这不是 compute-bound，跳到第 ③ 步（重叠），
             而不是第 ② 步（PIPE）。

【第 ③ 步】查三层重叠。
   kernel 内：DeepEP 自己的流水，你改不了；先跳过。
   调用级  ：检查代码里有没有 defer_epilogue=True
             + async_with_compute_stream=True。若缺一个 → 补上，
             消掉「最慢 rank 决定整步」的那段纯等待（第 38 章 §2.5）。
   流水线级：检查有没有用 DualPipe；若没有，第 ④ 阶段的目标就先定在这里。

【第 ② 步】通信藏好之后，回头看不依赖通信的那几段。
   → 如果 grouped GEMM 的 aic_mad 不满：进 L1/L0 流水深度（第 35 章 §4.4）。
   → 如果 aic_fixpipe 高而 aic_mad 不高：这是 MQA logits 的特征
     （第 35 章 §5.5），去查能不能少搬。

【第 ④ 步】看 layout。KV / A 的 NZ 转换次数、512 B 对齐、L1 两 bank 配平
   （第 36 章 §5.3、§5.5）。

【第 ⑤ 步】数 kernel。超连接、量化、Engram 门控是不是各自一个 launch？
   第 34 章 §3.1 说融合的收益主要在这里。

【第 ⑥ 步】最后复核口径：是不是把某一列的口径搞错了（§3.7、§4.4）。
```

这段走查里没有出现“调 tile 尺寸”。这**不是巧合**。在 128 卡这个规模上，
端到端的第一步几乎总是“账本 + 重叠”，而不是“单算子调参”。
第 28 章 §1.1 说的三种典型失败，第一种就是“先动手、后判类”。

---

---

## 6. 从“能跑”到“跑满”的路线图

### 6.1 五个阶段

把上面五节收成一条可执行的路线。每个阶段都有**目标、判据、以及本教程里对应的
章节**；阶段之间是严格顺序，跳步得到的优化通常会在下一阶段被发现是错的。

| 阶段 | 目标 | 判据（可检查） | 对应章节 |
| --- | --- | --- | --- |
| ① 正确性 | 每个算子与参考实现对得上 | 第 14 章的对拍口径；第 18 章 §7.3 的报告清单 | 第 13–18 章 |
| ② 单算子利用率 | 每个热点算子贴着它自己的 roof | roofline 判类 + 对应 PIPE 的利用率；不跨形状抄数字 | 第 28 章 §2–§4；第 35 章 §5；第 36 章 §3 |
| ③ 层内融合 | 减少 kernel 数量与重复读写 | 融合前后 IO 字节数与 kernel 数（第 28 章 §5.1 的四个问题） | 第 13 章、第 33 章 §5.5、第 34 章 §3.1 |
| ④ 卡间重叠 | 通信不再落在关键路径上 | 通算流水图上的 Wait Time Ratio 下降 | 第 38 章 §3.4、§9.3；第 32 章 §5.2 |
| ⑤ 全局 MFU | 端到端算力利用率稳定 | `C ≈ 6·N·D` 折算出的有效 TFLOPs ÷ 峰值；第 17 章 §2.3 的算法 | 第 17 章 §2.3、§9；第 18 章 §7 |

### 6.2 为什么是这个顺序

* **① 在 ② 之前**：没有正确性就没有性能可言。第 17 章 §7.3 把“对不对”
  变成了可执行的检查，而不是一句态度。
* **② 在 ③ 之前**：先让每个算子贴着自己的 roof，再谈融合。否则你无法判断
  融合省下的时间是不是本来就不该花（第 28 章 §5.1 的四个问题）。
* **③ 在 ④ 之前**：融合会改变 kernel 数量与形状，从而改变通信与计算的配比。
  第 35 章 §5.6 的 MegaMoE 就是“先融合、再谈重叠”的极端版本。
* **④ 在 ⑤ 之前**：MFU 是端到端指标，通信没重叠时它一定难看；
  但反过来，通信重叠好了、单算子利用率低，MFU 一样难看。
  **两者不能互相替代**。这也是第 28 章 §1.1 那三种典型失败里的第一种。

### 6.3 一个必须承认的天花板

这套路线图有一个前提：**你能拿到一台可复现的机器**。第五部分的所有性能数字
都带条件，而其中最关键的一个条件是，**DeepEP-Ascend 的带宽表来自“手工配置的
PoC HDK”，不是公开版本**。README 在 `Recommended HDK and firmware` 一节写明
商用 Q3 HDK **计划**在 2026-10-15 前后发布，并明确
“these measurements are not results from that unreleased commercial version”
（`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:40-42`；第 38 章 §7.1）。

所以阶段 ④ 之后能不能真的“跑满”，取决于你手上的固件版本。
这不是本章能回答的问题，只能登记在 §7 的现状清单里。

### 6.4 每个阶段的常见失败

五个阶段各有自己的“典型死法”。这些不是本章发明的，都是第 32–38 章末尾
“常见坑”小节里已经登记过的条目，这里只是把它们按阶段归位：

| 阶段 | 最常见的失败 | 出处 |
| --- | --- | --- |
| ① 正确性 | 把“仅静态核对”当成“跑过”。第五部分的示例**没有一条来自本机实测** | 第 32 章 §8.6、第 19 章的开头 |
| ① 正确性 | 对拍本身不适用却当成 kernel 错（例如 ACLNN 参考实现对 scale 形状有约束） | 第 35 章 §8 P8 |
| ② 单算子 | 用张量核的 roof 给逐元素算子定价，判类直接反掉 | 第 28 章 §2.3 M2 |
| ② 单算子 | 拿 dense GEMM 的 99% 去推 `M=128` 的推理性能 | 第 35 章 §8 P10 |
| ② 单算子 | 在 FixPipe-bound 的算子上调 MAD 的形状，调错单元了 | 第 35 章 §5.5、第 36 章 §3.3 |
| ③ 融合 | 为了融合而融合，把本该拆开的相位并到一起（例如确定性模式下的 K-split 串行相位） | 第 28 章 §5.2、第 35 章 §8 P12 |
| ③ 融合 | 忘了迭代状态是编译期常量，导致 `repeat` 展开后代码体积失控 | 第 34 章 §5.1；第 36 章 §5.4 的 icache 一节 |
| ④ 重叠 | 只设 `defer_epilogue=True` 而没配 `async_with_compute_stream=True`，直接断言失败 | 第 38 章 §3.4；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/csrc/buffers/ep.hpp:398` |
| ④ 重叠 | 以为可以把 EP 入 graph capture，目前不支持 | 第 38 章 §8.3 |
| ⑤ 全局指标 | 报 MFU / TPOT 不写条件（预热、缓存、图模式、取数方式） | 第 18 章 §7.2、§7.3 |
| ⑤ 全局指标 | 把“有效吞吐”当成“硬件上限的百分比”，两者是不同的分母 | 第 37 章 §6.1、§6.3 |

### 6.5 阶段之间会回退

上表列的是“顺序”，但实际做的时候一定会回退。三种回退是正常的，
而且每一次回退都说明你在做对的事：

| 你做到了 | 发现 | 该退回 |
| --- | --- | --- |
| ③ 层内融合 | 融合后 kernel 更大，单算子利用率掉了 | **②**：重新给融合后的形状定 roof（第 28 章 M4 讲的正是这件事） |
| ④ 卡间重叠 | 为了重叠加了新的小 kernel，kernel 数量反而上升 | **③**：先把小 kernel 融掉，再谈重叠 |
| ⑤ 全局 MFU | MFU 上不去，但每个算子都贴着上限 | **①**：多半是某处的数值/形状约束让你多做了一次搬运 |

唯一**不该**回退的是 ① → ②：正确性没锁死之前不要做任何性能优化。
第 17 章 §7.3 把“对不对”做成可执行检查，就是为了让这道门能真的拦住人。

---

## 7. 这套生态还缺什么

### 7.1 一份诚实的现状清单

第 32–38 章每一章末尾都写了“本章没有验证的东西”。把它们集中起来，就是这套生态
在 2026-09-30 这次发布时点的**真实能力边界**。清单分三档，判据全部来自各仓库
README 的 `Ongoing` / `Features` 一节或源码里的断言。

**第一档：现在可以用**

| 能力 | 状态 | 依据 |
| --- | --- | --- |
| TileKernels 的 MoE 路由 / 量化 / RoPE / Engram / mHC，两个后端 | 可用 | 第 33、34 章；`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md:9-19` |
| DeepGEMM-Ascend 的 BF16/FP8/FP4 dense、M-grouped、MQA logits、MegaMoE | 可用 | 第 35 章；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:91-176` |
| DeepEP-Ascend 的 EP dispatch / combine、cached handle、展开布局、确定性布局 | 可用 | 第 38 章 §8.1；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:60-74` |
| DeepSelect 的 Lightning Indexer TopK（bf16，`topk=512`） | 可用 | 第 37 章；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:17-28` |
| FlashMLA 的昇腾稀疏 prefill / decode | 可用 | 第 36 章；`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:13-15` |
| DeepJIT 的运行时编译 / 缓存 / 装载 / 启动 | 可用 | 第 38 章 §6；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepJIT/README.md:11-16` |

**第二档：experimental（接口在，验证未完成）**

| 能力 | 状态 | 依据 |
| --- | --- | --- |
| `PPBuffer`（相邻流水 rank 的 send/recv） | 正确性与性能验证进行中 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:67,82` |
| `EngramBuffer`（NPU 侧 BF16/FP8 表 + 每层 fetch hook） | experimental | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:68,82` |
| `BucketBuffer` 的 batched all-gather | 可用但归入 experimental；**用前必须设 `EP_AVOID_RECORD_STREAM=1`** | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:69,80,234` |
| TileKernels 两个后端是否**语义等价**（不只是各自能跑） | 未经两台机器对拍 | 第 33 章 §8.3 |
| `T.view` 在昇腾上的实际可用性 | 只核对了契约与调用方式 | 第 33 章 §8.3 |

**第三档：不支持（现在别用）**

| 能力 | 状态 | 依据 |
| --- | --- | --- |
| Bucket 的 **reduce-scatter / all-reduce** | 昇腾 kernel 仍在做 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:80` |
| `lb_prefetch_weights` / `lb_reduce_grads`（EPLB 的搬运侧） | 接口已对齐，**昇腾通信 kernel 未实现** | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:81`；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/csrc/buffers/ep.hpp:79,85`（`EP_HOST_UNREACHABLE`） |
| **把 EP 入 graph capture** | 不支持 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:84` |
| 混合通信（hybrid）、CPU-backed Engram、`do_handle_copy=True`、非展开（low-latency）布局、奇数 rank 数 | 不支持 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md:29,72,84`；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/csrc/buffers/ep.hpp:116` |
| DeepGEMM-Ascend 的 masked grouped GEMM | 未实现 | 第 35 章 §8 P4；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/csrc/apis/gemm.hpp:591,615` |
| MegaMoE 的通用性 | 能力被裁到最窄：只支持 `fp8xfp4` + `swiglu` + `fast_math=True`，核数写死 32 | 第 35 章 §5.6；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/csrc/apis/mega_moe.hpp:145-150,159` |
| FlashMLA 的融合 kernel（norm-RoPE-attn-RoPE-cast） | **CUDA only，昇腾不支持** | 第 36 章 §6.4；`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:39,53` |
| DeepSelect 的昇腾实现支持 fp32 采样场景 | 只支持 bf16；昇腾也只支持 `torch.int32` 索引，`sorted=True` 只在 CUDA FP32 变体上支持 | 第 37 章 §5.1；`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md:37,90-95` |
| DeepJIT 的缓存预热与 Python 编译 API | WIP，尚未可用 | `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepJIT/README.md:18-21` |
| “TileLang 昇腾版”是单一实现 | 实际是两棵树（主干 `tilelang/ascend/` 与适配器仓库），API 不通用 | 第 19 章 §7、第 32 章 §3.5 |
| 128 卡超节点的硬件细节 | 官方文章只有一句话，corpus 里没有对应白皮书 | 第 32 章 §2.4 与“待核事项”第 3 条 |

![图 5：这套生态的能力状态](figures/05-ecosystem-status.md)

（源描述见 [`figures/05-ecosystem-status.md`](figures/05-ecosystem-status.md)；
SVG 规格 `src/tutorial/figures/39-deepseek-e2e-supernode/05-ecosystem-status.json`。）

### 7.2 这些缺口对使用者的实际影响

三档清单不是“待办列表”，它会直接改变工程决策。四条最要紧的：

1. **训练侧的重叠只能靠 DualPipe + `defer_epilogue` 这一层。** 想要像 NVIDIA
   那样把 EP 入 graph capture 来降下发开销，现在做不到（第 38 章 §8.4）。
   decode 的小 batch 场景要特别小心每次 launch 的固定开销。
2. **动态冗余专家（EPLB）暂时不能端到端用。** EPLB 本身有算法、有
   `rebalance_experts` 接口，但把权重搬来搬去的 `lb_prefetch_weights` 在昇腾上
   还没实现，“在线负载均衡”目前只有规划、没有搬运。
3. **FlashMLA 在昇腾上只覆盖稀疏这一段。** 融合了 Q-norm / RoPE / cast 的那条
   优化路线是 CUDA only，所以昇腾侧要自己承担这些“小算子”的开销，
   这也是 TileKernels 在昇腾上更重要的原因之一。
4. **DeepSelect 在昇腾上只有 bf16。** 采样场景（fp32、`vocab ≈ 128K`）目前
   没有昇腾实现，所以“decode 最后一步选 token”这件事在昇腾上要么换精度、
   要么走别的路径。

### 7.3 怎么自己维护这份清单

清单会过期。好消息是，清单里不同条目的**保鲜期差别很大**，按这个差别去维护，
成本最低：

| 依据的类型 | 例子 | 保鲜期 | 怎么核验 |
| --- | --- | --- | --- |
| **硬件指令集语义** | gather2 的 `burst_count`、L2 hint、SS buffer（第 36 章 §6.4 的 C 类） | 最长（以芯片代际计） | 查芯片编程指南，不看仓库 |
| **编译器能力边界** | 哪些 `T.Parallel` 能被自动张量化、`T.simd.*` 的覆盖面（A/B 类的边界） | 中等（随 release 移动） | 对比 `_vendor/` 两个 tag（第 36 章 §6.5 明确说这条边界会随版本移动） |
| **上游仓库的成熟度标注** | 各 README 的 `Ongoing` / `Features` 一节 | 最短（随 commit 变） | 绑定 commit 查（第 32 章 §8.4） |

维护的动作只有四步，都在第 32 章 §8.1–§8.4 里讲过了：

1. **读 `Requirements`**：确认硬件与工具链版本（决定这条结论对你的机器成不成立）；
2. **读 `News`**：确认这条能力是哪个日期加的（决定它是“已发布”还是“实验性”）；
3. **读 `Ongoing`**：区分“已发布 / 实验性 / 未实现”，
   这三档的措辞在这套仓库里是统一的（第 32 章 §8.3）；
4. **记下 commit**：`PROVENANCE.md` 里每个仓库的 commit 就是这份清单的版本号；
   行号随仓库更新会漂移，所以引用时要连 commit 一起引。

一句提醒：**不要把“README 里写着”当成“硬件上成立”。** 本章 §4 那张表
里，唯一来自实测的一行（卡间带宽）在 README 自己那里都写着“PoC HDK”。

---

## 8. 收口到贡献

### 8.1 把缺口分到三条路径上

第 31 章把贡献分成三条路径：**写算子/示例、改编译器、加/改后端**。
第 30 章讲的是“只有 main / 未合入的方向”怎么跟踪。把 §7 的缺口按这两章的方法
归档一遍：

| 缺口 | 属于哪条路径 | 为什么 |
| --- | --- | --- |
| Bucket 的 reduce-scatter / all-reduce | 路径一（算子）或路径三（后端/通信） | 是新的通信 kernel；昇腾侧要操作 URMA 队列 |
| `lb_prefetch_weights` / `lb_reduce_grads` | 路径一（算子） | 接口已经对齐、只差 kernel 实现（第 38 章 §8.3） |
| masked grouped GEMM | 路径一（算子） | 有明确的对拍口径（第 35 章 §3.4） |
| 把 EP 入 graph capture | 路径二（编译器/运行时）或路径三 | 需要图捕获与通信生命周期对齐；DeepEP 现在是 host 侧 launch |
| FlashMLA 融合 kernel 的昇腾版 | 路径一（算子） | 第 36 章已经逐条给出哪些是 A 类、哪些要下钻到 `T.simd.*`、哪些必须手写 |
| DeepSelect 的非 bf16 路径 | 路径一或路径三 | 取决于瓶颈是算法常数还是向量指令宽度 |
| 两棵树（主干 / 适配器）的收敛 | 路径三（后端） | 第 31 章 §7.4 已标注：这条流程在两个仓库里都没有成文规定 |
| TileKernels 两后端的语义等价测试 | 路径一（示例/测试） | 属于“可被复核”的贡献，不改编译器 |

### 8.2 用第 30、31 章的方法判断“值不值得做”

不要一开始就写代码。第 30 章给的是**跟踪状态**的方法（先确认它是不是
“仅 main / 未进 tag”、有没有人已经在做），第 31 章给的是**流程**（issue →
（大改动）RFC → PR → review → CI）。把两者合成一条判断路径：

1. **先确认状态**：这个缺口在锁定 commit 里是“未实现”“experimental”还是
   “已实现但你没用对”？三档的处理完全不同，第三档是文档/用法问题，
   不值得写代码。
2. **再确认它是不是瓶颈**：用 §5 的六步顺序。如果它在你的模型里不在关键路径上，
   补上它不会让 MFU 提高。
3. **估一下它的边界**：这个缺口的收益是**数学层的**还是**常数层的**？
   第 37 章那句“算法决定上限、硬件决定常数”在这里最好用，
   数学层的（例如 reduce-scatter 的字节量）值得做；常数层的（例如某个
   `num_stages` 再调一档）通常不值得单独开一个 PR。
4. **找对门**：第 31 章 §2.3 的四道门（格式与 lint、本地测试、上游 CI、
   按需的性能回归）是硬门槛；昇腾侧还有独立的 `ascend` CI job 与 PR 模板要求
   （第 31 章 §7）。
5. **写清口径**：性能类贡献必须附上完整条件（第 18 章 §7.3 的清单）；
   第 31 章 §9.3 的性能回归判定规则是“至少 2 个 benchmark 同时掉破 0.90”，
   这条规则本身就是“要看整体、不看单点”的示范。

### 8.3 读完本教程之后的第一步

最后给一条可执行的路径。它把全书（不只是第五部分）串起来，按顺序做：

```text
第 1 步 · 确认环境        读一个组件的 Requirements + News，
                          确认硬件型号与 CANN / TileLang 版本；
                          对照第 32 章 §8 的核验清单
第 2 步 · 跑通最小用例    从该组件的 tests/ 或 examples/ 挑最小的那个，
                          先跑正确性（第 19 章 §4.5 的「前提/命令/期望输出」格式）
第 3 步 · 建立自己的账本  用第 2 章的记账法 + 本章 code/01 的模板，
                          把你的模型形状代进去，算出每个算子的 flops/bytes
第 4 步 · 判类            按第 28 章 §2.2 的 roofline 判类，
                          确定每个热点算子该看哪条 PIPE
第 5 步 · 单算子          用第 28 章 §3–§4 的 tile 模型调一个算子，
                          用第 12 章的工具验证
第 6 步 · 融合            用第 28 章 §5.1 的四个问题决定要不要融，
                          参考第 33、34 章的融合写法
第 7 步 · 重叠            按第 38 章 §9.3 的三层重叠逐层打开，
                          用通算流水图验证
第 8 步 · 全局指标        用第 17 章 §2.3 的 MFU 与第 18 章 §7 的 TTFT/TPOT
                          收口，并写下第 18 章 §7.3 的报告清单
```

这条路径的每一步都能只用本教程的内容完成，也都能独立停下来。
**第 4 步是分水岭**：在此之前你是在“跑通”，在此之后你才开始“跑满”。

### 8.4 一个具体的起步项目

如果 §8.3 的八步让你不知道从哪里下手，这里给一个**两周内能做完**的项目。
它刻意选得小，但能走完 §6 的前三个阶段：

**题目**：在昇腾上写一个融合的 **SwiGLU + 量化** kernel，与 TileKernels 的同名
算子对拍并在真机上测量。

**为什么选它**：

* 它有现成的参照物，`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/quant/swiglu_forward_kernel.py:40-50`
  就是官方实现，正面证明“你写的与它的差距”；
* 它是第 13 章的融合算子在昇腾上的一个实例，且是第 33 章 §7.2 里
  “共享算法、重写数据流动”的典型；
* 它是**访存为主**的算子（§4.2 的分档），所以你能在不用碰 Cube 的情况下
  走完 ① 正确性 → ② 单算子利用率 → ③ 层内融合 三个阶段；
* 它在真实流水线里真的被调用（§2.1 的第 5 步），不是玩具。

**交付物（四件）**：

1. 一份 `README.md`：运行前提（Ascend 950 + CANN 版本 + TileLang 版本）、
   运行命令、期望输出、你实际做到的验证级别，格式照第 19 章 §4.5 的四段式；
2. 一份对拍报告：与 TileKernels 的参考实现比数值（第 14 章的对拍口径）；
3. 一份性能报告：`do_bench(backend="msprof")` 的 `aic_*` / `aiv_*` / `mte*`
   字段，按第 19 章 §8 的方法判读瓶颈；**必须写全第 18 章 §7.3 那份清单**；
4. 一份“与官方实现的差异清单”：哪里一样、哪里不同、为什么。

**评判标准**：不看绝对数字，看三件事，数值是否与参考实现对得上、
你测得的那条 PIPE 是否与你从 roofline 判出来的那类资源一致、
你的报告里有没有把条件写全。这三件事对应第 31 章 §11 的自检清单里
“可被复核”的那一条。

---

## 9. 全书回顾：五个部分各给了什么

本章是第五部分、也是全书的收口。把五个部分放回一条时间线上，它们各自回答的
问题其实只有五个：

| 部分 | 章节 | 它回答的问题 | 读完之后你能做的第一件事 |
| --- | --- | --- | --- |
| 一 · 宏观 | 1–6 | 为什么需要 tile DSL？它在软件栈的哪一层？ | 判断一个算子“该不该自己写”（第 2 章 §1） |
| 二 · 实用 | 7–19 | 怎么用 TileLang 写出一个生产级算子、并训练/推理一个 7B 模型？ | 跑通第一个 kernel，并把热点算子换掉（第 13–18 章） |
| 三 · 深入 | 20–29 | 编译器与后端是怎么把这些程序变成机器码的？ | 读懂生成产物，并知道该改哪一层（第 20–25 章） |
| 四 · 前沿与参与 | 30–31 | 还没进 release 的方向在哪？怎么把它变成一次可复核的贡献？ | 提一个算子 PR 或修一个 pass（第 31 章 §10） |
| 五 · 真实开源 | 32–39 | 一套生产栈在昇腾上长什么样？怎么判断瓶颈、下一步做什么？ | 本章 §8.3 的八步 |

**如果只带走三句话**，本章建议是这三句：

1. **“先算再写”不是一句口号，而是有执行顺序的**：判类 → 定 roof → 定 tile →
   算融合账 → 再写代码（第 28 章）。本章 §5 的六步检查顺序是它在整条流水线上的
   版本。
2. **TileLang 的边界是“算”与“搬”的分界，也是“规整”与“协议”的分界**：
   访存为主、规整、可被算法描述的部分，TileLang 覆盖得很好（第 33、34 章、
   第 35 章 §5.7）；指令级细节、通信协议、FixPipe 位域这些落点，会主动下沉到
   Ascend C / PTX（第 36 章 §6、第 38 章 §9）。
3. **每个性能数字都带条件；每句“已实现”都要问“在哪个 commit、哪台机器、
   什么形状”**。这是本书里那些“看起来对、其实不成立”的结论的共同解药
   （第 18 章 §7.3、第 32 章 §8、本章 §4.4）。

---

## 本章小结

一页纸复习：

1. **训练侧一条链**：mHC pre → 路由 TopK → dispatch → grouped GEMM ① →
   SwiGLU+量化 → grouped GEMM ② → combine → mHC post。TileLang 覆盖两端与中间
   一小段，最贵的两段 GEMM 与两段通信在 Ascend C 里（§2.1、§2.2）。
2. **反向的两条特殊机制**：mHC 的中间激活**重算**（论文 §4.3.2，`L_r*` 由式 (20)
   给出、重算块与流水线 stage 对齐），以及超连接参数的**梯度二次归约**
   （第 34 章 §4.3）。训练一步的 backward 形状见 §2.3。
3. **mHC 论文 §4.3 的三条基础设施结论**：kernel 融合（读写从 `(3n+1)C`/`3nC`
   降到 `(n+1)C`/`nC`）、重算（换显存）、DualPipe 重叠（MLP 的
   `ℱ_post,res` 放高优先级 stream、attention 长算子不用 persistent kernel）。
   收益数字“6.7% 额外开销”要连带 `n=4`、论文自报、in-house 大规模训练三个条件
   一起引用（§2.5）。
4. **推理侧两条链**：prefill 的形状是大 M，瓶颈在算力 / FixPipe / 卡间带宽；
   decode 的形状是小 M，瓶颈在片内带宽与启动开销。**瓶颈是形状的函数**，
   不是算子的属性（§3）。
5. **三类资源、三套口径**：算力 432.5/865/1730.1 TFLOPS（本书从 `HWDesc` 复算）、
   片内带宽 HBM 3500 / L2 4500 GB/s（源码硬件模型常量）、卡间带宽
   dispatch 313–375 / combine 272–347 GB/s（厂商实测，PoC HDK）。
   **不许相减、不许相乘、不许跨机器比**；唯一允许的一次除法是量级比值
   3500/374 ≈ 9.4（§4）。
6. **瓶颈检查顺序**：判类 → 哪条 PIPE 满 → 通信有没有重叠 → layout/对齐 →
   启动开销 → 回头复核模型。每一步的仪器与判据都在 §5.2 那张表里。
7. **从能跑到跑满五个阶段**：正确性 → 单算子利用率 → 层内融合 → 卡间重叠 →
   全局 MFU。顺序不能换（§6）。
8. **现状清单**：EP 主线可用；Bucket 只有 all-gather；PP / Engram 实验性；
   reduce-scatter / all-reduce、LB 通信、hybrid、CPU Engram、EP 入图、
   masked grouped GEMM、昇腾上的融合 attention kernel、DeepSelect 的非 bf16
   路径都还没有（§7）。
9. **收口到贡献**：把缺口分到第 31 章的三条路径上，用第 30 章的状态跟踪方法确认
   “该不该做”，用 §8.3 的八步作为读完本教程后的第一步。

## 延伸阅读

* **[R1] 前面七章**：第 32 章（全栈地图）、第 33/34 章（TileKernels）、
  第 35 章（DeepGEMM-Ascend）、第 36 章（FlashMLA 昇腾）、
  第 37 章（DeepSelect）、第 38 章（DeepEP-Ascend），本章的每一节都以它们为依据。
* **[R2] mHC 论文**：
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md`
  §4.3.1 Kernel Fusion / §4.3.2 Recomputing / §4.3.3 Overlapping
  Communication in DualPipe（§2.3、§2.5）；同目录另有
  `paper-deepseek-mhc.pdf`。
* **[R3] 官方发布文章**：
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/deepseek-article-20260930.md`
  六个组件的清单与“昇腾 950 的 128 卡超节点”的原始表述（§1.1、§4.1）。
* **[R4] 训练侧调度件**：`../tilelang-corpus/14-deepseek-ascend-stack/raw/DualPipe/README.md`（双向流水与气泡对照表）、
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/EPLB/README.md`（冗余专家与分组受限路由），§2.6。
* **[R5] 性能表**：`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md`（dense / M-grouped /
  MQA / MegaMoE / HC prenorm）、`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md`（EP8–EP128）、
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md`（410 / 360 TFlops）、
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSelect/README.md`（2–20× 与有效内存带宽），§3、§4。
* **[R6] 方法论**：第 17 章 §2.3（`6·N·D` 与 MFU）、第 18 章 §7（TTFT/TPOT）、
  第 28 章 §2–§5（roofline 判类、tile 模型、融合账本），§5、§6。
* **[R7] 贡献路径**：第 30 章（前沿状态跟踪）、第 31 章（三条路径与四道门），§8。

完整的引用清单（含行号与核验状态）见 [`references.md`](references.md)；
每张图的源描述见 [`figures/`](figures/)；示例代码与验证级别见
[`code/README.md`](code/README.md)。
