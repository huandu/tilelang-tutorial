# 第 36 章 FlashMLA 昇腾稀疏注意力：算法与流水

## 学习目标

读完本章，你应当能够：

1. 把一个「单 Q token 的稀疏 attention」写成可执行的形式：给定
   `Q/KV/indices/sm_scale`，说出 `gather → P = QKᵀ → softmax → PV` 四步各自
   的 shape，并写出 online softmax 的四个状态量（`running_max`、
   `running_max_for_softmax`、`running_sumexp`、`out_accum`）在**每个 KV 块**
   上的更新顺序。
2. 用一段 20 行的 Python，把官方技术报告里的**算存比脚本**复现出来，读懂
   `cube_cycles` 与 `mem_cycles` 两列为什么能推出「每个 AI Core 处理 ≥ 64 个 q head
   才进入 compute bound」，并独立推出取 128 时 L0C / UB 会被什么撑爆。
3. 复述官方报告里**四个设计讨论**的推理链，而不是只背结论：head 数怎么定、
   output accumulator 放哪里、为什么不能用 1C1V、KV 的 ND2NZ 转换放在哪一步。
4. 画出 CUBE 侧与 VECTOR 侧的两条流水，说清「`O += S@V` 为什么延后一拍」，
   以及 skip-scale 信号为什么会把 SIMD VF 和 Scalar Core 卡在一起。
5. 把官方那六条优化技巧（gather2、KV cache layout、高速 ND2NZ、icache、
   L1 bank 配平、skip-scale）各归到一条硬件资源上，并说得出**它在省哪一份带宽、
   绕开哪一个上限**。
6. 面对本章任何一个技术点，能回答「它属于三类里的哪一类」：TileLang 直接可表达 /
   可表达但要下钻到 `T.simd.*` / 必须落到 Ascend C、PTO 或 SIMD VF 内建，
   并给出判断依据（对照第 25 章的 `T.*` 原语清单与本章引用的真实指令）。

## 前置知识

* 第 15 章：attention 家族、online softmax 的递推与等价性证明。本章把
  「三趟折成一趟」的推导直接拿来用，但换了硬件，所以要重新算一遍代价。
* 第 6 章、第 19 章 §2：昇腾的机器模型——AIC（Cube）与 AIV（Vector）、
  GM / L1 / L0A / L0B / L0C / UB 的层级、MTE1/2/3 与 FixPipe 这四条搬运/收尾流水线。
  本章所有「谁搬、搬到哪一级」的判断都建立在这张图上。
* 第 22 章：流水线与同步——多缓冲、`num_stages`、Set/Wait 旗标的语义。
  本章的 CUBE / VECTOR 双流水是它在真实算子上的一个极端例子。
* 第 23 章：昇腾自动调度（task-based 建模 + SMT 求解）。本章会反复问「这一步
  编译器能不能自动排」，答案要和第 23 章的模型对上。
* 第 25 章：昇腾后端与 `T.*` 原语清单——`T.alloc_l1/l0a/l0b/l0c`、`T.copy`、
  `T.gemm`、`T.dual_copy`、`T.SimtVF` / `T.SimdVF`、`T.Pipelined`、
  `T.annotate_buffer_versions`、`T.ascend_cross_core_set_flag` / `_wait_flag`。
  本章 §6 的三分类表要拿它当字典。
* 第 35 章：DeepGEMM-Ascend 里已经出现过「FixPipe bound」「scale factor 打包」
  「协程流水」这些词。本章的其中一条 bound 就是 FixPipe bound 的另一种算法。

不需要你有昇腾 NPU。本章的验证边界见下一节——**所有代码的最高验证级别是
「仅静态核对」或「在本机 CPU 上用 NumPy 对拍」**，没有任何一条性能数字来自本机实测。

---

## 本章的验证边界（务必先读）

写作环境是 macOS (Apple Silicon)、Python 3.11，**没有 NVIDIA GPU，也没有昇腾 NPU**
（见 [`../../VERSIONS.md`](../../VERSIONS.md)）。因此本章遵守三条硬规矩：

1. **所有算子源码只做静态阅读。** 本章引用的
   `14-deepseek-ascend-stack/raw/FlashMLA/csrc/ascend_kernels/**` 是昇腾侧的
   Ascend C / SIMD VF 源码，只能由 `bisheng` + CANN 编译器在真实 NPU 上构建。
   **本章作者没有构建、没有运行、没有 profile 过这些 kernel**，任何
   「跑一遍看结果」的说法都不成立。
2. **性能数字只引用一手来源，并写清测量条件。** 本章出现的性能数字只有两类：
   官方 FlashMLA README 里的公开数字（带硬件型号与口径），以及本书 `code/` 里
   纯 Python / NumPy 复算出来的**周期数**（由公式推导，不是实测）。
   例如「prefill 410 TFlops = 95% 硬件峰值」是 **Huawei Ascend 950 NPU** 上的口径
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/README.md:13,53`），
   不是本机数字。
3. **`code/` 里的程序分两档**：
   * `01-head-roofline`、`02-fixpipe-bound`：纯算术，**在本机跑过**，输出与官方报告的
     表格逐行一致（见各自的 `README.md`）；
   * `03-online-softmax-skip-scale`：NumPy 实现 + 与「一次算完的参考实现」对拍，
     **在本机跑过**；它能验证的只是**算法等价性**，不涉及任何硬件行为；
   * `04-tilelang-expressibility`：把本章的技术点映射成「三类」清单，
     并在本机逐个 grep 锁定源码树确认原语存在，**是静态核对工具，不是 kernel**。

本章还会跨章引用第 15、19、22、23、25、35 章，一律写成「见第 N 章」，不写死路径。

---

## 1. 问题的由来：稀疏注意力为什么要在昇腾上重写一遍

### 1.1 DSA 一句话概括

DeepSeek 的 **DeepSeek Sparse Attention（DSA）** 是「token 级稀疏注意力」：
每个 query token 不再看全部 KV，而是先由上游的小算子（DeepSelect 里的
Lightning Indexer + TopK）挑出 `topk` 个「最重要的」KV token，写下**索引表**，
attention 算子再按这张表**只算那 `topk` 个位置**。

DSA 第一次公开是在 DeepSeek-V3.2-Exp，官方把它定位为「长上下文里省训练/推理算力、
同时几乎不掉点」的机制：

> DeepSeek Sparse Attention (DSA) achieves fine-grained sparse attention for the
> first time, delivering substantial improvements in long-context training and
> inference efficiency while maintaining virtually identical model output quality.
> —— `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepSeek-V3.2-Exp/README.md:47-49`

到 DeepSeek-V4.1，这套机制用的是 FP8 / FP4 的 KV cache，并且 **prefill 与 decoding
共用一套设计**。本章讲的就是它在昇腾上的实现：FlashMLA 仓库里的
`sparse attention prefill` 与 `decoding` 两个算子，以及官方为它专门写的技术报告
`../tilelang-corpus/14-deepseek-ascend-stack/raw/FlashMLA/docs/20260930-ascend-prefill-deep-dive-zh.md`
（下称「报告」）。本章所有「报告 `:N`」都指这个文件的第 N 行。

报告的公开口径是：在 DeepSeek V4.1 的典型工况下，该算子在华为昇腾 950 NPU 上
**prefill 达到 410 TFlops（硬件峰值 95%）、decoding 达到 360 TFlops（硬件峰值 83%）**
（报告 `:5`；同一组数字在 `FlashMLA/README.md:13,53,63`）。这两个数字是**上游自报的**，
测量条件（模型形状、CANN 版本、benchmark 脚本）见 §5.7。

### 1.2 这篇报告真正教的是什么

如果把报告当成一份「性能说明」，你会看到一堆结论：

* 每个 AI Core 处理 64 个 q head；
* output accumulator 放在 UB（也就是 VECTOR 侧）；
* 用 **1C2V**（一个 Cube 配两个 Vector），不用 1C1V；
* KV 先从显存搬到 UB、用 SIMD VF 转 NZ，再进 L1；
* gather2、KV cache layout、高速 ND2NZ……

但这些都是**结论**。报告真正的价值在于：**每一个结论之前都有一段代价计算**，
而且那些计算只需要纸笔和几行 Python 就能复现。本章的写法就是把这些推理链补全，
让读者能自己走一遍，而不是背结论。

具体来说，报告在动手排流水之前做了**四个设计讨论**（报告 §讨论，`:77-164`），
它们构成了本章的骨架：

| 讨论 | 报告位置 | 要回答的问题 | 本章 |
| --- | --- | --- | --- |
| 每个 AI Core 负责多少个 q head | `:81-116` | 算存比往哪边倒？L0C / UB 装得下吗？ | §3.1 |
| output accumulator 在哪里做 | `:118-130` | L0C 不能原地乘标量，怎么办？ | §3.2 |
| 能不能用 1C1V | `:132-144` | 三条 bound 哪一条先到？ | §3.3 |
| KV 的 ND2NZ 何时做 | `:146-154` | 随路转换为什么不行？ | §3.4 |

四个讨论之后才排流水（`:166-210`）、才有六条优化技巧（`:212-248`）。
本章**严格按这个顺序**推进。

### 1.3 一个必须先建立的直觉：这是「搬运」与「计算单元」的赛跑

昇腾的一个 AI Core 不是「一个处理器」，而是**一组固定分工的单元**（见第 19 章 §2）：

* **Cube（AIC / MAD）**：矩阵乘，输入必须在 L0A/L0B，输出落在 L0C；
* **Vector（AIV）**：向量与标量运算，输入输出都在 UB；950 上每个 AI Core 有**两个**；
* **MTE1/2/3**：三条搬运流水线（L1→L0、GM→片上、片上→GM）；
* **FixPipe（FIX）**：把 L0C 的结果搬出去（到 UB 或 GM）；
* **Scalar（S）**：标量与配置，负责发指令、读写同步旗标。

一个 attention kernel 慢在哪，几乎总是等价于问：**这些单元里谁先忙满、谁在等谁**。
报告的四个讨论，本质上是在四个不同维度上做这件事：算力 vs L2 带宽（§3.1）、
L0C 的写权限（§3.2）、MTE2 的队列深度 vs FixPipe 带宽（§3.3）、
BIU 的访存合并条件（§3.4）。

本章的全部图示、公式、代码都指向同一件事：**把「谁等谁」算清楚**。这也是第 28 章
「算子设计方法论」在昇腾上的一个具体样本。

### 1.4 本章的记号表

后面几节会反复用到下面这些符号，先在这里固定：

| 记号 | 含义 | 本章取值 |
| --- | --- | --- |
| `h_q` | 一个 q token 的 query head 数 | 64 或 128（V4.1 是 128，但一个 AI Core 只拿 64） |
| `MM_M` | 一个 AI Core 负责的 head 数 | 64（`config.h:28`） |
| `MM_M_PER_V` | 分给每个 VECTOR Core 的 head 数 | 32（`MM_M / 2`，`config.h:54`） |
| `d_qk` / `d_v` | QK 与 VO 的维度 | 均为 512（`config.h:39-40`） |
| `topk` | 每个 q token 关注的 KV 数 | 512 或 1024（本章讨论）；实际是 64 的倍数 |
| `B_TOPK` | 沿 KV 方向的分块大小 | 64（`config.h:55`）；报告说「一般取 64 或 96」（`:43`） |
| `B_TOPK_PER_V` | 每个 VECTOR Core 分到的 KV 块 | 32（`B_TOPK / 2`，`config.h:56`） |
| `sm_scale` | softmax 的缩放标量 | 通常 `d_qk ** -0.5 ≈ 0.044` |
| `1C1V` / `1C2V` | 一个 Cube 配几个 Vector | 本算子用 **1C2V**（`config.h:53`） |

一句话：**`h_q`、`MM_M`、`MM_M_PER_V` 是「谁算多少」，`B_TOPK`、`B_TOPK_PER_V`
是「谁搬多少」，这两个维度上的切分决定了后面所有的 bound。**

---

## 2. 算法回顾：单个 Q token 的稀疏 attention

本节对应报告的「算法回顾」一节（报告 `:11-73`）。本章不重推 online softmax 的
等价性（那是第 15 章 §2.3–§2.5 的事），只做两件事：把形状固定下来，
把报告里那个**非标准**的 `running_max_for_softmax` 说清楚。

### 2.1 数学形式：gather → P → softmax → PV

报告明确写：**不同的 q token 总是分开处理的**，所以「只考虑单个 Q token」
（报告 `:15-17`）。给定（报告 `:19-27`）：

* `Q`：`[h_q, d_qk]`
* `KV`：`[s_kv, d_qk]`
* `indices`：`[topk]`，每个元素是一个 `[0, s_kv)` 的 token 下标（非法位置填 `-1`
  或任意 `>= s_kv` 的值，见 `FlashMLA/README.md:157`）
* `sm_scale`：一个标量，通常是 `d_qk ** -0.5`

前向传播要算四步（报告 `:29-37`）：

$$gathered\_kv = KV[indices]\ [topk, d_{qk}]$$

$$P = Q \cdot gathered\_kv^\top\ [h_q, topk]$$

$$S = softmax(P \cdot sm\_scale)\ [h_q, topk]$$

$$out = S \cdot gathered\_kv[:,\ :d_v]\ [h_q, d_v]$$

本章和报告都用 `d_qk = d_v = 512`（V4.1 的 `head_dim`），`h_q = 64` 或 `128`，
`topk` 一般是 512 / 1024——这三个数后面会反复出现，因为**几乎所有 bound
都由它们算出来**。

注意第一步：`gathered_kv = KV[indices]` 是一次**按索引的 gather**，不是连续读。
这是整章所有麻烦的源头——稀疏访问打碎了「随路做 layout 转换」的可能（§3.4），
也让 MTE2 的请求数爆炸（§3.3、§5.1）。

![图 1：DSA 的稀疏访问与 TopK 索引](figures/01-dsa-sparse-topk.md)

图 1 把「KV 是连续的、访问是稀疏的」这件事画了出来：`KV` 在显存里是
`[s_kv, 512]` 的连续矩阵，而 `indices` 表把每个 q token 的读取位置打散。
读的时候是一次一个 token（每个 512 个 bf16，即 1 KB）。

### 2.2 online softmax 的递推

softmax 有全局依赖（分母要对所有 `topk` 个位置求和），所以不能「读一块算一块」。
报告用的是 Flash Attention 同款的 **online softmax**（报告 `:41`）。把 `topk`
切成若干块、每块 `B_TOPK` 个 token（报告 `:43`；`B_TOPK` 一般取 64 或 96，
但**本章引用的 `config.h` 里 `B_TOPK = 64`**，见 §3.3 末尾的说明）。

报告维护的状态量有四个（报告 `:45`），shape 都是 `[h_q]`（`out_accum` 是
`[h_q, d_v]`），**全部 float32**：

| 状态量 | shape | 初值 | 语义 |
| --- | --- | --- | --- |
| `running_max` | `[h_q]` | $-\infty$ | 到目前为止每行 P 的**真实**最大值 |
| `running_max_for_softmax` | `[h_q]` | $-\infty$ | 上一次 rescale 时**用来做 exp 的**那个 max |
| `running_sumexp` | `[h_q]` | 0 | 到目前为分母累加的和 |
| `out_accum` | `[h_q, d_v]` | 0 | 到目前的未归一化输出 |

关键点：**这里有两个 max，不是一个**。教科书的 online softmax 只需要一个
`m`（每次更新都立刻把累加器乘上 `exp(m_old − m_new)`）。报告这里多了一个
`running_max_for_softmax`，它**故意允许滞后**——只有滞后超过阈值才去更新。
这是 §2.3 的 skip-scale 的前提。

对每个 KV 块 `k`，报告的更新流程是（报告 `:47-73`）：

1. 按 `indices[B_TOPK*k : B_TOPK*(k+1)]` 稀疏读入 `gathered_kv`，`[B_TOPK, d_qk]`；
2. 算 `P = Q @ gathered_kv^T`，`[h_q, B_TOPK]`；
3. 算 `P` 每行的 `cur_max`，`[h_q]`；
4. `running_max = max(running_max, cur_max)`；
5. **检查**是否有某行满足 `running_max[i] - running_max_for_softmax[i] > RESCALE_THRES`；
6. 若有（**只有此时才**）做 online softmax 的 rescale：
   1. `out_accum <- out_accum * exp(running_max_for_softmax - running_max)`；
   2. `running_sumexp <- running_sumexp * exp(running_max_for_softmax - running_max)`；
   3. `running_max_for_softmax <- running_max`；
7. `S = exp(P - running_max_for_softmax)`，cast 成 bfloat16，`out_accum += S @ gathered_kv`；
8. `running_sumexp += S.sum(dim=-1)`。

![图 2：online softmax 的递推](figures/02-online-softmax-recurrence.md)

图 2 强调第 5–6 步是一个**条件分支**：大多数块会跳过 rescale，只做第 7–8 步。
这也解释了为什么这是一个可以省下大量带宽的优化（§5.6）。

### 2.3 skip-scale 与 `RESCALE_THRES ≈ 6`

`RESCALE_THRES` 是本章第一个需要「算一下」的超参数。报告的原话是（`:57`）：

> 检查是否有某一行 `i` 满足 `running_max[i] - running_max_for_softmax[i] > RESCALE_THRES`，
> 其中 `RESCALE_THRES` 为一超参数，在我们的 kernel 中一般取 6

为什么要用一个阈值，而不是「一变就 rescale」？报告在第 6.4 步给了动机（`:67-68`）：

> 这相当于更新了 online softmax 所使用的 max，因此
> `exp(running_max - running_max_for_softmax)` 总是 $\le \exp(6)$ 的

把这句话翻成数学。记 `m* = running_max`，`m_s = running_max_for_softmax`。
每次 rescale 之后二者相等；之后随着新块到来，`m*` 只会不减。设
`δ = m* − m_s ≥ 0`。那么在**下一次 rescale 之前**，用来算 exp 的是 `m_s`，
而累计到现在为止的真实 max 是 `m*`，于是：

* 对**新来的块**，`S = exp(P − m_s) = exp((P − m*) + δ) ≤ exp(0 + δ)`，
  即最大值最多被放大 `exp(δ)` 倍；
* 对**已经累加的旧块**，`out_accum` 与 `running_sumexp` 里存的是按 `m_s` 排序的量，
  一旦 rescale 就要整体乘 `exp(m_s − m*) = exp(−δ)`。

`δ` 越大，`S` 被放大的倍数越大；`δ` 越小，rescale 越频繁。取 `δ ≤ 6` 的意义是：
**`S` 中最大的元素不会超过 `exp(6) ≈ 403`，而 rescale 时乘的因子不会小于
`exp(−6) ≈ 2.5e−3`**——两者都还在 float32 的舒适区间里，同时 rescale 的次数被压到最低。

为什么是 6 而不是 5 或 8？报告没有给推导，只给了「一般取 6」这个工程取值
（`:57`；代码里同样写死 `RESCALE_THRESHOLD = 6`，
`csrc/ascend_kernels/prefill/sparse/config.h:43`）。本章不替它编一个推导，
只指出两点**可核验**的事实：

1. `exp(6)` 是 float32 能轻松表示的量级，不会溢出；
2. 阈值是**作用在 `sm_scale` 之后**判断的——代码把阈值预先除掉 `sm_scale`
   （`config.h:43` 的注释写的是 `(cur_max - max_for_scale) * sm_scale <= RESCALE_THRESHOLD`，
   `run()` 里预计算 `aux_params = { RESCALE_THRESHOLD / params.sm_scale }`，
   `kernel_body.h:1660-1662`）。也就是说 **`6` 是作用在 `P * sm_scale` 上的阈值**，
   不是作用在原始 `P` 上的。这一点在读源码时最容易看错，本章习题 E4 会专门考。

---

## 3. 四个关键设计讨论

本节是整章的重心，对应报告 §讨论（`:77-164`）。四个讨论的结论合起来，就是
报告在排流水前的「设计定稿」：

> 64 个 head / AI Core，1C2V，output accumulator 放在 UB 上，
> KV 经由 UB 和 SIMD VF 完成 ND2NZ 的中转。
> —— 报告 `:168`

下面四个小节把这句话拆开，逐个给推理。

### 3.1 每个 AI Core 负责多少个 q head：先算存比

**问题是**：一个 AI Core 上的 Cube 一次要扛多少个 q head，才能让矩阵乘单元
不闲着？报告的思路是算**算存比**：把「这个 head 数下 Cube 要花多少周期」和
「要喂这么多 head，L2 最少要花多少周期搬数据」都算出来，谁大谁就是瓶颈。

报告给的脚本（`:85-100`）是：

```python
d = 512
l2_total_bw = 5e12
num_cores = 32
freq = 1650e6
l2_bw = l2_total_bw / num_cores / freq  # Per AI Core per cycle

for h_q in [16, 32, 64, 128]:
    for topk in [512, 1024]:
        cube_cycles = h_q * topk * (d+d) // 4096
        qo_mem_vol = h_q * d * 2 * 2
        kv_mem_vol = topk * d * 2
        mem_cycles = (qo_mem_vol + kv_mem_vol) / l2_bw
        print(f"{h_q:3d}, {topk=:4d}, {cube_cycles=:6d}, {mem_cycles=:6.0f}")
```

把每一行读懂：

* `l2_bw`：把 **L2 总带宽 5 TB/s** 平均分到 **32 个 AI Core**（`GRID_SHAPE = 32`，
  `config.h:74`），再除以 **1650 MHz** 的主频，得到「每个 AI Core 每个周期能拿到
  多少字节」。算出来约 **94.7 B/cycle**。
* `cube_cycles = h_q * topk * (d+d) // 4096`：`P = QKᵀ` 要 `h_q·topk·d` 次乘加，
  `PV` 再要 `h_q·topk·d` 次（`d = d_qk = d_v = 512`，所以 `d+d`），
  一共 `2·h_q·topk·d` 次 MAC；Cube 打满是 **4096 FMA/cycle/AICore**（报告 `:85`），
  所以周期数是 `h_q·topk·(d+d) / 4096`。
* `qo_mem_vol`：每个 q token 要读 `Q`（`h_q·d` 个 bf16 = 2 字节）和写 `O`
  （同样多），所以 `h_q·d·2·2` 字节。
* `kv_mem_vol = topk·d·2`：这个 q token 要读的 `gathered_kv` 是 `topk·d` 个 bf16。
* `mem_cycles`：上面两者相加除以 `l2_bw`。

报告跑出来的表（`:104-111`）是：

```yaml
16, topk= 512, cube_cycles=  2048, mem_cycles=  5883
 16, topk=1024, cube_cycles=  4096, mem_cycles= 11419
 32, topk= 512, cube_cycles=  4096, mem_cycles=  6229
 32, topk=1024, cube_cycles=  8192, mem_cycles= 11765
 64, topk= 512, cube_cycles=  8192, mem_cycles=  6921
 64, topk=1024, cube_cycles= 16384, mem_cycles= 12457
128, topk= 512, cube_cycles= 16384, mem_cycles=  8305
128, topk=1024, cube_cycles= 32768, mem_cycles= 13841
```

于是报告下结论（`:114`）：**`head >= 64` 才是 compute bound；`head = 64` 时
compute 与 memory 大致均衡。**

自己验证一遍：`h_q = 32, topk = 1024` 时 `cube_cycles = 8192 < mem_cycles = 11765`，
memory 更贵 → memory bound；`h_q = 64, topk = 1024` 时 `16384 > 12457`，翻过来了；
`h_q = 64, topk = 512` 时 `8192` vs `6921`，恰好有点余量。你可以直接跑
本章的 `code/01-head-roofline/head_roofline.py` 复现这张表（本机可跑，
输出与报告逐行一致）。

**为什么不是 128？** 报告给了三条理由（`:116`）：

1. **L0C 装不下。** `O`（128 × 512，float32）就是 256 KB，而 L0C 只有 128 KB
   （第 19 章 §2.2 的表是 Atlas A2 口径；950 的 L0C 容量以平台为准，
   但报告明确说 128 个 head 会把 L0C「完全占满」）。于是 accumulator 只能挪到
   VECTOR 侧的 UB，就必须用两个 AIV，还要做**细粒度 Pipeline（拷贝 - 累加）**，
   「很难写」。
2. **没必要。** 根据 micro-benchmark，**MMAD 的 `M` 取 64 就足以打满 Cube 吞吐**
   （报告 `:116`）。也就是说 `head = 128` 买不到更多算力。
3. 这条与第 1 条联动：一旦 accumulator 挪走，`O` 的每块结果都要经 FixPipe
   搬到 UB 再乘标量再搬回去，白白多出一整轮搬运。

这个「M ≥ 64 就够」的判断在源码里能对上：`config.h:28` 的注释写
`MMA_M = ku::ceil(H_Q, 32u)` 并解释「MMA_M = 32 已经 memory-bound（L2 bw bound）；
MMA_M = 32 对应 MMA_M_PER_V = 16，正好是一个 L0C fractal 的高；MMA_M 小于 32
会让 FixPipe → UB 的拷贝出问题」。也就是说 M 方向的有效下限是 32，
而 64 是**打满吞吐**的取值——两个数一致地指向「一个 AI Core 管 64 个 head」。

#### 3.1.1 这个模型故意忽略了什么

理解一个性能模型，和记住它的结论一样重要。报告的算存比模型只算了两项：

* **Cube 的 MAC 数**（除以 4096 FMA/cycle）；
* **L2 的字节数**（Q、O、gathered KV，除以 5 TB/s ÷ 32 核 ÷ 1.65 GHz）。

它**故意没有算**的东西至少有四类，每一类都在后面的章节里以别的方式回来：

1. **HBM → L2 那一段。** 模型用的是 L2 带宽，隐含假设「KV 已经/即将在 L2 里」。
   真实情况是 KV 在 HBM，第一次读要过 HBM；报告在 decode 侧专门加了
   L2 hint 与「8 MB 小 cache 复用」的启发式（`kernel_body.h:1675-1685`），
   就是在处理这段。
2. **L1 → L0 与 L0C → UB 这两段。** 它们不走 L2，模型里没有它们；但 §3.3 的
   FixPipe bound 与 §5.5 的 L1 bank 配平分别处理了这两段。
3. **UB 的读写。** §3.4 决定的 ND2NZ 绕路会把 KV 在 UB 里读一遍写一遍，
   这部分流量在算存比模型里是零。§5.3 的四条技巧全是在优化它。
4. **同步与发射开销。** 模型假设每个周期都在干活；§4.4 说的阻塞式同步、
   MTE2 队列深度，都是纯开销。

所以正确的用法是：**算存比模型用来做第一刀的粗筛**（head 数取 16/32/64/128
里的哪个），细活要交给后面四个讨论和六条技巧。这也是第 28 章「先建解析模型、
再用 profiler 校正」的一个具体例子。

### 3.2 output accumulator 放在哪：L0C 还是 UB

rescale 要把 `out_accum` 整体乘一个标量 `exp(m_s − m*)`。问题在于：

> L0C 上的数据只能被 FixPipe 搬出或被 CUBE Core 用作累加器，无法原地乘以标量。
> —— 报告 `:120`

这条限制是昇腾存储层级的直接后果（第 19 章 §2.2）：L0C 是 Cube 的专用累加器，
没有「Vector 就地读写 L0C」这条路。报告列了两种可行方案（`:120-130`）：

**方案一**：在 VECTOR Core 上维护 accumulator。每次需要 rescale 时，把 Cube 上
这一次的累加结果经 FixPipe 搬到 UB，加到 VECTOR 侧的 accumulator 上（再乘标量）。

**方案二**：accumulator 始终留在 L0C。rescale 时把 L0C 拉到 UB，乘完标量，
再**放回 L0C**——而「放回 L0C」这件事只能通过**乘一个单位矩阵**来做。
报告列了它的两笔代价（`:126-128`）：

1. 多消耗 Cube 算力：把 64 × 512 的矩阵从 UB 搬回 L0C，至少要做一个
   `64 × 64 × 512` 的 float 矩阵乘（乘以单位矩阵）；
2. 若要**不损失精度**，必须关掉 HF32 模式，此时 Cube 算力降到
   **256 FMA/cycle/AICore**，那个 `64 × 64 × 512` 的矩阵乘要 **8192 周期**，
   「根本无法接受」。

**结论**（报告 `:130`）：选**方案一**，accumulator 放 VECTOR 侧（UB）。

顺手核对一下那 8192 周期怎么来的：`64 × 64 × 512 = 2,097,152` 次 MAC；
除以 256 FMA/cycle = **8192 周期**。完全对得上。这也解释了为什么 §3.1 里
「head = 128 必须把 accumulator 挪到 UB」是个坏消息——它会把 rescale 也一起
复杂化。

在源码里，这个决定表现为：`config.h:83` 有 `float output_accum[MMA_M_PER_V][D_VO]`，
它落在 UB 的结构体 `UnifiedBufferMemoryPlan` 里；而 Cube 侧每算完一块 `O`
就用 FixPipe 把 `cur_output_frag` 推到 UB（`kernel_body.h:1493-1524` 的
`run_sv_gemm` → `trigger_o_copy_out`），VECTOR 侧再用 `add_and_scale_o_vf`
把它加进 `output_accum`（`kernel_body.h:686-726`，调用点在 `run_scale_o`，
`kernel_body.h:1195-1250`）。报告的「方案一」在代码里就是这么落的。

### 3.3 为什么不能 1C1V：三条 bound

昇腾 950 每个 AI Core 是「**1 个 Cube + 2 个 Vector**」（第 25 章 §2.1：
32/28 个 Cube、64/56 个 Vector）。自然的想法是：既然 Cube 是瓶颈，就把工作量
按 1C1V 配，另一个 Vector 去跑别的小算子或通信，整体 MFU 更高。报告说
**不行**，并给了三条独立的 bound（`:132-144`）。

**bound ①：MTE2 Issue Queue Entry。** 单个 VECTOR Core 同一时刻最多只能有
**16 个**「已发出但未完成」的 MTE2 拷贝请求。哪怕用了 gather2（一次拷两个 token），
一轮也只能拷 `16 × 2 = 32` 个 token——对 `B_TOPK = 64` 来说，「太少了」
（报告 `:138`）。

**bound ②：Dequantize。** decoding 时 KV 是 FP8/FP4，要先反量化成 bfloat16
再送 Cube。报告说根据 micro-benchmark，**1C1V 时反量化速度赶不上 Cube 计算速度**，
反量化成了瓶颈（`:140`）。这和第 35 章讨论的 DeepGEMM/MQA 是同一类问题：
量化格式的收益是省显存带宽，代价是引入一个必须在 VECTOR 侧做的转换步骤。

**bound ③：FixPipe。** 只用 1C1V 时，FixPipe 向外搬运的速度只有
**128 B/cycle/AI Core**。报告接着算了一笔账（`:142`）：

* 每个块的 MMA 计算周期：`MMA_M · B_TOPK · (512+512) / 4096 = B_TOPK · 16`
  （`MMA_M = 64`）；
* FixPipe 搬运周期：`(MMA_M · 512 + MMA_M · B_TOPK) · sizeof(float) / 128 = 1024 + B_TOPK · 2`
  （`out` 的 64×512 个 float32，加上 `P` 的 64×B_TOPK 个 float32）；
* 「FixPipe 不成为瓶颈」的条件：`B_TOPK · 16 > 1024 + B_TOPK · 2`，
  化简得 `14 · B_TOPK > 1024`，即 **`B_TOPK > 73.14`**。

报告由此推出：要留余量就得取 96 或 128，而更大的 `B_TOPK` 会让 UB 更紧张，
也让 bound ① 更严重——三条 bound 互相加重。

**结论**（报告 `:144`）：给每个 Cube 配**两个 VECTOR Core（1C2V）**，
把 MTE2 发射、反量化、FixPipe 搬运、softmax 全都分摊开。

![图 3：1C1V 的三条 bound](figures/03-1c1v-bounds.md)

图 3 把三条 bound 与它们各自的来源量画在一起：队列深度是**指令级**的硬上限，
反量化是**吞吐级**的竞争，FixPipe 是**带宽级**的等式。

**一个容易读错的点**：报告算的是 **1C1V 的** FixPipe 条件，所以得到
`B_TOPK > 73.14`。但**最终代码用的是 1C2V，`B_TOPK = 64`**
（`config.h:53-55`）。这两者不矛盾——1C2V 把 `MMA_M` 与 `B_TOPK` 的搬运
都分给两个 AIV（`MMA_M_PER_V = 32`、`B_TOPK_PER_V = 32`），FixPipe 的压力与
1C1V 不同。把「报告里的 `B_TOPK > 73.14`」当成「代码里 `B_TOPK` 必须 > 73」
是本章最常见的误读，习题 E6 会考它。

### 3.4 KV 的 ND2NZ 何时做

这是四个讨论里最「硬件」的一个。

**背景**：KV 在显存里是 **ND layout**（row-major），shape `[s_kv, d_qk]`，
stride `[d_qk, 1]`；而 Cube 的输入**必须是 NZ layout**（fractal 排布，
见第 21 章的 layout 章节）。正常做法是：**从显存搬到 L1 时随路做 ND2NZ**，
昇腾的 `asc_copy_gm2l1_nd2nz` 就是干这个的——Q 就是这么搬的
（`kernel_body.h:1379-1397` 的 `run_load_q`）。

**问题**：KV 是**按 token 稀疏 gather 的**，一次一个 token，无法启用随路的
ND2NZ（报告 `:150`）。

**第一个想法**（报告 `:152`）：把 GM → L1 拷贝里的一个 token（512 个 bf16）
当成 32 个 32 B 的数据块，这 32 块在显存里连续、在 L1 里以 `B_TOPK × 32 B`
为 stride 排布，看起来像是一次带 stride 的搬运。**但实验显示不行**：

> AI Core 内部的 BIU（负责发送访存指令的单元）只会在**源和目标都连续**的时候
> 才会触发访存合并，而像这里这种「只有源连续，而目的不连续」的情况是无法合并
> 访存的，导致 L2 带宽利用率很低，整体瓶颈位于 L2 带宽。
> —— 报告 `:152`

这是全章最值得记住的一条硬件事实：**访存合并（memory coalescing）需要源和目的
都连续**。「源连续、目的 strided」在 NV 上常常还能靠 coalescing 兜住，
在这里不行。

**结论**（报告 `:154`）：**唯一的做法**是

1. 先把 KV 从显存拷到 **UB**（MTE2）；
2. 用 **SIMD VF** 在 UB 里把 ND 转成 NZ；
3. 再把 NZ 结果从 UB 拷到 **L1**（MTE3），供 Cube 的 `L1 → L0B` 使用。

![图 5：KV 的 ND → UB → NZ → L1 路径](figures/05-kv-nd2nz-path.md)

图 5 把这条绕路画清楚：直接 GM→L1 的箭头被打叉（无法随路转换 + 无法合并访存），
实际路径是 GM→UB→(SIMD VF)→UB(NZ)→L1。

代码完全对应：`run_issue_kv_gather` 把 KV 拷进 UB 的 `gathered_kv`
（`kernel_body.h:1018-1075`，decode 时是 `gathered_kv_fp4`），
`run_kv_nd2nz_and_push_to_cube` 先调 `kv_nd2nz_for_prefill_vf`
（prefill）或 `kv_dequant_and_nd2nz_for_decode_vf`（decode）得到
`gathered_kv_in_nz`，再用 `asc_copy_ub2l1` 推到 L1
（`kernel_body.h:1077-1112`）。

这里的 NZ 排布还要注意一个细节：UB 里的 NZ buffer 是
`bf16 gathered_kv_in_nz[(B_TOPK_PER_V+1)*D_VO]`（`config.h:81`）——
**故意多 pad 一行**（`B_TOPK_PER_V+1` 而不是 `B_TOPK_PER_V`），
注释写明「Add extra padding to avoid bank conflict during nd->nz」。
这是 §5.3 的第二条技巧。

![图 6：ND 与 NZ 的排布对照](figures/06-nd-vs-nz.md)

图 6 把同一个 `[B_TOPK, 512]` 矩阵在两种排布下的样子并排画出来，并标出
「多 pad 一行」在 NZ buffer 里的位置。

### 3.5 四个讨论的汇总

把 §3.1–§3.4 的结论压成一张表，方便对照报告原文：

| 讨论 | 报告位置 | 结论 | 关键推理 / 数字 | 落到源码 |
| --- | --- | --- | --- | --- |
| head 数 | `:81-116` | 每个 AI Core 64 个 head | `head ≥ 64` 才 compute bound；128 时 `O` = 256 KB 会把 L0C 占满；MMAD 的 M=64 已打满吞吐 | `MMA_M = 64`（`config.h:28`） |
| accumulator | `:118-130` | 放 VECTOR 侧的 UB | L0C 不能原地乘标量；搬回 L0C 要 `64×64×512` 矩阵乘，关 HF32 后 8192 周期 | `output_accum` 在 UB 结构体里（`config.h:83`） |
| 1C1V vs 1C2V | `:132-144` | 1C2V | MTE2 队列 16 个 entry；反量化跟不上；FixPipe 128 B/cycle → `B_TOPK > 73.14` | `NUM_VEC_CORE_PER_AI_CORE = 2`（`config.h:53`） |
| ND2NZ 时机 | `:146-154` | GM→UB→(SIMD VF)→L1 | 稀疏 gather 无法随路转换；BIU 只在源和目的都连续时合并访存 | `kv_nd2nz_for_prefill_vf`（`kernel_body.h:277-331`） |

这四行合起来就是报告 `:168` 那句「设计定稿」。**注意它们的顺序不是随意的**：
先定 head 数（决定 L0C 够不够），才需要决定 accumulator 放哪；定了 accumulator
才暴露出 Vector 侧的负载，才需要回答 1C1V 行不行；而 Vector 侧的负载又直接决定
ND2NZ 必须放在 UB 上做。**每一条结论都在给下一条创造约束**——这是本章最值得学的
「设计过程」形状。

---

## 4. 流水线设计

设计定稿之后，报告才开始排流水（报告 §流水线，`:166-210`）。这里的目标只有一句话：

> 让 CUBE Core 上的矩阵乘法计算单元时时刻刻都能打满，并且不要让其他操作
> （比如 softmax）成为瓶颈。—— 报告 `:170`

### 4.1 CUBE 侧 5 步与 VECTOR 侧 7 步

报告先给的是**最基本的流水思路**（报告 `:172-180`）。用 `[i]` 表示第 i 个 KV 块：

```text
1. 计算 P[i]   = Q @ K[i]^T
2. 计算 O      += S[i-1] @ V[i-1]
3. 计算 P[i+1] = Q @ K[i+1]^T
4. 计算 O      += S[i] @ V[i]
5. ...
```

注意这里的写法：**它是把两个 GEMM 交错排**，而不是「先算完一个块的所有事
再算下一个块」。CUBE 侧真正的 5 个动作，其实就是这四行的循环体：
`QKᵀ[i]`、`SV[i-1]`、`QKᵀ[i+1]`、`SV[i]`，加上开头的 `Q` 装载
（`run_load_q`）和结尾的 drain。

对应到源码，CUBE 侧是**两段嵌套循环**（`kernel_body.h:1619-1637` 的主循环）：

```text
run_load_q(cur_job)
run_qk_gemm(cur_job, 0)
do {
    run_load_q(nxt_job)                 // 下一个 q token 的 Q 先搬
    for k in [0, num_topk_blocks-1):
        run_qk_gemm(cur_job, k+1)       // QKᵀ[k+1]
        run_sv_gemm(cur_job, k)         // SV[k]
    run_qk_gemm(nxt_job, 0)
    run_sv_gemm(cur_job, num_topk_blocks-1)
} while (nxt_job is valid)
```

也就是：**先发 `QKᵀ[k+1]`，再发 `SV[k]`**。这正是「延后一拍」在代码里的形状。

VECTOR 侧因为要同时管 index、搬 KV、做 softmax、做 O scale、传 skip-scale 信号，
排出来是 7 步（报告 `:188-200`）。用「当前正在处理的块」记为 `i`：

1. 发起第 `i+5` 个 KV 块的 index 读取（GM → UB）；
2. 发起第 `i+3` 个块的 ND2NZ 转换与 UB → L1 的 MTE3 拷贝；
3. 等 `P[i+1] = Q @ K[i+1]^T` 完成，发起第 `i+1` 个块的 softmax，把结果写回；
4. 发起第 `i+5` 个块的 index 预处理（算 gather2 的 MTE 参数）；
5. 发起第 `i` 个块的 O scale（前提是这一块没有 skip-scale O）；
6. 和 SIMD VF 同步，拿第 `i+1` 个块要不要 skip-scale O，写进 SS buffer；
7. 发起第 `i+5` 个块的 MTE2 拷贝，把 KV 从显存拷进 UB。

把 7 步按「它服务的是哪个块」重新排一下，就很容易看出这是一条**深度 6 的流水线**：

| 距当前块的偏移 | 动作 |
| --- | --- |
| `i` | O scale（第 5 步） |
| `i+1` | softmax（第 3 步）、写 skip-scale 信号（第 6 步） |
| `i+3` | ND2NZ + UB→L1（第 2 步） |
| `i+5` | index 读 + index 预处理 + KV 的 MTE2 拷贝（第 1、4、7 步） |

源码里这个 6 就是 **`SYSTOLIC_ARR_DEPTH = 6`**（`kernel_body.h:1273`）：
VECTOR 侧维护一个长度 6 的环形数组 `systolic_arr_items[6]`，每轮把新块推进去，
再用 `ACTIVATE(offset, func)` 按偏移取出来执行相应的 7 个 `run_*` 函数
（`kernel_body.h:1273-1290`）。偏移量与报告 7 步的对应写在
`systolic_arr_activate` 里（`kernel_body.h:1289-1322`）：

```text
ACTIVATE(5, run_issue_index_fetch)          // 报告第 1 步
ACTIVATE(3, run_kv_nd2nz_and_push_to_cube)  // 报告第 2 步
ACTIVATE(1, run_softmax)                    // 报告第 3 步
ACTIVATE(5, run_process_index)              // 报告第 4 步
ACTIVATE(0, run_scale_o)                    // 报告第 5 步
ACTIVATE(1, run_set_is_rescale_triggered_flag)  // 报告第 6 步
ACTIVATE(5, run_issue_kv_gather)            // 报告第 7 步
```

报告的 7 步与代码的 7 条 `ACTIVATE` **一一对应**：softmax 是偏移 1、
ND2NZ + UB→L1 是偏移 3、O scale 是偏移 0、index / KV 相关的三个动作是偏移 5。
这说明报告那份流水不是示意图，而是代码的直接描述。

把报告 7 步、`ACTIVATE` 偏移、源码函数名三列对齐：

| 报告第几步 | `ACTIVATE(offset)` | 源码函数 | 起始行 |
| --- | --- | --- | --- |
| 1（index 读） | `ACTIVATE(5, ...)` | `run_issue_index_fetch` | `kernel_body.h:954` |
| 2（ND2NZ + UB→L1） | `ACTIVATE(3, ...)` | `run_kv_nd2nz_and_push_to_cube` | `kernel_body.h:1078` |
| 3（softmax） | `ACTIVATE(1, ...)` | `run_softmax` | `kernel_body.h:1114` |
| 4（index 预处理） | `ACTIVATE(5, ...)` | `run_process_index` | `kernel_body.h:983` |
| 5（O scale） | `ACTIVATE(0, ...)` | `run_scale_o` | `kernel_body.h:1195` |
| 6（写 skip-scale 信号） | `ACTIVATE(1, ...)` | `run_set_is_rescale_triggered_flag` | `kernel_body.h:1186` |
| 7（KV 的 MTE2 拷贝） | `ACTIVATE(5, ...)` | `run_issue_kv_gather` | `kernel_body.h:1018` |

一个小陷阱：**第 1、4、7 步共用 `offset = 5`**，所以它们不是「三个独立阶段」，
而是**在同一个阶段里按顺序发起三件事**（先读 index，再预处理，最后才发起 KV 拷贝——
因为 KV 拷贝要用到预处理算出来的地址）。这也是为什么 `NUM_INDEX_BUFS = 4`
（`config.h:69`）：`ACTIVATE(1, run_softmax)` 之后紧跟 `ACTIVATE(5, run_process_index)`，
至少得留 4 份 index buffer 才不会被覆盖。

![图 4：CUBE / VECTOR 双流水](figures/04-cube-vector-pipeline.md)

图 4 用时间线画出两条流水：上排是 CUBE 侧的 `QKᵀ` / `SV` 交替，
下排是 VECTOR 侧的 7 个 `run_*`；标出「`P[i]` 从 L0C 出来到被 `SV[i]` 用掉」
中间那段空窗，正好被 softmax + S 回写填满。

### 4.2 为什么把 `O += S@V` 延后一拍

报告给的答案很直接（`:184`）：

> 相当于将前一个块的 `O += S @ V` 延后计算，以给 softmax 操作留出时间：
> `P[i]` 在 (1) 结束后就已就绪，直到 (4) 开始时才会被用到，期间有充足的时间
> （大约 1024 周期）供 FixPipe 将 `P[i]` 从 L0C 拷贝至 UB、softmax、
> 以及将 `S[i]` 从 UB 拷贝至 L1 使用。

换句话说：`P[i]` 有两条下游依赖。它既要被 softmax 消费（算出 `S[i]`），
又要间接供 `SV[i]` 使用。如果让 `QKᵀ[i]` 和 `SV[i]` 背靠背执行，
`SV[i]` 会一直等 `S[i]`，而 `S[i]` 要等 FixPipe 把 `P[i]` 搬出 L0C、
VECTOR 做完 exp、再搬进 L1——这段时间 CUBE 就空转了。

把 `SV[i]` 推到下一轮（即紧跟 `QKᵀ[i+1]` 之后），CUBE 在等 `S[i]` 之前
先去算 `QKᵀ[i+1]`，**用一段独立的、不依赖 softmax 的矩阵乘把空窗填上**。
这就是「软件流水」里经典的「用独立工作填依赖延迟」。

报告给的量级是**约 1024 周期**（`:184`）。这个数从哪来？它和 §3.3 里 FixPipe 的
一笔账是同一个量级：`P` 是 64 × B_TOPK 的 float32，`B_TOPK = 64` 时
`64 × 64 × 4 / 128 = 128` 周期只是搬 `P`；再加上 softmax 的 VF 计算与
`S`（bf16）回写 L1。报告没有给推导，只说「大约 1024 周期」，本章按原文引用，
不替它补算（习题 E8 会让你自己用报告给的公式验一遍量级）。

### 4.3 skip-scale 信号：SS buffer + cross core flag

§2.3 说的 skip-scale 在 NPU 上还有第二层意义，报告写得很清楚（`:160`）：

> 因为 NPU 上 scale-O 很麻烦，需要使用 FixPipe 将 O 从 L0C 拷贝到 UB，
> 然后通过 VF 累加到 UB 的 `out_accum` 上，浪费 FixPipe、UB 与 SIMD VF 的带宽。

于是「这个块要不要 scale-O」这个判断一旦能省，就值得省。难点在于信号的两端
**不是同一类核**（报告 `:162`）：

* **生产者**：VECTOR Core 上的 softmax VF（它知道 `running_max` 的增量）；
* **消费者**：既有 CUBE Core（它要决定是否把 L0C 上的 `O` 清空 / 是否触发
  `trigger_o_copy_out`），也有 VECTOR Core（它要决定是否做 `add_and_scale_o_vf`）。

报告的做法是用 **SS buffer + cross core flag** 传递（报告 `:162`）。

源码里这套机制是三个部件：

1. **SS buffer**：一块被 scalar core 读写的特殊存储
   `volatile __ssbuf__ SSBufferMemoryPlan`（`kernel_body.h:914`），
   里面是 `uint32_t is_rescale_triggered[8][8][NUM_VEC_CORE_PER_AI_CORE]`
   （`config.h:110-112`）。两个 `8` 分别是 job_idx 与 kv_block_idx 取模的槽位——
   注释写明「Allocate 8 slots for job_idx and 8 slots for kv_block_idx
   so that there will be no data race」，即**用环形的槽位避免读写同一个位置**。
2. **写**：VECTOR 侧的 `run_set_is_rescale_triggered_flag`
   先 `wait_flag(PIPE_V, PIPE_S, ROW_MAX_MAX_DELTA_FULL)`（等 VF 把
   `row_max_max_delta` 写好），再写
   `ss_buf.is_rescale_triggered[...] = row_max_max_delta > threshold`，
   最后发 `CrossCoreSetFlag<4, PIPE_S>(SS_BUFFER_FULL_V2C)`
   （`kernel_body.h:1186-1193`）。
3. **读**：CUBE 侧在 `run_sv_gemm` 的第一个 tile 里
   `cross_core_wait_flag_2aiv<PIPE_S>(SS_BUFFER_FULL_V2C)`，读 SS buffer，
   `should_trigger_o_copy_out = is_rescale_triggered > 0`，
   若为真就 `trigger_o_copy_out<false>()`（先把上一块的 O 拉出去）
   （`kernel_body.h:1583-1590`）。VECTOR 侧的 `run_scale_o` 也读同一个格子
   （`kernel_body.h:1205-1207`）。

`CrossCoreSetFlag<4, PIPE>(id)` 里的 **`4` 是 mode**，含义是「发给同一个 AI Core
里的两个 AIV」，helper 会同时发 `id` 与 `id+16` 两份
（`kernel_body.h:69-81`）。这是本章唯一一处「一个 Cube 核要同时通知两个 Vector 核」
的同步，而它恰好是 1C2V 这个配置带来的。

![图 7：skip-scale 的信号传递](figures/07-skip-scale-signal.md)

图 7 画的是信号从「softmax VF 算出 `row_max_max_delta`」到「CUBE 决定是否清空
L0C」的整条路径，以及沿途要经过的两次跨核 flag。

### 4.4 「顺序发射、顺序执行」为什么放大问题

报告说这条流水复杂，给了两个原因（`:202-206`）。第一个就是上文的 SS buffer 同步：

> 算子依靠 SS buffer 传递 skip-scale 的信号，但能读写 SS buffer 的只有 scalar core，
> 而 skip-scale 的信号由 SIMD VF 生成，因此这会导致 SIMD VF → Scalar Core 的
> **阻塞式**同步。倘若此处没有处理好，便会导致 scalar core 等待 SIMD VF 完成、
> 其他单元出现空闲的情况。—— 报告 `:204`

关键词是**阻塞式**：不是「发个信号就走」，而是 scalar core 必须**停下来等** VF。
而 scalar core 一旦停了，它就发不出后续指令，别的单元也会跟着饿着。

第二个原因是全章另一条要记住的硬件事实（报告 `:206`）：

> 华为的各个硬件管线（PIPE）都是**顺序发射、顺序执行**的，比如对于 `PIPE_V` 来说，
> 发射 SIMD VF 的顺序与执行的顺序必须完全一致，如果前面的 SIMD VF 由于某些条件
> 不满足（比如 `asc_sync_wait` 条件尚不满足）而不能发射的话，就会阻塞整条硬件管线。

把这两条合起来看，就得到一个反直觉的结论：**在这个 kernel 里，「把某个同步
往前提一点」常常比「多算一点」更划算**，因为它可能让整条 `PIPE_V` 停摆。

代码里有一段**专门为这件事写的注释**，就在 `systolic_arr_activate` 里
（`kernel_body.h:1306-1319`）：

> If we've just launched `run_process_index`, then we launch `issue_kv_gather`
> after `set_is_rescale_triggered_flag` because
> 1) `run_process_index` is using VF, issuing `set_is_rescale_triggered_flag`
>    now won't cause VF to be idle
> 2) `run_issue_kv_gather` must wait for `run_process_index`. If we launch
>    `set_is_rescale_triggered_flag` after `issue_kv_gather`, the scalar core must
>    wait for `process_index` to complete which cause cube core to be idle

翻译过来：**当 `run_process_index` 正要占用 VF 时，就把
`set_is_rescale_triggered_flag` 插在它前面**——这样 VF 不闲着，而且 scalar core
也不会因为「先发 gather 再等 index 预处理」而卡住。这就是报告那句
「我们需要谨慎设计各个操作的发起顺序」在代码里的落地。

对应到源码，这个「要不要插到前面」的开关是 `issue_kv_gather_later`：
只有当即将处理的是一个 index 组的**组首**（`kv_block_idx % (INDEX_BLOCK_SIZE/B_TOPK) == 0`）时，
才把 `set_is_rescale_triggered_flag` 提到 `run_issue_kv_gather` 之前
（`kernel_body.h:1306-1318`）。因为这个情况下 `run_process_index` 会真的干活；
其余情况它是个空调用，提前插反而是噪音。

---

## 5. 优化技巧

报告最后一节列了六条优化（`:212-252`）。本节逐条讲它**在绕开哪个上限**，
并给出源码位置。

先给一张总览：**每条技巧都在照顾一个具体的资源，而且几乎都对应着前面某个 bound。**

| 技巧 | 报告位置 | 照顾的资源 / 绕开的上限 | 与哪条 bound 呼应 |
| --- | --- | --- | --- |
| gather2 | `:214-216` | MTE2 发射次数（队列深度 16） | §3.3 bound ① |
| KV cache layout | `:218-220` | 每个 token 的拷贝请求次数 | §3.3 bound ① |
| 高速 ND2NZ | `:222-232` | UB 读写吞吐、VF 的标量开销 | §3.4 的绕路成本 |
| icache | `:236-238` | 指令缓存（32K/16K/8K） | ——（编译器/代码体积） |
| L1 bank 配平 | `:240-242` | L1 的 bank 并行带宽 | §3.4 的 L1 侧 |
| skip-scale | `:244-252` | FixPipe / UB / VF 三份带宽 + 功耗 | §4.3 的信号同步 |

可以看出：**六条里有四条是在解决「搬运」**（gather2、layout、ND2NZ、L1 bank），
一条在解决「指令体积」（icache），一条在解决「重复计算」（skip-scale）。
这反过来印证了 §1.3 的说法——**这个 kernel 的主战场上，CUBE 是主角，
但胜负手大半在搬运侧。**

### 5.1 gather2：一次拷两个 token

**要绕开的上限**：单 VECTOR Core 的 MTE2 Issue Queue 只有 **16** 个 entry
（§3.3 bound ①）。如果每个 KV token 发一次 MTE2 请求，`B_TOPK = 64` 个 token
就要发 64 次请求，远超队列深度，发射端会堵住。

**技巧**（报告 `:216`）：设置 MTE2 请求的 **`burst_count = 2`**，
`src_stride` 取**两个 token 起始地址之差**，于是一次请求拷两个 token。

源码里 prefill 分支是这么写的（`kernel_body.h:1055-1068`）：
`n_burst` 取 2，`asc_copy_gm2ub_align` 的第 3 个参数是 `n_burst`、
最后两个参数是 `src_stride` 与 UB 侧的行距：

```cpp
uint32_t n_burst = cur_global_kv_ptr_offset != invalid_global_kv_ptr_offset ? 2 : 0;
asc_copy_gm2ub_align(
    (__ubuf__ gathered_kv_t*)cur_ub_gathered_kv_ptr,
    cur_base_ptr,
    n_burst,                         // burst_count = 2：一次两个 token
    D_QK * sizeof(gathered_kv_t),    // len_burst：一个 token 512 个 bf16
    0, 0, false,
    asc_load_l2_cache_mode::NORMAL_LAST_VICTIM,
    *(cur_ub_src_stride_in_gather2_ptr + i),   // src_stride：两个 token 的地址差
    D_QK * sizeof(gathered_kv_t)
);
```

两个 token 怎么配对？看 `process_kv_index_vf`（`kernel_body.h:87-255`）：
它把 index 向量**按奇偶拆开**（`asc_deintlv`），再取
`smaller = min(even, odd)`、`larger = max(even, odd)`
（`kernel_body.h:166-172`），于是每个 pair 的 `src_stride` 恰好是
`(larger − smaller) × 512 × 2` 字节（`kernel_body.h:186-190`）。
把它写成 `src_stride` 就能用一条指令拷两个 token。

报告还强调：**这些 MTE2 参数用 SIMD VF 算**，避免阻塞 scalar core
（报告 `:216`）。这呼应 §4.4 的「谁都不能把 scalar core 拖住」。

一个附带细节值得看：如果两个 index 都非法，代码把 `n_burst` 设成 0，
**直接跳过这次拷贝**，注释说这是为了「avoid one L2 slice being congested」
（`kernel_body.h:1057`）——非法 token 不产生任何流量。

### 5.2 Decoding KV Cache Layout：让 data 与 scale 相邻

**要绕开的上限**：每个 token 的**拷贝请求次数**（还是 MTE2 队列）。

旧版 FlashMLA 的 KV cache 布局是「先放一个块里所有 token 的 data，
再放所有 token 的 scale」（报告 `:220`）。对 CUDA 无所谓，但对昇腾意味着
**每个 token 要两次 MTE2 拷贝**（一次 data、一次 scale）。

本算子的改动是：**把一个 token 的 data 和 scale 相邻放置**，于是每次拷贝一次到位
（报告 `:220`）。这个新格式在官方 README 里有精确描述
（`FlashMLA/README.md:138-143`，`kv_cache_format.h:5-19`）：

* **V41（FP8）**：每 token 528 字节 = 512 字节 `e4m3`（覆盖全部 512 维，
  包括 64 维 RoPE）+ 16 字节 `e8m0` scale（每 32 维一个）；
* **V41_FP4**：每 token 288 字节 = 256 字节 `e2m1`（2 个/字节，偶下标在低 nibble）
  + 32 字节 `e4m3` scale（每 16 维一个）。

这条改动的收益是**把 MTE2 请求数减半**，直接缓解 §3.3 的 bound ①。
代码里 decode 分支只发**一次** `asc_copy_gm2ub_align`，一次搬
`F::BYTES_PER_TOKEN` 字节（`kernel_body.h:1040-1050`），正对应这个新布局。

### 5.3 高速 ND2NZ 的四条技巧

§3.4 决定了「KV 必须走 UB、用 SIMD VF 转 NZ」。这个 VF 的效率直接决定了
启动第一个 `QKᵀ` 之前的「空窗期」有多长（报告 `:224`）。报告给了四条技巧
（`:226-232`），逐条对应源码：

**① `asc_loadalign` + `vsstb`。** 用 `asc_loadalign` 从 ND 读，
用 `vsstb` 把每个 32 B 数据块**以一定间隔**写回 UB，就完成了 ND→NZ
（报告 `:226`）。源码：`kv_nd2nz_for_prefill_vf` 里
`asc_loadalign_postupdate` 读、`vsstb(..., POST_UPDATE)` 写
（`kernel_body.h:304-330`）。注意代码里的注释解释了为什么**必须**用 `vsstb`
而不是 `asc_storealign_postupdate`：

> `asc_storealign_postupdate` is buggy, it won't update the pointer, since CANN
> forgets to pass the pointer as reference. As a result we have to use `vsstb`.
> —— `kernel_body.h:283-284`

这是本章唯一一处「CANN 的已知坑」，值得单独记住。

**② 多 pad 一行，避免 bank conflict。** 在 NZ buffer 里额外 pad 一行
（报告 `:228`）。源码：`config.h:81` 的 `(B_TOPK_PER_V+1)*D_VO`，
以及 `config.block_stride = B_TOPK_PER_V + 1`（`kernel_body.h:287`）。
为什么要 pad？因为写回时每个 32 B 块以 `B_TOPK_PER_V` 行为 stride 落到 UB，
如果行数是 2 的幂（32），不同列会落到同一个 bank；pad 成 33 就打散了。
报告说的是「保证写回时没有 UB 的 bank conflict」。

**③ 自增指针，别让 VF 的小 scalar 成为新瓶颈。** 用
`asc_loadalign_postupdate` 与 `vsstb` 自带的「自增 pointer」功能，
免去手动更新指针的开销（报告 `:230`）。源码里这一点被顶到了极致：
整个 VF 里**没有一个显式的指针加法**，位置全靠 `post_update` 的
`offset` 参数推进（`kernel_body.h:304-330`）。甚至循环的嵌套顺序都是为了让
这个自增 offset **恒为正**——代码里的注释：

> `vsstb` only accepts positive `offset` for "post_update", so we must find an
> order which only increases (instead of decreases) `cur_gathered_kv_in_nz_ptr`.
> This is why we use a column (head dim) major loop layout.
> —— `kernel_body.h:294-296`

也就是说：**循环顺序是被「post-update 只能正增量」这条硬件/编译器约束反推出来的**。
这是本章最漂亮的一个「约束驱动的代码形状」的例子。

**④ 一次处理一对相邻的 256 B。** 报告说这是「为了利用满 UB 的所有 bank 的带宽」
（报告 `:232`）。源码：外层循环步长是 2（`j += 2`），中括号里同时维护
`cur_data[0]` / `cur_data[1]` 两个向量，分别写到两个 256 B 的 bank group
（`kernel_body.h:290-330`）；注释写明「To balance traffic between two bank
groups, we process two adjacent `NUM_BF16_IN_VEC` elements together so that
two 256B bank groups are interleaved」（`kernel_body.h:296`）。

报告说这四条做完之后，**该 VF 可以打满 UB 的读、写吞吐**（`:232`）。
注意这是**上游的声明**，本章没有实测。

### 5.4 icache：代码体积就是性能

报告给的数字很具体（`:238`）：昇腾 NPU 的指令缓存
（icache）只有 **32 K（CUBE Core）、16 K（VECTOR Core）、8 K（SIMD VF）**。
代码太大就会频繁换入换出。两条技巧：

1. **在 SIMD VF 里用硬件循环 `VLOOP`，不要用 `#pragma unroll` 展开**（报告 `:238`）。
   源码里到处是 `#pragma unroll 1`——它的意思正是「**不要**展开」。
   例如 softmax 里最内层的行循环注释写着
   `#pragma unroll 1 // Don't unroll to reduce code size, which prevents VF's ICache miss`
   （`kernel_body.h:595`）。
2. **当心函数内联导致代码体积成倍增长**（报告 `:238`）。报告说因此在
   VECTOR Core 结尾**把 main loop 和 drain loop 融合**成一个循环。源码：

   > We fuse the main loop and the draining loop together to reduce code size and
   > avoid icache miss. —— `kernel_body.h:1331`

   对应代码是 `while (true) { for (k...) { ... } if (is_draining) break; ... }`
   （`kernel_body.h:1330-1345`）。

这条技巧的意义是方法论层面的：**在昇腾上，「少写代码」本身就是一个优化手段**。
第 23 章的自动调度器在估延迟时并不看 icache，所以这条要靠人判断。

### 5.5 L1 两个 bank 的配平

**硬件事实**（报告 `:242`）：Ascend 950 的 L1 分成两个 bank，
分别管 `0 ~ 256 K` 与 `256 K ~ 512 K` 两段地址空间。如果所有 Tensor 都落在
同一个 bank，另一个 bank 的带宽就白扔了。

**技巧**：把每个 Tensor（Q、量化好的 K 等）切成两份，分别放两个 bank，
随时打满两个 bank 的带宽（报告 `:242`）。

源码里这条写得非常直白（`kernel_body.h:907-912`）：

```cpp
// Since L1 buffer is splitted into two bank groups (the first one spans
// address 0 ~ 256K while the second one spans 256K ~ 512K), we split every
// buffer into two parts to balance load between those two bank groups
__cbuf__ bf16* l1_q_buf_lo          = (__cbuf__ bf16*)(0*1024);
__cbuf__ bf16* l1_gathered_kv_buf_lo = (__cbuf__ bf16*)(32*2*1024);
__cbuf__ bf16* l1_s_buf              = (__cbuf__ bf16*)(256*1024 - 64*64*sizeof(bf16)/2);
__cbuf__ bf16* l1_gathered_kv_buf_hi = (__cbuf__ bf16*)(288*1024);
__cbuf__ bf16* l1_q_buf_hi           = (__cbuf__ bf16*)((512-32*2)*1024);
```

每个张量都有 `_lo` 和 `_hi` 两份，`_hi` 的地址都落在 256 K 以上。
搬运时也要成对写：`run_load_q` 里 Q 用两次 `asc_copy_gm2l1_nd2nz` 分别写 lo/hi
（`kernel_body.h:1379-1397`），`run_qk_gemm` 用两次 `asc_copy_l12l0a`
（`kernel_body.h:1417-1434`），`run_sv_gemm` 用两次 `asc_copy_l12l0b_transpose`
（`kernel_body.h:1547-1560`）。**这是「地址规划即性能优化」的教科书例子**：
没有任何算法变化，只是把地址错开。

### 5.6 skip-scale 的额外收益：功耗与频率

报告列了 skip-scale 的三条好处（`:248-252`）：

1. 减少 FixPipe 的拷贝量与 VF 计算量；
2. 降低对 UB 读、写带宽的压力；
3. 通过减少计算与数据搬运量，**降低算子运行时所需的功率，进而获得更高的频率
   并提高性能**。

第 3 条是昇腾上特有的：芯片在功耗约束下会动态调频，省下来的功耗会**换成频率**。
这在本教材里是第一次出现「省算力 → 换频率 → 再换性能」这条链，
它说明**性能模型里不能只有周期数，还要有功耗**。第 28 章的 roofline 模型
同样没有覆盖这一项，这是昇腾平台上的一个已知建模缺口。

### 5.7 那两个 TFlops 数字是怎么测出来的（口径）

§1.1 引用的 410 / 360 TFlops 来自官方 README（`:13,53,63`）。
要正确理解它们，得知道官方 benchmark 脚本怎么算：

* 入口是 `tests/test-sparse-prefill.py`（prefill）与 `tests/test-sparse-decode.py`（decode）；
* 时间用 `bench(...).get_kernel_time("sparse_attn_fwd")` 取得（**kernel 时间**，
  不含 host 侧 launch 开销）；
* TFlops 用**理论 FLOPs 除以实测 kernel 时间**得到：
  `flops = count_flop_and_mem_vol(p, t).fwd_flop / ans_time / 1e12`
  （`FlashMLA/tests/test-sparse-prefill.py:46-51`）；
* 形状由脚本里的 `TestParam` 列表给出（`h_q ∈ {128, 64}`、`d_qk = 512`、
  `(s_kv, topk) ∈ {(128,128), (256,256), (512,512), ...}`），
  「典型工况」的具体取法**报告没有逐项列出**，本章按 README 的 410/360 引用。

也就是说，这两个数字是**「理论 FLOPs / kernel 时间」的吞吐口径**，
不含 host 开销，环境是 **Ascend 950 NPU + CANN 9.2.0 及以上**
（`FlashMLA/README.md:85-88`）。本章不做任何外推。

---

## 6. 回到 TileLang：哪些能表达，哪些要下钻，哪些必须手写

这是本章对读者最有价值的一张表。前面五节讲的全是「官方怎么做的」，
这一节回答「**我需要自己写的时候，用 TileLang 能写到哪一层**」。

判断依据有三个来源：

* 第 25 章给出的 `T.*` 原语清单（`T.alloc_*`、`T.copy`、`T.gemm`、`T.dual_copy`、
  `T.SimtVF`/`T.SimdVF`、`T.Pipelined`、`T.annotate_buffer_versions`、
  `T.ascend_set_flag`/`_wait_flag`/`_pipe_barrier`/`cross_core_*`、`T.simd.*`）；
* 本章引用的 FlashMLA 真实指令（`asc_loadalign*`、`vsstb`、`asc_mmad`、
  `asc_copy_gm2ub_align`、`asc_copy_ub2l1`、`CrossCoreSetFlag`……）；
* 在锁定源码树里逐个 grep 确认哪些原语真的存在（见 `code/04-tilelang-expressibility/`）。

### 6.1 三类的判据

| 类别 | 判据 | 你该怎么做 |
| --- | --- | --- |
| **A. 直接可表达** | 只需要「tile + 循环 + `T.gemm`/`T.copy`/逐元素运算」就能描述，没有依赖某条特定硬件指令的语义 | 直接写 TileLang，交给第 23 章的自动调度 |
| **B. 可表达但要下钻** | TileLang 有对应原语，但需要显式指定向量宽度、mask、buffer 版本、跨核 flag 或 `T.SimdVF` 区域，不能再指望自动调度 | 用 `T.SimdVF` + `T.simd.*` + 注解原语手写，仍走 TileLang 的 codegen |
| **C. 必须内建** | 结论依赖某条 CCE/Ascend C 内建的名字、参数或位级语义（分布 `BRC`/`UNPK4`、`n_burst`/`src_stride`、`unit_flag_ctl`、`__ssbuf__`……） | 落到 Ascend C / PTO / SIMD VF，TileLang 只负责把它包起来 |

### 6.2 类别 A：TileLang 能直接表达的技术点

| 本章技术点 | 报告/代码位置 | TileLang 写法 | 对照第 25 章 |
| --- | --- | --- | --- |
| online softmax 的四个状态量与递推 | 报告 `:45-73` | 普通 TIR 变量 + `T.Parallel` 逐元素；`T.reduce_max` / `T.reduce_sum` 做行归约 | §4.6 的 `T.SimtVF` 区域里的 `T.Parallel` |
| tile 分块（`B_TOPK`、`MMA_M`、`D_QK` 切块） | `config.h:28,43,55,59` | `T.Kernel(grid)` + `T.alloc_l1/l0a/l0b/l0c` 的 shape | §2.2、§4.3 |
| `P = Q @ K^T` 与 `O += S @ V` | `kernel_body.h:1403-1491, 1493-1638` | `T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=...)` | §4.4（昇腾上 `T.gemm` 只有 L0 MAD 一条降级路径） |
| gather（按 `indices` 读 KV） | `kernel_body.h:1018-1075` | `T.Tensor` 索引 + 循环 / `T.copy` 的索引写法 | §4.3 |
| rescale 的「乘标量」 | `kernel_body.h:686-726` | `T.Parallel` 里逐元素乘 | §4.6 |
| `O += S@V` 延后一拍的**结构** | 报告 `:172-186` | 把 `T.gemm(Q,K)` 与 `T.gemm(S,V)` 写成两个 stage，靠 `T.Pipelined` 的依赖分析错开 | §4.7、第 22 章 |

A 类里最值得注意的是最后一行：**「延后一拍」这个调度意图，TileLang 的
`T.Pipelined` 表达的是「软件流水」这个抽象，不是「offset = 1」这个具体实现**。
你写的是「这两个 GEMM 都依赖 softmax，本迭代彼此独立」，调度器自己去决定错开几拍
（第 23 章）。所以 A 类里包含「结构」但不包含「具体的拍数」。

### 6.3 类别 B：能表达，但要下钻到 `T.simd.*` 与注解原语

| 本章技术点 | 代码位置 | TileLang 原语 | 核验 |
| --- | --- | --- | --- |
| softmax 的向量区域 | `kernel_body.h:466-685` | `with T.SimdVF(...):`（或 `T.SimtVF(threads=...)`） | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py:107,179` |
| rescale 的 exp / 乘加 | `kernel_body.h:565-580, 686-726` | `T.simd.vexp`/`vmul`/`vadd`（在 `T.SimdVF` 内） | 同 `frame.py`；`simd.py` 有 100 个 `def` |
| ND2NZ 的**读**（`asc_loadalign*`） | `kernel_body.h:304-330` | `T.simd.vld(addr, dist, post_inc=...)` | `tilelang/ascend/language/simd.py:324-416`（`post_inc` 即 `asc_..._postupdate`） |
| ND2NZ 的**写**（`vsstb`） | `kernel_body.h:315-330` | `T.simd.vsstb(src, base, stride, mask, update=True)` + `T.simd.make_ubuf_ptr` | `simd.py:482-552` |
| gather2 的地址计算（两两配对、`src_stride`） | `kernel_body.h:87-255` | `T.simd.vld2` / `vdintlv` / `vmin`/`vmax` / `vsub`；地址仍是 UB 里的 int64 数组 | `simd.py` 的 `vld2`/`vdintlv` 等 |
| 多缓冲的档数（L1 KV 4 档、UB KV 3 档、index 4 档） | `config.h:67-69` | `T.Pipelined(iters, num_stages=k)` + `T.annotate_buffer_versions({buf: k})` | 第 25 章 §4.7；`annotations.py:19` |
| 显式向量宽度与尾掩码 | `kernel_body.h:466-470`（`PAT_ALL`/`PAT_VL8`） | `T.simd.pset`/`pge`/`update_mask` | `simd.py:111-128` |
| pattern（`MODE_MERGING`、`E2B_B32`） | `kernel_body.h:517,495-497` | `T.simd.vld(..., dist=...)` 的 `dist` 取值表 | `simd.py:303-321` |
| 跨核同步（C↔V flag） | `kernel_body.h:69-81, 1186-1207` | `T.ascend_cross_core_set_flag(mode_id, pipe, flag_id)` / `_wait_flag` | `sync.py:124-175`；`ascend_set_flag`/`ascend_wait_flag` 在 `sync.py:28,51` |
| L1 两 bank 拆分 | `kernel_body.h:907-912` | 分配两块 `T.alloc_l1`，自己在地址上加偏移 | 第 25 章 §2.2 |

注意 B 类里的 **`T.simd.vld(..., post_inc=...)` 与 `T.simd.vsstb(..., update=True)`**：
它们和报告里那两条「自增指针」技巧是**同一件事**，只是名字不同。
这是本章一个很实用的结论——**报告里的高速 ND2NZ，在 M 树上是可以用
`T.SimdVF` + `T.simd.*` 复刻的**，不必退回 Ascend C。

### 6.4 类别 C：必须用 Ascend C / PTO / SIMD VF 内建

| 本章技术点 | 代码位置 | 为什么 TileLang 表达不了 |
| --- | --- | --- |
| 分布式的加载/转换：`asc_loadalign_brc_elem2datablock_postupdate`、`asc_loadalign_unpack4_postupdate`、`asc_e4m32float`、`asc_e2m1x22bfloat16` | `kernel_body.h:380, 383, 386, 428-454` | 这些是**带数据分布语义**的格式转换（`BRC` = 广播、`UNPK4` = 解包、`E2M1→BF16` 反量化）。`T.simd.*` 覆盖了 `vld`/`vsstb` 这些通用指令，但**没有**覆盖这些格式专用的转换名 |
| MTE2 的**参数级**控制：`burst_count`、`src_stride`、`n_burst=0` 跳过 | `kernel_body.h:1043-1067` | gather2 的收益完全来自 `burst_count=2` 这一位参数；TileLang 的 `T.copy` 抽象层没有暴露它 |
| L2 cache hint（`NORMAL_LAST_VICTIM` / `NOTALLOC_CLEAN`）与运行时启发式 | `kernel_body.h:1046, 1675-1685` | decode 时按「KV 能否装进 8 MB、复用次数 ≥ 4」选 hint（`kernel_body.h:1675-1685`）。这是**运行期决策 + 指令级参数**，不是 tile 级结构 |
| FixPipe 的位级参数：`unit_flag_ctl`、`NZ2ND_en`、`dual_dst_ctl` | `kernel_body.h:1475-1540` | `T.dual_copy` 暴露了 `unit_flag_ctrl`（第 25 章 §4.5），但 `dual_dst_ctl`/`NZ2ND_en` 这类位域没有对应的 `T.*` |
| SS buffer（`__ssbuf__`）+ 用 8×8 槽位避免 data race | `config.h:110-112`、`kernel_body.h:1189, 1205-1207` | 这是一块 scalar core 才能读写的特殊存储，属于「同步机制本身」而不是「计算」 |
| cross core flag 的 `mode_id=4`（= 同时发给两个 AIV）+ `flag_id+16` 约定 | `kernel_body.h:69-81` | `T.ascend_cross_core_set_flag` 的第一个参数就是 `mode_id`，**可以表达**；但 `flag_id+16` 这个「一份给 AIV0、一份给 AIV1」的约定是本 kernel 自己定的协议，需要你显式管理 |
| icache 控制：`#pragma unroll 1` / `VLOOP` / 手动 fuse main+drain loop | `kernel_body.h:294-330, 595, 1331` | 「不要展开」「代码体积」是编译期指令级决策，TileLang 的循环注解里目前没有等价开关（第 23 章的调度器也不建模 icache） |
| `asc_set_gm2ub_loop_size` / `asc_set_ctrl` / `asc_set_mmad_direction` 这类**核级配置** | `kernel_body.h:921-923, 1405, 1535` | 这些是「给这个核设定全局模式」的指令；`T.set_mmad_direction` 有（第 25 章 §4.6），其余没有 |

把 C 类里最常被问到的一条单独说清楚：**`vsstb` 与 `asc_loadalign` 本身
并不在 C 类**——它们在 M 树上有对应的 `T.simd.vsstb` / `T.simd.vld`
（见 §6.3）。真正落在 C 类的，是**围绕它们的那层「数据分布 + 参数 + 协议」**：
`BRC`/`UNPK4` 的转换、`n_burst` 的取值、SS buffer 的槽位分配、`flag_id+16`。
换句话说：**指令可以复用，协议不可以**。

### 6.5 这张表怎么用

三条可操作的结论：

1. **先按 A 类写，写到写不动为止。** 一个稀疏 attention 的骨架
   （分块、两个 GEMM、online softmax、rescale）在 TileLang 上是**完全可写的**，
   而且能吃到第 23 章自动调度的全部好处。先让它正确、再让它快。
2. **性能瓶颈出现之后，再决定下钻到哪一层。** 如果 profiler 显示是「Cube 等数据」，
   那多半是 B 类问题（多缓冲档数、跨核 flag、向量宽度）；
   `T.SimdVF` + 注解原语基本能覆盖。§8 的常见坑清单就是按这个思路排的。
3. **只有当结论本身绑在某条指令的参数上时，才去写 Ascend C。**
   gather2（`burst_count`）、高速 ND2NZ 的**分布选择**、L2 hint、SS buffer
   ——这些是 C 类。判据是：「**如果我把这条指令换成另一条语义相近的，
   结论还成立吗？**」不成立的就是 C 类。

第 23 章的调度器、第 25 章的 codegen 决定了「A 与 B 的边界在哪里会随版本移动」：
今天要手写的 B 类，明天可能进了自动调度。C 类则相对稳定，因为它绑的是
硬件指令集本身。

---

## 7. 进阶与陷阱

### 7.1 十二个常见坑

这些都是从报告的推导与 FlashMLA 源码里「读」出来的坑，**不是本机踩出来的**
（本机没有 NPU）。每一条都给出处。

1. **把 `RESCALE_THRES = 6` 当成作用在原始 `P` 上。** 它作用在 `P * sm_scale` 上；
   代码预先把阈值除以 `sm_scale`（`config.h:43`、`kernel_body.h:1660-1662`）。
   `sm_scale ≈ 1/√512 ≈ 0.044`，两者差 22 倍。见 §2.3。
2. **以为 online softmax 只需要一个 max。** 这个 kernel 有 `running_max` 与
   `running_max_for_softmax` 两个，后者**故意滞后**（报告 `:45,57`）。少了它，
   skip-scale 就无从谈起。
3. **把 `B_TOPK > 73.14` 当成对代码的约束。** 那是 **1C1V** 的 FixPipe 条件；
   代码用 1C2V + `B_TOPK = 64`（`config.h:53-55`）。见 §3.3。
4. **以为 accumulator 可以留在 L0C 上乘标量。** L0C 不能被 Vector 原地读写
   （报告 `:120`）；「搬回去」要靠乘单位矩阵，代价是 8192 周期。
5. **忘了 L0C 只有 128 KB。** `head = 128` 时 `O` 就是 256 KB，直接放不下
   （报告 `:116`）。这也是「64 head」这个选择的硬约束之一。
6. **以为稀疏 gather 也能随路做 ND2NZ。** 只有**源和目标都连续**才会触发 BIU
   的访存合并（报告 `:152`）；「源连续、目的 strided」不行。见 §3.4。
7. **忘了 NZ buffer 要多 pad 一行。** `(B_TOPK_PER_V+1)*D_VO`
   （`config.h:81`），否则写回时有 UB bank conflict。见 §5.3②。
8. **在 SIMD VF 里手动更新指针。** 应用 `post_inc` / `POST_UPDATE`
   （`kernel_body.h:304-330`），否则 VF 的小 scalar 会成为新瓶颈（报告 `:230`）。
9. **在 SIMD VF 里用 `#pragma unroll` 展开循环。** icache 只有 8 K，展开会
   频繁换入换出（报告 `:238`）；应该用硬件循环。见 §5.4。
10. **让所有 Tensor 都落在 L1 的同一个 bank。** 950 的 L1 是两个 bank，
    必须把每个 Tensor 切成 lo/hi 两份（`kernel_body.h:907-912`）。见 §5.5。
11. **以为「scalar core 等 VF」只是慢一点。** 它是**阻塞式**的，而各条 PIPE
    是「顺序发射、顺序执行」的，一个 `asc_sync_wait` 不满足会卡住整条管线
    （报告 `:204-206`）。见 §4.4。
12. **以为 skip-scale 只省带宽。** 它还会**省功耗 → 换频率**
    （报告 `:252`）。这条在别的平台上基本不会被提到。见 §5.6。

### 7.2 与既有章节的关系（交叉一致性）

本章有几个概念和第 15、25、35 章重叠，说法必须一致：

* **online softmax**：第 15 章 §2.3–§2.5 推的是「一个 max 的等价性证明」；
  本章是**两个 max + 阈值**的工程版本。二者不冲突——第 15 章的 `m` 就对应本章的
  `running_max`，本章多出来的 `running_max_for_softmax` 是一个**近似**，
  它的合法性由 §2.3 的 `exp(δ) ≤ exp(6)` 界定。
* **`T.gemm` 在昇腾上的降级**：第 25 章 §4.4 说昇腾的 `T.gemm`「不做选择，
  只有 L0 MAD 一条路」。本章的 `asc_mmad` 正是那条路（`kernel_body.h:1462, 1600`），
  两个说法一致。
* **`T.dual_copy` 与 FixPipe**：第 25 章 §4.5 说 `L0C → UB` 走 FIX（硬件双目的拷贝）。
  本章的 `asc_copy_l0c2ub`（`kernel_body.h:1475, 1512`）就是它，
  `dual_dst_ctl=1` 那个参数正是「双目的」的开关。
* **FixPipe bound**：第 35 章在 DeepGEMM/MQA 里已经给过一种「FixPipe 是瓶颈」
  的算法；本章 §3.3 的 bound ③ 是同一个单元、同一类等式，只是形状换成 attention。
  两章的结论可以互相印证：**FixPipe 的 128 B/cycle 在这个平台上是个反复出现的
  约束**。
* **`T.simd.*` 的覆盖面**：第 25 章 §4.6 说 `T.SimdVF` 里要写「显式 `T.simd.*`
  指令」。本章 §6.3 把这条落到了具体名字上（`vld`/`vsstb`/`vld2`/
  `make_ubuf_ptr`）。**没有冲突，是同一件事的两个粒度。**

### 7.3 资料缺口与「待核」项

按本书的引用规范，本章有几处**核不到、或报告没写清**的地方，明确登记：

1. **`RESCALE_THRES = 6` 的选值依据。** 报告只说「一般取 6」（`:57`），
   没给推导。本章给出的是「`exp(6)` 量级安全 + rescale 次数最少」的解释，
   属于**合理推断**而非原文结论。
2. **「大约 1024 周期」的来源。** 报告 `:184` 给了这个数，没有给公式。
   本章按原文引用，未复算。
3. **Ascend 950 的片上容量。** 报告说 L1 分成两个 bank（0–256K 与 256K–512K，
   `:242`），并说 `head = 128` 时 `O`（256 KB）会把 L0C「完全占满」（`:116`）；
   但报告没有给出 L0C 的容量数字，而第 19 章 §2.2 表里的容量是 **Atlas A2** 口径。
   本章引用时都标了来源；具体数值以平台文档为准。
4. **410 / 360 TFlops 的「典型工况」。** README 与报告都没列出精确的形状与批大小
   （见 §5.7），只给了硬件与 CANN 版本。
5. **`NUM_VEC_CORE_PER_AI_CORE = 2` 在 950 上是否存在例外配置。**
   代码把 2 写死为常量（`config.h:53`），没有运行时分支。本章不推断其他配置。

---

## 8. 本章小结

把这一章缩成一张卡片：

**算法**：DSA 让每个 q token 只看 `topk` 个 KV。数学上是
`gather → P = QKᵀ → softmax(P·sm_scale) → PV`；工程上用 online softmax，
维护 `running_max` / `running_max_for_softmax` / `running_sumexp` / `out_accum`
四个 float32 状态；再叠一个 skip-scale，只有当
`running_max − running_max_for_softmax > RESCALE_THRES`（≈6，作用在
`P·sm_scale` 上）时才做 rescale。

**四个设计讨论**（报告的推进顺序）：

1. **head 数**：算存比（`cube_cycles` vs `mem_cycles`）显示 `head ≥ 64` 才是
   compute bound；取 128 会撑爆 L0C（256 KB > 128 KB），而且 MMAD 的 M=64
   已足够打满吞吐。
2. **accumulator 放哪**：L0C 不能原地乘标量；搬回 L0C 要一次 `64×64×512`
   的矩阵乘，关 HF32 后是 8192 周期，不可接受 → 放 VECTOR 侧的 UB。
3. **1C1V 还是 1C2V**：三条 bound——MTE2 队列深度 16、反量化跟不上、
   FixPipe 128 B/cycle 推出 `B_TOPK > 73.14`——合起来否掉 1C1V → 用 1C2V。
4. **ND2NZ 何时做**：稀疏 gather 无法随路转换；「源连续、目的不连续」不触发
   访存合并 → 走 GM→UB→(SIMD VF)→L1。

**流水**：CUBE 侧把 `QKᵀ[k+1]` 排在 `SV[k]` 前面（延后一拍，让 softmax 有时间）；
VECTOR 侧是 7 个动作、深度 6 的流水（源码 `SYSTOLIC_ARR_DEPTH = 6`）。
skip-scale 信号用 **SS buffer + cross core flag** 传递，而
「PIPE 顺序发射、顺序执行」使得任何一次同步阻塞都会被放大——所以
**发起顺序本身就是优化对象**。

**六条优化技巧**：gather2（`burst_count=2` + `src_stride`）绕 MTE2 队列深度；
KV cache layout（data 与 scale 相邻）把每 token 的拷贝请求数砍半；
高速 ND2NZ 四条技巧（`asc_loadalign`+`vsstb`、多 pad 一行、自增指针、
一次一对 256 B）；icache（32K/16K/8K，用 `VLOOP` 不用 `#pragma unroll`）；
L1 两 bank 配平（每个 Tensor 切 lo/hi）；skip-scale 的额外收益（省功耗换频率）。

**TileLang 的落点**：分块、两个 GEMM、online softmax、rescale 属于
**A 类（直接可表达）**；`T.SimdVF` + `T.simd.vld/vsstb/vld2` +
`T.annotate_buffer_versions` + `T.ascend_cross_core_set_flag` 属于
**B 类（可表达但要下钻）**；`BRC`/`UNPK4` 的分布转换、`burst_count`/`src_stride`、
L2 hint、`unit_flag_ctl` 位域、`__ssbuf__`、`flag_id+16` 协议、icache 控制
属于 **C 类（必须手写）**。判据是那句：
**「换一条语义相近的指令，结论还成立吗？」**

一句话记住本章：**这是一个「每一步都先算代价、再决定往哪放」的算子；
它的每一条结论都能在报告里找到一行算式，也都能在源码里找到一行指令。**

## 延伸阅读

* [references.md](references.md) 的 R1–R8：本章所有「报告 `:N`」与代码行号的完整清单。
* 想动手复现报告里的两张表：`code/01-head-roofline/` 与 `code/02-fixpipe-bound/`。
* 想验证 skip-scale 的算法等价性：`code/03-online-softmax-skip-scale/`。
* 想知道某个技术点能不能用 TileLang 写：`code/04-tilelang-expressibility/`。
* 想补前置：第 15 章（online softmax）、第 19 章 §2（机器模型）、
  第 22 章（流水与同步）、第 23 章（自动调度）、第 25 章（`T.*` 原语与昇腾后端）。
* NV 侧对照：`FlashMLA/docs/20250929-hopper-fp8-sparse-deep-dive.md`
  （dequantization-bound 与 crossover、DSM）与
  `FlashMLA/docs/20250422-new-kernel-deep-dive.md`（compute-bound 判据的另一种算法）。
  这两篇讲的是**同一类问题在不同机器上的不同解**，对照读能看清「哪些结论是
  算法的、哪些是机器的」。
