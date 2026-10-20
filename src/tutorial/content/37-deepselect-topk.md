# 第 37 章 DeepSelect：算法先行的 TopK

## 学习目标

读完本章，你应当能够：

1. 说清 TopK 在 DeepSeek 推理里出现的**两个位置**（DSA 的 Lightning Indexer 与
   采样器），并**自己算出**朴素实现的两笔账：为什么“读一遍输入”和“排序”是两件
   不同的事，为什么 `torch.topk` 在这个形状下必然慢。
2. 完整复述 DeepSelect 的算法：**阈值 T + 随机块序扫描 + 筛选 + compact** 三步骤，
   并解释为什么块顺序**必须**是随机的、候选缓冲区的上界为什么是 `k + B2 + B`、
   额外空间 `O(k+B+B2)` 为什么能放进片上存储。
3. **独立走一遍期望复杂度证明**：定义 `B_i` / `C_i` / `A_i` / `W` / `H_m`，
   说明每一步不等式背后的直觉，最后得到
   `E[W] ≤ (1+k/B2)·L·H_m` 与 `O((1+k/B2)(k+B+B2)·log(N/B))`，并说清当
   `B,B2 = Θ(k)` 时为什么是 `O(k log(N/k))`。
4. 读懂 CUDA 侧的五个关键技巧（mask 一遍扫描、`set/prmt/dp4a`、浮点加法替整数加法、
   warp 前缀和/字节提取/直方图、Threadblock Cluster），并回答**每个技巧在压哪个瓶颈**。
5. 说清昇腾侧的差异：哪些技巧是 CUDA 专有的、昇腾靠什么达到同样的效果、
   两边共同的硬约束是什么。
6. 用判据回答一个更一般的问题：**什么类型的算子适合用 TileLang 写，什么适合手写**；
   并解释本章为什么不是“TileLang 章节”。

## 前置知识

* 第 13 章：reduction 类算子的写法、`T.reduce_*` 的语义、逐元素算子的访存记账法。
  本章 §1 的“账单”用的就是那一套方法。
* 第 28 章：算子设计方法论与性能模型，“先把算子变成 `(flops, bytes)`”、roofline、
  memory-bound 与 compute-bound 的判据。本章是它在“几乎没有浮点计算”的算子上的案例。
* 第 16 章的 §4.3：DSA（indexer + top-k + sparse MLA）在模型里的位置；本章只补
  “top-k 这一步为什么难、怎么把它做成 memory-bound”。
* 第 6 章（可选）：NV GPU 与昇腾 NPU 的抽象对照。本章 §5 会用到 shared memory 与
  UB 的对照，但不需要你先读第 6 章也能看懂。

不需要你会写 CUDA，也不需要你有 GPU 或 NPU：本章的所有内核代码都只做**静态阅读**，
所有可运行的示例都在 CPU 上（见下面的“验证边界”）。

---

## 本章的验证边界（务必先读）

写作环境是 macOS (Apple Silicon)、Python 3.11，**没有 NVIDIA GPU，也没有昇腾 NPU**
（见 [`../../VERSIONS.md`](../../VERSIONS.md)）。因此本章遵守四条硬规矩：

1. **本章的任何 kernel 代码都没有被运行过。** CUDA 与 Ascend C 源码只做了静态阅读：
   在 corpus 快照里逐行核对函数、指令与常量，**没有编译，也没有执行**。
2. **性能数字只有一个来源**：DeepSelect 仓库自述与它自带的性能图
   （`.../DeepSelect/README.md:39-62`、`assets/perf_*.png`）。本章引用时会写清“来源”与
   “测量条件”，并明确标注**不是我们测的**。我们不从图上读出的数字当作精确值。
3. **复杂度结论来自报告里的证明**（`.../DeepSelect/docs/DeepSelect-deep-dive.zh.md:51-102`），
   本章 §3 把它讲成人话，并用 `code/03` 的纯 Python 表复算；这些数字是**公式算出来的**，
   不是测量值。
4. **本机可运行的东西只有算法模型**：`code/01`、`code/02`、`code/03` 三个纯 Python 脚本，
   它们在写作机上实际跑过，验证级别是“CPU 逻辑验证”。它们**不测性能**，
   只验证算法正确性与代价量级。

一句话：**本章的“快”是引用来的，“对”是自己算的。**

---

## 1. 问题的由来：TopK 为什么在 DeepSeek 里是个瓶颈

### 1.1 TopK 出现的两个位置

先看这张图（图 1，源文件 `figures/01-topk-in-model.md`）。DeepSeek 的推理里有
**两处**要用 TopK，它们看起来毫不相干，但形状惊人地相似。

**位置一：DSA 的 Lightning Indexer。** DeepSeek Sparse Attention（DSA）不再让每个
query 看全部 context token，而是先用一个**轻量打分**（indexer）给每个 context token
打个分，再从中选出最相关的 `topk` 个位置，只让后续的 MLA 读这些位置的 KV。仓库自述
把这件事写成：“在 DeepSeek Sparse Attention（DSA）中，需要使用 TopK 算子从大量
context token 中选出最相关的位置”
（`.../DeepSelect/docs/DeepSelect-deep-dive.zh.md:3`）。

**位置二：采样器。** 生成下一个 token 时，模型输出的是一整张词表的 logits，采样之前
要从中挑出候选 token。报告同一行接着写：“在采样阶段，也需要使用 TopK 从较大的词表中
选出候选 token”（同上，`:3`）。

两处的共同点，是本章全部讨论的起点：

> **`k` 很小（≤ 4096，主要是 512），而 `N` 很大（context 可以到百万，词表约 128K）。**

`k=512` 这个数字不是随便挑的：仓库自述说它“mainly optimized for `topk=512`
（即 DeepSeek V4 的 `topk`）”（`.../DeepSelect/README.md:23`）。

### 1.2 两个场景的约束矩阵

两个位置虽然形状相似，工程约束却差得很远。图 6（`figures/06-constraint-matrix.md`）
把它们和两个后端的支持情况摆在一起。

从这张表里要记住三件事：

1. **唯一的硬约束是 `topk <= 4096`。** 代码里是一条显式的检查
   （`.../DeepSelect/csrc/api.cpp:57`），仓库也直说“larger values are not
   supported”。
2. **两个场景的 dtype 不同，后端能力也不同。** Lightning Indexer 用 `bfloat16`，
   Sampler 用 `float32`；而“目前只有 CUDA 实现支持 `torch.float32` 作为输入，
   昇腾实现只支持 `torch.bfloat16`”（`.../DeepSelect/README.md:37`）。
3. **输入张量的行 stride 有对齐要求**：CUDA 侧是 1024 字节，昇腾侧是 32 字节
   （`.../DeepSelect/csrc/structs.h:17-18,39-40`）。不满足就得自己 padding，
   这类“库对外暴露的对齐契约”是工程实现的一部分，别忽略。

### 1.3 朴素实现的账单：读一遍 + 排序

现在算账。用第 28 章的记账法，把“选出一行里最大的 `k` 个”拆成两件事：

**第一笔：你必须把这一行读完。** 这是物理下限。没有一条信息能让你不读某个元素就
断定它进不了 top-k，任何实现至少要读 `N` 个元素一次。这就是为什么 DeepSelect 的
指标是**带宽**而不是算力（详见 §6.1）。

**第二笔：排序。** `torch.topk` 的通用做法是做一个大小为 `k` 的选择/排序。无论具体
算法是什么，它都**额外引入一趟与 `N` 同阶的工作**：要么沿 `N` 做一次近似线性但常数
很大的选择，要么把数据推进堆里做 `N·log k` 量级的比较。也就是说，朴素实现的代价结构是

```text
朴素 TopK  ≈  读一遍 N          （不可避免）
           +  与 N 同阶的选择/排序  （可避免）
```

而 DeepSelect 的整个设计目标，就是**把第二笔砍掉**：让“排序”只发生在一个大小为
`O(k+B+B2)` 的小集合上，而不是在 `N` 上。`code/03-complexity-tables` 里把这两笔账
画成了表（Table 4）：在 `N=1048576, k=512` 下，读一遍是 1,048,576 个元素，
而一个 `N·log2(k)` 的排序模型是 9,437,184 个元素，**贵了近 10 倍**。

把这笔账换算成带宽：如果设备能跑到 5 TB/s，读 1M 个 bf16（2 MB）只要 0.4 µs，
而多做一趟等量的工作就多花一倍时间。这解释了为什么 README 把指标写成
“effective memory bandwidth”。它想说的是“我们离『只读一遍』这条物理下限有多近”。

### 1.4 本章要回答的问题

把上面几段收成一句话：

> **怎么把“选 top-k”做成一个只读一遍输入、把选择限制在一个与 `N` 无关的小集合上的
> 算法？以及，为了实现它，需要在硬件上多花多少功夫？**

第一半是**算法**问题（§2、§3），第二半是**常数**问题（§4、§5）。本章的标题
“算法先行”说的就是：先解决第一半，第二半才有意义。

---

## 2. 概念与模型：把 TopK 变成“读一遍 + 有界选择”

### 2.1 记号

| 符号 | 含义 | 报告里的位置 |
| --- | --- | --- |
| `N` | 一行的长度（DSA 里是 compressed seqlen，采样里是 vocab_size） | `...zh.md:18` |
| `k` | 要选出的元素个数（topk） | `...zh.md:18` |
| `B` | 扫描块大小，每次读入的连续元素个数 | `...zh.md:15` |
| `B2` | 候选缓冲区的“松弛量”，决定何时触发 compact | `...zh.md:15` |
| `T` | 当前的 top-k 阈值，初值 `-inf` | `...zh.md:9` |
| `topk_candidate` | 候选缓冲区，存的是一批 `(value, index)` | `...zh.md:21` |

报告给的推荐值：“对于 `k=512`，一组较为合理的参数配置是 `B=B2=1024`”
（`...zh.md:15`）。这句话很重要。它意味着 `B`、`B2` 都是 `Θ(k)` 量级，
这是后面渐近复杂度能化简成 `O(k log(N/k))` 的前提。

> 注意：一个要提前指出的**资料不一致**：报告给的推荐是 `B=B2=1024`（`k=512` 时），
> 但仓库里真正发布的 CUDA 实例用的是 `B`（`elements_per_round`）`=4096` 或 `8192`、
> `B2`（`reconstruct_threshold`）`=4096`
> （`.../DeepSelect/csrc/api.cpp:173-186`）。两者不是矛盾，而是**算法建议值与
> 硬件调优值**的区别：更大的 `B` 会抬高复杂度界，但换来更大的 TMA 传输与更好的
> 带宽。这正是本章主题的一个实例，**算法决定上限，硬件决定常数**。详见 §6.3。

### 2.2 算法三步骤

图 2（`figures/02-deepselect-steps.md`）是算法的全貌：扫描 → 筛选 → 压缩，
输入只被读一遍。

报告的伪代码只有 20 行，我们逐行读一遍
（`.../DeepSelect/docs/DeepSelect-deep-dive.zh.md:17-39`）：

```text
DeepSelectTopk(x[0:N), k, B, B2):
    require 1 <= k <= N and B >= 1 and B2 >= 1

    topk_candidate = []                     # shared memory
    topk_threshold = -inf                   # current top-k threshold
    p = random_permutation(ceil_div(N, B))  # independent of x

    for block in p:
        lo, hi = block * B, min((block + 1) * B, N)

        for j in range(lo, hi):
            if x[j] > topk_threshold:
                topk_candidate.append((x[j], j))

        if len(topk_candidate) >= k + B2:
            topk_candidate = RadixSelectTopK(topk_candidate, k)
            topk_threshold = min(v for v, _ in topk_candidate)

    return RadixSelectTopK(topk_candidate, k)
```

逐行解释：

* **`topk_candidate = []`，`topk_threshold = -inf`**：候选缓冲区一开始是空的，
  阈值是 `-inf`。阈值取 `-inf` 有一个直接后果，**第一次 compaction 之前，所有元素
  都会被接受**（因为任何数都 `> -inf`）。这段“预热期”是算法必需的，也是复杂度
  证明里 `A_i` 在开头较大的原因。
* **`p = random_permutation(ceil_div(N, B))`**：块的**处理顺序**是一个随机排列，
  而且注释特意写了 `independent of x`。**这个随机性与输入数据无关**。这一点在
  §2.3 和 §3 里是承重的。
* **外层 `for block in p`**：按随机顺序取块。注意每次访问的是 `[lo, hi)` 这一段
  **连续**的内存。这保证了 coalesced 的访存，也是 TMA / DMA 能高效工作的前提
  （报告 Notes 里写：“`x` 中的每个元素都恰好读取一次，并且以大小为 `B` 的连续块
  进行访问”，`...zh.md:48`）。
* **`for j in range(lo, hi)` 里的 `if x[j] > topk_threshold`**：这就是“筛选”。
  一旦阈值涨上去，绝大多数元素连一次 `append` 都换不到。
* **`if len(topk_candidate) >= k + B2`**：这就是“compact 的触发条件”。这里有个
  细节值得强调：触发线是 `k + B2` 而不是 `k`。为什么不一到 `k` 就压缩？
  因为压缩本身有成本，而且压缩一次只能“留下 `k` 个”，如果每次都是刚超一点就压，
  压缩次数会爆炸。`B2` 就是给缓冲区留的松弛量，**让它多攒一点再一次性处理**。
* **`topk_candidate = RadixSelectTopK(topk_candidate, k)`**：用 radix select
  （而不是全排序）把候选精简回 `k` 个。选 radix select 而不是 sort 的理由很简单：
  我们只要“第 `k` 大是谁”，不需要“前 `k` 个排好序”，而 radix select 是线性的。
* **`topk_threshold = min(v for v, _ in topk_candidate)`**：压缩之后，缓冲区里正好是
  当前见过的最大 `k` 个元素，所以**它们的最小值就是当前的 top-k 阈值**。这是整个算法
  最重要的不变量。
* **`return RadixSelectTopK(topk_candidate, k)`**：扫描完之后，缓冲区里可能还留着
  不到 `k+B2` 个元素，最后再做一次 compact 就是答案。

### 2.3 为什么块顺序必须随机

报告把这句话写成一条硬性要求：“为了避免极端输入下的性能退化，块处理顺序必须是随机的”
（`...zh.md:43`）。为什么？用一个反例就清楚了。

假设输入是**升序**的（大值都在尾部），而扫描是**从左到右**的：

1. 头两个块进来，阈值是 `-inf`，所有元素被接受，缓冲区到 `2B`；
2. 触发 compact，保留前 `k` 个，阈值变成“最小块里的最大值”，一个很小的数；
3. 下一个块的值**全都比这个阈值大**（因为输入升序），于是整块被接受；
4. 于是每读一个块就触发一次 compact，`A` 一路涨到 `N`。

结果是：候选量从“`k` 量级”退化成了“`N` 量级”，算法彻底失去意义。而**同样的
输入**只要换成随机块序，就完全正常。因为随机顺序下，早期块里就可能出现大值，
阈值很快涨到位，后续块被大量过滤掉。

这三个脚本里的 `code/02-block-order-experiment` 就是干这件事的。它的实际输出：

```text
input      order               A     R           W      W/N  correct
--------------------------------------------------------------------
random     sequential      6,599     5       9,159    0.035  True
random     random          6,597     5       9,157    0.035  True

ascending  sequential    262,144   255     392,704    1.498  True
ascending  random          6,144     5       8,704    0.033  True

descending sequential      2,048     1       2,560    0.010  True
descending random          8,192     7      11,776    0.045  True

staircase  sequential    262,144   255     392,704    1.498  True
staircase  random          6,144     5       8,704    0.033  True
```

（`A` = 进入候选缓冲区的元素数，`R` = 循环内 compact 次数，`W = A + k·R`；
`N=262144, k=512, B=B2=1024`。）

读这张表要读出三点：

* **随机块序是稳的**：四种输入下 `W/N` 都在 0.03–0.05，和输入分布无关。
* **顺序扫描好不好，取决于大值在哪里**：`descending`（大值在前）时顺序扫描最省；
  `ascending`/`staircase`（大值在后）时 `A` 直接等于 `N`，`W ≈ 1.5N`。
* **最后一列全是 `True`**：随机化影响的是**代价**，不是**结果**。这一点很重要，
  后面 §3 的证明只讨论 `W`，默认正确性已经由算法结构保证。

> **一句话记住**：随机块序不是“让结果随机”，而是“让**代价**不依赖输入顺序”。

### 2.4 阈值只升不降，候选缓冲区有界

报告里有两句关于不变量的陈述，必须背下来：

1. **阈值单调**：“随着不断扫描，`topk_threshold` 只会单调增大”（`...zh.md:41`）。
   原因很直接：阈值是“当前缓冲区最小值”，而缓冲区每次 compact 后都是“当前见过的
   最大 `k` 个”；见过的元素越多，第 `k` 大只会更大，不会更小。
2. **候选有界**：“任意时候，我们都有 `len(topk_candidate) <= k + B2 + B`”
   （`...zh.md:47`）。原因同样是结构性的：触发线是 `k+B2`（达到即压缩），而两次
   压缩之间最多再扫进一个 `B` 块。

图 3（`figures/03-threshold-and-buffer.md`）把这两件事画在了一起。

这张图的柱长是**候选缓冲区的元素数**，标注是**当前阈值**。读它的正确方式是：
柱长在 `k`（512）与 `k+B2`（1536）附近来回振荡，而标注里的阈值（`-inf → 0.975 →
0.992 → 0.997`）**只升不降**。

而“阈值只升不降”的直接后果，是报告那句话的下半句：“因此大体上来说，后续元素通过
阈值筛选的概率会逐渐降低，候选缓冲区的增长速度也会越来越慢”（`...zh.md:41`）。
这一句就是 §3 里 `E[A_i] ≤ L/i` 的直观来源，**第 `i` 个块接受元素的概率大致是 `1/i`
量级**。

### 2.5 额外空间 `O(k+B+B2)` 为什么放得进片上

报告 Notes 的最后一条是：“除去随机排列，算法所需要的额外空间为 `O(k+B+B2)`，因此可以
存放在容量更小但速度更快的存储空间（例如 shared memory）中”（`...zh.md:49`）。

这句话是怎么成立的？把它算成字节数就清楚了。在 CUDA 侧的配置里，候选是按
`(value, index)` 的二元组存的；在昇腾侧，`config.h` 把这件事写成了编译期常量：

$$
\begin{aligned}
C_{\text{candidate}}
  &=C_{\text{topk}}+C_{\text{block}}+C_{\text{filtered}}\\
  &=k+B+B_2.
\end{aligned}
$$

对应 `csrc/ascend_kernels/config.h:195`。

注意右边**正是 `k+B+B2`**。这个缓冲区被存成三份等长的 part（value / index_low /
index_high）（`config.h:196-203`）。注意 `k` 与 `B2` 都不是默认值。它们由
`Topk512/1024/4096Config` 显式给出（`config.h:130-135`），而 `B` 取默认的
`kBlockSize = 16 × 1024`：

| 配置 | `k` | `B` | `B2` | `k+B+B2` | 每份 part | 三份合计 |
| --- | --- | --- | --- | --- | --- | --- |
| `Topk512Config` | 512 | 16384 | 4096 | 20,992 | ≈ 41.5 KB | ≈ 124.5 KB |
| `Topk4096Config`（Lower） | 4096 | 16384 | 2560 | 23,040 | ≈ 45.5 KB | ≈ 136.5 KB |
| `Topk4096Config`（Upper） | 4096 | 16384 | 3072 | 23,552 | ≈ 46.5 KB | ≈ 139.5 KB |

（每份 part 的字节数按 `aligned((k+B+B2+TOPK_COMPACT_PADDING) × 2 B, 32)` 算，
`config.h:17,196-199`。）三份合计都在 **140 KB 以内**，而整张 UB 布局由
`static_assert` 卡在代码声明的预算 `248 KB` 以内（`config.h:19,238`）。
**这就是“`O(k+B+B2)` 能放进片上”的具体含义**：它不是一句渐近说法，
而是一张能通过 `static_assert` 的内存预算表。

> `Topk4096Config` 有两组参数（Lower / Upper），由**运行期行长**在两者之间选
> （`config.h:120-135` 的 `TopkSelectPolicyConfig` 注释）。这也是一种“常数调优”：
> 短行长与长行长用不同的参数，代价是要把两组都编进 kernel。

---

## 3. 期望复杂度证明（讲成人话）

这一节是本章的教学价值所在。报告的证明只有一页半
（`.../DeepSelect/docs/DeepSelect-deep-dive.zh.md:51-102`），但它把“随机化到底买到了
什么”这件事说清楚了。我们把它拆成八步，**每一步都先讲直觉，再写式子**。

### 3.0 先明确“要证什么”

算法的代价由两部分组成：

* 每个元素被**读一次**（不可省，`N` 次）；
* 每个进入候选缓冲区的元素，会被 compact 步骤**处理**一次。

所以真正要控制的是第二个量。报告定义了一个单一的标量：

> **`W` = 所有 `RadixSelectTopK` 调用的输入长度之和**（包括最后一次调用）
> （`...zh.md:57`）。

`W` 就是“额外工作量”的总和。**我们要证的是 `E[W]` 的上界**，期望是对
“随机块顺序”取的，注意，**随机性只来自块顺序，不来自输入**：

> “固定输入，下方分析中的随机性仅来自一个与输入独立的均匀随机块排列”
> （`...zh.md:53`）。

还有一条技术假设：**假设所有元素互不相同**（`...zh.md:55`）。报告说“存在相同元素时，
算法不会产生性能下降，这里不仔细讨论”，本章沿用这条假设，§3.8 会说明重复元素为什么
不会让情况变坏。

### 3.1 符号表

先把所有符号摊开。这张表最好抄下来，因为它把“证明的语言”和“代码的变量”一一对上了：

| 符号 | 定义 | 报告位置 | 代码里的对应 |
| --- | --- | --- | --- |
| `B_i`（花体） | 随机处理顺序中的**第 i 个块** | `...zh.md:57` | `for block in p` 的循环变量 |
| `C_i` | 开始处理 `B_i` 时的 `topk_candidate` | `...zh.md:57` | `topk_candidate` |
| `A_i` | 处理 `B_i` 时**加入**候选缓冲区的元素个数 | `...zh.md:57` | 每次 `append` 的计数 |
| `A` | `Σ_i A_i`，即总共被加入的元素个数 | `...zh.md:78` | `code/01` 的 `stats.accepted` |
| `R` | **循环内部**调用 `TopK` 的次数（不含最后一次） | `...zh.md:80` | `stats.compactions` |
| `W` | 所有 `TopK` 调用的输入长度之和 | `...zh.md:57` | `stats.compact_work` |
| `m` | `ceil(N/B)`，块的总数 | `...zh.md:60` | `code/01` 的 `n_blocks` |
| `L` | `k + B + B2`，一个常量 | `...zh.md:61` | `code/03` 的 `L` |
| `H_m` | 第 `m` 个**调和数** `Σ_{i=1}^{m} 1/i` | `...zh.md:62` | `code/03` 的 `harmonic(m)` |
| `U_i` | 前 `i` 个已处理块构成的集合 `{B_1, …, B_i}` | `...zh.md:67` | 没有单独的集合对象；由 `for block in p` 已经走过的前缀表示 |
| `T_i` | `U_i` 里所有元素中**前 `L` 大**的那些元素（不足 `L` 个则取全部） | `...zh.md:69` | 由阈值 `topk_threshold` 和候选缓冲区共同刻画 |

两个立刻要注意的点：

* **`L = k+B+B2` 是“候选缓冲区的上界 + 一个块”**。它不是巧合。它在 §3.2 里会变成
  “任何被接受的元素，在前 `i` 块里的排名都不会超过 `L`”。
* **`H_m` 就是 `ln m + O(1)`**。调和数出现，是因为每一块的接受概率大致是 `1/i`；
  把 `i` 从 1 加到 `m`，得到的就是 `H_m`。这是整个证明里最“漂亮”的一步。

### 3.2 第一块砖：被加入的元素一定是“前 L 大”

**直觉**：一个元素之所以被加入候选缓冲区，是因为它**大于当前阈值**。而阈值是
“缓冲区里最小元素”，所有落选的历史元素都不大于它。所以“比它大的历史元素”
只可能藏在缓冲区里。而缓冲区长度有上界。于是这个元素在“已见过的东西”中的排名
必然靠前。

**式子**：报告的原话是（`...zh.md:65`）：

```text
由于 |C_i| < k + B2，并且此前已经处理但不在 C_i 中的所有元素都不大于当前阈值，
因此，从 B_i 中加入 topk_candidate 的任意元素，
在前 i 个已处理块中的排名至多为 |C_i| + |B_i| ≤ |C_i| + B < L。
```

拆开看，唯一需要“想一下”的是 `|C_i| < k + B2` 为什么成立：它是**循环不变量**，
每次处理完一个块，只要 `|C| ≥ k+B2` 就立刻 compact 回 `k`，所以进入下一个块时
必然有 `|C| < k+B2`。而 `|B_i| ≤ B` 是因为块大小是 `B`（最后一块可能更小）。

于是得到本节要用的结论：

> **（★）处理 `B_i` 时被加入的每一个元素，都属于 `U_i` 中前 `L` 大的元素。
> 换句话说，它属于 `T_i`。**

### 3.3 第二块砖：给定 `U_i`，`B_i` 在块集合里是均匀的

这是**唯一**用到随机性的地方。报告的原话是：“在给定 `U_i` 的条件下，`B_i` 在 `U_i`
中的 `i` 个块之间是均匀分布”（`...zh.md:69`）。

**直觉**：`U_i` 是“前 `i` 个被处理的块”这个**集合**。我们已经知道这个集合是哪些块了
（这是条件），但不知道其中哪一个是**最后处理的**（第 `i` 个）。因为顺序是完全随机排列的，
在这个集合里，“谁是最后一个”是均匀随机的，每个块被选中的概率都是 `1/i`。

这一步是**随机的全部来源**，也是为什么 §2.3 里那个反例会退化：顺序扫描时
“谁是最后一个”是确定的（地址最大的那个），没有这个均匀性，后面的期望就不成立。

### 3.4 合起来：`E[A_i | U_i] ≤ L/i`

把 ★（被加入的元素属于 `T_i`）和 §3.3（`B_i` 均匀）合起来，就得到报告的核心不等式
（`...zh.md:71-76`）：

```text
E[A_i | U_i]
  ≤ (1/i) · Σ_{b ∈ U_i} |b ∩ T_i|     ← B_i 均匀，且加入的元素都在 T_i 里
  = |T_i| / i                           ← 求和就是把 T_i 的元素数按块拆开再加回去
  ≤ L / i                               ← 因为 |T_i| ≤ L
```

逐个符号解释：

* `|b ∩ T_i|` 是“块 `b` 里有多少个元素属于 `T_i`”。把 `b` 遍历 `U_i` 求和，
  得到的就是 `|T_i|`（每个 `T_i` 的元素恰好属于某一个块）。
* 除以 `i` 是因为 `B_i` 均匀，**“第 i 个块是哪个”的期望贡献，就是把 `U_i` 里所有块
  的贡献取平均**。
* 最后一步：`T_i` 被定义为“`U_i` 里前 `L` 大”，所以 `|T_i| ≤ L`。当 `U_i` 的
  元素总数不足 `L` 时取全部，此时 `|T_i| < L`，不等式更松，仍然成立。

**这一步的物理含义**：第 `i` 个块期望只“贡献”`L/i` 个新候选。`i` 越大，贡献越小，
这就是“阈值涨上去以后接受率下降”的定量版本。

### 3.5 求和：`E[A] ≤ L·H_m`

现在对 `i` 求和。因为期望是线性的（`A = Σ_i A_i`），

$$
\begin{aligned}
\mathbb{E}[A]
&=\sum_{i=1}^{m}\mathbb{E}[A_i]\\
&\le\sum_{i=1}^{m}\frac{L}{i}
=LH_m.
\end{aligned}
$$

这一步见 `...zh.md:78`。

`H_m` 就是调和数。这一步的直觉是：**如果第 `i` 块的接受概率是 `1/i` 量级，
那么总接受量就是 `log` 量级，而不是 `N` 量级。** 这正是随机化的全部价值。

用 `code/03` 的 Table 1 感受一下数量级（`k=512, B=B2=1024`）：

| `N` | `m = N/B` | `H_m` | `E[W]` 的上界 |
| --- | --- | --- | --- |
| 16,384 | 16 | 3.38 | 12,982 |
| 262,144 | 256 | 6.12 | 23,517 |
| 1,048,576 | 1,024 | 7.51 | 28,835 |
| 16,777,216 | 16,384 | 10.28 | 39,480 |

**`N` 涨了 1000 倍，界只涨了 3 倍**。因为 `H_m` 是 `log`。

### 3.6 从 `A` 到 `W`：`W = A + kR` 与最终结论

还差最后两步，报告在 `...zh.md:80-82`：

**第一步：`R·B2 ≤ A`。** 每一次循环内的 compact 都是因为缓冲区达到了 `k+B2`，
压缩后只剩 `k` 个，所以**这次调用至少“消耗掉”了 `B2` 个元素**。把这些消耗加起来
不能超过总共接受过的 `A`，所以 `R·B2 ≤ A`，即 `R ≤ A/B2`。

**第二步：`W = A + kR`。** 为什么？

* 每个被加入的元素，都会**被某一次 compact 读到**（作为那次调用的输入）。这是 `A`；
* 每一次循环内的 compact 会留下 `k` 个幸存者，这 `k` 个元素在**下一次** compact 时
  又会被读一遍，一共有 `R` 次循环内 compact，所以多了 `k·R`。

（最后一次 compact 的输入已经算在 `A` 里，且它之后不再有调用，所以不额外贡献。）

把两步代进去：

$$
\begin{aligned}
\mathbb{E}[W]
&=\mathbb{E}[A]+k\mathbb{E}[R]\\
&\le\mathbb{E}[A]+\frac{k}{B_2}\mathbb{E}[A]
&& (R\le A/B_2)\\
&=\left(1+\frac{k}{B_2}\right)\mathbb{E}[A]\\
&\le\left(1+\frac{k}{B_2}\right)LH_m
&& (\mathbb{E}[A]\le LH_m).
\end{aligned}
$$

报告把它框成最终结论（`...zh.md:86-92`）：

$$
\boxed{
\mathbb{E}[W]
\le\left(1+\frac{k}{B_2}\right)LH_m
=\left(1+\frac{k}{B_2}\right)L(\ln m+O(1))
}
$$

### 3.7 渐近形式与 `Θ(k)` 的取法

把 `m = ceil(N/B)` 代进 `ln m`，并用 `L = k+B+B2`：

$$
\mathbb{E}[W]
=O\left(\left(1+\frac{k}{B_2}\right)(k+B+B_2)\log\frac{N}{B}\right)
$$

这一步见 `...zh.md:96-100`。

再取 `B, B2 = Θ(k)`（也就是报告推荐的 `B=B2=1024` 配 `k=512` 这种情况）：

* `(1 + k/B2) = Θ(1)`，
* `(k + B + B2) = Θ(k)`，
* `log(N/B) = Θ(log(N/k))`，

于是

$$
\mathbb{E}[W]=O\left(k\log\frac{N}{k}\right)
$$

这一步见 `...zh.md:102`。

**这就是全章最重要的一个不等式**：TopK 的额外计算量是 `O(k log(N/k))`，
而输入规模是 `N`。当 `k = 512`、`N = 10⁶` 时，`k log(N/k) ≈ 512 × 7.6 ≈ 3900`，
相对 `N` 是千分之几。报告的措辞是“远小于输入规模 `N`”（`...zh.md:102`）。
图 5（`figures/05-complexity-scale.md`）把这几个量级摆在一起，**注意它是渐进的
量级对比，不是实测**。

`code/03` 的 Table 2 把三种形式并排放在一起，可以验证
`(1+k/B2)·L·H_m` 与 `(1+k/B2)·(k+B+B2)·ln(N/B)` 的比值随 `N` 趋于 1：

```
         N      bound(N)    asymptotic    k*log(N/k)  bound/asym
     4,096         8,000         5,323         1,065        1.50
    65,536        18,217        15,970         2,484        1.14
 1,048,576        28,835        26,617         3,904        1.08
16,777,216        39,480        37,264         5,323        1.06
```

### 3.8 每一步的直觉，以及三个常见误读

把证明压缩成一句话链条：

> 随机块序 → “第 i 块是谁”均匀 → 期望贡献 `≤ L/i` → 求和得 `L·H_m`
> → compact 次数 `R ≤ A/B2` → `E[W] ≤ (1+k/B2)·L·H_m` → `O(k log(N/k))`。

**误读一：“随机化是为了保证正确性。”** 不是。报告明确说这是为了“避免极端输入下的
性能退化”（`...zh.md:43`），`code/02` 的表里顺序扫描的结果也是**对的**（最后一列全
`True`）。正确性来自算法结构（阈值筛选 + 精确的 radix select），随机化只影响 `W`。

**误读二：“`L = k+B+B2` 里的 `B` 是块大小，所以 `B` 越小越好。”** 不是。
`L` 随 `B` 增大而增大（界变松），但 `B` 太小又会让 `m = N/B` 变大，
把 `H_m` 推高。两边都在涨，所以 `B` 有个最优点。`code/03` 的 Table 3 用 `B2` 演示了
同一个 trade-off：界的最小值落在 `B2 ≈ k` 附近（见表里 `B2=1024` 那一行）。

**误读三：“`W = A + kR` 里的 `kR` 项不重要。”** 它恰恰是 `(1+k/B2)` 这个因子的来源。
如果只看 `A`（`O(L log m)`）就会低估代价；把 `kR` 加进来，`B2` 取得小（compact 频繁）
时 `k/B2` 会显著放大界。这就是为什么 §2.2 强调“触发线是 `k+B2` 而不是 `k`”。

最后补一句关于**重复元素**：报告假设元素互不相同。若存在相等元素，算法在边界上
（阈值处）可能多收或多漏几个元素，但 `|C_i| < k+B2` 这条不变量不变，§3.2 的排名界
仍然成立（并列只会让“前 `L` 大”这个集合变大，不会让单个块贡献更多），
所以 `E[W]` 的界不恶化。报告的原话是“存在相同元素时，算法不会产生性能下降”
（`...zh.md:55`）。

---

## 4. CUDA 实现：算法定上限，硬件定常数

§2、§3 已经把**上限**定下来了：读一遍输入 + `O(k log(N/k))` 的额外工作。
但从“算法是这样”到“kernel 真的跑到带宽上限”，中间全是常数问题。

报告的原话很直白：“我们使用 CUDA C++ 实现这个 DeepSelect kernel，并大量应用底层 PTX
优化技巧与位运算技巧**减少常数、降低耗时、提高显卡的带宽使用率**”
（`...zh.md:105`）。这一节逐个讲这五类技巧，**重点不是“用了什么指令”，而是“它在压哪个
瓶颈”**。

### 4.1 整体数据流

一个 CTA（线程块）负责**一行**（`batch_idx = blockIdx.x`，
`csrc/cuda_kernels/v3/topk_select.cuh:48-49`）。数据流是：

```text
   GM（输入行）
      │  TMA 载入（多缓冲、swizzled）
      ▼
   shared memory 里的一个 B 块
      │  每个线程取 16 个元素（NUM_ELEMS_PER_THREAD_PER_ROUND == 16）
      ▼
   阈值筛选（生成 hit_mask）
      │  算写回偏移 → 把命中元素写进 incoming 缓冲区
      ▼
   incoming 缓冲区（片上）
      │  攒够触发量
      ▼
   reconstruct：两趟直方图 + pivot + census → 新的 top-k 与新的阈值
      │  阈值回传，继续下一个 B 块
      ▼
   epilogue：输出 (value, index)
```

几个从配置里能读出来的事实（`csrc/cuda_kernels/config.h`、`csrc/api.cpp:151-186`）：

* 每个线程每轮处理 **16 个元素**（`common_parts.cuh:309` 有一条
  `static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND == 16)`）。
* 载入用 **TMA**（Tensor Memory Accelerator），并且是**多缓冲**的：配置里
  `tma_buffer_depth` 是 3、4、5 或 16（`api.cpp:173-186`）。缓冲越多，越能把
  下一块的载入与当前块的计算重叠起来。这就是第 11 章讲的软件流水，只是这里是手写的。
* 块扫描的顺序不是按地址走，而是**按段（segment）做随机排列**：kernel 里维护了一个
  `perm_state`，用模乘映射
  `get_permuted_seg_idx(linear_id, perm_len, perm_mul) = (linear_id·perm_mul + PERM_ADD_BASE) mod perm_len`
  把“第几次访问”映到“访问哪一段”，再用 `advance_perm_state(...)` 推进
  （`common_parts.cuh:452-462`）。**这个映射是双射当且仅当 `perm_mul` 与 `perm_len`
  互素**。这是“打乱但不重复”的实现方式，也是 §2.3 的随机块序在 CUDA 侧的落地。

### 4.2 一遍扫描：用 mask 把两遍压成一遍

先看报告对瓶颈的描述（`...zh.md:107`）：

> 正常来说，在筛选那一步中，每个 CUDA 线程需要扫描两遍输入：第一遍统计出每个线程
> 读到的数字中有多少个大于阈值的元素，以此得知每个线程应该把结果写入到
> `topk_candidate` 数组中的哪个地方；第二遍再次扫描每个元素，并将大于阈值的元素
> 放置在 `topk_candidate` 中的正确位置。此处哪怕使用了 shared memory 避免反复读取
> 显存，其 **CUDA Core 操作的开销也会限制整个算子的性能**。

关键句是最后一句：数据已经在 shared memory 里了，**带宽不是瓶颈，指令数才是**。
两遍扫描意味着每个元素至少被“摸”两次（一次比较、一次（可能）写回），而多数元素其实
是**不命中的**（阈值涨上去之后），第二遍纯属浪费。

DeepSelect 的做法是：第一遍里**顺手生成一个 `uint32_t` mask**，
第 `i` 个 bit 记录第 `i` 个元素是否大于阈值；之后就不用再扫元素了，
直接**枚举 mask 里的 1**：

```cuda
// 枚举命中的位置（示意，摘自 logic，实际在内联 PTX 里）
uint32_t mask = hit_mask & (hit_mask - 1u);   // 关键技巧一：先消掉最低的 1
while (mask != 0) {
    uint32_t e = __ffs(mask) - 1u;            // 关键技巧二：最低的 1 的下标
    mask &= mask - 1u;                        // 再消掉它
    // ... 把第 e 个元素写出去
}
```

（`csrc/cuda_kernels/common_parts.cuh:1476-1484`；报告里对应的描述在
`...zh.md:107`。还有一个优化细节：**最低的那一个命中元素会被提前读出并单独处理**，
让它的访存延迟和后面的前缀和重叠：预取在 `common_parts.cuh:1449-1452`，
写回在 `:1471-1475`。）

两个小技巧值得单独记：

* `__ffs(mask)` 返回最低的 1 的下标（“find first set”）；
* `mask &= mask - 1u` 是**清掉最低的 1** 的标准写法，`mask-1` 会把最低的 1 借位
  变成 0、把更低位的 0 全变成 1，按位与之后就只有这一位被消掉。

于是循环体的执行次数**恰好等于命中的元素个数**，而不是块大小。命中越稀疏，
收益越大。这直接兑现了 §2.4 里“接受率下降”的好处。

图 4（`figures/04-cuda-mask-flow.md`）把这条链路画了出来。

### 4.3 mask 是怎么生成的：`set / prmt / dp4a`

mask 本身也要用指令生成，而且它在一个内层循环里跑，所以要快。报告点了三条指令：
`set.s32.bf16x2`、`prmt.b32`、`dp4a.s32.s32`（`...zh.md:107`）。核心循环长这样
（`csrc/cuda_kernels/common_parts.cuh:1413-1428`，注释来自源码）：

```cuda
for (int32_t j = NUM_ELEMS_PER_THREAD_PER_ROUND - 4; j >= 0; j -= 4) {
    asm volatile (
        "{\n"
        ".reg .b32 r0, r1, ind, nib;\n"
        "set.gtu.s32.bf16x2 r0, %1, %3;\n"   // 一次比较两个 bf16；.gtu 把 NaN 也算命中
        "set.gtu.s32.bf16x2 r1, %2, %3;\n"
        "prmt.b32 ind, r0, r1, 0x7531;\n"    // 各取一个 bit，凑成 4 个 mask 位
        "dp4a.s32.s32 nib, ind, 0xF8FCFEFF, 0;\n"  // 4 个命中压成 0/1/2/4/8
        "mad.lo.u32 %0, %0, 16, nib;\n"      // hit_mask = hit_mask*16 + nib
        "}\n"
        : "+r"(hit_mask) : ...);
}
```

逐条读：

* **`set.gtu.s32.bf16x2`**：一条指令比较**两个** bf16（`bf16x2` = 打包进 32 位），
  条件成立时把对应的 16 位填成 `0xFFFF`。所以一次比较产出两个 bit 的信息。
  `.gtu` 的 `u` 表示无符号比较，源码注释说明这是为了让 **NaN 也被算作命中**
  （NaN 会一路进下游，由后面的 census 报出来）。
* **`prmt.b32`** 是字节置换（permute），在这里把两个 32 位结果里需要的位挑出来，
  拼成一个 4 位的小整数 `ind`。
* **`dp4a.s32.s32`** 是“四路点积并累加”，本来用于量化推理。这里借它把 `ind` 的四个
  字节（取值 `0x00/0xFF`）按权重 `0xF8FCFEFF` 转成 `1/2/4/8` 再求和，**一条指令
  就把 4 个命中打包进 nibble**。
* `mad.lo.u32 hit_mask, hit_mask, 16, nib` 把这一段追加到 `hit_mask` 的低 4 位。

所以“生成 mask”这件事，是**4 个元素一组、用 5 条指令**完成的。这就是 `code` 里
“16 个元素一组”的原因：`16 / 4 = 4` 轮，正好填满一个 `uint32` 的 16 位。

### 4.4 用浮点加法替换整数加法：一个反直觉的技巧

这是全章最“花哨”、也最值得理解的一个技巧。报告的原话（`...zh.md:109`）：

> 该算子中的比较指令和位运算指令数量极多。根据我们的观察，此类指令会和整数加法指令
> 抢占部分硬件资源，造成算子性能下降。哪怕将整数加法换成与比较、位运算指令没有冲突的
> 整数乘加（`mad`）指令，其吞吐也只有浮点乘加的一半，成为性能瓶颈。观察到，当 `x` 和
> `y` 满足 `0 <= x, y <= 2^22` 时，
> `x+y = __float_as_uint(__uint_as_float(x) + __uint_as_float(y))`。

把它拆成三步理解：

1. **冲突来自哪里**：这个 kernel 里比较（`set`）、位运算（`prmt`/`lop3`/`and`）是主力，
   而整数加法（`add.u32`）和它们在 GPU 上**共用同一组执行端口/发射槽**。
   于是算子被这类指令的吞吐卡住。
2. **换成 `mad` 行不行**：不行。整数乘加的吞吐只有浮点乘加的一半，本身就成了瓶颈。
3. **换成浮点加法**：浮点加法和比较/位运算**不抢资源**（走的是另一类端口），
   吞吐翻倍。问题只剩：怎么让浮点加法算的**正好是整数加法**？

**为什么成立。** `float32` 的位布局是“1 位符号 + 8 位指数 + 23 位尾数”。当
`0 ≤ x < 2^23` 时，`x` 的位模式里符号位和指数位全为 0，剩下 23 位就是 `x` 本身，
也就是说，**把它当成 float 读出来，它是一个次正规数（subnormal），而其尾数恰好是 `x`**。
对两个 `≤ 2^22` 的整数，`x+y ≤ 2^23` 仍然落在次正规范围内，浮点加法不会发生规格化、
不会把进位“串”到指数位，因此

```text
整数加法  x + y          的位模式
  == 浮点加法  as_float(x) + as_float(y)  的位模式
```

用源码里的写法就是（`csrc/cuda_kernels/common_parts.cuh:1041,1086`）：

```cuda
// USE_CLUSTER_ADDRESSING 分支用 add.u32；
// 否则就退化成"把偏移当成浮点加"：
"add.f32 %0, %0, 0f00000008;\n"        // 等价于 dst_ptr += 8
```

`0f00000008` 就是整数 `8` 的位模式；`%0` 是 shared memory 指针的整数地址。
两个“小于 `2^23` 的正整数”做浮点加法，结果仍是同一个整数的位模式，于是
**地址自增这个最频繁的整数加法，被搬到了浮点流水线上**。

> 注意：一个需要你自己核对的细节：报告给的条件是 `x, y ≤ 2^22`
> （`...zh.md:109`），而代码里另有一个常量
> `MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION = 1u << 23`
> 用来约束 `vocab_size`（`csrc/structs.h:20-22`）。两者不是同一个量（前者约束
> **单个加数**，后者约束**索引范围**），不要混用；`code/` 里也没有复现这个技巧
> （它需要真实硬件才有意义）。这一点登记在 `references.md` 的“待补资料”里。

### 4.5 其它压低常数的技巧：前缀和、字节提取、直方图

报告还点了三处（`...zh.md:111`），它们分别对应“算写回偏移”、“读字节”、
“做 radix select”：

**（1）warp 级前缀和。** 每个线程知道自己命中了几个元素，但**不知道前面线程命中了
多少**，所以算不出写回位置。这需要一个 warp 内的前缀和。源码用 `shfl.sync.up`
做 log 步的 Hillis-Steele 扫描（`csrc/cuda_kernels/utils.cuh:6-25`）：

```cuda
for (uint32_t i = 1; i <= 16; i <<= 1) {
    asm volatile ("shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff;\n"
                  "@p add.u32 %0, %0, t;\n" : "+r"(x) : "r"(i));
}
```

注意 `@p` 这个谓词：越界的 lane（`shfl` 无效）**不加**，所以不需要额外的掩码指令。

> 有意思的是，主循环里最后**没有**用这个函数，而是用了另一种写法：每次
> `__ballot_sync` 取一个 bit 平面、再 `__popc`，用 5 次 ballot 完成前缀和
> （`common_parts.cuh:1457-1465`）。源码注释解释了这个选择：命中数 ≤ 16，
> “ballot 之间互相独立，关键路径比 5 步 shuffle 扫描短得多”。同一个 warp 前缀和，
> 在这个 kernel 里有两个版本，`shfl` 版用在 init window 的计数上
> （`common_parts.cuh:778`），`ballot` 版用在主筛选循环里，服务两种不同的数据形态。
> **这是“常数”层面的权衡。**

**（2）字节提取。** 在 bf16 的 radix select 里，要取一个 16 位值的**高字节**或
**低字节**。直接的写法是移位加掩码，但那要占用移位端口。源码用 `prmt` 一次取两个
字节（`common_parts.cuh:944-968`）：

```cuda
"prmt.b32 hi, %0, 0, 0x5341;\n"     // 两个半字的高字节，各放到一个 0x00hh 半字
"lop3.b32 lo, %0, %4, 0x00ff00ff, 0x28;\n"   // 一次逻辑运算完成"异或+与"
```

`lop3` 是“三输入逻辑运算”，一条指令完成任意 3 输入布尔函数，把“异或、与非”合起来，
少发指令。

**（3）直方图构建。** radix select 的第一步是统计每个 8 位桶里有多少元素。源码用
`red.shared.add.u32`（shared memory 的原子加）直接累加，并用 `bfe`/`mad` 同时算出
两个桶的地址（`common_parts.cuh:902-925`）：

```cuda
"bfe.u32 i0, %0, 8, 8;\n"      // 取第一个值的第 8..15 位（高字节）
"bfe.u32 i1, %0, 24, 8;\n"     // 取第二个值的第 24..31 位
"mad.lo.u32 a0, i0, 4, %1;\n"  // 桶地址 = base + bucket*4
"mad.lo.u32 a1, i1, 4, %1;\n"
"red.shared.add.u32 [a0], 1;\n"
"red.shared.add.u32 [a1], 1;\n"
```

**一次循环处理两个元素、一次算两个桶地址、两条原子加**，把“地址计算”和“累加”
的指令数都压到最低。

### 4.6 radix select 本身：pivot 是怎么来的

前面一直在说 compact 用 radix select，但没展开它怎么做。锁定源码里的实现是**两趟
8 位直方图**（`common_parts.cuh:1106-1128`）：

```text
① 对候选元素的高 8 位建直方图（256 个桶）
② 从高位到低位累加桶计数，找到「第 k 大落在哪个高字节桶」→ 记为 pivot_hi8
③ 只对落在 pivot_hi8 桶里的元素，再建一次低 8 位的直方图
④ 同样找到低字节桶 → pivot_lo8，拼成完整的 16 位 pivot
⑤ 拿 pivot 去和所有候选比一次：数出 > pivot 的个数、= pivot 的个数（census）
⑥ 用这两个数算出「= pivot 的要取多少个」（eq_quota），以及写回起始位置
```

几个关键点：

* **为什么是 radix 而不是 sort**：我们只要第 `k` 大（以及它两侧的划分），
  radix select 是线性的，sort 是 `n log n`。
* **为什么要先“distort”**：IEEE 浮点的位表示**不保持数值顺序**（负数的位模式比正数
  大）。所以要把浮点序映射成无符号整数序，才能用 radix 桶的**单调性**。源码用
  `xor` 技巧：正数翻符号位、负数取反（`csrc/cuda_kernels/bit_utils.cuh:22-45`），
  bf16 的 SIMD 版本用 `prmt` 一次处理两个（`:53-63`）。
* **census 用 `set.gt` + `add.rn.bf16x2` 累加**（`common_parts.cuh:971-999`）：
  比较结果直接当 `1.0/0.0` 用 bf16 加法累加，**用浮点累加器统计整数计数**，
  又一次把工作搬到了浮点流水线上（和 §4.4 同一个思路）。源码里还有一条
  `static_assert(2 * N <= 256)`，因为 bf16 累加的精度假设是“计数不超过 256”。
* **`eq_quota`**：如果 pivot 有并列，`cnt_gt < k ≤ cnt_gt + cnt_eq`，
  就必须从相等的那些里**精确地取 `k - cnt_gt` 个**。这就是 §3.8 提到的“并列”在实现
  层的处理方式（`compute_equal_quota_and_prefix`，`common_parts.cuh:483-508`）。

### 4.7 小 batch 与 wave quantization：Threadblock Cluster

前面所有优化都假设“有足够多的行来填满 GPU”。但 Lightning Indexer 场景的 batch
可以小到 1，而 GPU 有上百个 SM，**一行只用一个 CTA，就只用了一个 SM**。
更隐蔽的是 **wave quantization**：即使 batch 不小，最后一个“波”里也可能只剩几个 CTA
在跑，其余 SM 空转。

报告的做法是（`...zh.md:113`）：

> 对于 batch size 较小的场景（或者 wave quantization 效应下最后一个 wave 中活跃 SM
> 数量较少的场景），我们使用 Threadblock Cluster 将一个序列的计算切分至多个 SM 上。
> 具体来说，每条序列会被切成若干份，每份首先经由一个 CTA 求出这一份中 top-k 大的
> 元素。紧接着，每个 CTA 会将这 top-k 个元素发送至其所在的 Threadblock Cluster 的
> CTA0，再由 CTA0 取出这些元素中的前 k 大元素。

这就是标准的“两阶段并行 top-k”：**每个 CTA 先算本段的 top-k，再由 cluster 的 CTA0
归并**。注意它用的是 **Threadblock Cluster**（Hopper 引入的机制）而不是“写完再去
全局内存读回来”，cluster 内可以用 **DSM（Distributed Shared Memory）** 直接访问
邻居 CTA 的 shared memory，省掉一趟全局内存往返。

触发条件是代码里的一个硬编码判据（`csrc/api.cpp:156-161`）：

```cpp
if (batch_size <= 6 && vocab_size >= 512*1024 && topk <= 1024) {   // TODO Tune
    // cluster 版本：cluster_size = 16
}
```

注释写着 `TODO Tune`，说明这是**调出来的经验值**，不是理论上算出来的。这一点也很
说明问题：**算法告诉你“可以并行划分”，硬件告诉你“从哪个 batch/vocab 开始划才划算”**。

另外还有一个更基础的 wave 判据：`num_waves = ceil(batch_size / num_sm)`，
当它等于 1 时用一个配置（512 线程 / `B=8192`），否则用另一个（256 线程 / `B=4096`）
（`csrc/api.cpp:151-177`）。同样是“常数层面的回退规则”。

### 4.8 小结：每个技巧在压哪个瓶颈

把 §4 的技巧列成表，这是本章最该记住的一张表：

| 技巧 | 压的瓶颈 | 一句话原理 | 出处 |
| --- | --- | --- | --- |
| `hit_mask` + `__ffs` + `mask &= mask-1` | 指令数（两遍扫描 → 一遍） | 只枚举命中的位置，命中越少越省 | `...zh.md:107`、`common_parts.cuh:1476-1484` |
| `set.bf16x2` / `prmt` / `dp4a` | mask 生成的指令数 | 4 个元素一组、5 条指令出 4 个 bit | `common_parts.cuh:1413-1428` |
| 浮点加法替整数加法 | 执行端口争用 | `≤2^22` 的整数加法 = 它们的位模式做浮点加法 | `...zh.md:109`、`common_parts.cuh:1041,1086` |
| warp 前缀和（ballot 版） | 写回偏移计算的延迟 | 命中数 ≤ 16，5 次 ballot 比 5 步 shuffle 短 | `common_parts.cuh:1457-1465`、`utils.cuh:6-25` |
| `prmt` 取字节 / `lop3` | 移位与掩码指令数 | 一条指令取两个字节 / 完成三输入布尔 | `common_parts.cuh:944-968` |
| `red.shared.add` 建直方图 | radix select 的指令数 | 一次循环两个元素、两条原子加 | `common_parts.cuh:902-925` |
| census 用 `add.rn.bf16x2` 累加 | 又一次端口争用 | 用浮点累加器数整数计数 | `common_parts.cuh:971-999` |
| Threadblock Cluster | 小 batch 与 wave quantization | 一条序列切到多个 SM，CTA0 归并 | `...zh.md:113`、`api.cpp:156-161` |

**这张表的读法**：左边全是“常数”，右边全是“瓶颈”。**没有一条改变了渐近复杂度**，
复杂度还是 §3 证的那个 `O(k log(N/k))`。这就是本章标题
“算法决定上限，硬件决定常数”的完整含义。

---

## 5. 昇腾侧：同一套算法，另一组常数

### 5.1 只有 `csrc/ascend_kernels/`，而且约束更窄

本章的材料里，昇腾侧只有三个文件：

```text
DeepSelect/csrc/ascend_kernels/kernel.h     19 行   接口声明
DeepSelect/csrc/ascend_kernels/config.h    245 行   编译期参数与 UB 内存布局
DeepSelect/csrc/ascend_kernels/kernel.asc  946 行   全部实现（Ascend C，bisheng 编译）
```

它**不是** TileLang，也**不是** Ascend C 的 TileLang 包装：它是一个用 Ascend C
（`kernel_operator.h`、`asc_*` 向量内建、`AscendC::Sort`）直接写的 kernel，
后缀名 `.asc`（`csrc/ascend_kernels/kernel.asc:1-12`）。它由昇腾后端编译（毕昇），
与 CUDA 版本共享的只有**算法**，不共享一行代码。

约束也比 CUDA 侧更窄，`api.cpp` 的昇腾分支里是三条硬断言
（`csrc/api.cpp:223-225`）：

```cpp
TORCH_CHECK(value_t == at::kBFloat16, "input dtype must be bfloat16 on Ascend");
TORCH_CHECK(output_index_t == at::kInt, "output_idx dtype must be torch.int32 on Ascend");
TORCH_CHECK(!sorted_value, "sorting by value is not supported on Ascend");
```

再加上两边共有的 `topk <= 4096`（`csrc/api.cpp:57`），以及昇腾侧更小的
`MAX_VOCAB_SIZE = 2 * 1024 * 1024`（`csrc/structs.h:42`，CUDA 侧是 `1 << 23`）。

**所以“同一套算法，两个后端”这句话要加一个限定词**：算法相同，**能力集不同**。
能用 CUDA 的 fp32 采样路径，不代表昇腾能用。

### 5.2 昇腾怎么把同一套算法落地

昇腾的 kernel 结构与 CUDA 侧**同构但不同形**。逐段对照：

**（1）分页 + 随机页序，代替随机块序。** `config.h` 里有一个
`TopkPermGenerator`（`csrc/ascend_kernels/config.h:43-88`），把输入切成
`shuffle_page_size = 2048` 个元素的“页”，再给出一张**逻辑页 → 物理页**的映射表。
`config.h:93-101` 的注释把算法完整写了一遍（中文）：

```text
// 算法流程：
// 1. 将输入序列划分为大小为 shuffle_page_size 的页，并随机打乱各页的处理顺序。
// 2. 取乱序后的前 warmup_pages 页作为预热数据，通过 radix selection 求出其中的
//    topk，并将 topk 中的最小值记为 threshold。…
// 3. 将剩余页按 batch 处理，每个 batch 包含若干完整页，其元素总数由 block_size 指定。
//    处理每个 batch 时，将大于 threshold 的元素依次写入候选区；当累计候选元素数达到
//    filtered_capacity 时，… 再通过一轮 radix selection 得到新的 topk…
```

这就是 §2 的算法：**页 = 块**（`shuffle_page_size` ≈ 块大小），
**batch = 扫描块**（`block_size`），**`filtered_capacity` = `B2`**。
另外它显式写出了一个**warmup 阶段**：先读几页建立初始阈值，再进入稳态扫描
（`config.h:93-101` 的第 2 步）。CUDA 侧有对应的机制，一个 init window，
用 `NUM_ELEMS_IN_INIT_WINDOW` / `NUM_INIT_ROUNDS_MAX` / `NUM_SEGS_IN_INIT_WINDOW`
三个常量描述（`common_parts.cuh:321,350`）。两边都要“先读一小段把阈值抬起来”，
理由是同一个：`k` 相对 `N` 很小时，`-inf` 开局的预热期越短，候选量越少。

> 注意： **一处需要与作者确认的偏差**：报告要求块排列“independent of x”
> （`...zh.md:23,53`），但昇腾实现把随机种子取自输入的**前 8 个字节**
> 来源：`input_hash = ld_dev(row_in, 0)`，`permutation_seed = input_hash ^ valid_len`
> （`csrc/ascend_kernels/kernel.asc:737-740`）。这样每一行会得到不同的顺序、
> 又不需要维护随机数状态，工程上说得通；但它与证明里“与输入独立”的假设
> **并不严格一致**。这一点登记在 `references.md` 的“待补资料”里，需要向上游确认。

**（2）片上内存是一张手工算好的表。** `UnifiedBufferMemoryPlan`
（`config.h:168-243`）把所有 buffer 的偏移写成编译期常量，并用 `static_assert`
把总量卡在 `248 KB` 以内（`config.h:19,238`）。这张表里最值得看的是
**候选区与 warmup 区共用一段 UB**，注释写得很清楚：“The leading region is
deliberately reused by two phases: warmup owns it first; candidate value/low/high
own it after warmup selection starts”（`config.h:210-211`）。**复用**是这里的
核心手段：UB 只有那么大，算法需要的所有东西必须挤进去。

**（3）直方图是向量化的。** radix select 的直方图用 `asc_cumulative_histogram_bin0/bin1`
一次累加两个 128 通道的向量（`csrc/ascend_kernels/kernel.asc:18-74`），
低字节那一趟用 `asc_eq` 先把“高字节等于 pivot”的元素筛出来再统计
（同文件 `:52-70`）。这和 CUDA 侧的 `red.shared.add` 是同一个思路的两种实现。

**（4）排序只发生在最后。** 如果用户要 `sorted_index`，kernel 在**输出前**才调用
`AscendC::Sort<int32_t, true>`，而且只对 `topk` 个元素排序
（`kernel.asc:844-846`）。**扫描全程不排序**。这正是 §1.3 说的“把排序从 `N` 上
挪到 `k` 上”。昇腾的 `Sort` 还需要 `topk_capacity` 大小的 permutation 与临时区
（`config.h` 的 `kSortIndexBytes` / `kSortTemporaryBytes`），这些也都在那张 UB
预算表里，并且复用了输入暂存区（`config.h:237-241` 的注释“Sorting reuses the
input staging region after all loads end”）。

**（5）流水是手工的 flag 配对。** `kernel.asc:701-710` 建了 `kNumStages` 组
`event_t`，每组一对 `set_flag(PIPE_V, PIPE_MTE2, ...)` / `wait_flag(...)`，
循环体里 `stage = stable_idx % kNumStages`（`kernel.asc:782`）。这就是第 22 章讲的
软件流水在 Ascend C 层的手写版本：**手动多缓冲 + 手动同步配对**。
对应到 TileLang，`T.Pipelined` 与编译器自动插入同步（第 19 章 §2.4、§7）会把
这一整段包起来。**这正是 TileLang 擅长、而手写最费劲的那部分**。

### 5.3 逐项对比：哪些技巧是 CUDA 专有的

把 §4 的八个技巧原样搬到昇腾，会发生什么？

| CUDA 侧技巧 | 昇腾能做吗 | 昇腾靠什么达到同样效果 |
| --- | --- | --- |
| `hit_mask` + `__ffs` + `mask &= mask-1` | **不直接支持** | 用**向量比较 + 向量选择**产出掩码（`vector_bool`），再交给向量的 compact / 直方图路径；昇腾侧没有 `__ffs` 这类标量位技巧，走的是 SIMD 路数 |
| `set.bf16x2` / `prmt` / `dp4a` | **PTX 专有** | 昇腾的对应物是 `asc_lt/asc_ge/asc_eq`（向量比较）+ `asc_select`（向量选择）+ `asc_cumulative_histogram_bin*`（向量直方图），见 `kernel.asc:18-74` |
| 浮点加法替整数加法 | **不需要** | 这个技巧是为绕开 CUDA 的**执行端口争用**；昇腾的向量/标量指令集与流水组织不同，没有这条动机。昇腾侧强调的是**减少同步、平衡 Cube/Vector**（第 19 章 §8） |
| warp 级前缀和 | **没有 warp 概念** | 昇腾用 `asc_reduce_sum` 这类**向量归约**，或用 `T.alloc_reducer`（TileLang M 树）替代；见 `kernel.asc:100-104` 的 `asc_reduce_sum` |
| `prmt` 字节提取 / `lop3` | **指令不同** | 昇腾用 `vxor` 直接对向量做位运算（`kernel.asc:44-51,58-61`），需要 `MODE_MERGING` 时甚至要退回裸 intrinsic（源码注释：“No asc_* C-API exposes MODE_MERGING for vxor”，`kernel.asc:48-49,58-59`） |
| `red.shared.add` 建直方图 | **有对应** | `asc_cumulative_histogram_bin0/bin1`（向量累加直方图）；256 桶被劈成两个 128 通道的向量（`kernel.asc:18-20`） |
| radix select + distort | **同一个算法** | 昇腾侧同样做两趟字节直方图 + `topk_find_bucket` 找桶（`kernel.asc:110-128`），同样用“高位字节 + 低位字节”拼出 pivot（`kernel.asc:406-473`）；浮点序到无符号序的映射用的是同一套 xor 思路 |
| Threadblock Cluster | **不适用** | 昇腾侧没有 cluster/DSM 这套抽象；多行并行是靠“一行一个 AI Core”的网格铺开（`kernel.asc:892-893` 用 `get_block_idx()` 取行号），小 batch 的原语不在本章材料里 |

一句话总结这张表：

> **相同的只有算法（页式随机序 + 阈值 + radix select）；从指令往下，两边没有任何
> 一条技巧是共用的。** 昇腾用 SIMD 向量原语 + 手工 UB 布局 + “一行一核”的网格，
> 达到了同样的效果，代价是**换了一整套实现**。

### 5.4 为什么本章不是“TileLang 章节”

到这里必须正面回答：**本章的题目里有 TopK，为什么主角不是 TileLang？**

因为 **DeepSelect 根本不是用 TileLang 写的**。它的两个后端分别是：

* CUDA 侧：`csrc/cuda_kernels/**`，C++/CUDA + 大量内联 PTX；
* 昇腾侧：`csrc/ascend_kernels/kernel.asc`，Ascend C + `asc_*` 向量内建。

而 TileLang 里**确实有一个 TopK 原语**，但它解决的不是这个问题。A 树的
`T.tile.topk` 的文档第一句就写明了做法，

> “Performs a TopK operation by **sorting** the source data and extracting the top K
> elements.”
> （`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend_tile.py:550-560`）

而且它要求 `src` **等形（static shape）**，因为要给临时缓冲做容量计算
（同文件 `:587` 抛出 `topk requires src buffer with static shape`）。

两条就足够解释了：

1. **它排序**。而 DeepSelect 的全部意义就是**不排序**（§1.3）。用排序的 topk
   去实现一个“为了不排序而设计”的算法，等于把已经省下来的那趟又加回去。
2. **它要求整块数据放进片上**。DSA 的一项 `compressed_seqlen` 可以到百万，
   一行 bf16 是 2 MB，远超 UB / shared memory。DeepSelect 的算法恰恰是为了
   **不把整行放进片上**而设计的（§2.5 只放 `O(k+B+B2)`）。

所以本章的定位是清楚的：**这是一个“算法先行”的案例，它的价值在于示范
“先算复杂度、再谈实现”这条方法论，而不是示范 TileLang 的写法。** 教材把这一章放进
第五部分，正是为了让读者看到：**这套开源栈里并不是每个组件都用 TileLang，
哪些用了、哪些没用、为什么，本身就是需要学的知识。**

（顺带一提：第 33 章的 TileKernels 是“一份算法、两个后端都写 TileLang”的正例；
本章是它的对照面。两章一起看，“什么时候该用 TileLang”这个问题才完整。）

### 5.5 判据：什么算子适合 TileLang，什么适合手写

把上一节展开成一份可以拿去用的判据。它综合了本章 §4 的八条技巧、第 19 章 §7 的
“TileLang 省掉了什么”、以及第 28 章的“先算再写”。

**先问一类问题：这个算子的收益从哪来？**

| 如果收益主要来自…… | 更适合 | 理由 |
| --- | --- | --- |
| 规则的 tile 划分、片上复用、多级流水 | **TileLang** | 这些正是 DSL 的调度空间（第 3、11 章）；编译器自动推导 layout 与多缓冲 |
| 矩阵乘 / attention / 归一化 / 逐元素 / 量化转换 | **TileLang** | 有对应的原语（`T.gemm` / `T.copy` / `T.reduce_*` / `T.tile.*`），且两个后端都能覆盖 |
| 需要同时支持 CUDA 与昇腾 | **TileLang** | 一份算法写两遍后端代码 vs 一份 TileLang 代码两个 target，成本差一个量级（见第 33 章） |
| 细粒度的线程协作（warp shuffle、ballot、`__ffs` 枚举） | **手写** | tile 抽象里没有“warp”这个概念；这类技巧必须在 SIMT 层面表达 |
| 位级 / 标量密集的控制流（本 chapter 的 mask、distort、pivot 选择） | **手写** | 收益来自指令级技巧，DSL 无法表达，也无法验证它有没有生成对 |
| 依赖 DSL 未暴露的硬件机制（Threadblock Cluster / DSM、特定 PTX） | **手写** | 没有对应原语；第 24 章会讲 cluster 在 TileLang 里的进展，但 v0.1.15 的通用路径没有 |
| 只有“单个 kernel 打满某台机器的最后 20%” | **手写（在 TileLang 版之上）** | 先有 TileLang 的正确版本，再定点替换热点。这是最省力的顺序 |

**再问三个“是不是”（三条否决线）：**

1. **整块数据放得进片上吗？** 放得进 → TileLang 有戏；放不进（比如一行 1M 的元素）
   → 要么改成流式算法，要么手写。
2. **收益能不能用 `(flops, bytes)` 说清？** 能 → 第 28 章的方法论适用，TileLang 的
   自动调度能搜到好配置。不能（收益来自“少发几条指令”这类微观性质）→ 手写。
3. **能不能容忍“编译器不保证把 `T.Parallel` 张量化”？** 第 19 章引用的昇腾手册
   明确写了“自动调度…不保证把任意 `T.Parallel` 变成 SIMD 指令”
   （`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:132-154`）。如果要的是
   **指令级的确定性**，就得往 `T.SimdVF` + `T.simd.*`（M 树）或手写走。

**最后给一条经验规则。** 把上面压成一句话：

> **收益能写成“少搬了多少字节、少做了多少次乘加”的算子，交给 TileLang；
> 收益只能写成“少发了几条指令、把某类指令挪到别的流水线上”的算子，手写。**

DeepSelect 属于后者。所以它手写，而且两个后端各写一遍。这不是 TileLang 的失败，
而是**分工**：TileLang 负责那 90% 规则的部分，手写负责剩下 10% 的极限。

---

## 6. 性能怎么读

### 6.1 指标是“有效内存带宽”，不是 FLOPs

DeepSelect 的性能图纵轴是 **effective memory bandwidth（有效内存带宽，TB/s）**。
仓库对为什么这么选说得很清楚：

> “Measured with the benchmark in `tests/test.py` (`python3 tests/test.py
> --perf-only`), which reports the ratio against `torch.topk` on the same input.
> The metric is **effective memory bandwidth: TopK does no floating-point math, so a
> FLOP rate would not be meaningful here**.”
> （`.../DeepSelect/README.md:41-44`）

这个指标怎么算的，看 `tests/test.py:150-170` 就知道：

```python
total_size = (sum of input bytes) + (output value bytes) + (output index bytes)
...
print(f"deepselect     : {time_usage*1e6:9.3f} us, {total_size/time_usage/1e12:.3f} TB/s")
...
print(f"torch.topk     : {torch_time*1e6:9.3f} us, {total_size/torch_time/1e12:.3f} TB/s"
      f"  (speedup {torch_time/time_usage:.2f}x)")
```

所以“有效带宽”= **（输入字节 + 输出字节）/ 时间**。要读出三件事：

1. **它是一个“导出的”指标**，不是硬件规格表上的峰值带宽。分子里连输出也算进去了，
   所以严格说它是“这个算子总共搬了多少字节”除以时间。
2. **它衡量的是“离只读一遍有多近”**。§3 已经证明，DeepSelect 相对输入只多做
   `O(k log(N/k))` 的工作，所以它的有效带宽应当逼近“读一遍”的理论上限；
   而 `torch.topk` 因为多一趟选择/排序，同样的分子下时间更长，带宽就更低。
3. **它不能跨算子比较**。一个 GEMM 的有效带宽和一个 TopK 的有效带宽不是一回事，
   §6.3 会再说一遍。

### 6.2 “2–20 倍”是相对加速，不是绝对带宽上限

仓库自述的加速比是（`.../DeepSelect/README.md:3-5`）：

> “supports NVIDIA CUDA and Huawei Ascend platforms, and achieves **2 ~ 20x speedup
> compared to vanilla `torch.topk`**”

这个数字**是相对于 `torch.topk` 的倍数**，不是“达到了硬件带宽的百分之多少”。
读性能图（`assets/perf_*.png`）要按形状分情况：

**（1）bf16 Lightning Indexer，CUDA**（`assets/perf_bf16_cuda.png`）。
横轴是 `vocab_size`（16K → 1M），三个子图分别是 `batch = 6 / 512 / 4096`。
定性结论：

* **大 batch + 长行**（`batch=4096, vocab=1M`）时两根柱子差得最远，DeepSelect 的柱高
  在 5 TB/s 量级、`torch.topk` 在 0.3 TB/s 量级，接近 README 说的 20× 那一端。
  **但这两个数字是从图上读的近似值，不是精确测量**。
* **小 batch**（`batch=6`）时，**两个实现的柱高都很低**（DeepSelect 也在 0.5 TB/s 以内）。
  这正是 §4.7 的两个问题：活跃 SM 太少，以及 wave quantization。仓库为此专门做了
  Threadblock Cluster 版本，并用 `batch_size <= 6` 作为触发条件之一
  （`csrc/api.cpp:156`）。
  **读法**：小 batch 那一列低，不代表算法差，代表并行度不够。

**（2）bf16 Lightning Indexer，昇腾**（`assets/perf_bf16_ascend.png`）。
形状与 CUDA 版一致（`topk=512`，三个 batch 档）。同样是“大 batch 高、小 batch
低”的形态，但纵轴上限更低（到 4 TB/s 刻度）：大 batch 长行时 DeepSelect 的柱高在
3.5 TB/s 量级，`torch.topk` 仍在 0.2 TB/s 量级。
**这两张图不能直接比较绝对数值**，机器不同、形状相同不等于可比。

**（3）fp32 采样场景，CUDA**（`assets/perf_fp32_cuda.png`）。
`vocab_size = 129280, topk = 512`，横轴是 `batch_size`。
这一张的形态与前两张不同：

* 大 batch（4096）时 DeepSelect 约 2.8 TB/s、`torch.topk` 约 1.1 TB/s，
  差距大约 **2.5×**。这就是 README 里“2×”那一端。
* `batch=6` 时两者都很低（DeepSelect ≈ 0.1 TB/s）。

**为什么同一个算子会有 2× 到 20× 这么宽的区间？** 因为加速比是**两个实现之比**：

* `torch.topk` 在 bf16 + 超长行 + 小 `k` 时最弱（它要处理 `N` 上的一趟选择），
  所以分母小、倍数大；
* 在 fp32 + 固定长度 128K 时 `torch.topk` 表现相对好一些，分母变大、倍数变小。

**结论**：看到“2～20×”时，第一件事是问“什么形状、哪台机器、相对谁”。
仓库自述里这三件事都写了（README §Supported Cases + §Performance），
这正是我们要引用的原因。

### 6.3 我们不能测什么，以及一个必须说的矛盾

先说明白：**本章没有、也不可能有任何本机实测数字**（见开头的验证边界）。
本机既没有 NVIDIA GPU，也没有昇腾 NPU，所以：

* 上面所有的 TB/s 都是**读图得到的量级**，标注为“从图上读，近似”；
* “2～20×”是**引用仓库自述**（`README.md:3-5`）；
* 我们唯一自己算出来的是 §3 的复杂度界与 `code/03` 的公式表。那是**渐近量级**，
  不是带宽，也不是加速比（图 5 的 notes 里专门写了这一条）。

最后回到本章的题眼。回头看 §2.1 里的那个矛盾：

| | 报告的建议 | 仓库发布的 CUDA 实例 |
| --- | --- | --- |
| `B`（块大小） | 1024（配 `k=512`） | 4096 或 8192（`api.cpp:173-186`） |
| `B2`（compact 松弛量） | 1024 | 4096 |

报告给的 `B=B2=1024` 是**让复杂度界最小**的那一组；代码用的 `B` 大得多，
按 §3.6 的式子算，界反而更松（`N=1M` 时从 ≈28.8k 涨到 ≈78k）。为什么代码不按报告来？

**因为复杂的界不是唯一的成本项。** `B` 更大意味着：

* 每次 TMA 传输更大、更接近带宽峰值；
* 单位元素上的循环开销（地址计算、同步、分支）更少；
* 相应地 compact 触发得更频繁（`B2` 也更大，部分抵消）。

反过来，`B=B2=1024` 让候选缓冲区更小、片上压力更低，但传输更碎。
**哪一组更快，只能测。而测出来的是常数，不是复杂度。**

这就是本章标题的最后一层含义：

> **算法（§2–§3）决定的是“不可能比 `N + O(k log(N/k))` 更省”；
> 硬件（§4–§5）决定的是“在给定的机器上，常数取多少”。**
> 前者是数学，可以证；后者是工程，只能调。

§5.5 那张判据表，本质上就是在问“这个算子的收益是数学层的还是常数层的”，
决定了它该用 TileLang 还是手写，也决定了你要花多少时间在 proof 上、多少时间在 profiler 上。

---

## 7. 本章小结

把这一页当作复习卡片。

**问题。** TopK 在 DeepSeek 里出现两次：DSA 的 Lightning Indexer（bf16，行长可到百万）
和采样器（fp32，词表约 128K）。两处的共同形状是 **`k` 小（≤ 4096，主要 512）、
`N` 大**。朴素实现的代价是“读一遍 `N`（不可省）+ 一趟与 `N` 同阶的选择/排序
（可省）”。

**算法（三步骤）。**

```text
循环：① 按随机块序读一个 B 块
      ② 只保留 > T 的元素，追加到 topk_candidate
      ③ 候选 ≥ k+B2 时 compact 回 k 个，T ← 其中最小值
结束：最后一次 compact → top-k
```

**三条不变量。**

1. 块顺序必须是**随机**的（避免大值靠后时退化成 `A ≈ N`）；
2. `|topk_candidate| ≤ k + B2 + B`（触发线 `k+B2`，两块之间最多再加 `B`）；
3. 阈值 `T` **只升不降**，所以接受率递减、缓冲区增长递减。

**复杂度证明（四步）。**

```text
★ 被加入的元素 ≤ 前 L 大（L = k+B+B2）                ← 缓冲区有界 + 历史落选者 ≤ T
  B_i 在 U_i 里均匀                                    ← 随机块序
  E[A_i|U_i] ≤ |T_i|/i ≤ L/i                          ← 均匀 + |T_i| ≤ L
  E[A] ≤ L·H_m；R ≤ A/B2；W = A + kR
  ⇒  E[W] ≤ (1+k/B2)·L·H_m = O((1+k/B2)(k+B+B2)log(N/B))
  ⇒  取 B,B2 = Θ(k)：E[W] = O(k log(N/k))  ≪  N
```

**CUDA 的常数技巧（都在压指令数/端口争用，不改复杂度）。**

| 技巧 | 压什么 |
| --- | --- |
| mask + `__ffs` + `mask &= mask-1` | 两遍扫描 → 一遍 |
| `set.bf16x2` / `prmt` / `dp4a` | mask 生成的指令数 |
| 浮点加法替整数加法（`≤2^22`） | 执行端口争用 |
| ballot 版 warp 前缀和 | 关键路径延迟 |
| `prmt` 取字节 / `lop3` / `red.shared.add` | radix select 的指令数 |
| Threadblock Cluster | 小 batch 与 wave quantization |

**昇腾的差异。** 只有 `csrc/ascend_kernels/**`（Ascend C，`.asc`）；只支持 bf16、
只支持 int32 索引、不支持按值排序、`MAX_VOCAB_SIZE` 更小。**相同的只有算法**：
分页随机序（`TopkPermGenerator`）、warmup 页建立初始阈值、UB 预算表、向量直方图、
最后才对 `k` 个元素排序、手工 flag 流水。所有 CUDA 专有的技巧在昇腾侧都换成了
SIMD 向量原语。

**为什么本章不是 TileLang 章节。** DeepSelect 用 CUDA/PTX 与 Ascend C 写成，
与 TileLang 无关；TileLang 的 `T.tile.topk` 是**排序式**且要求静态形状，
正好解决不了这里的问题。把这一章放进教材，是为了示范“先算复杂度、再谈实现”，
以及回答“哪些算子不该用 TileLang”。

**性能数字只来自一手材料。** 指标是有效内存带宽（= (输入+输出字节)/时间），
加速比是**相对 `torch.topk`** 的倍数（2～20×，随形状与机器变化），
不是绝对带宽上限。本机无 GPU/NPU，本章所有 TB/s 都是从仓库自带的图上读的近似量级。

**最后一句话。** 算法决定上限（`N + O(k log(N/k))`），硬件决定常数（`B` 取 1024 还是
8192）。前者可以证，后者只能调。

---

## 延伸阅读

下面每条都指向本章 [`references.md`](references.md) 里的条目 id。

* **一手算法材料**：`R1`（DeepSelect 算法与实现简析，中文，**本章 §2、§3、§4 的
  主要依据**）、`R2`（英文版，措辞略有差异，可对照读）、`R3`（仓库自述：两个场景、
  约束、性能图说明）。
* **源码（CUDA）**：`S1`（`common_parts.cuh`：mask 生成、census、reconstruct）、
  `S2`（`bit_utils.cuh`：distort / un-distort）、`S3`（`utils.cuh`：warp 前缀和）、
  `S4`（`config.h` / `api.cpp`：配置与派发、cluster 触发条件）、`S5`（`structs.h`：
  对齐要求与 `MAX_VOCAB_SIZE`）。
* **源码（昇腾）**：`A1`（`kernel.asc`：向量直方图、候选筛选、索引恢复、排序）、
  `A2`（`ascend_kernels/config.h`：`TopkPermGenerator`、配置策略、UB 内存预算）、
  `A3`（`ascend_kernels/kernel.h`：接口）。
* **接口与测试**：`P1`（`deep_select/interface.py`：参数语义与对齐要求）、
  `P2`（`tests/test.py`：正确性判据与性能用例的形状清单）。
* **本机可运行**：`code/01`（纯 Python 复刻 + 与 `torch.topk` 对拍）、
  `code/02`（随机块序 vs 顺序扫描）、`code/03`（复杂度与代价模型小表）。
  三个示例的验证级别见 [`code/README.md`](code/README.md)。
* **跨章**：第 13 章（reduction 与逐元素算子的记账法）、第 16 章 §4.3（DSA 在模型里的
  位置）、第 19 章（昇腾上写算子的整体流程与 msProf 调优）、第 22 章（多缓冲与同步插入，
  §5.2 的手工 flag 流水对应它）、第 28 章（算子设计方法论，本章 §1 的账单与
  §5.5 的判据都建立在它之上）、第 33 章（TileKernels：同样一份算法用 TileLang
  写两个后端的正例，与本章互为对照）。
* **本节发现的资料问题**（昇腾侧种子取自输入、报告与发布配置的 `B/B2` 取值差异、
  `2^22` 与 `1<<23` 的关系）见 [`references.md`](references.md) 的“待补资料”。
