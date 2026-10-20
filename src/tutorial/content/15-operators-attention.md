# 第 15 章 算子实战 III：Attention 家族

## 学习目标

读完本章，你应当能够：

1. 数清一次朴素 attention 到底在 HBM 上读写了几趟、每一趟搬了多少元素，并用同一套
   记账法说明 FlashAttention 把哪几趟消掉了。
2. **自己推导 online softmax**：写出“分块 → 运行最大值 → 重标度 → 累加”的递推式，
   说明为什么每一步的重标度因子是 `exp(m_prev - m_new)`，并证明最终结果与先算完整
   softmax 再乘 `V` 完全等价。不是背结论，而是知道每一步为什么可以这么做。
3. 写出 attention 的四级递进实现：① 朴素版（materialize `S×S`）→ ② 分块 + online
   softmax → ③ 加流水与 shared memory 复用 → ④ 对齐官方 `flash_attention` 例子，并
   说清每一级“上一版为什么不够、这一版改了什么”。
4. 面对 causal / varlen / GQA / decode / paged KV / sliding window / attention sink
   七个变体，能在锁定版本的 `examples/` 里指出各自实现在哪个文件、改了哪几行。
5. 说清昇腾上 attention 的瓶颈与 NVIDIA 有什么不同：为什么昇腾的 `QK^T` 结果必须
   绕一趟 workspace，为什么“Cube 与 Vector 的核间流水”是主要矛盾。
6. 识别数值稳定性、causal 边界、varlen 打包、split-KV 归约顺序这四类坑，并知道
   每一类在锁定源码里的具体位置。

## 前置知识

* 第 9 章：`T.copy` / `T.gemm` / `T.reduce_max` / `T.reduce_sum` / `T.fill` 的语义。
* 第 10 章：shared memory 与 fragment 的区别、layout 为什么是接口。
* 第 11 章：`T.Pipelined`、多缓冲、`stage`/`order`、“流水优化吞吐不优化延迟”。
* 第 13 章：reduction 类算子的写法与 `T.reduce_*` 的 `clear` 语义。
* 第 14 章：`T.gemm` 的 M/N/K 协议、warp 划分与 `GemmWarpPolicy`（本章大量复用）。
* 第 6 章（可选）：NV 的 SIMT + Tensor Core 与昇腾的 Cube/Vector 分工。

**本章的验证边界（很重要）**：写作环境是 macOS (Apple Silicon)，**没有 NVIDIA GPU，
也没有昇腾 NPU**。因此本章出现的数字只有两类，来源都写清楚：

* **本机模型的输出**：来自 `code/01`–`code/04` 的纯 CPU 程序（标准库 + numpy），
  可以在任意机器上复算；本章把它标为“CPU 逻辑验证”。
* **厂商/上游自带的数字**：来自锁定版本源码里的 README 与 benchmark 文档，带原始
  文件路径与测量条件；本章把它标为“厂商自报，作者未复现”。

本章**没有任何一条本机实测的 kernel 性能数字**，也**不声称任何 kernel 被运行过**。
所有 `code/` 目录里的 kernel 源码都只是“可以抄到有卡机器上跑”的骨架，验证级别写在
每个示例的 `README.md` 里。

---

## 1. 问题的由来：attention 的时间花在哪

### 1.1 三个矩阵乘与一个 softmax

Attention 的形状约定本章统一用 `BHSD`（batch × head × seq × dim），因为它和
`examples/flash_attention/example_mha_fwd_bhsd.py` 的索引顺序一致；`BSHD` 只是换一下
转置，本章末尾会说明它的实现差别。

对每个 `(batch, head)` 对，输入 `Q, K, V ∈ R^{N×d}`，输出 `O ∈ R^{N×d}`：

```text
S = Q Kᵀ · s            # [N, N]   第一处矩阵乘，s = 1/√d
P = softmax(S, dim=-1)  # [N, N]   逐行的 softmax
O = P V                 # [N, d]   第二处矩阵乘
```

想要“会写 attention kernel”，第一步是看清楚：`S` 和 `P` 都是 `N×N`。当 `N = 32K`
时，`N×N` 在 fp32 下是 **4 GB**（单个 batch、单个 head）。这还没有算上读写它的时间。

### 1.2 朴素实现的三趟访存

把上面三行写成最直白的 GPU 实现（三个独立 kernel，或一个 kernel 里来回读写全局内存），
每个 `(batch, head)` 的 HBM 流量是：

| 步骤 | 读 | 写 |
| --- | --- | --- |
| ① `S = QKᵀ` | `Q`(Nd) + `K`(Nd) | `S`(N²) |
| ② `P = softmax(S)` | `S`(N²) | `P`(N²) |
| ③ `O = PV` | `P`(N²) + `V`(Nd) | `O`(Nd) |

合计约 `4N² + 4Nd` 个元素。`N` 一大，`4N²` 就是全部：**计算是 `O(N²d)`，访存也是
`O(N²)`，但 `d` 只有 64–256，而 `N` 可以到 10⁵**。这就是 1.3 节要说的“算得不多、
搬得很多”。

第 1 章引用的厂商资料给出过这两个数量级的差距：未命中缓存的全局访存是“hundreds of
clock cycles”，而多数算术指令“typically 4 cycles”
（`../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md`，条件：
未缓存的 local/global 访存与 compute capability 7.0 设备）。所以把 `N²` 的中间结果
写出去再读回来，代价不是“多一点带宽”，而是把整个 kernel 的时间轴拉长。

### 1.3 算得不多，搬得很多

第 2 章给过算术强度（arithmetic intensity）的定义：**每搬一个字节做了多少次浮点运算**。
把 `N×N` 中间矩阵留在片上不出 HBM，等价于说“让人均算力更高的设备少等数据”。

一句话概括本章要解决的问题：

> **朴素 attention 的瓶颈不是 `O(N²d)` 的乘加，而是 `O(N²)` 的中间结果在 HBM 上来回。**

FlashAttention 的贡献是：**保持结果完全精确（没有近似），把中间结果的读写留在片上**。
它在论文摘要里的原话是“an IO-aware exact attention algorithm that uses tiling to reduce
the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU
on-chip SRAM”（`../tilelang-corpus/03-tilelang-papers/raw/arxiv-2205-14135.md`）。

### 1.4 本章的四级阶梯

本章按 PyTorch tutorial 的方式，一次只改一件事：

| 级 | 版本 | 相比上一版**唯一**的改变 | 这一版为什么还不够 |
| --- | --- | --- | --- |
| ① | 朴素版 | 无 | `S`、`P` 落地到 HBM，`4N²` 流量 |
| ② | 分块 + online softmax | 中间结果只在片上；softmax 改成单趟递推 | 搬运与计算没有重叠；`Q` 反复从 HBM 读 |
| ③ | 加流水与 shared memory 复用 | `T.Pipelined` + 多缓冲 + 寄存器复用 | 形状写死、没有覆盖真实模型的变体 |
| ④ | 对齐官方实现 | 与 `examples/flash_attention/` 逐行对齐 | 无 |

第 4、5 节再把 ④ 扩成“变体矩阵”与“两台机器的瓶颈差异”。

---

## 2. 概念与模型

### 2.1 记号

固定一个 `(batch, head)` 和一行 query，令：

* `x_j = q · k_j` 是第 `j` 个 key 的**未缩放得分**；`s = 1/√d` 是缩放因子；
* `v_j` 是第 `j` 个 value 向量；
* `m` 是任意常数（下面会说明它为什么可以是任意常数）。

标准的 safe softmax 写成：

$$
\begin{aligned}
P_j &= \frac{\exp(s x_j-m)}{\sum_k \exp(s x_k-m)}\\
O &= \sum_j P_j v_j
\end{aligned}
$$

### 2.2 第一步：为什么可以减去任意常数

这是全部推导的基石，只有两行：

$$
\begin{aligned}
&\frac{\exp(sx_j-m)}{\sum_k\exp(sx_k-m)}\\
&\quad=\frac{\exp(sx_j)\exp(-m)}
{\sum_k\exp(sx_k)\exp(-m)}\\
&\quad=\frac{\exp(sx_j)}{\sum_k\exp(sx_k)}
\end{aligned}
$$

**`exp(−m)` 在分子分母上各出现一次，约掉了。** 所以：

* 数学上 `m` 取什么都一样；
* 工程上 `m` 只有一个作用，**让 `exp` 的指数不为正**，避免上溢。

正因为它是“约掉”的，后面所有看起来复杂的重标度步骤，都只是在维护“当前用的是哪个
`m`”，而不是在改变结果。这一步想通了，online softmax 就没有魔法了。

### 2.3 第二步：单趟 softmax 的递推（连续情形先推一遍）

假设我们沿 key 方向**分块**处理，第 `t` 块记作 `B_t`。处理完前 `t` 块后，我们想
持有三个量，使得：

$$
\begin{aligned}
\ell^{(t)} &= \sum_{j\in B_1\cup\cdots\cup B_t}\exp(s x_j-m^{(t)}),\\
O^{(t)} &= \sum_{j\in B_1\cup\cdots\cup B_t}\exp(s x_j-m^{(t)})v_j.
\end{aligned}
$$

第一行是分母的运行和，第二行是尚未归一化的分子。

其中 `m⁽ᵗ⁾ = max_{j ∈ B₁∪…∪B_t} (s·x_j)` 是**到目前为止**见过的最大值。

处理新一块 `B_{t+1}` 时，会出现一个新候选最大值 `M = max_{j ∈ B_{t+1}}(s·x_j)`，
于是新的运行最大值是

$$
m^{(t+1)}=\max(m^{(t)},M)
$$

问题来了：`ℓ⁽ᵗ⁾` 和 `O⁽ᵗ⁾` 是按**旧的** `m⁽ᵗ⁾` 归一化的，而现在要换成按
`m⁽ᵗ⁺¹⁾` 归一化。代进去看：

$$
\exp(sx_j-m^{(t+1)})
=\exp(sx_j-m^{(t)})\exp(m^{(t)}-m^{(t+1)})
$$

所以只要把旧的两个量和乘同一个因子 `c = exp(m⁽ᵗ⁾ − m⁽ᵗ⁺¹⁾)`，再补上新块的贡献：

$$
\begin{aligned}
c &= \exp(m^{(t)}-m^{(t+1)}),\\
\ell^{(t+1)} &= c\ell^{(t)}
  +\sum_{j\in B_{t+1}}\exp(sx_j-m^{(t+1)}),\\
O^{(t+1)} &= cO^{(t)}
  +\sum_{j\in B_{t+1}}\exp(sx_j-m^{(t+1)})v_j.
\end{aligned}
$$

第一行是重标度因子，后两行更新运行和与未归一化输出。

最终输出

$$
O=\frac{O^{(T)}}{\ell^{(T)}}
$$

**这就是 online softmax 的全部推导。** 只有两条性质被用到：

1. 指数可以拆开：`exp(a − b)·exp(b − c) = exp(a − c)`；
2. 求和可以分段再合并（加法结合律）。

### 2.4 等价性证明（不是“看起来对”）

**命题**：按上面递推算出来的 `O⁽T⁾/ℓ⁽T⁾`，等于先算完整 softmax 再乘 `V` 的结果
`Σ_j P_j v_j`。

**证明**：用数学归纳法先证一个更强的结论，对任意 `t`，

$$
\begin{aligned}
O^{(t)} &= \exp(-m^{(t)})\sum_{j\le t}\exp(sx_j)v_j && (A)\\
\ell^{(t)} &= \exp(-m^{(t)})\sum_{j\le t}\exp(sx_j) && (B)
\end{aligned}
$$

*`t = 1`*：`O⁽¹⁾ = Σ_{j∈B₁} exp(s·x_j − m⁽¹⁾) v_j`，把 `exp(−m⁽¹⁾)` 提出去就是 (A)。

*归纳步*：假设 (A)(B) 对 `t` 成立，讨论 `t+1`。

$$
\begin{aligned}
O^{(t+1)}
&= \exp(m^{(t)}-m^{(t+1)})O^{(t)}
  +\sum_{j\in B_{t+1}}\exp(sx_j-m^{(t+1)})v_j\\
&= \exp(m^{(t)}-m^{(t+1)})\exp(-m^{(t)})
  \sum_{j\le t}\exp(sx_j)v_j\\
&\quad+\exp(-m^{(t+1)})
  \sum_{j\in B_{t+1}}\exp(sx_j)v_j\\
&= \exp(-m^{(t+1)})
  \left(\sum_{j\le t}\exp(sx_j)v_j
  +\sum_{j\in B_{t+1}}\exp(sx_j)v_j\right)\\
&= \exp(-m^{(t+1)})\sum_{j\le t+1}\exp(sx_j)v_j.
\end{aligned}
$$

第二行到第三行使用了归纳假设。

(B) 同理。归纳成立。

由 (A)(B)，比值

$$
\frac{O^{(t)}}{\ell^{(t)}}
=\frac{\sum_{j\le t}\exp(sx_j)v_j}
{\sum_{j\le t}\exp(sx_j)}
$$

**`exp(−m⁽ᵗ⁾)` 又一次约掉了**。所以运行最大值取大取小、什么时候更新，都不影响最终
结果，只影响中间数的动态范围。取 `t = T`，右边正是 `Σ_j P_j v_j`。∎

### 2.5 从“三趟”到“一趟”

把 2.3 节的递推放进 tile 级代码，就得到 FlashAttention 的主循环（NVIDIA 版，非
causal）：

```text
for k in 0 … ceil(N/block_N):
    S_blk = Q_blk · K_blkᵀ · s          # 只在片上：fragment
    m_new = max(m_prev, rowmax(S_blk))
    c     = exp(m_prev − m_new)          # 重标度因子
    P_blk = exp(S_blk − m_new)
    ℓ     = c · ℓ + rowsum(P_blk)
    O     = c · O + P_blk · V_blk
O_final = O / ℓ
```

对比 1.2 节的三趟：

| | 朴素版 | FlashAttention |
| --- | --- | --- |
| `S` 是否落地 | 写 `N²`、读 `N²` | 只在 fragment 里 |
| `P` 是否落地 | 写 `N²`、读 `N²` | 只在 fragment 里 |
| 需要几趟遍历 key | 3 趟（max、sum、`PV`） | 1 趟（块内同时更新 max/sum/`O`） |
| 每个 key 被读几次 | 每趟一次，共 3 次 | 每个 `Q` 块读一次 |

论文摘要给出的定性结论是“it requires fewer HBM accesses than standard attention,
and is optimal for a range of SRAM sizes”
（`../tilelang-corpus/03-tilelang-papers/raw/arxiv-2205-14135.md`）。
**本 corpus 只保存了这三篇 FlashAttention 论文的标题与摘要**（见
`references.md` 的“待补资料”），所以本章不引用摘要之外的论文数字；具体的
`4N² + 4Nd` 记账是本章 `code/01-traffic-model` 里的**模型**，不是论文数据。

### 2.6 与第 11 章的关系：这里也有“搬运与计算”

flash attention 的循环体是“搬 `K`、算 `QKᵀ`、写 `S` 到 fragment、softmax、搬 `V`、
算 `PV`”。只要把 `T.copy(K…)` 与 `T.gemm(Q, K)` 放进同一个 `T.Pipelined` 循环，
第 11 章那套“搬运与计算重叠”就自动生效。这正是下一节 ③ 版要做的事。

---

## 3. 用法：四级递进实现

四个版本的完整代码都在 `code/` 下，本节只讲“改了什么、为什么”。

### 3.0 共同骨架

所有版本共享同一段 TileLang 骨架（以 `BHSD`、`block_M × block_N` 分块为例）：

```python
with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
    Q_shared = T.alloc_shared([block_M, dim], dtype)
    K_shared = T.alloc_shared([block_N, dim], dtype)
    V_shared = T.alloc_shared([block_N, dim], dtype)
    acc_s    = T.alloc_fragment([block_M, block_N], accum_dtype)   # 分数
    acc_o    = T.alloc_fragment([block_M, dim], accum_dtype)       # 输出累加器
```

三个维度的含义与官方例子一致：`bx` 扫 query 方向、`by` 是 head、`bz` 是 batch。
**这是 `T.Kernel` 的网格语义（第 8 章）**：一个 `(bx, by, bz)` 就是一个线程块，
块内所有 tile 都在片上。

### 3.1 ① 朴素版：把 `N×N` 物化出来

最直白的写法是“一个线程块负责一个 query 行块，把整行分数都算出来，softmax，再乘 V”：

```python
with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
    Q_shared = T.alloc_shared([block_M, dim], dtype)
    K_shared = T.alloc_shared([seq_kv, dim], dtype)      # ← 整个 K
    V_shared = T.alloc_shared([seq_kv, dim], dtype)      # ← 整个 V
    S_shared = T.alloc_shared([block_M, seq_kv], dtype)  # ← 整个 S
    P_local  = T.alloc_fragment([block_M, seq_kv], accum_dtype)
    acc_o    = T.alloc_fragment([block_M, dim], accum_dtype)
    ...
```

**为什么不够。** 三个理由，按严重程度排在下面：

1. **放不下。** `S_shared` 是 `block_M × seq_kv` 的 fp16。`seq_kv = 8192`、`block_M = 128`
   就是 `128×8192×2 B = 2 MB`，已经超过一块 GPU 的 shared memory 预算；`seq_kv = 32K`
   直接不可能。
2. **就算放得下也不划算。** 中间结果在片上停留的时间再长，它也是“用完全部 key 才能
   开始算输出”，一次只服务一个 `Q` 块。
3. **softmax 需要整行。** 朴素版为了取行最大值必须先把整行算完，于是“算分数”与
   “算输出”被硬性串起来。

`code/04-block-flash-attention/block_flash_attention.py` 里的 `exact_masked_attention()`
就是这个版本（直接构造完整的 `[Sq, Skv]` 分数矩阵，再整行 softmax），它用 numpy 在
CPU 上跑。**它的存在价值是当参考实现**，3.2 节的每一处改动，都以“结果跟它逐元素
相等”为判据。

### 3.2 ② 分块 + online softmax：把三趟折成一趟

改了什么：

* 循环沿 key 方向分块，`K_shared` / `V_shared` 只装一个 `block_N`；
* `S` 不再物化，只保留一个 `block_M × block_N` 的 fragment `acc_s`；
* 引入三个“每行一个数”的状态：`scores_max`（运行最大值 `m`）、`scores_sum`（运行
  分母 `ℓ`）、`acc_o`（未归一化分子 `O`）；
* 循环末尾统一 `acc_o /= logsum`。

```python
for k in T.Pipelined(loop_range, num_stages=num_stages):
    T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
    T.clear(acc_s)                       # causal 时换成写 -∞ / 0 的 mask，见 3.4 节
    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    T.copy(scores_max, scores_max_prev)          # 保存旧 m
    T.fill(scores_max, -T.infinity(accum_dtype)) # 清成 −∞ 再 reduce
    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
    for i in T.Parallel(block_M):
        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])   # m_new = max(m_prev, M)
    for i in T.Parallel(block_M):
        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
    for i, j in T.Parallel(block_M, block_N):
        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
    T.reduce_sum(acc_s, scores_sum, dim=1)
    for i in T.Parallel(block_M):
        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]     # ℓ = c·ℓ + ℓ_blk
    T.copy(acc_s, acc_s_cast)
    for i, j in T.Parallel(block_M, dim):
        acc_o[i, j] *= scores_scale[i]                              # O = c·O
    T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared)
    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
```

这段代码与 2.3 节的公式是**逐项对应**的：

| 公式 | 代码 |
| --- | --- |
| `m⁽ᵗ⁺¹⁾ = max(m⁽ᵗ⁾, M)` | `scores_max[i] = T.max(scores_max[i], scores_max_prev[i])` |
| `c = exp(m⁽ᵗ⁾ − m⁽ᵗ⁺¹⁾)` | `scores_scale[i] = T.exp2(m_prev·scale − m_new·scale)` |
| `ℓ⁽ᵗ⁺¹⁾ = c·ℓ⁽ᵗ⁾ + ℓ_blk` | `logsum[i] = logsum[i]*scores_scale[i] + scores_sum[i]` |
| `O⁽ᵗ⁺¹⁾ = c·O⁽ᵗ⁾ + P_blk·V_blk` | `acc_o[i,j] *= scores_scale[i]` 之后 `T.gemm(acc_s_cast, V_shared, acc_o)` |
| `O / ℓ` | 循环后的 `acc_o[i, j] /= logsum[i]` |

三个实现细节值得单独点出来：

* **为什么用 `exp2` 而不是 `exp`。** 官方例子的 `scale` 里乘了 `log2(e)`：
  `scale = (1.0 / dim) ** 0.5 * 1.44269504`
  （`_vendor/tilelang-v0.1.15/examples/flash_attention/example_mha_fwd_bhsd.py:24`）。
  这样 `exp(x)` 变成 `exp2(x·log₂e)`，硬件上 `exp2` 与乘加可以合成一条指令。
* **为什么 `T.fill(scores_max, -inf)` 之后还要 `clear=False`。** `T.reduce_max` 的
  `clear=True` 本来就会先把输出初始化成 `-inf`
  （`_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:176-204`）；
  官方写成“显式 fill + `clear=False`”是同一个语义的另一种写法，读代码时不要以为
  `clear=False` 在这里意味着“累加到旧值上”。
* **为什么缩放只施在 `exp` 的指数上，而不缩放 `scores_max` 本身。** 因为
  `x ↦ s·x` 单调且 `s > 0`，`max(s·x) = s·max(x)`，所以 `reduce_max` 拿到的未缩放
  最大值乘以 `scale` 就是缩放后的最大值。这也是官方源码把 `scale` 放在 `exp2` 里、
  而不放在 `T.gemm` 之后的原因。

**为什么还不够。** 这一版已经把 `4N²` 的流量干掉了，但留下了两件事：

1. **搬运与计算没有重叠。** `T.copy(K…)` 与 `T.gemm(Q, K…, acc_s)` 在同一个循环体里
   前后排着，`num_stages=1` 时它们是串行的（第 11 章 1.1 节）。
2. **`Q` 被反复读。** 每个 `Q` 块只在块内复用；`K`、`V` 每个 `Q` 块都要重读一遍。
   这不是 bug，而是“用带宽换片上空间”的必然结果。

### 3.3 ③ 加流水与片上缓冲复用（shared memory / register）

改了什么，只有三处：

1. **`k` 循环换成带 `num_stages` 的 `T.Pipelined`**，让第 `k+1` 块的 `K`、`V` 搬运
   与第 `k` 块的 `QKᵀ`、`PV` 重叠（第 11 章的机制）。
2. **`policy=T.GemmWarpPolicy.FullRow`**：让每个 warp 负责 `acc_s` 的完整若干行。原因
   见 4.2 节，`reduce_max` / `reduce_sum` 是行内归约，FullRow 让归约尽量落在
   warp 内部。
3. **`acc_s` 复用**：softmax 后的概率直接写回 `acc_s`，再用 `T.copy(acc_s,
   acc_s_cast)` 做一次 fp32 → fp16 转换给第二个 `T.gemm`。fp16 的 `P` 不能直接原地改，
   因为 `T.gemm` 的 A 操作数要求 `P` 是低精度。

```python
for k in T.Pipelined(loop_range, num_stages=num_stages):   # ← 改动 1
    T.copy(K[...], K_shared)
    ...
    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
           policy=T.GemmWarpPolicy.FullRow)                # ← 改动 2
    ...
    T.copy(acc_s, acc_s_cast)                              # ← 改动 3
    T.copy(V[...], V_shared)
    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
```

**片上占用要自己算。** 加了 `num_stages` 之后，`K_shared`、`V_shared` 各变成
`num_stages` 份。以 `block_M=block_N=128, dim=128, fp16` 为例：

| 缓冲 | 单份大小 | 份数 | 合计 |
| --- | --- | --- | --- |
| `K_shared` | `128×128×2 B = 32 KB` | `num_stages` | `32·ns` KB |
| `V_shared` | `32 KB` | `num_stages` | `32·ns` KB |
| `Q_shared` | `128×128×2 B = 32 KB` | 1 | 32 KB |

`num_stages=2` 就已经是 160 KB；`num_stages=3` 是 224 KB。这就是官方例子里
`block_M=128, block_N=128, num_stages=2` 作为回归配置的原因
（`_vendor/tilelang-v0.1.15/examples/flash_attention/example_mha_fwd_bhsd.py:167`）。
**编译器不会替你把这个账算小**：超了就是编译失败或启动失败，不是变慢。

**为什么还不够。** 这一版已经是一个可用的 flash attention 了，但它把形状写死了：
没有 causal 的循环裁剪、没有 `past_len`（KV cache 的前缀）、没有 varlen、没有 GQA。
下一节把它和官方实现对齐，就会看到这些“真实模型需要的东西”分别落在哪几行。

### 3.4 ④ 与官方 `flash_attention` 对齐

官方入口是 `_vendor/tilelang-v0.1.15/examples/flash_attention/example_mha_fwd_bhsd.py`
（另有 `example_mha_fwd_bshd.py` / `example_gqa_fwd_bshd.py` / `example_mha_fwd_varlen.py`
等布局与变体）。把我们的 ③ 版和它逐行对齐，差别只有下面五组：

**（1）`scale` 折进 `log2(e)`，且把缩放放在 `exp2` 里。**

```python
scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)      # 第 24 行
...
acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)   # 第 85 行
```

**（2）支持 `seq_kv ≥ seq_q`（KV cache 场景），用一个 `past_len` 把因果对角线右移。**

```python
past_len = seq_kv - seq_q                              # 第 30 行
q_idx = bx * block_M + i + past_len                    # 第 69 行
k_idx = k * block_N + j                                # 第 70 行
acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
```

**（3）causal 的循环裁剪：不是“扫完再 mask”，而是根本不去扫。**

```python
loop_range = (
    T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M + past_len, block_N))
    if is_causal else T.ceildiv(seq_kv, block_N)
)                                                      # 第 59–63 行
```

`(bx+1)*block_M + past_len` 是这一块 query 里**最大的** query 下标；比它大的 key
不可能被看到（causal 是 `q ≥ k`），所以直接不迭代。这一行把 causal 的算术量砍掉一半，
也把“mask 只在下三角附近出现”这件事变成了循环上界。

**（4）非 causal 时也有一个边界 mask。**

```python
acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_kv, -T.infinity(acc_s.dtype), 0)
```
（`_vendor/tilelang-v0.1.15/examples/flash_attention/example_mha_fwd_bhsd.py:73-74`）

`seq_kv` 不是 `block_N` 的整数倍时，最后一块 `K` 的尾部会越界。`T.copy` 会自动补零
（第 9 章 4.2 节），但**补出来的 0 是真实得分 0，不是“该被忽略”**，所以必须显式
把这些位置的得分设成 `-∞`。

顺带一提：**causal 那一支不需要这个额外的边界 mask**。因为 `q_idx ≤ seq_q − 1 +
past_len = seq_kv − 1 < k_idx`，越界的 key 天然被 `q_idx >= k_idx` 判成不可见。
两分支的 mask 条件不一致，是读官方代码时最容易看漏的一处。

**（5）epilogue 走一次 `O_shared` 再写回全局。**

```python
for i, j in T.Parallel(block_M, dim):
    acc_o[i, j] /= logsum[i]                            # 第 97–98 行
T.copy(acc_o, O_shared)                                 # 第 99 行
T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])   # 第 100 行
```

`acc_o` 是 fragment（寄存器布局，可能带 padding / swizzle），直接写全局内存会产生
非合并访问；先落到 shared memory 再按连续地址写出，是标准的 epilogue 写法。

### 3.5 四个版本放在一起

| | ① 朴素 | ② 分块 online | ③ + 流水/复用 | ④ 官方 |
| --- | --- | --- | --- | --- |
| `S`、`P` 落地 HBM | 是 | 否 | 否 | 否 |
| 遍历 key 的趟数 | 3 | 1 | 1 | 1 |
| 搬运/计算重叠 | 否 | 否 | 是（`T.Pipelined`） | 是 |
| 归约的 warp 划分 | 未指定 | 默认 `Square` | `FullRow` | `FullRow` |
| causal | 无 | 无 | 无 | 循环裁剪 + mask |
| `seq_kv > seq_q` | 无 | 无 | 无 | `past_len` |
| epilogue | 直接写 | 直接写 | 直接写 | 经 `O_shared` |

---

## 4. 实现机制

### 4.1 一件事交给谁

| 想做的事 | 前端写的 | 谁实现 |
| --- | --- | --- |
| 分数只在片上 | `T.alloc_fragment` + `T.gemm` | layout inference 给 fragment 定布局（第 21 章） |
| `K`/`V` 分块搬运 | `T.copy(global → shared)` | copy lowering 选指令：SIMT / `cp.async` / TMA（第 9 章 4.1） |
| 搬运与计算重叠 | `T.Pipelined(..., num_stages)` | `PipelinePlanning` + `InjectSoftwarePipeline`（第 11 章 4.2/4.3） |
| 行内归约 | `T.reduce_max` / `T.reduce_sum` | cross-thread allreduce，≥32 线程走 shared、<32 走 shuffle（第 9 章 4.4） |
| 概率转 fp16 给第二个 GEMM | `T.copy(acc_s, acc_s_cast)` | 元素级 cast + layout 转换 |
| 越界 key 的 mask | 显式 `T.if_then_else` | 前端；**编译器不会替你判断“补零 = 有效得分”** |

### 4.2 为什么是 `FullRow`

`GemmWarpPolicy` 有三个取值（`_vendor/tilelang-v0.1.15/tilelang/tileop/base.py:5-12`）：
`Square`（默认，按接近正方形的比例切）、`FullRow`（所有 warp 分到行方向）、
`FullCol`（所有 warp 分到列方向）。

对 attention 来说，后续的 `reduce_max(acc_s, ..., dim=1)` 与 `reduce_sum(..., dim=1)`
都是**沿 `block_N` 的行内归约**。用 `FullRow` 让每个 warp 负责整行，行内归约主要在
warp 内部完成，跨 warp 的 shared memory 归约最少。这是 `T.gemm` 的 `policy` 参数
第一次真正发挥作用的地方，**GEMM 的 warp 划分不是性能细节，而是被后续算子绑定的
布局约束**。第 14 章讲 GEMM 家族时会看到同一个参数在纯 GEMM 里的含义。

### 4.3 流水在 attention 上到底重叠了什么

第 11 章的机制原样适用：`T.Pipelined` 把循环体按数据依赖打上 `stage`/`order` 注解，
算出每个 buffer 需要的份数（`use - def + 1`），再重排成 prologue / body / epilogue。
在 attention 里，被重叠的是：

* `K_shared` 的搬运 与 上一块的 `QKᵀ` + softmax；
* `V_shared` 的搬运 与 上一块的 softmax + `PV`；
* 第二个 `T.gemm` 与下一块的第一个 `T.gemm`（不同硬件单元）。

**注意 `acc_s` 不能被流水**：它是 `T.gemm` 的累加器，又同时被 softmax 读写，具有
跨语句的读写依赖。官方例子里 `acc_s` 只有一份，`K_shared`/`V_shared` 才是多份的。

### 4.4 Blackwell（sm100）变体：`T.alloc_tmem` 与 `T.tcgen05_gemm`

`examples/flash_attention_sm100/` 的注释写得很直接：

> Blackwell (SM100) Flash Attention Forward using TCGEN05MMA with TMEM accumulators.
> Replaces the Hopper WGMMA-based Flash Attention for Blackwell GPUs.
> 来源：`_vendor/tilelang-v0.1.15/examples/flash_attention_sm100/mha_fwd_bshd.py:1-8`

与 Hopper 版的差别集中在分配与指令：

```python
S_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)   # 分数落在 TMEM
D_tmem = T.alloc_tmem([block_M, dim], accum_dtype)       # 输出累加器落在 TMEM
mbar_s = T.alloc_barrier(1)
...
T.tcgen05_gemm(...)                                       # 显式异步 MMA
```

`T.alloc_tmem` 的 docstring 说明了这块内存的约束：按列分配、列数必须是 2 的幂且
≥32、只能被 TCGEN5.MMA 与特定的 TMEM load/store 指令访问
（`_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:254-270`）。
`T.gemm` 的 docstring 则说清了它和手工异步接口的关系：

> For manual asynchronous scheduling, use `T.wgmma_gemm(...)` with `T.wait_wgmma(...)`
> on Hopper, or `T.tcgen05_gemm(...)` with `T.mbarrier_wait_parity(...)` on Blackwell.
> 来源：`_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:177-179`

（sm100 的 `variant="ss"/"ts"`、warp specialization 等细节属于第 24 章的题目，
本章只把它当作“同一套 tile 抽象在不同硬件上换了指令”的证据。）

### 4.5 昇腾：为什么 `QKᵀ` 的结果必须绕一趟

昇腾 AI Core 把“矩阵计算”和“向量计算”放在不同的核上（Cube / Vector），中间隔着
几级专用存储。锁定版本的编程指南有一张搬运关系表
（`_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md:627-636`）：

| src | dst | 说明 |
| --- | --- | --- |
| GM | L1 | 全局 → L1 |
| L1 | L0A / L0B | L1 → L0A / L0B（Cube 的左/右矩阵） |
| L0C | GM | L0C → 全局 |
| GM | UB | 全局 → Unified Buffer |
| UB | L1 | Unified Buffer → L1 |

**注意这张表里没有 `L0C → UB`。** 而 flash attention 的主循环恰恰要求“把 Cube 算出的
分数交给 Vector 做 softmax，再把概率交回 Cube 做 `PV`”。于是昇腾版必须绕：

```text
Cube:  Q·Kᵀ → L0C → (GM workspace) ─┐
                                     ├─→ Vector: softmax / rescale
Vector: ─────────────────────────────┘  → (GM workspace) → L1 → L0 → Cube: P·V
```

锁定版本的适配器例子把这个绕行写成了显式参数：`flash_attention_fwd` 的签名里有
`workspace_1 / workspace_2 / workspace_3` 三个全局缓冲，分别是“分数”“概率”“部分输出”
的落脚点（`_vendor/tilelang-ascend-v0.1.2.000/examples/flash_attention/flash_attn_bhsd.py:26-34`），
Cube 侧用 `T.set_cross_flag("FIX", 0)` 通知 Vector、Vector 侧 `T.wait_cross_flag(0)`
取数（同文件 `:98-100` 与 `:130-131`）。

主干仓库里的昇腾实现把这件事做得更进一步：编程指南 4.1.5.2 节说明编译器有一个
**Workspace 消除（workspace reduction）** pass，会把 `T.copy(acc_s_l0c, acc_s_ub)`
这种“跨核传递”自动改写成 `copy_l0c_to_gm`（写 workspace）+ `copy_gm_to_ub`（读
workspace）（`_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md:1345-1390`）。
实现就在 `_vendor/tilelang-ascend-v0.1.2.000/src/transform/ascend_workspace_reduction.cc:999-1004`：
那里有一张显式的改写表，`copy_l0c_to_ub` 被改写成 `copy_l0c_to_gm`（同文件
`:1455-1481` 是 pass 的注册入口）。
也就是说：**同一句 `T.copy`，在 NVIDIA 上是 shared ↔ register，在昇腾上可能变成
“L0C → GM → UB”两跳。** 这是第 25 章要展开的事情，但写 attention kernel 时必须
先知道它存在。

### 4.6 变体矩阵在代码里的落点

下表是本章承诺的“七个变体 → 官方文件”映射。全部路径都在锁定版本里核对过。

| 变体 | 锁定版本里的实现 | 关键改动 |
| --- | --- | --- |
| 基础 MHA（`BHSD`） | `examples/flash_attention/example_mha_fwd_bhsd.py` | 3.4 节的五处 |
| 基础 MHA（`BSHD`） | `examples/flash_attention/example_mha_fwd_bshd.py` | 索引顺序改为 `[b, s, h, d]` |
| causal | 同上，`is_causal=True` | 循环裁剪 + `q_idx >= k_idx` mask |
| GQA / MQA | `examples/flash_attention/example_gqa_fwd_bshd.py`（固定长度）、`example_gqa_fwd_varlen.py`（varlen） | `kv_head_idx = head_idx // groups`，KV 头数 = `heads // groups` |
| varlen（padding 打包） | `examples/flash_attention/example_mha_fwd_varlen.py`、`example_gqa_fwd_varlen.py` | `cu_seqlens_q` / `cu_seqlens_k` + `max_seqlen_q` + 越界 guard |
| decode / split-KV | `examples/flash_decoding/example_mha_inference.py`、`example_gqa_decode.py` | 两个 kernel：split 与 combine，中间存 `Output_partial` + `glse` |
| paged KV | `_vendor/tilelang-ascend-v0.1.2.000/examples/flash_attention/paged_flash_attn_bhsd.py` | `block_table[b, k]` 间接寻址 KV 页 |
| sliding window | `examples/attention_sink/example_mha_sink_fwd_bhsd.py` 的 `window_size` | 循环起点 `start` + 上界裁剪 + `check_inf` |
| attention sink | 同上，`Sinks` 输入 | epilogue 里 `logsum += exp2(sink − m·scale)` |
| block-causal（噪声/干净分区） | `examples/block_causal_attention/block_causal_attention.py` | 逐 tile 决定是否 mask、是否只走对角线 |

**关于 `BHSD` 与 `BSHD`。** 二者只是索引顺序不同：`BHSD` 下 `T.copy(Q[bz, by, bx*block_M
:(bx+1)*block_M, :], Q_shared)` 是“batch、head 固定，搬一段 sequence”；`BSHD` 下对应
的是 `T.copy(Q[bz, bx*block_M:(bx+1)*block_M, by, :], Q_shared)`。**搬的那一维在最后
意味着地址连续**，这直接决定 `T.copy` 能不能走 TMA（第 9 章 4.1）。真实推理框架里
KV cache 常按 `[batch, seq, head, dim]` 存储，所以 `BSHD` 变体不是“重复劳动”。

### 4.7 昇腾侧同样有七个变体

`_vendor/tilelang-ascend-v0.1.2.000/examples/` 下与 attention 相关的目录：

| 变体 | 文件 | 与 NV 版的差别 |
| --- | --- | --- |
| MHA forward（Cube/Vector 分离） | `flash_attention/flash_attn_bhsd.py` | 显式 `T.Scope("C")` / `T.Scope("V")` + 三个 workspace |
| paged KV | `flash_attention/paged_flash_attn_bhsd.py` | 与上同构，KV 用 `block_table` 间接寻址 |
| sink（+ sliding window） | `mha_sink_fwd_bhsd/mha_sink_fwd_bhsd.py` | sink 用 mask-after-exp + 双缓冲；注释详细写了 `m* = max(sink, m_final)` 的原因 |
| GQA varlen | `gqa_fwd_varlen/gqa_fwd_varlen.py` | 用 `T.mma` + L0A/L0B/L0C 双缓冲替代 `T.gemm_v0`，外部传入 `Mask` |
| 稀疏索引（DSA/lightning indexer） | `lightning_indexer/example_lightning_indexer.py` | Cube 算 `Q·Kᵀ`，Vector 做加权 reduce + `T.tile.topk` |
| 稀疏 flash attention | `sparse_flash_attention/` | 按 top-k 索引 gather KV |
| 训练/推理端到端 | `deepseek_v4/`、`xllm_kernels/` | 面向具体模型的算子组合 |

`mha_sink_fwd_bhsd.py` 里有一段注释非常值得读，它把本章 5.1 节的数值问题讲清楚了：

> Why `m*` (not `m_i_final`): the kernel uses mask-after-exp, so `m_i_final` may come
> from tail blocks (which contribute 0 via mask-after-exp). … When sinks > `m_i_final`,
> `exp(sinks − m_i_final)` overflows / dominates `sumexp` → output collapses to 0.
> 来源：`_vendor/tilelang-ascend-v0.1.2.000/examples/mha_sink_fwd_bhsd/mha_sink_fwd_bhsd.py:565-577`

### 4.8 三个“只为省带宽”的变体

4.6 节的表把七个变体并列了，但它们背后的动机并不一样。有三个变体**完全不改变
单个 (query head, KV head) 对的数学**，存在的唯一理由是省带宽或换并行度：
GQA/MQA、paged KV、split-KV。它们最容易在设计评审里被问到，值得单独说清楚。

**（1）MQA / GQA：让 KV cache 变小。**

MHA 里每个 query head 都有自己的 `K`、`V`。MQA（multi-query attention）让**所有 query
head 共用一份 `K`/`V`**；GQA（grouped-query attention）折中：把 `heads` 个 query head
分成 `groups` 组，每组共用一份 `K`/`V`。于是：

```text
MHA:  kv_heads = heads          （每个 query head 一份 KV）
GQA:  kv_heads = heads / groups
MQA:  kv_heads = 1              （groups = heads）
```

对 kernel 来说改动只有两处（官方实现见
`_vendor/tilelang-v0.1.15/examples/flash_attention/example_gqa_fwd_varlen.py:19-24,50-52`）：

```python
head_kv = heads // groups          # KV 头数
kv_head_idx = head_idx // groups   # 本 query head 该读哪一份 KV
```

收益在**访存量**上：KV cache 的读取量正比于 KV 头数，GQA 把它降到 `1/groups`。
这对 decode 尤其关键，decode 阶段几乎没有计算、全是读 KV cache，
**KV cache 的字节数就是延迟**。这也解释了为什么现在的开源模型普遍用 GQA：
不是因为它难写，而是因为它把推理时的带宽需求按 `groups` 倍砍掉了。

**（2）paged KV：让 KV cache 能动态增长。**

推理服务里每个请求的上下文长度不同、还会随时增长（continuous batching）。预分配
“最大长度 × batch”的连续 KV cache 会浪费大量显存，于是把 cache 切成固定大小的
**页（page / block）**，用一张表记录“逻辑块 → 物理页”的映射。KV 的地址不再由算术
算出，而是查表：

```python
bidx = block_table[bz, k]           # 逻辑第 k 块 → 物理页号
T.copy(K_cache[bidx, ...], K_shared)
```

（`_vendor/tilelang-ascend-v0.1.2.000/examples/flash_attention/paged_flash_attn_bhsd.py:94`；
`block_table` 是 `[batch, table_blocks]` 的整数输入，见同文件 `:50-58`。）

对 kernel 的影响有三点：① 地址空间里多了一次索引加载；② `T.copy` 的源地址变成
“查表 + 偏移”，**TMA 那种静态描述符不再直接适用**（第 9 章 4.1 节说过 TMA 需要
编译期可描述的地址模式）；③ 页与页在物理内存里不连续，跨页时要么拆成多次 copy、
要么放弃合并访问。**“paged 不改变 attention 的数学，只改变地址来源”**。这句话
要记住。

**（3）split-KV：换并行度，不是省带宽。**

decode 的算术强度极低：一行 query 要读完全部 K/V。按 1.2 节的记账，不管怎么切，
K/V 反正都要读一遍，所以 **split-KV 在带宽上没有任何收益**（`code/01` 的第 3 段模型
也验证了：`Sq = 1` 时两种实现的流量比是 1x）。它买到的是**并行度**：把 KV 切成
`num_split` 段之后，可用的线程块数从 `ceil(Sq/block_m) × heads × batch` 变成它的
`num_split` 倍，长 context 下才够填满机器。

代价有两个，而且都必须付：① 多一个 combine kernel（多一趟读写部分结果的带宽）；
② 数值上多一次归约。这正是 5.4 节要展开的坑。

把三个变体放在一起，可以总结成一句话：

> **GQA 减少要读的字节数；paged KV 改变字节在哪里；split-KV 让更多的核同时去读。**
> 三者都不改变 attention 的数学定义，也都需要 kernel 作者显式处理。这就是“变体矩阵”
> 存在的意义。

---

## 5. 进阶与陷阱

### 5.1 数值稳定性：`m` 到底在保护什么

回到 2.3 节：`c = exp(m_prev − m_new)`。因为 `m_new = max(m_prev, M) ≥ m_prev`，
所以**这个差永远 ≤ 0**，`c ∈ (0, 1]`：

* 如果新块里出现了更大的分数，`m_new > m_prev`，`c < 1`，旧的累加器被**等比缩小**；
* 如果新块没有更大的分数，`m_new = m_prev`，`c = 1`，什么也不用做。

这就是 `max` 的全部作用：保证 `exp` 的指数**永远不为正**，因此永远不会上溢。

顺带说清一个容易被想当然的性质：**`ℓ` 和 `O` 并不单调。** 只要这一行至少有一个可见
key，取到运行最大值的那个 key 就贡献 `exp(0) = 1`，所以 `ℓ ≥ 1`、最后的除法安全；
但运行最大值一旦变大，`c < 1` 会把旧的 `ℓ`、`O` 一起按比例缩小，**并且缩的是同一个
比例**。所以它们的**比值**始终不变（这正是 2.4 节证明的东西）。把“`ℓ` 单增”
当成前提去写代码，就会在“新块带来一个远大于历史的分数”时判断错。

两个必须知道的边界：

**（1）`−∞ − (−∞) = NaN`。** 当一个 query 行在前若干块里**一个可见 key 都没有**
（sliding window 的早期块、或 causal 且 `seq_q > seq_kv` 的行），
`m_prev = m_new = −∞`，于是 `exp2(−∞ − (−∞)) = exp2(NaN) = NaN`。
官方 sink/sliding-window 例子用 `check_inf` 修掉了它：

```python
# NOTE(wt): check_inf is necessary for sliding window attention.
for i in T.Parallel(block_M):
    if window_size is not None:
        scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
```
（`_vendor/tilelang-v0.1.15/examples/attention_sink/example_mha_sink_fwd_bhsd.py:103-109`；
源码注释里把这一过程称作 FlashAttention-3 里的 `Check_inf`，但本 corpus 只保存了
FA3 论文的摘要，见 `references.md` 的“待补资料”。）

**（2）`check_inf` 只解决 NaN，不解决“整行都看不见”。** 如果一个 query 行到最后
仍然没有任何可见 key，那么 `ℓ = 0`，`O/ℓ` 是 `0/0`。官方 varlen 例子显式处理了这
一情形：

```python
# When sq > skv, some tokens can see nothing
acc_o[i, j] = 0 if is_causal and bx * block_M + i + offset < 0 else acc_o[i, j] / logsum[i]
```
（`_vendor/tilelang-v0.1.15/examples/flash_attention/example_gqa_fwd_varlen.py:117-119`）

**（3）mask 用什么值，是一个真实的实现选择。** 固定形状的例子用
`-T.infinity(acc_s.dtype)`；varlen 例子用的是 `-1e9`
（`example_gqa_fwd_varlen.py:84`）。在 v0.1.15 里 `T.infinity` 落到 IEEE `+∞`
（`_vendor/tilelang-v0.1.15/src/op/math.cc:80-108`），而不是 dtype 的最大有限值。
这与“`exp2(−∞ − m) = 0`”是自洽的；换成 `-1e9` 则依赖 `1e9` 足够大到让结果下溢为 0。
**读别人的 kernel 时，先看清它用的是哪一个。**

### 5.2 causal mask 的边界

causal 的三处边界，任何一处写错都会得到“看起来对、只在某些形状错”的结果：

1. **对角线是 `q ≥ k`，不是 `q > k`。** 官方是
   `T.if_then_else(q_idx >= k_idx, 0, -inf)`（`example_mha_fwd_bhsd.py:71`）。
2. **有 `past_len` 时，`q` 的下标要加偏移**：`q_idx = bx*block_M + i + past_len`
   （`example_mha_fwd_bhsd.py:69`）。否则 KV cache 场景下整条对角线会错位。
3. **`k` 的循环上界要用“本块 query 里最大的下标”**：
   `T.ceildiv((bx + 1) * block_M + past_len, block_N)`
   （`example_mha_fwd_bhsd.py:59-63`）。用 `T.ceildiv(seq_kv, block_N)` 不会错，
   但会白白算一半的块，而且多算的块**必须**被 mask 掉。

window 的边界另加一条：窗口是 `q − window_size < k ≤ q`，对应的判断在
`example_mha_sink_fwd_bhsd.py:91-92`；循环起点也要跟着裁：
`start = T.max(0, (bx*block_M + past_len − window_size) // block_N)`
（同文件 `:84`）。

### 5.3 varlen 的打包

varlen（variable length）解决的问题是：一个 batch 里序列长度不同，padding 到同一个
长度会浪费算术。做法是把整个 batch 的 token 拼成一条扁平序列，再用前缀和数组
`cu_seqlens` 记录每个样本的起止：

```python
Q_unpad: [UQ, heads, dim]        # UQ = Σ_b seqlen_q[b]
cu_seqlens_q: [batch_size + 1]   # int32，cumsum
max_seqlen_q: int32              # 编译期常量，决定 grid 大小
```

四个容易踩的点：

1. **`max_seqlen_q` 是编译期常量。** `T.Kernel(T.ceildiv(max_seqlen_q, block_M), ...)`
   要求它在 kernel 编译时就确定，所以不同 batch 的最大长度会触发重编译（第 7 章的
   JIT 缓存）。
2. **grid 大小按 `max_seqlen` 开，短样本要靠 guard 跳过。** 官方 varlen 例子的写回
   有 `if bx * block_M + i < q_current_seqlen`（`example_gqa_fwd_varlen.py:123`）。
3. **causal 对齐方式是“右对齐”**：`offset = kv_current_seqlen − q_current_seqlen`
   （`example_gqa_fwd_varlen.py:68`），注释写明 `# always align on the right`。
   这是“增量解码 + 前缀 KV”的语义：最后一个 query token 与最后一个 key 对齐。
4. **`cu_seqlens` 是 `int32`，不是索引张量**。它每次都从全局内存读一次，属于
   “每个 block 读 4 个数”的极小开销。

### 5.4 split-KV 的归约顺序

decode 阶段只有一个 query（`seq_q = 1`），KV 却有几十万。此时 `Q` 块数量太少，
GPU 根本喂不饱；**Flash-Decoding 的做法是把 KV 方向切成 `num_split` 段，每段独立算
一份部分结果，再用第二个 kernel 合并**。

官方实现在 `_vendor/tilelang-v0.1.15/examples/flash_decoding/example_mha_inference.py`：

**split 段**：每个 `sid ∈ [0, num_split)` 只扫
`[sid·(seqlen_kv//num_split), (sid+1)·(seqlen_kv//num_split))`，输出两样东西，
归一化后的部分输出 `Output_partial[…, sid, :]`，以及这一段的 log-sum-exp：

```python
for i in T.Parallel(block_M):
    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
T.copy(logsum, glse[bid, hid, sid, mid * block_M:(mid + 1) * block_M])
```
（同文件 `:109-111`）

**combine 段**（`T.Kernel`，同文件 `:115-152`）：

```python
T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)     # 各 split 的 lse 最大值
for k in T.Pipelined(num_split):
    lse_logsum_local[i] += T.exp2(lse_local_split[i] - lse_max_local[i])
lse_logsum_local[i] = T.log2(lse_logsum_local[i]) + lse_max_local[i]   # logsumexp(lse)
...
scale_local[i] = T.exp2(lse_local_split[i] - lse_logsum_local[i])      # 权重
o_accum_local[i, j] += po_local[i, j] * scale_local[i]
```

**为什么必须这样合并？** 因为每一段输出已经**各自除过自己的 `ℓ`**：

$$
O_s=\frac{1}{\ell_s}
\sum_{j\in \mathrm{segment}\ s}\exp(sx_j-m_s)v_j
$$

而真实的输出是

$$
O=\sum_j P_jv_j=\frac{\sum_s \ell_sO_s}{\sum_s \ell_s}
$$

所以权重必须是 `ℓ_s / Σ_{s'} ℓ_{s'}`，而 `ℓ_s = 2^(lse_s)`（代码里的 LSE 是
`log2(ℓ)` 的形式；注意固定形状的例子里 `scale` 已经折进 `log2(e)`）。代码用 `exp2`
与 `log2` 的配对实现这个比值，并在求权重前先减一次最大值。**这是 online softmax 的
同一个技巧用在了 split 维度上**。

这一步把“数值稳定性”和“归约顺序”绑在了一起：

* **数学上**，`Σ_s ℓ_s O_s / Σ_s ℓ_s` 与求和顺序无关；
* **浮点上**，`exp2(lse_s − max)` 之后的相加顺序会改变舍入，而且如果 `ℓ_s` 用低精度
  存（例如 fp16），大 context 下各段权重会明显失真。

昇腾的 backward 例子给了一个真实的量级：把 dQ 的 partial gradient **先转 bf16 再
对 64 个 KV tile 做原子累加**，官方 README 报告的相对 L2 误差约 **1.024%**
（dK/dV 分别是 0.316% / 0.037%），并明确说明“The larger dQ error comes from repeated
BF16 rounding during atomic accumulation across 64 KV tiles”，同时提醒
“atomic reduction does not guarantee bitwise determinism”
（`_vendor/tilelang-v0.1.15/examples/ascend/flash_attention/README.md`，测量环境：
Ascend 950DT、CANN/Bisheng 9.2.0、TileLang `8121f415`，fast-math 打开；
**厂商自报，作者未复现**）。这是本章唯一一处“归约顺序影响正确性”的实测数字，
所以它值得记住。

### 5.5 稀疏与索引类：attention 侧只剩一个结构问题

第 16 章会把 DeepSeek NSA / DSA、mInference、kpool 这条线讲全，本章只回答一个问题：
**稀疏 attention 在 kernel 结构上到底改了什么？**

答案是：**key 的遍历不再是一个连续区间，而是一个由索引张量给出的集合。**

| 方案 | 索引的粒度 | 锁定版本里的实现 |
| --- | --- | --- |
| block-sparse | `block_M × block_N` 的 0/1 块表 | `examples/blocksparse_attention/` |
| NSA（native sparse attention） | 每个 query 选 `selected_blocks` 个 `block_size` 块 | `examples/deepseek_nsa/example_tilelang_nsa_fwd.py` |
| mInference（vertical-slash） | 每个 query 一组“竖向 + 斜向”的列索引 | `examples/minference/example_vertical_slash_sparse_attn.py` |
| DSA / lightning indexer | 先算一个轻量打分，再 top-k 出 token 索引 | `_vendor/tilelang-ascend-v0.1.2.000/examples/lightning_indexer/example_lightning_indexer.py` |
| HISA（两级） | 先选 pool block，再选 block 内 token | `examples/dsa_hisa/hisa.py` |
| SeerAttention | 学习到的块稀疏 | `examples/seer_attention/block_sparse_attn_tilelang.py` |

NSA 的前向 kernel 把这一点写得很清楚。它的 kernel 签名里有一个 `BlockIndices`
张量，grid 直接按 `seq_len` 展开：

```python
def native_sparse_attention(Q, K, V, BlockIndices, dim, is_causal, ...,
                            block_size=64, groups=1, selected_blocks=16):
    ...
    block_indices_shape = [batch, seq_len, head_kv, selected_blocks]
```
（`_vendor/tilelang-v0.1.15/examples/deepseek_nsa/example_tilelang_nsa_fwd.py:18-29`）

昇腾的 lightning indexer 则展示了“索引算子自己也是一个 GEMM + reduce + topk”：
Cube 侧算 `Q·Kᵀ` 并写入 `QK_RES`，Vector 侧按 `WEIGHTS` 加权、沿 group 维 reduce、
最后 `T.tile.topk` 选出 `TOP_K`
（`_vendor/tilelang-ascend-v0.1.2.000/examples/lightning_indexer/example_lightning_indexer.py:34-124`）。

**结构上只有两类新增的东西**：

1. **间接寻址**：KV block 的位置来自索引张量，而不是 `k * block_N` 这样的算术地址。
   这会让 `T.copy` 的源地址变成“查表 + 偏移”，TMA 的静态描述符不再直接适用。
2. **top-k / gather 这一类的原语**：它们本身不是 attention，但决定了 attention 能
   不能省下带宽。第 16 章和第 9 章的 `T.tile.topk` 一节会展开。

### 5.6 昇腾与 NVIDIA：瓶颈为什么不一样

把 4.4、4.5 两节合起来看，差异可以归结成三句话：

| | NVIDIA | 昇腾 |
| --- | --- | --- |
| 谁算矩阵 | Tensor Core，与 SIMT 单元同处一个线程块 | Cube 核，与 Vector 核**分开** |
| 分数怎么给 softmax | `T.gemm` 直接累加到 fragment（寄存器） | `L0C → GM workspace → UB`，**必须绕**（4.1.5.2 节） |
| 主要矛盾 | 异步搬运（`cp.async`/TMA）与计算的 overlap、非矩阵指令的占比 | **Cube 与 Vector 的核间流水**、跨核同步次数、scalar 指令下发 |

昇腾侧的调优文档把这件事说得很直：性能优化的核心原则是“掩盖短流水、减少气泡、
优化到单一 bound”，而核间优化的两步就是“找到最佳 `num_stages`”与“减少同步次数”
（`_vendor/tilelang-ascend-v0.1.2.000/examples/flash_attention/fa_opt/flash_attention_performance_optimization_zh.md`
第 4 节）。同一份文档给出一条**只适用于昇腾**的限制：

> **重要**：`T.pipelined` 不支持嵌套使用。

（同文件“5.2 使用限制”；措辞已按锁定版本核对，`code/06` 会把这一行定位并打印出来。
**未在 NPU 上运行验证。**）

厂商自报的优化收益（**厂商自报，作者未复现**）为：把 7 项优化全部做上以后，FA 达到
原生 AscendC 算子 **80%** 的性能；其中 T.pipelined 核间流水一项对应“原生 AscendC
的 60%”，减少核间同步次数对应 72%，用 axpy 合并指令对应 80%
（`_vendor/tilelang-ascend-v0.1.2.000/examples/flash_attention/fa_opt/bench_mark_zh.md`；
测试条件：`B=1, Q_N=12, KV_N=1, D=128, S=32K/64K/128K, block_size=128`）。

主干仓库里的昇腾 FA 报告了另一组数字：`SEQ_LEN=4096, D=128` 的 MHA 前向在
Ascend 950（dav-3510，bf16）上 **26.8 µs / 320.1 TFLOPS**，对照 Torch SDPA 的
**28.94 µs / 296.8 TFLOPS**；GQA 前向（`S1=8192, G=32, S2=8192, D=128`）则是
TileLang **3030 µs** 对 Torch **2864 µs**
（`_vendor/tilelang-v0.1.15/examples/ascend/flash_attention/README.md`；
**厂商自报，作者未复现**）。注意这两组数字**互相之间不可比**：形状、硬件、
测量口径都不同。本章引用它们的唯一目的是说明“瓶颈在哪”。

---

## 6. 本章小结

把这一页当作复习卡片。

**为什么要有 flash attention。** 朴素 attention 的瓶颈不是 `O(N²d)` 的乘加，而是
`O(N²)` 的中间矩阵在 HBM 上来回：`S` 与 `P` 各写一次读一次，合计约 `4N²` 个元素。
flash attention 把它们留在片上，结果**完全精确**。

**推导的主线只有两条性质。**

1. softmax 对指数平移不变：`exp(−m)` 在分子分母各出现一次、约掉；
2. 求和可分段：块的贡献可以直接加到前面的累加器上。

把“换了一个运行最大值”这件事写成公式，就是

```text
m_new = max(m_prev, M_blk)
c     = exp(m_prev − m_new)                     ← 永远 ≤ 1
ℓ     = c · ℓ      + rowsum(exp(S_blk − m_new))
O     = c · O      + exp(S_blk − m_new) · V_blk
O_out = O / ℓ
```

`m` 只负责让 `exp` 的指数不为正；它取什么值都不改变最终比值（2.4 节的归纳证明）。

**四级阶梯。** ① 朴素版物化 `N×N`（放不下也不划算）→ ② 分块 + online softmax
（`4N²` → 片上，但没有重叠）→ ③ 加 `T.Pipelined` 与 `FullRow`、寄存器复用
（形状写死）→ ④ 对齐官方实现（causal 循环裁剪、`past_len`、越界 mask、`O_shared`
epilogue）。

**三个参数各自绑着一件事。**

* `num_stages` 决定 `K_shared`/`V_shared` 的份数，也就决定了片上占用；
* `policy=T.GemmWarpPolicy.FullRow` 让后面的行内 `reduce_*` 尽量待在 warp 内；
* `scale` 折进 `log2(e)`，让 `exp` 变成 `exp2`。

**七个变体各自改了哪一行。** causal：循环上界 + `q ≥ k`；varlen：`cu_seqlens` +
越界 guard；GQA：`kv_head_idx = head_idx // groups`；decode：split 与 combine 两个
kernel；paged KV：`block_table` 间接寻址；sliding window：循环起点 + `check_inf`；
sink：epilogue 里把 `exp2(sink − m·scale)` 加进分母。

**四个坑。** ① `−∞ − (−∞) = NaN`（sliding window 必须 `check_inf`）；
② causal 的对角线、偏移、循环上界三处都要对；③ varlen 的 `max_seqlen` 是编译期常量、
短样本靠 guard；④ split-KV 必须按 `ℓ_s / Σℓ` 加权合并，低精度累加会真的伤正确性
（昇腾 backward 的 dQ 相对 L2 误差约 1.024%，厂商自报）。

**两台机器。** NVIDIA 上分数直接落在 fragment；昇腾上 `QKᵀ` 必须绕
`L0C → workspace → UB`，所以瓶颈从“异步搬运的 overlap”变成“Cube 与 Vector 的
核间流水与同步次数”。

## 延伸阅读

下面每条都指向本章 `references.md` 里的条目 id。

* **论文**：`R1`（FA1，摘要给出“IO-aware exact attention”的定位）、`R2`（FA2，
  并行划分与 warp 划分的动机）、`R3`（FA3，Hopper 上的 warp specialization 与低精度）。
  **注意**：corpus 里这三篇只有标题与摘要，见 `references.md` 的“待补资料”。
* **官方例子（本章的主角）**：`S1`（`example_mha_fwd_bhsd.py`，3.4 节逐行对照的对象）、
  `S2`（`README.md`，那篇把整个 kernel 贴出来的讲解）、`S3`（varlen + GQA）、
  `S4`（decode / split-KV）、`S5`（sink + sliding window + `check_inf`）、
  `S6`（block-causal）、`S7`（sm100 / TMEM）。
* **API 出处**：`S8`（`T.reduce_max` 的 `clear` 语义）、`S9`（`T.gemm` 的
  `policy` 与异步接口的关系）、`S10`（`GemmWarpPolicy`）、`S11`
  （`T.alloc_tmem` 的约束）、`S12`（`T.infinity` 落到 IEEE `+∞`）。
* **昇腾**：`A1`（适配器版 `flash_attn_bhsd.py`，workspace + 跨核 flag 的写法）、
  `A2`（`mha_sink_fwd_bhsd.py` 里关于 `m*` 的长注释）、`A3`（`gqa_fwd_varlen.py`）、
  `A4`（`lightning_indexer`）、`A5`（Cube/Vector 调优文档）、`A6`（FA benchmark）、
  `A7`（编程指南 4.1.5.2 workspace 消除）、`A8`（主干仓库的昇腾 FA 与它的性能表）、
  `A10`（workspace 消除 pass 的实现与那张改写表）、`A11`（paged KV 的昇腾实现）。
* **稀疏与索引**：`S13`（NSA 前向）、`S14`（mInference / vertical-slash）、
  `S15`（HISA 两级索引）。它们是第 16 章的入口，本章只讲“结构上多了什么”。
* **代码**：`code/README.md` 里有六个示例的验证级别总表；`code/02`、`code/03`、
  `code/04` 是本机可运行的推理验证（CPU 逻辑验证）。
