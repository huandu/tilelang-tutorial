# 第 15 章 习题

说明：

* 难度：`★`（基础，只需读懂本章）/ `★★`（需要综合本章多节）/ `★★★`（需要动手或
  深入读源码）。
* 「需要硬件」一栏里，`否` 表示只需要一台能跑 Python 的机器。本章**没有**强制要求
  硬件的题目；所有必须在加速器上完成的实验都标为「可选」。
* 所有题目只依赖第 1–15 章的知识。涉及第 16 章的内容（MoE / MLA / 稀疏算子的完整
  展开）会显式提示。
* 本章代码示例见 `code/`；题目里提到的 `code/0X` 都指本章的示例目录。

---

## E1 概念题 ★ 不需要硬件

请用三句话回答：

1. 朴素 attention 的瓶颈为什么不是 `O(N²d)` 的乘加？
2. FlashAttention 说自己「exact」（精确）是什么意思？它到底近似了什么、没有近似什么？
3. 为什么把中间结果留在片上之后，`K`、`V` 反而可能被**读得更多**？

---

## E2 概念题 ★ 不需要硬件

写出分块 online softmax 的递推式（四个式子：`m_new`、`c`、`ℓ`、`O`），并且：

1. 说明 `c` 的取值范围以及为什么；
2. 用一句话说明「为什么 `exp(−m)` 可以约掉」；
3. 如果 kappa 把运行最大值换成一个**常量** `m ≡ 0`，结果在数学上还对不对？
   在数值上会发生什么？

---

## E3 概念题 ★★ 不需要硬件

关于 masking 与数值：

1. 在 v0.1.15 里，`T.infinity("float32")` 得到的是 IEEE `+∞` 还是 `float32` 的最大
   有限值？它的实现在哪个文件？（给出路径与行号区间。）
2. 于是 `-T.infinity(acc_s.dtype)` 作为 mask 值时，`exp2(x − m)` 在什么条件下会得到 0？
3. 一个 query 行在前若干块里一个可见 key 都没有，会发生什么？`check_inf` 修好了什么？
4. `example_gqa_fwd_varlen.py` 用的是 `-1e9` 而不是 `-T.infinity(...)`。请说明这两种
   取值在「整行全被 mask」这一极端情形下的行为差异。

---

## E4 概念题 ★★ 不需要硬件

设 `Sq = Skv = 4096`、`d = 128`、fp16（2 字节/元素），只考虑一个 `(batch, head)`。

1. 按本章 1.2 节的记账法，朴素版的 HBM 元素流量是多少？其中 `N²` 项占多少？
2. 分块融合版（`block_m = block_n = 128`）的流量是多少？请逐項列出。
3. 两者相差几倍？如果换成 `Sq = 1, Skv = 4096`，结论会变成什么？为什么？
4. 你可以用 `code/01-traffic-model/traffic_model.py` 验证上面三个答案。请说明这个模型
   **高估**了什么、**低估**了什么。

---

## E5 代码阅读题 ★ 不需要硬件

下面这段来自官方例子（`example_mha_fwd_bhsd.py`）：

```python
T.copy(scores_max, scores_max_prev)
T.fill(scores_max, -T.infinity(accum_dtype))
T.reduce_max(acc_s, scores_max, dim=1, clear=False)
for i in T.Parallel(block_M):
    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
```

1. `T.reduce_max` 的 `clear` 参数语义是什么？（指出它在锁定版本源码里的位置。）
2. 既然 `clear=True` 会自己把输出初始化成 `-inf`，为什么这里写成「先 `T.fill` 再
   `clear=False`」？这两种写法在语义上等价吗？
3. 最后那个 `T.max` 在公式里对应哪一步？

---

## E6 代码阅读题 ★★ 不需要硬件

下面是一个 causal attention 的循环上界计算，它有一个**真实的 bug**：

```python
past_len = seq_kv - seq_q
loop_range = T.ceildiv(seq_kv, block_N)          # ← 这里
for k in T.Pipelined(loop_range, num_stages=num_stages):
    T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
    for i, j in T.Parallel(block_M, block_N):
        q_idx = bx * block_M + i + past_len
        k_idx = k * block_N + j
        acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
    ...
```

1. 这段代码**结果正确吗**？如果不正确，错在哪一行？
2. 如果结果仍然正确但很慢，慢在哪？请给出官方的写法。
3. 把 `q_idx >= k_idx` 写成 `q_idx > k_idx` 会得到什么？请描述它在 `is_causal=True`
   且 `past_len = 0` 时对输出的影响（提示：思考对角线元素）。

---

## E7 代码阅读题 ★★ 不需要硬件

下面是 flash decoding 的 combine kernel 的核心（`example_mha_inference.py`）：

```python
T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
for k in T.Pipelined(num_split):
    T.copy(lse_local[k, :], lse_local_split)
    for i in T.Parallel(block_M):
        lse_logsum_local[i] += T.exp2(lse_local_split[i] - lse_max_local[i])
for i in T.Parallel(block_M):
    lse_logsum_local[i] = T.log2(lse_logsum_local[i]) + lse_max_local[i]
for k in T.Pipelined(num_split, num_stages=2):
    T.copy(Output_partial[...k...], po_local)
    ...
    scale_local[i] = T.exp2(lse_local_split[i] - lse_logsum_local[i])
    o_accum_local[i, j] += po_local[i, j] * scale_local[i]
```

1. `lse_logsum_local` 在数学上是什么量？
2. `scale_local` 在数学上是什么？为什么它必须等于 `ℓ_s / Σℓ`？
3. 如果偷懒写成 `o_accum_local += po_local`（直接相加），结果会怎样？请给出一个定性
   的理由，并说明 `code/03-split-kv-combine/` 里的实验验证了这一点。
4. 为什么要先减去 `lse_max_local`，而不是直接 `exp2(lse_local_split)`？

---

## E8 代码阅读题 ★★ 不需要硬件

下面摘自昇腾适配器版本的 flash attention（`flash_attn_bhsd.py`）：

```python
with T.Scope("C"):
    ... T.gemm_v0(q_l1, k_l1, acc_s_l0c, transpose_B=True, init=True)
    T.copy(acc_s_l0c, workspace_1[cid, :, :])
    T.set_cross_flag("FIX", 0)
    T.wait_cross_flag(1)
    ...
with T.Scope("V"):
    T.wait_cross_flag(0)
    T.copy(workspace_1[cid, ...], acc_s_ub_)
    ... softmax on acc_s_ub ...
    T.copy(acc_s_half, workspace_2[cid, ...])
    T.set_cross_flag("MTE3", 1)
```

1. `T.Scope("C")` 与 `T.Scope("V")` 分别对应什么硬件？它们为什么不能直接共享片上缓冲？
2. 为什么分数要经过 `workspace_1`？请引用锁定版本编程指南里的那条依据。
3. `workspace_2` / `workspace_3` 各装的是什么？
4. 一个 KV 块的处理里，Cube 与 Vector 之间大约要同步几次？为什么「减少同步次数」
   会出现在调优文档的收益清单里（见 `fa_opt/bench_mark_zh.md`）？

---

## E9 动手题 ★★ 不需要硬件

用 numpy（或你熟悉的语言里的数组库）实现一个**分块 online softmax**，并让它在
`code/02-online-softmax/online_softmax.py` 的基础上多满足一个要求：

1. 每处理完一块，打印 `(块号, m, c, ℓ, O/ℓ)`——注意最后一项是**当前的归一化输出**，
   它会随着块的推进而变化；
2. 证明：当最后一块处理完时，`O/ℓ` 与「先算完整 softmax 再乘 V」逐元素相等（误差
   `< 1e-12`）；
3. 把 key 的顺序**倒过来**再跑一次，观察中间的 `m`/`c`/`O/ℓ` 序列完全不同，但最终
   结果相同。请用一句话解释为什么。

---

## E10 动手题 ★★ 不需要硬件

用 `code/01-traffic-model/traffic_model.py` 的模型做一次参数扫描：

1. 固定 `Sq = Skv = 8192`、`d = 128`，让 `block_m` 取
   `32, 64, 128, 256, 512`，把「tiled 的总流量」画成一张表；
2. 找出这个模型里 tiled 流量最小的 `block_m`，并解释为什么它会随 `block_m` 单调下降
   （提示：看 K/V 重读项的表达式）；
3. 然后回答：既然模型说「`block_m` 越大越好」，为什么真实 kernel 不把
   `block_m` 一路加到 512 或 1024？至少给出两个理由（提示：片上容量、warp 划分、
   尾部浪费、layout）。

---

## E11 概念题 ★★ 不需要硬件

sliding window + attention sink 的 epilogue 是：

```python
for i in T.Parallel(block_M):
    logsum[i] += T.exp2(sinks[i] * 1.44269504 - scores_max[i] * scale)
for i, j in T.Parallel(block_M, dim):
    acc_o[i, j] /= logsum[i]
```

（`example_mha_sink_fwd_bhsd.py:125-128`）

1. 这个 sink 在 softmax 里相当于什么？为什么它只影响分母、不影响分子？
2. 写出它在数学上的等价形式（即：把 sink 当成一个额外的 key 之后，完整 softmax
   长什么样）。
3. 昇腾版 `mha_sink_fwd_bhsd.py` 的注释里说要用 `m* = max(sink, m_i_final)` 而不是
   `m_i_final`。请用你上面写出的公式解释：为什么用 `m_i_final` 时
   `exp(sink − m_i_final)` 可能溢出？

---

## E12 动手题 ★★★ 不需要硬件（需要能读源码的耐心）

把 `code/04-block-flash-attention/block_flash_attention.py` 当作**规格说明**，然后
完成下面两件事：

1. 给 `block_flash_attention()` 增加一个 `softcap` 参数：在 softmax 之前对分数做
   `z = softcap * tanh(z / softcap)`（这是 Gemma-2 一类的 logit soft-capping）。
   同时修改参考实现，并让两者在 `softcap ∈ {None, 50, 30}` 下都一致。
2. 说明：为什么 soft-capping **不能**简单地放进 `T.gemm` 之后的那一行
   `acc_s[i,j] *= scale` 里？（提示：`exp2` 的输入与 `scores_max` 的取值必须来自
   **同一个**变换后的分数。）

不需要真的写 TileLang；本机用 numpy 验证逻辑即可。

---

## E13 代码阅读题 ★★★ 不需要硬件（可选：在真机上验证）

官方 varlen 例子里有这样一段：

```python
offset = kv_current_seqlen - q_current_seqlen  # always align on the right
max_visible_k_idx = offset + (bx + 1) * block_M
loop_range = (T.min(T.ceildiv(max_visible_k_idx, block_N), T.ceildiv(kv_current_seqlen, block_N))
              if is_causal else T.ceildiv(kv_current_seqlen, block_N))
```

1. `offset` 为什么是 `kv_current_seqlen - q_current_seqlen`？它体现了什么语义？
2. 为什么 `loop_range` 里要取 `T.min(...)`，两个上界分别防的是什么？
3. 该例子的 mask 用的是 `-1e9`，请构造一个小例子（长度 3 的 query、长度 3 的 key、
   `is_causal=True`），说明如果把 `-1e9` 换成 `-T.infinity("float32")`，哪一行会最先
   出问题。（可以用 `code/02` 或 `code/04` 的代码改造验证。）

**可选（需要硬件）**：在 NVIDIA GPU 上跑
`python examples/flash_attention/example_gqa_fwd_varlen.py --q_seqlen 2048 --k_seqlen 2048 --groups 16`
（需要 `flash_attn` 包做对照），比较 `is_causal` 打开与关闭时的一致性检查是否都通过。

---

## E14 动手题 ★★★ 可选（需要 NVIDIA GPU）

在真机上做一次「配置 → 现象」的对照实验（**只在有卡的机器上做**）：

1. 用 `examples/flash_attention/example_mha_fwd_bhsd.py` 跑
   `--seq_q 4096 --seq_kv 4096 --dim 128 --heads 16 --is_causal`；
2. 把 `block_M`、`block_N`、`num_stages` 改成三组：
   `(64, 64, 1)`、`(128, 128, 2)`、`(128, 128, 4)`；
3. 记录每组是否**编译成功**、是否**通过正确性检查**、以及 `do_bench` 的延迟；
4. 用第 11 章 4.5 节的片上占用公式解释「哪一组会因为 shared memory 超限而失败」，
   并解释「为什么 `num_stages=1` 的那一组最慢」。

报告要求：写明 GPU 型号、CUDA 版本、TileLang commit，以及你实际观察到的数字。
**不要引用本章任何数字**——本章没有实测数字。
