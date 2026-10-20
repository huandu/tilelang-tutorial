# 第 18 章 习题

说明：

* 难度标注为 `★`（基础）/ `★★`（需要综合本章内容）/ `★★★`（推理链条较长或要动手）。
* 「需要硬件」一栏中，`否` 表示只需要一台能跑 Python 的机器；`是（GPU）` /
  `是（NPU）` 表示必须用真实加速器。凡标注「可选」的硬件实验，本机环境
  （macOS、无 GPU/NPU）做不了，但题目本身可以用纸笔或纯 Python 完成。
* 所有题目只依赖第 18 章及更早的知识；涉及后续章节的地方会显式提示。

---

## E1 概念题 ★ 不需要硬件

回答「为什么 prefill 与 decode 的优化目标不同」：

1. 写出 `M ≪ N, K` 时 GEMM 算术强度的简化式，并说明推导里哪一项被省略了、
   为什么可以省略。
2. 用 bf16 与 H100 的公开平衡点 333 FLOP/Byte 算翻转点 `M*`，写出过程。
3. 第 14 章说方形 GEMM 的翻转点约 `10³`，本章说约 375。两者为什么不同？
   报这个数字时必须一起报什么？

---

## E2 概念题 ★ 不需要硬件

Qwen2.5-7B 的模型卡给出：28 层、28 个 query head、4 个 KV head、
上下文 131072、参数量 7.61B（非嵌入 6.53B）。

1. 写出「每 token 每层 KV 字节数」的公式，并算出 bf16 下的值。
2. 算出 context = 131072 时一条序列的 KV cache 大小（GiB）。
3. 如果把 GQA 换成 MHA（28 个 KV head），同一个 context 下是多少？
4. 写出 decode attention 的算术强度公式，并比较 GQA 与 MHA（bf16）
   两种情况下各是多少。
5. 用一个句子说明「省显存」与「算术强度」在这里为什么是一枚硬币的两面。

---

## E3 概念题 ★★ 不需要硬件

关于 paged KV：

1. 分页解决的是什么问题？和「按最大长度连续预留」相比，浪费从哪里转移到了哪里？
2. 它对 attention kernel 提出的新要求是什么？为什么这会与 TMA 的静态描述符冲突？
3. `block_size` 从 16 提到 128，会分别改善和恶化什么？用 Qwen2.5-7B 的
   块账给出两组数字。

---

## E4 概念题 ★★ 不需要硬件

关于 continuous batching：

1. 用一句话说明它为什么能同时提高吞吐、降低平均延迟（在槽位没被占满时）。
2. 它给 kernel 带来了哪三件事的新要求？（提示：varlen、paged KV、
   prefill/decode 混步。）
3. 为什么 `max_seqlen` 必须是编译期常量？这会带来什么运维上的现象？
4. 如果并发请求数超过槽位数，吞吐和延迟会怎么变？请说明你判断的依据。

---

## E5 代码阅读题 ★★ 不需要硬件

读下面这段来自
`_vendor/tilelang-v0.1.15/examples/flash_decoding/example_gqa_decode.py` 的片段：

```python
kv_group_num = heads // groups
valid_block_H = min(block_H, kv_group_num)
...
with T.Kernel(batch, heads // valid_block_H, num_split, threads=threads) as (bx, by, bz):
    ...
    cur_kv_head = hid // (kv_group_num // valid_block_H)
```

1. `heads = 28`、`groups = 4`、`block_H = 64` 时，`kv_group_num` 与
   `valid_block_H` 各是多少？
2. `cur_kv_head` 这一行的作用是什么？如果它算错，会出现什么现象——崩溃，
   还是「结果看起来对但数值偏」？
3. 三个 grid 维度分别扫的是什么？为什么第二维是 `heads // valid_block_H`
   而不是 `heads`？

---

## E6 代码阅读题 ★★ 不需要硬件

读下面这段来自
`_vendor/tilelang-v0.1.15/examples/flash_decoding/example_mha_inference.py`
的 combine 段：

```python
T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
for k in T.Pipelined(num_split):
    ...
    lse_logsum_local[i] += T.exp2(lse_local_split[i] - lse_max_local[i])
lse_logsum_local[i] = T.log2(lse_logsum_local[i]) + lse_max_local[i]
...
scale_local[i] = T.exp2(lse_local_split[i] - lse_logsum_local[i])
o_accum_local[i, j] += po_local[i, j] * scale_local[i]
```

1. 为什么合并权重是 `exp2(lse_s − logsumexp(lse))`，而不是简单地把各段输出相加？
   写出对应的数学式。
2. 为什么要在累加前先减 `lse_max`？这和 online softmax 里的重标度是不是同一个
   技巧？
3. 这段代码在数值上有什么风险？请引用本章提到的某个实测误差来说明。

---

## E7 代码阅读题 ★★ 不需要硬件

读下面这段来自
`_vendor/tilelang-v0.1.15/examples/fusedmoe/example_fusedmoe_tilelang.py`
的片段：

```python
group_padded_offsets = [0 for _ in range(len(group_sizes))]
for i in range(1, len(group_sizes)):
    group_padded_offsets[i] = group_padded_offsets[i - 1] + math.ceil(
        (counts[i - 1] + 1) / self.padding_M) * self.padding_M
...
group_idx_for_bx[bx] = i   # 满足 m_start_padded >= group_padded_offsets[i] 的最后一个 i
```

与 kernel 侧：

```python
cur_group_idx = group_idx_for_bx[bx]
m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))
```

1. 为什么 `group_padded_offsets` 的递推里用 `counts[i-1] + 1` 而不是 `counts[i-1]`？
   多出来的那一行起了什么作用？
2. `actual_rows` 里的 `T.max(0, ...)` 是防御什么情况的？
3. 如果 host 侧的 `group_sizes` 与 kernel 拿到的 `X` 的行顺序不一致，
   会出现什么现象？

---

## E8 代码阅读题 ★★★ 不需要硬件

下面是从本章 `code/08-ascend-aclgraph/aclgraph_decode_block.py` 摘的一段：

```python
with T.Kernel(n_blocks, is_npu=True) as (cid, vid):
    row_x = cid * block_m + vid * rows_per_vec
    ...
    T.reduce_sum(sum_square_ub, rms_ub, dim=-1)
```

1. `is_npu=True` 这个参数在主干的 `_vendor/tilelang-v0.1.15` 里能不能用？
   请说明你是怎么验证的，以及主干写法的差别。
2. `(cid, vid)` 两个索引分别是什么？为什么这里要按 `vec_num` 把
   `block_m` 行切成两半？
3. 如果把这个 kernel 放进 `torch.npu.graph(g)` 里 capture，需要保证什么？

---

## E9 动手题 ★★ 需要一台能跑 Python 的机器（不需要加速器）

用 `code/01-prefill-decode-roofline/prefill_decode_roofline.py` 的公式，
自己写一个小脚本回答：

1. 把 dtype 从 bf16 换成 fp8（1 字节）后，翻转点 `M*` 变成多少？
2. 权重从 bf16 换成 fp8 后，「读一遍权重」的时间下界变成多少毫秒？
   单请求速度上界变成多少 tok/s？
3. 这两件事为什么会让「量化对 decode 的收益」远大于「量化对 prefill 的收益」？
   用一句话解释。

（可以先用脚本自带的 `--json` 拿到形状参数，再手改公式复算。）

---

## E10 动手题 ★★ 需要一台能跑 Python 的机器（不需要加速器）

用 `code/03-decode-step-share/decode_step_share.py` 做一次敏感度分析：

1. 固定 context = 8192，让 batch 从 1 增到 128，画出（或列表输出）
   attention 占比与单步下界时间的变化；
2. 解释为什么单步下界时间在 batch 很大时会明显变长，而权重类算子的
   **绝对**时间并没有变；
3. 如果用这份账本来决定「该不该加 batch」，你还缺哪些信息？至少写三条。

---

## E11 动手题 ★★★ 可选硬件（GPU）

在装有 NVIDIA GPU 与 TileLang v0.1.15 的机器上：

1. 跑通 `_vendor/tilelang-v0.1.15/examples/flash_decoding/example_mha_inference.py`，
   记录它的正确性判据与它打印的延迟；
2. 把 `num_split` 从 4 改成 1 与 8，各跑一次，记录延迟；
3. 用第 7 章的口径写出你的对比表（必须包含：卡型、`seqlen_kv`、`dim`、
   `block_M`、`block_N`、`num_split`、`num_stages`、预热方式、取数方式）；
4. 说明你观察到的趋势与本章 4.1 节的解释是否一致。

（本机没有 GPU，本题无法在写作环境完成；**如果没跑过，就不要在报告里写任何
延迟数字**。）

---

## E12 动手题 ★★★ 可选硬件（NPU）

在装有昇腾 NPU 与 TileLang-Ascend 适配器的机器上：

1. 用 `code/08-ascend-aclgraph/aclgraph_decode_block.py` 跑通正确性校验；
2. 用 host 日志确认 ACLGraph 是否真的生效（写出你用的环境变量与搜索关键字）；
3. 用 `msprof op` 采集一次，说明你看到了哪几张图、它们分别回答什么问题；
4. 把 `torch.npu.graph` 换成 eager 再跑一遍，比较两者的**端到端**时间，
   并解释差异来自哪里。

（本机没有 NPU，本题无法在写作环境完成。若只做了静态核对，
请在报告里明确写「仅静态核对」。）
