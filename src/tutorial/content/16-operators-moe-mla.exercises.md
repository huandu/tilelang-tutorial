# 第 16 章 习题

说明：

* 难度标注为 `★`（基础）/ `★★`（需要综合本章内容）/ `★★★`（推理链条较长或需要动手）。
* 「需要硬件」一栏：`否` 表示一台能跑 Python 的机器即可；`是（GPU）` /
  `是（NPU）` 表示必须用真实加速器。凡标「可选」的硬件题，本机环境
  （macOS、无 GPU/NPU）做不了，但题目本身可以用纸笔或纯 Python 完成。
* 所有题目只依赖第 16 章及更早的知识；涉及后续章节的地方会显式提示。
* 账本口径与第 02 章一致：GEMM 记 `2MNK` FLOP 与 `b(MK+KN+MN)` 字节，
  纯搬运算子只记字节。

---

## E1 概念题 ★ 不需要硬件

MoE 的一层在执行时间上可以拆成四段。

1. 按顺序写出这四段的名字，并说明每一段是「矩阵乘」还是「纯搬运/选择」。
2. 写出 permute 与 combine 各自的字节公式（用 `T`、`k`、`d`、`b` 表示），
   并说明为什么它们的 FLOP 是 0。
3. router 的 GEMM 形状是什么？为什么它「不吃算力」但仍可能成为瓶颈？
4. 本章说「MoE 的瓶颈常常不在矩阵乘」。请给出**一个反例**：什么样的形状下
   矩阵乘就是主角？用本章的账本数字支持你的回答。

---

## E2 概念题 ★★ 不需要硬件

单个专家对 `n` 个 token 做 `gate(d→d_e) + up(d→d_e) + down(d_e→d)` 三个 GEMM。

1. 写出这个专家自己的 FLOP 与字节（下界）公式，并给出算术强度 `AI(n)`。
2. 取 `d=7168`、`d_e=2048`、`b=2`，求 `n` 从多大开始该专家的 GEMM 变成
   compute-bound（用 H100 公开规格的平衡点 ≈333 FLOP/Byte）。写出推导，
   并说明为什么这个 `n` 由「权重字节」而不是「激活字节」决定。
3. V3.2 有 256 个路由专家、每 token 激活 8 个。如果这一步只有 64 个 token，
   平均每个专家分到多少 token？按第 2 问的结论，这一步的专家 GEMM 是
   compute-bound 还是 memory-bound？
4. 这个判断对「该不该把 gate 和 up 融成一个 dual GEMM」有什么影响？

（可用 `code/01-cost-ledger/decode_step_ledger.py` 的“单专家”一节复算。）

---

## E3 代码阅读题 ★★ 不需要硬件

下面是 `_vendor/tilelang-v0.1.15/examples/fusedmoe/example_fusedmoe_tilelang.py`
里 routed kernel 的核心片段（已省略上下文）：

```python
M = math.ceil(group_sum / block_token) + group_count        # 第 120 行
...
with T.Kernel(M, T.ceildiv(dexpert, block_dexpert), threads=threads) as (bx, by):
    m_start_padded = bx * block_token
    cur_group_idx = group_idx_for_bx[bx]                     # 第 157 行
    cur_group_size = group_sizes[cur_group_idx]
    m_start = m_start_padded - group_padded_offsets[cur_group_idx] + group_offsets[cur_group_idx]
    actual_rows = T.max(0, T.min(block_token, cur_group_size - (m_start_padded - group_padded_offsets[cur_group_idx])))
    ...
    for i, j in T.Parallel(block_token, block_dexpert):
        if i < actual_rows:                                   # 第 189 行
            up_logits[m_start + i, by * block_dexpert + j] = up_logits_local[i, j]
```

回答：

1. `M` 为什么写成 `ceil(group_sum / block_token) + group_count`，而不是
   `ceil(group_sum / block_token)`？
2. `group_idx_for_bx` 是一张什么表？它是在 host 侧还是 device 侧算出来的？
   为什么要用它而不是在 kernel 里循环比较 `group_padded_offsets`？
3. `m_start` 与 `m_start_padded` 的差是什么？为什么两者都要有？
4. 如果没有 `if i < actual_rows` 这个判断，会在什么情况下产生错误结果？
   会报错吗？

---

## E4 代码阅读题 ★★ 不需要硬件

读 `_vendor/tilelang-v0.1.15/examples/grouped_gemm/example_grouped_gemm_fwd.py`：

```python
total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)   # 第 64 行
...
in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]                         # 第 87 行
cur_batch_size = batch_sizes[cur_batch_idx]
m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
actual_rows = T.max(0, T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded))
```

回答：

1. `batch_offsets`、`batch_padded_offsets`、`batch_sizes` 三者分别是什么？
   为什么不能只用其中两个？
2. 同目录的 `example_grouped_gemm_fwd_ptr.py` 用指针表（`T.make_tensor`）代替
   拼接，但把 `copy_num_stages` 固定成 1。文件头注释给了什么理由？
3. 如果某个 batch（专家）的 `size` 是 0，这段代码会怎么处理？会浪费一个 block 吗？
4. 本章的 `code/02-moe-routing/routing_sim.py` 里，`padded_offsets` 与
   `block_metadata` 是怎么生成的？它和这里的 `batch_padded_offsets` 是什么关系？

---

## E5 概念题 ★★ 不需要硬件

MLA 的核心是「把上投影吸收掉」。设潜向量 `c` 的维度是 `d_c`，第 `h` 个 head 的
上投影矩阵是 `W_h`（形状 `d_h × d_c`），query 是先投影到 `d_h` 再参与打分的
`q_h`，value 是 `v = W_h c`。

1. 推导 `q_h · (W_h c) = (W_h^T q_h) · c`，并说明这一步为什么能让 kernel
   **只和 `c` 打交道**。
2. 对 `o = Σ_s a_s v_s` 做同样的推导，说明为什么「输出还在潜空间、需要最后再乘
   一次上投影」。
3. 在 `examples/deepseek_mla/example_mla_decode.py` 里，`Q` 是
   `[batch, heads, dim]`、`KV` 是 `[batch, seqlen_kv, 1, dim]`。用第 1、2 问的
   结论解释：这里的 `dim` 是多少？为什么 `kv_head_num` 是 1？
4. RoPE 分支（`Q_pe` / `K_pe`）为什么**不**做低秩压缩？它在 kernel 里如何与
   主分支合并（看第 65-66 行）？

---

## E6 概念题 ★★ 不需要硬件

取 `H_q = 128`、`d_c = 512`、`d_r = 64`、`b = 2`。

1. 写出 MHA / GQA 与 MLA 的「每 token 每层 KV 字节」公式，并各算一个值。
2. 写出 MHA/GQA 的 decode 算术强度 `AI = 2H_q/(H_kv·b)`，并解释为什么它与
   序列长度 `L` 无关。
3. 写出 MLA 的 decode 算术强度公式并算值。它为什么能比 GQA（`H_kv=8`）高一个
   数量级？
4. 已知 H100 公开规格的平衡点约 333 FLOP/Byte。解出「MLA decode 要在多少个
   query head 时才跨过去」。128 个 head 的结果告诉你什么？这与 3.4 节官方的
   「fp8 没有比 bf16 更快，因为受限于 softmax / shared memory」是否矛盾？

---

## E7 代码阅读题 ★★ 不需要硬件

读 `_vendor/tilelang-v0.1.15/examples/deepseek_mla/example_mla_decode.py`：

```python
glse = T.alloc_global([batch, heads, num_split], dtype)                 # 第 30 行
Output_partial = T.alloc_global([batch, heads, num_split, dim], dtype)  # 第 31 行
...
T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)   # 第 65 行
T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)  # 第 66 行
```

回答：

1. 第 65、66 行是两次 `T.gemm`，但它们写的是**同一个** `acc_s`。这样做的语义是
   什么？为什么不写成一个 K=576 的 GEMM？
2. `policy=T.GemmWarpPolicy.FullCol` 是什么意思？它和「`acc_o` 太大、
   要按 `dim` 切开」有什么关系？
3. `glse` 与 `Output_partial` 各存什么？第二个 kernel（第 96 行起）做了什么？
   为什么用两个 kernel 而不是原子加？
4. 假设 `num_split=1`，第 30、31 行的两个 global 缓冲还有用吗？官方为什么仍
   保留 `main_split` 与 `main_no_split` 两个版本？

---

## E8 概念题 ★★ 不需要硬件

1. 列出至少三条「MLA 对 kernel 友好」的理由，每条给出一个可指认的依据
   （公式或源码行）。
2. 列出至少三条「MLA 对 kernel 不友好」的理由，并说明官方文档给出的对策。
3. `block_H` 能不能从 64 降到 32 来减小 `acc_o`？为什么？
4. 为什么 fp8 版本要求 KV 是 K-major 布局？如果不是会怎样（给出量化后果）？

---

## E9 代码阅读题 ★ 不需要硬件

读 `_vendor/tilelang-v0.1.15/examples/deepseek_v32/topk_selector.py`：

```python
def convert_to_uint16(x):
    hval = T.cast(x, T.float16)
    bits_uint = T.reinterpret(hval, T.uint16)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8
```

回答：

1. 这个函数把 float 变成什么？为什么要保证「大的 float 映射成大的整数」？
2. 为什么只取高 8 位（`>> 8`）？这样做对 `topk` 的精确性有什么影响，
   官方又是怎么补救的？
3. top-k selector 输出的是**位置**还是**值**？为什么不能直接输出值？
4. 本章说「top-k 是排序/直方图问题，不是 GEMM」。请从这段代码里找出至少两个
   支持这个说法的证据。

---

## E10 概念题 ★★★ 不需要硬件

设 indexer 有 `I` 个 index head、每个 head 维 `di`，并在整条序列 `L` 上打分；
被选中的 attention 有 `H_q` 个 query head、每个选中位置花 `2·d_c + d_r` 个 MAC。

1. 写出 indexer 的 FLOP（每个 query token）与被选中的 attention 的 FLOP，
   并求出两者相等时的 `topk`（用 `L` 表示）。
2. 取 V3.2 的参数（`I=64`、`di=128`、`H_q=128`、`d_c=512`、`d_r=64`、
   `index_topk=2048`），算 `L = 131072` 时的比值。这说明什么？
3. 换成**字节**口径：分别算 dense KV、indexer KV、sparse KV 的每 token 读数，
   并给出「稀疏/稠密」的比值。
4. 综合第 2、3 问，用一句话说明「稀疏 attention 省的是什么、不省什么」。

（可用 `code/05-sparse-index-map/sparse_index_map.py` 复算。）

---

## E11 动手题 ★★ 不需要硬件

运行本章的三个账本：

```bash
cd chapters/16-operators-moe-mla/code/01-cost-ledger && python3 decode_step_ledger.py
cd ../02-moe-routing && python3 routing_sim.py
cd ../03-mla-kv && python3 kv_budget.py
```

回答：

1. `code/01` 里三组 MoE 形状的 grouped GEMM 算术强度分别是多少？哪一组
   compute-bound、哪一组 memory-bound？
2. `code/01` 的「单专家」表里，`tokens_for_expert` 到多少才跨过平衡点？
   和你在 E2 手算的结果一致吗？
3. `code/02` 输出的「最热专家 / 平均专家」与「padding 浪费」是多少？把
   `--block-m` 从 32 改成 64，这两个数字怎么变？为什么？
4. `code/03` 里 MLA 与 MHA 的 KV 比值是多少？它与 `--seq` 有没有关系？

---

## E12 动手题 ★★★ 需要 GPU（可选）

在一台有 NVIDIA GPU（SM90 或更新）的机器上：

1. 运行 `examples/deepseek_mla/example_mla_decode.py`，记录它打印的
   `Latency` 与 `TFlops`，并写清 `batch` / `kv_ctx` / `dim` / `pe_dim` /
   `block_N` / `block_H` / `num_split`。
2. 把 `num_split` 从 1 调到 4、8、16，画出「split 数–延迟」曲线，指出拐点，
   并解释拐点之后为什么不再变快。
3. 用第 14 章 7.1 节的清单写全口径（dtype、shape、epilogue、计时后端、预热、
   重复次数、正确性判据），并说明为什么缺一项就不算对比。
4. 本章 3.5 节说 `acc_o` 吃掉 128 个寄存器/线程。请在真机上用 profiler 看
   register per thread，与这个下界比较，并解释差异来自哪里。

> 如果你没有 GPU，本题只做第 3 问的「口径设计」，结果写成「未运行」。

---

## E13 动手题 ★★★ 需要 NPU（可选）

读 `_vendor/tilelang-ascend-v0.1.2.000/examples/` 下的四个目录：
`fusedmoe/`、`moe_token_permute/`、`dispatch_combine/`、`deepseek_v4/`。

1. 把 `fusedmoe/example_fusedmoe.py` 里的 `single_gemm_kernel`、
   `dual_gemm_kernel`、`grouped_gemm_kernel`、`grouped_dual_gemm_kernel`
   与本章第 2 节的四段一一对应，填一张「段 → kernel」表。
2. 找出该文件里三处显式的容量/资源约束（注解、注释或常量），并说明它们分别
   约束了什么。
3. `dispatch_combine_shmem.py` 需要什么样的硬件与软件前置条件？为什么它
   「不支持 graph 模式」这件事对推理部署有影响？
4. 设计一份「对齐手写基线」的实验清单：基线选什么（AscendC 还是 CATLASS）、
   什么 shape、什么 dtype、什么测量协议（本题**不要运行**，只交方案，
   不超过 15 行）。
