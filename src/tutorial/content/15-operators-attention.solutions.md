# 第 15 章 习题答案

每题给出**解题思路 + 参考答案 + 常见错误**。答案里的行号一律指锁定版本：
TileLang `v0.1.15`（`_vendor/tilelang-v0.1.15/`）与 TileLang-Ascend
`v0.1.2.000-release`（`_vendor/tilelang-ascend-v0.1.2.000/`）。

---

## E1 概念题 ★

**思路**：把「计算量」和「访存量」分开数，再看两者的增长阶数。

**参考答案**

1. 因为 attention 的算术强度很低：计算是 `O(N²d)`，但朴素实现要往 HBM 写/读
   `S` 和 `P` 两个 `N×N` 矩阵，访存是 `O(N²)`。当 `d = 64~256` 而 `N` 可以到 10⁵
   时，`N²` 那部分占绝对多数（本章 `code/01` 的账：`N = 8192, d = 128` 时，
   `S`/`P` 的读写占全部流量的 ~98%）。换句话说：**矩阵乘本身不是瓶颈，把中间结果
   搬出去再搬回来才是。**
2. 「exact」指的是输出与标准 attention **逐元素相等**（在浮点误差内），没有做
   稀疏化、没有丢 key、没有近似 softmax。它改变的是**中间结果的存放位置与遍历顺序**
   （分块 + online softmax），改变的是 IO 复杂度，不是数学定义。
3. 因为分块之后每个 `Q` 块都要扫一遍完整的 `K`、`V`。原来的朴素实现只读一次 `K`、`V`
   （但换来的是 `N²` 的中间矩阵落地）。这是典型的**用带宽换片上存储**：把
   `O(N²)` 的存放需求换成 `O(N/block_m)` 次 K/V 重读。

**常见错误**：把「省流量」说成「K/V 只读一次」。事实上融合版读 K/V 的次数更多，
省的只是 `S`/`P`。另一类错误是说 FlashAttention「近似」或「采样」——它不是。

---

## E2 概念题 ★

**思路**：推导的出发点是「softmax 对指数平移不变」。

**参考答案**

```text
m_new = max(m_prev, rowmax(S_blk))          （S_blk 已缩放）
c     = exp(m_prev − m_new)
ℓ     = c · ℓ      + rowsum(exp(S_blk − m_new))
O     = c · O      + exp(S_blk − m_new) · V_blk
```

1. 因为 `m_new = max(m_prev, …) ≥ m_prev`，所以指数 `m_prev − m_new ≤ 0`，于是
   `c = exp(≤0) ∈ (0, 1]`。`c = 1` 表示最大值没变（不需要重标度），`c < 1` 表示
   出现了更大的分数，旧的累加器要等比缩小。
2. 因为 `exp(sx_j − m) = exp(sx_j)·exp(−m)`，而 `exp(−m)` 在分子与分母上各出现一次，
   直接约掉；`m` 只是为了让指数不为正。
3. **数学上仍然正确**：把 `m` 固定成任何常数，最终比值不变（`code/02` 里把块大小与
   key 顺序换掉仍在 1e-16 内一致，是同一件事的另一种体现）。**数值上会出事**：
   `x` 一大 `exp(x)` 直接上溢成 `inf`，`inf/inf = NaN`（`code/02` 第 3 段给了
   一个 `x ≈ 700` 的例子）。所以 `m` 是纯粹的范围控制装置。

**常见错误**：把 `c` 说成「误差修正因子」。它不是修正误差，它是在不同 `m` 之间做
**精确的坐标变换**。

---

## E3 概念题 ★★

**思路**：先去锁定源码看 `T.infinity` 到底落在哪个值，再讨论 NaN。

**参考答案**

1. 是 IEEE `+∞`。实现是
   `_vendor/tilelang-v0.1.15/src/op/math.cc:80-108`：`infinity_op` 对 16/32/64 位浮点
   与 bfloat16 都返回 `std::numeric_limits<float>::infinity()`；前端入口在
   `_vendor/tilelang-v0.1.15/tilelang/language/tir/op.py:361`。
   **注意这不等于 dtype 的最大有限值**，很多教材会假设后者。
2. `exp2(−∞ − m)`：当 `m` 有限时结果是 0；这条路径保证被 mask 的元素对 `ℓ` 与 `O`
   的贡献恰好是 0。
3. 该行的 `m_prev` 与 `m_new` 都还是 `−∞`，于是 `c = exp2(−∞ − (−∞)) = exp2(NaN) = NaN`，
   `acc_o` 被污染。`check_inf` 的做法是把「仍是 `−∞` 的运行最大值」置成 0，从而
   `c = exp2(−∞ − 0) = 0`，整块贡献 0，不再出现 NaN。
   见 `_vendor/tilelang-v0.1.15/examples/attention_sink/example_mha_sink_fwd_bhsd.py:103-109`。
4. 用 `-1e9` 时不会出现 `−∞ − (−∞)`：`m` 从第一块起就是有限的 `-1e9`，于是
   `c = exp2(−∞ − (−1e9·scale)) = 0`（第一块）或 `1`（后续块），全程不产生 NaN。
   代价是**被 mask 的元素不再保证贡献 0**：在 `m = -1e9·scale` 的那一块里，
   `exp2(−1e9·scale − (−1e9·scale)) = 1`，于是 `rowsum` 会把整块的「假峰」也加进
   `logsum`。在官方 varlen 例子里这不影响正确性，因为这些行是 padding 行、最后不写回
   （`example_gqa_fwd_varlen.py:122-124` 的 guard）；但它说明 `-1e9` 与 `-∞`
   **不是等价的两种写法**，换一种 kernel 结构就可能出错。

**常见错误**：认为 `T.infinity` 是 `FLT_MAX`；或者认为 `check_inf` 能修「整行都看不见」
（`ℓ = 0`）的情形——那是另一种问题，见 E3.3 与正文 5.1 节。

---

## E4 概念题 ★★

**思路**：直接套本章 1.2 节的四项与 2.5 节的两项；`Sq = Skv = N = 4096, d = 128`。

**参考答案**

1. 朴素版 `4·Sq·Skv + 2·(Sq+Skv)·d`：

   ```text
   4 × 4096 × 4096 = 67,108,864         （S/P 的写+读+写+读）
   2 × (4096+4096) × 128 = 2,097,152    （Q/K/V/O 各一次）
   合计 69,206,016 个元素
   ```

   `N²` 项占 `67,108,864 / 69,206,016 ≈ 97.0%`。
2. 融合版（`block_m = block_n = 128`，`Sq/block_m = 32` 个 Q 块）：

   ```text
   Q 读   = 4096 × 128 = 524,288
   K 读   = 32 × 4096 × 128 = 16,777,216      （每个 Q 块重读一遍）
   V 读   = 32 × 4096 × 128 = 16,777,216
   O 写   = 4096 × 128 = 524,288
   S/P    = 0
   合计 34,603,008 个元素
   ```

3. 相差 `69,206,016 / 34,603,008 = 2.00x`。
   换成 `Sq = 1, Skv = 4096` 时：朴素版 `4×1×4096 + 2×4097×128 = 1,065,216`，
   融合版 `1,048,832`，比值 `≈ 1.02x` —— **几乎不省**。因为此时「`N×N` 中间矩阵」
   只有 `1×4096`，朴素版本来就不浪费流量；decode 的瓶颈是并行度而不是流量。
4. **高估**：它假设每次访问都真的落到 HBM，忽略了缓存复用、忽略了 TMA 一次搬一整块、
   也忽略了「`N×N` 根本放不进 SRAM」这件事实（后者意味着朴素版在长序列上根本无法
   实现）。**低估**：它只数元素个数，不数这些访问的**代价差异**——写回 `N×N` 与
   读 `Q` 的分块，在时间轴上的效果完全不同（写要占用带宽与容量，还可能引发同步）。
   所以 `2.00x` 这个数字**不能**当作性能预测。

**常见错误**：忘记把 `Q`/`O` 的账算进融合版；或者把 `4N²` 写成 `2N²`（少算了
`softmax` 这一趟的读与写）。

---

## E5 代码阅读题 ★

**思路**：去读 `T.reduce_max` 的 docstring。

**参考答案**

1. `clear=True`（默认）：先把输出缓冲初始化成 `-inf` 再做归约；`clear=False`：
   在输出缓冲**已有的值**上继续归约（取 max / 累加）。
   见 `_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:176-204`。
   `reduce_sum` 的语义在 `:230-256`，那里还额外说明了 `clear=True` 时内部会先
   复制一份 temp 以免 warp 归约把同一个值重复累加。
2. 两种写法**语义等价**：`T.fill(buf, -inf)` + `clear=False` 与直接 `clear=True`
   都得到「从 `-inf` 起做 max」。官方写成显式 `T.fill` 只是因为紧接着要 `T.fill`
   别的缓冲、读起来更整齐。**容易误读的地方是**：看到 `clear=False` 会以为「要累加到
   旧值上」，但这里的旧值刚刚被 fill 成 `-inf`。
3. 对应递推里的 `m_new = max(m_prev, M_blk)`：`scores_max_prev` 是上一步的 `m`，
   `scores_max` 是本块的最大值，取 max 之后得到新的运行最大值。

**常见错误**：把 `clear=False` 理解成「不清零就累加」而不去看前一行已经 fill 过。

---

## E6 代码阅读题 ★★

**思路**：先判断「结果对不对」，再判断「快不快」。

**参考答案**

1. **结果是对的，但慢。** 因为 mask 已经把 `q_idx < k_idx` 的位置压成 `-∞`，
   多算的块不会污染结果（`exp2(−∞ − m) = 0`）。错的地方是第 2 行
   `loop_range = T.ceildiv(seq_kv, block_N)`：causal 下这一块 query 最多只能看到
   `(bx+1)·block_M + past_len` 这个 key 下标，后面的 KV 块是白算的。
   （**唯一会真的错的情形**是 `m` 全为 `−∞` 时产生 NaN——见 E3.3；在 causal 且
   `past_len ≥ 0` 时对角线永远可见，所以不会发生。）
2. 慢在「大约一半的 KV 块被算了两遍却全部被 mask 掉」，而且这些块还要花带宽去搬。
   官方写法见 `_vendor/tilelang-v0.1.15/examples/flash_attention/example_mha_fwd_bhsd.py:59-63`：

   ```python
   loop_range = (
       T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M + past_len, block_N))
       if is_causal else T.ceildiv(seq_kv, block_N)
   )
   ```

3. 写成 `q_idx > k_idx` 会把**对角线元素**也 mask 掉。`past_len = 0` 时，
   `q_idx = bx·block_M + i`、`k_idx = k·block_N + j`，对角线是 `i == j` 那些位置；
   它们被 mask 之后，每一行至少还有一个可见 key（更早的 key），所以不会变成 NaN，
   但**输出会错**：等于把 attention 从「包含自己」改成「只看严格过去」，最后一行
   （`i = 0`）会一个 key 都看不到，`ℓ = 0` → `0/0 = NaN`。这就是 5.2 节说的
   「对角线边界差一个等号，会在特定形状上静默出错」。

**常见错误**：认为「多算的块会被 mask，所以完全无害」。它对正确性无害，但对性能和
数值稳定性都不是无害（多算的块增加了出现全 `−∞` 行的概率）。

---

## E7 代码阅读题 ★★

**思路**：把每一段映射回数学公式，注意 `lse` 是**对数**。

**参考答案**

1. `lse_logsum_local = log2( Σ_s 2^(lse_s − lse_max) ) + lse_max`，即
   **`logsumexp(lse_s)`**，也等于 `log2( Σ_s ℓ_s )`（因为 `lse_s = log2(ℓ_s)` 相位化）。
   换句话说：它是「把各段的分母加起来」之后取对数。
2. `scale_local = 2^(lse_s − lse_logsum) = ℓ_s / Σ_{s'} ℓ_{s'}`。
   必须等于 `ℓ_s / Σℓ` 的理由：每一段的 `O_s` 已经**各自除过自己的 `ℓ_s`**
   （是归一化输出），而真实输出是
   `O = Σ_j P_j v_j = Σ_s Σ_{j∈s} P_j v_j = Σ_s (ℓ_s/Σℓ) · O_s`。少乘这个
   `ℓ_s/Σℓ`，相当于假设每段的质量一样大。
3. 直接相加会得到 `Σ_s ℓ_s O_s / 1` 量级的结果，而正确结果是
   `Σ_s (ℓ_s/Σℓ) O_s`——每一项都被放大了 `Σℓ` 倍，且各段相对权重也错了。
   `code/03-split-kv-combine/` 的实测：
   加权合并误差 `5.6e-17`，直接相加误差 `6.1e-01`（同一个问题、同一个随机种子）。
4. 因为 `lse_s` 的绝对值可以很大（它随 `skv` 增长），直接 `exp2(lse_s)` 会溢出；
   减去 `lse_max` 之后指数 `≤ 0`，`exp2` 的结果落在 `(0, 1]`。这与 2.2 节
   「减去 `m`」是**同一个技巧**，只是这次作用在段维度上。

**常见错误**：把 `lse` 当成 `ℓ`。它们是 `log2` 关系；`scale_local` 写成
`lse_s / Σlse` 就错了（这是最容易犯的一个）。

---

## E8 代码阅读题 ★★

**思路**：昇腾把矩阵计算与向量计算放在不同的核上，跨核只能借道 GM。

**参考答案**

1. `T.Scope("C")` 是 **Cube**（矩阵）核，`T.Scope("V")` 是 **Vector**（向量）核。
   它们各有自己的片上存储（Cube 用 L1/L0A/L0B/L0C，Vector 用 Unified Buffer），
   两者之间**没有直接的搬运通道**，因此不能直接共享缓冲。
2. 分数必须经过 workspace，因为唯一的出路是 `L0C → GM`。依据是锁定版编程指南里的
   搬运关系表（`_vendor/tilelang-ascend-v0.1.2.000/docs/TileLang-Ascend Programming Guide.md:629-636`）：
   合法对里包括 `GM→L1`、`L1→L0A/L0B`、`L0C→GM`、`GM→UB`、`UB→GM`、`UB→UB`、`UB→L1`，
   **没有 `L0C → UB`**。指南另在 4.1.5.2 节说明编译器会把
   `T.copy(acc_s_l0c, acc_s_ub)` 这类的跨核 `T.copy` 自动改写成
   「写 workspace + 读 workspace」两条指令（同文件 `:1345-1390`）。
3. `workspace_2` 装 **fp16 的概率 `P`**（softmax 的输出，要送回 Cube 做 `P·V`）；
   `workspace_3` 装 **`P·V` 的 fp32 部分输出**（Cube 算完再送回 Vector 做累加与
   归一化）。见
   `_vendor/tilelang-ascend-v0.1.2.000/examples/flash_attention/flash_attn_bhsd.py:26-34`。
4. 每个 KV 块至少两次跨核握手：Cube 通知 Vector「分数好了」（`T.set_cross_flag("FIX", 0)`
   / `T.wait_cross_flag(0)`），Vector 通知 Cube「概率好了」（`set_cross_flag("MTE3", 1)`
   / `wait_cross_flag(1)`），返回时还要再握一次手。同步指令本身要下发、要等待，
   次数一多就会进入 **scalar bound**，所以调优文档把它列成一条独立的优化项
   （`fa_opt/bench_mark_zh.md` 里「优化核间同步下发次数」对应「原生 AscendC 的 72%」，
   **厂商自报，作者未复现**）。

**常见错误**：把 `T.Scope("C")/(V)` 理解成「两种写法」而不是「两个物理核」。

---

## E9 动手题 ★★

**思路**：直接改 `code/02-online-softmax/online_softmax.py` 的第 1 段：在循环里同时
打印归一化输出。

**参考答案（关键代码）**

```python
for start in range(0, n, block):
    xb = x[start:start + block]
    vb = v[start:start + block]
    M = float(np.max(s * xb))
    m_new = max(m, M)
    c = float(np.exp(m - m_new))
    p = np.exp(s * xb - m_new)
    l = c * l + float(np.sum(p))
    o = c * o + p @ vb
    m = m_new
    print(blk, m, c, l, (o / l).ravel())     # ← 多出来的这一行
```

**观察到的结论**

* 中间的 `o/l` 会**随块推进而变化**：每来一块新数据，分母变大、旧的权重被重新归一化，
  所以它不是「近似值逐步逼近」，而是「当前已知 key 集合上的精确归一化输出」。
* 最后一块处理完时，它与一次性 softmax 的结果在 1e-16 内一致（脚本里有 assert）。
* **倒序 key**：中间序列完全不同（第一次见到的 key 不同 → `m` 的爬升路径不同 →
  `c` 出现的时机不同），但最终比值相同——因为最终公式里 `exp(−m)` 会被约掉，
  中间路径不影响比值。这正是 2.4 节归纳证明的内容。

**常见错误**：把「中间输出在变」误读成「算法不稳定」。它是**精确的**当前归一化结果；
不稳定的只是中间量的量级（那正由 `m` 控制）。

---

## E10 动手题 ★★

**思路**：tiled 流量 = `2·ceil(Sq/block_m)·Skv·d + 2·Sq·d`，只有第一项随 `block_m` 变。

**参考答案**

`Sq = Skv = 8192, d = 128`：

| `block_m` | Q 块数 | K/V 重读 | Q+O | 合计 |
| --- | --- | --- | --- | --- |
| 32 | 256 | 536,870,912 | 2,097,152 | 538,968,064 |
| 64 | 128 | 268,435,456 | 2,097,152 | 270,532,608 |
| 128 | 64 | 134,217,728 | 2,097,152 | 136,314,880 |
| 256 | 32 | 67,108,864 | 2,097,152 | 69,206,016 |
| 512 | 16 | 33,554,432 | 2,097,152 | 35,651,584 |

1/2. **单调下降**，因为 K/V 的重读次数就是 Q 块个数 `ceil(Sq/block_m)`，`block_m` 越大
重读越少；模型里没有任何惩罚项，所以它一路下降。

3. 真实 kernel 不能用大 `block_m`，至少三个理由：

   * **片上容量**：`Q_shared` 是 `block_m × dim`，`acc_s` 是 `block_m × block_n`
     的 fp32 fragment（寄存器！）。`block_m = 512, block_n = 128, dim = 128` 时
     `acc_s` 单个就是 64K 个 fp32 累加器，远超寄存器预算。
   * **warp 划分与 layout**：`block_m × block_n` 必须能被 warp 数整除并映射到 MMA
     tile（第 14 章）；过大的 tile 会让 `FullRow` 策略下的每 warp 行数变得不合理。
   * **尾部浪费**：`seq_len` 不是 `block_m` 整数倍时，越大的 `block_m` 浪费越多
     （`T.ceildiv` 会开出多余的行），varlen 场景尤其明显。

   另外注意：这个模型**忽略了朴素版最大的问题**——`N×N` 根本放不进片上。
   所以「模型说越小越差」并不说明朴素版更好。

**常见错误**：只看模型结论就断定「`block_m` 越大越好」，忘了模型的假设边界。

---

## E11 概念题 ★★

**思路**：把 sink 理解成一个「值向量为 0 的额外 key」。

**参考答案**

1. sink 相当于 softmax 分母里额外的一项：它对输出没有任何贡献（分子 += 0），
   但会占掉一部分概率质量，因此**只会让输出变小、不会改变方向**。它的用途是给
   softmax 提供一条「什么都不看」的出路（原论文语境是让模型可以放弃关注任何 token）。
2. 把 sink 记作第 `N+1` 个 key，其分数为 `sink`、值为 `0`，则

   ```text
   P_j    = exp(z_j − m*) / ( Σ_k exp(z_k − m*) + exp(sink − m*) )
   O      = Σ_j P_j v_j
   m*     = max( max_k z_k , sink )
   ```

   代码里的 `logsum[i] += exp2(sinks[i]·log2e − scores_max[i]·scale)`
   就是给分母补上 `exp(sink − m*)` 那一项（`scores_max` 已经是缩放后的 `m`）。
3. 昇腾例子的注释说它用 `m_i_final = max(z_k)` 而不是 `m* = max(sink, m_i_final)`：
   如果 `sink > m_i_final`，那么 `exp(sink − m_i_final)` 的指数是**正数**，
   量级可以达到 `exp(几百)` → 溢出/主导整个 `sumexp`，输出会塌成 0。
   用 `m*` 之后指数 `≤ 0`，恒不超过 1。（原文见
   `_vendor/tilelang-ascend-v0.1.2.000/examples/mha_sink_fwd_bhsd/mha_sink_fwd_bhsd.py:565-577`；
   该文件还提醒要把 `sumexp`/`acc_o` **从 `m_i_final` 重标度到 `m*`**，因为分母和分子
   必须在同一个 `m` 上。）

**常见错误**：只把 sink 加到分母而不检查 `m` 是否包含 sink——`m*` 必须覆盖所有
出现在 `exp` 里的分数，包括 sink。

---

## E12 动手题 ★★★

**思路**：softcap 是**对数的变换**，而 kernel 的 `m` 必须与 `exp` 的输入来自同一个
变换后的分数。

**参考答案（关键代码）**

```python
def cap(z, softcap):
    return z if softcap is None else softcap * np.tanh(z / softcap)

# 参考实现
z = (q @ k.T) * scale                      # 先缩放
z = cap(z, softcap)                        # 再 softcap
z = np.where(mask, -inf, z)

# 分块实现
acc_s = q_blk @ k_blk.T
acc_s = np.where(masked, -inf, acc_s * scale)
acc_s = cap(acc_s, softcap)
m_new = max(m_prev, rowmax(acc_s))         # ← 必须在 cap 之后取 max
p = np.exp2(acc_s * LOG2E - m_new * LOG2E)
```

(1) 用 `softcap ∈ {None, 50, 30}` 与参考实现逐元素比对，误差应在 `1e-12` 内。

(2) **为什么不能只改 `acc_s[i,j] *= scale` 那一行**：

* `tanh` 需要的是**缩放后的 logit** `s·x`，而 kernel 里从未显式形成它——它一直是
  `x·(s·log2e)`，`log2e` 被折进了 `exp2`。要做 softcap，必须先还原出 `s·x`，
  再做 `cap·tanh(·/cap)`，再乘回 `log2e`。
* 更关键的是 `reduce_max`：`scores_max` 必须是**变换后分数**的逐行最大值。虽然 `tanh`
  单调，`argmax` 不变，但**最大值本身变了**（`cap·tanh(z/cap) ≠ z`）。
  如果 `m` 来自未变换的分数而 `exp2` 的输入是变换后的分数，那么
  `c = exp2(m_prev − m_new)` 与 `p = exp2(x' − m)` 就不在同一个坐标系里，
  rescale 因子会错——结果不是 NaN，而是**静默地算错**。

**常见错误**：以为「`tanh` 单调所以顺序无关，随便在哪一步加都行」。
单调性只保证 argmax 不变，不保证 `m` 的数值一致。

---

## E13 代码阅读题 ★★★

**思路**：`offset` 是「右对齐」的因果偏移；`-∞` 与 `-1e9` 的差别在整行全 mask 时显现。

**参考答案**

1. 因为这是「Q 是增量解码的新 query、K/V 是包含前缀的历史」这种场景：最后一个 query
   token 应该能看见最后一个 key。所以让 query 的尾部与 key 的尾部对齐，
   `offset = kv_len − q_len`；这样 `q_idx + offset` 才是它在全局序列里的绝对位置。
   源码注释写得很直白：`# always align on the right`。
2. 两个上界分别防的是：
   * `T.ceildiv(max_visible_k_idx, block_N)`：causal 下**不可能看到**的 KV 块，
     用不着扫（性能）；
   * `T.ceildiv(kv_current_seqlen, block_N)`：**超出这个样本的 KV 长度**的块，
     那是下一个样本的数据，扫了就是错的（正确性）。
   varlen 打包之后所有样本首尾相接，第二个上界是必须的。
3. 长度 3 的例子：`Sq = Skv = 3`（为简单起见 `offset = 0`），取
   `block_M = block_N = 4`，于是最后一个（第 3）query 行以及第 4 行的所有 key
   都在 `q_idx < k_idx` 的范围内 → 整行被 mask。此时

   ```text
   用 -1e9:  m_new = -1e9（有限）  →  c = exp2(−∞ − (−1e9·scale)) = 0，无 NaN
   用 -inf :  m_new = −∞           →  c = exp2(−∞ − (−∞)) = NaN
   ```

   最先出问题的是 **`scores_scale` 那一行**（`c` 变成 NaN），随后 NaN 通过
   `acc_o *= scores_scale` 污染整个输出累加器。**注意**：在这个具体 kernel 里，
   被污染的是越界的 padding 行，而它们在写回时被
   `if bx * block_M + i < q_current_seqlen` 挡掉了（`example_gqa_fwd_varlen.py:122-124`），
   所以「NaN 存在但看不出来」。这正是 `-1e9` 这种写法危险的地方：**它对结果无害，
   但对后续改动（比如加一个跨行的 epilogue）不健壮**。

**可选（需要硬件）**：在 NVIDIA GPU 上跑真机一致性检查；失败会直接抛出
`torch.testing.assert_close` 的错误。

---

## E14 动手题 ★★★（可选，需要 NVIDIA GPU）

**思路**：这是一个「配置 → 现象」的实验，答案取决于你的卡，所以这里只给**判据**
和**预期方向**，不给数字。

**参考答案（判据与推理）**

1. 三组配置里，`(128, 128, 4)` 最可能因 shared memory 超限而**编译/启动失败**：
   按第 11 章 4.5 节的公式，流水相关的片上占用上界是
   `num_stages × (block_M×block_K + block_K×block_N) × sizeof(dtype)`。
   在 attention 里被多缓冲的是 `K_shared`、`V_shared`（各 `block_N × dim`），
   所以 `(128,128,4)` 大约需要 `4 × 2 × 128 × 128 × 2 B = 256 KB`，再加
   `Q_shared`（32 KB）—— 已经超过多数卡的 shared memory 上限（H100 为 227 KB/块）。
   `(128,128,2)` 则约 128 KB + 32 KB，能过。
2. `(64, 64, 1)` 最慢：`num_stages=1` 意味着搬运与计算**不重叠**（第 11 章 1.1 节），
   每轮成本是「搬运 + 计算」而不是二者的较大者；同时更小的 tile 意味着更多的
   K/V 重读与更多的循环迭代（本章 `code/01` 的账）。
3. 正确性检查三组都应该通过（除非第一组因为容量而直接失败）。

**报告要求**：写明 GPU 型号、CUDA/toolchain 版本、TileLang commit
（`a35f8ddf45eba16c21211ec8822d56ce5363036f`）、benchmark 口径（warmup/rep、
是否开启 fast-math），以及原始输出。**不要引用本章的任何数字**——本章没有实测数字，
所有引用都必须来自你自己的测量。

**常见错误**：把「编译失败」当成「TileLang 有 bug」。容量约束是**设计上的硬约束**，
它的错误信息本身就是提示。
