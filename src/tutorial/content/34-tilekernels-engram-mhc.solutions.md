# 第 34 章 习题参考答案

每题给出**解题思路 + 参考答案 + 常见错误**。所有行号都指 corpus 快照
`../tilelang-corpus/14-deepseek-ascend-stack/raw/` 下的文件（TileKernels 的引用在站点上
会链接到 `deepseek-ai/TileKernels` 对应 commit 的源码）。

本章没有需要硬件才能作答的题；凡是涉及实测的地方，答案只给“怎么测”。

---

## E1 概念题 ★

**思路**：先分清“稀疏”在不同轴上的含义，再谈寻址。

**参考答案**

1. MoE 稀疏的是**计算**：每个 token 只激活少数专家，用运行期的 hidden state 路由，
   参数（专家）是有条件地被使用的。Engram 稀疏的是**记忆**：每个 token 只取固定
   数量的静态 embedding 行，用 token id 做确定性索引。论文把前者叫
   conditional computation，后者叫 conditional memory，两者互补
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:64-80`）。
2. 指“embedding index 只由输入 token id 决定”，不依赖 hidden state，
   因此**前向开始前**就能算出要读哪些行。有了这个可预测性，系统可以在算前几层
   的时候异步把行从 host memory 预取到设备上，把 PCIe 延迟藏在计算后面
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:97-99,341-351`）。
3. 指的是**取行数不随表增大**：`code/04` 里表从 25.8 万槽放大到 1000 万槽，
   每 token 每层仍然只取 `(max_ngram-1) × heads = 16` 行。反过来，
   “延迟不是常数”的证据是论文 §2.5 专门讨论 Engram 该放在第几层：
   放得深才有更多计算窗口来盖住预取延迟（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:341-351`）。

**常见错误**：把 `O(1)` 读成“访问一次是常数时间”。它约束的是**访问次数**，
不是单次访问的代价；表放在 NVMe 上，次数不变但每次更慢。

---

## E2 概念题 ★★

**思路**：分别找“论文的证据等级”与“仓库自我声明”。

**参考答案**

1. 不能。demo 的作用是演示数据流，它把 Attention/MoE/mHC 全部 mock 掉了，
   而且只有一个 `forward`，没有训练循环、没有 loss、没有容量分配开关。
   验证 scaling law 需要训练几十个不同 `ρ` 的模型并比验证 loss，
   demo 连训练都不做。
2. `../tilelang-corpus/14-deepseek-ascend-stack/raw/Engram/README.md:89`：
   `注意： Note: The provided code is a demonstration version intended to illustrate the
   data flow. It mocks standard components (like Attention/MoE/mHC) to focus on the
   Engram module.`
3. 一致。论文式 (1) 说 `φ_{n,k}` 是“lightweight multiplicative-XOR hash”
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:167-176`）；内核实现是先做
   `T.bitwise_xor(hash, x[n]*mult[n])` 的累进 XOR，再对每个 head 的素数取模并
   加 offset（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_hash_cuda.py:54-63`）。

**常见错误**：把“仓库有代码”等同于“结论已被验证”。判断证据等级要问
“这份代码跑出来的数能支持这句话吗”，而不是“有没有代码”。

---

## E3 概念题 ★

**思路**：照抄 §2.1 与 §3.1，不要自己发明。

**参考答案**

记 `z = (Σ_d x_d·w_h,d·w_e,d·k_d) · rstd_x · rstd_k · d^{-1/2}`，则

$$
\begin{aligned}
r_x &= \operatorname{rsqrt}(\operatorname{mean}(x^2)+\epsilon),\\
r_k &= \operatorname{rsqrt}(\operatorname{mean}(k^2)+\epsilon),\\
\alpha &= \operatorname{sigmoid}
  \left(\operatorname{sign}(z)
  \sqrt{\operatorname{clamp}(|z|,\mathrm{clamp\_value})}\right),\\
\mathrm{out} &= x+\alpha v.
\end{aligned}
$$

1. 因为两个 RMSNorm 都是“逐元素乘权重”：
   `RMSNorm(x,w_h)ᵀ·RMSNorm(k,w_e) = rstd_x·rstd_k·Σ x·(w_h⊙w_e)·k`，
   乘积是逐元素可交换的，所以能先算 `w_fused = w_h⊙w_e`。
2. `signed-sqrt` 把“大负相关”压缩成有限区间（`sign·sqrt(|·|)`），
   让 sigmoid 的两侧都有合理斜率；`clamp_value` 防止 `|z|→0` 时 `sqrt` 的
   导数发散（反向里对应 `T.Select(|z| < clamp_value, 0, …)`，
   `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_bwd_cuda.py:249-253`）。
3. `α ∈ (0, 1)`（sigmoid 的值域）。`α → 0` 说明 RMSNorm 后的 hidden state 与
   key 方向不相关，检索到的记忆与当前上下文不一致，于是这段记忆被抑制、
   输出退化为 `x`。

**常见错误**：把 `rstd_x`/`rstd_k` 的位置写错。它们是**标量**（每个 token 每个
head 一个），乘在点积外面，而不是逐元素乘进去。

---

## E4 代码阅读题 ★★

**思路**：逐行对照 §3.1 的恒等式与 §3.2 的段号。

**参考答案**

1. `Σ_d x_d²`（用于 `rstd_x`）、`Σ_d k_d²`（用于 `rstd_k`）、
   `Σ_d x_d·w_fused,d·k_d`（`raw_dot`，用于 `α` 与 `grad_w`）。
2. 因为恒等式允许把两个权重先乘好。若分别读两个权重再相乘，每个元素要多读一次
   （两个权重 → 四个操作数），寄存器压力与 L2/global 流量都增加；而 `w_fused`
   只在权重更新后重算一次（`fused_weight` kernel）。
3. 等的是“已经发出、但还没完成”的 `async_copy` 组数。`T.ptx_wait_group(2)`
   允许最多 2 组在飞，等于“保留双缓冲的流水”。若写 `0` 就变成每搬一块都等全部
   完成，流水被打断。
4. `x`/`k` 的第一块 tile 在进入循环前（prologue，`:92-95`）就已经搬进
   `kv_smem[0]` 与 `x_smem[0:blk_d]` 了；循环体处理的是 `prev_phase = i_b - 1`，
   也就是“上一轮搬进来的那块”。所以 `i_b` 从 1 开始。

**常见错误**：把 `ptx_wait_group(n)` 理解成“等 n 个拷贝完成”。它是
**允许最多 n 组未完成**，语义方向正好相反。

---

## E5 代码阅读题 ★★★

**思路**：先填表，再回答问题；重点是“哪些概念在两份实现里同名同义”。

**参考答案**

| 关注点 | CUDA 版 | 昇腾版 |
| --- | --- | --- |
| 一个任务处理几个 head | 1 | `heads_per_task = 2` |
| 多缓冲怎么实现 | 手写 `phase = i_b % 2` + `async_copy` + `ptx_wait_group` | `T.annotate_buffer_versions({buf: num_stages})`，由 pass 展开 |
| 三个累加量的实现 | 三个局部标量，`T.warp_reduce_sum` 归约 | `S.vmula` 累加到向量寄存器，`S.vcadd` 归约 |
| 把闸门广播到每条向量 | 用一个标量 `gate_score_reducer[0]` 直接乘 | `S.vld(..., dist='BRC_B32')` 广播成向量 |
| 输出怎么回到显存 | 直接 `output_target[...] = ...` | 先写 UB（`S.vsts(... 'PK_B32')`），再 `T.copy` 回 GM |

**一一对应**的部分是**数学**：三个累加量、`rstd`、`α` 的公式、两遍扫描、
`v` 与 `k` 的生命周期复用、`save_for_backward` 的四个小张量，完全一致。
**各自机器独有**的是**工程**：切任务的方式、多缓冲的表达、归约原语、
UB vs shared memory、SIMD 内建 vs 编译器向量化。

这正是“一份算法、两个后端”的含义：把机器的差异都收进“怎么切、怎么同步、
向量多宽”这三件事里，而让算法部分保持**可逐行对照**。

**常见错误**：认为“两个后端”是两份独立实现、只是结果接近。恰恰相反，
本章 §3.4 的表格能逐项对上，昇腾融合版的注释甚至明确写“op-for-op /
bitwise identical”。

---

## E6 计算题 ★★

**思路**：先算元素数，再乘字节数；注意 `kv` 是 `hc+1` 行。

**参考答案**

`T = 8192`、`hc = 4`、`d = 4096`、bf16 = 2 B：

1. `hidden_states` = `8192 × 4 × 4096 × 2 B` = **256 MiB**；
   `kv` = `8192 × 5 × 4096 × 2 B` = **320 MiB**；
   `output` = `256 MiB`。
2. 必读 `hidden_states` + `kv` = 256 + 320 = **576 MiB**；
   必写 `output` = **256 MiB**；合计 **832 MiB**。
3. 四个 `[8192, 4]` 的 fp32 张量 = `4 × 8192 × 4 × 4 B` = **512 KiB**，
   约占总流量（832 MiB）的 **0.06%**。所以推理时把 `save_for_backward` 关掉
   几乎不影响访存，但训练时必须开。
4. 两个 `[8192,4,4096]` 的 fp32 中间张量各写一遍读一遍：
   `2 × 2 × (8192×4×4096×4 B)` = `2 × 2 × 512 MiB` = **2 GiB**。
   这比整条前向的“必读必写”还大一倍多。`code/01` 的口径只数元素个数，
   **忽略了**：L2 cache 的命中、编译器可能的融合、以及真实 kernel 里
   `v` 与 `x` 的复用带来的额外节省。

**常见错误**：忘乘 bf16 的 2 字节，或把 `kv` 当成 `[T, 4, d]`。

---

## E7 概念题 ★★

**思路**：从“并行划分”出发，而不是从公式出发。

**参考答案**

1. 因为反向 kernel 是按 token 划分并行的：每个 CTA 只处理一部分 token，
   它算出的 `grad_w` 只是**局部和**，必须按 CTA 维度堆叠起来等后续归约
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/engram/engram_gate_bwd_cuda.py:114,332-336`）。
2.

$$
\begin{aligned}
g_w &= \sum_{\mathrm{blocks}}g_{w,\mathrm{partial}}[b],\\
g_{w_h} &\mathrel{+}= g_w\odot w_e,\\
g_{w_e} &\mathrel{+}= g_w\odot w_h.
\end{aligned}
$$

3. 两条理由任答其一即可：① 跨 CTA 求和若用原子加，浮点结果依赖执行顺序，
   不可复现；TileKernels 选择“写部分和 + 确定性归约”。② 求和只需要做一次，
   而它要被两个参数共用，放在一个独立 kernel 里正好顺路。
4. 带 `main_grad` 时，`grad_w_reduce` 把结果**原地累加**到参数的 fp32 梯度缓冲上，
   autograd 层返回 `None`；不带 `main_grad` 时，它接收一个新分配的 fp32 张量并
   写入（本质也是 `+=`，因为初值是 0）。见 `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/modeling/engram/engram_gate.py:78-99`。

**常见错误**：直接写 `grad_weight_hidden = grad_w_sum * weight_embed`
（覆盖而不是累加），这会丢掉梯度累积语义。

---

## E8 概念题 ★★

**思路**：对照 §5.1 的表格逐项回答。

**参考答案**

1. mHC 用**和（L1）**：行列和都归一到 1；状态是片上的一整块
   `[token_block, 4, 4]` 片段（加反向的 `2 × repeat` 份中间量）。
   Engram 用**L2 范数**：行/列按 `‖·‖₂` 缩放；状态是两个小向量
   `row_scale[num_rows]` 与 `col_scale[head_dim]`。
2. 因为稳定性和语义：先减行最大值再 exp，避免大 logits 溢出；同时让每一行
   天然变成一个概率分布，后续行列归一化才有“双随机”的收敛目标。
3. 因为列归约是**跨 CTA** 的：每个 CTA 只处理一部分行，只能算出列平方和的
   部分和；必须再有一个 kernel 把所有部分和归约起来（并且要保持确定性顺序）。
4. 迭代状态小到能留在片上，所以“一个 kernel 内迭代”只付一次进出的代价；
   若每轮启动一个 kernel，`repeat = 10` 就变成 20 次 launch + 20 次 HBM 往返
   （§5.4）。

**常见错误**：以为两处 Sinkhorn 是“同一份代码的两个副本”。它们的归一化目标、
范数、状态、拆分的 kernel 数都不同。

---

## E9 代码阅读题 ★★★

**思路**：先看网格与线程绑定，再看注释里的四个函数名。

**参考答案**

1. `T.Kernel(num_tokens, threads=96)`：**每个 token 一个 CTA**，96 个线程
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_big_fuse_cuda.py:44`）。CTU 内按线程绑定再分两组。
2. 前 32 个线程（`T.get_thread_binding() < 32`）做“reduce 部分和 + rmsnorm、
   split mixes、Sinkhorn、写 `comb_mix`”；其余线程做 `pre_mix` 的计算与
   `pre_apply_mix` 的逐元素加权和（`:50-106` 与 `:107-134`）。
3. 四段是：
   * `_mhc_reduce_partials_and_rmsnorm_fwd`（`:47-63`）
   * `_mhc_pre_split_mixes_fwd`（`:65-72` 与 `:108-117`，pre 与 post/comb 分在两处）
   * `_mhc_sinkhorn_fwd`（`:74-103`）
   * `_mhc_pre_apply_mix_fwd`（`:119-134`）
4. 启发是：**融合的正确性可以用“与拆分版本逐 op 对齐”来定义**。
   昇腾版直接声明结果与组合路径 bitwise identical（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/pre_big_fuse_asc.py:43-51`），
   于是正确性测试只需要和已有的单算子 kernel 或 CPU 基线比一次
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tests/mhc/test_pre_big_fuse.py` 就是这么做的：用 `big_fuse_reference`
   把四个算子串起来当 golden）。

**常见错误**：以为这个 kernel 是一个“标准的 warp 分工”。它不是 SIMT 意义上的
warp specialization，而是**用线程编号当条件把 CTA 切成两组独立流水**
（CUDA 版），昇腾版则改用 `T.SimtVF(threads=32)` / `T.SimdVF()` 显式区分。

5. 论文 §4.3.1 的四条融合是：

```text
   ① 式 (14)–(15)：x⃗ φ 的两次扫描 + 反向两个矩阵乘   → 一个 kernel（GEMM，不用 TileLang）
   ② 式 (16)–(18)：α/r 与 b 的轻量运算、σ、2σ        → 一个 kernel
   ③ 式 (19)：Sinkhorn-Knopp 迭代                    → 一个 kernel
   ④ 两个「应用映射」kernel：F_pre 与 F_post,res      → 两个 kernel
   （`paper-deepseek-mhc.md:417-429,501-502`）
```

`pre_big_fuse` 覆盖的是 ② + ③ + ④ 里的 `F_pre`（外加把 GEMM 的 split-K 部分和
归约与 RMSNorm 一起做），也就是**把论文里三个独立 kernel 合成一个**；
论文里被拆开的 ① 与 ④ 里的 `F_post,res` 仍然各自独立
（`pre_big_fuse_cuda.py:44-135` 只输出 `post_mix` / `comb_mix` / `layer_input`，
`post` 是另一个 kernel）。

多融一层要满足两个条件：**共享同一份数据**（`residual` 与 GEMM 的部分和被读进
同一个 CTA，省掉一次往返）与**没有跨 CTA 的中间归约**（`mixes` 只有 24 个分量，
一个 CTA 就能算完）。反过来，一旦某一步需要全 token 的统计量或跨 CTA 归约，
就必须再拆一个 kernel。这正是论文把式 (14)–(15) 单独留下的原因。

---

## E10 概念题 ★★

**思路**：围绕“存 vs 算”做权衡。

**参考答案**

1. 省下的是**每层的 `residual` 与 `layer_input` 的保存**。
   `residual` 是 `[n, M, h]`，比 `layer_input` 大 `M = 4` 倍；
   与其为 `L` 层各存一份，不如在反向时用“剩下的输入 + 层内小张量”重算一遍。
2. 因为每一层的激活是**各自分配**的张量，只有一个 `T.Tensor[(L,), T.ptr]`
   指针表能把它们整批传进同一个 kernel；kernel 内用
   `T.make_tensor(ptr, shape, dtype)` 还原。代价是**类型与 shape 不再被编译器
   检查**，写错 shape 只会得到静默的错值。
3. 因为前向保存下来供下一层使用的残差是 bf16；重算必须复现这次
   fp32 → bf16 → fp32 的往返，否则重算出的数与真实前向不一致，
   后续梯度会偏（`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels/mhc/multilayer_recompute_cuda.py:111-112`）。
4. 这个记账只数“读写字节数”，忽略了：重算本身的**计算时间**（多一遍前向）、
   指针表的构建与上传、以及 `T.make_tensor` 带来的间接寻址开销。
   所以“IO 更少”不等于“墙钟更快”；它只是把瓶颈从带宽搬到算力。
5. 论文 Table 3（`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:510-531`）
   把激活分三类，对应到 kernel 的七个参数是：
   * “每 `L_r` 层存一次”的 `x_{l0}` → `initial_residual`；
   * “每层都存”的 `F(H^pre x_l, W_l)` → `layer_output_list`；
   * “块内瞬态重算”的 `x_l`、`H^pre x_l`、`RMSNorm(H^pre x_l)` →
     `residual_list` / `layer_input_list`（kernel 的输出），
     而 `pre_mix_list` / `post_mix_list` / `comb_mix_list` 就是论文说的
     “lightweight coefficients”。它们**不在** Table 3 里，因为很小，存下来更划算。
   论文式 (20) 的最优块长 `L*_r ≈ sqrt(nL/(n+2))` 在实现里由**调用方**决定：
   kernel 的接口是 `pre_mix_list` 等列表的长度（`num_layers`/`num_post`），
   `tests/mhc/test_multilayer_recompute.py:49-51` 还要求 `num_post` 是
   `num_layers-1` 或 `num_layers`。本章没有找到“库里自动按式 (20) 选 `L_r`”的代码，
   **这一点属于待核**：实现是否在更上层（训练脚本）里按式 (20) 与流水级对齐来切块，
   本章无法确认。

**常见错误**：把重计算说成“省计算”。它恰恰是**多做一遍前向**，
省的是显存与带宽。

---

## E11 设计题 ★★

**思路**：用 §7.4 的两问：瓶颈在字节还是指令？归约跨不跨 CTA？

**参考答案**

1. **TileLang。** 每 token 一个 4×4，迭代次数是编译期常量，归约维度 4
   完全放得进片上；瓶颈是“迭代 + 访存往返”，TileLang 的一个 kernel
   就能装下全部迭代。反过来（每轮一个 kernel 或下钻手写）会付出上百次 launch
   与 HBM 往返，收益却几乎没有。
2. **先用 TileLang，接近上限后再考虑下钻。** `N = 24` 很小，但它是真正的
   GEMM（`K = 28672`），瓶颈落回“算力 + 数据布局”。DeepGEMM-Ascend 选择用
   MAD 原语抽象 + 稀疏加载 + 协程流水把分形布局、对齐、地址计算都抠到位
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md:5`）。如果一开始就用 TileLang 的
   `T.gemm`，很快能拿到“对的数 + 不错的性能”；要吃到最后一截才需要下钻。
   论文的划分与这个判断**一致**：它把式 (14)–(15)（也就是这个 prenorm GEMM）
   明确画在 TileLang 之外，其余 kernel 才用 TileLang
   （`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-mhc.md:502`）。
   所以本题的答案不是“谁更先进”，而是**两边的边界落在同一个地方**：
   分形布局、对齐、协程流水这类指令级细节，决定了这一段必须离开 tile 级 DSL。
3. **下钻（PTX / 位运算）。** TopK 的瓶颈在“比较次数与指令数”：
   算法（阈值筛选 + RadixSelect）决定上限，硬件指令决定常数；
   这类优化要靠位运算、向量压缩比较，不是 tile 级并行能表达的（第 37 章展开）。

**常见错误**：把“小”等同于“简单就该用高层”。小 shape 反而常常需要手工
控制（对齐、padding、间接寻址）。

---

## E12 动手题 ★★

**思路**：改参数、跑脚本、看数字怎么变。

**参考答案**（在写作机上用 torch 2.1.1、CPU 实际跑过）

1. `eps = 1e-3` 时，`max|row_sum−1|` 的下限升到约 `1e-3` 量级。因为每一步都是
   “除以 `(sum+eps)`”且 softmax 后“加 `eps`”，所以固定点附近必然留下
   `O(eps)` 的偏差。这也是为什么生产里 `eps = 1e-6` 而不是更小：
   太小会让除法在数值上不稳，太大则精度不够。
2. 收敛速度基本不变。行 softmax 对输入整体加/乘常数免疫：
   加常数被减最大值消掉，乘正常数被 softmax 归一化消掉。
   所以矩阵的整体尺度只影响你“看到的输入”，不影响迭代轨迹。
3. 每行的 L2 范数变成 `1.0`（而不是 `alpha`）。因为
   `finalize` 的输出是 `alpha × (归一化后的行)`，行的 L2 = `alpha`。
   所以 `alpha` 控制的是**整体尺度**（测试里用 `sqrt(head_dim)`），
   行列的相对平衡由前面的迭代决定。
4. 以本章脚本里那个 2×4×4 的随机矩阵为例，`repeat = 10` 与 `repeat = 20` 的
   `max|row_sum−1|` 会再降大约一个数量级（`code/02` 的输出里，
   `repeat=10` 是 `5.7e-3`，`repeat=30` 是 `1.4e-6`），但**两个 `repeat`
   本身就不是同一个口径**：论文式 (9) 的一次迭代是“行 + 列”各一次
   （`paper-deepseek-mhc.md:402-406`），而代码的 `repeat` 从
   “softmax 之后做一次列归一化”开始计数（`mhc/sinkhorn_cuda.py:35-52`）。
   所以“论文 `t_max = 20`”不能直接翻译成“代码 `repeat = 20`”；
   要比较必须把两者的迭代定义对齐，或者干脆只用同一种实现比不同次数。

**常见错误**：把 `alpha` 当成“收敛精度”参数。它不参与迭代，只乘在最终结果上。

---

## E13 动手题 ★★

**思路**：把 shape 换大，把“多出来的中间张量”显式做出来。

**参考答案**

1. 把 `hidden_size` 改成 5120 后，`max|gate_naive − gate_fused|` 仍在
   `1e-5` 量级（fp32 累加 5120 项的误差）。恒等式不依赖 `d`，
   只依赖“RMSNorm = 归一化 × 逐元素权重”这个事实。
2. 显式写出 `normed_x` / `normed_k` 需要多访问
   `2 × T × hc × d × 4 B`（fp32）的写 + 同样的读。在 `T = 8192, hc = 4, d = 4096`
   时就是 E6 算的 2 GiB。但真实 kernel 的收益**不完全等于**这个字节差，因为：
   ① 中间张量可能命中 L2；② 真实 kernel 还要读 `v`、写输出，本身就是访存主导；
   ③ 融合还可能省掉一次 kernel 启动。所以“少访问的字节数”是**收益的上界估计**，
   不是实测加速比。

**常见错误**：把记账当成性能预测。本章所有字节数都是**元素计数模型**。

---

## E14 分析题 ★★★

**思路**：先读论文的两个端点解释，再把它翻译成系统前提，最后评估复现成本。

**参考答案**

1. `ρ = 100%` 退化成**纯 MoE**：所有稀疏容量都给了路由专家，没有专门承载静态
   模式的记忆，模型只能靠深度与计算去“重建”这些模式。`ρ = 0%` 退化成
   **纯 Engram**：稀疏容量全给了记忆，缺少条件计算能力，需要动态推理的任务掉下来。
   论文对两者的解释见 `../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:420-430`。
2. 系统前提至少有三条：① **表能放得下**，容量要么吃 HBM、要么能卸载到
   host/NVMe，否则“给 Engram 更多容量”根本装不进设备；② **访问次数不随表增长**
   （`O(1)` 取行数），否则把容量换成记忆以后每次前向的访存也涨；
   ③ **寻址可预取**，索引只依赖 token id，才能把搬运藏在计算后面；
   如果路由依赖 hidden state，就没法提前动。三条合起来才是论文“< 3% 开销”
   成立的条件。
3. 最小验证计划：
   * 训练规模：至少要在**同算力档位**下扫 `ρ`，每个点都要训到收敛；
     论文用的是 `2e20` / `6e20` FLOPs 两档、`P_tot/P_act ≈ 10`
     （`../tilelang-corpus/14-deepseek-ascend-stack/raw/paper-deepseek-engram.md:400-412`）。
   * 硬件：能承载这些参数量与 All-to-All 的集群（论文 §2.5 说训练时表要在
     GPU 之间 sharding），还要足够的存储放表。
   * 判别指标：**固定 token 数下的验证 loss**（不是下游 benchmark 单项），
     并保证 iso-parameter + iso-FLOPs。
   * 为什么本书做不到：写作环境没有 GPU/NPU，也没有训练数据与算力；
     复现它需要的是“几十次多十亿参数模型的预训练”，不是一次 kernel 运行。
     所以本章只能引用论文的数字，并把它标注为“论文自报，作者未复现”。

**常见错误**：把“不需要额外 FLOPs”理解成“不需要额外成本”。Engram 的容量
省的是算力，代价是显存/存储与通信；这正是 §2.3 要读者分清的地方。
