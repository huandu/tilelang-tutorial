# 第 16 章 习题答案

与 `exercises.md` 一一对应（E1–E13）。每题给「解题思路 → 参考答案 → 常见错误」。
凡出现数字，都标注它是「本书推导 / 模型输出」还是「锁定版本里的第一方表述」。

---

## E1 概念题 ★

**思路**：回到 2.1 节的四段图，再回 1.2 节的两张表。

**参考答案**：

1. router/gating（小矩阵乘 + softmax + top-k）、dispatch/permute（纯搬运）、
   grouped GEMM（矩阵乘）、combine（纯搬运 + 归约）。四段里前两段中的
   「重排」和后一段的 combine 都不是矩阵乘，只有 router 与 grouped GEMM 是。
2. permute 与 combine 各 `2·T·k·d·b` 字节（读一遍、写一遍）；它们只做数据搬移与
   选择/归约，没有乘加，所以 FLOP = 0。
3. router 的 GEMM 是 `M=T, N=E, K=d`；FLOP 只有 `2·T·d·E`，算术强度低
   （默认形状下 8.0 FLOP/Byte），所以不吃算力。但它决定了每个 token 去哪个专家，
   于是决定了后续所有重排与 GEMM 的布局；它还是「小而多」的 kernel，
   启动延迟与 top-k/排序是关键路径。
4. 反例：**大 prefill 批 + 专家数少**。默认形状（T=8192、d=7168、d_e=2048、
   E=8、k=4）下，grouped GEMM 的算术强度是 1365 FLOP/Byte（`code/01` 输出），
   远高于 333 的平衡点——此时矩阵乘是主角，permute+combine 的 1.75 GB 只是一次性
   成本。（本书推导 / 模型输出。）

**常见错误**：把 dispatch/permute 也算成矩阵乘；只算 `2MNK` 而忽略重排字节。

---

## E2 概念题 ★★

**思路**：单个专家的三个 GEMM 共享同一批权重，所以权重是「固定成本」。

**参考答案**：

1. `FLOP = 6·n·d·d_e`（gate、up、down 各 `2·n·d·d_e`）；
   字节下界 `= 3·d·d_e·b + 4·n·d·b`（三份权重 + 读激活 + 写中间/输出）；
   `AI(n) = 6·n·d_e / (b·(3·d_e + 4·n))`。注意 **`d` 被约掉了**：
   算术强度只由 `d_e`、`b` 和 `n` 决定。
2. 令 `AI(n)=333`，`d_e=2048`、`b=2`：
   `AI(n) = 12288n/(12288+8n) = 1536n/(1536+n)`；令其等于 333，
   `1536n = 333·(1536+n)` → `1203n = 511488` → `n ≈ 425`。
   （当 `n→∞` 时 `AI → 1536`，即上限由 `d_e/b` 决定。）
   为什么由权重决定：`3·d_e·b` 是与 `n` 无关的常数项，`n` 小的时候它占满分母；
   只有 `4n` 涨到和 `3·d_e` 可比时，激活才不再被权重淹没。
3. `64 tokens × 8 = 512` 个 assignment，除以 256 个专家 =
   **平均 2 个 token/专家**。`AI(2) = 1536·2/(1536+2) ≈ 2.0 FLOP/Byte`
   → 远远是 memory-bound。
4. 影响是「dual GEMM 值得做」：在小 M 下瓶颈是**读权重与读激活**，
   gate/up 共享一次 input 读能直接减掉一次激活读取；此时省下的是稀缺项。
   （对比：大 M 时 GEMM 已 compute-bound，融 dual 的收益主要落在
   减少一次激活读取与一次启动，而不是算力。）

**常见错误**：忘记权重项；把 `n` 当成 `T`（`n` 是「分到这个专家的 token 数」，
它等于 `T·k/E` 的平均值，只有在均衡且不漏时才近似成立）。

---

## E3 代码阅读题 ★★

**参考答案**：

1. 因为每个专家的 `size` 都要**各自向上对齐到 `block_token`**，所以总块数是
   `Σ_e ceil(size_e/block)`；而 `ceil(Σ size_e / block) ≤ Σ_e ceil(size_e/block)`。
   `+ group_count` 是给这种舍入差留的余量（上界），保证 grid 足够覆盖所有专家的
   最后一个不满块。多出来的 block 靠 `actual_rows=0` 被整体 mask 掉。
2. 是一张长度 `M` 的 int32 表，`group_idx_for_bx[bx]` 给出第 `bx` 个 M-block
   属于哪个专家。它在 **host 侧**用 Python 循环算出来（第 373-382 行），再作为
   kernel 参数传进去。用它是因为 kernel 内逐个比较 `group_padded_offsets` 要
   `O(group_count)` 的串行比较（kernel 里做分支代价高），而查表是 `O(1)`。
3. `m_start_padded` 是**padding 布局**里的起点（决定这个 block 在 grid 中的位置），
   `m_start` 是**真实 token 布局**里的起点（决定去哪个地址读数据）；
   两者相差该专家前面累计的 padding 量。读数据要用后者，算 grid 要用前者。
4. 不 mask 会写坏 result 缓冲区里 padding 区域的行（越界之外的脏写或覆盖别的
   专家的数据）。**不会报错**——这是最危险的一类 MoE bug。

**常见错误**：把 `group_idx_for_bx` 说成 device 侧计算；混淆两个 `_start`。

---

## E4 代码阅读题 ★★

**参考答案**：

1. `batch_sizes[e]` 是第 `e` 个专家的 token 数；`batch_offsets[e]` 是它在**真实
   拼接张量**里的起始行；`batch_padded_offsets[e]` 是它在**对齐后**的起始行。
   缺一不可：grid 划分要按 padded，读写数据要按真实，且两个索引都要与
   `cur_batch_idx` 配套。
2. 文件头注释说 “Multi-stage software pipelining on ptr-backed tensors is not
   correct yet”，所以把 `copy_num_stages` 固定为 1。
3. `ceil(0/block_M)=0`，所以 **size=0 的专家不贡献任何 block**，不会浪费
   （这正是「拼接 + 偏移表」相对「给每个专家固定留块」的优势）。
4. `code/02` 用同一套规则：先 `group_sizes`，再 `group_offsets`（紧凑布局），
   再 `padded_offsets`（对齐后布局），最后由 `block_metadata()` 产出
   `[expert, m_start, valid_m]`。昇腾的 `block_metadata` 与这里的
   `batch_padded_offsets` 是同一份信息的两种编码。

**常见错误**：认为 `batch_offsets` 与 `batch_padded_offsets` 可以互相推出
（它们之差依赖前面所有专家的 padding，不能只看当前专家）。

---

## E5 概念题 ★★

**参考答案**：

1. `q_h · (W_h c) = q_h^T W_h c = (W_h^T q_h)^T c = (W_h^T q_h) · c`。
   左端是「先算 K 再打分」，右端是「把 W_h 吸收进 query，再用潜向量打分」。
   只要 query 侧愿意多存/多算一个 `W_h^T q_h`，KV 就永远不必展开成 per-head 的
   K——缓存里只留 `c`。
2. `o = Σ_s a_s (W_h c_s) = W_h (Σ_s a_s c_s)`。也就是 attention 可以在潜空间做，
   输出仍是 `d_c` 维，最后乘一次 `W_h`（上投影）即可。
3. 这里的 `dim` 就是 `d_c = 512`（配置里的 `kv_lora_rank`），即**潜向量维度**；
   `kv_head_num=1` 是因为 `c` 是所有 head 共享的，不存在「每个 head 自己的
   KV head」这个概念（第 20 行的断言 `kv_head_num == 1` 就是在强制这一点）。
4. RoPE 是**位置相关**的旋转，如果作用在被压缩的潜向量上，会破坏低秩结构、
   也无法与「吸收」兼容；所以单独留一条 `d_r=64` 的高维分支，不压缩。
   在 kernel 里它被当成**第二个 QK 乘积，累加到同一个 `acc_s`**
   （第 65-66 行），即 `acc_s = Q@KV^T + Q_pe@K_pe^T`。

**常见错误**：以为 `dim=512` 是「每个 head 的维度」（不是，per-head 维度由
`qk_nope_head_dim / v_head_dim` 给出，是 128）；把 `kv_head_num=1` 当成 GQA 的
一个特例（本质是「共享潜向量」，不是「共享 K/V head」）。

---

## E6 概念题 ★★

**参考答案**：

1. MHA/GQA：`2·H_kv·d·b`；MLA：`(d_c+d_r)·b`。
   数值（fp16，`b=2`）：MHA(128,128)=65536 B；GQA(8,128)=4096 B；
   MLA=1152 B。（模型输出 / 本书推导。）
2. MHA/GQA decode 的 FLOP 是 `4·H_q·L·d`（QK 与 PV 各 `2·H_q·L·d`），
   字节是 `2·H_kv·L·d·b`；`L` 与 `d` 同时出现在分子分母上，约掉后
   `AI = 2·H_q/(H_kv·b)`。**它与 `L` 无关，只由「每个 KV 元素被多少 query head
   复用」决定。**
3. `AI(MLA) = 2·H_q·(2·d_c+d_r)/(b·(d_c+d_r)) = 2·128·1088/(2·576) ≈ 241.8`。
   比 GQA-8 高一个数量级，是因为分母只有**一份**潜向量（`H_kv` 等效为 1），
   而分子里 128 个 head 都在复用它。
4. 令 `AI=333`：`H_q = 333·b·(d_c+d_r)/(2·(2·d_c+d_r)) = 333·2·576/(2·1088)
   ≈ 176`。128 个 head 落在平衡点之下一点——**账面仍是 memory-bound，
   但已经贴着脊线**。这不矛盾：账本只比较「FLOP 与 HBM 字节」，
   而真实 kernel 还花在 softmax、shared memory 与跨 warpgroup 交换上；
   官方 README 的结论（fp8 没比 bf16 快）正说明这些开销才是当时的上限。

**常见错误**：把 MLA 的 AI 写成 `2H/(1·b)`（漏掉潜向量与 per-head 维度的差别）。

---

## E7 代码阅读题 ★★

**参考答案**：

1. 语义是 `acc_s = Q@KV^T + Q_pe@K_pe^T`，即「潜向量部分 + 解耦 RoPE 部分」的
   分数之和。不写成一个 K=576 的 GEMM，是因为缓存里本来就是两份独立张量
   （512 的潜向量与 64 的 rope key），而且 576 不是 2 的幂，K 方向的对齐与
   swizzle 都不友好。
2. `FullCol`（`tilelang/tileop/base.py:5`，值 2）表示把 warp 全分到**列**方向，
   即每个 warpgroup 负责输出矩阵的一部分列。这正是「`acc_o` 太大、要按 `dim`
   切开、两个 warpgroup 各算一半」的实现方式。
3. `glse` 存每个 split 的 log-sum-exp（在线 softmax 的统计量），
   `Output_partial` 存每个 split 的部分输出。第二个 kernel（第 96 行起）
   按 `glse` 重新缩放并加权求和，得到最终输出。用两个 kernel 而不是原子加，
   是为了**确定性**：原子加的顺序不定，而两阶段合并的顺序是固定的；
   代价是多一份 `Output_partial` 的显存。
4. `num_split=1` 时这两个 global 缓冲仍然「有用」（代码路径统一、不需要分支），
   只是不带来并行度。官方保留两个版本是因为 split 路径与 non-split 路径的
   grid 形状、缓冲分配不同，合在一起反而更难读、也更难让编译器优化——
   `if num_split > 1: return main_split else: return main_no_split`（第 178-180 行）。

**常见错误**：把两次 `T.gemm` 写进同一个 `acc_s` 理解为「覆盖」（实际是累加，
第一次前要有 `T.clear`/`clear_accum=True`）。

---

## E8 概念题 ★★

**参考答案**：

1. 友好：
   * KV 字节只有 MHA 的 1.76%（`code/03`），长上下文下决定「能不能装下」；
   * 一份潜向量服务 `H_q` 个 head，decode AI 从 1.0 抬到 241.8；
   * 每个 token 的 KV 是连续的 `512+64`、单 kv-head，天然适配 paged/流式读取；
   * K 维可以拆成 512 与 64 两段（第 65-66 行），避开非 2 的幂宽度。
2. 不友好：
   * head 维 576/512 → `acc_o` 吃 128 个寄存器/线程；
   * `wgmma` 要求 M ≥ 64 → 不能靠缩小 `block_H` 减 `acc_o`；
   * 只能按 `dim` 切（`FullCol`），且两个 warpgroup 都要完整 `acc_s`，要过一次
     shared memory；
   * 片上容量紧张，MI300X（约 64 KB）上必须把 Q 放寄存器、降流水级数。
   官方对策：`policy=FullCol` 切 `acc_o` + 走 shared 交换 `acc_s` +
   （AMD 上）Q 放 fragment。
3. 不能。`wgmma.mma_async` 的最小 M 是 64（4 个 warp 组成一个 warpgroup），
   `block_H=32` 会掉到 `mma.sync` 或直接不被支持，性能反而下降。
4. SM90 上 fp8 的 `wgmma.mma_async` 没有 `imm-trans-a/b` 操作数，要求两个
   操作数都是 K-major；PV 乘积要在 `block_I` 上收缩，而 `KV_shared` 是
   `[block_I, dim]`——所以要么把 V 在 shared 里转置成 K-major，要么退回
   `mma.sync`。官方记录的量化后果是整个 kernel **慢 1.81×**
   （`examples/deepseek_v32/README.md:189`；该数字是该 example 在 H800 SXM 上的
   测量，不是本书测量）。

**常见错误**：把「Q 放寄存器」当成 NV 上的通用做法（它是 AMD/MI300X 因容量不足
才采取的对策）。

---

## E9 代码阅读题 ★

**参考答案**：

1. 把 float 映射成**保序的 uint16**：正数按位反转符号位设 1，负数取反；
   这样浮点的可比顺序变成无符号整数的可比顺序，可以用整数基数排序来处理。
2. 只取高 8 位是为了把 65536 个桶压到 256 个，第一遍只需 256 个桶的直方图。
   代价是阈值一样本内部无法区分——官方补救办法是对「阈值桶」里的元素
   **再做一轮 8-bit 精化**（原文 Stage 2：多轮 8-bit radix，逐步取更高位）。
3. 输出的是**位置**（`index[bx, pos] = input_idx`）。因为下游的 sparse MLA 要
   拿这些位置去 gather KV 行（`KV[b_i, Indices[...], g_i, :]`），而不是要
   logits 的值；值只在选择阶段有用。
4. 证据（任取两条）：(a) 用 `T.atomic_add(s_histogram[bin], 1)` 建直方图，
   再做前缀和/后缀计数找阈值；(b) 判断 `s_histogram[tx] >= topk and
   s_histogram[tx+1] < topk` 来找阈值 bin；(c) 用 `atomic_add` 抢占输出位置。
   三者都是排序/统计操作的典型形态，没有矩阵乘。

**常见错误**：以为 `convert_to_uint16` 是量化（它是**保序映射**，不是精度转换）。

---

## E10 概念题 ★★★

**参考答案**：

1. indexer 每个 query token：`2·I·di·L` FLOP（MQA，KV 只算一次）；
   被选中的 attention：`2·H_q·(2·d_c+d_r)·topk` FLOP。
   相等 → `topk = L · I·di / (H_q·(2·d_c+d_r))`。
2. 系数 `= 64·128 / (128·1088) = 8192/139264 ≈ 0.0588`；
   `topk = 0.0588 × 131072 ≈ 7708`。配置里 `index_topk=2048` 远小于它，
   所以此时 indexer 的 FLOP（2.147 GFLOP）是被选中 attention（0.570 GFLOP）的
   **3.76 倍**。稀疏不省 FLOP。
3. 字节（fp16）：dense `= L·(d_c+d_r)·2 = 151 MB`；indexer `= L·di·2 = 33.6 MB`；
   sparse `= topk·(d_c+d_r)·2 = 2.36 MB`。
   稀疏/稠密 `= (33.6+2.36)/151 ≈ 0.24×`。
4. 稀疏 attention 省的是**KV 的访存字节**（这里约省 4 倍），不省 FLOP——
   它把「读完整 KV」换成「读索引 + 读少量 KV」，而 indexer 自己还要扫全序列。

**常见错误**：只算 attention 那一项就下结论「稀疏更快」；忘记 indexer 的
KV 也要读一遍（`di=128` 虽然不是 576，但也是全序列）。

---

## E11 动手题 ★★

**参考答案**（均来自 `code/` 的实际输出，可在任意机器复算）：

1. 三组分别是 **1365.3**（T=8192,E=8,k=4 → compute-bound）、
   **11.9**（T=128,E=64,k=6 → memory-bound）、
   **175.5**（T=8192,E=256,k=6,EP=16 → memory-bound）。
2. `tokens_for_expert` ≈ **425** 才跨过 333 的平衡点（表里 128 时 118.2、
   512 时 384.0，解方程 `1536n/(1536+n)=333` 得 425），与 E2 手算一致。
3. `block_M=32`：最热/平均 = 53/16 = **3.31×**，padding 浪费 **150%**，grid 10 块。
   改成 64：浪费 **300%**，grid **8** 块；改成 16：浪费 **75%**，grid **14** 块。
   规律：**block_M 越大，grid 越少但 padding 越多；越小则相反**——这正是
   §6.3 说的「两头夹」。
4. MLA/MHA = **1.76%**（`code/03` 的 A 部分）；它是结构常数，**与 `--seq` 无关**
   （`--seq` 只改变绝对字节数）。

**常见错误**：以为「grid 少」就等于「快」（padding 出来的块也要跑完 kernel）。

---

## E12 动手题 ★★★（GPU，可选）

**思路**：本题的关键不是数字，而是**把口径写全**并解释曲线的形状。

**参考答案**：

1. （需要真机）记录 `Latency` 与 `TFlops`，并附 `batch` / `kv_ctx` / `dim` /
   `pe_dim` / `block_N` / `block_H` / `num_split`。本机无 GPU，**未运行**。
2. 预期形状：`num_split` 从 1 增大时延迟下降（并行度上升），到某个值后持平甚至
   上升（合并 kernel 的开销与重复读取 Q 的成本开始占主导）。**拐点必须在真机上
   测，本书不给数字。**
3. 口径清单（第 14 章 7.1 节）：dtype（输入/累加/输出）、shape（B、H、kv_ctx、
   dim、pe_dim、paged？）、epilogue（是否含输出上投影）、计时后端
   （`event` / `cupti` / `cudagraph`，含不含 launch）、预热与重复次数（目标毫秒）、
   是否 flush L2、取 mean/median/min、正确性判据（rtol/atol 与 golden）。
   缺任何一项，两个数字就不可比。
4. 用 `ncu` 看 register/thread：应大于 128（因为除了 `acc_o` 还要装
   Q/K/V 的 fragment、地址、循环变量），但不应出现 spill（`ld.local`）。
   差异来源：`acc_o`（128）+ `acc_s`（16）+ Q/Q_pe/B_shared 相关的寄存器 +
   编译器分配的临时量。

**常见错误**：只报 TFlops 不报 shape；把 `min` 和 `mean` 混用。

---

## E13 动手题 ★★★（NPU，可选）

**参考答案**：

1. 「段 → kernel」对照（
   `_vendor/tilelang-ascend-v0.1.2.000/examples/fusedmoe/example_fusedmoe.py`）：

   | 本章的段 | 昇腾 example |
   | --- | --- |
   | 共享专家（第 7 行「Dense 投影」） | `single_gemm_kernel`（第 39 行）/ gate+up 用 `dual_gemm_kernel`（第 89 行）+ `single_gemm_kernel_fixed`（第 780 行）做 down |
   | 路由专家 grouped GEMM | `grouped_gemm_kernel`（第 146 行）/ `grouped_dual_gemm_kernel`（第 212 行）→ `grouped_dual_gemm_kernel_fixed`（第 886 行） |
   | 重排 | host 侧 `host_preprocess`（第 1184 行）产出 `block_metadata`；核内的 permute 在 `examples/moe_token_permute/` |
   | 合并 | routed 路径在 `routed_expert_kernel` 里用 `output_gm * expert_weights` 完成加权；跨 rank 的归约在 `dispatch_combine` |

2. 三处显式约束：
   * `block_M=64 / block_N=256 / block_K=128` 三个模块级常量（第 25-27 行）
     ——受 L0C 与 L1 容量约束；
   * 注释 `Capacity ... L1 = 144KB (< 512KB); L0C = 2*(64*256*4) = 128KB
     (= 128KB, at limit)`（第 228-230 行）——L0C 恰好打满；
   * `SIG_L0AB / SIG_L0C / SIG_K_L1 ...` 等 flag id（第 336 行起）
     ——developer 模式下手写流水的同步资源编号（每核可用的同步 id 有限）。
3. 前置条件：Atlas A3 系列（800I A3 / 800T A3）、Linux ARM、Ascend HDK
   25.3.RC1.2 + CANN 8.2.RC1.alpha003 以上（D2D 功能）、额外编译 shmem
   （`install_ascend.sh --enable-shmem`）。README 明确写 **“Graph mode is not
   currently supported”**：ACLGraph 是把 kernel 序列整体入图来消启动开销的，
   不支持图模式意味着这部分开销只能靠别的手段摊（例如 persistent kernel），
   对推理部署的端到端延迟有直接影响。
4. 方案模板（不运行）：

   ```text
   基线：AscendC 手写算子（若可得）或 torch_npu 同类算子；写明版本与来源
   对比：TileLang-Ascend 的 grouped_dual_gemm_kernel_fixed（同一 example）
   shape：至少 4 组，覆盖「每专家 token 数」从个位数到数百（如 T=1/128/1024/8192）
   dtype：fp16 输入 → fp32 累加 → fp16 输出；固定是否含 epilogue（本题：不含）
   测量：msprof 的 device Task Duration；丢第一次启动（冷启），其余取平均
   报数：正确性用 torch.testing.assert_close 对同一 golden（rtol/atol 明确）；
        性能报等效 TFLOP/s（2*M*N*K 口径，写明 padding 是否计入）
   记录：block_M/N/K、core_num、是否用 combineCV、block_metadata 的 padding 量
   ```

**常见错误**：拿带后处理的 example 去比纯 GEMM；把「精度通过」当成「性能对齐」。
