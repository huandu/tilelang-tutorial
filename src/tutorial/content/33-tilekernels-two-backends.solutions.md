# 第 33 章 习题答案

说明：以下每题给“解题思路 + 参考答案 + 常见错误”。引用正文时用“§N.M”，
引用语料时给 corpus 路径与行号（相对于
`../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/`），
引用锁定源码时给 `_vendor/tilelang-v0.1.15/` 路径与行号。
凡是需要真实硬件才能得到的数字，答案只给**判据与流程**，不给具体测量值。

---

## E1

**思路**：§2.2 的两张表就是答案的判据，
外层回答“要不要算、算什么形状、用哪块芯片”，内层回答“这块芯片上怎么算”。

**参考答案**

1. `assert ... is_contiguous()` → **外层 `*_kernel.py`**。这是 API 契约的一部分
   （`topk_gate_kernel.py:26`），两个后端共享。
2. `if is_ascend(): from ..._asc import ...` → **外层**，而且是“后端分派”这件事的
   本体（`topk_gate_kernel.py:35-36`）。它必须在外层，因为内层不该知道另一种芯片存在。
3. `T.alloc_reducer` → **内层 `*_cuda.py`**（`topk_gate_cuda.py:28`）。
   它是 CUDA 侧的归约实现方式；昇腾侧用的是 `S.vcmax`/`S.vcmin`。
   注意：如果某个原语两个后端都支持（如 `T.copy`、`T.Kernel`），两边都能写，
   **但仍然是各写各的**。
4. `torch.empty` → **外层**（`topk_gate_kernel.py:31`）。分配输出是 API 层的职责。
5. `T.annotate_buffer_versions` → **内层 `*_asc.py`**（`topk_gate_asc.py:35`）。
   这是昇腾特有的缓冲版本声明，CUDA 侧没有对应物。
6. `get_packed_ue8m0_pack_factor()` → **既不是外层也不是内层，而是共享配置模块**
   `config.py`（`config.py:70-75`）。它需要知道“现在是哪块芯片”，
   但它是被两个内层共同调用的**共享算术/配置**，所以放在 `quant/common.py` 旁边一类的位置。
7. `shared_memory_per_multiprocessor` 探测 → **共享配置模块** `config.py:74-77`
   （`get_max_smem_per_sm`）。理由同上：它是“这台机器有什么”的事实，
   不是“怎么算”。

**常见错误**：把 6、7 答成“外层”。外层是**每个算子**一个的文件，
而探测值要跨算子复用，所以它们住在共享的 `config.py`。另一个常见错误是
以为“两棵树的共性原语可以只写一份”，只要它在 `T.prim_func` 里，
就必须在两个后端文件里各出现一次。

---

## E2

**思路**：§3.1–§3.2。

**参考答案**

1. 判据是**设备节点 `/dev/davinci_manager` 是否存在**（`config.py:24-26`）。
   用 `lru_cache(maxsize=1)` 是因为它**不可能在进程生命周期内改变**
   （一台机器不会中途变成另一台），缓存一次即可；`maxsize=1` 也顺带保证了
   “同一进程只有一个后端”这个不变量。
2. 昇腾分支在函数内 import，是因为 CUDA 工厂模块在**模块顶层**被导入
   （`topk_gate_kernel.py:4`），而昇腾工厂模块只有在真的要跑昇腾时才需要；
   延迟 import 让“在 CUDA 机器上 import 这个包”不需要经过昇腾专属模块
   （反之亦然）。这是“两条后端路径互不干扰”在 import 层面的落实。
3. 因为昇腾的一个 AI Core 由 1 个 Cube 核与 **2 个 Vector 核**组成
   （第 19 章 §2.1），所以纯向量算子的并行单位是 `cube_core_num * 2`
   （`config.py:70-71`）。错用 `get_num_sms()` 会让 kernel 只开出**一半**的
   Vector 核（每个核要处理的 token 数翻倍），结果正确但吞吐明显偏低，
   而且因为结果对，这种错很难被发现。
4. 因为 `num_vec_cores` 是**编译期常量**：它会参与
   `num_vregs_per_group` / `num_groups` / `num_aligned_experts` 的计算
   （`topk_gate_asc.py:14-18`），还会决定快路径还是慢路径。
   当成运行期变量的话，这些常量全部退化，布局无法特化，而且每个新值都可能触发重新编译。
5. 因为 `pdl_sync` / `pdl_trigger` 定义在 **CUDA 方言**里
   （`_vendor/tilelang-v0.1.15/tilelang/cuda/language/pdl.py:10-21`），
   而 `tilelang.language` 通过 `from tilelang.cuda.language import *` 把它带进来
   （`.../tilelang/language/__init__.py:11`）；`tilelang.ascend.language` 建立在
   `tilelang.language.common` 之上，**没有** PDL。所以昇腾分支如果传 `use_pdl`
   会在工厂里就 `AttributeError`。

**常见错误**：第 3 题答成“少了 2 倍核数就会算错”。不会算错，
网格形状与结果无关，只影响并行度；这类“对了但慢了”的错误才是最难查的。

---

## E3

**思路**：§3.3 的表，加上第 19 章 §2.6 的“两棵树”。

**参考答案**

1. 用的是 **M 树（主干）**。证据（任取两处）：
   * `topk_gate_asc.py:2` `from tilelang.ascend import language as T`；
   * `topk_gate_asc.py:31` 用 `T.alloc_shared`（A 树是 `T.alloc_ub`）；
   * `topk_gate_asc.py:35` 用 `T.annotate_buffer_versions`（A 树没有这个原语）；
   * `rand/randn_asc.py:7` `target='ascend'`（A 树是 `'ascendc'` / `'pto'`）；
   * `topk_gate_asc.py:151` 只出现 `S.mem_bar` 而没有 `barrier_all` / `set_flag`，
     说明同步是自动插入的（M 树特性）。
2. （a）它要 **Ascend 950**（`README.md:30` 明写），而 950 是 M 树的验证目标；
   （b）它要 **自动调度**：几十个算子上手写 flag 不可维护，
   而 M 树的 AutoSchedule 默认开启（第 23 章）。
3. 会失败。A 树的 `T` 与 M 树的 `T` 是**两个命名空间**
   （`tilelang.language` vs `tilelang.ascend.language`），
   `T.alloc_L1` 在 M 树里不存在（M 树只有小写的 `T.alloc_l1`），
   `T.gemm_v0` 只在 A 树里，`is_npu=True` 也不是 M 树 `Kernel` 的参数
   （M 树签名是 `Kernel(*blocks, prelude=None)`，
   `_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:139-142`）。
   具体报错形态取决于哪一条先被执行：通常是 `AttributeError`
   （找不到属性）或 `TypeError`（多余关键字参数）。
4. 不矛盾。它只有**一句手写同步**：`topk_gate_asc.py:151` 的
   `S.mem_bar('VST_VLD')`。它保护的是“刚用 `S.vsts` 写进 UB 的值，
   下一轮迭代要 `S.vld` 读回来”这条 SIMD 内部的读写依赖，
   这不是跨 PIPE 的 flag，而是同一条指令流里的写后读，所以必须手写；
   而真正的跨 PIPE 依赖（MTE2/V/MTE3 之间）由编译器的 InsertSync 自动处理。

**常见错误**：第 4 题答成“因为它用了 A 树，所以要什么自己写”。
恰恰相反：**正因为用了 M 树，它才只需要一句手写同步**。

---

## E4

**思路**：逐条对照 §4.4 的表。

**参考答案**

1. **Ascend 版（M 树）**。判据（任三条）：
   （a）`T.Kernel(num_vec_cores)` ，网格是核数，没有 `threads=`；
   （b）`T.alloc_shared`（在昇腾方言里就是 UB，`ascend/language/__init__.py:8-18`）；
   （c）`T.annotate_buffer_versions`（昇腾专属原语）；
   （d）`T.Pipelined(..., annotations={'multi_buffer_eligible': ...})`；
   （e）`row = w * num_vec_cores + core_id` 的按核交错划分。
2. 分配的是 **UB（Unified Buffer）**。Ascend 的 Vector 单元只能直接读写 UB，
   寄存器（fragment）不是 Vector 的地址空间；这也是为什么 `alloc_shared`
   在昇腾方言里被重新指向。
3. 好处是**访存连续**：每一轮迭代里，一个核要搬的是“一整行”连续的专家打分
   （`scores[row, :]`），与 DMA/MTE2 的连续搬运天然匹配。
   相反，如果按 token 连续切给每个核，同一轮里各核访问的是相隔很远的行，
   每个核内部连续、核间不连续，对批量搬运不友好。
   （这是从代码结构推出的论证，**未在硬件上验证**。）
4. 没有对应物。CUDA 版的网格就是 `num_tokens`（`topk_gate_cuda.py:22`），
   一行一个 block，天然整除；而昇腾版的网格是核数，
   `ceildiv(num_tokens, num_vec_cores)` 之后最后一轮可能多出空位，
   所以需要 `row < num_tokens` 守卫（`topk_gate_asc.py:47`）。
5. 大概率仍能编译：AutoSchedule 会自己推断版本数
   （`annotations.py:19-24` 说它在 `T.Persistent` + `num_stages > 1` 时会计算最优份数）。
   但你失去两件事：（a）`invalid_logits_ub` 的“只有 1 份”这条**语义**保证，
   如果它被推断成多份，它就不再是跨迭代共享的累积器，`T.device_assert` 会漏检；
   （b）流水深度不再由你保证。**这两点本章未在硬件或编译器上验证**，
   属于“读文档 + 读代码得出的风险提示”。

**常见错误**：第 5 题答成“会编译失败”。文档说调度器会推断，
所以更可能的现象是**编译通过但语义悄悄变了**。这比编译失败危险得多。

---

## E5

**思路**：§4.7 与 §4.3(b)。

**参考答案**

1. 它把“写一个 int64 下标”拆成“写两个 int32 槽”：
   `T.view` 把 `[T, num_topk] int64` 重新解释成 `[T, 2*num_topk] int32`；
   低位槽写索引（`S.vsts(out_ub[2*k], idx, one, 'ONEPT_B32')`，单 lane 写），
   高位槽保持 0（由 `T.clear(out_ub)` 保证）；最后整段搬过去。
2. `T.view` 的契约（`_vendor/tilelang-v0.1.15/tilelang/language/customize.py:101-140`）：
   （a）shape 与 dtype 必须**保持总位数不变**；
   （b）源必须有**零 `elem_offset`**；
   （c）**紧凑**（不传 `strides` 时源必须是已紧凑布局；非紧凑需要显式 strides）。
   这里 `[T,k] int64` 与 `[T,2k] int32` 位数相同，源是刚 `torch.empty` 出来的连续张量，
   所以满足。
3. 必需，因为高位必须为 0。删掉的话，int64 的高 32 位会是 UB 里的垃圾值，
   索引变成巨大的数字（低位正确、高位随机）。
   在小 shape 的测试里之所以可能看不出来：如果 UB 复用到的恰好是刚被清零的内存，
   高位碰巧是 0，测试就过了；换一个 shape、换一段历史，结果就变。
   这类“依赖未初始化内存”的 bug 是间歇性的。
4. CUDA 版直接 `T.copy(topk_idx_shared, topk_idx[pid, 0], disable_tma=True)`
   （`topk_gate_cuda.py:64`），靠 `T.copy` 在 dtype 不同时逐元素转换
   （`copy_op.py:122-127`）。两者前提不同：CUDA 版依赖 copy 的隐式转换；
   Ascend 版依赖“高位为 0”这个由 `T.clear` 建立的不变量。

**常见错误**：以为 `T.view` 会做转换。它**不做**任何转换，只重新解释同一段内存，
所以“总位数必须相等”是硬约束。

---

## E6

**思路**：把“断言”和“循环体”当成两件互相竞争的目标。

**参考答案**

1. 会失去**循环的规则性**。在 `T.Parallel` 循环体里逐个断言，意味着循环体里
   出现了带控制流的副作用分支，编译器就不能把整段载入向量化/展开成规整访存。
   折叠成一次断言后，循环体只剩“累加 0/1”这一条纯计算，向量化得以保留
   （注释原话：“single deferred assert to keep the load loop vectorized”）。
2. 因为 top-k 的语义依赖“最大值”与“相等”，而 `NaN` 让两者都失去定义；
   同时“越界填 `-inf`”这个补齐技巧也要求**真实元素里不能有 `-inf`**，
   否则无法区分“补齐位”与“真实的最低分”。
   所以：**补齐技巧 + 有限性检查是一对互相依赖的设计**，缺一个就不安全。
3. 昇腾版用 SIMD 位运算做等价的事：
   `S.vsub(logit, logit)`（自己减自己）+ `S.vcmps(..., 0.0, op='ne')` 得到标记，
   再用 `S.vmax` 累积进一个 `T.int32x64` 的向量寄存器
   （`topk_gate_asc.py:66-71`），最后由循环外的 `T.device_assert`（`:155`）统一检查。
   关键原语：`S.vsub` / `S.vcmps` / `S.vmax` / `S.vsts`。

**常见错误**：第 2 题只答“防止 NaN 传播”。更准确的说法是：
**它保护的是 tie-break 语义与 `-inf` 补齐技巧的前提**。

---

## E7

**思路**：§4.5。把“SIMT”与“SIMD”各自展开成一句具体的执行模型。

**参考答案**

1. （a）CUDA：硬件同时执行 32 个**各自独立**的线程，每个线程有自己的寄存器和
   自己的一串标量指令；
   （b）昇腾：一条指令一次作用于 **64 条 lane**，你写的每条原语都是
   “对 64 个元素同时做同一件事”。
2. 会直接报错。M 树的昇腾 `T.Kernel` 签名是
   `Kernel(*blocks: int | PrimExpr, prelude: str | None = None)`
   （`_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:139-142`），
   **没有** `threads` 参数，且只接受一个网格维度；传 `threads=32` 会得到
   `TypeError: Kernel() got an unexpected keyword argument 'threads'`。
   设计意图写在它的 docstring 里：“There is no SIMT thread domain at this scope ...
   Use `T.SimtVF(threads=...)` inside the body to run thread-parallel code”。
3. 用 `with T.SimtVF(threads=...)`。本章的例子：
   `rand/randn_asc.py:34` 的 `with T.SimtVF(threads=512)`，
   以及 `moe/normalize_weight_asc.py:31` 的 `with T.SimtVF(threads=128)`。
4. `vector_length = 64` 定的是**一条 vreg 的宽度**（fp32 lane 数）；
   `max_vregs_per_group = 6` 定的是**一组最多用几个 vreg**，
   于是 `experts_per_group ≤ 6 × 64 = 384`。
   当 `num_groups == 1`（专家数不超过一组能覆盖的范围）走快路径
   （`topk_gate_asc.py:73-105`）；否则走慢路径（`:106-151`），
   每一轮 k 被拆成“逐组求最大值 → 逐组求最小下标 → 逐组屏蔽中奖 lane”三个阶段，
   并因为要“写 UB 再读回来”而在末尾插一句 `S.mem_bar('VST_VLD')`。

**常见错误**：第 2 题答成“会退化成一个线程”。不是退化，是**参数不存在**。

---

## E8

**思路**：§4.6。

**参考答案**

1. `scores_ub` 是 `num_stages`（6）份，因为它是流水数据：
   “搬第 w+5 行、算第 w 行、写第 w-1 行”需要多份同时在飞；
   `invalid_logits_ub` 是 1 份，因为它是**跨所有迭代共享的累积器**，
   它的语义就是“从 kernel 开始到结束有没有出现过非有限值”，不是一个可以并行的流。
2. 调度器求的是**依赖上至少需要几份**（下限），
   而 `num_stages` 表达的是**我想重叠多深**（性能意图）。
   前者可由依赖图推出，后者是人的选择，所以要么由人显式给出，
   要么接受“求解器替你选”。二者不矛盾：一个解约束，一个给目标。
3. 它就不再是“全程一个”的累积器。可能出现“某个版本记录了坏值、
   但最终被读的那个版本恰好是干净的”，于是 `T.device_assert` 漏检，
   也就是**错误被静默地放过**。这正是“版本号是语义而不只是性能”的证明。
4. 因为 CUDA 版根本没有“同一时刻多个迭代在飞”这种结构：
   一个 block 处理一行、做完就退出（`topk_gate_cuda.py:22-66`），
   缓冲区不存在跨迭代复用的时间窗口，也就没有版本数这个自由度。
   **流水是昇腾侧为了掩盖搬运延迟而引入的结构**，CUDA 侧靠网格并行度掩盖延迟。

**常见错误**：第 4 题答成“因为 CUDA 的 shared memory 更大/更小”。
与容量无关，与**结构**有关。

---

## E9

**思路**：`sf_block = (num_per_tokens, num_per_channels)`，
第一个分量是“几个 token 共享一个尺度”，第二个是“几个通道共享一个尺度”。

**参考答案**

1. `(1, 128)` → **per-token**：每行独立，尺度沿**通道**方向按 128 一段变化
   （即 `per_token_cast` 的 `sf_block`，`per_token_cast_kernel.py:28`）。
2. `(32, 32)` → **per-block**：行与列都按 32 分块，块内共享一个尺度
   （`per_block_cast` 的典型配置）。
3. `(128, 1)` → **per-channel**：每列的尺度跨 128 行共享，
   沿**token** 方向变化（`per_channel_cast` 的 `sf_block`，
   `per_channel_cast_kernel.py:53`）。
4. 昇腾侧只支持 `(32, 32)`：`per_block_cast_asc.py:22` 的
   `assert (num_per_tokens, num_per_channels) == (32, 32)`。

**常见错误**：把 `(128, 1)` 说成 per-token。注意第一个分量是 **token 维**：
`(128, 1)` 表示“128 个 token 共享”，所以是 per-channel；
`(1, 128)` 才是“每个 token 独立”。

---

## E10

**思路**：§5.1（打包因子）与 §5.6（两条更多差异）。

**参考答案**

1. CUDA 是 **4**（`torch.int32`），昇腾是 **2**（`torch.int16`）
   （`quant/common.py:70-75`）。
2. 打包因子不同意味着**同一个 `2^e` 序列被切成了不同长度的字**，
   于是 sf 张量的**形状**（以及内存里的字节顺序）都不同。
   直接把 CUDA 的 sf 张量交给昇腾 kernel，读到的是错误的指数，
   表现为整片数值偏差（而不是某个元素错）。所以**必须重新打包**。
3. （a）**`use_e4m3_sf` 只支持 CUDA**：昇腾上开这个选项会命中断言
   （`common.py:118`、`cast_back_kernel.py:41`）；
   （b）**列主序 sf 的对齐要求不同**：CUDA 需要 16 字节对齐（TMA 加载要求），
   昇腾不需要（`common.py:215-217` 的注释）。
   这两条都不会造成数值口径分歧，因为：（a）是“支持与否”的开关，
   不影响已支持路径的算术；（b）只影响**内存布局/分配**，不影响尺度值本身。
   真正的算术只有一份：`get_sf_and_inv`（`common.py:251-282`）。

**常见错误**：认为“打包因子不同”只影响性能。它直接改变**数据布局**，
是正确性问题。

---

## E11

**思路**：SwiGLU 的两半拼接 → 输出减半；记账时统一用“每个元素占几字节”。

**参考答案**

1. SwiGLU 需要两个投影（gate 与 up），实现里把它们**拼在一行**：
   `x[:, :hidden]` 是 gate、`x[:, hidden:]` 是 up，
   逐元素算 `swish(gate) * up` 之后输出只剩 `hidden` 列。
   所以输入宽 `2*hidden`、输出宽 `hidden`
   （`swiglu_forward_kernel.py:24-27`，其中 `hidden = hidden // 2`）。
2. 以 fp16（每元素 2 字节）为基准，把“元素数 × 2B”换成“元素数 × 0.5B”
   （fp4 是每个元素半字节）。方案 A（两个 kernel）：
   读 x（2H 个元素）→ 写 swiglu 结果（H）→ 读（H）→ 写 fp4（H）
   = `2H + H + H + H = 5H` 个元素的访存；
   方案 B（融合）：读 x（2H）→ 写 fp4（H）= `3H`。
   比值 A:B = **5:3 ≈ 1.67**（正文用 fp16 输出时是 4.5H : 2.5H = 1.8:1，
   数量级一致）。**这是本书的记账，不是实测。**
3. 代价是**片上存储压力**：融合版要在 UB 里同时放左右两半的输入、
   输出、fp32 中间值、amax、尺度倒数、sf，还有一条 `use_fast_certificate`
   分支下的第二套缓冲（`swiglu_forward_asc.py:285-311`）。
   UB 容量（库自述 216/248 KB，`config.py:81-83`）因此成为 tile 大小的上界。
4. 因为 SwiGLU 几乎不做乘加以外的计算（每元素几次浮点运算），
   而每元素要搬好几个字节，算术强度很低，属于 memory-bound。
   此时减少访存**直接**等于减少时间。前提是“瓶颈确实在访存”；
   如果某天它变成 compute-bound（例如融合了很重的后处理），
   省访存的收益就会被计算时间掩盖，结论不再成立。

**常见错误**：把“融合总是更好”当结论。融合的收益取决于瓶颈在哪一侧，
而且它**增加**片上存储压力；UB 不够时只能缩小 tile，可能反而变差。

---

## E12

**思路**：读 `rand/randn_kernel.py` 与 `rand/randn_asc.py`，再用 §2 的三层结构解释。

**参考答案**

1. 在 CUDA 路径上它**直接调用 `torch.randn`**
   （`rand/randn_kernel.py:31-32`）。这是合理的，因为：
   （a）文件第一行的注释就写明这个模块是“extremely slow `torch_npu.randn` 的临时替代”，
   动机只存在于昇腾侧；（b）CUDA 上 `torch.randn` 本身已经很快，
   再写一个 kernel 只是徒增维护面。
2. `assert not torch.npu.is_current_stream_capturing()`（`:37-39`）。
   它挡住的错误是：在 NPU 图捕获期间调用 randn。
   因为 `reserve_seed` 在 **host** 上执行（`rand/seed.py:13-20`），
   图捕获会把当时那个种子固化下来，replay 时产生的随机数完全相同，
   训练里会表现为“每一步的 dropout 噪声一模一样”这类极难定位的问题。
   库选择**显式失败**，而不是给出静默错误的结果。
3. 最合理的解释是**收益与维护成本的比较**：
   （a）CUDA 侧没有收益（原生就够快），三层结构里“`torch/` 参考实现 + 原生实现”
   已经覆盖了这个需求；（b）多一个 `_cuda.py` 就多一份要跟着
   `T.rng_*` 语义、dtype 支持、seed 管理一起维护的代码；
   （c）跨后端一致性的要求（同样的 seed 出同样的数）在随机数上是**不可满足**的，
   所以这个算子本来就不适合“同一份 API、两个后端”这个模式。
   于是库里选择了**只在需要的地方写**，并在外层用一句 `if not is_ascend()`
   明确让路。

**常见错误**：第 3 题答成“因为 TileLang 在 CUDA 上做不了 rng”。
两边的 `T.rng_*` 都存在（`cuda/language/random.py` 与
`ascend/language/random.py`），技术上是可行的。这是**工程取舍**，不是能力缺失。

---

## E13

**思路**：先按 §4.4 的表做预测，再运行 `code/03-map-and-diff/map_and_diff.py` 核对。

**参考答案**

1. 预测（要点即可，不要求逐条命中）：
   * `moe/moe_topk_gate_forward`：左边（CUDA）应有 `T.Kernel(num_tokens ...)`、
     `threads=`、`T.alloc_local`、`T.shfl_xor` / `T.warp_reduce_*`；
     右边（Ascend）应有 `T.Kernel(num_vec_cores)`、`T.alloc_shared`、
     `T.annotate_buffer_versions`、`T.Persistent`、`T.SimdVF`、`T.SimtVF`。
   * `quant/per_token_cast`：左边有 `T.Kernel(...)`+`threads=`、`T.alloc_fragment`、
     `T.annotate_layout`、`T.reduce_absmax`；右边有 `T.Kernel(num_vec_cores)`、
     `T.annotate_buffer_versions`、`T.Persistent`、大量 `S.*`。
2. 实际输出（作者运行过）里比较值得注意的两条：
   * `moe/topk_gate`：`显式向量宽度` 是 **cuda 0 : asc 15**、
     `fragment（寄存器）` 是 **cuda 5 : asc 0**、`位视图` 是 **cuda 0 : asc 1**。
   * `quant/swiglu_forward`：`内联设备代码`（`T.import_source`）**只在 cuda 侧出现**，
     `SIMT 线程作用域`**只在 asc 侧出现**。这正是 §7.3 那两个“下沉点”。
   如果预测与输出不符，最常见原因是：把 `T.Parallel` 误当成了 CUDA 专属
   （它在两个方言里都有），或者忘了 Ascend 的 `normalize_weight` 之类算子
   也会用 `T.SimtVF` + fragment。
3. 因为脚本第 2 步只遍历**同时存在 `_cuda.py` 与 `_asc.py`** 的算子家族，
   而 `rand/randn` 只有 `_asc.py`（第 1 步的清单里它的 `cuda` 列是 `—`）。
4. **能**证明：这些 API 名在锁定源码树里存在，并且**名字确实被定义成了函数**
   （而不是某个对象属性或拼写错误）。**不能**证明：签名对不对、参数含义对不对、
   语义对不对、跑起来会不会错。所以它只是“引用可达性”的机械检查，
   真正的 API 结论仍然来自逐行读定义（见 `../references.md`）。

**常见错误**：把这一步当成“代码能跑”的证据。它连 import 都没做。

---

## E14

**思路**：先弄清“要一起参与 FMA 的两个元素在哪”，再看它们在 vreg 里怎么排。

**参考答案**

1. 它们在**把两个操作数重排到同一 lane 位置**：
   旋转是对 (first, second) 两个向量**逐 lane**做 FMA，
   所以两个操作数必须一个在 `first` 的 lane j、一个在 `second` 的 lane j。
   `S.vintlv`（interleave，定义在
   `_vendor/tilelang-v0.1.15/tilelang/ascend/language/simd.py:1052`）与
   `S.vdintlv`（de-interleave，同文件 `:1067`）是这一对互逆的重排指令。
   必须对称，是因为写回时数据要恢复成原来的 lane 顺序
   （否则 `T.copy` 写回 GM 的就不再是原来的张量）。
2. CUDA 版用**两次带不同偏移的访存**直接得到两个配对向量：
   `first_local[i, j] = x[..., vec_offset*vec + j]` 与
   `second_local[i, j] = x[..., half_dim + vec_offset*vec + j]`
   （`rope_cuda.py:98-100`；`interleaved` 时先按 `vec*2` 读再取奇偶，
   `:92-96`）。每个线程处理一个标量，配对关系由**地址**表达，
   所以不需要重排；重排是“一条指令吃 64 个 lane”带来的额外义务。
3. 关键在 `load2` 的 `packed_bf16=True` 路径：它把一整段 128 条 bf16 lane
   用 `S.vcvt(..., part=0/1)` 拆成两个 vreg，而 `part` 的语义是
   `PART_EVEN`(0) / `PART_ODD`(1)（`simd.py:1247-1249`），
   也就是**偶 lane / 奇 lane**，不是前半 / 后半。
   * GPT-J（`interleaved=True`）的搭档是相邻的 (2i, 2i+1)，
     正好就是“偶 / 奇”，所以拆完即可用，**不需要**重排；
   * NeoX（`interleaved=False`）的搭档是 (j, j+half)，与偶奇无关，
     所以要先 `S.vintlv`（把偶奇合回去，得到前后两半），旋转后再
     `S.vdintlv`（拆回偶奇）交给打包写回。
   反过来，fp32 路径（`:79-81`）是两次独立的 `S.vld`，
   天然给出“前半 / 后半”，所以这次需要重排的是 GPT-J：
   `interleaved` 为真时用 `S.vdintlv` 变出偶奇。
4. 没关系。旋转公式一行没变，变的只是**操作数怎么排进 lane**。
   这正是“同一份算法、不同的并行映射”最干净的一个例证。

**常见错误**：把 `part` 当成“前半 / 后半”。
它是偶 / 奇拆分，把这个搞反，就会得出“bf16 路径不需要重排”的错误结论。
