# 第 38 章 习题参考答案

每题给「解题思路 + 参考答案 + 常见错误」。行号都指向 DeepEP-Ascend 快照
commit `3b25377d04b2`；引用格式见 [`references.md`](references.md)。

---

## E1（★）通信量从哪来

**思路**：先算「一个 token 送到一个目标 rank」的 payload，再乘 k。

**参考答案**：

1. **BF16 dispatch 上界**：每 token 每目标 rank 搬
   `hidden×2 + topk×8（int64 索引）+ topk×4（float32 权重）`
   `= 7168×2 + 48 + 24 = 14408 B`；k=6 个不同 rank ⇒ 每 token `86448 B`。
   乘 8192 个 token ⇒ 约 **708 MB/层**。
   （若按第 16 章的 `k·d·b` 粗略口径，是 `6×7168×2=86016 B/token`，
   元数据只占约 0.5%。）
2. **FP8 dispatch 上界**：hidden 变成 1 字节，还要带 SF
   （`ceil(7168/128)=56` 个 fp32 scale = 224 B，`utils/math.py:30-38`）：
   每 token 每 rank `7168 + 224 + 48 + 24 = 7464 B`；乘 6 ⇒ `44784 B`；
   乘 8192 ⇒ 约 **367 MB/层**（约为 BF16 的 52%）。
3. **combine 上界**：BF16 输入 + 权重，`6×(14336+24)=86160 B/token`
   ⇒ 约 **706 MB/层**。
4. **为什么不随 GEMM 调优变小**：dispatch/combine 是**零 FLOP 的搬运**，
   它的大小只由 `T、k、hidden、字节宽度` 决定，与 GEMM 的 tile 尺寸、
   指令选择无关（第 16 章 §2.5–§2.6）。

**常见错误**：把 SF 当成「每个 token 一份」而不是「每 token 每目标 rank 一份」；
忘了 FP8 仍然要传 SF。

---

## E2（★）三个参数各自在防什么坑

**思路**：每个参数都对应一条会出错的假设。

**参考答案**：

1. `num_max_tokens_per_rank` 决定接收 buffer 的容量与布局。所有 rank 都要按**同一套
   偏移**写对方的内存（对称内存，§4.1），容量不一致会让偏移对不上，
   所以必须所有 rank 一致（`README.md:90`）。
2. `expert_alignment` 决定展开后每个专家的行数按多少对齐。后面 grouped GEMM
   要求 M 是 block_M 的整数倍，对齐不一致会让 GEMM 读到错的边界
   （`README.md:131` 的注释「Match the grouped GEMM's token alignment」）。
3. `do_zero_padding` 只保证**对齐空隙**被清零。`do_cpu_sync=False` 时输出按
   **容量**分配，尾部可能是没被写过的内存；`do_zero_padding` 不会把它变成
   有效数据（`DeepEP/README.md:287`）。

**常见错误**：以为 `do_zero_padding=True` 就等于「输出全部有效」。

---

## E3（★★）读 `get_buffer_size_hint`

**思路**：`calculate_buffer_size` 是纯整数运算，直接翻译。

**参考答案**：

1. 取 **dispatch 布局与 combine 布局的较大值**（`csrc/buffers/ep.hpp:596-604`）。
2. 对齐常量是 `kNumAllocationAlignmentBytes`，等于 **2 MB**
   （`common/compiled.hpp:48`）。
3. hint 用 `ceil_div(hidden, 32)`（`:594`），实际运行时是
   `ceil_div(hidden, 128)×2 = ceil_div(hidden, 64)`（`tests/ep/test_ep.py:91-93`），
   相差 **2 倍**。hint **偏大**——这是源码注释里写明的近似
   （`:593`「approximate the SF pack count」），用于安全规划，
   不会影响实际运行的字节流量。`code/03-buffer-size/` 给出了具体数字
   （hidden=7168 时多预留 112 个 pack，共 224 B/token）。

**常见错误**：把 hint 的数字当成实际通信量。

---

## E4（★★）FP8 dispatch 的 payload

**思路**：`view(torch.int16)` 是字节重解释，不是数值转换。

**参考答案**：

1. `ceil(7168/128) = 56` 个 fp32 scale；每个 fp32 是 4 字节，`view(int16)`
   把它拆成 2 个 int16 ⇒ `num_sf_packs = 56 × 2 = **112**`
   （也就是 `7168/64`）。
2. 每 token 的 SF 字节 = `112 × 2 = **224 B**`。
3. **字节重解释**。判据：`56 × 4 = 224` 字节，`112 × 2 = 224` 字节——
   **元素数翻倍、字节总数不变**。数值转换会改变字节数与值域。

**常见错误**：以为 SF 被压缩了。这里 hidden 压缩了（7168→7168 字节的 fp8，
相对 BF16 省一半），但 **SF 本身没压缩**。

---

## E5（★★★）为什么 EP 越大越慢

**思路**：从「字节量」「固定开销」「同步」三个角度各找一条。

**参考答案**（列出四条，答对三条即可）：

1. **去重率下降**：EP8 时 32 个专家在本 rank，token 的 6 个专家更容易落回本地；
   EP128 时只有 2 个本地专家，几乎每个 top-k 专家都在别的 rank 上，
   去重率与本地命中率都下降。依据：`dispatch.hpp:183` 的
   `simt::deduplicate<kNumRanks>` 与正文 §1.3 的 `R·(1-(1-1/R)^k)`。
2. **固定开销随 rank 数上升**：barrier 的源码注释写「this costs 4 us (8 ranks)」
   （`dispatch.hpp:643`），kernel launch 与 SoC 初始化「cost 1.9 us」
   （`dispatch.hpp:573`）。这些开销与 payload 无关，rank 越多占比越高。
3. **上游还没优化完**：README 明确说「Larger EP sizes ... remain under
   optimization」（`README.md:21`），所以下降里有一部分是实现余量，
   不是物理必然。
4. **（加分）padding 更多**：expert alignment=128 在大 EP 下产生更多空行，
   它们不产生有效 payload 却占用 grid 与显存（§2.2）。

**常见错误**：只说「通信量变大」而说不出为什么——关键是**去重率**与
**本地专家数**这两个随 R 变化的量。

---

## E6（★★★）为什么 combine 比 dispatch 慢

**参考答案**：

1. **payload 翻倍**：dispatch 走 FP8（1 字节/element），combine 走 BF16
   （2 字节/element）。同一 token 的 hidden 部分，combine 是 dispatch 的 2 倍
   （`README.md:11`）。这是**网络字节**的原因。
2. **额外的本地归约开销**（本地开销）：combine 要把收到的 k 份 BF16 先写进
   UB 接收环、再读出来做 FP32 累加、再转回 BF16
   （`combine_utils.hpp:56-90`）。这段不占网络，但占流水线时间。
   README 原话「additional local reduction overhead」（`README.md:21`）。
3. **与 HBM 抢带宽**（与 HBM 有关）：URMA 的数据落地要写显存，
   本地归约读显存、写结果也写显存，两者争抢 HBM。
   README 原话「HBM contention with URMA」（`README.md:21`），
   并说华为固件改进就是针对这条。

**常见错误**：只答「combine 是归约所以慢」，不同时指出**payload 翻倍**这一条——
它其实是三者里最容易量化的一条。

---

## E7（★★）读 dispatch 的三段

**参考答案**：

1. **针对「目标 rank」去重**：同一次 token 的多个 top-k 专家如果落在同一个 rank，
   只往那个 rank 传一次。用的是 `simt::deduplicate<kNumRanks>(dst_rank_idx, ...)`
   （`dispatch.hpp:183`）。
2. **数据和元数据是同一条 SQE 里的两个 SGE**：源码注释 `:360-369` 说明
   SGE 0 指向 token 数据（`x`），SGE 1 指向元数据（`metadata_send_buffer`）。
   好处是**一次网络搬运同时带走数据和来源表**，不需要第二趟。
3. **本地 token 走 UB 中转的本地拷贝**：
   `asc_copy_gm2ub_align` → `asc_copy_ub2gm_align`（`:818-838`），
   **不占 URMA 带宽**。

**常见错误**：以为元数据是单独发一次的。

---

## E8（★★）`defer_epilogue` 的两个硬约束

**参考答案**：

1. 必须同时打开 **`async_with_compute_stream=True`**；
   断言在 `csrc/buffers/ep.hpp:398`。
2. 因为通信**之前**排的 kernel 决定了输入何时就绪；`buffer.capture()` 记录
   「输入就绪」事件，作为 `previous_event` 传进去，库才知道可以安全开始搬数据
   （`DeepEP/README.md:289`）。
3. 当**没有别的计算可以重叠**时：延迟 epilogue 只是把「收尾」往后挪，
   却多了一次 `.wait()` 的调度与事件开销，反而更慢。
   README 明确说「Without other work to overlap, call `.wait()` immediately」
   （`DeepEP/README.md:287`）。

**常见错误**：以为「延迟」本身就会变快。

---

## E9（★★）核心数与 handle 的关系

**参考答案**：

1. `num_sms` 在昇腾上选的是 **AI core 数**（`README.md:74`）。
2. 会用 **handle 里存的 32**。cached dispatch 直接把 `num_sms` 覆盖成
   `handle.num_sms`（`deep_ep/buffers/ep.py:409`），combine 也是
   `num_sms = handle.num_sms`（`:566`）。
3. `num_sms=0` 会走 `get_theoretical_num_sms()`，它在昇腾上返回
   `get_num_ai_cores()`（`deep_ep/buffers/ep.py:305-309`），
   也就是当前设备的 AI core 总数。

**常见错误**：以为 cached 调用可以换核数。

---

## E10（★★★）画通信栈

**思路**：按「谁在什么时候被调用」分层。

**参考答案**：

1. **连接建立走 host 侧的 HCCL + HCOMM**：HCCL 注册内存、查拓扑、
   拿对端地址（`hccl.hpp:76-97`、`:184-193`）；HCOMM 建端点、建信道、
   拿 Jetty（`:196-281`）。**数据搬运走 device kernel 里的 URMA**：
   kernel 直接拼 SQE 并敲门铃（`comm/handle.hpp:346-355`、
   `dispatch.hpp:327-429`）。
2. 因为**整个 EP group 是一个 UB 域**：`get_physical_domain_size()` 与
   `get_logical_domain_size()` 都返回 `{1, num_ranks}`
   （`csrc/kernels/comm/api.hpp:82-89`），对应 NVIDIA 的
   「rdma_ranks=1、nvlink_ranks=num_ranks」。所以 scale-out 域大小为 1。
3. 检查在 `csrc/comm/handle.hpp:54-64`：对每个非自身 peer 调
   `HcclRankGraphGetLinks`，要求至少有一条 `COMM_PROTOCOL_UB_MEM` 链路，
   否则 `EP_HOST_ASSERT(has_ubmem and "Ascend currently requires direct UBMEM
   connectivity to every peer")`。

**常见错误**：把 URMA 说成「双边 RDMA」——它的数据路径只有单边 WRITE
（`comm/handle.hpp:257-258`），没有 READ。

---

## E11（★★★）DeepJIT 与 TileLang JIT 的职责

**参考答案**：

| 维度 | TileLang（第 26 章） | DeepJIT（本章） |
| --- | --- | --- |
| 编译的输入 | TileLang DSL，经 TIR/pass 生成的 kernel 源码 | **作者手写的 Ascend C 源码字符串** |
| 缓存键里包含什么 | 源码 + TIR + pass 配置 + 目标等 | 源码 + 被跟踪的 include 树 + 编译器版本 + 生效选项 + 调用方给的依赖签名 |
| host 侧 ABI 谁生成 | TileLang（`MakePackedAPI` 生成 packed function） | **没人生成**；调用方自己传指针与启动参数 |
| 使用者要自己管什么 | 一个可调用的函数、adapter 的框架差异 | 参数校验、张量分配、stream 选择、启动配置 |

**一句话总结**：DeepEP 这类库的 host 侧本来就有一大堆依赖 shape 的逻辑
（`EPBuffer.dispatch` 里几十行断言），而且它的 kernel 是**为某个 shape
现场拼出来的模板实例**，不需要一个通用的 packed 入口；
给它套一层自动 ABI 反而会挡住作者对启动参数（`num_blocks`、
`num_ubuf_bytes`、`jetty_ptrs`）的控制。

**常见错误**：把两者说成「谁更好」。它们解决的是**不同层次**的问题。

---

## E12（★★）选型题

**参考答案**：

1. **Bucket all-reduce**：**不能**。Bucket 只实现了 batched all-gather；
   reduce-scatter / all-reduce 的昇腾 kernel「still being built」
   （`README.md:80`）。替代：用 all-gather 的结果在本地做归约
   （或等上游补齐）。
2. **动态冗余专家**：**不能端到端做**。`lb_prefetch_weights` /
   `lb_reduce_grads` 的接口已按 V2.5 对齐，但昇腾通信 kernel 未实现
   （`README.md:81`；`csrc/buffers/ep.hpp:79`、`:85` 直接
   `EP_HOST_UNREACHABLE`）。EPLB 只能给出**规划**（`EPLB/README.md:35`），
   搬运这一步缺失。
3. **把 decode 的 EP 通信 capture 进图**：**不能**。README 明确
   「graph capture remain unsupported」（`README.md:84`）。
   替代：走 eager 路径，用 `defer_epilogue` 与算子的融合来压低开销。
4. **7 张卡**：**不能**。多卡通信要求 rank 数为偶数
   （`README.md:29`；`csrc/kernels/comm/hccl.hpp:85` 断言 `num_ranks % 2 == 0`）。

**常见错误**：把「接口存在」当成「功能可用」——第 2 问就是这个陷阱。

---

## E13（★★★）复算「90–95%」的含义

**参考答案**：

1. 表里的 GB/s = **有效 payload 字节 ÷ 时间**，payload 口径是
   `num_dispatch_urma_bytes = num_recv_tokens × (count_bytes(x, topk_idx, topk_weights)/num_tokens)`
   （`tests/ep/test_ep.py:404-406`）。它**不含 padding**，
   计时也**不含 barrier**（barrier 是 benchmark 的独立阶段）；
   README 还说计时含 issue 与 drain、不含最后的 epilogue（`README.md:21`）。
2. 除以 0.90 与 0.95：
   * EP8：393–417 GB/s
   * EP16：366–391 GB/s
   * EP32：353–378 GB/s
   * 并集 ≈ **353–417 GB/s/rank**（量级 350–420）。
3. 不能当硬件规格的两个理由：
   * README 原文是 "roughly 90–95%"，比例本身是模糊的；
   * corpus 里没有 Ascend 950 UBMEM 链路的官方带宽规格，无法交叉验证
     （见 `references.md` 的「待补资料」第 1 条）。

**常见错误**：把 373–375 直接当成「链路带宽」而不做除法；
或者忘了带宽是**每 rank** 的量。

---

## E14（★★）TileLang 主线

**参考答案**：

1. * `topk_idx`、`topk_weights`：`tile_kernels/moe/topk_gate_kernel.py` 与
     `tile_kernels/moe/moe_topk_gate_forward_kernel.py`。
   * `x`（FP8 数据 + SF）：`tile_kernels/quant/per_token_cast_kernel.py`
     （`topk_gate_kernel.py:7-43`、`moe_topk_gate_forward_kernel.py:9-53`、
     `per_token_cast_kernel.py:9-60`）。
2. 因为 dispatch/combine 的核心不是「在 AIV 上做 tile 级计算」，而是
   **直接写 URMA 队列、操作 Jetty、逐 lane 拼 64 字节的 SQE**
   （§4.3、§5.2）。这属于「按字节/按协议操作硬件」的领域，
   TileLang 的 tile 级数据流抽象在这个层次帮不上忙——
   与第 35 章「瓶颈在通信与协议时，编译器路线收益有限」是同一结论。
3. MegaMoE 把 **dispatch + 两个 grouped GEMM + SwiGLU + combine** 融进
   一个 kernel（`DeepGEMM-Ascend/.../mega_moe.hpp:26-33`）。
   两条路的适用场景：**算法还在动 / 要复用 / 要接不同模型 → DeepEP-Ascend**；
   **形状冻结、要把通信压到最低 → MegaMoE**（代价是核数写死 32、形状提前注册，
   见第 35 章 §4.5）。

**常见错误**：以为 TileLang 与 DeepEP 是对立的——它们是**分工**，
接口就是三个张量（`topk_idx` / `topk_weights` / `x`）。
