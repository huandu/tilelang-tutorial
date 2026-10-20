# 第 38 章 DeepEP-Ascend：专家并行通信与计算重叠

## 学习目标

读完本章，你应当能够：

1. 用第 16 章的账本口径算出 MoE **一层** dispatch + combine 的通信字节数，
   说清它为什么随专家并行（EP）规模变成核心成本，以及 FP8 dispatch 把这个数字
   压掉了多少。
2. 逐行解释 `DeepEP-Ascend/README.md` 里那段 `EPBuffer` 用法：构造参数、
   `get_buffer_size_hint()`、`destroy()`、缓存 handle（cached handle）、
   expert padding、`defer_epilogue` 与 `async_with_compute_stream` 各自表达什么。
3. 说清 FP8 dispatch 为什么省带宽、**row-major scale** 的粒度与含义，
   并与第 27 章的 block scaling 口径对齐；能指出它和第 35 章 SF 打包的异同。
4. 画出 HCCL / HCOMM / UBMEM / URMA 在这个库里的分工，说明为什么昇腾超节点
   用 UBMEM + URMA 而不是纯 RDMA。
5. 逐项对照第 26 章 TileLang 自己的 JIT / ABI 与 DeepJIT 在**职责**上的异同，
   并说出 DeepJIT 对使用者的四个实际影响（首次调用编译、共享缓存、
   `ASCEND_HOME_PATH` 与 Bisheng、`EP_JIT_*` / `DJ_JIT_*`）。
6. 会读 128 卡超节点的带宽表：说出测量条件，解释为什么 EP 越大越慢、
   为什么 combine 比 dispatch 慢，以及「可持续 dispatch 达到物理 payload
   带宽上限的 90–95%」这句话该怎么读。
7. 说清 TileLang 这条主线在本章的两处落点：**通信库自己一行 TileLang 也没有**，
   但它两端的算子（MoE 路由、量化）是 TileLang 写的。

## 前置知识

* 第 16 章：MoE 的四段结构（router / dispatch / grouped GEMM / combine）、
  `bytes = 2·T·k·d·b` 这套字节账，以及「跨卡时每 token 每方向 `k·d·b` 字节」的
  结论。本章的 §1 直接在这张账本上继续算，不再重复定义。
* 第 26 章：TileLang 自己的运行时与 JIT——`@tilelang.jit` 的生命周期、
  三层缓存、host packed ABI、adapter 与框架的转接。§6 会把这条链与 DeepJIT
  逐项摆在一起。
* 第 27 章：低精度与 block scaling。本章只用到两个结论：**scale 的粒度、
  以及 scale 的布局（行主序 / 分组主序）会直接改变数值结果**，
  以及 E8M0 的语义（只存指数，只能是 2 的整数次幂）。
* 第 35 章：MegaMoE 把 dispatch → grouped GEMM → SwiGLU → combine 融进一个
  kernel。§9 会把「独立通信库 + 外部 GEMM」与「全融合」两条路摆在一起比。
* 第 32 章：本次开源的六个组件与版本核验方式。

不需要你会写 Ascend C，也不需要你有昇腾硬件。本章有两段 C++ 需要读
（§4 的 dispatch 主循环、§6 的 JIT 调用点），都可以只靠源码读懂。

---

## 本章的验证边界（务必先读）

写作环境是 macOS (Apple Silicon)、Python 3.11，**没有 NVIDIA GPU，也没有昇腾 NPU**
（见 [`../../VERSIONS.md`](../../VERSIONS.md)）。因此本章遵守四条硬规矩：

1. **所有代码结论一律标注「静态阅读」。** 引用的 DeepEP-Ascend 源码逐行读过，
   行号与文件内容一致；但**没有一行 Ascend C 被编译过、没有被任何 NPU 执行过**。
2. **性能数字只有一个来源：README 的性能表**（`DeepEP-Ascend/README.md:13-21`）。
   本书没有任何实测数字。凡是「本书复算」的地方，都出自 `code/` 下的纯 Python
   程序，能在写作机上跑出结果，正文会明确标注。
3. **通信栈的头文件不在 corpus 里。** `hccl/hccl.h`、`hcomm/hcomm.h`、
   `kernel_operator.h`、`c_api/*`、`simt_api/*` 都由 CANN 提供，
   快照里只有 `#include` 和调用点。所以 §5 只根据 **README 的措辞**与
   **DeepEP-Ascend 源码里怎么用它们**来描述分工，**不对 CANN 的内部实现下结论**。
4. **硬件规格缺失的地方要留白。** 本章会反推一个「物理 payload 带宽上限」的量级，
   但 corpus 里没有 Ascend 950 UBMEM 链路的官方带宽规格，
   这个反推写进 `references.md` 的「待补资料」，并标注为区间估计。

三个记号先约定好：

| 记号 | 指什么 |
| --- | --- |
| `DeepEP-Ascend/…` | corpus 里的 `../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/…`；站点会把它换成上游 GitHub 链接（commit `3b25377d04b2`） |
| `DeepEP/…` | NVIDIA 侧对照物 `…/raw/DeepEP/…`（commit `93eb6eb23812`） |
| `DeepJIT/…` | corpus 里的 `…/raw/DeepJIT/…`（服务器端主分支快照，见 `PROVENANCE.md`） |

**本章与第 35 章的关系**是镜像的：第 35 章讲「昇腾上怎么把 GEMM 压到极限」，
落脚在 `hc_prenorm.py` 这个 TileLang kernel；本章讲「昇腾上怎么把 all-to-all
压到极限」，落脚在**一个不用 TileLang 的通信库**。两章合起来回答同一个问题：
**一个生产级栈会在哪里用 TileLang，在哪里绕开它。**

---

## 1. 问题的由来：省下来的算力，被 all-to-all 吃回去

### 1.1 一句话定位

第 16 章说过：MoE 用「每个 token 只激活 k 个专家」把 FLOP 降下来，代价是引入
两段**零 FLOP 的搬运**——dispatch（把 token 送到专家所在的卡）和 combine
（把结果送回原卡）。当专家被切到不同 rank 上（专家并行，EP），这两段搬运
就变成了**跨设备的 all-to-all**。

这就是本章要处理的东西。DeepEP-Ascend 的自我定位只有一句话
（`DeepEP-Ascend/README.md:3`）：

> 面向华为昇腾 NPU 的高性能机器学习训练与推理通信库，提供 MoE dispatch/combine
> 的专家并行（EP）all-to-all 操作，支持 FP8 dispatch 和延迟 epilogue。

请注意这份定位里没有出现的词：**没有「TileLang」**。这是一条重要的线索，
§9 会回到它。

### 1.2 先把账算出来

用第 16 章的口径：一层 MoE，一个 token 被路由到 k 个专家，hidden 维是 d，
每元素 b 字节。**跨卡时每 token 每方向 `k·d·b` 字节**（第 16 章 §2.6 第 3 条）。
现在把它放到本次发布所用的规模上：

| 量 | 取值 | 出处 |
| --- | --- | --- |
| 每 rank token 容量 T | 16384 | `DeepEP-Ascend/README.md:11` |
| hidden d | 7168 | 同上 |
| top-k | 6 | 同上 |
| 总专家数 | 256 | 同上 |
| EP 规模 | 8 / 16 / 32 / 64 / 128 | `README.md:13-19` |

把 `k·d·b` 代进去，一个 token 在 dispatch 方向上传出去的**有效载荷**是：

```text
FP8  dispatch：6 × 7168 × 1          = 43008 字节 ≈ 42.0 KB
BF16 dispatch：6 × 7168 × 2          = 86016 字节 ≈ 84.0 KB
BF16 combine ：6 × 7168 × 2 + 6×4    = 86040 字节 ≈ 84.0 KB   （含路由权重）
```

再乘上容量 16384 个 token，**一层**的字节量就到 GB 级别。而同一层的 grouped
GEMM 在默认形状下只有 ~2.9 TFLOP（第 16 章 §2.4 的账）：在一个平衡点 333
FLOP/Byte 的机器上，2.9 TFLOP 对应的「算力等价访存」约 8.7 GB。
**也就是说，dispatch + combine 的通信量和这一层 GEMM 的访存是同一量级。**

### 1.3 为什么这个成本随 EP 规模「越来越不划算」

上表的数字还不完整，因为它假设「一个 token 恰好去 k 个不同的 rank」。
真实的 EP 里有两件事会改变这个数字：

1. **去重**。如果同一个 token 的 k 个专家里有几个落在同一个 rank，它对这个
   rank 只传一次。所以一个 token 实际覆盖的**不同目标 rank 数**是
   `R·(1-(1-1/R)^k)`（R 个 rank、均匀路由的期望），随 R 增大而增大。
2. **本地专家不占网络**。落在本 rank 的专家走本地拷贝，不占 URMA 带宽
   （`DeepEP-Ascend/deep_ep/include/deep_ep/impls/ep/dispatch.hpp:436-445`
   的元数据本地拷贝与 `:806-841` 的 hidden 本地拷贝流水）。

把这两条合起来看，EP 越大，一个 token 越可能被拆到更多 rank 上，
跨设备字节量越大；同时每个 rank 上的专家越少（128 卡、256 专家时只有 2 个），
每张卡的**算力摊薄**，通信占比反而更高。这正是 128 卡超节点上
「算力省下来了、通信吃回去了」的具体形态。`code/02-comm-ledger/` 会把
这两条算成一张表。

### 1.4 本章要回答的六个问题

1. **接口**：`EPBuffer` 长什么样？为什么构造它要指定 token 容量、hidden、top-k？
2. **数据流**：一次 dispatch / combine 在设备上到底发生了什么？
3. **精度**：为什么 dispatch 可以用 FP8，combine 不能？scale 放在哪、怎么排？
4. **重叠**：`defer_epilogue` 与 `async_with_compute_stream` 表达的是什么？
5. **底座**：HCCL / HCOMM / UBMEM / URMA 各管什么？
6. **现状**：哪些能力现在能用，哪些还没实现——以及没实现对使用者意味着什么？

---

## 2. 概念与模型：一次 dispatch / combine 到底发生了什么

### 2.1 一次 all-to-all 的两半

MoE 的前向里，通信出现两次（图 1）：

* **dispatch**：每个 rank 持有自己的 token，按路由结果把 token 的行**复制**到
  专家所在的 rank。目标端收集到的 token 是「按专家分组」的，叫
  **展开布局（expanded layout）**。
* **combine**：专家算完之后，把每个 token 的 k 份结果**归约**回它原来的 rank，
  乘上路由权重。

![图 1：一个 token 的 dispatch / combine 数据流](figures/01-dispatch-combine-flow.md)

（源描述见 [`figures/01-dispatch-combine-flow.md`](figures/01-dispatch-combine-flow.md)；
SVG 规格 `src/tutorial/figures/38-deepep-ascend/01-dispatch-combine-flow.json`。）

图里有一个容易被忽略的细节：**dispatch 传的不只是 token 数据**。
接收端要能把结果送回去，所以每个 token 还要带一份「来源信息」：

* `topk_idx`：这个 token 选中了哪些专家（`[num_tokens, num_topk]`，int64）；
* `topk_weights`：对应的路由权重（float32）；
* 源 token 的全局下标（`src_token_global_idx`）。

这三样加起来的字节数写在**每 token 的布局（token layout）**里，
§4.2 会逐字段拆开。先把结论放这里：在 hidden=7168 时它们是 ~300 字节，
相对 7168 字节的 hidden 只是零头，但**它们决定了 combine 能否正确还原**——
第 16 章 §2.5 说的「来源表本身也要存、也要传」，指的就是它。

### 2.2 展开布局与 expert padding

接收端拿到的 token 不是 `[T, d]`，而是「每个 token 的每个 top-k 槽位一行」。
设这个 rank 从所有来源 rank 一共收到 `num_recv_tokens` 个 token（去重后
按 rank 计），那么展开后的行数是

```text
num_expanded_tokens = Σ_e ceil(num_recv_tokens_per_expert[e] / expert_alignment)
                                                       × expert_alignment
```

**为什么要 padding 对齐？** 因为下一步是 grouped GEMM，它的 M 必须是
block_M 的整数倍。第 16 章 §2.3 第 2 条讲过同一件事（官方例子用
`padding_M = 128`）。DeepEP-Ascend 把它做成一个参数 `expert_alignment`，
README 的注释是「Match the grouped GEMM's token alignment」
（`README.md:131`）。

两个后果：

1. **padding 是浪费的显存与 grid**，但换来的是 GEMM 的规整 grid。
   这是第 16 章 §2.6 第 4 条「不均衡与 varlen 直接变成浪费」的具体实现。
2. **padding 槽位里是脏数据还是零，取决于 `do_zero_padding`**。
   `do_zero_padding=True` 才会把对齐空隙清零（`README.md:134`）；
   在 `do_cpu_sync=False` 的模式下，尾部未用的容量**不会被清零**，
   README 明确提醒「does not make unused output capacity valid data」
   （`DeepEP/README.md:287`）。

### 2.3 combine 是 dispatch 的逆；两个反向正好互换

MoE 的前向是

```text
dispatch → 专家 GEMM → combine
```

反向传播时，梯度按相反顺序穿过这段链（`README.md:140-172`）：

* **dispatch 的反向是一次 combine**（把发出去的梯度收回来）；
* **combine 的反向是一次 dispatch**（把回来的梯度再发出去）。

所以 `EPBuffer` 的两个方法互为反向，而且**复用同一个 handle**：
前向 dispatch 产生的路由信息（谁去了哪）保存下来，反向直接拿来用，
不需要重新算一遍路由。这就是 handle 存在的第一个理由。

### 2.4 handle 是什么，为什么可以缓存

`EPHandle` 是 dispatch 返回的一包**路由结果**（`deep_ep/buffers/ep.py:21-51`），
里面装着：

* 去重后的每 rank 接收计数与前缀和（`psum_num_recv_tokens_per_rank`）；
* 每个本地专家的接收计数、对齐后的前缀和（`psum_num_recv_tokens_per_expert`）；
* 每个 token 每个 top-k 槽位在目标 buffer 里的**槽位下标**
  （`dst_buffer_slot_idx`）与 SGE 下标（`dst_gsge_idx`）；
* 回传元数据 `recv_src_metadata`（`[num_recv_tokens, num_topk + 2]`）。

一旦有了它，下一次 dispatch 只要**输入形状不变**，就可以跳过布局计算，
直接复用这套槽位分配——README 把这个模式叫 **cached handle**
（`README.md:66`、`README.md:74`）。代码里的入口是
`dispatch(..., handle=handle)`（`deep_ep/buffers/ep.py:327`），
走的是 `cached_mode` 分支（`csrc/buffers/ep.hpp:119-132`）。

**cached handle 有一个前提**：所有 rank 必须用同样的容量、同样的对齐、
同样的 top-k；`deep_ep/buffers/ep.py:412-414` 会断言
`(num_experts, expert_alignment, num_max_tokens_per_rank)` 与 handle 一致。

还有一个坑：handle **按引用**持有 `topk_idx`，且不允许在 handle 存活时修改它。
`EPHandle.topk_idx` 这个属性会在读取时检查 PyTorch 的版本计数器
（`deep_ep/buffers/ep.py:94-98`），但**通过 `.data` 或裸指针的写入检查不到**。

### 2.5 训练一步里，通信站在哪里

把 dispatch / combine 放回训练一步里看（图 2）：

```text
attention / MLP 前向 ──▶ dispatch ──▶ 专家 GEMM ──▶ combine ──▶ 输出
                                ▲
                                └── 这一段可以和「共享专家（shared experts）」
                                    或其它独立计算重叠
```

![图 2：训练一步里通信的位置](figures/02-train-step-timeline.md)

（源描述见 [`figures/02-train-step-timeline.md`](figures/02-train-step-timeline.md)；
SVG 规格 `src/tutorial/figures/38-deepep-ascend/02-train-step-timeline.json`。）

为什么重叠是**必须**的？因为 dispatch 是「同步阻塞」的：专家 GEMM 要等
token 到齐才能开始，而 token 到齐需要等**最慢的 rank**。第 16 章 §2.6 第 3 条
已经点过这条：**最慢的 rank 决定整步时间**。如果不把通信藏到别的计算后面，
这段等待就是纯粹的流水线气泡。

DeepEP 这一代人给出的答案是「**先发起、后等待**」：把 dispatch 立刻下发，
然后去算**不依赖它**的东西（例如共享专家），需要结果时再 `.wait()`
（`README.md:175-196`）。这个模式在训练侧还有一个更激进的版本——
DeepSeek-V3 的 DualPipe，把前向与反向的计算-通信整体错开
（`DualPipe/README.md:3`：full overlap of forward and backward
computation-communication phases）。§9.3 会把它与本库的关系讲清楚。

### 2.6 buffer 的生命周期

图 3 把 `EPBuffer` 的一生画出来。要点有四条，都来自 README 的
`Buffer initialization` 一节（`README.md:88-116`）：

1. **每个 EP group 只建一个长生命周期 buffer**，跨层、跨 micro-batch 复用；
2. **所有 rank 必须对 token 容量达成一致**；
3. **增删改 buffer 之前必须等所有在途操作完成**；
4. 用完之后**显式 `destroy()`**，再拆进程组。

![图 3：EPBuffer 的生命周期](figures/03-buffer-lifecycle.md)

（源描述见 [`figures/03-buffer-lifecycle.md`](figures/03-buffer-lifecycle.md)；
SVG 规格 `src/tutorial/figures/38-deepep-ascend/03-buffer-lifecycle.json`。）

---

## 3. 用法：把 README 的样例读成一段能落地的代码

这一节把 README 里的三段代码逐行读一遍。**它们需要真实的多卡昇腾环境**，
本机跑不了（`code/01-ep-minimal/README.md` 写明验证级别是「仅静态核对」）。

### 3.1 构造 `EPBuffer`：参数逐个说清

README 的初始化样例（`README.md:100-113`）是：

```python
ep_group = dist.group.WORLD
max_tokens_per_rank = 4096
hidden = 7168
num_topk = 6
num_experts = 256

buffer = EPBuffer(
    ep_group,
    num_max_tokens_per_rank=max_tokens_per_rank,
    hidden=hidden,
    num_topk=num_topk,
    use_fp8_dispatch=False,  # Reserve enough storage for BF16 and FP8 dispatch.
    explicitly_destroy=True,
)
```

对应到源码，签名其实更长（`deep_ep/buffers/ep.py:175-191`）。这里按
「读者必须先懂的那几个」分组：

| 参数 | 语义 | 为什么需要它 |
| --- | --- | --- |
| `group` | 通信进程组 | 决定 `num_ranks`；buffer 的所有容量计算都按这个规模来 |
| `num_max_tokens_per_rank` | 每 rank 的 token 容量上限 | 决定接收 buffer 的大小；**所有 rank 必须一致**（`README.md:90`） |
| `hidden` | token 的 hidden 维 | 决定每 token 的字节数 |
| `num_topk` | 每 token 选几个专家 | 决定展开行数与元数据大小 |
| `use_fp8_dispatch` | 是否为 FP8 dispatch 预留空间 | `False` 时按 BF16 预留，**已经足够同时容纳 FP8**（样例注释） |
| `explicitly_destroy` | 是否要求手动 `destroy()` | `True` 时析构不自动清理，忘写 `destroy()` 会打印泄漏警告并直接泄漏（`csrc/buffers/base.hpp:19-29`） |
| `deterministic` | 是否使用确定性布局 | 打开后 dispatch 的输出按源 token 排序（`deep_ep/buffers/ep.py:100-160`） |
| `num_bytes` | 手动指定 buffer 字节数 | 不填就用 `get_buffer_size_hint()` 算；必须 2 MB 对齐 |

构造时**不在设备上分配**，而是拼一个 `comm::Context`（`csrc/buffers/ep.hpp:44-58`）：
它先在 HCCL 里注册一块**对称内存（symmetric memory）**，再申请一块
host workspace 用于 CPU 侧同步。构造结尾有两个 `synchronize()` 夹一个
`dist.barrier(group)`（`deep_ep/buffers/ep.py:242-244`）——保证所有 rank
的 buffer 都建好了，任何一个 rank 才能开始通信。

### 3.2 `get_buffer_size_hint()`：先算容量，再决定分配

当你要管一个 buffer 池时，第一个问题总是「要留多大」。库给的入口是
`EPBuffer.get_buffer_size_hint(...)`（`deep_ep/buffers/ep.py:251-275`），
它**不构造 buffer**，只算一遍布局并返回**对齐到 2 MB**的字节数。

它对内部做的事在 `csrc/buffers/ep.hpp:581-605`：分别算

* **dispatch 布局**：`BufferLayout(TokenLayout(hidden, sf, topk), num_ranks, num_max_tokens_per_rank)`；
* **combine 布局**：`BufferLayout(combine_token_layout, min(num_topk, num_ranks), num_max_tokens_per_rank)`；

取**两者较大值**，再向上对齐到 `kNumAllocationAlignmentBytes`（2 MB，
`deep_ep/include/deep_ep/common/compiled.hpp:48`）。

一个容易误读的细节：size-hint 里 SF 的 pack 数用的是
`ceil_div(hidden, 32)`（`csrc/buffers/ep.hpp:594`），而实际运行时
`num_sf_packs = ceil_div(hidden, 64)`（一个 `int16` 装两个 UE8M0，
见 `DeepEP-Ascend/tests/ep/test_ep.py:91-93` 的 `sf.view(torch.int16)`）。
也就是说**这个 hint 为 SF 预留的是实际用量的两倍**，源码注释也写明了
「approximate the SF pack count」（`:593`）。规划容量时可以放心按它来，
但不要把它的数字当成实际流量。`code/03-buffer-size/` 会把这段公式复算一遍。

### 3.3 dispatch / combine 的四个 helper

README 给训练 / 推理各写了一个 helper 对（`README.md:122-172`）。
把它们摊平看，真正的调用只有两个方法：

```python
# dispatch：把 token 送到专家所在 rank
recv_x, _, recv_weights, handle = buffer.dispatch(
    x,                       # bf16，或 (fp8_data, sf) 元组
    topk_idx=topk_idx,       # [num_tokens, num_topk], int64
    topk_weights=topk_weights,
    num_experts=num_experts,
    expert_alignment=expert_alignment,
    do_cpu_sync=do_cpu_sync,
    do_expand=True,          # 昇腾只支持 True
    do_zero_padding=True,
    async_with_compute_stream=True,
    defer_epilogue=True,
)

# combine：把 k 份专家输出归约回原 rank
combined_x, combined_weights = buffer.combine(
    expert_output,           # bf16
    handle=handle,
    bias=bias,               # 可选：融合残差 / 共享专家输出
    async_with_compute_stream=True,
    defer_epilogue=True,
)
```

逐项解释**必须懂的五个参数**：

| 参数 | 语义 | 为什么这么设计 |
| --- | --- | --- |
| `do_expand=True` | 使用「每个 token 每个专家槽位一行」的展开布局 | 昇腾后端**只支持**这个模式（`csrc/buffers/ep.hpp:116`）；NVIDIA 侧还有非展开的 low-latency 模式，昇腾没有 |
| `expert_alignment` | 每个本地专家的接收数向上对齐到这个值 | 让 grouped GEMM 的 M 严格对齐；测试用 128（`README.md:11`） |
| `do_cpu_sync` | 是否同步回 CPU 拿到精确的接收计数 | 训练 / prefill 常开（要精确形状）；decode 关掉，用设备侧计数，**省掉一次 CPU 往返**（`README.md:132`） |
| `do_zero_padding` | 是否清零对齐空隙 | 给 GEMM 一个干净输入；关掉会读到脏数据 |
| `bias` | combine 阶段额外加上的一块 bf16 | 把「共享专家输出」或残差融进 combine，**省一次单独的 elementwise kernel**（`README.md:157`） |

再补两个**必须保持默认**的参数：

| 参数 | 约束 | 出处 |
| --- | --- | --- |
| `do_handle_copy` | 必须 `False`，handle 复制已不支持 | `README.md:72`；`deep_ep/buffers/ep.py:392` |
| `sl_idx` / `num_allocated_qps` | 必须保持 `None` / `0`，显式 RDMA 服务等级与 QP 数不支持 | `README.md:74`；`deep_ep/buffers/ep.py:205-206` |

`num_sms` 这个参数沿用了 NVIDIA 的名字，但在昇腾上选的是 **AI core 数**
（`README.md:74`）：`0` 表示走默认值 `get_theoretical_num_sms()`，
而它在昇腾上直接返回 `get_num_ai_cores()`（`deep_ep/buffers/ep.py:305-309`）。
注意 cached dispatch 与 combine **一律复用 handle 里存的核数**，
传别的值不起作用（`deep_ep/buffers/ep.py:409`、`:566`）。

### 3.4 重叠：`defer_epilogue` + `async_with_compute_stream`

这两个参数一起用，表达的是**「先把通信发出去，先不取结果」**：

* `async_with_compute_stream=True`：这次调用返回一个 `EventOverlap` 对象
  （而不是结果张量），表示「当前 stream 上的工作」；
* `defer_epilogue=True`：**把收尾（epilogue）也推迟**——调用立刻返回，
  但真正的「等接收计数 / 拷贝数据」要等到 `.wait()` 才执行。

README 给出的骨架（`README.md:177-194`）：

```python
pending = dispatch_forward(buffer, x, topk_idx, topk_weights, num_experts, ...)

# 通信在跑，同时算别的（这里：共享专家）
shared_output = shared_experts(x)

recv_x, _, recv_weights, handle = pending.wait()   # 这时才真正取结果
expert_output, expert_ctx = local_experts_forward(recv_x, recv_weights, handle)

pending = combine_forward(buffer, expert_output, handle, bias=shared_output)
# ... 再算点别的 ...
output, _ = pending.wait()
```

代码侧的对应关系：`defer_epilogue=True` 时，C++ 侧不立即跑 epilogue，
而是把它包成一个 hook 返回（`csrc/buffers/ep.hpp:396-402`），
Python 侧在 `.wait()` 时回调（`deep_ep/buffers/ep.py:500-504`）。

**两个硬约束**：

1. `defer_epilogue=True` 必须搭配 `async_with_compute_stream=True`，
   C++ 侧直接断言（`csrc/buffers/ep.hpp:398`）；
2. 如果通信之前已经排了别的 kernel，要用 `buffer.capture()` 先把
   「输入就绪」的事件记下来，作为 `previous_event` 传进去。

**代价**：延迟 epilogue 意味着「结果在第一次 `.wait()` 时产生」，
拿到之后要自己留住；如果不重叠、立即 `.wait()`，反而可能比不延迟更慢。
README 的原话是「Without other work to overlap, call `.wait()` immediately」
（`DeepEP/README.md:287`）。

### 3.5 确定性布局（`deterministic=True`）

默认情况下，dispatch 的输出顺序**由设备上 SIMT 线程的原子操作顺序决定**，
同一份输入两次运行的行顺序可能不同。打开 `deterministic=True` 后，
`EPHandle.deterministic_sort()`（`deep_ep/buffers/ep.py:100-160`）
会在 epilogue 之后做一次**按源 token 的稳定排序**：

* 排序键 = `expert_idx * 1e10 + 源 token 全局下标`（`:125-130`）；
* 只排「展开后的数据张量」，再改写元数据槽位（`:145-160`）；
* cached dispatch 走 `cached_recv_src_metadata_before_sort` 复用上一次的确定性布局
  （`:116-121`）。

这是第 16 章「一致性（确定性）」那条工程约束的具体实现：**用一次额外的
排序换「逐位可复现」**。代价写在 `deep_ep/buffers/ep.py:145-160` 里——
被排序的数据量是全部展开 token，k 路路由下是 `T·k` 行。

### 3.6 FP8 dispatch 的 scale：粒度、排布与两个方向

§3.2 留下了一个问题没讲透：FP8 dispatch 的 scale 到底长什么样。这一节补齐，
并把第 27 章的数值口径接上。

**（1）为什么要 FP8。** dispatch 是纯搬运，字节数就是成本。把 hidden 从
BF16 换成 FP8，**hidden 部分直接省一半**（§1.2 的账：84 KB → 42 KB）。
代价是精度：FP8 必须带 scale。库把这件事做成一个显式的模式开关
`use_fp8_dispatch`——注意它管的是**容量预留**（`README.md:111` 的注释），
FP8 的实际数据由调用方按约定传进来。

**（2）scale 的粒度。** 输入 `x` 在 FP8 模式下是一个**二元组**
`(x_data, sf)`（`deep_ep/buffers/ep.py:341-345` 的文档字符串）：

* `x_data`：`[num_tokens, hidden]`，`torch.float8_e4m3fn`；
* `sf`：`[num_tokens, num_sf_packs]`，每个元素是一个 `int16`
  （`sf_pack_t`，`common/compiled.hpp:33`）。

每 128 个 hidden 元素一个 fp32 scale（`deep_ep/utils/math.py:30-38` 的
`per_token_cast_to_fp8` 用 `view(m, -1, 128)` 分组）。所以
`num_sf_packs = ceil(hidden / 128) × 2`：那个 `×2` 来自
`sf.view(torch.int16)` 的**字节重解释**（一个 fp32 变两个 int16），
不是「两个 scale 合成一个 int16」。

**（3）排布：row-major 还是 column-major。**
`dispatch` 有一个参数 `use_tma_aligned_col_major_sf`（`ep.py:372-373`），
默认 `False` 表示 **row-major**：

```text
row-major   ： sf.stride() == (num_sf_packs, 1)        # 同一 token 的 scale 连续
col-major   ： sf.stride() == (1, num_expanded_tokens) # 同一 scale 位置跨 token 连续
```

测试里对这两种排布都有断言（`tests/ep/test_ep.py:143-145`）；
host 侧从 row-major 转 col-major 就是把输出转置再转回来
（`csrc/buffers/ep.hpp:380-381`）。文档字符串还写明昇腾**不会**在 token
维度上额外补 padding（`ep.py:372-373`）。

**（4）与第 27 章对齐。** 第 27 章 §2.3 讲过 block-scaled 的三要素
（块粒度、E8M0 的语义、打包）；§6.2 讲过分组 GEMM 的 scale 布局最容易写错，
§7 第 1 条给了一个具体教训：**行主序的 bug 会「藏起来」**——
当 K 不超过 4 个粒度块时，两种索引恰好等价。
这里的 `use_tma_aligned_col_major_sf` 就是同一类「布局开关」，
只不过它作用在 dispatch 的输出上：**如果接收侧按 row-major 读、
而发送侧写了 col-major，得到的 scale 会整体错位**。
它比第 27 章的 MX 打包简单（没有「一个 word 装 4 个 E8M0」那种位运算），
但踩坑的方式是同一种。

**（5）为什么 combine 不能也走 FP8。** combine 要做的是一次**加权求和**，
而且是 k 份结果的归约；在低精度下累加会引入额外的舍入。库的选择是
**combine 输入固定 BF16**（`csrc/buffers/ep.hpp:434` 断言
`x.scalar_type() == torch::kBFloat16`），归约在 FP32 里做（§4.5），
最后转回 BF16。这是「通信省一半」与「归约要准」之间的取舍。

---

## 4. 实现机制：buffer、layout 与两个 kernel

这一节读代码。目标是能回答「一次 dispatch 在设备上分几步」「每一步动数据了吗」。

### 4.1 两块内存：workspace 与 buffer

`EPBuffer` 在 HCCL 里注册的是一整块对称内存，但内部切成两段
（`csrc/comm/api.hpp:48-63`）：

```text
symmetric memory  =  [ workspace | buffer ]
                        ↑            ↑
                    信号量/计数    收发数据
```

* **workspace**：放 `EPSignals`，也就是各种直方图计数器、前缀和、
  host 侧通知位（`deep_ep/include/deep_ep/layout/ep/workspace.hpp:8-18`）；
* **buffer**：放收发的 token 数据（dispatch 用发送缓冲，combine 用接收环）。

「对称」这个词在这里是**字面意思**：每个 rank 都按同样的公式算出同一段偏移，
于是 `rank i` 想写到 `rank j` 的某个位置时，它只需要知道 **`j` 这块内存的
基地址**，剩下的偏移两边一致。基地址表存在 `CommonSignals.peer_ptrs` 里
（`deep_ep/include/deep_ep/layout/common/signals.hpp:9-17`），
取对端地址的辅助函数是 `get_sym_ptr()`（同文件 `:19-24`）。

### 4.2 每 token 的布局

`TokenLayout`（`deep_ep/include/deep_ep/layout/ep/token.hpp:9-83`）
定义了「一个 token 在 buffer 里占的字节」：

```text
[ hidden bytes ][ SF packs（32B 对齐）][ topk_idx（int64×k，32B 对齐）]
                [ topk_weights（float32×k）][ src_token_global_idx（int32）]
```

对齐单位是 32 字节（`kNumUbAlignmentBytes = 32`，
`common/compiled.hpp:38`）。两个入口分别对应 dispatch（带 SF 和元数据）
与 combine（不带 SF、不带元数据，`csrc/kernels/ep/combine.hpp:14-16`）。

### 4.3 dispatch kernel：三段

`dispatch_impl`（`deep_ep/include/deep_ep/impls/ep/dispatch.hpp:559-865`）
在 `num_vec_cores` 个 AIV 上跑，每个 AIV 负责一段连续的 token 区间。三段是：

**第一段：SIMT 并行做路由（`:86-549`）。** 这一段跑在 SIMT 模拟的
AIV 线程上（`__simt_vf__`），做四件事：

1. 每个 token 的 k 个专家做**直方图**统计（`asc_atomic_add` 到
   `expert_histogram`，`:179`）；
2. 用 `simt::deduplicate` 把「同一 token 的多个专家落在同一 rank」**去重**，
   一个目标 rank 只记一次（`:183`）；
3. 分配**槽位**：对每个目标 rank 做前缀和，给每个 entry 定一个
   `dst_slot_idx`（`:260-294`）；
4. 把跨 rank 的接收数汇总到 workspace，算出
   `psum_num_recv_tokens_per_rank` / `_per_expert`（`:451-548`）。

这一段还有两个进度标记：`local_copy_ready` / `metadata_ready`
（`:297-298`、`:431`、`:801`），后面的搬运流水会等它。

**第二段：写 URMA 的 SQ（`:327-429`）。** 每个 warp 负责一批 SQE：
把「源地址 + 目标 rank + 目标槽位」写进一个 64 字节的 WQEBB
（`kNumWQEBBBytes = 64`，`comm/handle.hpp:19`）。这里有一个细节值得记：
**一次 WRITE 的 SQE 里，token 数据与元数据是两个 SGE**
（源码注释 `:360-369`），所以网络上一次搬运就带上了「数据 + 来源表」，
不需要第二趟。最后一条物理 SQE 会带 `fence + CQE` 标记
（`kWriteSQEFinalFlags`，`comm/handle.hpp:31-32`），
保证顺序与完成通知。

**第三段：本地拷贝 + 收尾（`:806-865`）。** 落在本 rank 的 token
不占网络，走 `asc_copy_gm2ub_align` → `asc_copy_ub2gm_align`
（`:818-838`）搬进接收 buffer。最后 `ring_doorbell()` 通知硬件
（`comm/handle.hpp:346-355`），一次 dispatch 的**发起侧**就结束了。

注意顺序：**SIMT 先算好路由并写好 SQE，硬件再去搬数据**，
所以「通信」与「路由计算」不重叠；真正能重叠的是**这次 dispatch 与
调用方后续的独立计算**。

### 4.4 dispatch 的 epilogue：为什么可以推迟

接收端的收尾在 `dispatch_copy_epilogue_impl` 里：等所有 rank 的通知到齐、
拿到精确的接收计数、再按槽位把数据从 buffer 拷到用户张量
（`csrc/buffers/ep.hpp:353-377` 是 host 侧的调用）。

它被单独拆成一个 kernel，正好配合 `defer_epilogue`：
**发起（issue）与收尾（epilogue）分成两次 launch**，
中间那段空白就可以塞进别的计算（`README.md:175`）。

### 4.5 combine kernel：环形归约

combine 的正向流程是「把 k 份结果送回原 rank，再加起来」。设备侧
`combine_impl`（`deep_ep/include/deep_ep/impls/ep/combine.hpp:162-...`）
的做法是：

1. 根据 `src_metadata` 算出每个 token 该发到哪、放到接收环的哪个槽
   （`load_meta_issue_wqe`，`:59-151`）；
2. 一个 token 只落在一个 rank 上（`master`）时，直接发原始数据；
   落在多个 rank 上时，先复制到 `send_buffer` 再发（`:138-140`）；
3. 接收侧用 `RingReducer`（`combine_utils.hpp:14-...`）把 k 份 BF16
   加载、**展宽成 FP32 累加**、再转回 BF16（`:56-90`）。

第 3 步是理解 combine 成本的关键：

* **累加在 FP32 里做**（`acc_regs` 是 `vector_float`，`:61`），
  只是最后 `asc_float2bfloat16_rn` 转回 BF16（`:88-89`）——
  与第 16 章 §2.5 的「k 路求和的顺序决定结果」是同一件事；
* **参与归约的数据都先落到 UB 的接收环里**，所以 combine 比 dispatch
  多了一次「写环 + 读环」的本地流量——这正是 README 说的
  「additional local reduction overhead」（`README.md:21`）。

### 4.6 同步：barrier 与信号

device 侧的 barrier 是 `comm::scalar::barrier`（`impls/ep/dispatch.hpp:651`），
它用 `CommonSignals` 里的 `barrier_counter` / `barrier_signals`
（`layout/common/signals.hpp:14-16`）在 rank 之间对齐时间。
源码注释给了两个数：一次 barrier 在 8 rank 时约 4 µs（`:643`），
一次 kernel launch + `InitSocState()` 约 1.9 µs（`:573`）——
这两个数字是**源码里写的观察**，不是本书实测。

还有一条容易忽略的：`EP_DISABLE_BARRIER_PROFILING` 关掉的是
**基准测试用的额外同步**，不影响正常调用的 barrier（`README.md:236`）。

---

## 5. 底座：HCCL / HCOMM / UBMEM / URMA 各管什么

README 用一句话交代了底座（`README.md:3`、`:5`）：

> Ascend C kernels use HCCL/HCOMM, UBMEM and URMA.

一句话里有四个名字，它们的**分工**可以从源码的调用点反推出来。先看图 4。

![图 4：DeepEP-Ascend 的通信栈分层](figures/04-comm-stack-layers.md)

（源描述见 [`figures/04-comm-stack-layers.md`](figures/04-comm-stack-layers.md)；
SVG 规格 `src/tutorial/figures/38-deepep-ascend/04-comm-stack-layers.json`。）

### 5.1 HCCL：进程组、拓扑与内存注册

HCCL 是昇腾的集合通信库（对标 NCCL）。在这个库里它承担四件事
（`csrc/kernels/comm/hccl.hpp`）：

1. **拿到通信域句柄**：按名字取 `HcclComm`
   （`HcclCommGetHandleWithName`，`:76`）；
2. **注册对称内存**：`HcclCommMemReg(comm, buffer_uid, &mem, &mem_handle)`
   （`:72-77`）。注册之后这块内存在对端可见；
3. **查拓扑**：`HcclRankGraphGetLayers` 拿网络层列表，然后**固定用
   `layers[1]`**（`:87-92`），注释写明「Use net layer 1 for UBMEM channels
   and Jetties」。README 的性能表也写着「All measurements use netlayer 1,
   the supernode's external Clos network」（`README.md:9`）；
4. **拿对端地址**：`HcclChannelGetRemoteMems` 返回对端注册过的内存，
   取最后一段（就是用户 buffer）填进 `peer_ptrs`
   （`:184-193`）。

还有一个**重要的硬约束**：这个后端要求**每个 peer 之间都有直连的
UBMEM 链路**；`get_physical_domain_size()` 会逐个 peer 检查链路协议里
有没有 `COMM_PROTOCOL_UB_MEM`，没有就断言失败
（`csrc/comm/handle.hpp:54-64` 的注释与断言：
「Ascend currently requires direct UBMEM connectivity to every peer」）。

### 5.2 HCOMM：端点、信道与 Jetty（URMA 的队列）

HCOMM 是更底层的通信引擎接口。这个库用它做三件事
（`csrc/kernels/comm/hccl.hpp:196-281`）：

1. **建端点**：`HcommEndpointCreate`，每个 AIV 一个
   （`num_jetties == num_vec_cores`，`:224-229`）；
2. **建信道**：`HcommChannelCreateWithConfig`，用来做 rank 间的
   **连接建立与端口交换**；低 rank 当 server、高 rank 当 client
   （`:263`），信道名字必须 ≤ `HCOMM_CHANNEL_NAME_MAX_LEN`
   （`:255`）；
3. **拿 Jetty**：设备侧的 `handle::HcommJetty`
   （`deep_ep/include/deep_ep/comm/handle.hpp:167-...`）把一条 URMA
   队列（SQ/CQ）的地址、深度、门铃地址读进 UB，然后**在 kernel 里
   自己写 SQE、自己敲门铃**。

这就是「Ascend C kernel 直接操作 URMA 队列」的全部含义：
**连接建立走 HCOMM 的 host API，数据搬运走 kernel 里的裸 SQE**。

### 5.3 UBMEM 与 URMA：为什么不是纯 RDMA

这两个词要分开讲：

* **UBMEM** 是「统一总线内存」——超节点里各 rank 的显存可以被对端
  **直接按地址访问**，所以本例里每对 rank 都是直连（§5.1 的硬约束）。
  在这个模型下，`rank j` 的 buffer 有确定的基地址，要写的地址就是
  「本地偏移 + 对端基址」，这正是对称内存能成立的原因。
* **URMA** 是昇腾的单边内存访问指令族。kernel 里构造的是 **WRITE** 和
  **WRITE_WITH_NOTIFY** 两种 opcode
  （`comm/handle.hpp:257-258`），也就是「我把数据写到你那儿，
  可选地顺手给你一个通知」。**没有任何一次 READ 或双边握手**在数据路径上。

**为什么不是纯 RDMA？** README 的 Requirements 里写明了硬件前提
（`README.md:28`）：需要 **Ascend 950 (A5) 的 UBMEM 连通性**，
以及 rank 之间的 **UBC_CTP / URMA 信道**。从这一点可以合理推断：
这套实现要的就是超节点内的 UB 内存语义，而不是通用 RDMA——
它本来就是为超节点内部设计的
（`deepseek-article-20260930.md:13` 的「基于昇腾 950 的 128 卡超节点方案」）。

**两个「域」的定义**：`get_physical_domain_size()` 与
`get_logical_domain_size()` 都返回 `{1, num_ranks}`
（`csrc/kernels/comm/api.hpp:82-89`）。换成 NVIDIA 的词就是
「rdma_ranks = 1、nvlink_ranks = num_ranks」——**整个 EP group
被当成单个 UB 域**。所以 `EPBuffer.__init__` 里
`num_scaleout_ranks = 1`（`deep_ep/buffers/ep.py:238-241`）。

### 5.4 DualPipe 与 EPLB 在这张图上的位置

这两个仓库不在 DeepEP-Ascend 里，但它们是理解「为什么要做重叠」和
「为什么接口里有个 LB 区」的关键。

**DualPipe**（`DualPipe/README.md:3`）是 DeepSeek-V3 技术报告里的
双向流水并行调度：它把前向和反向的计算-通信阶段**完全重叠**，
顺带压低流水线气泡。它的表格（`DualPipe/README.md:26-31`）用
`(PP/2-1)(F&B+B-3W)` 的公式把气泡写清楚——注意气泡里出现的是
`F&B`（两个方向互相重叠的 chunk），这是 DualPipe 相对 1F1B 的核心区别。

**EPLB**（`EPLB/README.md:5-8`）解决另一个问题：不同专家的负载不一样。
它的做法是**冗余专家（redundant experts）**——把热点专家复制多份，
再启发式地摊到各卡上；分两个策略：分层（先把专家组摊到节点、
再在节点内复制）与全局（无视专家组，全局复制摊牌）。

这两件事最终都落到 DeepEP 的接口上：EPLB 产生的「冗余专家」需要
把权重**提前搬到**副本所在的卡，反向再把副本的梯度**收回**原专家——
对应 `EPBuffer.lb_prefetch_weights` / `lb_reduce_grads`
（`README.md:81`）。**但这两个能力在昇腾后端还没实现**：
Python 侧接口已经按 V2.5 对齐，通信 kernel 还没写
（`README.md:81`；C++ 侧直接 `EP_HOST_UNREACHABLE`，
`csrc/buffers/ep.hpp:75-86`）。§8 会把它归进「现在别用」的清单。

---

## 6. DeepJIT：device kernel 为什么在运行时编译

### 6.1 一句话定位

README 说（`README.md:37`）：

> Installation builds the host extension; device kernels are compiled on
> first use. Keep the CANN toolkit and Bisheng available at runtime.

也就是：**安装时只编 host 侧扩展，device kernel 第一次用到才编**。
干这件事的是 DeepJIT（`DeepJIT/README.md:3`）：
一个 header-only 的 C++20 JIT 运行时，同时支持 CUDA 与昇腾两个后端。

### 6.2 调用形态

DeepEP-Ascend 用它的方式是「**拼一段源码字符串，交给 JIT 编译**」。
以 dispatch 为例（`csrc/kernels/ep/dispatch.hpp:51-68`）：

```cpp
const auto kernel = jit->compile("dispatch", std::format(R"(
#include <deep_ep/impls/ep/dispatch.hpp>
using namespace deep_ep;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&dispatch_impl<{}, {}, {}, ...>);
}}
)", num_ranks, num_experts, num_topk, ...));

jit->launch(kernel, {.num_blocks = num_vec_cores, .num_ubuf_bytes = num_ub_bytes}, ...);
```

读这段要注意两点：

1. **模板参数就是 shape 与规模**。`num_ranks`、`num_experts`、`num_topk`、
   `expert_alignment`、`num_hidden_bytes`、`num_sf_packs` 全都进模板参数——
   它们直接决定 UB 布局的大小（`UBLayoutWithBuffer`，
   `impls/ep/dispatch.hpp:56-75`）。换一个 shape 就等于换一份 kernel 源码。
2. **host 参数与 device 参数分开**。`num_blocks` / `num_ubuf_bytes` 是启动
   配置，其余是 kernel 实参；指针参数里还夹着 `jetty_ptrs`（§5.2 的队列表）。

`refresh runtime` 侧的 Device 查询也来自 DeepJIT + ACL：
`runtime->get_num_aicore_cores()` / `get_num_vec_cores()` /
`get_num_ubuf_bytes_per_vec_core()`（`csrc/runtime/jit.hpp:33-59`），
这些值直接喂给 UB 布局计算。

### 6.3 缓存与环境变量

DeepJIT 的缓存是「内存 + 磁盘」两层（`DeepJIT/README.md:22-27`）：

* **磁盘缓存**：默认 `$HOME/.dj`，可以用**冒号分隔的多个根**；
  按顺序查找，miss 只写第一个根（`DeepJIT/README.md:39`、`:71-79`）。
  多个进程/节点可以共享一个只读查找缓存（`:63-69`）；
* **共享缓存的三个前提**：同一文件系统内可原子 rename、目录 `fsync`
  可用、目录可信（`:45-58`）。

DeepEP 把前缀设成 `EP`（`csrc/runtime/jit.hpp:77`：
`deep_jit::Config(library_root, "EP", {}, {include_dir}, {"deep_ep/"})`），
所以它的变量是 `EP_JIT_*`，回落顺序是
`EP_JIT_* → DJ_JIT_* → 内置默认`（`README.md:238`）。
README 里列了七个（`README.md:242-248`），日常最有用的是：

| 变量 | 默认 | 用途 |
| --- | --- | --- |
| `EP_JIT_CACHE_DIR` | `$HOME/.dj` | 缓存根，或冒号分隔的多个根 |
| `EP_JIT_DEBUG` | `0` | 打印编译命令、二进制路径、加载耗时 |
| `EP_JIT_DUMP_ASM` | `0` | cache miss 时把昇腾汇编存到 `asm/` |
| `EP_JIT_PRINT_COMPILER_COMMAND` | `0` | 打印 Bisheng 与链接器命令 |
| `EP_JIT_LAUNCH_TIMEOUT` | `300` | kernel launch 超时秒数，`0` 关闭 |

还有一条来自 README Requirements 的硬要求（`README.md:31`、`:37`）：
**运行时必须能拿到 CANN toolkit 与 Bisheng**，并且
`ASCEND_HOME_PATH`（或 `ASCEND_TOOLKIT_HOME`）要指向当前的 CANN 安装。
DeepJIT 的 Ascend 后端需要的正是 `bin/bisheng`、`bin/ld.lld` 与
`asc/include/adv_api`（`DeepJIT/README.md` 的 Ascend 一节）。

### 6.4 与第 26 章 TileLang 的 JIT / ABI 逐项对照

第 26 章把 TileLang 的 JIT 拆成四段：装饰 → 取 TIR → 缓存查询 →
lower/codegen → 加载/调用，出口是一个 **packed host function**。
把 DeepJIT 摆在同一张表上：

| 维度 | TileLang（第 26 章） | DeepJIT（本章） |
| --- | --- | --- |
| 编译什么 | 由 TileLang DSL 经 TIR/pass 生成的 kernel 源码 | **作者手写的 Ascend C 源码字符串** |
| 谁决定编译期常量 | TileLang 的 JIT 参数 + tensor shape | C++ 模板参数，作者自己拼 |
| 缓存键 | 源码 + TIR + pass 配置 + 目标等（`kernel_cache.py`） | 源码 + 被跟踪的 include 树 + 编译器版本 + 生效选项 + 调用方给的依赖签名（`DeepJIT/README.md:37-...`） |
| 磁盘缓存 | `<cache>/<version>/<platform>/<key>/` | `<cache-root>/cache/<tag>.<digest>/`（`DeepJIT/README.md` 的 Cache-key 一节） |
| 共享缓存 | 支持，靠 `TILELANG_CACHE_DIR` | 支持多根查找 + 只读共享根（`DeepJIT/README.md:39-79`） |
| host 侧 ABI | **packed function**：形状/stride/device 检查、输出分配都在里面（`make_packed_api.cc`） | 没有生成 host wrapper；**调用方自己传指针和启动参数** |
| 框架适配 | adapter 层（torch / cython / tvm_ffi） | 用当前 `torch_npu` stream，其余由调用方负责 |
| 语言/后端 | 一套 DSL，两个后端（第 25 章） | 两个后端共享运行时，**源码各不相同** |

一句话总结它们的**分工差异**：

> TileLang 的 JIT 把「**从 DSL 到可调用函数**」整条链包住，
> 所以它必须管 ABI、管 adapter、管输出分配；
> DeepJIT 只管「**把一段 C++ 源码变成一次可发起的 launch**」，
> host 侧的参数校验、张量分配、stream 管理都留给调用方。

这不意味着谁更好，而是**它们解决的是不同层次的问题**：
DeepEP 这类库的 host 侧本来就有一大堆依赖 shape 的逻辑
（`EPBuffer.dispatch` 里几十行断言），再多一层自动 ABI 反而碍事；
而 TileLang 的用户是「写算子的研发」，必须要有个能直接调用的函数。

### 6.5 对使用者的四个实际影响

1. **首次调用慢**。第一次用到某个 shape 的 dispatch/combine，要现场
   调 Bisheng 编一遍（`README.md:37`）。生产部署要**预热**，
   或者把磁盘缓存挂成共享目录（`README.md:239-242`）。
2. **缓存目录要共享**。多节点/多进程场景下，把所有 rank 指向同一个
   `EP_JIT_CACHE_DIR`，才能避免每个 rank 各编一遍。
3. **环境要完整**。`ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` 必须
   指对，`bin/bisheng` 与 `bin/ld.lld` 必须可执行；容器里缺 CANN
   工具链会直接在第一次调用时失败（`README.md:31`）。
4. **调参要会看汇编**。`EP_JIT_DUMP_ASM=1` 把昇腾汇编存下来，
   `EP_JIT_PRINT_COMPILER_COMMAND=1` 打印 Bisheng 命令——
   这是排查「为什么这次编译慢 / 为什么会 trap」的第一手材料。

---

## 7. 性能表怎么读：128 卡超节点的带宽数字

### 7.1 先把表贴出来，连同它的条件

（`DeepEP-Ascend/README.md:9-21`，逐字抄录）

| EP size | Dispatch (GB/s) | Combine (GB/s) |
| --- | ---: | ---: |
| EP8 | 373–375 | 345–347 |
| EP16 | 348–352 | 338–341 |
| EP32 | 335–340 | 320–324 |
| EP64 | 323–327 | 294–298 |
| EP128 | 313–320 | 272–278 |

**测量条件**（同一个 README 的 `Performance` 一节，`:9-11`、`:21`）：

| 条件 | 取值 |
| --- | --- |
| 硬件 | Ascend 950DT NPU，**手工配置的 PoC HDK**（非公开版本） |
| 软件栈 | CANN 9.2.0 |
| 网络 | netlayer 1，即超节点的 **external Clos 网络** |
| 每 rank token 容量 | 16,384 |
| hidden | 7,168 |
| 路由 | top-6 / 256 experts |
| 核数 | 32 AI cores / 64 AIVs |
| expert alignment | 128，无 bias |
| dispatch 精度 | **FP8 + row-major scale** |
| combine 精度 | **BF16** |
| 迭代 | 10 warmup + 50 samples，**刷 cache** |
| 口径 | 计时含 issue 与 drain，**不含最后的 epilogue**；带宽范围覆盖所有 rank |

这张表是「本书唯一引用的性能数字」。要提醒的是它的**两个不确定性**：

* HDK 是 **PoC（proof-of-concept）版本，带人工配置，不是公开版本**；
  README 在 `Recommended HDK and firmware` 一节写明商用 Q3 HDK 计划在
  2026-10-15 前后发布，并明确「these measurements are not results from
  that unreleased commercial version」（`:41-47`）。所以**这张表不能当成
  你现在能买到的硬件的成绩**。
* README 也写明「Kernel support on other Ascend generations or CANN
  versions has not been established by these measurements」（`:35`）。

### 7.2 为什么 EP 越大越慢

两条单调下降都能从 §1.3 和源码里找到解释：

1. **去重率下降、跨设备字节量上升**。EP8 时 256/8 = 32 个专家在本 rank，
   一个 token 的 6 个专家更容易落回本地；EP128 时只有 2 个本地专家，
   几乎每个 top-k 专家都在别的 rank 上。每 rank 的有效接收 token 数上升，
   而有线带宽固定，于是单位时间能完成的「逻辑 token 数」下降。
   依据：`impls/ep/dispatch.hpp:183` 的去重、以及正文 §1.3 的
   `R·(1-(1-1/R)^k)`。
2. **同步与启动的固定开销随 rank 数上升**。源码注释里写着
   barrier「this costs 4 us (8 ranks)」（`impls/ep/dispatch.hpp:643`）、
   kernel launch 与 SoC 初始化「cost 1.9 us, which is slow」（`:573`）。
   这些开销与 payload 大小无关；EP 越大，它们相对有效 payload 的占比越高，
   而分母（时间）里包含了它们。
3. **上游还没压到底**。README 自己也说
   「Larger EP sizes and combine remain under optimization」（`:21`）——
   所以这张表的下降里，有一部分是**当前实现的余量**，不是物理必然。

### 7.3 为什么 combine 比 dispatch 慢

同样是 EP8，dispatch 373–375 GB/s，combine 只有 345–347。三条原因：

1. **combine 的 payload 更大**。README 写明 dispatch 走 FP8、
   combine 走 BF16。同样一个 token，combine 的 hidden 部分是 dispatch 的
   **2 倍字节**（§7.1 的账：42 KB → 84 KB）。
2. **额外的本地归约开销**。README 原话是「combine has additional local
   reduction overhead」（`:21`）。设备侧对应 §4.5 的 `RingReducer`：
   数据要先写进 UB 接收环、再读出来做 FP32 累加、再转回 BF16，
   这一段不占网络带宽但占流水线时间。
3. **与 HBM 抢带宽**。README 接着说「HBM contention with URMA」（`:21`）——
   URMA 的数据落地要写显存，而本地归约也要读显存，
   两者在 HBM 上互相争抢。README 说华为固件改进计划就是针对这条
   （「Huawei firmware improvements are also planned to reduce this
   contention」）。

### 7.4 「90–95% 的物理 payload 带宽上限」怎么读

README 的那句是（`:21`）：

> Sustained dispatch reaches roughly 90–95% of the physical payload
> bandwidth limit for EP sizes up to 32.

三个词要拆开：

* **physical payload**：真正在链路上搬的有效字节。测试代码里这个量是
  `num_dispatch_urma_bytes = num_recv_tokens × (每 token 的逻辑字节数)`，
  其中「每 token 的逻辑字节数」= `count_bytes(x, topk_idx, topk_weights) / num_tokens`
  （`tests/ep/test_ep.py:404-406`）。注意它**不含 padding、不含 barrier**。
* **bandwidth limit**：这条链路（UBMEM/URMA）能搬的字节率上限。
* **90–95%**：实测 GB/s ÷ 上限。所以这句话等价于
  「dispatch 已经把链路吃掉了九成以上」。

**本书复算**：把表里的数字除以 0.90 与 0.95，可以反推出一个量级：

```text
EP8  : 373–375 ÷ (0.90…0.95) ≈ 393–417 GB/s
EP16 : 348–352 ÷ (0.90…0.95) ≈ 366–391 GB/s
EP32 : 335–340 ÷ (0.90…0.95) ≈ 353–378 GB/s
```

也就是**量级在 350–420 GB/s/rank 之间**。注意三件事：

1. 这是一个**区间估计**，不是实测；README 自己是「roughly」的说法；
2. 它随 EP 规模略降，符合 §7.2 的解释（规模越大、固定开销占比越高）；
3. corpus 里**没有 Ascend 950 UBMEM 链路的官方带宽规格**，
   所以这个反推**不能**当成硬件手册上的数字。它只用来回答一个定性问题：
   **瓶颈现在在链路上，不在算法上**——这也是为什么 README 说
   「Larger EP sizes and combine remain under optimization」。

`code/04-bandwidth-normalize/` 会把这张表连同口径一起复算成
「相对上限」的百分比，并明确标注假设。

![图 5：带宽随 EP 规模变化](figures/05-bandwidth-vs-ep.md)

（源描述见 [`figures/05-bandwidth-vs-ep.md`](figures/05-bandwidth-vs-ep.md)；
SVG 规格 `src/tutorial/figures/38-deepep-ascend/05-bandwidth-vs-ep.json`。
数据源只有 `README.md:13-19` 一张表。）

---

## 8. 状态的诚实标注：现在能用什么，别用什么

这一节按「能用 / 实验性 / 不支持」三档把整个库排一遍。判据全部来自
README 的 `Features and API compatibility`（`:60-74`）与 `Ongoing`（`:76-84`），
以及源码里的断言。图 6 是速查版。

![图 6：能力状态清单](figures/06-capability-status.md)

（源描述见 [`figures/06-capability-status.md`](figures/06-capability-status.md)；
SVG 规格 `src/tutorial/figures/38-deepep-ascend/06-capability-status.json`。）

### 8.1 现在可以用（EP 主线）

| 能力 | 状态 | 依据 |
| --- | --- | --- |
| `EPBuffer` 的 BF16 / FP8 dispatch | 可用，展开布局（`do_expand=True`） | `README.md:66`、`:72` |
| combine（BF16 输入，FP32 累加） | 可用 | `README.md:66`；`impls/ep/combine_utils.hpp:56-90` |
| cached handle（复用布局） | 可用；**`do_handle_copy` 必须 `False`** | `README.md:66`、`:72` |
| 确定性布局（`deterministic=True`） | 可用，代价是一次排序 | `deep_ep/buffers/ep.py:100-160` |
| expert padding / `do_zero_padding` | 可用 | `README.md:66` |
| 延迟 epilogue（`defer_epilogue`） | 可用，必须配 `async_with_compute_stream` | `csrc/buffers/ep.hpp:398` |
| 训练与推理共用同一接口 | 可用（prefill / decode 用 `do_cpu_sync` 区分） | `README.md:62`、`:120` |
| 专家接收计数统计（`cumulative_local_expert_recv_stats`） | 可用，可做在线负载监控 | `deep_ep/buffers/ep.py:351-352` |

**一句话**：如果你要做的是「标准的 MoE 专家并行 dispatch/combine」，
昇腾后端这条路已经通了。`README.md:262-265` 给了两条可直接复现的
8 卡命令（bf16 与 fp8 各一条）。

### 8.2 实验性（接口在，验证未完成）

| 能力 | 状态 | 依据 |
| --- | --- | --- |
| `PPBuffer`（相邻流水 rank 的 send/recv） | experimental，正确性与性能验证进行中 | `README.md:67`、`:82` |
| `EngramBuffer`（NPU 侧 BF16/FP8 表 + 每层 fetch hook） | experimental | `README.md:68`、`:82` |
| `BucketBuffer` 的 **batched all-gather** | 可用但归入 experimental；**用前必须设 `EP_AVOID_RECORD_STREAM=1`** | `README.md:69`、`:80`、`:234` |
| `BufferAllocator` / `BufferBase` / `deep_ep.comm` | 可用（分配规划、生命周期、通信域查询、barrier、stream） | `README.md:70` |

Bucket 还有一组容易踩的约束（`README.md:220`）：所有 rank 的 shard 必须等长；
bucket 必须足够大，**session 不会自动扩容**；离开 session 时输出必须属于 bucket；
批量最多 64 个 tensor；多 group 时用 `group=` 选；**session 退出不加 barrier**。

### 8.3 不支持（现在别用）

| 能力 | 状态 | 依据 |
| --- | --- | --- |
| Bucket 的 **reduce-scatter / all-reduce** | 昇腾 kernel **仍在做** | `README.md:80` |
| `lb_prefetch_weights` / `lb_reduce_grads` | 接口已对齐，**昇腾通信 kernel 未实现** | `README.md:81`；`csrc/buffers/ep.hpp:79`、`:85`（`EP_HOST_UNREACHABLE`） |
| **混合通信（hybrid）** | 不支持 | `README.md:84`；`deep_ep/buffers/ep.py:194-196` |
| **CPU-backed Engram 存储** | 不支持（NVIDIA 侧支持） | `README.md:84` |
| **graph capture**（把 EP 入图） | 不支持 | `README.md:84` |
| 显式 RDMA 服务等级 / QP 数 | 不支持（`sl_idx=None`、QP 用默认） | `README.md:74` |
| `do_handle_copy=True` | 不支持 | `README.md:72` |
| 非展开布局（low-latency 模式） | 不支持（`do_expand` 必须 `True`） | `csrc/buffers/ep.hpp:116` |
| 奇数 rank 数 | 多卡通信要求 rank 数为偶数 | `README.md:29`；`csrc/kernels/comm/hccl.hpp:85` |

### 8.4 这些状态对读者的实际影响

把上面对号入座，能得出四条可执行的结论：

1. **训练侧的重叠只能靠 DualPipe + defer_epilogue 这一层**。
   想要「像 NVIDIA 那样把 EP 入 graph capture 降下发开销」——
   现在做不到（`README.md:84`）。decode 的小 batch 场景要特别小心
   每次 launch 的固定开销。
2. **动态冗余专家（EPLB）暂时不能端到端用**。EPLB 本身有算法、
   有 `rebalance_experts` 接口（`EPLB/README.md:35`），但把权重搬来搬去的
   `lb_prefetch_weights` 在昇腾上还没实现。所以「在线负载均衡」这条路
   目前只有规划、没有搬运。
3. **Bucket 只能 all-gather**。CP/DP 用 all-gather 没问题；
   要用 reduce-scatter / all-reduce 就得先自己在 all-gather 上叠。
4. **hybrid 模式没有**。NVIDIA 侧那套「scale-up + scale-out 混跑」
   在昇腾上退化成单域（§5.3 的 `{1, num_ranks}`），
   所以容量规划只能按「一个 UB 域」来。

---

## 9. TileLang 这条线：通信库不用它，它服务的两端用

### 9.1 分工线在哪里

回到 §1.1 的那句话：DeepEP-Ascend 的自我定位里没有 TileLang。
这不是遗漏——`deep_ep/` 下**一行 TileLang 也没有**（`grep` 无命中），
kernel 全部是 **Ascend C**（`__global__ __vector__ __aicore__`，
`#include <kernel_operator.h>`、`c_api/*`、`simt_api/*`，
见 `impls/ep/dispatch.hpp:1-20`）。

但**它服务的两端是 TileLang**：

```text
          TileKernels（TileLang）                DeepEP-Ascend（Ascend C）
   ┌────────────────────────────┐        ┌────────────────────────────┐
   │ moe_topk_gate / topk_gate  │───────▶│  EPBuffer.dispatch         │
   │ per_token_cast（量化到 FP8）│───────▶│  EPBuffer.combine          │
   └────────────────────────────┘        └────────────────────────────┘
             产生 topk_idx                    消费 topk_idx
             产生 fp8 x + sf                  消费 fp8 x + sf
```

* **进 dispatch 的 `topk_idx`** 由 `tile_kernels.moe.topk_gate` /
  `moe_topk_gate_forward` 产出（`TileKernels/tile_kernels/moe/topk_gate_kernel.py:7-43`，
  实现分别在 `topk_gate_cuda.py` 与 `topk_gate_asc.py`）。输出的
  `[num_tokens, num_topk] int64` 正是 `EPBuffer.dispatch` 要的类型。
* **进 dispatch 的 FP8 数据与 scale** 由 `tile_kernels.quant.per_token_cast_impl`
  产出（`TileKernels/tile_kernels/quant/per_token_cast_kernel.py:9-60`），
  它同时有 CUDA 与 Ascend 两个 TileLang 实现。

**为什么通信库不用 TileLang？** 因为 dispatch/combine 的核心不是
「在 AIV 上算」，而是「**直接写 URMA 队列、操作 Jetty、按字节拼 SQE**」
（§5.2、§4.3）。这类代码需要的是「能拿到队列地址、能逐 lane 组装 64 位字、
能在 SIMT 与 SIMD 之间互操作」——这是 Ascend C 的领域，不是
「tile 级数据流」的领域。第 35 章的结论在这里有一个镜像的表述：
**当瓶颈在通信协议、不在算术上时，TileLang 的抽象层次帮不上忙**。

### 9.2 与 MegaMoE 的两种做法

第 35 章的 MegaMoE 走了**相反的方向**：它把 dispatch、两个 grouped GEMM、
SwiGLU、combine 全部融进**一个 kernel**
（`DeepGEMM-Ascend/deep_gemm/include/deep_gemm/layout/mega_moe.hpp:26-33`
的四个进度阶段 `kDispatched → kLinear1Ready → kLinear2Ready → kCombineSent`），
中间结果不再往返 HBM，通信与计算在同一批核上交错。

两种做法摆在一起：

| 维度 | DeepEP-Ascend（独立通信库） | MegaMoE（融合 kernel） |
| --- | --- | --- |
| 谁写通信 | DeepEP 的 Ascend C kernel | DeepGEMM-Ascend 的 Ascend C kernel |
| 谁写 GEMM | 外部库（如 DeepGEMM-Ascend） | 同一个 kernel |
| 中间结果 | 展开布局的 token，落在 HBM | 不落 HBM，留在片上/对称缓冲 |
| 通信与 GEMM 的关系 | 分两次 launch，靠「先发起后等待」重叠 | 同一批核内按进度阶段交错 |
| 可复用性 | 高：任意 MoE 结构都能接 | 低：形状/专家数/核数都写死 |
| 约束 | 需要对称缓冲区、rank 数偶数 | **核数固定 32**、形状提前注册、每次 launch 清 progress 区 |

适用场景的判断很简单：

* **算法还在动、要换模型结构、要复用到不同层** → 用 DeepEP-Ascend。
  它把「通信」做成一个可组合的库，代价是中间结果过一遍 HBM。
* **形状已经冻结、要把通信开销压到最低** → 走 MegaMoE 那条路。
  代价是灵活性：第 35 章 §4.5 记着
  `DJ_HOST_ASSERT(runtime->get_num_sms() == kNumAICores)`，
  核数写成 32。

### 9.3 与 DualPipe 的关系（再补一句）

§5.4 提到 DualPipe 做的是**训练侧的流水线级重叠**。把它和本章的
`defer_epilogue` 区分开：

| 层次 | 重叠的是什么 | 谁负责 |
| --- | --- | --- |
| kernel 内 | dispatch 的 issue 与本地拷贝；combine 的发送与归约 | DeepEP-Ascend 自己 |
| 调用级 | 「这次 dispatch」与「调用方后续的独立计算」 | 使用者的代码（`defer_epilogue`） |
| 流水线级 | 前向 chunk 与反向 chunk 的计算-通信 | DualPipe 的调度 |

三者**可以叠加**：DualPipe 提供一个调度骨架，骨架里的每个 MoE 层
用 `defer_epilogue` 把本层的通信藏起来，而 DeepEP 的 kernel 内部
再做一层细粒度重叠。本章只证明第一、二层能用；第三层需要真实的
多机环境，留给第 39 章。

### 9.4 一句话总结这条主线

> **TileLang 负责「算」，DeepEP-Ascend 负责「搬」。**
> 两者的接口是三个张量：`topk_idx`（int64）、`topk_weights`（float32）、
> `x`（bf16 或 fp8+sf）。只要这三个张量的类型与布局对得上，
> 一边是 TileLang、一边是 Ascend C，也能拼成一条流水线。

---

## 本章小结

一页纸复习：

1. **通信量从哪来**：MoE 的 dispatch 把 token 送到专家所在卡、combine 收回，
   跨越 rank 时是 all-to-all。用第 16 章的账本，每 token 每方向 `k·d·b` 字节；
   在 16384 tokens/rank、hidden 7168、top-6 的规模下，**一层就是 GB 级**。
   EP 越大，token 被拆到越多 rank 上、本地专家越少，通信占比越高。
2. **接口**：`EPBuffer(group, num_max_tokens_per_rank, hidden, num_topk,
   use_fp8_dispatch, explicitly_destroy, ...)`；用 `get_buffer_size_hint()`
   规划容量（对齐 2 MB）；`dispatch` / `combine` 互为反向；
   `handle` 缓存路由、按引用持有 `topk_idx`。
3. **精度**：dispatch 用 FP8（hidden 只占 1 字节）+ row-major scale，
   combine 必须是 BF16（要精确归约）。scale 的粒度与布局影响数值，
   与第 27 章的 block scaling 是同一套语言。
4. **重叠**：`defer_epilogue=True` + `async_with_compute_stream=True`
   把「发起」与「收尾」拆开，中间塞独立计算；训练侧的整体重叠交给 DualPipe。
5. **底座**：HCCL 管通信域、拓扑与内存注册（固定用 netlayer 1）；
   HCOMM 管端点、信道与 Jetty；UBMEM 提供对称内存地址；
   kernel 直接写 URMA 的 WRITE / WRITE_WITH_NOTIFY。
   整个 EP group 是一个 UB 域（`{1, num_ranks}`）。
6. **DeepJIT**：device kernel 首次使用时才编；缓存可共享；
   调用形态是「拼一段 C++ 源码 → `compile` → `launch`」。
   与 TileLang 的 JIT 相比，DeepJIT **不管 host ABI 与 adapter**，
   只管「把源码变成一次 launch」。
7. **性能表**：EP8–EP128，dispatch 373→313 GB/s、combine 345→272 GB/s；
   条件是 Ascend 950DT + CANN 9.2.0 + PoC HDK + netlayer 1；
   dispatch 在 EP≤32 时达到链路物理上限的 90–95%。
   combine 更慢是因为 payload 翻倍、多一次本地归约、且与 HBM 抢带宽。
8. **现状**：EP 主线可用；Bucket 只有 all-gather；PP / Engram 实验性；
   reduce-scatter / all-reduce、LB 通信、hybrid、CPU Engram、
   graph capture 都不支持。选型前先过一遍 §8 的三张表。
9. **TileLang 主线**：通信库自己不用 TileLang（它要直接操作 URMA 队列），
   但它消费的 `topk_idx` 与 `fp8 x + sf` 来自 TileLang 算子；
   MegaMoE 走的是把通信融进 GEMM kernel 的另一条路——
   一条可组合、一条极致融合。

---

## 延伸阅读

* **[R1] DeepEP-Ascend 自述**：`DeepEP-Ascend/README.md`
  —— 定位、Requirements、已验证组合、HDK 说明、性能表、接口兼容表、
  `Ongoing` 清单、三段 EP 用法代码、环境变量。
* **[R2] Python 接口与 handle**：`DeepEP-Ascend/deep_ep/buffers/ep.py`
  —— `EPHandle` 的字段、`EPBuffer` 的构造与断言、`dispatch` / `combine`
  的参数与文档字符串、`deterministic_sort`、LB 两个入口。
* **[R3] host 侧 launch 与 buffer 尺寸**：`DeepEP-Ascend/csrc/buffers/ep.hpp`
  与 `csrc/kernels/ep/{dispatch,combine}.hpp`
  —— epilogue 的拆分、容量公式、模板参数是怎么拼出来的。
* **[R4] device 侧 kernel**：`deep_ep/include/deep_ep/impls/ep/dispatch.hpp`、
  `.../combine.hpp`、`.../combine_utils.hpp`、`.../dispatch_copy_epilogue.hpp`
  —— §4 的全部依据。
* **[R5] URMA 队列与 SQE 模板**：`deep_ep/include/deep_ep/comm/handle.hpp`
  —— `HcommJetty`、`HcommPeerInfo`、WRITE / WRITE_WITH_NOTIFY 的字段布局。
* **[R6] 通信栈**：`DeepEP-Ascend/csrc/kernels/comm/hccl.hpp`、
  `csrc/kernels/comm/symmetric.hpp`、`csrc/comm/api.hpp`
  —— §5 的全部依据。
* **[R7] JIT 运行时**：`DeepJIT/README.md`（缓存、共享缓存、Ascend 后端）
  与 `DeepEP-Ascend/csrc/runtime/jit.hpp`（`Config` 与设备查询）。
* **[R8] 测试与口径**：`DeepEP-Ascend/tests/ep/test_ep.py`
  —— 测量循环、`count_bytes` 的口径、`bench_msprof` 的 warmup/sample/flush。
* **[R9] NVIDIA 对照物**：`DeepEP/README.md`
  —— V2.5 的接口约定、cached handle、deferred epilogue、
  expert load balancing 的完整语义（昇腾缺失的部分在这里可以看到原貌）。
* **[R10] 配套调度与均衡**：`DualPipe/README.md`、`EPLB/README.md`。
* **[R11] TileLang 侧两端**：`TileKernels/tile_kernels/moe/topk_gate_kernel.py`
  与 `tile_kernels/quant/per_token_cast_kernel.py`。
* **[R12] 前置章节**：第 16 章（MoE 四段结构与字节账）、
  第 26 章（TileLang 的 JIT 与 ABI）、第 27 章（低精度与 block scaling）、
  第 35 章（MegaMoE 融合）。

完整的引用清单（含核验状态与上游链接）见 [`references.md`](references.md)；
每个代码示例的运行前提与验证级别见 [`code/README.md`](code/README.md)。
