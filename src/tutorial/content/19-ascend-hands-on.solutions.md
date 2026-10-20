# 第 19 章 习题答案

说明：以下每题给「解题思路 + 参考答案 + 常见错误」。引用正文时用「§N.M」，
引用源码时给 `_vendor/` 路径与行号。凡是需要真实硬件才能得到的数字，
答案只给**判据与流程**，不给具体测量值。

---

## E1

**思路**：§1.1 把「kernel 正确」拆成三层义务，这三层各有独立的失败模式。

**参考答案**

1. 三层义务：
   * **算法正确**：这段代码在数学上算的是你要的东西吗（例如 GEMM 的累加顺序、
     ReLU 的位置、量化的舍入方式）。
   * **tiling 正确**：每一级片上存储都装得下对应的 tile 吗（L1/L0A/L0B/L0C/UB 各自的容量）。
   * **同步正确**：当一份数据被 A 单元写完、B 单元要读，或者一份缓冲要被复用时，
     有没有保证先后顺序（跨 PIPE 的依赖必须显式表达）。
2. 「算法对、tiling 也对、同步错」的典型症状：结果**偶发**错误、
   或者**第一次调用对、连续调用后错**、或者只在某些 shape/核数下错。
   因为读到的是「上一轮还没被覆盖的数据」或「还没搬完的数据」，取决于时序。
3. 从编译器角度看：NVIDIA 路径的 `cp.async`/`mbarrier` 由 pass 隐式处理，
   而昇腾（尤其是 A 树的手工模式）要求程序显式表达跨 PIPE 依赖，所以错的概率更高。
   从可见性角度看：同步错误的症状是**时变的**，小规模测试常常蒙混过关，
   因此很难定位（§11 的 P10 就是这个症状）。

**常见错误**：把「同步正确」理解成「加几个 barrier 就行」。`barrier_all` 只等本核
此前发出的异步操作，**不承担 buffer ownership 的交接**（P11）。

---

## E2

**思路**：§2.4 给了两个原语的语义；§11 的 P11 给了 `barrier_all` 的边界反例。

**参考答案**

1. `PipeBarrier<PIPE_X>()` 在**同一条执行序列内部**约束顺序：它保证前序指令的读写
   全部完成后，后序指令才开始。`SetFlag`/`WaitFlag` 在**两条序列之间**建立一对锁：
   `SetFlag` 在前序指令读写完成后把标志位置 1，`WaitFlag` 发现为 0 就阻塞、为 1 就
   清 0 并放行。
2. `T.barrier_all()`（`PIPE_ALL`）只等**它之前已发出的本核异步操作**。它：
   （a）不等价于缓冲区 ownership 的交接；（b）不能保护「屏障之后又出现的新 store
   复用同一物理缓冲」。P11 的反例就是「加了 barrier 还是错」。
3. 同一条 PIPE 内的指令是**按发射顺序**进入队列的，但硬件允许前一条还没执行完就开始
   执行后一条（流水）。因此当两条指令之间存在数据依赖时，必须用 `PipeBarrier`
   或 `Set/Wait` 把顺序「锁死」。

**常见错误**：把 `barrier_all` 当成万能的「全芯片同步」。它是**本核**的全 PIPE 屏障，
核间同步要用 `set_cross_flag`/`wait_cross_flag`（A 树）或交叉核同步（M 树自动）。

---

## E3

**思路**：按 §7.1 的对照表判断，判据是「API 名 + 前端命名空间 + target」。

**参考答案**

| 序号 | 归属 | 判据 |
| --- | --- | --- |
| 1 | A 树 | `tilelang.language` 是 A 树的前端命名空间 |
| 2 | M 树 | 小写 `alloc_l0c` 是 M 树方言（A 树是大写 `alloc_L0C`） |
| 3 | A 树 | `is_npu=True` 且返回 `(cid, vid)` 是 A 树写法 |
| 4 | M 树 | `T.MixedKernel` 只在 M 树导出 |
| 5 | A 树 | `TL_ASCEND_*` 这些 `PassConfigKey` 只存在于 A 树（`pass_config.py:38-62`） |
| 6 | M 树 | `T.dual_copy` 是 M 树方言 |
| 7 | A 树 | `T.gemm_v0` 是 A 树接口（M 树用 `T.gemm`） |
| 8 | M 树 | M 树的昇腾 target 是 `"ascend"`（A 树是 `"ascendc"`/`"pto"`） |
| 9 | A 树 | `T.printf` 是 A 树导出（M 树只有 `T.print`） |

混用几乎一定失败，因为两套名字来自不同的 Python 包与不同的枚举：
最典型的报错是 `AttributeError: module 'tilelang.language' has no attribute 'alloc_l1'`，
或反过来 `AttributeError: module 'tilelang.ascend.language' has no attribute 'alloc_L1'`；
用 A 树的 `PassConfigKey` 去驱动 M 树则可能因为未知 key 而静默无效。

**常见错误**：只记 API 名不看命名空间。**判据是「import 哪一行」，不是「函数叫什么」。**

---

## E4

**思路**：§4.2 逐行解释 + §4.3 的数据流图。

**参考答案**

1. `VEC_NUM = 2` 表示每个逻辑块（一个 AI Core）里有**两个 Vector 子单元**；
   `vid ∈ {0,1}` 区分它们。UB buffer 形状是 `block_M // VEC_NUM`，因为一块 tile 被
   两个 `vid` 各处理一半，每个子单元只需要自己那一半的缓冲。
2. 第一个 `barrier_all()` 保护 `MTE2 → V`（等两次 `T.copy` 搬运完成，Vector 才能读）；
   第二个保护 `V → MTE3`（等 `T.tile.add` 算完，MTE3 才能把 `c_ub` 写回 GM）。
   写错（删掉或位置错）会造成「读到未搬完的输入」或「写回未算完的输出」，结果错。
3. 因为硬件在单次小规模执行里可能恰好没触发竞争，或者上一次执行的残值恰好是对的。
   这说明**「这次跑对了」不能证明同步正确**——这正是 §11 P10 强调「实测通过不能
   替代配对检查」的原因。
4. `GM(A), GM(B) --MTE2--> UB --Vector--> UB --MTE3--> GM(C)`。

**常见错误**：以为 `T.tile.add` 会自动等搬运完成。`T.tile.add` 是 Vector 计算指令，
它不会替你等 MTE2。

---

## E5

**思路**：§5.3 的通路表 + §5.4 的容量约束。

**参考答案**

1.

| 步骤 | 搬运/计算单元 | PIPE | 由谁发起 |
| --- | --- | --- | --- |
| GM → L1 | MTE2 | `PIPE_MTE2` | 用户写的 `T.copy` |
| L1 → L0A/L0B | MTE1（`LoadData`） | `PIPE_MTE1` | `T.gemm_v0` 内部 |
| 累加进 L0C | Cube（MAD） | `PIPE_M` | `T.gemm_v0` 内部 |
| L0C → GM | FixPipe | `PIPE_FIX` | 用户写的 `T.copy` |

2. `init=(k == 0)` 让**第一次迭代清零 L0C 累加器**，后续迭代在其上累加。
   写成 `init=True` 会在每轮都清零，结果只等于最后一次 K 分块的贡献；
   写成 `init=False` 则第一次迭代不清零，结果里混进残值（P9）。
3. L1 只有 512 KB，放下整个 K 列不现实；只开 `K_L1` 列就能沿 K 反复复用同一块 L1，
   把「搬入量」与「片上容量」解耦。§5.4 的第 3 条就是这个约束。
4. `128 × 256 × 4 = 131072` 字节 = **128 KB**，恰好等于 Atlas A2 的 L0C 容量
   （`atlasA2_hardware_info.md`）。也就是说这个配置顶到了 L0C 上限，
   想再大 `block_N` 就得靠 `kL0Size` 调整 L0 双缓冲或换更小的累加 dtype。

**常见错误**：把 `loop_k = T.ceildiv(K, K_L1)` 写成 `K // K_L1` 却在非整除时
不做尾块处理——本示例假定整除，非整除 shape 需要尾块方案（P7）。

---

## E6

**思路**：`dual_copy` 的形状判据是「rank ≥ 2 时恰有一个尾部维度是 2:1；
rank = 1 时唯一维度必须是 2:1」。

**参考答案**

| 序号 | 结果 | 说明 |
| --- | --- | --- |
| 1 | 接受，**M-split** | 尾部倒数第二维 128→64，`dual_dst_ctl=0b01` |
| 2 | 接受，**N-split** | 尾部最后一维 256→128，`dual_dst_ctl=0b10` |
| 3 | **拒绝**（`ValueError`） | 256→257 不是 2:1 |
| 4 | 接受 | rank=1，唯一维度 4096→2048 是 2:1（`split_candidates=[(-1, 2)]`） |
| 5 | 接受，**N-split** | 前导维相同，最后一维 256→128 |

关于「能不能单独把纯 Vector kernel 变成两 AIV」：**不能**。
`dual_copy` 的 docstring 明确写它是 **mixed-kernel operation**——用户必须在同一个
kernel 里也提供 Cube 侧的工作；编译器按这个契约处理，不会替你合成 Cube 侧工作。

**常见错误**：忘记「静态已知的整体 extent 必须是偶数」；奇数会直接报错而不是
四舍五入（docstring 原文：odd extents are rejected rather than rounded）。

---

## E7

**思路**：L0C 是 Cube 的累加器，它的容量直接限制输出 tile 的面积。

**参考答案**

1. 约束是 `block_M × block_N × sizeof(accum_dtype) ≤ L0C 容量`。
   L0C 占用算式：`L0C_bytes = block_M * block_N * bytes(accum_dtype)`。
   （多缓冲不适用于 L0C——它是累加器，整块 tile 全程占用。）
2. `131072 B / 4 B = 32768` 个 fp32 元素。`block_M = 128` 时
   `block_N ≤ 32768 / 128 = 256`。所以 `block_N` 最大 **256**（不考虑 layout padding）。
3. 正文 §5.4 给的路是调 `kL0Size`（以及相应调整 L0 tile），
   本质是**改变 L1→L0 的 K 方向切分粒度**，让 L0A/L0B 的双缓冲装得下、同时把
   `block_M×block_N` 的面积做上去以提高 L0C 利用率。文档给的推荐：
   half 精度、`block_M=128, block_N=256` 时 `kL0Size=64`。
4. 在 `../code/12-tuning-sweep/` 下跑：

```bash
python3 tune_sweep.py --stages 1
```

你会看到 `BM=256 BN=256 ...` 这一行被判 `OVER: l0c: need 262144 B > cap 131072 B (2.00x)`，
而 `BM=128 BN=256` 的 `l0c=128KB` 是 OK。

**常见错误**：以为「L0C 只是缓存、超了会自动变慢」。它是配额，超了要么编译失败，
要么被后端拒绝。

---

## E8

**思路**：§7.5 给了三种模式的定义；M 树 `references/synchronization.md` 给了
「条件执行」下的差别。

**参考答案**

1. `iteration` 用**扁平化后的词法循环下标**（迭代号取模版本数）；
   `counter` 用**活跃 buffer epoch 的计数器**（只在一次完整 epoch 完成后推进）。
2. 当 `active(i)` 为假时：`iteration` 模式下槽位时钟**照走**（跳过的迭代也算一个
   词法迭代），所以 handshake 跟着词法 epoch 走；`counter` 模式下**跳过就不推进**
   计数器，槽位选择与 handshake 跟着活跃 guard 走。
3. `auto` 在以下情况选 `counter`：条件执行、不规则循环（外层依赖的循环界、带 break
   的循环）、多个 owner，或与 stage 组合后 `iteration` 不合适时。
4. 因为 M 树里编译器会**按注解自己给 buffer 加版本维**。如果你已经在 buffer 形状里
   手动加了一个 stage 维，再加注解就等于要求双缓冲两次（浪费容量，甚至导致
   capacity 超限或错误的槽位推断）。

**常见错误**：把 `num_stages` 与「物理槽位数」当成一回事。`num_stages` 是**上界**，
最终版本数由 Z3 调度器结合生命周期与容量决定（§8.4）。

---

## E9

**思路**：ReLU 是逐元素运算，必须在 **Vector** 上做；而 Cube 的输出在 **L0C**。
所以关键是「L0C 的结果怎么交给 Vector」。

**参考答案**

1. 改动思路：Cube 算出 `C_L0` 后，先把它搬到一个 Vector 能读的地方，再做 ReLU，最后写出。
   在 A 树上最直接的路是 `L0C → GM → UB`（两段拷贝），或者用 M 树的 `T.dual_copy`
   把 `L0C → UB` 做掉。省搬运的做法是**不要多写一次 GM**：即
   `L0C → UB`（单次拷贝）→ Vector ReLU → `UB → GM`。
2. 因为 L0C 是 **Cube** 的输出/累加器，Vector 单元**不能直接读写 L0C**；
   Cube 与 Vector 之间必须经过 GM/L2（或在 M 树里由编译器用 workspace 机制中转）。
3. 改动后的数据流：

```
GM(A), GM(B) --MTE2--> L1 --MTE1--> L0A/L0B --Cube--> L0C
                                                        │  (L0C→UB 的通路)
                                                        ▼
                                                       UB --Vector(ReLU)--> UB --MTE3--> GM(C)
```

4. 若换 M 树：`T.dual_copy(C_l0c, C_ub)` 一行就把 L0C 的结果劈给两个 AIV 落到 UB，
   再用 `T.SimtVF` 做 ReLU，最后 `T.dual_copy(C_ub, C[...])` 写回。相比 A 树手写，
   省掉了：`vid` 手动切分、L0C→UB 的搬运细节、以及这些搬运之间的 set/wait 配对
   （编译器自动插同步）。

**常见错误**：想着「在 L0C 上做 ReLU」。Cube 只做矩阵乘，L0C 上没有通用的逐元素运算；
而且 Vector 读不到 L0C。**必须先搬到 UB。**

---

## E10

**思路**：这是流程题。答案是「怎么测、判据是什么」，不是数值。

**参考答案**

1. `python3 ../01-env-check/env_check.py` 应显示 `binary:bisheng`、`env:ASCEND_HOME_PATH`、
   `import:tilelang`、`torch.npu.is_available` 均为 `ok`。若不齐，先修环境。
2. 上板采集：

```bash
msprof op --kernel-name="vec_add" python vec_add_adapter.py --m 1024 --n 1024
```

判据：在 msProf 输出的「计算内存热力图 / Roofline 瓶颈分析图」里看这次运算是
**偏内存受限还是计算受限**（逐元素 add 的算术强度极低，预期偏内存受限——
这只是一条**预期**，要用你的图证伪或证实）。
3. 记录 `aic_mad`、`aiv_vec`、`aiv_mte2`、`aiv_mte3` 四个比值（口径：M 树
   `do_bench(backend="msprof_detail")` 返回的 `KernelProfile`）。若你只有 A 树，
   请显式说明你用的是 msProf 的哪张图/哪张表，并给出你采用的替代口径。
4. 判据（§8.3）：逐元素 add 的情况下，若 `aiv_mte2`/`aiv_mte3` 明显高于 `aiv_vec`，
   结论是「Vector 侧受搬运带宽限制」；下一步要调的单一变量建议是
   **多缓冲深度**（`num_stages`）或**每核 tile 大小**，二者一次只动一个。
5. 记录表模板来自 `python3 ../12-tuning-sweep/tune_sweep.py` 的输出；
   把设备型号、CANN 版本、torch_npu 版本、哪棵树、pass config、正确性结果、
   延迟中位数、瓶颈结论、原始样本填进去。

**常见错误**

* 只跑一次 `do_bench` 就下结论（msprof 后端不带 median 选择）。
* 把 AIC 与 AIV 的比值相加当成统一时间轴占比。
* 改了多个变量（tile + stages + pass config）之后比较延迟——结论不可归因。

---

## E11

**思路**：按 §7.1 的对照表逐项替换。核心是「命名空间 → 分配 → 网格/子核 → 计算原语 →
同步」。

**参考答案**

A → M 至少要改这 8 处：

1. `import tilelang.language as T` → `import tilelang.ascend.language as T`；
2. `@tilelang.jit(...)` 加 `target="ascend"`；
3. `T.alloc_ub(...)` → `T.alloc_shared(...)`（M 树里 `alloc_shared` 就是 UB）；
4. `T.Kernel(m_num * n_num, is_npu=True) as (cid, vid)` →
   `T.Kernel(num_blocks) as bx`（无 `threads=`、无 `vid`）；
5. 二维坐标还原：M 树如果上 `T.Persistent`，改用 `for tile in T.Persistent([...], num_blocks, bx)`；
6. `T.Scope("V")` / `T.Scope("C")` → 去掉（M 树按操作自动判定核），或改用
   `T.SimtVF`/`T.SimdVF` 明确表达 Vector 区；
7. 逐元素计算：`T.tile.add(c, a, b)` → 可在 `T.SimtVF(threads=N)` 里写
   `T.Parallel` 的 `c[i] = a[i] + b[i]`；
8. 两个 `T.barrier_all()` → 删掉（M 树 `InsertSync` 自动插入）。

能删掉的同步：所有**核内、PIPE 之间**的显式 barrier/set/wait 都可以删，
因为 M 树默认启用 AutoSchedule 并在 `InsertSync` 阶段推导依赖、插入成对 flag
（`tilelang/ascend/pipeline.py:106-115`）。**跨核（Cube↔Vector）的交接**在
识别出 GM-workspace 模式后也由 `TL_ASCEND_AUTO_CV_*` 相关的自动机制处理，
但要注意它只覆盖「被识别的模式」。

从 M 树照抄回 A 树，最可能先失败在 import 后的**分配函数名**（`T.alloc_shared` 在 A 树
指 CUDA 风格的 shared，而不是 UB）或 `T.Kernel` 不接受 `is_npu=` 之外的子核约定，
典型报错是 `AttributeError` 或编译期 scope 检查失败。

**常见错误**：改了 API 名但保留 `vid` 的手工切分——M 树里 `vid` 不存在，
子核要用 `T.MixedKernel(...) as (bx, sid)`。

---

## E12

**思路**：ACLGraph 治的是 **Host-bound**（下发开销），不是设备计算。

**参考答案**

1. **该入图**。20 个小算子、每个单独 launch，Host 侧的下发开销会累加，
   正是 capture+replay 的主场。判据：`dur_ns`（设备侧）远小于端到端时间。
2. **不一定**。一个 8192³ 的大 GEMM 本身是设备侧大块计算，launch 开销占比很小；
   入图通常收益有限。判据：比较「单算子 launch 时间」与「算子执行时间」的比值。
3. `torch_tl_ascend` 解决的是**把 TileLang 算子作为一等公民接进 PyTorch** 的问题：
   它把算子编译/打包成 Python C 扩展（`torch.ops.tl_ascend.*`），并支持 libtorch C++ 侧
   集成。相比直接 JIT 出 kernel 收 `.npu()` 张量，多出的是：**注册成 torch 算子
   （可被 PyTorch 的 dispatcher 识别）、可打包分发、可被 C++ 应用调用**。
4. 支持反向传播时，§9.4 清单里最关键的几项是：**输出语义**（in-place 不设 `out_idx`；
   有输出才设）、**设备上下文一致**、**dtype/布局一致**，以及（如果要接进
   `torch.compile` 图）**不要在 kernel 里夹带 host 侧同步**。autograd 还需要额外的
   反向 kernel 与 `torch.autograd.Function` 包装——这是本章范围之外的下一步。

**常见错误**：把 ACLGraph 当成「图编译优化」。它**不承载图编译和图优化**，
只做 capture+replay 的下发加速（A 树编程指南 §7.1 原文）。
