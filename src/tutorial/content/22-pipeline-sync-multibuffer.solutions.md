# 第 22 章 习题答案

说明：每题给“解题思路 + 参考答案 + 常见错误”。所有源码行号基于
`_vendor/tilelang-v0.1.15/`；模型数字来自 `code/02`–`code/04` 的**本机 Python
3.11 运行输出**（模型结果，不是硬件测量）。

---

## E1 ★

**思路**：先把“依赖”说清楚，再说“距离”是这个依赖跨越了几次迭代。

**参考答案**

1. 距离 `d` 表示“消费者第 `k + d` 次迭代用到生产者第 `k` 次迭代的结果”。
   `d = 0` 是同迭代依赖（在同一次循环体内按顺序就能满足，不需要多版本）；
   `d ≥ 1` 是循环携带依赖（要么多留版本、要么让消费者等）。
2. `-1` 是“分析阶段还没确定距离”的占位。源码注释给出了它的两种归宿：
   如果那个 storage 被选中做多缓冲，就把距离解析成 **ring 宽度（版本数）**；
   否则就是 **1 次迭代**（消费者等一轮）。对应 `CollectSyncPoints` 里的
   `if (loop == enclosing_owner->loop && distance < 0) distance = info->num_versions;`
   与紧接着的 `if (distance < 0 && dep.kind == DependencyKind::kData) distance = 1;`。
3. 两个理由：(a) 重排可能改变程序语义，原顺序是有依赖的，必须先证明没有会出错的
   跨迭代边；(b) 手工下标（如 `A_shared[ko % 2]`）把“版本协议”的责任推给用户，
   编译器无法验证也无法推导。正确的做法是声明意图（`stage`/`order`/`num_stages`），
   由编译器推导版本数与同步。

**常见错误**：把 `d = 0` 说成“同一语句内的依赖”。同一条语句里也可能有
RAW（例如 `C_local[i] = A_shared[i]`），但本题问的是**跨迭代**的语义。

---

## E2 ★

**参考答案**

1. 两条校验（`src/transform/inject_pipeline.cc:3100-3138`）：
   * **`order` 唯一**：`ICHECK(!used_orders.count(order)) << “ValueError: Two
     statements in the software pipeline cannot have the same order”` ，
     防止“同一时刻有两条语句争着发射”，否则 prologue/body/epilogue 的展开
     次序不确定。
   * **生产者不晚于消费者**：对有 buffer 依赖的 `(src → dst)` 要求
     `stage(src) ≤ stage(dst)`；同 stage 时还要 `order(src) < order(dst)`
     防止“把消费者排到生产者前面”，那会让依赖方向与时间方向矛盾。
2. 违反第二条：`copy` 是生产者、`gemm` 是消费者，而
   `stage(copy) = 1 > stage(gemm) = 0`。运行时会报
   `ValueError: statement ... in stage 0 cannot depends on statement ... in a later
   stage 1`。
3. 报 `ValueError: Two statements in the software pipeline cannot have the same
   order`。关键词是 `cannot have the same order`。

**常见错误**：以为 stage 相同就可以任意交换顺序。同 stage 内顺序受限：
内部依赖必须保持 `order(src) < order(dst)`。

---

## E3 ★★

**思路**：先把每个 buffer 的 `def` / `use` 从表里读出来，再用公式。

**参考答案**

1. `A_shared`：def = 0（`copy_a` 写），use = 2（`gemm` 读）→ **3**。
   `B_shared`：同理 → **3**。
   `C_local`：def = 2（`gemm` 写），use = 2（`scale` 读，同 stage）→ **1**。
2. `depth = max(stage) + 1 = 2 + 1 = **3**`。
3. 用 **`num_stages` = 4**。源码注释（`inject_pipeline.cc:3626-3632`）：
   “Use the actual pipeline depth (number of buffer copies) for barrier sizing,
   not the SW pipeline stage count (max_stage + 1)”，随后
   `pipeline_depth = pipelined_num_stages.defined() ? num_stages : max_stage + 1`。
   理由是：**barrier/版本槽的数量要跟“同时在飞的份数”一致**，而那个数由
   `num_stages` 决定；stage 数组的跨度在自动规划时刚好相等，但手写 stage 时不相等。

**常见错误**：用 `max(stage)+1` 给 barrier 定尺寸。手写 stage 时这会少分配槽位，
表现为偶发的 parity 错配。

---

## E4 ★★

**参考答案**

1. **不是。** 守卫写作 `if (num_versions >= 2)`，所以 3、4、…… 也会进入这段逻辑；
   注释里说“A special case when `use - def + 1 == 2`”，但那是注释对意图的描述，
   与代码守卫不完全一致。在规整的 GEMM 形态里，`use − def + 1 ≥ 3` 时总能找到
   一对相交的 (writer, reader)，所以实际上只在 `== 2` 时才看到效果。但严格读代码，
   不能断言“只在 ==2 生效”。
2. 例如：`copy` 写 `S_shared[0:64]`（stage 0, order 0），另一个语句
   `prep` 写 `S_shared[64:128]`（stage 0, order 1），`consume` 读
   `S_shared[64:128]`（stage 1, order 2）。`S_shared` 的 def = 0、use = 1 →
   `use − def + 1 = 2`。检查 writer/reader 对：只有 `prep → consume` 满足
   `order(prep) < order(consume)` 且 `stage(prep) < stage(consume)`，且两者区域
   **相交**。所以这里 `need_multi_version` 为真。反过来，若 `consume` 只读
   `S_shared[0:64]`（`copy` 写的那段），则 `copy → consume` 也满足两个不等式且相交，
   依旧为真。要构造 `need_multi_version == false`，需要“写者与读者在时间上错开、
   且被读的区域不是被同轮或更晚的写者覆盖”这种形态。这在真实 kernel 里少见，
   通常是 `if` 分支下的部分写。
3. 若把 `MayConflict` 换成恒真，则只要存在“更晚 order 的 writer + 更早 order 的
   reader、且 stage 递增”的语句对，就会判定需要多版本。后果：
   * **正确性**：安全（宁可多留版本）；
   * **性能**：多分配共享内存，可能直接编译失败（超预算），或者挤掉本来能复用的
     空间。属于“保守但浪费”。

**常见错误**：断言“版本数一定等于 `use − def + 1`”。这是**上界**，
源码会收紧。

---

## E5 ★★

**参考答案**

1. `stage_expr` = 槽位号 = `(k − min) mod ns`；`parity_cycle` = 这是第几圈 =
   `((k − min) / ns) mod 2`。`ns = 3` 时：

   | k | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | stage_expr（槽位） | 0 | 1 | 2 | 0 | 1 | 2 | 0 | 1 |
   | parity_cycle | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |

2. 因为 **同一个槽位会被反复使用**：槽 0 在第 0 圈与第 1 圈各用一次。只知道槽位号，
   无法区分“我在等本次使用”还是“我在等上一次使用”。parity 就是这个区分位，
   它等价于第 11 章说的“第几轮到达”。
3. tcgen05 MMA 之类由 tile-op 内部管理 arrive 的 barrier，其 arrive 次数与槽位
   语义由那条指令自己定义（例如按 MMA 的发射轮次 arrive）。如果 ISP 再按 stage
   把槽位扩成 `num_stages × N`，**生产者的 arrive 目标与消费者的 wait 目标就会
   不一致**，典型症状是 wait 永远等不到，或者提前通过（读到不完整数据）。

**常见错误**：把 `parity_cycle` 当成 `k mod 2`。它是 `(k / ns) mod 2`。

---

## E6 ★★

**参考答案**

1. 三个来源与顺序（`src/transform/common/pipeline_utils.h:87-110`）：
   `num_stages` → `tl_pipelined_num_stages` → `max(software_pipeline_stage) + 1`。
2. 因为下游有些 pass（内部 buffer 版本化 `MultiVersionBufferRewriter`、
   warp specialization）把 `num_stages` 当作 **active pipeline marker**，
   并且**不支持嵌套流水**。保留它会让这些 pass 误以为还有一层流水。
   原文见 `src/transform/pipeline_planning.cc:1287-1293`。
3. **(c)**。前半句“取决于来源”对，(b) 错在“永远”，(a) 错在“永远”。
   `InjectSoftwarePipeline` 为 barrier 定尺寸时优先用 `GetPipelineNumStages`
   （它本身就按上面三级顺序读），定义不了才退回 `max_stage + 1`
   （`inject_pipeline.cc:3626-3640`）。

**常见错误**：认为 `num_stages` 与 `stage` 数组是同一件事的两种写法。
只在自动规划路径上它们等价。

---

## E7 ★★★

**思路**：严格按“排序 → 先 retire 再 allocate → best fit → 归还时合并”执行。
下面的偏移与本目录模型脚本的实际输出一致（可用 `code/02` 的 `pack()` 复算）。

**参考答案**

排序后：`P(0)`、`Q(2)`、`R(3)`、`S(6)`、`T(10)`。

1. 逐条走一遍：
   * `P`：free 为空 → offset **0**，arena_top = 32；
   * `Q`：free 仍为空 → offset **32**，arena_top = 48；
   * `R`：free 仍为空 → offset **48**，arena_top = 64；
   * `S`：`Q` 到期（end = 6）归还 `(32,16)`，但 16 < 24 放不下 →
     offset **64**，arena_top = 88；
   * `T`：`P`、`R`、`S` 全部到期，归还 `(0,32)`、`(48,16)`、`(64,24)` 并合并成
     `(0,88)` → 选 offset **0**。

   **arena = 88**，偏移 `{P:0, T:0, Q:32, R:48, S:64}`。
2. `P` 活到 12 之后：`T` 分配时 `P` 未到期，free 是
   `(48,40)`（`R` 与 `S` 归还后合并），`T` 被放到 **48**。**arena 仍是 88**
   （没有新的顶端分配），只是布局挪了位置。
3. 对齐 32 后：`P`→0、`Q`→32、`R`→**64**（32 对齐把 48 顶到 64）、
   `S`→**96**（`Q` 归还的 16 字节空洞放不下 24，只能开新顶端）、`T`→**64**
   （`R` 归还的 16 字节空洞是最佳适配）。
   **arena = 120**，比原来的 88 多出 **32 字节**。
4. 因为对齐会把“空洞的起始地址”向上取整，取整吃掉的那一段既不属于任何 buffer、
   也不足以容纳下一个请求。它就从“可用空洞”变成“永久碎片”。这就是为什么
   `MergeSharedMemoryAllocations` 里 `align_bytes` 与 `kSmemAlignmentMap`
   （TMA/MMA swizzle 的显式对齐）会直接推高 arena 大小。

**常见错误**：忘记“先 retire 再 allocate”。如果先分配再回收，`T` 在这道题里
会被放到 88，arena 变成 96，与真 pass 不一致。

---

## E8 ★★

**参考答案**（本机实际运行结果）

1. `num_stages=4` 时：
   * `epilogue`：no-reuse 160 KiB，**arena 128 KiB**；
   * `plain`：no-reuse 128 KiB，**arena 128 KiB**；
   * `carry`：no-reuse 192 KiB，**arena 192 KiB**。
   差异来源：`epilogue` 里的 `C_shared` 只在循环后用一次，活跃区间与 A/B 不重叠，
   被塞进 A 腾出的洞里（offset 0），于是 arena 比 no-reuse 少 32 KiB。
   `carry` 里的 `S_shared` 每个迭代都活，任何 tile 都躲不开，完全不能复用。
2. `carry` 从 **`num_stages=5`** 开始 `[OVER]`（arena 240 KiB > 228 KiB）。
   把 `--budget-kib` 调到 **小于 192**（例如 160）就能让 `num_stages=4` 也
   `[OVER]`，实测命令
   `python3 buffer_versions.py --scenario carry --num-stages 4 --budget-kib 160`
   输出 `[OVER]`。
3. 因为 `C_shared` 的活跃区间是 `[51, 53)`，在所有 A/B 区间 `[3, 51)` 结束之后；
   排序时它最后一个被分配，此时之前所有 buffer 都已归还，free list 里 offset 0
   是最大的空洞，best fit 自然选中它。
4. `--iterations 4` 时，**arena 与版本数完全不变**（仍是 128 KiB /
   `A_shared=4, B_shared=4, C_shared=1`），变的只有活跃区间的下标
   （例如 A 从 `[9,57)` 变成 `[9,21)`）。原因：arena 打包只看**区间的前后顺序与
   重叠关系**，不看绝对长度；只要 prelude/steady/epilogue 的相对次序不变，
   布局就不变。这也解释了为什么“循环轮数”不影响 shared 占用。

**常见错误**：以为迭代次数越多占用越大。那是对“模型输出”的误解；
轮数只影响时间，不影响空间。

---

## E9 ★★

**参考答案**（本机实际运行结果）

1. 默认（intra-core，8 槽，不允许复用）：

   | depth | allocator sum | 结果 |
   | --- | --- | --- |
   | 1 | 4 | 放得下，空 4 槽 |
   | 2 | 8 | 放得下，空 0 槽 |
   | 3 | 12 | **溢出**：保留 `acquire_A`、`acquire_B`（6 槽），溢出 `release_A`、`release_B` |
   | 4 | 16 | 溢出：保留 `acquire_A`、`acquire_B`（8 槽），溢出两个 release |

   第一次出现 `spilled` 是 **depth = 3**。
2. `--sharing` 把 `release_A` 与 `release_B` 标成可复用，`assign_bases` 把它们
   并到同一个物理通道（base 4，宽度取两者中较大的 2），于是 8 → 6。
   真 pass 里对应的机制是 **`FlagAllocator` 的 `reusable_pairs_`**
   （`src/ascend/transform/insert_sync.cc:1729`、`:1742`）：两个不可能同时在飞的
   同步点可以共用 base。
3. 因为跨核命名空间是 **16 槽**（`kCrossCoreFlagLimit`，`insert_sync.cc:2136`），
   而 depth=4 时需求正好 16，恰好放满（空 0 槽）。这也解释了为什么跨核同步
   “更贵也更宽”。
4. 例子：把“同一个 buffer 的 release 边”从“每个 slot 一条”改成“整圈一条”
   （即 ring 只有一份 release 事件，靠 ring 宽度隐式区分），同步点数量下降但每条
   需要的槽位变宽。反过来，“同步点变多、需求不变”的做法是：让新增的同步点
   都被判为**可复用**（例如时间上串行、不会同时在飞）。真实 kernel 里对应
   “同一个 hard event 上，不同 K-tile 的同步点复用同一批 flag id”。这正是
   `reusable_pairs_` 想抓的模式。

**常见错误**：以为溢出到 `asc_lock/asc_unlock` 之后同步就“没了”。溢出是**换了
原语**（事件标志 → 阻塞锁），语义还在，代价不同。

---

## E10 ★★

**参考答案**

1. 最小场景：ring 深度 2，槽 = `k mod 2`。
   第 `k` 轮：`copy → slot(k)`，然后 `gemm` 读 `slot(k)`。
   第 `k+2` 轮的 `copy` 会重新写 `slot(k)`。如果没有 WAR 屏障，
   `copy(k+2)` 可能在 `gemm(k)` 读完之前开始写，于是 `gemm(k)` 读到**半新半旧**
   的数据。是否出错取决于“copy 与 gemm 的相对快慢”。这正是**偶发**的来源。
2. 因为“有没有出错”是时序的函数，而不是输入的函数。一台机器上
   `copy` 恰好总比 `gemm` 慢，就永远不出错；换一个 tile 形状、换一个
   occupancy、换一个驱动版本，就可能出错。
   能反证的检验方案（不需要真跑）：
   * **构造性检查**：用 §4 的路径把读写集列出来，逐条问“这条边有没有覆盖”；
   * **放大竞态窗口**：把 `block_K` 调小、线程数调大，让 copy 与 gemm 的耗时接近；
   * **改变起点**：给循环加一个会改变的偏移（例如换 `K` 的非整除形状），
     让 prologue/epilogue 的守卫形态变化；
   * **开/关中间同步**：在 kernel 里人为插一个 `__syncthreads()`，如果结果从
     “偶发错”变成“稳定对”，就锁定是这一处的同步缺失（反之不能证明没有别的
     同步问题）。
3. 因为有 guard 时，“迭代编号”与“实际执行的 epoch 数”不再是同一个时钟。
   `iteration` 模式下槽位按**词法迭代号**取（`i mod nv`），被跳过的迭代照样推进
   时钟；`counter` 模式按**实际执行的 epoch** 取。生产者与消费者如果用了不同的
   时钟（例如一个在 guard 内、一个在 guard 外），就会算出不同的槽位，
   于是“等错了槽”，表现为偶发的数据错乱或死等。

**常见错误**：把“多跑几次没出错”当成证据。请记住 §5.1 的第一条建议。

---

## E11 ★★

**思路**：把稳态写成周期 `T` 的不等式。设 `depth = D`、带宽 `L = latency`、
计算 `c`、屏障 `b`。要让第 `k` 轮的 copy 能开始，必须等 `k − D` 轮的计算结束
再加一道屏障，于是 `D·T ≥ L + c + b`；而计算本身受 `T ≥ c` 约束。

**参考答案**

1. 无屏障时 `T_none = max(c, L/D)`。有 `war` 时 `T_war = max(c, (L + c + b)/D)`。
   两者相等（即 barrier 被完全隐藏）的条件是

   ```text
   (L + c + b) / D ≤ c   ⟺   D ≥ 1 + (L + b) / c
   ```

   代入 `L = 30, c = 20, b = 8`：`D ≥ 1 + 38/20 = 2.9` → **`D ≥ 3`**。
   这正是默认输出里 `depth ≥ 3` 时 `war` 与 `none` 完全相同的原因。
   （这就是第 11 章 `depth ≥ 1 + latency/compute` 判据在加入同步成本后的推广。）
2. `sync-copy` 的稳态步长是 **128**。它由 `latency + barrier` 决定
   （`L + b = 128`），与 `compute` 和 `depth` 都无关。因为在这种模型里每次拷贝
   都要等上一次拷贝落地。
3. 预测：`+cb` 与 `war+raw` 的**绝对**差距不变（都是一个 `b = 8`），
   但**相对**差距变小。实跑 `--compute 60`：

   ```text
   depth=2: none=60.0  war=60.0  war+raw=60.0  +cb=68.0  sync-copy=60.0
   ```

   `+cb` 的稳态是 68 = 60 + 8，而 `war+raw` 是 60：绝对差仍是 8 周期，
   相对差从默认配置的 40% 降到约 13%。另外注意 `sync-copy` 在这里**也变成免费**
   （60）：当计算远慢于搬运时，失去异步重叠不再有代价。这说明
   “异步到底值不值钱”取决于 `latency` 与 `compute` 的相对大小。

**常见错误**：以为“加了 barrier 就一定慢”。必需的 barrier 在深度足够时是免费的。

---

## E12 ★★★

**参考答案**

1. 安全，因为同一 PIPE 内的指令**天然按序执行**。依赖的具体函数是
   （`src/ascend/transform/insert_sync.cc:766-772`）：

   ```cpp
   // These are the same-pipe dependencies the materializer satisfies without a
   // barrier. MTE/FIX issue order alone does not protect overlapping accesses.
   bool HasImplicitSamePipeOrder(const std::string &pipe) {
     return pipe == "S" || pipe == "V" || pipe == "M";
   }
   ```

   注意注释的第二句：它明确说这个豁免**只适用于 S/V/M**。
2. 因为 MTE 系列虽然也是“同一个引擎”，但**发射顺序不保证访问顺序不冲突**
   （注释原话：“MTE/FIX issue order alone does not protect overlapping
   accesses”）。当两次访问的区域相交时，必须显式 `PipeBarrier`。
3. `MTE2_M` → `MTE2_MTE1` 是在**规范化事件名**：Cube 单元在 TIR 里被标记为
   `M`，但它实际依赖的搬运通路是 `MTE1`（L1→L0 的通路）。规范化保证
   （a）硬件 `HardEvent` 枚举有对应项，（b）同一个物理事件不会因为书写差异被
   分成两个 flag 域（`FlagDomain` 用 `GetHardEvent` 的返回值做 key，
   `insert_sync.cc:2083-2088`）。否则同一个依赖会占用两份 8 槽预算。

**常见错误**：以为“同一个 PIPE 就不需要同步”。对 S/V/M 成立，对 MTE/FIX 不成立。

---

## E13 ★★★

**参考答案**

1. WS 在前。`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:105-110`：
   `AutoWarpSpecialization → MaterializeWSSchedule → ProducerConsumerWarpSpecialized`；
   流水在 `:129-130`：`PipelinePlanning → InjectSoftwarePipeline`。
   也就是说 ISP 看到的已经是 WS 之后的 IR。
2. `InjectSoftwarePipeline` 的 TMA barrier 管理（`RewritePipelineTmaBarriers`，
   `inject_pipeline.cc:3616-3640` 的注释）只在 **kernel 里存在 TMA copy 且没有
   做 warp specialization** 时生效，条件写作
   “when TMA copies are present (without warp specialization)”。
3. 因为下游有些 pass（内部 buffer 版本化、warp specialization）把 `num_stages`
   当成 **active pipeline marker**，而它们**不支持嵌套流水**；保留它会导致误判。
   原文在 `pipeline_planning.cc:1287-1293`。
4. 两种症状：
   * **同一个位置上出现两层 barrier**（WS 插了一层、ISP 又插一层）→ 性能掉，
     对应 §5.2“多一个同步会掉性能”；
   * **一层都没有**（每一层都以为对方负责）→ 偶发数据错误，对应 §5.1
     “漏一个同步会偶发出错”。

**常见错误**：以为“多插同步只会慢一点”。它也可能把“谁负责”这件事搞乱，
从而在另一处少插一层。

---

## E14 ★（TileLang，不需要 GPU）

**参考答案（预期形态）**

1. 预期（取自上游单测对同一形态的断言，**不是本机跑出来的**）：

   | num_stages | `software_pipeline_stage` |
   | --- | --- |
   | 1 | `[0, 0, 1]` |
   | 2 | `[0, 0, 1]` |
   | 3 | `[0, 0, 2]` |
   | 4 | `[0, 0, 3]` |

   对不上的一行最可能是 **`num_stages=1`**：很多人预期它等于“不流水”，
   实际上它仍然是流水循环（只是注释里那句“把 copy 挪到 order 开头并整体减一”
   要求 `num_stages ≥ 2`，见 `pipeline_planning.cc:1266-1277`）。
2. A/B 的 `__shared__` 首维应该与 `use − def + 1` 一致：`num_stages=3` 时是 **3**，
   `num_stages=2` 时是 **2**（除非 §2.4 的收紧规则把 2 降成 1。那就说明这段代码
   在那个形态下不需要双缓冲）。
3. `cp.async.wait_group` 的参数**不保证单调**。它是“允许在飞的组数”，由
   `AsyncPipelineLoopWaitRelaxer` / `RelaxTrailingConsumerWaits` 在
   `InjectSoftwarePipeline` 里算出来（`inject_pipeline.cc:2493-2540`），
   与 tail drain 的形态有关。看到非单调不要立刻判定是 bug，先对照 IR。
4. `TL_LOWER_TRACE=terminal` 的 diff 里，`InjectSoftwarePipeline` 那一步应当出现：
   * `alloc_buffer` 的形状多一维（`(BM, BK)` → `(N, BM, BK)`）；
   * 每个 `A_shared[...]` 下标前面多一项 `(ko - min) % N`；
   * `ptx_wait_group` / `mbarrier_wait_parity` 被插入或改写；
   * 循环被拆成 prologue + 滚动 body + epilogue 三段。

**如果你和作者一样没有 TileLang**：请改为“阅读 `code/01/README.md` 与上游单测，
写出预期输出”，并在答案里明确列出**你验证了什么、没有验证什么**。作者本人只做到
“仅静态核对”（`py_compile` + API 名核对），没有运行过这段脚本。

**常见错误**：把“预期观察点”当作“已观察到的结果”。这两者在本书里必须分开写。
