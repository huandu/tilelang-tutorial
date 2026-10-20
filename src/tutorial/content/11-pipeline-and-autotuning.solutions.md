# 第 11 章 习题答案

与 `exercises.md` 一一对应。每题给「解题思路 + 参考答案 + 常见错误」。
所有涉及数字的地方都标明是**模型输出**还是**厂商资料**；本章没有实测性能数字。

---

## E1 参考答案

**解题思路**：回到「一份 buffer」这个物理事实上——生产者要写的那块内存，正是消费者
正在读的那一块。

1. 因为 `A_shared` / `B_shared` 只有一份。第 `ko+1` 轮的搬运必须等第 `ko` 轮的
   `T.gemm` **读完**才能覆盖，所以两件事在时间上无法重叠，只能首尾相接：
   每轮 = `latency + compute`（模型里 `latency` 已含发出成本）。
2. `issue` 是发出一次搬运要占用搬运引擎多久；`latency` 是数据从发出到可读的飞行
   时间；`compute` 是一次计算占用计算单元多久。
3. 因为二者差两个数量级。第 1.2 节引用的资料里，未缓存的 local/global 访存是
   「hundreds of clock cycles」，而多数算术指令是「typically 4 cycles」。如果只记住
   一个「搬运耗时」的合成数字，就没法解释为什么「让计算等数据」特别贵，也推不出
   「需要几份 buffer 才能盖住延迟」这个结论。

**常见错误**：把 `issue` 和 `latency` 当成一回事，于是得出「2-stage 永远够用」——
实际上一份 buffer 里的等待主要来自 `latency`，而不是 `issue`。

---

## E2 参考答案

1. **stage**：一条被调度的语句属于管线的第几级，数字越小越早，由 `stage=` 给出。
   **order**：同一级（以及整体）里语句的发射顺序，由 `order=` 给出。
   **depth**：缓冲区份数，由 `num_stages=` 直接给出，或在手动模式下由
   `max(stage) + 1` 推出。
2. `num_stages=4` → 深度 4。`stage=[0, 0, 3]` → 深度 `max(stage) + 1 = 4`
   （注意最大值是 3，不是 2）。
3. `num_stages=0` 是**关闭流水**：`src/ir.cc:138-144` 里只有 `if (num_stages > 0)`
   才写 `num_stages` 注解，所以这个循环在下游看来根本不是流水循环。
   `num_stages=1` 按 docstring 是「生产者与消费者之间最多留几份 buffer」，
   也就是**只留一份**——性能上等价于不做重叠，但它仍然是一个带注解的流水循环，
   下游会走流水那条（包括屏障分配等）路径。两者在自动调优的搜索空间里必须区别对待。
   另外值得记一笔：源码把「缓冲区份数」（`num_stages`）与「SW 流水级数」
   （`max(stage)+1`）当成两个概念处理
   （`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:3628-3638`），
   `num_stages` 才是权威的深度来源。

**常见错误**：把 `max(stage)+1` 误写成 `max(stage)`；或者以为 `stage` 的值必须从 0
连续到 N−1（不必，`[0, 0, 3]` 合法，只是中间两级没有语句）。

---

## E3 参考答案

1. 流水优化的是**吞吐**。单次操作的耗时（延迟）几乎不变；变的是单位时间能完成多少轮。
2. **(b)** 最值得。理由是填充的代价与轮数无关，512 轮的 reduction 能把填充摊薄到
   几乎可以忽略；稳态收益（模型里 2.11× 这个量级）能完全兑现。(a) 只有 3 轮，
   填充占主导；它更该考虑的是别的优化（合并算子、加大 tile）。
3. (c) 的瓶颈在搬运。稳态步长是 `max(issue, latency/depth, compute)`——当搬运那一项
   最大时，加深流水只是让搬运之间的间隔变小，但 `compute` 已经不是限制项了，
   加深不会让总时间继续下降。应该去优化搬运本身：提高每次搬运的字节数（加大
   `block_K`）、改善访问的连续性、或者换一种数据布局。

**常见错误**：认为「加 stage 总是能加快」——在 (c) 上就会得到「加了深度没变化，
但片上占用涨了」的结果。

---

## E4 参考答案

1. 它对应「这一轮循环**先搬后算**」。规划中途为了让「搬运跟着它的最后一个消费者排」，
   会把搬运排到 order 的末尾；而真实的一次迭代应该是「先搬下一块，再算这一块」。
   循环右移把搬运挪回开头。
2. 如果不做这一步，`gemm` 的 stage 会是 `num_stages`，于是
   `max(stage) + 1 = num_stages + 1`——深度比用户要的多一份，多占一整份 A/B tile，
   而且语义上也多了一格空转。`pipeline_planning.cc` 里那段代码的注释说的
   「shrink the stage offset by 1」就是这个意思。
3. 典型「两搬一算」的结果：

```text
copy A → (stage=0, order=0)
copy B → (stage=0, order=1)
gemm   → (stage=2, order=2)

深度 = max(stage) + 1 = 3 = num_stages
版本数 = use - def + 1 = 2 - 0 + 1 = 3
```

**常见错误**：把 `(stage, order)` 写成 `gemm → (stage=3, order=2)`，忘了收尾整形会减 1。
可以用 4.5 节的容量公式自检：深度必须是 3（而不是 4），否则 256×256×64 的 tile 会
算成「放不下」。

---

## E5 参考答案

1. 因为片上要放的是**输入 tile**，不是输出 tile。`block_M×block_K` 是 A 的一块，
   `block_K×block_N` 是 B 的一块。累加结果 `block_M×block_N` 在 fragment（寄存器）里，
   不占 shared。`block_M×block_N×block_K` 是这一轮的乘加**次数**，不是字节数。
2. 它是上界，因为它假设两个 tile 都拿到满额版本、且完全不复用。实际占用会因
   `MergeSharedMemoryAllocations` 更小——那个 pass 会合并 shared 分配，并让
   live range 不重叠的存储共用空间
   （`_vendor/tilelang-v0.1.15/src/transform/merge_shared_memory_allocations.cc:1576-1660`）。
3. 因为版本数是 `use - def + 1`。同一 stage 内写后立即读，`use - def + 1 = 1`，
   一份就够：生产者写完就立刻被读掉，不存在「读者还在读、写者要覆盖」的时间窗，
   所以不需要复制（`ComputeBufferVersions` 里 `def == -1` 时直接返回 1，
   `num_versions >= 2` 的分支才做多版本判断）。

**常见错误**：把 `sizeof(dtype)` 漏掉，或者按 `block_M×block_N` 估输出 tile——
后者在 `block_K` 很大时会严重低估。

---

## E6 参考答案

1. 当 `use - def + 1` 恰好等于 2，且**找不到**任何一对
   「写者 + 读者」同时满足
   `order(写) < order(读)`、`stage(写) < stage(读)`、访问区域重叠 时，
   版本数从 2 减回 1。
2. 安全的原因是那三个条件合起来正是「写者可能在读者还没读完时就覆盖同一块内存」的
   完整判据：stage 差保证了它们真的会被重排到不同的逻辑迭代上，区域重叠保证了它们
   真的碰同一块字节。三者不同时成立，就不存在冲突，一份 buffer 足够。
   （`ValidatePipelineBody` 会先保证依赖方向正确，所以这里只需要检查「是否可能
   同时可见」。）
3. 版本数会变成 1（因为 `MayConflict` 为 false → `need_multi_version = false` →
   `num_versions--`）。代价是：两个语义上不同的值被塞进同一块内存，如果它们分处
   不同 stage，就可能互相覆盖。所以真实内存布局里**不该**人为制造这种「不重叠」的
   假象——这类判断是由访问区域（region）算出来的，不是由下标字面值算出来的。

**常见错误**：以为 `num_versions--` 是一个无条件优化。它是被 `MayConflict` 严格
把关的。

---

## E7 参考答案

1. 各两项。数组对齐的是**可调度语句**：`T.copy` 与 `B[...] = ...` 两条；
   `base: T.int32 = ko * BK` 是可重放的标量 `Bind`，不占位置。
2. 编译器会接受它（为兼容旧代码保留），但**忽略** `base` 那一项，剩下两项按顺序应用
   到 copy 与 store 上。官方文档把这个行为写成「Legacy Annotation Form」，并建议新
   代码只标注真正有副作用的语句。
3. 两种情形：(a) `Bind` 读的 buffer 在同一个流水体内被写过（例如
   `value: T.float32 = A_shared[tx]`），此时这个读依赖流水生产者，不能重放；
   (b) 你想要「算一次、多处共享」的语义，那就应该把值**物化**成 buffer/寄存器写，
   而不是依赖 `Bind` 重放。

**常见错误**：数成 3 项并给 `base` 也编一个 stage，然后困惑为什么行为不符合预期。
另一个常见错误是相反方向：把一个真实的 producer 当成 `Bind` 省略掉，导致流水体里
少了一个 stage。

---

## E8 参考答案

1. 顺序是：`num_stages` → `tl_pipelined_num_stages` → `software_pipeline_stage`
   的最大值 + 1。这个顺序对应「信息的确切程度从高到低」：用户显式给的深度最权威；
   没有就用 `PipelinePlanning` 记下来的原始深度；再没有就用 stage 注解反推；
   三者都没有就返回空（`nullopt`），表示这不是流水循环。
2. 因为下游有些 pass（内部 buffer 版本化、warp specialization）把 `num_stages` 当作
   「这里有一个活动流水」的标记，而它们**不支持嵌套流水**。规划完成后，深度这个信息
   已经被翻译成 stage/order 注解了，把旧键删掉可以避免下游误判成嵌套。
   （源码注释：`pipeline_planning.cc:1287-1293`。）
3. 返回空（`nullopt`）。下游会把它当成普通循环，不做流水改写——也就等于
   `num_stages=0` 的效果。

**常见错误**：以为 `num_stages` 键会一直留在 IR 上，于是去 dump IR 找不到它。
规划之后应该找 `tl_pipelined_num_stages` 和 `software_pipeline_stage`。

---

## E9 参考答案

1. **硬约束**是第四条 `shared memory over budget`——超了会直接编译/启动失败。
   **正确性**相关的是 `tile not a multiple of 16`（对齐要求）和
   `tile does not divide the problem`（会导致越界或结果错误，视实现而定）；
   `illegal thread count` 两者都涉及（既是硬件合法性，也影响正确性）。
2. 因为流水对「迭代轮数」有假设：prologue / body / epilogue 三段的分界是
   `max_stage_`，如果最后一轮是部分 tile，那么每一级 stage 在最后一轮要处理的范围
   都不一样，需要额外的边界逻辑。本章的模型与 `code/01` 都明确假设「每轮计算量相同、
   没有尾部不整除的迭代」；把整除作为可行性条件，是把这个假设显式写进搜索空间。
   部分 tile 的处理属于第 14 章的范围（那里的做法通常是 OOB padding 或专门的
   尾巴 kernel）。
3. 会发生至少两类后果：
   * **浪费编译时间**：不可行的候选也要走一遍编译，很多会以编译错误或启动失败告终；
     调优器的 `timeout` 是给测量用的，编译失败会以异常的形式被记录（
     `AutoTuner.run` 里每个候选都有 `Compilation failed ...` 的路径）。
   * **污染结论**：如果只剩少数候选活下来，最优结果只是在「幸存者」里最好，
     不能代表这个形状的可达性能。更糟的是如果全部失败，整轮调优一无所获。

**常见错误**：认为「让调优器自己筛掉不可行的候选就行」。筛选是它被迫做的兜底，
不是设计。

---

## E10 参考答案

**动手题的思路**：`pipeline_timeline.py` 的 `required_depth()` 已经把公式实现进去了，
但要真正回答「公式对不对」，必须自己跑一遍看 `makespan` 在哪里停止下降。

**1. 固定 `issue=1, compute=20`，扫 `latency`**（本机实际运行，`--iterations 16`；
「最小够用深度」= `compute_util` 第一次达到 100% 的 stage 数）：

```text
latency | 公式 1+ceil(L/20) | 实测最小够用深度
   10   |        2          |        2
   20   |        2          |        2
   40   |        3          |        3
   80   |        5          |        5
  120   |        7          |        7
```

五个点全部吻合。所以在这个模型里，`1 + ceil(latency / compute)` **是精确判据**，
不只是估计。

**2. 固定 `latency=30`，扫 `compute`**：

```text
compute | 公式 1+ceil(30/C) | stages=2 的 compute_util
   5    |        7          |   31%
  10    |        4          |   53%
  20    |        3          |   82%
  30    |        2          |  100%   ← 从这个点起 2 就够了
  40    |        2          |  100%
```

也就是说：当 `compute ≥ latency` 时，深度 2 就够了。

**3. 为什么 stage 数不能是常量**：从上面两张表看，同一个 `latency=30` 下，
够用深度随 `compute` 从 7 变到 2；同一个 `compute=20` 下，够用深度随 `latency`
从 2 变到 7。而真实机器上 `latency`、`issue`、`compute` 又都随 tile 尺寸、dtype、
访存模式变化。所以「开 3 层」这种常量建议只在特定配置下成立。

**常见错误**：只看 `makespan` 的绝对值就下结论，而没看它在哪个深度停止变化。
另外注意 `--iterations` 要够大，否则填充还没被摊薄，曲线看不出平台。

---

## E11 参考答案

**1. 128×128、fp16 时的最大深度**（`smem_budget.py` 实测输出）：

```text
block_K = 64 → 每 stage 32768 B → 最大 7 层
block_K = 16 → 每 stage  8192 B → 最大 28 层
```

差 **4 倍**：`block_K` 减到 1/4，每份 tile 也减到 1/4。

**2. 最大可行 stage 数 = 3 的组合**（`--sweep` 输出的表里，228 KiB 预算）：

```text
block_M x block_N | BK= 16 | BK= 32 | BK= 64
   64 x 64        |   57    |   28    |   14
  128 x 128       |   28    |   14    |    7
  128 x 256       |   19    |    9    |    4
  256 x 128       |   19    |    9    |    4
  256 x 256       |   14    |    7    |    3   ← 只有这一个等于 3
```

等于 3 的组合只有 `(256, 256, 64)` 这一行。

**3. 可以放进评审清单的规则**：

```text
修改 num_stages 前必须做的检查：
  1. 用 num_stages × (block_M×block_K + block_K×block_N) × sizeof(dtype)
     算出新配置的片上占用上界；
  2. 与目标机器的 shared 预算比较（NVIDIA 侧参考 per-SM 可配置上限）；
  3. 确认新深度确实还落在「1 + ceil(latency/compute) 判据」没有满足的区间里；
     如果已经满足，加深只会增加占用，不会变快；
  4. 如果预算是紧的，把第 1 步作为回归测试的一部分（例如放进 code/02 那种脚本）。
```

**常见错误**：只记住「256×256 的 tile 最多 3 层」这个结论，而忘了它是
`block_K = 64` 与 228 KiB 预算共同决定的——两个前提一变，数字就变。

---

## E12 参考答案

**动手题的思路**：题目的核心不是写出一份「最好」的空间，而是证明你写的空间**值得跑**。
所以答案必须包含「数量」与「为什么不是坏空间」两部分，两者都要有依据。

**一个合格的答案（示例）**：

```python
# 1. 维度与取值
block_M   = (64, 128, 256)
block_N   = (64, 128, 256)
block_K   = (16, 32, 64)
num_stages = (1, 2, 3, 4)
threads   = (128, 256)

# 2. 预剪枝规则
#    (a) tile 必须整除 M/N/K（本例 4096 能被上述所有取值整除）
#    (b) num_stages × (block_M*block_K + block_K*block_N) * 2 <= 228 KiB

# 3. 数量（4096³、fp16、228 KiB 预算）
#    剪枝前：3 × 3 × 3 × 4 × 2 = 216
#    剪枝后：214（被拒的两个是 256×256×64 + num_stages=4 的两个线程变体）
```

**为什么它不属于三种坏空间**：

* **不是「全部不可行」**：`legal = 214 > 0`。可以用
  `code/03-autotune-config-space/config_space.py` 的 `well-formed` 模式复现这组数字。
* **不是「只剩一个可行」**：214 个候选横跨 3 种 tile 形状组合与 4 种深度，维度上有
  真正的取舍（大 tile 省全局往返但容量受限，小 tile 相反）。
* **参数确实被用上**：内核里必须写 `T.Pipelined(n, num_stages=num_stages)`、
  `threads=threads`、`T.alloc_shared((block_M, block_K), ...)`，逐项对应到候选 dict
  的键。**改完之后要确认生成的 kernel 里每个参数都出现了**——否则会出现「所有候选
  性能相同、调优照样给出一个赢家」的假成功（这也是 `AutoTuner` 会检查
  `Unused keys in config` 的原因）。

**第 4 问（可选，需要 GPU）的答题要点**：

* 把 `artifact.config` 抄成函数默认值，写在带注释的位置（说明机器型号与问题形状），
  并保留一个 `enable_autotune` 开关。
* **不要固定 `latency`**：它是这次测量在特定机器、特定时刻、特定负载下的结果，
  换机器或换形状就失效；而 `config` 是一组可复现的调度参数。

**常见错误**：把空间开得极大（例如 `block_K` 从 8 取到 256）却不做预剪枝，
结果大部分预算花在编译非法候选上；或者只在 4096³ 上调优，却把结果当成所有形状的
默认值。

---

## 关于本章答案中的数字

以下数字全部来自**本章模型或厂商公开资料**，不是实测：

| 数字 | 来源 | 口径 |
| --- | --- | --- |
| 400 / 220 / 190 周期与 1.00/1.82/2.11× | `code/01-pipeline-timeline/pipeline_timeline.py` | 模型：`issue=1, latency=30, compute=20, iterations=8` |
| E10 的两张扫描表 | 同一个模型 | `--iterations 16`，各 `latency` / `compute` 取值 |
| 最大深度 7 / 28 / 3 / 4 倍 | `code/02-shared-memory-budget/smem_budget.py` | `num_stages × (A + B)` 的上界，预算 228 KiB |
| 216 / 214 个候选 | `code/03-autotune-config-space/config_space.py` 与 `code/05` 的 `grid_configs` | 4096³、fp16、228 KiB 预算 |
| 228 KiB、256 KB shared+L1 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md` | H100，厂商公开资料 |
| 访存「hundreds of clock cycles」、算术「typically 4 cycles」 | `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md` | 未缓存 local/global 访存；compute capability 7.0 |

**没有任何一条来自本机实测**——本章写作环境没有 GPU/NPU。
