# 第 22 章 流水线、多缓冲与同步插入

## 学习目标

读完本章，你应当能够：

1. 把「第 `k` 轮写、第 `k+d` 轮读」写成一条带**依赖距离（dependence distance）**的
   边，并说清 `d = 0` 与 `d ≥ 1` 在流水里的区别。
2. 独立推导 `num_versions = use_stage − def_stage + 1`，并解释它为什么是**上界**、
   以及源码在什么特例下把 2 降成 1。
3. 说清「深度」有两个来源（`num_stages` 注解、`max(stage) + 1`）以及
   `pipeline_depth` 与「SW 流水级数」在源码里被刻意区分的那个注释在讲什么。
4. 从 buffer 的读写集推出它需要哪些同步点（RAW / WAR / WAW + 循环携带），
   并说明 CUDA 侧 `ThreadSync` 与昇腾侧 `InsertSync` 各自用什么数据结构表达这件事。
5. 解释「漏一个同步为什么是偶发错误」「多一个同步为什么掉性能」，并给出
   同步点数量与吞吐之间的定性关系。
6. 在**没有 GPU** 的机器上设计并执行「改 `num_stages` 看 IR 变化」的实验，
   说出每一步应该看到什么。

## 前置知识

* 第 11 章：`T.Pipelined` 的用法、`stage`/`order`/`num_stages` 三个词、
  时空图与最小够用深度的判据。本章是它的深入版，**假设你已经会写流水循环**。
* 第 20 章：TIR / `SBlock` / pass pipeline 的读法，以及如何 dump IR。
* 第 10 章：shared memory 与 fragment 的区别。
* 第 6 章（可选）：NV 与昇腾在存储层级与同步模型上的差异。
* 会在本地装 `rg` 与 Python 3；读源码时用
  `rg -n "ComputeBufferVersions" _vendor/tilelang-v0.1.15/src/` 这类命令定位。

本章的算法描述**逐条对应 v0.1.15 源码**；凡是我无法在锁定版本里确认的说法，
都在句中标注「**待核**」。没有 GPU/NPU 时无法观察到的现象，也明确写出来。

---

## 1. 问题的由来：把「重叠」变成可判定的事

### 1.1 第 11 章留下的三个悬问

第 11 章把「为什么要流水」讲清楚了：把搬运与计算放到两条能同时工作的通路上，
每轮的成本从 `latency + compute` 降到 `max(compute, issue, (latency + compute)/depth)`。
但它留下三个没有回答的问题：

1. **谁决定哪条语句在 stage 几、按什么顺序发射？** 编译器自动推断的规则是什么，
   我手写注解时它又会校验什么？
2. **「多准备几份」具体多几份？** 为什么是 `use − def + 1`，而不是简单等于
   `num_stages`？
3. **多份 buffer 之间怎么保证不互相破坏？** 谁来插中间那道「等你读完我再写」的
   屏障，插在哪里，插多少？

这三个问题分别对应本章 §2、§3、§4。

### 1.2 一个反例：把 copy 全挪到前面，程序就错了

先看一个把「重叠」做错的最朴素方式。把第 11 章那个循环手写成
「先把所有搬运做完，再统一计算」：

```python
for ko in T.serial(T.ceildiv(K, block_K)):
    T.copy(A[by * block_M, ko * block_K], A_shared[ko % 2, :, :])
```

——这已经不是流水，而是「把循环体拆开重排」。它有两个问题：

* **它改写了程序语义**：`T.copy` 的顺序在原始程序里是有依赖的（后一轮的写可能
  覆盖前一轮还没读的数据），重排必须证明这些依赖没有跨迭代的边；
* **它引入了一个新的正确性问题**：`A_shared[ko % 2]` 这个下标是**用户手写**的，
  编译器不知道它要保证什么，也无法检查。

所以正规做法不是「让用户重排」，而是**让用户声明意图（stage/order、深度），
让编译器推导并物化**。这正是 TileLang 的分工：`T.Pipelined` 只声明
「这是一个流水循环」，剩下的三件事——排程、版本数、同步——全部由 pass 完成。

---

## 2. 软件流水的形式化

### 2.1 依赖与依赖距离（图 1）

**图 1 · 依赖距离** 见 `figures/01-dependence-distance.md`。它把「第 `k` 轮写、
第 `k+d` 轮读」画成一条带标注的边。三个定义：

* **访问集** `R(s)` / `W(s)`：语句 `s` 读、写哪些 buffer region。
  TileLang 里这就是 `SBlock` 的 `reads` / `writes` 字段，由
  `GetSBlockReadWriteRegion` 推出。
* **依赖边** `p → c`：`p` 的结果被 `c` 用到，因此 `p` 必须先发生。
* **依赖距离** `d`：`c` 的第 `k + d` 次迭代用到 `p` 的第 `k` 次迭代的结果。
  `d = 0` 是同迭代依赖，`d ≥ 1` 是**循环携带依赖**（loop-carried dependence）。

昇腾侧的依赖分析把这件事写成了显式的数据结构
（`_vendor/tilelang-v0.1.15/src/ascend/transform/auto_schedule/dependency_analysis.h:44-75`）：

```cpp
enum class DependencyKind {
  kData,           // 有向的生产者→消费者数据依赖
  kOwnerExclusion, // 同一个可多缓冲 storage 的两个互斥 owner
};

struct DepInfo {
  IRStructure *prod_node;
  IRStructure *cons_node;
  ffi::Optional<Var> storage;
  std::vector<DependencyTaskPair> task_pairs;
  int distance;    // 0 同迭代 / -1 未解析的跨迭代 / >=1 显式距离
  DependencyKind kind{DependencyKind::kData};
};
```

这段注释里有一句对理解整章很关键：

> `-1` : unresolved cross-iteration dependency. A versioned storage may resolve
> it to its ring width; otherwise it means one iteration.

也就是说：**跨迭代依赖既可以用「等一轮」解决，也可以用「多一个版本」解决**。
谁来解决，取决于那个 storage 有没有被选中做多缓冲。

### 2.2 stage / order 的语义与校验

`(stage, order)` 是一对**逻辑坐标**：

* `stage` 说「这条语句属于管线的第几级」，数字越小越早发生；
* `order` 说「同一级内按什么顺序发射」。

编译器对用户给的注解有硬校验，
`ValidatePipelineBody`（`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:3097-3138`）
只允许两件事：

```cpp
// 1. 不能有两条语句拿到相同的 order
ICHECK(!used_orders.count(order))
    << "ValueError: Two statements in the software pipeline cannot have "
       "the same order";

// 2. 有 buffer 依赖的 (src → dst) 必须满足 stage(src) <= stage(dst)；
//    若两级相同，还必须满足 order(src) < order(dst)
ICHECK_LE(src_info.stage, dst_info.stage)
    << "... cannot depends on statement ... in a later stage ...";
if (src_info.stage == dst_info.stage) {
  ICHECK_LT(src_info.order, dst_info.order)
      << "... cannot be reordered";
}
```

注意第二条：**生产者不能晚于消费者**。这条规则排除了「把下一轮的计算提到这一轮的
搬运前面」这种写法——不是因为编译器做不到，而是因为那样写会让版本数失控。

还有一条很容易踩的**对齐**规则：`stage` / `order` 数组是按**源码顺序**对齐到
「可调度的语句」的，而「可调度的语句」**不包括可重放的标量 `Bind`**。
`PipelinePlanning` 用一个掩码把 `Bind` 标出来再过滤注解数组
（`_vendor/tilelang-v0.1.15/src/transform/pipeline_planning.cc:1046-1053`、
`kPipelineReplayableScalarBinds`，键名定义在
`src/transform/common/pipeline_utils.h:27-38`）。`Bind` 的语义第 11 章已经讲过，
这里只强调结论：**数组长度 == 有副作用的语句数**。

### 2.3 模调度的直觉（图 2）

**图 2 · 从 stage/order 到模调度** 见 `figures/02-modulo-schedule.md`。
模调度（modulo scheduling）的核心一句话：

> 把「第 `k` 次迭代的第 `s` 级」放到「第 `k + s` 个时间槽」执行。

`InjectSoftwarePipeline` 就是这么写的。它在 `EmitImpl` 里对每条语句做一次**偏斜**
（skew）：

```cpp
// _vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:2891-2899
int stage = pipeline_anno.stage;
PrimExpr inbound = Bool(true);
PrimExpr skewed_loop_var = new_loop_var - stage;   // ← 模调度的偏斜
if (need_bound_check)
  inbound = And(pipeline_loop_->min <= skewed_loop_var,
                (skewed_loop_var < pipeline_loop_->min + pipeline_loop_->extent));
```

`skewed_loop_var` 就是「这条语句在自己的时间槽里应该处理第几次逻辑迭代」。
`inbound` 是边界检查：在 prologue / epilogue 里，某些语句的那一轮还不存在或不
再存在，于是被门掉。

整条流水被 `BuildPipeline` 拆成三段（`inject_pipeline.cc:1237-1254`）：

```cpp
// prologue：只展开、带边界检查
Stmt prologue = EmitImpl(pipeline_loop_->min,
                         pipeline_loop_->min + max_stage_, true, true);
// body：滚动循环、不做边界检查（稳态里每一轮都是满的）
Stmt body     = EmitImpl(pipeline_loop_->min + max_stage_,
                         pipeline_loop_->min + pipeline_loop_->extent,
                         false, false);
// epilogue：同样展开并带边界检查
Stmt epilogue = EmitImpl(pipeline_loop_->min + pipeline_loop_->extent,
                         pipeline_loop_->min + pipeline_loop_->extent + max_stage_,
                         true, true);
```

三个参数的含义是 `(start, end, unroll_loop, need_bound_check)`：
**prologue 与 epilogue 逐轮展开并做边界检查，稳态保留成一个滚动循环**。
这就是「模调度 + 剥壳（peeling）」的标准形状，也解释了第 11 章为什么说填充段
与迭代次数无关。

### 2.4 版本数 = use − def + 1

多缓冲的第一问是「一个 buffer 要留几份」。答案是：

```text
num_versions = use_stage − def_stage + 1
```

其中 `def_stage` 是**最早写**它的 stage，`use_stage` 是**最晚读**它的 stage。
源码只有十几行（`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1488-1542`）：

```cpp
int ComputeBufferVersions(const Buffer &buffer,
                          const BufferAccessInfo &buffer_info) {
  if (buffer_info.def == -1) {
    // 不在流水体里被定义的 buffer 不改（比如外部分配的累加缓冲）
    return 1;
  }
  // `use - def + 1` is a upper bound of the needed versions
  int num_versions = buffer_info.use - buffer_info.def + 1;
  if (num_versions >= 2) {
    // 特例：use - def + 1 == 2 时，只有当确实存在
    // order(writer) < order(reader) 且 stage(writer) < stage(reader)
    // 且两个访问区域相交时，才真的需要双缓冲
    bool need_multi_version = false;
    ...
    if (!need_multi_version) num_versions--;
  }
  return num_versions;
}
```

`def` / `use` 从哪来？`GetBufferAccessInfo`
（`inject_pipeline.cc:1421-1457`）遍历流水体里每个 `SBlock` 的 `reads` / `writes`：

```cpp
for (const BufferRegion &write : block->writes) {
  auto &info = infos.at(write->buffer);
  if (info.def == -1) info.def = stage;
  else                info.def = std::min(info.def, stage);
}
for (const BufferRegion &read : block->reads) {
  auto &info = infos.at(read->buffer);
  info.use = std::max(info.use, stage);
}
```

三个可以直接用的结论：

1. **版本数看的是「读写跨越了几个 stage」，不是深度。** 一个 buffer 写于 stage 0、
   读于 stage 2，就是 3 份；另一个同样在 stage 0 写、但在 stage 0 读（同 stage 内
   先写后读），就只有 1 份。
2. **`def == -1` 的 buffer 不动。** 「在流水体外分配、流水体内不被写」的缓冲
   （例如累加器 fragment）不会被复制。
3. **`use − def + 1` 是上界，源码会收紧。** 当它算出来 `≥ 2` 时（源码的守卫写作
   `if (num_versions >= 2)`，注释则把这一支描述成 `== 2` 的特例——两者不完全一致，
   这里以代码为准），源码逐个检查「是否存在一个 writer 和一个 reader，满足
   `order(writer) < order(reader)`、`stage(writer) < stage(reader)` 且两个访问区域
   可能相交」——只有存在这样一对，多缓冲才是真的必要，否则把版本数减 1。

这是本章第一个「源码比直觉更聪明」的地方，也是常见错误的来源：很多 kernel 在
`num_stages=2` 时根本没用到第二份 buffer，就是因为这个收紧规则生效了。
（注释与守卫的这一点差异**不是笔误级别的细节**：它意味着「版本数一定等于
`use − def + 1`」这句话在源码层面是错的，只有把 `MayConflict` 与 order/stage 的
相对关系一起考虑才是完整判据。）

### 2.5 深度从哪来，以及「深度 ≠ stage 数」

同一份源码里，「多深」有两个来源，读注解的顺序是固定的
（`ComputeBufferVersions` 那条路径在 `BuildPipeline` 用到的
`GetPipelineNumStages`，`_vendor/tilelang-v0.1.15/src/transform/common/pipeline_utils.h:87-110`）：

```text
1. 循环注解 "num_stages"               ← 用户给的（T.Pipelined(num_stages=N)）
2. 循环注解 "tl_pipelined_num_stages"  ← PipelinePlanning 存下来的原始深度
3. max(software_pipeline_stage) + 1    ← 手写 stage 数组时由它反推
```

第 3 条解释了「只给 `stage`/`order` 时深度怎么来的」。第 2 条则是一个**有意的
设计**：`PipelinePlanning` 会**删掉**原来的 `num_stages` 注解，只留
`tl_pipelined_num_stages`，理由写在注释里
（`_vendor/tilelang-v0.1.15/src/transform/pipeline_planning.cc:1287-1293`）：

> We intentionally do NOT keep the legacy key "num_stages" here because multiple
> downstream passes (e.g. internal buffer versioning / warp specialization) treat
> it as an active pipeline marker and do not support nested pipelines.

更微妙的是，**「流水深度」和「SW 流水级数」在源码里是两个东西**，
`InjectSoftwarePipeline` 里有一段注释专门提醒
（`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:3626-3640`）：

```cpp
// Use the actual pipeline depth (number of buffer copies) for barrier sizing,
// not the SW pipeline stage count (max_stage + 1).
Optional<Integer> pipelined_num_stages = GetPipelineNumStages(op);
int pipeline_depth = pipelined_num_stages.defined()
                         ? static_cast<int>(pipelined_num_stages.value().IntValue())
                         : max_stage + 1;
```

两者在「自动规划」路径上通常相等（因为规划器故意把最后一级放在
`num_stages − 1`），但在**手写 stage** 时不相等：`stage = [0, 1]` 的深度是 2，
而 `num_stages = 4` 的深度是 4。混用这两者是多缓冲容量算错的常见原因。

---

## 3. 多缓冲的物化

### 3.1 版本维与下标改写（图 3）

**图 3 · 版本维的物化** 见 `figures/03-version-dim-materialization.md`。
从「逻辑上要留 N 份」到「IR 上真的有 N 份」需要两步。

**第一步：给 buffer 的 shape 最前面插一个版本维**
（`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1551-1560`）：

```cpp
Buffer RewriteAllocBuffer(const Buffer &buffer, int num_versions) {
  ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*(buffer.get()));
  new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_versions));
  if (!new_buffer->strides.empty()) {
    ICHECK(new_buffer->strides.size() + 1 == new_buffer->shape.size());
    PrimExpr stride_0 = new_buffer->strides[0] * new_buffer->shape[1];
    new_buffer->strides.insert(new_buffer->strides.begin(), stride_0);
  }
  return Buffer(new_buffer);
}
```

于是 `(block_M, block_K)` 变成 `(num_stages, block_M, block_K)`；如果原 buffer 已有
显式 strides，strides 也要同步前插一项。

**第二步：把每个访问的下标前置 `(k − min) mod nv`**
（`inject_pipeline.cc:1115-1145`）：

```cpp
Stmt VisitStmt_(const BufferStoreNode *op) final {
  ...
  PrimExpr version = floormod(
      (pipeline_loop_->loop_var - pipeline_loop_->min), new_buffer->shape[0]);
  n->indices.insert(n->indices.begin(), version);
  ...
}
PrimExpr VisitExpr_(const BufferLoadNode *op) final {
  ... // 与 Store 完全对称
}
```

对**指针式**访问（`tvm_access_ptr`）还要单独换算字节偏移
（`RewriteBufferAccess`，`inject_pipeline.cc:1059-1093`）：新下标 =
`旧下标 + version × 单版本元素数`。写成公式：

```text
访问下标 = 旧下标 + ((k − min) mod nv) × product(buffer.shape)
```

**第三步（容易漏）：放宽访问区域。** 版本维一加，依赖分析看到的 region 就多了一维。
在稳态循环里，一个块的读/写可能落在**任何一个版本**上；而在 prologue / epilogue 里
实际只有一份版本活着。`RewritePipelineBufferRegion`
（`inject_pipeline.cc:1038-1056`）用一个布尔参数区分这两种情形：

```cpp
Range accessed_version =
    access_all_versions_
        ? Range::FromMinExtent(0, new_buffer->shape[0])          // 整个版本维
        : Range::FromMinExtent(
              floormod((pipeline_loop_->loop_var - pipeline_loop_->min),
                       new_buffer->shape[0]),
              Integer(1));                                        // 只碰一个槽
new_region.insert(new_region.begin(), accessed_version);
```

`access_all_versions_` 的取值就是 `max_stage_ != 1`（见 `:2903-2905` 的构造），
也就是说：**只有在真正做了多级流水时，访问区域才放宽到整个版本维**。

### 3.2 barrier 也必须多版本

如果只给数据 buffer 加版本、忘了给 barrier 加，程序会在「第 `k` 轮等第 `k+1` 轮」
上死等或者提前通过。`ExpandPipelineBarriers`
（`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:785-886`）专门做这件事：

```cpp
PrimExpr ns = IntImm(DataType::Int(32), num_stages);
PrimExpr stage_expr  = FloorMod(loop_var - loop_min, ns);
PrimExpr parity_cycle = FloorMod(FloorDiv(loop_var - loop_min, ns), 2);
...
new_node->shape = {PrimExpr(num_stages) * buf->shape[0]};   // [N] → [num_stages × N]
```

注意 `parity_cycle`：**mbarrier 的等待需要一个 parity（奇偶轮次）**，它等于
「这是第几圈绕过的」——`((k − min) / nv) mod 2`。同一个槽位被第 0 圈和第 1 圈
各用一次，parity 负责把两圈区分开。少了这一项，wait 会与 arrive 配错。

哪些 barrier 会被展开？源码的判据很具体（`inject_pipeline.cc:792-825`）：
**该 barrier 在本循环体里有显式的 `ptx_arrive_barrier`，或者是 ISP 自己刚创建的
`pipeline_mbar`**。由 tile-op 内部管理 arrive 的 barrier（注释里点名 tcgen05 MMA）
**不**展开。

### 3.3 分配与合并：一次线扫描 arena 打包

版本维加完之后，每个 buffer var 的字节数是 `nv × 单版本字节`。接下来
`MergeSharedMemoryAllocations`
（`_vendor/tilelang-v0.1.15/src/transform/merge_shared_memory_allocations.cc:1576-1668`）
把**所有** shared 分配合并成一次分配——因为「每个 kernel 只允许有一个 dynamic 或
static shared 分配」（文件头注释）。它的算法是三段：

1. **线性访问模式扫描**：把语句体展开成一个线性序列，记下每个 buffer var 第一次和
   最后一次被触碰的位置，得到活跃区间 `Interval{start, end, size, align}`
   （`SharedMemLinearAccessPatternFinder`，`:111-370`）。
2. **区间按程序序排序**，用优先队列维护「还活着的区间」，谁结束了就归还到
   freelist。排序键是 `(start, -size, name)`，即**先来先分配、同起点大的先放**。
3. **best-fit 分配**：在 freelist 里挑一个「对齐后浪费最少」的空洞；都没有就在
   arena 顶部对齐后开新空间。freelist 每次归还都与邻居合并（coalesce），
   保持「有序且无相邻碎片」的不变量。

```cpp
// _vendor/tilelang-v0.1.15/src/transform/merge_shared_memory_allocations.cc:862-880
std::optional<size_t> Allocate(size_t need, size_t alignment) {
  // Best-fit search: pick the slot that wastes the least space after
  // alignment.
  ...
}
```

这个算法有两个直接后果：

* **它做的是「buffer var 之间」的复用，不是「buffer var 内部」的复用。**
  一个已经乘了 `nv` 的多版本 buffer 是一个不可分割的块，它的字节数不会因为
  「其中只有一份在用」而缩小。
* **对齐要求会吃掉碎片。** `align_bytes` 默认 16，而 TMA/MMA swizzle 会通过
  `kSmemAlignmentMap` 追加更严格的对齐（源码里明确写着：**对齐不足不是启动错误，
  而是静默的数据损坏**，`:400-410`）。

### 3.4 容量约束与冲突：哪些 buffer 能复用

把 3.3 的算法反过来读，就得到「哪些 buffer 能共用同一段 shared memory」的判据：

```text
可以复用 ⟺ 两者的活跃区间 [start, end) 不相交
         ∧ 对齐要求能被满足
         ∧ 都愿意参与合并（都在同一次 MergeSharedMemoryAllocations 的输入里）
```

以及一条**反向**的判据——「多缓冲为什么会吃掉复用机会」：

```text
一个 buffer 的版本数每 +1，它的活跃区间就向前延伸一段、字节数也乘一倍；
前者让它更容易与别的 buffer 撞上，后者让撞上之后的代价更大。
```

`code/02-buffer-version-model/` 就是这个判据的可复算版本。它的运行结果（本机
Python 3.11 实际跑出来的模型输出，不是硬件测量）在一个 128×128×64 的 fp16
GEMM 上给出：

```text
scenario: epilogue（多一个只在 epilogue 用一次的 32 KiB scratch）
num_stages=2  versions: A_shared=2, B_shared=2, C_shared=1
    no-reuse sum    96.0 KiB | arena    64.0 KiB | budget   228.0 KiB  [OK  ]
      A_shared  offset 0.0 KiB | C_shared offset 0.0 KiB | B_shared offset 32.0 KiB

scenario: carry（多一个每次迭代都活着的 16 KiB buffer）
num_stages=4  versions: A_shared=4, B_shared=4, S_shared=4
    no-reuse sum   192.0 KiB | arena   192.0 KiB   ← 完全不能复用
num_stages=5  versions: A_shared=5, B_shared=5, S_shared=5
    no-reuse sum   240.0 KiB | arena   240.0 KiB | budget   228.0 KiB  [OVER]
```

两行对照就是本章第一个「工程结论」：

> **epilogue 型（短命）的 buffer 会被 arena 自动塞进别人腾出来的洞里；
> 循环携带型（长命）的 buffer 会把整段 arena 撑开，并且在 `num_stages` 变大时
> 第一个顶穿预算。**

这解释了为什么很多 kernel 在 `num_stages` 从 3 加到 4 时突然编译失败，而
`block_K` 同样加一倍却没事——失败的不是「总量」，而是「总量里能被复用的那部分」。

### 3.5 昇腾侧的物化：owner、epoch 与两种时钟

昇腾侧的多缓冲不是「按 stage 跨度算版本数」，而是把这件事建模成一个**所有权**
（ownership）问题，再由调度器决定版本数。三个概念（源码里给的定义在
`_vendor/tilelang-v0.1.15/src/ascend/transform/auto_schedule/multi_buffer.h:159-200`）：

* **owner**：一个循环，它的一次迭代包含某个 storage 的**一个完整 epoch**
  （写 + 消费这次写的所有读）。一个 storage 可以有多个互不嵌套的 owner。
* **版本数**：由调度器在「任务生命周期 + 显存容量」下选出来，`num_stages` 只是
  **搜索上界**（`src/ascend/transform/auto_schedule.cc:705-736` 读取注解，缺省为 1，
  没有注解时用 `max_manual_stage + 1`）。
* **索引模式**：`iteration`（按词法迭代号取槽）或 `counter`（按**实际执行的**
  epoch 计数取槽）。两者在「有 guard 的循环」上会分叉：`iteration` 会把被跳过的
  迭代也算进去，`counter` 只数真正执行的 epoch
  （上游文档 `tilelang-ascend/references/synchronization.md` 的
  「Conditional execution and epochs」一节把这张表列全了）。

物化由两个 pass 完成（`_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:111-114`）：

```python
mod = ascend_transform.PrepareMultiBuffer()(mod)     # 选版本数、建逻辑时钟
mod = ascend_transform.ResolveCore()(mod)
mod = ascend_transform.InsertSync()(mod)             # 插同步（§4.4）
mod = ascend_transform.MaterializeMultiBuffer()(mod) # 把逻辑时钟落成物理版本
```

顺序值得注意：**同步插入在物理版本物化之前**。`InsertSync` 处理的是
`(storage, epoch, num_versions)` 这样的逻辑对象，`MaterializeMultiBuffer` 才把它
变成真实的 `buffer[slot]`。这使得「同步协议」与「内存布局」可以分开推理——
这是昇腾侧与 CUDA 侧在工程上最大的差异之一。

---

## 4. 同步插入

### 4.1 从读写集到 hazard

同步的全部依据是**访问冲突**。给定两个访问（可能属于不同语句、也可能属于同一条
语句的不同迭代），需要同步当且仅当：

```text
1. 访问同一个 buffer（或已知别名的两段存储）
2. 访问范围可能相交
3. 至少一个是写
4. 两次访问的执行者（线程 / 核 / PIPE）不完全相同
```

其中第 4 条是最容易被忽略的一条：**如果两次访问由同一组线程、以相同顺序执行，
就不需要 barrier**。CUDA 侧的 `FindConflict` 把这条写成 Z3 查询；
昇腾侧它体现为「site 的 core/PIPE 相同且存在隐式顺序时不生成 flag」。

### 4.2 CUDA 侧：`tl.ThreadSync` 的算法（图 4）

**图 4 · 从读写集到同步点** 见 `figures/04-hazard-to-sync.md`。
`ThreadSync` 本身就是三个 pass 串起来
（`_vendor/tilelang-v0.1.15/src/transform/thread_storage_sync.cc:2058-2100`）：

```cpp
PrimFunc TileLangThreadSync(PrimFunc func, const std::string &storage_scope) {
  ...
  if (sync_scope.rank == StorageRank::kShared && sync_scope.tag.empty())
    stmt = ThreadSyncAfterWaitGroupInserter(sync_scope)(stmt);  // ① 特殊照顾 cp.async
  TileLangThreadSyncPlanner planner(sync_scope, warp_size);     // ② 找冲突点
  planner(stmt);
  stmt = ThreadSyncInserter(sync_scope, planner.syncs_inserted_)(std::move(stmt));
  n->body = ThreadPartialSyncRewriter::Rewrite(std::move(stmt), warp_size); // ③ 收窄
  return func;
}
```

**第一步：收集访问条目。** `TileLangThreadSyncPlanner`
（`:603-700`）逐语句收集 `AccessEntry`：

```cpp
struct AccessEntry {
  Array<IterVar> threads;          // 谁在访问
  Array<PrimExpr> buffer_indices;  // 访问哪个下标
  ConstrSet cset;                  // 当前所在的约束（if / loop 条件）
  Array<arith::IntSet> touched;    // 触碰的区间
  Var buffer; Buffer buffer_name; DataType dtype;
  AccessType type;                 // kRead / kWrite / kSync / kAlloc / kReadAcquire
  StorageScope scope;
  bool is_pointer_access = false;
  bool is_async_copy = false;      // TMA / cp.async 上下文
  bool is_atomic = false;          // atomic RMW
};
```

这里每个字段都有用：`threads` + `cset` 是「谁」，`touched` 是「哪里」，
`is_async_copy` / `is_atomic` 是两个**豁免标签**。

**第二步：两两查冲突。** `FindConflict`（`:1759-2043`）依次排除：

```cpp
// 两个异步写之间不加 barrier（例如两条 TMA load 写不同槽）
if (prev.type == kWrite && curr.type == kWrite &&
    prev.is_async_copy && curr.is_async_copy) return false;
// 不同 buffer 无关
if (!prev.buffer.same_as(curr.buffer)) return false;
// 两个 atomic 之间不加 barrier（加了会改变可观察语义）
if (prev.is_atomic && curr.is_atomic) return false;
```

然后进入真正的判据。如果下标**逐项恒等**，就用 Z3 检查「两边由同一批线程执行」
（`prev ⇒ curr` 且 `curr ⇒ prev`），是则判定无冲突。如果下标不等，则把两边换算成
**字节区间**，用 `CanProve(prev_max < curr_min || curr_max < prev_min)` 证明不相交；
证不出来就保守认定冲突。循环携带依赖的处理方式是**符号移位**：把下一轮的访问
里的循环变量替换成 `loop_var + step`，再和本轮比：

```cpp
// For loop-carry analysis, we compare: Iter(i) vs Iter(i+step)
loop_shift_sub.Set(loop->loop_var, loop->loop_var + step);
```

这条设计注释还顺手回答了一个常见疑问：

> This approach can prove that accesses like `A[i]` and `A[i+1]` are disjoint
> (no loop-carry dependency), while correctly detecting dependencies like
> `A[i]` and `A[i-1]` (loop-carry dependency with distance 1).

**第三步：把冲突变成 barrier，再收窄。** `ThreadSyncInserter`（`:212-260`）在冲突
两侧插入 `tvm_storage_sync(scope)`——codegen 之后就是 `__syncthreads()`。
`ThreadPartialSyncRewriter`（`:263-430`）最后把「全块 barrier」收窄成只包含真正
参与的 warp 集合（并给每个 barrier 分配不同的 barrier id 与参与线程数）。

### 4.3 异步拷贝与 TMA：为什么还要 wait_group 和 parity

`__syncthreads()` **不会**等异步拷贝完成——这是源码里专门写了一段注释强调的事实
（`_vendor/tilelang-v0.1.15/src/transform/thread_storage_sync.cc:60-75`）：

> `tvm_storage_sync` barriers inserted by `ThreadSyncPlanner` are based on memory
> conflicts and may end up *before* the `wait_group`, which is incorrect for
> cp.async because `__syncthreads()` does not wait for outstanding asynchronous
> copies.
> Correct usage requires: `ptx_wait_group(N); tvm_storage_sync("shared");`

于是有了 `ThreadSyncAfterWaitGroupInserter`（`:76-120`）：**在每个
`ptx_wait_group` 之后强制补一个 shared 同步**（若还没有）。这一条对 NV 与
昇腾都不适用（它是 CUDA 专属，昇腾路径跳过，见 §4.5）。

被流水搬运的异步写还有第二个坑：上一小节说的 `access_all_versions` 只放宽了
**数据** buffer 的区域；barrier 自己的槽位必须显式展开，且 wait 必须带 parity。
两者的分工是：

| 机制 | 解决什么 | 源码 |
| --- | --- | --- |
| `ptx_wait_group(N)` | 我这一轮的异步拷贝到齐了吗 | `inject_pipeline.cc` 里 `AsyncPipelineLoopWaitRelaxer` / `RelaxTrailingConsumerWaits`（`:2493-2540`） |
| `__syncthreads()` | 别人写的 shared 我能不能看见了 | `thread_storage_sync.cc:76-120` |
| `mbarrier` + parity | 第 `k` 圈的 arrive 与第 `k` 圈的 wait 配对 | `ExpandPipelineBarriers`（`inject_pipeline.cc:785-886`） |

还有一条与 warp specialization 有关的分支：**只有在没有做 WS 的情况下**，
`InjectSoftwarePipeline` 才会自己把 TMA copy 改写成 `tl.tileop.tma_copy` 并插
`mbarrier_wait_parity`（`inject_pipeline.cc:3616-3640`）。做了 WS 的 kernel，
这套 barrier 归 WS 那一趟管（见 §8.3）。

### 4.4 昇腾侧：`InsertSync` 的算法

昇腾侧的同步不是「插 barrier」，而是「建一条**有向的完成协议**」。数据模型三件套
（`_vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc`）：

```cpp
struct SyncInsertionSite {   // :82   ——「在哪里插」
  IRStructure *path_begin; IRStructure *path_end; bool at_beginning;
  CoreMask core;             // Cube / Vector / MTE...
  std::string pipe;          // PIPE_M / PIPE_V / PIPE_MTE1 ...
  int domain_id;             // 属于哪个 epoch 域
  PrimExpr guard{Bool(true)};// 条件执行时的谓词
};

struct DepEdge { size_t src; size_t dst; int distance; bool strict; };  // :212

struct SyncPoint {           // :245
  DepEdge edge;
  DependencyKind dependency_kind;
  int num_versions;          // ← 多缓冲直接进入同步协议
  PrimExpr flag_iteration;
  ControlNode *flag_loop;
  ...
};
```

`CollectSyncPoints`（`:1226-1400`）把一条依赖边变成一个同步点：

```cpp
// _vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc:1330-1345
if (within_owner) {
  num_versions = info->num_versions;   // 在 owner 内：用该 ring 的版本数
  flag_loop = enclosing_owner->loop;
  ...
  if (loop == enclosing_owner->loop && distance < 0)
    distance = info->num_versions;     // 未解析的跨迭代依赖 → 跨整个 ring
} else if (info == nullptr) {
  if (manual_versions)
    num_versions = dep.distance > 0 ? dep.distance : manual_versions;
}
if (distance < 0 && dep.kind == DependencyKind::kData) distance = 1;
```

注意最后两行：**手工多缓冲时，同步的版本数来自用户声明的距离**；自动 ring 时，
它来自调度器选出的 `num_versions`。这就是「多缓冲的完成协议」在代码里的样子。

方向由 PIPE 决定，`GetHardEvent`（`:751-772`）把两端的 PIPE 名拼成一个事件名，
并对四种组合做了规范：

```cpp
std::string GetHardEvent(const std::string &producer_pipe,
                         const std::string &consumer_pipe) {
  std::string event = producer_pipe + "_" + consumer_pipe;
  if (event == "MTE2_M")  return "MTE2_MTE1";
  if (event == "M_MTE2")  return "MTE1_MTE2";
  if (event == "MTE3_M")  return "MTE3_MTE1";
  if (event == "M_MTE3")  return "MTE1_MTE3";
  return event;
}
```

这里的 `M` 指 Cube（矩阵）单元，`MTE1/2/3` 是三条搬运通路。也就是说
**「同步」在昇腾上是「PIPE 之间的有向事件」，而不仅仅是一个位置**。

最后 `SyncMaterializer`（`:2344-2480`）按三条规则落地：

```cpp
// 1. 非跨核、且两侧同一个 PIPE
if (!site_registry_.IsCrossCore(sync_point) && producer_pipe == consumer_pipe) {
  if (producer_pipe == "PIPE_S" || producer_pipe == "PIPE_V" ||
      producer_pipe == "PIPE_M") continue;          // S/V/M 有隐式顺序，不需要
  InsertStatement(dst.path_end, MakeAscendPipeBarrier(producer_pipe), true, ...);
  continue;
}
// 2. 跨核
if (site_registry_.IsCrossCore(sync_point)) { ... CrossCoreSetFlag / WaitFlag ... }
// 3. 核内跨 PIPE
InsertStatement(src.path_end, MakeAscendSetFlag(producer_pipe + "_" + consumer_pipe,
                                                event_id), false, ...);
InsertStatement(dst.path_end, MakeAscendWaitFlag(...), true, ...);
```

`event_id` 的算法是同步与多缓冲耦合得最紧的一处
（`MakeFlagEventId`，`insert_sync.cc:2299-2307`）：

```cpp
PrimExpr MakeFlagEventId(const PrimExpr &iteration, int num_versions, int base_id) {
  PrimExpr event_id = num_versions == 1
                          ? IntImm(dtype, base_id)
                          : indexmod(iteration, IntImm(dtype, num_versions))
                                + IntImm(dtype, base_id);
  ...
}
```

即 `event_id = iteration % num_versions + base_id`。**一个版本一个事件槽**，
所以版本数直接乘在 flag 需求上。

### 4.5 `AscendThreadSync` 与 `RewriteFlagToBuf`：把协议落成有限资源

名字容易让人误会的是 `AscendThreadSync`
（`_vendor/tilelang-v0.1.15/src/ascend/transform/thread_storage_sync.cc:2054-2124`）。
它在昇腾上**只处理 `SIMT_VF` 块**：

```cpp
if (is_ascend) {
  // Ascend: process each SimtVF block independently.
  // ThreadSyncAfterWaitQueueInserter and ThreadPartialSyncRewriter are
  // CUDA-specific and skipped for Ascend.
  struct SimtVFThreadSyncMutator : StmtExprMutator {
    Stmt VisitStmt_(const SBlockNode *op) final {
      if (op->name_hint == "SIMT_VF") {
        TileLangThreadSyncPlanner planner(scope, warp_size);
        ...
        body = ThreadSyncInserter(scope, planner.syncs_inserted_)(body);
      }
      ...
    }
  };
}
```

也就是说：**昇腾上「PIPE 之间的同步」由 `InsertSync` 负责，
`AscendThreadSync` 只负责 `T.SimtVF` 里那点线程级同步**。它出现在 pipeline 的
两处（`tilelang/ascend/pipeline.py:163` 的 `"global"`、`:181-182` 的
`"shared"` / `"shared.dyn"`），而 `InsertSync` 出现在调度链里（`:113`）。

最后一关是资源：每个 hard event 只有 **8 个 event id 槽**
（`kIntraCoreFlagLimit = 8`，`insert_sync.cc:2135`；跨核 16，`:2136`），
而每个同步点要占 `num_versions` 个**连续**槽。`FlagAllocator::Plan`
先做加法（`insert_sync.cc:1856-1868`）：

```cpp
std::string domain = FlagDomain(sync_point);
required_ids[domain] += sync_point.num_versions;
```

超了怎么办？`RewriteFlagToBuf`
（`_vendor/tilelang-v0.1.15/src/ascend/transform/rewrite_flag_to_buf.cc`）用
0/1 背包（容量 8）挑一组「填得最满」的保留为 `set_flag/wait_flag`，其余的
**溢出**到共享的 32 槽互斥池：

```text
set_flag<PROD_CONS>(id)  →  asc_lock(PIPE_PROD, buf, ASC_LOCK_NON_BLOCK);
                            asc_unlock(PIPE_PROD, buf, ASC_LOCK_NON_BLOCK);
wait_flag<PROD_CONS>(id) →  asc_lock(PIPE_CONS, buf, ASC_LOCK_BLOCK);
                            asc_unlock(PIPE_CONS, buf, ASC_LOCK_BLOCK);
```

溢出不是消失，而是**从事件标志变成阻塞锁**。`code/04-ascend-flag-budget/`
把这段算术跑了一遍（本机 Python 3.11 模型输出）：

```text
depth=2  blocks: acquire_A=2, acquire_B=2, release_A=2, release_B=2
  allocator sum = 8 ids → fits in 8 slots; no spill  (0 slots spare)
depth=3  blocks: 4 × 3 = 12 ids
  -> exceeds 8 slots: 0/1 knapsack keeps 6/8 slots as set/wait flags
     kept    : acquire_A, acquire_B
     spilled : release_A, release_B  -> asc_lock/asc_unlock mutex pool (blocking)
```

四个同步点、深度 3 就已经溢出。真实 kernel 的同步点远不止四个——这就是
「多缓冲会把瓶颈从 shared memory 转移到 flag 槽位」的具体含义（§5.2 的
第二条路径）。

### 4.6 两侧对照

| 维度 | CUDA（`src/transform/thread_storage_sync.cc`） | 昇腾（`src/ascend/transform/insert_sync.cc` + `rewrite_flag_to_buf.cc`） |
| --- | --- | --- |
| 冲突表示 | `AccessEntry` + Z3 判定字节区间是否相交 | `DepInfo{prod, cons, storage, distance, kind}`（`dependency_analysis.h:44-97`） |
| 同步节点 | 冲突点之间插 `tvm_storage_sync` → `__syncthreads()` | `SyncPoint{edge, kind, num_versions}` → `SetFlag/WaitFlag` 或 `PipeBarrier` |
| 方向性 | 无：barrier 只说「等到齐」 | 有：`HardEvent = PROD_CONS` 天生区分方向 |
| 与多缓冲耦合点 | barrier 槽位展开 + parity（`ExpandPipelineBarriers`） | `event_id = iteration % num_versions + base` |
| 资源上限 | 异步组数 / barrier 数量 | 每个 hard event 8 槽（跨核 16），超出溢出到互斥池 |
| 还有谁在插同步 | `ThreadSyncAfterWaitGroupInserter`（cp.async 专属） | `AscendThreadSync`（只处理 `SimtVF` 块） |

---

## 5. 正确性：漏一个同步 vs 多一个同步

### 5.1 漏一个同步为什么是「偶发」错误

同步要防的是**两个硬件单元之间的竞态**。竞态的错误表现取决于相对时序，因此它在
三类条件下最容易表现为「偶发」：

1. **两次访问的延迟接近。** 例如搬运与计算耗时相当时，读与覆盖几乎同时到达同一
   个地址。此时少一个 barrier 只在某些 tile 形状、某些 K 上出错。
2. **有多个版本的槽位在轮转。** 深度 `d` 越大，同一个物理槽被复用的间隔越远，
   但**只要每个槽的 WAR 边没有被保护，任何一次「读者还没读完、写者已经写」都会
   破坏数据**。深度大只是降低了发生概率，没有消除它。
3. **尾部/条件路径。** prologue 与 epilogue 的轮次不完整，guard 让部分语句不执行。
   上游文档把这类情形单独列成一节（conditional execution and epochs）：被跳过的
   迭代会让 `iteration` 模式的槽位号与 `counter` 模式的 epoch 号分叉，从而让
   生产者与消费者算出**不同的槽**。

为什么「偶发」比「稳定错」更难处理：它会让同一份代码在 CI 上过、在线上挂。
所以本章给的第一条工程建议是：

> **不要把「跑出来的结果对了」当作同步正确的证据。** 要验证的是
> 「所有 hazard 都被覆盖」，方法是审阅 §4 的推导（读写集 → 冲突 → 同步点），
> 而不是重复运行。

### 5.2 多一个同步为什么掉性能

一个 barrier 的代价有两层：

* **直接的等待代价**：它本身要花时间（barrier 指令 + 等所有参与者到齐）；
* **间接的结构代价**：它把原本可以重叠的两条通路**合流**了。

第二层才是主要的。第 11 章的时空图已经说明：流水的收益来自「两条车道同时在忙」。
一个放在生产者与消费者之间的多余 barrier，会把「车道并行」变回「车道串行」。
`code/03-sync-count-vs-perf/` 用第 11 章同款贪心模型量化了这件事。下面是模型输出
（`issue=1, latency=30, compute=20, barrier=8, iterations=16`，抽象周期数，
**不是硬件测量**）：

```text
depth |    none |     war | war+raw |      +cb | sync-copy
    3 | 350cyc  | 350cyc  | 388cyc  | 478cyc  | 628cyc
    6 | 350cyc  | 350cyc  | 358cyc  | 478cyc  | 628cyc

稳态每轮步长（循环后一半的平均）：
  depth=3: none=20.0  war=20.0  war+raw=22.6  +cb=28.0  sync-copy=38.0
  depth=6: none=20.0  war=20.0  war+raw=20.0  +cb=28.0  sync-copy=38.0
```

读法：

* `none` 是**不正确**的上界（没有任何 WAR 保护），它给出理论下限 20 周期/轮；
* `war`（最低限度的释放屏障）在 `depth ≥ 3` 时与 `none` **完全一样**：
  8 个周期的屏障被「已经在路上的那一份拷贝」盖住了。**必需的同步可以是免费的。**
* `+cb`（在消费者链里多插一道屏障）把稳态从 20 抬到 28，正好是一个 `barrier`；
  加到 `depth = 6` 也救不回来。**多余的同步不会免费**，因为它落在了关键路径上。
* `sync-copy`（退化成同步拷贝、每语句一道 barrier）稳态 38；把 `latency` 从 30
  调到 120 时它变成 128——这就是「把异步流水写成了串行搬运」的代价。

### 5.3 同步点数量 vs 性能：定性结论

把上面两节合成一张定性图（这张图是**推理**结论，不是实测）：

```text

  吞吐
   ▲
   │                        ╭──────────────  正确性上界（无同步）
   │                    ╭───╯
   │                ╭───╯  ← 必需的同步：只要 depth 够，成本可被隐藏
   │        ╭───────╯
   │    ╭───╯
   │ ╭──╯                    ╲
   │─╯                        ╲
   │                           ╰──────────  多余的同步：每多一道 ≈ 一个 barrier 的代价
   │
   └──────────────────────────────────────────────► 每轮同步点数
      0        1(必需)      2          3         4
                 ↑                          ↑
           "漏一个同步会偶发出错"      "多一个同步会掉性能"
           在这条线的左边              在这条线的右边

  附加约束（只在昇腾侧显现）：同步点 × 版本数 超过每 hard event 8 个 flag 槽时，
  一部分同步点溢出到阻塞互斥池，吞吐再掉一级。
```

三条可以带走的判据：

1. **同步点数应当等于「依赖边数」，不多不少。** 少了是错误，多了是浪费。
   编译器两个方向的目标都是逼近这个数：CUDA 侧靠 Z3 证「不相交从而不需要」，
   昇腾侧靠 `reusable_pairs_` 证「不可能同时在飞从而可以复用同一个 flag」。
2. **必需同步的成本与深度成反比，多余同步的成本与深度无关。**
   所以「加深度」只能治第一类问题。
3. **同步有第二种预算。** 在昇腾上它是 flag 槽位（8/16），在 CUDA 上它是
   异步拷贝的在飞组数。当主预算（shared memory）还没到顶而性能已经开始掉时，
   先去看第二个预算。

---

## 6. 与硬件特性的关系

同步模型不是纯软件的抽象，它直接对应硬件提供的原语。本节只讲**这些原语在同步模型
里的地位**，机制细节留给第 24 章（NV）与第 25 章（昇腾）。

### 6.1 NV 侧：`cp.async` / `mbarrier` / TMA 各自解决什么

| 硬件特性 | 它提供的语义 | 在同步模型里的位置 |
| --- | --- | --- |
| `cp.async` + `commit_group` / `wait_group` | 「我发的这批拷贝到齐了几个组」 | **生产者内部的完成计数**。它不跨越线程，所以 `__syncthreads()` 不能替代它，反过来也不能（§4.3） |
| `__syncthreads()`（`tvm_storage_sync`） | 「本块所有线程都到这里了」 | **消费者侧的可见性**。由 `ThreadSync` 按访问冲突插入 |
| `bar.sync` 带 barrier id / 线程数（`ThreadPartialSyncRewriter`） | 「只有参与的 warp 到齐」 | 把全块 barrier 收窄，减少无关 warp 的等待 |
| `mbarrier` + parity | 「第 `k` 圈的 arrive 与第 `k` 圈的 wait 配对」 | **环形的完成协议**：`ExpandPipelineBarriers` 把槽位从 `[N]` 扩到 `[nv × N]`，wait 带 `((k−min)/nv) mod 2` 的 parity |
| TMA（`tl.tileop.tma_copy` + `ptx_arrive_barrier`） | 一次搬运可以只由一个线程发起、由 barrier 通知全部消费者 | 在流水里由 ISP 统一管理（**没有 WS 时**，`inject_pipeline.cc:3616-3640`），或在 WS 那一趟里由 WS 管理 |

一句话：**NV 的同步模型是「组内计数 + 块级屏障 + 环形 parity」三层叠加**。
缺任何一层都会留下一个具体的坑（漏 `wait_group` → 读到半截数据；漏
`__syncthreads` → 别的 warp 看不见；漏 parity → 等错轮次）。

这三层里最特别的是 mbarrier。厂商资料把它称为**异步事务屏障**
（asynchronous transaction barrier），它与普通 barrier 的区别是「不只数到达的线程，
还数事务」：

> It too is a split barrier, but instead of counting just thread arrivals, it also
> counts transactions. …blocks threads at the `Wait` command until all the
> producer threads have performed an `Arrive`, and the sum of all the
> transaction counts reaches an expected value.
> —— `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md`
> 第 314、316 行（NVIDIA Hopper Architecture In-Depth，厂商公开资料；条件：H100）

这解释了两件事：**为什么 TMA 只需要一个线程发起**（其余线程在 barrier 上等事务计数），
以及**为什么流水必须区分 parity**（同一个事务屏障被反复使用，必须知道当前算第几圈）。

### 6.2 昇腾侧：Set/Wait 与 MTE / Cube / Vector

昇腾的执行单元被组织成若干净水线（PIPE）：`PIPE_S`（标量）、`PIPE_V`（向量）、
`PIPE_M`（Cube/矩阵）、`PIPE_MTE1/2/3`（三条搬运通路）等。同一 PIPE 内的指令**天然
按序执行**，跨 PIPE 就要显式同步——这正是 `HasImplicitSamePipeOrder` 的含义：

```cpp
// _vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc:766-772
// These are the same-pipe dependencies the materializer satisfies without a
// barrier. MTE/FIX issue order alone does not protect overlapping accesses.
bool HasImplicitSamePipeOrder(const std::string &pipe) {
  return pipe == "S" || pipe == "V" || pipe == "M";
}
```

于是同步模型变成一张**PIPE 之间的有向图**：

| 原语 | 语义 | 什么时候用 |
| --- | --- | --- |
| `PipeBarrier<PIPE_X>` | 同一个 PIPE 内部的一道栅栏 | 同 PIPE 且该 PIPE 没有隐式顺序（MTE / FIX）时 |
| `SetFlag<HardEvent>(id)` / `WaitFlag<HardEvent>(id)` | 「PIPE_A 完成了，通知 PIPE_B」 | 跨 PIPE 的依赖；`HardEvent = A_B`，id 由 `iteration % nv + base` 给出 |
| `CrossCoreSetFlag` / `CrossCoreWaitFlag` | Cube 核与 Vector 核之间的事件 | 核间流水（inter-core pipeline） |
| `asc_lock` / `asc_unlock` | 8 槽用完后的溢出通道 | `RewriteFlagToBuf` 背包溢出的同步点 |
| `AscendC::SetFlag` 的**版本化** | 同一个事件在 ring 上有 `nv` 个槽 | 多缓冲：`num_versions` 直接进 `SyncPoint` |

昇腾侧还有一条 NV 侧没有的语义：**核间流水**。同一个 `T.Pipelined` 里可以让
Cube 写 workspace、Vector 读 workspace，两侧的完成协议由 cross-core flag 承担
（`_vendor/tilelang-ascend-v0.1.2.000/docs/tutorials/t_pipelined.md` 的
「Inter-core Pipeline」一节给了逐拍时间表）。该仓库的 `T.Pipelined` 还多一个
`cross_interval` 参数控制跨核同步的频率（默认 1，即每轮同步一次）。

**待核**：`cross_interval` 只出现在 `_vendor/tilelang-ascend-v0.1.2.000` 的
`tilelang/language/pipeline.py:19-52`；我没有在 `_vendor/tilelang-v0.1.15` 的
`tilelang/language/loop.py` 里找到同名参数（该文件的 `Pipelined` 只有
`start/stop/num_stages/order/stage/annotations`）。因此本书把 `cross_interval`
标注为**昇腾适配器仓库的能力**，不主张它在 v0.1.15 主干里存在。

---

## 7. 实战：改 num_stages，看 IR 怎么变

本节给出一个**可复现的实验方案**。它的关键设计是：**IR 部分不需要 GPU**。
`tilelang.tools.compile_only.compile_kernel_source` 内部调用
`tilelang.lower(..., enable_device_compile=False)`
（`_vendor/tilelang-v0.1.15/tilelang/tools/compile_only.py:169-200`），
只跑 lowering + codegen，不编译 nvcc、不启动设备。完整脚本在
`code/01-num-stages-ir-diff/lower_num_stages.py`。

### 7.1 实验设计

被观察的 kernel 就是 §2.3 的那个形态（两条 global→shared 拷贝 + 一次 `T.gemm`），
只把 `T.Pipelined(..., num_stages=N)` 的 N 换成 `1, 2, 3, 4`。每次观察四组事实：

```text
A. 循环注解      software_pipeline_stage / order / tl_pipelined_num_stages
B. 版本维        生成 CUDA 里 __shared__ 数组的首维
C. 同步点        __syncthreads() / cp.async.wait_group / mbarrier 的出现次数与参数
D. pass 逐段 diff TL_LOWER_TRACE=terminal 的 PipelinePlanning / InjectSoftwarePipeline / ThreadSync
```

其中 A 由脚本直接跑 `tilelang.transform.PipelinePlanning()` 得到（不依赖 codegen），
D 由官方调试工具给出（`tilelang/tools/lower_trace/__init__.py:1-30`；
文档在 corpus：`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tutorials/debug_tools_for_tilelang.md`）。

### 7.2 有卡机器上的步骤

```bash
pip install tilelang==0.1.15           # 不需要 GPU 就能装
python3 lower_num_stages.py --num-stages 1 2 3 4 --dump-dir /tmp/tl22
diff -u /tmp/tl22/gemm_num_stages_2.cu /tmp/tl22/gemm_num_stages_3.cu
TL_LOWER_TRACE=terminal python3 lower_num_stages.py --num-stages 3
```

如果你有卡，可以再补一段性能测量（脚本不代做）：

```bash
# 用同一个 kernel 编译 5 档，再 do_bench
python3 -c "..."    # 见 code/01/README.md 的步骤；用 kernel.get_profiler().do_bench()
```

### 7.3 预期观察点

| 观察点 | `num_stages=1` | `num_stages=2` | `num_stages=3` | 为什么 |
| --- | --- | --- | --- | --- |
| `software_pipeline_stage` | `[0, 0, 1]` | `[0, 0, 1]` | `[0, 0, 2]` | 自动规划把 copy 放 stage 0、消费者放最后一级；「整体减一」那一步要求 `num_stages ≥ 2`（`pipeline_planning.cc:1266-1277`） |
| A/B 的版本数 | 2 | 2 | 3 | `use − def + 1`；§2.4 |
| `__shared__` 首维 | 2 | 2 | 3 | §3.1 |
| mbarrier 的槽数 | 2 | 2 | 3 | `ExpandPipelineBarriers`：`[nv × 原槽数]` |
| `cp.async.wait_group` 的参数 | 小 | 中 | 大 | 在飞组数随深度变化 |
| 稳态循环里是否还有 `if` 守卫 | 无 | 无 | 无 | body 段 `need_bound_check=false` |
| prologue / epilogue 是否有守卫 | 有 | 有 | 有 | 两段都是 `unroll=true, need_bound_check=true` |

三点提醒：

* **`num_stages=1` 也仍然是「流水循环」**（注解存在、深度 2）。「不做流水」是
  `num_stages=0`：那时 `src/ir.cc:138-144` 根本不写入注解。
* **`num_stages=2` 的版本数是否真的是 2**，取决于 §2.4 那个收紧特例在不在。
  如果你观察到它降成 1，那不是 bug——那是一次成功的不必要复制消除。
* **不要指望同步点数量单调。** 深度变大通常让必需的同步点被隐藏，但也可能因为
  ring 变长而让某些冲突从「被版本化消解」变成「需要显式同步」。看到非单调，
  按 §4 的路径逐条对。

关于这张表的可信度，必须分开说：**`num_stages=3` 那一列有上游单测的直接断言**
（R3）；`num_stages=1` 那一列是**从源码守卫 `num_stages >= 2` 推出来的**，
我在上游测试里没有找到覆盖它的用例，因此这一列标注为「**待核**」。
`num_stages=2` 那一列介于两者之间：同目录的
`test_pipeline_planning_before_after_copy_consumer_num_stages_plan` 对
**两语句**形态（copy/copy，没有 gemm）给出的是 `stage = [0, 1]`、`order = [0, 1]`，
与「消费者落在 `num_stages − 1`」一致，但三语句 GEMM 形态在 v0.1.15 的测试里
只覆盖了 `num_stages=3`。

### 7.4 本机的验证边界（必须写清）

作者本机是 macOS (Apple Silicon)，**没有装 TileLang**，也**没有 NVIDIA GPU 和
昇腾 NPU**。因此：

* `code/01` 只做到**仅静态核对**（`py_compile` 通过 + API 名在 v0.1.15 里逐个
  核对），在本机运行会打印 `cannot import TileLang` 并退出，这是预期行为；
* 上面「预期观察点」一栏里的数组取自上游单测对同一形态的断言
  （`test_pipeline_planning_before_after_copy_gemm_num_stages_plan` 的
  `software_pipeline_stage: [0, 0, 2]`），**不是作者跑出来的**；
* `code/02`–`code/04` 是纯标准库模型，在**本机实际运行过**，验证级别为
  「CPU 逻辑验证」；它们的输出是模型结果，不是硬件测量。

---

## 8. 常见坑

### 8.1 stage 与 buffer 版本数不是一回事

最常被混起来的两组数：

```text
stage                 每条语句的位置，由 stage 数组给出，范围 [0, max_stage]
depth / num_stages    有几个时间槽，由注解或 max(stage)+1 给出
num_versions          某个 buffer 要留几份 = use_stage − def_stage + 1
```

它们之间的关系是**不等式**，不是等式：

* `num_versions ≤ depth`（因为 `use ≤ max_stage = depth − 1`，`def ≥ 0`）；
* 一个 buffer 在 `num_stages=4` 时完全可能只留 1 份（如果它在同一个 stage 里
  写后即读）；
* 反过来，手写 `stage = [0, 1, 2, 3]` 让某条依赖跨了 3 个 stage，就会得到
  4 份版本，即使你没写 `num_stages`。

**工程建议**：讨论容量时永远说「哪个 buffer 的版本数」，不要说「这个 kernel 的
版本数」——后者不存在。

### 8.2 多缓冲与 buffer 复用冲突

§3.4 的结论可以用一句更短的句子概括：

> 多缓冲吃掉的不只是「绝对字节数」，还有「可复用的概率」。

两个具体的坑：

1. **把长命 buffer 和短命 buffer 放在同一个 kernel 里。** 长命 buffer 会让 arena
   从头到尾都被占一块，短命 buffer 只能在剩下的碎片里找位置；对齐要求一上来就
   可能放不下。诊断办法是把 `MergeSharedMemoryAllocations` 的 `disable_reuse`
   打开做对照：如果打开后占用几乎不变，说明**本来就没在复用**，瓶颈是绝对大小；
   如果打开后暴涨，说明复用一直在起作用，问题在别处。
   （开关是 pass config `TL_DISABLE_SHARED_MEMORY_REUSE`，
   见 `_vendor/tilelang-v0.1.15/tilelang/transform/pass_config.py:210`。）
2. **以为「加了版本之后还能复用」。** 版本维是加在 buffer 最外层的，整个多版本
   buffer 作为一个整体参与打包；不同版本之间不会互相复用。

### 8.3 同步插入与 warp specialization 的交互

warp specialization（WS）把「生产者 warp」与「消费者 warp」显式拆开，各自跑自己
的代码。它对本章的三件事都有影响：

1. **顺序**：WS 的三趟 pass 在流水规划**之前**跑
   （`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:105-110`：
   `AutoWarpSpecialization → MaterializeWSSchedule → ProducerConsumerWarpSpecialized`），
   而 `PipelinePlanning → InjectSoftwarePipeline` 在 `:129-130`。也就是说，
   ISP 看到的已经是 WS 之后的 IR。源码注释解释了原因：

   > Shared buffers are multi-versioned internally only for functions where the
   > WS transformation actually applies.

2. **barrier 归属**：做了 WS 的 kernel，TMA 的 mbarrier 由 WS 那一趟管；
   `InjectSoftwarePipeline` 的「pipeline-level TMA barrier management」这一步
   **只对没有 WS 的 kernel**生效（`inject_pipeline.cc:3616-3640` 的注释明确写了
   "when TMA copies are present (without warp specialization)"）。

3. **注解清理**：`PipelinePlanning` 会把 `num_stages` 换成
   `tl_pipelined_num_stages`，理由正是「下游有些 pass（内部 buffer 版本化 /
   warp specialization）把它当作 active pipeline marker，且不支持嵌套流水」
   （`pipeline_planning.cc:1287-1293`）。所以**不要**自己往循环上补回
   `num_stages` 注解；那会让下游误以为还有一层流水。

一句话：**WS 与流水是两套都会插同步的机制，同时打开时必须搞清「这一次是谁在管
barrier」**。搞混的典型症状是同一个位置上出现两层 barrier（性能掉），或者一层都
没有（偶发错误）。

### 8.4 另外四个小坑

* **嵌套流水不被支持。** `NestedLoopChecker` 把 `num_stages` /
  `tl_pipeline_order` / `tl_pipeline_stage` 列为流水注解键，并对嵌套情形给
  「corner cases and undefined behaviours」的警告
  （`_vendor/tilelang-v0.1.15/tilelang/analysis/nested_loop_checker.py:12-15`、`:65-100`）。
  这是 `PipelinePlanning` 只删 `num_stages`、保留 `tl_pipelined_num_stages` 的
  另一个原因。
* **`num_stages` 与 `order`/`stage` 同时给。** docstring 明确说
  「Avoid setting `num_stages` together with manual `order`/`stage` unless you
  intend an explicit depth override」
  （`tilelang/language/loop.py:112-186`）。两者同时给时，一个是深度、一个是要被
  反推深度的数组，很容易写出「深度 4、stage 只到 1」这种自相矛盾的组合。
* **ROCm 上会静默退化。** `PipelinePlanning` 对非 `gfx950` 的 ROCm target 会
  **删掉** `num_stages` 注解，让循环退回顺序执行
  （`pipeline_planning.cc:1150-1170`）。也就是说同一份代码在 NV 上是流水的、
  在 MI300X 上不是。
* **`Bind` 占不占注解位。** 可重放的标量 `Bind` 不占；但如果这个 `Bind` 读了
  同一个流水体里被写过的 buffer，它就变成有依赖的语句，必须占位
  （`language/loop.py` docstring 里那段 `idx = Ids[i]` 的例子）。

---

## 9. 本章小结

把这一章压缩成七句话：

1. **形式化**：软件流水的输入是「带距离的依赖边」。距离 `d ≥ 1` 的边，可以用
   「多一份版本」或「多等一轮」解决，选哪个由规划器决定。
2. **排程**：`(stage, order)` 是逻辑坐标，模调度把它变成
   「第 `k` 轮的第 `s` 级放在第 `k + s` 个时间槽」；prologue / body / epilogue 是
   这条映射在边界上的余数（body 无守卫，两段有守卫）。
3. **版本数**：`num_versions = use_stage − def_stage + 1`，是上界；
   `use − def + 1 == 2` 时源码会检查「是否真的存在相交的晚写早读」来决定要不要降成 1。
4. **物化**：给 buffer 的 shape 前置版本维，把访问下标改成 `(k − min) mod nv`，
   放宽稳态里的访问区域，并且**别忘了 barrier 也要按 stage 展开并带 parity**。
5. **分配**：多版本 buffer 是一个不可分割的块，进 arena 做 linear-scan + best-fit
   打包；能不能复用只取决于「活跃区间是否相交 + 对不对齐」。
6. **同步**：CUDA 用 `AccessEntry` + Z3 找冲突，插 `__syncthreads()`，另用
   `wait_group` 与 mbarrier parity 覆盖异步；昇腾用「site + 有向 edge + 版本数」
   生成 `SetFlag/WaitFlag`，并在每个 hard event 的 8 个槽里做背包分配。
7. **代价**：漏一个同步是偶发的正确性错误；多一个同步是确定的性能损失，而且
   深度救不了它。同步还有第二个预算（flag 槽位 / 在飞组数），它可能先于
   shared memory 成为瓶颈。

## 延伸阅读

* 想先复习流水的基本用法与时空图：见第 11 章。
* 想理解 `SBlock` / pass pipeline / IR dump：见第 20 章，尤其是「怎么自己 dump IR」。
* 想看这些 pass 在 NVIDIA 硬件上的完整后果（TMA、wgmma、cluster）：
  见第 24 章。
* 想看这些 pass 在昇腾硬件上的完整后果（Ascend C / PTO、L1/UB/L0、dual_copy）：
  见第 25 章。
* 想看昇腾的自动调度与约束求解（延迟估计、Z3、多缓冲物化的完整链条）：见第 23 章，
  其中 3.5 节讲的就是本章 §3.5 提到的 owner / epoch / clock。
* 想看布局推导如何在多版本 buffer 之后接上：见第 21 章
  （`InjectSoftwarePipeline` 会在加版本维之后同步更新 `layout_map` 的
  `UpdateExpandedLayoutMapForRemappedAllocs`）。
* 本章引用的全部源码位置、corpus 资料与核验状态见 `references.md`；
  可复算的实验与模型见 `code/`。
