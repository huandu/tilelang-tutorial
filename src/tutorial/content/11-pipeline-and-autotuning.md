# 第 11 章 流水线与自动调优

## 学习目标

读完本章，你应当能够：

1. 用“搬运与计算的重叠”解释为什么同一个 GEMM，加了 `num_stages=3` 之后能明显变快，
   并说清流水优化的是**吞吐**而不是单次延迟。
2. 写出一个循环的时空图：哪一段是填充（prologue）、哪一段是稳态、哪一段是排空
   （epilogue），并据此判断“这个循环值不值得做流水”。
3. 读懂 `T.Pipelined(..., num_stages=N)` / `order=` / `stage=` 的语义，知道每种写法
   下编译期会走哪条路径，以及 `num_stages` 在什么情况下等于“不做流水”。
4. 说清 `PipelinePlanning` 与 `InjectSoftwarePipeline` 各自负责什么，并把
   `stage`/`order` 注解、`use - def + 1` 的版本数、多缓冲版本维这三件事串起来。
5. 在动手编译之前估算一份配置的片上占用，判断 `num_stages` 还能不能再加。
6. 用 `@tilelang.autotune` 或 `AutoTuner.from_kernel` 写一个可信的搜索空间，读出最优
   配置，并把它固定成函数默认值。
7. 说出 carver / roller 的 cost model 在整条链路里干什么、不干什么。

## 前置知识

* 第 7 章：安装、`@tilelang.jit`、编译缓存、`get_kernel_source()`。
* 第 8 章：`T.Kernel`、buffer 与内存作用域、`T.serial` / `T.Parallel`。
* 第 9 章：`T.copy` / `T.gemm` / `T.clear` 的语义。
* 第 10 章：shared memory 与 fragment 的区别、layout 为什么是接口。
* 第 6 章（可选，本章 2.6 节要用）：NV 与昇腾在存储层级与同步模型上的差异。

本章几乎不引入新的硬件名词；所有新概念都先用“它解决什么问题”讲一遍。

**本章的验证边界（很重要）**：写作环境是 macOS (Apple Silicon)，**没有 NVIDIA GPU，
也没有昇腾 NPU**。因此本章出现的数字只有两类，来源都写清楚：

* **模型输出**（来自 `code/01`–`code/03`，纯标准库，可在任意机器复算）；
* **厂商公开资料或上游论文里的数字**（带 corpus 路径）。

本章**没有任何一条实测性能数字**，也不声称任何 kernel 被运行过。

---

## 1. 问题的由来：不做流水会怎样

### 1.1 一个 GEMM 主循环里，时间都去哪儿了

第 14 章会展开 GEMM 的全家族，这里先只看最小的那个形态。一个 tile 化的 GEMM 把
`M×N` 的输出切成 `block_M × block_N` 的块，然后沿 `K` 方向迭代：

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_K), dtype)
    B_shared = T.alloc_shared((block_N, block_K), dtype)
    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

    T.clear(C_local)
    for ko in T.serial(T.ceildiv(K, block_K)):               # ← 不做流水的版本
        T.copy(A[by * block_M, ko * block_K], A_shared)      # 搬运
        T.copy(B[bx * block_N, ko * block_K], B_shared)      # 搬运
        T.gemm(A_shared, B_shared, C_local, transpose_B=True)  # 计算

    T.copy(C_local, C[by * block_M, bx * block_N])
```

每一轮做两件事：**把下一块数据搬进片上**，**对上一块数据做矩阵乘**。它们由不同的硬件
负责，搬运走访存通路，计算走矩阵运算单元。所以这两条通路**天生可以同时工作**。

但上面这段代码里它们没有同时工作。原因是 `A_shared` / `B_shared` 各自只有一份：

1. 第 `ko` 轮的 `T.copy` 写 `A_shared`；
2. `T.gemm` 读 `A_shared`；
3. 只有 `T.gemm` **读完**之后，第 `ko+1` 轮的 `T.copy` 才敢覆盖 `A_shared`。

于是每一轮的时间是“搬运 + 计算”，而不是“搬运与计算取较大者”：

```text
不做流水（教学简化：把 num_stages 当作缓冲份数，一份 buffer）

  搬运   [ c0 ]····[ c1 ]····[ c2 ]····[ c3 ]
  计算   ····[ g0 ]····[ g1 ]····[ g2 ]····[ g3 ]
         （· = 空闲。两条车道从不同时忙，每轮 = 搬运 + 计算）

2-stage（num_stages = 2，两份 buffer）

  搬运   [ c0 ][ c1 ][ c2 ][ c3 ][ c4 ][ c5 ]
  计算   ····[ g0 ][ g1 ][ g2 ][ g3 ][ g4 ]
         （两条车道同时在忙 = 重叠；深度足够时每轮 ≈ max(搬运, 计算)）
```

图 1 会把不做流水、2-stage、3-stage 三档画成带具体周期数的时空图，并附上 1..8 的
扫描表。

“不做流水会怎样”的答案就是这句话：**搬运与计算各自都要排队等对方，二者的和成了
每一轮的下界。**

### 1.2 把“等”量化：三个量

要把上面的直觉变成可以算的东西，只需要三个量（都以“周期”为单位）：

| 记号 | 含义 | 直觉 |
| --- | --- | --- |
| `issue` | 发一次搬运要占用搬运引擎多久 | 引擎的**事务处理时间** |
| `latency` | 数据从发出到可以被读，隔了多少周期 | 数据在路上的**飞行时间** |
| `compute` | 一次计算占用计算单元多久 | 计算单元被**占住的时间** |

为什么要把 `latency` 单独拎出来，而不是把搬运当成一件整体的活儿？因为现代加速器把
“发出”和“到达”拆开了。H100 的厂商资料原话是：

> It enables applications to build end-to-end asynchronous pipelines that move data into
> and off the chip, completely overlapping and hiding data movement with computation.
> 来源：`../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md`
> 第 79 行（NVIDIA Hopper Architecture In-Depth，厂商公开资料；条件：H100）

同一份资料在解释异步执行时又说：

> Asynchronous execution involves finding independent tasks to overlap with memory
> transfers and other processing. The goal is to keep all the units in the GPU fully
> used. ，同文件第 241 行

“发出之后不用干等”就是为什么一份 buffer 不够：**要重叠，就得让下一轮的数据已经在
路上，而这一轮的计算还没结束。**

至于 `latency` 为什么要单独考虑，第 1 章引用过的数据给出了量级：访存“hundreds of
clock cycles”，而算术指令“typically 4 cycles”（来源：
`../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md` 第 3629 行与
第 3145 行；条件分别是未缓存的 local/global 访存，以及 compute capability 7.0 设备上
多数算术指令的延迟）。两者差两个数量级，所以“让计算等数据”和“让数据等计算”的
代价完全不同。

### 1.3 用本章的模型算一遍

`code/01-pipeline-timeline/` 是一个 60 行的离散事件模型：给定
`(issue, latency, compute, stages)`，用贪心调度算出每一轮搬运与计算的起止时间。
把它跑起来（作者在 macOS + Python 3.11 上实际运行过，输出见该目录 README）：

```text
model: copy -> compute, greedy list schedule, abstract cycles
  iterations=8 issue=1 latency=30 compute=20
  latency/compute = 1.50 -> smallest depth that reaches the steady state = 3

== summary ==
+--------+--------------------------------------------------------------------+
| stages | makespan  speedup  issue_iv  compute_iv                            |
+--------+--------------------------------------------------------------------+
| 1      |   400  x1.00  issue_iv= 50.0  compute_iv= 50.0  compute_util=  43% |
| 2      |   220  x1.82  issue_iv= 24.3  compute_iv= 24.3  compute_util=  84% |
| 3      |   190  x2.11  issue_iv= 18.6  compute_iv= 20.0  compute_util= 100% |
| 4      |   190  x2.11  issue_iv= 15.7  compute_iv= 20.0  compute_util= 100% |
+--------+--------------------------------------------------------------------+
```

这里是**模型**在说话，不是硬件。但三条结论可以直接搬到真实 kernel 上：

1. **不做流水时，每轮 = `latency + compute` = 50 个周期**。计算单元有 57% 的时间
   在等数据（利用率 43%）。
2. **2-stage 已经能拿到大部分收益**（1.82×），但还没到底：每轮 24.3 周期，
   说明深度 2 只盖住了一部分飞行时间。
3. **3-stage 之后到顶**：每轮 20 周期，正好等于 `compute`；再加到 4、5、6… 总时长
   一动不动（`--sweep` 可复现）。多出来的 stage 只做一件事：**占内存**。

模型还给出一个可以背下来的判据：

$$
D_{\min}=1+\left\lceil\frac{L}{C}\right\rceil
=1+\left\lceil\frac{30}{20}\right\rceil=3.
$$

这里 $D_{\min}$ 是最小够用深度，$L$ 是搬运延迟，$C$ 是每轮计算时间。

直觉是：要盖住 `latency` 那么长的飞行时间，生产者得能领先消费者
`ceil(latency / compute)` 轮；再加上消费者正在读的那一份，就是需要的份数。

### 1.4 为什么长 reduction 循环收益最大

流水有一段**填充（prologue）**：第一个 `copy` 没有任何计算可以与之重叠。这段填充的
绝对长度与迭代次数无关，所以迭代越多、填充占比越小。在 `code/01` 里把
`--iterations` 从 8 改成 16，加速比从 2.11× 升到 2.29×。

这解释了为什么 GEMM 的 `K/BK` 循环、Attention 的 KV 长度循环、MoE 的 token 循环都是
流水的主战场：它们的 reduction 维度很长，填充可以被摊薄到几乎可以忽略。反过来，
一个只跑两三轮的短循环，做流水的意义就有限。**这也是一类值得先算一笔账再动手的
优化**。

---

## 2. 概念与模型

### 2.1 stage、order、depth：先把三个词钉死

TileLang 的流水线里只有三个概念，但它们的名字很容易混：

| 词 | 定义 | 在 API 里的位置 |
| --- | --- | --- |
| **stage** | 一条被调度的语句属于管线的第几级。**数字越小越早发生** | `T.Pipelined(..., stage=[...])` |
| **order** | 同一级内部，语句按什么顺序发射 | `T.Pipelined(..., order=[...])` |
| **depth**（深度） | 缓冲区份数，也就是同时能有几份数据在飞 | `T.Pipelined(..., num_stages=N)` |

`T.Pipelined` 的 docstring 把三者的关系写得很直白
（`_vendor/tilelang-v0.1.15/tilelang/language/loop.py:112-186`）：

> `num_stages`: the max number of buffer used between pipeline producers and consumers.
> For compiler-inferred pipelines, if `num_stages` is 0, pipeline will not be enabled.
> For manually scheduled pipelines, prefer specifying `order` and `stage` without
> `num_stages`; the pipeline depth is inferred as `max(stage) + 1`.

> **口径提醒（与第 22 章一致）**：本章为了讲清直觉，把 `num_stages` 当作“缓冲份数”；
> 严格说它是给 planner 的**上界/提示**，最终版本数由 `ComputeBufferVersions` 按
> `use − def + 1` 推出，两者可能不同（`num_stages=1` 仍可能得到一个带多份 buffer 的
> 流水循环）。教学示意与实际 IR 的差异见第 22 章 §2.4 与 §7.3。

也就是说，**深度有两种给法**：要么直接给 `num_stages`（让编译器自己决定 stage/order），
要么给 `stage`/`order`（深度由 `max(stage) + 1` 推出来）。**不要两个一起给**，
除非你确实想要覆盖深度，docstring 的原话是“Avoid setting `num_stages` together with
manual `order`/`stage` unless you intend an explicit depth override”。

还有一个容易被忽略的细节：`num_stages=0` 时，IR 上根本不会出现 `num_stages` 注解。
参数在 `src/ir.cc` 里就被丢掉了：

```cpp
// _vendor/tilelang-v0.1.15/src/ir.cc:138-144
Map<String, Any> anno = annotations;
if (num_stages > 0)
  anno.Set("num_stages", PrimExpr(num_stages));
if (!order.empty())
  anno.Set("tl_pipeline_order", order);
if (!stages.empty())
  anno.Set("tl_pipeline_stage", stages);
```

`num_stages=0` 于是意味着“这个循环不是流水循环”，下游的 `PipelinePlanning`、
`InjectSoftwarePipeline` 都不会碰它。自动调优时把 `0` 放进搜索空间，就是在搜索里包含
“不做流水”这个选项；`_vendor/tilelang-v0.1.15/examples/gemm/example_gemm_autotune.py`
正是这么做的（它的 roller 映射里有一句
`config["num_stages"] = hint.pipeline_stage if hint.pipeline_stage > 1 else 0`）。

### 2.2 时空图（图 1）

**图 1 · 流水时空图（不做流水 vs 2-stage vs 3-stage）** 见
`figures/01-pipeline-space-time.md`（Mermaid 版 + ASCII 版）。它把 `code/01` 的输出
画成了时空图。读图只要看三件事：

1. **有没有重叠**：图里 `B`（搬运与计算同时进行）出现的次数，就是重叠的周期数。
   不做流水时 `B` 一次都不出现。
2. **填充有多长**：最左边那段“只有搬运、没有计算”的区间就是 prologue。
3. **稳态的步长是多少**：稳态里相邻两次计算开始时间的差，就是每一轮的真实成本。
   它等于

```text
每轮周期 = max( compute, issue, (latency + compute) / depth )
                                 ↑ 在「够深」之前，瓶颈在这一项
```

   直觉是：每一个缓冲区从“被写入”到“被读完”要占 `latency + compute` 那么久，
   `depth` 个缓冲区同时在飞，所以产能上限是 `depth / (latency + compute)`。
   把这个上限与 `1/compute` 一比，就得到 1.3 节那条判据：
   `depth ≥ 1 + latency / compute` 之后，瓶颈才从缓冲区数量转到计算本身。

### 2.3 多缓冲：为什么一份 buffer 不够（图 2）

**图 2 · 多缓冲的内存布局与槽位轮转** 见 `figures/02-multibuffer-layout.md`。
要让第 `ko+1` 轮的搬运与第 `ko` 轮的计算重叠，第 `ko+1` 轮就不能写第 `ko` 轮正在读的
那块内存。唯一的办法是**多准备几份**：把“一份 A tile”变成“N 份 A tile”。

编译期做这件事的方式非常具体（图 2 有完整示意）：

1. 先做活跃期分析，得到每个 buffer 的“被定义的 stage”与“最后被读的 stage”；
2. 版本数 = `use_stage - def_stage + 1`；
3. 把版本数作为**最外层新维度**插到 buffer 的 shape 前面，例如
   `(block_M, block_K)` → `(num_stages, block_M, block_K)`；
4. 每个访问点改写为 `buffer[ko % num_stages, ...]`。

代价是**片上存储线性增长**：深度 `N` 意味着 `N` 份 A tile 与 `N` 份 B tile。
第 4.5 节会把这件事算清楚，因为它是本章唯一一个“加 stage 会直接把 kernel 编译失败”
的硬约束。

### 2.4 吞吐 vs 延迟：流水到底优化了什么

这是最容易说错的一点，值得单独讲清楚。

* **延迟（latency）**：一次操作从开始到结束要多久。流水**几乎不改变**它，第 `ko` 轮
  的计算在数据到齐之后要跑 `compute` 那么久，这个数字没变。
* **吞吐（throughput）**：单位时间内能完成多少轮迭代。流水**改变的就是它**，
  从 `1/(latency + compute)` 变成 `min(depth / (latency + compute), 1/compute)`
  （这里忽略 `issue` 的影响；把 `issue` 算进来就是上一小节的完整式子）。

一个只跑一次的算子（单次 `reduce`、batch size = 1 的 decode 首步），流水能做的只是把
填充藏到别的工作后面；一个跑几千轮的循环（大 `K`、长 context），流水能把吞吐抬到
接近计算单元的物理上限。所以“要不要加流水”“加多深”这两个问题的答案，取决于
**这个循环有多少轮**：

```text
迭代次数 N 很大  →  填充被摊薄，深度越大越接近稳态上界，值得加
迭代次数 N 很小  →  填充占主导，加深度几乎没用，还可能因为占用片上加不上
每轮计算很短     →  稳态上界由搬运决定，加计算深度没用，该去优化搬运
```

### 2.5 自动调优在解决什么

上面这些参数，`block_M`、`block_N`、`block_K`、`num_stages`、`threads`，**没有一组
能在所有形状、所有 GPU 上都最优的值**。原因很朴素：

* tile 越大，片上占用越大，能同时驻留的 block 越少（occupancy 下降）；
* `block_K` 越大，每轮的搬运越“整块”，但循环轮数变少，填充占比上升；
* `num_stages` 越大，延迟藏得越好，但片上占用线性增长，超过预算直接编译失败；
* 不同 SM 数的 GPU 对同一份 tile 的 wave 划分不同，尾部浪费也不同。

这些效应互相耦合，靠解析式算不准。**自动调优就是用真实测量替代猜测**：把参数暴露成
函数参数、给一个候选集合、让编译器并排编译并测量、选出最好的那一组。它的输入是
“搜索空间”，输出是“一组具体参数 + 它的实测延迟”。

### 2.6 两台机器的流水抽象对照

NVIDIA 与昇腾都做“搬运与计算重叠”，但**抽象层次不一样**。下表只做定位，机制细节
留给第 23 章。

| 维度 | NVIDIA 侧（v0.1.15 的 CUDA pipeline） | 昇腾侧（v0.1.15 的 Ascend pipeline） |
| --- | --- | --- |
| 用户写什么 | `T.Pipelined(..., num_stages=N)`，必要时给 `stage`/`order` | 同样写 `T.Pipelined`，通常只给深度；`T.alloc_*` 落到 L1 / UB / L0A-L0C 等不同存储 |
| 谁决定调度 | `PipelinePlanning` 按数据流自动定 stage/order，`InjectSoftwarePipeline` 落地 | `AutoSchedule` 把算子任务化后用 **Z3 求解**开始时间；`EstimateLatency` 先估延迟 |
| 多缓冲怎么来 | `ComputeBufferVersions` 算版本数，`RewriteAllocBuffer` 加版本维 | `PrepareMultiBuffer` → `MaterializeMultiBuffer` 两个 pass 物化 |
| 同步怎么插 | `InjectSoftwarePipeline` 内部生成异步拷贝与其等待；`ThreadSync` 补线程同步 | `InsertSync` 统一插入成对的 flag 读写 |
| 深度信息从哪来 | 循环注解 `num_stages` / `tl_pipelined_num_stages` / `max(stage)+1` | 同一套注解，但求解器还会自己决定任务顺序与缓冲区生命周期 |
| 有没有“核间”流水 | 没有这个概念：同一个 block 内所有单元在同一套线程/同步模型下 | 有：同一个 `T.Pipelined` 里可以让 Cube 与 Vector 交叠，见 4.6 节 |
| 你要不要手写 stage/order | 需要时才写（例如循环体不规整） | 一般不需要 |

对应的源码位置：

* NVIDIA：`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:129-142`
  （`PipelinePlanning → InjectSoftwarePipeline → LayoutInference → ...`）。
* 昇腾：`_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:107-115`
  （`EstimateLatency → AutoSchedule → AssignCore → PrepareMultiBuffer → ResolveCore →
  InsertSync → MaterializeMultiBuffer → ...`）。

**本章只需要记住两个结论**：

1. `T.Pipelined` 这个**接口**在两边一样；不一样的是它背后那把“决定 stage/order、
   决定缓冲区份数、决定同步点”的引擎。
2. 因此第 4.5 节的容量约束两边都存在，但**约束的具体资源不同**：NVIDIA 侧主要看
   shared memory，昇腾侧要看 L1 / UB / L0 各自的大小以及它们之间搬运的粒度。

---

## 3. 用法

本节从“最小可用”走到“自动调优”，每一步都有可对照的代码：
`code/04-tilelang-pipelined-gemm/` 覆盖 3.1–3.4，`code/05-tilelang-autotune-gemm/`
覆盖 3.5–3.8。两个示例都需要 GPU 才能运行，本章只做到静态核对，验证级别写在各目录
的 README 里。

### 3.1 最小可用：只给 `num_stages`

回到 1.1 节那个循环，把 `T.serial` 换成 `T.Pipelined` 就够了：

```python
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.copy(B[bx * block_N, ko * block_K], B_shared)
    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
```

这三行就是 GEMM 流水的全部。`matmul_plain` 那个 `T.serial` 版本不是凑数：它把 1.1 节的
“不做流水”写成了可编译的对照组。

### 3.2 参数逐个解释

`T.Pipelined` 的完整签名
（`_vendor/tilelang-v0.1.15/tilelang/language/loop.py:112-118`）：

```python
def Pipelined(
    start, stop=None, num_stages=0,
    order=None, stage=None, annotations=None,
) -> frame.ForFrame:
```

| 参数 | 语义 | 什么时候用 |
| --- | --- | --- |
| `start` / `stop` | 循环范围；只给一个参数时它就是 `stop`，`start` 默认为 0 | 总是 |
| `num_stages` | 生产者与消费者之间的缓冲区份数上限；`0` 表示不启用流水 | 编译器能自己推断时（GEMM 这类规整体） |
| `order` | 被调度语句的发射顺序 | 需要把“未来轮次的搬运”插到“当前轮次的计算”之前时 |
| `stage` | 每条被调度语句的逻辑 stage | 循环体不规整、自动推断结果不合意时 |
| `annotations` | 透传给下游 pass 的额外注解 | 高级用法 |

关于 `order` 和 `stage` 的**对齐规则**，官方文档说得很清楚
（`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/software_pipeline.md`）：
两个数组**按源码顺序**对齐到“被调度的语句”，而且**只数有副作用的语句**，
copy、fill、gemm、reduce、store、atomic，以及显式写出的 wait / commit / 同步语句。

还要注意它们**不是**“数组下标 = 源码行号”。下面两种写法都不行：

* 数了不该数的（把可重放的标量 `Bind` 也算进去，见 3.4 节）；
* 用了一套顺序去描述另一个顺序（例如按发射顺序数语句，而数组要按源码顺序对齐）。

### 3.3 手动 stage/order

最常见的两种手动写法。

**（a）标准的两级拆分**，搬运在 stage 0，计算在 stage 1：

```python
for ko in T.Pipelined(
    T.ceildiv(K, block_K),
    stage=[0, 0, 1],     # copy A, copy B, gemm —— 按源码顺序
    order=[0, 1, 2],
):
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.copy(B[bx * block_N, ko * block_K], B_shared)
    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
```

深度由 `max(stage) + 1 = 2` 推出，恰好对应 2.3 节说的“两份 buffer”。

**（b）调整发射顺序**，让“下一轮的生产者”先于“这一轮的消费者”发射：

```python
for ko in T.Pipelined(num_tiles, stage=[0, 1], order=[1, 0]):
    T.copy(A[ko * BK], A_shared)              # stage 0, order 1
    T.gemm(A_shared, B_shared, C_local)       # stage 1, order 0
```

官方文档对这一对的解释是：编译期生成 prologue / 稳态 / epilogue 时，会为每个 stage 取
**正确的逻辑迭代号**；而源码顺序只负责把注解的每一项对到哪条语句上，`order` 管的是
“分类之后按什么顺序发射”。这两个数组的职责是分开的，这一点在手写调度时最容易搞混。

编译器会校验你的注解。`InjectSoftwarePipeline` 的 `ValidatePipelineBody` 有两条硬规则
（`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:3104-3138`）：

1. **两条语句不能有相同的 `order`**，报错文案是
   `ValueError: Two statements in the software pipeline cannot have the same order`；
2. **生产者不能晚于消费者**：有 buffer 依赖的 `(生产者 src → 消费者 dst)` 必须满足
   `stage(src) ≤ stage(dst)`；若两级相同，还必须满足 `order(src) < order(dst)`。

这两条读数阵的时候可以直接当检查清单用。

### 3.4 那条会咬人的规则：replayable scalar `Bind`

下面这段代码里，`stage`/`order` 该写几个数？

```python
for ko in T.Pipelined(num_tiles, stage=[0, 1], order=[1, 0]):
    base: T.int32 = ko * BK          # ← 这一行要不要占一个位置？
    T.copy(A[base + tx], A_shared[tx])
    B[base + tx] = A_shared[tx]
```

答案是 **两个**，`base` 不占位置。官方文档把这条规则称为“可重放的标量 `Bind`
（replayable scalar `Bind`）”：

* 一个 `Bind` 是**值别名**，没有存储、没有副作用；
* 它可能被不同 stage 的多条语句使用，而每条语句需要的是**自己那一轮的** `base`；
* 因此“给 `base` 选一个 stage/order”在语义上就没有唯一正确答案。

编译器的处理是：把可重放的 `Bind` 从可调度语句里剔除，在**每个消费者前面按需重建**它，
并代入该消费者自己的逻辑迭代号。

两种情况下 `Bind` **不是**可重放的，必须占一个注解位置：

1. 它读的 buffer **在同一个流水体内被写过**。例如
   `value: T.float32 = A_shared[tx]`。这个读依赖前一级的生产者，不能随意重放。
2. 你想要的是“算一次、多处共享”的语义。那就把它**物化**成一次 buffer 或寄存器写，
   而不是靠 `Bind` 重放。

旧代码把可重放的 `Bind` 也写进 `order`/`stage` 的做法仍然被接受（为了兼容），但那几项
会被忽略；新代码请只标注真正有副作用的语句。

### 3.5 自动调优之一：装饰器路径

`@tilelang.autotune` 叠在 `@tilelang.jit` 上，接口是声明式的
（`_vendor/tilelang-v0.1.15/tilelang/autotuner/tuner.py:1318-1345`）：

```python
import tilelang
import tilelang.language as T


def matmul_configs(M, N, K):
    return [
        dict(block_M=BM, block_N=BN, block_K=BK, num_stages=S, threads=TH)
        for BM in (64, 128, 256)
        for BN in (64, 128, 256)
        for BK in (16, 32, 64)
        for S in (1, 2, 3, 4)
        for TH in (128, 256)
    ]


@tilelang.jit(out_idx=[-1])
def matmul(M: int, N: int, K: int,
           block_M: int = 128, block_N: int = 128, block_K: int = 32,
           num_stages: int = 3, threads: int = 128,
           dtype: str = "float16", accum_dtype: str = "float32"):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype),
               C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return kernel


# 再叠一层 autotune，就得到「声明式」的调优入口
matmul_tuned = tilelang.autotune(
    configs=matmul_configs, warmup=25, rep=100, timeout=60,
)(matmul)

from tilelang.autotuner import set_autotune_inputs
import torch

M = N = K = 1024
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)
C = torch.empty(M, N, device="cuda", dtype=torch.float16)

with set_autotune_inputs(A, B, C):
    tuned = matmul_tuned(M, N, K)   # 编译 + 调优 + 返回最优 kernel
tuned(A, B, C)
```

写成装饰器也一样：`@tilelang.autotune(configs=...)` 直接叠在 `@tilelang.jit` 之上。
`tilelang.autotune(...)` 返回的就是一个 decorator，而它内部会断言被装饰的对象是
`JITImpl`（`_vendor/tilelang-v0.1.15/tilelang/autotuner/tuner.py:1397-1398`），
所以“叠装饰器”和“显式调用”是同一件事。

四个要点：

* **被调参数必须是有默认值的函数参数**。调优器会用候选值覆盖它们；没有默认值就没法单独
  编译某一个候选。
* **`configs` 里的键必须真的在函数签名里**。有不认识的键时 `AutoTuner.run` 会直接
  报错 `Unused keys in config`（见 5.3 节）；反过来，签名里有、内核里却不用的参数
  不会被报错，但会让整轮调优失去意义。
* **`configs` 可以是列表，也可以是函数**。函数的入参是调用时的实参（这里是
  `M, N, K`），返回值是候选列表。这让搜索空间可以随问题尺寸变化。
  `AutoTuner.run` 里明确处理了这个分支
  （`_vendor/tilelang-v0.1.15/tilelang/autotuner/tuner.py:866-870`）：
  如果 `configs` 是 `Callable`，就先用实参调它一次，把结果当成真正的候选列表。
* **`set_autotune_inputs` 让所有候选跑在同一份输入上**。这是可复现性的关键：否则每个
  候选可能拿到不同的随机张量，“最优”就可能只是运气。官方文档把它列为输入供给的最高
  优先级（`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/autotuning.md`）。

### 3.6 自动调优之二：`AutoTuner` 路径

程序式入口多给你两样东西：显式控制编译/测量参数，以及一个可以读、可以存的
`AutotuneResult`。

```python
from tilelang.autotuner import AutoTuner, set_autotune_inputs

tuner = (
    AutoTuner.from_kernel(matmul, configs=matmul_configs(M, N, K))
    .set_compile_args(out_idx=[-1], target="auto")
    .set_profile_args(
        warmup=25, rep=100, timeout=60,
        supply_type=tilelang.TensorSupplyType.Auto,
        ref_prog=lambda a, b: (a.float() @ b.float().T).to(a.dtype),
        skip_check=False,
    )
)
with set_autotune_inputs(A, B, C):
    artifact = tuner.run()

print(artifact.latency)      # 最优候选的实测延迟
print(artifact.config)       # 最有用的那一项：这一组参数
print(artifact.ref_latency)  # 参考实现的延迟（若提供了 ref_prog）
artifact.kernel(A, B, C)     # 直接拿最优 kernel 用
```

`AutotuneResult` 的字段
（`_vendor/tilelang-v0.1.15/tilelang/autotuner/param.py:155-180`）：

| 字段 | 含义 |
| --- | --- |
| `latency` | 最优候选的实测延迟，单位是毫秒；没有测量则为 `None` |
| `config` | 产生这个延迟的参数组合（一个 dict）；**这是唯一值得抄回代码的东西** |
| `ref_latency` | 参考实现的延迟；没给参考实现时为 `None` |
| `libcode` | 生成库的源码 |
| `func` | 被优化后的 `PrimFunc` |
| `kernel` | 编译好的 kernel，可直接调用 |

`AutoTuner.from_kernel` 的第一个参数是**被调优的 kernel 或 kernel 工厂**
（`_vendor/tilelang-v0.1.15/tilelang/autotuner/tuner.py:144-154`）：

```python
tuner = AutoTuner.from_kernel(matmul, configs=matmul_configs(M, N, K))
```

这里传的是 3.5 节那个 `@tilelang.jit` 工厂本身（**不是**已经叠了自动调优的
`matmul_tuned`）。官方文档的说法是“the function above (already `@tilelang.jit`)”，
也正是这个意思。

### 3.7 参数空间怎么写

官方文档给了三条实践建议
（`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/programming_guides/autotuning.md`）：

1. **从一个能跑通的基线出发**，先搜一个小的、有意义的空间；
2. **遵守硬件限制**（shared memory 字节、每线程/每 block 寄存器、每 block 最大线程数），
   **提前剔掉不可能的候选**；
3. 让 block 尺寸是向量宽度与 warp 大小的倍数；用 `set_autotune_inputs` 保证每个候选
   用同一份数据测量；定下来之后把最优配置**写回默认值**。

`code/05-tilelang-autotune-gemm/autotune_gemm.py` 里的 `grid_configs` 就是按第 2 条做的
预剪枝：先剔掉不能整除问题尺寸的 tile，再剔掉超 shared 预算的组合。对
4096³ 的问题、`(64,128,256) × (64,128,256) × (16,32,64) × (1,2,3,4) × (128,256)`
这个网格，216 个候选里有 214 个合法，被拒的恰好是
`block_M = block_N = 256, block_K = 64, num_stages = 4` 的那两个线程数变体
（`4 × 65536 B = 256 KiB`，超了）。这个数字可以用
`code/03-autotune-config-space/config_space.py` 复算。

### 3.8 结果怎么读、怎么固定

调优跑完之后有三件事要做，缺一件这次调优就白跑了：

1. **读 `config`，不读 `latency`**。延迟是这台机器、这个时刻的值；`config` 才是可以
   搬到代码里的东西。
2. **把 `config` 写成函数默认值**，并在注释里写清它是在什么机器、什么形状上调出来的。
   `examples/gemm/example_gemm_autotune.py` 的 `get_heuristic_config()` 就是这么做的：
   按 CUDA compute capability 返回不同的默认配置。
3. **保留关闭自动调优的开关**。官方文档给的写法很直接：

```python
if enable_autotune:
    kernel = flashattn(batch, heads, seq_len, dim, is_causal)
else:
    kernel = flashattn(batch, heads, seq_len, dim, is_causal,
                       groups=groups, block_M=128, block_N=128,
                       num_stages=2, threads=256)
```

   第一次调用带自动调优，之后走固定配置。否则每次启动都要重跑一遍搜索。

另外，v0.1.15 的调优器会把结果缓存到 `$TILELANG_CACHE_DIR/autotuner`，缓存键包含
TileLang 版本、函数源码、闭包自由变量、候选列表与编译/测量参数；落盘的产物里有人可读的
`best_config.json` 与 `latency.json`
（`_vendor/tilelang-v0.1.15/tilelang/autotuner/param.py:26-31`）。想强制重搜时用环境变量
`TILELANG_AUTO_TUNING_DISABLE_CACHE=1`。

### 3.9 Carver 与 cost model 的角色

手写笛卡尔积的问题是**它不知道硬件**：`block_M = 256, block_N = 256` 在某些 GPU 上
根本驻留不下，而 `block_K = 16` 在某些机器上会让每轮搬运太碎。Carver 就是来填这个
缺口的。

Carver 的自我定位是“a lightweight framework for generating and ranking tile
configurations … for common GPU, CPU, and accelerator backends”
（`_vendor/tilelang-v0.1.15/tilelang/carver/README.md`）。它做三件事：

1. **建模硬件**：`TileDevice` 上挂着 `smem_cap`（每 block 可用的 shared 字节）、
   `warp_size`、可用张量指令等
   （`_vendor/tilelang-v0.1.15/tilelang/carver/arch/arch_base.py:8`）。
2. **枚举 tile 结构**：`MatmulTemplate` / `GeneralReductionTemplate` /
   `FlashAttentionTemplate` 等把“循环该怎么分块”参数化。
3. **按 cost model 排名并返回 top-k 提示**：`recommend_hints(topk=20)`。

它的 cost model 里就包含本章关心的那类约束。例如 tensor-core policy 里有一句
`if td.smem_cost > (self.arch.smem_cap): ...` ，
共享内存算不过去就退到动态 shared
（`_vendor/tilelang-v0.1.15/tilelang/carver/roller/policy/tensorcore.py:320-335`），
default policy 里还有更直接的一版：超了就丢弃这个候选
（`_vendor/tilelang-v0.1.15/tilelang/carver/roller/policy/default.py:557-560`）。
提示里还会带上 pipeline 的深度建议，`example_gemm_autotune.py` 就是把它读成
`num_stages` 的。

**但 cost model 只负责排序，不负责定案。** 它的作用是：

* 把明显不可行的候选挡在门外（省编译时间）；
* 把搜索排序，让 `early_stop` 之类的优化有机会生效；
* 给出一个“从哪里开始测”的建议集合。

它**不**替代测量。本章 `code/03-autotune-config-space/config_space.py` 就是刻意做这个
演示：它的解析代理在 214 个合法候选上只产出 12 个不同的 cost 值，也就是说模型已经用
完了分辨率，排名里大量的是并列。**并列就是“该去测量”的信号。**

---

## 4. 实现机制

这一节回答“编译器到底做了什么”。所有引用都指向锁定版本
`_vendor/tilelang-v0.1.15/`，行号在写作时逐条核对过。

### 4.1 编译期两步（图 3）

**图 3 · 从 `T.Pipelined` 到多缓冲 IR 的编译期两步** 见
`figures/03-pipeline-compile-flow.md`，它是这一节的路线图。CUDA pipeline 里的位置写得
很直白（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:126-142`）：

```python
# Run pipeline planning and software-pipeline rewriting before layout
# inference so inferred layouts see the final pipelined structure directly.
mod = tilelang.transform.PipelinePlanning()(mod)
mod = tilelang.transform.InjectSoftwarePipeline()(mod)
mod = tilelang.transform.Simplify()(mod)

# Infer memory layouts for fragments and shared memory
mod = tilelang.transform.LayoutInference()(mod)
...
mod = tilelang.transform.LowerTileOp()(mod)
```

三个细节值得注意：

1. **流水在 layout 推导之前**。注释里给了理由：让 layout 推导直接看到最终的流水结构，
   不用先按旧结构推一遍再返工。
2. **两步之间只有注解在传递**。`PipelinePlanning` 不重排语句，它只把循环体规范化
   （摊平嵌套的 `SeqStmt`、剔除不该调度的语句）并往循环上写注解；真正搬语句的是
   `InjectSoftwarePipeline`。这也意味着你可以在两步之间 dump IR，观察“规划结果对
   不对”，第 12 章讲的 IR diff 调试法就靠这个。
3. **`Simplify` 夹在中间**，把规划阶段暴露出来的常量表达式折掉。

### 4.2 `PipelinePlanning`：自动定 stage/order

入口是 `PipelinePlanner::Substitute`，pass 注册在
`_vendor/tilelang-v0.1.15/src/transform/pipeline_planning.cc:1356-1365`
（pass 名 `tl.PipelinePlanning`）。核心逻辑在同一文件的 `:1160-1310`，步骤是：

**第一步，把循环体摊平、去掉不该调度的语句。**

```text
NormalizePipelineBody  →  把循环体规范成语句列表
SeqStmtFlattener       →  把嵌套的 SeqStmt 摊成一维
AnalyzeScheduledStmts  →  标出哪些语句是可重放的标量 Bind（不参与调度）
```

**第二步，给每条语句分类。** `PropagateBufferProducersForCopy` 做一件容易被忽略的事：
它把“为搬运做准备的写操作”和搬运绑在一起。源码里的例子是：

```python
fill(shared, 0)        # 写 shared
copy(global, shared)   # 可能依赖刚填进去的值
```

如果搬运被挪到生产者一侧，这个 `fill` 必须**跟着一起挪**，否则搬运可能读到未初始化或
错误的值。代码注释把这个传播称为“buffer-level backward dependency closure”。

然后是 `AnalyzeCopyLastUse`：找出每个搬运的**最后一个消费者**。这一步决定了搬运该排到
哪里，“放在它最后一个消费者之前”通常比“放在循环体最前面”更能缩短数据在片上的
存活时间。

**第三步，分配 stage 与 order。** 主循环的逻辑是（源码注释的复述）：

```text
for 每条语句 pinfo:
    # 搬运类语句先跳过，它们要跟着自己的最后一个消费者排
    if pinfo.IsFirstStage() and pinfo.IsLastUseStmtIndexValid():
        continue
    pinfo.order = order_idx++          # 消费者按源码顺序拿 order
    pinfo.stage = num_stages           # 先放到「最后一格」，下面再回落
    for 每条以当前语句为最后一个消费者的搬运 p1:
        p1.order = order_idx++
        p1.stage = 0                   # 搬运固定在最前面一格
```

接着有一段“收尾整形”：如果所有搬运的 `order` 都排在了非搬运语句之后，说明这一轮
循环是从“先算后搬”开始的，于是把整个 order 数组循环右移，并把非搬运语句的
`stage` 减 1。**这一步就是“深度 = num_stages”的来历**：

```text
典型的两搬一算 GEMM 体，num_stages = 3：

  规划中途：  copy A order=1 stage=0
             copy B order=2 stage=0
             gemm   order=0 stage=3          ← 先被放到第 3 格

  收尾整形后：copy A order=0 stage=0
             copy B order=1 stage=0
             gemm   order=2 stage=2          ← 减 1，深度 = max(stage)+1 = 3
```

**第四步，写注解。** 规划的主要输出是下面四个键（还有若干给特定后端或特定优化的
辅助键，例如可重放 `Bind` 的掩码与逐语句的异步生产者信息）：

```text
software_pipeline_stage        # 每条被调度语句的 stage
software_pipeline_order        # 每条被调度语句的 order
tl_pipelined_num_stages        # 原始深度，供下游用
software_pipeline_tma_copies   # 逐语句的 TMA 可用性
```

源码在写 `tl_pipelined_num_stages` 的地方留了一段注释，解释为什么不保留原来的
`num_stages`：

> We intentionally do NOT keep the legacy key `num_stages` here because multiple
> downstream passes (e.g. internal buffer versioning / warp specialization) treat it as
> an active pipeline marker and do not support nested pipelines.
> 来源：`_vendor/tilelang-v0.1.15/src/transform/pipeline_planning.cc:1287-1293`

**这也连着一条更普遍的规则：流水不能嵌套。** 上游对嵌套循环的态度写在
`NestedLoopChecker` 的文档里，“Nested loops is an annoying problem in tilelang or
other polyhedral-style compilers. It contains many corner cases and undefined
behaviours”，并且明确规定 `T.Pipelined` **不能**嵌在 `T.Parallel` 里
（`_vendor/tilelang-v0.1.15/tilelang/analysis/nested_loop_checker.py:65-100`）。
那个文件里判断“这是不是一个流水循环”用的正是 2.1 节提到的三个注解键
（同文件 `:12-15`）。昇腾侧的文档把这条规矩写得更直白：

> `T.Pipelined` **does not support nesting**. You cannot place a `T.Pipelined` loop
> inside another `T.Pipelined` loop.
> 来源：`../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/README.md`
> 第 521 行

所以：需要嵌套的流水结构时，先把它拆成两个 kernel，或者重新组织循环，而不要把两个
`T.Pipelined` 摞起来。

### 4.3 `InjectSoftwarePipeline`：prologue / body / epilogue

pass 注册在 `_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:4033-4048`
（pass 名 `tl.InjectSoftwarePipeline`）。真正干活的是 `PipelineRewriter::BuildPipeline`
（同文件 `:1208-1265`），它做四件事：

**（1）算每个 buffer 需要几份版本**（见 4.4 节），需要多版本的直接改造分配。

**（2）按 `order` 排序**所有被调度的语句，得到发射序列。

**（3）生成三段。** 这是软件流水的标准形态：

```cpp
// _vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1236-1246
Stmt prologue = EmitImpl(pipeline_loop_->min,
                         pipeline_loop_->min + max_stage_, true, true);
Stmt body     = EmitImpl(pipeline_loop_->min + max_stage_,
                         pipeline_loop_->min + pipeline_loop_->extent, false, false);
Stmt epilogue = EmitImpl(pipeline_loop_->min + pipeline_loop_->extent,
                         pipeline_loop_->min + pipeline_loop_->extent + max_stage_,
                         true, true);
```

`max_stage_` 是注解里最大的 stage。三段的意思就是 1.4 节讲的：

```text
prologue  前 max_stage 轮：只做前面的 stage，把流水填满
body      中间所有轮：所有 stage 都在跑，这是稳态，也是性能所在
epilogue  最后 max_stage 轮：前面的 stage 已经没活干，只剩下收尾
```

**（4）收紧等待。** 生成完三段之后还会跑
`AsyncPipelineLoopWaitRelaxer` 与 `RelaxTrailingConsumerWaits`，把能省的等待去掉，
并把循环内多轮叠加的 commit/wait 收敛到安全的最少数量。最后 `ConvertSSA` 把重写产生的
大量变量定义规范化（`:4038`）。

### 4.4 多缓冲物化：版本数怎么算

核心函数是 `PipelineRewriter::ComputeBufferVersions`
（`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1474-1544`）：

```cpp
// use - def + 1 是这个 buffer 需要的版本数上界
int num_versions = buffer_info.use - buffer_info.def + 1;
if (num_versions >= 2) {
  // 特例：上界恰好是 2 时，只有真的存在
  // order(写) < order(读) 且 stage(写) < stage(读) 且访问区域重叠
  // 的一对语句，才需要双缓冲；否则把版本数减回去
  ...
  if (!need_multi_version) num_versions--;
}
```

`def` 是 buffer 被写入的 stage，`use` 是它最后被读的 stage。对 4.2 节那个“copy 在
stage 0、gemm 在 stage `num_stages-1`”的典型结果：`use - def + 1 = num_stages`，
于是 A/B tile 各要 `num_stages` 份。

关于“深度”还有一点值得单独记下来：源码里明确区分了两个数（
`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:3628-3638`）：

$$
\begin{aligned}
\text{实际流水深度（buffer 份数）} &= \text{num\_stages},\\
\text{软件流水 stage 数} &= \max(\text{stage})+1.
\end{aligned}
$$

注释原话是“Use the actual pipeline depth (number of buffer copies) for barrier sizing,
not the SW pipeline stage count (max_stage + 1)”。在“copy 在 stage 0、gemm 在
stage `num_stages-1`”这个常见形态下两者相等（都是 `num_stages`），但它们是两个概念：
前者说的是**内存里放几份**，后者说的是**循环被切成几段**。手写 stage/order 时如果
只盯着 `max(stage)+1`，就会忽略 `num_stages` 这个更权威的深度来源。

需要多版本时，形状改写由 `RewriteAllocBuffer` 完成
（同文件 `:1546-1560`）：

```cpp
Buffer RewriteAllocBuffer(const Buffer &buffer, int num_versions) {
  ObjectPtr<BufferNode> new_buffer = make_object<BufferNode>(*(buffer.get()));
  new_buffer->shape.insert(new_buffer->shape.begin(), PrimExpr(num_versions));
  ...
}
```

新维度插在 `shape` 的**最前面**。这就是图 2 里
`(block_M, block_K) → (num_stages, block_M, block_K)` 的来源，也是你在生成码里看到
`A_shared[ko % 3][...]` 这种下标的原因。

还有一条容易被忽略的注释：如果 buffer 的定义 stage 是 `-1`（也就是它**不**在这个流水
体内被写），函数直接返回 1，注释是“Keep the original number of versions as buffers
defined outside the software pipeline should not be mutated”。**只读的输入 buffer
不会被复制**。这一点决定了容量估算要不要把它算进去。

### 4.5 多缓冲与片上容量的硬约束：stage 数不是越大越好

把 4.4 节的结果乘上 dtype 大小，就得到本章最重要的一个估算式：

$$
\begin{aligned}
S_A &= B_MB_K\,s,\\
S_B &= B_KB_N\,s,\\
S_{\text{pipe}} &\le N_{\text{stages}}(S_A+S_B),
\end{aligned}
$$

其中 $B_M,B_N,B_K$ 是三个 tile 维度，$s$ 是 dtype 的字节数，
$N_{\text{stages}}$ 是 `num_stages`。

用 `code/02-shared-memory-budget/smem_budget.py` 算两个具体例子（作者在本机实际运行过）：

```text
tile: A[128,32] + B[32,128] @ fp16 (2 B/elem)
per stage (A tile + B tile): 16384 B
largest stage count that fits: 14        # 预算 228 KiB

tile: A[256,64] + B[64,256] @ fp16 (2 B/elem)
per stage (A tile + B tile): 65536 B
largest stage count that fits: 3

+------------+----------------------------------------------------------+
| num_stages | upper-bound shared footprint                             |
+------------+----------------------------------------------------------+
| 2          |    131072 B    128.0 KiB   fits (56% of budget)          |
| 3          |    196608 B    192.0 KiB   fits (84% of budget)          |
| 4          |    262144 B    256.0 KiB   does NOT fit (112% of budget) |
+------------+----------------------------------------------------------+
```

同一个 `num_stages=4`，在 `128×128×32` 的 tile 上只用了 64 KiB（宽裕），在
`256×256×64` 的 tile 上要 256 KiB（**直接放不下**）。这就是“stage 数不是越大越好”的
全部含义：**能加多少 stage，是 tile 尺寸的函数，不是常量。**

判断当前配置放不放得下，按这个顺序做：

1. **数清楚哪些 buffer 会被多版本化。** 只有“在本流水体内被写”的 buffer 才复制
   （4.4 节那条 `def == -1` 的规则）；只读输入不算。
2. **按上面的公式算上界**，用 `code/02` 的工具算，别心算。
3. **和真实预算比，并留余量。** 上界是“不做任何复用时”的占用；TileLang 会合并
   shared 分配并复用 live range 不重叠的存储
   （`_vendor/tilelang-v0.1.15/src/transform/merge_shared_memory_allocations.cc:1576-1660`），
   所以实际可能更小。反过来，累加器 fragment 走寄存器、barrier 另有开销，别把预算
   用满。
4. **加 stage 之前先问：1.3 节的判据说还需要吗？** 如果
   `num_stages ≥ 1 + ceil(latency / compute)` 已经满足，再加只会增加占用。

超过预算时的表现是**编译或启动阶段直接失败**，不是“跑得慢一点”。这是好事：
它是一条**硬**约束，你没有调优到一半才发现的机会。`code/04` 里就留了一个故意超预算的
命令（`--block-m 256 --block-n 256 --block-k 64 --stages 4`），用来在有卡的机器上复现
这条边界。

### 4.6 昇腾侧：`T.Pipelined` 与 Z3 自动调度的关系

昇腾侧的接口同样是 `T.Pipelined`。`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md`
的官方 GEMM+ReLU 例子里就是 `for k in T.Pipelined(K // block_K, num_stages=2):`，
并且明确写着“`T.Pipelined` uses a two-stage pipeline for the reduction loop”。

Ascend 适配器仓库的 README 把 `T.Pipelined` 在这边的职责说得更完整：

> enables automatic pipeline scheduling to achieve intra-core computation and data
> movement overlap, as well as inter-core pipeline overlap between Cube and Vector units.
> 来源：`../tilelang-corpus/07-tilelang-ascend-adapter/raw/tilelang-ascend/README.md`
> 第 512 行

也就是说，同一句 `T.Pipelined` 在昇腾上还要负责**核间**流水（Cube 与 Vector 之间）。
同一份文档给了两种写法：把 Cube 与 Vector 操作放进**同一个** `T.Pipelined` 里做核间
流水，或者各自用**独立**的 `T.Pipelined` 只做核内流水（同文件第 511-535 行）。
这一点是 NVIDIA 侧没有的概念，第 25 章会展开。

不同的是它背后那把引擎。昇腾的 pass 序列是
（`_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:107-115`）：

```python
mod = ascend_transform.EstimateLatency()(mod)
mod = ascend_transform.AutoSchedule()(mod)
mod = ascend_transform.AssignCore()(mod)
mod = ascend_transform.PrepareMultiBuffer()(mod)
mod = ascend_transform.ResolveCore()(mod)
mod = ascend_transform.InsertSync()(mod)
mod = ascend_transform.MaterializeMultiBuffer()(mod)
```

其中 `AutoSchedule` 的调度器是一个 **Z3 求解器**。它的输入是一张任务表，每个任务带：

```text
latencies            每个任务的执行延迟
iis                  每个任务的 initiation interval
resource_flags       任务占用的硬件 pipe 掩码
                     （MTE1=1, MTE2=2, MTE3=4, Cube=8, Vector=16,
                       Fixpipe=32, Scalar=64）
data_deps            有向 (i, j, latency)：j 必须在 i 之后至少 latency 才开始
resource_deps        不能并发发射的任务对
owner_exclusion_deps 共享存储的所有权冲突：两者不能同时活着
pipe_order_deps      共用同一条硬件 pipe、必须保持源码顺序的任务对
```

输出是**每个任务的开始时间**（`_vendor/tilelang-v0.1.15/tilelang/ascend/transform/z3_scheduler.py:67-140`）。
C++ 侧通过 `tl.transform.z3_schedule_python` 这个 FFI 全局函数调用它，因为 Z3 不是线程
安全的，Python 侧还专门加了一把锁（同文件 `:228-256`）。

把它和 NVIDIA 侧放在一起看，差别就很清楚了：

| | NVIDIA（v0.1.15） | 昇腾（v0.1.15） |
| --- | --- | --- |
| 调度形式 | 按数据流给语句打 stage/order 注解 | 把语句变成带延迟与资源约束的任务，交给 SMT 求解 |
| 谁决定缓冲区份数 | `num_stages` 注解驱动的版本数计算 | 求解结果 + `PrepareMultiBuffer` / `MaterializeMultiBuffer` |
| 同步 | 异步拷贝自带的等待 + `ThreadSync` | `InsertSync` 统一插入成对 flag |
| 你写的代码 | 一样是 `T.Pipelined` | 一样是 `T.Pipelined` |

**本章的结论只需要两句**：接口收敛在 `T.Pipelined`，机制完全不同；想知道昇腾侧
`EstimateLatency` 怎么估、Z3 的约束怎么写、求解失败时回退到什么，去看第 23 章。

---

## 5. 进阶与陷阱

### 5.1 陷阱一：stage 加太多，直接撑爆片上存储

这是本章最硬的一条约束，也是最容易犯的错。典型症状是：调优空间里放了
`num_stages = 1..8`，每个 tile 尺寸都试，结果一半候选在编译期或启动期直接报错，
浪费大量编译时间。

**为什么会发生**：4.4 节的版本数计算是“`num_stages` 越大、版本数越多”，
而版本数最终乘到字节数上（4.5 节）。`num_stages` 从 3 加到 4 不是“多花一点”，
而是**多一整份 A tile 加一整份 B tile**。

**怎么避免**：

* 在定义搜索空间时就按 4.5 节的公式做**预剪枝**，`code/05` 里的 `grid_configs`
  就是示例，它把 4096³ 上的 216 个候选择到 214 个；
* 把 tile 尺寸和深度**一起**考虑：`256×256×64` 的 tile 最多只能配 3 深，
  而 `128×128×32` 可以配到 14 深；
* 记住上界是上界：`MergeSharedMemoryAllocations` 会复用 live range 不重叠的存储
  （`_vendor/tilelang-v0.1.15/src/transform/merge_shared_memory_allocations.cc:1576-1660`），
  所以实际占用可能比乘积小；但如果按上界都不放不下，那就一定放不下。

**昇腾侧同理，只是资源换了名字**：你要看的不再是 shared memory 一个数，而是 L1、UB、
L0A/L0B/L0C 各自的大小，以及它们之间搬运的粒度。约束的方向一样，细节在第 25 章。

### 5.2 陷阱二：把流水当成了“自动加快”，忘了它的前提

流水不是魔法，它成立需要两个前提，缺一个就会得到错误结果或没有收益。

**前提一：搬运与计算之间真的没有数据依赖。** 第 `ko+1` 轮的搬运要能和第 `ko` 轮的
计算并行，前提是第 `ko+1` 轮的搬运**不依赖**第 `ko` 轮计算的结果。GEMM 满足这一点
（A、B 都来自全局内存，与累加器无关）。但如果你写的是“上一轮的输出作为下一轮的输入”
的循环。例如某些 scan / 累积型算子。那就不满足，编译器要么拒绝流水，要么给你一个
依赖链串起来的伪流水。**这类循环需要换算法，不是换 num_stages。**

**前提二：同步必须跟着多缓冲一起变。** 一份 buffer 时，同步靠顺序就能表达；
多缓冲之后，正确的同步变成：

```text
写 slot s 之前：等「上一次读 slot s 的消费者」读完
读 slot s 之前：等「写 slot s 的生产者」写完
```

这就是图 2 里“写 ← 覆盖”那一格的含义。编译期负责插这些同步
（NVIDIA 侧在 `InjectSoftwarePipeline` 内部生成异步拷贝及其等待，昇腾侧由 `InsertSync`
统一插成对的 flag），但前提是你**不要手动破坏它**：不要在流水体内手写
`T.sync_threads` 之类的原语去“帮忙”，也不要对同一块 buffer 做编译期看不到的别名访问。

判断一个循环能不能流水，最省事的自检是：**把循环体单独看，如果第 `ko+1` 轮完全不
依赖第 `ko` 轮的任何结果，那就可以。**

### 5.3 陷阱三：搜索空间定义不当，调优等于没调

这是自动调优里最常见的浪费。三种坏空间的**症状**都可以在动手之前看出来，
`code/03-autotune-config-space/config_space.py` 把它们做成了可复现的输出：

```text
mode = all-infeasible
configurations requested: 48
configurations legal:     0
  rejected (shared memory over budget): 48
=> This space is wasted: every candidate is infeasible.

mode = degenerate
configurations requested: 1
configurations legal:     1
distinct cost values:     1
=> Only one legal configuration: tuning cannot rank anything.
```

| 坏空间的形态 | 症状 | 怎么改 |
| --- | --- | --- |
| 全部不可行 | `legal = 0`，整轮调优只产出编译错误 | 按 4.5 节做容量预剪枝；检查 tile 是否整除问题尺寸 |
| 只剩一两个可行 | 调优耗时与“什么都不调”一样，甚至更慢 | 放宽 tile 尺寸/深度的取值范围，但保持可行 |
| 可行但分不开 | cost model 的取值大量并列，排名是噪声 | 加更多有物理意义的维度（深度、线程数），或直接上真机测量 |

还有一条**比上面三条更隐蔽**的：**搜索空间里的参数没有被内核真正用上**。
`AutoTuner.run` 会显式检查这件事。如果候选 dict 里有内核参数列表里没有的键，它会直接
报错：

```text
ValueError: Unused keys in config: {...}
```

（`_vendor/tilelang-v0.1.15/tilelang/autotuner/tuner.py:904-913`）。报错是好事；
真正危险的是**键存在但不起作用**。例如你把 `num_stages` 传进了工厂函数，但内核里写的
是 `T.Pipelined(n)` 而不是 `T.Pipelined(n, num_stages=num_stages)`。这时所有候选的
性能完全相同，调优会“成功”地给你一个毫无意义的赢家。**改完搜索空间后，务必确认
每个被调参数在生成的 kernel 里真的出现了。**

### 5.4 其它常见坑

**（1）把 `num_stages=1` 与 `num_stages=0` 当成一回事。** 不是。docstring 说
`num_stages` 是“the max number of buffer used between pipeline producers and
consumers”，所以 `1` 意味着“只留一份缓冲”，性能上等于不做重叠；而 `0` 是
**关闭流水**：注解根本不会写进 IR（见 2.1 节），下游根本不把这里当流水循环。
两者在自动调优里语义不同，别混着放。另外如 4.4 节所述，源码把“缓冲区份数”
（`num_stages`）与“流水级数”（`max(stage)+1`）分开处理，`num_stages` 才是权威的那一个。

**（2）给 `order`/`stage` 数错了对象。** 数组只数有副作用的语句。把可重放的标量
`Bind` 数进去（3.4 节）是最常见的错法；旧代码里这种写法被兼容并忽略，但新代码会让你
误以为注解生效了。

**（3）忘了 `order` 必须唯一。** `ValidatePipelineBody` 会直接报
“Two statements in the software pipeline cannot have the same order”。
手动写 `[0, 0, 1]` 这种 `order` 是错的；想表达“同一级里这两条并行”用的是**相同的
`stage` + 不同的 `order`**。

**（4）以为加了流水就等于“延迟变小”。** 如 2.4 节所述，流水把每一轮的成本从
`latency + compute` 压到 `max(compute, issue, (latency + compute) / depth)`，
但**单次操作的完成时间没变**。如果你的指标是“单个请求的端到端延迟”，需要看的是整个
kernel 的填充与排空占比，而不是只看稳态。

**（5）在循环太短时硬上流水。** 填充的绝对代价与轮数无关。轮数只有两三轮的循环，
加深度几乎必然净亏（多占片上、收益被填充吃掉）。

**（6）把 `latency` 当固定常量来推公式。** 1.3 节的 `1 + ceil(latency / compute)`
是一个**模型判据**，不是硬件的公开规格。真实机器上 `latency`、`issue`、`compute`
都随 tile 尺寸、dtype、访存模式变化。这个判据的用途是给你一个“从多深开始试”的
起点，不是终点。

---

## 6. 本章小结

把这一页当作复习卡片。

**为什么要有流水。** 搬运和计算由不同的硬件负责，天生可以同时工作。一份 buffer 的
循环里，它们必须互相等：每轮 = 搬运 + 计算。多准备几份 buffer 就能让第 `ko+1` 轮的
搬运与第 `ko` 轮的计算重叠，把每轮成本压到 `max(issue, latency/depth, compute)`。

**三个词。** `stage` 是语句属于第几级（越小越早），`order` 是同级的发射顺序，
`num_stages` 是缓冲区份数（深度）。深度要么直接给 `num_stages`，要么由
`max(stage) + 1` 推出；**不要两个一起给**。

**流水优化的是吞吐，不是延迟。** 单次操作的耗时不变；变的是单位时间能完成多少轮。
所以轮数多的长 reduction 循环收益最大，轮数少的短循环不值得。

**编译期两步。**

```text
PipelinePlanning        按数据流自动给每条语句定 stage/order，只写注解，不动语句
InjectSoftwarePipeline  按 order 重排、生成 prologue/body/epilogue、
                        按 use - def + 1 复制 buffer、插入异步拷贝与同步
```

`ComputeBufferVersions` 给被跨 stage 读写的 buffer 算出份数；`RewriteAllocBuffer`
在 shape 最前面插入版本维。只读的、或只在同一 stage 内被写读的 buffer 不会被复制。

**硬约束。** 流水相关的片上占用上界是
`num_stages × (block_M×block_K + block_K×block_N) × sizeof(dtype)`。
超了就是编译/启动失败，不是变慢。加 stage 之前先用 `code/02` 算一遍。

**自动调优。** 两条入口：`@tilelang.autotune(configs=...)` 与
`AutoTuner.from_kernel(...).set_compile_args(...).set_profile_args(...).run()`。
候选列表可以手写笛卡尔积，也可以由 carver 的 `MatmulTemplate.recommend_hints(topk)`
生成。结果里**只有 `config` 值得抄回代码**；把它写成函数默认值，并保留一个关闭调优的
开关。carver 的 cost model 负责**剪枝与排序**，不负责定案。

**两台机器。** 接口都是 `T.Pipelined`，机制不同：NVIDIA 侧是按数据流打注解再改写，
昇腾侧是把任务、延迟、资源约束交给 Z3 求解开始时间。细节在第 23 章。

**三个最容易踩的坑。** ①stage 太多撑爆片上存储；②忘了流水的两个前提
（无跨轮依赖、同步随多缓冲一起变）；③搜索空间定义不当，全部不可行、只剩一个可行、
或者参数根本没被内核用上。

## 延伸阅读

下面每条都指向本章 `references.md` 里的条目 id。

* **官方文档原文**：`R1`（`software_pipeline.md`，`order`/`stage`/replayable `Bind`
  的权威说明）、`R2`（`tutorials/auto_tuning.md`，GEMM 自动调优的入门版）、
  `R3`（`programming_guides/autotuning.md`，两条接口、输入供给、缓存与调参建议）。
* **锁定源码**：`S1`（`T.Pipelined` 的签名与 docstring）、`S2`（`order`/`stage` 如何
  变成循环注解）、`S3`/`S4`（`PipelinePlanning` 的实现与注册）、`S5`
  （`InjectSoftwarePipeline` 的 prologue/body/epilogue）、`S6`（`ComputeBufferVersions`
  与版本维的插入）、`S7`（注解的校验规则）、`S8`（pass 注册与 `ConvertSSA`）、
  `S9`（`GetPipelineNumStages`）、`S10`（CUDA 侧 pass 顺序）、`S11`（shared 分配的
  合并与复用）、`S12`（嵌套规则）。
* **自动调优源码**：`S13`（`autotune` 装饰器）、`S14`（`AutoTuner`）、`S15`
  （`AutotuneResult`）、`S16`（carver 的定位与模板清单）、`S17`（`TileDevice` 的
  `smem_cap` 等硬件建模字段）、`S18`（cost model 的容量约束）、`S19`、`S20`
  （两个官方 GEMM 示例）。
* **昇腾对照**：`S21`（昇腾 pass 序列）、`S22`/`S23`（Z3 调度器与其 FFI 包装）、
  `S24`（延迟估计）、`S25`（昇腾后端 README）、`R6`（Ascend 适配器 README 的
  `T.Pipelined` 一节）。
* **量化背景**：`R4`（H100 的异步执行与 256 KB shared+L1 / 228 KB shared）、
  `R5`（访存与算术指令的延迟量级）。
* **下一章**：第 12 章讲怎么把这些结论落到调试与性能分析上（`lower_trace`、
  `pass_diff`、Nsight Compute / msProf）；第 23 章展开昇腾的延迟估计与 Z3 求解；
  第 28 章把“tile 尺寸与深度怎么选”上升成方法论。
