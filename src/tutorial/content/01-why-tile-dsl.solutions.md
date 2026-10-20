# 第 01 章 习题答案

与 `exercises.md` 的编号一一对应。每题给出**解题思路**、**参考答案**与**常见错误**。
凡涉及性能数字的地方都标注了来源；本章没有任何需要在本机实测才能得出的结论。

---

## E1 概念题 ★

**解题思路**：从“算法正确 ≠ 运行快”这个区分入手，再落到“时间花在搬运还是计算”。

**参考答案**：

1. 因为绝大多数 AI 算子的数学公式早已固定，性能差异不来自公式，而来自**数据如何在存储
   层级之间流动**。同一个 `softmax`，写成四遍或两遍，运算量几乎一样，但访存量差一倍以上。
2. 至少两类动作：**搬运**（把数据从显存搬到片上/寄存器，或搬回去）和**计算**（加、乘、
   比较、归约）。例子：`MatMul` 的主要问题是反复从显存读取 `A`、`B`（搬运过多）；
   `softmax` 的主要问题是访存量本身就无法再少（memory-bound）。
3. `x` 被读取的次数直接决定时间：四遍实现里 `x` 至少被读 2 次（第 1、2 遍），`e` 被写
   1 次又读 2 次（第 3、4 遍）。而理论上只读 `x` 一遍、只写 `y` 一遍就够了。以
   $4096\times4096$ fp16 为例，理论下界是 32 MB 读 + 32 MB 写，四遍实现可能接近三倍。

**常见错误**：把“慢”解释成“GPU 不够快”或“PyTorch 没优化”。PyTorch 调用的就是高度
优化的库；问题在于算子本身的访存结构与是否融合。

---

## E2 概念题 ★

**解题思路**：先写定义式，再代入数字，最后按“谁先到瓶颈”判类。

**参考答案**：

1. `算术强度 AI = FLOP / Bytes`；`机器平衡点 = 峰值算力 / 峰值带宽`（FLOP/Byte）。
2. `1000e12 / 3.0e12 ≈ 333 FLOP/Byte`。来源：峰值与带宽取自
   `../tilelang-corpus/05-nvidia-backend/raw/nvidia-blog-hopper-architecture-in-depth.md`
   （H100 SXM5，厂商公开资料；带宽原文为“over 3 TB/sec”，FP16 密集 1000 TFLOPS 出自
   该文的 preliminary specs 表）。**这个比值是本书的推导，不是厂商给出的数字**，且厂商
   规格为未实测口径。
3. *`y = max(x, 0)`*：运算量约 $N$ 次比较，访存量 4 MB 读 + 4 MB 写（fp16，4096²）；
   AI 远低于 333，**memory-bound**。
   *`C = A B`（4096³）*：运算量 $2MNK = 1.37\times10^{11}$ FLOP；理论最小访存量
   $(MK+KN+MN)\times2 = 96$ MB；AI ≈ 1400 FLOP/Byte，高于 333，**compute-bound**。

**常见错误**：把 `MatMul` 算成 memory-bound（忘了它为了省访存做了分块，且分块之后计算量
远大于访存量）；或者忘了尺寸对结论的影响（E9 第 2 问就是在考察这一点）。

---

## E3 概念题 ★★

**解题思路**：把三个阶段的“新增管理对象”逐一列出，再说明“组合爆炸”。

**参考答案**：

1. 阶段一（SIMT）：新增的是“线程分工 + 数据在 global/shared 之间怎么搬”；阶段二
   （张量核）：新增的是“fragment 布局约束”（数据必须以特定形状/布局喂给矩阵单元）；
   阶段三（异步 + 专用存储）：新增的是“多流水之间的同步协议”“多缓冲的容量约束”
   “跨核/跨 SM 的依赖关系”。
2. 复杂度是组合的：要正确完成一次计算，必须同时满足“存储位置正确”“布局合法”“同步
   完整”“流水重叠可行”“算力单元被喂饱”。每一项都有多个取值，且它们互相约束。例如
   多开一份 buffer 能改善流水，但会挤占片容量；改一个 tile 尺寸会同时改变 layout、寄存器
   压力和同步点。可行解的数量随维度相乘，而不是相加。
3. 因为昇腾 950 后端本身就同时提供 SIMT 与 SIMD 两种向量执行方式（`T.SimtVF` 与
   `T.SimdVF`，见 `_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py:107` 与
   `:179`），而现代 NVIDIA GPU 也早已把异步搬运、专用矩阵单元、warp specialization 塞进
   同一套执行体系。两边都在往对方的方向长，“GPU=SIMT、NPU=SIMD”不再是有效概括。依据：
   `../tilelang-corpus/09-zhihu-da-zhi-ruo-yu/raw/zhihu-dazhiruoyu-deepseek-ascend.md`
   （社区文章，非官方）。

**常见错误**：把阶段三的“异步”理解成“更快的搬运”。异步的关键不是更快，而是**允许
重叠**，代价是引入必须被证明正确的完成协议。

---

## E4 概念题 ★★

**解题思路**：用“人负责什么 / 编译器负责什么”这条线统一三代。

**参考答案**：

1. CUDA 把**全部硬件细节**交给程序员：线程分工、搬运、布局、同步、流水。它换来的通用性
   体现在“任何算法都能用这套模型表达”，代价是这些细节必须手工维护，且随硬件代际变化。
2. Triton 用“tile 是第一等公民”换来了：程序更短、更容易跨代复用，编译器负责 tile 内部
   的线程分配与访存合并。代价写在原论文里，“The benefits of Triton come at the cost of
   increased programming efforts.”以及它无法像 Halide/TVM 那样把调度手工指定出来。
   来源：`../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md`，第 3 页。
3. TileLang 多保留了**“会影响合法性与性能的资源信息”**：tile 的形状、放在哪一级存储、
   在多条流水之间怎么流动。它要保留这类信息，是因为这些信息**编译期推不出来**，只有
   算法作者知道“这个中间结果必须留在片上”。TileLang 的做法是：把人该负责的数据流留下，
   把可以从数据流推导出来的调度（线程绑定、layout、tensorize、pipeline）交给编译器。
   依据：`../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577.md`（摘要）。

**常见错误**：认为三者是“越来越先进”的线性替代关系。实际是**设计取舍**：Triton 用
简洁换走可控性和硬件表达力；TileLang 再把一部分可控性买回来。

---

## E5 概念题 ★

**解题思路**：先复述定义，再对每个要素做“少一个会怎样”的反事实推理。

**参考答案**：

tile = 逻辑张量上的一个**定形子区域**，同时携带 (1) 形状、(2) 存储位置、(3) 归属
（归属的具体形态就是物理布局 layout）。措辞与第 3 章 §2.4 一致。

1. 缺**形状**：无法静态确定要搬多少数据、分配多少片上空间，编译期就不能做多缓冲与
   布局推导。
2. 缺**存储位置**：编译器只能猜数据在哪一级存储上，猜错就会引入隐式的搬运与同步；
   在昇腾上这类信息（L1 / L0C / UB）必须在前端说清。
3. 缺**归属/物理布局**：逻辑上正确的 `[M, K]` 到硬件矩阵单元那里可能不合法（fractal/对齐），
   编译器只能插入额外的转换，或者直接性能断崖。

“循环分块”只是一种循环变换：它改变了访问顺序，但没有表达“这块数据被放在哪一级存储、
以什么物理布局被消费”。所以分块是 tile 抽象的**必要条件，不是充分条件**。

**常见错误**：把 tile 理解成“尺寸是常量的数组”。常量形状只是第 1 条。

---

## E6 代码阅读题 ★★

**解题思路**：逐行归类到“数据流”或“调度”，再解释 `T.Kernel` 的网格语义。

**参考答案**：

1. 每次迭代负责计算输出 `C` 的一个 `block_M × block_N` 分块。`bx` 是网格 x 维（跨 N
   方向的分块编号），`by` 是 y 维（跨 M 方向）。依据：`T.Kernel(T.ceildiv(N, block_N),
   T.ceildiv(M, block_M))` 的参数顺序与 `A[by * block_M, ...]`、`B[..., bx * block_N]`
   的下标一致；`T.Kernel` 的 docstring 明确说它构造的是“a grid of tile programs”
   （`_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:431-455`）。
2. `A_shared`、`B_shared` 在片上共享存储（`T.alloc_shared`）；`C_local` 在累加器/寄存器
   级别（`T.alloc_fragment`）。判断依据就是分配函数本身，TileLang 里存储作用域由
   `T.alloc_*` 显式决定（`_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:49,81`）。
3. *数据流*：`T.copy(A[...], A_shared)`、`T.copy(B[...], B_shared)`、`T.gemm(...)`、
   最后一条 `T.copy(C_local, C[...])`。
   *调度决策*：`T.Kernel(..., threads=128)` 的线程数、`T.Pipelined(..., num_stages=3)`
   的流水深度、`T.alloc_*` 的位置选择（严格说位置选择属于数据流，但它是“可被覆盖的
   调度决策”，见第 3 章）。
4. 它想让**搬运（`T.copy`）**与**计算（`T.gemm`）**重叠：第 `ko` 轮在算的时候，第
   `ko+1` 轮的数据已经在搬。物理代价是片上空间要多开几份 buffer，份数受片上容量限制。
5. `T.alloc_fragment` 是给单线程可见的寄存器切片用的，不能被整个 block 共享。把 `A_shared`
   换成 fragment 会让“所有线程共享同一块 A”这个语义消失，`T.gemm` 的输入也就不再是
   共享 tile。**结论：存储位置不是性能选项，而是语义的一部分。**

**常见错误**：认为 `T.copy` 是“并行 for 循环的语法糖”。它在语义上是“一次 tile 级搬运”，
降级成 `cp.async`/TMA/SIMT 循环是编译期的事（第 9 章）。

---

## E7 代码阅读题 ★★

**解题思路**：把昇腾的层级背下来，再用 `dual_copy` 的 docstring 解释“为什么需要专用原语”。

**参考答案**：

1. 出现的存储作用域：`shared.l1`（`A_l1`、`B_l1`）、`shared.l0c`（`C_l0c`）、
   `shared.dyn`（`C_ub`，即 UB）。按昇腾层级从近到远：L0C（Cube 累加器）→ L1 → UB →
   GM（全局内存）。依据：`tilelang/ascend/language/allocate.py:26-63` 里各 `alloc_*` 的
   默认 scope。
2. 因为 `C_l0c → C_ub` 不是普通搬运：L0C 是 Cube 的累加器，要把它交给 Vector 单元做
   ReLU，必须走“M/N 对半拆分到两个 AIV”的专用路径。`dual_copy` 的 docstring 写明它
   支持 `L0C→UB` 的硬件事务（“lowered as a hardware dual-destination copy”），并且要求
   即使 M-split 也要保证静态形状为偶数。依据：
   `_vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py:53-100`。
3. `T.Parallel` 在这里描述的是**逐元素并行域**（对 `block_M//2 × block_N` 的每个元素做一次
   `max(..., 0)`），由 `T.SimtVF` 把它映射到 128 个 SIMT 线程上。它和 CUDA 的
   `threadIdx` **不是**同一回事：程序员没有指定“哪个线程处理哪个元素”，映射由编译器
   完成（第 8 章）。
4. 这些同步被编译器接走了。`tilelang/ascend/README.md:144-152` 的对照表里，Synchronization
   一栏写的是“None (Just write operations in order!)”，编译器“Infers pipeline and
   Cube/Vector dependencies; inserts paired flags”。
5. `clear_accum=(k == 0)` 表示只在 K 方向第一轮把累加器清零，后续轮次在同一个 `C_l0c`
   上继续累加。去掉它（或恒为 True/False）要么每轮都清零导致结果错，要么第一轮没有初值。

**常见错误**：把 `T.alloc_shared` 在昇腾上理解成“就是 C 语言的 shared array”。在昇腾
语境里，`alloc_shared` 默认落在 UB，而 L1 要用 `T.alloc_l1`，同一个函数名在两台机器上
指向不同的物理存储。

---

## E8 代码阅读题 ★★

**解题思路**：从 docstring 原文出发，再把它和“跨后端”这一点连起来。

**参考答案**：

1. 原文是“Construct a kernel launch frame: **a grid of tile programs**.”以及“This is the
   target-neutral launch: the part every backend shares.”请以实际文件为准：
   `_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:431-455`。
2. 意思是：**同一个 tile 前端语义，可以落到完全不同的执行模型上。** 在 CUDA 上，
   `T.Kernel` 的网格对应 `blockIdx`、`T.Parallel` 对应线程与 warp；在昇腾上，
   `T.Kernel` 的网格对应 core 编号，而向量计算由 `T.SimtVF`/`T.SimdVF` 这两个 frame 单独
   定义执行域（`tilelang/ascend/language/frame.py:107,179`）。所以“grid of tile programs”
   是不依赖具体机器的说法，backend dialect 再补上各家自己的 launch 注解。
3. 因为 CUDA 的“grid of threads”把**执行的最小单位**定义成线程，程序员必须从线程写起；
   而“grid of tile programs”把**执行的最小单位**定义成一个 tile 程序，线程/核的划分
   由后端决定。这正是本章第 2.4 节那张对照图的落点。

**常见错误**：以为 `T.Kernel` 就是“CUDA 的 kernel 换个名字”。语义上它绑定的是一组
tile 程序，而不是一组线程。

---

## E9 动手题 ★★★

**解题思路**：脚本已经把公式写成函数，改动参数即可；关键是要解释结论**为什么**变。

**参考答案（要点）**：

1. 默认参数下 `MatMul(4096³)` 的 AI ≈ 1400 FLOP/Byte（compute-bound），
   `Softmax(4096²)` 的 AI 约 1–2 FLOP/Byte（memory-bound）。
2. 缩到 `512³` 后：FLOP 变成 $2\times512^3=2.68\times10^8$，访存量变成
   $(512^2\times3)\times2=1.57$ MB，AI ≈ **170**，**低于** 333 的平衡点，结论从
   compute-bound 翻转成 memory-bound。原因是本模型里
   $AI = \dfrac{2M^3}{3M^2\cdot b} = \dfrac{2M}{3b}$（$b$ 为每元素字节数）**随边长线性
   增长**：矩阵越小，它越容易落在 memory-bound 一侧。这正是习题把规模改小的目的，
   “AI 判类”不是与规模无关的标签，它依赖 shape 与 dtype。
3. 当平衡点升到 `100e12 / 1.5e12 ≈ 67 FLOP/Byte` 时，`MatMul` 的 AI≈1400 仍然远高于
   平衡点，仍是 compute-bound；`softmax` 仍是 memory-bound。只有当矩阵变小到访存量与
   计算量可比（例如很小的 K 或极瘦长的矩阵）时，矩阵乘才会掉到平衡点附近。
4. `RMSNorm` 的算术强度更低：它每读一个元素只做“平方 → 累加 → 乘缩放”等少量运算，
   且需要至少两遍访存（一遍求均方，一遍归一化），分母不变、分子很小，所以 AI 比 softmax
   还低。

**常见错误**：以为 AI 与 shape 无关。只有当 $MK+KN+MN$ 这一项相对 $MNK$ 可以忽略
（即 K 很大、矩阵很“方”）时，AI 才近似与形状无关；一般情况下它随形状变化，
`MatMul(512³)` 与 `MatMul(4096³)` 就会给出不同的判类。

---

## E10 动手题 ★★★

**解题思路**：把 softmax 变成“行内分块 + 片上归约”。

**参考答案（要点）**：

```text
# 目标：读 x 一遍，写 y 一遍
for each row r:                      # 一个 tile 程序负责一行（或一行的若干段）
    x_tile = load x[r, :]  -> 片上 (shared)      # 全局读 1 次
    m = tile_reduce_max(x_tile)                  # 片上归约，不访存全局
    e = tile_exp(x_tile - m) -> 片上              # 就地覆盖，仍只在片上
    s = tile_reduce_sum(e)                       # 片上归约
    y_tile = e / s -> 就在片上算出
    store y_tile -> y[r, :]                      # 全局写 1 次
```

2. 成立的原因：softmax 是**逐行独立**的（分母只依赖同一行）。只要一行能放进片上，
   就可以把“归一化所需的统计量”在片上算出来，而不必让中间量回写显存。
3. 需要流水/多缓冲的地方：把第 `r+1` 行的 `load` 与第 `r` 行的计算重叠。需要同步的地方：
   `tile_reduce_max/sum` 是对片上数据的跨线程归约，必须先让所有写入 x_tile 的线程完成，
   再做归约（片内 barrier），否则会读到一半的数据。

**常见错误**：以为“一行放不下片上”就没法做。真实 attention 里用的是在线 softmax（分块
累积最大值与和），属于第 15 章的内容；本章只要求给出“能放进片上”的版本。

---

## E11 分析题 ★★

**解题思路**：分块前按“每个输出元素都要重新读一遍 A、B 的一整行/列”来算。

**参考答案**：

1. 朴素实现：为了算 `C` 的每个元素，要读 `A` 的一整行（K 个）和 `B` 的一整列（K 个）。
   共 $M\times N$ 个输出元素，各读 $K$ 个 fp16 = 2 字节：
   访存量 $\approx 2\times M\times N\times K\times 2 = 2\times4096^3\times2 \approx 2.7\times10^{11}$
   字节 ≈ 274 GB。算术强度 ≈ 1 FLOP/Byte，严重 memory-bound。
2. 使用 `128 × 32`（M×K）与 `32 × 128`（K×N）tile：每个输出 tile 读 $128\times32$ 的 A
   与 $32\times128$ 的 B 各一次，理论最小访存量
   $(MK + KN + MN)\times2 = 96$ MB，算术强度 ≈ 1400 FLOP/Byte。
3. 分块把同一个数据在多次计算之间**复用**（A 的一段被 N 方向多个 tile 共享），于是每个
   字节参与运算的次数暴涨。瓶颈因此从“带宽够不够”转移到“算力够不够”。
4. 代价是：每次都多做了“先把数据搬进片上、再算、再写回”的动作，需要额外的片上空间与
   同步；如果 tile 选得不好，收益会被搬运和同步吃掉。

**常见错误**：只算“读 A 一遍 + 读 B 一遍 + 写 C 一遍”就当成朴素实现的访存量。那是
**最优**情况，不是朴素情况。

---

## E12 概念题 ★★

**解题思路**：从官方对照表抄事实，再用社区文章解释难点。

**参考答案**：

1. 至少 5 项（`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:144-152`）：
   ① 片上存储规划（UB/L1 分配与复用）；② tile 级算子的低层实现（`T.gemm` → 硬件矩阵
   指令）；③ SIMT 计算的线程映射与 barrier 插入；④ `T.Pipelined` 的调度、重叠与多缓冲；
   ⑤ 同步（Set/Wait flag 的配对与 ID 管理）；⑥ 支持范围内的 layout 转换（如 ND→NZ）。
2. “把同步交给编译器”更难，是因为它需要在**时间维度**上推理：搬运指令“已经发出”不等于
   “已经完成”，而“什么时候可以覆盖一块 buffer”是运行时才确定的事。要正确解，编译器必须
   把硬件事实建模成约束（谁读谁写、延迟、发射间隔、片上容量），再用 SMT 求解器（Z3）求
   一个可行且尽量紧凑的流水计划。社区文章特别指出：**latency 与 issue interval 必须分开**
   一条指令 30 拍才出结果，不代表这条流水 30 拍内不能接下一项任务；混起来就会把本可
   重叠的机器排成串行。来源：
   `../tilelang-corpus/09-zhihu-da-zhi-ruo-yu/raw/zhihu-dazhiruoyu-deepseek-ascend.md`
   （社区文章）。代码落点见 `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py:108-114`
   的 `EstimateLatency → AutoSchedule → PrepareMultiBuffer → InsertSync →
   MaterializeMultiBuffer`。
3. 因为后端一旦错了，表现往往是“偶发数值错误”，而不是编译报错（同一篇文章里说的
   “Debug/DFX 信息缺失”）。这类 bug 在特定 shape、特定迭代、流水压得紧时才出现，只能靠
   一套系统性的验证体系才能长期维护，功能做出来只是开始。

**常见错误**：把“自动调度”理解成“自动把 `T.Parallel` 向量化”。官方 README 明确写了
“Automatic scheduling handles task order and dependencies; it does not imply automatic
tensorization of arbitrary `T.Parallel` code into SIMD instructions.”

---

## E13（可选，需要硬件）★★★

**解题思路**：这是一道实验题，重点在于**测量口径的自洽**。

**参考答案（要点）**：

1. 安装按官方 `docs/get_started/Installation.md`（第 7 章会详细展开）。
2. 生成代码里，`T.copy` 通常降级成 `cp.async`/TMA 一类的搬运循环，`T.gemm` 通常降级成
   `mma`/`wgmma`/`tcgen05` 一类的矩阵指令（具体取决于目标架构）。定位方法是搜索生成的
   `.cu` 源里与 `A_shared`/`B_shared`/`C_local` 对应的变量名。
3. 记录 `num_stages ∈ {1, 3, 4}` 的 `do_bench` 结果时，**必须**固定：同一台机器、同一
   shape、同一 dtype、同一 backend、同一 warmup/repeat、是否含 L2 刷新的口径。结论应当
   只在这组条件下成立。

**常见错误**：换 `num_stages` 时顺手改了 block size，导致无法归因；或者把首次编译时间
算进延迟里。

---

## E14（可选，需要硬件）★★★

**解题思路**：同类实验，但重点是**片上容量约束**。

**参考答案（要点）**：

1. 按 `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md` 的 Quick Start 运行；
   需要 CANN 工具链（`bisheng`、`ld.lld`）与 `torch_npu`，并确认 `torch.npu.is_available()`。
2. 生成源码里可以找到矩阵乘相关的低层调用（社区文章与官方文档都提到 `T.gemm` 最终落到
   `asc_mmad` 一类的指令）以及各级 MTE 搬运。
3. 把 `num_stages` 从 2 改成 3 会增加一份多缓冲占用。由于 L1/UB 容量是硬约束，你很可能
   会看到编译期直接失败（约束不可解），而不是运行期报错。把报错原文抄下来，它通常就写着
   是哪一级存储超了容量。

**常见错误**：把“编译通过”当成“计划可行”。在昇腾侧，调度是解出来的；不可解时编译器
通常在编译期就会拒绝，这与 CUDA 侧“编得过但跑得慢”的表现不同。
