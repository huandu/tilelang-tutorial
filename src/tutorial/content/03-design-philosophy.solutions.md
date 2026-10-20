# 第 03 章 习题答案

与 `exercises.md` 的编号一一对应。每题给出**解题思路**、**参考答案**与**常见错误**。
所有源码引用都指向锁定版本 TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）。本章没有任何需要在本机实测才能得出的
结论；凡是涉及硬件的部分都明确标注为「有卡时做」。

---

## E1 概念题 ★

**解题思路**：回到论文摘要那一句，先抓动词（decouples），再看宾语（scheduling space
与 dataflow），最后看宾语里的括号（四个维度的清单）。

**参考答案**：

1. 核心主张：**把调度从数据流里拆出去**。论文原文是 "TileLang decouples scheduling space
   (thread binding, layout, tensorize and pipeline) from dataflow, and encapsulated them as
   a set of customization annotations and primitives."（来源：
   `../tilelang-corpus/03-tilelang-papers/raw/arxiv-2504-17577.md`）。用中文说：
   用户只负责描述「数据怎么流动」，硬件相关的调度决定被打包成注解与原语，
   写不写由用户决定，不写就由编译器决定。
2. 四个词，按括号里的顺序：**thread binding**、**layout**、**tensorize**、**pipeline**。
3. 解耦的是**调度空间**与**数据流**。
   * 属于调度的例子：`T.gemm(A_s, B_s, C_f, policy=T.GemmWarpPolicy.FullRow)` 里的
     `policy`——它决定了 warp 怎么划分，即 thread binding
     （`tilelang/language/gemm_op.py:160`、`tilelang/tileop/base.py:5-12`）。
   * 属于数据流的例子：`T.copy(A[by * block_M, ko * block_K], A_shared)`——
     它说的是「把 A 的这一块搬到 shared memory」，与用几个线程搬无关
     （`tilelang/language/copy_op.py:59`）。

**常见错误**：把「解耦」理解成「数据流和调度完全无关」。它们是**分开表达**，不是**互不影响**：
存储位置（数据流的一部分）会约束布局（调度的一部分），这也是 LayoutInference 必须同时
看两者的原因。

---

## E2 概念题 ★

**解题思路**：逐句拆开，先看这句决定「数据经过哪里、算什么」，还是「为了跑得快怎么安排」。

**参考答案**：

1. **前半句对，理由要改。** `alloc_shared` 确实属于数据流，因为它决定了「这块数据要经过
   shared memory 这一级」；但理由不是「scope 是数据本身的属性」——scope 恰恰是一个
   **硬件相关**的选择（shared 是片上的、线程块内共享的）。它归到数据流，是因为数据流
   的定义就是「数据经过哪些存储与计算」，而不是「这段代码和硬件有没有关系」。
   （定义来源：`_vendor/tilelang-v0.1.15/docs/get_started/overview.md:78-84`。）
2. **错。** 改 `num_stages` 改的是**多缓冲深度**（pipeline 维度），循环的迭代次数
   `T.ceildiv(K, block_K)` 一个字都没变。「迭代次数」与「缓冲级数」是两件事。
   依据：`T.Pipelined` 的 `num_stages` 文档写的是
   「The max number of buffer used between pipeline producers and consumers」
   （`tilelang/language/loop.py:123-133`）。
3. **错。** 不写 `policy` 时，缺省是 `GemmWarpPolicy.Square`
   （`tilelang/language/gemm_op.py:166`，枚举定义在 `tilelang/tileop/base.py:10`）。
   thread binding 这一维度仍然**被决定了**，只是决定权交给了编译器的缺省值。
   这正是「有入口、也有缺省」的含义。

**常见错误**：把「用户没有写」等同于「没有被决定」。本章 2.3 节那张表最后一列就是在说
每个维度的缺省行为是什么。

---

## E3 概念题 ★★

**解题思路**：第 3 问的关键是抓住「语言层看见的是一个整体，编译期才把它切开」这个反差。

**参考答案**：

1. 三要素：**一段有形状的数据**、**一处明确的归属**（哪个线程块 / warp 拥有它）、
   **一个显式的存储位置**（global / shared / fragment / 专用 buffer）。定义见正文 2.4 节，
   官方版本的措辞在 `docs/get_started/overview.md:74`。
2. 它被**摊到每个线程的寄存器里**了。`alloc_fragment` 申请的是「整块 tile 的逻辑缓冲」，
   编译期的 **LayoutInference** pass 会推导出一个 `Fragment` 布局，把逻辑索引
   `(i, j)` 映射成 `(线程号, 该线程内的局部下标)`。官方文档原话：
   「`TileLang` uses a Layout Inference Pass during compilation to derive a Layout object
   `T.Fragment`, which determines how to allocate the corresponding register files for each
   thread」（`docs/get_started/overview.md:82`）。
   所以「逻辑 tile」在语言层存在，在运行时不存在。
3. 可以这样说：**你申请的东西比硬件真实拥有的东西大得多**——你按 `(block_M, block_N)`
   申请一块缓冲，而单个线程根本装不下它；能成立，只能是因为语言层把 tile 当成了一个
   整体（一等概念），由编译期负责把它折叠成线程私有的碎片。

**常见错误**：说成「`alloc_fragment` 就是寄存器数组」。它是**逻辑**缓冲，物理形态由
`Fragment` 布局决定；同一段代码换一个布局，每个线程分到的元素就变了。

---

## E4 概念题 ★★

**解题思路**：这一题的答案必须落到「原话 + 行号」，因为它是本章最容易写错、也最容易被
读者质疑的一句。

**参考答案**：

1. 原话是 "**Note: This interface is not yet fully implemented.**"
   出现在 `_vendor/tilelang-v0.1.15/docs/get_started/overview.md:20`，
   紧跟在 Beginner Level 的三条说明之后。corpus 转存的同一句话在
   `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/overview.md`
   里也能看到。
2. 现存的是 **eager 前端**：`@tilelang.jit` 的 eager 模式 + `T.const` / `T.dynamic` +
   `T.Tensor[[...]]` 注解 + `T.empty`（`tilelang/language/eager/builder.py:1177`、
   `tilelang/language/symbolics.py:13`、`tilelang/language/allocate.py:400`；
   示例见 `examples/eager_jit/eagerjit.en.ipynb`）。
   * **解决了**：形状/类型从实际张量参数里推出来，签名风格接近 Triton，
     调用即编译执行（`tilelang/jit/__init__.py:277-312` 的 lazy / eager 两种模式）。
   * **没有解决**：kernel 体里**仍然要写存储层级**——`T.alloc_shared`、`T.gemm`、
     `T.Pipelined` 一个都不能少。也就是说，用户仍然得知道 shared memory 存在。
3. 因为「方向」和「交付」是两件事。如果把 eager 前端说成「TileLang 已经有硬件无关层」，
   读者第一次写 kernel 时就会按「不需要懂硬件」的期待去写，然后在 `T.alloc_shared` 上
   卡住——这是对读者最有害的错误期待。官方文档自己都标了未完成，本章没有理由替它加戏。

**常见错误**：把 `T.const` 当成通用能力。它的 docstring 明确写着 eager mode only，
在非 eager 上下文调用会抛 `JITNoBuilderError`（`tilelang/language/eager/builder.py:1179,1196`）。

---

## E5 概念题 ★★

**参考答案**：

1. 藏起来的是 **tile 内部的实现细节**。论文原话：
   "the existence of built-in tile types and operations ... **simplifies the structure of
   tensor programs by hiding important performance details pertaining to intra-tile memory
   coalescing, cache management and specialized hardware utilization**"
   （`../tilelang-corpus/11-related-compilers/raw/triton-mapl-2019.md:325-329`，加粗为本书所加）。
   即三类：**tile 内的访存合并**、**cache 管理**、**专用硬件（张量核一类）的使用**。
2. 代价是「更多的实现工作量由语言/编译器承担，换来的是用户侧的简单」：
   "The benefits of Triton come at the cost of increased programming efforts"
   （同文件 `:248-256`）。注意这句话的语境是：Triton 靠往传统编译管线里**加** tile 级操作
   与优化来获得灵活性，这份灵活性不是免费的。
3. TileLang 多保留的是**显式的存储位置与（可选）布局**。第一项代价就是
   **beginner 层尚未交付**：想「不懂硬件也写好 kernel」的用户，在当前版本里没有完整的
   落脚点（`docs/get_started/overview.md:20`；另外官方文档也提醒 Level 1
   「may not handle some complex kernel fusion patterns (e.g., Flash Attention) as easily」，
   `docs/deeplearning_operators/matmul.md:41`）。

**常见错误**：把「hiding」理解成「Triton 不支持 tile」。正好相反：Triton 是围绕 tile 建的，
它藏起来的是 **tile 内部** 的细节。

---

## E6 代码阅读题 ★

**解题思路**：先认 tile op 的 IR 名字，再数「没写出来的信息」，最后把信息对应到 pass。

**参考答案**：

1. 三个 tile op：`T.copy` 两个、`T.gemm` 一个。它们在 IR 里的名字分别是
   **`tl.tileop.copy`** 与 **`tl.tileop.gemm`**
   （`tilelang/language/copy_op.py:130`、`tilelang/language/gemm_op.py:199-200`）。
2. 没写出来的信息，至少包括：
   * **线程怎么分**：`A_shared` 由哪些线程写、`T.gemm` 里 warp 怎么划分、
     多少个线程协作——一个都没写；
   * **同步**：`T.copy` 写入 shared 与 `T.gemm` 读取 shared 之间需要 barrier，
     代码里没有任何 `__syncthreads()` 或 `T.sync_threads()`；
   * （还可以说）**搬运用什么机制**：同步 SIMT 拷贝、`cp.async` 还是 TMA，也没写。
3. 对应的 pass（都在 `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py` 里）：
   * `MaterializeKernelLaunch(lower_thread_binding=True, default_threads=128)`（`:70`）
     负责把 tile-program 的并行循环落到线程上；
   * `LowerTileOp()`（`:142`）把 tile op 拆成具体实现；
   * `ThreadSync("shared")` / `ThreadSync("shared.dyn")`（`:258-259`）负责插入
     shared memory 的同步。

**常见错误**：以为 `T.copy` 在 Python 层就展开了循环。它只产出一个 intrinsic call，
真正的展开发生在 lowering 阶段。

---

## E7 代码阅读题 ★★

**参考答案**：

1. 属于 **expert 层**。两条依据：
   * 用了 `T.get_thread_binding(0)`——CUDA 方言的 docstring 明确说它
     「exposes `threadIdx` for thread-level code」（`tilelang/cuda/language/kernel.py:23`）；
   * 用了 `T.alloc_local` 与 `T.serial` 手写循环，并且**没有**用 `T.copy` 搬数据——
     搬运逻辑（`A_shared[tk] = A[...]`）是手写的。
   （对照：developer 层会写 `T.copy(...)` + `T.gemm(...)` + `T.Parallel(...)`。）
2. `A_shared` 存的是**当前 K 分块的 A 向量**，它对块内**所有**线程都是同一个值——
   每个线程都要读同样的一份，所以用 `tn` 之外的下标（只有 `tk`）就够了。
   `B_shared` 存的是 **N 方向上一整段 B 的若干行**，每个线程只关心自己那一行 `tn`，
   所以下标必须带 `tn`。这就是「归属」在代码里的样子：谁读，决定它怎么被写。
3. `C_reg` 是**每个线程一个标量**的累加器，语义上就是线程私有的，所以用
   `T.alloc_local((1,), accum_dtype)`。如果换成 `T.alloc_fragment((1,), ...)`，
   就变成了「一块块级逻辑缓冲」，会被送进 LayoutInference 去推导
   `Fragment` 布局（`tilelang/transform/__init__.py:30-38`）——但它只有 1 个元素而线程有
   `BLOCK_N` 个，语义上说不通。**这正是三层的边界**：一旦你开始按线程号写代码，
   就应该用 `alloc_local`，而不是让布局推导去猜。

**常见错误**：说「`alloc_local` 是共享内存的另一种写法」。`alloc_local` 是线程私有的
（`local` scope），`alloc_shared` 才是块内共享的（`shared.dyn` scope）——
见 `tilelang/language/allocate.py:49,67` 两个函数的默认 scope。

---

## E8 代码阅读题 ★★★

**参考答案**：

1. `if fast_dequant:` 决定的是**同一个位置用哪一层的实现**：
   * `fast_dequant=True` 走 `fast_dequant_bf16_fp4_twiddling`，它是 **expert 层**——
     用 `T.get_thread_binding()`（`:225`）、`T.alloc_local`（`:228-231`）、
     `T.serial` / `T.vectorized`、以及 `T.call_extern`（`:247`）去调一个外部反量化函数；
   * `fast_dequant=False` 走 `simple_dequant_bf16_fp4`，它是 **developer 层**——
     用 `T.alloc_fragment`（`:298-299`）、`T.copy`（`:302,313`）、`T.Parallel`（`:303`）。
2. 因为**两者的接口相同**：都是「读 `B_shared` + `Scale` + `k`，写 `B_dequantize_shared`」。
   在下游 `T.gemm` 眼里，它只关心 `B_dequantize_shared` 里最终是不是 BF16 数据，
   不关心是谁写的、用什么层次写的。这正是 composable 的含义：
   **层次的边界不是编译单元边界**——不同层次的代码最终都变成同一种东西（IR 里的
   语句与 tile op），因此可以放进同一个流水循环、由同一个 pass 一起处理。
   官方表述见 `docs/deeplearning_operators/matmul.md:214-218`。
3. 拆成两个 kernel 会损失（至少两条）：
   * **多一次 global memory 往返**：反量化的结果必须写回显存、再被 GEMM 读回来，
     而不是留在 shared memory 里直接被 `T.gemm` 消费；
   * **失去跨阶段的流水重叠**：现在反量化与 GEMM 在同一个 `T.Pipelined` 循环里，
     可以和第 k+1 块的搬运重叠；拆开之后两个 kernel 只能串行，
     重叠只能靠 kernel 之间的并发，而那是另一类优化。
   （还可以说：多一次 kernel 启动开销，见第 26 章。）

**常见错误**：认为「同一个 kernel 里混用不同层次」需要特殊的注解或开关。
上面这段代码里没有任何开关，只有一句普通的 Python `if`。

---

## E9 动手题 ★★

**解题思路**：直接跑，观察输出里的 `def` / `export` 两列。

**参考答案**：

1. 作者运行结果（2026-10-01，v0.1.15）：

   ```
   == 官方文档核对：beginner 层的实现状态 ==
     [有] docs/get_started/overview.md:20  含 “not yet fully implemented”

   合计 47 个名字，找到 47 个，缺失 0 个。
   ```

   （最后三行的行号随脚本版本略有差异，关键是「缺失 0 个」与退出码 0。）

2. 例如 `Tensor`、`StridedTensor`、`GemmWarpPolicy`。`export` 表示脚本只在**导出清单**
   里找到了这个名字——它们在 `tilelang/language/common.py` 的 `__all__` 里被再导出，
   而不是在本文件里 `def` 出来的（`Tensor` 的实际定义是 `tilelang/language/proxy.py`
   里的代理类）。这一列的用意就是防止把「出现在导出清单里」误读成「这里定义了它」。
3. 例如把 `"clamp"` 加进 developer 那一组。它会命中
   `tilelang/language/customize.py:52`（该模块由 `common.py:107` 的
   `from .customize import (...)` 再导出），输出形如
   `[有] clamp def tilelang/language/customize.py:52`，退出码仍是 **0**——只要名字真的在
   导出面上，加进去就能通过。若故意写错一个名字（如 `"clamp2"`），退出码变成 **1**，
   并在末尾列出缺失清单。（这三条结果作者都实际跑过。）

**常见错误**：把脚本当成「运行验证」。它不导入 tilelang，只做源码字符串核对；
「这个 API 在我的环境里能用」是另一回事（见该目录 README 的注意事项）。

---

## E10 动手题 ★★

**解题思路**：沿着两条线走——一条是 `tilelang.language` 这个门面，一条是
`common → cuda 方言` 的再导出。关键是发现 **CUDA 方言把 `gemm` 遮蔽（shadow）了一次**。

**参考答案**（每步都给出文件与行号）：

1. `tilelang/language/__init__.py:11-12`
   —— `from tilelang.cuda.language import *`，并把这个模块的 `__all__` 直接当作自己的
   `__all__`。所以 `T` 命名空间里的东西**主要来自 CUDA 方言**。
2. `tilelang/cuda/language/__init__.py:5`
   —— `from tilelang.language.common import *`，先把「后端无关」的那一套装进来。
3. `tilelang/language/common.py:79`
   —— `from .gemm_op import (gemm, gemm_blockscaled)`，于是先有了后端无关的 `gemm`
   （定义在 `tilelang/language/gemm_op.py:160`）。
4. `tilelang/cuda/language/__init__.py:88`
   —— `from .gemm_op import gemm, gemm_blockscaled, gemm_sp`。**这一句把上一步的
   `gemm` 覆盖掉了**；CUDA 版本的签名多出 `mbar` 等硬件参数。
5. `tilelang/cuda/language/__init__.py:59-65` 的注释也解释了这件事：
   「The CUDA dialect shadows a handful of common constructs with versions that expose
   CUDA's knobs as typed keywords: T.Kernel (threads, prelude, cluster_dims),
   T.copy / T.im2col (TMA and cache hints), T.gemm (mbar), ...」。
6. 最终 `tilelang/language/__init__.py:12` 把 CUDA 方言的 `__all__` 定为
   `tilelang.language.__all__`，于是 `T.gemm` 指向
   `tilelang/cuda/language/gemm_op.py:32` 的 CUDA 版本。

一句话总结：**`import tilelang.language as T` 拿到的永远是 CUDA 方言**
（`tilelang/language/__init__.py:32` 的 `__tilelang_dialect__ = "cuda"`），
想要别的后端要显式去 `tilelang.<backend>.language`。

**常见错误**：以为 `tilelang.language` 是「中立层」。它是**门面**，且默认门面就是 CUDA。
这也是正文 5.4 节第 (d) 项边界的一部分。

---

## E11 动手题 ★★★

**参考答案**：

1. 四行都在 `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py`：

   ```
   :129    mod = tilelang.transform.PipelinePlanning()(mod)
   :130    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
   :134    mod = tilelang.transform.LayoutInference()(mod)
   :142    mod = tilelang.transform.LowerTileOp()(mod)
   ```

2. 注释抄录与翻译：

   > 原文：`# Run pipeline planning and software-pipeline rewriting before layout\n
   > # inference so inferred layouts see the final pipelined structure directly.`
   > （`tilelang/cuda/pipeline.py:127-128`）
   >
   > 译文：在布局推导**之前**先做流水线规划与软件流水重写，这样推导出来的布局能直接
   > 看到最终的流水线结构。

3. 会出问题的场景（推断，不要求实测）。流水线重写会：把多缓冲变成多份 buffer
   （buffer 版本化）、把循环体里的语句按 stage 重新排序、插入等待/提交语句
   （`docs/programming_guides/software_pipeline.md:44-60` 描述了 stage / order 的语义）。
   如果 LayoutInference 先跑：
   * 它看到的还是「单缓冲 + 原始语句顺序」的循环，推导出的 `Fragment` / shared 布局
     可能建立在**后来会消失的结构**上；重写之后若 buffer 被版本化、访问被重新排序，
     原先的布局注解可能不再成立，轻则需要插入额外的布局转换，重则编译期报错；
   * 更隐蔽的问题是：同一个 shared buffer 在不同流水阶段被不同线程访问，
     布局的一致性约束只有在最终结构上才能判对。

   所以这句注释不是风格偏好，而是**依赖顺序**：布局推导的输入必须已经是最终形态。

**常见错误**：把 pass 顺序当成「随便排」。TileLang 的 pipeline 文件里几乎每一段都写了
为什么这么排（例如 `:113` 解释 `LowerBlackwell2SM` 必须在 `LayoutInference` 之前、
`:208-210` 解释 `ConfigIndexBitwidth` 必须在 `FlattenBuffer` 之后）。

---

## E12 动手题 ★★★

**参考答案**：

1. 四个变体各改一个维度（源码见
   `code/03-schedule-as-annotation/schedule_as_annotation.py`）：

   | 变体 | 维度 | 改动 |
   | --- | --- | --- |
   | `v0` | —— | 全部缺省（`threads=128`、`num_stages=2`、无布局注解、无 `policy`） |
   | `v1` | thread binding | `T.gemm(..., policy=T.GemmWarpPolicy.FullRow)` |
   | `v2` | layout | `T.annotate_layout({B_s: make_swizzled_layout(B_s)})` |
   | `v3` | pipeline | `T.Pipelined(..., stage=[0,0,1], order=[0,1,2])`（**不**传 `num_stages`） |

2. 无 GPU 时 `--static-only` 的输出形如：

   ```
   调度维度对照：
     v0  （无）             全部走缺省值
         依据：tilelang/cuda/language/kernel.py:32-35（threads 默认 128）
     v1  thread binding  T.gemm(..., policy=T.GemmWarpPolicy.FullRow)
         依据：tilelang/language/gemm_op.py:166 与 tilelang/tileop/base.py:11
     v2  layout          T.annotate_layout({B_s: make_swizzled_layout(B_s)})
         依据：tilelang/language/annotations.py:45 与 tilelang/layout/swizzle.py:66
     v3  pipeline        T.Pipelined(..., stage=[0,0,1], order=[0,1,2])
         依据：tilelang/language/loop.py:112-118 与 docs/programming_guides/software_pipeline.md:44-60
   ```

3. 正确性验证的设计（有卡时）：
   * **固定随机种子** `torch.manual_seed(0)`，四个变体**复用同一对输入张量** a/b；
   * 参考实现用 `torch.relu(a @ b)`（fp32 比较）；
   * 判据统一为 `rtol=1e-2, atol=1e-2`（fp16 输入、fp32 累加下的常规容差），
     四个变体必须**全部**通过；
   * 需要记录的测量条件（即使本题不做性能对比，也要写下来，供第 28 章使用）：
     GPU 型号与驱动、CUDA / TileLang 版本、`M/N/K` 与 block 尺寸、dtype、
     单卡还是多卡、warmup 与重复次数、计时口径（wall clock / CUDA event / cupti）。

   **本题不下性能结论**：正确性的一致性与性能无关；性能对比属于第 28 章的方法论。

4. （可选进阶）把 `v2` 的布局换成恒等 `T.Layout(B_s.shape, lambda i, j: (i, j))`。
   预期观察：*理论上*「显式恒等布局」与「不写注解」应当等价，但**是否真的等价取决于
   LayoutInference 是否把它当作已给约束、以及该约束是否与 `T.gemm` 期望的 shared 布局
   兼容**。如果不等价，最可能的表现是：显式恒等布局与 `T.gemm` 的期望冲突，导致要么
   编译报错、要么在布局之间多插一次转换（代价是额外的 shared memory 访问）。
   **本题的正确做法是承认这里没有把握**：请在你自己的机器上跑，并把观察写下来；
   作者没有 GPU，没有做过这项实验，不给出结论。

**常见错误**：把四个变体的性能差异当成「哪个更好」的结论。不同调度在不同 shape、
不同 GPU 上的优劣完全不同，这正是第 28 章要建立性能模型的原因。
