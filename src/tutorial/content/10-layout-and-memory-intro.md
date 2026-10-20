# 第 10 章 布局与存储层级入门

## 学习目标

读完本章，你应当能够：

1. 用“逻辑张量 → 物理存储”这条线索解释：为什么 **layout（布局）是接口**，而不是
   编译器的内部细节。它在生产者和消费者之间传递约定。
2. 说清 TileLang 里 `T.Layout` 与 `T.Fragment` 各自的语义：前者是“逻辑下标 → 物理
   下标”的纯函数，后者在此之上再加“线程映射 + 每线程寄存器槽 + 可复制”（replicate）。
3. 说出 global / shared / fragment / local（以及昇腾的 UB / L1 / L0A-L0C）四类存储
   各自的用途与容量量级，并解释 **bank conflict** 是怎么产生的、swizzle 为什么能消它。
4. 用 `T.annotate_layout` / `make_swizzled_layout` / `T.use_swizzle` 这三个入口干预布局，
   并说清各自的代价，尤其是“`T.use_swizzle` 是线程块栅格化，不是共享内存 swizzle”。
5. 用 `tilelang.tools.plot_layout` 和编译器的 layout visualization 把一份布局画出来、
   读出来，并据此判断布局是否合理。
6. 说出昇腾侧 fractal（分形）存储、ND→NZ 重排与 OOB padding 的直觉用途（细节见第 21 章）。

## 前置知识

* 第 7–9 章：能写出并读懂一个 tiled kernel，知道 `T.alloc_shared`、`T.alloc_fragment`、
  `T.copy`、`T.gemm`、`T.Parallel` 在程序里长什么样。
* 第 6 章：对“存储层级”有抽象概念，越靠近计算单元，容量越小、延迟越低。
* 会看 Python 的 lambda；能读懂 `(i, j) -> (j, i)` 这类下标变换。
* 本章**不要求**任何 CUDA 或昇腾编程经验。所有硬件名词第一次出现时都会先说明它解决
  什么问题。

本章是全书第一个真正的“为什么 TileLang 不是语法糖”的关口：第 1 章说 TileLang 的
价值在编译期，本章给出第一个具体证据，**布局推导**。第 11 章会在此之上讲流水线与
调优，第 21 章会把这里的直觉升级成完整的布局推导算法。

---

## 1. 问题的由来：同一个张量，为什么会有“摆法”问题

### 1.1 一个看起来无法解释的“慢”

设想你用 TileLang 写了一个 GEMM，参数、tile 大小都照着官方示例抄，结果：

* 结果**正确**；
* 但性能比预期差一截；
* 把 `block_K` 从 32 改成 64，性能突然又回来了。

代码里既没有多算一次乘法，也没有多读一次显存。变化的是**数据在线程、寄存器与片上
存储里的摆法**。同一份逻辑数据，摆法不同，硬件每拍能同时服务的请求数就不同。这就是
本章要讲的东西。

### 1.2 逻辑张量与物理存储是两张不同的表

你在 TileLang 里写下的是**逻辑形状**：

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)   # 逻辑：[128, 32]
C_local  = T.alloc_fragment((block_M, block_N), accum) # 逻辑：[128, 128]
```

这两行说的是“逻辑上有一块 128×32 的数据”和“逻辑上有一块 128×128 的累加器”。但它们
没有说：

* `A_shared` 的第 `(i, j)` 个元素在 shared memory 里的**字节偏移**是多少；
* `C_local` 的第 `(i, j)` 个元素属于**哪个线程**、是该线程的**第几个寄存器**；
* 这两个 buffer 被读写时，一个 warp 的 32 个线程会命中 shared memory 的哪些 **bank**。

这三件事合起来叫这份数据的**布局（layout）**。逻辑形状由你写；物理布局由编译器推导，
也可以被你覆盖。这就是“接口”的含义（图 1）。

```text
逻辑视图                     物理视图
──────────────              ──────────────────────────────────────────
A[M, N]                     fragment ─▶ (线程 t, 寄存器槽 l)
                            shared   ─▶ 字节偏移 ─▶ bank
                            global   ─▶ base + i*stride + j
```

### 1.3 为什么这件事不能留给“实现细节”

如果布局真的只是实现细节，那它就应该像寄存器分配一样，用户永远不用管。但它是接口，
原因有三条：

**（一）布局必须跨语句保持一致。** 生产者和消费者可能对同一块 buffer 有不同要求。
`T.copy` 把数据搬进 `A_shared` 时有一个自然的写入顺序；`T.gemm` 读 `A_shared` 时，被
选中的 MMA 指令有它自己期望的排布。两者不一致，要么插入一次额外的转换（多读写一轮片
上存储），要么直接编译失败。TileLang 的 `T.copy` 文档把这件事写得很直白：作用域或
向量宽度不被支持时，**编译失败**而不是悄悄降级
（`_vendor/tilelang-v0.1.15/docs/programming_guides/instructions.md`）。

**（二）布局直接决定片上容量。** 一个 `[128, 128]` 的 fp32 累加器是 64 KB。如果每个
线程都存一份完整的副本（replicate），容量立刻乘上线程数。所以“谁在哪个线程上持有
哪些元素”不是美学问题，而是**能不能装下**的问题。

**（三）布局决定访存效率。** shared memory 的 bank conflict、global memory 的合并
访存（coalescing）、张量核的 operand 排布，都由布局决定。第 1 章讲过“搬数据比算数据
贵”；布局就是决定搬运效率的那一层。

这三条合起来回答了一个常见疑问：*为什么不能像写 NumPy 一样写 kernel？* 因为 NumPy
不需要你关心数据在硬件上怎么摆，而 kernel 的每一分性能都来自这里。

---

## 2. 概念与模型

### 2.1 `T.Layout`：逻辑下标到物理下标的纯函数

TileLang 用一个对象描述布局：`T.Layout`。它的定义极其简单
（`_vendor/tilelang-v0.1.15/tilelang/layout/layout.py:12-43`）：

```python
class Layout(Node):
    def __init__(self, shape, forward_fn):
        """shape: 每个逻辑维度的大小；forward_fn: 逻辑下标 -> 物理下标"""
        ...
        forward_index = forward_fn(*vars)   # vars = [i0, i1, ...]
```

也就是说，布局就是一张**下标映射表**：

```text
forward: (i0, i1, ...)  ──▶  [f0, f1, ...]      （输入空间的坐标 → 输出空间的坐标）
```

`Layout` 提供了几个正交的能力（都在 `layout.py` 里）：

| 能力 | 方法 | 直觉 |
| --- | --- | --- |
| 正向映射 | `map_forward_index(indices)` | 给逻辑坐标，算物理坐标 |
| 取逆 | `inverse()` | 给物理坐标，反推逻辑坐标（存在才可逆） |
| 展平 | `get_linearized_forward_index()` | 物理坐标 → 行主序的一维偏移 |
| 分块复制 | `repeat(dim, factor)` | 把小块的布局“拼”成大块 |
| 提升维度 | `expand(leading_shape)` | 前面加几维，原样透传 |

> **为什么要有 `repeat` / `expand`？** 因为 MMA 之类的原子布局只定义在一个很小的
> “微内核”上（例如 8×8 或 16×16），真实 tile 是把微内核重复铺开。`repeat`/`expand`
> 就是这套“用小布局拼大布局”的代数（对照 CuTe 的 layout algebra，见
> `../tilelang-corpus/05-nvidia-backend/raw/nvidia-cutlass-media-docs-cute.md`）。

一个最小例子，把 4×4 矩阵转置的布局：

```python
import tilelang.language as T

transpose = T.Layout([4, 4], lambda i, j: (j, i))   # 逻辑 (i, j) → 物理 (j, i)
```

### 2.2 `T.Fragment`：给每个元素配一个线程和一个寄存器槽

`T.Layout` 只描述“数据摆在哪”，不描述“谁去访问”。它用的是逻辑下标，不涉及线程。
而寄存器（fragment / local）里的数据本质上是**被特定线程持有**的，于是需要更进一步的
描述：`T.Fragment`。

`T.Fragment` 是 `Layout` 的子类
（`_vendor/tilelang-v0.1.15/tilelang/layout/fragment.py:12-99`），它在
`forward_index` 之外多两样东西：

```text
1. forward_thread : (i0, i1, ...) ─▶ 线程号
2. replicate      : 同一份数据在几个线程上各存一份
```

合起来，一个 `Fragment` 回答的问题就是：**逻辑元素 `(i, j)` 落在哪个线程的哪个槽里，
以及有几份副本。** 图 4 给了 `MakeGemmFragment8x8` 的完整对应表；源码只有两行
（`_vendor/tilelang-v0.1.15/src/layout/gemm_layouts.cc:47-52`）：

```cpp
PrimExpr forward_thread = FloorDiv(j->var, 2) + 4 * i->var;  // 线程号 = 4*i + j/2
PrimExpr index          = FloorMod(j->var, 2);               // 槽号   = j % 2
```

这里已经藏着一个非常重要的直觉：

> **`T.alloc_fragment((128, 128))` 里写的是“整个线程块的寄存器切片”，不是“每个线程
> 各存 128×128”。** 官方文档专门提醒了这一点：`T.alloc_fragment` 分配的是整个 block
> 的寄存器文件，编译器再用 layout inference 推出 `T.Fragment`，决定每个线程分到哪些
> 元素（`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/get_started/overview.md`）。

所以 `block_M × block_N = 128×128` 的累加器，在 128 个线程上平均就是每线程 128 个槽；
而 8×8 的例子（图 4）在 32 个线程上每线程只有 2 个槽。**布局决定了每线程寄存器压力**，
这也直接连到第 11 章的占用率与调优。

### 2.3 存储层级：用途与容量约束

TileLang 在语言层暴露的 `scope` 决定了数据放在哪一级（
`_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:49-103,171`）：

| 语言原语 | 默认 scope | 用途 | 谁持有 | 容量量级（NVIDIA，公开规格） |
| --- | --- | --- | --- | --- |
| `T.alloc_shared(shape, dtype)` | `shared.dyn` | 线程块内共享、给张量核喂数据 | 整个 block | 每 SM 可配到 **228 KB**；每 block 最多 **227 KB**；静态 shared 上限 **48 KB** |
| `T.alloc_fragment(shape, dtype)` | `local.fragment` | 累加器、tile 级中间结果 | 每线程一段 | 每 SM 的寄存器文件（几十万字节量级） |
| `T.alloc_local(shape, dtype)` | `local` | 每线程私有数组 | 单线程 | 同上（也可能被溢出到 local memory） |
| `T.alloc_var(dtype, init, scope)` | `local.var` | 标量变量、标志位 | 单线程 | 1 个元素 |
| `T.alloc_global(...)` | `global` | workspace | 全局 | 显存 |

> 上表里 228/227/48 KB 三个数字是 **H100（compute capability 9.0）** 的公开规格
> （`../tilelang-corpus/05-nvidia-backend/raw/nvidia-hopper-tuning-guide.md:50-51,226-228`）；
> B200（cc 10.0）的每 SM 容量同样标为 228 KB
> （`../tilelang-corpus/05-nvidia-backend/raw/nvidia-blackwell-tuning-guide.md:50-51,93`）。
> 这是**厂商文档口径，不是实测**，且不同架构 / 不同 cc 数值不同，写代码时不要当常数用。

一条必须记住的对偶关系：

```text
容量：  寄存器 < shared < global
延迟：  寄存器 < shared < global
```

所以“把数据放到离计算最近的地方”永远是对的，只要放得下。**放不下的时候会发生什么，
正是 5.2 节要讲的坑。**

### 2.4 bank conflict 与 swizzle：让 32 个线程别撞同一扇门

#### 为什么会有 bank

shared memory 为了做到“一个 warp 一拍读很多个字”，被划分为等宽的 **bank**：每个
bank 每拍能服务一个地址；连续的 32-bit 字依次落到连续的 bank；warp 是 32 线程，bank
也是 32 个。**只要 32 个线程访问的地址落在 32 个不同 bank 上，一次就能全部服务完；
多个地址落在同一个 bank 上，就被拆成多拍串行**。这就是 bank conflict
（`../tilelang-corpus/05-nvidia-backend/raw/nvidia-cuda-best-practices.md:1545-1554`）。

#### 经典的 32 路冲突

32×32 的 `float` tile，行主序，行长 32 个字。warp 里第 `t` 个线程读**同一列**：

$$
\operatorname{addr}(t)=32t,\qquad
\operatorname{bank}(t)=(32t)\bmod 32=0.
$$

于是 32 个线程全部撞在 bank 0，访问被拆成 32 拍。

官方最佳实践里给出的两种修法（`.../nvidia-cuda-best-practices.md:2622-2641`）：

1. **padding**：把行长从 32 改成 33，于是 `bank(t) = (t*33) mod 32 = t`，恰好铺满 32
   个 bank，冲突消失（代价：容量多出约 3%）。
2. **swizzle（异或重排）**：不增容量，只把“第 `s` 行第 `c` 个单元”搬到第
   `c xor s` 个单元。图 2 给了两者前后对照。

#### TileLang 的三种 swizzle 粒度

TileLang 在 `tilelang/layout/swizzle.py:94-126` 暴露了三个固定粒度的构造函数，它们对应
`src/layout/gemm_layouts.cc:442-530` 里的三个 2D swizzle：

```text
make_quarter_bank_swizzled_layout   32B   1-bit XOR   8 行周期
make_half_bank_swizzled_layout      64B   2-bit XOR   8 行周期
make_full_bank_swizzled_layout     128B   3-bit XOR   8 行周期
```

核心公式（以 128B 为例，`src/layout/gemm_layouts.cc:502-520`）：

$$
\begin{aligned}
v &= \frac{128}{\text{element\_size}},\\
t_s &= \left\lfloor\frac{i}{8}\right\rfloor,\qquad
s=i\bmod 8,\\
t_c &= \left\lfloor\frac{j}{8v}\right\rfloor,\qquad
c=\left\lfloor\frac{j}{v}\right\rfloor\bmod 8,\qquad
\mathit{vec}=j\bmod v,\\
c' &= \operatorname{xor8x8}(c,s),\\
\mathit{physical}
  &=(t_c,t_s,\mathit{vec}+(c'+8s)v).
\end{aligned}
$$

`xor2x2 / xor4x4 / xor8x8` 的定义在 `src/layout/gemm_layouts.cc:376-393`，本质就是低位
按位异或。图 3 把这三种粒度的置换表画了出来。

> **为什么叫“full bank”？** 因为 bank 是 4B，32 个 bank 合起来是 128B；“128B
> swizzle”的原子正好横跨一整组 bank。名字里的数字是**字节宽度**，不是 bank 数。

#### swizzle 的代价

swizzle 不是零成本，它换来的是“不涨容量”和“无冲突”：

* **地址计算变贵**：每次访存多一次异或（以及配套的除法/取模）。
* **对齐与形状有硬约束**：`MakeFullBankSwizzleLayout2D` 里有两个断言，
   `stride == 4 || stride % 8 == 0` 且 `continuous % (vector_size * 8) == 0`
  （`src/layout/gemm_layouts.cc:509-511`）；不满足就构造失败。
* **必须与消费者匹配**：源码注释明确写着，swizzle 是否启用取决于是否使用 TMA
  （`tilelang/layout/swizzle.py:64-65`）。用错了不是更慢，而是把 TMA 的描述符对齐
  搞坏，或者让本该廉价的访问变成先解一遍地址。

### 2.5 昇腾侧：fractal、ND→NZ 与 OOB padding

昇腾的 Cube 单元对操作数在 L1 里的排布有**硬性要求**：数据必须以**分形（fractal）**
形式存放，16 行为一组、宽度为 C0 的小块，其中 `C0 = 32 byte / element_size`
（bf16/fp16 是 16，fp8 是 32，fp32 是 8）
（`_vendor/tilelang-v0.1.15/src/ascend/layout/ascend_layouts.h:1-30`）。

逻辑 `[rows, cols]` 映射到物理 `[cols/C0, rows/16, 16, C0]`，正映射是
`(r, c) -> [c/C0, r/16, r%16, c%C0]`（`tilelang/layout/ascend.py:20-37`）。不同角色用
不同的分形方向：

$$
\begin{aligned}
\text{MajorK}:&\quad [MN,K]\to[k/C_0,mn/16,mn\bmod16,k\bmod C_0],\\
\text{MajorMN}:&\quad [K,MN]\to[mn/C_0,k/16,k\bmod16,mn\bmod C_0],\\
\text{L0C}:&\quad [M,N]\to[n/16,m/16,m\bmod16,n\bmod16].
\end{aligned}
$$

MajorK 把归约轴 $K$ 放在外层，MajorMN 把输出轴 $MN$ 放在外层，L0C 使用固定的
累加器布局。

真实的张量在 GM / UB 上是普通的 **ND（row-major）** 数组，Cube 需要 **NZ**，两者之间
必须做一次重排搬运（pass `InsertNd2Nz`，见
`tilelang/ascend/pipeline.py:91` 与 `src/ascend/transform/insert_nd2nz.cc`）。而分形要求
`rows` 是 16 的倍数、`cols` 是 C0
的倍数，真实 shape 常常不满足，于是需要 **OOB padding**：在逻辑 region 之外补零，把
物理块填满（`src/ascend/transform/insert_oob_padding.cc`）。图 5 画了这条链路。

`T.alloc_l0c` 会在分配时自动挂上 L0C 的固定布局，因为那是硬约束，不是可选项
（`tilelang/ascend/language/allocate.py:55-70`：`layout=True` 时执行
`make_ascend_l0c_layout`）。

昇腾侧的 `scope` 比 NVIDIA 多一层
（`tilelang/ascend/language/allocate.py:15-70`）：

```text
T.alloc_shared(...)  → UB（Unified Buffer，Vector 侧片上存储）
T.alloc_l1(...)      → shared.l1   （Cube 的操作数暂存）
T.alloc_l0a/l0b(...) → shared.l0a / shared.l0b  （Cube 的左右矩阵）
T.alloc_l0c(...)     → shared.l0c  （Cube 输出/累加，自动带 L0C 布局）
```

**本章只需要记住一句话**：昇腾把“逻辑 → 物理”这件事从“编译器自由推导”变成了
“硬件规定 + 编译器归一化”。细节（`NormalizeAscendFractalStorage`、`InsertNd`、
`AscendInsertOOBPadding` 三个 pass 的顺序、约束与转换代价）留给第 21 章。

---

## 3. 用法

### 3.1 默认路径：让编译器推

绝大多数情况下你**什么都不用写**。TileLang 的编译期有一个
`LayoutInference` pass（`src/transform/layout_inference/layout_inference.cc`），在
“tile op 图”上传播并求解布局约束：每个 `fragment`/`local` 元素落到哪个线程的哪个
寄存器槽、shared buffer 用哪种 swizzle、是否需要插入 layout 转换。它的输入是
`T.copy` / `T.gemm` / `T.Parallel` 这些 op 的语义和 buffer 的 scope，输出就是上面那些
`T.Layout` / `T.Fragment`。

一段“什么都不写”的最小 GEMM：

```python
import tilelang
import tilelang.language as T


@tilelang.jit
def matmul(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C
```

骨架与官方示例一致（`_vendor/tilelang-v0.1.15/examples/gemm/example_gemm.py:4-26`）。
`A_shared` 会被推导出一个布局（通常是某种 swizzle），`C_local` 会被推导出一个
`Fragment`。你一行布局代码都没写。

### 3.2 `T.annotate_layout`：覆盖推导结果

当默认结果不满足需要（对齐某个外部约定、或调优发现某个 swizzle 更快），用
`T.annotate_layout({buffer: layout})` 显式指定。定义很短
（`_vendor/tilelang-v0.1.15/tilelang/language/annotations.py:45-66`）：

```python
def annotate_layout(layout_map: dict):
    _layout_map = {}
    for buffer, layout in layout_map.items():
        if is_reducer(buffer):
            assert isinstance(layout, PartialFragment), (...)
        elif is_fragment(buffer):
            assert isinstance(layout, Fragment), (...)   # fragment buffer 必须是 Fragment
        if isinstance(layout, Layout):
            _layout_map[buffer.data] = layout
        elif isinstance(layout, Callable):                # 也可以直接给一个函数
            _layout_map[buffer.data] = Layout(buffer.shape, layout)
        else:
            raise ValueError(f"Invalid layout: {layout}")
    return sblock_attr({"layout_map": _layout_map})
```

要点（照抄自源码，不是概括）：

* 参数是 **`{buffer: layout}` 的字典**，键是 buffer 对象，值是 `Layout`/`Fragment`，
  也可以是一个**回调函数**（会被包成 `Layout(buffer.shape, fn)`）。
* 对 **fragment buffer**，layout **必须是 `Fragment`**（含线程映射），否则直接断言失败。
* 对 **reducer**（`T.alloc_reducer`），layout 必须是 `PartialFragment`。因为 reducer 的
  replica 是“待求和的加数”，不是“相等的副本”，普通 `Fragment` 语义会说错。

常见写法（官方 example 里就是这么用的，例如
`_vendor/tilelang-v0.1.15/examples/gdn/example_chunk_delta_bwd.py:270-273`）：

```python
from tilelang.layout import make_swizzled_layout

T.annotate_layout({
    A_shared: make_swizzled_layout(A_shared),
    B_shared: make_swizzled_layout(B_shared),
})
```

或者用 CUDA 侧更贴 MMA 的构造器（`tilelang/cuda/intrinsics/layout/mma_layout.py:336`）：

```python
from tilelang.cuda.intrinsics import make_mma_swizzle_layout

T.annotate_layout({
    A_shared: make_mma_swizzle_layout(A_shared),
    B_shared: make_mma_swizzle_layout(B_shared),
})
```

### 3.3 `T.use_swizzle`：线程块栅格化，**不是**共享内存 swizzle

名字里都有 swizzle，但这两件事完全不同。`T.use_swizzle` 的定义是
（`_vendor/tilelang-v0.1.15/tilelang/language/annotations.py:30-42`）：

```python
def use_swizzle(panel_size: int, order: str = "row", enable: bool = True):
    if order == "row":
        device_func = "rasterization2DRow"
    elif order == "column":
        device_func = "rasterization2DColumn"
    elif order == "mlx":
        device_func = "rasterization2DMLX"
    ...
    return attr(None, "threadblock_swizzle_pattern", tvm_tuple(device_func, panel_size))
```

它改的是**线程块（thread block）被调度的顺序**（栅格化），目的是让相邻的 block 更可能
复用同一段 L2 数据。它**不**改任何 buffer 在 shared memory 里的排布。判定依据是源码
里的键名：`threadblock_swizzle_pattern`。

```python
with T.Kernel(...):
    T.use_swizzle(panel_size=10, enable=True)   # 只影响 block 的栅格化顺序
```

对照表（这一节最容易被误记，建议背下来）：

| 你想要的效果 | 用哪个 | 落在哪 |
| --- | --- | --- |
| 消除 shared memory 的 bank conflict | `T.annotate_layout({buf: make_swizzled_layout(buf)})` | buffer 的物理排布 |
| 让相邻 block 复用 L2 | `T.use_swizzle(panel_size=..., order=...)` | 线程块的启动顺序 |

### 3.4 观察 layout：`plot_layout` 与编译器可视化

**不要靠猜。** TileLang 提供两种观察手段（官方文档
`_vendor/tilelang-v0.1.15/docs/tools/layout_visualization.md`）：

**（a）直接画一个 `T.Layout` / `T.Fragment`。** 适合设计映射时自检：

```python
import tilelang.language as T
from tilelang.tools import plot_layout

transpose = T.Layout([4, 4], lambda i, j: (j, i))
plot_layout(transpose, save_directory="./tmp", name="transpose_4x4", formats="png")
# → 写出 ./tmp/transpose_4x4.png，并打印路径
```

* 对 `T.Layout`：`view="input"`（默认）时格子是**输入**空间、标签是输出的展平坐标；
  `view="output"` 反过来。
* 对 `T.Fragment`：格子按**线程**染色，标签同时给出 `T`（线程号）与 `L`（线程内槽号）
这正是把图 4 画出来。
* 高维 layout 会被展平成二维网格：除最后一维外全部并入行坐标。

**（b）让编译器把推导出来的 layout 打印/画出来。** 适合检查一个编译好的 kernel：

```python
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
        tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "txt,svg",
    }
)
def kernel(A, block_M, block_N):
    ...
```

`TL_LAYOUT_VISUALIZATION_ENABLE` 默认 `False`（
`tilelang/transform/pass_config.py:252`）；打开后，pass 会对每个 inferred fragment 打印：

```text
C_local inferred layout:
  Shape: [128, 128] -> [..]
  Thread: <线程表达式>
  Index:  [<寄存器槽表达式>]
  Replicate:  1
```

`TL_LAYOUT_VISUALIZATION_FORMATS`（`pass_config.py:255`）接受 `txt` / `png` / `pdf` /
`svg` / `all` 或逗号组合；`txt` 不额外成图。图写到 `./tmp/<buffer>_layout.svg` 这类
路径下。

**已知限制**（`docs/tools/layout_visualization.md` 末尾）：直接画 `T.Fragment` 只支持
二维输入、且线程与局部索引必须取值唯一（single-valued）；编译器生成的图只覆盖二维 inferred
fragment，其它形状只打印文本并给警告；绘图是在 Python 里逐格枚举，**大 shape 会很慢、
很占内存**；output view 对非双射/稀疏映射会覆盖或留空，只当诊断视图，不要当校验。

---

## 4. 实现机制：编译器在背后做了什么

### 4.1 layout inference：约束传播 + 冲突检测

`LayoutInference` 把每个 tile op 的“布局要求”当成约束，在这个图上做传播与求解：

```text
T.copy(global → shared)  → 对 shared 布局提出「写入要合并/可向量化」的约束
T.gemm(shared → fragment) → 对 shared 提出「MMA 期望的排布」，对 fragment 提出 Fragment
T.copy(fragment → global) → 对 fragment 提出「写出要合并」的约束
```

当一个 buffer 收到两个**不一致**的要求时，处理策略分三种（照源码
`src/transform/layout_inference/layout_inference.cc`）：

1. **能合并就合并**：如果两边都是 swizzle 布局，
   `MergeSwizzleLayouts` 取**更小的粒度**并继续（`.../layout_inference.cc:454-461`，
   底层在 `src/layout/gemm_layouts.cc:999-1026`）；
2. **fragment 能包含就取更宽的那个**（`ProveFragmentContains` 路径）；
3. **否则报错**：`LOG(FATAL) << “Get different layout for ” << buffer
   << “\n current layout: ... \n previous layout: ...”`，并附上 buffer 的 source span
   （`.../layout_inference.cc:463-468`）。对 reducer 还有一条更早的、可被捕获的
   `LayoutConflictException`（`.../layout_inference.cc:378-393`）。

这条路径解释了一个常见现象：你手写的 `T.annotate_layout` 与 op 要求冲突时，报错信息里
会同时出现 `current layout` 和 `previous layout`。那是**推出来的**和**你写的**两块布局
在打架。

### 4.2 swizzle 的代码落点

* 对外 API：`tilelang/layout/swizzle.py:66-126`（`make_swizzled_layout`、
  `make_{volta,wgmma,tcgen05mma}_swizzled_layout`、三个固定 bank 粒度）；
* 真实构造：`src/layout/gemm_layouts.cc:442-530`（2D 公式）+ `:921-947`
  （`MakeSwizzledLayout` 按 buffer 的 `stride`/`continuous`/`element_size` 选粒度）；
* 粒度识别与合并：`DetectSwizzleMode` / `MergeSwizzleLayouts`
  （`src/layout/gemm_layouts.cc:966-1026`）。

### 4.3 容量与合并：共享内存不是随便加的

一个 kernel 只能有一份动态或静态 shared memory 分配，所以
`MergeSharedMemoryAllocations` 要把多个 TIR 级分配合并成一整块，并处理对齐
（`_vendor/tilelang-v0.1.15/src/transform/merge_shared_memory_allocations.cc:1-6`）。
它就是“shared 容量超限”这条报错最后会撞上的地方，也解释了为什么 padding 的 +1 列
会真的让占用变大（5.2 节）。

### 4.4 昇腾：把自由选择换成归一化

昇腾 pipeline 在 layout/向量化之前插了一整段“调度”与归一化，其中和本章直接相关的是
（`tilelang/ascend/pipeline.py:83-101`；综合概述见
`../tilelang-corpus/90-synthesis/02-compiler-internals-map.md`）：

```text
NormalizeAscendFractalStorage   fractal 存储归一化      (pipeline.py:86)
InsertNd2Nz                     插入 ND→NZ 变换          (pipeline.py:91)
AscendInsertOOBPadding          边界填充
```

（`AscendInsertOOBPadding` 在 `pipeline.py:101`；`AscendLayoutInference` 在 `:83`。）
对应实现分别在 `src/ascend/transform/{normalize_fractal_storage,insert_nd2nz,insert_oob_padding}.cc`。
**NVIDIA 那边布局是“推导出来的”，昇腾这边是“硬件规定 + 编译器归一化”**。这是两条
后端路线最本质的差别之一。

---

## 5. 进阶与陷阱

### 5.1 布局不匹配 → 性能断崖（不是错误）

布局不匹配有两种后果，第二种更隐蔽：

* **显式失败**：约束无法合并时编译直接报 `Get different layout for ...`（4.1 节）。
* **隐性降级**：约束能合并，但合并后的布局对某一条路径不友好。典型是
  `MergeSwizzleLayouts` 为了兼容取**更小的** swizzle 粒度。它不报错，但可能让本该
  128B 的访问退化成 32B，或者破坏与 TMA 描述符的对齐假设。这类问题**不会出现在正确性
  测试里**，只会出现在性能上。

排查顺序：先证明布局是问题（用 3.4 的工具把 inferred layout 打出来），再决定是换
tile 参数、还是显式 `annotate_layout`。**不要在没看布局前就乱加注解。**

### 5.2 shared memory 容量超限

shared memory 有硬上限（H100/B200 每 block 227 KB，静态 48 KB 起，见 2.3 节）。超了会
编译失败或启动失败。即使**没有**超限，占用太多 shared memory 也会限制每个 SM 上能
同时驻留的 block 数。这是占用率问题的常见来源，解决办法通常是减小
`block_M/block_N/block_K`。注意 padding 也占容量：2.4 节那个 +1 列放在 `[128, 128]`
fp32 上是 128 个字 = 512 B，看着不大；但如果同时有多缓冲、且形状更大，累计起来会
明显挤压可用空间。第 11 章讲多缓冲时会再回到这个约束。

### 5.3 错误 swizzle 反而更慢

swizzle 是“用一点地址计算成本换消除 bank conflict”。当**本来就没有冲突**时，这笔交易
是亏的：

* 数据布局本来是合并访存友好的，加 swizzle 后地址表达式变复杂，反而多花指令；
* swizzle 与 TMA 描述符的对齐不匹配时，会把廉价的 TMA 搬运打回手动搬运
  （源码注释见 `tilelang/layout/swizzle.py:64-65`）；
* 粒度选错（本该 128B 却用了 32B）会牺牲带宽。

判断依据仍是“有没有冲突”：没有冲突就不加。**注解是用来消问题的，不是用来“更优化”
的。**

### 5.4 把 `T.use_swizzle` 当成共享内存 swizzle

这是本章最值得记住的一个混淆点。`T.use_swizzle` 只改**线程块栅格化顺序**（3.3 节），
它既不会消除 bank conflict，也不会改变任何 buffer 的物理排布。想消 bank conflict，
要用 `T.annotate_layout` + `make_swizzled_layout` 系列。

### 5.5 过度注解

`T.annotate_layout` 是**覆盖**，不是“加强版优化提示”。你写的布局会成为硬约束，编译器
不能再放宽；一旦它与某个 op 的要求不一致，就会走 4.1 的报错路径。正确姿势是：
**先让编译器推（3.1），用 3.4 的工具看清结果，确认有问题再针对单个 buffer 覆盖
（3.2）。**

---

## 6. 本章小结

把这一页当复习卡片：

* **布局是接口**：逻辑形状 `[M, N]` 不等于物理摆法；布局把“生产者写”和“消费者读”
  连起来，所以它必须被显式表达、可被观察、可被覆盖。
* **两种表达**：`T.Layout` = 逻辑下标 → 物理下标；`T.Fragment` = 再加线程映射、寄存器
  槽、replicate。`T.alloc_fragment((M,N))` 说的是**整个 block** 的寄存器切片。
* **四级存储**：global / shared / fragment / local；越近越快越小。shared 有硬上限
  （H100/B200 每 block 227 KB）。
* **bank conflict**：一个 warp 内多个线程命中同一 bank 就串行化。修法有两种：padding
  （涨容量）和 swizzle（不涨容量，按行异或，代价是地址计算 + 对齐约束）。
* **三个干预入口**：默认让 `LayoutInference` 推；用 `T.annotate_layout` 覆盖某个
  buffer；用 `T.use_swizzle` 改**线程块栅格化顺序**（与 bank conflict 无关）。
* **观察工具**：`plot_layout` 直接画 `Layout`/`Fragment`；`TL_LAYOUT_VISUALIZATION_*`
  让编译器打印/落盘 inferred layout。
* **昇腾对应物**：fractal 存储（物理 `[cols/C0, rows/16, 16, C0]`）、ND→NZ 重排、
  OOB padding；那里布局更多是**硬件规定**而非自由推导。

## 延伸阅读

* 想复习“为什么搬数据比算数据贵”：第 1 章 1.1–1.2。
* 想看布局推导的完整算法与昇腾 fractal 细节：第 21 章。
* 想把这些布局放进流水线看效果：第 11 章。
* 想在真机上把布局问题定位出来：第 12 章（Nsight Compute / msProf）。
* 上游一手材料索引：见本章 `references.md` 的 R1–R14。
