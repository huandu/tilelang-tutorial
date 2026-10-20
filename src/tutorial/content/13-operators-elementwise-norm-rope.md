# 第 13 章 算子实战 I：逐元素、归一化与 RoPE

## 学习目标

读完本章，你应当能够：

1. 用第 02 章的记账口径，给任意一个逐元素 / 归一化 / RoPE 算子写出它的
   `(bytes, flops)` 账本，算出融合前后的访存量比值，并据此判断「该不该融合」
   ——而不是先写再测。
2. 说清逐元素算子的性能几乎只由两件事决定：**每个字节被读/写几次**、**每次访存有多宽**；
   并能从 `kernel.get_kernel_source()` 的输出里指出哪一行是 128-bit 访存、哪一段退化成了标量。
3. 手写并解释逐元素算子的四级实现：`T.Parallel(N)`、显式「每线程 n 个元素」、
   `T.copy` 到 fragment 再算、跨 dtype 的 cast；能说明每一级为什么变快或变慢。
4. 用 `T.reduce_sum` / `T.reduce_max` / `T.reduce_absmax` 写出 RMSNorm 与 LayerNorm，
   解释 `clear` 的语义、共享内存/寄存器四种 scope 组合各自动了什么手脚，
   以及「线程内归约 + 跨线程 butterfly 归约」这条分工线。
5. 写出 RoPE 的两种配对约定（interleaved 与 NeoX half-split），知道写错了不会报错、
   只会得到数值错误的模型，并知道为什么要在 fp32 上算完再 cast。
6. 把 RMSNorm + RoPE（+ per-token FP8 量化）融进一个 kernel，写出它的收益（bytes）
   与代价（寄存器 / 片上存储压力），并说出昇腾侧同款实现的三个差异
   （UB、`T.SimtVF`、`T.alloc_ub`）。

## 前置知识

* 第 02 章 §2.1–2.2：算术强度、机器平衡点、以及本书采用的记账口径。本章的账本
  沿用同一套规则，只是把它算到具体算子上。
* 第 08 章：`T.Kernel` 的网格语义、`T.Parallel`、buffer 与内存作用域。
* 第 09 章：`T.copy` / `T.reduce_*` / `T.fill` / `T.clear` 的语义与降级行为。本章会
  反复用到其中的「归约是两段的」「`T.copy` 在执行完就一定可用」这两条结论。
* 第 10 章：fragment / shared 的区别，以及为什么 layout 是接口。
* 第 12 章：怎么打印生成代码、怎么定位编译错误。本章的每个优化都用它来验证。
* 可选：第 11 章的多缓冲与 `T.Pipelined`——本章的昇腾部分会用到，NVIDIA 部分暂不需要。

本章的代码分两级验证：`code/01-ledger/` 与 `code/07-api-audit/` 是**本机跑过**的
纯 Python 模型与静态检查；`code/02-` 到 `code/06-` 是需要真实加速器的 TileLang 骨架，
只做到**静态核对**（`py_compile` + API 名核对）。写作环境里**没有 NVIDIA GPU，
也没有昇腾 NPU**（见 [`../../VERSIONS.md`](../../VERSIONS.md)），所以本章不含任何
本机实测的性能或正确性结论。

---

## 1. 问题的由来：最"简单"的算子也值得认真写

### 1.1 一行 PyTorch 的代价

第 02 章讲过，eager 模式下 `torch.sin(torch.cos(x))` 是两次 kernel launch，中间结果
必须落到显存再读回来。把这句话换成账本，就是一串「读一遍、写一遍」叠加：

```text
t = cos(x)         读 x 一遍，写 t 一遍
y = sin(t)         读 t 一遍，写 y 一遍
合计：x、t、y 各走两趟显存，中间张量 t 的存在完全不必要
```

逐元素算子是所有大模型算子里最不起眼的一类，但它们在训练和推理里出现的次数最多：
残差相加、缩放、激活、位置编码、量化、去量化、mask、droput……一次前向里可能有几十个。
它们几乎全是 **memory-bound**：算术强度只有零点几到几个 FLOP/Byte，远低于任何加速器
的机器平衡点。换句话说，**它们的运行时间几乎完全等于「搬了多少字节 ÷ 带宽」**。

于是优化方向就只剩两条：

1. **少搬**：把本来要落到显存再读回来的中间量留在片上（融合）；
2. **搬宽一点**：让每次访存尽量是一次 128-bit（或更宽）的对齐访问，而不是逐元素 32-bit。

本章的三个实战，就是在这两条线上一级一级往上走。

### 1.2 先算账，再写代码

`code/01-ledger/ledger.py` 把上面的口径写成了可复算的代码。它不 import 任何框架，
只用整数运算，所以结论可以在任何机器上复现。五个场景的结果：

```text
场景                                    逐算子写法        融合成一个 kernel    比值
scale+add (y = x*a + b, fp32, N=2^20)   20.97 MB          12.58 MB            1.67×
RMSNorm (4096×4096, fp16)              234.99 MB          67.12 MB            3.50×
LayerNorm (4096×4096, bf16)            335.59 MB          67.13 MB            5.00×
RoPE (4096×32 行, rope_dim=128, fp16)  369.10 MB         100.66 MB            3.67×
RMSNorm+RoPE+FP8 量化 (4096×4096, bf16) 185.62 MB         51.40 MB            3.61×
```

对应地，这十个方案的算术强度全部落在 `0.100 – 2.315 FLOP/Byte`。只要目标机器的
平衡点高于这个区间的上界，它们就都是 memory-bound——这就是「融合不是优化技巧，
而是把账本上重复的项删掉」的意思。

三个要注意的口径问题（第 02 章也强调过）：

* 这是**下界**，真实 kernel 只会更慢；
* 中间张量不免费：上一个 kernel 写、下一个 kernel 读，两次都要计；
* `ledger.py` 里「6 步分解」「4 步分解」是**一种假设的拆法**，不是对 PyTorch 实际
  产生的 kernel 数的追踪——PyTorch 的调度和融合策略会随版本变化。

### 1.3 三个实战的递进关系

本章的三个实战共用同一条主线，但每一步都多一个难点：

| 实战 | 算子 | 新增的难点 | 需要的新原语 |
| --- | --- | --- | --- |
| 一 | add / scale / cast | 访存宽度、边界谓词 | `T.Parallel` 的多层写法、`T.copy` 到 fragment |
| 二 | LayerNorm / RMSNorm | 跨线程归约、片上容量 | `T.reduce_*`、`T.clear`、split-K |
| 三 | RoPE、RMSNorm+RoPE+量化 | 跨 lane 数据交换、多算子融合 | shared 做配对、第二次归约、fp8 cast |

读完之后你应该能独立完成一件事：拿到一个新的 memory-bound 算子，
先算出它的账本，再决定用几趟访存、tile 取多大、要不要融合。

---

## 2. 概念与模型

### 2.1 逐元素算子的唯一自由度是访存

一个逐元素算子的计算部分简单到没有优化空间：每个输出元素只依赖对应位置的输入。
所以能动的只有访存。GPU 上一条 32-bit 的 `ld.global` 每个线程只能带回 4 字节，
一个 warp（32 线程）合并起来是 128 字节——这刚好是一条 cache line。如果改成
`ld.global.v4`（128-bit），每个线程带回 16 字节，一个 warp 就是 512 字节，
**同样多的访存指令搬了 4 倍的数据**。

fp16/bf16 下这个宽度可以到 8 个元素一次（128 bit）。所以在 fp16 的逐元素算子里，
一句 `T.Parallel(threads, num_per_thread)` 里的 `num_per_thread=8` 通常不是随便挑的
——它正好对应一次 128-bit 访存。

官方文档给了一个非常有代表性的对照：同样是

```python
for i in T.Parallel(threads):
    C[b_x * threads + i] = A[b_x * threads + i] + B[b_x * threads + i]
```

在 `threads=256` 下，TileLang 生成的 CUDA 是「32 个线程各做一次 `uint4` 访存」，
**而 launch config 仍然是 256 线程**（
`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/deeplearning_operators/elementwise.md:153-176`）。
编译器自己做了「减少参与线程数、加宽访存」这件事。理解这一点很重要：你写的是
**块级逻辑**，线程与向量宽度是编译器的决定；但你可以用写法去引导它。

### 2.2 归约：两段，而不是一段

逐元素算子不需要任何跨线程通信；归一化需要。以 `y = x * rsqrt(mean(x²) + eps)` 为例，
一行里的 N 个元素分散在不同线程的寄存器里，要得到这一行的平方和，必须先：

1. **线程内归约**：每个线程把自己负责的那几个元素先加起来（纯寄存器操作，没有同步）；
2. **跨线程归约**：所有线程的值合成一个（要通信）。

第 2 步在 v0.1.15 里由 `tl::AllReduce` 完成，它对不同的参与宽度走不同的机制：

```text
offset ≥ 32   →  shared memory + __syncthreads()   （跨 warp 只能走共享内存）
offset <  32  →  shfl_xor_sync                     （warp 内用寄存器交换）
```

这就是为什么「跨线程归约」在共享内存紧张时也要留出一块工作区：参与线程数超过 32 时
会通过 `T.add_workspace` 分配一块工作区
（`_vendor/tilelang-v0.1.15/src/op/reduce.cc:183-188`）。

### 2.3 片上存储是有限的：tile 尺寸是被容量反推出来的

融合之所以不是免费的，就在于中间量必须在片上某个地方存活。本书涉及的三层：

| 位置 | CUDA 侧 | 昇腾侧 | 容量感觉 |
| --- | --- | --- | --- |
| 寄存器 fragment | `T.alloc_fragment` | `T.alloc_fragment`（`T.SimtVF` 内） | 每线程几十到两百多个寄存器，决定 occupancy |
| 共享内存 | `T.alloc_shared` | `T.alloc_shared`（**就是 UB**） | 每 SM 一两百 KB |
| 全局显存 | `T.Tensor` | GM | 大，但带宽低一个数量级 |

所以一个 `(1, 4096)` 的 fp32 fragment 就是 16 KiB 的寄存器压力，`(4, 4096)` 就是 64 KiB。
官方昇腾 RMSNorm 示例的做法很有代表性：先算 UB 预算
（`ub_budget = 208 * 1024`），据此决定一次处理几行、是否复用输入 buffer
（`_vendor/tilelang-v0.1.15/examples/ascend/example_rmsnorm.py:17-33`）。
**tile 尺寸是容量的函数，不是调参的起点。**

### 2.4 融合的三种形态

本章讲的融合都发生在「一次 kernel 调用内部」，但实际工程里可以分三层看：

| 形态 | 例子 | 账本收益 |
| --- | --- | --- |
| 同一算子内减少访存 | 归一化只读一遍 x | 去掉一次整行读 |
| 相邻算子合并 | RMSNorm + RoPE | 去掉一次中间张量的写 + 读 |
| 跨模块融合 | Q/K RMSNorm + RoPE + 量化写 KV cache | 去掉整条中间链 |

TileOPs 的 `fused_qk_norm_rope_roofline` 就是第三种：它把「每个 q/k 值做一次带权
RMSNorm（4 FLOP）再加一次 RoPE（3 FLOP）」写成一个记账公式，读 q/k 列、写回 q/k 列，
v 列完全不碰（`../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/perf/formulas.py:910-923`）。
这就是「融合」在工程上真正的样子：**先写清账本，再写 kernel**。

---

## 3. 用法：三个递进的实战

这一节的代码都在 `code/` 下。每个实战都按同一个顺序走：
**PyTorch 参考实现 → 朴素 TileLang 版 → 优化版 → 为什么变快**。

### 3.1 实战一：逐元素（add / scale / cast）

#### 3.1.1 PyTorch 参考实现

```python
def ref_add(a, b):
    return a + b

def ref_scale_cast(x, alpha=0.5):
    return (x * alpha).to(torch.float16)
```

参考实现要尽量短——它的作用是**定义语义**（谁加谁、什么 dtype、要不要饱和），
不是性能基线。

#### 3.1.2 朴素版：一块数据交给一个 block

```python
with T.Kernel(T.ceildiv(N, threads), threads=threads) as bx:
    for i in T.Parallel(threads):
        idx = bx * threads + i
        C[idx] = A[idx] + B[idx]
```

这里**故意不写边界判断**。当 `N` 不是 `threads` 的整数倍时，`idx` 会越界，
而 TileLang 会在编译期自动插入谓词——这件事发生在
`legalize_safe_memory_access` 这一步。官方 elementwise 文档的原话是
「TileLang automatically incorporates boundary-checking conditions; however, this
comes at a cost」（corpus `elementwise.md:100-102`）。代价是什么，§5.1 会展开。

朴素版最短，但它把「每线程算几个元素」完全交给编译器：长度对齐时编译器可能自己
做出 8 元素的向量化（下面 3.1.3 会看到），非对齐时则整段退化成标量。

#### 3.1.3 优化一：显式告诉编译器「每线程算几个」

```python
with T.Kernel(T.ceildiv(N, threads * num_per_thread), threads=threads) as bx:
    for i, j in T.Parallel(threads, num_per_thread):
        idx = (bx * threads + i) * num_per_thread + j
        C[idx] = A[idx] + B[idx]
```

关键在最内层索引：`(bx * threads + i) * num_per_thread + j`。对最内层循环变量 `j`
来说，它是**步长为 1 的仿射表达式**，所以这一层可以被向量化成一次
128-bit 访存：fp32 取 4 个一组，fp16/bf16 取 8 个一组。

官方文档在同一个例子上的对照非常直接：`num_per_thread=8` 时生成的代码里出现了
`uint4 v_ = *(uint4*)(A + ...)`，而且是"每个线程连续 8 个元素"的形态
（corpus `elementwise.md:216-252`）。文档给的 CuTe 版本也是同一个思路：
`local_tile` 取出每线程 8 个元素，然后 `recast<nv_bfloat162>` 做向量算术。

**为什么快**：同样的数据量，访存指令数少了 4–8 倍，而 memory-bound 算子的时间
就花在访存指令和内存事务上。

#### 3.1.4 优化二：先 `T.copy` 到 fragment，再算

```python
a_frag = T.alloc_fragment((block,), dtype)
b_frag = T.alloc_fragment((block,), dtype)
c_frag = T.alloc_fragment((block,), dtype)
T.copy(A[start:start + block], a_frag)     # 整块搬进寄存器
T.copy(B[start:start + block], b_frag)
for i in T.Parallel(block):
    c_frag[i] = a_frag[i] + b_frag[i]
T.copy(c_frag, C[start:start + block])
```

对单纯的 add 来说，这一版**不会更快**——数据只被用一次，先搬进寄存器再算反而多占
寄存器。它的价值是**当一份输入要被用多次时**：比如
`y = x * rsqrt(mean(x²))`，x 既要用在平方和里，又要用在乘 rstd 那一步；
搬进 fragment 之后，第二遍用的是寄存器而不是显存。

这也是官方文档特意说明的一点：显式 `T.copy` 到寄存器后，生成的代码里会多出
`A_register` / `B_register` 两个数组，但「during the actual computation, these
registers are simply reassigned to `v_` and `v__1`」（corpus `elementwise.md:279`）。
换句话说：**这一级写法的收益完全取决于复用次数，没有复用时它只是噪音。**

#### 3.1.5 cast：简单的和需要归约的

跨 dtype 的 cast 仍然是逐元素的：

```python
for i, j in T.Parallel(block_M, block_N):
    Y[by * block_M + i, bx * block_N + j] = T.Cast(
        out_dtype, X[by * block_M + i, bx * block_N + j] * alpha
    )
```

但它的向量宽度上限更低：一次向量化的 cast 要求源侧和目标侧的 lane 数都能被同一个
宽度整除（`_vendor/tilelang-v0.1.15/src/transform/loop_vectorize.cc:966-987`）。
fp32 → fp16 时源侧最多 4 lane（128 bit / 32 bit），所以这条路径的访存宽度天然不如
同 dtype 的 add。

而当 cast 变成**量化**时，事情就变了：

```python
T.reduce_absmax(x_local, amax_local, dim=1)      # 按组求 amax → 这是归约！
for i in T.Parallel(blk_m):
    scale_local[i] = T.max(amax_local[i], 1e-4) / fp8_max
for i, j in T.Parallel(blk_m, group_size):
    q_local[i, j] = T.clamp(x_local[i, j] / scale_local[i], fp8_min, fp8_max)
T.copy(q_local, q_fp8)
```

这正是官方 `_vendor/tilelang-v0.1.15/examples/cast/example_per_token_cast_to_fp8.py:8-41` 的结构。
**「cast 类算子」一旦带上按组的 scale，就立刻变成「归约 + 逐元素」**——所以它同时
属于实战一和实战二，也是 §3.3.4 那个三算子融合的第三块拼图。

#### 3.1.6 两种前端写法：eager 与 classic

v0.1.15 里同时存在两种写法，本章的示例都用了 classic 风格，但你会在官方示例里
两个都看到：

| 风格 | 外形 | 符号维度 | 例子 |
| --- | --- | --- | --- |
| classic | `@tilelang.jit` 包 `@T.prim_func` | `T.Tensor((M, N), dtype)` + `T.dynamic("M")` | `_vendor/tilelang-v0.1.15/examples/norm/rms_norm.py` |
| eager | `@tilelang.jit` 直接包函数，函数体就是 kernel | `M, N = T.const("M, N")` | `_vendor/tilelang-v0.1.15/examples/elementwise/example_elementwise_add.py` |

`T.const` 只在这个 eager 模式下可用（
`_vendor/tilelang-v0.1.15/tilelang/language/eager/builder.py:1177-1200` 里明确写了
"eager mode only"）；classic 风格里请用 `T.dynamic`
（`_vendor/tilelang-v0.1.15/tilelang/language/symbolics.py:13-31`）。两者对生成代码没有本质影响，
但**不能混用**，报错信息（`JITNoBuilderError`）也不太好懂。

### 3.2 实战二：归一化（LayerNorm / RMSNorm）

#### 3.2.1 PyTorch 参考实现与账本

```python
def ref_rms_norm(x, weight=None, eps=1e-6):
    rstd = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    y = x.float() * rstd
    return (y * weight.float()).to(x.dtype) if weight is not None else y.to(x.dtype)

def ref_layer_norm(x, gamma, beta, eps=1e-5):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), gamma, beta, eps=eps)
```

按 PyTorch 语义逐步展开，RMSNorm 是 6 次调用：`pow → mean → add(eps) → rsqrt → mul
→ mul(weight)`；LayerNorm 类似，只是方差要用两遍归约（`sum(x)` 与 `sum(x²)`）。
账本上 4096×4096 的 RMSNorm 从 234.99 MB 降到 67.12 MB（3.50×），LayerNorm 从
335.59 MB 降到 67.13 MB（5.00×）——**归一化的全部优化空间都在「别把中间量写回显存」
这一件事上**。

这和第 02 章例 2 的结论一致：归一化的算术强度只有 1–2 FLOP/Byte 量级，
比任何 GEMM 低两个数量级。

#### 3.2.2 朴素版：读两遍 x

```python
with T.Kernel(N, threads=threads) as bx:          # 一个 block 一行
    x_sh = T.alloc_shared((D,), dtype)
    sq = T.alloc_fragment((D,), "float32")
    acc = T.alloc_fragment((1,), "float32")

    T.copy(X[bx, :], x_sh)                        # 第一遍：读进 shared
    for j in T.Parallel(D):
        sq[j] = T.Cast("float32", x_sh[j]) * T.Cast("float32", x_sh[j])
    T.reduce_sum(sq, acc, dim=0)
    for j in T.Parallel(D):                       # 第二遍：从 shared 读出来算
        Y[bx, j] = T.Cast(dtype, T.Cast("float32", x_sh[j]) * T.rsqrt(acc[0] / D + eps))
```

这一版其实还不错：**x 只从显存读了一遍**（第二遍读的是 shared）。但它已经把行完整地
放进了 shared，而且把平方和单独存了一份 fragment。它的价值在于最短、最容易看懂，
也是理解「先求统计量、再归一化」这条数据流的最短路径。

#### 3.2.3 优化版：整行留在寄存器里

```python
with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
    x_local = T.alloc_fragment((blk_m, N), dtype)
    sq_local = T.alloc_fragment((blk_m, N), dtype)
    sumsq = T.alloc_fragment((blk_m,), dtype)

    T.copy(X[bx * blk_m : (bx + 1) * blk_m, :], x_local)
    for i, j in T.Parallel(blk_m, N):
        sq_local[i, j] = x_local[i, j] * x_local[i, j]
    T.reduce_sum(sq_local, sumsq, dim=1)
    for i in T.Parallel(blk_m):
        sumsq[i] = T.rsqrt(sumsq[i] / N + eps)
    for i, j in T.Parallel(blk_m, N):
        x_local[i, j] = x_local[i, j] * sumsq[i]     # 原地写回，不再需要第二个 buffer
    T.copy(x_local, Y[bx * blk_m : (bx + 1) * blk_m, :])
```

与官方 `_vendor/tilelang-v0.1.15/examples/norm/rms_norm.py:39-62` 的结构基本一致：
官方单独留了一份 `A_pow_local` 存平方，本章直接在 `x_local` 上原地写回，
少占一份 fragment、多一次原地读写；`blk_m` 则被提成了参数。
数据流是：**global → fragment（一遍）→ 平方 → 归约 → 乘 rstd → fragment → global**。
中间结果全程在寄存器里，既没有第二遍 global 读，也没有额外的中间张量。

`blk_m` 的作用是摊薄「每行的固定开销」：跨线程归约的 barrier、`rsqrt` 的调用、
以及第二个 `T.Parallel` 循环的启动成本，都由 `blk_m` 行分摊。代价是寄存器压力
线性增长——`blk_m * N` 个 fp32 累加器。所以这个参数有一个由寄存器文件容量决定的
上限，不是越大越好。

#### 3.2.4 行太长时：split-K 两趟 + 倒序

如果 N 大到整行放不进 fragment，就把它切成 `num_k_step` 段
（官方 `_vendor/tilelang-v0.1.15/examples/norm/rms_norm.py:6-36`，本章 `code/03-norm` 里的 `rms_norm_splitk`）：

```python
T.clear(x_local)
for k in T.Serial(num_k_step):                    # 第一趟：只累加平方和
    T.copy(X[bx * blk_m, k * blk_k], x_sh)
    for i, j in T.Parallel(blk_m, blk_k):
        x_local[i, j] += x_sh[i, j] * x_sh[i, j]
T.reduce_sum(x_local, sumsq, dim=1)
for i in T.Parallel(blk_m):
    sumsq[i] = T.rsqrt(sumsq[i] / N + eps)

for k in T.Serial(num_k_step):                    # 第二趟：倒序，利用 cache
    T.copy(X[bx * blk_m, (num_k_step - 1 - k) * blk_k], x_sh)
    for i, j in T.Parallel(blk_m, blk_k):
        x_sh[i, j] *= sumsq[i]
    T.copy(x_sh, B[bx * blk_m, (num_k_step - 1 - k) * blk_k])
```

三个细节值得单独说：

1. **`T.clear(x_local)` 是必须的**。累加器要先清零，`+=` 才有意义；
   漏掉它得到的是一个依赖残留寄存器的错误结果（而且往往"看起来像随机噪声"）。
   官方源码里在同一个位置也有一行 `T.clear(A_local)`，并且带着注释
   `# reverse, better cache hit rate`。
2. **第二趟倒序**。第一趟结束时刚访问过的是最后一段，倒着走能让它还在 cache 里。
   这是纯访存的、零成本的优化——只要循环写成从 `num_k_step - 1` 往下。
3. **x 被读了两遍**。这是 split-K 的固有代价，账本上就要多算一遍；
   它换来的是「片上只留一个 `(blk_m, blk_k)` 的累加器」。所以它适合 N 很大
   （比如 8192 甚至 32768）的场景，N=1024 时应该用 §3.2.3 的写法。

官方示例在 `__main__` 里用的正是 `M, N, blk_m, blk_k = 8192, 8192, 1, 512` 这组
「大 N、小 blk_k」的参数（`_vendor/tilelang-v0.1.15/examples/norm/rms_norm.py:75-80`）。

#### 3.2.5 `T.reduce_*` 的用法要点

本章用到的四个归约：

| 调用 | 语义 | 典型用途 |
| --- | --- | --- |
| `T.reduce_sum(src, dst, dim=1)` | 沿 dim 求和，**覆盖写** dst | 平方和、`sum(x)` |
| `T.reduce_max(src, dst, dim=1)` | 沿 dim 取最大 | softmax 的 max、amax |
| `T.reduce_absmax(src, dst, dim=1)` | 最大绝对值 | 量化 scale |
| `T.reduce_min(src, dst, dim=1)` | 沿 dim 取最小 | 各种 clamp 边界 |

三条容易踩的语义：

1. **`clear` 默认是 `True`**，意思是"结果覆盖 dst"。写成 `clear=False` 才表示
   "把这次的结果和 dst 里已有的值合并"；而且它不是简单地把 dst 当累加器——
   跨线程归约会把同一个值计入多次，所以 `clear=False` 会先把 dst 拷进一个临时
   fragment，归约完再合并回去（`_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:94-109,247-253`、
   `_vendor/tilelang-v0.1.15/src/op/reduce.cc:179-182`）。这是为什么 `clear=False` 的 sum 会**多一份片上存储**。
   对 max/min 来说合并是幂等的，不存在这个问题。
2. **src 和 dst 的 scope 组合有四种**，前端会替你补上中间 fragment
   （`_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:94-171`）：shared→shared 建两个 fragment，
   shared→fragment 建一个，fragment→shared 建一个，fragment→fragment 直发。
   你在 kernel 里写的 `T.reduce_sum(shared_buf, fragment_out, dim=1)` 其实会多一次
   shared→fragment 的拷贝——所以**能直接在 fragment 上归约就直接做**。
3. **dst 的形状必须对**：要么是 `src.shape` 去掉 dim，要么是把 dim 留成 1。
   形状不对会在前端直接抛 `ValueError`，不会走到编译期。

### 3.3 实战三：RoPE 与多算子融合

#### 3.3.1 RoPE 是什么，为什么它值得单独占一节

RoPE（Rotary Position Embedding）把位置信息编码成一个**旋转**：对第 `p` 对维度，
按位置 `t` 的角度 `θ(t, p)` 把二维向量转过去。数学上没有新东西，但工程上有一个
特别容易出错的点——**"哪两个元素是一对"是训练时就定下来的**，而且有两种主流约定：

```text
interleaved（原版 RoFormer / non-neox）       half-split（NeoX / Llama / Qwen）
索引： 0 1 2 3 4 5 6 7                       索引： 0 1 2 3 │ 4 5 6 7
        └─┘ └─┘ └─┘ └─┘                              └─────────┘ └─────────┘
y[2p]   = x[2p]·c[p] − x[2p+1]·s[p]           y[p]   = x[p]·c[p] − x[p+H]·s[p]
y[2p+1] = x[2p]·s[p] + x[2p+1]·c[p]           y[p+H] = x[p+H]·c[p] + x[p]·s[p]
```

（`H = R/2`，`c/s` 就是预计算好的 cos/sin 表，形状 `(seq_len, R/2)`，
**一个 pair 占一行**。）

两种约定都能跑完、都不会报错，只是数值完全不同——所以后面 §5.3 会专门给一个
30 秒的自检方法。

另外两件事在真实模型里很重要：

* **只有一部分通道被旋转**（decoupled RoPE / partial RoPE）：比如 MLA 里只有
  `qk_rope_head_dim` 那么宽的一段被旋转。通用写法是
  `dim_start = hidden_size - rope_dim`（适配器
  `_vendor/tilelang-ascend-v0.1.2.000/examples/pos_embedding/rope.py:23,38`）。
* **cos/sin 表怎么算不是 kernel 的事**。频率基数、YaRN / Llama-3.1 的分段缩放、
  position id 的选取，全在 Op 层（Python）算好再传给 kernel
  （`../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/kernels/rope.py:1-20`）。
  kernel 只负责：读 x、读 cos/sin、写 y。

#### 3.3.2 PyTorch 参考实现

```python
def ref_rope_neox(x, cos, sin):
    half = x.shape[-1] // 2
    x_lo, x_hi = x.float()[..., :half], x.float()[..., half:]
    c, s = cos.float(), sin.float()
    return torch.cat([x_lo * c - x_hi * s, x_hi * c + x_lo * s], dim=-1).to(x.dtype)

def ref_rope_interleaved(x, cos, sin):
    head = x.float().reshape(*x.shape[:-1], -1, 2)
    x0, x1 = head[..., 0], head[..., 1]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2).to(x.dtype)
```

先写这个、先用它验一遍 `c=1,s=0`（应该原样返回 x）和 `c=0,s=1`（两种约定应该不同），
再动 TileLang。这是本章「先能用」的第一步。

#### 3.3.3 朴素版与优化版

**朴素版**：一行一个 block，一对一地读 `x[i, p]` 与 `x[i, p + H]`，算完直接写 global。

```python
with T.Kernel(S, threads=threads) as bx:
    for p in T.Parallel(half):
        c = T.Cast("float32", Cos[bx, p])
        s = T.Cast("float32", Sin[bx, p])
        lo = T.Cast("float32", X[bx, p])
        hi = T.Cast("float32", X[bx, p + half])
        Y[bx, p] = T.Cast(dtype, lo * c - hi * s)
        Y[bx, p + half] = T.Cast(dtype, hi * c + lo * s)
```

它是正确的，也是最容易看出"两种约定差在哪"的版本。两个问题：一是每行一个 block，
行数少的时候并发度不够；二是 cos/sin 被逐行从显存读，没有片上复用。

**优化版**：一个 block 吃 `blk_s` 行，并且把 x 与 cos/sin 都先搬进 shared。

```python
with T.Kernel(S // blk_s, threads=threads) as bx:
    x_sh = T.alloc_shared((blk_s, R), dtype)
    c_sh = T.alloc_shared((blk_s, half), "float32")
    s_sh = T.alloc_shared((blk_s, half), "float32")

    base = bx * blk_s
    T.copy(X[base : base + blk_s, :], x_sh)
    for i, p in T.Parallel(blk_s, half):
        c_sh[i, p] = T.Cast("float32", Cos[base + i, p])
        s_sh[i, p] = T.Cast("float32", Sin[base + i, p])
    for i, p in T.Parallel(blk_s, half):
        lo = T.Cast("float32", x_sh[i, p])
        hi = T.Cast("float32", x_sh[i, p + half])
        Y[base + i, p] = T.Cast(dtype, lo * c_sh[i, p] - hi * s_sh[i, p])
        Y[base + i, p + half] = T.Cast(dtype, hi * c_sh[i, p] + lo * s_sh[i, p])
```

三处改动，各自对应一笔账：

1. **块数少了 `blk_s` 倍**：`T.Kernel(S)` 变成 `T.Kernel(S // blk_s)`，每块的固定开销
   （启动、同步、cos/sin 的取数）被 `blk_s` 行分摊。
2. **cos/sin 在片上复用**：一行里 R/2 个系数被读进 shared 一次，之后所有 lane 从
   shared 取。这让 `Cos`/`Sin` 的**显存**读从「每行 R/2 次」变成「每行 R/2 个元素
   一次搬完」，虽然总量一样，但访存变成了规整的整块搬运。
3. **计算统一升到 fp32**，只在写回时 cast 一次。fp16 全程算的话，`x*c - y*s`
   会舍入两次；官方昇腾实现也这么做（`_vendor/tilelang-ascend-v0.1.2.000/examples/pos_embedding/rope.py:41-46` 先
   `T.copy(x_half_ub, x_ub)` 把 fp16 转成 fp32 再算）。代价是 fp32 中间量占的
   片上空间翻倍。

**注意两条 store 不是同一次向量访存**：`Y[..., p]` 与 `Y[..., p + half]` 的索引都
对 `p` 是步长 1 的仿射表达式，所以它们各自可以被向量化，但它们是两条独立的 store。
这正是为什么 half-split 约定对访存友好——如果换成 interleaved 约定，
按 pair 循环时两条 store 的步长就变成 2 了。

#### 3.3.4 融合：RMSNorm + RoPE (+ per-token 量化)

真实模型里这三个算子几乎总是连在一起（QKV 投影之后立刻归一化、旋转、量化写缓存），
所以它们是融合的最佳候选。先把账算出来：

```text
三个算子分别写  ：185.6 MB
融合成一个 kernel： 51.4 MB   （3.61×）
```

融合后的数据流（`code/05-fusion/rms_norm_rope_quant.py` 的
`rms_norm_rope_fused`）：

```text
x ──T.copy──▶ shared ──cast──▶ fragment(fp32)
  ──square + reduce_sum──▶ rstd ──乘 w[j]──▶ 写回 shared（降回 dtype）
  ──在 shared 上就地做尾部 RoPE──▶ T.copy 到 Y
```

三个设计决定，每个都有明确的理由：

* **RoPE 那一步必须绕回 shared**。NeoX 约定要求第 `p` 个和第 `p + H` 个元素同时可见，
  而它们在 fragment 里可能属于不同线程——寄存器之间不能互相访问，
  shared 是唯一能让"跨 lane 配对"成立的地方。
* **归一化之后立刻降回 dtype 再写进 shared**，而不是继续用 fp32 占着 shared。
  这直接决定 tile 能做多大。
* **加量化时（`rms_norm_rope_quant_fused`）要付第二次归约的代价**：
  按 128 一组取 amax。这里的写法是把一行 reshape 成 `(n_groups, group_size)` 的
  三维 fragment，再用 `T.reduce_max(..., dim=2)` 降一维——而不是手写
  read-modify-write 的循环，因为后者的 store 索引不随最内层循环推进，
  会被向量化规划器直接标量化
  （`_vendor/tilelang-v0.1.15/src/transform/loop_vectorize.cc:1050-1060` 里那段
  "is_store && (is_invariant || is_independent) → return {1, requires_scalarization=true}"）。

**融合的代价要说清楚**：三件事共享同一块 fragment 和 shared，`blk_m` 只能取 1，
而 `x_f32` 一个 `(1, 4096)` 的 fp32 fragment 就是 16 KiB。想加大 `blk_m` 就必须
二选一：按 K 切行（回到 §3.2.4 的两趟写法），或者改用 per-thread partial 归约
（TileOPs 的 RMSNorm 在小 N 且网格能占满设备时会走这条分支，
`../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/kernels/norm/rms_norm.py:30-52`）。

### 3.4 昇腾侧的同款实现

同一批算子在昇腾上有两个写法，差别不在算法，而在**语言面**。

#### 3.4.1 主干 v0.1.15 的昇腾方言

```python
with T.Kernel(num_blocks) as bx:                 # 一维 AI Core 网格，没有 threads=
    a = T.alloc_shared((tile,), "float32")       # 在昇腾上这就是 UB
    T.annotate_buffer_versions({a: num_stages, ...})
    for step in T.Pipelined(iterations, num_stages=num_stages):
        T.copy(A[begin : begin + tile], a)       # MTE2：DMA 搬进 UB
        with T.SimtVF(threads=2048):
            for i in T.Parallel(tile):           # Vector：在 UB 上向量化计算
                c[i] = a[i] + b[i]
        T.copy(c, C[begin : begin + tile])       # MTE3：DMA 搬出去
```

四个与 CUDA 侧不同的点：

1. **UB 就是 `T.alloc_shared`**：主干里没有 `T.alloc_ub`
   （`_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:15-23` 的 docstring 直接写 "Allocate a UB buffer"）。
2. **`T.Kernel` 没有 `threads=`**：它给的是"一维 AI Core 网格"
   （`_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:139-160`）；线程域要用 `T.SimtVF(threads=...)`
   另外开，或者用 `T.SimdVF()` 直接写向量寄存器。
3. **搬运与计算是两条流水线**：`T.copy`（MTE）+ `T.Pipelined` + 多缓冲，
   让"搬下一块"和"算这一块"重叠。这是 CUDA 侧逐元素 kernel 通常不需要、
   昇腾侧却很关键的一件事。
4. **tile 尺寸由 UB 容量反推**：官方 RMSNorm 示例先算 `ub_budget = 208 * 1024`，
   再据此挑一次处理几行、要不要复用输入 buffer
   （`_vendor/tilelang-v0.1.15/examples/ascend/example_rmsnorm.py:17-33`）。

#### 3.4.2 适配器 v0.1.2.000 的写法

适配器仓库里，同样的代码变成：

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):   # 多一个向量子核维度
    a_ub = T.alloc_ub((rows, block_N), dtype)              # 主干没有这个 API
    with T.Scope("V"):
        T.copy(A[bx, by], a_ub)
        T.barrier_all()
        T.tile.add(c_ub, a_ub, b_ub)                        # 直接下达 UB 上的向量指令
        T.barrier_all()
        T.copy(c_ub, C[bx, by])
```

#### 3.4.3 三个必须记住的差异

| 对象 | 主干 v0.1.15 | 适配器 v0.1.2.000 |
| --- | --- | --- |
| UB 的分配 | `T.alloc_shared`（`_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:15-23`） | `T.alloc_ub`（`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/allocate.py:156`） |
| 向量计算 | `T.SimtVF(threads=…)` + `T.Parallel`，或 `T.SimdVF()` + `T.simd.*` | `T.Scope("V")` + `T.tile.*`（`_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/__init__.py:86` 导出 `tile`） |
| 启动与同步 | `T.Kernel(*blocks)` 一维网格；同步由 pass 自动插入 | `T.Kernel(..., is_npu=True) as (cid, vid)`；`TL_ASCEND_AUTO_SYNC` 等要显式打开（`_vendor/tilelang-ascend-v0.1.2.000/tilelang/transform/pass_config.py:34-48`） |

这三条不是传闻，`code/07-api-audit/api_audit.py` 把它们做成了可执行的断言：
在主干 v0.1.15 的语言面上，`alloc_ub` 不存在，`tile` 命名空间也没有被任何
`__init__.py` 导出。把适配器的代码原样搬到主干，第一步就会 `AttributeError`。

---

## 4. 实现机制：编译器把这一行变成了什么

### 4.1 `T.Parallel` 是怎么变成向量访存的

你在 kernel 里写的是**块级逻辑**，让编译器去决定线程与向量宽度。对逐元素循环，
它要回答的问题是「最内层循环能不能一次处理 `k` 个元素」。判定链（源码在
`_vendor/tilelang-v0.1.15/src/transform/loop_vectorize.cc`）大致是：

1. **索引必须是可向量化的**：访问索引对最内层循环变量是仿射的、步长为 1
   （`IndicesCanVectorize`，`:1283`）。
2. **谓词必须在向量组内部恒定**：越界判断如果依赖 lane（比如 `idx < 2047`），
   就不能把 8 个元素合成一次访存（`IsExprInvariantInVectorBoundary`，`:1248-1281`）。
3. **跨 dtype 的 cast 取两边 lane 上限的公因数**
   （`CastNode` 分支，`:966-987`）：

   ```text
   fp32 → fp16 ： 源侧 128/32 = 4 lane，目标侧 128/16 = 8 lane → 取 4
   int32 → fp8 ： 源侧 4 lane，目标侧 16 lane → 取 4
   ```

4. **宽度上限还受目标属性约束**：`max_vector_bits` 以及
   `tl.disable_vectorize_256` 这类开关会把它压下来（`:1273-1276`）。
5. **store 的索引必须随向量轴推进**。如果一条 store 的索引在向量组内不变
   （典型的 read-modify-write 累加），规划器会直接把它标量化
   （`:1050-1060`）。

最后一步是把并行循环按算出来的宽度切到线程上（`_vendor/tilelang-v0.1.15/src/transform/loop_partition.cc:112-130`）。
这解释了官方文档里那个看起来"反直觉"的现象：`T.Parallel(threads)` 的 extent 只有
256，但编译器算出向量宽度是 8 之后，只需要 32 个线程干活，**而 launch config 依然是
256 线程**（corpus `elementwise.md:174-178`）。想让"每线程 8 个元素"这件事可控，
就显式写出 `T.Parallel(threads, num_per_thread)` 的二维形态。

### 4.2 归约的 lowering

`T.reduce_sum(src, dst, dim, clear)`（前端在
`_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:25-92`）先把参数校验成几件事：

* `dst` 的形状必须是「去掉 dim」或「dim 留成 1」；
* src/dst 的 scope 必须是 shared / fragment 的组合，四种组合各自补中间 fragment；
* 之后发射一条 `tl.tileop.reduce`，或者（当 src 是 local、dst 是 local.var 时）
  直接发射一条本地归约的 intrinsic。

真正生成代码的是 `_vendor/tilelang-v0.1.15/src/op/reduce.cc:168-198` 里描述的四步：

```text
① 可选初始化（clear=True，或 sum/abssum 需要清空）
② 线程内归约：展开的内层循环
③ 跨线程归约：tl::AllReduce<Reducer, threads, scale, ...>::run
   - 参与宽度 > 32 时，会通过 T.add_workspace 分配一块工作区
④ 把结果按 dst 的并行维度包一层循环写回；用了临时 clear buffer 就在这里分配
```

`tl::AllReduce` 的两条路径与四条编译期断言在
`_vendor/tilelang-v0.1.15/src/tl_templates/cuda/reduce.h:211-253`：`threads` 必须能被 `scale` 整除，
且 `threads / scale` 是 2 的幂。不满足会直接 `static_assert` 失败——
这个断言在编译期就报，不用等运行。

一个容易忽略的点：**fragment→fragment 的归约比 shared→fragment 少一次拷贝**。
`T.reduce_sum(shared_buf, frag_out)` 会在前端插入一个 `copy(shared_buffer,
red_frag_in)`（`_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:125-140`）。所以在归一化里，
正确做法是把数据先读进 fragment，然后在 fragment 上归约——这也正是 §3.2.3
比 §3.2.2 省一遍片内搬运的原因之一。

### 4.3 越界谓词从哪里来

`legalize_safe_memory_access` 会遍历所有可能越界的访问，给它们包上
`if_then_else(condition, access, safe_value)`
（`_vendor/tilelang-v0.1.15/src/transform/legalize_safe_memory_access.cc:419-477`）。两个细节：

* **安全值可配置**：第 09 章讲过的 `T.annotate_safe_value` 改的就是这里的默认值。
* **手工写了边界检查就会跳过自动插入**：源码里有一条分支专门检测
  "store 的值本身就是一个 `IfThenElse`"，此时不再叠加谓词
  （同文件 `:471-477`）。这解释了一个常见现象：**自己写的 `if idx < N` 与编译器
  自动插入的谓词，对向量化的影响完全不同**——前者在 IR 里就是一个显式的分支，
  后者是"访问谓词"。

### 4.4 文件与行号速查

| 你想知道 | 去哪看 |
| --- | --- |
| `T.Parallel` 的参数与 `coalesced_width` | `_vendor/tilelang-v0.1.15/tilelang/language/loop.py:13-33,73-74` |
| `T.copy` 的语义与 region 归一化 | `_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py:59-130` |
| `T.reduce_*` 的签名与 scope 分支 | `_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:25-92,94-171,230-259` |
| reducer（epoch）的语义 | `_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:358-460` |
| `T.fill` / `T.clear` | `_vendor/tilelang-v0.1.15/tilelang/language/fill_op.py:11-72` |
| 归约的 lowering 行为 | `_vendor/tilelang-v0.1.15/src/op/reduce.cc:168-198` |
| 归约的 layout 冲突报错 | `_vendor/tilelang-v0.1.15/src/op/reduce.cc:200-250` |
| `tl::AllReduce` 的实现与断言 | `_vendor/tilelang-v0.1.15/src/tl_templates/cuda/reduce.h:211-253,279-296` |
| 向量化判定链 | `_vendor/tilelang-v0.1.15/src/transform/loop_vectorize.cc:966-1000,1016-1066,1248-1281` |
| 越界谓词插入 | `_vendor/tilelang-v0.1.15/src/transform/legalize_safe_memory_access.cc:419-477` |
| 并行循环切到线程 | `_vendor/tilelang-v0.1.15/src/transform/loop_partition.cc:112-130` |
| 昇腾 UB 的分配 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:15-23` |
| 昇腾 `T.Kernel` 的网格语义 | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:139-160` |
| `T.SimtVF` / `T.SimdVF` | `_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py:107-140,146-190` |
| 适配器的 `T.alloc_ub` | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/allocate.py:156-157` |
| 适配器的 `T.tile` 命名空间 | `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/__init__.py:86` |

---

## 5. 进阶与陷阱

这一节按「现象 → 原因 → 怎么查」组织。所有条目都是本章示例里真实可能遇到的，
不是泛泛的提醒。

### 5.1 陷阱一：向量化被悄悄关掉

**现象**：kernel 能跑、结果也对，但生成代码里全是逐元素的 `A[i]`，没有 `uint4`/`float4`。

**四个常见原因**（对应 §4.1 的判定链）：

1. **边界谓词依赖 lane**。最典型的就是非对齐长度：官方文档里 `N=2047`、
   `threads=256`、每线程 8 个元素的例子生成的 CUDA 是

   ```c++
   #pragma unroll
   for (int i = 0; i < 8; ++i) {
     if (((i * 256) + ((int)threadIdx.x)) < 2047) { ... }
   }
   ```

   文档的原话是「TileLang did not apply optimizations such as vectorization or
   coalesced memory access」（corpus `elementwise.md:104-133`）。
   **怎么查**：把 `N` 改成整除的假数据跑一遍，如果生成代码突然出现 `uint4`，
   就确认是这个原因。
2. **最内层索引不是步长 1**：被向量化的那一维实际步长大于 1，例如访问的是一张
   转置视图或带 stride 的张量，或者 `loop_layout` 把最内层循环变量映射到了行索引。
   **怎么查**：把索引改写成 `(bx * threads + i) * num_per_thread + j` 这种
   「最内层步长为 1」的形态，再看生成代码有没有变化。
3. **跨 dtype cast 把宽度压到 4 或更小**。fp32→fp16 上限就是 4 lane。
   **怎么查**：算一下 `vector_load_bits_max / max(src.bits, dst.bits)`。
4. **store 的索引在向量组内部不变**，典型的 read-modify-write 累加：

   ```python
   for i in T.Parallel(N):
       acc[i // 8] += v[i]          # 向量宽度取 8 时，acc 的索引在组内恒定
   ```

   规划器会直接把这条 store 标量化（
   `_vendor/tilelang-v0.1.15/src/transform/loop_vectorize.cc:1050-1060` 里那段
   "store 的索引与循环变量无关 → return {1, requires_scalarization=true}"）。
   **怎么查**：把归约改写成「先落到一个与循环变量同步的 fragment，再用
   `T.reduce_*` 降维」。

还有一条"不是陷阱但是容易看不懂"的现象：即使向量化成功，`threads` 也可能只用到
一部分——因为编译器把 1 个元素的 256 线程改成了 8 个元素的 32 线程。
想看真实的线程利用率，要看生成代码里的 `if (threadIdx.x < ...)`，而不是 launch 参数。

### 5.2 陷阱二：归约的 `clear` 与初值

**现象 A**：`clear=True`（默认）时以为 dst 会被"累加"，结果发现旧值被覆盖了。
这是**正确行为**：`clear=True` 就是覆盖写。

**现象 B**：写了 `clear=False` 想自己控制初值，结果结果偏大或偏小，而且偏差随
线程数变化。原因见 §3.2.5：跨线程归约是 butterfly，同一个值会被计入多次；
`clear=False` 的实现是"先把 dst 拷进临时 fragment，再合并回来"
（`_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:94-109`、`_vendor/tilelang-v0.1.15/src/op/reduce.cc:179-182`），
所以它做的是**合并一次**，不是**累加 N 次**。

**现象 C**：`Reduce for shared memory not implemented.` 或
`Layout may conflict with ReduceOp for buffer ...`。前者是 src/dst 的 scope 组合
不合法；后者是 fragment layout 不相容（`_vendor/tilelang-v0.1.15/src/op/reduce.cc:200-250` 会打印出
`src_layout` / `orig_dst_layout` 供对比）。**怎么查**：先检查 dst 的形状，
再考虑加一层 shared 把 layout 转过来。

**现象 D**：`T.clear(x_local)` 忘了写，累加器带着上一次迭代的残留值。
split-K 写法里这一步是必须的（§3.2.4）。

### 5.3 陷阱三：RoPE 三种错法

| 错法 | 表现 | 怎么发现 |
| --- | --- | --- |
| 用错了配对约定（把 NeoX 当 interleaved） | 能跑、不报错，数值全错 | `c=1, s=0` 时应该原样返回 x；`c=0, s=1` 时两种约定输出必须不同 |
| 只旋转了整行而不是尾部 rope_dim | 能跑，前半段被破坏 | 和 PyTorch 参考逐元素比 |
| 全程 fp16 计算 | 能跑，误差比预期大一档 | 和 fp64 参考比，看误差是否超出该 dtype 的机器 eps 量级 |

第一条特别值得强调：**RoPE 的错误不会崩，只会让模型变傻**。所以本章的验证顺序
永远是"先对齐 PyTorch 参考，再看性能"。

### 5.4 陷阱四：非对齐长度

归一化算子里最容易被忽略的一条：为了对齐把行读到 `N_padded > N` 时，
padding 的列必须填 0（这样它对平方和没有贡献），而且**分母必须用 N 而不是
`N_padded`**。TileOPs 的 RMSNorm 在代码里专门写了一行注释：

```python
# N, not N_padded: the pad contributes zero to the sum.
for i in T.Parallel(block_m):
    rrms[i] = T.rsqrt(sumsq[i] / float(N) + eps)
```

（`../tilelang-corpus/12-operator-libraries/raw/TileOPs/src/tileops/kernels/norm/rms_norm.py:72-74`。）
用 `N_padded` 会让结果整体偏小，而且偏差随 N 的对齐情况变化——
典型症状是"单元测试用 1024 通过，用 1000 就失败"。

### 5.5 陷阱五：融合不是免费的

融合的收益写在账本上（3.5–5 倍访存），代价却藏在三个地方：

1. **寄存器压力**。`x_f32` 这样的 fp32 fragment 直接决定 occupancy；
   `blk_m` 一大，能同时在跑的 block 就少了。
2. **片上存储**。共享内存/UB 是固定的，放不下就得减 tile 或者减缓冲份数。
3. **layout 约束互相打架**。融合的算子越多，布局推导要同时满足的约束越多
   （比如「归约要求某种 fragment 划分」与「RoPE 要求 p 和 p+H 在同一个线程」）。
   这也是 §5.1 里 layout 冲突最常见的来源。

判断标准很简单：**先看账本能省多少，再看片上放不放得下**。如果省的那部分访存
相对片上开销只是一点点，那就不值得融合。

### 5.6 陷阱六：昇腾特有的三件事

1. **`T.alloc_ub` 在主干的 v0.1.15 里不存在**。写了会 `AttributeError`，
   而且报错位置在 Python 层，看起来像"环境装错了"。
2. **`T.tile.*` 也不在主干里**。第 09 章已经登记过这条；本章把正反两个方向都
   写成了脚本断言。
3. **`TL_ASCEND_AUTO_SYNC` 这类 pass config 只在适配器仓库里**。主干没有这些键，
   传了会被当成未知配置（未必报错，但也不会有任何效果）。

---

## 6. 本章小结

**一句话**：这三类算子的优化空间几乎全在访存上；融合的价值可以先在纸上算清楚，
再用"少搬 + 搬宽"两条原则去实现。

复习卡片：

| 问题 | 答案 |
| --- | --- |
| 怎么判断一个算子是不是 memory-bound？ | 算 `(flops, bytes)` 得算术强度，和机器平衡点比；本章三类算子都远低于平衡点 |
| 逐元素算子的两条优化路线？ | ① 减少重复访存（融合、把中间量留在片上）；② 加宽单次访存（每线程 n 个元素、128-bit） |
| `T.Parallel(threads)` 与 `T.Parallel(threads, n)` 的区别？ | 前者 extent 只有 `threads`，编译器可能自己重排到 32 线程 × 8 元素；后者显式声明每线程 n 个连续元素，写法可控 |
| 归约为什么是两段？ | 线程内先各自累加（寄存器，无同步），再做跨线程 butterfly（≥32 走 shared + barrier，<32 走 shuffle） |
| `clear=True` 与 `clear=False`？ | 前者覆盖 dst；后者与 dst 的旧值**合并一次**，sum/abssum 会多一份临时片段 |
| 归一化为什么要升 fp32？ | 累加与平方在低精度上会累积误差；官方示例统一用 fp32 累加、写回时再 cast |
| RoPE 的关键风险？ | 配对约定写错不会报错，只会数值全错；用 `c=1,s=0` 自检 |
| 融合的代价是什么？ | 寄存器压力、片上存储、layout 约束；tile 尺寸由容量反推 |
| 昇腾与 NVIDIA 的三处代码差异？ | UB 用 `alloc_shared`（主干）/`alloc_ub`（适配器）；线程域要显式开 `T.SimtVF`；tile 由 UB 容量决定 |
| 本章能在本机做哪些验证？ | `code/01-ledger` 账本模型、`code/07-api-audit` API 名核对；这两个是真跑过的 |

五张速查图：`figures/01-ledger-of-fusion.md`（账本）、
`figures/02-vectorization-decision.md`（向量化判定链）、
`figures/03-reduce-two-stage.md`（两段归约）、
`figures/04-rope-two-conventions.md`（两种配对）、
`figures/05-ascend-ub-pipeline.md`（昇腾 UB 数据流）。

---

## 延伸阅读

* 想复算本章的所有字节数：`code/01-ledger/ledger.py`（纯标准库，CPU 可跑）。
* 想看每个 API 名到底在不在锁定版本里：`code/07-api-audit/api_audit.py`。
* 想先打牢原语语义再回来看优化：第 09 章（`T.copy` / `T.reduce_*` 的降级行为）。
* 想理解 fragment / shared 与 layout 的关系：第 10 章。
* 想看生成代码、定位编译错误：第 12 章。
* 想把这里的 kernel 用到真实模型里：第 14–18 章（GEMM、Attention、端到端）。
* 昇腾侧的完整调优流程：第 19 章。
* 上游资料：官方 elementwise 教程（R1）、TileOPs 的归一化与 RoPE 实现（R5–R8）、
  TileLang-Ascend 的 RoPE 与融合示例（S42、S43）。完整清单与核验状态见
  [`references.md`](references.md)。
