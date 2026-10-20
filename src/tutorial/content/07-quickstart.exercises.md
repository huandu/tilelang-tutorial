# 第 07 章 习题

说明：

* 难度标注为 `★`（基础）/ `★★`（需要综合本章内容）/ `★★★`（推理链条较长或需要动手）。
* 「需要硬件」一栏中，`否` 表示只需要一台能跑 Python 的机器；`是（GPU）` / `是（NPU）`
  表示需要真实加速器。**需要硬件的题都标了「可选」**，本章不强制。
* 所有题目只依赖本章与更早章节（第 4、6 章）的知识；涉及后续章节时会显式提示。

---

## E1 概念题 ★ 不需要硬件

本章 3.1 节列出了 pip 安装与源码构建两套前置条件。请回答：

1. 哪一条要求对**两条路径**都成立？它的具体数值是多少？依据在哪个文件的哪一行？
2. 「CUDA >= 10.0」与「pip 提供的 CUDA 工具链 >= 13.0」分别在什么情况下用？
3. 一个人说：「我在 macOS 上按 3.3 节从源码构建，要编 CUDA 后端。」
   这句话哪里有问题？给出依据。

---

## E2 概念题 ★ 不需要硬件

本章 2 节说一个 kernel 剥到最里面是四个部件。请：

1. 把四个部件、它们在 Python 里的写法、以及它们"回答的问题"列成一张表。
2. 说明其中哪两个是"人写的"、哪两个是"编译器补的"，并各举一个本章代码里的例子。
3. 如果去掉「搬运」这个部件（即写成版本 A 那样完全不显式搬运），程序还成立吗？
   为什么？这说明四个部件里哪一个才是**必需的**？

---

## E3 概念题 ★★ 不需要硬件

`T.Kernel` 的 CUDA 方言有 `threads=` 参数。请回答：

1. 本章 4.2 节说 `threads=256` "不是总线程数"。那它是什么？给了默认值是多少？
2. 官方文档说 `T.Kernel` 里的代码运行在 **block level**，不是 thread level。
   请用本章 5.2 节的对照表说明：既然你从没写过 `threadIdx.x`，它为什么会出现在
   生成代码里？
3. 网格大小（`T.ceildiv(N, block_N)`）与线程数（`threads`）是两个独立的量吗？
   在版本 A 里 `block_N == threads == 256`，是巧合还是必须？

---

## E4 代码阅读题 ★★ 不需要硬件

读下面这段（本章 4.1 节版本 A 的核心，`N=1024, block_N=256, threads=256`）：

```python
with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
    for i in T.Parallel(block_N):
        C[bx * block_N + i] = A[bx * block_N + i] + B[bx * block_N + i]
```

请回答：

1. 这个网格有多少个 tile 程序？`bx` 的取值范围是什么？
2. 第 `bx` 个 tile 程序负责 `C` 的哪一段下标？请用 `bx` 写出这段区间的表达式。
3. `i` 是线程号吗？如果不是，它是什么？
4. 如果 `block_N` 从 256 改成 512、而 `threads` 保持 256，网格会变成几块？每个 tile
   程序负责的元素数会变吗？语义是否仍然正确？

---

## E5 代码阅读题 ★★ 不需要硬件

下面是官方文档给出的生成结果（`elementwise.md:155-172`），
Python 输入是 `elementwise_add(1024, threads=256, dtype=bfloat16)`，
`T.Parallel(threads)` 里是 `C[b_x*threads+i] = A[...] + B[...]`：

```c++
extern "C" __global__ void __launch_bounds__(256) main_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  if (((int)threadIdx.x) < 32) {
    uint4 __1;
      uint4 v_ = *(uint4*)(A + ((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 8)));
      uint4 v__1 = *(uint4*)(B + ((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 8)));
      ((nv_bfloat162*)(&(__1.x)))->x = (((nv_bfloat162*)(&(v_.x)))->x+((nv_bfloat162*)(&(v__1.x)))->x);
      ...   // 其余成对加法
    *(uint4*)(C + ((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 8))) = __1;
  }
}
```

请回答：

1. `blockIdx.x` 在 Python 里对应哪个名字？它的最大值是多少（`N=1024, threads=256`）？
2. `if (threadIdx.x < 32)` 里的 32 是怎么来的？256 个线程里另外 224 个去做什么了？
3. `uint4` 是多少字节？为什么一次能搬 `8` 个 `bfloat16`？
4. Python 里那个循环变量 `i` 去哪了？请在生成代码里找出它"变形后"的位置。
5. 如果作者本来想"256 个线程各做 1 个元素"，这份生成代码是不是违背了他的意图？
   官方文档对这种情况给出的建议是什么（提示：`num_per_thread`）？

---

## E6 代码阅读题 ★★ 不需要硬件

下面是同一个 kernel 在 `N=2047` 时的生成结果（`elementwise.md:122-130`）：

```c++
extern "C" __global__ void __launch_bounds__(256) main_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (((i * 256) + ((int)threadIdx.x)) < 2047) {
      C[((i * 256) + ((int)threadIdx.x))] = (A[((i * 256) + ((int)threadIdx.x))] + B[((i * 256) + ((int)threadIdx.x))]);
    }
  }
}
```

请回答：

1. 这个边界判断是谁加的？Python 里出现过类似的条件吗？
2. 与 E5 的向量化版本相比，这一版每个线程处理几个元素？访存宽度是多少字节？
3. 官方文档对这份代码的评语是什么（用一句话概括）？为什么边界判断会挡住向量化？
4. 由此回答：为什么"tile 尺寸要和 shape 对齐"？这是性能问题还是正确性问题？

---

## E7 动手题 ★★ 不需要硬件（可在 CPU 上做静态检查）

把本章 4.1 节版本 A 的**一维**向量加改成**二维**版本：输入 `A`、`B`、输出 `C` 都是
`(M, N)`，输出满足 `C[m, n] = A[m, n] + B[m, n]`。要求：

1. 写出完整的 TileLang kernel 骨架（`@T.prim_func` + `T.Kernel` + `T.Parallel`）。
2. 明确写出 `T.Kernel` 两个维度参数的顺序：哪个是 `bx`、哪个是 `by`，
   并对应到哪一维。
3. 说明如果你把 `T.Kernel` 的两个参数写反，会发生什么（会报错吗？会算错吗？）。

（提示：本章 4.4 节有官方二维示例；注意参数顺序是"N 在前、M 在后"。）

---

## E8 动手题 ★★ 不需要硬件（可在 CPU 上做静态检查）

从版本 A 出发，改写成版本 B：先 `T.alloc_fragment` 分配三个 buffer，用 `T.copy` 把
`A`、`B` 的 tile 搬进来，算完再 `T.copy` 搬回 `C`。请：

1. 写出改动后的 kernel 骨架。
2. 逐条说明 `T.copy(A[bx * block_N], A_local)` 里，源只写了 `A[bx * block_N]`
   （一个"首地址"），编译器是怎么知道要搬多少个元素的。
3. 版本 A 与版本 B 的**计算结果**是否相同？它们在语义上真正的差别是什么？

---

## E9 概念题 ★★ 不需要硬件

`@tilelang.jit` 有 lazy 与 eager 两种模式，自动推断。请回答：

1. 判断依据是什么？（看被修饰函数的哪一个特征？）
2. 分别写出两种模式的一段最小代码，并指出「编译在哪一行发生、执行在哪一行发生」。
3. 什么场景该选 lazy？什么场景该选 eager？各举一个本章提到的官方示例。

---

## E10 概念题 ★★★ 不需要硬件

本章 6.3 节给出了缓存 key 的构成。假设你的 kernel 是版本 A，参数
`N=1<<20, block_N=256, threads=256`。有人做了下面四种改动，请分别判断"缓存命中/
不命中"，并说明依据是 key 里的哪个字段：

1. 只在 kernel 的 docstring 里加了一行注释。
2. 把 `block_N` 从 256 改成 512。
3. 把 `target` 从 `"cuda"` 改成 `"ascend"`。
4. 改了 `src/transform/` 里某个 C++ pass 的实现（`pass_configs` 不变，
   且 `TILELANG_KERNEL_CACHE_USE_LIB_STAMP` 没打开）。
5. 对第 4 种情况，给出两种"强制重新编译"的办法。

---

## E11 动手题 ★★ 不需要硬件

写一段 Python 脚本（可以在本机跑，不需要 tilelang）：

1. 打印"Python ↔ 生成 CUDA"的对照条目，至少覆盖：网格 → `blockIdx` /
   `__launch_bounds__`；`T.Parallel` → 线程 + 向量宽度；`T.copy` → 向量化访存。
2. 打印一份检查清单，列出"拿到真实机器上的 `get_kernel_source()` 输出后，按顺序
   该看的 5 个地方"。

（参考本章 `code/02-inspect-source/inspect_source.py`，但请自己组织内容。）

---

## E12 概念题 ★★ 不需要硬件

本章第 7 节写了昇腾侧的第一个 kernel。请回答：

1. 与 CUDA 版相比，列出**三处结构性差异**，每处说明"为什么它不是语法糖"。
2. 昇腾的 `T.Kernel` 如果传两个维度参数会怎样？依据在哪个文件的哪个函数？
3. 主干 `tilelang.ascend.language` 与独立仓库 `tilelang-ascend` 的方言，至少列出
   三处不同写法，并说明"为什么不能把独立仓库的示例直接抄到主干上"。

---

## E13 代码阅读题 ★★ 不需要硬件

下面是主干昇腾 `T.Kernel` docstring 里自带的最小例子
（`_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:165-168`）：

```python
with T.Kernel(NUM_CORES) as bx:
    with T.SimtVF(threads=128):
        for i in T.Parallel(128):
            out[bx * 128 + i] = x[bx * 128 + i] * 2.0
```

请回答：

1. `NUM_CORES` 与 `128`（`T.SimtVF` 的 `threads`）分别是什么？两者是同一层的东西吗？
2. 为什么 CUDA 版写 `T.Kernel(..., threads=128)`，昇腾版却要分成两层？
3. 这段代码里没有任何 `T.copy`。这说明前面 E2 的"四个部件"里哪一个对昇腾也是可选的？

---

## E14（可选，需要硬件）★★★

在一台有 NVIDIA GPU 的机器上（或一台昇腾机器上，二选一）：

1. 按本章第 3 节装好对应的 TileLang，并用 `code/06-env-check/check_env.py`
   确认环境。
2. 跑 `code/01-first-kernel/vecadd_tilelang.py`（或昇腾版 `code/05-ascend-first-kernel/vecadd_ascend.py`），
   把它打印的源码保存下来。
3. 与本章 `code/02-inspect-source/` 打印的两份官方生成结果对比：命中"向量化"还是
   "带边界判断"？把差异点逐条写下来。
4. 把 `N` 改成 2047，重跑一次，再用 `get_kernel_source()` 看源码变化。
5. 记录你的环境信息（Python 版本、CUDA 版本 / CANN 版本、GPU / NPU 型号、
   `tilelang.__version__`）。**不要写任何你没有实测过的性能数字。**

（提示：这是一道"测量口径自洽"题——环境、shape、dtype、backend 都要记全，
下次才能复现。）
