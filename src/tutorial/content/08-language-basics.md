# 第 08 章 语言基础

## 学习目标

读完本章，你应当能够：

1. 说清 `T.Kernel` 的每一个参数：位置参数是**块网格**沿 x/y/z 的 extent，返回值
   是该轴的块索引；`threads=` 是每块的 SIMT 线程数（省略时 SIMT 后端默认 128）；
   并能写出 1 维、2 维、3 维网格以及块内取索引的三种写法。
2. 说出 v0.1.15 支持的数据类型家族（bool / 有符号与无符号整数 / float16、
   bfloat16、float32、float64 / float8 八种 / float6 两种 / float4 两种存储模型 /
   tfloat32），以及 `'float16'`、`T.float16`、`torch.float16` 三种写法为什么会
   归一到同一个类型；并能正确写出 fp8 / fp4 的 dtype。
3. 按“谁看得到这份数据”“它活多久”两条标准，在
   `T.alloc_shared` / `alloc_fragment` / `alloc_local` / `alloc_var` 之间做出选择，
   并解释为什么 `alloc_fragment` 写成块形状却落在寄存器里；同时认得昇腾侧
   `T.alloc_l1` / `alloc_l0a` / `alloc_l0b` / `alloc_l0c` 与 UB 的分配方式。
4. 说出 `T.serial` / `T.unroll` / `T.vectorized` / `T.Parallel` / `T.Pipelined`
   各自解决什么问题，并能复述 `T.Parallel` 的默认维度映射规则
   （lane → thread → 每线程串行复制）。
5. 在 `if/else`、`T.if_then_else`、`T.Select`、`while` 之间做出正确选择，并说出
   控制流对向量化与软流水造成的两种典型代价。
6. 判断一段 Python 代码在 kernel 里能不能写：知道哪些语法会被 trace 期展开、
   哪些会变成设备循环、哪些直接不可用；并知道“编译期值”与“设备值”混用是
   最常见的坑。

## 前置知识

* 第 3 章：tile / dataflow / schedule 三层解耦，以及“数据流留给人、调度空间
  交给编译器”这条主线。
* 第 6 章：`grid → block → warp → thread`、NV 的 shared memory、昇腾的
  UB / L1 / L0A / L0B / L0C 与 MTE 路径。本章会直接使用这些机器图，不再重复。
* 第 7 章：装好 TileLang，写出并跑通第一个 kernel，会打印生成的 CUDA 源码。
* 会用 Python；写过 PyTorch；**不需要**写过 CUDA。

> **本章的验证边界。** 写作环境是 macOS（Apple Silicon）、Python 3.11，
> **没有 NVIDIA GPU，也没有昇腾 NPU**（见 [../../VERSIONS.md](../../VERSIONS.md)）。
> 本章所有需要加速器的代码示例都只做到“**仅静态核对**”：`python3 -m py_compile`
> 通过，且每个 API 名与签名都在 `_vendor/tilelang-v0.1.15/` 里逐条读过。
> 凡是在本机真正运行过的脚本（都是纯标准库的小模型），会明确写成
> “**CPU 逻辑验证**”。本章**不含任何实测性能数字**。

---

## 1. 问题的由来：kernel 里到底要“表达”什么

在 PyTorch 里写 `y = x * 2`，你表达的是**数学意图**。写 TileLang kernel 时，
你要表达的是一组**工程事实**。这些事实在 PyTorch 里由框架替你决定，在 kernel
里必须由你说清楚：

1. **这次 launch 有多少个块、每块多少线程。** 决定硬件怎么把工作铺开，也决定
   你能用多少并行度。
2. **每个数的位宽与语义。** fp16 和 bf16 都占 2 字节，但一个截尾数、一个截指数；
   fp8 有八种拼法，选错了不是“精度差一点”，而是数值范围完全不同。
3. **每份数据住在哪一级存储、谁能看见它。** shared 是块内共享的暂存区，
   fragment 名义上也是块形状但实际切到寄存器，`alloc_var` 只是一个标量。
4. **哪些迭代是并行的、哪些是顺序的、哪些要流水。** 同一段循环写成
   `T.Parallel` 和写成 `T.serial` 是两种完全不同的硬件行为。
5. **什么条件下走哪条路。** 边界处理、稀疏分支、提前退出。
6. **哪些是 Python，哪些是设备代码。** 这是 TileLang 作为“Python 嵌入 DSL”
   独有的边界问题：同一个 `with` 块里混着两种语义的东西，分不清就会写出
   “看起来对、编译不出来”或者更糟的“编译出来但语义错了”的代码。

第 7 章你已经跑通过第一个 kernel，看过它生成的 CUDA 源码。本章要做的是把那份
源码往回拆：**每个语言构造对应到 IR 里的什么、代价是什么、什么时候该用哪个。**

阅读本章时请始终记住第 4 章的那条管线：你写的 Python 先被解析成 TIR，
再经过一系列 pass 变成具体后端的代码。本章每一节都会给出“你写的”和
“编译器看到的”两侧。

---

## 2. 概念与模型：六个构件

把整章压成一张表。一个 kernel 里必须显式表达的东西，只有六类：

| 构件 | 语言表面 | 落到 IR 里是 | 本章 |
| --- | --- | --- | --- |
| 定义与网格 | `with T.Kernel(gx, gy, ...) as (bx, by)` | thread-binding frame（`blockIdx.x/y/z` + `threadIdx.x/y/z`） | §3 |
| 数据类型 | `'float16'` / `T.bfloat16` / `torch.float32` | `tvm.DataType` | §4 |
| buffer 与作用域 | `T.alloc_shared / alloc_fragment / alloc_local / alloc_var` | `Buffer` + `storage_scope` | §5 |
| 迭代 | `T.serial / T.unroll / T.vectorized / T.Parallel / T.Pipelined` | `For` 节点，`ForKind` 不同 | §6 |
| 控制流 | `if/else`、`x if c else y`、`T.if_then_else`、`while` | `IfThenElse` 语句 / `if_then_else` 表达式 / `While` | §7 |
| Python 边界 | 哪些语法能用、哪些会展开、哪些不行 | trace 期求值 vs 运行期 TIR | §8 |

结构关系见 `figures/01-kernel-anatomy.md`。

**一条贯穿全章的分界线：trace 期的 Python 与运行期的 TIR。**
你在 kernel 体里写的每一行，要么在**构造 IR 的时候**（trace 期）就被求值掉，
要么变成**设备上真正执行**的 TIR 语句。`for k in [0, 1, 2]` 是第一种（trace 期
把循环体复制三份），`for i in T.serial(4)` 与 `for i in range(4)` 都是第二种；
`if True:` 是第一种，`if i < N:`（其中 `i` 是 TIR 变量）是第二种。
本章 §6–§8 的所有坑，几乎都能归到“我以为是这一种，其实是那一种”。

---

## 3. kernel 定义与网格：`T.Kernel`

### 3.1 参数含义

`T.Kernel` 的定义在
`_vendor/tilelang-v0.1.15/tilelang/cuda/language/kernel.py:13-55`：

```python
def Kernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple[int, ...] | None = None,
    prelude: str | None = None,
    cluster_dims: int | tuple[int, int, int] | list[int] | None = None,
) -> KernelLaunchFrame:
```

四个参数各自的含义：

* **`*blocks`（位置参数，1–3 个）**：块网格沿每个轴的 extent，对应
  `gridDim.(x|y|z)`。它返回的绑定就是这个轴上的块索引。**顺序就是轴的顺序**：
  `T.Kernel(gx, gy)` 的 `gx` 是 x 轴，`gy` 是 y 轴。
* **`threads=`（关键字）**：每块的线程数。可以是一个整数（写进 `blockDim.x`），
  也可以是长度 ≤ 3 的列表/元组（`blockDim.(x|y|z)`）。**省略时，SIMT 后端用
  `DEFAULT_SIMT_THREADS = 128`**（`tilelang/transform/__init__.py:201`）。
* **`prelude=`**：注入到生成代码之前的源码（比如 `#include` 或辅助函数）。
* **`cluster_dims=`**：CUDA 的线程块簇（SM90+）。`2` 或 `(2,1,1)` 表示
  2-CTA 簇。`T.ClusterKernel` 是要求必须给 `cluster_dims` 的同款 launch。

这四条**不是**“随便什么关键字都能塞”：每个后端方言用自己 `Kernel` 的关键字
参数声明它认识哪些注解，其它一律拒绝。CUDA 方言认 `threads` / `prelude` /
`cluster_dims`；ROCm 与 Metal 认 `threads` / `prelude`；CPU 只认 `prelude`。

### 3.2 网格是“块”的网格，不是“线程”的网格

这是从 PyTorch 走过来最容易带错的观念。**`T.Kernel` 里的一切默认是块级的**：

```python
with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
    for i in T.Parallel(128):        # 不是 "threadIdx.x 线程做第 i 件事"
        C[bx * 128 + i] = A[bx * 128 + i] + B[bx * 128 + i]
```

`T.Parallel(128)` 的 128 是**逻辑迭代数**，不是“128 个线程各做一次”，
虽然在这个例子里结果恰好一样。编译器的布局推导会决定把这 128 次迭代切成
“哪几条 lane、哪些线程、每线程几次”。所以你不需要、通常也不应该手写
`threadIdx`。真需要线程级行为时，CUDA 方言提供
`T.get_thread_binding(dim=0)`（`tilelang/language/kernel.py:587`），
而昇腾侧要先开一个 `T.SimtVF(threads=...)` 区域（见 3.5）。

### 3.3 最小示例：一维网格

`code/01-kernel-grid/kernel_grid.py` 的第一个函数：

```python
@tilelang.jit
def elementwise_scale(N, block=256, dtype="float32"):
    @T.prim_func
    def kernel(X: T.Tensor((N,), dtype), Y: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:   # ①
            for i in T.Parallel(block):                            # ②
                gi = bx * block + i                                # ③
                Y[gi] = X[gi] * 2.0                                # ④
    return kernel
```

逐行：

* **①** `T.ceildiv(N, block)` 是 x 轴的块数，上取整，所以最后一轮可能盖住
  一部分越界位置（由后面的守卫处理）。`threads=block` 把每块线程数定死成
  `block`；`as bx` 拿到 x 轴块索引，等价于 `blockIdx.x`。
* **②** `T.Parallel(block)` 声明“这一块要做的 `block` 件事”。它是块级意图，
  映射由编译器做。
* **③** 全局下标 = 块号 × 每块元素数 + 块内偏移。这就是所有 elementwise
  kernel 的通用公式。
* **④** 直接写全局内存。TileLang 会为 `gi < N` 自动插边界守卫
  （`LegalizeSafeMemoryAccess`），所以这里省掉了显式 `if`。

### 3.4 二维与三维网格

二维网格就是把 N 方向给 x、M 方向给 y，这是 `examples/quickstart.py` 的写法：

```python
with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
    T.copy(A[by * BM, ko * BK], A_s)
    T.copy(B[ko * BK, bx * BN], B_s)
```

返回的是**列表**（多个轴时），可以解包；只有一个轴时返回单个 Var，写成
`as bx` 或 `as (bx,)` 都行。这个行为写在
`tilelang/language/kernel.py:88-96` 的 `_normalize_bindings` 里。

块索引的分解遵循“**x 变得最快**”：

当网格为 $(g_x,g_y,g_z)$ 时，一次 launch 的线性块号 $p$ 分解为

$$
\begin{aligned}
b_x &= p \bmod g_x,\\
b_y &= \left\lfloor \frac{p}{g_x} \right\rfloor \bmod g_y,\\
b_z &= \left\lfloor \frac{p}{g_x g_y} \right\rfloor.
\end{aligned}
$$

这不是 TileLang 自己的约定，而是 CUDA 的 `blockIdx` 顺序。TileLang 只是在
`src/ir.cc:236-262` 里给每个轴建一个标签为 `blockIdx.x/y/z` 的绑定帧。

图见 `figures/02-grid-mapping.md`；本目录
`code/01-kernel-grid/grid_index_model.py`（CPU 逻辑验证）把上表算了出来：
`M=1000, N=700, BM=256, BN=128` → `grid=(6,4)`、24 个块、M 方向多 24 行、
N 方向多 68 列。

### 3.5 昇腾侧：一维核网格 + 核内线程域

昇腾方言的 `Kernel` 是另一个形状
（`tilelang/ascend/language/kernel.py:139-172`）：

```python
def Kernel(*blocks, prelude=None) -> KernelLaunchFrame:
    if len(blocks) != 1:
        raise ValueError(f"Ascend targets a 1-D core grid: T.Kernel(N) takes exactly one grid extent. ...")
```

* **没有 `threads=`**：NPU 的核网格是一维的，核内线程域要显式开。
* **多维网格直接报错**，而不是被悄悄压平。
* 线程域用 `with T.SimtVF(threads=128):`（真实的 `threadIdx` 作用域）或
  `with T.SimdVF():`（寄存器级 SIMD，没有线程维）打开。

这解释了第 6 章那张机器图在语言层的落点：**NV 是“块网格 + 块内隐式线程”，
昇腾是“核网格 + 核内显式线程域”。**

### 3.6 块内取索引的三种写法

```python
with T.Kernel(N, M, B, threads=(64, 2)) as (bx, by, bz):
    tx, ty, tz = T.get_thread_bindings()   # ① threadIdx.x / y / z（固定三个）
    n_blk_x = T.get_block_extent(0)    # ② 该轴的块数
    bx2 = T.get_block_binding(0)       # ③ 等价于 bx
```

* ① `T.get_thread_binding(dim)` / `T.get_thread_bindings()`：线程索引。
  **返回的永远是三个**（x、y、z），`threads=(64, 2)` 时第三个是 1。
  注意一个限制，**`T.Kernel` 没写 `threads=` 时，trace 期不知道线程数**，
  此刻调 `get_thread_extent()` 会抛
  `"The thread extent is not known at trace time"`（`kernel.py:215-226`）。
* ② `T.get_block_extent(dim)`：该轴的网格 extent；没启动的轴返回 1
  （`kernel.py:199-208`）。
* ③ `T.get_block_binding(dim)`：该轴的块索引 Var，和 `as` 绑定拿到的是同一个
  东西。

`threads=(64, 2)` 这种按维度的写法，会在 `_normalize_threads`
（`kernel.py:99-135`）里补齐成三维：`[64, 2]` → `[64, 2, 1]`。

---

## 4. 数据类型

### 4.1 三种写法，一个类型

`docs/programming_guides/type_system.md:7-11` 列出了三种写法，它们**等价**：

```python
dtype = "float16"           # 字符串
dtype = T.float16           # TileLang dtype 对象
dtype = torch.float16       # 框架 dtype
```

归一化在 `tilelang/language/dtypes.py` 里做：字符串走
`_CANONICAL_TO_DISPLAY_STR`（`:105`），把 TVM 的规范名换成显示名
（`float → float32`、`half → float16`、`int → int32`、`long → int64`、
`short → int16`、`double → float64`、`uint → uint32`）；torch / numpy 的
dtype 走 `_TORCH_DTYPE_TO_STR` / `_NUMPY_DTYPE_TO_STR` 映射表；最后都进
`dtype.__new__`（`:245-253`）得到同一个 `tvm.DataType`。

**这条规则的实际好处**：你不用为了“传参”和“写 buffer”记两套名字。
`torch.randn(..., dtype=torch.bfloat16)` 得到的张量，直接喂给
`T.Tensor((N,), T.bfloat16)` 的 kernel 就行；第 7 章的 JIT 会用张量的实际
dtype 去推断目标与类型。

### 4.2 家族清单

v0.1.15 里 `tilelang.language` 导出的 dtype 名字一共 **173 个**
（`dtypes.py:891` 的 `__all__` 由 `_all_dtypes` 展开）。按家族分：

| 家族 | 成员 | 向量形（lane） |
| --- | --- | --- |
| 布尔 | `bool` | 无 |
| 有符号整数 | `int4`、`int8`、`int16`、`int32`、`int64` | `int8/16/32/64` 有 `x2…x64`；`int4` 无 |
| 无符号整数 | `uint8`、`uint16`、`uint32`、`uint64` | 各有 `x2…x64` |
| 浮点 | `float16`、`bfloat16`、`float32`、`float64` | `float16/32/64` 有 `x2…x64`；`bfloat16` 只有 `x2` |
| 张量浮点 | `tfloat32`（`custom[tfloat32]`） | `x2…x64` |
| float8 | `float8_e3m4`、`float8_e4m3`、`float8_e4m3b11fnuz`、`float8_e4m3fn`、`float8_e4m3fnuz`、`float8_e5m2`、`float8_e5m2fnuz`、`float8_e8m0fnu` | 各有 `x2…x64` |
| float6 | `float6_e2m3fn`、`float6_e3m2fn` | 各有 `x2…x64` |
| float4 | `float4_e2m1fn`（打包）、`float4_e2m1_unpacked`（SMEM） | `float4_e2m1fn` 有 `x2…x64`；`unpacked` 无 |

外加 7 个**规范别名**：`short`、`int`、`long`、`uint`、`half`、`float`、
`double`。

这张表不是抄文档抄来的：`code/02-dtypes/dtype_families.py --diff` 会去读
`dtypes.py` 的 `_all_dtypes`，逐条比对。当前输出是 `exact match`（173 条）。
图见 `figures/03-dtype-families.md`。

> **注意 `bfloat16` 的特殊性。** 它只有 `bfloat16` 和 `bfloat16x2` 两个名字，
> 没有 `bfloat16x4` 及以上。如果你下意识地写 `T.bfloat16x8`，那是错的。
> 需要“一次处理 8 个 bf16”时，只能靠循环或展开表达，或者改用
> `float8`（如果数值范围允许）。

### 4.3 fp8：八个拼法不是一回事

fp8 是“8 位浮点”的统称，E4M3 与 E5M2 的**数值范围完全不同**；同一份 E4M3 在不同
平台还有不同的 NaN 规则。列出八个成员时，要同时说清“指数/尾数怎么分”和
“NaN / 溢出怎么处理”：

| dtype | 指数 / 尾数 | 备注 |
| --- | --- | --- |
| `float8_e3m4` | 3 / 4 | 早期格式 |
| `float8_e4m3` | 4 / 3 | 抽象 E4M3，NaN 规则由后端决定 |
| `float8_e4m3b11fnuz` | 4 / 3 | bias-11 的 FNUZ 变体 |
| `float8_e4m3fn` | 4 / 3 | 有限值 + NaN，**CUDA 侧的默认拼法** |
| `float8_e4m3fnuz` | 4 / 3 | FNUZ，**ROCm 侧的默认拼法** |
| `float8_e5m2` | 5 / 2 | 指数范围更大，精度更低 |
| `float8_e5m2fnuz` | 5 / 2 | FNUZ 变体 |
| `float8_e8m0fnu` | 8 / 0 | **只有指数**，用作 MX 的块 scale 因子 |

不想背这张表也可以：`tilelang/language/fp8.py` 提供了
`determine_fp8_type(fp8_format="e4m3")`，它按运行时平台返回正确的拼法，
CUDA 返回 `float8_e4m3fn`，ROCm 返回 `float8_e4m3fnuz`，gfx950 例外地回到
OCP 拼法。

```python
in_dtype = T.float8_e4m3fn
accum_dtype = T.float32        # 累加 dtype 必须显式给
```

**混合精度时累加 dtype 永远是显式的。** `type_system.md:38-39` 把这条写在
“Notes” 里：要 fp16 输入、fp32 累加，就把 `accum_dtype` 写成 `T.float32`，
不要指望推导。

### 4.4 fp4：两种存储模型

FP4 E2M1 在 v0.1.15 里有**两个** dtype，区别不在数值，在**存储**：

| dtype | bits | 用法 |
| --- | --- | --- |
| `T.float4_e2m1fn` | 4 | **打包**：一字节两个 FP4 值，全局张量与打包 SMEM 往返用它，占一半空间 |
| `T.float4_e2m1_unpacked` | 8 | **解包**：一个 FP4 值占一个字节槽，只用于 SMEM/TMA，配对 tcgen05 的 `kind::f8f6f4` / `mxf8f6f4` 混合精度路径 |

两者的 docstring 写在 `dtypes.py:268-304`。`is_float4()` 对两者都返回 `True`；
`is_float4_e2m1fn()` / `is_float4_e2m1_unpacked()` 才区分它们。用错模型的
典型症状是“访存对不上”或者“TMA 描述符对不上”。因为它们对应不同的
`CU_TENSOR_MAP_DATA_TYPE_*`。

还有一个实际约束：**PyTorch 没有 `float4_e2m1fn` 这个 dtype**，所以
`as_torch()` 会退到 `float4_e2m1fn_x2`（打包版）或 `int8`
（`dtypes.py:187-236`）。传张量进去时要注意这一点。

### 4.5 混合 A/B dtype 的硬约束

`dtypes.py:319-340` 的 `validate_gemm_ab_dtypes` 规定了一条硬约束：

```text
A 与 B 的 dtype 不同时：
  - 只有「f8f6f4 家族」之间允许混用（且调用方要显式 opt-in）
  - 其它任何组合都会抛 ValueError
```

所以 `T.gemm` 里 A 用 fp8、B 用 fp16 是**不允许**的；要混精度就在 fp8 家族内部
混（比如 fp8 × fp4），或者先转成同一类型。

---

## 5. buffer 与内存作用域

### 5.1 四个常用分配函数

`tilelang/language/allocate.py` 里的签名（括号里是默认 scope）：

| API | 默认 scope | 语义 | 行号 |
| --- | --- | --- | --- |
| `T.alloc_shared(shape, dtype)` | `shared.dyn` | 块内共享的片上暂存区 | `:49` |
| `T.alloc_fragment(shape, dtype)` | `local.fragment` | 逻辑上是块形状，物理上切到每线程寄存器 | `:81` |
| `T.alloc_local(shape, dtype)` | `local` | 每线程私有、普通下标空间 | `:67` |
| `T.alloc_var(dtype, init=...)` | `local.var` | 单元素标量 | `:103` |

还有三个不常用但要知道的：

* `T.alloc_global(shape, dtype)`（`:171`）：后端直接分配的工作区
  （`cudaMalloc` 风格）。docstring 明确建议**优先把工作区从 Torch 侧传进来**，
  这个 API 主要用于测试和特定场景。
* `T.alloc_tmem(shape, dtype)`（`:254`）：Blackwell 的 Tensor Memory，
  scope `shared.tmem`。约束写在 docstring 里：列数必须是 2 的幂且 ≥ 32，
  分配与释放必须由同一个 warp 执行，只有 TCGEN5.MMA 和特定的 TMEM 载入/存储
  指令能碰它。
* `T.alloc_barrier(arrive_count)`（`:194`）：mbarrier，scope `shared.barrier`。

### 5.2 怎么选

选择依据只有两条：**谁看得到这份数据**，以及**它要活多久**。

```text
放一个 tile 的暂存副本，块内所有线程共享         → alloc_shared
放一个"块形状"的累加器 / 中间结果                → alloc_fragment
放一个每线程私有、需要自己下标的小数组            → alloc_local
放一个标量：计数器、标志、临时值                 → alloc_var
放一块跨块的工作区（少见）                       → alloc_global
放 Blackwell 的 TMEM 累加器                      → alloc_tmem
```

最小示例（`code/03-memory-scopes/alloc_scopes.py`）：

```python
with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
    A_s = T.alloc_shared((BM, BK), "float16")     # ①
    B_s = T.alloc_shared((BK, BN), "float16")     # ①
    C_f = T.alloc_fragment((BM, BN), "float32")   # ②
    scratch = T.alloc_local((BK,), "float32")     # ③
    scale = T.alloc_var("float32", init=0.5)      # ④
    T.clear(C_f)
    for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
        T.copy(A[by * BM, ko * BK], A_s)
        T.copy(B[ko * BK, bx * BN], B_s)
        T.gemm(A_s, B_s, C_f)
    for i, j in T.Parallel(BM, BN):
        C_f[i, j] = C_f[i, j] * scale[0]          # ⑤
    T.copy(C_f, C[by * BM, bx * BN])
```

* **①** A/B 是“搬上来、反复用”的 tile，需要块内共享 → `alloc_shared`。
* **②** C 是累加器。写成 `(BM, BN)` 的块形状，但 `local.fragment` 意味着
  布局推导会把它切成每线程的寄存器片段。
* **③** `alloc_local` 是一个每线程私有的小数组，索引空间普通（`scratch[k]`）。
* **④** `T.alloc_var` 只分配**一个元素**。docstring 里的例子给了五种写法，
  常用的是 `T.alloc_var("float32", init=0.5)` 或
  `T.alloc_var("int32", 0)`。用的时候是 `scale[0]`。它仍然是一个 buffer，
  只是长度为 1。
* **⑤** 注意 `scale[0]` 而不是 `scale`。

### 5.3 为什么 `fragment` 写成块形状却落在寄存器里

`docs/get_started/overview.md:82-84` 把这件事讲得最清楚：`alloc_fragment` 分配
的是**整个线程块**的寄存器文件，TileLang 用布局推导（LayoutInference）推一个
`T.Fragment` 对象，决定“这块寄存器怎么分给每个线程”。

所以下面这段算术是成立的（`code/03-memory-scopes/scope_footprint.py`
现算）：

```python
C_f = T.alloc_fragment((128, 128), "float32")  # 逻辑上 128×128
```

$$
128\times128\times4\ \mathrm{B}=64\ \mathrm{KiB},
$$

$$
\frac{64\ \mathrm{KiB}}{128\ \text{线程}}
=512\ \mathrm{B}/\text{线程},
$$

$$
\frac{512\ \mathrm{B}/\text{线程}}{4\ \mathrm{B}/\text{寄存器}}
=128\ \text{个 fp32 寄存器}/\text{线程}.
$$

**128 个寄存器/线程是个很大的数字**（NV 每线程最多 255 个 32-bit 寄存器，
见第 6 章）。这就是为什么 `BM × BN` 不能随便放大：它直接吃掉 occupancy。
这也是为什么第 10 章要专门讲 layout，`fragment` 的布局不是装饰，它决定
寄存器压力与后续 `T.copy` 的合并方式。

### 5.4 一个特例：bool buffer 在 CUDA 侧会被改写 scope

```python
if dtype == "bool":
    scope = "shared"      # 不是 shared.dyn
```

`allocate.py:60-62` 的注释解释了原因：smem 合并 pass 目前处理不了 bool 类型，
所以 `T.alloc_shared(..., "bool")` 会落到静态 `shared` 上。昇腾方言**故意覆盖**
了这个函数，去掉了这个 workaround。它的 docstring 写着“Ascend UB handles
bool buffers in the requested scope directly”
（`tilelang/ascend/language/allocate.py:15-24`）。这是第 6 章说过的
“同一个高层概念在两台上落点不同，且被显式承认”的又一个例子。

### 5.5 昇腾侧：UB / L1 / L0A / L0B / L0C

主库 v0.1.15 的昇腾方言有五个分配函数
（`tilelang/ascend/language/allocate.py`）：

```python
A_l1 = T.alloc_l1((BM, BK), "bfloat16")       # scope shared.l1   (:26)
B_l1 = T.alloc_l1((BN, BK), "bfloat16")       # scope shared.l1
C_l0c = T.alloc_l0c((BM, BN), "float32")      # scope shared.l0c  (:55)
C_ub = T.alloc_shared((BM // 2, BN), "float32")  # UB，scope shared.dyn (:15)
```

`alloc_l0a`（`:37`）与 `alloc_l0b`（`:46`）的 docstring 各写了一句关键的话：
**“Layout is deferred to the consuming gemm operation, which infers K-major or
MN-major from its transpose flags.”** 也就是说，L0A/L0B 的物理排布不是你在
分配时定的，而是由消费它的 `T.gemm` 的 `transpose_A` / `transpose_B` 反推。
这是第 21 章布局推导要处理的核心问题，本章只需要记住“**别自己假设 L0 的排布**”。

`alloc_l0c` 还会额外挂一个 `make_ascend_l0c_layout` 的布局注解（`:55-66`），
因为 L0C 是固定布局。

### 5.6 一个必须记住的版本差异：`alloc_ub` 在哪

**主库 v0.1.15 里没有 `T.alloc_ub`。** UB 就用 `T.alloc_shared` 分配，
昇腾后端 README 的原话是：

> “Use `T.alloc_shared` for Unified Buffer storage, `T.alloc_l1` for L1 storage,
> and `T.alloc_l0a`, `T.alloc_l0b`, or `T.alloc_l0c` for Cube buffers.”
> 来源：`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:138-139`

`alloc_ub` 是**适配器仓库**的拼法。`_vendor/tilelang-ascend-v0.1.2.000/`
的 `tilelang/language/allocate.py:140-157` 里同时有
`alloc_L1` / `alloc_L0A` / `alloc_L0B` / `alloc_L0C` / `alloc_ub`（注意大驼峰），
scope 名也不同：

```text
适配器仓库                            主库 v0.1.15 的 ascend 方言
────────────────────────────────     ──────────────────────────────
T.alloc_ub    → "shared.ub"           （无）UB 用 T.alloc_shared → "shared.dyn"
T.alloc_L1    → "shared.l1"           T.alloc_l1  → "shared.l1"
T.alloc_L0A   → "wmma.matrix_a"       T.alloc_l0a → "shared.l0a"
T.alloc_L0B   → "wmma.matrix_b"       T.alloc_l0b → "shared.l0b"
T.alloc_L0C   → "wmma.accumulator"    T.alloc_l0c → "shared.l0c"
```

两套拼法**不能互换**。写昇腾代码之前先确认自己用的是哪棵树；本书统一以
`_vendor/tilelang-v0.1.15/` 为准，出现适配器仓库的拼法时会显式标注。

---

## 6. 并行与循环

### 6.1 五种循环构造子

`tilelang/language/loop.py` 里定义了它们：

| 构造子 | 签名（节选） | 语义 |
| --- | --- | --- |
| `T.serial(start, stop=None, step=None)` | `:189` | 普通设备循环，`ForKind = Serial` |
| `T.unroll(start, stop=None, step=None, *, explicit=False)` | `:230` | 请求展开；`explicit=True` 要求真的展开 |
| `T.vectorized(start, stop=None)` | `:322` | 把一段迭代合成一条向量操作 |
| `T.Parallel(*extents, coalesced_width=None, loop_layout=None)` | `:13` | 嵌套并行循环，线程映射由布局推导决定 |
| `T.Pipelined(start, stop=None, num_stages=0, order=None, stage=None)` | `:112` | 生产者—消费者软流水 |

大写版本 `T.Serial` / `T.Unroll` / `T.Vectorized` 只是小写版的别名
（`:297`、`:309`、`:349`），用大写强调“这是 tile 级循环”。

### 6.2 语义差异：谁执行、为了什么

```text
T.serial      谁执行：默认单个线程顺序走完；除非被外层 T.Parallel 分活
              目的：表达"必须按序"的迭代（累加、扫描、K 方向归约）

T.unroll      谁执行：同 serial，但请求编译器展开
              目的：摊掉循环开销、把连续访存合并成更宽的请求
              额外：CUDA dialect 支持 unroll_factor=N → #pragma unroll N

T.vectorized  谁执行：一段迭代被合成一条向量指令
              目的：一个 request 搬多个元素，寄存器复用

T.Parallel    谁执行：编译器决定（lane → thread → 每线程串行）
              目的：表达"这些迭代之间没有依赖"

T.Pipelined   谁执行：外层是循环，"谁"由循环体里的语句决定
              目的：让 copy 和 compute 在时间上重叠
```

最小示例（`code/04-loops/loops_demo.py`）：

```python
base = bx * BLK                             # trace 期算出的块内起点
acc = T.alloc_var("float32", init=0.0)     # ①
for i in T.serial(BLK):                    # ②
    acc[0] += X[base + i]
for i in T.vectorized(4):                  # ③
    Y[base + i] = X[base + i] + acc[0]
for i in T.unroll(4):                      # ④
    Y[base + 4 + i] = X[base + 4 + i] * 2.0
for i, j in T.Parallel(BM, BN):            # ⑤
    Y[by * BM + i, bx * BN + j] = X[by * BM + i, bx * BN + j]
```

* **①** 归约的累加器用一个标量 `acc`（`alloc_var`），所以它必须是串行的。
* **②** `T.serial` 表达“一次加一个”。这条依赖链本身不能并行。
* **③** `T.vectorized(4)` 要求“这 4 次迭代拼成一条向量操作”。它的前提是
  `X[base..base+3]` 在内存里连续、且对齐。
* **④** `T.unroll` 只是请求展开；能不能展开由后端决定。需要保证时用
  `explicit=True`，或者 CUDA 方言的 `unroll_factor=N`。
* **⑤** `T.Parallel` 把线程映射全部交给编译器。这是最常用、也最该优先用的
  一种。

### 6.3 `T.Parallel` 的维度映射规则

这是本章技术含量最高的一节。**`T.Parallel` 的 extents 是逻辑迭代空间，
不是“哪个维度给线程”的声明。** 默认映射由
`src/transform/loop_partition.cc` 的 `PlanLoopPartition` 决定
（`:248-271`）：

对 $k$ 维的 `T.Parallel`，记其 extents 为 $e_0,\ldots,e_{k-1}$，默认映射是

$$
\begin{aligned}
\mathit{flattened}
  &= e_0(e_1(\cdots(i_0)+i_1)\cdots)+i_{k-1},\\
\mathit{access\_idx}
  &= \left\lfloor\frac{\mathit{flattened}}{v}\right\rfloor,\\
\mathit{thread}
  &= \mathit{access\_idx}\bmod n,\\
\mathit{local\_idx}
  &= \left\lfloor\frac{\mathit{access\_idx}}{n}\right\rfloor v
     +\mathit{flattened}\bmod v,
\end{aligned}
$$

其中 $v$ 是 `vector_size`，$n$ 是 `num_thread`。访存粒度从连续 lane 开始，
再轮到相邻线程，剩余部分变成每线程串行复制。

三个结论：

1. **最内层是最快的**。行优先展平意味着 `i_{k-1}` 变得最快，所以最内层的
   连续取值会落进同一组 lane → 可以被向量化。
2. **线程是中间那一层**。lane 分组之后按 `% num_thread` 轮转，所以相邻的
   线程拿到相邻的分组 → 访存可以合并。
3. **剩下的变串行复制**。当总迭代数超过 `num_thread × vector_size` 时，
   多出来的部分变成每个线程的串行循环。

`code/04-loops/loop_mapping.py`（CPU 逻辑验证）把这个公式实现了一遍。跑
`--compare` 会看到一件反直觉的事：

```text
threads=8, vector_size=2, 总共 64 次迭代

T.Parallel(8, 8)      per-thread=8    thread 0 owns (0,0) (0,1) (2,0) (2,1) ...
T.Parallel(2, 32)     per-thread=8    thread 0 owns (0,0) (0,1) (0,16) (0,17) ...
T.Parallel(4, 4, 4)   per-thread=8    thread 0 owns (0,0,0) (0,0,1) (1,0,0) (1,0,1) ...
```

**每线程的工作量完全相同，但归属不同。** 所以“nest 形状无所谓，反正次数
一样”是错的：形状决定了 `thread 0` 读的是哪几个逻辑下标，也就决定了
load 会不会合并。

### 6.4 什么时候必须自己给布局

不写 `loop_layout=` 时，布局推导会自己推一个并挂在最外层循环上。需要自己
给的时候，`loop_layout=` 接的是一个 `T.Fragment`，并且有硬约束
（`tilelang/language/loop.py:40-79`）：

* 嵌套深度为 `k` 时，`Fragment` 的 `InputDim` 必须等于 `k`；
* 布局注解只能挂在**最外层**并行循环上，内层挂会报错；
* 每个并行循环在布局推导之后都必须被某个布局覆盖，漏了也是编译错误。

```python
layout = T.Fragment([BM, BN], forward_thread_fn=lambda i, j: (i * BN + j) % 128)
for i, j in T.Parallel(BM, BN, loop_layout=layout):
    ...
```

另一个 hint 是 `coalesced_width=`，它告诉向量化规划器“期望的合并宽度”，
用于向量化可行性判断（`loop.py:22-30`）。

### 6.5 `T.Pipelined`：把 copy 和 compute 叠起来

```python
for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    T.copy(A[by * BM, ko * BK], A_s)     # 生产者
    T.copy(B[ko * BK, bx * BN], B_s)     # 生产者
    T.gemm(A_s, B_s, C_f)                # 消费者
```

`num_stages` 是“生产者和消费者之间最多用几份 buffer”。它的**代价是片上
容量**：第 5 章算过，`(128,256)` fp32 开两份就放不进示例里的 192 KiB UB。

想手工排调度就用 `order` / `stage`（两个长度相同的列表，按顺序给循环体里的
可调度语句编号）。这里有一条容易踩的规则（`loop.py:139-168`）：**本地别名
产生的可重放标量 `Bind` 不算可调度语句，不占槽位。** 例如

```python
for i in T.Pipelined(n, order=[1, 0], stage=[0, 1]):
    base: T.int32 = i * block          # 可重放，不占槽位
    T.copy(A[base], shared)            # slot 0
    T.copy(shared, B[base])            # slot 1
```

只写了两个 slot，而不是三个。如果这个 `Bind` 读的 buffer 恰好也被流水写，
它就不再可重放，会变回一个调度语句（因为它带了流水依赖）。

---

## 7. 控制流

### 7.1 `if` / `elif` / `else` 与三元表达式

标准 Python 的 `if/elif/else` 在 kernel 里可用，条件是 **TIR 表达式**
（`docs/programming_guides/control_flow.md:15-40`）：

```python
for i in T.Parallel(BLK):
    gi = bx * BLK + i
    if gi < N:                      # ① TIR 条件 → 设备侧分支
        Y[gi] = X[gi]
    else:
        Y[gi] = 0.0

Y[by, 0] = S[by] if by < M else 0.0   # ② 三元 → if_then_else 表达式
```

* **①** 条件是 TIR 表达式，会变成设备上真正的分支。
* **②** `x if cond else y` 构造的是 `if_then_else`，是**值**不是语句。

**编译期常量会被折叠。** 条件是纯 Python 的 `bool` 时
（比如 `if USE_FAST_PATH:` 里 `USE_FAST_PATH = True`），trace 期就定下来了，
不会在设备上留下分支（`control_flow.md:19-21`）。这也意味着：从 Python 常量
推导出来的 `if` 可以放心用来“按参数开关一段实现”。

### 7.2 `T.if_then_else` 与 `T.Select`：不是一回事

这两个都是“取值”，但求值方式不同。`tilelang/language/tir/ir.py:172-179`
的 `Select` docstring 把区别写死了：

> “``Select`` evaluates both value expressions. Use ``if_then_else`` when a
> branch must guard side effects or out-of-bounds memory accesses.”

```python
# 安全：条件为假时，X[gi] 这个 load 根本不会发生
Y[gi] = T.if_then_else(gi < N, X[gi], 0.0)

# 危险：两个操作数都会被求值，gi >= N 时 X[gi] 照样被读
Y[gi] = T.Select(gi < N, X[gi], 0.0)
```

**规则**：做越界保护、或者一边有副作用时，只能用 `T.if_then_else`；
两边都安全、只想按条件选值时，`T.Select` 更接近硬件 select，通常更省。

### 7.3 `T.all_of` / `T.any_of` 的一个陷阱

`docs/programming_guides/control_flow.md:33-38` 给的例子是（同一份文件在
`:141` 又用了一次）：

```python
if T.all_of(i < M, j < N):     # ← 这个写法在 v0.1.15 里是错的
```

但 v0.1.15 的实现在 `tilelang/language/logical.py:12, 51`：

```python
def any_of(buffer: BufferLikeType) -> tirx.PrimExpr: ...
def all_of(buffer: BufferLikeType) -> tirx.PrimExpr: ...
```

它们接受的是一个**缓冲区或缓冲区切片**，语义是“这个切片里是否至少有一个 /
全部元素为真”。对应测试的用法是
`T.all_of(BlockMask[by, bx, k, :])`
（`testing/python/language/test_tilelang_language_all_of.py:56`）。

**多个条件的与，用 `and`。** Python 的 `and` / `or` / `not` 作用在 TIR 表达式上
时会降级成 `tir.And` / `tir.Or` / `tir.Not`
（`tilelang/language/eager/builder.py:820-846`）：

```python
if (i < M) and (j < N):
    ...
```

这是本章发现的一处**文档落后于源码**的地方，请以源码为准。

### 7.4 `while`、`break`、`continue`

```python
while i[0] < BLK:                # 条件必须是 TIR 表达式
    if X[i[0]] < 0.0:
        i[0] += 1
        continue                 # 跳过本次
    Y[i[0]] = X[i[0]]
    i[0] += 1
```

* `while` 条件是 TIR 表达式；**常量恒真会被编译期拒绝**，防止写出死循环
  （`control_flow.md:112-116`）。
* `break` / `continue` 在 `T.serial` / `T.unroll` / `T.Parallel` / `while` 里
  都可用。有一个限制要注意：**trace 期的 Python 循环里，只有编译期条件才支持
  `break` / `continue`**；要按运行期的值跳转，就用 `T.serial`
  （`python_compatibility.md:42-48`）。
* 设备侧的 `T.loop_break()`（`tilelang/language/customize.py:163`）是
  `break` 的等价内建，某些场景下更明确。

### 7.5 控制流对 layout 与流水的两种代价

**代价一：守卫会挡住向量化。** 如果每次迭代都带一个谓词，向量化器没法把它
合成一个宽的、对齐的访存请求，只能退化成带条件的标量循环。
`docs/deeplearning_operators/elementwise.md:113-135` 给了一个生成代码的例子：
`N=2047`、`threads=256`、`num_per_thread=8` 时，生成的是

```c++
#pragma unroll
for (int i = 0; i < 8; ++i) {
  if (((i * 256) + ((int)threadIdx.x)) < 2047) {
    C[...] = A[...] + B[...];
  }
}
```

没有向量化，也没有合并访存。同一份文档紧接着指出：**除了尾部那一组，
其它线程本来可以走更优化的代码。** 这正是 `LegalizeSafeMemoryAccess`
自动插守卫的代价，也是“默认安全”与“默认快”之间的取舍。

**代价二：分支会切开流水。** `T.Pipelined` 的 `order` / `stage` 是按循环体里的
**可调度语句**编号的。一段 `if` 会把循环体切成多个基本块，`Bind` 的可重放性
也可能因此改变（见 6.5）。写流水循环时，尽量让循环体的结构稳定：把
数据相关的分支留在 `T.Parallel` 内部，而不是让整个流水循环分叉。

---

## 8. Python 兼容性边界

### 8.1 一张表

`docs/programming_guides/python_compatibility.md` 是这一节的权威来源。
归纳成三类：

| 类别 | 构造 | 说明 |
| --- | --- | --- |
| 直接可用 | `for i in range(n)` / `range(a,b,s)` | 映射成 `T.serial` |
| 直接可用 | `for x in list/tuple/enumerate/zip` | **trace 期展开**，必须有限 |
| 直接可用 | `while`、`if/elif/else`、`x if c else y`、`break`/`continue` | |
| 直接可用 | `a[i]`、`a[i,j,k]`、`a[i:j]`、`a[-1]` | 切片会构造 `BufferRegion` |
| 直接可用 | `x = expr`、`+ - * / %`、`+=` 等增强赋值 | |
| 直接可用 | 列表/字典/集合推导式 | 边界必须在 trace 期可求值；推导变量不外泄 |
| 有限制 | `with` | 只认 `T.Kernel` / `T.ws` 这类 DSL 上下文 |
| 有限制 | `assert` | 用 `T.device_assert` 或 `T.assert` |
| 有限制 | `print()` | 只在 Python 表达式上有效；设备侧用 `T.print()` |
| 不可用 | 函数与类 | kernel 里不能有 `def` / `class`；要复用写 `@T.macro` |
| 不可用 | `a = b = c` | 链式赋值，拆开写 |
| 不可用 | `len()` | 用 `buffer.shape[dim]` |
| 不可用 | `type()`、`isinstance()` | |

自查工具：`code/06-python-boundary/py_compat_scan.py`（CPU 逻辑验证）把这张表
做成了一个 AST 扫描器，能在编译之前先把显然越界的构造挑出来。

### 8.2 `@T.macro`：kernel 里的“函数”

kernel 里不能写函数，但可以写宏：

```python
@T.macro
def load_tile(src, dst, r, c):
    T.copy(src[r, c], dst)
```

`T.macro` 在**编译期**把宏体内联到调用点，语义上接近 CUDA 的 `__device__`
函数（`python_compatibility.md:89-91`）。v0.1.15 里它同时挂在 lazy 解析器
（`tilelang/language/parser/entry.py:105`）与 eager builder
（`tilelang/language/eager/builder.py:1075`）上，两种前端都能用。

### 8.3 最容易搞错的一件事：编译期值 vs 设备值

```text
N              Python int（来自 @jit 工厂的参数）  →  编译期常量
A.shape[0]     从 buffer 形状读出来的符号          →  既可用于 trace，也可进 TIR
T.dynamic("K") 显式声明的符号变量                  →  设备值（TIR Var）
bx             T.Kernel 返回的块索引               →  设备值（TIR Var）
X[i]           buffer load                         →  设备值（TIR 表达式）
```

两种典型错误：

* **把设备值当 Python 值用**：写 `if bx > 4:` 是没问题的（`bx` 是 TIR Var，
  `>` 会构造 TIR 表达式），但写 `len(X)` 或 `X.shape` 去当 Python 的整数用，
  就会掉进 trace 期的坑。
* **把应该展开的东西写成设备循环**：Python 的 `range` 在 `for` 语句里**会变成
  设备侧的 `T.serial`**，包括边界是动态的时候（`python_compatibility.md:32-37`）。
  “我只想展开 4 次”和“我要一个 4 次的设备循环”在写法上要分清：
  循环一个具体的 Python 列表是展开，`range` 是设备循环。

```python
for k in [0, 1, 2]:        # 展开：trace 期把循环体复制 3 份
    ...
for k in range(3):         # 设备循环：相当于 T.serial(3)
    ...
for k in T.unroll(3):      # 请求展开的设备循环
    ...
```

还有一个更隐蔽的点：**Python list 不是设备 buffer。** 用运行期变量去索引
Python list 不会生成设备侧查表，要用 buffer 或直接把下标算出来
（`python_compatibility.md:68-70`）。

图见 `figures/08-python-boundary.md`。

---

## 9. 类型系统与隐式转换的坑

第 4 节讲了“有哪些类型”，这一节讲“编译器会替你转什么、不会替你转什么”。

**① 整数字面量与浮点混用。** `T.max(C_f[i, j], 0)` 是
`examples/quickstart.py` 的写法：`0` 是 Python int，`C_f` 是 fp32 的 fragment。
这类“字面量与 buffer 元素”的混用会被接受并做类型提升。但**两个不同 dtype 的
buffer 直接运算就未必**，尤其在有 lanes（向量形）的时候。遇到“类型不匹配”的
报错，先把操作数显式转成同一个 dtype 再看。

**② `bool` 的存储。** CUDA 侧 `T.alloc_shared(..., "bool")` 会落到静态
`shared`（见 5.4）；`bool` 也没有向量形。用 uint8 做掩码在很多场景下更省心
（`control_flow_demo.py` 的 `buffer_predicates` 就是这么写的）。

**③ int4 / float4 是打包的。** `int4` 与 `float4_e2m1fn` 的 `bits` 是 4，
也就是**一字节存两个值**。`as_torch()` 会退到 `int8` / `float4_e2m1fn_x2`
（`dtypes.py:187-236`）。做指针算术或算 buffer 大小时如果按 1 字节/元素算，
会差一倍。

**④ `float8_e4m3` 与 `float8_e4m3fn` 是两个不同的类型。** 前者是抽象 E4M3，
`as_torch()` 会按平台在 `float8_e4m3fn`（CUDA）与 `float8_e4m3fnuz`（ROCm）
之间切换（`dtypes.py:187-206`）。同一段代码在两个平台上会得到不同的物理类型，
这正是 `fp8.py` 的 `determine_fp8_type()` 存在的理由。

**⑤ 累加 dtype 必须显式。** 见 4.3。fp16 输入 + fp16 累积在 K 很大时会明显
损失精度；用 fp32 累积是默认推荐（第 27 章会量化讨论这一点）。

**⑥ 混合 A/B dtype 只允许 f8f6f4 家族内部。** 见 4.5。

**⑦ 字符串别名会被改写。** `"half"` 就是 `"float16"`、`"float"` 就是
`"float32"`（`dtypes.py:105-113`）。这本身没问题，但如果你在日志里看到
`float32` 而代码里写的是 `"float"`，不要以为是编译器换了类型。

---

## 10. 实现机制：编译器把这些语义落到哪里

这一节把前面各节的语言表面接回第 4 章那条编译管线。

**① `T.Kernel` → `MaterializeKernelLaunch`。**
`launch_kernel` 先在 trace 期做三件归一化：块绑定
（`_normalize_bindings`，`kernel.py:88-96`）、线程数
（`_normalize_threads`，`:99-135`，补成三维）、簇维度
（`_normalize_cluster_dims`，`:138-151`，`[1,1,1]` 视为没给）。
然后 `DEFAULT_SIMT_THREADS = 128`（`transform/__init__.py:201`）在 pass
阶段补上默认线程数。这个 pass 还决定“这个后端有没有线程维度”：
`lower_thread_binding=True`（CUDA/ROCm/Metal）把线程占位符绑成
`threadIdx.*` 作用域；`False`（CPU、Ascend）丢掉它们，**body 里引用线程索引
会被拒绝**。昇腾因此把线程域做成 `T.SimtVF` 自己的作用域。

**② `T.alloc_*` → `Buffer` + `storage_scope`。**
每个分配函数最终都调到 `T.sblock_alloc_buffer(shape, dtype, scope=...)`
（`allocate.py:49-193`）。`scope` 是唯一区分它们的东西：
`shared.dyn` / `local.fragment` / `local` / `local.var` / `global` /
`shared.tmem` / `shared.barrier`。**同一个 API 换一个 scope 就是换一种物理存储。**

**③ `T.Parallel` → `PlanLoopPartition` + `LowerParallelLoop`。**
布局推导先给每个并行循环找一个 `Fragment` 布局（没给就自己推），
`PlanLoopPartition`（`loop_partition.cc:248-271`）按 6.3 的公式算出映射，
`LowerParallelLoop`（`:324-350`）再分两步落地：先 `PartitionLoop` 按布局切分，
再 `VectorizeLoop` 按访存分析决定向量宽度（1 就是标量）。最后
`PragmaUnrollLoop` 处理展开 pragma。

**④ 类型归一化 → `DataType`。**
三个入口（字符串 / TileLang 对象 / 框架 dtype）都走到
`dtype.__new__`（`dtypes.py:245-253`），产出同一个 `tvm.DataType`。之后
`validate_gemm_ab_dtypes`（`:319-340`）之类的检查才有统一的比较基准。

**⑤ 布局推导是“所有并行循环都必须被覆盖”的。**
`ParallelLoopLayoutValidator` 在 `tl.transform.LayoutInference` 里检查：
每个并行循环最终都要有布局注解，且必须在最外层（`loop.py:40-58`）。
这条规则是第 10、21 章的主角，本章只要求你知道“`T.Parallel` 不是免费的，
它背后有一次布局推导”。

---

## 11. 进阶与陷阱

1. **以为 `threads=` 可以随便省。** 省略时默认 128，看起来无害；但一旦
   kernel 体里要用 `T.get_thread_extent()`，trace 期就会报
   `"The thread extent is not known at trace time"`（`kernel.py:215-226`）。
   需要线程数就显式给。

2. **以为 `T.Parallel` 的 extents 直接对应线程维度。** 不是。它先展平，
   再按 lane → thread → 串行 切。要控制映射就写 `loop_layout=`。

3. **用 `T.Select` 做越界保护。** 两边都会求值，越界访问照样发生。用
   `T.if_then_else`。

4. **忽略 `T.alloc_fragment` 的寄存器代价。** `128×128` fp32 就是
   128 寄存器/线程。`BM × BN` 是 occupancy 的第一杀手。

5. **把 `bfloat16x8` 当成合法 dtype。** bf16 只有 x2。

6. **把适配器仓库的 `alloc_ub` / `alloc_L1` 当成主库 API。** 主库 v0.1.15
   没有 `alloc_ub`；scope 名也不一样。

7. **在流水循环里放会分叉的 `if`。** `order` / `stage` 是按可调度语句编号的，
   分叉会改变编号与 `Bind` 的可重放性。

8. **把 Python 的 `range` 当成“展开”。** 在 `for` 语句里它会变成设备侧的
   `T.serial`。要展开就循环一个具体的 Python 序列。

9. **相信文档里的 `T.all_of(i < M, j < N)`。** v0.1.15 的 `all_of` 接的是
   buffer。多条件用 `and`。

10. **忘了 fp4 有两种存储模型。** `float4_e2m1fn` 是打包的（4 bit），
    `float4_e2m1_unpacked` 是 SMEM 专用的（8 bit 槽）。算大小、配 TMA
    描述符时选错就会对不上。

---

## 12. 本章小结

* **六个构件**：网格（`T.Kernel`）、dtype、buffer/作用域、迭代、控制流、
  Python 边界。前五个直接决定 IR 的形状，第六个决定“哪些东西根本进不了 IR”。
* **`T.Kernel`**：位置参数按顺序对应 x/y/z 的块数，返回块索引；`threads=` 是
  每块线程数（默认 128）；`prelude` / `cluster_dims` 是后端注解。块内默认是
  **块级**世界，线程级要用 `T.get_thread_binding()`（NV）或 `T.SimtVF`（昇腾）。
* **dtype**：173 个名字，`'float16'` / `T.float16` / `torch.float16` 三种写法
  归一到一个 `DataType`。fp8 有八个拼法，fp4 有两种存储模型，bf16 只有 x2。
  混合精度时累加 dtype 必须显式；A/B dtype 只允许在 f8f6f4 家族内混。
* **buffer 与作用域**：`alloc_shared`（块内共享暂存）、`alloc_fragment`（块形状
  逻辑 / 寄存器物理）、`alloc_local`（每线程私有数组）、`alloc_var`（标量）。
  昇腾侧还多出 UB / L1 / L0A / L0B / L0C，且主库用 `alloc_shared` 分配 UB。
* **迭代**：`T.serial` 顺序、`T.unroll` 展开、`T.vectorized` 向量化、
  `T.Parallel` 交给编译器、`T.Pipelined` 软流水。`T.Parallel` 的默认映射是
  **展平 → lane → thread → 每线程串行**，所以 nest 的形状决定访存能否合并。
* **控制流**：`if/else` 是语句，`T.if_then_else` 只求值被选中的一边，
  `T.Select` 两边都求值。守卫会挡住向量化，分支会切开流水。
  `T.all_of` / `T.any_of` 接的是 buffer，不是条件列表。
* **Python 边界**：`range` 在 `for` 里是设备循环，具体序列是 trace 期展开；
  没有函数/类（用 `@T.macro`）、没有链式赋值、没有 `len()`。
  分清“编译期值”与“设备值”是这一节最关键的一件事。

下一章（第 9 章）会在这些构件之上讲**核心原语**：`T.copy`、`T.gemm`、
`T.reduce_*`、`T.atomic_add`、`T.fill`、`T.tile.*`，每一个的语义、参数与
降级行为，以及它们各自会展开成什么样的 `T.Parallel` / `T.serial`。

## 延伸阅读

* 想回到编译管线本身：第 4 章。
* 想回到两台机器的硬件图（shared / UB / L1 / L0 的物理动机）：第 6 章。
* 想按存储作用域写第一个程序：第 7 章。
* 想深入布局推导与 `T.Fragment`：第 10 章、第 21 章。
* 想深入软流水与多缓冲：第 11 章、第 22 章。
* 想把低精度用对：第 27 章。
* 本章引用的全部原始资料与核验状态：见 `references.md`。
* 想亲手复算网格、dtype、作用域与循环映射：`code/` 下六个示例。
