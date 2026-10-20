# 第 07 章 习题答案

说明：本文件与 [exercises.md](exercises.md) 一一对应（E1…E14）。每题给**解题思路**、
**参考答案**与**常见错误**；动手题给思路与关键片段，不要求逐字照抄。

---

## E1 概念题 ★

**解题思路**：先把 3.1 节的两张表并排看，再回答“哪条对两者都成立”。

**参考答案**：

1. 对两条路径都成立的是 **Python >= 3.10**（`_vendor/tilelang-v0.1.15/docs/get_started/Installation.md:8`
   与 `:75`）。另外 **CUDA >= 10.0** 在两条路径的条件里也重复出现
   （`:9` 与 `:76`），但它在源码构建里可以换成 pip 提供的工具链。glibc >= 2.28
   只出现在 pip 安装的前置条件里（`:7`）；“Linux 或 Windows”只出现在源码构建里（`:74`）。
2. “CUDA >= 10.0”指 **host 上装了 CUDA 工具链**（能跑 `nvcc`）时用它可以；
   “pip 提供的 CUDA >= 13.0”指 **host 没装 CUDA**、改用 pip 装的 CUDA 包
   （`nvidia-cuda-nvcc` 等）时用，对应 `Installation.md:101-124`。
3. 有问题。源码构建的前置条件写的是 “Operating System: **Linux or Windows**”
   （`Installation.md:74`）；而且 CMake 选项一栏写着 `-DUSE_CUDA=ON|OFF` “default ON
   when a CUDA toolkit is found, **except on macOS**”（`Installation.md:160`）。
   换句话说 macOS 上不会默认编 CUDA，也不在支持的构建系统列表里。

**常见错误**：把 glibc 要求当成两条路径都要满足（它只写在 pip 一节）；或者以为
macOS 能照 3.3 节编 CUDA。

---

## E2 概念题 ★

**解题思路**：先复述 2 节那张表，再按“谁写的 / 谁补的”归类。

**参考答案**：

1.

| 部件 | Python 写法 | 回答的问题 |
| --- | --- | --- |
| 网格 | `with T.Kernel(...) as bx:` | 这段计算被切成多少块并行执行？ |
| buffer | `T.alloc_shared` / `T.alloc_fragment` | 数据放在哪一级存储上？ |
| 搬运 | `T.copy(src, dst)` | 一个 tile 怎么从一级存储到另一级？ |
| 并行域 | `for i in T.Parallel(n):` | 这 n 个元素各做一次，具体谁做？ |

2. **人写的**是①网格与②buffer 的位置选择；**编译器补的**是③搬运的低层循环与④并行域
   到线程/向量化的映射。例子：人写 `T.Kernel(T.ceildiv(N, block_N), threads=256)`，
   编译器把它落成 `__launch_bounds__(256)` 与 `gridDim.x`；人写 `T.Parallel(block_N)`，
   编译器落成 `if (threadIdx.x < 32)` + `uint4`（本章 5.2 节）。
3. 成立，版本 A 就没有显式 buffer，也没有显式 `T.copy`：`T.Parallel` 里的
   `C[bx*block_N+i] = A[...] + B[...]` 既读全局、又算、又写全局。所以四个部件里
   **必需的只有“网格”和“并行域”**：没有网格就没有并行切分，没有并行域就没有
   逐元素语义。buffer 与 copy 是**可选的表达力**（当你需要控制数据落哪一级存储、
   或需要 `T.gemm` 这类原语时它们才变成必需）。

**常见错误**：把“四个部件”当成“四个都要显式写”。TileLang 的默认路径允许省略后两个。

---

## E3 概念题 ★★

**解题思路**：回到 `T.Kernel` 的 docstring 与 5.2 节的对照表。

**参考答案**：

1. 它是“**每个 block 的线程数**”，也就是 `blockDim.x` 的取值，同时也是
   `__launch_bounds__` 的数字。默认值是 **128**
   （`_vendor/tilelang-v0.1.15/tilelang/cuda/language/kernel.py:25-27`）。
2. 因为 `T.Parallel` 是块级并行域，不指定线程号；`LayoutInference` 会把它**推导**成
   “每个线程负责哪些元素”，这一步会插入 `threadIdx.x` 并选定向量宽度。所以你没写
   `threadIdx.x`，它却出现在生成代码里。那是编译器补的映射。
3. 是两个独立的量：网格大小由 `T.ceildiv(N, block_N)` 决定（**多少块**），
   `threads` 决定每块多少线程（**每块多少线程**）。版本 A 里 `block_N == threads`
   只是**巧合般的方便**（让“每线程 1 元素”的直觉成立），不是必须，E4 第 4 问就会
   让它们不等，语义照样正确。

**常见错误**：把 `threads=` 当成“总线程数”。它是每块线程数，总数大致是
`网格大小 × threads`（这里“大致”是因为编译器可能只用其中一部分线程，见 E5）。

---

## E4 代码阅读题 ★★

**解题思路**：先算网格，再算每块负责的下标区间。

**参考答案**：

1. 网格大小 = `ceil(1024/256) = 4` 个 tile 程序；`bx ∈ {0, 1, 2, 3}`。
2. 第 `bx` 块负责 `C[bx*256 .. bx*256+255]`，即闭区间
   `[bx*256, (bx+1)*256 - 1]`。用表达式写就是 `C[bx*block_N + i]`，`i ∈ [0,256)`。
3. `i` 不是线程号，而是**块内的元素下标**（0…255）。它和线程的对应关系由编译器
   决定（可能一个线程负责多个 `i`）。
4. `block_N=512` 时网格变成 `ceil(1024/512) = 2` 块；每块负责 512 个元素；
   语义仍然正确，每块内 `T.Parallel(512)` 平铺 512 个元素，`threads=256` 只是说
   每块最多 256 个线程，编译器可以让每个线程做 2 个（或向量化后更少线程做更多）。
   这正是“`block_N` 与 `threads` 解耦”的体现。

**常见错误**：把 `bx` 的范围写成 `0..N`（那是下标不是块号）；或者认为
`block_N != threads` 会报错。

---

## E5 代码阅读题 ★★

**解题思路**：逐行找 `blockIdx.x`、`threadIdx.x`、向量宽度三个量。

**参考答案**：

1. `blockIdx.x` 对应 Python 里的 `b_x`；`N=1024, threads=256` 时网格大小 4，
   所以 `blockIdx.x ∈ {0,1,2,3}`（最大值 3）。
2. 32 = 256 / 8，即“每线程做 8 个元素”时需要的线程数。另外 224 个线程不干活
   （warps 仍然被 launch，但 `if` 判断把它们挡住了）。
3. `uint4` = 4 × 4 = **16 字节**；`bfloat16` 是 2 字节，所以一次能搬 `16/2 = 8` 个。
4. 循环变量 `i` 消失了，它被**向量化 + 线程映射**吸收进
   `((int)threadIdx.x) * 8`：`threadIdx.x` 决定“第几组 8 个”，组内 8 个由向量类型一次
   处理。`i` 在生成代码里不再作为变量出现。
5. 违背了“256 线程各做 1 个”的字面意图。但没违背“256 个元素各做一次”的语义。
   官方文档（`elementwise.md:176-192`）的建议是：**显式指定每线程元素数**
   （`num_per_thread`，即把 `T.Parallel` 写成 `T.Parallel(threads, num_per_thread)`，
   让线程维度显式化），这样生成的代码更贴近你的设计意图。

**常见错误**：以为“只用了 32 个线程”是 bug。它是优化结果，代价是“launch 了 256 但只
激活 32”；官方把它归为“可能反直觉、需要显式引导”的情形。

---

## E6 代码阅读题 ★★

**解题思路**：对比两份生成代码的“循环结构 + 访存宽度”。

**参考答案**：

1. 边界判断是**编译器加的**。Python 里没有这个条件（`T.Parallel(threads)` 里只有
   一句加法），是 lowering 时为了让尾部安全而插入的。
2. 每个线程处理 **8** 个元素（`for (int i = 0; i < 8; ++i)`）；
   访存宽度是 **1 个 bfloat16 = 2 字节**（直接 `C[...] = A[...] + B[...]`，
   没有 `uint4`），即从 16 字节退化到 2 字节。
3. 官方评语是：**除了尾部那一组，其余线程本可以执行更优的代码，但边界判断挡住了
   向量化**（`elementwise.md:133`）。原因是向量化要求“每线程固定处理连续的 N 个元素
   且都在界内”，一旦有一个元素可能越界，就不能安全地把 8 个打包成一次 128-bit 访问，
   只能逐个判断。
4. 所以“tile 尺寸和 shape 对齐”主要是**性能问题**（挡住向量化），但如果写代码时
   自己漏了边界处理，就会变成**正确性问题**（越界读写）。TileLang 的默认行为是
   帮你补齐正确性，代价是丢掉一部分性能。

**常见错误**：认为边界判断只是“多一个分支”、开销可忽略。它真正的影响是**阻止了
向量化**，让访存宽度从 16 字节掉到 2 字节。

---

## E7 动手题 ★★

**解题思路**：一维 → 二维，多一维网格、多一个 `T.Parallel` 维度。

**参考答案（要点）**：

```python
@T.prim_func
def main(A: T.Tensor((M, N), dtype), B: T.Tensor((M, N), dtype), C: T.Tensor((M, N), dtype)):
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        for local_y, local_x in T.Parallel(block_M, block_N):
            y = by * block_M + local_y
            x = bx * block_N + local_x
            C[y, x] = A[y, x] + B[y, x]
```

* **参数顺序**：第一个维度是 `bx`（沿 `N` 方向，`blockIdx.x`），第二个是 `by`
  （沿 `M` 方向，`blockIdx.y`）。所以 `A[by*block_M + local_y, bx*block_N + local_x]`。
* **写反会怎样**：不会报错，会**算错**（访问位置错位），而且在越界时可能出现
  非法访存。这也是它危险的地方，没有编译期错误提示。

**常见错误**：把 `T.Kernel(M_blocks, N_blocks)` 当成“先 M 后 N”。记住本章 4.4 节的
  “N 在前、M 在后”。

---

## E8 动手题 ★★

**解题思路**：把版本 A 的“读写全局”拆成“先搬进 buffer、算、再搬出去”。

**参考答案（要点）**：

```python
with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
    A_local = T.alloc_fragment((block_N,), dtype)
    B_local = T.alloc_fragment((block_N,), dtype)
    C_local = T.alloc_fragment((block_N,), dtype)
    T.copy(A[bx * block_N], A_local)
    T.copy(B[bx * block_N], B_local)
    for i in T.Parallel(block_N):
        C_local[i] = A_local[i] + B_local[i]
    T.copy(C_local, C[bx * block_N])
```

2. `T.copy` 的 docstring 说它接受 `Buffer` / `BufferRegion` / `BufferLoad`；
   当源写成一个 `BufferLoad`（这里是 `A[bx * block_N]`）时，它会**从目标推断区域**
   （把源当成“首地址”，形状取目标的形状），docstring 里叫 “Syntactic Sugar”
   （`tilelang/language/copy_op.py:96-99`）。所以 `A` 搬 `block_N` 个元素这个信息，
   来自 `A_local` 的形状。
3. **结果完全相同**。真正差别是：版本 B 把数据显式地经过片上一级存储（这里是
   fragment / 寄存器），版本 A 直接在全局内存上读写。这个差别在简单逐元素加法里
   不改变结果，但在 GEMM、reduction 等需要复用的算子里是**语义必需**的
   （tile 必须落在合适的一级存储上才能被 `T.gemm` / 归约消费）。

**常见错误**：以为 `T.copy(A[bx*block_N], A_local)` 里的 `A[bx*block_N]` 只搬一个
元素。它是“首地址”，搬几个由目标形状决定。

---

## E9 概念题 ★★

**解题思路**：回到 `JITImpl` 的 docstring（`tilelang/jit/__init__.py:271-330`）。

**参考答案**：

1. 看被修饰的函数**返回什么**：返回 `PrimFunc`（`@T.prim_func` 的结果）→ **lazy**；
   用 DSL 直接写 body、返回输出张量（`T.empty` / `T.const` 那套）→ **eager**。
   模式是自动推断的。
2. lazy：

   ```python
   @tilelang.jit(out_idx=-1, target="cuda")
   def build_add(N, block_N=256):
       @T.prim_func
       def main(A, B, C): ...
       return main

   kernel = build_add(1 << 20)   # ← 这一行编译（或命中缓存）
   c = kernel(a, b)              # ← 这一行执行
   ```

   eager：

   ```python
   @tilelang.jit
   def add_eager(A, B, block_N=256):
       N = T.const("N"); ...; return C

   c = add_eager(a, b)           # ← 编译 + 执行都在这一行
   ```

3. **lazy**：要复用同一个 kernel 对象、要看生成源码、要接 autotuner、要控制
   `out_idx` / `target` 等编译选项。官方例子：`examples/ascend/example_vecadd.py`
   （返回 `@T.prim_func`）。
   **eager**：想“像调用普通函数”、形状从输入推断。官方例子：
   `examples/quickstart.py` 与 `examples/elementwise/example_elementwise_add.py`。

**常见错误**：以为 “lazy/eager” 要手动指定。它在 `JITImpl.__call__` 里按函数行为
自动推断（`tilelang/jit/__init__.py:519-540`）。

---

## E10 概念题 ★★★

**解题思路**：把四种改动分别映射到 key 的字段上。

**参考答案**：

1. **命中**。注释不进 `func.script(show_meta=True)` 的语义内容，key 不变。
   （注意：如果你改的是 `T.prim_func` 函数体里的语句，那是另一回事。）
2. **不命中**。`block_N` 出现在编译参数里（`args_repr`），key 变了；旧 key 的记录
   还在缓存里，可复用。
3. **不命中**。`target` 是 key 的一个字段（不同后端本来就是不同产物）。
4. **可能命中**（这是坑）。`pass_configs` 没变，而 key 里默认**不含库内容哈希**
   只有打开 `TILELANG_KERNEL_CACHE_USE_LIB_STAMP` 才会把
   `libtilelang` 等库的哈希加进去（`tilelang/env.py:375-377`、
   `tilelang/cache/kernel_cache.py:141-142`）。你改了 pass 本体但 key 没变，就会
   拿到旧产物。
5. 两种办法：① `TILELANG_DISABLE_CACHE=1`（或 `tilelang.disable_cache()`）跑一次；
   ② 换 `TILELANG_CACHE_DIR` 到一个空目录，或手动清掉 `~/.tilelang/cache` 下对应
   条目。也可以打开 `TILELANG_KERNEL_CACHE_USE_LIB_STAMP`，让库哈希进 key。

**常见错误**：以为“改了代码就一定重新编译”。要重新编译，必须让 key 里的某个字段变，
或者关掉缓存。

---

## E11 动手题 ★★

**解题思路**：把“读生成代码”的方法固化成脚本与清单。

**参考答案（要点）**：

一个可借鉴的结构（本章 `code/02-inspect-source/inspect_source.py` 就是这么做的）：

1. 一张对照表，每条包含“Python 片段 / 对应 CUDA / 为什么”三元组；至少覆盖：
   * `T.Kernel(ceil(N/256), threads=256)` → `__launch_bounds__(256)` + `gridDim.x`；
   * `T.Parallel(threads)` → `if (threadIdx.x < 32)` + 向量宽度 8；
   * `C[b_x*threads+i] = A[...] + B[...]` → `uint4` 向量化 + `nv_bfloat162`；
   * `T.alloc_fragment` → 局部数组 `bfloat16_t A_local[8]`；
   * `T.copy` → `*(uint4*)(...)` 的 128-bit 装载/存储。
2. 五步检查清单（示例）：
   ① 函数签名，kernel 参数是否都成了指针参数？
   ② 网格，`blockIdx` 用在哪些下标上，和 Python 的 `bx`/`by` 是否一致？
   ③ 线程，有没有出现 `threadIdx`，向量宽度是几（`uint4` → 8×bfloat16）？
   ④ 边界，有没有 `if (... < N)`？出现在哪些循环里？
   ⑤ 搬运，`T.copy` 落成了什么（`uint4` / `cp.async` / TMA）？有没有被向量化？

**常见错误**：只把 CUDA 抄下来不对照。对照的价值在于“发现编译器的意图和你写的不一样”。

---

## E12 概念题 ★★

**解题思路**：三处差异各自回答“为什么不是语法糖”。

**参考答案**：

1.
   * **方言不同**：`tilelang.language` 是 CUDA 方言的 re-export
     （`tilelang/language/__init__.py:11-12`）。昇腾用
     `tilelang.ascend.language`。混用会拿到签名不同的 `T.Kernel`。这不是别名问题，
     是**两套 launch 语义**。
   * **网格是一维核网格**：昇腾 `T.Kernel` 只接受一个维度，多给会 `raise ValueError`
     （`tilelang/ascend/language/kernel.py:139-172`）。CUDA 的“网格 + 线程”两层，
     在昇腾上被拆成“核网格 + `T.SimtVF` 线程域”两个独立概念。
   * **片上存储映射不同**：`T.alloc_shared` 在 CUDA 是 shared memory，在昇腾落到
     UB（`tilelang/ascend/language/allocate.py:15`），L1 / L0A / L0B / L0C 另有
     `alloc_l1` / `alloc_l0a` / …。同一个函数名指向不同物理存储，**决定了数据放在
     哪一级是必须显式表达的**。
2. 会直接报错：`raise ValueError(“Ascend targets a 1-D core grid: T.Kernel(N) takes
   exactly one grid extent ...”)`，见 `tilelang/ascend/language/kernel.py:172-173`。
3. 至少三处：`T.Kernel(n)` vs `T.Kernel(n, is_npu=True) as (cid, vid)`；
   `T.alloc_shared` vs `T.alloc_ub`；`T.SimtVF` / `T.SimdVF` vs `T.Scope("V")` +
   `T.barrier_all`（还可加：`T.Parallel` vs `T.tile.add` 家族；有无
   `tilelang.cache.clear_cache()`）。不能直接抄，因为两者是**不同的前端实现**，
   同名 API 的签名与语义都可能不同；照抄的典型后果就是 `AttributeError` / 参数报错。

**常见错误**：以为“都是 TileLang，代码应该通用”。**前端表面共享，方言和物理映射不共享。**

---

## E13 代码阅读题 ★★

**解题思路**：分清“核网格”与“线程域”两层。

**参考答案**：

1. `NUM_CORES` 是**核网格大小**（多少个 AI core 参与），`128` 是
   `T.SimtVF` 声明的一个**线程域**里的线程数。两者不是同一层的东西：前者是 kernel
   最外层的并行切分，后者是某个 region 内部的线程并行。
2. 因为 CUDA 的 `threads=` 只是给同一层 launch 附加的“每块线程数”，而昇腾的核网格
   本身没有线程概念（`kernel.py:139-152` 的 docstring 说 “There is no SIMT thread
   domain at this scope”），线程并行必须单独用 frame 声明。这正是本章 7.1 节的
   第 2 处差异。
3. 说明**“搬运”这个部件也是可选的**：`out[...] = x[...] * 2.0` 直接在全局内存上
   计算。与 E2 第 3 问一致，必需的只有网格与并行域。

**常见错误**：把 `T.SimtVF(threads=128)` 理解成 CUDA 的 `threads=128`。它声明的是
一个 region 级的执行域，作用范围与 `T.Kernel` 不同。

---

## E14（可选，需要硬件）★★★

**解题思路**：这是一道实验题，重点在于**验证级别与测量口径自洽**。

**参考答案（要点）**：

1. 记录环境：`python --version`、`nvcc --version` 或 CANN 版本、GPU / NPU 型号、
   `tilelang.__version__`、`torch.__version__`。
2. `code/01-first-kernel/vecadd_tilelang.py` 会打印 `variant a` 的生成 CUDA。
   把它保存下来。
3. 对照 `elementwise.md` 的两份形态：如果看到 `uint4` / `nv_bfloat162`，说明命中
   向量化；如果看到 `#pragma unroll` + `if (... < N)`，说明走了边界判断版本。
   差异点至少写：向量宽度、线程利用率、是否有边界分支。
4. `N=2047` 时重跑：网格数、`if` 条件里的常数、访存宽度都会变。**注意**：
   如果你把 `N` 写成 2047，tile 不整除，预期会看到边界判断版本。
5. **不要写没实测的性能数字**。要报延迟/带宽，必须固定 shape、dtype、backend、
   warmup / repeat；用 `kernel.get_profiler().do_bench()` 或
   `tilelang.profiler.do_bench`，并注明测量口径。

**常见错误**：
* 换了 `N`（或 `block_N`）之后没有重新记录环境/口径，导致无法归因；
* 把“编译耗时”算进 kernel 延迟；
* 在没有 `torch.npu.is_available()` 确认的情况下就宣称“跑通了昇腾”。
