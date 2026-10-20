# 第 08 章 习题答案

与 `exercises.md` 的 E1–E15 一一对应。每题给**解题思路**、**参考答案**与
**常见错误**。凡涉及源码的答案都给出 `_vendor/` 里的位置，方便复核。

---

## E1 概念题 ★

**思路**：这一类题只要把「位置参数 → 轴」和「关键字参数 → launch 注解」两条
规则分开记，就不会乱。

**参考答案**

1. `A` 对应 x 轴的块数（`gridDim.x`），`B` 对应 y 轴的块数（`gridDim.y`）。
   它们被写进 `T.Kernel` 的 `*blocks` 位置参数，顺序就是轴的顺序。
2. `bx` 等价于 `blockIdx.x`，`by` 等价于 `blockIdx.y`。这一点在
   `_vendor/tilelang-v0.1.15/src/ir.cc:236-262` 里被写死：每个轴建一个标签为
   `blockIdx.x/y/z` 的绑定帧，变量名依次是 `bx`、`by`、`bz`。
3. 删掉 `threads=` 不会报错。SIMT 后端会用 `DEFAULT_SIMT_THREADS = 128`，
   定义在 `_vendor/tilelang-v0.1.15/tilelang/transform/__init__.py:201`，
   由 `MaterializeKernelLaunch` 在 pass 阶段补上。
4. 因为 `_normalize_bindings`（`_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:88-96`）
   在只有一个绑定时返回**单个 Var**，多个绑定时返回列表；而
   `src/ir.cc:29-36` 附近还给 `Var` 补了 `__iter__` / `__len__`，所以
   `as (bx,)` 也能解包。

**常见错误**：把 `A`、`B` 说成「线程数」；或者以为 `as (bx, by)` 的顺序与
`A`、`B` 的书写顺序相反（x 是先写的那个）。

---

## E2 概念题 ★

**思路**：三种写法能等价，是因为它们都走同一个归一化入口。八个 fp8 成员要
按「指数/尾数」记，不能只记名字。

**参考答案**

1. 是同一个 dtype。字符串走 `_CANONICAL_TO_DISPLAY_STR`（`dtypes.py:105`）、
   torch dtype 走 `_TORCH_DTYPE_TO_STR`，最后都进 `dtype.__new__`
   （`dtypes.py:245-253`），产出同一个 `tvm.DataType`。
2. `float8_e3m4`（3/4）、`float8_e4m3`（4/3）、`float8_e4m3b11fnuz`（4/3，
   bias-11 FNUZ）、`float8_e4m3fn`（4/3，CUDA 默认）、`float8_e4m3fnuz`（4/3，
   ROCm 默认）、`float8_e5m2`（5/2）、`float8_e5m2fnuz`（5/2）、
   `float8_e8m0fnu`（8/0，只有指数，用作 MX 块 scale）。
3. `float4_e2m1fn` 是 4 bit（打包，一字节两个值）；`float4_e2m1_unpacked` 是
   8 bit（一个 FP4 值占一个字节槽）。前者用于全局张量与打包 SMEM 往返，后者
   只用于 SMEM/TMA，配对 tcgen05 的 `kind::f8f6f4` / `mxf8f6f4` 路径 —— 对应
   不同的 TMA 描述符类型，所以必须是两个 dtype。见 `dtypes.py:268-304`。
4. 不合法。`bfloat16` 只有 `bfloat16` 和 `bfloat16x2` 两个名字，
   没有 x4 及以上（`dtypes.py` 的 `_all_dtypes` 里可以逐个核。
   `code/02-dtypes/dtype_families.py --diff` 会打印 `exact match`）。

**常见错误**：把 `float8_e4m3` 与 `float8_e4m3fn` 当成同一个类型；
以为所有 dtype 都有 `x2…x64`。

---

## E3 代码阅读题 ★★

**思路**：先看 `threads=` 怎么被补齐成三维，再看 `get_thread_bindings` 返回什么，
最后算下标范围。

**参考答案**

1. 补齐后是 `(64, 2, 1)`，所以 `blockDim.x = 64`、`blockDim.y = 2`、
   `blockDim.z = 1`。补齐发生在 `_normalize_threads`
   （`_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:99-135`）：列表/元组
   会被补成三维，整数会被写成 `[threads, 1, 1]`。
2. 报错在 `tx, ty = T.get_thread_bindings()` 这一行：
   `get_thread_bindings()` **永远返回三个**（`kernel.py:244-249`），
   两个变量解包三个值会抛 `ValueError: too many values to unpack`。
   顺序固定是 `threadIdx.x`、`threadIdx.y`、`threadIdx.z`
   （dim=0 → x，1 → y，2 → z，见 `kernel.py:237-243`）。
   改法：

   ```python
   tx, ty, _tz = T.get_thread_bindings()
   ```

   注意 `tz` 一定存在且 `threads=(64, 2)` 时它的 extent 是 1。
3. `tx ∈ [0, 64)`、`ty ∈ [0, 2)`，所以 `ty*64 + tx ∈ [0, 128)`。
   因此要求 `N >= 128`；否则要加 `if ty*64 + tx < N` 的守卫（或让
   `LegalizeSafeMemoryAccess` 自己插）。
4. 能用，但语义变了：不写 `threads=` 时 trace 期不知道线程数，而这段代码并没有
   调用 `get_thread_extent()`，所以不会报错；`get_thread_bindings()` 仍然返回
   三个占位 Var，实际线程数由后端决定（默认 128）。
   **但是**这里最危险的地方是：`ty*64 + tx` 这个公式是**硬编码 64** 的，
   一旦实际 `blockDim.x` 不是 64，索引就是错的。

**常见错误**：以为 `threads=(64, 2)` 会得到 `blockDim=(64,2)`（其实是三维）；
以为 `get_thread_bindings()` 只返回用到的那几个（固定三个）；
以为返回值顺序是 y、x。

---

## E4 概念题 ★★

**思路**：两个问题——「谁看得到」（块内共享 / 每线程私有）和「活多久」
（一次迭代 / 整个块 / 整个 kernel）。

**参考答案**

1. `T.alloc_shared((128, 32), "float16")`。它要在 K 循环里被反复读，块内所有
   参与 GEMM 的线程都要用它，且要跨循环迭代存活。
2. `T.alloc_fragment((128, 128), "float32")`。累加器逻辑上是块形状，但应该切成
   每线程的寄存器；用 shared 会把它放到片上 SRAM 里去，既慢又占地方。
3. `T.alloc_local((32,), "float32")`。每线程私有、需要自由下标。
4. `T.alloc_var("int32", init=0)`。单元素标量。

为什么把累加器放 shared 通常是错的：`local.fragment` 映射到寄存器，访存
延迟最低、且天然是每线程私有的；`shared.dyn` 是块内共享的 SRAM，
(a) 每次读写都要过 SRAM 端口，(b) `128×128` fp32 就是 64 KiB，
要和 A/B 的 tile 抢同一块片上空间，(c) 还得自己处理线程间的写冲突。
见 `docs/get_started/overview.md:82-84`。

**常见错误**：只说「shared 更快」。**shared 比 global 快，但比寄存器慢**；
这条题的关键是「累加器应该是每线程私有的寄存器」。

---

## E5 代码阅读题 ★★

**参考答案**

1. `128 × 128 × 4 B = 65536 B = 64 KiB`。
2. `64 KiB / 128 线程 = 512 B/线程`；`512 / 4 = 128` 个 fp32 寄存器/线程。
   要警惕是因为 NV 每线程最多 255 个 32-bit 寄存器，128 个已经吃掉一半，
   直接压低 occupancy。
3. 至少两种：
   * 把 dtype 从 fp32 换成 fp16/bf16：`128×256×2×2 = 128 KiB`，放得下，
     代价是累加精度（如果这其实是累加器，就不要这么做）。
   * 把 tile 切成两半，或减少 `num_stages`（例如从 2 改成 1，即
     `128×256×4 = 128 KiB`），代价是复用量或流水重叠度下降。
   * 也可以把 tile 形状改小（例如 `(64, 256)`）。
4. `196608` 是昇腾官方文档里出现的**某个型号示例**的 192 KiB UB 口径，
   不是通用规格。正确做法是在真实设备上用 `GetCoreMemSize` 查询，
   或查对应型号文档。这一点第 6 章 §7 也强调过。

**常见错误**：把 `64 KiB / 128` 算成 `512 KiB`；把 192 KiB 当成
「昇腾 UB 都是 192 KiB」。

---

## E6 概念题 ★★

**参考答案**

1. `T.Parallel(128)`。元素间无依赖，交给布局推导最省心。
2. `T.serial(32)`。这是归约，必须顺序累加（或者用 `T.reduce_*`，见第 9 章）。
3. `T.Pipelined(..., num_stages=...)`。它专门表达「搬运与计算重叠」。
4. `T.vectorized(8)`。它就是「把这几次迭代拼成一条向量操作」。
5. `T.unroll(4)`。次数少、希望摊掉循环开销。

默认差异：`T.serial` 由**单个线程**顺序执行整段循环（除非被外层 `T.Parallel`
分了活）；`T.Parallel` 是**编译器决定**谁来执行——先展平、再按
lane → thread → 每线程串行复制切分。

**常见错误**：把第 1 题答成 `T.serial`（「反正 128 次」）；把第 5 题答成
`T.unroll` 时说不清「请求」与「保证」的区别。

---

## E7 代码阅读题 ★★★

**思路**：照着 `PlanLoopPartition` 的四步公式算一遍。

**参考答案**

1. `(0, 2)`：`flattened = 0*4 + 2 = 2`；`access_idx = 2 // 2 = 1`；
   `thread = 1 % 4 = 1`；`local_idx = (1 // 4)*2 + (2 % 2) = 0`。所以是
   `(thread=1, slot=0)`，与输出一致。
2. 因为映射是**先对整个 nest 做行优先展平**再切分的。两种 nest 的总迭代数
   一样（64），所以每线程的次数一样；但展平后的「第 k 个迭代点」对应哪一组
   `(i, j)` 不同，因此归属不同。见
   `_vendor/tilelang-v0.1.15/src/transform/loop_partition.cc:248-271`。
3. `_vendor/tilelang-v0.1.15/src/transform/loop_partition.cc:248-271`
   （`class LoopPartitioner` 的 `Partition`）。
4. `T.Parallel(BM, BN)`（最内层是 N）更可能合并。行优先展平让最内层
   （`j`，对应列）变得最快，于是相邻线程拿到相邻的列 → 相邻地址。
   换成 `T.Parallel(BN, BM)` 时最快变化的维度变成 `i`（行），相邻线程会跨行
   访问，stride 变成 N，访存就散了。

**常见错误**：以为 `T.Parallel(a, b)` 里的第一个维度一定是线程维度。

---

## E8 动手题 ★★（CPU）

**思路**：跑脚本，读输出，注意「每线程次数」与「归属」是两件事。

**参考答案（判据）**

1. 默认参数下每个迭代点都有一行 `index → (thread, slot)`，且
   `slot` 从 0 单调递增。
2. `--compare` 会打印五种 nest 形状，每种都给出 `thread 0` / `thread 1`
   拿到的前四个下标。可以看到 `T.Parallel(8,8)` 的 `thread 0` 拿到
   `(0,0) (0,1) (2,0) (2,1)`，而 `T.Parallel(2,32)` 的 `thread 0` 拿到
   `(0,0) (0,1) (0,16) (0,17)`。
3. 例如 `--extents 4 4 --threads 32 --vector 2`：总迭代数 16，而
   `access_idx` 最大只有 `16/2 - 1 = 7`，所以 `thread = access_idx % 32`
   只会取到 0..7 —— **32 个线程里有 24 个一个点都分不到**。
4. `--vector 1` 之后 lane 那一层退化：`access_idx = flattened`，
   `local_idx = (flattened // num_thread)`，也就是「一个迭代点一个元素」，
   不再有向量合并。

**常见错误**：把 `--compare` 的输出读成「nest 形状无所谓」。

---

## E9 动手题 ★★（CPU）

**参考答案（判据）**

1. `per-axis tiles: M -> 4, N -> 6`；`grid = (6, 4)`；24 个块；
   尾部 M 多 24 行、N 多 68 列。
2. `x_is_n=True` 时块号先扫 N（`pid 0..5` 都是 `by=0`），
   `x_is_n=False` 时块号先扫 M（`pid 0..3` 全是 `bx=0..3`）。也就是
   **x 轴（位置参数的第一个）变得最快**。
3. 因为 `bx = pid % gx`，`pid` 每加一，`bx` 就加一；直到 `pid` 达到 `gx`
   才进位到 `by`。所以按线性块号走一遍会先把 `by=0` 那一行的所有 `bx`
   扫完。
4. `pid=7`、`gx=3, gy=2, gz=2`：`bx = 7 % 3 = 1`、
   `by = (7 // 3) % 2 = 2 % 2 = 0`、`bz = 7 // (3*2) = 1`。所以是
   `(bx=1, by=0, bz=1)`。

**常见错误**：把 `bz` 算成 `pid // gx`（漏掉 `gy`）。

---

## E10 概念题 ★★

**参考答案**

1. `T.Select` **两边都求值**；`T.if_then_else` 只求值被选中的那一支。
   原始说明在 `_vendor/tilelang-v0.1.15/tilelang/language/tir/ir.py:172-179`
   （`Select` 的 docstring：「``Select`` evaluates both value expressions.
   Use ``if_then_else`` when a branch must guard side effects or out-of-bounds
   memory accesses.」）。
2. A 不会读 `X[gi]`；B 会读。
3. 两边都安全、只想按条件选一个值时。`T.Select` 通常更接近硬件 select，
   没有分支开销。
4. P 会把全部 128 个值乘 2；Q 只把前 64 个乘 2（其余保持原值）；
   R 里的 `x = 3 if 1 > 2 else 4` 是**纯 Python** 的条件表达式，
   `1 > 2` 在 trace 期就求值成 `False`，所以 `x = 4`，
   循环体等价于 `C[i] = C[i] * 4`——也就是说 R 会把 128 个值都乘 4。

**常见错误**：第 4 题里把 R 当成「不会执行」。纯 Python 的 `if/else` 只是
被 trace 期折叠，不是被删掉。

---

## E11 代码阅读题 ★★

**参考答案**

1. `T.all_of(buffer)` 的语义是「这个缓冲区（或缓冲区切片）里的**所有元素**都为
   真」。它接受 `BufferLikeType` —— 具体是 `Buffer` 或 `BufferRegion`
   （`_vendor/tilelang-v0.1.15/tilelang/language/logical.py:12-87`）。
2. 不一致。文档给的是条件列表形式，源码接的是缓冲区。正确写法是用 `and`：

   ```python
   if (i < M) and (j < N):
       ...
   ```

   （`and` / `or` / `not` 在 TIR 表达式上会降级成 `tir.And` / `tir.Or` /
   `tir.Not`，见 `tilelang/language/eager/builder.py:820-846`。）
   如果确实想表达「某个 mask 切片全为真」，就按缓冲区形式写：
   `T.all_of(mask_shared)` 或 `T.all_of(mask[by, bx, k, :])`。
3. 例如 `_vendor/tilelang-v0.1.15/testing/python/language/test_tilelang_language_all_of.py:56`
   的 `if T.all_of(BlockMask[by, bx, k, :]):`。
4. **文档与源码冲突时，以锁定版本的源码为准**（并用该版本的测试交叉验证）。

**常见错误**：看到文档例子就照抄，然后在编译期报一个「参数个数不对」的错却
不知道从哪查起。

---

## E12 概念题 ★★

**参考答案**

1. 能用 —— 但它在 `for` 语句里会被当成 `T.serial(N)`，即**设备循环**，
   不是 trace 期展开。
2. 能用 —— 同样是设备侧的 `T.serial`，这次边界是运行期符号。
3. 能用 —— 循环一个具体的 Python 列表是**trace 期展开**（必须有限）。
4. 不能。`len()` 在 kernel 里不可用；用 `X.shape[0]`。
5. 不能。链式赋值要拆成 `a = 0`、`b = 0`。
6. 不能。kernel 里不能定义函数；用 `@T.macro` 定义可复用代码块。
7. 不能直接用。用 `T.device_assert` 或 `T.assert`。
8. 设备侧要用 `T.print(...)`；`print()` 只对 Python（host）表达式有效。
9. 能。列表推导式和 `enumerate()` 都是 trace 期展开；但要注意
   `range(16)` 的边界必须在 trace 期可求值，并且**这个 list 不是设备 buffer**——
   用运行期变量去索引它不会生成设备侧查表。

**常见错误**：把 1 和 3 混为一谈（一个设备循环、一个 trace 期展开）。

---

## E13 动手题 ★★（CPU）

**参考答案（判据）**

1. `5/5 self-tests pass`（`len()` BAD、链式赋值 BAD、class BAD、range 循环 OK、
   while OK）。
2. 内置 demo 报出 2 个 `BAD`：`len()`（不可用，用 `buffer.shape[dim]`）
   与 `a = b = c`（链式赋值不支持）。另有 4 个 `WARN`：两个 `def`、
   一个 `T.Parallel` 循环、一个 `open()` 上下文管理器。
3. 命令形如：

   ```bash
   python3 py_compat_scan.py ../05-control-flow/control_flow_demo.py
   ```

   输出里会出现若干 `WARN`（`def kernel_factory`、`def kernel` 等）与
   可能出现的 `BAD`。这些**基本是假阳性**：脚本不区分 kernel 体与 host 侧，
   把每一个 `def` 都当成了「kernel 里定义函数」。真正的判定要交给
   TileLang 自己编译。
4. 它**不能替代 TileLang 的解析器/编译器**——只是一个编译前的粗筛。

**常见错误**：把 `WARN` 当成错误；或者反过来，以为没有 `BAD` 就一定能编译过。

---

## E14 综合题 ★★★

**思路**：按「网格 → dtype → 作用域 → 控制流 → Python 边界」五条线索扫。

**参考答案（至少五处）**

1. **网格维度不够（§3）。** `T.Kernel(T.ceildiv(N, BN), threads=128)` 只有一维，
   但 kernel 明显要覆盖 M 和 N 两个方向（最后写的是 `C[0, bx*BN]`）。
   修法：改成二维网格并解包两个绑定：

   ```python
   with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
   ```

   并把 A 的 tile 起点从 `A[0, a]` 改成 `A[by * BM, a]`、
   回写改成 `C[by * BM, bx * BN]`。

2. **混合 A/B dtype 越界（§4.5）。** `A` 是 `float8_e4m3`、`B` 是 `float16`，
   两者既不同类也不都在 f8f6f4 家族里，`validate_gemm_ab_dtypes`
   （`dtypes.py:319-340`）会抛 `ValueError`。修法：两边统一到同一家族
   （例如都用 fp8），或先把 B 转成 fp8。

3. **`T.bfloat16x8` 不存在（§4.2）。** bf16 只有 `bfloat16` 和 `bfloat16x2`。
   而且即便存在，**GEMM 的累加器也不该用 bf16**——应该是 `float32`。
   修法：`C_f = T.alloc_fragment((BM, BN), "float32")`。

4. **`alloc_var` 是每线程私有，不能当块级标志（§5.1、§5.3）。** `mask` 被当成
   一个跨语句的「标志」用，但 `local.var` 是每个线程各有一份。如果它确实要
   在块内共享，应该用 `T.alloc_shared((1,), "float32")`；如果只是每线程自己的
   临时值，那就不该在条件里跨线程语义地使用。这里语义不清，建议改写或换 scope。

5. **`T.all_of(cond1, cond2)` 参数个数不对（§7.3）。** v0.1.15 的 `all_of` 接受
   一个 buffer / buffer 切片，不是条件列表。修法：

   ```python
   if (mask[0] > 0.0) and (ko < 4):
   ```

6. **`len(A)` 不可用（§8）。** 改成 `A.shape[0]`，或直接删掉（这个变量没被用到）。

7. **链式赋值 `a = b = ko * BK` 不可用（§8）。** 拆成两行。

8. **流水循环体里放条件分支（§7.5）。** `T.Pipelined` 的 `order` / `stage` 是按
   可调度语句编号的，循环体里出现 `if` 会改变基本块结构与 `Bind` 的可重放性。
   建议把这类数据相关的条件挪到独立的 `T.Parallel` 里，或至少确认它不会
   破坏流水划分。

**常见错误**：只找出 dtype 与 `len()` 两处，漏掉网格维度与 `alloc_var` 的
语义问题；或者把 `mask[0] > 0.0` 的写法当成合法 `all_of`。

---

## E15 动手题 ★★★（可选，需要硬件）

**这道题在本机**（macOS，无 GPU 无 NPU）**没有执行**，所以下面只给「预期判据」
与「怎么记录」，不给结果数字。

**思路与判据（NVIDIA 侧）**

1. 用 `guard_costs_vectorization.compile(N=2047)` 与
   `guard_free_full_blocks.compile(N=2047)`，各自 `get_kernel_source()`。
2. 预期判据：带守卫那一份应当出现「标量循环 + 谓词」（形如
   `if (gi < 2047)` 包着单个元素访问），而无守卫那一份因为
   `gi < full <= N` 恒成立，规划器**有可能**发出向量化访问（如 `float4`）。
   **请以你实际看到的源码为准**；如果两份都没出现向量化，请看
   `coalesced_width=` 与 `T.Parallel` 的 extents 是否满足了向量化的前提条件
   （第 6.4 节）。
3. 改 `num_per_thread` 会改变每线程的元素数，进而改变可用的向量宽度。
   记录每次改动后的源码差异。
4. 记录模板：

   ```text
   验证级别：在真实硬件上运行过
   GPU 型号：____      CUDA 版本：____      TileLang 版本：v0.1.15
   编译出的向量化宽度：____      观察到的差异：____
   ```

**判据（昇腾侧）**

1. 跑 `GetCoreMemSize` 记录该设备的 UB / L1 / L0C 容量。
2. 与 `scope_footprint.py` 里的 `196608`（192 KiB）对照。**大概率不相等** ——
   192 KiB 只是官方文档里某个型号的示例值。这正是第 6 章与本章 §5 反复强调
   「容量数字一定要带型号和出处」的原因。

**常见错误**：把「预期判据」抄成「实验结果」。本章禁止在没有硬件的情况下
声称跑过。
