# 第 08 章 习题

说明：

* 难度标注为 `★`（基础）/ `★★`（需要综合本章内容）/ `★★★`（链条较长或需要动手）。
* “需要硬件”一栏中，`否` 表示只需要一台能跑 Python 的机器；`是（GPU）` /
  `是（NPU）` 表示需要真实加速器。本章**没有**强制要求硬件的题目，需要硬件的
  实验一律标为“可选”。
* 所有题目只依赖第 1–8 章的知识；涉及后续章节时会显式提示。
* 本章的代码示例分两种验证级别：纯标准库脚本是“CPU 逻辑验证”（本机跑过），
  TileLang 示例是“仅静态核对”（只保证 `py_compile` 通过、API 名与签名在
  `_vendor/tilelang-v0.1.15/` 里核对过，**没有**在硬件上运行）。作答时不要
  假设 TileLang 示例跑过。

---

## E1 概念题 ★ 不需要硬件

`with T.Kernel(A, B, threads=128) as (bx, by):`

1. `A` 和 `B` 分别对应哪个网格维度？它们写进哪个 launch 参数？
2. `bx` / `by` 分别等价于 CUDA 里的什么？
3. 如果把 `threads=128` 删掉会发生什么？默认值是多少，写在哪一行？
4. 为什么写成 `as (bx, by)` 和写成 `as (bx,)` 在只有一个位置参数时都能工作？
   请指出实现这一点的函数名与文件位置。

---

## E2 概念题 ★ 不需要硬件

1. 下面三种写法得到的 dtype 是同一个吗？为什么？

   ```python
   d = "float16"
   d = T.float16
   d = torch.float16
   ```

2. 说出 v0.1.15 里 float8 的**全部八个**成员名，并为每一个写出“指数位数 / 尾数
   位数”。
3. `float4_e2m1fn` 与 `float4_e2m1_unpacked` 的 `bits` 分别是多少？为什么同一个
   “FP4 E2M1”会有两种 dtype？
4. `T.bfloat16x8` 合法吗？给出理由。

---

## E3 代码阅读题 ★★ 不需要硬件

读 `code/01-kernel-grid/kernel_grid.py` 的 `two_dim_threads`（不需要运行）：

```python
with T.Kernel(M, threads=(64, 2)) as bx:
    tx, ty = T.get_thread_bindings()
    Y[bx, ty * 64 + tx] = X[bx, ty * 64 + tx]
```

1. `threads=(64, 2)` 会让 `blockDim.x` / `blockDim.y` / `blockDim.z` 分别是多少？
   补齐三维这一步在哪个函数里做？给出文件位置。
2. 上面这段代码有一处会立刻报错，是哪一处？`T.get_thread_bindings()` 实际返回
   几个值、顺序是什么？应该怎么改？
3. `ty * 64 + tx` 覆盖的下标范围是什么？要让这个 kernel 不越界，对 `N` 有什么
   要求？
4. 如果改成 `with T.Kernel(M, N) as (bx, by):`（不加 `threads=`），这段代码还能
   用吗？为什么？

---

## E4 概念题 ★★ 不需要硬件

为下面四份数据各选一个分配函数，并说明理由（用“谁看得到”“活多久”两条）：

1. 一个 `(128, 32)` 的 fp16 A 矩阵 tile，在 K 方向的循环里被反复读；
2. 一个 `(128, 128)` 的 fp32 累加器；
3. 一个每线程私有、长度 32 的 fp32 临时数组；
4. 一个用来记“当前块是否还要继续算”的 int32 标志。

再回答：为什么把累加器写成 `T.alloc_shared` 通常是错的？

---

## E5 代码阅读题 ★★ 不需要硬件

读 `code/03-memory-scopes/scope_footprint.py` 的输出（不需要运行）：

```text
  C fragment (fp32, per block)       (128, 128)   float32   x1  =    64 KiB
  C tile in UB (fp32, 2-stage)       (128, 256)   float32   x2  =   256 KiB   cap 192 KiB  DOES NOT FIT
```

1. 第一行的 64 KiB 是怎么算出来的？请写式子。
2. 如果把 `threads=128` 补上，这块 fragment 平均每个线程占多少字节、多少个
   fp32 寄存器？请写式子，并说明为什么这个数字要警惕。
3. 第二行要放得下，有哪些改法？（至少给两个，并说明各自的代价。）
4. 脚本里的 `196608` 是什么口径？直接用这个数当“昇腾 UB 有多大”合适吗？
   正确做法是什么？

---

## E6 概念题 ★★ 不需要硬件

对下面五种需求，各选一个循环构造子（`T.serial` / `T.unroll` / `T.vectorized` /
`T.Parallel` / `T.Pipelined`）：

1. “把 128 个元素各乘 2，彼此没有依赖”；
2. “沿 K 方向把 32 个数累加到同一个标量上”；
3. “把 A 的 tile 从 GM 搬到 shared，同时上一个 K 块的 GEMM 还在算”；
4. “希望这 8 次迭代拼成一条 `ld.global.v4`”；
5. “一个每块只走 4 次、开销想摊掉的小循环”。

再回答：`T.serial` 和 `T.Parallel` 在“谁执行”这件事上的默认差异是什么？

---

## E7 代码阅读题 ★★★ 不需要硬件

读 `code/04-loops/loop_mapping.py` 的输出（不需要运行）：

```text
$ python3 loop_mapping.py --extents 4 4 --threads 4 --vector 2
  index          flat  thread  slot
  (0, 0)            0       0     0
  (0, 1)            1       0     1
  (0, 2)            2       1     0
  ...
$ python3 loop_mapping.py --compare
  T.Parallel(8, 8)      per-thread=8    thread 0 owns (0,0) (0,1) (2,0) (2,1) ...
  T.Parallel(2, 32)     per-thread=8    thread 0 owns (0,0) (0,1) (0,16) (0,17) ...
```

1. 手算验证 `(0, 2)` 落在 `thread=1, slot=0`：写出 `flattened`、`access_idx`、
   `thread`、`local_idx` 四步。
2. 为什么 `T.Parallel(8, 8)` 与 `T.Parallel(2, 32)` 的“每线程次数”一样，
   但 `thread 0` 拿到的下标不同？
3. 这段映射公式的原始出处在哪里？给出文件与行号区间。
4. 假设 `Y[by*BM+i, bx*BN+j]` 里的行是连续的、`X` 是行主序二维张量。
   `T.Parallel(BM, BN)` 与 `T.Parallel(BN, BM)` 哪个更可能合并访存？为什么？

---

## E8 动手题 ★★ 不需要硬件（CPU）

在本机运行 `code/04-loops/loop_mapping.py`：

1. 默认参数跑一遍，记录每个迭代点 → (thread, slot) 的表；
2. 跑 `--compare`，记录五种 nest 形状的 `thread 0` 归属；
3. 自己构造一组 `--extents` / `--threads` / `--vector`，使得**有些线程完全分不到
   工作**，并解释为什么（提示：总迭代数 < threads 会发生什么）；
4. 把 `vector_size` 改成 1，观察 lane 那一层退化成什么。

---

## E9 动手题 ★★ 不需要硬件（CPU）

在本机运行 `code/01-kernel-grid/grid_index_model.py`：

1. 记录 `M=1000, N=700, BM=256, BN=128` 时的网格形状、块数与尾部 padding；
2. 把 `x_is_n` 从 `True` 改成 `False`（或直接改脚本），说明 `thread 0`/`pid 0`
   之后的块顺序发生了什么变化；
3. 用脚本里的 `linear_to_block` 解释：为什么“x 变得最快”意味着按线性块号
   走一遍会**先扫完一行**？
4. 新增一个 `grid = (3, 2, 2)` 的例子，手工验证 `pid=7` 分解成 `(bx, by, bz)`。

---

## E10 概念题 ★★ 不需要硬件

```python
Y[gi] = T.if_then_else(gi < N, X[gi], 0.0)      # A
Y[gi] = T.Select(gi < N, X[gi], 0.0)            # B
```

1. A 与 B 在求值行为上有什么区别？引用 `_vendor/tilelang-v0.1.15/` 里的
   原始说明（文件 + 行号）。
2. 当 `gi >= N` 时，A 会读 `X[gi]` 吗？B 呢？
3. 在什么情况下 B 反而更合适？
4. 下面这三段代码，哪一段会把 `C` 里的值乘 2、哪一段不会？为什么？

   ```python
   # P
   for i in T.Parallel(128):
       C[i] = C[i] * 2
   # Q
   for i in T.Parallel(128):
       if i < 64:
           C[i] = C[i] * 2
   # R
   x = 3 if 1 > 2 else 4
   for i in T.Parallel(128):
       C[i] = C[i] * x
   ```

---

## E11 代码阅读题 ★★ 不需要硬件

读 `_vendor/tilelang-v0.1.15/tilelang/language/logical.py:12-87`（不需要运行）：

```python
def any_of(buffer: BufferLikeType) -> tirx.PrimExpr: ...
def all_of(buffer: BufferLikeType) -> tirx.PrimExpr: ...
```

1. `T.all_of(buffer)` 的语义是什么？它接受的是什么类型？
2. `docs/programming_guides/control_flow.md:33-38` 里的
   `if T.all_of(i < M, j < N):` 与源码一致吗？如果不一致，正确的写法是什么？
3. 请到 `testing/python/language/test_tilelang_language_all_of.py` 里找一个
   **正确**的调用例子，并抄下来。
4. 这条题想说明什么工程习惯？请用一句话概括。

---

## E12 概念题 ★★ 不需要硬件

判断下面每一小段代码在 kernel 里能不能写；能写的说明它会变成什么，不能写的
说明该改成什么：

1. `for i in range(N):` （`N` 是 Python int）
2. `for i in range(N):` （`N` 是 `T.dynamic("N")` 出来的 Var）
3. `for k in [0, 1, 2]:`
4. `n = len(X)`
5. `a = b = 0`
6. `def helper(x): return x + 1`
7. `assert i < N`
8. `print("hello")` / `T.print(C, msg="C:")`
9. `scale_idx_table = [j * 2 for j in range(16)]`，然后
   `for idx, val in enumerate(scale_idx_table): out[idx] = val`

---

## E13 动手题 ★★ 不需要硬件（CPU）

在本机运行 `code/06-python-boundary/py_compat_scan.py`：

1. `--self-test` 的输出是什么？几项通过？
2. 默认（不带参数）时会报出哪些 `BAD`？它们各自的原因是什么？
3. 把 `py_compat_scan.py` 指向本章的
   `code/05-control-flow/control_flow_demo.py`（一条命令），记录输出里
   `BAD` 与 `WARN` 的数量，并解释：这里的 `BAD`/`WARN` 是真的越界，还是这个
   教学模型的假阳性？
4. 用一句话说明这个脚本**不能**替代什么。

---

## E14 综合题 ★★★ 不需要硬件

下面是一段“看起来对”的 kernel，请找出**至少四处**问题，并逐条给出修法。
（提示：网格、dtype、作用域、控制流、Python 边界各有一处。）

```python
@tilelang.jit
def bad(M, N, K, BM=128, BN=128, BK=32):
    @T.prim_func
    def kernel(A: T.Tensor((M, K), "float8_e4m3"),
               B: T.Tensor((K, N), "float16"),
               C: T.Tensor((M, N), "float16")):
        with T.Kernel(T.ceildiv(N, BN), threads=128) as bx:
            A_s = T.alloc_shared((BM, BK), "float8_e4m3")
            B_s = T.alloc_shared((BK, BN), "float16")
            C_f = T.alloc_fragment((BM, BN), T.bfloat16x8)
            mask = T.alloc_var("float32", 1.0)
            n = len(A)
            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
                a = b = ko * BK
                T.copy(A[0, a], A_s)
                T.copy(B[a, bx * BN], B_s)
                T.gemm(A_s, B_s, C_f)
                if T.all_of(mask[0] > 0.0, ko < 4):
                    C_f[0, 0] = C_f[0, 0] * mask[0]
            T.copy(C_f, C[0, bx * BN])
    return kernel
```

---

## E15 动手题 ★★★ 需要硬件（可选）

**（可选，只有有加速器的读者需要做。本章写作环境既无 GPU 也无 NPU，故本题未在
本机验证。）**

在 NVIDIA GPU 机器上：

1. 把 `code/05-control-flow/control_flow_demo.py` 的
   `guard_costs_vectorization` 与 `guard_free_full_blocks` 都编译到
   `N=2047`，打印两者的 CUDA 源码；
2. 回答：哪一份出现了向量化访存（如 `float4`）？哪一份只有标量 + 谓词？
   这与第 7.5 节的说法一致吗？
3. 把 `num_per_thread` 从 8 改成 4、16，观察向量化是否出现变化；
4. 明确记录你的验证级别是“在真实硬件上运行过”，并写清 GPU 型号、CUDA 版本、
   TileLang 版本（必须是 `v0.1.15`）。

在昇腾 NPU 机器上（二选一）：

1. 跑通 `code/03-memory-scopes/alloc_scopes.py` 里的 `ascend_scope_showcase`
   （需要把 `T.copy` 的目标尺寸调成合法值），用 `GetCoreMemSize` 记录该设备
   真实的 UB / L1 / L0C 容量；
2. 与 `code/03-memory-scopes/scope_footprint.py` 里的 192 KiB 对照，说明那是不是
   该设备的真实值。
