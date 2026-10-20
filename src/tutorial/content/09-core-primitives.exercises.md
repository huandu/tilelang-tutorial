# 第 09 章 习题

说明：

* 难度：`★`（基础，只需读懂本章）/ `★★`（需要综合本章多节）/ `★★★`（需要动手或读源码）。
* “需要硬件”一栏里，`否` 表示只需要一台能跑 Python 的机器。本章**没有**强制要求硬件的
  题目；所有必须在加速器上完成的实验都标为“可选”。
* 所有题目只依赖第 1–9 章的知识；涉及第 10 章及以后的 layout 细节时会显式提示。

---

## E1 概念题 ★ 不需要硬件

请用一句话分别回答：

1. “原语是接口而不是实现”这句话里的“接口”指什么？
2. 前端函数 `T.copy` 在绝大多数情况下真正做的最后一件事是什么（给出函数调用形态）？
3. 有哪一类 `T.copy` 会**根本不产生 tile op**，而是直接变成一条赋值语句？判据是什么？

---

## E2 概念题 ★ 不需要硬件

把下面五种 `T.copy` 的（源 scope，目标 scope）与它最可能的指令形态连起来，
并说明每条连线的判据在 `src/cuda/op/copy_analysis.cc` 里对应哪一条检查：

```text
(global, shared)   , 对齐 且最后一维位宽是 128 的倍数
(global, shared)   , 显式 T.async_copy
(shared, fragment)
(fragment, shared)
(tmem,   fragment)
```

可选答案：`cp.async`、`cp.async.bulk.tensor`（TMA）、`ldmatrix`、`stmatrix`、
`tcgen05.ld`。

---

## E3 概念题 ★ 不需要硬件

关于“越界与 padding”：

1. SIMT 路径下，源越界时读到的是什么值？默认值从哪来，怎么改？
2. 目标越界时行为是什么？和源越界有什么不同？
3. TMA 路径的 OOB 处理与 SIMT 路径有何本质不同？
4. 昇腾 `T.copy(..., pad_value=v)` 的 `pad_value` 是“越界填充”还是别的东西？
   请说清它的触发条件和单位（多少字节）。

---

## E4 概念题 ★★ 不需要硬件

写出下面四个 `T.gemm` 调用各自对 A、B 的**逻辑形状**要求（用 M/N/K 表示），
并指出哪一个在 v0.1.15 里会直接报参数错误：

```python
T.gemm(A, B, C)
T.gemm(A, B, C, transpose_B=True)
T.gemm(A, B, C, transpose_A=True, transpose_B=True)
T.gemm(A, B, C, accum_dtype="float32")
```

---

## E5 概念题 ★★ 不需要硬件

1. `T.gemm(..., clear_accum=True)` 除了“语义上先清零”之外，还会在 IR 层面改变什么？
   指出具体字段与源码行。
2. `GemmWarpPolicy` 的三个取值是什么？在 CUDA 后端，`tcgen05`、`wgmma`、`fma`
   三条路径分别怎么使用它？
3. 为什么说“`T.gemm` 从 mma 换成 wgmma 会连带改变 shared layout”？
   用源码里的哪条规则解释？

---

## E6 概念题 ★★ 不需要硬件

1. `T.reduce_sum(A, out, dim=1)` 与 `T.reduce_sum(A, out, dim=1, clear=False)` 的
   数值语义分别是什么？
2. 前端为什么会走两条不同的代码路径？请指出 `reduce_op.py` 里对应的分支。
3. 一个 `T.reduce_sum` 在 256 线程的 block 上、`scale=1`、归约宽度 256，会经过哪些
   通信层级？请按 `tl::AllReduce` 的 butterfly 顺序描述（提示：看 `offset` 从多少开始、
   在哪个阈值换机制）。
4. `batch=4` 为什么能省时间？省的是什么开销？

---

## E7 概念题 ★★ 不需要硬件

1. `T.atomic_add` 的标量路径与 tile-region 路径各自的判据是什么？
2. 哪条路径支持 `return_prev=True`？另一条会怎样？
3. CUDA 上 `use_tma=True` 走的是什么指令？它对 dtype 有什么限制？
4. 昇腾侧的 `T.set_atomic("add", "float32")` 为什么被称为“有状态”？
   忘记配对调用会出现什么后果？

---

## E8 代码阅读题 ★★ 不需要硬件

下面这段代码（节选自 `_vendor/tilelang-v0.1.15/examples/norm/rms_norm.py:14-36`，
已核对，未运行）：

```python
with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
    A_shared = T.alloc_shared((blk_m, blk_k), dtype)
    A_local = T.alloc_fragment((blk_m, blk_k), dtype)
    A_powsum = T.alloc_fragment((blk_m,), dtype)

    num_k_step = T.ceildiv(N, blk_k)
    T.clear(A_local)
    for k in T.Serial(num_k_step):
        T.copy(A[bx * blk_m, k * blk_k], A_shared)
        for i, j in T.Parallel(blk_m, blk_k):
            A_local[i, j] += A_shared[i, j] * A_shared[i, j]
    T.reduce_sum(A_local, A_powsum, dim=1)
```

请回答：

1. `T.clear(A_local)` 会生成什么？为什么它不能省掉？
2. `T.copy(A[bx * blk_m, k * blk_k], A_shared)` 里两边的 extent 分别从哪里来？
   这里有没有写 region 切片？为什么这样写是合法的？
3. `A_powsum` 的合法 shape 有几种？`(blk_m,)` 属于哪一种？
4. 这个 kernel 里 `T.reduce_sum` 的 `clear` 参数用了什么默认值？如果这里写
   `clear=False`，结果会错在哪里（假设 `A_powsum` 没被初始化过）？

---

## E9 代码阅读题 ★★ 不需要硬件

下面是从 `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend_tile.py` 摘出的
`select` 的 docstring 与 `gather` 的语义描述（未运行）：

```text
T.tile.select(dst, selMask, src0, src1, selMode)
    dst[i] = selMask.bit[i] ? src0[i] : src1[i]

T.tile.gather(dst, src, src_offset, src_base_addr)
    dst[i] = src[(src_base_addr + src_offset[i]) / sizeof(dtype)]
```

请回答：

1. `selMask` 的 dtype 是什么？它的第 i 个元素控制几个输出元素？
2. `T.tile.gather` 的 `src_offset` 单位是元素下标还是字节？请给一个数值例子：
   dtype 为 `float32`，`src_base_addr=64`，`src_offset[0]=8`，则 `dst[0]` 取自
   `src` 的第几个元素？
3. `T.tile.sort` 与 `T.tile.topk` 的输出布局有什么共同点？它们各自如何处理
   `actual_num` 之外的填充位置？

---

## E10 代码阅读题 ★★★ 不需要硬件（可运行）

运行 `code/02-copy-predicate-model/copy_predicate_model.py`（纯标准库），
然后回答：

1. 该脚本把 `src_predicate` 与 `dst_predicate` 分开计算，分别对应
   `src/op/copy.cc` 里的哪两行？为什么源侧要用 `select`、目标侧要用 `if`？
2. 把 `--safe-value -1.0` 传进去，输出有什么变化？对应源码里的哪个变量？
3. 脚本里 `extent` 与 `access` 的关系是模拟什么？如果 `extent` 正好整除 tile 大小，
   脚本还会生成谓词吗？这和源码里 `CanProve` 的短路逻辑是否一致？

---

## E11 动手题 ★★★ 不需要硬件（可运行）

打开 `code/03-thread-reduce-model/thread_reduce_model.py`：

1. 把 `--threads 256 --scale 1` 与 `--threads 32 --scale 1` 分别跑一遍，
   记录输出里 `shared-memory rounds` 与 `shuffle rounds` 的数量。
2. 用 `tl::AllReduce` 的实现解释这两个数字（`src/tl_templates/cuda/reduce.h:270-295`）。
3. 把 `--threads 48` 跑一遍，脚本应该报错。请说明这个约束在源码里的哪一条
   `static_assert` 上，并解释为什么必须是 2 的幂。
4. 用 `--clear true` 与 `--clear false` 各跑一次，说明 `clear` 如何改变结果，
   并对照 `tilelang/language/reduce_op.py:141-158` 的分支。

---

## E12 动手题 ★★★ 不需要硬件，但需要读源码

用 `T.copy` 的映射表（本章 §2.2）为下面四个调用各写一行“预测”，
然后用 `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc` 的
`SelectCopyInstForLowering` 逐条核对（只读源码，不编译）：

```python
T.copy(A_g, A_s)                                    # global → shared, fp16, 行宽 64
T.copy(A_g, A_s, disable_tma=True)                  # 同上
T.copy(A_g, A_s, prefer_instruction="cp_async")
T.async_copy(A_g, A_s)
```

要求给出：

* 每一行最可能选到的 `CopyInst`（`kNormal`/`kCPAsync`/`kBulkLoad`/`kBulkStore`/…）；
* 判定链上的第几步决定了它（用 §4.1 的编号）；
* 哪两个调用在 `analyze` 层会因“约束不满足”而**直接编译失败**而不是回退。

（可选硬件题：在有 NVIDIA GPU 的机器上编译这四个 kernel，打印生成的 CUDA 源码，
与你预测的结果对照。请在答案里写明是哪一个硬件/驱动/CUDA 版本。）
