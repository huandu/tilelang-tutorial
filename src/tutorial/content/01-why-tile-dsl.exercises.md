# 第 01 章 习题

说明：

* 难度标注为 `★`（基础）/ `★★`（需要综合本章内容）/ `★★★`（需要动手或推理链条较长）。
* 「需要硬件」一栏中，`否` 表示只需要一台能跑 Python 的机器；`是（GPU）`/`是（NPU）`
  表示需要用真实加速器才能完成实验。本章没有强制要求硬件——所有必须在硬件上做的实验都
  被标注为「可选」。
* 所有题目只依赖第 1 章及更早的知识；涉及后续章节的地方会显式提示。

---

## E1 概念题 ★ 不需要硬件

阅读 1.1 和 1.2 节，回答：

1. 为什么说「一次算子慢，通常不是算法算错了」？
2. 一个算子的时间可能花在哪些动作上？请举出至少两类，并各给一个本章提到的例子。
3. 为什么同一个 `softmax` 在「先减最大值 → exp → 求和 → 除」的四遍实现下，会比理论上
   只需要两遍的实现慢？请从「访存次数」的角度解释。

---

## E2 概念题 ★ 不需要硬件

1. 写出算术强度（arithmetic intensity）与机器平衡点（machine balance）的定义式。
2. 用本章的 H100 数字（峰值 FP16 张量核 1000 TFLOPS，HBM3 带宽 over 3 TB/sec）估算该
   机器的平衡点，写出计算过程。
3. 判断下面两个算子在 H100 上分别属于 memory-bound 还是 compute-bound，并说明依据：
   * 对 $x \in \mathbb{R}^{4096\times4096}$（fp16）做逐元素 `y = max(x, 0)`；
   * $C = A B$，其中 $A, B, C \in \mathbb{R}^{4096\times4096}$（fp16，fp32 累加）。

---

## E3 概念题 ★★ 不需要硬件

本章 4.2 节提出：硬件演进分三个阶段（SIMT 线程模型 → 张量核 → 异步搬运与专用存储）。

1. 每个阶段分别给程序员新增了哪一种「必须人工管理的资源」？
2. 为什么说「复杂度是组合的，不是线性叠加的」？请用本章的「存储位置 × 布局 × 同步 ×
   流水 × 算力单元」这条线索说明。
3. 昇腾 950 的例子（Cube/Vector + L1/L0A/L0B/L0C/UB + MTE 流水）说明「GPU 是 SIMT、
   NPU 是 SIMD」这个概括为什么已经失效？

---

## E4 概念题 ★★ 不需要硬件

对照 4.5 节的表格，回答：

1. CUDA 主要把哪一类信息交给了程序员？它的通用性代价是什么？
2. Triton 用「tile 是语言第一等公民」换来了什么？又付出了什么代价？请引用 Triton
   2019 年原论文里的一句话作为依据。
3. TileLang 相比 Triton 多保留了哪一类信息？它为什么要保留这一类信息？

---

## E5 概念题 ★ 不需要硬件

给出本书对 tile 的定义，并逐条说明缺少其中任意一条会带来什么问题：

1. 形状（静态形状）；
2. 存储位置（storage scope）；
3. 归属（ownership）——它在编译期落实为具体的物理布局（layout）。

然后回答：为什么「对循环做分块（loop tiling）」不等价于「使用 tile 抽象」？

---

## E6 代码阅读题 ★★ 不需要硬件

读下面这段来自锁定版本官方 quickstart 的代码（`_vendor/tilelang-v0.1.15/examples/quickstart.py`，
已省略 `import`）：

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    A_shared = T.alloc_shared((block_M, block_K), dtype)
    B_shared = T.alloc_shared((block_K, block_N), dtype)
    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
    T.clear(C_local)
    for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
        T.copy(A[by * block_M, ko * block_K], A_shared)
        T.copy(B[ko * block_K, bx * block_N], B_shared)
        T.gemm(A_shared, B_shared, C_local)
    T.copy(C_local, C[by * block_M, bx * block_N])
```

请回答：

1. `T.Kernel` 的每次迭代负责输出矩阵的哪一部分？`bx`、`by` 分别对应哪一维？
2. 三个缓冲区分别被放在哪一级存储上？请指出你判断的依据（哪一行代码）。
3. 这段代码里，哪些语句在描述**数据流**，哪些在描述**调度决策**？请分成两组。
4. `num_stages=3` 想让哪两件事重叠？它的物理代价是什么？
5. 如果你把 `T.alloc_shared` 换成 `T.alloc_fragment`，会有什么后果？（提示：可结合
   第 2 章的存储层级，但仅凭本章也能判断。）

---

## E7 代码阅读题 ★★ 不需要硬件

读下面这段昇腾 GEMM + ReLU 的例子（`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:83-98`）：

```python
with T.Kernel(num_blocks) as bx:
    A_l1 = T.alloc_l1((block_M, block_K), T.bfloat16)
    B_l1 = T.alloc_l1((block_N, block_K), T.bfloat16)
    C_l0c = T.alloc_l0c((block_M, block_N), T.float32)
    C_ub = T.alloc_shared((block_M // 2, block_N), T.float32)
    for tile in T.Persistent([M // block_M * n_tiles], num_blocks, bx):
        m, n = tile // n_tiles * block_M, tile % n_tiles * block_N
        for k in T.Pipelined(K // block_K, num_stages=2):
            T.copy(A[m, k * block_K], A_l1)
            T.copy(B[n, k * block_K], B_l1, l2_cache_ctrl="NOTALLOC_KEEP")
            T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=(k == 0))
        T.dual_copy(C_l0c, C_ub)
        with T.SimtVF(threads=128):
            for i, j in T.Parallel(block_M // 2, block_N):
                C_ub[i, j] = T.max(C_ub[i, j], 0)
```

请回答：

1. 这段代码里出现了哪几种「存储作用域」？请把它们按昇腾的层级从近到远排序。
2. 为什么 `C_l0c` 与 `C_ub` 之间要用 `T.dual_copy`，而 `A`、`B` 用 `T.copy`？（提示：
   阅读 `_vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py:53-100` 的 docstring。）
3. `with T.SimtVF(threads=128):` 与 `T.Parallel` 一起出现，这里的 `T.Parallel` 是
   在描述什么？它和 CUDA 的 `threadIdx` 是同一回事吗？
4. 整段代码里**没有出现任何显式的同步语句**。请对照
   `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:144-152` 的表格，说明这些同步
   被谁接走了。
5. `T.gemm(...)` 的 `clear_accum=(k == 0)` 是什么意思？如果去掉它会怎样？

---

## E8 代码阅读题 ★★ 不需要硬件

读 `_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:431-455` 中 `T.Kernel` 的 docstring，
回答：

1. docstring 把一次 `T.Kernel` launch 描述成什么？请引用原文。
2. docstring 说 `T.Kernel` 是「target-neutral launch: the part every backend shares」。
   结合 `tilelang/ascend/language/frame.py:107` 的 `T.SimtVF`，说说这句话意味着什么：
   同一个 tile 前端如何落到两种完全不同的执行模型上？
3. 为什么把「网格」这个概念与「tile 程序」绑在一起（grid of tile programs），是一种
   与 CUDA 的「grid of threads」不同的抽象？

---

## E9 动手题 ★★★ 不需要硬件（可在 CPU 上完成）

本章 `code/01-op-cost-model/op_cost_model.py` 给出了一个算术强度估算脚本。请：

1. 运行它，记录默认参数下 `MatMul(4096³, fp16)` 与 `Softmax(4096×4096, fp16)` 的算术
   强度与结论。
2. 把矩阵乘的规模改成 `M=N=K=512`，重算并解释结论是否改变。
3. 把机器参数改成「峰值算力 100 TFLOPS，带宽 1.5 TB/s」，重新判断上面两个算子的类别，
   并说明在什么条件下 `MatMul(4096³)` 会从 compute-bound 变成接近平衡点。
4. 在脚本里再加一个 `RMSNorm`（按行归一化）的估算，并说明它的算术强度为什么更低。

交付：修改后的脚本 + 一小段结论（不超过 200 字）。

---

## E10 动手题 ★★★ 不需要硬件

下面是 PyTorch 风格的四遍 softmax（示意，不是可运行代码）：

```text
m = max(x, dim=-1)          # 第 1 遍：读 x，写 m
e = exp(x - m)              # 第 2 遍：读 x、m，写 e
s = sum(e, dim=-1)          # 第 3 遍：读 e，写 s
y = e / s                   # 第 4 遍：读 e、s，写 y
```

请：

1. 用「tile 视角」把它改写成一段伪代码：显式写出每个中间量放在哪一级存储、有哪几次
   全局访存。目标是把全局访存降到接近理论下界（读 `x` 一遍、写 `y` 一遍）。
2. 说明你的改写为什么能成立（提示：softmax 是逐行的，每行可以独立处理；在线 softmax
   的思想属于第 15 章，本章只需给出「行内 offline」的版本）。
3. 指出你的方案里哪一步需要多缓冲或流水，哪一步需要同步。

要求：用伪代码，不要求 TileLang 语法；但必须显式标出每一块数据的存储位置。

---

## E11 分析题 ★★ 不需要硬件

给定 $C = A @ B$，$M = N = K = 4096$，fp16 输入、fp32 累加。假设你只能选择一对
`block_M × block_K` 与 `block_K × block_N` 的片上 tile。

1. 写出一次「朴素无分块」实现在最坏情况下对 `A` 与 `B` 的总访存量（以字节为单位，
   只算最外层的重复读取，忽略 cache）。
2. 写出使用 `128 × 32` 的 tile 之后的理论最小访存量。
3. 分别算出两种情况下的算术强度，并解释为什么分块能把一个「带宽不够」的问题变成
   「算力够不够」的问题。
4. 指出分块的物理代价：你为了多做几次浮点运算，付出的代价是什么？

---

## E12 概念题 ★★ 不需要硬件

本章 4.2 节说「需要被同时正确管理的东西变多了」。请：

1. 列出至少 5 个在 v0.1.15 的昇腾后端里由编译器负责、而在 Ascend C 里要由程序员手工
   维护的事项（可参考 `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:144-152`）。
2. 解释为什么「把同步交给编译器」比「把 layout 交给编译器」更难，需要额外引入什么机制
   （提示：社区文章提到了 SMT 求解器 Z3 与「issue interval 和 latency 必须分开」）。
3. 从工程角度说明：为什么 TileLang@Ascend 团队把大量工程量花在**验证**上，而不是功能上。

---

## E13（可选，需要硬件）★★★

如果你有 NVIDIA GPU（sm90 及以上更佳）：

1. 安装锁定版本的 TileLang（安装步骤属于第 7 章，这里只需按官方
   `docs/get_started/Installation.md` 操作）。
2. 运行 `_vendor/tilelang-v0.1.15/examples/gemm/example_gemm.py`，打印
   `kernel.get_kernel_source()`，找出生成的 CUDA 里哪一段对应你写的 `T.copy`、哪一段
   对应 `T.gemm`。
3. 只在**同一台机器、同一 shape、同一 dtype、同一口径**下对比：把 `num_stages` 从 3 改
   成 1 和 4，记录每次 `do_bench` 的结果。写出你的测量命令与观察结论。

说明：本教程的写作环境没有 GPU，没有跑过这项实验；请以你自己机器的实测为准，并在结论里
写明测试条件。

---

## E14（可选，需要硬件）★★★

如果你有昇腾 NPU（Ascend 950，CANN 环境）：

1. 按 `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md` 的 Quick Start 跑通
   GEMM + ReLU 示例（shape 为 `M=256, N=7168, K=2048`）。
2. 用 `kernel.get_kernel_source()` 导出生成的 PTO / Ascend C 源码，找出 `asc_mmad`
   （矩阵乘）与 MTE 搬运相关的调用。
3. 把 `T.Pipelined(..., num_stages=2)` 改成 `num_stages=3`，记录是否仍然能通过编译
   （提示：片上容量是硬约束）；如果编译失败，把报错信息抄下来并解释原因。

说明：同样地，本教程没有 NPU，未运行过；请以你自己的环境为准。
