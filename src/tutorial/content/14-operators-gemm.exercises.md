# 第 14 章 习题

说明：

* 难度标注为 `★`（基础）/ `★★`（需要综合本章内容）/ `★★★`（需要动手或推理链条较长）。
* “需要硬件”一栏中，`否` 表示只需要一台能跑 Python 的机器；`是（GPU）` /
  `是（NPU）` 表示必须用真实加速器。凡标注“可选”的硬件实验，本机环境
  （macOS、无 GPU/NPU）做不了，但题目本身可以用纸笔或纯 Python 完成。
* 所有题目只依赖第 14 章及更早的知识；涉及后续章节的地方会显式提示。

---

## E1 概念题 ★ 不需要硬件

1. 写出 GEMM $C_{M\times N}=A_{M\times K}B_{K\times N}$ 的计算量、最小 DRAM
   访存量与算术强度公式。
2. 用 H100 SXM5 的公开规格（fp16 张量核 1000 TFLOPS、HBM3 带宽 over
   3 TB/s）估算它的机器平衡点，并写出计算过程。
3. 用第 2 问的结果，求出 **fp16 正方形 GEMM** 大约在什么 $M$ 上从
   memory-bound 翻转到 compute-bound（保留推导过程）。

---

## E2 概念题 ★ 不需要硬件

对 $block\_M=128$、$block\_N=256$、$block\_K=32$、fp16 输入、fp32 累加、
`threads=128`、`num_stages=3` 的一份配置，回答：

1. 单级流水的 shared memory 占用是多少 KB？三级呢？
2. 每个线程的累加器要占多少个 32 位寄存器？
3. 如果 H100 每个 SM 的 shared memory 可配到 228 KB，这份配置最多能同时
   驻留几个 block（只看 shared memory）？
4. 把 `block_M` 从 128 提到 256，第 1、2 问的答案怎么变？代价是什么？

（可用 `code/01-gemm-ledger/gemm_ledger.py` 复算。）

---

## E3 概念题 ★★ 不需要硬件

本章把 GEMM 的实现分成五级阶梯。回答：

1. 第 ② 级（`T.gemm` + shared memory）相对第 ① 级（朴素分块）多解决了
   什么？用“访存量随 $m_1, n_1$ 的公式”说明。
2. 第 ③ 级（`T.Pipelined`）为什么不能提高第 ② 级算出的 DRAM 流量下界，
   却能提高性能？请从“搬运延迟”和“张量核利用率”两个角度解释。
3. 第 ④ 级的两条注解 `T.annotate_layout` 与 `T.use_swizzle` 作用在**不同**
   的层次上。分别说明它们改什么、服务哪个指标、以及为什么都不改变 FLOPs。

---

## E4 概念题 ★★ 不需要硬件

本章 4 节给出四个变体：split-K、stream-K、grouped GEMM、batched GEMM。

1. 用一个句子写出每个变体的“什么时候用它”判据。
2. 对下面四个场景各选一个变体，并说明理由：
   * (a) MoE 的 64 个专家，每个专家收到的 token 数不同；
   * (b) 单个 GEMM，$M=256$、$N=256$、$K=16384$；
   * (c) attention 的 32 个 head，每个 head 的 $QK^\top$ 形状相同；
   * (d) 一个 $8192\times8192\times8192$ 的 GEMM，输出块数远大于 SM 数，
     但 profiler 显示最后一波只有三分之一的 SM 在跑。
3. split-K 用“带宽换并行度”，请写出这句话对应的两个量：
   哪个量变差了、哪个量变好了。

---

## E5 代码阅读题 ★★ 不需要硬件

下面这段来自 `_vendor/tilelang-v0.1.15/examples/gemm_splitk/example_tilelang_gemm_splitk.py`
（已省略 import）：

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), split_k, threads=128) as (bx, by, bz):
    A_shared = T.alloc_shared((block_M, block_K), dtype)
    B_shared = T.alloc_shared((block_K, block_N), dtype)
    C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

    T.clear(C_local)
    for ko in T.Pipelined(T.ceildiv(splitK, block_K), num_stages=0):
        T.copy(A[by * block_M, bz * splitK + ko * block_K], A_shared)
        T.copy(B[bz * splitK + ko * block_K, bx * block_N], B_shared)
        T.gemm(A_shared, B_shared, C_local)

    for i, j in T.Parallel(block_M, block_N):
        T.atomic_add(C[by * block_M + i, bx * block_N + j], C_local[i, j])
```

回答：

1. `T.Kernel` 有三个维度 `(bx, by, bz)`。第三维 `bz` 在这里代表什么？
2. `T.Pipelined(..., num_stages=0)` 是什么意思？参考 `T.Pipelined` 的
   docstring 回答。
3. 为什么最后一步用 `T.atomic_add` 而不是 `T.copy`？如果换成 `T.copy`，
   `split_k=4` 时会得到什么结果（用一句话描述）？
4. 这段代码把 `C` 声明成 `out_dtype=T.float32`。为什么不用 fp16 累加？

---

## E6 代码阅读题 ★★ 不需要硬件

下面是 `examples/grouped_gemm/example_grouped_gemm_fwd.py` 的核心片段：

```python
total_m_blocks = sum((size + block_M - 1) // block_M for size in batch_sizes_list)
...
with T.Kernel(total_m_blocks, T.ceildiv(N, block_N), threads=threads) as (bx, by):
    m_start_padded = bx * block_M
    cur_batch_idx = 0
    for i in range(batch_count):
        in_cur_batch_idx = m_start_padded >= batch_padded_offsets[i]
        cur_batch_idx = T.if_then_else(in_cur_batch_idx, i, cur_batch_idx)
    cur_batch_size = batch_sizes[cur_batch_idx]
    m_start = m_start_padded - batch_padded_offsets[cur_batch_idx] + batch_offsets[cur_batch_idx]
    actual_rows = T.max(0, T.min(block_M, cur_batch_size + batch_padded_offsets[cur_batch_idx] - m_start_padded))
```

回答：

1. 为什么 grid 的第一维是 `total_m_blocks` 而不是 `ceildiv(batch_sum, block_M)`？
2. `batch_offsets` 与 `batch_padded_offsets` 的区别是什么？为什么要同时存在？
3. `actual_rows` 用来做什么？
4. 同目录的 `example_grouped_gemm_fwd_ptr.py` 用指针表代替拼接，但把
   `copy_num_stages` 固定成 1。文件头注释给了什么理由？

---

## E7 代码阅读题 ★ 不需要硬件

读下面这段（来自 `examples/gemm/example_gemm_intrinsics.py`，已省略上下文）：

```python
mma_emitter = TensorCoreIntrinEmitter(
    a_dtype=in_dtype, b_dtype=in_dtype, accum_dtype=accum_dtype,
    a_transposed=False, b_transposed=True,
    block_row_warps=2, block_col_warps=2,
    warp_row_tiles=64, warp_col_tiles=64, chunk=chunk,
)
...
for ki in T.serial(0, (block_K // micro_size_k)):
    mma_emitter.ldmatrix_a(A_local, A_shared, ki)
    mma_emitter.ldmatrix_b(B_local, B_shared, ki)
    mma_emitter.mma(A_local, B_local, C_local)
```

回答：

1. 这段代码属于本章阶梯的第几级？它的“PyTorch 参考”是什么？
2. `block_row_warps × block_col_warps` 决定什么？在这段里 `threads`
   应该是多少？
3. `ldmatrix_a` / `ldmatrix_b` / `mma` / `stmatrix` 四个调用分别对应
   数据在哪两级存储之间流动？
4. 本章说“绝大多数生产算子停在 ③–④ 级”。给出一个理由，说明为什么
   第 ⑤ 级的可维护性更差。

---

## E8 概念题 ★ 不需要硬件

本章 5 节把低精度 GEMM 分成 fp8 / int4 / dequantize / block-scaled 四类。

1. 填表：每一类的输入 dtype、累加 dtype、额外输入、结构上多出的东西。
2. `T.gemm_blockscaled(A, B, C, SFA, SFB, k_start=..., sf_a_granularity_k=...,
   sf_b_granularity_k=...)` 里，`sf_a_granularity_k` 的物理含义是什么？
3. 为什么 block-scaled 在 sm100 上用 `sf_granularity_k=128`，而昇腾侧把
   `sf_*_granularity_k` 固定成 32？这说明“同一个概念”在哪一层上依赖硬件？
4. 本章把低精度的数值正确性交给第 27 章。列出至少两个“属于第 27 章、
   不属于本章”的问题。

---

## E9 概念题 ★★ 需要硬件（可选用模型替代）

本章 6.3 节列出“对齐手写基线”必须固定的五件事。回答：

1. 逐条写出这五件事。
2. CATLASS README 里说“在定制 shape 下的性能能达到相应算子标杆性能的
   0.98~1.2 倍”。这句话给出了哪三个要素？为什么它是“对齐”的模板？
3. 主干 v0.1.15 的昇腾 README 用的对比对象是什么？它报的两个单位分别
   对应哪类算子？
4. 如果你拿 TileLang 的 `example_gemm.py` 和 cuBLAS 比，只报了两个
   TFLOP/s，这个对比可能在哪些口径上不公平？至少说出三条。

---

## E10 动手题 ★★ 不需要硬件

运行 `code/01-gemm-ledger/gemm_ledger.py`：

```bash
cd chapters/14-operators-gemm/code/01-gemm-ledger
python3 gemm_ledger.py --m 4096 --n 4096 --k 4096
python3 gemm_ledger.py --m 256 --n 256 --k 16384
python3 gemm_ledger.py --m 128 --n 128 --k 128
```

回答：

1. 三组 shape 的判定分别是什么？哪一组是 memory-bound？
2. 第二组（$256\times256\times16384$）为什么是 memory-bound？它的 K 很大，
   为什么没让它 compute-bound？
3. 结合 4.1 节，说明为什么第 2 组适合 split-K：它的输出块数够不够？
   瓶颈在哪个方向？

---

## E11 动手题 ★★ 不需要硬件

仿照 `code/01-gemm-ledger/gemm_ledger.py`，给“片上占用估算”加一个函数
`tile_budget(block_m, block_n, block_k, stages, dtype_bytes, threads)`，
返回：

* 单级与多级 shared memory 字节数；
* 每线程累加器寄存器数；
* 在给定的 `smem_limit`（默认 228 KB）下能驻留的 block 数。

然后用它回答：

1. `(128, 128, 32, 3)`、`(128, 256, 32, 3)`、`(256, 128, 64, 2)` 三组配置
   各自的占用。
2. 哪一组最可能因为寄存器压力掉 occupancy？说明你的判断依据。
3. 在不超过 228 KB shared memory 的前提下，把 `num_stages` 从 3 提到 4
   还可行吗？对哪一组不行？

---

## E12 动手题 ★★★ 需要 GPU（可选）

在任意一台有 NVIDIA GPU 的机器上，运行
`code/02-tilelang-ladder/gemm_ladder.py`（它按 `--rung` 选择阶梯级别）。

1. 把 `RUNG_1`（shared，不流水）、`RUNG_2`（`num_stages=3`）、
   `RUNG_3`（加 swizzle/layout 注解）各跑一遍，记录 latency。
2. 用第 7.2 节的协议报数：`warmup`、`rep`、`backend`、`return_mode` 各是
   什么？为什么必须写清？
3. 对 `RUNG_2` 把 `num_stages` 从 1 扫到 5，画出“阶段数–延迟”曲线，
   指出拐点，并解释为什么继续加阶段不再变快。
4. 与 `torch.matmul`（cuBLAS）比，要求：同样的 dtype、同样的 shape、
   同样的 flap 口径（`2MNK`）。写下你实际得到的比值，并标注口径。

> 如果你没有 GPU，本题可只做第 2、4 问的“口径设计”部分，并把结果写成
> “未运行”。

---

## E13 动手题 ★★★ 需要 NPU（可选）

读 `_vendor/tilelang-ascend-v0.1.2.000/examples/batch_gemm/batch_gemm.py` 和
`_vendor/tilelang-ascend-v0.1.2.000/examples/gemm/example_gemm.py`：

1. 找出两段代码里与 NV 版本一一对应的四个要素：grid 维度、片上缓冲分配、
   tile 级矩阵乘调用、同步。
2. `batch_gemm.py` 的 grid 是一维的 `B * m_num * n_num`，在 kernel 内用
   `bid = cid // (m_num * n_num)` 反解 batch 号。这样做和 NV 侧直接用
   三维 `T.Kernel` 相比，各有什么取舍？
3. 设计一个“对齐手写基线”的实验方案：要用什么基线（CATLASS 的哪个
   example）、什么 shape、什么 dtype、什么测量协议？写成一份不超过
   15 行的清单。**不要运行它**，本题只考方案设计。
