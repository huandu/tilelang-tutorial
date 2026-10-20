# 第 14 章 习题答案

与 `exercises.md` 一一对应（E1–E13）。每题给“解题思路 → 参考答案 →
常见错误”。涉及数字的地方标出它是“本书推导”还是“厂商/上游资料数字”。

---

## E1 概念题 ★

**思路**：直接套第 1.1 节的三个公式，注意 $M=N=K$ 时化简。

**参考答案**：

1. 计算量 $=2MNK$ FLOP；最小 DRAM 访存 $=b(MK+KN+MN)$ 字节（$b$ 为每元素
   字节数）；算术强度 $\mathrm{AI}=\dfrac{2MNK}{b(MK+KN+MN)}$，
   正方形时 $\mathrm{AI}=\dfrac{2M}{3b}$。
2. 平衡点 $=1000\ \mathrm{TFLOPS} / 3\ \mathrm{TB/s} \approx 333\ \mathrm{FLOP/Byte}$
   （本书推导：用 H100 SXM5 的两个厂商公开规格相除）。
3. 令 $\dfrac{2M}{3b}=333$，$b=2$ → $M = 333\times3\times2/2 \approx 1000$。
   所以 fp16 正方形 GEMM 在 $M\approx10^3$ 附近翻转。低于它的规模做 tile
   收益有限，该去减少访存。

**常见错误**：把 $2MNK$ 写成 $MNK$（漏掉乘加的 2）；把访存写成只算 $A$、$B$
不算 $C$。注意这里的 $b$ 是“元素字节数”，不同 dtype 结果不同。

---

## E2 概念题 ★

**思路**：套 1.3 节的两个公式。$block\_M=128, block\_N=256, block\_K=32$，
$b=2$（fp16），$S=3$，`threads=128`。

**参考答案**：

1. 单级 $=(128\times32+32\times256)\times2$ 字节
   $=(4096+8192)\times2=24576$ 字节 $\approx 24$ KB；三级 $=72$ KB。
2. 累加器元素数 $=128\times256=32768$，fp32；`threads=128` →
   $32768/128=256$ 个 fp32/线程 = 256 个 32 位寄存器。
3. $228/72\approx3.17$ → 最多 3 个 block（只看 shared memory）。
4. `block_M` 提到 256：单级变成 $(8192+8192)\times2=32$ KB，三级 96 KB；
   累加器变成 $256\times256/128=512$ 个寄存器/线程。
   代价：寄存器压力翻倍（512 远超 CUDA 常规上限 255），几乎必然会溢出到
   local memory；同时 shared memory 也翻倍，occupancy 下降。

**常见错误**：忘记乘流水级数 $S$；把寄存器数按 bit 而不是 32 位寄存器折算。

---

## E3 概念题 ★★

**思路**：把每一级放回“账本”的三个维度（DRAM 流量、重叠、片上带宽）。

**参考答案**：

1. 第 ① 级每次乘加都直接读 global，访存量约 $O(MNK)$（每个输出元素读
   $2K$ 个元素）。第 ② 级把 tile 搬进 shared，每个输出块的搬运量变成
   $b(m_1k_1+k_1n_1)$/ 迭代，总读取量降到 $b\,MNK(1/m_1+1/n_1)$，比下界
   只差一个 $(1/m_1+1/n_1)$ 的倍率。
2. 流水不改变搬运的**总量**，所以不改 DRAM 流量下界；它改变的是**时间分布**：
   第 ② 级里搬运（内存延迟）与计算（张量核吞吐）串行，张量核在等数据；
   流水让第 $i$ 次计算与第 $i+1$ 次搬运重叠，把搬运延迟藏进计算时间里，
   于是张量核利用率上升。
3. `T.annotate_layout` 改的是 buffer 的**物理布局**（数据在 shared memory 里
   怎么摆），服务“shared→fragment 的片上带宽、bank conflict”；
   `T.use_swizzle` 改的是 **block 的遍历顺序**（哪些 block 先跑），服务
   “L2 命中率”。两者都只影响数据怎么流动，不改变乘加次数，所以不改变 FLOPs。

**常见错误**：把两个注解都说成“优化内存”而不区分对象；把 `T.use_swizzle`
误认为改 shared 内部排布（那是 `T.annotate_layout`）。

---

## E4 概念题 ★★

**参考答案**：

1. 判据：
   * split-K：输出块数小于核数，且 $K$ 足够大可以切；
   * stream-K：输出块数够多但不是核数的整数倍、最后一波明显不满；
   * grouped GEMM：多个形状不同（或动态）的 GEMM，典型是 MoE；
   * batched GEMM：多个形状一致的 GEMM（或可 padding 成一致）。
2. 场景匹配：
   * (a) → grouped GEMM（每个专家 M 不同）；
   * (b) → split-K（$256/128\times256/128=4$ 个输出块，远小于 SM 数，K=16384
     很大）；
   * (c) → batched GEMM（形状一致）；
   * (d) → stream-K（块数够多，问题是尾波量化）。
3. “用带宽换并行度”：变差的是**写出流量**（变成原来的 `split_k` 倍，
   $b\,MNK/k$）以及原子合并的开销；变好的是**并行度/负载均衡**（grid 变大，
   核不再闲着）。

**常见错误**：把 (d) 也选成 split-K。注意 split-K 解决的是“块太少”，
(d) 的问题是“最后一波不满”，不同。

---

## E5 代码阅读题 ★★

**参考答案**：

1. `bz` 是 **K 方向的段号**：把 $K$ 切成 `split_k` 段，每个 block 只算
   `splitK = K // split_k` 这一段的部分和。
2. `num_stages=0` 表示**不做软件流水**，`T.Pipelined` docstring 明确写
   “if `num_stages` is 0, pipeline will not be enabled”。这里写
   `T.Pipelined` 只是为了保留循环语法，实际是串行迭代。
3. 因为 `split_k` 个 block 都要往同一个输出位置累加，必须原子相加；
   用 `T.copy` 会变成“最后一个写的 block 覆盖前面的”，结果只等于
   某一段的部分和（$K$ 的某一 1/4），而不是完整结果。
4. 原子加的加法顺序不确定，用 fp16 累加会因舍入误差放大而丢精度；
   官方示例把输出声明为 fp32 就是为了让合并过程在高精度下进行。

**常见错误**：认为 `num_stages=0` 是“默认最优值”；把 `T.atomic_add`
当成可有可无的写法。

---

## E6 代码阅读题 ★★

**参考答案**：

1. 因为各专家 M 不同、且要对齐到 `block_M` 的整数倍（padding），
   grid 必须按“每个专家各自的 M 块数”累加得到 `total_m_blocks`，
   而不是按总 M 除以 block_M。
2. `batch_offsets[g]` 是专家 $g$ 在**拼接张量**里的真实起始行；
   `batch_padded_offsets[g]` 是它在**对齐到 block_M 后**的起始偏移。
   同时存在是因为：grid 按 padding 后的边界划分，而实际读写数据要回到
   真实偏移，二者差一个 padding 量。
3. `actual_rows` 是当前 block 覆盖的**有效行数**（去掉 padding），用于最后
   写回时只写有效行、不越界也不写脏 padding 区域。
4. 注释说 “Multi-stage software pipelining on ptr-backed tensors is not
   correct yet”，所以把 `copy_num_stages` 固定为 1。

**常见错误**：把 `batch_offsets` 和 `batch_padded_offsets` 当成同一个量；
忽略“ptr-backed 多级流水尚不正确”这条限制。

---

## E7 代码阅读题 ★

**参考答案**：

1. 第 ⑤ 级（官方 intrinsic / 手动 MMA）。它没有 PyTorch 参考，对照物是
   CUTLASS 的 warp-level MMA 代码。
2. 决定 warp 在 M/N 方向怎么划分（分区策略）。`threads`
   $=32\times(block\_row\_warps\times block\_col\_warps)=32\times4=128$。
3. `ldmatrix_a/b`：shared → 寄存器 fragment；
   `mma`：寄存器 → 寄存器（累加器）；
   `stmatrix`：寄存器 → shared。
4. 第 ⑤ 级要自己管 warp 划分、微形状、fragment 大小，对每种 dtype/形状
   组合都要重新核对；一旦换了架构或 dtype 就可能失效，可移植性与可维护性
   都比 TileLang 的高层原语差很多。

**常见错误**：把 `threads` 算成 `block_row_warps × block_col_warps`（忘了
乘 32）；把 `stmatrix` 说成“写回 global”。

---

## E8 概念题 ★

**参考答案**：

1. 表：

   | 变体 | 输入 dtype | 累加 dtype | 额外输入 | 结构上多出的东西 |
   | --- | --- | --- | --- | --- |
   | fp8 | e4m3 / e5m2 | fp32 | 无 | 更严的 K 对齐 |
   | int4 | int4（打包） | int32 | 无 | 打包 / 解包 |
   | dequantize | int4/int8/fp4 | fp32 | scale（+可选 zero，粒度可变） | 搬入时反量化 |
   | block-scaled | fp8 / fp4 | fp32 | SFA / SFB | scale 按 K 块施加 |

2. `sf_a_granularity_k` = **一个 A scale factor 覆盖多少个 K 元素**。
3. 因为 scale 的块大小是**硬件指令定义的**：sm100 的 MXFP8 用 128，
   MX 规范在昇腾侧是 32。这说明“block-scaled”这个概念虽然统一，但
   **粒度是硬件相关的**，前端只能把粒度作为参数暴露。
4. 例：量化误差如何传播、累加顺序对结果的影响、scale 的舍入/上溢策略、
   如何做数值验证（rtol/atol 怎么定），都属于第 27 章。

**常见错误**：把 `sf_*_granularity_k` 理解成“scale 的个数”而不是
“每个 scale 覆盖的 K 宽度”。

---

## E9 概念题 ★★

**参考答案**：

1. 五件事：基线是谁；同一个 shape；同一个 dtype 与累加精度；同一个 epilogue；
   同一套测量协议。
2. 三个要素：对比对象（标杆性能）、条件（定制 shape）、区间（0.98–1.2×）。
   它是“对齐”的模板，因为它把“跟谁比、在什么条件下、差多少算合格”都写全了，
   而不是只给一个孤立的数字。
3. 对比对象是 **Torch NPU**。compute-bound 的算子报 TFLOP/s（GEMM、GQA
   backward），memory-bound 的报有效 GB/s（FP8 casting）。
4. 可能不公平的口径（举三条即可）：
   * dtype 不一致：TileLang 用 fp16+fp32 累加，cuBLAS 若默认开 TF32（fp32 输入时）
     或用了不同累加精度；
   * epilogue 不一致：一方融合了 bias/激活，另一方没有；
   * 计时口径不一致：一个用 `event`（含间隙/launch），一个用 `cupti`
     （kernel 纯耗时）；
   * 预热/重复次数、是否 flush L2 不一致；
   * shape 是否 padding、是否 batched（cuBLAS 可能走了 strided-batched API）。

**常见错误**：只写“dtype 可能不同”而不写清哪一侧开了 TF32；把
`do_bench` 默认的 `mean` 和 `min` 混用。

---

## E10 动手题 ★★

**参考答案**（以下判定来自 `code/01` 的模型输出，可复算）：

1. $4096^3$ → compute-bound；$256\times256\times16384$ → memory-bound；
   $128^3$ → memory-bound。memory-bound 的是后两组。
2. $256\times256\times16384$ 的 $M$、$N$ 都很小，虽然 $K$ 大，但算术强度
   由 $M$、$N$ 主导（访存量里 $MK$、$KN$ 占大头），所以仍然 memory-bound。
   直觉：$K$ 再大，也只是把同一批 $A$、$B$ 元素重复用很多次，而输出只有
   $256\times256$，摊不开。
3. 输出块数 $=\lceil256/128\rceil\times\lceil256/128\rceil=4$，远小于 SM 数
   （H100 有大量 SM），grid 太小；同时 K 很大可以切且不影响读入量，
   所以适合 split-K。瓶颈在“并行度”，不在带宽本身。

**常见错误**：只看 K 大就以为 compute-bound。算术强度看的是“总 FLOPs/总
字节”，小 M/N 会把 AI 拉低。

---

## E11 动手题 ★★

**思路**：实现 `tile_budget`，注意两点：单级字节数 $=b(block\_M\cdot block\_K + block\_K\cdot block\_N)$，寄存器数 $=block\_M\cdot block\_N/\text{threads}$
（fp32 累加时，每个元素一个 32 位寄存器）。

**参考答案**：

1. （$b=2$、fp32 累加、`threads=128`）

   | 配置 | 单级 | 多级 | 每线程累加寄存器 |
   | --- | --- | --- | --- |
   | (128,128,32,3) | 16 KB | 48 KB | 128 |
   | (128,256,32,3) | 24 KB | 72 KB | 256 |
   | (256,128,64,2) | 48 KB | 96 KB | 256 |

2. 后两组最可能掉 occupancy：累加器就占 256 个寄存器，已经超过 CUDA 常规
   的上限（255），必然溢出到 local memory。判断依据是 1.3 节的寄存器公式
   与 CUDA 最佳实践里“cc7.0 每线程 32 个寄存器才能满 occupancy”的结论。
3. 第三组 (256,128,64,2) 单级就 48 KB，加到 3 级是 144 KB（仍 < 228 KB），
   第一组 (128,128,32,3) 加到 4 级是 64 KB（可以）。真正的限制通常先卡在
   寄存器而不是 shared memory。所以答案要以“哪个约束先到”为准。

**常见错误**：只算 shared memory 不算寄存器；忘记 `accum_dtype` 的
`/4 Byte` 折算。

---

## E12 动手题 ★★★（GPU，可选）

**思路**：本题的关键不是跑出某个数字，而是**把口径写全**。没有 GPU 时，
至少要能答出第 2、4 问。

**参考答案**：

1. （需要真机）应记录三级各自的 latency。本机无 GPU，**未运行**。
2. 应写：`warmup`（如 25 ms）、`rep`（如 100 ms）、`backend`
   （`event` / `cupti` / `cudagraph`）、`return_mode`（默认 `mean`）、
   是否 flush L2。必须写清是因为不同 backend 的计时范围不同（含不含 launch、
   含不含 kernel 间隙），`mean`/`min` 也会改变结论。
3. 预期形状：`num_stages` 从 1 增加到某个值延迟下降，之后平甚至上升
   （片上缓冲占用增大、occupancy 下降）。**具体拐点必须在真机上测，本
   书不给数字。**
4. 与 `torch.matmul` 比时，要写清：同一 dtype、同一 shape、`2MNK` 口径、
   同一 `do_bench` 参数、是否 TF32、对比的是 `cublas` 还是 `cublasLt`。

**常见错误**：直接抄别人的“比 cuBLAS 快 x 倍”而不写条件；用 `min` 和
`mean` 混着比。

---

## E13 动手题 ★★★（NPU，可选）

**参考答案**：

1. 对应要素：

   | 要素 | NV（`examples/gemm/example_gemm.py`） | 昇腾（适配器 `examples/gemm/example_gemm.py`） |
   | --- | --- | --- |
   | grid | `T.Kernel(ceildiv(N, bN), ceildiv(M, bM), ...)` 二维 | `T.Kernel(m_num * n_num, is_npu=True)` 一维，用 `//`、`%` 反解 |
   | 片上缓冲 | `T.alloc_shared` | `T.alloc_L1` / `T.alloc_L0C` |
   | 矩阵乘 | `T.gemm` | `T.gemm_v0` |
   | 同步 | 隐式（`T.Pipelined` 自动插） | 显式 `T.barrier_all()`（主干后端由 `InsertSync` 自动做） |

2. 一维 grid + 除法反解：好处是 grid 维度简单、kernel 内可控，便于做
   persistent 遍历；代价是要在 kernel 里做整数除法，且丢失了“x 维、y 维”
   的语义信息，不利于编译器做 swizzle 优化。NV 侧直接用多维 grid 更贴近
   硬件。
3. 实验方案模板（不运行）：

   ```text
   基线：CATLASS examples/00_basic_matmul（Common 模板 + MmadAtlasA2Pingpong）
   对比：TileLang 昇腾 example_gemm.py，同 shape / dtype / 输出类型
   shape：至少 4 组，覆盖 (256,512,1024) 与 (8192,1024,8192) 等
   dtype：fp16 -> fp32 累加 -> fp16 输出；固定是否含 epilogue（本题：不含）
   测量：同一 do_bench（warmup=25ms, rep=100ms），同一 backend
   报数：TFLOP/s = 2MNK / t；正确性用 torch.testing.assert_close
         （rtol=1e-2, atol=1e-2）对同一 golden
   记录：block_M/N/K_L1、num_stages、swizzle 策略、使用的 example 路径
   ```

**常见错误**：用 CATLASS 的某个带后处理的 example 去比 TileLang 的纯 GEMM；
或把“精度比对通过”当成“性能对齐”。
