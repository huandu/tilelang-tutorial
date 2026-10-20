# 第 13 章 习题答案

说明：每题给出**解题思路**、**参考答案**与**常见错误**。账本题请自己在
`code/01-ledger/ledger.py` 上复算；源码行号基于
[`../../VERSIONS.md`](../../VERSIONS.md) 锁定的 v0.1.15 / v0.1.2.000。

---

## E1 概念题 ★

**思路**：不要盯着比值，先看「融合后的下界是什么」。融合后的账本永远等于
「把输入读一遍 + 把输出写一遍」，所以同一组形状下不同算子的融合版字节数会趋同。

**参考答案**

1. 因为两者的融合版都只需要「读 x 一遍 +（读权重/偏置这些 O(N) 的小张量）+
   写 y 一遍」。4096×4096 的 fp16 输入是 33.55 MB，读一遍 + 写一遍就是 67.1 MB；
   `w` 只有 4096 个元素（8 KB），在账本上是可以忽略的加项。
2. 因为 LayerNorm 的**逐算子版本**把更多的中间量落到了显存：

   ```text
   RMSNorm  ：x² 一份中间张量            → 7 个整张量 pass ≈ 235 MB
   LayerNorm：x² 一份 + 逐趟的累加张量   → 10 个整张量 pass ≈ 336 MB
   ```

   分母（融合后）几乎一样，分子（逐算子）更大，所以比值更大。
3. **不变**。账本里所有的项都正比于 `sizeof(dtype)`（每个张量读/写一次，元素数相同），
   比值是无量纲的。

   **唯一的例外**：如果你的逐算子版本把中间张量存成 fp32 而输入输出是 fp16
   （真实实现里很常见，因为累加通常要在 fp32 上做），那么分子会变大、比值会略增。
   这也是为什么记账时要把「中间张量的 dtype」写清楚——不然两本账不可比。

**常见错误**：把「融合后字节数一样」当成「两个算子的耗时一样」。耗时还取决于
带宽能不能打满、occupancy、归约的 barrier 数量，这些都不在账本里。

---

## E2 概念题 ★

**思路**：一个 op 一个 kernel 时，中间张量要被写一次、读一次；融合之后它就不存在了。

**参考答案**

1. 逐算子：

   ```text
   kernel 1：读 x（N 元素）+ 写 t（N 元素）
   kernel 2：读 t（N）+ 读 b（N）+ 写 y（N）
   合计 = 5N 个元素 = 5N × 4 字节 = 20N 字节；flops = 2N（两次基本运算）
   AI = 2N / 20N = 0.100 FLOP/Byte
   ```

2. 融合：

   ```text
   读 x（N）+ 读 b（N）+ 写 y（N）= 3N 个元素 = 12N 字节；flops = 2N
   AI = 2N / 12N = 0.167 FLOP/Byte
   ```

3. **严格说，这个结论依赖于机器**：memory-bound 的定义是 `AI < 平衡点`，
   而平衡点是「峰值算力 ÷ 峰值带宽」，因机器与精度而异。
   但在工程上它非常稳健：0.1–0.17 FLOP/Byte 比任何加速器在同一精度下的平衡点
   低一到两个数量级，所以实践中不需要查表也能判断方向。
   真要下结论时，按第 02 章 §2.2 的方法自己算一遍平衡点。

**常见错误**：忘了把中间张量 `t` 计两次（写一次、读一次），于是算出 3N / 3N，
得出「融合没有收益」的错误结论。

---

## E3 概念题 ★★

**思路**：把「循环怎么切」与「启动配置」分开看。

**参考答案**

1. 循环规划器先根据 dtype 与目标属性算出可用的向量宽度（bf16 下 128 bit = 8 个元素），
   再把 extent 为 256 的最内层循环按「8 个元素一组」切开——256 / 8 = 32 组，
   于是只需要 32 个线程各处理一组。这就是生成代码里出现 `uint4` 与
   `threadIdx.x < 32` 的原因。
2. `threads=256` 是**你在 `T.Kernel(..., threads=256)` 里声明的启动配置**，
   会写进 `__launch_bounds__(256)`。循环切分不会回头改启动维度——多出来的 224 个
   线程只是空转。官方文档把这称为「counterintuitive」：
   「applying vectorization can result in each thread computing 8 data elements —
   effectively utilizing only 32 active threads. Interestingly, the kernel launch
   configuration still retains the original allocation of 256 threads」
   （corpus `elementwise.md:174-178`）。
3. 写法二把「每线程几个元素」变成了你**显式声明**的东西：最内层 extent 就是
   向量宽度，且相邻元素按构造属于同一个线程。于是「用多少线程、每次访多宽」
   从编译器的推断变成了你的输入。

**常见错误**：以为写法一只有 32 个线程在启动（看 launch config 会否掉），
或者以为写法一更慢就一定是因为线程少——实际原因是**访存宽度**，
两种写法的有效访存都是 128-bit。

---

## E4 代码阅读题 ★★

**思路**：把生成代码里的 `if` 与「向量组内部是否恒定」联系起来。

**参考答案**

1. 那个 `if` 的条件依赖 `threadIdx.x`（也就是 lane），在「8 个元素一组」的
   向量组内部并不恒定。`IsExprInvariantInVectorBoundary` 证明不了不变性，
   规划器只能把向量宽度往最小宽度退，最终退到 1——
   见 `_vendor/tilelang-v0.1.15/src/transform/loop_vectorize.cc:1016-1066`
   （`ComputeBufferVectorSize` 里的最小宽度回退）与 `:1248-1281`。
2. 谓词是 `legalize_safe_memory_access` 插入的
   （`_vendor/tilelang-v0.1.15/src/transform/legalize_safe_memory_access.cc:419-477`）。
   它在无法证明访问安全时包一层 `if_then_else(cond, access, safe_value)`。
3. `N = 2048` 时，一个 block 恰好覆盖 2048 个元素 = 256 线程 × 8 元素，
   没有尾块，谓词消失，规划器可以取满 8 的向量宽度，生成代码里就会出现
   `uint4` 的整块读写。官方文档给的正是这一对对照
   （corpus `elementwise.md:153-176` 与 `:194-211`）。
4. 两种可行做法：

   * **主循环 + 尾块分离**：对 `floor(N / block) * block` 个元素跑一个无谓词的
     kernel（或同一 kernel 里的一个无谓词分支），剩下的元素交给一个带谓词的小循环；
   * **手工把边界写进值里**：`legalize_safe_memory_access` 有一条分支会跳过
     "store 的值本身已经是 `IfThenElse`" 的访问（同文件 `:471-477`），
     所以把 `if idx < N:` 改成 `C[idx] = T.if_then_else(idx < N, val, 0)` 之外，
     还要注意不要让它破坏向量化——对纯逐元素算子，`0` 这样的安全值写入越界地址
     仍然可能越界，所以实践上更稳的是第一种做法。

    第 09 章的 `T.annotate_safe_value` 是相关的另一个旋钮：它改的是"谓词为假时读到
    什么值"，不是"要不要插谓词"。

**常见错误**：把 `#pragma unroll` 当成向量化。展开是展开，向量化是向量化——
这段代码被展开了 8 次，但每次都是标量访存。

---

## E5 概念题 ★★

**思路**：先想清楚「归约在物理上是怎么合成的」，再谈 `clear`。

**参考答案**

1. 忽略它。`clear=True` 的语义是"结果覆盖 dst"：实现上要么先给 dst 写初值
   （sum 写 0、max 写 -inf、min 写 +inf），要么另开一个干净的 clear buffer
   （`_vendor/tilelang-v0.1.15/src/op/reduce.cc:179-182`）。
2. 因为跨线程归约是 butterfly：每一级里每个参与线程的部分和都会被组合一次，
   同一个值会被计入多次。如果把 dst 直接当累加器，dst 的旧值就会被重复累加
   若干次（次数与线程数有关）——所以它不是「加一次」，而是「加错了」。
3. max / min（以及 bitand、bitor）的合并是**幂等**的：`max(a, a) = a`，
   合并一次和合并多次结果相同。`reduce.cc` 里那句"duplicate temporary buffer"
   明确写的是 "for non-clearing sum/abssum"。
4. `reduce_macro` 的三步（`_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:103-131`，以
   shared→shared 为例）：

   ```text
   ① 建两个 fragment：red_frag_in / red_frag_out
   ② 因为 clear=False：copy(out, red_frag_out)     ← 保住 dst 的旧值
   ③ copy(buffer, red_frag_in) → 发射 tl.tileop.reduce(clear=False)
      → copy(red_frag_out, out)                   ← 把合并结果写回
   ```

**常见错误**：以为 `clear=False` 是"保留 dst 当累加器"。它的真实语义是
「把 dst 的旧值当作一次额外的输入参与一次合并」。

---

## E6 代码阅读题 ★★

**思路**：逐条对应到「数据流」而不是「语法」。

**参考答案**

1. 不能省。`A_local[i, j] += …` 是累加；`A_local` 是 fragment，生命周期里带着
   上一次使用留下的内容。不 `clear` 的话第一次迭代就加到脏值上，
   结果错得没有规律。官方源码在同一个位置也有一行 `T.clear(A_local)`。
2. 不会被标量化，因为这条 store 的索引 `A_local[i, j]` **随最内层循环变量 j
   以步长 1 推进**，与读 `A_shared[i, j]` 的索引完全同步。规划器里那条
   "store 的索引不随向量轴推进就标量化"的规则
   （`_vendor/tilelang-v0.1.15/src/transform/loop_vectorize.cc:1050-1060`）针对的是索引在向量组内**恒定**
   的 store（典型是沿向量轴做归约），这里不满足那个条件。
3. 倒序是为了利用 cache：第一趟结束时，最后一段 `k` 的数据刚被读过，第二趟
   从最后一段开始能命中 cache，减少一次 DRAM 往返。
   代价是这个写法的 **x 被读了两遍**（每个 pass 一遍），账本上比「整行放
   fragment」多一整个读的量。
4. 整行放 fragment：N 能放进寄存器时用，账本最优（读一遍写一遍）；
   split-K：N 很大（官方示例用 8192）或者片上预算紧张时用，
   用"多读一遍"换"每块只用一个小累加器"。

**常见错误**：把 `T.clear` 当成「清零输出张量 B」——它清的是累加器 `A_local`，
与输出无关。

---

## E7 代码阅读题 ★★

**思路**：看 `reduce_macro` 的四个分支，注意"建 fragment"和"copy"是两件事。

**参考答案**

| 组合 | 额外 fragment | 额外 copy |
| --- | --- | --- |
| ① (shared, shared) | 2 个（in / out） | 至少 2 次（src→frag_in、frag_out→out），`clear=False` 时再加 1 次 |
| ② (shared, fragment) | 1 个（in） | 1 次（src→frag_in） |
| ③ (fragment, shared) | 1 个（out） | 1 次（frag_out→out），`clear=False` 时再加 1 次 |
| ④ (fragment, fragment) | 0 | 0（直接发射 `tl.tileop.reduce`） |

（`_vendor/tilelang-v0.1.15/tilelang/language/reduce_op.py:94-171`。）

"先在 shared 上归约"等于走 ②：前端必须先把 whole tile 从 shared 拷进一个
fragment 才能归约。而"先把数据读进 fragment 再归约"走的是 ④：数据本来就在
fragment 里，归约之后还能**原地**继续参与后面的逐元素计算（本章的
`rms_norm` 就是这种写法），等于省掉一次 shared↔fragment 的来回。

**常见错误**：以为"src 是 shared、dst 是 fragment"是零成本直达。

---

## E8 动手题 ★★

**思路**：先证明正确性，再谈向量化；这两件事在非对齐长度上是冲突的，
所以答案必须把冲突讲清楚。

**参考答案**

朴素可用的版本（正确性优先）：

```python
@tilelang.jit(out_idx=[2])
def scale_add(N, threads=256, npt=8, alpha=0.5, dtype="float32"):
    block = threads * npt

    @T.prim_func
    def main(X: T.Tensor((N,), dtype), B: T.Tensor((N,), dtype), Y: T.Tensor((N,), dtype)):
        with T.Kernel(T.ceildiv(N, block), threads=threads) as bx:
            for i, j in T.Parallel(threads, npt):
                idx = (bx * threads + i) * npt + j
                Y[idx] = X[idx] * alpha + B[idx]

    return main
```

* `N = 1 << 20`：`grid = N / block`，每个 block 恰好对齐，无谓词，
  最内层 8 个元素可以向量化成 128-bit 访存。
* `N = 1000`：`block = 2048`，`grid = 1`，整个 block 都是"尾块"。
  正确性由 `legalize_safe_memory_access` 自动插入的谓词保证；
  代价是这段循环被标量化（E4 的原因）。

想让非对齐情况也快，标准做法是**主循环 / 尾块分离**：

```python
full = (N // block) * block          # 能整块处理的部分
# 主 kernel：只处理 [0, full)，无谓词
with T.Kernel(full // block, threads=threads) as bx:
    for i, j in T.Parallel(threads, npt):
        idx = (bx * threads + i) * npt + j
        Y[idx] = X[idx] * alpha + B[idx]

# 尾 kernel（或同一 kernel 里的第二个分支）：处理 [full, N)，逐元素带谓词
with T.Kernel(1, threads=threads) as _:
    for i in T.Parallel(threads):
        idx = full + i
        if idx < N:
            Y[idx] = X[idx] * alpha + B[idx]
```

**验证**：`torch.testing.assert_close(out, x * alpha + b, rtol=1e-5, atol=1e-5)`；
再打印生成代码，确认主循环里有 `uint4`/`float4`、尾块的谓词只出现在尾 kernel 里。

**常见错误**：以为"编译器总会自动帮我做尾块分离"。官方文档那个 `N=2047` 的例子
就是反例：它对整个循环标量化，而不是只对尾块。

---

## E9 动手题 ★★

**思路**：多一个输入张量，账本上只多 O(N) 的读；实现上多一块片上存储。

**参考答案**

1. 参考实现：

   ```python
   def ref_rms_norm(x, w, eps=1e-6):
       rstd = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
       return (x.float() * rstd * w.float().unsqueeze(0)).to(x.dtype)
   ```

2. kernel 只需在 `rms_norm` 的基础上加一个 shared 权重和一个乘法：

   ```python
   w_sh = T.alloc_shared((N,), dtype)
   T.copy(W, w_sh)
   ...
   for i, j in T.Parallel(blk_m, N):
       x_local[i, j] = x_local[i, j] * sumsq[i] * T.Cast("float32", w_sh[j])
   ```

   注意 `w_sh[j]` 的索引只依赖 `j`，在向量组内部是"对最内层循环变量依赖"的
   仿射表达，所以它可以和 `x_local[i, j]` 一起被向量化；
   如果把它写成一个不随 `j` 变化的标量，就变成了广播，向量化规则不同。

3. 账本（4096×4096、fp16、M=N=4096）：

   ```text
   逐算子 6 步（含乘 w）：235.0 MB
   融合（含乘 w）      ： 67.1 MB（x 读一遍 + y 写一遍）+ w 的 8 KB ≈ 67.13 MB
   比值 ≈ 3.50×
   ```

   也就是权重带来的增量在账本上可以忽略——**它不值得为它单独设计一个 kernel**，
   但必须放进融合 kernel 里。

4. 放 shared：好处是"每块从显存读一次 w，块内所有行复用"，代价是
   `N × sizeof(dtype)` 的 shared 容量（4096 个 fp16 = 8 KB）。
   放 fragment：好处是不占 shared，代价是它的 layout 必须与整行 fragment 在 `j`
   这一维一致（否则要在两个 layout 之间做转换，或者直接 layout 冲突报错）。
   实践上**除非 N 很小、且这个 fragment 本来就要常驻，否则放 shared 更稳**。

**常见错误**：把 `w` 直接当全局张量在循环里读（`W[j]`），结果每个元素都去一次
显存——账本上等价，实际会被 L2 兜住一部分，但访存指令数陡增。

---

## E10 代码阅读题 ★★

**思路**：看"哪两个索引是一对"。

**参考答案**

1. 甲 = half-split（NeoX / Llama / Qwen 风格），把行从中间劈成两半；
   乙 = interleaved（原版 RoFormer），相邻两个元素成一对。
2. 都应当**原样返回 x**（`cos ≡ 1`、`sin ≡ 0` 时旋转角为 0，是恒等变换）。
   这个性质把"配对索引写错"这件事暴露得非常快：任何一个把 `p` 和 `p+H`
   写反、或者错位一个元素实现，都会在这一步就失败。
3. 都是 `(S, R/2)`，FP16 下 R=128 就是 `(S, 64)`：
   **一个 pair 占 cos/sin 表的一行**，与配对约定无关。
4. **甲更容易**。甲按 `p` 循环时，两次 store 的索引分别是 `p` 与 `p + H`，
   步长都是 1，各自可以整段向量化；乙按 pair 循环时，两次 store 的索引是
   `2p` 与 `2p + 1`，步长是 2，单独看都不能向量化——它的正确做法是按
   128-bit 向量读整行，让一对元素落在同一个向量里。

**常见错误**：以为两种约定只是"实现上等价、写法不同"。它们是**语义不同**的
两种旋转，用错了模型会变傻但不报错。

---

## E11 分析题 ★★★

**思路**：把 `blk_m` 理解成"每块的飞行高度"：飞得高摊得薄，但油耗（寄存器）大。

**参考答案**

1. `blk_m` 是一个 block 一次处理的行数。调大它等于把「每行的固定开销」——
   跨线程归约的 barrier、`rsqrt` 调用、循环启动、`w`/`cos`/`sin` 的取数——
   摊到更多行上；块数也相应减少。
2. 三件事：

   * **寄存器**：`x_f32` 这类 fp32 fragment 的大小是 `blk_m × N × 4` 字节，
     直接压低 occupancy；
   * **片上存储**：`x_sh` 是 `blk_m × N` 个 dtype，shared / UB 是固定的；
   * **layout 约束**：归约要求一种 fragment 划分，RoPE 要求 `p` 与 `p+H` 同时可见，
     两者都要在同一个 kernel 里满足，`blk_m` 越大可选的划分越少。
3. 两条路线：① 按 K 切行、分段累加（split-K，见 §3.2.4）；
   ② per-thread partial 归约——每个线程直接以自己的步长走一遍 shared 上的行，
   先得到 per-thread 部分和，再做一次跨线程合并
   （TileOPs 的 RMSNorm 在小 N 且网格能占满设备时走这条分支）。
4. 一份可用的验证清单：

   ```text
   ① python3 -m py_compile                    —— 语法
   ② 只编译不运行：k = rms_norm_rope_fused(m, n, rope_dim)
      —— 最可能在这里暴露 layout 冲突；报错会打印 src_layout / orig_dst_layout
   ③ 打印 k.get_kernel_source()，检查三件事：
      · 平方/归约那一步有没有被向量化
      · RoPE 的两条 store 是不是各自连续
      · fp8 的 cast 有没有被拆成标量
   ④ 小形状（M=8, N=256, rope_dim=32）跑正确性，与 PyTorch 参考对齐
   ⑤ 放大到目标形状，再与账本下界比有效带宽
   ```

**常见错误**：一上来就用目标形状跑，出错了分不清是 layout 冲突、数值错误还是
形状假设不成立。

---

## E12 概念题 ★★

**思路**：看四个特征——`T.Kernel` 的签名、UB 的 API、同步方式、向量计算的 API。

**参考答案**

* **甲 = 适配器 tilelang-ascend v0.1.2.000**。依据：`with T.Kernel(..., is_npu=True) as
  (cid, vid)` 返回两个维度；`T.alloc_ub`；`T.Scope("V")` + `T.barrier_all()`；
  `T.tile.add`。
* **乙 = 主干 v0.1.15 的昇腾方言**。依据：`with T.Kernel(num_blocks) as bx` 只有一维
  核下标、没有 `is_npu=`；`T.alloc_shared`（在昇腾侧就是 UB）；
  `T.Pipelined` + `T.SimtVF(threads=...)` + `T.Parallel`。

把甲改写成乙需要改四处：

```text
① T.Kernel(m_num, is_npu=True) as (cid, vid)  →  T.Kernel(核数) as bx
② T.alloc_ub(...)                             →  T.alloc_shared(...)
③ T.Scope("V") + T.barrier_all()              →  T.SimtVF(threads=...) 包裹（同步由 pass 处理）
④ T.tile.add(c_ub, a_ub, b_ub)                →  T.Parallel 循环，或 T.simd.* 显式向量指令
```

（本章的 `code/07-api-audit/api_audit.py` 可以在两轮检查里复现
「主干没有 `alloc_ub` / `tile` 命名空间」这条判断。）

**常见错误**：以为 `T.alloc_ub` 只是 `T.alloc_shared` 的别名。它们在两套代码里，
各自只存在于自己的那一份 tilelang 里。

---

## E13 设计题 ★★★

**思路**：先写账本，再设计数据流，最后想清楚"融合把哪个中间值吃掉了"。

**参考答案**

1. 参考实现：

   ```python
   def ref_fused_add_rms_norm(x, residual, w, eps=1e-6):
       s = (x + residual).to(x.dtype)
       rstd = torch.rsqrt(s.float().pow(2).mean(-1, keepdim=True) + eps)
       return (s.float() * rstd * w.float().unsqueeze(0)).to(x.dtype), s
   ```

   注意参考实现**把 `s = x + residual` 也返回了**——这正是第 4 问的伏笔。

2. 账本（M 行、N 列、fp16）：

   ```text
   两个算子分别写：
     add      ：读 x(N) + 读 residual(N) + 写 s(N) = 3M·N 个元素
     rms_norm ：读 s(N) + 写 y(N)                 = 2M·N 个元素
     合计 5M·N 个元素 = 160.5 MB（M=N=4096, fp16）
   融合写：
     读 x + 读 residual + 写 y = 3M·N 个元素 = 100.7 MB（+ w 的 8 KB）
   比值 ≈ 5 / 3 ≈ 1.67×
   ```

3. 数据流草图：

   ```text
   T.copy(X block)      → fragment(x_f32)      ← 也可以是 shared 中转
   T.copy(Residual block)→ fragment(r_f32)
   s_f32[i, j] = x_f32[i, j] + r_f32[i, j]      （fp32 相加，避免二次舍入）
   sq[i, j]    = s_f32[i, j] * s_f32[i, j]
   T.reduce_sum(sq, sumsq, dim=1)
   rstd[i]     = T.rsqrt(sumsq[i] / N + eps)
   y[i, j]     = T.Cast(dtype, s_f32[i, j] * rstd[i] * T.Cast("float32", w_sh[j]))
   T.copy(y_frag, Y block)
   ```

   `s_f32` 全程留在 fragment 里：既参与归约，又参与最终输出，一次都不用落显存。

4. **副作用**：`s = x + residual` 这个中间张量**不再存在于显存里**。
   在很多模型里它并不"中间"——它是残差流（residual stream），
   同一个 block 的下游算子（或者下一个 block）还要用它。
   一旦融合，你只有三个选择：

   * 让这个融合 kernel **多写一个输出** `s`（账本上加一整趟写，1.67× 的收益会掉到
     5/4 = 1.25×）；
   * 下游**重新算**一遍 `x + residual`（把省下的访存又花回去）；
   * 重新组织下游算子，让它直接吃 `y` 而不需要 `s`。

   这就是"融合的收益与代价"最真实的样子：账本只回答"能省多少"，
   **省下来的前提是"那个中间量真的没人要了"**。

5. 验证：

   * 正确性：与参考实现比 `rtol=atol=1e-2`（fp16）；再用 fp64 参考在小形状上
     确认误差在 dtype 的 eps 量级内，排除"约定/索引写错"。
   * 性能：有效带宽 = `(3M·N·sizeof(dtype) + N·sizeof(dtype)) / latency`，
     与该卡 peak bandwidth 比；同时和"两个算子分别写"的实测对照，
     看是否真的拿到了接近 1.67× 的收益。**没有实测就不能下结论**——
     本章所有数字都只是账本下界。

**常见错误**：只算"融合能省多少字节"，不问"被省掉的那个中间量还有没有别的消费者"。
一个被别处需要的中间张量，融合掉它等于把成本转移到了别的地方。
