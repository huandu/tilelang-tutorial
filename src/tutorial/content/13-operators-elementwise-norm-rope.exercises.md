# 第 13 章 习题

说明：

* 难度：`★`（基础，只需读懂本章）/ `★★`（需要综合本章多节）/ `★★★`（需要动手、
  读源码或设计一个方案）。
* 「需要硬件」一栏里，`否` 表示只需要一台能跑 Python 的机器。本章**没有**强制要求
  硬件的题目；所有必须在加速器上完成的实验都标为「可选」，并在题面里写明。
* 账本题一律使用第 02 章的记账口径（输入读一次、输出写一次、中间张量各计一次、
  基本运算与超越函数各记 1 FLOP）。
* 所有题目只依赖第 1–13 章的知识；涉及第 14 章及以后的内容时会显式提示。

---

## E1 概念题 ★ 不需要硬件

`code/01-ledger/ledger.py` 在 4096×4096 下给出两个数字：

```text
RMSNorm  逐算子 6 步：234.99 MB   →  融合：67.12 MB   （3.50×）
LayerNorm 多趟分解 ：335.59 MB   →  融合：67.13 MB   （5.00×）
```

请回答：

1. 为什么两者的**融合后**字节数几乎一样（67.12 与 67.13 MB）？
2. 为什么 LayerNorm 的**比值更大**？
3. 如果把 dtype 从 fp16/bf16 换成 fp32，这两个比值会变大、变小还是不变？为什么？

---

## E2 概念题 ★ 不需要硬件

一个逐元素算子是 `y = x * a + b`，其中 `x`、`b`、`y` 都是长度 N 的 fp32。

1. 按账本口径，写出「一个 op 一个 kernel」写法的 `(flops, bytes)`；
2. 写出「融合成一个 kernel」写法的 `(flops, bytes)`；
3. 分别算出算术强度，并说明「这两个值都远低于机器平衡点」这个结论是否依赖
   具体是哪一款加速器。

---

## E3 概念题 ★★ 不需要硬件

下面两种写法都能算 `C = A + B`：

```python
# 写法一
with T.Kernel(T.ceildiv(N, 256), threads=256) as bx:
    for i in T.Parallel(256):
        C[bx * 256 + i] = A[bx * 256 + i] + B[bx * 256 + i]

# 写法二
with T.Kernel(T.ceildiv(N, 2048), threads=256) as bx:
    for i, j in T.Parallel(256, 8):
        idx = (bx * 256 + i) * 8 + j
        C[idx] = A[idx] + B[idx]
```

官方文档称，写法一在 `threads=256` 下生成的 CUDA 是「32 个活跃线程、每个做一次
`uint4` 访存」，而 launch config 仍然是 256 线程。请解释：

1. 为什么 extent 是 256 的循环最后只需要 32 个线程？
2. 为什么 launch config 不跟着改？
3. 写法二相比写法一，把哪一件事变成了「你可以控制的」？

---

## E4 代码阅读题 ★★ 不需要硬件

`N = 2047`、`threads = 256`、每线程 8 个元素的 TileLang kernel，生成的 CUDA 是：

```c++
extern "C" __global__ void __launch_bounds__(256) main_kernel(
    bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (((i * 256) + ((int)threadIdx.x)) < 2047) {
      C[(i * 256) + ((int)threadIdx.x)] = (A[...] + B[...]);
    }
  }
}
```

请回答：

1. 这段代码里为什么没有出现 `uint4`？（用本章 §4.1 的判定链回答）
2. 这个 `if` 是谁插进去的？在哪个 pass 里？（给文件名）
3. 如果 `N` 改成 2048，生成的代码会有什么变化？为什么？
4. 如果想在保留 `N = 2047` 的前提下让「前面 99.9% 的元素」走到向量化路径，
   你会怎么改 kernel？（提示：第 09 章讲过一种让编译器跳过自动谓词的方式）

---

## E5 概念题 ★★ 不需要硬件

关于 `T.reduce_*` 的 `clear` 参数：

1. `clear=True`（默认）时，`T.reduce_sum(src, dst, dim=1)` 对 `dst` 的旧值做什么？
2. `clear=False` 时，"把结果和 dst 的旧值合并"为什么不能直接把 dst 当累加器？
3. 哪个归约类型的 `clear=False` 不需要额外的临时 buffer？为什么？
4. 写出 `clear=False` 的 `T.reduce_sum` 在实现上的三步（提示：`reduce_op.py` 的
   `reduce_macro`）。

---

## E6 代码阅读题 ★★ 不需要硬件

下面是 split-K 版 RMSNorm 的骨架：

```python
T.clear(A_local)
for k in T.Serial(num_k_step):
    T.copy(A[bx * blk_m, k * blk_k], A_shared)
    for i, j in T.Parallel(blk_m, blk_k):
        A_local[i, j] += A_shared[i, j] * A_shared[i, j]
T.reduce_sum(A_local, A_powsum, dim=1)
for i in T.Parallel(blk_m):
    A_powsum[i] = T.rsqrt(A_powsum[i] / N + 1e-12)

for k in T.Serial(num_k_step):
    # reverse, better cache hit rate
    T.copy(A[bx * blk_m, (num_k_step - 1 - k) * blk_k], A_shared)
    for i, j in T.Parallel(blk_m, blk_k):
        A_shared[i, j] *= A_powsum[i]
    T.copy(A_shared, B[bx * blk_m, (num_k_step - 1 - k) * blk_k])
```

请回答：

1. `T.clear(A_local)` 能不能省？省掉会发生什么？
2. 第一个循环里的 `+=` 为什么不会破坏向量化？（提示：看 §4.1 判定链的第 5 条，
   注意这条 store 的索引与循环变量的关系）
3. 第二个循环为什么倒序？代价是什么（写出这个写法下 x 被读了几遍）？
4. 这个版本与「整行放 fragment」的版本相比，各自的适用场景是什么？

---

## E7 代码阅读题 ★★ 不需要硬件

`T.reduce_sum` 的 src 与 dst 可以是 shared 或 fragment。下面四种组合，前端
分别做了什么：

```text
① (shared,   shared)
② (shared,   fragment)
③ (fragment, shared)
④ (fragment, fragment)
```

请对每一种写出：前端是否插入额外的 fragment、是否插入 `T.copy`。
然后回答：为什么在归一化里，"先把数据读进 fragment 再归约"比"在 shared 上归约"
更省片内搬运？

---

## E8 动手题 ★★ 不需要硬件（真机验证可选）

写一个 TileLang kernel 计算 `y = x * a + b`（`a` 是标量，`b` 是长度 N 的向量），
要求：

1. 在 `N = 1 << 20`（对齐）和 `N = 1000`（非对齐）两种情况下都正确；
2. 对齐情况下生成代码里能看到 128-bit 访存；
3. 在你的解里说明清楚：非对齐时用什么机制保证了正确性，以及它对向量化的影响。

提示：`code/02-elementwise/elementwise_variants.py` 里的 `add_flat` 与
`add_thread_vec` 是两个可以直接改的起点。

---

## E9 动手题 ★★ 不需要硬件（真机验证可选）

把本章的 RMSNorm 改成**带权重**的版本：

```text
y = x * rsqrt(mean(x^2) + eps) * w
```

要求：

1. 写出 PyTorch 参考实现；
2. 写出 TileLang kernel（提示：权重要在片上，考虑放 shared 还是 fragment）；
3. 用账本口径写出「逐算子写法」与「融合写法」的字节数，并给出比值；
4. 说明权重放在 shared 与放在 fragment 各有什么代价。

---

## E10 代码阅读题 ★★ 不需要硬件

下面两段 PyTorch 参考实现分别对应哪一种 RoPE 约定？

```python
# 甲
half = x.shape[-1] // 2
lo, hi = x[..., :half], x[..., half:]
y = torch.cat([lo * cos - hi * sin, hi * cos + lo * sin], dim=-1)

# 乙
head = x.reshape(*x.shape[:-1], -1, 2)
x0, x1 = head[..., 0], head[..., 1]
y = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1).flatten(-2)
```

请回答：

1. 甲、乙分别是哪一种？
2. 用 `cos ≡ 1`、`sin ≡ 0` 时，两者应该输出什么？这个性质为什么可以用来做自检？
3. 如果 `x` 的最后一维是 128，两种约定下 `cos` 的形状各是什么？
4. 在 TileLang 里实现时，哪一种约定的两条 store 更容易各自向量化？为什么？

---

## E11 分析题 ★★★ 不需要硬件（真机验证可选）

本章的 `rms_norm_rope_fused` 把 `blk_m` 默认取 1。请回答：

1. `blk_m` 代表什么？把它调大在账本上带来什么好处？
2. 限制它调大的三件事分别是什么（提示：寄存器、片上存储、layout 约束）？
3. 如果要支持 `blk_m > 1` 且 N 很大，有哪两条路线？（本章正文提到过）
4. 写出一份「在真机上验证这个 kernel」的步骤清单（至少 4 步，要包含怎么排查
   layout 冲突）。

---

## E12 概念题 ★★ 不需要硬件

下面两段代码片段各属于哪一个方言？把判断依据（具体到 API 名）写出来。

```python
# 甲
with T.Kernel(m_num, is_npu=True) as (cid, vid):
    a_ub = T.alloc_ub((rows, block_N), dtype)
    with T.Scope("V"):
        T.copy(A[bx, by], a_ub)
        T.barrier_all()
        T.tile.add(c_ub, a_ub, b_ub)

# 乙
with T.Kernel(num_blocks) as bx:
    a = T.alloc_shared((tile,), "float32")
    for step in T.Pipelined(iterations, num_stages=2):
        T.copy(A[begin : begin + tile], a)
        with T.SimtVF(threads=2048):
            for i in T.Parallel(tile):
                c[i] = a[i] + b[i]
```

另外回答：如果要把甲改写成乙所在的方言，至少需要改哪四处？

---

## E13 设计题 ★★★ 不需要硬件（真机验证可选）

设计一个 fused kernel：`y = rms_norm(x + residual)`，即「残差相加 + RMSNorm」
融合（这是每个 Transformer block 都会出现的模式）。要求：

1. 写出 PyTorch 参考实现；
2. 用账本口径写出「2 个算子分别写」与「融合写」的字节数，给出比值；
3. 给出 TileLang 的数据流草图（哪几步在 fragment、哪几步在 shared）；
4. 指出这个融合新增的**一个**副作用（提示：残差那条路下游还需要 x + residual
   这个中间值，想一想 fused 之后它去哪了）；
5. 说明你会怎么验证它（正确性判据 + 性能口径）。
