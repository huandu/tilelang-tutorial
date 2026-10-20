# 第 33 章 习题

说明：

* 难度：`★`（读完正文即可作答）/ `★★`（需要综合本章多处内容）/ `★★★`（需要动手、
  多步推理，或需要对照源码）。
* 「需要硬件」一栏中，`否` 表示只要一台能跑 Python 的机器；`是（GPU/NPU）` 表示必须用
  真实的 NVIDIA GPU 或昇腾 NPU；`否（但要读源码）` 表示不需要硬件，但要在
  corpus 快照或 `_vendor/tilelang-v0.1.15/` 里核对。
* 所有题目只依赖第 01–33 章的知识。涉及后续章节的地方会显式标注。
* 写作环境没有 GPU/NPU（见 `../../VERSIONS.md`），所以涉及实测的题都只要求
  「说明怎么测、判据是什么」，不要求给出测量值。
* 题中出现的行号一律指 corpus 快照
  `../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/` 下的文件。

---

## E1 概念题 ★ 不需要硬件

正文 §2.2 把 TileKernels 的源码切成三层。判断下面每一条应该放进哪一层，
并说明理由（如果某一层都不合适，指出它该放哪里）：

1. `assert scores.dim() == 2 and scores.is_contiguous()`
2. `if is_ascend():` 分支里 import 昇腾工厂函数
3. `T.alloc_reducer((1,), T.int32, op='min')`
4. `torch.empty((num_tokens, num_topk), dtype=torch.int64, device=scores.device)`
5. `T.annotate_buffer_versions({scores_ub: 6})`
6. `def get_packed_ue8m0_pack_factor() -> int: return 2 if is_ascend() else 4`
7. 一段把 `shared_memory_per_multiprocessor` 读出来的探测函数

---

## E2 概念题 ★ 不需要硬件

正文 §3.1–§3.2 讲了运行期后端选择。回答：

1. `is_ascend()` 的判据是什么？为什么它用 `functools.lru_cache(maxsize=1)`？
2. 为什么昇腾分支用**函数内 import**，而 CUDA 分支在**模块顶层** import？
3. `get_num_vec_cores()` 为什么是 `get_num_sms() * 2`？如果纯向量算子错用了
   `get_num_sms()`，会有什么现象？
4. 为什么 `get_num_vec_cores()` 要在**调用工厂函数时**求值，而不是在 kernel 里
   当成一个变量？
5. `get_pdl()` 为什么只出现在 CUDA 分支的参数里，不出现在昇腾分支？

---

## E3 概念题 ★★ 不需要硬件（但要读源码）

第 19 章讲了昇腾的「两棵树」（适配器 A 树 / 主干 M 树）。本章的 TileKernels：

1. 用的是哪一棵？给出文件与行号作为证据（至少两处）。
2. 正文 §3.3 给了两个「为什么选它」的理由。说出这两条。
3. 如果你把 A 树的写法（例如 `T.alloc_L1`、`T.gemm_v0`、`is_npu=True`）搬进
   `topk_gate_asc.py`，会出现什么后果？为什么？
4. 一个陷阱题：为什么 `topk_gate_asc.py` 里**没有**一句 `barrier_all` 或 `set_flag`，
   而正文 §4.3 却说它有一句手写同步？它在哪一行、保护什么？

---

## E4 代码阅读题 ★ 不需要硬件

下面是某个 kernel 的片段（已做删减）：

```python
with T.Kernel(num_vec_cores) as core_id:
    scores_ub = T.alloc_shared((num_aligned_experts,), T.float32)
    invalid_logits_ub = T.alloc_shared((vector_length,), T.int32)
    T.annotate_buffer_versions({scores_ub: 6, invalid_logits_ub: 1})
    for w in T.Pipelined(T.ceildiv(num_tokens, num_vec_cores), num_stages=6,
                         annotations={'multi_buffer_eligible': [scores_ub]}):
        row = w * num_vec_cores + core_id
        if row < num_tokens:
            T.copy(scores[row, :], scores_ub[:num_experts])
```

1. 这是**哪个后端**的写法？列出至少三条判据。
2. `alloc_shared` 在这里分配的是哪一级存储？为什么不是寄存器？
3. `row = w * num_vec_cores + core_id` 这种划分方式，与「把 token 连续切给每个核」
   相比，有什么好处？
4. `if row < num_tokens:` 这个守卫在 CUDA 版里有没有对应物？为什么？
5. 如果有人把 `T.annotate_buffer_versions` 整行删掉，代码能不能编译？会不会出错？

---

## E5 代码阅读题 ★★ 不需要硬件

下面是 Ascend 版里处理 `int64` 输出的三行：

```python
topk_idx_view = T.view(topk_idx, (num_tokens, num_topk * 2), T.int32)
...
with T.SimdVF():
    T.clear(out_ub)
...
S.vsts(out_ub[2 * k], idx, one, 'ONEPT_B32')
...
T.copy(out_ub[: num_topk * 2], topk_idx_view[row, :])
```

1. 这四行合起来在做什么？把「写一个 int64」拆成了什么？
2. `T.view` 的契约是什么（写三条）？为什么这里能满足？
3. `T.clear(out_ub)` 为什么是必需的？删掉它会怎样？为什么这种错误可能在小 shape
   的测试里看不出来？
4. CUDA 版是怎么做同一件事的？两种做法的前提有什么不同？

---

## E6 代码阅读题 ★★ 不需要硬件

下面是 CUDA 版的非有限值检查：

```python
T.reducer_init(nonfinite_reducer)                       # op='sum'
for i in T.Parallel(num_aligned_experts):
    if i < num_experts and not T.isfinite(scores_fragment[i]):
        T.reducer_update(nonfinite_reducer[0], 1)
T.finalize_reducer(nonfinite_reducer, nonfinite_result)
T.device_assert(nonfinite_result[0] == 0, msg='Score input for topk gate must be finite')
```

1. 注释里写「single deferred assert to keep the load loop vectorized」。
   如果改成在 `T.Parallel` 循环体里逐个 `T.device_assert(is_finite, ...)`，
   会失去什么？
2. 为什么这里需要检查有限性？只用一句话答，但要指出它和「越界填 `-inf`」的依赖关系。
3. 昇腾版没有用 reducer，它是怎么等价地做这件事的？给出关键原语名。

---

## E7 概念题 ★★ 不需要硬件

正文 §4.5 说「CUDA 的并行是 32 个独立的小计算，昇腾的并行是一条 64 宽度的向量指令」。

1. 用两句话解释这句话分别指什么。
2. 在昇腾的 `T.Kernel(num_vec_cores)` 里写 `threads=32` 会发生什么？为什么？
3. 昇腾侧如果要真的用「线程」，应该用什么？给出一个本章出现过的例子（文件 + 行号）。
4. `topk_gate_asc.py` 里 `vector_length = 64`、`max_vregs_per_group = 6`
   这两个常量是怎么分工的？`num_groups > 1` 时会发生什么？

---

## E8 概念题 ★★ 不需要硬件

正文 §4.6 解释了 `T.annotate_buffer_versions`。

1. `scores_ub` 与 `invalid_logits_ub` 的版本数分别是多少？各自的理由是什么？
2. 用一句话说明「调度器会算版本数」和「你必须写版本数」这两件事不矛盾。
3. 如果给 `invalid_logits_ub` 也写上 `num_stages`（6 份），会有什么后果？
4. CUDA 版为什么完全不需要这一套？（提示：从「一行一批做完」这个结构想。）

---

## E9 概念题 ★ 不需要硬件

给出下面三个 `sf_block`，分别说出它属于 per-token / per-block / per-channel 里的哪一种，
并说明缩放尺度沿哪个维度变化：

1. `sf_block = (1, 128)`
2. `sf_block = (32, 32)`
3. `sf_block = (128, 1)`

再回答：`per_block_cast` 在昇腾上只支持哪一种 `sf_block`？这一限制体现在哪一行？

---

## E10 概念题 ★★ 不需要硬件

1. UE8M0 的打包因子在 CUDA 与昇腾上分别是多少？对应的 torch dtype 是什么？
2. 把在 CUDA 上量化并打包好的权重直接搬到昇腾上使用，会出什么问题？
3. 除了打包因子，正文 §5.6 还列了另外两条后端差异（一个是不支持的选项，
   一个是对齐要求）。分别说出它们，并说明为什么它们不会让两个后端的**数值口径**
   产生分歧。

---

## E11 概念题 ★★ 不需要硬件

SwiGLU 前向 + per-token 量化的融合（§5.5）。

1. 输入是 `[num_expanded_tokens, 2*hidden]`，输出是 `[num_expanded_tokens, hidden]`。
   解释这个 2 倍关系从哪来。
2. 正文给了一个「方案 A（两个 kernel）4.5H vs 方案 B（一个 kernel）2.5H」的记账
   （以 fp16 计、忽略 sf 与缓存）。用同样的口径，把量化目标改成 **fp4**（每个元素
   0.5 字节）重算一遍，给出 A 与 B 的访存比。
3. 融合带来的代价是什么？（提示：看 `swiglu_forward_asc.py` 里的缓冲区数量。）
4. 为什么说「SwiGLU 是 memory-bound 算子，所以省下的就是实打实的时间」？
   这句话依赖什么前提？如果 SwiGLU 变成 compute-bound，结论还成立吗？

---

## E12 动手题 ★★ 不需要硬件（但要读源码）

`rand/randn_kernel.py:24-51` 是本章唯一「只有一个后端」的例子。

1. 在 CUDA 路径上，它做了什么？为什么这样是合理的？
2. 昇腾路径上有哪一条断言是「正确性优先于功能」的典型？解释它挡住了什么错误。
3. 既然 `T.rng_init` / `T.rng_rand_float` 在**两个**方言里都存在
   （`_vendor/tilelang-v0.1.15/tilelang/cuda/language/random.py` 与
   `.../tilelang/ascend/language/random.py`），为什么这个库还是不给 CUDA 写一个
   自己的 randn kernel？给出你认为最合理的解释（可以从三层结构与维护成本角度谈）。

---

## E13 动手题 ★★★ 需要 Python，不需要硬件

用本章示例 `code/03-map-and-diff/map_and_diff.py` 做一次机械核对。

1. 先不看输出，对下面两个算子各预测一下：哪些结构性标记会「只在一边」？
   - `moe/moe_topk_gate_forward`（CUDA 版有 `T.alloc_shared` 与 warp 归约；
     Ascend 版有 `T.Persistent`、`T.SimtVF`、`T.annotate_buffer_versions`）
   - `quant/per_token_cast`
2. 运行脚本，把你的预测与实际输出逐条对照，写出猜错的项与原因。
3. 再针对 `rand/randn` 回答：为什么它不会出现在脚本第 2 步的对照表里？
4. 脚本第 3 步的输出是「21/21 命中」。说明这一步**能**证明什么、**不能**证明什么。

---

## E14 代码阅读题 ★★★ 不需要硬件（但要读源码）

`transform/rope_asc.py` 的 `rotary_dim = 128` 分支里有这么一段（`:182-192`）：

```python
first, second = load2(x_ub, row, head, 0, head, half_dim, True)
if dtype == T.bfloat16:
    first, second = (first, second) if interleaved else S.vintlv(first, second)
elif interleaved:
    first, second = S.vdintlv(first, second)
first, second = rotate(first, second, cosine, sine, neg_sine)
if dtype == T.bfloat16:
    first, second = (first, second) if interleaved else S.vdintlv(first, second)
elif interleaved:
    first, second = S.vintlv(first, second)
```

1. 这两处重排为什么是「一次合、一次拆」的对称结构？为什么必须对称？
2. CUDA 版（`rope_cuda.py:89-100`）做同一件事用的是什么写法？为什么它不需要这一步？
3. 为什么在 `dtype == T.bfloat16` 时重排的是 NeoX（`interleaved=False`）一侧，
   而在 fp32 时重排的是 GPT-J（`interleaved=True`）一侧？
   （提示：`S.vcvt(..., part=0/1)` 的语义是 `PART_EVEN` / `PART_ODD`，
   见 `_vendor/tilelang-v0.1.15/tilelang/ascend/language/simd.py:1247-1249`。）
4. 用一句话说明：这一步和旋转公式本身有关系吗？
