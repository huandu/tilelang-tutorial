# 第 03 章 习题

说明：

* 难度标注为 `★`（基础，读完正文即可作答）/ `★★`（需要综合本章多处内容）/
  `★★★`（需要动手、多步推理，或需要对照源码）。
* “需要硬件”一栏中，`否` 表示只要一台能跑 Python 的机器；`是（GPU）` 表示必须用真实
  NVIDIA GPU 才能完成；`否（但是要读源码）` 表示不需要硬件，但必须在
  `_vendor/tilelang-v0.1.15/` 里核对。
* 所有题目只依赖第 01–03 章的知识。涉及后续章节的地方会显式标注。
* 本章没有任何必须在本机实测才能得出的结论，因此**没有强制要求硬件的题目**；
  需要 GPU 的题都标为“可选”或“有卡时做”。

---

## E1 概念题 ★ 不需要硬件

阅读正文 2.1 节，回答：

1. 用一句话复述 TileLang 论文摘要的核心主张（可以引用英文原文，但要解释）。
2. 论文把“调度空间”括号展开成了哪四个词？请按顺序写出来。
3. “解耦”在这句话里，解的到底是**哪两样东西**？各举一个 TileLang 里的具体例子
   （一个属于调度，一个属于数据流）。

---

## E2 概念题 ★ 不需要硬件

正文 1.2 节把“决定”分成了两类。请针对下面的说法判断对错，并说明理由：

1. “`T.alloc_shared((128, 32), 'float16')` 属于数据流，因为 storage scope 是数据本身
   的属性。”
2. “把 `T.Pipelined(..., num_stages=3)` 改成 `num_stages=2` 属于数据流改动，因为它改变了
   循环的迭代次数。”
3. “`T.gemm(A_s, B_s, C_f)` 里不写 `policy=...` 时，thread binding 这一维度就没有被决定。”

---

## E3 概念题 ★★ 不需要硬件

正文 2.4 节给出了本书对 tile 的三要素定义。

1. 写出这三要素。
2. `T.alloc_fragment((block_M, block_N), accum_dtype)` 申请的那块缓冲，物理上**并不真的存在**。
   请解释：程序运行时，这块“逻辑 tile”去哪了？由哪个编译期环节负责？
3. 如果用一句话向刚学完第 01 章的读者解释“为什么 `alloc_fragment` 是一等概念的最好证据”，
   你会怎么说？

---

## E4 概念题 ★★ 不需要硬件

正文 2.5 节说 beginner 层“方向存在，但层本身未完成”。请回答：

1. 官方文档里原话是怎么标注的？它出现在哪个文件的第几行附近？
2. v0.1.15 里现存的那个“雏形”叫什么前端？它解决了哪一件事，又**没有**解决哪一件事？
3. 为什么本章坚持把它标注为“未完成”，而不是简单说“TileLang 有硬件无关层”？

---

## E5 概念题 ★★ 不需要硬件

正文 5.1 节引用了 Triton 2019 年论文里的一个词：**hiding**。

1. 它藏起来的是什么？（至少写出三类决定）
2. 论文自己承认这条路线要付出什么代价？引用原文一句。
3. TileLang 相对它多保留了哪一类信息？这个选择带来的**第一项代价**是什么（见 5.4 节）？

---

## E6 代码阅读题 ★ 不需要硬件（要读源码）

下面这段来自锁定版本（`_vendor/tilelang-v0.1.15/examples/quickstart.py:29-39`，节选）：

```python
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    T.copy(A[by * block_M, ko * block_K], A_shared)
    T.copy(B[ko * block_K, bx * block_N], B_shared)
    T.gemm(A_shared, B_shared, C_local)
```

请回答：

1. 这段循环体里出现了几个“tile op”？分别叫什么（用 IR 里的名字）？
2. 这段代码**没有**写出哪些信息？（至少两条：一条关于线程，一条关于同步）
3. 这三条信息最终由哪个 pass 补上？请给出 v0.1.15 里那个 pass 的名字与文件路径。

---

## E7 代码阅读题 ★★ 不需要硬件（要读源码）

下面这段来自 `_vendor/tilelang-v0.1.15/examples/gemv/example_gemv.py:28-40`（节选）：

```python
with T.Kernel(T.ceildiv(N, BLOCK_N)) as bn:
    tn = T.get_thread_binding(0)  # tn = threadIdx.x
    A_shared = T.alloc_shared((BLOCK_K,), dtype)
    B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
    C_reg = T.alloc_local((1,), accum_dtype)
    T.clear(C_reg)
    for bk in T.serial(T.ceildiv(K, BLOCK_K)):
        for tk in T.serial(BLOCK_K):
            A_shared[tk] = A[bk * BLOCK_K + tk]
            B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
        for tk in T.serial(BLOCK_K):
            C_reg[0] += A_shared[tk].astype(accum_dtype) * B_shared[tn, tk].astype(accum_dtype)
    C[bn * BLOCK_N + tn] = C_reg[0]
```

请回答：

1. 这段代码属于三层接口的哪一层？给出**两条**判断依据（要看它用了哪些原语）。
2. `A_shared` 与 `B_shared` 都放在 shared memory 里，但它们的数据流**不一样**：
   `A_shared` 的什么特点让它可以被所有线程共用，而 `B_shared` 为什么必须带 `tn` 下标？
3. `C_reg` 为什么用 `T.alloc_local` 而不是 `T.alloc_fragment`？如果换成 `alloc_fragment`，
   下一步会发生什么（用第 4 节的机制回答）？

---

## E8 代码阅读题 ★★★ 不需要硬件（要读源码）

下面两段来自**同一个文件**（`_vendor/tilelang-v0.1.15/examples/dequantize_gemm/example_dequant_gemm_bf16_mxfp4_hopper.py`）：

```python
# 片段 A（:366-373）
for k in T.Pipelined(K // block_K, num_stages=num_stages):
    T.copy(A[by * block_M, k * block_K], A_shared)
    T.copy(B[bx * block_N, k * block_K // num_elems_per_byte], B_shared)
    if fast_dequant:
        get_fast_dequant_twiddling_func()(B_shared, B_dequantize_shared, Scale, k)
    else:
        get_simple_dequant_func()(B_shared, B_dequantize_shared, Scale, k)
    T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)

# 片段 B（:225-231，属于 fast_dequant 那条路的 macro 内部）
tx = T.get_thread_binding()
B_local_thread = T.alloc_local((local_compress_size,), storage_dtype)
B_dequantize_local_thread = T.alloc_local((local_size,), out_dtype)
Scale_local_thread = T.alloc_local((1,), storage_dtype)
Scale_local_thread_exponent = T.alloc_local((1,), out_dtype)
```

请回答：

1. 片段 A 里那一行 `if fast_dequant:` 决定的是什么？请用本节的术语（**层次**）来回答。
2. 为什么这两条路的**接口**可以完全一样（都是“输入 B_shared + Scale + k，输出
   B_dequantize_shared”），而**实现**落在两个不同的层次上？这与“composable”是什么关系？
3. 假设有人把 `fast_dequant` 那条路拆成**另一个 kernel**，先跑反量化再跑 GEMM。请说出
   这样做在数据流上的两个具体损失。

---

## E9 动手题 ★★ 不需要硬件

运行本章 `code/01-layer-inventory/`：

```bash
python3 chapters/03-design-philosophy/code/01-layer-inventory/layer_inventory.py
```

1. 贴出它的最后三行输出（合计 / 缺失）。
2. 从输出里找出**只看 `export` 而不是 `def`** 的三条，并说明“`export`”在这里是什么意思。
3. 往 `LAYERS` 元组里加一个本章正文提到过的名字（例如 `tilelang/language/customize.py`
   里的 `clamp` 或 `reshape`），再跑一次。改完后退出码是多少？

---

## E10 动手题 ★★ 否（但是要读源码），有 GPU 时可加做

写一个**只有 6–10 行**的“层次地图”：从
`_vendor/tilelang-v0.1.15/tilelang/language/__init__.py` 出发，说明
`import tilelang.language as T` 时，`T.gemm` 这个名字是**怎么一步步被暴露出来**的。
要求给出每一步的文件与行号。

提示：注意 `tilelang/language/__init__.py:11-12`、`tilelang/cuda/language/__init__.py`
的再导出、以及 `tilelang/language/gemm_op.py`。

---

## E11 动手题 ★★★ 不需要硬件（要读源码，需推理）

正文 4.3 节给出了 CUDA pipeline 里四个 pass 的顺序：

```text
PipelinePlanning → InjectSoftwarePipeline → LayoutInference → LowerTileOp
```

1. 在 `_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py` 里找到这四行的行号。
2. 源码注释解释了“为什么 LayoutInference 必须在 pipeline 重写**之后**”。把这句话抄下来
   并翻译成中文。
3. 假设有人把 `LayoutInference` 挪到 `PipelinePlanning` 之前，请基于那句注释推断**至少一类**
   会出问题的场景（不需要真的改代码运行）。

---

## E12 动手题 ★★★ 需要硬件（有卡时做；无卡时只做前两问）

阅读并（在有 GPU 的机器上）运行
`code/03-schedule-as-annotation/schedule_as_annotation.py`。

1. 说出四个变体 `v0`–`v3` 分别改动了调度空间的哪一个维度。
2. 无 GPU 时：运行 `--static-only`，贴出“调度维度对照”那段输出。
3. 有 GPU 时：四个变体的**正确性**应当一致。请设计一个实验验证这一点，并说明你会记录
   哪些测量条件；**不要**在本题里下性能结论，性能对比属于第 28 章的方法论。
4. 有 GPU 时（可选，进阶）：把 `v2` 的 `make_swizzled_layout(B_s)` 换成
   `T.Layout(B_s.shape, lambda i, j: (i, j))`（恒等布局）再跑一次。猜测并验证
   “显式恒等布局”与“完全不写 `annotate_layout`”是否等价，说明你的观察。
