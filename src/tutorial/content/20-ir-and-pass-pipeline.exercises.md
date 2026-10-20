# 第 20 章 习题

难度：`★` 入门 / `★★` 需要读一点源码 / `★★★` 需要动手或综合。
「需要硬件」一栏标明是否必须在有 GPU/NPU 的机器上做；本章绝大多数习题都可以在
任意机器上完成。

---

### E1 概念题 ★

把本章 2.2 节的七个节点（`IRModule` / `PrimFunc` / `Buffer` / `Var` / `PrimExpr` /
`SBlock` / 轴）各用一句话写出来，并回答：`"tilelang_root"` 这个名字硬编码在
锁定版本的哪个文件、哪一行？用什么函数判断一个块是不是 kernel 主块？

*需要硬件：否*

---

### E2 概念题 ★★

CUDA 与 Ascend 调用 `MaterializeKernelLaunch` 时的参数不同。写出两侧的参数，并解释：

1. `lower_thread_binding=False` 对 Ascend 意味着什么？
2. `launch_dim_tags=["cthread"]` 里的 `cthread` 是什么维度，谁读回它？
3. 为什么「保留线程域」这件事会直接决定两侧编程模型的形状？

*需要硬件：否*

---

### E3 代码阅读题 ★★

`PipelinePlanning` 把一个 `num_stages=3` 的循环（体里是「copy A / copy B / gemm」
三条语句）改写成：

```python
annotations={
    "software_pipeline_order": [0, 1, 2],
    "software_pipeline_stage": [0, 0, 2],
    "software_pipeline_async_producers": [1, 1, 0],
    "tl_pipelined_num_stages": 3,
}
```

请回答：

1. `stage=[0,0,2]` 说明这三条语句分别属于第几级？为什么 gemm 是 2 而不是 3？
2. `async_producers=[1,1,0]` 里 `1` 与 `0` 分别代表什么？
3. 为什么这个 pass「循环体一个字都没改」？谁负责真正的改写？

*需要硬件：否*

---

### E4 概念题 ★★

本章把 pass 分成「先规划、后改写」两类。请分别举出两个例子，并说明「规划类 pass」
通常把信息写在 IR 的什么地方。

*需要硬件：否*

---

### E5 代码阅读题 ★★

看本章 3.1 节那段真实 IR：

```python
@T.prim_func
def foo(x_handle: T.handle):
    x = T.match_buffer(x_handle, (2,), strides=(1,))
    ...
    with T.block("tilelang_root"):
        T.reads()
        idx = T.Buffer((1,), "int32", scope="local.var")
        ...
```

请解释：

1. 为什么参数是 `x_handle: T.handle` 而不是 `x: T.Tensor((2,), "float32")`？
2. `T.reads()` 与 `T.writes(...)` 是「指令」还是「声明」？证据是什么？
3. `scope="local.var"` 与 `scope="shared"` 在后续 pass 里的命运差别是什么？

*需要硬件：否*

---

### E6 动手题 ★

运行本章的顺序提取脚本：

```bash
cd chapters/20-ir-and-pass-pipeline/code/01-pass-order
python3 dump_pass_order.py
```

报告三个数字：CUDA 的 pass 调用总数、Ascend 的 pass 调用总数、两侧「模块限定名去重后
取交集」的同名 pass 数。再回答：CUDA 的 `Simplify` 在 Ascend 里也出现吗？

*需要硬件：否*

---

### E7 动手题 ★★

运行版本对照脚本：

```bash
cd chapters/20-ir-and-pass-pipeline/code/04-version-diff
python3 compare_ascend_pipelines.py --grep InsertNd2Nz
python3 compare_ascend_pipelines.py --grep 'InsertNd('
```

报告两次输出的差别，并说明：为什么第二条命令没有输出？这说明 `InsertNd` 与
`InsertNd2Nz` 是什么关系？

*需要硬件：否*

---

### E8 代码阅读题 ★★★

`InjectFenceProxy` 的官方文档给出了一对 before/after IR（见
`code/02-ir-snippet-extract/` 抽出的 `--only fence` 输出）。请回答：

1. after 里比 before 多了哪一行？
2. 它插在什么位置？为什么必须插在那里（提示：generic proxy 与 async proxy）？
3. 这个 pass 在 CUDA pipeline 的哪一行被调用？为什么非 Hopper/非 TMA 目标是 no-op？

*需要硬件：否*

---

### E9 概念题 ★★★

本章说 CUDA 与 Ascend「共享一条 lowering 走廊，在三处主题上分岔」。写出这**三个主题**，
并为每个主题各列出至少两个「一侧有、另一侧用不同 pass 替代」的例子。

*需要硬件：否*

---

### E10 动手题 ★★★

设计一个「逐 pass 读 IR」的可复现流程，针对下面这个最小 kernel：

```python
@T.prim_func
def main(A: T.Tensor((N,), "float16"), B: T.Tensor((N,), "float16"),
         C: T.Tensor((N,), "float16")):
    with T.Kernel(T.ceildiv(N, 256 * 8), threads=256) as bx:
        for i, j in T.Parallel(256, 8):
            off = (bx * 256 + i) * 8
            C[off + j] = A[off + j] + B[off + j]
```

写出命令序列（用 `TL_LOWER_TRACE` 或 `lower_trace()`），并列出你预期在
`LayoutInference` / `LowerTileOp` / `VectorizeLoop` / `ThreadSync` / `MakePackedAPI`
之后各看到什么变化。本题不要求真的运行（本机无 GPU）。

*需要硬件：是（在 GPU 机器上）*

---

### E11 概念题 ★★

`ThreadSync` 会打印一条 `LOG(WARNING) "[ThreadSync] Hoisting sync out of an if ...
This is not a fix"`。请回答：

1. 这条警告为什么说「不是修复」？
2. 正确的修法是什么？为什么加 `T.assume` 有效？
3. 另外两条 `LOG(FATAL)`（`:327` 与 `:1068`）分别在什么条件下触发？

*需要硬件：否*

---

### E12 概念题 ★★

`StorageRewrite` 在 v0.1.15 里为什么「reuse 默认关闭」？那么共享内存的复用实际由谁
完成？CUDA 与 Ascend 各自的入口是什么？给出源码位置。

*需要硬件：否*

---

### E13 综合题 ★★★

把两条 pipeline 按本章的九组职责各画一张表（组名 → pass 列表）。要求：至少为每组
写出 CUDA 与 Ascend 各一个 pass，并标出「只有一侧有」的组项。最后用一句话总结
「两条 pipeline 在哪些组上重叠、在哪些组上分岔」。

*需要硬件：否*
