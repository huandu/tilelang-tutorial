# 第 19 章 习题

说明：

* 难度：`★`（读完正文即可作答）/ `★★`（需要综合本章多处内容）/ `★★★`（需要动手、
  多步推理，或需要对照源码）。
* 「需要硬件」一栏中，`否` 表示只要一台能跑 Python 的机器；`是（NPU）` 表示必须用
  真实的昇腾 NPU；`否（但要读源码）` 表示不需要硬件，但必须在 `_vendor/` 里核对。
* 所有题目只依赖第 01–19 章的知识。涉及后续章节的地方会显式标注。
* 本章的**必做部分不要求任何硬件**；标「是（NPU）」的题都是选做的动手题。
* 写作环境没有 GPU/NPU（见 `../../VERSIONS.md`），所以参考答案里凡是涉及实测的，
  都只给「怎么测」与「判据」，不给「测出来是多少」。

---

## E1 概念题 ★ 不需要硬件

正文 §1.1 说「一份 kernel 正确」被拆成三个互相独立的义务。

1. 写出这三个义务的名称，并用一句话解释每个。
2. 举一个例子说明「算法对、tiling 也对，但同步错」会表现成什么样（给出症状）。
3. 为什么说「同步正确」这一条最容易错、最难调？请从「编译器默认行为」和「错误的可见性」
   两个角度各说一句。

---

## E2 概念题 ★ 不需要硬件

正文 §2.4 介绍了 Ascend C 的两种同步原语。

1. `PipeBarrier<PIPE_X>()` 与 `SetFlag`/`WaitFlag` 分别在什么粒度上工作？各解决什么问题？
2. `T.barrier_all()`（等价 `PIPE_ALL`）**不能**做什么？正文 §11 的 P11 给了什么反例？
3. 为什么说「同一条 PIPE 内的指令按顺序发射，不等于按顺序执行完」？这对写同步意味着什么？

---

## E3 概念题 ★★ 不需要硬件

判断下面每一条属于 **A 树（适配器）** 还是 **M 树（主干 v0.1.15）**，并说明判据：

1. `import tilelang.language as T`
2. `T.alloc_l0c((block_M, block_N), "float32")`
3. `with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):`
4. `T.MixedKernel(N, sids=2)`
5. `pass_configs={tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True}`
6. `T.dual_copy(res, temp)`
7. `T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))`
8. `target="ascend"`
9. `T.printf("cid:%d\n", cid)`

然后回答：为什么「一份 kernel 里混用两棵树的 API」几乎一定失败？给出一个具体的报错形态。

---

## E4 代码阅读题 ★ 不需要硬件

下面是 A 树 `examples/elementwise/elementwise_add.py` 的核心片段：

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
    bx = cid // n_num
    by = cid % n_num
    a_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
    b_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
    c_ub = T.alloc_ub((block_M // VEC_NUM, block_N), dtype)
    with T.Scope("V"):
        T.copy(A[bx * block_M + vid * block_M // VEC_NUM, by * block_N], a_ub)
        T.copy(B[bx * block_M + vid * block_M // VEC_NUM, by * block_N], b_ub)
        T.barrier_all()
        T.tile.add(c_ub, a_ub, b_ub)
        T.barrier_all()
        T.copy(c_ub, C[bx * block_M + vid * block_M // VEC_NUM, by * block_N])
```

1. `VEC_NUM = 2` 是干什么的？为什么 UB buffer 的形状是 `block_M // VEC_NUM` 而不是
   `block_M`？
2. 两次 `T.barrier_all()` 各保护哪两个 PIPE 之间的依赖？写错会怎样？
3. 如果把两个 `barrier_all()` 都删掉，代码**可能**在小规模测试里仍然通过。为什么？
   这说明了什么？
4. 这段代码用了哪条数据流（按 §2 的术语写出完整路径）？

---

## E5 代码阅读题 ★★ 不需要硬件

下面是 A 树 `examples/gemm/example_gemm.py` 的核心片段：

```python
with T.Kernel(m_num * n_num, is_npu=True) as (cid, _):
    bx = cid // n_num
    by = cid % n_num
    A_L1 = T.alloc_L1((block_M, K_L1), dtype)
    B_L1 = T.alloc_L1((K_L1, block_N), dtype)
    C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype)
    with T.Scope("C"):
        loop_k = T.ceildiv(K, K_L1)
        for k in T.serial(loop_k):
            T.copy(A[bx * block_M, k * K_L1], A_L1)
            T.copy(B[k * K_L1, by * block_N], B_L1)
            T.barrier_all()
            T.gemm_v0(A_L1, B_L1, C_L0, init=(k == 0))
            T.barrier_all()
        T.copy(C_L0, C[bx * block_M, by * block_N])
```

1. 填出下表（每一步用哪一个搬运/计算单元，对应哪条 PIPE）：

| 步骤 | 搬运/计算单元 | PIPE | 由谁发起 |
| --- | --- | --- | --- |
| GM → L1 | ? | ? | ? |
| L1 → L0A/L0B | ? | ? | ? |
| 累加进 L0C | ? | ? | ? |
| L0C → GM | ? | ? | ? |

2. 为什么 `init=(k == 0)` 是必需的？如果写成 `init=True` 或 `init=False` 分别会怎样？
3. 为什么 L1 缓冲只开 `K_L1` 列而不是 `K` 列？这和 §5.4 的「容量约束」有什么关系？
4. 用 `block_M=128, block_N=256, accum_dtype="float"` 算一下 C_L0 占多少字节，
   并与 L0C 的容量比较。

---

## E6 代码阅读题 ★★ 不需要硬件（但要读源码）

`T.dual_copy` 的规则（M 树 `tilelang/ascend/language/copy_op.py:53` 的 docstring）是：
对 rank ≥ 2 的区域，**恰有一个尾部维度是 2:1**；对 rank = 1，**唯一的维度必须是 2:1**。

判断下面每一对 src/dst 能否被 `T.dual_copy` 接受；能接受的写出是 M-split 还是 N-split：

1. `src: (128, 256)` → `dst: (64, 256)`
2. `src: (128, 256)` → `dst: (128, 128)`
3. `src: (128, 256)` → `dst: (128, 257)`
4. `src: (4096,)` → `dst: (2048,)`
5. `src: (8, 128, 256)` → `dst: (8, 128, 128)`

再回答：`T.dual_copy` 能不能单独把一个纯 Vector kernel 变成「两个 AIV 一起跑」？为什么？

---

## E7 概念题 ★★ 不需要硬件

正文 §2.2 说「NVIDIA 的累加器在寄存器，昇腾的累加器在 L0C」。

1. 这句话对 **tile 尺寸的选择** 有什么直接约束？请写出 L0C 占用的算式。
2. 已知 Atlas A2 的 L0C 是 128 KB。如果输入是 fp16、累加用 fp32，
   问 `block_M = 128` 时 `block_N` 最大能取多少（不考虑 layout padding）？
3. 如果要把 `block_N` 再调大，正文给了哪条路？为什么它有效？
4. 用 `../code/12-tuning-sweep/tune_sweep.py` 验证你在第 2 问得到的边界
   （把 `--stages` 设成 1，看哪些配置被判 OVER）。

---

## E8 概念题 ★★ 不需要硬件

正文 §7.5 介绍了 `T.annotate_buffer_versions` 的三种索引模式。

1. `iteration` 与 `counter` 在选择「用哪个物理槽位」时，分别以什么为时钟？
2. 一个循环里 load/计算/store 共享一个 `if active(i)` 条件，当 `active(i)` 为假时：
   两种模式的行为有什么差别？
3. 哪些情况下 `auto` 会自动选 `counter`？
4. 为什么正文强调「M 树里不要再手动给 buffer 加一个 stage 维」？

---

## E9 动手题 ★★ 不需要硬件（但要读源码）

目标：给 A 树的 GEMM 加一个 ReLU epilogue（`C = relu(A @ B)`）。

1. 从 `../code/04-gemm-l1-l0c/gemm_adapter.py` 出发，写出改动的**思路**：数据要经过哪些
   存储层级？在哪一级做 ReLU 最省搬运？
2. 为什么不能直接在 L0C 上做 ReLU？（提示：L0C 归谁管？谁能直接读写它？）
3. 画出改动后的数据流（用 §2 的层级名）。
4. （选做）如果换成 M 树方言，正文 §7.3 给的 `T.dual_copy` + `T.SimtVF` 方案相比
   A 树手写，省掉了哪些东西？

---

## E10 动手题 ★★★ 是（NPU）

目标：为 `../code/02-elementwise-ub/vec_add_adapter.py` 采一次性能并做判读。

1. 先用 `../code/01-env-check/env_check.py` 确认环境齐备。
2. 用 `msprof op --kernel-name=... python vec_add_adapter.py` 上板采集，
   记录它给出的图里「内存 vs 计算」的走向。
3. 再用 M 树风格的 `do_bench(backend="msprof_detail")` 思路，记录 `aic_mad`、
   `aiv_vec`、`aiv_mte2`、`aiv_mte3` 四个值（如果你手边只有 A 树，请说明你用的替代口径）。
4. 按 §8.3 的判据给出一句结论，并写出你**下一步要调的单一变量**。
5. 把这次实验的完整条件填进 `../code/12-tuning-sweep/tune_sweep.py` 输出的记录表模板。

（参考答案只给判据与流程；**任何具体数值以你现场采集为准**。）

---

## E11 动手题 ★★★ 是（NPU）

目标：把 A 树的逐元素 add 改写成 M 树方言（或反过来）。

1. 列出**必须**修改的每一处（至少 6 处），例如 import、`target`、分配函数名、
   网格与子核写法、计算原语的写法等。
2. 改完之后，哪些地方原本手写的同步可以删掉？为什么？
3. 如果先跑通 M 树版本，再照抄回 A 树，最可能在哪一行先失败？报什么错？

---

## E12 综合题 ★★ 不需要硬件

给定两个真实场景，判断「该不该入图（ACLGraph）」，并说明理由：

1. 一个推理服务，每个请求跑一串 20 个小算子，每个算子单独 launch；
2. 一个大 GEMM（M=N=K=8192），反复调用。

再回答：

3. `torch_tl_ascend` 解决的是哪一类问题？和直接 JIT 出 kernel 收 `.npu()` 张量相比，
   多出了什么能力？
4. 如果要把一个 TileLang 算子接进 `torch.nn.Module` 并支持反向传播，
   本章 §9.4 的检查清单里哪几项最关键？
