# 第 06 章 习题

说明：

* 难度标注为 `★`（基础）/ `★★`（需要综合本章内容）/ `★★★`（链条较长或需要动手）。
* 「需要硬件」一栏中，`否` 表示只需要一台能跑 Python 的机器；`是（GPU）` / `是（NPU）`
  表示需要用真实加速器。本章**没有**强制要求硬件的题目——所有必须在硬件上做的实验都
  标为「可选」。
* 所有题目只依赖第 1、5、6 章及更早的知识；涉及后续章节时会显式提示。

---

## E1 概念题 ★ 不需要硬件

1. 为什么加速器必须有片上存储？请从「延迟」「容量」「带宽」三个量的不对称来解释，
   并引用本章给出的两个量级数字（一个访存延迟、一个算术延迟）。
2. 「片上存储」和「分块（tiling）」各自解决什么问题？请用一句话分别概括。
3. 为什么说「同一份片上数据被复用的次数，就是分块把 HBM 流量压下来多少倍」？请以
   矩阵乘为例说明。

---

## E2 概念题 ★ 不需要硬件

把下面的 NV 侧层级与昇腾侧层级做一一对照，并说明**哪一对并不是真正的对应物**、
为什么：

```text
NV:     寄存器 / shared memory / L2 / HBM
昇腾:   L0A/L0B/L0C / L1 / UB / GM
```

提示：注意「shared memory」在昇腾上要拆成几种角色；注意 L0A/L0B/L0C 在 NV 侧有没有
同名对应物。

---

## E3 概念题 ★ 不需要硬件

一次 $M=N=K=4096$ 的 fp16 矩阵乘：

1. 理论运算量是多少 FLOP？
2. 理论最小访存量是多少字节？（提示：读 $A$、$B$，写 $C$。）
3. 如果不做分块、每个输出元素都重新读一遍 $A$ 的一行和 $B$ 的一列，访存量大约放大
   多少倍？
4. 第 06 章说的「把 HBM 流量压下来」具体是靠哪一个动作实现的？

---

## E4 概念题 ★ 不需要硬件

「GPU 是 SIMT、NPU 是 SIMD」这句话今天为什么不准确？请至少给出两条来自本章
**厂商来源**的反例，并说明每条反例反驳的是这句话的哪一半。

---

## E5 代码阅读题 ★★ 不需要硬件

读 `code/03-target-sketch/nv_hopper_sketch.py`（不需要运行）：

1. `T.alloc_shared` 和 `T.alloc_fragment` 分别对应本章哪一级存储？
2. `T.Pipelined(..., num_stages=3)` 对应本章 1.4 节的哪个概念？它的代价是什么？
3. `T.gemm(A_shared, B_shared, C_local)` 在 Hopper 上可能落到哪一代 Tensor Core 指令？
   在 Blackwell 上呢？请引用锁定版本的源码位置。
4. 如果给 `T.copy` 加上 `disable_tma=True`，会关掉什么？

---

## E6 代码阅读题 ★★ 不需要硬件

读 `code/03-target-sketch/ascend_950_sketch.py`（不需要运行），把下面每一行映射到
本章 4.3 节的搬运单元（MTE1/MTE2/MTE3/Fixpipe）或计算单元：

1. `T.copy(A[m, k * block_K], A_l1)`
2. `T.gemm(A_l1, B_l1, C_l0c, transpose_B=True, clear_accum=(k == 0))`
3. `T.dual_copy(C_l0c, C_ub)`
4. `with T.SimtVF(threads=128): ...`
5. `T.dual_copy(C_ub, C[m : m + block_M, n : n + block_N])`

对第 2 行，请注明它隐含了哪一段 MTE1 搬运。

---

## E7 概念题 ★★ 不需要硬件

按「把什么从寄存器里拿了出来」这条线索，说明 Tensor Core 三代
（`mma` / `wgmma` / `tcgen05`）的演进：

1. 每一代的指令粒度（warp / warpgroup / CTA）是什么？
2. 每一代累加器 `D` 住在哪里？
3. 每一代的同步方式是什么？
4. Blackwell 的 Tensor Memory 在 `sm_100a`/`sm_100f` 上的结构是什么（列 × 行 × 位宽）？

---

## E8 代码阅读题 ★★ 不需要硬件

读 `code/02-pipeline-timeline/pipeline_timeline.py` 的 `schedule()` 函数（不需要运行）：

1. `reuse_ready = starts["copy_out"][i - versions] + copy_out` 这一行在表达什么约束？
2. 如果把这一行删掉、让 `reuse_ready = 0`，模型会给出什么样的（错误的）结论？为什么
   这个错误在真实硬件上会变成「数据被提前覆盖」？
3. 这个模型为了简单，忽略了哪些真实效应？（至少列三条。）

---

## E9 动手题 ★★ 不需要硬件（CPU）

运行 `code/02-pipeline-timeline/pipeline_timeline.py`：

1. 用默认参数跑一遍，记录 `versions=1` 与 `versions=2` 的 makespan 和计算利用率。
2. 把它改成 `--copy-in 10 --compute 40 --copy-out 10` 再跑一遍，记录两组利用率。
3. 用本章的「机器平衡点 / memory-bound vs compute-bound」语言解释：为什么同一个
   多缓冲机制，在两种参数下带来的利用率提升幅度不同？
4. 自行构造一组参数，使 `versions=2` 相对 `versions=1` **几乎没有**提升，并解释原因。

---

## E10 动手题 ★★ 不需要硬件（CPU）

运行 `code/01-onchip-footprint/onchip_footprint.py`：

1. 记录 Case 3（`C_ub(128,256)` fp32）在 1 份和 2 份缓冲下各占多少字节，以及是否
   放得进文档示例里的 192 K UB。
2. 如果要求在 192 K UB 里放下 2 份缓冲，这个 fp32 tile 的最大元素数大约是多少？
   （给出计算过程。）
3. 如果把 dtype 换成 fp16，结论会怎么变？
4. 在真实设备上，`192 K` 这个数能不能直接用？如果不能，正确做法是什么？

---

## E11 概念题 ★★ 不需要硬件

昇腾用成对的 SetFlag/WaitFlag 管理流水依赖。

1. 对「MTE2 写 L1、MTE1 读 L1」这条路径，写出正向（数据就绪）与反向（缓冲归还）
   两对信号的方向。
2. 为什么**只有正向信号是不够的**？请以 double buffer 的第 2 轮覆盖 slot 0 为例说明。
3. TileLang-Ascend 编程指南说「event ID 按有向 pipe pair 分配」。这句话有什么工程
   含义？为什么正反两个方向可以复用同一个编号？

---

## E12 概念题 ★★★ 不需要硬件

下面三段话分别声称了一个「事实」。请判断每一段的证据强度属于
**厂商文档（需注明条件）** / **社区说法（未确认）** / **本章自己的推导**，并写出
你会去 `references.md` 的哪一条核对：

> (a)「昇腾 950PR 最大配备 128GB、1.6TB/s 高速内存。」
> (b)「SIMT 路径的访问 UB 的带宽只有 SIMD 的一半。」
> (c)「一次 $4096^3$ 的 fp16 GEMM，理论最小访存量约 96 MiB。」

对每一段，说明「要把它升级为可引用的结论，还缺什么」。

---

## E13 动手题 ★★★ 需要硬件（GPU，可选）

**（可选，只有有 NVIDIA GPU 的读者需要做。本章写作环境没有 GPU，故本题未在本机验证。）**

在一台有 NVIDIA GPU（Hopper 或更新）的机器上：

1. 安装 TileLang（见第 7 章），把 `nv_hopper_sketch.py` 的 kernel 从
   `try/except` 中取出，以 $M=N=K=4096$、fp16 输入跑通。
2. 用 `torch.relu(a.float() @ b.float())` 做正确性对照（`rtol=1e-2, atol=1e-2`）。
3. 打印 TileLang 生成的 CUDA / PTX，回答：
   * 目标架构选择的是 `mma`、`wgmma` 还是 `tcgen05`？
   * `T.copy` 有没有降级成 TMA？
   * `num_stages=3` 生成了几份 shared buffer？
4. 明确记录你的验证级别是「在真实硬件上运行过」，并写清 GPU 型号、CUDA 版本、
   TileLang 版本。

---

## E14 动手题 ★★★ 需要硬件（NPU，可选）

**（可选，只有有昇腾 NPU 的读者需要做。本章写作环境没有 NPU，故本题未在本机验证。）**

在一台有 Atlas 350 / 昇腾 950 的机器上：

1. 安装 CANN + `torch_npu` + TileLang，确认 `torch.npu.is_available()` 为 `True`。
2. 跑通 `ascend_950_sketch.py` 的完整 Quick Start（$M=256, N=7168, K=2048$，
   bf16 输入、fp32 输出），与 `torch.relu(a.float() @ b.float().T)` 对照。
3. 用 `msProf`（见第 12、19 章）采集 trace，回答：
   * Cube 与 Vector 的占用在时间线上是否重叠？
   * 有没有出现某条 MTE 流水长时间空等？
   * 编译器插入了哪些 Set/Wait？（可与 `T.set_flag`/`T.wait_flag` 的手写版本对照。）
4. 用 `GetCoreMemSize` 记录该设备上 UB / L1 / L0C 的真实容量，回到 E10 的问题 4
   对照本章的「示例数字」。
