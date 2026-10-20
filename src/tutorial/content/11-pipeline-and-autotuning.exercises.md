# 第 11 章 习题

说明：

* 难度标注为 `★`（基础）/ `★★`（需要综合本章内容）/ `★★★`（需要推理链条较长或动手）。
* “需要硬件”一栏中，`否` 表示只需要一台能跑 Python 的机器；`是（GPU）`/`是（NPU）`
  表示需要用真实加速器才能完成实验。本章**没有强制要求硬件的题目**，所有必须上机的
  实验都被标成“可选”。
* 所有题目只依赖第 11 章及更早章节的知识；涉及后续章节的地方会显式提示。
* 本章写作环境没有 NVIDIA GPU，也没有昇腾 NPU。题面里出现的所有数字都来自
  `code/01`–`code/03` 的模型输出或 corpus 里的厂商资料，**不是实测值**；请照抄口径。

---

## E1 概念题 ★ 不需要硬件

1. 用一句话说明“不做流水”时每一轮循环的时间为什么是“搬运 + 计算”而不是
   “搬运与计算取较大者”。
2. `issue`、`latency`、`compute` 三个量分别描述什么？
3. 为什么“数据在路上的飞行时间”（`latency`）需要和“搬运占用引擎的时间”（`issue`）
   分开考虑？请从第 1.2 节引用的访存延迟与算术延迟的数量级差异上说明。

---

## E2 概念题 ★ 不需要硬件

1. 写出 stage、order、depth 三个词各自的定义，并说明它们分别由
   `T.Pipelined` 的哪个参数给出。
2. 只给 `num_stages=4` 时，深度是多少？只给 `stage=[0, 0, 3]` 时深度是多少？
3. `num_stages=0` 与 `num_stages=1` 在语义上分别是什么？为什么说它们是两个不同的东西？

---

## E3 概念题 ★★ 不需要硬件

本章 2.4 节区分了“延迟”与“吞吐”。

1. 用自己的话说明：流水优化的是哪一个？
2. 下面三种循环中，哪一种最值得加流水？说明理由。
   * (a) 一个只有 3 轮的 reduction 循环；
   * (b) 一个 512 轮的 GEMM reduction 循环（`K/block_K`）；
   * (c) 一个每轮计算量远小于搬运量的循环。
3. 为什么 (c) 那一类循环“加计算深度没用”？应该去优化什么？

---

## E4 概念题 ★★ 不需要硬件

本章 4.2 节说，`PipelinePlanning` 最后一步会“把整个 order 数组循环右移，并把非搬运
语句的 stage 减 1”。

1. 这一步在物理上对应什么？（提示：哪条语句先发生）
2. 如果不做这一步会发生什么？（提示：从 `max(stage) + 1` 反推深度）
3. 对“copy A、copy B、gemm”这个循环体、`num_stages = 3`，写出收尾整形**之后**每个
   语句的 `(stage, order)`，并说出深度。

---

## E5 概念题 ★★ 不需要硬件

本章 4.5 节给出的容量估算式是
`num_stages × (block_M×block_K + block_K×block_N) × sizeof(dtype)`。

1. 为什么是 `block_M×block_K + block_K×block_N`，而不是 `block_M×block_N×block_K`？
2. 为什么说这个数是**上界**？哪一类 pass 会让实际占用更小？
3. 为什么“只在同一个 stage 内被写、然后立即被读”的 buffer 不需要多版本？

---

## E6 代码阅读题 ★★ 不需要硬件

阅读这段 `InjectSoftwarePipeline` 里的代码
（`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:1488-1544`）：

```cpp
int num_versions = buffer_info.use - buffer_info.def + 1;
if (num_versions >= 2) {
  bool need_multi_version = false;
  for (...writer_block...) {
    for (...reader_block...) {
      if (writer_info.order < reader_info.order &&
          writer_info.stage < reader_info.stage &&
          MayConflict((*it1)->region, (*it2)->region)) {
        need_multi_version = true;
        break;
      }
    }
  }
  if (!need_multi_version) {
    num_versions--;
  }
}
return num_versions;
```

1. 这段代码在什么情况下会把版本数从 2 减回 1？
2. 为什么减版本是**安全**的？请从“写者与读者是否可能同时看到同一块内存”的角度解释。
3. `MayConflict` 检查的是访问区域是否重叠。如果两个赋值语句都只访问 buffer 的第一个
   元素（区域不重叠），版本数会变成多少？这样做有什么代价？

---

## E7 代码阅读题 ★★ 不需要硬件

下面是一段用户代码：

```python
for ko in T.Pipelined(num_tiles, stage=[0, 1], order=[1, 0]):
    base: T.int32 = ko * BK
    T.copy(A[base + tx], A_shared[tx])
    B[base + tx] = A_shared[tx]
```

1. `stage` 和 `order` 各有多少项？为什么是这个数量？
2. 如果用户写成 `stage=[0, 0, 1]`、`order=[1, 2, 0]`（把 `base` 也算进去），
   编译器会怎么处理？
3. 什么情况下 `base` 这种 `Bind` **不能**被当成可重放的？给出本章提到的两种情形。

---

## E8 代码阅读题 ★★ 不需要硬件

阅读 `_vendor/tilelang-v0.1.15/src/transform/common/pipeline_utils.h:74-110`：

```cpp
inline ffi::Optional<Integer> GetPipelineNumStages(const ForNode *loop) {
  if (auto num_stages = loop->annotations.Get("num_stages")) { ... }
  if (auto num_stages = loop->annotations.Get("tl_pipelined_num_stages")) { ... }
  if (auto stages_anno = loop->annotations.Get(s_tir::attr::software_pipeline_stage)) {
    auto stages = Downcast<ffi::Array<Integer>>(stages_anno.value());
    int max_stage = -1;
    for (const auto &stage : stages) max_stage = std::max(max_stage, (int)stage->value());
    if (max_stage >= 0) return Integer(max_stage + 1);
  }
  return ffi::Optional<Integer>();
}
```

1. 这个函数按什么顺序读取深度？为什么要按这个顺序？
2. `PipelinePlanning` 为什么要在写入 `tl_pipelined_num_stages` 的同时**删掉**原来的
   `num_stages` 键？（提示：看 `pipeline_planning.cc:1287-1293` 的注释）
3. 如果循环上三个键一个都没有，函数返回什么？下游会把这个循环当成流水循环吗？

---

## E9 代码阅读题 ★★ 不需要硬件

下面是从 `code/03-autotune-config-space/config_space.py` 抄出来的判定函数的一部分：

```python
if cfg.threads % 32 != 0 or cfg.threads > problem.max_threads:
    return "illegal thread count"
if cfg.block_m % 16 or cfg.block_n % 16 or cfg.block_k % 16:
    return "tile not a multiple of 16"
if problem.m % cfg.block_m or problem.n % cfg.block_n or problem.k % cfg.block_k:
    return "tile does not divide the problem"
if smem_bytes(cfg, problem.dtype) > problem.smem_budget:
    return "shared memory over budget"
return None
```

1. 四条规则里，哪一条对应本章的“硬约束”？哪一条对应的是**正确性**而不是性能？
2. 为什么“tile 不整除问题尺寸”在第 11 章被当成不可行，而不是留到第 14 章去处理
   边界？（提示：想想流水对迭代轮数的假设）
3. 如果把这四条规则全删掉，直接把 216 个候选交给调优器，会发生什么？请给出至少两条
   具体后果。

---

## E10 动手题 ★★ 不需要硬件

用 `code/01-pipeline-timeline/pipeline_timeline.py` 做实验，回答以下问题，并给出你实际
跑出来的数字：

1. 固定 `issue=1, compute=20`，把 `latency` 从 10 调到 120（例如
   `--latency 10 20 40 80 120`）。对每个 `latency` 记录“达到稳态所需的最小深度”，
   并和公式 `1 + ceil(latency / compute)` 对照。
2. 固定 `latency=30`，把 `compute` 从 5 调到 40。什么时候 `stages=2` 就够了？
3. 用你跑出来的数据解释：为什么“stage 数该开多大”不能是一个常量。

（本题只需要 Python；不需要 GPU。把命令与输出粘进你的笔记。）

---

## E11 动手题 ★★★ 不需要硬件

1. 用 `code/02-shared-memory-budget/smem_budget.py` 回答：在 228 KiB 的预算下，
   `block_M = block_N = 128`、`block_K = 64`、fp16 时，最多能开多少 `num_stages`？
   换成 `block_K = 16` 呢？两个答案差几倍？
2. 用同一份脚本的 `--sweep` 输出，找出“最大可行 stage 数 = 3”的所有
   `(block_M, block_N, block_K)` 组合（在脚本给的那张表里）。
3. 基于 1、2 的结论，写一条**可以放进代码评审清单**的规则：当有人想把
   `num_stages` 从 3 改成 4 时，应该先做什么检查？

---

## E12 动手题 ★★★ 不需要硬件（可选上机）

设计一个 GEMM 的自动调优搜索空间，并论证它值得跑。

1. 写出候选的维度与取值：`block_M`、`block_N`、`block_K`、`num_stages`、`threads`。
2. 说明你用什么规则做预剪枝，并算出剪枝前后各有多少个候选。
   （可以用 `code/03` 的 `check()` 逻辑手算，也可以直接改那个脚本跑。）
3. 说明为什么你的空间**不是**“全部不可行”“只剩一个可行”“参数没被用上”这三种
   坏空间中的任何一种。
4. （可选，需要 GPU）在真实机器上跑一遍 `code/05-tilelang-autotune-gemm/`，
   记录最优 `config`，并说明你会把它固定到哪里、为什么不是固定 `latency`。
