# 第 22 章 习题

说明：

* 难度标注为 `★`（基础）/ `★★`（需要综合本章内容）/ `★★★`（推理链条较长或动手）。
* “需要硬件”一栏中，`否` 表示只需要一台能跑 Python 的机器；`TileLang` 表示需要装
  TileLang v0.1.15（**不需要 GPU**）；`是（GPU）` 表示需要真实加速器。
* 所有题目只依赖第 22 章及更早章节的知识；涉及后续章节的地方会显式提示。
* 本章写作环境没有 NVIDIA GPU，也没有昇腾 NPU。题面里出现的所有数字都来自
  `code/02`–`code/04` 的模型输出或 corpus / 上游单测里的原文，**不是实测值**；
  请照抄口径。

---

## E1 概念题 ★ 不需要硬件

1. 给出“依赖距离”的定义；`d = 0` 与 `d ≥ 1` 分别对应什么样的程序行为？
2. `_vendor/tilelang-v0.1.15/src/ascend/transform/auto_schedule/dependency_analysis.h`
   用 `distance = -1` 表示“未解析的跨迭代依赖”。结合源码注释说明：这个 `-1`
   最终可能被翻译成哪两种不同的东西？
3. 为什么说“把 copy 全挪到循环前面”不是流水的合法实现方式？至少给出两个理由。

---

## E2 概念题 ★ 不需要硬件

`ValidatePipelineBody`（`_vendor/tilelang-v0.1.15/src/transform/inject_pipeline.cc:3097-3138`）
对用户的 `stage`/`order` 注解有两条硬校验。

1. 抄出这两条校验，并各用一句中文说明它在防什么。
2. 下面这组注解违反了哪一条？

   ```python
   for ko in T.Pipelined(n, stage=[1, 0], order=[0, 1]):
       T.copy(A[ko * BK], A_shared)          # 生产者
       T.gemm(A_shared, B_shared, C_local)   # 消费者
   ```

3. 如果两条注解都合法但 `order` 有重复，会抛什么错误？报错文案的关键词是什么？

---

## E3 概念题 ★★ 不需要硬件

下表是某个循环体的 `(stage, order)` 与访问：

| 语句 | stage | order | 读 | 写 |
| --- | --- | --- | --- | --- |
| `copy_a` | 0 | 0 | `A`（global） | `A_shared` |
| `copy_b` | 0 | 1 | `B`（global） | `B_shared` |
| `gemm` | 2 | 2 | `A_shared`, `B_shared` | `C_local` |
| `scale` | 2 | 3 | `C_local` | `C_local` |

1. 用 §2.4 的公式算出 `A_shared` / `B_shared` / `C_local` 各自的版本数。
2. 这个循环的 `depth`（SW 流水级数）是多少？
3. 如果 `num_stages` 注解写的是 4，而 `max(stage)+1 = 3`，`InjectSoftwarePipeline`
   在为 **barrier** 定尺寸时用哪一个？为什么？（提示：源码里有一段专门解释这件事的
   注释。）

---

## E4 代码阅读题 ★★ 不需要硬件

下面是 `ComputeBufferVersions` 的骨架（`src/transform/inject_pipeline.cc:1488-1542`，
已删去循环体细节）：

```cpp
int num_versions = buffer_info.use - buffer_info.def + 1;
if (num_versions >= 2) {
  bool need_multi_version = false;
  for (writer in blocks) {
    for (reader in blocks) {
      if (writer.order < reader.order &&
          writer.stage < reader.stage &&
          MayConflict(writer.region, reader.region)) {
        need_multi_version = true;
        break;
      }
    }
  }
  if (!need_multi_version) num_versions--;
}
```

1. 这段收紧逻辑只在 `use − def + 1 == 2` 时生效吗？为什么？
2. 给一个 `use − def + 1 == 2` 但 `need_multi_version == false` 的具体例子
   （用 stage/order 描述即可），并说明这时程序为什么不需要双缓冲。
3. 如果把条件里的 `MayConflict` 换成恒真，会发生什么？（从正确性与性能两个角度答。）

---

## E5 代码阅读题 ★★ 不需要硬件

`ExpandPipelineBarriers`（`src/transform/inject_pipeline.cc:785-886`）里有三行：

```cpp
PrimExpr ns = IntImm(DataType::Int(32), num_stages);
PrimExpr stage_expr   = FloorMod(loop_var - loop_min, ns);
PrimExpr parity_cycle = FloorMod(FloorDiv(loop_var - loop_min, ns), 2);
...
new_node->shape = {PrimExpr(num_stages) * buf->shape[0]};
```

1. `stage_expr` 与 `parity_cycle` 分别是什么？用 `num_stages = 3`、
   `k = 0..7` 列一张表。
2. 为什么等待 mbarrier 时必须带 `parity`，只给“槽位号”不够？
3. 为什么注释特别说“只有显式带 `ptx_arrive_barrier` 的 barrier，或 ISP 自己创建的
   `pipeline_mbar`”才展开？如果连 tcgen05 MMA 内部管理的 arrive barrier 也一起
   展开，会出什么问题？（这一问只需给出合理推断。）

---

## E6 概念题 ★★ 不需要硬件

1. 写出 `GetPipelineNumStages`（`src/transform/common/pipeline_utils.h:87-110`）
   读取深度的三个来源与顺序。
2. `PipelinePlanning` 为什么要把 `num_stages` 删掉、只留
   `tl_pipelined_num_stages`？原文怎么说？
3. 下面哪一句是对的，为什么？

   * (a) 流水深度等于 `max(stage) + 1`，永远如此；
   * (b) 流水深度等于 `num_stages`，永远如此；
   * (c) 取决于注解的来源，`InjectSoftwarePipeline` 在为 barrier 定尺寸时优先用
     `num_stages`，缺失时才退回 `max(stage) + 1`。

---

## E7 代码阅读题 ★★★ 不需要硬件

`MergeSharedMemoryAllocations` 的 arena 打包规则是：区间按 `(start, -size, name)`
排序；最优先分配、同起点大者先分配；分配时选“对齐后浪费最少”的空洞（best fit）；
归还时与相邻空洞合并。

给定（对齐 0，容量无限，分配顺序按 `(start, -size, name)`）：

| buffer | size | start | end |
| --- | --- | --- | --- |
| `P` | 32 | 0 | 10 |
| `Q` | 16 | 2 | 6 |
| `R` | 16 | 3 | 10 |
| `S` | 24 | 6 | 9 |
| `T` | 8 | 10 | 14 |

1. 手算每个 buffer 的起始偏移与最终 arena 大小。
2. 如果把 `P` 的活跃区间延长成 `[0, 12)`（即 `P` 活到 `T` 开始之后），
   `T` 会被放到哪里？arena 大小会变吗？
3. 现在把对齐从 0 改成 32（其它不变），重新算一遍。arena 大了多少？
   多出来的空间花在哪里？
4. 用一句话解释：为什么“对齐要求”在这个模型里等价于“凭空多出一段不能用的碎片”。

---

## E8 动手题 ★★ 不需要硬件

```bash
cd code/02-buffer-version-model
python3 buffer_versions.py --scenario epilogue
python3 buffer_versions.py --scenario carry
python3 buffer_versions.py --scenario plain
```

1. 三个场景里，`num_stages=4` 时的 arena 大小分别是多少？解释差异的来源。
2. 在 `carry` 场景里，从哪一档开始 `[OVER]`？把 `--budget-kib` 调到多少能让它
   在 `num_stages=4` 就 `[OVER]`？
3. 解释为什么 `--scenario epilogue` 里 `C_shared` 的 offset 总是 0。
4. （延伸）把 `iterations` 调到 4 再跑一遍，哪些数字变了、哪些没变？为什么？

---

## E9 动手题 ★★ 不需要硬件

```bash
cd code/04-ascend-flag-budget
python3 flag_budget.py
python3 flag_budget.py --sharing
python3 flag_budget.py --cross-core
```

1. 在默认（intra-core，不允许复用）下，第一次出现 `spilled` 的是哪一档？
   溢出的是哪些同步点？
2. `--sharing` 在第 2 档把 `allocator sum` 从 8 降到 6：这 2 个槽是怎么省出来的？
   真 pass 里对应的机制叫什么？
3. 为什么 `--cross-core` 在第 4 档仍然放得下？
4. 请设计一个“同步点数变多、但 flag 需求不变”的场景，并说明它在真实 kernel 里
   对应什么写法。

---

## E10 概念题 ★★ 不需要硬件

本章 §5.1 说“漏一个同步会偶发出错”。

1. 用“WAR hazard + 环上槽位轮转”描述一个会**偶发**产生错误结果的最小场景。
2. 为什么“重复跑 1000 次都过”不能作为同步正确的证据？请给出一个能反证的
   检验方案（不需要真跑）。
3. 上游文档把“条件执行”单独列成一节。请说明：一个带 `if active(i)` 的循环，
   在 `iteration` 与 `counter` 两种索引模式下，生产者与消费者可能算出不同槽位的
   原因是什么。

---

## E11 动手题 ★★ 不需要硬件

```bash
cd code/03-sync-count-vs-perf
python3 sync_count_model.py
python3 sync_count_model.py --latency 120 --compute 20 --barrier 8
python3 sync_count_model.py --latency 30 --compute 60 --barrier 8
```

1. 默认参数下，为什么 `war` 在 `depth ≥ 3` 时与 `none` 的 makespan 完全相同？
   给出一个用 `latency`、`compute`、`barrier` 表示的条件。
2. 第二次运行时 `sync-copy` 的稳态步长是多少？它由哪些量决定？
3. 第三次运行（`compute = 60`）时，`+cb` 与 `war+raw` 的差距会变大还是变小？
   先预测再跑，并解释。

---

## E12 代码阅读题 ★★★ 不需要硬件

昇腾侧 `SyncMaterializer::InsertSyncPoints`
（`src/ascend/transform/insert_sync.cc:2359-2470`）按三条规则落地同步：

```cpp
if (!IsCrossCore(sp) && producer_pipe == consumer_pipe) {
  if (producer_pipe == "PIPE_S" || producer_pipe == "PIPE_V" ||
      producer_pipe == "PIPE_M") continue;          // 规则 1
  InsertStatement(..., MakeAscendPipeBarrier(producer_pipe), ...);
  continue;
}
if (IsCrossCore(sp)) { ...CrossCoreSetFlag / WaitFlag... }   // 规则 2
... SetFlag / WaitFlag ...                                    // 规则 3
```

1. 规则 1 里的 `continue` 为什么是安全的？它依赖 `HasImplicitSamePipeOrder` 的什么
   假设？请给出该函数的三行源码。
2. 为什么 MTE 系列（`PIPE_MTE1/2/3`）**不能**享受同样的豁免？源码注释怎么说？
3. `GetHardEvent` 把 `MTE2_M` 规范化成 `MTE2_MTE1`。解释这一步在做什么，
   以及它为什么必要。

---

## E13 概念题 ★★★ 不需要硬件

本章 §8.3 说“WS 与流水是两套都会插同步的机制”。

1. 在 v0.1.15 的 CUDA pass 顺序里，WS 的三趟 pass 与
   `PipelinePlanning` / `InjectSoftwarePipeline` 谁先谁后？给出文件与行号。
2. `InjectSoftwarePipeline` 的 TMA barrier 管理在什么条件下**不**接管？
3. `PipelinePlanning` 为什么坚持把 `num_stages` 换成 `tl_pipelined_num_stages`？
4. 如果你在一个已经做了 WS 的 kernel 上又手写 `num_stages` 注解，最可能出现哪两种
   症状？分别对应本章 §5 的哪一条结论？

---

## E14 动手题 ★ TileLang（不需要 GPU）

在有 TileLang v0.1.15 的机器上（**不需要 GPU**）：

```bash
cd code/01-num-stages-ir-diff
python3 lower_num_stages.py --num-stages 1 2 3 4 --dump-dir /tmp/tl22
diff -u /tmp/tl22/gemm_num_stages_2.cu /tmp/tl22/gemm_num_stages_3.cu
TL_LOWER_TRACE=terminal python3 lower_num_stages.py --num-stages 3
```

1. 把四次运行的 `software_pipeline_stage` 抄下来，和 §7.3 的预期表对照；
   有哪一行对不上？
2. 在 diff 里找出 A/B 的 `__shared__` 首维。它和你在第 1 问里算出的版本数一致吗？
3. 统计四次运行里 `cp.async.wait_group` 的参数取值。它们是单调的吗？
4. 用 `TL_LOWER_TRACE=terminal` 的输出，指出 `InjectSoftwarePipeline` 那一步
   具体改了哪几行（贴出 diff 片段）。

> 注意：这一题**不需要** NVIDIA GPU；但要装 TileLang。如果你和作者一样在没有
> TileLang 的机器上，请改为“阅读 `code/01/README.md` 与上游单测
> `test_pipeline_planning_before_after_copy_gemm_num_stages_plan`，写出你预期的输出”
> 并明确指出哪几项你**没有**验证过。
