# 第 23 章 习题答案

每题给「思路 → 参考答案 → 常见错误」。动手题给关键片段，不要求你写出完整实现。
所有行号基于 `_vendor/tilelang-v0.1.15/`（commit 见 `VERSIONS.md`）。

---

### E1

**思路**：区分「什么必须成立」（约束）与「什么尽量小/少」（目标）。

**参考答案**：三件事都能写成不等式。Set/Wait 配对本质是「消费者不能在生产者完成前
开始」，即 `start_consumer >= start_producer + latency`——这是**约束**；Cube/Vector 的
生产消费是同一类有向约束，只是生产者在 AIC、消费者在 AIV；跨迭代复用则是
「下一轮的写不能早于上一轮的读」，是带 `distance` 的约束。**目标**是让流水尽量紧凑，
在 TileLang 里就是最小化 II（相邻两轮迭代的启动间隔，`z3_scheduler.py:398-540`）。
至于「开几份 buffer」，它既是约束（`sum(buf*size) <= capacity`）又是决定跨迭代距离的
变量，所以不能事后决定。

**常见错误**：把「Set/Wait 配对」当成语法问题（其实是时序约束）；把目标写成
「最小化总执行时间」而不落到 II 上（循环里 II 才是可控目标，makespan 由 II 乘轮数决定，
见 `auto_schedule.cc:1272-1281`）。

### E2

**参考答案**：

1. `1<<0 MTE1`（L1→L0A/L0B）、`1<<1 MTE2`（GM→UB / GM→L1）、`1<<2 MTE3`（UB→GM / UB→L1）、
   `1<<3 Cube`（MAD）、`1<<4 Vector`、`1<<5 Fixpipe`（L0C→UB/GM）、`1<<6 Scalar`
   （`src/ascend/transform/ascend_pipe.h:52-62`）。
2. 因为**发射端口是同一条**：`GM→UB` 与 `GM→L1` 由同一个 MTE2 单元按序发射，
   建模时只需要知道「它们互相争用」，不需要区分目的地。`GetResourcePipeName`
   也证实掩码只用于标识端口。
3. `kAll` 是全 1 掩码，注释写明 `kAll is used only for a full-core PIPE_ALL fence`。
   任务必须落在**具体**某条流水上，否则依赖/资源分析无法判断它和别的任务是否冲突；
   `GetResourcePipeName` 对「既不是 0 也不是单 bit」的掩码直接 ICHECK 报错。

**常见错误**：以为 `MTE2` 位是「GM→UB」专用；把 `kAll` 当成「不限流水」的合法标签。

### E3

**参考答案**：`latency` 是任务发射到**输出可用**的周期数，约束的是消费者；
`ii` 是同一条流水**下一次发射**的最小间隔，约束的是生产者自己。两者都在
`TaskNode`（`task_annotations.h:41-42`）上，且必须满足 `latency >= ii`。

* 把 latency 当成 ii（都取小）：消费者以为数据很快就好，但同一条流水的下一次发射
  被低估 → 资源约束放松 → 计划排得过密，消费者**提前启动**，读到未完成的数据 → **错**。
* 把 ii 当成 latency（都取大）：消费者被要求等更久，本来能重叠的搬运算被串行化 →
  **慢**。
  这正是原文说的「两个概念一旦混起来，编译器就会把本来能够充分重叠的机器重新排成
  串行，反过来只盯着 issue interval，又会让 consumer 在数据还没真正 ready 的时候
  提前启动」。

**常见错误**：只写「一个快一个慢」，没有把两种错误方向与后果对应起来。

### E4

**参考答案**：`distance < 0` 表示「这里有一条跨迭代依赖，但跨几轮还没定，交给求解器
用 ring 宽度（buffer 份数）决定」。`distance = 0` 是同迭代依赖；`-1` 是未定版的跨迭代
依赖；`≥1` 是用户手工多缓冲、已经写死的跨迭代距离（`dependency_analysis.h:62-73`）。

关键在于约束形式：`start[v] - start[u] >= latency - II * buf[-distance-1]`
（`z3_scheduler.py:449-455`）。**buf 出现在约束右边**，也就是说「开几份」直接决定了
这条依赖允许的重叠量；反过来 II 又决定了最小可用份数。两者互相牵制，所以必须同一个
问题里解，不能先解 II 再补 buffer。

**常见错误**：以为多缓冲只是「把 buffer 变大」，与时间无关。

### E5

**参考答案**：

1. 返回 `"MTE2_MTE1"`：`GetHardEvent` 里有一条特判 `if (event == "MTE2_M") return "MTE2_MTE1";`。
2. 因为 Cube 的输入来自 L0A/L0B，而 L0 是由 MTE1 装填的。硬件的 `HardEvent` 是
   **端口到端口**的事件，`M` 不是这条路径上的发射端；写成 `MTE2_M` 会得到一个硬件不
   存在的同步事件（无法表达「等数据真正进 L0」）。
3. 结果是 `Evaluate(Call(ascend_set_flag(), {StringImm("MTE2_MTE1"), event_id}))`
   与对应的 `ascend_wait_flag`（`insert_sync.cc:2270-2280`）；codegen 再把它落成
   `AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(eventID)` /
   `WaitFlag<...>`（见 `tilelang/ascend/language/sync.py` 的 `ascend_set_flag` 文档）。

**常见错误**：把它当成字符串拼接的细节，而忽略「HardEvent 必须对应硬件真实端口对」。

### E6

**参考答案**：位置在 `z3_scheduler.py:449-455`：

```python
else:
    buffer_var_id = -distance - 1  # Convert to 0-based index
    ...
    solver.add(start_vars[v] - start_vars[u] >= latency - ii_mid * buffer_vars[buffer_var_id])
```

`distance = -1` 对应 `buf` 数组下标 0，`-2` 对应下标 1，依此类推；也就是说
「跨迭代距离」这个语义被复用成了「某个自动 buffer 的版本数」的编号。
`buffer_var_id` 就是把 1-based 的编码（`-1`、`-2`）翻译成 0-based 数组下标。

**常见错误**：误以为 `buffer_var_id` 是 buffer 在 IR 里的编号——它是**本循环内自动
多缓冲 buffer 列表**的下标，由 C++ 侧 `storage_sizes` 的构造顺序决定。

### E7

**参考答案**：默认 `enable_offset = False` 时加入的约束是（`z3_scheduler.py:502-514`）：
对每个 `pipe_bit`，把使用该流水的任务分组，组内任意两个任务 a、b 都要求
`start[a] - start[b] < II` 且 `start[b] - start[a] < II`，等价于
`max(start) - min(start) < II`——即**一条流水上的任务必须落在同一个 II 窗口内**，
不允许把某条流水的任务摊到多个流水级上。

手写非零 `T.Stage` 时，用户明确要求某些任务处在不同阶段，这与「同一 II 窗口」矛盾，
所以必须 `enable_offset=True`。如果不打开而硬写非零 stage，会在
`z3_scheduler.py:369-370` 直接抛
`ValueError("Non-zero manual stages require enable_offset=True")`（手动模式），
或约束自相矛盾导致求解失败。

**常见错误**：以为 `enable_offset` 与 `T.Stage` 无关；实际上它们是绑定关系，
`T.Stage` 的 docstring 也写了 ``T.Pipelined(..., num_stages=...)`` 不限制 `T.Stage`
的取值。

### E8

**参考答案**：

1. 取 `max`：`num_versions = std::max((*existing).second, num_versions);`
   （`auto_schedule.cc:257`）。
2. 注释承认：合并之后**不再复核每个 owner 的 per-owner 内存上限**，理由是那些上限只是
   粗略的调度启发式、不建模下游的存储复用，所以全局求和并不权威。后果是**可能产出一个
   超过物理片上容量的合并分配**；兜底者是下游或运行时校验
   （注释原文：`such a kernel may only be rejected by downstream/runtime validation`）。

**常见错误**：以为求解器给出的版本数一定满足容量——它只在**求解时刻**满足模型里的
容量，合并与后续复用变换之后就未必了。

### E9

**参考答案**：

1. 13 个（`NormalizeControlFlowForSchedule` … `RestoreWhileLoops`），脚本会逐条打印
   实现文件与注册行号，最后输出
   `ok: 顺序一致，且每个 pass 都在 src/ascend/transform 里有 FFI 注册`。
2. 唯一调用求解器的是 `AutoSchedule`（`auto_schedule.cc:1400`）；前一个
   `EstimateLatency`（`estimate_latency.cc:2271`）负责给每个任务算出 latency/II，
   后一个 `AssignCore`（`assign_core.cc:401`）负责给任务标核。
3. 依据两个 pass 的 docstring：`AnnotateMultiBufferEligible` 明确要求先跑
   `NormalizeControlFlowForSchedule` 与 `MaterializeScheduleUnits`，因为「buffer 相关的
   循环边界必须是可调度的任务」，它才能判断写先读后。对调之后，多缓冲资格会在
   **尚未规范化**的语句上推断，可能漏判或误判 owner，从而给出错误的多缓冲集合
   （表现为依赖过保守 → 慢，或漏掉对旧数据的读 → 错）。

**常见错误**：把 pass 顺序当成实现细节；对调顺序后仍认为「只是一个优化开关」。

### E10

**参考答案**：

1. 预测：容量 1500 只能放一份 1000 字节的 L1 buffer，所以 `buf[0] = 1`，
   于是 `II >= 55`；求解器会给出 `II = 55`、`buffer versions = [1]`。
2. 实测（把 `memory_groups=[[4096, 0]]` 改成 `[[1500, 0]]` 后运行 `demo`）：
   `first feasible II = 55, buffer versions = [1]`。
3. 因为「先挑大 II」并不能让容量变大。容量约束里 `buf` 的上界由 `num_stages` 与
   `capacity // size` 共同决定；1500 < 2 × 1000，所以 `buf = 1` 是唯一可行取值，
   **与 II 无关**。求解器在 II 增大时也不会放松容量约束（`z3_scheduler.py:431-436`）。

**常见错误**：以为「性能换空间」在模型里是可逆的——它可以换（加大 II 来减少版本需求），
但不能凭空造出容量。

### E11

**参考答案**：设两个 owner 为 i、j，共享同一段物理 ring，交接延迟为
`latency_i_j`（i 先）与 `latency_j_i`（j 先）。

1. 二选一布尔 `O`（i before j）后，`O` 为真时加两条：

   ```text
   r[j] - r[i] >= latency_i_j                 # 正向交接：同一轮内 i 把 ring 交给 j
   r[i] - r[j] + II >= latency_j_i            # 绕回交接：下一轮 i 要等本轮 j 读完
   ```

   第二条就是 `z3_scheduler.py:479-483` 里的
   `solver.add(z3.Implies(i_before_j, r_vars[i] - r_vars[j] + ii_mid >= latency_j_i))`。
   对称地，`¬O` 时把 i、j 互换。
2. 必须二选一，是因为两个 owner **在语义上无先后**：任意一方先发都可以，物理上只需要
   保证它们的存储生命周期不重叠。写死一个顺序会丢掉一半可行解，甚至把本可实现的
   流水判成不可行。
3. 注释原文说明：owner 之间交换的是**同一物理 slot**，交接发生在各自的 issue 相位上，
   而不是逻辑流水级上。如果把 pipeline stage offset 也算进去，会把「谁先拿到 slot」
   和「谁在第几轮」混为一谈，约束就错了。

**常见错误**：只加正向约束、漏掉绕回那一条——`r` 是模 II 的，环上必然有两个交接点。

### E12

**参考答案**：

1. **自变量**：把 `example_gemm.py` 的 `NUM_STAGES` 从 1 扫到 4；
   可选再加一个「人为制造求解压力」的变量（更多任务、更稠密的跨迭代依赖）。
   **因变量**：单次调用的 msProf 时间（`do_bench(..., backend="msprof")`）与
   `AutoSchedule` 之后 IR 里的 buffer 版本维度 / Set-Wait 数量。
2. **两条独立证据**：
   - 代码侧：dump `AutoSchedule` 前后的 IR（`tl.enable_dump_ir` / `tl.dump_ir_path`），
     看 buffer 第一维是否始终为 1、Set/Wait 是否退化到「每对任务只有一条」；
   - 性能侧：实测时间与 `II = sum(latencies)` 对应的串行模型的预测值吻合
     （即 `num_stages` 从 1 加到 4 完全不改变时间）。
3. **排除容量干扰**：把 `T.annotate_unlimit_memory("shared")`（以及 `"shared.l1"`）
   打开再测一次。如果解除容量上限后版本数 > 1、时间也改善了，说明原来的瓶颈是容量；
   如果解除后版本数仍然是 1，那就不是容量问题，而是求解退化或依赖过保守
   （可继续用 `T.assume_no_conflict` 做对照）。

**常见错误**：只看性能、不看 IR；或只用一次测量就下结论（msProf 结果需要多次重复与
固定 shape 才能比较）。
