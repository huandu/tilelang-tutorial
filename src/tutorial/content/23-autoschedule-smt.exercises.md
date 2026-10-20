# 第 23 章 习题

难度标记：`★` 基础 / `★★` 需要动手或读源码 / `★★★` 需要设计。
硬件标记：`[无卡]` 本机可完成；`[需 NPU]` 必须有昇腾机器（或 camodel 仿真环境）。

---

## 概念题

**E1（★，[无卡]）** 用你自己的话说明：为什么 Ascend C 里“Set/Wait 配对”
“Cube/Vector 生产消费”“跨迭代复用同一块 UB”这三件事可以合起来写成一个调度问题？
请明确指出哪一部分是**约束**、哪一部分是**目标**。

**E2（★，[无卡]）** `src/ascend/transform/ascend_pipe.h` 把硬件流水定义成 7 位掩码。
回答：

1. 写出这 7 个名字与对应位值；
2. 为什么 `MTE2` 同时负责 `GM→UB` 和 `GM→L1` 却只有一位？
3. `kAll` 为什么不能出现在任务的 `pipe_mask` 里？

**E3（★，[无卡]）** 解释 `latency` 与 `ii` 的区别。分别给出“把两者混为一谈”的
两种错误方向，以及每种方向造成的后果。

**E4（★★，[无卡]）** 在 `z3_scheduler.py` 的循环模型里，`data_deps` 的
`distance` 取负值时代表什么？为什么这个设计使得“buffer 开几份”不能作为调度之后的
独立步骤来处理？请用一句话概括，并说明 `distance` 取 `0`、`-1`、`≥1` 三种情况的区别。

## 代码阅读题

**E5（★★，[无卡]）** 阅读 `_vendor/tilelang-v0.1.15/src/ascend/transform/insert_sync.cc`
里的 `GetHardEvent`（约 `:751`）。回答：

1. `producer_pipe = "MTE2"`、`consumer_pipe = "M"` 时它返回什么？
2. 为什么要把 `M` 映射成 `MTE1`，而不是直接产出 `MTE2_M`？
3. 顺着 `MakeAscendSetFlag`/`MakeAscendWaitFlag`（约 `:2270`）说明这对结果最终变成
   什么形式的 IR。

**E6（★★，[无卡]）** 在 `_vendor/tilelang-v0.1.15/tilelang/ascend/transform/z3_scheduler.py`
的 `z3_schedule_loop_python` 里找到“自动 buffer 决定跨迭代距离”的那几行，给出
行号与表达式，并解释 `buffer_var_id = -distance - 1` 这一行的作用。

**E7（★★，[无卡]）** `enable_offset` 默认是 `False`。请说明这个默认值具体加了什么
约束（写出不等式），并回答：为什么手写非零 `T.Stage` 时必须把它打开？如果关不掉又
硬写非零 stage，代码会怎么失败？

**E8（★★，[无卡]）** 阅读 `_vendor/tilelang-v0.1.15/src/ascend/transform/auto_schedule.cc`
中 `MergeSelectedBufferVersion`（`:245`）及函数内的注释（`:251-256`）。回答：

1. 两个 owner 共用一段物理 ring 时，合并后的版本数取什么？
2. 注释里承认了什么限制？这个限制可能导致什么后果？由谁来兜底？

## 动手题

**E9（★★，[无卡]）** 运行本章的 `code/pass_order/ascend_pass_order.py`。

1. 贴出输出，数出 `if auto_schedule_enabled:` 块里有几个 pass；
2. 找出其中唯一真正调用求解器的那个，说明它前后两个 pass 各自的职责；
3. 假设把 `AnnotateMultiBufferEligible` 与 `MaterializeScheduleUnits` 的顺序对调，
   依据这两个 pass 的 docstring 推断会发生什么，并说明你的判断依据。

**E10（★★，[无卡]）** 修改 `code/toy_scheduler/toy_schedule.py` 里玩具实例的
`memory_groups` 容量，从 `4096` 改成 `1500`，然后运行 `demo`。

1. 先不运行，用 §4.3 的手推方法预测 `II` 与 `buffer versions`；
2. 再运行验证；
3. 解释为什么求解器不能“先挑大 II、再多开版本”来救这个实例。

**E11（★★★，[无卡]）** 玩具模型目前没有 owner-exclusion。请为
“两个不相关 owner A、B 共用同一段物理 ring”补上这个约束：

1. 写出 `A before B` 时要加的两条不等式（正向交接 + 绕回交接），说明每一条保护的是
   哪一对访问；
2. 解释为什么它必须写成**二选一**（用布尔变量），而不是一个固定的顺序；
3. 说明为什么 z3_scheduler 的注释强调“pipeline stage offsets 故意不参与”。

**E12（★★★，[需 NPU]）** 设计一个实验，用来在真实昇腾机器上判定
“自动求解失败 → 静默退化成串行”是否发生（§5.2 形态 (3)）。要求：

1. 给出自变量（你要改什么）与因变量（你测什么）；
2. 给出至少两条互相独立的证据（一条来自生成代码/IR，一条来自性能）；
3. 说明如果只是“性能持平”还能用什么手段排除“容量约束把版本数压回 1”这个干扰项。
