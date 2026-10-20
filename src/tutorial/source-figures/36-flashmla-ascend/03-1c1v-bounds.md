# 图 3：1C1V 的三条 bound

**用途**：正文 §3.3。

| | 1C1V（报告否掉） | 1C2V（最终采用） |
| --- | --- | --- |
| **MTE2 队列深度** | 单 VECTOR Core 未完成请求上限 16；gather2 后一轮最多 32 个 token，装不下 `B_TOPK = 64` | 每核只搬 `B_TOPK_PER_V = 32` 个 token |
| **反量化吞吐** | 微基准显示反量化赶不上 Cube 计算 → 反量化成为瓶颈 | 两个 VECTOR Core 分摊反量化 |
| **FixPipe 带宽** | 128 B/cycle；`B_TOPK·16 > 1024 + B_TOPK·2` → `B_TOPK > 73.14` | 每个 AIV 的 `MMA_M_PER_V = 32`、`B_TOPK_PER_V = 32` |
| **结论** | 三条 bound 互相加重，不可行 | 分摊 MTE2 发射、反量化、FixPipe 搬运、softmax |

## 读图要点

1. 三条 bound 是**互相加重**的：为了满足 FixPipe 要把 `B_TOPK` 抬到 96 或 128，
   但更大的 `B_TOPK` 让 UB 更紧张、也让 MTE2 队列更塞。
2. `B_TOPK > 73.14` 是 **1C1V 的条件**。最终代码用 1C2V + `B_TOPK = 64`
   （`config.h:53-55`），两者不矛盾，报告正是用这条不等式否掉 1C1V。
3. 另一条独立 bound 是 MTE2 的队列深度（16 个 entry），它是**指令级**的硬上限，
   与带宽无关。

## 来源

* 报告 `:132-144`（1C1V 的三条 bound 与结论）。
* `config.h:53-55`（`NUM_VEC_CORE_PER_AI_CORE = 2`、`B_TOPK = 64`）。
