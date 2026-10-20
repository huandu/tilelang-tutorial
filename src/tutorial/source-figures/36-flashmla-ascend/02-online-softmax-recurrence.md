# 图 2：online softmax 的递推（含 skip-scale 分支）

**用途**：正文 §2.2–§2.3。

```mermaid
flowchart TD
    S0[读入第 k 个 KV 块] --> S1[P = Q · gathered_kvᵀ]
    S1 --> S2[cur_max = max(P, dim=-1)]
    S2 --> S3[running_max = max(running_max, cur_max)]
    S3 --> S4{running_max - running_max_for_softmax<br/>&gt; RESCALE_THRES ?}
    S4 -- 是 --> R1[out_accum *= exp(m_s - m*)]
    R1 --> R2[running_sumexp *= exp(m_s - m*)]
    R2 --> R3[running_max_for_softmax = running_max]
    S4 -- 否（skip scale） --> S5
    R3 --> S5[S = exp(P - running_max_for_softmax)]
    S5 --> S6[out_accum += S · gathered_kv]
    S6 --> S7[running_sumexp += S.sum(-1)]
```

## 读图要点

* 中间那个菱形是**条件分支**：绝大多数块走“否”，即 skip-scale
  （报告 `:69`：“如果没有任何一行 i 触发了这个条件，就跳过这一步”）。
* 之所以能跳过，是因为 `exp(P − m_s)` 里用的 `m_s`（`running_max_for_softmax`）
  允许比真实 max `m*` 小，只要差值不超过 `RESCALE_THRES`（报告取 6）。
* 阈值作用在 `P * sm_scale` 上；代码把阈值预先除以 `sm_scale`
  （`config.h:43`，`kernel_body.h:1660-1662`）。

## 来源

* 报告 `:39-73`（计算流程的 8 步）；阈值在 `:57`，`exp(6)` 的说明在 `:67-68`。
