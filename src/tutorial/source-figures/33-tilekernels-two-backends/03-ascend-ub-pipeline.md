# 图 3：Ascend 版的 UB / Vector Core 流水

**用途**：正文 §4.6。

事实来源：`.../TileKernels/tile_kernels/moe/topk_gate_asc.py:30-48,151,153`；
`_vendor/tilelang-v0.1.15/tilelang/ascend/language/annotations.py:19-45`
（`annotate_buffer_versions` 的语义与 `num_stages > 1` 时 Z3 的行为）；
`.../tilelang/ascend/language/__init__.py:8-18`（`alloc_shared` 在昇腾方言里指向 UB）。

```text
  时间 ────────────────────────────────────────────────────────────────►

  PIPE: MTE2                PIPE: V（SIMD 计算）        PIPE: MTE3
  ────────────              ────────────────────         ────────────
  GM ──copy──► scores_ub
   (第 w+5 行)                       ...

                    S.vld ─► vmax/vmin 归约
                              ─► vsts out_ub
                              （第 w 行）
                                                          out_ub ──copy──► GM
                                                          （第 w-1 行）

  三个缓冲的版本数由 T.annotate_buffer_versions 显式声明：
    scores_ub        : num_stages = 6
    out_ub           : num_stages = 6
    invalid_logits_ub: 1          ← 跨所有迭代共享的累积器
```

## 每轮迭代做了什么（快路径，`num_groups == 1`）

| 步骤 | 位置 | 说明 |
| --- | --- | --- |
| 取一行 token | `:46` `row = w * num_vec_cores + core_id` | 按核交错分配，保证每次搬的是连续一整行 |
| GM → UB | `:48` `T.copy(scores[row, :], scores_ub[:num_experts])` | MTE2 |
| 非法值打标 | `:65-71` `vsub/vcmps/vmax` + 单 lane 写回 | 结果累积进 `invalid_logits_ub` |
| 载入 vreg | `:75-81` | `scores_vec[i]` / `index_vec[i]` |
| 重复取 max | `:85-99` | `vmax` 组内折叠加 `vcmax` 跨 lane |
| 写第 k 个 idx | `:101` `vsts(out_ub[2*k], idx, one, 'ONEPT_B32')` | 单 lane 写 |
| 屏蔽中奖 lane | `:104-105` `vsel(neg_inf, ...)` | 下一轮不再考虑 |
| UB → GM | `:153` `T.copy(out_ub[:num_topk*2], topk_idx_view[row, :])` | MTE3 |
| 循环后统一断言 | `:155` | 因为 `invalid_logits_ub` 只有 1 份 |

## 为什么必须显式给版本数

`annotations.py:19-45` 的原文要点：

* `1`：退出多缓冲资格（含显式 owner 声明）；
* `num_versions >= 2`：固定版本数，交给调度器安排索引模式；
* `mode`（`"auto"` / `"iteration"` / `"counter"`）：只选模式，版本数交给调度器；
* 显式固定份数的用途是「无论求解器怎么选，都保证这份关键缓冲被多缓冲」。

也就是说：**调度器求的是「依赖上至少要几份」，而 `num_stages` 表达的是
「我想重叠多深」。后者是性能意图，必须由写 kernel 的人给出。**
