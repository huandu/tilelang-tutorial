# 图 5：瓶颈判读决策树

**用途**：正文 §8.3。输入是 M 树 `do_bench(backend="msprof_detail")` 返回的
`KernelProfile`（`tilelang/profiler/msprof.py:60-134`）。

```mermaid
flowchart TD
    S[dur_ns 与算术强度] --> P{dur 高，问题在哪侧?}
    P -->|compute-bound| M[aic_mad 高不高?]
    M -->|aic_mad 已接近上限| ALG[Cube 已打满<br/>改算法/tile/dtype，别再叠流水]
    M -->|aic_mad 低，Cube 在等| W{谁最忙?}
    W -->|aic_mte2 高| GM2[GM→L1 带宽是瓶颈<br/>L2 提示 / swizzle / 减少重复读]
    W -->|aic_mte1 高| L0L1[L1→L0 太频繁<br/>调 kL0Size / L0 tile]
    W -->|aic_fixpipe 高| FIX[L0C 写回太慢<br/>调 L0C 使用方式/写回策略]
    W -->|都不高| SYNC[可能在等同步<br/>查 flag 配对 / 多缓冲]
    P -->|memory-bound| V[aiv_mte2 / aiv_mte3 高不高?]
    V -->|高| BAND[搬运带宽吃满<br/>做算子融合 / 减少 GM 往返]
    V -->|低但 dur 高| VSYNC[在等同步或多缓冲不够<br/>检查 T.Pipelined / 版本数]
```

## 三条核心判据

1. **Cube 忙不忙**：`aic_mad` 高 + `dur` 高 → 计算瓶颈，改算法；`aic_mad` 低 →
   Cube 在等数据，看 `aic_mte1/2` 与 `aic_fixpipe`。
2. **Cube 在等谁**：`aic_mte2` → GM→L1；`aic_mte1` → L1→L0；`aic_fixpipe` → L0C 写回。
3. **Vector 侧**：`aiv_mte2/3` 高而 `aiv_vec` 低 → 纯搬运受限，做融合。

## 读这些比值时的两个陷阱

* **AIC 与 AIV 的比值各自相对自己的 `total_cycles`**，不要相加当成一个统一时间轴的占比。
* `do_bench(backend="msprof")` 不带 median/quantile 选择，多取几次自己求中位数。

来源：M 树 `tilelang/profiler/msprof.py:60-134`（字段定义）、
`references/profiling.md`（口径与用法）、`SKILL.md` §12（实验方法）。
本图是**判读方法**，不含任何实测数字。
