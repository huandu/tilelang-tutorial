# 图 03 · 三条 padding 路径不是同一件事

```mermaid
flowchart TB
    subgraph SIMT["SIMT 路径（src/op/copy.cc:449-525）"]
        S1["src_predicate = MakePredicate(src->shape)"] --> S2["value = if_then_else(src_pred, load, safe_value)"]
        S2 --> S3["safe_value 默认 = 0<br/>可被 T.annotate_safe_value 覆盖"]
        S4["dst_predicate = MakePredicate(dst->shape)"] --> S5["body = IfThenElse(dst_pred, store)"]
    end
    subgraph TMA["TMA 路径"]
        T1["TMADesc.oob_fill"] --> T2["越界语义由 descriptor 定义<br/>没有 Python 层安全值"]
    end
    subgraph ASC["昇腾 T.copy pad"]
        A1["pad_value=v"] --> A2["行宽向上取整到 32B 边界"]
        A2 --> A3["pad lane 填 v"]
        A3 --> A4["先发 ascend_set_copy_pad_value<br/>供 AutoSchedule 插入 PIPE_S→PIPE_MTE2 同步"]
    end
```

对照表：

| 路径 | 触发条件 | “填充”发生在哪 | 填什么 | 能不能改 |
| --- | --- | --- | --- | --- |
| SIMT（NV） | 默认回退路径 | 每条越界 lane 的取值 | 默认 0 | 能，`T.annotate_safe_value({A: v})` |
| 目标侧越界（NV） | dst 区域超出 dst buffer | 整条 store | 不写 | 不需要改 |
| TMA（NV） | `prefer_instruction="tma"` / `T.tma_copy` / TMA store | tensor map descriptor | `oob_fill` | 只能在 descriptor 参数层面 |
| 昇腾 DMA | `T.copy(..., pad_value=v)` | 行尾对齐到 32B 的位置 | `v` | 传参；也可 `data_select=True` 复用上次的 pad 寄存器 |

**来源**：`_vendor/tilelang-v0.1.15/src/op/copy.cc:449-481,507-525`；
`src/cuda/op/copy.h:49-60`；`tilelang/ascend/language/copy_op.py:278-287,326-353`。

**要点**：昇腾的 `pad_value` 是**DMA 对齐填充**，与“越界读替一个值”是两件事；
把两者混在一起写 kernel 会在 tail 处理上出错。
