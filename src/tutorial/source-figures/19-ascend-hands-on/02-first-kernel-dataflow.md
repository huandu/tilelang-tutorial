# 图 2：逐元素 add 的数据流与同步点

**用途**：正文 §4.3。对应 A 树 `examples/elementwise/elementwise_add.py`。

```text
        PIPE: MTE2                    PIPE: V                     PIPE: MTE3

   GM(A) ──T.copy──┐
                   ├──▶ a_ub ┐
   GM(B) ──T.copy──┘         │
                   ├──▶ b_ub ┤
                             │
                    T.barrier_all()  ◀── 同步点 ①：等两次 MTE2 全部完成
                             │
                             ▼
                    c_ub = a_ub + b_ub   （T.tile.add）
                             │
                    T.barrier_all()  ◀── 同步点 ②：等 Vector 全部完成
                             │
                             ▼
   GM(C) ◀──T.copy──────── c_ub
```

## 两个同步点分别是哪条依赖

| 同步点 | 保护的关系 | 不写会怎样 |
| --- | --- | --- |
| ① | `MTE2 → V`：a_ub/b_ub 的数据就绪 | Vector 可能读到旧数据（结果错） |
| ② | `V → MTE3`：c_ub 的计算结果就绪 | MTE3 可能把未算完的 c_ub 写回 GM（结果错） |

## 为什么是“全屏障”而不是精确 flag

`T.barrier_all()` 等价于 `PIPE_ALL` 的全屏障，语义是“等本核此前发出的异步操作全部完成”。
它在最小示例里够用，但代价是**没有流水重叠**：MTE2 与 V 无法并行。
要做到“搬下一块的同时算这一块”，就要用多缓冲 + 精确 flag，见
`../code/03-elementwise-pipeline/` 与正文 §4.4。
