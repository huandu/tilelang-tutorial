# 图 3：GEMM 的数据流与两级同步

**用途**：正文 §5.3。对应 A 树 `examples/gemm/example_gemm.py` 与 M 树
`SKILL.md` 第 4 节。

```text
                 MTE2                MTE1               M                  FixPipe
  GM(A) ─┐
         ├─T.copy─▶ A_L1 ─┐
  GM(B) ─┘                ├─(gemm_v0 内部)─▶ L0A/L0B ─▶ Cube ─▶ C_L0 ──▶ GM(C)
                          │                                    (L0C)
                 ┌────────┴────────┐
                 │  barrier_all()  │  同步点 ①：L1 数据就绪（MTE2 → MTE1/M）
                 └────────┬────────┘
                          ▼
                    T.gemm_v0(A_L1, B_L1, C_L0, init=(k==0))
                          │
                 ┌────────┴────────┐
                 │  barrier_all()  │  同步点 ②：Cube 算完，L1 可被下一轮覆盖（M → MTE2）
                 └─────────────────┘

  整段放在 for k in T.serial(ceildiv(K, K_L1)) 里：
    L1 缓冲沿 K 反复复用，这正是「沿 K 分块」写进代码结构的地方。
```

## 这一行展开了多少件事

| 阶段 | PIPE | 谁在做 |
| --- | --- | --- |
| GM → L1 | MTE2 | 你写的 `T.copy` |
| L1 → L0A/L0B | MTE1 | `T.gemm_v0` 内部 |
| Cube MAD（累加到 L0C） | M | `T.gemm_v0` 内部 |
| L0C → GM | FixPipe | 你写的 `T.copy` |

## 容量账（Atlas A2，fp16 输入 / fp32 累加）

* L1 需放下 `block_M×K_L1 + K_L1×block_N` 个 fp16 → 512 KB 预算；
* L0C 需放下 `block_M×block_N` 个 fp32 → 128 KB 预算。
  官方示例 `block_M=128, block_N=256` 的 fp32 累加恰好是 128 KB，**顶到 L0C 上限**。

来源：A 树 `examples/gemm/example_gemm.py`、`T.gemm_v0` 的 docstring；
catlass `atlasA2_hardware_info.md`（容量）；M 树 `SKILL.md` §4–§5（通路与约束）。
