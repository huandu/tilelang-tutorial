# 图 07 · 昇腾一次 GEMM+ReLU 的原语与存储层级

```mermaid
flowchart LR
    GM[("GM")]
    L1["L1 Buffer"]
    L0A["L0A"]
    L0B["L0B"]
    CUBE["Cube / AIC<br/>T.gemm → ascend.mad"]
    L0C["L0C"]
    UB["UB"]
    VF["Vector / AIV<br/>T.SimtVF / T.SimdVF<br/>（ReLU）"]

    GM -- "T.copy" --> L1
    L1 -- "T.copy" --> L0A
    L1 -- "T.copy" --> L0B
    L0A --> CUBE
    L0B --> CUBE
    CUBE --> L0C
    L0C -- "T.dual_copy（M/N 各半）" --> UB
    UB --> VF
    VF --> UB
    UB -- "T.dual_copy / T.copy" --> GM
```

ASCII 版：

```text
  GM ──T.copy(GM→L1)──▶ L1 ──┬──T.copy(L1→L0A)──▶ L0A ──┐
                             └──T.copy(L1→L0B)──▶ L0B ──┤
                                                  T.gemm (ascend.mad)
                                                         ▼
                                                       L0C
                                                         │
                                              T.dual_copy（两个 AIV 各一半）
                                                         ▼
                                                        UB ──T.SimtVF/T.SimdVF──▶ UB
                                                         │
                                                T.dual_copy / T.copy ──▶ GM
```

**来源**：`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:60-155`
（Quick Start 与“Writing Ascend Kernels”）；
`tilelang/ascend/language/copy_op.py:53-163`；
`tilelang/ascend/language/frame.py:107-195`。

**要点**：`T.dual_copy` 在这里的职责是把 L0C 累加器按 M 或 N 切开交给两个 AIV，
否则 Vector 侧只有一半并行度；它不是“把纯 Vector kernel 变成双 AIV”的开关。
