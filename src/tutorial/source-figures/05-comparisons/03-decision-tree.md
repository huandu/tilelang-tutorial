# 图 3：什么时候该选谁

用于正文 §5 与 §7。判定顺序：**硬件 → 目标是算子还是编译器 → 愿意让出多少控制权**。

```mermaid
graph TD
  START["要写一段 kernel"] --> HW{"目标硬件？"}

  HW -->|NVIDIA| N1{"要改编译器，还是写算子？"}
  HW -->|昇腾| A1{"要改编译器，还是写算子？"}
  HW -->|其他/跨厂商| X1["TileLang（一个前端，多后端 lowering）"]

  N1 -->|写编译器| C1["TVM 底盘 + 给它写 pass (R10/R11/R12, S10/S11)"]
  N1 -->|写算子| N2{"愿意让出多少控制权？"}
  N2 -->|要完全控制 epilogue / 对齐库| N3["CUTLASS-CuTe (R5/R6)"]
  N2 -->|要自动调度、tile 级| N4["Triton (R2/R4) 或 TileLang (S1)"]

  A1 -->|写编译器| C1
  A1 -->|写算子| A2{"愿意让出多少控制权？"}
  A2 -->|贴硬件、显式 L1/L0/UB| A3["Ascend C (R14) 或 PTO-DSL (R17/R18)"]
  A2 -->|自动 layout/schedule/sync| A4["TileLang@Ascend (S3/S11)"]

  N3 --> NOTE["注意：NVIDIA 的 cuTile 与 CuTe DSL 是两个不同的东西；<br/>CuTe DSL 在低层 (R5/R27)，cuTile 资料不足 (R17)"]
  N4 --> NOTE
```

判定要点（都来自正文）：

1. **先选硬件**：NVIDIA / 昇腾 / 跨厂商，决定可选项集合（S2）。
2. **再问“改编译器还是写算子”**：改编译器 → TVM 底盘（§4）；写算子 → tile 层或低层。
3. **最后问“让出多少控制权”**：让得多 → tile 层自动调度；让得少 → 低层手写。
4. **不要用代码行数决定**：社区文章的“差几倍”没有公开基线（R22 的核验备注）。
