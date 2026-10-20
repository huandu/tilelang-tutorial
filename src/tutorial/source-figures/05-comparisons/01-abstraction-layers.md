# 图 1：抽象层次纵轴与 TVM 底盘

用于正文 §2.1 与 §4。图里的证据编号对应 `references.md`。

```mermaid
graph TB
  subgraph HIGH["tile 级：用户给 tile，编译器管调度/布局/同步（部分或全部）"]
    TR["Triton — 编译器选调度 (R2)"]
    TL["TileLang — 数据流留在语言里 (S1/S10)"]
    CT["cuTile — 资料不足 (R17/R9)"]
  end

  subgraph LOW["线程 / 指令级：用户直接面对线程、存储层级与同步"]
    CUDA["CUDA (R7/R8)"]
    CUTE["CUTLASS / CuTe & CuTe DSL (R5/R6/R27)"]
    AC["Ascend C (R14)"]
    PTO["PTO-DSL (R17/R18/R19)"]
  end

  subgraph ISA["硬件 ISA"]
    PTX["PTX / SASS"]
    PTOISA["PTO-ISA / CCE"]
  end

  TVM["TVM 底盘：IR + pass + target/device 抽象 (R10/R11/R12)"]

  TR --> PTX
  TL --> PTX
  CUDA --> PTX
  CUTE --> PTX
  AC --> PTOISA
  PTO --> PTOISA
  TVM -.->|横切| HIGH
  TVM -.->|横切| LOW
```

ASCII 版（方便在纯文本环境阅读）：

```text
高层  ▲  tile 级         :  Triton        cuTile        TileLang
      │
      │  线程 / 指令级    :  CUDA          CUTLASS-CuTe
      │                     Ascend C      PTO-DSL
      ▼
低层     硬件 ISA        :  PTX / SASS            PTO-ISA / CCE

横穿所有层的底盘： TVM（IR + pass 基础设施 + target/device 抽象）
```

读图要点：

* TileLang 是唯一同时出现在“tile 级”和“底盘使用方”的对象，前端在 tile 层，
  实现落在 TVM 上。
* CuTe DSL 虽然用 Python，但仍在**低层**（R5/R27）；不要因为它不是 C++ 就把它归到 tile 级。
* TVM 不是某一层的一员，而是横切面（§4）。
