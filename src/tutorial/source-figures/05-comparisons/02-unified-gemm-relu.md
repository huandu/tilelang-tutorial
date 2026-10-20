# 图 2：统一例子 `C = relu(A @ B)` 的数据流对照

用于正文 §3。横轴是六个系统，纵轴是「一件事由谁负责」。

```mermaid
graph LR
  GM["Global memory (A, B, C)"]

  subgraph CUDA["CUDA (hand-written)"]
    CU1["thread computes element"]
    CU2["__shared__ aTile/bTile"]
    CU3["__syncthreads()"]
    CU4["epilogue relu in same kernel"]
    GM --- CU2 --- CU1 --- CU4
    CU3 -.->|user| CU2
  end

  subgraph TRITON["Triton (tile)"]
    T1["tl.load block"]
    T2["tl.dot"]
    T3["tl.maximum(acc, 0)"]
    T4["tl.store block"]
    T1 --- T2 --- T3 --- T4
  end

  subgraph TILELANG["TileLang (tile + explicit dataflow)"]
    L1["T.copy (GM->shared)"]
    L2["T.gemm"]
    L3["T.Parallel relu on C_local"]
    L4["T.copy out"]
    L1 --- L2 --- L3 --- L4
  end

  subgraph ASCEND["Ascend C (instruction)"]
    A1["CopyIn (DataCopy)"]
    A2["Compute / Cube"]
    A3["CopyOut"]
    A1 --- A2 --- A3
  end

  subgraph PTO["PTO-DSL (instruction)"]
    P1["pto.load GM->L1"]
    P2["tile.extract L1->L0"]
    P3["tile.matmul(_acc)"]
    P4["pto.store L0C->GM"]
    P1 --- P2 --- P3 --- P4
  end
```

「谁负责什么」对照（正文 §3.7 的表就是它的文字版）：

| 关注点 | CUDA | Triton | TileLang | Ascend C | PTO-DSL |
| --- | --- | --- | --- | --- | --- |
| 片上存储 | 用户 | 编译器 | 用户声明，pass 布局 | 用户 | 用户 |
| 多缓冲 | 用户 | 编译器 | `num_stages` | 用户 (`BUFFER_NUM`) | 自动/可选手写 |
| 同步 | 用户 | 编译器 | 编译器 pass | 用户 `EnQue/DeQue` | PTOAS InsertSync |
| 融合 ReLU | 用户 epilogue | 块级一行 | fragment 上一行 | 手写 Vector 段 | 手写 Vector 段 |

证据：CUDA=R7/R8；Triton=R2/R4；TileLang=S1/S10/S11；Ascend C=R14；PTO-DSL=R18/R19。
