# 图 2：三层接口的定位、形态与实现状态

对应正文 **2.5 节**。**注意中间那一层（beginner）是虚线框**：官方文档明确写着
“This interface is not yet fully implemented”
（`_vendor/tilelang-v0.1.15/docs/get_started/overview.md:20`）。本章如实标注，不把它画成已完成。

```mermaid
flowchart TB
    subgraph L3["Expert Level（已完成，示例中大量使用）"]
        E["T.get_thread_binding() · T.alloc_local<br/>T.serial / T.vectorized<br/>CUDA 方言：T.ldmatrix · T.tma_load · T.mbarrier_*<br/>TIR 内建：T.tvm_thread_allreduce"]
    end

    subgraph L2["Developer Level（v0.1.15 的主体）"]
        D["T.Kernel · T.alloc_shared / alloc_fragment<br/>T.copy · T.gemm · T.reduce_* / cumsum<br/>T.Parallel · T.Pipelined · T.annotate_layout"]
    end

    subgraph L1["Beginner Level（未完成）"]
        B["方向：eager 前端<br/>@tilelang.jit + T.const / T.dynamic<br/>+ T.Tensor[[...]] 注解 + T.empty<br/>（kernel 体里仍要写存储层级）"]
    end

    L1 -. "文档标注：not yet fully implemented" .-> L2
    L2 --> L3

    classDef done fill:#eef6ff,stroke:#3b6ea5;
    classDef todo fill:#f7f7f7,stroke:#999,stroke-dasharray: 6 4;
    class E,D done;
    class B todo;
    style L1 stroke-dasharray: 6 4;
```

三层的官方类比（`_vendor/tilelang-v0.1.15/docs/deeplearning_operators/matmul.md:17-21`）：

| 层 | 类比对象 | 文档原话的关键词 |
| --- | --- | --- |
| Level 1 / beginner | TVM | “conceptually similar to the idea behind TVM” |
| Level 2 / developer | Triton | “somewhat comparable to Triton's programming model” |
| Level 3 / expert | 手写 CUDA/HIP | “almost as explicit as a hand-written CUDA/HIP kernel” |

三层**可以混用**，这是 TileLang 相对 Triton 的官方差异之一
（`_vendor/tilelang-v0.1.15/docs/deeplearning_operators/matmul.md:214-218`）。
