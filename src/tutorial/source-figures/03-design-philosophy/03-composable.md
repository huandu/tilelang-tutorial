# 图 3：composable ，一个 kernel 里的两层

对应正文 **2.6 与 3.3 节**。图为
`_vendor/tilelang-v0.1.15/examples/dequantize_gemm/example_dequant_gemm_bf16_mxfp4_hopper.py`
的流水循环（`:366-373`）的结构简化：**同一个 `T.Pipelined` 循环体里，一个是 thread 级
macro，一个是 tile 级 `T.gemm`，中间没有任何模式切换。**

```mermaid
flowchart TB
    K["with T.Kernel(grid, threads=threads)"] --> S["T.annotate_layout({B_shared: make_swizzled_layout(...)})   ← layout 维度"]
    S --> P["for k in T.Pipelined(K // block_K, num_stages=N)   ← pipeline 维度"]
    P --> C1["T.copy(A[...], A_shared)"]
    C1 --> C2["T.copy(B[...], B_shared)"]
    C2 --> DQ{"dequant 走哪条路？（同一个位置的两种实现）"}
    DQ -->|"fast_dequant = True"| F1["**expert 层**<br/>macro: T.get_thread_binding()<br/>T.alloc_local · T.serial · T.vectorized<br/>T.call_extern(外部反量化函数)"]
    DQ -->|"fast_dequant = False"| F2["**developer 层**<br/>macro: T.alloc_fragment<br/>T.copy · T.Parallel"]
    F1 --> G["T.gemm(A_shared, B_dequantize_shared, C_local, transpose_B=True)   ← tensorize 维度"]
    F2 --> G
    G --> OUT["T.copy(C_local, C_shared); T.copy(C_shared, C[...])"]

    classDef exp fill:#fff4e6,stroke:#b5722a;
    classDef dev fill:#eef6ff,stroke:#3b6ea5;
    class F1 exp;
    class F2 dev;
```

同一份代码里，四个调度维度全部出现过：

| 维度 | 在这个 kernel 里的出现位置 |
| --- | --- |
| thread binding | `threads=threads`（`:336`），macro 内 `T.get_thread_binding()`（`:225`） |
| layout | `T.annotate_layout` + `make_swizzled_layout`（`:344-348`） |
| tensorize | `T.gemm(..., transpose_B=True)`（`:373`） |
| pipeline | `T.Pipelined(..., num_stages=num_stages)`（`:366`） |

第二个例子（低层写在高层的旁边）见
`_vendor/tilelang-v0.1.15/examples/gemm/example_gemm_intrinsics.py:122-142`：
`T.Parallel` 搬运与 `mma_emitter.ldmatrix_a/b`、`mma_emitter.mma` 直接相邻。

事实来源：实体文件的对应行号已在正文 3.3 节逐条给出并核验（见 `../references.md` S12）。
