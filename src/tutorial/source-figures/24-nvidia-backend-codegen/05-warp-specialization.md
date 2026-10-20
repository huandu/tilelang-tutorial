# 图 5：warp specialization 的角色与握手

以 `_vendor/tilelang-v0.1.15/examples/warp_specialize/example_warp_specialize_gemm_copy_0_gemm_1.py`
为例（256 线程 = 2 个 warp group）。

```mermaid
flowchart LR
    subgraph PG["producer: T.ws(0)，tx &lt; 128"]
        P1["barrier_wait(compute_is_done, (ko+1)%2)"]
        P2["tma_copy(A_g -> A_s, barrier=data_is_ready)"]
        P3["tma_copy(B_g -> B_s, barrier=data_is_ready)"]
        P4["barrier_arrive(data_is_ready)"]
        P1 --> P2 --> P3 --> P4
    end
    subgraph CG["consumer: T.ws(1)，128 &lt;= tx &lt; 256"]
        C1["barrier_wait(data_is_ready, ko%2)"]
        C2["T.gemm(A_s, B_s, C_local)"]
        C3["barrier_arrive(compute_is_done)"]
        C1 --> C2 --> C3
    end
    P4 -->|data_is_ready 翻转| C1
    C3 -->|compute_is_done 翻转| P1
```

编译期顺序（仅当目标是 CUDA 且 `have_tma(target)`，也就是 `arch >= 9`）：

```text
AutoWarpSpecialization      （opt-in：tl.enable_auto_warp_specialization）
        ↓
MaterializeWSSchedule       （物化 WSSchedule：角色分支 + 多版本 buffer + mbarrier）
        ↓
ProducerConsumerWarpSpecialized （在 tile-op 层改写 pipelined loop）
        ↓
... LayoutInference / LowerTileOp ...
        ↓
AnnotateWarpGroupRegAlloc   （注入 set_max_nreg）
```

源码位置：`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:101-109`、
`:27-38`（门禁）、`:268-271`（寄存器）。指令分类在
`_vendor/tilelang-v0.1.15/src/cuda/transform/ws_analysis.h:33-41`。
