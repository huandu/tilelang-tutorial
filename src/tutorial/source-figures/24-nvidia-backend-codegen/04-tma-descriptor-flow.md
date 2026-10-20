# 图 4：TMA 描述符的流转

```mermaid
sequenceDiagram
    participant P as 前端 (Python)
    participant L as LowerTileOp / Copy::LowerBulk
    participant H as LowerHopperIntrin
    participant W as TLCUDASourceWrapper (host)
    participant D as device kernel
    participant HW as TMA 硬件单元

    P->>L: T.copy(A_g, A_s) / T.tma_copy(...)
    L->>L: 选 CopyInst，构造 TMADesc
    L->>H: create_tma_descriptor(...) 留在 device IR 里
    H->>H: 收集 desc 初始化语句
    H->>W: 挂到 host 函数属性 tma_descriptor_args
    W->>W: 生成 cuTensorMapEncodeTiled(...)
    W->>D: 以 __grid_constant__ const CUtensorMap 参数传入
    D->>D: 选出单个线程（elected）
    D->>HW: expect_tx + cp.async.bulk.tensor.Nd
    HW-->>D: 完成后 arrive 到 mbarrier
    D->>D: 消费者 mbarrier_wait_parity
```

源码位置：

* 构造描述符与发射：`_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:2108-2215`；
* 提升为 host 属性：`_vendor/tilelang-v0.1.15/src/cuda/transform/lower_hopper_intrin.cc:27-44`；
* host 侧编码模板：`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:111-131`（`cuTensorMapEncodeTiled`）与
  `:133-159`（`cuTensorMapEncodeIm2col`）；
* device 侧 PTX：`_vendor/tilelang-v0.1.15/src/tl_templates/cuda/copy_sm90.h:212-270`（普通）
  与 `:88-210`（multicast）；
* mbarrier 原语：`_vendor/tilelang-v0.1.15/src/tl_templates/cuda/barrier.h:11`（`mbarrier_init`）、
  `:112`（`mbarrier_arrive_expect_tx`）、`:34`（`mbarrier_wait`）、
  `:184`（`tma_store_arrive`）、`:194`（`tma_store_wait`）。
