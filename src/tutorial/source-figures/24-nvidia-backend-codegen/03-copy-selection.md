# 图 3：`T.copy` 的搬运指令选择判定树

对应 `_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:716-891`。

```mermaid
flowchart TD
    S["T.copy / T.tma_copy / T.async_copy / T.copy_cluster"] --> G{"gather4 / scatter4 标注?"}
    G -- 是 --> RG["BulkLoadGather4 /<br/>BulkStoreScatter4"]
    G -- 否 --> CM{"cluster_mask != 0?"}
    CM -- 是 --> RC["必须 kBulkLoad<br/>（否则报错）"]
    CM -- 否 --> ET{"显式 T.tma_copy?"}
    ET -- 是 --> RT{"TMA 约束满足?<br/>sm_90+ / global<->shared / 16B 对齐"}
    RT -- 是 --> RB["kBulkLoad / kBulkStore / ...1D"]
    RT -- 否 --> ERR["编译期报错（不降级）"]
    ET -- 否 --> EA{"显式 async?"}
    EA -- 是 --> RA["kCPAsync"]
    EA -- 否 --> PI{"prefer_instruction?"}
    PI -- tma --> RT
    PI -- cp_async --> RA
    PI -- sync --> RS["LDSM / STSM / TMem / Normal"]
    PI -- 未指定 --> DEF{"store 方向且 TMA 可用?<br/>allow_load=false, allow_store=true"}
    DEF -- 是 --> RB
    DEF -- 否 --> RS2["SelectSyncLikeInst<br/>LDSM -> STSM -> TMemLoad/Store -> Normal"]
```

**默认路径只对 store 自动选 TMA**（`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:797-803`）。load 想走 TMA 需要在
`T.tma_copy`、`prefer_instruction="tma"`、或 warp-specialized producer 分支三者之一里。

TMA 的硬条件（`CheckBulkLoad` / `CheckBulkStore` / `CheckGlobalStrides`，
`_vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc:143-261`）：

* `arch >= 90`（`src/cuda/target_utils.cc:117-122`）；
* 作用域必须是 `global` ↔ `shared`/`shared.dyn`；
* 最后一维位数是 128 的整数倍；
* 最内层 stride 为 1，其余 stride 的字节数能被 16 整除且 < 2^40；
* dtype 对（含 FP4 只支持 load 方向的特殊规则）。
