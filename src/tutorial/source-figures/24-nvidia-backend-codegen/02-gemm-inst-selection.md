# 图 2：`T.gemm` 的指令选择判定树

对应 `_vendor/tilelang-v0.1.15/src/cuda/op/gemm.cc:336-375` 的 `Gemm::SelectInst`。

```mermaid
flowchart TD
    S["T.gemm / T.wgmma_gemm / T.tcgen05_gemm"] --> Q1{"显式 wgmma?"}
    Q1 -- 是 --> W1{"AllowWgmma?"}
    W1 -- 是 --> R1["cuda.wgmma"]
    W1 -- 否 --> F1["编译期报错<br/>FatalWgmmaUnavailable"]

    Q1 -- 否 --> Q2{"显式 tcgen05?"}
    Q2 -- 是 --> T1{"AllowTcgen5Mma?"}
    T1 -- 是 --> R2["cuda.tcgen05"]
    T1 -- 否 --> F2["编译期报错<br/>FatalTcgen5Unavailable"]

    Q2 -- 否 --> Q3{"use_2cta?"}
    Q3 -- 是 --> T2{"AllowTcgen5Mma?"}
    T2 -- 是 --> R2
    T2 -- 否 --> F3["编译期报错<br/>（2CTA 无单 CTA 回退）"]

    Q3 -- 否 --> T3{"AllowTcgen5Mma?<br/>sm_100 且 C 在 TMEM<br/>且形状在 meta 白名单"}
    T3 -- 是 --> R2
    T3 -- 否 --> W2{"AllowWgmma?<br/>sm_90 且 M>=64<br/>且 warp 数是 4 的倍数<br/>且 CheckWgmma"}
    W2 -- 是 --> R1
    W2 -- 否 --> Q4{"sm_70 / sm_75 且<br/>native MMA 不可用?"}
    Q4 -- 是 --> R3["cuda.fma"]
    Q4 -- 否 --> R4["cuda.mma"]
```

`AllowTcgen5Mma` 的三条硬条件（`src/cuda/op/gemm.cc:80-87`）：

1. `TargetIsSm100(target)`；
2. `A` 在 shared 或 TMEM、`B` 在 shared、**`C` 在 TMEM**；
3. `GetTCGEN5MMAMeta(M, N, K, ab_dtype, c_dtype).first` 为真。

`AllowWgmma` 的三条硬条件（`src/cuda/op/gemm.cc:89-97`）：Hopper、`M >= 64`、
`num_warps % 4 == 0`，外加 `CheckWgmma` 的 dtype/转置/K 整除约束。
