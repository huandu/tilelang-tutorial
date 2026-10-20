# 图 · MoE 一层的数据流

```mermaid
flowchart TD
    X["激活 x: T x d"] --> R["router<br/>x @ W_g -> T x E<br/>softmax + top-k"]
    R --> C["计数 / 前缀和<br/>group_sizes, offsets,<br/>padded_offsets"]
    X --> P["permute<br/>T*k 行 x d<br/>2 x T*k*d*b 字节"]
    C --> P
    P --> G["grouped GEMM<br/>每专家 gate+up+down<br/>6*T*k*d*de FLOP"]
    G --> S["combine / unpermute<br/>scatter-reduce<br/>2 x T*k*d*b 字节"]
    C --> S
    S --> O["输出: T x d"]

    subgraph EP["专家并行（>1 rank）"]
        D["dispatch: all-to-all k 份"]
        K["combine: all-to-all k 份"]
    end
    P -.->|跨 rank| D
    D -.-> G
    S -.->|跨 rank| K
```

```text
每 token 的账（b = 每元素字节数）
--------------------------------------------
router      : 2*T*d*E FLOP          （很小，GEMV 形状）
permute     : 2*k*d*b 字节/token    （零 FLOP）
grouped GEMM: 6*k*d*de FLOP/token   （gate+up+down，各 2*k*d*de）
combine     : 2*k*d*b 字节/token    （零 FLOP）
EP dispatch :   k*d*b 字节/token    （网络）
EP combine  :   k*d*b 字节/token    （网络）
--------------------------------------------
重排 + 通信 = (4*k*d*b) 字节/token，全部不是矩阵乘。
```

依据：`_vendor/tilelang-v0.1.15/examples/fusedmoe/example_fusedmoe_tilelang.py`
（routed kernel 与 `scatter_reduce`）、
`_vendor/tilelang-ascend-v0.1.2.000/examples/dispatch_combine/dispatch_combine_shmem.py`
（shmem put 的 dispatch/combine 一对 kernel）。
