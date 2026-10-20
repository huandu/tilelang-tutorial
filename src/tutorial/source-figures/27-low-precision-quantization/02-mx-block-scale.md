# 图 2：MX 块缩放（block-scaled）的生产与消费

用于正文第 2.3 节与第 4.1 节。左边是量化（生产 scale），右边是 kernel（消费
scale）。两边必须共享同一个粒度与同一个打包约定，否则结果直接错。

```mermaid
flowchart TB
    subgraph P["生产者（离线或单独的量化 kernel）"]
        A1["X: [MN, K] fp32/fp16"] --> A2["按 gran_k 分块<br/>求每块 amax"]
        A2 --> A3["scale = ceil_to_ue8m0(amax / 448)<br/>即 2^k，clamp 到 [1, 254]"]
        A3 --> A4["Xq = round_to_e4m3(X / scale)"]
        A3 --> A5["sf_u8 = fp32 的 biased exponent 字节<br/>= log2(scale) + 127"]
        A5 --> A6["每 4 个字节拼 1 个 uint32<br/>w0 | w1&lt;&lt;8 | w2&lt;&lt;16 | w3&lt;&lt;24"]
        A6 --> A7["转置成 group-major 扁平数组<br/>SFA[g*M + m]"]
    end
    subgraph C["消费者（T.tcgen05_gemm_blockscaled）"]
        B1["TMA 每 sf_load_period 次迭代搬一个 word<br/>sf_load_period = gran_k * 4 / block_K"]
        B1 --> B2["tcgen05_sf_warp_transpose<br/>4x32 word 视图 -> 32x4"]
        B2 --> B3["tcgen05_cp_warpx4<br/>每 128 word 块写入 4 个 TMEM 列"]
        B3 --> B4["MMA 时由 SFA_ID/SFB_ID 选字节子列<br/>id = (k_start / gran_k) % 4"]
        B4 --> B5["C += (A * SFA) @ (B * SFB)"]
    end
    A4 --> B5
    A7 --> B1
```

## ASCII 版

```text
生产者                                          消费者
─────────────────────────────────────────      ─────────────────────────────────────────
X[MN, K] fp16                                  SFA: [ceil(ceil(K/gran)/4) * M] uint32
  │ 按 gran 分块求 amax                          │ 每 sf_load_period = gran*4/block_K 次迭代搬 1 个 word
  ▼                                             ▼
scale = 2^ceil(log2(amax/448))                 tcgen05_sf_warp_transpose（4x32 -> 32x4）
  │                                             ▼
  ├─► Xq = round_e4m3(X / scale) ──► A/B tile   tcgen05_cp_warpx4（128 word -> 4 个 TMEM 列）
  │                                             ▼
  └─► sf_u8 = 指数字节 = log2(scale)+127        SFA_ID = (k_start/gran) % 4 选子列
        │                                       ▼
        └─► word = b0 | b1<<8 | b2<<16 | b3<<24  C += (A*SFA) @ (B*SFB)
            再按 group-major 展平：SFA[g*M+m]
```

## 三个必须写进接口的量

| 量 | 含义 | 锁定版本里的取值 |
| --- | --- | --- |
| `sf_granularity_k` | 一个 scale 字节覆盖多少个 K 元素 | SM100 MXFP8 示例是 128；PTX/CATLASS 的原生 MX block 是 32 |
| 一个 word 覆盖的 K | `4 × sf_granularity_k` | 128 粒度时是 512 |
| `sf_load_period` | 每多少次 K 迭代搬一个 word | `sf_granularity_k * 4 // block_K`；`block_K=128` 时是 4 |

## 来源

* `_vendor/tilelang-v0.1.15/examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py:1-73,509-572`
* `_vendor/tilelang-v0.1.15/examples/blockscaled_gemm_sm100/mxfp8_illustrated.md`（Scale-Factor Layout 一节）
* `_vendor/tilelang-v0.1.15/tilelang/language/builtin.py:1440-1479`
* `../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md:32061-32240`
