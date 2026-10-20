# 图 4：量化算子族地图

**用途**：正文 §5.1、§5.3–§5.5。

事实来源：`.../TileKernels/tile_kernels/quant/__init__.py:1-11`（导出清单）；
`quant/common.py:19-33,70-75,174-185,251-282`；三个 `*_kernel.py` 的分派段
（`per_token_cast_kernel.py:37-62`、`per_block_cast_kernel.py:42-61`、
`per_channel_cast_kernel.py:57-62`）；`swiglu_forward_kernel.py:40-76`。

```mermaid
flowchart TD
    X["输入 x<br/>[num_tokens, hidden]（bf16 / fp32）"] --> G{"选哪种粒度？"}

    G -->|"每行沿通道切"| PT["per_token_cast<br/>sf_block = (1, C)<br/>C ∈ {16,32,64,128}"]
    G -->|"2D 分块"| PB["per_block_cast<br/>sf_block = (Rt, C)<br/>昇腾只支持 (32,32)"]
    G -->|"每列跨多个 token"| PC["per_channel_cast<br/>sf_block = (T, 1)<br/>T ∈ {32,128}；昇腾只支持 32"]

    PT --> SF["get_sf_and_inv<br/>clamp → amax/max → 幂次化 → 打包"]
    PB --> SF
    PC --> SF
    SF --> Q1["fp8 (e4m3)"]
    SF --> Q2["fp4 (e2m1, int8 两两打包)"]

    Q1 --> CB["cast_back / per_token_cast_back<br/>反量化 → bf16 / fp32"]
    Q2 --> CB

    PT --> SW["swiglu_forward_and_per_token_cast<br/>SwiGLU + 量化融合"]
    SW --> Q1

    CFG["BaseCastConfig (frozen dataclass)<br/>sf_block / use_packed_ue8m0 / use_e4m3_sf"] -.-> PT
    CFG -.-> PB
    CFG -.-> PC
    CFG -.-> SW

    subgraph DIFF["后端差异"]
        D1["UE8M0 打包因子：CUDA 4 / 昇腾 2"]
        D2["use_e4m3_sf：只支持 CUDA"]
        D3["列主序 sf：CUDA 需 16B 对齐，昇腾不需要"]
    end
    SF -.-> DIFF
```

## 读图提示

* **粒度是 `sf_block` 的形状**，不是「块大小」：`(1, C)` 是 per-token、
  `(Rt, C)` 是 per-block、`(T, 1)` 是 per-channel。
* 三个 cast 共用同一段算术（`get_sf_and_inv`）、同一套配置（`BaseCastConfig`），
  因此两个后端的舍入与 clamp 口径**不可能不一致**。
* 融合路径（SwiGLU + per-token cast）省掉的是「SwiGLU 结果写回 GM、再读回来量化」
  这一趟往返；代价是多套 UB 缓冲。
* 图中的三条后端差异会让 sf 张量的**形状或取值**不同，因此
  跨设备搬运权重时必须重新打包。
