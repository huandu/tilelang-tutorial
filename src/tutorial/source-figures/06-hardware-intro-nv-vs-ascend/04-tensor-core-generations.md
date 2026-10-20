# 图 4 · Tensor Core 三代：把什么从寄存器里拿了出来

正文位置：第 3.3 节。

```mermaid
flowchart TB
    subgraph G1["第 1 代：warp 级 mma（Volta / Turing / Ampere）"]
        direction LR
        A1["A: 寄存器 fragment"] --> M1["mma.sync<br/>warp = 32 线程"]
        B1["B: 寄存器 fragment"] --> M1
        M1 --> D1["D: 寄存器 fragment"]
    end

    subgraph G2["第 2 代：warpgroup 级 wgmma（Hopper, sm_90a）"]
        direction LR
        A2["A: 寄存器 或 shared memory"] --> M2["wgmma.mma_async<br/>warpgroup = 128 线程<br/>异步"]
        B2["B: shared memory<br/>（描述符）"] --> M2
        M2 --> D2["D: 寄存器 fragment"]
        M2 -.-> C2["commit_group / wait_group"]
    end

    subgraph G3["第 3 代：CTA 级 tcgen05（Blackwell, sm_100a/f）"]
        direction LR
        A3["A: Tensor Memory 或 shared memory"] --> M3["tcgen05.mma<br/>CTA（可选 2-CTA）<br/>异步"]
        B3["B: shared memory"] --> M3
        M3 --> D3["D: Tensor Memory（TMEM）<br/>512 列 × 128 行 × 32-bit / CTA"]
        M3 -.-> C3["mbarrier 完成通知"]
    end

    G1 --> G2 --> G3
```

## 每代“解决了什么 / 留下了什么”

* **`mma`**：第一次提供远超通用 FP32 的矩阵乘吞吐。代价是操作数与累加器全在寄存器里，
  fragment 布局必须由程序员显式安排，寄存器压力限制了 tile 大小。
* **`wgmma`**：粒度升到 warpgroup（4 个连续 warp），变成异步；操作数可以放在 shared
  memory，不再全部占寄存器；发射与完成分离换来更深的流水。代价是累加器仍在寄存器，
  且要手工维护 commit/wait。
* **`tcgen05`**：把累加器搬进专用的 **Tensor Memory**；`A` 可在 TMEM 或 SMEM，`B` 在
  SMEM，`D` 在 TMEM；支持 block scaling（MXFP 格式）；完成通过 `mbarrier` 通知。
  代价是多了一种必须显式管理的存储和一套新的同步协议。

## 事实来源

* `../tilelang-corpus/05-nvidia-backend/raw/nvidia-ptx-isa.md`
  （9.7.16 `mma` / 9.7.17 `wgmma` / 9.7.18 `tcgen05`；TMEM 的形状与用途见 9.7.18.1
  与 9.7.18.10）。
* TileLang 侧的选择逻辑：`_vendor/tilelang-v0.1.15/tilelang/language/gemm_op.py:177-181`。

均为厂商公开文档，**非实测**。

## 许可

自绘（Mermaid），随本书正文。
