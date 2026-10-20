# 图 5 · 昇腾 AI Core 存储层级与搬运单元

正文位置：第 4.1、4.2、4.3 节。

```text
  AI Core（分离模式：一组 Cube Core / AIC + Vector Core / AIV）
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Cube 侧（AIC）                          Vector 侧（AIV）              │
  │                                                                        │
  │   L0A Buffer ─┐                        Unified Buffer（UB）            │
  │               ├─► Cube（矩阵）──► L0C ─┐   ▲  向量/标量的输入输出       │
  │   L0B Buffer ─┘   每次 16×16 fp16     │   │                            │
  │                                        │   │                           │
  │   L1 Buffer（Cube 中转区）             │   │                           │
  │   BT Buffer（Bias）                    │   │                           │
  │   FP Buffer（量化/ReLU 参数）          │   │                           │
  └────────────────────────────────────────┼───┼───────────────────────────┘
                                            │   │
   搬运单元（每条路径固定归属）：            │   │
     MTE2   GM ─────────────────────────► L1    UB  ◄─────┐
     MTE1   L1 ─────────────────────────► L0A/L0B, BT     │
     MTE3   UB ─────────────────────────► GM              │
     Fixpipe  L0C ──────────────────────► GM 或 L1        │
     Fixpipe  L1 ───────────────────────► FP Buffer        │

  GM（Global Memory / 外部显存）
```

## 两条官方给出的典型数据流

```text
Vector： GM → UB → Vector → UB → GM
Cube：   GM → L1 → L0A/L0B → Cube → L0C → Fixpipe → GM
                                              └─────→ L1
```

## 容量与对齐

* 官方要求用 `GetCoreMemSize` 在运行时查询容量；**本章不给 L0A/L0B/L0C 的容量数字**
  （corpus 内找不到可跨型号引用的出处）。
* 文档示例里出现过的数字：UB 192 K（某 Atlas A2 类产品的 bank 冲突示例）、
  L1 512 KB（某个 `GetMatmulApiTiling` 示例）。**都是示例，不是通用规格。**
* 对齐要求：UB 32 B、L1 32 B、L0A/L0B 512 B、L0C 64 B、BT 64 B、Fixpipe 64 B。

## 事实来源

* `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_0008_html.md`
  （计算/存储/搬运单元、两条数据流、MTE 路径、对齐、`GetCoreMemSize`）。
* `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_best_practices_10_00020_html.md`
  （UB 192 K / 48 bank 示例；Atlas A2 上下文）。
* `../tilelang-corpus/06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/canncommercial_900_API_ascendcopapi_atlasascendc_api_07_0666_html.md`
  （L1 512 KB 示例注释）。

均为厂商公开文档，**非实测**。

## 许可

自绘，随本书正文。
