# 图 1：昇腾片上存储与搬运通路

**用途**：正文 §2.2–§2.3。

```text
                    ┌──────────────────────── AI Core ────────────────────────┐
                    │                                                         │
   ┌────────┐  MTE2 │  ┌──────────┐  MTE1   ┌────────┐                        │
   │  GM    │──────▶│  │  L1      │────────▶│ L0A    │──┐                     │
   │ (HBM)  │       │  │ (CBuf)   │────────▶│ L0B    │  │  ┌────────┐         │
   │        │       │  │ 512 KB   │────────▶│ 各64KB │  ├─▶│  Cube  │         │
   │        │       │  └──────────┘         └────────┘  │  │ (MAD)  │         │
   │        │       │        ▲                           │  └───┬────┘         │
   │        │       │        │                           ▼      ▼              │
   │        │ MTE2  │  ┌──────────┐                  ┌────────────┐            │
   │        │──────▶│  │    UB    │                  │    L0C     │            │
   │        │       │  │ 192 KB   │                  │  128 KB    │            │
   │        │◀──────│  │ (Vector) │                  │ (fp32 累加)│            │
   │        │ MTE3  │  └──────────┘                  └─────┬──────┘            │
   │        │◀──────│       ▲                              │ FixPipe           │
   │        │FixPipe│       │                              │ (L0C→GM/L1)       │
   └────────┘       │       └──────────────────────────────┘                   │
                    │                                                         │
                    └─────────────────────────────────────────────────────────┘
```

## 要点

* **Cube 的输入必须先到 L0A/L0B，输出在 L0C**；L1 只是中转与复用缓冲。
* **Vector 的输入输出都在 UB**；UB 与 L1 之间也可以互相搬（`UB→L1`）。
* Cube 与 Vector **不能直接交换数据**，要经过 GM/L2（M 树用“Workspace 消除”自动化）。
* 每条搬运通路都是**异步的**，跨通路的数据依赖必须显式同步。

## 通路与指令对照

| 通路 | 搬运单元 | 常见指令 |
| --- | --- | --- |
| GM → L1 | MTE2 | `DataCopy` |
| GM → UB | MTE2 | `DataCopy` |
| L1 → L0A/L0B | MTE1 | `LoadData` |
| Cube 计算 | M | `Mmad` |
| L0C → GM/L1 | FixPipe | `Fixpipe` / `DataCopy` |
| UB → GM | MTE3 | `DataCopy` |

来源：corpus `06-huawei-ascend/raw/hiascend/llms-content/document/detail/canncommercial/
canncommercial_900_programug_Ascendcopdevg_atlas_ascendc_10_0008_html.md`；
catlass `docs/zh/2_Design/01_kernel_design/00_basics/atlasA2_hardware_info.md`。
容量数字为 **Atlas A2** 的，Ascend 950 请以平台接口为准。
