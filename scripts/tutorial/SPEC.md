# 图表规范（TileLang 教程）

教程里所有图都由 **JSON 规格** 渲染成 SVG，渲染器是 `scripts/tutorial/svg.mjs`。
你只需要写规格，不要手写 SVG、不要指定颜色——颜色由站点主题（DaisyUI 语义 token）
统一提供，明暗两套主题自动生效。

## 文件位置

每个图一个文件：

```
src/tutorial/figures/<chapter-slug>/<figure-id>.json
```

`<figure-id>` 用原图文件的序号加语义名，例如 `01-storage-hierarchy`。

## 通用字段

| 字段      | 必填 | 说明                                                                              |
| --------- | ---- | --------------------------------------------------------------------------------- |
| `id`      | 是   | 必须与文件名一致（不含 `.json`）                                                  |
| `title`   | 是   | 一行标题，同时作为 SVG 的 `aria-label`（无障碍）                                  |
| `caption` | 否   | 显示在图下方的说明文字（中文，一到两句）                                          |
| `kind`    | 是   | `flow` / `layers` / `steps` / `compare` / `timeline` / `bars` / `grid` / `custom` |
| `notes`   | 否   | 图下方的小字备注列表，用于「读图提示」                                            |

节点/条目的 `tone` 取值：`default`（默认）、`accent`（强调，主色描边）、`store`（存储）、
`compute`（计算单元）、`muted`（次要）、`warn`（风险/陷阱）。**不要**写颜色值。

## 1. `flow`：流程图（最常用）

```json
{
  "id": "01-compile-flow",
  "title": "一次编译的主要阶段",
  "kind": "flow",
  "direction": "LR",
  "nodeMaxEm": 20,
  "nodes": [
    { "id": "src", "label": "Python kernel", "sub": "T.Kernel / T.copy", "tone": "default" },
    { "id": "tir", "label": "TIR IRModule", "sub": "官方口径：TVM TIR", "tone": "accent" },
    { "id": "cuda", "label": "CUDA 源码", "code": "tilelang/cuda/codegen", "tone": "compute" }
  ],
  "edges": [
    { "from": "src", "to": "tir", "label": "前端" },
    { "from": "tir", "to": "cuda" }
  ],
  "groups": [{ "label": "编译期", "nodes": ["tir"], "tone": "default" }],
  "notes": ["虚线表示可选路径"]
}
```

- `direction`：`LR`（从左到右）或 `TB`（从上到下）。节点多、标签长时用 `TB`。
- `edges[].dashed: true` 画虚线，`edges[].label` 标在连线中点。
- `groups` 用来表达 Mermaid 里的 `subgraph`。
- 节点支持 `badge`（右上角小标）与 `shape`（`rect` 默认 / `pill` / `diamond`）。

## 2. `layers`：分层/层级图

适合存储层级、软件栈、抽象层次。

```json
{
  "id": "02-storage-hierarchy",
  "title": "存储层级对照",
  "kind": "layers",
  "layers": [
    { "label": "寄存器 / Register File", "sub": "最快、最小", "right": "NV", "tone": "store" },
    {
      "label": "Shared Memory（与 L1 合并）",
      "sub": "H100 可配到 228 KB",
      "right": "NV",
      "tone": "store"
    },
    { "label": "L0A / L0B / L0C", "sub": "Cube 专用", "right": "昇腾", "tone": "store" }
  ]
}
```

## 3. `steps`：步骤序列

适合算法步骤、pass 顺序、操作流程。`direction` 为 `LR`（默认）或 `TB`。

```json
{
  "id": "03-online-softmax",
  "title": "online softmax 的递推",
  "kind": "steps",
  "steps": [
    { "label": "读入一个 K 块", "sub": "计算块内 max" },
    { "label": "重标定累加器", "sub": "exp(m_old − m_new)" },
    { "label": "累加 P·V", "tone": "accent" }
  ]
}
```

## 4. `compare`：并列对照

适合「NV vs 昇腾」这类逐项对比；列数 2–4，行数不限。

```json
{
  "id": "04-nv-vs-ascend",
  "title": "两台机器的对照",
  "kind": "compare",
  "columns": ["NVIDIA GPU", "昇腾 950"],
  "rows": [
    { "label": "执行单元", "values": ["线程 / warp / 协作组", "Cube(AIC) + Vector(AIV)"] },
    { "label": "片上存储", "values": ["寄存器、shared memory", "UB、L1、L0A/L0B/L0C"] }
  ]
}
```

## 5. `timeline`：时间线

```json
{
  "id": "05-ascend-timeline",
  "title": "昇腾后端的关键节点",
  "kind": "timeline",
  "items": [
    { "label": "tilelang-ascend 开源", "at": "2025-09" },
    { "label": "PTO 后端进入 main", "at": "2026-09-30", "tone": "accent" }
  ]
}
```

## 6. `bars`：量级对比

适合容量、带宽、延迟的量级比较。`value` 决定条长，`display` 是显示文本。

```json
{
  "id": "06-alignments",
  "title": "对齐字节数",
  "kind": "bars",
  "items": [
    { "label": "UB 对齐", "value": 32, "display": "32 B" },
    { "label": "L0A/L0B 对齐", "value": 512, "display": "512 B", "tone": "store" }
  ]
}
```

## 7. `grid`：矩阵 / 表格

适合 tile 分块、layout、数据排布。`headers` 可选；`rows[].label` 可选。

```json
{
  "id": "07-fragment-layout",
  "title": "fragment 与线程的对应",
  "kind": "grid",
  "headers": ["j=0", "j=1", "j=2", "j=3"],
  "rows": [
    { "label": "i=0", "cells": ["T0", "T1", "T2", "T3"] },
    { "label": "i=1", "cells": ["T4", "T5", "T6", "T7"], "tone": "accent" }
  ]
}
```

## 8. `custom`：自由摆放（最后手段）

只有在上面 7 种都表达不了时才用：自己给坐标（坐标系是 1000×高度 的自定义单位）。

```json
{
  "id": "08-swizzle",
  "title": "swizzle 前后",
  "kind": "custom",
  "width": 1000,
  "height": 320,
  "shapes": [
    {
      "type": "rect",
      "x": 60,
      "y": 40,
      "width": 380,
      "height": 220,
      "label": "线性地址",
      "tone": "muted"
    },
    {
      "type": "rect",
      "x": 560,
      "y": 40,
      "width": 380,
      "height": 220,
      "label": "XOR swizzle",
      "tone": "accent"
    },
    { "type": "text", "x": 440, "y": 140, "width": 120, "height": 40, "label": "重排" }
  ],
  "arrows": [{ "points": [450, 150, 550, 150] }]
}
```

## 写规格的硬性要求

1. **忠于原图**：信息必须来自对应的 `figures/*.md`（Mermaid 或 ASCII 版）。不要新增原图
   没有的结论；原图里的文字尽量保留。
2. **不要出现颜色、字体、像素**：颜色由 `tone` 决定；字号由渲染器管理。
3. **标签要短**：节点标题 ≤ 18 个汉字（或 36 个半角字符），副标题 ≤ 28 个汉字；
   超长会自动折行，但折行后的图会变高。
4. **每张图只表达一件事**：一张图里节点建议 ≤ 12 个；超过就拆成两张图（用 `-a`/`-b` 后缀）。
5. **JSON 必须合法**：不要有注释、不要有尾随逗号。
6. **`id` 必须与文件名一致**，且只含小写字母、数字与连字符。
