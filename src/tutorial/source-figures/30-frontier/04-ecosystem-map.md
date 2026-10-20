# 图 4：TileLang 生态的仓库地图（自绘）

```text
                  ┌──────────────────────────────────────────────┐
                  │  tile-ai/tilelang  （主干，本教材基线）        │
                  │  v0.1.15 / main                              │
                  │  tilelang/{cuda,rocm,metal,cpu,webgpu,ascend} │
                  │  src/{cuda,rocm,metal,cpu,webgpu,ascend,backend/common}
                  └───────┬──────────────────────────┬───────────┘
                          │                          │
        接入 / 适配        │                          │  上层使用者
                          ▼                          ▼
   ┌───────────────────────────────┐   ┌──────────────────────────────────┐
   │ tile-ai/tilelang-ascend       │   │ tile-ai/TileOPs                  │
   │  A2/A3 适配器（另一棵树）      │   │  spec 驱动的 LLM 算子库           │
   │  Ascend C & PTO 两条 codegen  │   │  （GemmFwdOp 等，可 autotune）    │
   └───────────────────────────────┘   └────────────┬─────────────────────┘
                                                    │ 被 agent 生成/优化
   ┌───────────────────────────────┐                ▼
   │ tile-ai/tilelang-mlir-ascend  │   ┌──────────────────────────────────┐
   │  AscendNPU IR（MLIR）路线      │   │ tile-ai/TileFoundry              │
   │  target kind = npuir          │   │  「编译器交给 agent」的生成流水线  │
   └───────────────────────────────┘   │  产出打上 foundry 标签的 PR        │
                                       └──────────────────────────────────┘
   ┌───────────────────────────────┐   ┌──────────────────────────────────┐
   │ tile-ai/TileRT                │   │ tile-ai/TileSight                │
   │  超低延迟推理 runtime          │   │  tile 级性能解析模型（CPU 上算）  │
   │  算子被它调度、被它约束        │   │  回答「为什么这个 tiling 更好」    │
   └───────────────────────────────┘   └──────────────────────────────────┘

   tile-ai/tilelang-benchmark —— 已归档，README 指向 TileOPs
```

## 每个仓库解决什么问题、算不算“代码”

| 仓库 | 定位 | 读者什么时候需要它 |
| --- | --- | --- |
| `tile-ai/tilelang` | 编译器本体：语言、pass、后端、JIT | 写算子、改编译器 |
| `tile-ai/tilelang-ascend` | A2/A3 的社区适配器（主干之外） | 在 A2/A3 上开发 |
| `tile-ai/tilelang-mlir-ascend` | 昇腾的 AscendNPU IR（MLIR）路线 | 研究第二条昇腾 codegen |
| `tile-ai/TileOPs` | 生产算子的参照实现（spec 驱动） | 想知道“工业界怎么写” |
| `tile-ai/TileRT` | 推理 runtime，约束回到编译期 | 关心部署与启动开销 |
| `tile-ai/TileFoundry` | agent 驱动的内核生成流水线 | 关心“AI 写算子”的现状 |
| `tile-ai/TileSight` | 解析性能模型 | 想不靠暴力调参选 tile |
| `tile-ai/tilelang-benchmark` | 已归档，被 TileOPs 取代 | 只作历史参考 |

## 事实边界

* 本图里各仓库的**描述**来自 corpus 里各自 README 快照的原文，抓取日期
  2026-09-30（`TileFoundry` 为 2026-09-29 的最后 push）。
* 仓库之间的依赖关系（例如 TileFoundry 的 PR 进 TileOPs）来自 TileOPs README 的
  badge 指向与 TileFoundry README 的 News 段落，是**上游自述**，本章未做实测。
* 注意命名：本章提到的是 **TileFoundry**（`tile-ai/TileFoundry`）。不存在
  `TileFindry` 这个仓库；若在别处看到该拼写，按拼写错误处理。
