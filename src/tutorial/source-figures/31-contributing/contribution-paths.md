# 三种贡献路径（图 31-1）

```mermaid
flowchart TB
    A[你的改动] --> B{改的是什么？}
    B -->|一个算子或一个示例| C["路径 1：算子 / 示例<br/>examples/&lt;op&gt;/"]
    B -->|pass / layout / codegen| D["路径 2：编译器<br/>src/transform/, src/layout/,<br/>tilelang/&lt;backend&gt;/pipeline.py"]
    B -->|新芯片或新执行模式| E["路径 3：后端<br/>tilelang/&lt;backend&gt;/ + src/&lt;backend&gt;/"]

    C --> C1["门槛：examples 能在 CI 矩阵里跑<br/>+ 正确性对照 + 形状/精度覆盖"]
    D --> D1["门槛：testing/python/transform/<br/>+ 布局改动跑 maint/layout_inference/<br/>+ 必要时性能回归"]
    E --> E1["门槛：backend manifest 契约<br/>+ testing/python/backend/<br/>+ 源码级 codegen 测试<br/>+ 打包与文档"]

    C1 --> F{维护者评审}
    D1 --> F
    E1 --> F
    F -->|合入| G[main]
    F -->|要求补充| A
```

| | 路径 1 · 算子/示例 | 路径 2 · 编译器 | 路径 3 · 后端 |
| --- | --- | --- | --- |
| 典型落点 | `examples/<op>/` | `src/transform/*.cc`、`tilelang/transform/__init__.py`、`tilelang/<backend>/pipeline.py` | `tilelang/<backend>/`、`src/<backend>/`、`CMakeLists.txt` |
| 上手难度 | 低（会 TileLang 即可） | 中（要读 TIR 与 pass 顺序） | 高（要同时懂 lowering、codegen、构建） |
| 需要硬件 | 视算子而定 | 通常不需要（可做 IR 级测试） | 通常需要在目标硬件上冒烟 |
| 最容易卡在哪 | 没有测试、只在本地能跑 | 把风格改动和语义改动混在一个 PR | 在 `engine/lower.py` 里加 target 分支 |

参考：`.agents/skills/tilelang-backend/references/integration-checklist.md` 的
Scope Matrix（新目标后端 / 同 target kind 的变体 / 新执行模式三条子路径）。
