# 图 3：一个 pass 的四层测试，以及每层“能证明什么、不能证明什么”（自绘）

四层不是可选项，而是四种**不同**的证据。缺哪一层，就有一类错误永远逃得掉。

```text
                    证据强度 / 运行成本
     便宜、快、离线                         昂贵、慢、要硬件
      │                                                          │
      ▼                                                          ▼
 ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  ┌──────────────────┐
 │ ④ IR diff     │  │ ③ 单元测试   │  │ ② 边界条件    │  │ ① 端到端对拍     │
 │   基线        │  │   单个 pass  │  │   同一 pass 的│  │   真编译 + 真跑  │
 │              │  │   的 before/  │  │   极端输入    │  │   数值对比        │
 │              │  │   after       │  │               │  │                  │
 └──────────────┘  └──────────────┘  └───────────────┘  └──────────────────┘
  证明：         证明：             证明：             证明：
  · 这次改动的    · pass 在给定      · 边界不被吞      · 整条 pipeline
    IR 变化是     IR 上确实做了      · 假警报不会      + codegen +
    你预期的      你预期的事           把检查关掉        运行时都活着
  不能证明：     不能证明：         不能证明：         不能证明：
  · 变化是对的   · 没被别的 pass    · 大规模输入下的   · 性能没退（见
  · 性能没退       之后覆盖          行为               图 4）
```

## 每层在本仓库里的现成范式

| 层 | 现成范式（`_vendor/tilelang-v0.1.15/`） | 关键写法 |
| --- | --- | --- |
| ④ IR diff 基线 | `testing/python/transform/test_tilelang_transform_hoist_broadcast_values.py:56-64` | 把 before/after 各写成一个 `@T.prim_func`，`BindTarget` 对齐后 `tvm.ir.assert_structural_equal(...)` |
| ③ 单元测试 | `testing/python/transform/test_tilelang_transform_verify_parallel_loop.py:23-43` | 手搓 `IRModule` → 调 pass → 断言 stderr 文本 |
| ② 边界条件 | `testing/python/transform/test_tilelang_transform_verify_buffer_init.py:106-313` | 每个“反向用例”都是一条边界：extern 填充、global scope、参数 buffer、只在读取之后才写…… |
| ① 端到端对拍 | 同文件 `:447-483` | `@tilelang.testing.requires_cuda` + `tilelang.compile(...)`，验证新检查不打断真实编译 |

## 注意：④ 和 ③ 的差别比看起来大

* ④ 断言的是**结构相等**：IR 必须逐节点一样。它最严格，也最容易因为一次无关的
  命名/顺序变化而变红。所以基线要由脚本生成，不要手写。
* ③ 断言的是**行为**：看到某段文本、或某个 `Call` op 名字出现。它对文本格式敏感，
  但对实现细节宽容。

两者都写，是因为一种 bug 只有一种断言抓得住：

* “把不该改的地方改了”→ 只有 ④ 能抓；
* “该报的没报”→ 只有 ③ 能抓。

来源：作者自绘；表格里的行号取自 `_vendor/tilelang-v0.1.15/`（2026-10-01 核对）。
