# 图 2：昇腾 950 相关改动的落点（自绘）

```text
 2026-09-02        2026-09-08          2026-09-29 / 09-30        2026-09-30
 v0.1.14           tilelang-ascend     v0.1.15 tag               corpus 的 main 快照
 7e3bbf37          v0.1.2.000-release  a35f8ddf (#3309)           994b44e (#3310)
    │              89a9364e                │                        │
    │                   │                   │                        │
    │                   │                   │                        │
    │    A2/A3 适配器线  │                   │                        │
    ├───────────────────┤                   │                        │
    │  tile-ai/tilelang-ascend              │                        │
    │  (社区维护，另一棵树)                  │                        │
    │                                       │                        │
    │                                       │                        │
    │            昇腾进主干（AscendC 路线）   │                        │
    │            PR #3308  "Introduce Ascend  950 backend"            │
    │            ├─ tilelang/ascend/  (dialect)          ─── 在 ───▶ │
    │            ├─ src/ascend/codegen/codegen_ascend.cc      v0.1.15│
    │            ├─ src/ascend/transform/{auto_schedule,…}           │
    │            ├─ tilelang/profiler/msprof.py                      │
    │            └─ testing/ascend/, examples/ascend/, docs 的 README│
    │                                       │                        │
    │                                       │      PTO 路线           │
    │                                       │   PR #3310  "Add PTO backend for Ascend950"
    │                                       │   ├─ src/ascend/codegen/codegen_pto.{cc,h}
    │                                       │   ├─ src/ascend/codegen/rt_mod_pto.cc
    │                                       │   ├─ tilelang/contrib/ptodsl/  (9 文件)
    │                                       │   └─ tilelang/jit/adapter/pto/  (5 文件)
    │                                       │            └──── 只在 ──▶ │ 出现
    │                                       │                   main 快照，不在 v0.1.15
    │                                       │                        │
    │                                       │      其余未合入的 PR   │
    │                                       │   #3325 T.PersistentKernel + AutoPersistent
    │                                       │   #3320 AutoSimtVF pass
    │                                       │   #3329 Ascend 算子生成/调优 agents
    │                                       │   #3328 / #3322 / #3324 / #3321
    │                                       │            （corpus 未收录快照）
```

## 读法

* **横轴只有四个锚点**，都是可核验的 commit：`v0.1.14`、适配器 release、
  `v0.1.15` 的 tag commit、corpus 里 main 快照的 commit。
* 「在 v0.1.15」这一列是**结论**，不是位置：箭头指向它就代表文件确实出现在
  `_vendor/tilelang-v0.1.15/` 里（见 `code/01-state-audit/`）。
* 最下面那一组 PR 只有编号和标题，没有落点，因为 corpus 抓取时它们都还没进
  main 快照——**不要把它们画进 v0.1.15 或 main 那一列**。

## 图里哪些是事实

| 事实 | 出处 |
| --- | --- |
| `v0.1.15` 的 commit `a35f8ddf`，提交信息 `[Release] Bump version into 0.1.15 (#3309)` | `_vendor/tilelang-v0.1.15` 的 git 提交 |
| main 快照 commit `994b44e`，提交信息 `[Ascend] Add PTO backend for Ascend950 (#3310)` | `../tilelang-corpus/02-tilelang-core-repo/raw/tilelang` 的 git 提交 |
| `tilelang/ascend/`、`src/ascend/` 出现在 v0.1.15、不在 v0.1.14 | `ls` / `tree_diff.py` |
| PTO 的 20 个文件只出现在 main 快照 | `tree_diff.py --selftest` |
| `T.PersistentKernel` / `AutoSimtVF` / `agent/core` 三处均无命中 | `state_audit.py` |

## 图里哪些是推断

* `#3308` 与 `#3310` 的先后关系按 PR 编号与 commit 顺序推断：
  `#3309`（release）在 `#3310` 之前，而 `#3308` 的内容已在 `v0.1.15` 里。
* PR 提交时间取自 corpus 快照页面的「Sep 29 / Sep 30」字样，只精确到日。
