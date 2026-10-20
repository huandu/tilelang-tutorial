# 版本锁定（全项目唯一事实来源）

所有章节、所有代码、所有引用，**必须**基于下表版本。任何地方出现「最新版」这类
模糊说法都算缺陷。若上游发布新 tag，只更新本文件并整体重审，不允许单章漂移。

| 组件 | 版本 | commit | 日期 | 本地路径 |
| --- | --- | --- | --- | --- |
| TileLang（主干） | `v0.1.15` | `a35f8ddf45eba16c21211ec8822d56ce5363036f` | 2026-09-30 | `_vendor/tilelang-v0.1.15/` |
| TileLang-Ascend（昇腾适配器） | `TileLang-ascend-v0.1.2.000-release`（仓库内 `VERSION` 文件写 `0.1.4`，以 tag 为准） | `89a9364e47a341fffdd65ed94786b1d4f9b6a78c` | 2026-09-08 | `_vendor/tilelang-ascend-v0.1.2.000/` |

辅助（**仅供对照，不是分析基线**）：

| 组件 | 版本 | commit | 用途 |
| --- | --- | --- | --- |
| TileLang（上一个 release） | `v0.1.14` | `7e3bbf3703e769479757eb7ecec2f2626f3cd0a8` | 回答「某能力是不是 v0.1.15 新增的」，见 `_vendor/tilelang-v0.1.14/` |

取自 tag 而不是 `main`：`main` 每天都在动，无法让多个章节、多个作者得出可复核的结论。
两个仓库都是 `git clone --depth 1 --branch <tag>`，工作树干净，`git status` 为空。

## v0.1.15 里有什么（写章节前先确认过的事实）

* `tilelang/ascend/` 与 `src/ascend/` **已经存在**：昇腾后端已进主干，不只是适配器仓库。
* `tilelang/{language,transform,layout,engine,backend,jit,cache,autotuner,carver,cuda,rocm,metal,cpu,webgpu,tools,profiler}/`
  全部存在；`examples/` 有 54 个子目录 + 3 个顶层文件（`conftest.py`、`pytest.ini`、
  `quickstart.py`，合计 57 项）；`docs/` 有 37 个 Markdown。
* `VERSION` 文件 = `0.1.15`。

## 引用规则

* 引用源码时写成 `_vendor/tilelang-v0.1.15/<相对路径>`，并给出行号区间（`path:120-138`）。
* 引用资料时写成 corpus 路径：`../tilelang-corpus/<类别>/raw/<文件>`。
* 引用上游链接时必须是本 corpus 里已存在或已核验可达的 URL；无法核验的写
  「未核验」并说明原因。
* 涉及「这个功能在不在 release 里」的判断，一律以本表版本为准；如果要讨论 `main`
  上更新的东西，必须显式标注「仅 main / 未进 v0.1.15」，并且放到第 30 章（前沿）里。

## 第五部分（第 32–39 章）的版本锁定

第五部分分析的是 **2026-09-30 DeepSeek 面向昇腾开源的整套基础设施**。它的对象不是一个
release tag，而是 13 个仓库在 **2026-10-01 的 `main` 快照**，以及官方发布文章的正文转录。
逐个仓库的 commit 与日期记录在 `../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md`，
站点构建时会把 `14-deepseek-ascend-stack/raw/<仓库>/...` 的引用解析成对应 commit 的 GitHub
blob/tree 链接，所以行号是稳定可复核的。

| 组件 | 快照 commit | 说明 |
| --- | --- | --- |
| TileKernels | `66258df6175d` | 用 TileLang 写的双后端算子库（第 33、34 章） |
| DeepGEMM-Ascend | `8491bbb4b8c0` | 昇腾 BLAS 与 MegaMoE（第 35 章） |
| DeepEP-Ascend | `3b25377d04b2` | 昇腾专家并行通信（第 38 章） |
| FlashMLA | `2e5429fc5653` | 含昇腾稀疏注意力与技术报告（第 36 章） |
| DeepSelect | `bfa4507d935f` | TopK 算子与算法报告（第 37 章） |
| DeepJIT / Engram / DeepSeek-V3.2-Exp / DualPipe / EPLB / clangd-ascend | `main` 快照 | 配套组件与论文 |
| DeepGEMM / DeepEP | `main` 快照 | NVIDIA 侧对照物 |

引用规则同上：写成 `../tilelang-corpus/14-deepseek-ascend-stack/raw/<仓库>/<相对路径>:<行号>`。

**与 v0.1.15 的关系**：TileKernels 明确要求 `TileLang >= 0.1.15`，所以第五部分与全书其他
部分共用同一个 TileLang 基线；凡是第五部分用到 TileLang 语言层面的结论，一律回到
`_vendor/tilelang-v0.1.15/` 核对，不要用 DeepSeek 仓库里的二手说法。

**验证边界（第五部分特别重要）**：本机既没有昇腾 NPU，也没有 NVIDIA GPU，也没有 CANN。
第五部分所有代码都只能**静态阅读**；所有性能数字都来自各仓库 README / 技术报告并注明
测量条件（硬件型号、CANN 版本、形状、口径）。任何地方不得出现本机实测的说法。

## 环境记录

写作时使用的本机环境（供代码章节标注测试条件）：

* macOS (Apple Silicon)，Python 3.11（miniconda）
* **没有** NVIDIA GPU，**没有** 昇腾 NPU

这条限制决定了两件事：

1. 任何需要真实加速器才能运行的代码，章节必须提供：可离线静态检查的版本（语法/API
   一致性）、代码阅读说明、以及「在有卡机器上如何验证」的步骤。**禁止宣称跑过没跑过的
   结果**。
2. 性能数字一律引用 corpus 内的原始出处（论文/官方文档），并标注来源 id。
