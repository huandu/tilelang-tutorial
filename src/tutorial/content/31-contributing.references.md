# 第 31 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
文中不出现“最新版”这类表述；讨论 `main` 上更新的东西时会显式标注。

写作环境：macOS (Apple Silicon)、Python 3.11；**没有 NVIDIA GPU，也没有昇腾 NPU**。
本章没有需要真实硬件才能得出的结论，因此没有硬件验证项；所有代码示例的最高验证级别
是“CPU 逻辑验证”。

“已核验（YYYY-MM-DD）”的含义：在锁定版本的工作树上用 `ls` / `grep -n` / `sed -n`
定位到对应行并读完内容；行号区间与文件内容一致。**这不是运行验证。**

---

## A. 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

### A1 · 贡献入口与流程

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `CONTRIBUTING.md:18-25` | 报 bug 的要求：先搜索、要最小复现、要清楚说明 | 已核验（2026-10-01）：`grep -n` 定位小节，逐行读 |
| S2 | `CONTRIBUTING.md:27-29` | 提问走 issue | 已核验（2026-10-01） |
| S3 | `CONTRIBUTING.md:31-38` | `♥ help` 标签；“每个 PR 尽量带测试与文档” | 已核验（2026-10-01） |
| S4 | `CONTRIBUTING.md:40-62` | 编码风格：`bash format.sh --files <changed-file>...`、Ruff、clang-format、C++ 风格指南、C++ API 审计脚本 | 已核验（2026-10-01） |
| S5 | `CONTRIBUTING.md:64-93` | 开发环境：fork、`--recurse-submodules`、`git remote add upstream`、`uv venv --seed`、`requirements-dev.txt`、`pre-commit install --install-hooks` | 已核验（2026-10-01） |
| S6 | `CONTRIBUTING.md:95-109` | editable 安装与卸载 | 已核验（2026-10-01） |
| S7 | `CONTRIBUTING.md:111-117` | lint：`pre-commit run --all-files` | 已核验（2026-10-01） |
| S8 | `CONTRIBUTING.md:119-127` | 本地测试：`python3 -m pytest testing` | 已核验（2026-10-01） |
| S9 | `CONTRIBUTING.md:129-135` | `Build Wheels` 与 `Documentation` 两节内容为 `_TBA_` | 已核验（2026-10-01）：这两节**没有**实质内容，因此本章不描述“wheels/文档贡献流程” |

### A2 · issue / PR 模板

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S10 | `.github/ISSUE_TEMPLATE/bug-report.yml` | 必填：已读文档、已搜索 issue、版本、系统信息、描述、最小复现 | 已核验（2026-10-01）：逐字段读 `validations.required` |
| S11 | `.github/ISSUE_TEMPLATE/feature-request.yml` | `Motivation` 必填；`Solution`/`Alternatives` 可选 | 已核验（2026-10-01） |
| S12 | `.github/ISSUE_TEMPLATE/questions.yml` | 提问模板 | 已核验（2026-10-01） |
| S13 | `.github/ISSUE_TEMPLATE/release-plan.yml` | 发布计划与就绪检查 | 已核验（2026-10-01） |
| S14 | `.github/ISSUE_TEMPLATE/config.yml` | `blank_issues_enabled: true` | 已核验（2026-10-01） |
| S15 | `.github/`（目录清单） | **没有** `pull_request_template.md`，也**没有** RFC 模板 | 已核验（2026-10-01）：`ls -R .github/` |

### A3 · CI 与自动化

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S16 | `.github/workflows/ci.yml:1-14` | PR 触发事件（6 种）与 `workflow_dispatch` | 已核验（2026-10-01） |
| S17 | `.github/workflows/ci.yml:37-76` | `lint` job：`compileall`、`audit_cpp_api_style.py --github-warnings --limit 20`（warning only）、`pre-commit --all-files` | 已核验（2026-10-01）：逐 step 读 |
| S18 | `.github/workflows/ci.yml:77-441` | `tests` 矩阵：self-hosted nvidia `CUDA-auto` / self-hosted amd `ROCm-7.2` / macos-latest `Metal`，Python 3.12，`needs: [lint]` | 已核验（2026-10-01） |
| S19 | `.github/workflows/ci.yml:366-379` | examples 步骤：`pytest ../examples`，排除 `grouped_gemm` 与 `examples/ascend` | 已核验（2026-10-01） |
| S20 | `.github/workflows/ci.yml:380-392` | CUDA 单元测试：`pytest ./python` | 已核验（2026-10-01） |
| S21 | `.github/workflows/ci.yml:393-419` | ROCm：`./python` + 4 个 portable example | 已核验（2026-10-01） |
| S22 | `.github/workflows/ci.yml:420-441` | Metal：`pytest -k metal ./python` | 已核验（2026-10-01） |
| S23 | `.github/workflows/ci.yml:442-508` | `ascend` job：`[self-hosted, ascend]`，`USE_ASCEND=ON`，`pytest -n 4 ../examples/ascend` 与 `pytest -n 32 ascend` | 已核验（2026-10-01） |
| S24 | `.github/workflows/ci.yml:509-663` | `cutedsl` job：`needs: [tests]`，`TILELANG_TARGET=cutedsl` | 已核验（2026-10-01） |
| S25 | `.github/workflows/pr-regression-test-bot.yml:34-56` | 性能 bot 的权限门：评论含 `@regression-perf`，且评论者权限为 admin/maintain/write | 已核验（2026-10-01） |
| S26 | `.github/workflows/pr-regression-test-bot.yml:58-320` | checkout `refs/pull/<n>/merge`、装 main 基线、跑 `test_perf_regression.py`、回帖 Markdown | 已核验（2026-10-01） |
| S27 | `.github/workflows/pr-reminder-bot.yml` | 新 PR 自动提醒 `pre-commit run --all-files` | 已核验（2026-10-01） |
| S28 | `.github/workflows/publish-docs.yml` | 文档在 PR merge 后（或手动）构建 | 已核验（2026-10-01） |
| S29 | `.github/dependabot.yml` | 每周一检查 GitHub Actions，commit 前缀 `[CI]` | 已核验（2026-10-01） |

### A4 · 代码规范与格式化

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S30 | `docs/developer_guide/cpp_style.md:1-10` | 以 TVM 惯例为基线；只管 `src/` 与 TileLang 自有头文件；不动 `3rdparty/` | 已核验（2026-10-01） |
| S31 | `docs/developer_guide/cpp_style.md:11-30` | 格式化：`bash format.sh --files`；`.clang-format` 改动单独提 PR；风格改动与行为改动分开 | 已核验（2026-10-01） |
| S32 | `docs/developer_guide/cpp_style.md:32-50` | 命名表（11 类实体） | 已核验（2026-10-01）：逐行对照表格 |
| S33 | `docs/developer_guide/cpp_style.md:72-114` | TVM Object/FFI：handle 跨边界、裸 `*Node` 局部、`.same_as`、`ObjectPtrHash`/`ObjectPtrEqual`、`Optional<T>` | 已核验（2026-10-01） |
| S34 | `docs/developer_guide/cpp_style.md:115-149` | ObjectNode 反射字段与 FFI 可见名 | 已核验（2026-10-01） |
| S35 | `docs/developer_guide/cpp_style.md:150-170` | 函数参数：不要用 `T`；`const&` | 已核验（2026-10-01） |
| S36 | `docs/developer_guide/cpp_style.md:171-201` | 头文件：公开头文件不要 `using namespace` | 已核验（2026-10-01） |
| S37 | `docs/developer_guide/cpp_style.md:202-259` | 命名空间策略是评审指南而非机械重写 | 已核验（2026-10-01） |
| S38 | `docs/developer_guide/cpp_style.md:280-295` | 错误处理：`ICHECK`/`LOG(FATAL)` 与用户级 validation 区分 | 已核验（2026-10-01） |
| S39 | `docs/developer_guide/cpp_style.md:309-330`、`maint/scripts/audit_cpp_api_style.py:601-623` | C++ API 审计脚本：advisory。脚本只在传 `--fail-on-findings` 时返回 1；CI 的调用（`ci.yml:64-69`）没传该参数，因此退出码恒为 0 → 仅告警 | 已核验（2026-10-01）：读 `main()` 的返回表达式 |
| S40 | `docs/developer_guide/cpp_style.md:331-349` | 例外与增量迁移政策：不要把大范围格式化与行为改动混在一起 | 已核验（2026-10-01） |
| S41 | `.clang-format:1-8` | `BasedOnStyle: LLVM`、`IndentWidth: 2`、`ColumnLimit: 80`、`Standard: c++17` | 已核验（2026-10-01） |
| S42 | `.pre-commit-config.yaml:10-60` | 钩子清单：clang-format `v23.1.0`、ruff `v0.16.6`、codespell `v2.4.3`、pymarkdown `v0.9.39` | 已核验（2026-10-01） |
| S43 | `pyproject.toml:176-201` | Ruff：`target-version=py310`、`line-length=140`、`select=[E,W,F,UP,FA,B,SIM]`（`I` 被注释） | 已核验（2026-10-01） |
| S44 | `pyproject.toml:194-197` | `testing/**.py` 与 `examples/**.py` 忽略 `UP`/`FA` | 已核验（2026-10-01） |
| S45 | `pyproject.toml:215-236` | Ruff 忽略项列表 | 已核验（2026-10-01） |
| S46 | `pyproject.toml:238-255` | pytest markers：`perf`、`slow`、`cpu`、`gpu`、`multi_gpu`、`cuda`、`llvm`、`metal`、`rocm` | 已核验（2026-10-01） |
| S47 | `requirements-lint.txt:1-5` | 工具 pin：clang-format `21.1.8`、codespell `2.4.1`、ruff `0.14.14` | 已核验（2026-10-01）：与 S42 的 rev 不一致，见正文 §8.4 |
| S48 | `format.sh` | `--files` / 默认 changed / `--all` 三种模式；末尾检查工作区是否变脏 | 已核验（2026-10-01）：读全文 |

### A5 · 测试、维护工具与基准

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S49 | `testing/`（目录树） | `conftest.py`、`cpp/`（空）、`ascend/`、`python/` | 已核验（2026-10-01）：`find testing -type f \| wc -l` = 487；`ls testing/cpp` 为空 |
| S50 | `testing/conftest.py:44-75` | `--run-perf` 选项；默认跳过 `perf` 标记测试 | 已核验（2026-10-01） |
| S51 | `tilelang/testing/__init__.py:11-20,105-240` | re-export `requires_cuda/rocm/metal/llvm`；定义 `requires_cdna`、`requires_cuda_or_cdna`、`requires_gfx950`、`requires_cuda_compute_version*` | 已核验（2026-10-01） |
| S52 | `tilelang/testing/perf_regression.py` | 结果收集与 `__TILELANG_PERF_RESULTS_JSON__=` 标记；`TL_PERF_REGRESSION_FORMAT=json` | 已核验（2026-10-01） |
| S53 | `maint/scripts/test_perf_regression.py:22-32` | 阈值校准注释：33 份报告/1901 样本；`THRESHOLD=0.90`、`MIN_COUNT=2` | 已核验（2026-10-01） |
| S54 | `maint/scripts/run_perf_regression.sh:1-30,55,66-67` | 本地性能回归：`BASE_REF=origin/main`、`FAIL_ON_REGRESSION=0`、`REGRESSION_THRESHOLD=1.0`、wheel 缓存等 | 已核验（2026-10-01） |
| S55 | `maint/scripts/ascend_perf_regression.py:1-24` | 昇腾回归需要常驻 worker：基线可能硬崩（LOG(FATAL)/abort/hang） | 已核验（2026-10-01） |
| S56 | `maint/scripts/run_local_ci_test.sh:1-30` | 本地复现 CUDA/ROCm/Metal 流程；worker 数按 GPU 数计算 | 已核验（2026-10-01） |
| S57 | `maint/scripts/run_local_ci_test_ascend.sh:1-52` | 本地复现昇腾 CI：4 个 example worker、32 个 unit worker | 已核验（2026-10-01） |
| S58 | `maint/layout_inference/README.md` | 布局黄金用例工具：`run.py`、`--record`、`--anchor`、`--cute`；“Recording is not approval” | 已核验（2026-10-01）：读全文 |
| S59 | `maint/precision/README.md` | 数学函数精度对照表（TileLang/Triton/PyTorch/CUDA vs double） | 已核验（2026-10-01） |
| S60 | `maint/host_checks/README.md` | 10 个 host 校验复现脚本 + `run_all.py` | 已核验（2026-10-01） |
| S61 | `testing/ascend/README.md:1-14` | “每个回归放在拥有该行为的第一道边界”；共享行为放 `testing/python` | 已核验（2026-10-01） |
| S62 | `testing/ascend/README.md:24-114` | pass/设备边界与调度测试的组织、以及本地运行命令 | 已核验（2026-10-01） |
| S63 | `benchmark/`（目录树） | `blocksparse_attention`、`compile_speed`、`mamba2`、`matmul`、`matmul_fp8`、`matmul_metal` | 已核验（2026-10-01） |

### A6 · 编译期与后端骨架（路径二、路径三）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S64 | `.agents/skills/tilelang-submit-pr/SKILL.md` | PR 预检：diff 审计、测试可移植性、`requires_*` 用法、只报真实跑过的验证 | 已核验（2026-10-01）：读全文 |
| S65 | `.agents/skills/tilelang-backend/SKILL.md` | 后端四块实现区模型；pass 顺序留在后端包；`engine/lower.py` 不放 target 分派 | 已核验（2026-10-01）：读全文 |
| S66 | `.agents/skills/tilelang-backend/references/integration-checklist.md:1-378` | Scope Matrix、Preflight Decisions、9 个步骤、Definition of Done | 已核验（2026-10-01）：读全文 |
| S67 | `tilelang/backend/README.md` | 后端架构与所有权模型 | 已核验（2026-10-01）：文件存在 |
| S68 | `testing/python/backend/test_tilelang_backend_module.py`、`test_tilelang_backend_auto_schedule.py` | 后端 contract 测试落点 | 已核验（2026-10-01） |
| S69 | `src/transform/loop_unswitching.cc`（末尾） | `TVM_FFI_STATIC_INIT_BLOCK` 注册 `tl.transform.LoopUnswitching` | 已核验（2026-10-01）：读文件末尾 |
| S70 | `tilelang/transform/__init__.py`（`LoopUnswitching`） | Python wrapper 与 `_ffi_api` 调用 | 已核验（2026-10-01） |
| S71 | `tilelang/cuda/pipeline.py:129-142` | CUDA pass 顺序片段：`PipelinePlanning`→`InjectSoftwarePipeline`→`LayoutInference`→`LowerTileOp` | 已核验（2026-10-01） |
| S72 | `tilelang/ascend/pipeline.py:108-114` | 昇腾 pass 顺序片段：`EstimateLatency`→`AutoSchedule`→`PrepareMultiBuffer`→`InsertSync`→`MaterializeMultiBuffer` | 已核验（2026-10-01） |
| S73 | `testing/python/transform/`（52 个文件） | 每个 pass/行为一个测试文件 | 已核验（2026-10-01）：`ls testing/python/transform \| wc -l` = 52 |
| S74 | `examples/`（54 个子目录 + 3 个顶层文件） | 一个算子一个目录的惯例 | 已核验（2026-10-01）：`ls` |

---

## B. 锁定的源码引用（`_vendor/tilelang-ascend-v0.1.2.000/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| A1 | `AGENTS.md:1-13` | 项目定位：Python DSL + `@tilelang.jit`，Developer/Expert 两种模式 | 已核验（2026-10-01） |
| A2 | `AGENTS.md:59-77` | 通用原则：如实报告、禁止伪完成；先验证再下结论；区分事实/推断/建议；最小必要改动 | 已核验（2026-10-01） |
| A3 | `AGENTS.md:79-115` | 核心算子开发原则 8 条（不猜 API、从示例入手、内存层级、新算子独立目录等） | 已核验（2026-10-01） |
| A4 | `AGENTS.md:117-121` | 开发规范：Python PEP 8 + 类型注解 + 行宽 100；C++ Google Style + clang-format + 行宽 100 | 已核验（2026-10-01）：与主干 `.clang-format` 的 80 列不同，见正文 §7.1 |
| A5 | `CONTRIBUTING.md:16-19` | 报 bug 要求：最小复现 + 清楚说明 | 已核验（2026-10-01） |
| A6 | `CONTRIBUTING.md:26-32` | `./format.sh` + 带测试与文档 | 已核验（2026-10-01） |
| A7 | `CONTRIBUTING.md:34-50` | `python setup.py install`；`python -m pytest testing` | 已核验（2026-10-01）：注意这比主干 `CONTRIBUTING.md` 简略 |
| A8 | `.github/pull_request_template.md:1-5` | 顶部说明：只勾实际完成的项，跳过项写原因与跟踪 Issue | 已核验（2026-10-01） |
| A9 | `.github/pull_request_template.md:10-19` | 变更类型标签 | 已核验（2026-10-01） |
| A10 | `.github/pull_request_template.md:21-31` | 算子合入标准：默认 target、双后端、性能 50%/2×、精度性能报告、2 分钟用例预算 | 已核验（2026-10-01）：逐条读 |
| A11 | `.github/pull_request_template.md:35-47` | 框架/接口合入标准：双后端验证、marker 分流、`ci/operator_test_manifest.yaml` 登记 | 已核验（2026-10-01） |
| A12 | `.github/pull_request_template.md:49-56` | 提交规范：英文 commit + 类型标签；检视意见逐条回复 | 已核验（2026-10-01） |
| A13 | `ci/operator_test_manifest.yaml:1-15` | 算子源文件 → pytest 测试的映射表；登记只在测试文件真实存在时生效 | 已核验（2026-10-01） |
| A14 | `.github/workflows/ci.yml` | 适配器 CI（`on: [pull_request]`） | 已核验（2026-10-01）：文件存在，`format-check` 的 `if: false` 被禁用 |
| A15 | `.github/workflows/ci_ascend.yml:1-53` | 适配器 CI：按改动文件跑 Ruff format/lint 与 clang-format（版本 18） | 已核验（2026-10-01） |
| A16 | `format.sh:15-40,205` | YAPF + Ruff + clang-format；用 `requirements-lint.txt` 做**版本校验** | 已核验（2026-10-01） |
| A17 | `.agents/skills/tilelang-custom-skill/tilelang-github-operations/pr-workflow-guide.md` | PR 工作流默认目标：`tile-ai/tilelang-ascend` 的 `ascendc_pto` 分支；commit 加 `🤖` 标识 | 已核验（2026-10-01） |
| A18 | `VERSION` | 文件内容为 `0.1.4` | 已核验（2026-10-01）：与 VERSIONS.md 记录的 tag `v0.1.2.000-release` 不同，已在正文 §7.1 说明 |
| A19 | `README.md`（Latest News 段） | 发布节奏：DeepSeek V4 kernels、Flash Attention guide、wheel 安装、ACLGraph、PTO backend 等按日期滚动 | 已核验（2026-10-01） |
| A20 | `maint/scripts/performance.py`、`maint/scripts/ci_performance.py` | 适配器的性能脚本（GEMM 配置与 `Latency`/`TFlops` 解析） | 已核验（2026-10-01）：读文件头 |
| A21 | `testing/{python,cpp}`、`docs/`、`tox.ini` | 适配器有独立的测试树与文档树 | 已核验（2026-10-01）：`ls` |

---

## C. corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/developer_guide/cpp_style.md` | https://tilelang.com | C++ 风格指南的官方文档快照；与 `_vendor/tilelang-v0.1.15/docs/developer_guide/cpp_style.md` **逐字节相同** | 已核验（2026-10-01）：`diff -q` 返回无差异 |
| R2 | `../tilelang-corpus/08-frontier-unmerged/raw/_index.md` | https://github.com/tile-ai/tilelang/issues | 14 条前沿资料索引，含 5 个 RFC/路线图 issue 与 5 个 `[Ascend]` PR | 已核验（2026-10-01） |
| R3 | `../tilelang-corpus/08-frontier-unmerged/raw/github-issue-2196.md` | https://github.com/tile-ai/tilelang/issues/2196 | RFC 范例：TVM 迁移；问题重构、可核验事实、拆分提案 | 已核验（2026-10-01）：读正文 |
| R4 | `../tilelang-corpus/08-frontier-unmerged/raw/github-issue-3178.md` | https://github.com/tile-ai/tilelang/issues/3178 | RFC 范例：CUDA cache policy；状态枚举、拟议签名、非目标、失败行为 | 已核验（2026-10-01）：读正文 |
| R5 | `../tilelang-corpus/08-frontier-unmerged/raw/github-issue-2897.md` | https://github.com/tile-ai/tilelang/issues/2897 | 语义级 RFC：reducer 所有权 | 已核验（2026-10-01）：文件存在、读标题与摘要 |
| R6 | `../tilelang-corpus/08-frontier-unmerged/raw/github-issue-2708.md` | https://github.com/tile-ai/tilelang/issues/2708 | 语义级 RFC：atomic 内存序 | 已核验（2026-10-01）：文件存在、读标题与摘要 |
| R7 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3308.md` | https://github.com/tile-ai/tilelang/pull/3308 | `[Ascend]` PR：Ascend 950 backend | 已核验（2026-10-01）：读标题与摘要 |
| R8 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3310.md` | https://github.com/tile-ai/tilelang/pull/3310 | `[Ascend]` PR：Ascend950 的 PTO backend | 已核验（2026-10-01） |
| R9 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3325.md` | https://github.com/tile-ai/tilelang/pull/3325 | `[Ascend]` PR：`T.PersistentKernel` + AutoPersistent pass；含 review 的 pre-merge checks 记录 | 已核验（2026-10-01）：读正文与检查结果段 |
| R10 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3320.md` | https://github.com/tile-ai/tilelang/pull/3320 | `[Ascend]` PR：AutoSimtVF pass | 已核验（2026-10-01） |
| R11 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3329.md` | https://github.com/tile-ai/tilelang/pull/3329 | `[Ascend]` PR：TileLang agents | 已核验（2026-10-01） |
| R12 | `../tilelang-corpus/12-operator-libraries/raw/github-tile-ai-tilelang-benchmark.md` | https://github.com/tile-ai/tilelang-benchmark | 官方 benchmark 仓库快照（README 顶部写明 Archived，指向 TileOps） | 已核验（2026-10-01）：读 README 与形状表 |
| R13 | `../tilelang-corpus/12-operator-libraries/raw/tilelang-benchmark/README.md` | 同上 | GEMM V0–V7 / M0–M7、FlashAttention FA0–FA4、Linear Attention CC/CT 形状表 | 已核验（2026-10-01） |
| R14 | `../tilelang-corpus/12-operator-libraries/raw/tilelang-benchmark/`（目录） | 同上 | `hopper_benchmark/`、`ampere_benchmark/`、`ada_benchmark/`、`cdna_benchmark/` 按架构分层 | 已核验（2026-10-01）：`find -maxdepth 2 -type d` |
| R15 | `../tilelang-corpus/07-tilelang-ascend-adapter/raw/_index.md` | https://github.com/tile-ai/tilelang-ascend | 适配器仓库、MLIR 昇腾路线、TileFoundry 的来源索引 | 已核验（2026-10-01） |
| R16 | `../tilelang-corpus/90-synthesis/03-ascend-route.md` | 本书内部综述 | 昇腾两仓库现状：“适配器是主战场，主干已接收 `[Ascend]` 系列 PR” | 已核验（2026-10-01）：该文件自身注明基于 corpus `main` 快照 `994b44e`，与锁定版本可能有差异 |
| R17 | `../tilelang-corpus/10-community-and-talks/raw/bilibili-tilelang-ascend-lessons.md` | B 站课程 | 适配器有 5 节视频课程（环境、编程基础、Developer 模式、调优调试、工程实践） | 已核验（2026-10-01） |

---

## D. 上游链接核验状态

| 链接 | 状态 |
| --- | --- |
| https://github.com/tile-ai/tilelang/issues?q=is%3Aissue+is%3Aopen+label%3A%22%E2%99%A5+help%22 | 未核验（写作环境未联网访问 GitHub 页面；链接文本抄自锁定版本 `CONTRIBUTING.md:33`） |
| https://github.com/tile-ai/tilelang/issues/2196 | 已核验（2026-10-01，通过 corpus 快照 R3） |
| https://github.com/tile-ai/tilelang/issues/3178 | 已核验（2026-10-01，通过 corpus 快照 R4） |
| https://github.com/tile-ai/tilelang/pull/3308 | 已核验（2026-10-01，通过 corpus 快照 R7） |
| https://github.com/tile-ai/tilelang-benchmark | 已核验（2026-10-01，通过 corpus 快照 R12） |
| https://pre-commit.com | 未核验（写作环境未联网；引用仅用于说明钩子框架） |

---

## E. “无法确认”的条目

本章有三处结论**无法在锁定版本里找到成文依据**，正文均已显式标注：

1. **两棵树的协作流程。** “能力先进 `tilelang-ascend` 适配器验证，再以 `[Ascend]` PR
   回主干”这条流程，在两个仓库的 `CONTRIBUTING.md`、`AGENTS.md`、`.github/` 里都没有
   成文规定。核验方式：对两个仓库检索 `上游`、`upstream`、`回主干`、`merge.*main`，
   无命中。能确认的只有：主干已有 `tilelang/ascend/`、`src/ascend/`、`examples/ascend/`、
   `testing/ascend/` 与独立的 `ascend` CI job；corpus 里存在 5 个 `[Ascend]` 前缀的
   PR 快照；适配器默认分支是 `ascendc_pto`。
2. **RFC 的强制章节。** 仓库没有 RFC 模板，也没有 RFC 标签配置。本章第 12 节的结构
   是从 corpus 里真实 RFC 反推的经验，不是上游要求。
3. **lint 工具的版本漂移。** `.pre-commit-config.yaml` 的 rev 与 `requirements-lint.txt`
   的 pin 不一致（clang-format `v23.1.0` vs `21.1.8`，ruff `v0.16.6` vs `0.14.14`，
   codespell `v2.4.3` vs `2.4.1`），而三处注释都声称“sync with requirements-lint.txt”。
   哪个是有意为之，仓库里没有说明。

另外两条属于“文件存在但没有实质内容”，因此本章不给结论：

* `CONTRIBUTING.md:129-135` 的 `Build Wheels` 与 `Documentation` 两节内容为 `_TBA_`，
  主干没有描述这两条贡献路径。
* `testing/cpp/` 在 v0.1.15 是空目录；C++ 侧是否计划恢复独立的 C++ 单测，仓库里
  没有说明。

---

## F. 待补资料

1. **缺少主干的 PR 模板。** 主干没有 `pull_request_template.md`，因此“一个主干 PR
   应该填什么”没有官方文本。本章第 10 节 Step 9 给出的四段式是写作建议，不是上游
   要求。建议后续版本抓取几个已合入的主干 PR（尤其是 `[Ascend]` 与非 `[Ascend]` 各一份）
   的正文与 review 记录，作为“主干 PR 实际长什么样”的补充证据。
2. **缺少 `[Ascend]` PR 的完整 diff 与评审意见全文。** corpus 里的 5 个 PR 快照是页面
   快照，含标题、摘要与部分检查结果（例如 `github-pr-3325.md` 里能看到 pre-merge
   checks 的计数），但没有完整 diff。要回答“一次后端 PR 到底动了多少文件”，
   还需要 PR 的 files-changed 列表。
3. **缺少 RFC 的讨论与收敛过程。** 现有快照是 issue 正文，没有后续评论。RFC 怎么从
   提案走到“可以开工”，需要评论串才能回答。
4. **缺少 `tilelang-ascend` 的 CI 运行记录。** 适配器仓库有
   `ci/operator_test_manifest.yaml` 与 `main`/`ci_ascend` 两条工作流，但没有运行记录
   快照，因此“每日全量任务实际跑哪些 marker”只能从 PR 模板的描述推断。
