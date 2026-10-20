# 第 31 章习题答案

与 `exercises.md` 的 E1–E13 一一对应。每题给“解题思路 → 参考答案 → 常见错误”。

---

## E1

**思路**：先问“改的是哪一层”，再问“这一层在仓库里的落点”，最后问“哪条测试能锁住它”。

**参考答案**

| 改动 | 路径 | 主要落点 | 测试目录 |
| --- | --- | --- | --- |
| 1. `fused_add_rms_norm` 示例 | 路径一 · 算子/示例 | `examples/fused_add_rms_norm/`（实现 + 测试） | 自己的示例测试，被 CI 的 `pytest ../examples` 收集 |
| 2. `T.copy` 加参数 | 路径二 · 编译器 | `tilelang/language/copy_op.py`、对应 lowering/op 实现、相关 pipeline | `testing/python/language/`、`testing/python/transform/` |
| 3. 自动向量化 pass | 路径二 · 编译器 | `src/transform/<pass>.cc` + `tilelang/transform/__init__.py` + `tilelang/<backend>/pipeline.py` | `testing/python/transform/` |
| 4. 支持自研加速器 | 路径三 · 后端 | `tilelang/<backend>/`（dialect、manifest、pipeline、codegen）+ `src/<backend>/` + CMake | `testing/python/backend/`、`testing/python/target/` |

**常见错误**：把第 2 题当成路径一（因为“只是加个参数”）；把第 4 题只写成
`tilelang/<backend>/codegen.py`（后端要四块实现区一起看）。

---

## E2

**思路**：直接列 `.github/ISSUE_TEMPLATE/` 的文件，再读每个 YAML 的 `validations.required`。

**参考答案**

四个模板：

* `bug-report.yml`：必填“已读文档 + 已搜索 issue”两个复选、TileLang 版本、系统信息、
  问题描述、可复现的最小代码；traceback 与 expected behavior 是可选。
* `feature-request.yml`：必填“已搜索 issue”复选与 `Motivation`；`Solution`、
  `Alternatives`、`Additional context` 可选。
* `questions.yml`：必填两个复选与问题正文。
* `release-plan.yml`：版本必填；里程碑、范围、任务、就绪检查、备注为可选项。

`config.yml` 里 `blank_issues_enabled: true` 表示**允许开空白 issue**，模板是引导而非
强制闸门。所以“没按模板填”不会被系统挡住，但内容不全的 issue 很难被处理。

**常见错误**：以为 `bug-report.yml` 的 traceback 是必填（它其实不是）；
以为没有 RFC 模板是因为漏了（不是，见 E3）。

---

## E3

**思路**：用“列目录 + 检索文件内容”两步验证，不要凭印象。

**参考答案**

（1）**没有 RFC 模板**。验证方法：`ls .github/ISSUE_TEMPLATE/` 得到
`bug-report.yml`、`config.yml`、`feature-request.yml`、`questions.yml`、
`release-plan.yml` 五个文件，没有任何与 RFC 相关的模板；再检索 `.github/` 下有没有
RFC 专属标签配置，也没有。corpus 里的 RFC 都是普通 issue
（`github-issue-2196.md`、`3178`、`2897`、`2708`）。

（2）判据（至少写两条即可）：改动会**改变已有代码的含义或行为**；改动**影响面跨模块**
（构建基座、公共 API、多个后端）；改动**需要先决定方向再写代码**（例如换 TVM、
改 reducer 所有权、定义 atomic 内存序）；改动**代价高、难以回退**。

（3）因为这个结论把“我观察到的”和“文件规定的”分开了。如果我写成“RFC 必须包含
某几节”，读者会以为那是上游要求；实际上仓库没写。把它标为无法确认，读者才知道
该去维护者那里确认，而不是照着一份自造规则提交。

**常见错误**：从 corpus 的 RFC 标题前缀 `[RFC]` 推断“有 RFC 标签就一定存在模板”。
标签和模板是两回事。

---

## E4

**思路**：先在 `CONTRIBUTING.md` 里找，找不到再去 `.agents/` 与后端文档里找。

**参考答案**

这条约束写在 `.agents/skills/tilelang-backend/SKILL.md`（以及它的
`references/integration-checklist.md` 第 3 节 Pass Pipeline 里），原文是
“Keep the complete pass order in `tilelang/<backend>/pipeline.py`”与
“Keep target dispatch out of `tilelang/engine/lower.py`”。

它更重要的一点是**不在 `CONTRIBUTING.md` 里**。主干仓库的 `CONTRIBUTING.md` 只覆盖
环境、格式化、lint 与测试入口；架构约束、后端边界、测试可移植性这些规则散落在
`.agents/skills/` 这类仓库内指南里。提醒是：**只读 `CONTRIBUTING.md` 会漏掉大量评审
会引用的规范。**

**常见错误**：答成 `docs/developer_guide/cpp_style.md`。那份文件管 C++ 命名与头文件，
不管 pass 顺序。

---

## E5

**思路**：读 `ci.yml` 的 job 边界（`lint:37`、`tests:77`、`ascend:442`、`cutedsl:509`）
与各自 `needs`、`if`。

**参考答案**

1. 四个 job：`lint`（无依赖）、`tests`（`needs: [lint]`）、`ascend`（`needs: [lint]`）、
   `cutedsl`（`needs: [tests]`）。
2. **不是阻断性的**。lint job 里那一步的名字就是
   `C++ API Style Audit (warning only)`，运行
   `python maint/scripts/audit_cpp_api_style.py --github-warnings --limit 20`；
   `cpp_style.md:309-330` 也说明它是 advisory。真正阻断的是同一 job 里的
   `compileall` 与 `pre-commit`。
3. **不会**。`tests` 的 `if` 包含 `!github.event.pull_request.draft`，草稿 PR 跳过测试
   矩阵（`ascend`、`cutedsl` 同）。
4. Metal 矩阵跑的是 `pytest ... -k metal ./python`，靠测试名过滤。名字里没有 `metal`
   的 Metal 专属测试**不会**在这个 job 里被执行，要么改名，要么放到别的、由
   `requires_metal` 控制并被收集的路径里。

**常见错误**：以为 `github.repository_owner == 'tile-ai'` 会跳过 fork PR。它是防止
fork 仓库自己同步时白跑 runner；向 `tile-ai/tilelang` 提的 PR 仍然满足条件。

---

## E6

**思路**：读 `maint/scripts/test_perf_regression.py:20-29`，再读
`pr-regression-test-bot.yml` 的 `permissions-check`。

**参考答案**

1. `THRESHOLD` 默认 `0.90`，`MIN_COUNT` 默认 `2`。
2. 注释记录了在 33 份 bot 报告、1901 个样本上的校准：20 份是干净的；6 份是真回归
   （都发生在 PR #2464），表现为 **14–23 个 benchmark 同时崩**；另外 7 份只有**一个**
   离群点（低到 0.044），其余 benchmark 都是 1.000。`MIN_COUNT=2` 正好把这两组分开，
   在 0.80–0.95 的阈值区间内做到 6/6 抓到真回归、0 误报。设成 1 就会被单点噪声触发。
3. 只有仓库权限为 `admin` / `maintain` / `write` 的人评论 `@regression-perf` 才触发。
   外部贡献者**不能自己触发**，只能在 PR 里请求维护者跑。

**常见错误**：把 `run_perf_regression.sh` 的默认阈值 1.0 当成 bot 的阈值。两者是两个
脚本、两套默认值。

---

## E7

**思路**：逐条读模板第 1 节的勾选项，注意“有参考实现”与“无参考实现”是两个分支。

**参考答案**

1. 有 AscendC 参考实现：TileLang 性能要达到 AscendC 的 **50% 以上**；
   若另有 GPU 基线，候选耗时不超过其 **2 倍**。
2. 没有 AscendC 参考实现：要检查仓库里的性能反模式清单
   （`.agents/skills/tilelang-perf-optimization/references/performance-antipatterns.md`），
   并记录保留项及原因。
3. **不能合入**。模板要求“适用时 AscendC 与 PTO 双后端均已通过；不支持的
   dtype/shape/后端行为已显式说明，没有静默跳过”。只有单后端通过且没有说明，
   属于未满足合入标准。

**常见错误**：只说“做了性能测试”而不给 50% 与 2 倍这两个数字；或忽略
“无参考实现”分支也要留证据。

---

## E8

**思路**：把 C++ 注册、Python `_ffi_api`、pipeline 挂载三段连起来。

**参考答案**

1. `tilelang.transform.LoopUnswitching()`（`tilelang/transform/__init__.py`）→
   `_ffi_api.LoopUnswitching()`（`tilelang/transform/_ffi_api.py` 由 FFI 生成/暴露）→
   native 全局函数名 `"tl.transform.LoopUnswitching"`（由
   `src/transform/loop_unswitching.cc` 末尾的 `TVM_FFI_STATIC_INIT_BLOCK` 注册）→
   `CreatePrimFuncPass(pass_func, 0, "tl.LoopUnswitching", {})` 返回的 pass。
2. 只加 `.cc`：native 侧能编译，但 Python 侧没有 `LoopUnswitching` 这个符号，
   调用会 `AttributeError`；FFI 全局名虽然存在，也没有正常的 Python 入口。
3. **不会自动执行**。pass 只有被挂进某条 `PassPipeline` 才会应用。
   以 v0.1.15 为例，CUDA 的流水线常以 `tilelang/cuda/pipeline.py` 的
   `CUDAPassPipelineBody` 为入口，昇腾的是 `tilelang/ascend/pipeline.py` 的
   `AscendPassPipelineBody`。

**常见错误**：认为“注册了 FFI 全局函数”就等于“接进编译流程”。注册与挂载是两件事。

---

## E9

**思路**：直接跑脚本，读输出；注意路径前缀规则与 `ascend` 后缀规则。

**参考答案**

（1）

| 路径 | 类别 |
| --- | --- |
| `examples/gemm/example_gemm.py` | operator / example |
| `src/transform/lower_tile_op.cc` | compiler pass |
| `tilelang/ascend/pipeline.py` | backend pipeline (Ascend) |
| `testing/python/transform/test_tilelang_transform_layout_inference.py` | tests |

（2）合并后的门禁包含：examples、transform unit tests、backend contract tests、
ascend examples + unit tests、ascend PR template、layout golden harness（路径里含
`layout` 时）、format + lint。其中**需要 maintainer 触发**的是性能回归
（`@regression-perf`）；脚本对这次输入不会列出它，因为没有任何一条路径命中
性能相关规则。这一点本身就是个提醒：**性能门禁不会自动出现**。

（3）换成 `tilelang/cuda/pipeline.py` 后，类别变为 `backend pipeline`（没有 Ascend 后缀），
不再出现 `ascend examples + unit tests` 与 `ascend PR template` 两条门禁，
其余门禁（transform unit tests、backend contract tests、format + lint）保持不变。

**常见错误**：把 `testing/...` 也判成 transform 门禁。脚本对 `testing/` 只列 lint，
因为“测试文件本身”不触发测试要求。

---

## E10

**思路**：先想清楚新原语会改变什么，再逐节填；重点是“非目标”和“兼容性”不能空。

**参考答案（要点）**

```markdown
# [RFC] Add T.tile.scan for last-dimension prefix sums

Source: https://github.com/tile-ai/tilelang/issues/<n>
Status: draft
Touched files (planned): tilelang/language/, src/op/, tilelang/cuda/pipeline.py,
                         testing/python/language/

## Motivation
Attention/SSM kernels need an in-tile prefix scan; today it is done with a
hand-written loop or an external call, which blocks vectorization.

## Current Behavior
There is no tile-level scan primitive; users write T.serial loops, and the
vectorizer cannot prove the recurrence.

## Proposal
    T.tile.scan(src, dst, dim=-1, op="add")
Adds a tile op that lowers to (a) an intra-thread serial scan plus a
block-level offset pass, or (b) a hardware primitive where available.

## Compatibility and Impact
New API only; no existing name changes. Codegen must be added for every
backend that claims support, otherwise raising an actionable error.

## Testing
testing/python/language/ for lowering, testing/python/transform/ for the new
pass, numerical fixture in testing/python/kernel/.

## Risks and Rollback
Risk: silent precision drift in fp16 accumulation. Rollback: revert the
primitive and its pipeline hook; kernels keep the old manual path.

## Non-goals
Not a general tensor scan; no reverse scan; no cross-CTA scan.

## Alternatives
Express it as T.cumsum sugar; rejected because it hides the lowering choice.

## Open Questions
Should the op be legal on fragment buffers or shared memory only?

## Staging
Land lowering + tests, then the backend-specific fast paths.
```

运行 `python3 rfc_lint.py scan-rfc.md` 应输出 `ok: scan-rfc.md`。

（3）示例理由：“非目标”明确写出不做反向 scan、不做跨 CTA scan，是为了防止评审时
范围被不断拉大；写清边界，评审者才能在有限范围内判断方案是否成立。

**常见错误**：只填 `Proposal`，把 `Motivation` 写成“因为很方便”；或者忘了给
`T.tile.scan` 的调用示例，被 `rfc_lint.py` 的“提到 T.* API 但没有代码块”规则抓到。

---

## E11

**思路**：脚本判定的是“未勾选且没有原因”，因此要观察两种输入的差别。

**参考答案**

（1）把某条改成例如：

```markdown
- [ ] 已跑改动路径对应的测试套件
```

运行 `python3 checklist_gate.py self-check.md`，会打印
`unexplained: <行号>: 已跑改动路径对应的测试套件`，退出码 1。

（2）改成：

```markdown
- [ ] 已跑改动路径对应的测试套件
  - 原因: 本机无 NPU，待 CI 的 ascend job 覆盖，见 #1234
```

再跑，输出 `ok: self-check.md`，退出码 0。

（3）“勾上但没做”会把一个**未验证的状态**伪造成已验证。下游（评审者、后续修改者）
会基于这个错误前提做决定；而不勾至少保留了“这里有缺口”的信息。这也对应昇腾仓库
`AGENTS.md` 的第一条通用原则：“未验证的结果，不得表述为『已完成』或『已通过』”。

**常见错误**：以为脚本能判断原因是否成立。它只检查“有没有写原因”。

---

## E12

**思路**：按“改动落在哪个 pass → 哪个测试目录 → 哪份黄金工具”的顺序规划。

**参考答案（要点）**

1. 单元测试放 `testing/python/transform/test_tilelang_transform_layout_inference.py`
   （或按现有命名习惯新增 `test_tilelang_transform_*.py`）。文件名要描述**保护的行为**，
   不是“新功能”。
2. 断言落在 **IR 节点**上：检查 layout 映射、区域表达式、buffer 身份，
   不要匹配 IR printer 的文本。`testing/ascend/README.md` 的原话是
   “Inspect IR nodes instead of matching the IR printer's text.”
3. **需要**跑 `maint/layout_inference/run.py`。如果 golden 大面积变化，`--record` 之后
   必须逐条读 `expected/` 的 diff，确认每个变化的 layout 都是有意为之；
   记录不是批准。若同时改动了评分公式，还要跑 `--cute` 保持
   `cute_model.py` 与 `layout_cost_model.cc` 同步。
4. `maint/precision/` **通常不需要**：它面向数学函数/精度的 lowering 误差表，
   与布局记录方式无关。除非这个改动影响了向量化宽度或数值路径，才需要补。
5. **需要**考虑性能回归：布局改动会改变访存与向量化。本地跑
   `bash maint/scripts/run_perf_regression.sh`；在 PR 上请维护者评论
   `@regression-perf`，并把 bot 回帖的结果贴在 PR 里。

**常见错误**：直接用 `--record` 覆盖 golden 就交差；或者只跑
`testing/python/transform/`，忘了布局黄金用例。

---

## E13

**思路**：把第 10 节的清单具体化到你自己的环境，并诚实标注跑不了的部分。

**参考答案（模板）**

1. 我会选路径一（算子/示例）：`♥ help` 标签里最常见的是“某算子缺示例/缺边界”，
   反馈周期短，且不需要先读懂 pass 框架。
2. Step 2–8 的命令按第 10 节照抄，替换掉仓库地址、venv 路径与设备参数；例如

   ```bash
   git clone --recurse-submodules git@github.com:<me>/tilelang.git
   cd tilelang
   git remote add upstream git@github.com:tile-ai/tilelang.git
   uv venv --seed .venv && source .venv/bin/activate
   uv pip install --requirements requirements-dev.txt
   pre-commit install --install-hooks
   python3 -m pip install --no-build-isolation --verbose --editable .
   bash format.sh
   pre-commit run --all-files
   python3 -m pytest testing
   ```

3. 本地无卡时的 PR 写法示例：

   ```text
   How verified:
   - bash format.sh            → pass
   - pre-commit run --all-files
   - pytest testing/python/transform/test_x.py -x → pass (IR-level, no GPU)

   Not run (no local NVIDIA GPU):
   - examples correctness on CUDA
   - performance regression
   Requesting maintainer to trigger @regression-perf.
   ```

**常见错误**：把“没跑”写成省略，或者用“should pass”这类推测语气替代真实的
验证状态。
