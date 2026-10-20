# 第 31 章 习题

每题标注难度 `★ / ★★ / ★★★` 与是否需要真实硬件。本章习题全部**不需要 GPU 或 NPU**：
它们检验的是你对流程、仓库结构、评审标准的理解，以及本地就能跑的工具。

---

## E1（★，不需要硬件）三种路径的判定

下面四个改动各属于本章的哪条贡献路径？分别写出你会去改的文件，以及会被哪个测试
目录覆盖。

1. 用 TileLang 实现一个 `fused_add_rms_norm` 算子并作为官方示例提交。
2. 给 `T.copy` 增加一个新的 `load_cache_policy` 参数。
3. 新增一个把 `T.Parallel` 循环自动向量化的 pass。
4. 让 TileLang 支持一个你公司自研的加速器。

---

## E2（★，不需要硬件）issue 模板

`.github/ISSUE_TEMPLATE/` 在 v0.1.15 里有哪几个模板？各有哪些必填项？
`config.yml` 里的 `blank_issues_enabled: true` 意味着什么？

---

## E3（★★，不需要硬件）RFC 的门槛

（1）仓库里有没有 RFC 模板？你怎么验证？

（2）从 corpus 里已有的 RFC 看，哪些类型的改动值得先写 RFC？给出至少两条判据。

（3）本章说「RFC 需要哪些章节」属于「无法确认」。为什么这个结论本身是有价值的？

---

## E4（★★，不需要硬件）约束写在哪里

「完整的 pass 顺序必须留在 `tilelang/<backend>/pipeline.py`，不要把 target 分派写进
`tilelang/engine/lower.py`」这条约束写在哪个文件里？为什么它**不在**
`CONTRIBUTING.md` 里这件事，对贡献者是一个重要提醒？

---

## E5（★★，不需要硬件）代码阅读：CI job 的能力与边界

读 `.github/workflows/ci.yml`，回答：

1. 一共几个 job？各自的依赖关系是什么？
2. `lint` job 里的「C++ API Style Audit」是阻断性的吗？依据是什么？
3. 草稿（draft）PR 会不会跑 `tests` 矩阵？
4. Metal 矩阵是怎么挑选测试的？一个名字里没有 `metal` 的 Metal 专属测试会怎样？

---

## E6（★★，不需要硬件）代码阅读：性能回归的判定规则

读 `maint/scripts/test_perf_regression.py` 开头注释里的 `THRESHOLD` 与 `MIN_COUNT`。

1. 默认值是多少？
2. 为什么不能把 `MIN_COUNT` 设成 1？注释里给出的两组数据分别是什么样的？
3. 谁有权触发这个 PR bot？外部贡献者能自己触发吗？

---

## E7（★★，不需要硬件）代码阅读：昇腾 PR 模板的性能条款

读 `_vendor/tilelang-ascend-v0.1.2.000/.github/pull_request_template.md` 第 1 节。

1. 一个新的昇腾算子，如果**有** AscendC 参考实现，性能门槛是什么？
2. 如果**没有** AscendC 参考实现，需要做什么？
3. 单选一个算子，只有 AscendC 后端通过、PTO 后端失败，能不能合入？模板怎么说？

---

## E8（★★★，不需要硬件）代码阅读：一个 pass 的注册链

读 `src/transform/loop_unswitching.cc` 末尾的 `TVM_FFI_STATIC_INIT_BLOCK` 与
`tilelang/transform/__init__.py` 里的 `LoopUnswitching`。

1. 写出一条从 Python 调用到 native 实现的完整链路。
2. 如果只加了 `.cc` 文件、忘了加 Python wrapper，会发生什么？
3. 如果 wrapper 加了、pass 也能跑，但没有把它挂进任何 pipeline，这个 pass 在真实
   编译流程里会被执行吗？

---

## E9（★★，不需要硬件）动手：跑一遍路径判定器

在 `code/01-pr-scope-linter/` 下运行：

```bash
python3 pr_scope_linter.py \
  examples/gemm/example_gemm.py \
  src/transform/lower_tile_op.cc \
  tilelang/ascend/pipeline.py \
  testing/python/transform/test_tilelang_transform_layout_inference.py
```

（1）四个路径各被判成什么类别？

（2）合并后的门禁集合里有哪几条？其中哪一条需要 maintainer 触发？

（3）把最后一个文件的路径改成 `tilelang/cuda/pipeline.py`，输出会怎么变？

---

## E10（★★，不需要硬件）动手：写一份 RFC 骨架

假设你要给 TileLang 增加一个新原语 `T.tile.scan`（对最后一个维度做前缀和），
它会改变 lowering 与至少一条 pipeline。

```bash
cd code/02-rfc-lint
python3 rfc_lint.py --template > scan-rfc.md
```

（1）把模板补成这样一份 RFC：动机、当前行为、提案签名、兼容性、验证计划、风险与回退、
非目标。

（2）运行 `python3 rfc_lint.py scan-rfc.md`，让它输出 `ok`。

（3）说明你为什么把某一条写进「非目标」，而不是留空。

---

## E11（★★，不需要硬件）动手：自检清单门禁

```bash
cd code/03-checklist-gate
python3 checklist_gate.py --template > self-check.md
```

（1）把其中一条改成「勾上但并没有做」的写法（`- [ ]` 且不带原因），运行脚本，
确认它报出 `unexplained`。

（2）把它补上一个带跟踪 issue 的原因，确认脚本输出 `ok`。

（3）解释为什么「勾上但没做」比「不勾」更危险。

---

## E12（★★★，不需要硬件）动手：为新 pass 设计测试落点

你要新增一个 pass：把布局推导的结果从「按 buffer 记录」改成「按 buffer 区域记录」。

请写出一份测试计划，至少回答：

1. 单元测试放哪个目录、文件名怎么起？
2. 断言应该落在 IR 节点上还是打印文本上？为什么？
3. 需不需要跑 `maint/layout_inference/`？如果 golden 全变了，`--record` 之后你要做什么？
4. 需不需要跑 `maint/precision/`？理由是什么？
5. 需不需要性能回归？如果需要在本地怎么跑、在 PR 上怎么请求？

---

## E13（★★，不需要硬件）动手：第一次贡献的完整计划

从 `♥ help` 标签的 issue 出发（入口见
`https://github.com/tile-ai/tilelang/issues?q=is%3Aissue+is%3Aopen+label%3A%22%E2%99%A5+help%22`），
写一份你自己的第一次贡献计划，必须包含：

1. 你会选哪一类路径（算子/编译器/后端），为什么；
2. Step 2–Step 8 的具体命令（按本章第 10 节，但换成你自己的环境）；
3. 如果最后一步（性能回归）你本地跑不了，你会在 PR 里怎么写。

---

## 习题与知识点的对应

| 题 | 对应正文 | 类型 |
| --- | --- | --- |
| E1 | §2、§4、§5、§6 | 概念 |
| E2 | §3.1、§12.1 | 概念 |
| E3 | §3.2、§12.2 | 概念 |
| E4 | §5.1、§6.3 | 概念 |
| E5 | §3.4 | 代码阅读 |
| E6 | §9.3 | 代码阅读 |
| E7 | §7.2、§9.4 | 代码阅读 |
| E8 | §5.1 | 代码阅读 |
| E9 | §4–§6、代码示例 01 | 动手 |
| E10 | §12.2、代码示例 02 | 动手 |
| E11 | §11、代码示例 03 | 动手 |
| E12 | §5.2、§9 | 动手 |
| E13 | §10 | 动手 |
