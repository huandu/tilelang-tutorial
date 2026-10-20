# 自检清单门禁（checklist_gate.py）

## 目的

真实项目的事故大多不是「没勾」，而是**跳过了却没说为什么**。昇腾仓库的 PR 模板写得很
直白：「请只勾选本 PR 已实际完成的项目；不适用或跳过的项目请写明原因和跟踪 Issue」
（`_vendor/tilelang-ascend-v0.1.2.000/.github/pull_request_template.md`）。

这个脚本把这条规范变成可执行的检查：每个未勾选项都必须带上原因或跟踪项。

## 运行前提

* Python 3.10+（只用到标准库）。
* 一份 Markdown 清单。

## 运行命令

```bash
python3 checklist_gate.py --template > self-check.md
python3 checklist_gate.py self-check.md
python3 checklist_gate.py --self-test
```

## 语法

```markdown
- [x] 已跑 bash format.sh
- [ ] 未跑 NPU 端到端 — 原因: 本机无 NPU，已在 PR 里标注待 maintainer 触发
- [ ] 未加性能报告
  - 原因: 本改动不触碰热点路径，见 #1234
```

判定规则：

* `- [x]` 视为已完成的勾选项，不再检查。
* `- [ ]` 视为未勾选项。它必须满足下面之一：
  1. 同一行 `—` 或 `--` 之后跟 `原因:` / `Because:` / `Reason:`；
  2. 紧随其后的子行以 `原因:` / `Because:` / `Reason:` 开头（可带 `- ` 列表符号）。
* 未勾选又没有原因的条目，脚本打印 `unexplained: <行号> <内容>` 并退出码 1。

## 期望输出

* `self-test passed`（自测）；
* `ok: <file>`（全部有解释）；
* `unexplained: ...`（需要补充原因）。

## 验证级别

**在 CPU 上验证过逻辑**：`python3 -m py_compile checklist_gate.py` 通过；
`python3 checklist_gate.py --self-test` 通过。

## 已知限制

脚本不能验证原因是否成立，也不能替 CI 判断测试是否真的跑过。它只防「悄悄跳过」。
