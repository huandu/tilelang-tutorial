# RFC 草稿校验器（rfc_lint.py）

## 目的

锁定版本的仓库里**没有** RFC 模板：`.github/ISSUE_TEMPLATE/` 只有 bug、feature、
question、release-plan 四个模板，RFC 是以普通 issue 形式提交的（见
`../tilelang-corpus/08-frontier-unmerged/raw/`）。模板缺失的后果是评审者要在评论里
反复索要同样的信息。这个脚本把「一份能被评审的 RFC 需要哪些章节」固化成检查项。

## 运行前提

* Python 3.10+（只用到标准库）。
* 一份 Markdown 草稿。

## 运行命令

```bash
python3 rfc_lint.py --template > my-rfc.md   # 生成骨架
$EDITOR my-rfc.md
python3 rfc_lint.py my-rfc.md                # 校验
python3 rfc_lint.py --self-test              # 自测
```

## 校验项

必需：动机（Motivation）、现状（Current Behavior）、提案（Proposal）、
兼容性与影响（Compatibility）、验证计划（Testing）、回退与风险（Risks / Rollback）、
非目标（Non-goals）。

建议：备选方案（Alternatives）、未决问题（Open Questions）、落地拆分（Staging）。

脚本还会检查两件常见的事故：

1. 提案里出现了 `T.` 开头的 API 名却没有任何代码块；
2. 通篇没有指向真实文件或 PR/issue 链接。

## 期望输出

* 合法草稿：打印 `ok: <file>`，退出码 0。
* 缺章节：逐条打印 `missing: ...`，退出码 1。
* `--self-test`：打印 `self-test passed`，退出码 0。

## 验证级别

**在 CPU 上验证过逻辑**：`python3 -m py_compile rfc_lint.py` 通过；
`python3 rfc_lint.py --self-test` 通过。

## 已知限制

脚本只能查「章节在不在」，不能判断论证是否成立。它不替代维护者对 RFC 的技术判断。
