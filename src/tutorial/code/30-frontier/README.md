# 第 30 章 代码

本章讲的是「一个能力到底在 v0.1.15、在 main、还是只在未合入的 PR 里」。
这类判断不能靠记忆，也不能靠 PR 页面上的措辞。所以本章的代码全部是
**审计工具**：只用 Python 标准库，只读文件，不编译、不联网、不需要任何硬件。

## 目录

```
code/
├─ README.md                       本文件
├─ 01-state-audit/
│  ├─ README.md
│  └─ state_audit.py               对 13 条能力逐条判定三种状态并自检
├─ 02-tree-diff/
│  ├─ README.md
│  └─ tree_diff.py                 两棵源码树的文件级差异 + 符号检索
└─ 03-upstream-watch/
   ├─ README.md
   └─ upstream_watch.py            上游跟踪手册（可选：检查本地 _vendor）
```

## 运行前提

Python 3.9+，只用标准库。**不需要 GPU、NPU，也不需要安装 TileLang。**

三个脚本默认从自身位置推断本书根目录（`../../../../..`），因此只要章节目录保持
在 `chapters/30-frontier/` 之下、`_vendor/` 与 `../tilelang-corpus/` 还在原位，
就可以直接运行。目录搬家时用 `--book-root` / `--corpus-root` / `--left` / `--right`
显式指定。

## 运行命令

```bash
python3 01-state-audit/state_audit.py          # 13 条能力的状态总表 + 自检
python3 02-tree-diff/tree_diff.py --selftest   # v0.1.15 与 main 快照的差异
python3 02-tree-diff/tree_diff.py --pattern PersistentKernel
python3 03-upstream-watch/upstream_watch.py --run
```

## 期望输出（摘要）

`state_audit.py` 打印一张 `v0.1.14 / v0.1.15 / main 快照` 三列的表，最后打印
`self-check: ok`。`tree_diff.py --selftest` 打印
`selftest: ok (main = v0.1.15 + the PR #3310 PTO files)`。

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-state-audit/state_audit.py` | Python 3.9+（只用标准库） | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过，`self-check: ok` |
| `02-tree-diff/tree_diff.py` | 同上 | **CPU 逻辑验证**：实际运行过，`--selftest` 通过 |
| `03-upstream-watch/upstream_watch.py` | 同上；`--run` 需要本机有 `git` | **CPU 逻辑验证**：`--run` 实际运行过（只跑只读的 `git describe/rev-parse/rev-list/status`） |

## 这三个脚本不能证明什么

* 它们不能证明「某个 PR 昨天还没合入，今天也没合入」。快照是**某个时刻**的照片，
  状态判定必须写上快照的 commit 与日期。
* 它们不做语义判断。`text` 探针只能回答「这段文字在不在这棵树里」，不能回答
  「这个功能对不对」。
* 它们不联网。`--run` 只读本地 `_vendor/`，因此无法替代
  `03-upstream-watch/upstream_watch.py` 里列出的 `git fetch` / `gh` 命令。

写作环境见 [../../VERSIONS.md](../../VERSIONS.md)：macOS (Apple Silicon)、
Python 3.11，**没有 NVIDIA GPU，也没有昇腾 NPU**。本章没有任何来自本机实测的
性能数字；出现的性能数字全部来自 corpus 内的 PR/文档原文，并标注了出处。
