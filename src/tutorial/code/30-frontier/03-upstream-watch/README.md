# 03 · 上游跟踪手册

## 目的

把“怎么持续跟踪 TileLang 上游”写成一份可以直接照抄的清单：四步状态判定流程、
需要 `git` / `gh` 的命令、要订阅的标签，以及本地 `_vendor/` 快照的局限。

这个脚本**不联网**。它默认只打印手册；加 `--run` 时额外用只读 git 命令检查
本地 `_vendor/` 三个 checkout。

## 运行前提

Python 3.9+。`--run` 需要本机有 `git`；如果没有，会打印提示而不是报错。

## 运行命令

```bash
python3 upstream_watch.py                    # 只打印手册
python3 upstream_watch.py --run              # 附带本地快照体检
python3 upstream_watch.py --repo tile-ai/tilelang-ascend
```

## 期望输出

`--run` 的结尾会打印一张 checkout 体检表，并给出与本章正文一致的提示：

```text
checkout                           | HEAD         | describe                     | history
----------------------------------------------------------------------------------------------------
tilelang-ascend-v0.1.2.000         | 89a9364e47a3 | TileLang-ascend-v0.1.2.000-release | 1 commit(s)
tilelang-v0.1.14                   | 7e3bbf3703e7 | v0.1.14                      | 1 commit(s)
tilelang-v0.1.15                   | a35f8ddf45eb | v0.1.15                      | 1 commit(s)

提示：这些是 --depth 1 快照，只有一个 commit，因此本地无法
`git diff v0.1.14..v0.1.15`。要比较两个 tag，要么用完整 clone，
要么用本目录 02-tree-diff/tree_diff.py 做文件级比较。
```

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上运行过 `--run`，
输出的 HEAD / describe 与 `_vendor/` 实际情况一致；脚本只执行
`git describe`、`git rev-parse`、`git rev-list --count`、`git status` 四条只读命令。

## 为什么这条“体检”值得做

`_vendor/` 是按 tag 做的 `git clone --depth 1`：每个 checkout 只有一个 commit，
没有共同祖先。这带来两个后果：

1. **本地无法** `git diff v0.1.14..v0.1.15`，也**无法** `git log` 看某个符号是
   什么时候进来的；
2. 想回答“这个能力在不在别的版本”，只能用文件级比较（`02-tree-diff/`）或
   在一个完整的 clone 里做。

把这条限制写进手册，是为了让“我本地 diff 不出来”不被误读成“两者没有差异”。
