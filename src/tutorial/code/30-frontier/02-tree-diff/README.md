# 02 · 两棵源码树的文件级差异

## 目的

回答「main 比 v0.1.15 多了什么」。脚本遍历两棵树、列出只在一侧存在的路径，
并可以顺带检索一个符号出现在哪一侧。

默认两侧是：

| 侧 | 路径 | 说明 |
| --- | --- | --- |
| left | `_vendor/tilelang-v0.1.15` | 本书锁定的 release |
| right | `../tilelang-corpus/02-tilelang-core-repo/raw/tilelang` | corpus 的主干快照（`main`） |

## 运行前提

Python 3.9+，只用标准库。不需要 GPU / NPU / TileLang。

## 运行命令

```bash
python3 tree_diff.py --selftest                 # 断言差异等于预期
python3 tree_diff.py --pattern PersistentKernel # 符号在哪一侧出现
python3 tree_diff.py --left <A> --right <B>     # 任意两棵树
```

## 期望输出

`--selftest` 的结尾必须是：

```text
only in right (new on main) : 20
only in left  (gone on main): 2
...
selftest: ok (main = v0.1.15 + the PR #3310 PTO files)
```

「only in right」的 20 个文件正是 PR #3310（PTO 后端）引入的内容：
`src/ascend/codegen/codegen_pto.*`、`src/ascend/codegen/rt_mod_pto.cc`、
`src/ascend/codegen/sfu_precision.h`、`tilelang/contrib/ptodsl/`（9 个文件）、
`tilelang/jit/adapter/pto/`（5 个文件），以及被搬到
`src/backend/common/codegen/` 的 `codegen_py.{cc,h}`。

「only in left」的 2 个文件是 `src/cuda/codegen/codegen_py.cc` 与 `.h`——它们不是
被删掉，而是被搬到 `src/backend/common/codegen/`。这说明同一份快照里
**新增和搬迁是混在一起的**，只看「有没有新文件」会漏掉搬迁类改动。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上运行过，
`--selftest` 通过；`--pattern` 的运行结果与手工 `rg` 一致。

## 与源码的对应关系

| 本脚本 | 对应的手工命令 |
| --- | --- |
| `rel_files()` 的差异集合 | `diff <(cd A && find . -type f \| sort) <(cd B && find . -type f \| sort)` |
| `text_hits()` | `rg -l '<pattern>' <tree>` |
| `--selftest` 的期望集合 | `../tilelang-corpus/08-frontier-unmerged/raw/github-pr-3310.md` 的改动清单 |

## 简化了什么

1. 不比文件**内容**，只比路径集合与符号是否出现。内容级比较需要 `git diff`，
   而 `_vendor/` 的 checkout 是 `--depth 1`，本地拿不到两个 tag 的共同祖先
   （这本身是 §8 要讲的一个坑）。
2. 跳过 `3rdparty/`、`.git/`、`build/`，以及本书自己的 `.git_commit.txt`。
3. 不做二进制比较：非文本后缀（`.png`、`.so` 等）只参与路径集合。
