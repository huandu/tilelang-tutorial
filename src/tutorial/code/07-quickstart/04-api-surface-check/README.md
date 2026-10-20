# 04-api-surface-check · 离线核对本章用到的 API

## 目的

把“本章说 v0.1.15 里有这个 API”从断言变成一次可复现的检查。脚本不 import
tilelang，只读源码文本，把本章（以及对照用的昇腾）每个 API 名在
`_vendor/tilelang-v0.1.15/` 与 `_vendor/tilelang-ascend-v0.1.2.000/` 里搜一遍。

它还带了三条**反向**检查，用来证明本章的一处分界说法：

* 主干 `tilelang/cache/__init__.py` 里**没有**顶层 `clear_cache()`；
* 主干的 `tilelang.ascend.language` 里**没有** `is_npu`、**没有** `alloc_ub`。

（这两条正是“主干昇腾方言”与“独立仓库 tilelang-ascend 方言”的区别，见本章 7.2 节。）

## 运行前提

只有 Python 3.10+，并且能看到本教材仓库（`_vendor/` 与脚本同在仓库内）。
**不需要 tilelang，不需要 GPU。**

## 运行命令

```bash
python check_api_surface.py
python check_api_surface.py --root /path/to/tilelang-book
```

## 预期输出

```text
仓库根：/.../tilelang-book

== 应当存在 ==
  ok   T.Kernel（target-neutral 网格）                _vendor/tilelang-v0.1.15/tilelang/language/kernel.py:431
  ok   T.Parallel（块级并行域）                        _vendor/tilelang-v0.1.15/tilelang/language/loop.py:13
  ...

== 应当不存在（用于证明主干与 tilelang-ascend 的分界） ==
  ok   主干 v0.1.15 的 tilelang/cache 没有顶层 clear_cache() ...

全部 36 项符合预期。
```

**正确性判据**：脚本退出码为 0，最后一行是“全部 N 项符合预期”。

## 验证级别

**在本机运行过（纯 Python，只读源码）。** 它依赖 `_vendor/` 里的锁定版本源码，
不是运行 TileLang。

## 在真实机器上的验证步骤

在被修改过的 checkout 上重跑本脚本，能立刻发现“API 被挪走 / 改名”。若某一条 FAIL，
先看它命中了哪个文件的哪一行，再去对照本章 `references.md` 里的行号。
