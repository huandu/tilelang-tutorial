# repo-map：仓库模块清单与数量统计

## 目的

正文第 2 节的「每个顶层模块负责什么」表，以及图 1 里的数字（24 个顶层子目录、
390 个 `.py`、42 个 `src/transform/*.cc`、54 个 examples 子目录、37 篇 docs），
都由这个脚本生成。它把「我数过」变成一条能重跑的命令。

## 运行前提

* Python 3.8+（标准库）。
* `_vendor/tilelang-v0.1.15/` 存在。
* **不需要** GPU/NPU，**不需要** 安装 tilelang。

## 运行命令

```bash
python3 inventory.py                 # 人读表格
python3 inventory.py --format json   # 机器可读
python3 inventory.py --vendor /path/to/checkout
```

## 期望输出（v0.1.15 上）

```text
VERSION: 0.1.15

| 模块 | 职责 | 存在 | .py | .cc/.h | 子目录 |
| `tilelang/language` | ... | 是 | 47 | 0 | 6 |
| `tilelang/engine`   | ... | 是 | 5  | 0 | 0 |
...
totals:
  tilelang_top_level_dirs = 24
  tilelang_python_files = 390
  src_transform_cc_files = 42
  examples_subdirs = 54
  examples_top_level_files = 3
  docs_markdown_files = 37
```

**正确性判据**：`examples_subdirs + examples_top_level_files = 57`，与
`VERSIONS.md` 里「`examples/` 有 57 个目录」的说法项数一致，但拆分口径不同
（54 个目录 + 3 个文件）。正文与 `references.md` 都记了这个差异。

## 常见误用

* 把本脚本指向 `main` 的另一份 clone，然后拿它的数字去核对正文。正文所有数字都来自
  **v0.1.15**（`VERSION` = `0.1.15`）；`main` 上 `examples/`、`tilelang/jit/adapter/`
  的数量都会变。
* 用 `git ls-files` 统计。仓库里有 `.gitignore` 掉的产物目录，`rglob` 与 `git ls-files`
  结果会略有差异。本章统一用 `rglob`（等价于「在你 clone 出来的工作树上数」）。

## 验证级别

**在本机运行过**（macOS Apple Silicon，Python 3.11，2026-10-01）。
