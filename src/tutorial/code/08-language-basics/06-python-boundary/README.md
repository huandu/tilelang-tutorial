# 06-python-boundary · Python 兼容性边界

对应正文第 8 节。

## 目的

把 `docs/programming_guides/python_compatibility.md` 的那张表变成一个**能跑的
检查器**：给一段（或一个文件）Python 代码，逐个报出里面出现了哪些
“可用 / 有限制 / 不可用”的构造。

## 文件

| 文件 | 类型 | 验证级别 |
| --- | --- | --- |
| `py_compat_scan.py` | 纯标准库静态扫描器 | **CPU 逻辑验证** |

## 运行

```bash
python3 py_compat_scan.py                 # 扫内置的示例片段
python3 py_compat_scan.py some_kernel.py  # 扫你自己的文件
python3 py_compat_scan.py --self-test     # 校验规则本身
```

前提：Python 3.9+（用到 `ast.unparse` 的分支需要 3.9+，只用标准库）。作者在
macOS (Apple Silicon) + Python 3.11 上实际运行过。

真实输出（节选）：

```text
$ python3 py_compat_scan.py --self-test
  pass  len() is BAD             statuses=['WARN', 'BAD ']
  pass  chained assign is BAD    statuses=['WARN', 'BAD ']
  pass  class is BAD             statuses=['BAD ']
  pass  range loop is OK         statuses=['WARN', 'OK  ']
  pass  while is OK              statuses=['WARN', 'OK  ', 'OK  ']

5/5 self-tests pass

$ python3 py_compat_scan.py
=== <built-in demo> ===
  WARN  :5    def kernel_factory          kernels allow no nested functions; use @T.macro
  WARN  :7    def kernel                  kernels allow no nested functions; use @T.macro
  OK    :8    with T.Kernel(...)          only T.Kernel / T.ws style contexts are recognized
  WARN  :9    for i in T.Parallel(BLK)    not recognized as range()/list/tuple/enumerate/zip
  OK    :11   if / elif / else            python bools fold at trace time; TIR conds stay
  BAD   :14   len()                       len() is not available in kernels -- use buffer.shape[dim]
  BAD   :15   a = b = c                   chained assignment is not supported; split it
  WARN  :16   with open('x')              not a known DSL context manager
  8 findings: 2 BAD, 4 WARN
```

## 它不是 TileLang 的 parser

这是**教学模型**：它只按上游那张表做模式匹配，因此有两类已知噪声——

1. 它扫**整个文件**，不区分 kernel 体和 host 侧；
2. `T.Parallel(...)` 这种 DSL 循环会被标 `WARN`（表里没有这一条）。

真正的判定仍然要交给 TileLang 自己编译。这个脚本的价值是“**在编译之前**先把
显然越界的东西挑出来”。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon) + Python 3.11 上实际运行过，
`--self-test` 5/5 通过，内置 demo 的输出见上。

## 原文出处

* `_vendor/tilelang-v0.1.15/docs/programming_guides/python_compatibility.md:1-101`
* `_vendor/tilelang-v0.1.15/tilelang/language/eager/builder.py:1075`（`T.macro`）
* `_vendor/tilelang-v0.1.15/tilelang/language/parser/entry.py:105`（parser 侧 `macro`）
