# 04-cite-check —— 校验正文里的每一个 `path:line` 引用

## 目的

第 25 章几乎每一句话都带一个源码行号。这个脚本把 `chapter.md`（以及可选的
`exercises.md` / `solutions.md` / `references.md`）里所有形如
`path/to/file.ext:123` 或 `path/to/file.ext:123-145` 的引用抽出来，逐个检查：

1. 文件是否存在（相对书根，或相对 `../tilelang-corpus/`）；
2. 起始行是否在文件行数以内；
3. 结束行 ≥ 起始行，且也在文件行数以内。

它会**拒绝**「文件存在但行号越界」这种最常见的引用错误。

## 运行前提

Python 3.10+，无第三方依赖。书根下要有 `_vendor/tilelang-v0.1.15/`。

## 运行命令

```bash
python3 check_citations.py
python3 check_citations.py --files chapter.md exercises.md solutions.md references.md
python3 check_citations.py --verbose      # 打印每一条引用
```

## 期望输出

```
scanned 4 file(s), 124 unique citations
[ok] 124/124 resolve, 0 out-of-range
```

## 验证级别

**CPU 逻辑验证**（本机已运行）。纯文本检查，不导入 tilelang、不编译、不联网。
