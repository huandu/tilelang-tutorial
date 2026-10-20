# 示例 1：源码锚点核对

## 目的

本示例做两层核对，把「正文说的位置」变成可重跑的检查：

1. **锚点表（172 条）**：手写一张「文件 : 行号 : 这一行应该包含什么 : 为什么重要」
   的表，逐条比对。这一层抓「行号漂移」。
2. **引用扫描（241 处）**：把整章 `.md` / `.py` 里出现的每一处 `file.py:123` 与
   `file.py:12-34` 都抽出来，检查「文件存在、行号在范围内、指向的不是空行」。
   这一层抓锚点表里没单独登记的行内区间笔误。

它把 `_vendor/tilelang-v0.1.15/` 与 `_vendor/tilelang-ascend-v0.1.2.000/`
当纯文本读，不 import 任何东西。

它**不 import tilelang / torch / tvm**，所以是本机（macOS、无 GPU、无 NPU）能真正跑通的
少数示例之一。

## 运行前提

* Python 3.9 以上（只用标准库）。
* 仓库里存在 `_vendor/tilelang-v0.1.15/` 与 `_vendor/tilelang-ascend-v0.1.2.000/`。
* **不需要** GPU / NPU / CUDA / CANN / torch / tilelang。

## 运行命令

```bash
cd chapters/26-runtime-jit-abi/code/01-runtime-anchor-check
python3 verify_runtime_anchors.py            # 两层核对，逐条打印
python3 verify_runtime_anchors.py --quiet    # 只打失败项与汇总
python3 verify_runtime_anchors.py --list     # 打印整张锚点表（含用途说明）
python3 verify_runtime_anchors.py --no-scan  # 只查锚点表
```

## 期望输出

失败时逐条列出「第几行应该是什么、实际是什么」，退出码 1。全部命中时退出码 0：

```text
citation scan: 241 line references checked

ok: 172/172 anchors verified; book root = <...>/tilelang-book
```

## 正确性判据

* 退出码 0；
* 锚点数 = 172；
* 引用扫描没有 `file not found` / `out of range` / `points at a blank line`；
* 任何一条失败都说明「本章某处引用的行号已经漂移」，必须修正文，不是修本脚本。

## 作者实际做到的验证级别

**在本机运行过（2026-10-01）**：输出
`citation scan: 241 line references checked` 与 `ok: 172/172 anchors verified`，退出码 0。
