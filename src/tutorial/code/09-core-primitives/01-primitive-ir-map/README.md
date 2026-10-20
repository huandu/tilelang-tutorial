# 01 · 原语 → IR → 指令 映射表（可自检）

## 目的

把正文 §2.2 的映射表变成一段可执行的代码，并顺手回答两个问题：

1. 表里引用的每一个源码文件与行号，在锁定版本的 `_vendor/` 工作树上是否真的存在、
   行号是否落在文件范围内？
2. 「`T.tile.*` 不在 v0.1.15 里」这条结论，能不能被机器复现？

脚本是纯标准库、纯文本检查，不 import TileLang、不编译任何 kernel。
它覆盖正文里最关键的 42 条源码引用（映射表的 36 条 + §5.3 的 6 条）。

## 运行前提

Python 3.9+。只用标准库（`argparse`、`dataclasses`、`pathlib`）。
**不需要 GPU / NPU / TileLang**。

自检部分需要锁定版本的源码在默认位置
（`<本书根>/_vendor/tilelang-v0.1.15` 与 `_vendor/tilelang-ascend-v0.1.2.000`）；
也可以用 `--vendor` / `--adapter` 指定别处。找不到时脚本会打印
`skipped`，仍然输出映射表并以 0 退出。

## 运行命令

```bash
cd 01-primitive-ir-map
python3 primitive_ir_map.py                 # 打印映射表 + 自检
python3 primitive_ir_map.py --no-check      # 只打印映射表
python3 primitive_ir_map.py --vendor /path/to/tilelang-v0.1.15
```

## 预期输出（摘要）

在 macOS（Apple Silicon）、Python 3.11 上实际运行：

```text
primitive            | IR                         | NV target                      | Ascend target
---------------------+----------------------------+--------------------------------+------------------------------
T.copy               | tl.tileop.copy             | SIMT / cp.async / TMA / LDSM   | DataCopy / ND2NZ / LoadData / Fixpipe
T.async_copy         | tl.tileop.async_copy       | cp.async (no wait)             | -
...
checked 30 source citations -> all in range (or file missing -> skipped)
claim "T.tile is absent from v0.1.15": confirmed (no `as tile` import under tilelang/ascend/language)
```

（实际输出以本机为准；上面只摘了表头与结论行。）

## 重要边界

* 自检只能证明「文件存在、行号在范围内」，**不能**证明行号里的内容与正文描述一致。
  内容一致性是作者人工逐行读过一遍的，这仍然**不是运行验证**。
* 映射表的「NV target / Ascend target」两列是**指令类别**，不是实测生成代码。

## 作者验证级别

**CPU 逻辑验证**：在 macOS（Apple Silicon）、Python 3.11 上运行过。
**未在加速器上运行，也未与任何真实编译产物对照。**
