# 第 32 章 代码

本章是导览章，没有需要在 NPU 上运行的 kernel。`code/` 里只放了一个**纯标准库**的
核验辅助脚本——它把第 32 章 §8 的「读 README」流程变成可重复执行的命令。

## 目录

```text
code/
├─ README.md                    本文件
└─ 01-verify-checklist/
   ├─ README.md                 目的、前提、命令、预期输出
   └─ extract_sections.py       从 README 抽取 News / Requirements / Ongoing
```

## 验证级别总表

| 示例 | 运行前提 | 作者实际做到的验证级别 |
| --- | --- | --- |
| `01-verify-checklist/extract_sections.py` | Python 3.9+，仅标准库 | **CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过 |

## 统一说明

* 写作环境（见 `../../VERSIONS.md`）：macOS (Apple Silicon)、Python 3.11，
  **没有 NVIDIA GPU，也没有昇腾 NPU**。
* 因此本章**不包含**任何需要加速器才能产生结果的示例；正文里所有性能数字都来自各仓库
  README 的公开口径，出处见 `../references.md`。
* 第五部分引用的 DeepSeek 仓库是 2026-10-01 的 corpus 快照，commit 见
  `../../../../tilelang-corpus/14-deepseek-ascend-stack/raw/PROVENANCE.md`。
