# 第 31 章代码

本章的代码不是 TileLang kernel，而是**把「推进上游」这件事本身变成可检查的流程**。
三个脚本都是纯 Python 标准库实现，不需要 GPU、NPU，也不需要安装 TileLang，
可以离线运行；它们读的是文件与路径，不联网、不写仓库。

## 验证级别

**在 CPU 上验证过逻辑**：三个脚本都在 macOS + Python 3.11 上用
`python3 -m py_compile` 与 `python3 <script> --self-test`（或实际输入）跑过。
没有真实硬件参与，也不需要。

## 目录

| 目录 | 目的 | 离线可跑 |
| --- | --- | --- |
| `01-pr-scope-linter/` | 输入改动路径列表，判定这次改动属于哪条贡献路径，并打印仓库里实际存在的门禁 | 是 |
| `02-rfc-lint/` | 校验一份 RFC 草稿是否具备被评审所需的章节（动机、方案、备选、兼容性、验证、回退） | 是 |
| `03-checklist-gate/` | 解析一份改动自检清单，把「勾了但没有说明原因」的跳过项揪出来 | 是 |

## 快速开始

```bash
cd 01-pr-scope-linter
python3 pr_scope_linter.py --self-test
python3 pr_scope_linter.py tilelang/language/copy_op.py src/transform/lower_tile_op.cc

cd ../02-rfc-lint
python3 rfc_lint.py --template > my-rfc.md
python3 rfc_lint.py my-rfc.md

cd ../03-checklist-gate
python3 checklist_gate.py --self-test
```

## 与正文章节的对应关系

* `01-pr-scope-linter/` 对应正文第 2 节（三种贡献路径）与第 4 节。
* `02-rfc-lint/` 对应正文第 10 节（社区与沟通）。
* `03-checklist-gate/` 对应正文第 9 节（改动自检清单）。

## 注意事项

这三个脚本是**写作辅助**，不是上游的官方工具。上游仓库里没有等价的脚本；
脚本里的门禁名称都抄自锁定版本的实际文件，文件路径写在每个脚本的 `GATES`
字典旁边。上游改了工作流之后，脚本需要跟着改。
