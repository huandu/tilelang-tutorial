# 01 · 能力状态审计：v0.1.14 / v0.1.15 / main 快照

## 目的

把第 30 章的版本纪律变成可执行的检查。脚本对 13 条能力各写一组“探针”
（某条路径存在 / 某个符号在某处出现），然后在三棵树上跑一遍，输出：

| 列 | 含义 |
| --- | --- |
| `v0.1.14` | 上一个 release，用来回答“是不是 v0.1.15 新增的” |
| `v0.1.15` | 本书锁定的基线 |
| `main 快照` | corpus 里的主干副本（`02-tilelang-core-repo/raw/tilelang`） |

判定规则很硬：**一条能力的所有探针都命中才算存在**。任何探针没命中就判为不存在，
避免“看到一处提到就以为功能有了”。

## 运行前提

Python 3.9+，只用标准库。不需要 GPU / NPU / TileLang。

## 运行命令

```bash
python3 state_audit.py
python3 state_audit.py --json                       # 机器可读
python3 state_audit.py --book-root /path/to/book    # 显式指定本书根目录
python3 state_audit.py --corpus-root /path/to/corpus
```

## 期望输出

标准输出形如（节选，实际 13 行）：

```text
capability                | v0.1.14 | v0.1.15 | main 快照 | 状态
--------------------------|---------|---------|-----------|-------------------
multi-backend-registry    |    Y    |    Y    |     Y     | v0.1.15 已具备
tvm-fork-still-pinned     |    Y    |    Y    |     Y     | v0.1.15 已具备（=迁移未落地）
cutedsl-backend           |    Y    |    Y    |     Y     | v0.1.15 已具备
ascend-backend-in-mainline |    -    |    Y    |     Y     | v0.1.15 已具备
ascend-pto-backend        |    -    |    -    |     Y     | 仅 main
ascend-persistent-kernel  |    -    |    -    |     -     | 仅未合入 PR/RFC
reducer-v2                |    Y    |    Y    |     Y     | v0.1.15 已具备
copy-cache-policy         |    -    |    -    |     -     | 仅未合入 PR/RFC
...

self-check: ok
```

`self-check` 会把每行的结果与脚本内嵌的期望值比对。任何一条不符就打印
`self-check: MISMATCH` 并以退出码 1 结束——这通常意味着 `_vendor/` 或 corpus
换版本了，应该停下来重新核对，而不是改期望值。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上运行过，
输出与上面的期望一致，`self-check: ok`，退出码 0。

## 与本章正文的对应

| 脚本中的能力 id | 正文小节 |
| --- | --- |
| `multi-backend-registry` | §3.1 多后端解耦 |
| `tvm-fork-still-pinned` / `vendored-z3-prover` | §3.2 TVM → apache/tvm 迁移 |
| `cutedsl-backend` | §3.3 CuteDSL 后端 |
| `ascend-backend-in-mainline` / `ascend-pto-backend` / `ascend-auto-schedule` | §4.1 Ascend 950 |
| `ascend-persistent-kernel` / `ascend-auto-simt-vf` / `ascend-op-agent` | §4.2–§4.4 |
| `reducer-v2` | §5.1 reducer RFC |
| `copy-cache-policy` / `copy-eviction-policy` | §5.3 cache policy |

## 简化了什么

1. 探针只做“存在性”判断，不做语义判断。例如 `reducer-v2` 命中的是 API 与两个
   pass 文件，它们具体管什么要在正文里读。
2. 文本检索只扫常见后缀（`.py/.cc/.h/.md/...`）且单文件上限 2 MB，`3rdparty/`、
   `.git/`、`build/` 一律跳过。
3. 每条能力最多收集 5 个命中文件用于展示，不做全文索引。
