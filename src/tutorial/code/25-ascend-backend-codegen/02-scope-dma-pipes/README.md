# 02-scope-dma-pipes —— scope / DMA 通路 / 流水线的静态抽取与一致性检查

## 目的

正文 §4.2–4.4 的三张表都来自源码，不是复述文档。本脚本把这些表**原样抽出来**，
再跑四条一致性检查：

1. `T.alloc_*` 产出的 scope 集合 ⊆ `GetAscendScopeQualifier` 能翻译的 scope 集合；
2. `DMAPath` 里每一个非 `kNone` 的通路都在 `src/ascend/op/copy.cc` 里真的被分支处理；
3. `DMAPath` 的数量与 `ResourcePipe` 覆盖的搬运路径一致（每条通路都能归到一个 pipe）；
4. 每个 scope → CCE 限定符的映射都能在 `codegen_ascend.cc` 里找到字面量。

## 运行前提

Python 3.10+，无第三方依赖。

## 运行命令

```bash
python3 scope_dma_pipes.py
```

## 期望输出

```
== scope -> CCE qualifier ==
  shared/shared.dyn        -> __ubuf__   (UB)
  ...
== DMAPath and the pipe that owns it ==
  kGMToUB    MTE2   ...
...
[ok] 4 consistency checks passed
```

## 验证级别

**CPU 逻辑验证**（本机已运行）。脚本只读源码，不导入 tilelang、不编译。
