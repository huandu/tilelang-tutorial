# 04-version-diff — 昇腾侧两条路的 pass 词汇对照

## 目的

全书锁定两份昇腾相关的源码：

* `_vendor/tilelang-v0.1.15/tilelang/ascend/pipeline.py`（**进主干**的昇腾后端）；
* `_vendor/tilelang-ascend-v0.1.2.000/tilelang/engine/phase.py`（**独立适配器仓库**）。

它们的 pass 词汇**不一样**。本脚本把两侧的 pass 名各抽一遍、做差集，并支持用
`--grep` 在整棵树里定位某个名字到底出现在哪里（例如 `InsertNd` 与 `InsertNd2Nz`）。

## 运行前提

* Python 3.10+（标准库）。
* checkout 内存在上面两份 `_vendor/`。
* 不需要 tilelang、torch、GPU、NPU。

## 运行命令

```bash
python3 compare_ascend_pipelines.py
python3 compare_ascend_pipelines.py --grep InsertNd2Nz
python3 compare_ascend_pipelines.py --grep AutoSchedule
python3 compare_ascend_pipelines.py --compare-main
```

## 预期输出

* `only in the in-tree ...` / `only in the adapter ...` 两段差集。据此可以确认：
  进主干的昇腾后端有整套 `AutoSchedule` / `InsertSync` / `MaterializeMultiBuffer`
  自动调度段与 `InsertNd2Nz`，适配器仓库走的是另一套
  `AscendLowerParallelToVector` / `CrossCorePipeline` / `AscendSyncInsert` 词汇。
* `--grep InsertNd2Nz` 会同时命中两份 `_vendor` 中的源码；而字面量 `InsertNd(`
  （不带 `2Nz`）在锁定版本里**找不到**——它是早期文档里的旧写法。
* `--compare-main` 会把进主干的 `v0.1.15` pipeline 与 corpus 的 **main 快照**
  （`../tilelang-corpus/02-tilelang-core-repo/raw/tilelang/...`）的 pass 名做差集。
  在 2026-10-01 的核验里两侧**互差为空**——也就是说正文列出的这些昇腾 pass
  **已经在 release 里**，不是「仅 main / 未进 v0.1.15」。这一条恰好反驳了
  「昇腾自动调度只在 main 上」的常见误解。

## 验证级别

**CPU 逻辑验证**：作者在本机跑通并核对了输出。脚本只做文本扫描，不编译任何 kernel。
