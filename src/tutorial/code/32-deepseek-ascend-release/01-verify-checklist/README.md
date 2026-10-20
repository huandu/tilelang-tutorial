# 01-verify-checklist：从 README 里抽核验材料

## 目的

实现第 32 章 §8“如何自己核验”里那条流程的第一步：**把一个组件 README 的
`News` / `Requirements` / `Ongoing` 三节原样抽出来**，让“能不能用、状态如何”的判断
建立在原文而不是转述上。

它**不做判断**：找不到某个小节就打印 `(not found)`，因为“README 没写”与
“README 说没有”是两回事（见正文 §8.3、§8.7）。

## 运行前提

* Python 3.9 及以上；
* **只使用 Python 标准库**——不需要 TileLang、不需要 PyTorch、不需要 NPU/GPU；
* 需要一个组件 README 文件（可以用 corpus 里的快照路径）。

## 运行命令

```bash
# 默认抽 News / Requirements / Ongoing
python3 extract_sections.py \
    ../../../../../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepEP-Ascend/README.md

# 换一组小节名
python3 extract_sections.py --sections Requirements,Performance \
    ../../../../../tilelang-corpus/14-deepseek-ascend-stack/raw/DeepGEMM-Ascend/README.md

# 只看这份 README 有哪些小节（先侦察，再决定抽什么）
python3 extract_sections.py --list \
    ../../../../../tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/README.md
```

（上面的相对路径是相对本目录：本目录 → `code/` → 章节目录 → `chapters/` →
`tilelang-book/` → `research/`，此时 `../../../../../tilelang-corpus/` 才指向 corpus。）

## 预期输出

* 每个文件以 `=== <路径>` 开头；
* 每个小节打印它自己的标题与正文；找不到时打印 `## <名字>: (not found)`；
* 文件不存在时打印到 stderr 并以退出码 1 结束。

实际输出（2026-10-01 在写作机上运行，节选）：

```text
=== .../DeepEP-Ascend/README.md
## News: (not found)
### Requirements
- Linux on an Ascend host with supported NPU drivers and firmware.
- Ascend 950 (A5) with UBMEM connectivity and UBC_CTP/URMA channels ...
The validated stack is Ascend 950DT, CANN 9.2.0, Python 3.12, PyTorch 2.13.0+cpu
and torch_npu 2.13.0rc1, with the PoC HDK configuration described below. Kernel
support on other Ascend generations or CANN versions has not been established by
these measurements.
## Ongoing
The Ascend backend is under active development:
- Bucket collectives: batched all-gather is available; Ascend reduce-scatter and
  all-reduce kernels are still being built.
- Expert load balancing: `EPBuffer.lb_prefetch_weights` and
  `EPBuffer.lb_reduce_grads` expose the aligned APIs, but their Ascend
  communication kernels are not implemented yet.
```

同一个脚本对 TileKernels 的输出则是“有 `News`、有 `Requirements`、
`Ongoing: (not found)`”——这正好说明为什么脚本必须把“找不到”当成一条信息打印出来。

## 验证级别

**CPU 逻辑验证**：本脚本在写作环境（macOS Apple Silicon、Python 3.11）上实际运行过，
输出见 `../README.md` 的“实际输出”。它**没有**在昇腾机器上运行过（也不需要）。

## 常见误用

* 把脚本输出当成结论。它只抽取文本；“这个组件能不能在 CANN 9.2.0 上用”仍然要人来
  读 `Requirements` 并判断。
* 只抽一个文件就下结论。同一个能力往往分布在多个仓库（例如 DSA 同时涉及 FlashMLA 与
  DeepSelect），应一起抽。
