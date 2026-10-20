# 第 20 章 代码目录

本章的示例都围绕「**观察**编译期」展开：要么解析锁定版本的源码，要么在真实机器上
抓 IR。**除 03 之外都不需要 tilelang、更不需要 GPU/NPU。**

## 验证级别总表

| 目录 | 作用 | 需要硬件吗 | 验证级别 |
| --- | --- | --- | --- |
| `01-pass-order/` | 静态提取两条 pipeline 的 pass 顺序、行号、守卫条件 | 否 | **CPU 逻辑验证**（已在本机运行） |
| `02-ir-snippet-extract/` | 从锁定版本抽取真实 IR 片段（含来源行号） | 否 | **CPU 逻辑验证**（已在本机运行） |
| `03-lower-trace-walkthrough/` | 逐 pass 抓 before/after IR（真实编译） | **是**（NVIDIA GPU） | **仅静态核对**（本机无卡，未运行） |
| `04-version-diff/` | 对照进主干的昇腾 pipeline 与适配器仓库的 pass 词汇 | 否 | **CPU 逻辑验证**（已在本机运行） |

## 定位 `_vendor/`

所有脚本都按「脚本所在目录向上数 4 层」定位书根，再拼
`_vendor/tilelang-v0.1.15/`。也就是说，只要章节目录还在
`.workspace/research/tilelang-book/chapters/20-ir-and-pass-pipeline/` 下，脚本就能
直接用相对路径找到锁定版本的源码，无需传参。

## 运行顺序建议

```bash
# 1) 先确认顺序表（不用任何依赖）
python3 01-pass-order/dump_pass_order.py

# 2) 再抽出正文引用的 IR 片段，逐段对照
python3 02-ir-snippet-extract/extract_ir_snippets.py

# 3) 昇腾两条路线的词汇对照
python3 04-version-diff/compare_ascend_pipelines.py

# 4) 有 GPU 时，做一次真正的逐 pass 观察
python3 03-lower-trace-walkthrough/lower_trace_walkthrough.py --target cuda
```

## 纪律

* 不修改 `_vendor/`，不修改 corpus。这些脚本只读。
* 需要硬件的示例（03）在 README 里写清楚了前提、命令、期望输出与「本机未运行」。
  不要把「仅静态核对」说成「运行验证」。
