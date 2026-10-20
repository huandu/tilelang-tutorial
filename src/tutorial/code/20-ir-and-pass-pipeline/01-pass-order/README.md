# 01-pass-order — 静态提取两条 pipeline 的 pass 顺序

## 目的

不安装 TileLang、不需要 GPU/NPU，只靠解析锁定版本的源码，就把
`tilelang/cuda/pipeline.py` 与 `tilelang/ascend/pipeline.py` 里的
**pass 调用顺序、行号、以及源码级守卫条件**打印出来，并统计两侧同名 pass。

## 运行前提

* Python 3.10+（只用标准库）。
* 当前 checkout 里存在 `_vendor/tilelang-v0.1.15/`（脚本按相对路径定位）。
* 不需要 tilelang、torch、CUDA、CANN。

## 运行命令

```bash
python3 dump_pass_order.py            # 两条 pipeline 的完整列表 + 对照统计
python3 dump_pass_order.py --pipeline ascend
python3 dump_pass_order.py --names    # 只打印 pass 表达式，方便 diff
```

## 预期输出

* 每个 pipeline 的 `total pass invocations`：CUDA 与 Ascend 各自的 pass 调用数。
* 每行格式 `序号 | L行号 | mod = <pass>(...)(mod) [if <guard>]`。
* 末尾的 shared / cuda-only / ascend-only 三段，用于确认「两侧共享的 lowering 走廊」
  与「各自特有的 pass」。

行号即 `tilelang/cuda/pipeline.py` 与 `tilelang/ascend/pipeline.py` 里的真实行号，
可直接与正文的表格对照。

## 验证级别

**CPU 逻辑验证**：作者在本机（macOS / Apple Silicon，无 GPU/NPU）跑过本脚本，
输出与正文表格一致。不涉及任何设备侧运行。
