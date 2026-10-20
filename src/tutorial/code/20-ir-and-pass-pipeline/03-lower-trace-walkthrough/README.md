# 03-lower-trace-walkthrough — 逐 pass 读一个真实 kernel 的 IR

## 目的

把第 12 章介绍的 IR Lower Trace 用到本章的读法上：编译一个最小 elementwise kernel，
打印「每个承重 pass 前后 IR 的增删行数」，再按需把生成的 CUDA 源码打出来对照。

## 运行前提

1. 锁定版本的 tilelang（`v0.1.15`）已安装，且 `tilelang.tools.lower_trace` 可导入。
2. **NVIDIA GPU + CUDA toolkit**，以及匹配的 PyTorch（脚本按 `target="cuda"` 编译）。
3. 磁盘可写：Lower Trace 会在 `./tmp/lower_trace_dir/<脚本名>/` 下写 `.tir` 快照
   与 HTML 报告。

## 运行命令

```bash
python3 lower_trace_walkthrough.py --target cuda
python3 lower_trace_walkthrough.py --target cuda --only LayoutInference,LowerTileOp
python3 lower_trace_walkthrough.py --target cuda --mode both --codegen-only
```

## 预期输出

* 每行一个 pass：`<pass 名>  +插入行数  -删除行数  changed=<bool>`。
* `--codegen-only` 追加打印生成的 CUDA 源码（应包含向量化的 `uint4` 访存）。
* 默认 `--mode terminal` 还会在终端打印彩色 diff；`--mode html` 会在
  `./tmp/lower_trace_dir/` 下生成可交互的 `report.html`。

## 判据

* `PipelinePlanning` 的记录应当**只改注解**（在 `T.Pipelined` 循环上写
  `software_pipeline_stage` / `software_pipeline_order` / `tl_pipelined_num_stages`）。
* `InjectSoftwarePipeline` 之后循环体应当被拆成 prologue / steady-state / epilogue，
  并出现多版本 buffer。
* `LayoutInference` 之后 fragment / shared buffer 上应出现 layout 注解。
* `LowerTileOp` 之后不再有 `tl.tileop.*` 调用。
* `ThreadSync` 之后出现 `T.tvm_storage_sync("shared")`。
* `MakePackedAPI` 之后函数参数变成 `(args, num_args)` 形式。

## 验证级别

**仅静态核对**。作者的本机（macOS / Apple Silicon）没有 NVIDIA GPU，也没有安装
tilelang，因此本脚本**没有在本机运行过**。脚本中引用的 API 名
（`tilelang.tools.lower_trace`、`tilelang.transform.PipelinePlanning` 等）已逐个在
`_vendor/tilelang-v0.1.15/` 里核对存在。上面的「判据」来自
`_vendor/tilelang-v0.1.15/docs/tools/lower_trace.md` 与
`testing/python/transform/` 下的对应测试，属于「应当如此」，不是本机实测结果。

在 GPU 机器上跑通后，请把 `tmp/lower_trace_dir/` 保留下来，作为回归基线
（第 29 章会讲怎么把它变成可对比的 IR 基线）。
