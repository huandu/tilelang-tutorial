# 01 · 改 num_stages，看 IR 与生成代码怎么变

## 目的

把第 22 章 §2–§4 的结论落到一次可复现的实验上：同一份 GEMM，只改
`T.Pipelined(..., num_stages=N)`，观察三件事怎么随 N 变：

1. `PipelinePlanning` 写到循环上的 `software_pipeline_stage` / `software_pipeline_order`；
2. `InjectSoftwarePipeline` 物化出来的**版本维**（生成 CUDA 里 `__shared__` 数组的首维）；
3. `ThreadSync` / 异步拷贝 / TMA barrier 这条链路上**同步点的数量与形态**。

关键点：**这个实验的 IR 部分不需要 GPU**。脚本走的是
`tilelang.tools.compile_only.compile_kernel_source`
（`_vendor/tilelang-v0.1.15/tilelang/tools/compile_only.py:169-200`），它内部调用
`tilelang.lower(..., enable_device_compile=False)`，只做 lowering + codegen，不编译
nvcc、也不启动任何设备。

## 运行前提

* Python 3.9+；
* TileLang `v0.1.15`（`pip install tilelang==0.1.15`），且该 wheel 带 CUDA codegen
  FFI（`tilelang.cuda._ffi_api.AnnotateDeviceBoundTmaCopies` 存在即可；脚本会自己判断）；
* **不需要** NVIDIA GPU，**不需要** nvcc；
* 想在真实硬件上补测延迟，另需一张 NVIDIA GPU + CUDA toolkit + PyTorch(CUDA)。

## 运行命令

```bash
# 只看 IR 事实（无卡可跑）
python3 lower_num_stages.py --num-stages 1 2 3 4

# 把每个变体的 CUDA 源写出来，方便 diff
python3 lower_num_stages.py --num-stages 2 3 4 --dump-dir /tmp/tl22
diff -u /tmp/tl22/gemm_num_stages_2.cu /tmp/tl22/gemm_num_stages_3.cu

# 看真正的 lowering 逐 pass diff（HTML / 终端）
TL_LOWER_TRACE=terminal python3 lower_num_stages.py --num-stages 3
TL_LOWER_TRACE=html     python3 lower_num_stages.py --num-stages 3
```

`TL_LOWER_TRACE` 是 v0.1.15 的官方调试入口
（`tilelang/tools/lower_trace/__init__.py:1-30`，
文档见 corpus：`../tilelang-corpus/01-tilelang-official-docs/raw/site/_sources/tutorials/debug_tools_for_tilelang.md`）。

## 期望输出（形态）

`num_stages=3` 时，脚本会打印出与上游单测
`test_pipeline_planning_before_after_copy_gemm_num_stages_plan` 一致的一行：

```text
  software_pipeline_stage = [0, 0, 2]
  software_pipeline_order = [0, 1, 2]
  tl_pipelined_num_stages = 3
```

这是**推导出来的预期**（该断言在
`../tilelang-corpus/02-tilelang-core-repo/raw/tilelang/testing/python/transform/test_tilelang_transform_pipeline_planning.py`
里逐字写着），不是作者本机的运行结果。

## 正确性判据

1. 对 `num_stages=N ≥ 2`，`software_pipeline_stage` 的最后一个元素应等于 `N - 1`，
   前两个（两条 copy）为 0；
2. `num_stages=1` 时上游 planner 仍会给出 `[0, 0, 1]`（「把 copy 挪到 order 开头并
   整体减一」这一步要求 `num_stages ≥ 2`，见
   `_vendor/tilelang-v0.1.15/src/transform/pipeline_planning.cc:1266-1277`），
   所以**不要**把「深度 1」当成「深度 0」；
3. `num_stages=0` 时循环上不应出现任何 `software_pipeline_stage` 注解——参数在
   `src/ir.cc:138-144` 就被丢掉了；
4. 生成 CUDA 里 A/B 的 shared 数组首维应随 N 增长（`N` 或 `N-1`，取决于
   planner 是否把 copy 归到 stage 0）；
5. `num_stages` 变大时，`cp.async.wait_group` 的**参数**（在飞组数）应变大，
   `__syncthreads()` 的数量不应无谓地线性增长——如果它线性增长了，说明有同步点
   落到了关键路径上（见本章 §5）。

## 作者实际做到的验证级别

**仅静态核对**：`python3 -m py_compile lower_num_stages.py` 通过；脚本用到的每个 API
（`tilelang.jit`、`JITImpl.get_tir`、`tilelang.transform.IfStmtBinding`、
`tilelang.transform.PipelinePlanning`、`tilelang.tools.compile_only.compile_kernel_source`、
`tilelang.lower`）都逐个在 `_vendor/tilelang-v0.1.15/` 里核对过签名与存在性。

**作者没有运行过这个脚本**：本机（macOS Apple Silicon）**没有安装 TileLang**，
也没有 NVIDIA GPU。在本机直接运行会打印
`cannot import TileLang (No module named 'tilelang')` 并退出——这是脚本的预期行为，
不是在硬件上失败。

## 在有卡机器上补做性能部分

IR 部分不需要卡；只有「多一个 stage 到底快多少」需要。步骤：

1. 用同一份 kernel（把脚本里的 `T.Pipelined` 换成 §7 的完整 GEMM），
   `@tilelang.jit` 编译 `num_stages ∈ {1, 2, 3, 4, 6}`；
2. 用 `kernel.get_profiler().do_bench()` 测延迟
   （`_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py`，`example_gemm.py` 的用法）；
3. 记录 **延迟** 与 **shared memory 占用**两列；延迟应在某档之后不再下降，
   而 shared 占用继续线性增长——两条曲线的拐点就是本章 §2.5 说的判据；
4. 换 3 组不同的 `(block_M, block_N, block_K)` 重复，确认拐点不是某一组 tile 的巧合。

**不要**把这里的任何数字当作基准结果：脚本本身不测量任何东西。
