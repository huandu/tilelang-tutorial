# 示例 1：调试与性能分析工具清点表

## 目的

本章提到的工具名字有三十多个。如果这些名字只停在正文里，读者没法判断「它到底在不在
锁定版本里」。这个脚本把「本章声称存在的工具入口」变成一张**可以复算的表**：直接读
`_vendor/` 下的锁定版本源码，用正则逐个确认入口存在，并打印 `文件:行号`。

它不导入 `tilelang`（本机没有 CUDA/NPU，编译好的 `.so` 不存在），因此不需要安装
TileLang，也不需要任何加速器。

## 运行前提

* Python 3.9+（只用标准库：`argparse` / `re` / `sys` / `pathlib`）。
* 需要本书的 `_vendor/tilelang-v0.1.15/` 与 `_vendor/tilelang-ascend-v0.1.2.000/`
  存在（脚本从自身位置向上寻找同时含 `VERSIONS.md` 与 `_vendor/` 的书根目录）。

## 运行命令

```bash
# 在本书根目录下
python3 chapters/12-debugging-and-profiling/code/01-tool-inventory/tool_inventory.py

# 或从示例目录里直接跑
cd chapters/12-debugging-and-profiling/code/01-tool-inventory
python3 tool_inventory.py

python3 tool_inventory.py --list-missing   # 只看缺了谁
```

## 期望输出

作者在写作机上的实际输出（2026-10-01，锁定版本 `a35f8dd` + `89a9364`，合计 36 项）：

```
== 编译期 · 日志 ==
  [有] TVM 日志等级表（LOG/DLOG/VLOG）               _vendor/tilelang-v0.1.15/src/runtime/logging.cc:18
  [有] LOG(FATAL) 报错通道（C++ 侧）                 _vendor/tilelang-v0.1.15/src/runtime/logging.cc:65
  [有] TVM_LOG_DEBUG 运行时说明符解析                 _vendor/tilelang-v0.1.15/src/runtime/logging.cc:73
  [有] TVM_LOG_DEBUG 编译期宏（Debug 构建开关）         _vendor/tilelang-v0.1.15/CMakeLists.txt:487

== 编译期 · 只看产物 ==
  [有] compile_only CLI（--output_file）        _vendor/tilelang-v0.1.15/tilelang/tools/compile_only.py:223
  [有] compile_kernel_source() 程序 API         _vendor/tilelang-v0.1.15/tilelang/tools/compile_only.py:169
  [有] get_kernel_source() / print_source_code() _vendor/tilelang-v0.1.15/tilelang/jit/kernel.py:532
  [有] codegen 后处理回调（CUDA）                    _vendor/tilelang-v0.1.15/tilelang/engine/callback.py:55
  [有] codegen 后处理回调（Ascend）                  _vendor/tilelang-v0.1.15/tilelang/ascend/callback.py:32

== 编译期 · pass 前后 ==
  [有] TL_LOWER_TRACE 环境变量                    _vendor/tilelang-v0.1.15/tilelang/env.py:390
  [有] TL_LOWER_TRACE_DIR 环境变量                _vendor/tilelang-v0.1.15/tilelang/env.py:391
  [有] lower_trace() 一次性 diff                 _vendor/tilelang-v0.1.15/tilelang/tools/lower_trace/__init__.py:49
  [有] enable() / disable() 全流程插桩             _vendor/tilelang-v0.1.15/tilelang/tools/lower_trace/core.py:693
  [有] Pass Diff 程序 API（legacy）               _vendor/tilelang-v0.1.15/tilelang/utils/pass_diff.py:1030
  [有] Pass Visualizer CLI（结构树）               _vendor/tilelang-v0.1.15/tilelang/tools/pass_visualizer/viewer.py:460
  [有] pass_timing 插桩                         _vendor/tilelang-v0.1.15/tilelang/tools/pass_timing.py:148

== 编译期 · 布局 ==
  [有] plot_layout()                          _vendor/tilelang-v0.1.15/tilelang/tools/plot_layout.py:6
  [有] TL_LAYOUT_VISUALIZATION_ENABLE         _vendor/tilelang-v0.1.15/tilelang/transform/pass_config.py:252
  [有] TL_EMIT_LINE_DIRECTIVES（#line 映射）      _vendor/tilelang-v0.1.15/tilelang/transform/pass_config.py:107

== 运行期 · 打印 ==
  [有] T.print（CUDA 方言）                       _vendor/tilelang-v0.1.15/tilelang/cuda/language/print.py:128
  [有] T.device_assert（CUDA 方言）               _vendor/tilelang-v0.1.15/tilelang/cuda/debug.py:23
  [有] T.print（Ascend 方言）                     _vendor/tilelang-v0.1.15/tilelang/ascend/language/print.py:31
  [有] Ascend debug 打印模板（printf）              _vendor/tilelang-v0.1.15/src/tl_templates/ascend/debug.h:15
  [有] T.printf（Ascend 适配器）                   _vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:534
  [有] T.dump_tensor（Ascend 适配器）              _vendor/tilelang-ascend-v0.1.2.000/tilelang/language/ascend.py:582

== 运行期 · 异常现场 ==
  [有] TL_ASCEND_EXCEPTION_DUMP 配置键           _vendor/tilelang-ascend-v0.1.2.000/tilelang/transform/pass_config.py:62
  [有] parse_exception_dump()                 _vendor/tilelang-ascend-v0.1.2.000/tilelang/tools/ascend_exception_dump_bin.py:67
  [有] read_msaicerr_bin()                    _vendor/tilelang-ascend-v0.1.2.000/tilelang/tools/ascend_exception_dump_bin.py:27
  [有] 异常 dump 的 MAGIC 搜索（C++ 模板）             _vendor/tilelang-ascend-v0.1.2.000/src/tl_templates/ascend/exception_dump.h:1

== 正确性 ==
  [有] torch_assert_close() 容差工具              _vendor/tilelang-v0.1.15/tilelang/utils/tensor.py:205
  [有] Profiler.assert_allclose()             _vendor/tilelang-v0.1.15/tilelang/profiler/__init__.py:104
  [有] AutoDD 自动缩小复现（--err-msg）               _vendor/tilelang-v0.1.15/tilelang/autodd.py:1399

== 性能 ==
  [有] Analyzer.analysis() 静态 roofline 估算     _vendor/tilelang-v0.1.15/tilelang/tools/Analyzer.py:216
  [有] do_bench(backend=...) 计时入口             _vendor/tilelang-v0.1.15/tilelang/profiler/bench.py:39
  [有] msprof 计时实现 bench_with_msprof()        _vendor/tilelang-v0.1.15/tilelang/profiler/msprof.py:40
  [有] KernelProfile（AIC/AIV pipe 利用率）        _vendor/tilelang-v0.1.15/tilelang/profiler/msprof.py:59

合计 36 个入口，找到 36 个，缺失 0 个。
注意：本表证明的是「入口在锁定版本源码里存在」，不是「它在你的机器上能跑通」。
```

**判据**：退出码为 `0`，且「缺失 0 个」。若某项找不到，退出码为 `1`，说明本章的工具表
与锁定版本已经对不上，需要重新核对——这正是升级版本后要做的第一件事。

## 注意事项

* 脚本证明的是「入口在源码里存在」，**不是**「这个 API 在你的环境里能跑通」。
  后者需要在装了匹配 CUDA/驱动或 CANN 的机器上实际编译运行。
* `T.printf` / `T.dump_tensor` / `TL_ASCEND_EXCEPTION_DUMP` 只在
  `_vendor/tilelang-ascend-v0.1.2.000/`（昇腾适配器仓库）里，不在主干 `v0.1.15` 里——
  这是昇腾「双仓库协作」的直接结果，见第 25、31 章。
* 位置取的是**首次匹配行**，可能是 `def` 定义，也可能是环境变量表 / 枚举里的一项。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上实际运行过，退出码 0。
未在任何加速器上运行过任何 kernel。
