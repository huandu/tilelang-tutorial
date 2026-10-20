# compile-path-check：逐跳校验编译路径

## 目的

正文图 2 与 4.1–4.4 节声称了一条具体的调用链：

```text
@tilelang.jit → JITImpl → get_tir → cache.cached → JITKernel._compile_artifact
 → engine.lower.lower_with_context → BackendContext.lower → PassPipeline.lower
 → <backend>/pipeline.py → device/host codegen → CompiledArtifact → adapter → 可调用对象
```

本脚本把这条链拆成 30 个「跳」，每跳要求：文件存在、且声称的行号上确实出现声称的符号。
任何一跳对不上，脚本以退出码 1 失败。

## 运行前提

* Python 3.8+（标准库）。
* `_vendor/tilelang-v0.1.15/` 存在。
* **不需要** GPU/NPU，**不需要** 安装 tilelang、TVM 或 torch。

## 运行命令

```bash
python3 check_compile_path.py
python3 check_compile_path.py --vendor /path/to/checkout
```

## 期望输出（v0.1.15 上）

```text
compile-path hops: 30/30 verified
  ok   E01 装饰器                    tilelang/jit/__init__.py:588
  ok   E02 装饰器 → JITImpl           tilelang/jit/__init__.py:270
  ...
  ok   E30 MakePackedAPI 的 C++ 本体 src/transform/make_packed_api.cc:2
```

**正确性判据**：`30/30 verified` 且退出码 0。

## 它能证明什么、不能证明什么

能证明：这条链上的每个符号在 v0.1.15 的源码里确实存在，且位置与正文一致——
也就是「正文没有把不存在的 API 写进去」。

不能证明：编译真的会这么走。真实编译会按 `pass_configs`、device codegen 开关
（`enable_host_codegen` / `enable_device_compile`）与缓存命中情况分岔；本脚本只做
静态可达性检查。要验证运行时的真实顺序，需要 `TILELANG_*` 环境变量 +
`--dump-ir`（见正文 4.7 节与官方 `lower_trace` 工具）。

## 验证级别

**在本机运行过**（macOS Apple Silicon，Python 3.11，2026-10-01）。
