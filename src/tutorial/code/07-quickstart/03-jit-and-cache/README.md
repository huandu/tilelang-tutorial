# 03-jit-and-cache · @tilelang.jit 与编译缓存

## 目的

把本章 6.1–6.3 节的三件事变成可执行的代码：

1. **lazy 模式**：`@tilelang.jit` 修饰“返回 `PrimFunc`”的函数 → 调用得到
   `JITKernel`，可以反复调用、查看生成源码；
2. **eager 模式**：`@tilelang.jit` 修饰“用 DSL 直接写 body”的函数 → 调用即编译并
   执行，返回结果张量；
3. **缓存**：同一份 kernel + 同样参数第二次编译应命中；改参数则 miss；
   演示 `tilelang.disable_cache()` / `tilelang.enable_cache()`。

## 运行前提

| 项 | 要求 |
| --- | --- |
| Python | >= 3.10 |
| tilelang | 已安装 |
| 加速器 | 一块 NVIDIA GPU |

## 运行命令

```bash
python jit_and_cache.py
TILELANG_VERBOSE=1 python jit_and_cache.py                  # 观察 cache hit / miss 日志
TILELANG_CACHE_DIR=/tmp/tl-cache python jit_and_cache.py    # 换缓存目录，强制冷启动
```

## 预期输出

```text
lazy  : type(kernel) = JITKernel
lazy  : correctness OK；同一个 kernel 调用了两次
lazy  : block_N=512 是另一份 kernel（另一条缓存）
eager : correctness OK；调用即编译 + 执行
----- generated CUDA (lazy, block_N=256) -----
extern "C" __global__ void __launch_bounds__(256) main_kernel(...) { ... }
cache : 已演示 disable_cache()/enable_cache() 一轮
```

**正确性判据**：所有 `assert_close` 通过。
**缓存判据**：第一次运行与第二次运行的编译耗时差距明显；`TILELANG_VERBOSE=1` 时
第二次应看到“命中缓存”相关日志，而不是重新走 nvcc。

## 验证级别

**仅静态核对。** 作者本机无 NVIDIA GPU、未安装 tilelang，未运行本脚本。
脚本用到的 API 名与签名在 `_vendor/tilelang-v0.1.15/` 里核对过：

* `tilelang.jit` 装饰器：`tilelang/jit/__init__.py:571-635`
* `JITImpl.compile`：`tilelang/jit/__init__.py:466`
* `JITKernel.get_kernel_source`：`tilelang/jit/kernel.py:532-543`
* `tilelang.disable_cache` / `enable_cache`：`tilelang/env.py:438-441`、`:558-559`
* 缓存目录默认值 `~/.tilelang/cache`：`tilelang/env.py:368`

## 在真实机器上的验证步骤

1. 装好 tilelang + CUDA 版 torch，确认 `torch.cuda.is_available()`。
2. 第一次 `python jit_and_cache.py`，记下总耗时。
3. 第二次再跑，观察是否明显变快、`~/.tilelang/cache/<version>/<platform>/` 下是否
   已经躺着上一次编译的产物。
4. 设 `TILELANG_DISABLE_CACHE=1` 再跑一次，确认回到“每次重新编译”，以确认缓存
   确实是你观察到的时间差的来源。
