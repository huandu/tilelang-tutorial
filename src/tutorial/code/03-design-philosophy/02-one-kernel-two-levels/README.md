# 示例 2：同一个 kernel 里的两层（composable 的最小实证）

## 目的

第 03 章 2.6 节给 "composable" 下了定义：*同一个 kernel 里，高层 tile 原语与低层 thread
原语可以共存，中间不需要任何模式切换或边界声明*。这个示例把这个定义写成一个最小可读的
kernel，并在注释里逐行标出每一句属于哪一层。

文件里只有一个 kernel，它做两件事：

1. **developer 层**：一个标准的多级分块 GEMM（`T.alloc_shared` / `T.alloc_fragment` /
   `T.copy` / `T.Pipelined` / `T.gemm`），全程没有线程索引；
2. **expert 层**：在同一段 kernel 体里，用 `T.get_thread_binding()` +
   `T.alloc_local` + `T.serial` 把 C 分块第一行的部分和算出来，每个线程各写一个槽位。

两份代码之间**没有 `if level == 2:` 之类的开关**，也不在不同的函数里——这就是本节要说明的
事情。

## 与官方示例的关系

两个片段都取自锁定版本 v0.1.15 的官方示例，只做了剪切与合并：

* developer 部分是 `examples/quickstart.py:18-46` 的结构；
* expert 部分是 `examples/gemv/example_gemv.py:46-76` 里
  “每个线程按 stride 跨步累加一部分”的写法。

没有引入任何本章正文没有提到的 API。

## 运行前提

* **NVIDIA GPU**（sm70 及以上）+ 可用的 CUDA toolkit + PyTorch(CUDA 版) +
  锁定版本的 TileLang v0.1.15（安装见 `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md`）。
* 不需要昇腾 NPU。本示例不使用任何昇腾 API。

在没有 GPU 的机器上仍可做两件离线检查：

```bash
python3 -m py_compile one_kernel_two_levels.py
python3 one_kernel_two_levels.py --static-only
```

## 运行命令

```bash
python3 one_kernel_two_levels.py
python3 one_kernel_two_levels.py --static-only   # 无 GPU 时
```

## 期望输出

**有 GPU 时**（读者应在自己机器上验证）：

1. 正确性判据：`torch.testing.assert_close(C, A @ B, rtol=1e-2, atol=1e-2)` 通过；
2. `Partial` 的校验：对某个 `(bx, by)`，`Partial[bx, by, :].sum()` 应等于
   `C_tile[0, :].sum()`（fp32 累加，允许 1e-2 级误差）；
3. 会打印生成设备源码中与 `T.gemm` 对应的片段行数，供对照。
4. **本示例不打印任何性能数字**——写作环境没有 GPU，任何延迟/吞吐都必须由读者在自己机器
   上产生，并写清测量条件。

**无 GPU 时**（作者实际跑过的路径）：

```
第 03 章示例 2：同一个 kernel 里的两层（仅静态核对）
--------------------------------------------------------
本文件在无 GPU 时只做语法与 API 核对，不能编译/运行 kernel。

用到的 API 与层次归属（v0.1.15）：
  T.Kernel                [developer]   tilelang/language/kernel.py:431
  T.alloc_shared          [developer]   tilelang/language/allocate.py:49
  ...
```

直接运行而不加 `--static-only` 时，脚本检测不到 `tilelang` 或 CUDA 会以退出码 2 结束，
并给出安装提示——这是预期行为，不是缺陷。

## 作者实际做到的验证级别

**仅静态核对。** 具体做了三件事：

1. `python3 -m py_compile one_kernel_two_levels.py` 通过；
2. 文件里出现的每个 TileLang API 都在 `_vendor/tilelang-v0.1.15/` 中 `grep -n` 核对过
   定义行与签名（清单见 `--static-only` 输出，也见 `../../references.md` 的 S1–S18）；
3. 两个片段与官方示例逐行对照过。

**没有**在真实硬件上编译或运行过。

## 一个必须说明的正确性前提

expert 段读的是 `C_shared`——它由上一句 tile 级 `T.copy(C_frag, C_shared)` 协作写入。
“协作写入 → 跨线程读取”之间的 barrier 由编译器的 `ThreadSync("shared")` pass 插入
（`_vendor/tilelang-v0.1.15/tilelang/cuda/pipeline.py:258-259`），用户不必手写
`__syncthreads()`。这一点在本章 4.3 节讲过，**但本示例没有在硬件上验证过**；
如果你把它改复杂了（例如在 expert 段里再写回 `C_shared`），请自己确认同步语义，
或显式插 `T.sync_threads()`。
