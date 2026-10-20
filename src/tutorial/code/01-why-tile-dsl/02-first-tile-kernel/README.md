# 示例 2：tile 视角的 GEMM + ReLU

## 目的

让读者在第 01 章就看见一个 **tile 程序长什么样**。重点不是语法（那是第 7–9 章），而是
四件事：

1. 网格从“元素的网格”变成“tile 程序的网格”；
2. 数据显式声明存储位置（`T.alloc_shared` / `T.alloc_fragment`）；
3. 搬运与计算是 tile 级原语（`T.copy` / `T.gemm`）；
4. 多缓冲与重叠只是一个参数（`T.Pipelined(..., num_stages=3)`）。

文件里还有一个逐元素相加的 kernel，用来对照“同一套 tile 写法如何覆盖最简单的算子”。

## 与官方示例的关系

本文件对标锁定版本 v0.1.15 中的：

```
_vendor/tilelang-v0.1.15/examples/quickstart.py
_vendor/tilelang-v0.1.15/examples/gemm/example_gemm.py
_vendor/tilelang-v0.1.15/examples/elementwise/example_elementwise_add.py
```

为了让文件能独立复制出去，本文件把 kernel 定义放进 `build_kernels()`，并把
`import torch / tilelang` 推迟到 `main()`。**除此之外，kernel 的写法与官方示例一致。**

## 运行前提

要真正编译运行，需要：

* **NVIDIA GPU**（sm70 及以上；sm90/sm100 能触达更多特性，但本示例只用最基础的原语）；
* 可用的 CUDA toolkit 与 PyTorch（CUDA 版）；
* 安装锁定版本的 **TileLang v0.1.15**：见
  `_vendor/tilelang-v0.1.15/docs/get_started/Installation.md`。

不需要昇腾 NPU。本示例不涉及昇腾 API。

在**没有 GPU** 的机器上，仍然可以做两项离线检查：

```bash
python3 -m py_compile tile_gemm_sketch.py    # 语法检查
python3 tile_gemm_sketch.py --static-only    # 打印 API 核对清单，不编译 kernel
```

## 运行命令

```bash
# 有 GPU 时
python3 tile_gemm_sketch.py

# 无 GPU 时
python3 tile_gemm_sketch.py --static-only
```

## 期望输出

**有 GPU 时**（用户应当在自己机器上验证）：

1. 正确性判据：`torch.testing.assert_close(c, torch.relu(a @ b), rtol=1e-2, atol=1e-2)`
   通过；逐元素相加部分用 `rtol=atol=1e-5` 通过。
2. 会打印生成的设备源码前 40 行，用于对照“哪一段对应 `T.copy`、哪一段对应 `T.gemm`”。
3. **本示例不打印性能数字。** 本教程的写作环境没有 GPU，任何延迟/吞吐数字都必须由读者
   在自己机器上产生，并写清测量条件（同一台机器、同一 shape、同一 dtype、同一口径）。

**无 GPU 时**（作者实际跑过的路径）：

```
第 01 章示例 2：tile 视角的 GEMM + ReLU（仅静态核对）
------------------------------------------------------------------
本文件在无 GPU 时只能做语法/API 核对，不能运行 kernel。

已核对的 API（v0.1.15）：
  T.Kernel          -> tilelang/language/kernel.py:431
  T.alloc_shared    -> tilelang/language/allocate.py:49
  ...
```

另外，直接运行而不加 `--static-only` 时，脚本会检测不到 `tilelang`/CUDA 并以退出码 2 或 3
结束，同时给出安装提示——这是预期行为，不是缺陷。

## 作者实际做到的验证级别

**仅静态核对。** 具体做了三件事：

1. `python3 -m py_compile tile_gemm_sketch.py` 通过；
2. 文件中出现的每个 TileLang API 都在 `_vendor/tilelang-v0.1.15/` 中 `grep -n` 核对过定义行
   与签名（清单见文件头部 docstring 与 `--static-only` 输出，也见 `../../references.md`
   的 S1–S16）；
3. 与三个官方示例逐行对照过用法。

**没有**在真实硬件上编译或运行过，因此文件中的正确性断言属于“官方示例同款写法”，
不是作者的实测结论。
