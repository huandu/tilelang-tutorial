# 01-first-kernel · 一维向量加法

## 目的

本章“第一个 kernel”的完整、可直接复制的版本。一个文件里包含两种写法：

* **版本 A（`vector_add_direct`）**：最朴素的写法。只有 `T.Kernel` + `T.Parallel`，
  直接在全局内存上读写。对应官方教程 `docs/deeplearning_operators/elementwise.md`
  的第一段示例。
* **版本 B（`vector_add_buffered`）**：先 `T.alloc_fragment` 分配片上 buffer，用
  `T.copy` 把 tile 搬进来，算完再搬出去。对应官方
  `examples/elementwise/example_elementwise_add.py` 的结构。

两段代码解决的问题完全相同（`C = A + B`），差别只在“数据先不先落到片上 buffer”。
对照读它们，是理解 `T.copy` 与 `T.Parallel` 分工的最快方式。

## 运行前提

| 项 | 要求 |
| --- | --- |
| Python | >= 3.10 |
| tilelang | 已安装（wheel 或源码构建，见本章 3.2 / 3.3 节） |
| 加速器 | 一块 NVIDIA GPU；脚本用 `torch.cuda` 造数据 |
| torch | 已安装 CUDA 版 |

## 运行命令

```bash
python vecadd_tilelang.py                  # 两种写法都跑
python vecadd_tilelang.py --variant a      # 只跑版本 A
python vecadd_tilelang.py --n 4096 --block-n 128 --threads 128
```

## 预期输出

```text
[variant a] correctness OK (n=1048576, block_N=256, threads=256)
----- generated CUDA (variant a) -----
extern "C" __global__ void __launch_bounds__(256) main_kernel(...) {
  ...   # 预期看到 128-bit 向量化访存（uint4）与半精度成对加法
}
[variant b] correctness OK (n=1048576, block_N=256, threads=256)
```

**正确性判据**：`torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)` 通过。
**性能口径**：本脚本不测性能。要测吞吐用 `kernel.get_profiler().do_bench()`
（见本章 6.3 节），并自行固定 shape / dtype / backend / warmup / repeat。

## 验证级别

**仅静态核对。**

作者本机是 macOS（Apple Silicon），既没有 NVIDIA GPU 也没有安装 tilelang，
**没有运行过这个脚本**。所谓“静态核对”指的是：

1. 脚本可以 `python3 -m py_compile` 通过（语法层面无误）；
2. 用到的每个 API 名与签名都在 `_vendor/tilelang-v0.1.15/` 里逐个核对过，
   见本章 `references.md` 的 S 系列条目。

## 在真实机器上的验证步骤

1. 按本章 3.2 或 3.3 节安装 tilelang；`python -c "import tilelang; print(tilelang.__version__)"`
   能打印版本号。
2. `python -c "import torch; print(torch.cuda.is_available())"` 打印 `True`。
3. 运行 `python vecadd_tilelang.py`，确认两条 `correctness OK`。
4. 把打印出的 CUDA 源码贴到官方 `elementwise.md` 的生成结果旁边逐行比对
   （见本章第 5 节）。若看到 `uint4` 或 `nv_bfloat162`，说明命中了向量化。
5. 想确认缓存生效：连续运行两次，第二次不应重新走 nvcc（可用
   `TILELANG_VERBOSE=1 python vecadd_tilelang.py` 观察日志）。
