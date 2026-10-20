# 05-ascend-first-kernel · 昇腾侧的第一个 kernel

## 目的

对照本章第 7 节：同一个 `C = A + B`，在昇腾上写会看到三处结构性差异——
方言换成 `tilelang.ascend.language`、`T.Kernel` 变成一维核网格（没有 `threads=`）、
线程域改由 `with T.SimtVF(threads=...)` 声明。

这份文件用 **v0.1.15 主干里**的 `tilelang.ascend.language` 写，与
`_vendor/tilelang-ascend-v0.1.2.000` 那个独立仓库的方言不是同一套。两个方言的区别
见本章 7.2 节，细节留到第 19、25 章。

**一处有意的简化（请知悉）**：官方 `example_vecadd.py` 在每个 UB buffer 上还加了一行
`T.annotate_buffer_versions({...})`（官方例子里是 `num_stages` 份）。本示例**没有**写它，
因为按 `annotate_buffer_versions` 的 docstring，`num_stages > 1` 时多缓冲份数是
**由自动调度器（Z3）算出来的**，这条注解只是"锁定份数 / 控制索引模式"的可选手段
（`_vendor/tilelang-v0.1.15/tilelang/ascend/language/annotations.py:19-38`）。
换句话说：本示例依赖自动调度，官方示例显式锁定。**这是语义上应当等价、但作者未在
NPU 上验证过的一处差异**，第 11、19、23 章会展开。

## 运行前提

| 项 | 要求 |
| --- | --- |
| 硬件 | 昇腾 NPU（`tilelang-ascend` 官方在 A2 / A3 上验证过） |
| CANN | >= 8.3.RC1 |
| Python | >= 3.10 |
| torch-npu | >= 2.6.0.RC1，`torch.npu.is_available()` 为 True |
| tilelang | Ascend 版（见本章 3.4 节） |

## 运行命令

```bash
# 先按本章 3.4 节把 CANN 与 Ascend 版 tilelang 装好并 source set_env.sh
python vecadd_ascend.py

# 官方完整版（SIMT + SIMD 两种模式、带性能回归）：
python -m examples.ascend.example_vecadd --mode simt
```

## 预期输出

```text
ascend vecadd: correctness OK
----- generated source (ascend) -----
...  # Ascend C / PTO 侧源码，可看到 MTE 搬运、Cube/Vector 分工与自动插入的同步
```

**正确性判据**：`torch.testing.assert_close(c, a + b)` 通过。
**性能口径**：本脚本不测性能。官方示例用 `do_bench(..., backend="msprof")` 报告
有效带宽，单位 GB/s，口径见 `examples/ascend/README.md:19-25`。

## 验证级别

**仅静态核对。** 作者本机是 macOS，没有昇腾 NPU，未运行本脚本。
API 名与签名在 `_vendor/tilelang-v0.1.15/tilelang/ascend/language/` 里核对过：
`kernel.py:139`（`T.Kernel`，一维核网格、无 `threads=`）、`frame.py:107`（`T.SimtVF`）、
`allocate.py:15`（`alloc_shared` → UB）。结构对照
`_vendor/tilelang-v0.1.15/examples/ascend/example_vecadd.py:11-44`。

## 在真实机器上的验证步骤

1. `source {cann}/ascend-toolkit/set_env.sh`；`python -c "import torch, torch_npu; print(torch.npu.is_available())"`
   打印 `True`。
2. 运行 `python vecadd_ascend.py`，确认 `correctness OK`。
3. 把 `get_kernel_source()` 的输出和 CUDA 版对照：找 MTE 搬运、`SimtVF` 对应的
   Vector 代码、以及**自动插入**的同步（在没有手写 barrier 的前提下）。
4. 用 `msprof` 复核带宽口径（第 12、19 章会展开）。
