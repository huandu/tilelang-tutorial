# 01-kernel-grid · `T.Kernel` 的网格与索引

对应正文第 3 节。

## 目的

1. 确认 `T.Kernel` 的**位置参数按顺序对应 x / y / z**，返回的绑定就是块索引；
2. 确认 `threads=` 是启动注解，省略时 SIMT 后端用 `DEFAULT_SIMT_THREADS = 128`；
3. 确认块内取索引的两条路：直接用 `as` 绑定的块索引，或用
   `T.get_thread_binding()` / `T.get_block_binding()` 取；
4. 确认昇腾方言的 `T.Kernel` 是**一维核网格、没有 `threads=`**。

## 文件

| 文件 | 类型 | 验证级别 |
| --- | --- | --- |
| `grid_index_model.py` | 纯标准库模型 | **CPU 逻辑验证** |
| `kernel_grid.py` | TileLang 示例 | **仅静态核对** |

## 运行 `grid_index_model.py`

```bash
python3 grid_index_model.py
```

前提：Python 3.9+，只用标准库。作者在 macOS (Apple Silicon) + Python 3.11 上
实际运行过。

典型输出（节选）：

```text
== x = N axis, y = M axis  (layout used by the quickstart) ==
M,N = 1000,700;  BM,BN = 256,128
per-axis tiles: M -> 4, N -> 6
grid = (6, 4)  ->  24 programs; x varies fastest
T.Kernel(6, 4, threads=128) as (bx, by)
pid   bx  by   row0  col0  m_tile n_tile
  0    0   0      0     0       0      0
  1    1   0      0   128       0      1
...
  6    0   1    256     0       1      0
...
tail: M pads 24 rows, N pads 68 cols -> LegalizeSafeMemoryAccess inserts the guards

== threads default ==
threads omitted -> the SIMT backend uses DEFAULT_SIMT_THREADS = 128
  threads=        None -> blockDim=(128, 1, 1) total=128
  threads=     (64, 2) -> blockDim=(64, 2, 1) total=128
```

这些都只是公式复算，**不是**硬件测量。

## `kernel_grid.py` 的运行前提（未在本机运行）

* 运行前提：NVIDIA GPU（CUDA）+ PyTorch(CUDA) + TileLang `v0.1.15`；
  文件末尾的 `ascend_style_core_grid` 需要昇腾 NPU + CANN + `torch_npu`。
* 静态检查命令（本机实际执行过）：

  ```bash
  python3 -m py_compile kernel_grid.py     # 通过
  ```

* 在有卡的机器上如何验证：

  ```python
  import torch, kernel_grid
  k = kernel_grid.elementwise_scale.compile(N=4096, block=256)
  x = torch.randn(4096, device="cuda", dtype=torch.float32)
  y = torch.empty_like(x)
  k(x, y)
  torch.testing.assert_close(y, x * 2.0)
  print(k.get_kernel_source())   # 对照 gridDim / blockIdx 是不是 (6,4)/(bx,by)
  ```

* 性能口径：本示例不做性能断言。
* 作者实际做到的验证级别：**仅静态核对**（`py_compile` 通过 + 上表列出的每个
  API 名与签名在 `_vendor/tilelang-v0.1.15/` 里核对过）。**未在硬件上运行**。

## 原文出处

* `_vendor/tilelang-v0.1.15/tilelang/cuda/language/kernel.py:13-55`
* `_vendor/tilelang-v0.1.15/tilelang/language/kernel.py:88-151, 392-427, 587-611`
* `_vendor/tilelang-v0.1.15/src/ir.cc:236-262`
* `_vendor/tilelang-v0.1.15/tilelang/transform/__init__.py:201`
* `_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py:139-172`
* 原型：`_vendor/tilelang-v0.1.15/docs/deeplearning_operators/elementwise.md:28-42, 75-95`
