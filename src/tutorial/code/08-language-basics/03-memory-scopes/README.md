# 03-memory-scopes · buffer 与内存作用域

对应正文第 5 节。

## 目的

1. 把 `alloc_shared / alloc_fragment / alloc_local / alloc_var / alloc_global` 五个
   分配函数与它们各自的 scope 对上；
2. 说明为什么 `fragment` “写成块形状、实际在寄存器里”；
3. 说明昇腾侧 `UB / L1 / L0A / L0B / L0C` 在主库与适配器仓库里的**两种拼法**；
4. 给出一个能算“这块 tile 占多少字节、开几份缓冲还放不放得下”的小工具。

## 文件

| 文件 | 类型 | 验证级别 |
| --- | --- | --- |
| `scope_footprint.py` | 纯标准库计算器 | **CPU 逻辑验证** |
| `alloc_scopes.py` | TileLang 示例 | **仅静态核对** |

## 运行 `scope_footprint.py`

```bash
python3 scope_footprint.py
python3 scope_footprint.py --tile 128 256 --dtype float32 --versions 2 --capacity 196608
```

前提：Python 3.9+，只用标准库。作者在 macOS (Apple Silicon) + Python 3.11 上
实际运行过。

典型输出（真实输出，节选）：

```text
Against a 192 KiB on-chip budget (196608 B):
  A tile (fp16, no buffering)        (128, 128)   float16   x1  =    32 KiB
  A tile (fp16, 2-stage)             (128, 128)   float16   x2  =    64 KiB
  C fragment (fp32, per block)       (128, 128)   float32   x1  =    64 KiB
  C tile in UB (fp32)                (128, 256)   float32   x1  =   128 KiB   cap 192 KiB  fits
  C tile in UB (fp32, 2-stage)       (128, 256)   float32   x2  =   256 KiB   cap 192 KiB  DOES NOT FIT
  same tile in fp16                  (128, 256)   float16   x2  =   128 KiB   cap 192 KiB  fits

  64 KiB fragment / 128 threads = 512 B = 128 fp32 registers per thread
```

**注意**：`196608`（192 KiB）只是昇腾官方文档里出现的**某个型号示例值**，不是
通用规格（见第 6 章）。真实设备上要用 `GetCoreMemSize` 查。这个脚本算的是
`元素数 × bits/8 × 份数`，不含任何分配器开销与对齐。

## `alloc_scopes.py` 的运行前提（未在本机运行）

* 运行前提：TileLang `v0.1.15`；`tmem_accumulator` 需要 Blackwell（sm_100+）；
  `ascend_scope_showcase` 需要昇腾 NPU + CANN + `torch_npu`。
* 静态检查命令（本机实际执行过）：

  ```bash
  python3 -m py_compile alloc_scopes.py    # 通过
  ```

* 在有卡的机器上如何验证：把 `scope_showcase.compile(M=..., N=..., K=...)` 跑通，
  然后看生成的 CUDA 源码里是不是有 `__shared__`（A_s/B_s）、
  `__launch_bounds__` 下的局部数组（C_f/scratch）与单标量变量（scale）。
* 性能口径：本示例不做性能断言。
* 作者实际做到的验证级别：**仅静态核对**（`py_compile` 通过 + 每个
  `alloc_*` 的签名与默认 scope 在
  `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py`、
  `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py`、
  `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/allocate.py` 里核对过）。
  **未在硬件上运行**。

## 一处必须记住的版本差异

主库 `v0.1.15` 的昇腾方言里**没有** `T.alloc_ub`：UB 用 `T.alloc_shared` 分配
（`_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:138`）。
`T.alloc_ub` / `T.alloc_L1` / `T.alloc_L0A` / ... 这套大驼峰拼法属于适配器仓库
`_vendor/tilelang-ascend-v0.1.2.000`，而且它的 scope 名也不同
（`shared.ub` / `wmma.matrix_a` / `wmma.matrix_b` / `wmma.accumulator`）。

## 原文出处

* `_vendor/tilelang-v0.1.15/tilelang/language/allocate.py:49-191, 254-289`
* `_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py:15-80, 103-125`
* `_vendor/tilelang-ascend-v0.1.2.000/tilelang/language/allocate.py:140-157`
* `_vendor/tilelang-v0.1.15/tilelang/ascend/README.md:131-151`
* `_vendor/tilelang-v0.1.15/docs/programming_guides/language_basics.md:152-172`
* `_vendor/tilelang-v0.1.15/docs/get_started/overview.md:76-90`
