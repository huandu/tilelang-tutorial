# 03 · 复现 TMA 描述符的参数与约束

## 目的

`T.copy` 走 TMA 时，真正的“描述符”是在 **host 上**编码的：
`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:111-131` 的模板会生成
`cuTensorMapEncodeTiled(&tmap, dtype, rank, gaddr, gdim, gstride + 1, boxdim,
estride, interleave, swizzle, l2promotion, oobfill)`。

这个脚本把这段参数推导与约束检查写成可运行的形式，用来回答：

* 描述符的 `globalDim` / `globalStride` / `boxDim` / `elementStrides` 分别是什么？
* 为什么传给 `cuTensorMapEncodeTiled` 的是 `gstride + 1`？
* 什么样的 tile 形状会被 `CheckBulkLoad` / `CheckGlobalStrides` 拒绝？

## 运行前提

Python 3.9+，只用标准库。**不需要 GPU。**

## 运行命令

```bash
python3 tma_descriptor.py
```

## 期望输出（片段）

```text
[ok]   2D K-major f16 tile  dtype=f16  gshape=(4096, 4096)  box=(128, 64)
       globalDim      = [4096, 4096]
       globalStride   = [8192]   # 单位是字节，最内层被丢掉
       boxDim         = [128, 64]
       elementStrides = [1, 1]
  CUtensorMap 2D_K_major_f16_tile_tmap;
  ... cuTensorMapEncodeTiled(&2D_K_major_f16_tile_tmap, /*dtype=*/FLOAT16, /*rank=*/2,
      /*gaddr=*/A, /*gdim=*/{4096, 4096}, /*gstride=*/{8192},
      /*box=*/{128, 64}, /*estride=*/{1, 1},
      /*interleave=*/NONE, /*swizzle=*/NONE,
      /*l2=*/L2_128B, /*oob=*/NONE);
       globalStride[0] bytes = 8192 （16 的倍数: True）

[rejected] box last dim 30: 最后一维 30 * 16 bits = 480 bits，不是 128 的倍数
[rejected] stride-2 innermost: global 最内层 stride 是 2，不是 1
[rejected] stride 12 elements (24 B): globalStride[0] 字节数 24 不是 16 的倍数
[rejected] box dim 512 > 256: boxDim 512 超过上限 256
self-check: ok
```

注意 `globalDim[0]` 与 `boxDim[0]` 都是**最快变化**的那一维；`globalStride` 的单位是
**字节**，而且它只有 `rank - 1` 项——模板传的是 `globalStride + 1`。这与
`_vendor/tilelang-v0.1.15/src/cuda/op/copy.cc:2061-2091` 里
`desc.global_stride.push_back(byte_stride)`、以及
`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:111-131` 里的
`{0}_globalStride + 1` 一致。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11 上运行过。

## 简化了什么

1. 只实现 `cuTensorMapEncodeTiled`（普通 tile 描述符），不实现
   `cuTensorMapEncodeIm2col`。
2. `CUtensorMapDataType` 与 `CUtensorMapSwizzle` 用**符号名**打印；真实代码里是
   `cuda.h` 的整数常量
   （`_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:111-131` 用
   `(CUtensorMapDataType){1}` 这种形式插入）。
3. 不做 swizzle 布局推导（`AnalyzeTMASharedLayout`）；swizzle 由 shared 侧 layout
   决定，本脚本只按元素数校验 box 上限 256
   （`_vendor/tilelang-v0.1.15/src/cuda/op/tma_layout.h:41-42`，`kTmaMaxBoxDim = 256`）。
4. 不做 2D/3D 坐标推导（`TMABulkCopyPlan::TmaCoords`），只算描述符本身。
5. 不做 1D（`kBulkLoad1D`）路径——那条路径根本不建描述符，直接传 device 侧地址。
