# 示例 04：低精度 GEMM 的结构与调用方式

## 目的

演示第 14 章第 5 节的四类低精度 GEMM 在**结构与调用方式**上的差别
（数值正确性见第 27 章）：

| 函数 | 变体 | 相对普通 GEMM 多出的东西 |
| --- | --- | --- |
| `gemm_fp8` | fp8（e4m3 / e5m2） | 只换 dtype；`block_K=64` 反映更严的 K 对齐 |
| `matmul_int4` | int4 | `T.int4` 输入、`T.int32` 累加、`pack_int4()` 打包 |
| `gemm_blockscaled` | block-scaled | 多一对 `SFA`/`SFB`，`k_start` 与 `sf_*_granularity_k` |
| （无独立函数） | dequantize | 在 `T.copy` 阶段顺手反量化，然后走普通 `T.gemm` |

## 运行前提

* NVIDIA GPU + CUDA + PyTorch(CUDA) + TileLang `v0.1.15`。
* fp8 需要 sm89 及以上；block-scaled 的真实示例（MXFP8）跑在 sm100 上。
* 本文件只用**通用** `T.gemm` / `T.gemm_blockscaled` 接口，不依赖某个具体
  示例的 TMEM/scale 打包细节，因此能否编译取决于后端对该 dtype 组合的支持。

## 运行命令

```bash
python3 low_precision_gemm.py
```

## 期望输出

```
fp8 dtype: <TileLang 返回的 fp8 类型> | 结构检查完成（未评估数值误差，见第 27 章）
int4 打包后形状: (1024, 512) （最后一维减半）
```

* **正确性判据**：本示例只断言「打包后的形状减半」这类结构性事实；
  fp8 / int4 的数值判据（误差上界、scale 表示、舍入顺序）属于第 27 章，
  这里不做断言。
* **本机没有 GPU，以上输出均未产生。** 作者实际做到的是静态核对：
  `py_compile` 通过；`T.int4`、`T.int32`、`determine_fp8_type`、
  `T.gemm_blockscaled(..., k_start, sf_a_granularity_k, sf_b_granularity_k)`
  逐个与 `_vendor/tilelang-v0.1.15/` 的源码/示例比对过。

## 已知局限

* `gemm_blockscaled` 里的 `SFA`/`SFB` 用了简化的 fp32 表示，**不是**真实
  硬件的打包格式。官方 sm100 示例用 `uint32` 每 4 个 K-block 打包一个
  E8M0 scale，并放在 TMEM 里（见 `examples/blockscaled_gemm_sm100/`）；
  昇腾侧则由 `T.gemm_blockscaled` 方言把粒度固定为 32。
* dequantize 没有独立函数：它在 TileLang 里表现为「搬入时做一次乘 scale」的
  `T.copy` + 普通 `T.gemm`，粒度（per-tensor / per-channel / per-group /
  fine-grained）决定元数据与额外算力。官方示例见
  `examples/dequantize_gemm/`。
