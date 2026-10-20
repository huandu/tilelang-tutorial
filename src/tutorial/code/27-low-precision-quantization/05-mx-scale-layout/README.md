# 示例 05：block-scaled 的 scale 打包与解包

## 目的

block-scaled GEMM 最容易错的地方不是 MMA，而是 scale 的打包与索引。脚本用 numpy
复刻锁定版本示例里的三个函数，把「正确打包 + 正确解包」作为基准，再看四种典型
错误各造成多大的结果偏差。

复刻对应的源码：

| 本脚本 | 锁定版本 |
| --- | --- |
| `ceil_to_ue8m0` | `examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py:543-546` |
| `pack_sf_u8_to_u32_1d` | 同上 `:518-525` |
| `unpack_sf_u32_1d` | 同上 `:509-515` |
| `quantize_and_pack` | 同上 `:528-572`（`quantize_fp8_with_packed_ue8m0`） |
| `blockscaled_gemm` | 同上 `:575-620`（`blockscaled_gemm_ref`） |

## 运行前提

Python 3.9+ 与 numpy。**不需要 GPU / NPU**。

## 运行命令

```bash
python3 mx_scale_layout.py
python3 mx_scale_layout.py --m 4 --n 8 --k 2048 --gran 128
```

## 期望输出

作者在写作机上的实际输出（2026-10-01，Python 3.11.7 + numpy 1.24.1，默认参数）：

```text
形状：A[4,1024]、B[8,1024]，sf_granularity_k=128
每行 scale 字节数 = ceil(K/128) = 8；每 4 个字节打包成 1 个 uint32
SFA 扁平长度 = ceil(8/4) * M = 8（group-major：SFA[g*M + m]）
打包后的第一个 word = 0x7a787a79 -> 字节 [121, 122, 120, 122]
解包回来与原始字节一致：True

正确布局下的输出范数 = 7532.512010（作为下面所有偏差的分母）

解码方式                                        相对 L2         说明
0 正确解包                                        0.000e+00       基准
1 字节序写反（w0 放最高字节）                             5.583e+00       scale 索引整体错位
2 用行主序索引（SFA[m*G + g]）                        2.469e+00       行与组互换
3 生产端粒度 32、消费端按 128 读                         2.773e+00       scale 覆盖范围差 4 倍

补充：当组数 = 1（K 不超过 512）时，group-major 与 row-major 的解包结果相同：
      True。所以行主序这个 bug 只在 K 超过 4 个粒度块时才暴露。
```

## 三条可从输出里直接读出来的判据

1. **三种错误的相对 L2 都在 1.5–5.6，也就是「和真值同一个量级」**。这不是精度问题，
   是逻辑错误；调容差没有意义。
2. **行主序的 bug 会藏起来**：当 K 只有 1 个组（K ≤ 4×粒度）时两种索引恰好等价
   （输出末尾的 `True`）。只在 K 变大之后才暴露——所以 layout 的单测必须用
   **至少 2 个组**的 shape。
3. **粒度约定必须写进接口**。生产端按 32 粒度打包、消费端按 128 粒度读，结果偏差
   2.77。锁定版本里 NVIDIA 的 SM100 MXFP8 示例用 128，昇腾 CATLASS 的
   `MX_SCALE_GROUP_NUM` 是 32，PTX 的原生 MX block 也是 32。

## 与硬件布局的对应关系

* 每个 word 覆盖 4 个 scale 字节，每个字节覆盖 `sf_granularity_k` 个 K 元素，
  所以一个 word 覆盖 `4 × sf_granularity_k` 个 K 元素；源码里
  `sf_load_period = sf_granularity_k * 4 // block_K` 就是按这个来的。
* 全局是 group-major：`SFA[g*M + m]`，`g` 是 4 个字节一组的组号。
* scale 字节就是 fp32 的 biased exponent，所以 `scale = 2^(byte - 127)`；
  填充值 127 表示 `2^0 = 1`。

## 注意事项

* 这里的 E8M0 与硬件一致（2 的整数次幂），但 fp8 元素的舍入用「最近码本值」，
  与硬件 round-to-nearest-even 不完全逐位一致。
* 没有模拟 TMEM：真实路径上 scale 还要经 `tcgen05_sf_warp_transpose` 与
  `tcgen05_cp_warpx4` 写入 TMEM 的 4 个子列，见正文第 4.1 节。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在 macOS (Apple Silicon)、Python 3.11.7 + numpy 1.24.1 上实际运行过，
退出码 0。未在任何加速器上运行过任何 kernel。
