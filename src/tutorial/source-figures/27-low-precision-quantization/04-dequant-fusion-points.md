# 图 4：反量化可以融合在哪一级

用于正文第 4.3 节。同一个「W 是低精度、要乘 scale 才能用」的问题，有三个融合位置，
代价和适用条件都不同。

```text
  ┌──────────────── 位置 A：搬入时（prologue） ────────────────┐
  │  GM(int4) --T.copy--> SMEM --解码--> 寄存器(fragment)       │
  │                                --T.gemm--> 累加器            │
  │  代表：examples/dequantize_gemm/*（_tir_packed_to_unsigned_  │
  │        convert / w4a8 的符号扩展 / mxfp 的 twiddling 解码）  │
  │  优点：反量化只做一次，之后整条 GEMM 路径是普通 fp16 GEMM     │
  │  缺点：SMEM 要按解压后的尺寸开（4 bit -> 16 bit 是 4 倍）     │
  └───────────────────────────────────────────────────────────┘

  ┌────────── 位置 B：主循环里（Ascend 的 L1 → L0）───────────┐
  │  GM --MTE--> L1(int8 + e8m0 scale) --L1→L0--> L0A/L0B      │
  │      在搬运指令里读 scale 并按粒度作用到每个 K 块           │
  │  代表：CATLASS 的 BlockMmadTla<MmadMx<...>>（L1_SCALE_      │
  │        FACTOR_K），以及 tilelang/ascend/language/           │
  │        gemm_op.py 里粒度固定为 32 的 T.gemm_blockscaled    │
  │  优点：Cube 直接吃低精度操作数，L1 带宽省下来               │
  │  缺点：粒度被硬件模板固定，改粒度要改模板                  │
  └───────────────────────────────────────────────────────────┘

  ┌──────────── 位置 C：累加之后（epilogue）──────────────────┐
  │  L0C/寄存器里的 int32 或 fp32 累加结果 --乘 scale--> 输出   │
  │  代表：tilelang-ascend 的 example_quant_batch_matmul：       │
  │        T.gemm_v0 累加到 int32 L0C，再 c_scale *= scale_ub    │
  │  优点：整数累加**精确**，scale 只做一次，误差只来自权重量化  │
  │  缺点：只适用于「scale 与 K 无关」的情形（per-tensor /       │
  │        per-channel）；如果是 per-K-block，就得回到位置 B     │
  └───────────────────────────────────────────────────────────┘
```

## 选择判据

| 情况 | 融合位置 | 理由 |
| --- | --- | --- |
| scale 与 K 无关（per-tensor / per-channel，如 W8A8、Qwen 系列的 weight-only） | C（epilogue） | 整数累加精确，scale 乘一次即可；不占额外 SMEM |
| 激活与权重都是 per-tensor 的 int8 | C 或 A | 两者等价，C 更省事 |
| 权重是 4 bit，需要 group 级 scale（group 沿 K） | A（搬入时） | 解压与 scale 在同一处完成，之后走普通 GEMM |
| 硬件提供块缩放的 MMA（SM100 的 tcgen05、昇腾 950 的 Cube） | B | 省一次解压与一次 SMEM 往返，代价是粒度被模板固定 |
| 需要 per-K-block 的 fp8 且没有块缩放硬件 | A + 分段 | 分成 K 段各自 scale，等价于把粒度换成 block_K |

## 一个容易写错的细节

位置 C 的「整数累加是精确的」只在**累加器是整数**时成立。如果权重是 fp8 而累加器
是 fp16，那么位置 C 的 scale 乘法仍然精确，但累加本身已经有误差——
这类错误在「只换累加器」的实验里就能看出来（见 `code/03-error-model/error_model.py`
实验 C：同样的输入，累加器从 fp16 换成 fp32，中位误差从 `1.2e-4` 掉到 `1.2e-9`）。

## 来源

* `_vendor/tilelang-v0.1.15/examples/dequantize_gemm/README.md`（位置 A 的原型）。
* `_vendor/tilelang-v0.1.15/examples/dequantize_gemm/example_dequant_gemm_w4a8.py:10-22`（符号扩展）。
* `_vendor/tilelang-ascend-v0.1.2.000/examples/quant_batch_matmul/example_quant_batch_matmul.py:49-88`（位置 C）。
* `../tilelang-corpus/06-huawei-ascend/raw/catlass/include/catlass/gemm/block/block_mmad_mx_tla.hpp:101-110,522-563`（位置 B）。
* `../tilelang-corpus/06-huawei-ascend/raw/catlass/experimental/attention/ascend950_fp8_mx_flash_attention_infer/mxfp8_flash_attention_infer.md`（「在 Cube 矩阵乘的 L1→L0 阶段完成反量化」）。
