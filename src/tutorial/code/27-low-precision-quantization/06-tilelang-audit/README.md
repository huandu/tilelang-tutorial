# 示例 06：把本章的源码结论变成可复现的检查

## 目的

回答两个问题，并且**可复现**：

1. 正文里引用的那些断言（有哪些 dtype、块缩放的粒度是多少、scale 怎么打包、
   `T.gemm_blockscaled` 的参数叫什么、昇腾侧粒度是 32 还是 128），在锁定版本的
   工作树上到底成不成立？
2. “v0.1.15 主干没有 `tilelang/quantize/` 包”“SM100 的 2CTA 变体把
   `sf_granularity_k` 固定成 128”这类版本边界，能不能被机器复现？

44 条检查按前缀分成四组：`S` = `_vendor/tilelang-v0.1.15/`，
`A` = `_vendor/tilelang-ascend-v0.1.2.000/`，`R` = `tilelang-corpus`，
`N` = 负向断言。

## 运行前提

Python 3.9+，只用标准库（`argparse`、`dataclasses`、`pathlib`）。
**不需要 GPU / NPU / PyTorch / TileLang。**

检查需要锁定版本的源码。脚本会从自身位置向上找
`<dir>/tilelang-v0.1.15` 或 `<dir>/_vendor/tilelang-v0.1.15`；找不到时打印 `skipped`
并以 0 退出，不会因为环境不齐而失败。

## 运行命令

```bash
# 在 code/06-tilelang-audit/ 下运行
python3 tilelang_audit.py

# 或者显式给根目录
python3 tilelang_audit.py \
    --vendor  ../../../../_vendor/tilelang-v0.1.15 \
    --ascend  ../../../../_vendor/tilelang-ascend-v0.1.2.000 \
    --corpus  ../../../../../tilelang-corpus
```

## 期望输出

作者在写作机上的实际运行结果（2026-10-01，Python 3.11.7）：

```text
root[vendor] = .../_vendor/tilelang-v0.1.15
root[ascend] = .../_vendor/tilelang-ascend-v0.1.2.000
root[corpus] = .../tilelang-corpus

ok  S1  ~ S30   dtype 系统 / fp8 / int4 / dequantize / block-scaled 的 30 条源码事实
ok  A1  ~ A5    昇腾方言粒度 32、e4m3_float8、quant GEMM 的 epilogue 反量化与判据
ok  R1  ~ R7    CATLASS 的 MX 粒度与模板、PTX 的块缩放与格式定义
ok  N1  ~ N2    「主干没有 tilelang/quantize/」「示例下有 dequantize_gemm/quantize/」

通过 44 / 失败 0 / 跳过 0（共 44 条）
```

草稿期这个脚本确实抓到过两类问题，说明它是有效的：

* `R6` 一开始没命中，因为 PTX 的两张表里同一行内容的换行位置不同，我原先写死的
  子串跨了行；改成两个更短、各自完整的锚点后通过。
* `S5` 一开始没命中，因为我凭印象写了一个不存在的锚点 `tilelang.language.dtypes`；
  改成源码里真实的断言文本（`hasattr(torch, "float8_e8m0fnu")` 与
  `upgrade torch >= 2.8.0`）后通过。

## 检查清单（正文引用编号）

| id | 结论 |
| --- | --- |
| S1 | `tilelang/dtypes.py` 是 `language/dtypes.py` 的再导出 |
| S2 | 语言面登记了 `int4` / `float8_e4m3fn` / `float8_e8m0fnu` 等名字 |
| S3 | torch 扩展 dtype 列表含 `e4m3fn` / `e5m2` / `e8m0fnu` / `float4_e2m1fnx2` |
| S4 | `int4` 落到 torch 时用 `int8` 存储 |
| S5 | `e8m0fnu` 需要 torch ≥ 2.8 |
| S6 | FP4 有 packed（type_code 17）与 unpacked（type_code 131）两种存储 |
| S7 | 混合 dtype GEMM 只允许 f8f6f4 家族 |
| S8 | `determine_fp8_type`：CUDA 用 e4m3fn/e5m2，ROCm 默认 FNUZ、gfx950 例外 |
| S9 | 有专门的 fp8 负零清理函数 |
| S10 | `gemm_fp8` 示例：fp32 累加、用相似度而非逐元素比较 |
| S11 | `gemm_int4` 示例：`T.int4` + `T.int32` 累加 + `rtol=0, atol=0` |
| S12 | int4 打包：两个 int8 的低 4 位合成一个字节 |
| S13 | dequantize GEMM 在搬入时用位运算解码 |
| S14 | w4a8 的 int4→int8 符号扩展技巧 |
| S15 | fine-grained 版本按 128 位事务决定每线程元素数 |
| S16 | SM100 MXFP8 示例：blockscale (1,1,128)、`sf_load_period` 公式 |
| S17 | 4 字节拼 `uint32` 再转置成 group-major 扁平数组 |
| S18 | E8M0 由 fp32 指数位得到（`ceil_to_ue8m0`） |
| S19 | scale 填充值是 127（即 1.0），不是 0 |
| S20 | 参考实现按 `sf_granularity_k` 分块乘 scale |
| S21 | 2CTA 变体断言 `sf_granularity_k == 128` |
| S22 | `T.gemm_blockscaled` 带 `k_start` / `sf_a_granularity_k` / `sf_b_granularity_k` |
| S23 | `T.tcgen05_gemm_blockscaled`：A/B 是 FP8/FP6/FP4，SFA/SFB 是 TMEM 里的 E8M0 |
| S24 | SF ID 推导公式 `(k_start / granularity) % 4` |
| S25 | `tcgen05_cp_warpx4` 把 128 word 的块映射到 4 个 TMEM 列 |
| S26 | 没有块缩放实现的后端直接编译失败，不退化成普通 GEMM |
| S27 | MX 指令描述符里 e4m3=0 / e5m2=1 / e2m1=5 |
| S28 | TCGEN5MMA 的 fp16/bf16 路径要求 fp32 累加器，K 是 16 的倍数 |
| S29 | TCGEN5MMA 的 f8f6f4 家族要求 K 是 32 的倍数，累加器可以是 fp32 或 fp16 |
| S30 | WGMMA 的 fp8 路径要求 A 不转置、B 转置、K 是 32 的倍数 |
| A1 | 昇腾方言的 `T.gemm_blockscaled` 把 MX 粒度写死为 32 |
| A2 | 昇腾侧 fp8 的 dtype 名是 `e4m3_float8` / `e5m2_float8` |
| A3 | 昇腾 quant GEMM：int8 输入、int32 累加（L0C）、epilogue 里乘 scale |
| A4 | 昇腾 quant GEMM 的对拍判据是 `rtol=1e-2, atol=1e-2` |
| A5 | 昇腾 FP8 GEMM 需要 A5（Ascend 950）的 Cube 核 |
| R1 | CATLASS 的 `MX_SCALE_GROUP_NUM = 32` |
| R2 | CATLASS MX 模板以 `L1_SCALE_FACTOR_K` 参数化 |
| R3 | A8W4 MX 示例的 scale 类型是 `float8_e8m0_t` |
| R4 | CATLASS MXFP8 FlashAttention：block=32、在 L1→L0 阶段反量化 |
| R5 | PTX：tcgen05 的 scale 在 TMEM，SFB 要复制到 32 个 lane 分区 |
| R6 | PTX：mxf8f6f4 用 ue8m0，SFA_ID 选的是 TMEM 里的字节子列 |
| R7 | PTX：e4m3 无 inf、NaN 只有 `0x7f`/`0xff`；ue8m0 是 8 位指数 0 位尾数 |
| N1 | v0.1.15 主干**没有** `tilelang/quantize/` 包 |
| N2 | v0.1.15 的 `examples/dequantize_gemm/quantize/` 存在（N1 的对照） |

## 注意事项

* **这是启发式文本检查，不是编译器。** 命中只说明那句话在锁定版本里存在，
  不说明我们理解对了，也不说明签名在别处被兼容。
* 它只能证明“文件里出现了这些字符串”。如果上游把同一句话写在注释里而代码路径
  已经不可达，它同样会通过。
* 换版本（例如换到 `main`）后必须重跑；`VERSIONS.md` 规定结论只对锁定版本负责。

## 作者实际做到的验证级别

**CPU 逻辑验证**：在写作机的锁定版本工作树上实际运行过，44/44 通过，退出码 0。
未编译、未运行任何 TileLang kernel。
