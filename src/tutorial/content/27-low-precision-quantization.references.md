# 第 27 章 参考资料

本章所有结论都基于下表的本地资料或锁定版本源码。**版本以
[../../VERSIONS.md](../../VERSIONS.md) 为准：TileLang `v0.1.15`
（commit `a35f8ddf45eba16c21211ec8822d56ce5363036f`）、TileLang-Ascend
`TileLang-ascend-v0.1.2.000-release`（commit `89a9364e47a341fffdd65ed94786b1d4f9b6a78c`）。**
全章以锁定版本为唯一事实来源，不做版本漂移式的表述；
讨论 `main` 上更新的东西时会显式标注。

写作环境：macOS (Apple Silicon)、Python 3.11.7 + numpy 1.24.1；
**没有 NVIDIA GPU，也没有昇腾 NPU**。因此本章

* 数值结论全部来自 `code/01`–`code/05` 的 CPU 实跑（验证级别：CPU 逻辑验证）；
* 硬件结论全部来自 `_vendor/` 的源码或 corpus 的厂商文档；
* **没有任何一条性能数字来自本机实测**，正文里唯一引用的性能数字（BitNet）
  是上游样例给出的，见「关于性能数字的说明」。

**编号约定**：下表的 `S*` / `A*` / `R*` 与
`code/06-tilelang-audit/tilelang_audit.py` 的检查编号**一一对应**；
每张表末尾那些没有对应检查的条目编号继续往下排（`S31+`、`A6+`、`R11+`），
表示「本章引用过，但没有做成机器检查」。一条命令复查全部对应关系：

```bash
# 在 code/06-tilelang-audit/ 下运行；期望：通过 44 / 失败 0
python3 tilelang_audit.py
```

---

## 锁定的源码引用（`_vendor/tilelang-v0.1.15/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| S1 | `tilelang/dtypes.py:1-3` | `tilelang/dtypes.py` 只是 `language/dtypes.py` 的再导出 | 已核验（2026-10-01）：逐行读；`code/06` 检查 S1 |
| S2 | `tilelang/language/dtypes.py:121-142` | 语言面的 dtype 登记表 `_STR_TO_TVM_DTYPE_CALL`：`int4` / `float8_e4m3fn` / `float8_e8m0fnu` 等 | 已核验（2026-10-01）；`code/06` S2 |
| S3 | `tilelang/language/dtypes.py:81-103` | torch 扩展 dtype 列表：`e4m3fn` / `e5m2` / `e8m0fnu` / `float4_e2m1fnx2` | 已核验（2026-10-01）；`code/06` S3 |
| S4 | `tilelang/language/dtypes.py:187-238` | `__dtype_as_torch__`：`int4`→int8 存储 | 已核验（2026-10-01）；`code/06` S4 |
| S5 | `tilelang/language/dtypes.py:213-217` | `float8_e8m0fnu` 要求 torch ≥ 2.8.0 | 已核验（2026-10-01）；`code/06` S5 |
| S6 | `tilelang/language/dtypes.py:268-308` | FP4 的 packed（type_code 17）与 unpacked（type_code 131）两种存储及其判断函数 | 已核验（2026-10-01）；`code/06` S6 |
| S7 | `tilelang/language/dtypes.py:319-341` | `validate_gemm_ab_dtypes`：混合 dtype 只允许 f8f6f4 家族 | 已核验（2026-10-01）；`code/06` S7 |
| S8 | `tilelang/language/fp8.py:9-27` | `determine_fp8_type`：CUDA 用 `e4m3fn`/`e5m2`，ROCm 默认 FNUZ、gfx950 例外 | 已核验（2026-10-01）；`code/06` S8 |
| S9 | `tilelang/utils/tensor.py:25-29` | `fp8_remove_negative_zeros_`：把 fp8 的负零写回 `0x00` | 已核验（2026-10-01）；`code/06` S9 |
| S10 | `examples/gemm_fp8/example_tilelang_gemm_fp8.py:7-11,16-35,53` | fp8 GEMM：`accum_dtype=T.float32`、`block_K=64`、相似度判据 `diff < 1e-3` | 已核验（2026-10-01）；`code/06` S10；**未运行** |
| S11 | `examples/gemm_int4/example_tilelang_gemm_int4.py:19-36` | `T.int4` 张量 + `T.int32` 累加 + `T.gemm` | 已核验（2026-10-01）；`code/06` S11；**未运行** |
| S12 | `examples/gemm_int4/example_tilelang_gemm_int4.py:59-67,101` | `pack_int4` 的打包公式与 `rtol=0, atol=0` 的精确相等断言 | 已核验（2026-10-01）；`code/06` S12；**未运行** |
| S13 | `examples/dequantize_gemm/README.md` | dequantize GEMM 的最小原型（`_tir_packed_to_unsigned_convert` + `T.gemm`） | 已核验（2026-10-01）；`code/06` S13；**未运行** |
| S14 | `examples/dequantize_gemm/example_dequant_gemm_w4a8.py:10-22` | int4→int8 的符号扩展技巧（左移 4 位 + 算术右移 4 位） | 已核验（2026-10-01）；`code/06` S14；**未运行** |
| S15 | `examples/dequantize_gemm/example_dequant_gemm_fine_grained.py:43-44,63-88` | 按 128 位事务决定每线程元素数；解码后写回 shared 再 `T.gemm` | 已核验（2026-10-01）；`code/06` S15；**未运行** |
| S16 | `examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py:1-52` | MXFP8 示例的头部约定：blockscale (1,1,128)、`sf_load_period`、`sf_k_groups`、`uint32` 打包注释 | 已核验（2026-10-01）；`code/06` S16；**未运行** |
| S17 | 同上 `:509-525` | `unpack_sf_u32_1d` / `pack_sf_u8_to_u32_1d`：4 字节拼 `uint32` 后转置成 group-major | 已核验（2026-10-01）；`code/06` S17；**未运行** |
| S18 | 同上 `:543-562` | `ceil_to_ue8m0` 与 `quantize_fp8_with_packed_ue8m0`：`amax/448` 向上取到 2 的幂、E8M0 字节取自 fp32 指数 | 已核验（2026-10-01）；`code/06` S18；**未运行** |
| S19 | 同上 `:566` | scale 的填充值是 127（即 `2^0 = 1`），不是 0 | 已核验（2026-10-01）；`code/06` S19；**未运行** |
| S20 | 同上 `:575-620` | `blockscaled_gemm_ref`：按 `sf_granularity_k` 分块、逐块乘 scale | 已核验（2026-10-01）；`code/06` S20；**未运行** |
| S21 | 同上 `:195` | 2CTA 变体断言 `sf_granularity_k == 128` | 已核验（2026-10-01）；`code/06` S21；**未运行** |
| S22 | `tilelang/language/gemm_op.py:275-320` | target-neutral 的 `T.gemm_blockscaled(A, B, C, SFA, SFB, k_start, sf_a_granularity_k, sf_b_granularity_k)` | 已核验（2026-10-01）；`code/06` S22 |
| S23 | `tilelang/cuda/language/gemm_op.py:313-395` | `T.tcgen05_gemm_blockscaled`：A/B 为 FP8/FP6/FP4、SFA/SFB 为 TMEM 里的 E8M0、2CTA 的约束 | 已核验（2026-10-01）；`code/06` S23 |
| S24 | `tilelang/cuda/intrinsics/macro/tcgen05_macro_generator.py:629-632` | SF ID 的推导公式 `(k_start / granularity) % 4` 与指令描述符的位域拼接 | 已核验（2026-10-01）；`code/06` S24 |
| S25 | `tilelang/language/builtin.py:1440-1471` | `tcgen05_cp_warpx4`：每个 128 word 的块映射到 4 个 TMEM 列 | 已核验（2026-10-01）；`code/06` S25 |
| S26 | `src/op/gemm_blockscaled.cc:94-99` | 没有块缩放实现的后端 `LOG(FATAL)`，明确不会退化成丢掉 scale 的普通 GEMM | 已核验（2026-10-01）；`code/06` S26 |
| S27 | `src/op/tcgen5_meta.h:283-297` | MX 指令描述符的 dtype 编码：E4M3=0、E5M2=1、E2M1=5 | 已核验（2026-10-01）；`code/06` S27 |
| S28 | `src/op/tcgen5_meta.h:42-45` | TCGEN5MMA 的 fp16/bf16 路径：累加器必须是 fp32，K 是 16 的倍数 | 已核验（2026-10-01）；`code/06` S28 |
| S29 | `src/op/tcgen5_meta.h:79-82` | f8f6f4 家族：K 是 32 的倍数，累加器可为 fp32 **或 fp16** | 已核验（2026-10-01）；`code/06` S29 |
| S30 | `src/cuda/op/gemm.cc:49-50,62-63` | WGMMA 的 fp8 路径要求 `!transA && transB && K%32==0` | 已核验（2026-10-01）；`code/06` S30 |
| S31 | `tilelang/language/dtypes.py:29` | `AnyDType = ir.Type \| str \| type \| torch.dtype \| dtype`（dtype 的三种写法） | 已核验（2026-10-01）；未做成机器检查 |
| S32 | `tilelang/language/dtypes.py:148-152` | `__dtype_call__`：多参数打包成 `Shuffle`（`T.bfloat16x2(a, b)`） | 已核验（2026-10-01）；未做成机器检查 |
| S33 | `tilelang/language/dtypes.py:245-255` | `__dtype_new__`：未登记的名字抛 `TypeError` | 已核验（2026-10-01）；未做成机器检查 |
| S34 | `tilelang/language/dtypes.py:349-353` | `get_tvm_dtype`：dtype 的统一入口 | 已核验（2026-10-01）；未做成机器检查 |
| S35 | `tilelang/language/dtypes.py:449-560` | 低精度家族的向量形态声明（`float8_e4m3x2`…`float4_e2m1_unpacked` / `bfloat16` / `tfloat32`） | 已核验（2026-10-01）；未做成机器检查 |
| S36 | `tilelang/language/fp8.py:30-40` | `determine_torch_fp8_type`：同一套判据映射到 torch dtype | 已核验（2026-10-01）；未做成机器检查 |
| S37 | `tilelang/utils/tensor.py:205-293` | `torch_assert_close(rtol=1e-2, atol=1e-3, max_mismatched_ratio=1e-3)` 的完整实现（含 fp8 的 cast 处理） | 已核验（2026-10-01）；未做成机器检查 |
| S38 | `tilelang/profiler/__init__.py:104-115` | `Profiler.assert_allclose(atol=1e-2, rtol=1e-2, max_mismatched_ratio=1e-2)` | 已核验（2026-10-01）；未做成机器检查 |
| S39 | `examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py:63-72,108-146` | SMEM 里每行一个 `uint32`；TMEM 的 `SFA_tmem [block_M,4]` 与 `SFB_tmem [block_M,8]`；每 4 次迭代搬一次 scale | 已核验（2026-10-01）；未做成机器检查；**未运行** |
| S40 | `examples/blockscaled_gemm_sm100/mxfp8_illustrated.md` | SM100 MXFP8 的数据通路说明：每 128 个 K 元素一个 ue8m0、word 打包公式、TMEM 形状、SF ID 的推导、grouped 的 `SFB[sf_group*E*N + eid*N + n]` | 已核验（2026-10-01）：逐段读 |
| S41 | `tilelang/language/builtin.py:1473-1482` | `tcgen05_sf_warp_transpose`：按 128 word 分块做 warp 级转置 | 已核验（2026-10-01）；未做成机器检查 |
| S42 | `src/cuda/op/gemm_blockscaled.cc:46-90` | CUDA 侧的块缩放指令选择：SM100 的 TCGEN05（A/B 在 SMEM、C 在 TMEM）或 SM120 的 `mma.sync` | 已核验（2026-10-01）；未做成机器检查 |
| S43 | `examples/dequantize_gemm/quantize/nvfp4.py:180-196` | `blockscaled_chunk_kmajor_word_offset`：CUTLASS `BlockScaledBasicChunk` 的 K-major 词坐标 | 已核验（2026-10-01）；`code/06` N2；**未运行** |
| S44 | `examples/bitnet-1.58b/utils_quant.py:16-30` | 三值权重（`1/mean|w|`）与 per-token int8 激活（`Qp/absmax`）的量化规则 | 已核验（2026-10-01）；**未运行** |
| S45 | `examples/bitnet-1.58b/README.md` | 上游给出的 A100 性能与显存数字（见「关于性能数字的说明」） | 已核验（2026-10-01）：文本存在；**未复现** |

**负向断言**（`code/06` N1）：主干 `tilelang` 包下**没有** quantize 子包——
v0.1.15 的量化工具只出现在 `examples/dequantize_gemm/quantize/` 与
`testing/python/quantize/` 两个示例目录里，以及
`examples/dequantize_gemm/quantize/nvfp4.py` 这个文件。
核验方式：在 `_vendor/tilelang-v0.1.15/` 下执行
`ls tilelang/quantize` 会报 `No such file or directory`；
`code/06` 的 N1 就是这个断言，N2 是对照组（N2 引用的文件存在）。

---

## 锁定的源码引用（`_vendor/tilelang-ascend-v0.1.2.000/`）

| id | 引用位置 | 用途 | 核验方式与状态 |
| --- | --- | --- | --- |
| A1 | `tilelang/ascend/language/gemm_op.py:110-160` | 昇腾方言的 `T.gemm_blockscaled`：MX 粒度写死为 32 | 已核验（2026-10-01）；`code/06` A1 |
| A2 | `tilelang/language/ascend.py:26` | fp8 的昇腾 dtype 名 `e4m3_float8` → `float8_e4m3_t` | 已核验（2026-10-01）；`code/06` A2 |
| A3 | `examples/quant_batch_matmul/example_quant_batch_matmul.py:25-116` | per-tensor / per-channel 量化 GEMM：int8 → int32 L0C 累加 → UB 乘 scale → cast；对拍判据 `rtol=1e-2, atol=1e-2` | 已核验（2026-10-01）；`code/06` A3、A4；**未运行** |
| A4 | `examples/quant_batch_matmul/example_quant_matmul.py:25-116` | 同一实现的非 batch 版本（对照） | 已核验（2026-10-01）；**未运行** |
| A5 | `examples/gemm/example_gemm_fp8_pto.py:10-16,35-93` | 昇腾 FP8 GEMM 需要 A5（Ascend 950）的 Cube 核，A2/A3 直接 `[SKIP]` | 已核验（2026-10-01）；`code/06` A5；**未运行** |
| A6 | `tilelang/utils/tensor.py:26-40` | `e4m3_float8` / `e5m2_float8` 与 torch dtype 的双向映射 | 已核验（2026-10-01）；未做成机器检查 |

---

## corpus 资料引用（`../tilelang-corpus/`）

抓取日期：2026-09-30（Asia/Shanghai），见 corpus 的 `README.md`。

| id | 本地路径 | 上游链接 | 用途 | 核验状态 |
| --- | --- | --- | --- | --- |
| R1 | `06-huawei-ascend/raw/catlass/include/catlass/catlass.hpp:48` | https://gitcode.com/cann/catlass | `MX_SCALE_GROUP_NUM = 32`——MX 粒度在昇腾侧的唯一定义 | 已核验（2026-10-01）；`code/06` R1 |
| R2 | `06-huawei-ascend/raw/catlass/include/catlass/gemm/block/block_mmad_mx_tla.hpp:101-145` | 同上 | `L1_SCALE_FACTOR_K` 与 `L1SCALEA/L1SCALEB_TILE_SIZE` 的公式：scale 在 L1 的深度是模板参数 | 已核验（2026-10-01）；`code/06` R2 |
| R3 | `06-huawei-ascend/raw/catlass/examples/59_ascend950_a8w4_mx_matmul/README.md` 与 `a8w4_mx.cpp:57` | 同上 | A8W4 MX 示例：`float8_e4m3_t` × `float4_e2m1x2_t`、scale 为 `float8_e8m0_t`、公式 `C = (MxScaleA x A) * (MxScaleB x B) + Bias` | 已核验（2026-10-01）；`code/06` R3；**未编译、未运行** |
| R4 | `06-huawei-ascend/raw/catlass/experimental/attention/ascend950_fp8_mx_flash_attention_infer/mxfp8_flash_attention_infer.md` | 同上 | MXFP8 FlashAttention：block size=32、`float8_e8m0_t` scale、在 L1→L0 阶段反量化、P 需要静态量化系数、Online Softmax 在 FP32 完成 | 已核验（2026-10-01）；`code/06` R4 |
| R5 | `05-nvidia-backend/raw/nvidia-ptx-isa.md:32061-32134` | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html | `tcgen05.mma` 的块缩放：`(A*scale_A)*(B*scale_B)+D`、scale 在 Tensor Memory、SFB 复制到 32 个 lane 分区 | 已核验（2026-10-01）；`code/06` R5 |
| R6 | 同上 `:32136-32300` | 同上 | Scale Factor A/B ID 的布局与子列选择（1/2/4 字节对齐） | 已核验（2026-10-01）；`code/06` R6 |
| R7 | 同上 `:1196-1252` | 同上 | 格式定义：`bf16`（8 指数 7 尾数）、`e4m3`（无 inf、NaN 只有 `0x7f`/`0xff`）、`e5m2`、`e2m1`、`ue8m0`（8 位指数 0 位尾数） | 已核验（2026-10-01）；`code/06` R7 |
| R8 | 同上 `:23561-23645` | 同上 | `mma.sync` 的块缩放：`.kind::mxf8f6f4` 配 `.ue8m0` 与 `.scale_vec::1X`，scale_A/scale_B 的形状表 | 已核验（2026-10-01） |
| R9 | 同上 `:32620-32660` | 同上 | `tcgen05.mma` 的语法，含 `.kind.block_scale{.scale_vectorsize}` 与 `[scale-A-tmem]`/`[scale-B-tmem]` 操作数 | 已核验（2026-10-01） |
| R10 | `06-huawei-ascend/raw/catlass/examples/54_ascend950_fp4_mx_matmul/README.md` | https://gitcode.com/cann/catlass | MXFP4 示例：L1 分块 256×256×448、L0 256×256×128 与 L1 容量约束的提醒 | 已核验（2026-10-01）；**未编译、未运行** |
| R11 | `05-nvidia-backend/raw/nvidia-blackwell-tuning-guide.md` | https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html | Blackwell 调优指南。**注：本 corpus 快照的正文只到第 1 节（架构概述），不含 tcgen05 / 块缩放的细节** | 已核验（2026-10-01）：全文 14.7 KB，`grep -i "tcgen05\|block.scal"` 无命中。因此本章的 Blackwell 细节一律引用 R5–R7 的 PTX ISA |
| R12 | `03-tilelang-papers/raw/arxiv-2407-08608-fulltext.md` | https://arxiv.org/abs/2407.08608 | FlashAttention-3：低精度 attention 的设计动机与 FP16/FP8 的数值误差对比 | 已核验（2026-10-01）：本地为全文 |
| R13 | `03-tilelang-papers/raw/arxiv-2407-00088.md` | https://arxiv.org/abs/2407.00088 | T-MAC / BitBLAS 系：TileLang 的低精度血统（低比特算子生成，而非通用编译器研究） | 已核验（2026-10-01） |

---

## 本章可复算的产出（`code/`）

| id | 位置 | 内容 | 验证级别 |
| --- | --- | --- | --- |
| C1 | `code/01-number-formats/number_formats.py` | 位模式枚举：各格式的 `u`、三个边界、E8M0 值域、E2M1 码本 | CPU 逻辑验证（实跑） |
| C2 | `code/02-quant-dequant/quant_dequant.py` | 五种量化方案的往返误差（相对 L2 / max err / SQNR） | CPU 逻辑验证（实跑） |
| C3 | `code/03-error-model/error_model.py` | `u/√K` 缩放律、参照选择、累加器精度三组实验 | CPU 逻辑验证（实跑） |
| C4 | `code/04-vs-reference/reference_matching.py` | 六种参照的对比（含两种会导致误判的写法） | CPU 逻辑验证（实跑） |
| C5 | `code/05-mx-scale-layout/mx_scale_layout.py` | scale 打包/解包的复刻与四种布局错误 | CPU 逻辑验证（实跑） |
| C6 | `code/06-tilelang-audit/tilelang_audit.py` | 44 条源码事实的静态核对 | CPU 逻辑验证（实跑，44/44 通过） |

---

## 关于性能数字的说明

本章只出现一处「性能数字」，并且明确标注了来源：

| 数字 | 来源 | 条件 | 状态 |
| --- | --- | --- | --- |
| BitNet b1.58-3B（Llama-3B 结构）：fp16 基线 177.67 vs int8×int2 实现 64.18 | `examples/bitnet-1.58b/README.md` 的 “BitBLAS Results / Performance” 表 | 设备 A100、batchsize=1、in_seq=1 | **上游样例给出的数字，本机未复现**（没有 GPU） |

除此之外，本章的全部数字分两类：

* `code/01`–`code/05` 的 CPU 实跑输出（可复算，命令写在各自的 README 里）；
* 从位模式或 IEEE-754 布局推出的解析值（`code/01` 的表格）。

**没有任何一条来自加速器实测。**

---

## 本章资料缺口

以下内容在写作时未能在 corpus 或锁定版本里找到足够的第一手材料。
它们不影响本章任何结论的成立，但会影响后续章节的取材：

1. **TileLang 官方文档里没有低精度/量化的专门页面。**
   `01-tilelang-official-docs/raw/site/_sources/programming_guides/type_system.md`
   只有一页 dtype 清单（列出 float8/6/4 家族），
   `01-tilelang-official-docs/raw/site/_sources/programming_guides/instructions.md`
   里只有 `T.gemm_blockscaled` 的一句签名说明。
   本章因此完全依赖源码与示例，而不是官方教程。
2. **R11（Blackwell Tuning Guide）的本地快照不完整**，只到架构概述一节；
   “block scaling” 的实际内容在 PTX ISA（R5–R7）里。
   如果后续要写硬件章节，建议补抓完整的 Blackwell tuning guide 与
   CUTLASS 的 block-scaled 文档。
3. **昇腾侧的 MXFP8/MXFP4 缺少端到端的数值误差公开数据。**
   CATLASS 的示例只给「Compare success」这样的通过/失败判据，
   没有给出误差量级；本章的昇腾数值结论因此只引用模板参数与数据流，
   不给误差数字。
4. **没有 MXINT8 / MXINT4 的硬件资料。** OCP MX 规范里包含整数变体，
   但锁定版本与 corpus 里都没有对应实现，本章因此只讲 fp8/fp4 的 MX 路径。
