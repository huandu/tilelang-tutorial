#!/usr/bin/env python3
"""第 27 章 · 示例 6：把本章的源码结论变成可复现的检查（纯标准库）。

本章引用了大量具体的文件与语义（哪些 dtype 存在、块缩放的粒度是多少、scale 怎么
打包、`T.gemm_blockscaled` 的参数叫什么）。这些结论如果只写在正文里，读者无法
验证；这个脚本把它们变成一组「在锁定版本的工作树上跑一遍」的断言。

每条检查包含：
    id       正文里的引用编号（与 references.md 一致）
    what     这条结论在讲什么
    相对路径  相对 _vendor 或 corpus 根目录
    必须出现  若干子串（全部命中才算通过）
    必须不出现 可选，命中即失败（用于负向断言）

**这是启发式文本检查，不是编译器。** 命中只说明「这句话在锁定版本里确实存在」，
不说明它的语义被正确理解了。

用法（在 code/06-tilelang-audit/ 目录下）：

    python3 tilelang_audit.py
    python3 tilelang_audit.py --vendor ../../../../_vendor/tilelang-v0.1.15 \
        --ascend ../../../../_vendor/tilelang-ascend-v0.1.2.000 \
        --corpus ../../../../../tilelang-corpus

找不到任一根目录时，相关检查打印 `skipped` 并以 0 退出。

作者实际做到的验证级别：**CPU 逻辑验证**（在写作机的工作树上实际运行过，
全部检查通过；输出见同目录 README.md）。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Check:
    cid: str
    what: str
    root: str  # "vendor" | "ascend" | "corpus"
    path: str
    must: list[str] = field(default_factory=list)
    must_not: list[str] = field(default_factory=list)
    need_missing: bool = False  # True 表示「这个路径必须不存在」


CHECKS: list[Check] = [
    # ---- dtype 系统 ------------------------------------------------------- #
    Check(
        "S1",
        "tilelang/dtypes.py 只是 language/dtypes.py 的再导出",
        "vendor",
        "tilelang/dtypes.py",
        ["from tilelang.language.dtypes import *", "dtype, AnyDType, get_tvm_dtype"],
    ),
    Check(
        "S2",
        "语言面的 dtype 只认字符串名，且登记了 e4m3/e5m2/e8m0/int4",
        "vendor",
        "tilelang/language/dtypes.py",
        ['"int4": "Int4"', '"float8_e4m3fn": "Float8E4M3FN"', '"float8_e8m0fnu": "Float8E8M0FNU"'],
    ),
    Check(
        "S3",
        "torch 的扩展 dtype 列表里有 e4m3fn/e5m2/e8m0fnu/float4_e2m1fnx2",
        "vendor",
        "tilelang/language/dtypes.py",
        ['("float8_e4m3fn",)', '("float8_e8m0fnu",)', '("float4_e2m1fnx2",)'],
    ),
    Check(
        "S4",
        "int4 落到 torch 时用 int8 存储（torch 没有 int4 dtype）",
        "vendor",
        "tilelang/language/dtypes.py",
        ['logger.debug("torch doesn\'t support int4, using int8 as storage dtype.")'],
    ),
    Check(
        "S5",
        "e8m0fnu 需要 torch >= 2.8（版本门槛写在断言里）",
        "vendor",
        "tilelang/language/dtypes.py",
        ['hasattr(torch, "float8_e8m0fnu")', "torch.float8_e8m0fnu is not supported in this version of torch. Please upgrade torch >= 2.8.0"],
    ),
    Check(
        "S6",
        "FP4 有 packed（4 bit）与 unpacked（8 bit 槽位）两种存储，用 type_code 区分",
        "vendor",
        "tilelang/language/dtypes.py",
        ["_FLOAT4_E2M1FN_TYPE_CODE = 17", "_FLOAT4_E2M1_UNPACKED_TYPE_CODE = 131"],
    ),
    Check(
        "S7",
        "混合 dtype 的 GEMM 只允许 f8f6f4 家族",
        "vendor",
        "tilelang/language/dtypes.py",
        ["def is_f8f6f4_family", "Mixed-dtype GEMM only supports f8f6f4 family operands currently"],
    ),
    # ---- fp8 ------------------------------------------------------------- #
    Check(
        "S8",
        "determine_fp8_type：CUDA 用 e4m3fn/e5m2，ROCm 默认 FNUZ、gfx950 例外",
        "vendor",
        "tilelang/language/fp8.py",
        ["def determine_fp8_type", 'T.float8_e4m3fn if fp8_format == "e4m3" else T.float8_e5m2', 'gcn_arch.startswith("gfx950")'],
    ),
    Check(
        "S9",
        "fp8 的负零有一个专门的清理函数",
        "vendor",
        "tilelang/utils/tensor.py",
        ["def fp8_remove_negative_zeros_", "Input tensor must be of float8 dtype"],
    ),
    # ---- 示例：fp8 / int4 ------------------------------------------------ #
    Check(
        "S10",
        "gemm_fp8 示例：fp32 累加、block_K=64、用相似度而非逐元素比较",
        "vendor",
        "examples/gemm_fp8/example_tilelang_gemm_fp8.py",
        ["accum_dtype=T.float32", "def calc_diff", "assert diff < 1e-3"],
    ),
    Check(
        "S11",
        "gemm_int4 示例：T.int4 张量 + T.int32 累加 + 精确相等断言",
        "vendor",
        "examples/gemm_int4/example_tilelang_gemm_int4.py",
        ["T.Tensor((M, K), T.int4)", "C: T.Tensor((M, N), T.int32)", "torch.testing.assert_close(C.cpu(), ref_c, rtol=0, atol=0)"],
    ),
    Check(
        "S12",
        "int4 打包：两个 int8 的低 4 位合成一个字节",
        "vendor",
        "examples/gemm_int4/example_tilelang_gemm_int4.py",
        ["packed = (tensor_i16[..., ::2] & 0x0F) | ((tensor_i16[..., 1::2] & 0x0F) << 4)"],
    ),
    Check(
        "S13",
        "dequantize GEMM：搬入时用位运算解码再送进 T.gemm",
        "vendor",
        "examples/dequantize_gemm/README.md",
        ["_tir_packed_to_unsigned_convert", "B_dequantize_local[i, j] = ", "T.gemm(B_dequantize_local, A_shared, Ct_local, transpose_B=True)"],
    ),
    Check(
        "S14",
        "w4a8 的 int4->int8 符号扩展技巧（先左移 4 位再算术右移 4 位）",
        "vendor",
        "examples/dequantize_gemm/example_dequant_gemm_w4a8.py",
        ["i8_shifted = tirx.reinterpret(T.int8, i4 << tirx.const(4, T.uint8))", "i8 = i8_shifted >> tirx.const(4, T.int8)"],
    ),
    Check(
        "S15",
        "fine-grained 版本按 128 位事务决定每线程处理多少元素",
        "vendor",
        "examples/dequantize_gemm/example_dequant_gemm_fine_grained.py",
        ["MAX_TRANSACTION_SIZE_IN_BITS = 128", "local_size = MAX_TRANSACTION_SIZE_IN_BITS // DataType(in_dtype).bits"],
    ),
    # ---- block-scaled：示例与 API --------------------------------------- #
    Check(
        "S16",
        "SM100 MXFP8 示例：blockscale (1,1,128)、uint32 打包、sf_load_period 公式",
        "vendor",
        "examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py",
        [
            "# Blockscale size: (M, N, K) = (1, 1, 128)",
            "sf_load_period = sf_granularity_k * 4 // block_K",
            "sf_k_groups = T.ceildiv(T.ceildiv(K, sf_granularity_k), 4)",
            "SFA: T.Tensor[[sf_k_groups * M], T.uint32]",
        ],
    ),
    Check(
        "S17",
        "打包公式：4 个字节拼成一个 uint32，再转置成 group-major 扁平数组",
        "vendor",
        "examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py",
        ["packed = (words[:, 0::4] | (words[:, 1::4] << 8) | (words[:, 2::4] << 16) | (words[:, 3::4] << 24))", "return packed.T.contiguous().reshape(-1)"],
    ),
    Check(
        "S18",
        "E8M0 由 fp32 的指数位直接得到，且向上取整到 2 的幂",
        "vendor",
        "examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py",
        ["def ceil_to_ue8m0(x):", "exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).ne(0).to(torch.int32)", "sf = ceil_to_ue8m0(x_amax / 448.0)"],
    ),
    Check(
        "S19",
        "scale 填充值是 127（即 2^0 = 1），不是 0",
        "vendor",
        "examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py",
        ["torch.full((mn, sf_k_padded), 127, device=x.device, dtype=torch.uint8)"],
    ),
    Check(
        "S20",
        "参考实现按 sf_granularity_k 分块，逐块乘 scale",
        "vendor",
        "examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py",
        ["def blockscaled_gemm_ref(", "a_block = a_f32[:, k_start:k_end] * sfa_scales[:, bi : bi + 1]"],
    ),
    Check(
        "S21",
        "2CTA 变体把 sf_granularity_k 固定为 128",
        "vendor",
        "examples/blockscaled_gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py",
        ["assert sf_granularity_k == 128"],
    ),
    Check(
        "S22",
        "T.gemm_blockscaled 的签名带 k_start / sf_a_granularity_k / sf_b_granularity_k",
        "vendor",
        "tilelang/language/gemm_op.py",
        ["def gemm_blockscaled(", "k_start: int | tirx.PrimExpr", "sf_a_granularity_k: int", "sf_b_granularity_k: int"],
    ),
    Check(
        "S23",
        "T.tcgen05_gemm_blockscaled：A/B 是 FP8/FP6/FP4，SFA/SFB 是 TMEM 里的 E8M0",
        "vendor",
        "tilelang/cuda/language/gemm_op.py",
        ["def tcgen05_gemm_blockscaled(", "SFA/SFB are E8M0 scale factors already", "resident in tensor memory"],
    ),
    Check(
        "S24",
        "SF ID 的推导公式：(k_start / granularity) % 4",
        "vendor",
        "tilelang/cuda/intrinsics/macro/tcgen05_macro_generator.py",
        ["runtime_sf_a_id = ((_sf_k_start + ki * micro_size_k) // int(sf_a_granularity_k)) % 4"],
    ),
    Check(
        "S25",
        "tcgen05_cp_warpx4：每个 128 word 的块映射到 4 个 TMEM 列",
        "vendor",
        "tilelang/language/builtin.py",
        ["def tcgen05_cp_warpx4(", "each 128-word chunk maps to", "4 TMEM columns"],
    ),
    Check(
        "S26",
        "没有块缩放实现的后端会直接编译失败，而不是退化成普通 GEMM",
        "vendor",
        "src/op/gemm_blockscaled.cc",
        ["a dense GEMM lowering would silently drop the SFA/SFB", "scale factors."],
    ),
    Check(
        "S27",
        "MX 指令描述符里 e4m3=0 / e5m2=1 / e2m1=5",
        "vendor",
        "src/op/tcgen5_meta.h",
        ["return 0u; // E4M3", "return 1u; // E5M2", "return 5u; // E2M1"],
    ),
    Check(
        "S28",
        "TCGEN5MMA 的 fp16/bf16 路径要求 fp32 累加器，且 K 是 16 的倍数",
        "vendor",
        "src/op/tcgen5_meta.h",
        ["if ((ab_dtype.is_bfloat16() || ab_dtype.is_float16()) &&", "(c_dtype.is_float() && c_dtype.bits() == 32)) {", "if (K % 16 != 0)"],
    ),
    Check(
        "S29",
        "TCGEN5MMA 的 f8f6f4 家族要求 K 是 32 的倍数，累加器可以是 fp32 **或** fp16",
        "vendor",
        "src/op/tcgen5_meta.h",
        [
            "} else if ((ab_dtype.is_float8() || ab_dtype.is_float6_e2m3fn() ||",
            "ab_dtype.is_float6_e3m2fn() || ab_dtype.is_float4()) &&",
            "(c_dtype.is_float16() && c_dtype.bits() == 16))) {",
            "if (K % 32 != 0)",
        ],
    ),
    Check(
        "S30",
        "WGMMA 的 fp8 路径要求 A 不转置、B 转置，且 K 是 32 的倍数",
        "vendor",
        "src/cuda/op/gemm.cc",
        ["if (op.a_->dtype.is_float8() && op.b_->dtype.is_float8())", "return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;"],
    ),
    # ---- 昇腾 ------------------------------------------------------------ #
    Check(
        "A1",
        "昇腾方言的 T.gemm_blockscaled 把 MX 粒度写死为 32",
        "vendor",
        "tilelang/ascend/language/gemm_op.py",
        ['annotations["sf_a_granularity_k"] = 32', 'annotations["sf_b_granularity_k"] = 32', "MX scales cover 32 K elements per factor"],
    ),
    Check(
        "A2",
        "昇腾侧 fp8 的 dtype 名是 e4m3_float8 / e5m2_float8",
        "ascend",
        "tilelang/language/ascend.py",
        ['"e4m3_float8": "float8_e4m3_t"'],
    ),
    Check(
        "A3",
        "昇腾 quant GEMM 示例：int8 输入、int32 累加、epilogue 里乘 scale",
        "ascend",
        "examples/quant_batch_matmul/example_quant_batch_matmul.py",
        ["T.alloc_L0C([block_M, block_N], accum_dtype)", "c_scale[bm_v, bn_v] *= scale_ub[bn_v]"],
    ),
    Check(
        "A4",
        "昇腾 quant GEMM 的对拍判据是 rtol=1e-2 / atol=1e-2",
        "ascend",
        "examples/quant_batch_matmul/example_quant_batch_matmul.py",
        ["torch.testing.assert_close(C.cpu(), ref_C.cpu(), rtol=1e-2, atol=1e-2)"],
    ),
    Check(
        "A5",
        "昇腾 FP8 GEMM 需要 A5（Ascend 950）的 Cube 核",
        "ascend",
        "examples/gemm/example_gemm_fp8_pto.py",
        ["FP8 GEMM (TMATMUL accepting float8_e4m3 / float8_e5m2) requires the A5"],
    ),
    # ---- corpus：CATLASS 与 PTX ------------------------------------------ #
    Check(
        "R1",
        "CATLASS 的 MX 粒度常量是 32",
        "corpus",
        "06-huawei-ascend/raw/catlass/include/catlass/catlass.hpp",
        ["constexpr uint32_t MX_SCALE_GROUP_NUM = 32;", "Data count for one MX-scale factor per group"],
    ),
    Check(
        "R2",
        "CATLASS MX 模板以 L1_SCALE_FACTOR_K 参数化 L1 上的 scale 深度",
        "corpus",
        "06-huawei-ascend/raw/catlass/include/catlass/gemm/block/block_mmad_mx_tla.hpp",
        ["static constexpr uint32_t L1_SCALE_FACTOR_K = DispatchPolicy::L1_SCALE_FACTOR_K;"],
    ),
    Check(
        "R3",
        "A8W4 MX 示例用的 scale 类型是 float8_e8m0_t，公式为 (MxScaleA*A)*(MxScaleB*B)",
        "corpus",
        "06-huawei-ascend/raw/catlass/examples/59_ascend950_a8w4_mx_matmul/README.md",
        ["float8_e8m0", "C = (MxScaleA x A) * (MxScaleB x B) + Bias"],
    ),
    Check(
        "R4",
        "CATLASS MXFP8 FlashAttention：block size=32、e8m0 scale、在 L1→L0 阶段反量化",
        "corpus",
        "06-huawei-ascend/raw/catlass/experimental/attention/ascend950_fp8_mx_flash_attention_infer/mxfp8_flash_attention_infer.md",
        ["block size=32", "float8_e8m0_t", "在 Cube 矩阵乘的 L1→L0 阶段完成反量化"],
    ),
    Check(
        "R5",
        "PTX：tcgen05.mma 的块缩放把 scale 放在 Tensor Memory，SFB 要复制到 32 个 lane 分区",
        "corpus",
        "05-nvidia-backend/raw/nvidia-ptx-isa.md",
        [
            "Block Scaling for tcgen05.mma",
            "where `scale_A` and `scale_B` are matrices residing in Tensor Memory.",
            "Scale factor for `B` matrix needs to be duplicated to all 32 lane partitions of tensor memory.",
        ],
    ),
    Check(
        "R6",
        "PTX：mxf8f6f4 的 scale 类型是 ue8m0，SFA_ID 选的是 Tensor Memory 里的字节子列",
        "corpus",
        "05-nvidia-backend/raw/nvidia-ptx-isa.md",
        [
            ".kind::mxf8f6f4 | .e4m3 , .e5m2",
            ".kind::mxf8f6f4 | E4M3, E5M2, E2M3",
            "SFA_ID specifies the byte offset in the",
            "Tensor Memory word",
        ],
    ),
    Check(
        "R7",
        "PTX：e4m3 没有 inf、NaN 只有 0x7f/0xff；ue8m0 是「8 位指数 + 0 位尾数」",
        "corpus",
        "05-nvidia-backend/raw/nvidia-ptx-isa.md",
        ["The `e4m3` encoding does not support infinity and `NaN` values are limited to", "This data format is an 8-bit unsigned floating-point format with 8 bits for exponent and 0 bits for", "`ue8m0` encoding does not support infinity"],
    ),
    # ---- 版本边界（负向断言） ------------------------------------------- #
    Check(
        "N1",
        "v0.1.15 主干里没有 tilelang/quantize/ 包（只在示例目录下）",
        "vendor",
        "tilelang/quantize",
        need_missing=True,
    ),
    Check(
        "N2",
        "v0.1.15 主干的 examples/ 下确实有 dequantize_gemm/quantize/（负向断言的对照）",
        "vendor",
        "examples/dequantize_gemm/quantize/nvfp4.py",
        ["def blockscaled_chunk_kmajor_word_offset"],
    ),
]


def find_root(start: Path, name: str) -> Path | None:
    """从 start 向上找 `<dir>/name` 或 `<dir>/_vendor/name`。"""

    current = start.resolve()
    for _ in range(12):
        for candidate in (current / name, current / "_vendor" / name):
            if candidate.exists():
                return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vendor")
    parser.add_argument("--ascend")
    parser.add_argument("--corpus")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    roots = {
        "vendor": Path(args.vendor) if args.vendor else find_root(here, "tilelang-v0.1.15"),
        "ascend": Path(args.ascend) if args.ascend else find_root(here, "tilelang-ascend-v0.1.2.000"),
        "corpus": Path(args.corpus) if args.corpus else find_root(here, "tilelang-corpus"),
    }
    for name, path in roots.items():
        print(f"root[{name}] = {path if path is not None else 'NOT FOUND'}")
    print()

    passed = failed = skipped = 0
    for check in CHECKS:
        root = roots[check.root]
        if root is None:
            print(f"skipped  {check.cid:<4}{check.what}（找不到 {check.root} 根目录）")
            skipped += 1
            continue
        target = root / check.path
        if check.need_missing:
            if target.exists():
                print(f"FAIL     {check.cid:<4}{check.what}（{target} 存在，但期望不存在）")
                failed += 1
            else:
                print(f"ok       {check.cid:<4}{check.what}")
                passed += 1
            continue
        if not target.is_file():
            print(f"FAIL     {check.cid:<4}{check.what}（文件不存在：{target}）")
            failed += 1
            continue
        text = target.read_text(encoding="utf-8", errors="ignore")
        missing = [needle for needle in check.must if needle not in text]
        forbidden = [needle for needle in check.must_not if needle in text]
        if missing or forbidden:
            detail = []
            if missing:
                detail.append("未命中：" + " | ".join(repr(m) for m in missing))
            if forbidden:
                detail.append("不应出现：" + " | ".join(repr(m) for m in forbidden))
            print(f"FAIL     {check.cid:<4}{check.what}（{'; '.join(detail)}）")
            failed += 1
        else:
            print(f"ok       {check.cid:<4}{check.what}")
            passed += 1

    print()
    print(f"通过 {passed} / 失败 {failed} / 跳过 {skipped}（共 {len(CHECKS)} 条）")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
