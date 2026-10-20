"""Chapter 08 / §4: how dtypes are written, with fp8 / fp6 / fp4 in focus.

Verification level: **static inspection only** (no GPU / NPU, no TileLang
installed). ``python3 -m py_compile fp8_fp4_exprs.py`` passes; every dtype
symbol and helper below was checked against ``_vendor/tilelang-v0.1.15/``:

  * dtype objects: ``tilelang/language/dtypes.py:633-711``
    (``float8_e4m3fn`` at :647, ``float8_e5m2`` at :661,
     ``float8_e8m0fnu`` at :675, ``float6_e2m3fn`` at :682,
     ``float4_e2m1fn`` at :696, ``float4_e2m1_unpacked`` at :704)
  * normalization of strings / torch dtypes: ``dtypes.py:105, 245-253``
  * fp8 selection helper: ``tilelang/language/fp8.py``
  * fp4 helpers: ``dtypes.py:305-317, 324-347``
  * GEMM A/B dtype validation: ``dtypes.py:319-340``
"""

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# 1. The three spellings of a dtype are interchangeable.
# ---------------------------------------------------------------------------
def three_spellings():
    return [
        "float16",          # string
        T.float16,          # TileLang dtype object
        __import__("torch").float16,  # framework dtype
    ]


# ---------------------------------------------------------------------------
# 2. fp8: pick the family by its exponent/mantissa split AND its NaN rule.
#    A GEMM that takes fp16 in / fp32 accumulate is the usual mixed-precision
#    shape; the accumulator dtype is passed explicitly, never inferred.
# ---------------------------------------------------------------------------
@tilelang.jit
def fp8_gemm(M, N, K, BM=128, BN=128, BK=128):
    in_dtype = T.float8_e4m3fn     # CUDA-side E4M3
    accum_dtype = T.float32        # explicit: fp8 * fp8 accumulate in fp32

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
            A_s = T.alloc_shared((BM, BK), in_dtype)
            B_s = T.alloc_shared((BK, BN), in_dtype)
            C_f = T.alloc_fragment((BM, BN), accum_dtype)
            T.clear(C_f)
            for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
                T.copy(A[by * BM, ko * BK], A_s)
                T.copy(B[ko * BK, bx * BN], B_s)
                T.gemm(A_s, B_s, C_f)
            T.copy(C_f, C[by * BM, bx * BN])

    return kernel


# ---------------------------------------------------------------------------
# 3. fp4: two storage models for the *same* logical 4-bit value.
#    - float4_e2m1fn      : packed, 2 values per byte; half the footprint,
#                           global tensors + packed SMEM round trips.
#    - float4_e2m1_unpacked: one FP4 value per byte slot, SMEM/TMA only; used
#                           by tcgen05 kind::f8f6f4 / mxf8f6f4 mixed paths.
# ---------------------------------------------------------------------------
def fp4_storage_model():
    packed = T.float4_e2m1fn
    unpacked = T.float4_e2m1_unpacked
    return {
        "packed_bits": packed.bits,          # 4
        "unpacked_bits": unpacked.bits,      # 8
        "packed_is_float4": packed.is_float4(),      # True
        "unpacked_is_float4": unpacked.is_float4(),  # True
    }


# ---------------------------------------------------------------------------
# 4. A block-scaled GEMM handle: the scale factor lives beside the data tile,
#    keyed to the tile address. On Ascend the SF handle is created by
#    alloc_l0a_sf / alloc_l0b_sf; on CUDA it rides along with gemm_blockscaled.
# ---------------------------------------------------------------------------
def blockscaled_signature():
    # print the accepted keyword surface without needing to compile
    import inspect

    sig = inspect.signature(T.gemm_blockscaled)
    return str(sig)


# ---------------------------------------------------------------------------
# 5. Selecting the fp8 flavour for the running platform.
# ---------------------------------------------------------------------------
def per_platform_fp8():
    from tilelang.language.fp8 import determine_fp8_type

    # CUDA -> float8_e4m3fn / float8_e5m2; ROCm -> the FNUZ variants,
    # except gfx950 which prefers the OCP spelling. See tilelang/language/fp8.py.
    return determine_fp8_type


# ---------------------------------------------------------------------------
# 6. Mixed A/B dtypes are only allowed inside the f8f6f4 family.
# ---------------------------------------------------------------------------
def mixed_dtype_guard():
    from tilelang.language.dtypes import validate_gemm_ab_dtypes

    # ok: both operands are f8f6f4 family and the caller opted in
    validate_gemm_ab_dtypes(T.float8_e4m3fn, T.float4_e2m1fn, allow_f8f6f4_mixed=True)
    # raises: mixed dtypes outside the f8f6f4 family
    try:
        validate_gemm_ab_dtypes(T.float16, T.bfloat16)
    except ValueError as exc:
        return str(exc)
    return "unexpectedly accepted"
