#!/usr/bin/env python3
"""Work out the cuTensorMapEncodeTiled arguments for a TMA tile copy.

The real descriptor is encoded on the host by
_vendor/tilelang-v0.1.15/tilelang/jit/adapter/wrapper.py:111-131 from the
`tma_descriptor_args` attribute that LowerHopperIntrin attaches to the host
function. The constraints checked here mirror CheckBulkLoad / CheckBulkStore /
CheckGlobalStrides in _vendor/tilelang-v0.1.15/src/cuda/op/copy_analysis.cc.

It is NOT the compiler. See README.md for the simplifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# From _vendor/tilelang-v0.1.15/src/cuda/op/tma_layout.h:40.
TMA_MAX_BOX_DIM = 256
TMA_SMEM_ADDR_ALIGN = 128

# Symbolic names for CUtensorMapDataType (numeric values live in cuda.h).
CU_TENSOR_MAP_DATA_TYPE = {
    "u8": "UINT8",
    "u16": "UINT16",
    "u32": "UINT32",
    "s32": "INT32",
    "u64": "UINT64",
    "s64": "INT64",
    "f16": "FLOAT16",
    "f32": "FLOAT32",
    "f64": "FLOAT64",
    "bf16": "BFLOAT16",
    "f32_f16": "FLOAT32_FLOAT32",
    "tf32": "TFLOAT32",
    "f8": "FLOAT8_E4M3",
    "f8_e5m2": "FLOAT8_E5M2",
    "f4_align16b": "16U4_ALIGN16B",  # packed FP4 global -> unpacked FP4 smem
}

F16_BITS = 16


class Rejected(Exception):
    pass


@dataclass(frozen=True)
class TensorSpec:
    """A global tensor plus the tile box we want to move with TMA."""

    name: str
    dtype: str
    elem_bits: int
    shape: tuple[int, ...]
    box: tuple[int, ...]
    strides: tuple[int, ...] = ()  # element strides, innermost last; () = row-major
    element_strides: tuple[int, ...] = ()
    interleave: str = "NONE"
    swizzle: str = "NONE"
    l2_promotion: str = "L2_128B"
    oob_fill: str = "NONE"
    extra: dict = field(default_factory=dict)

    def element_strides_or_default(self) -> tuple[int, ...]:
        return self.element_strides or tuple([1] * len(self.box))

    def strides_or_default(self) -> tuple[int, ...]:
        if self.strides:
            return self.strides
        strides = [1] * len(self.shape)
        acc = 1
        for i in range(len(self.shape) - 1, -1, -1):
            strides[i] = acc
            acc *= self.shape[i]
        return tuple(strides)


def check_global_strides(spec: TensorSpec) -> None:
    """Mirror CheckGlobalStrides (copy_analysis.cc:143-193)."""
    strides = spec.strides_or_default()
    if strides[-1] != 1:
        raise Rejected(f"global 最内层 stride 是 {strides[-1]}，不是 1")
    for i, stride in enumerate(strides[:-1]):
        stride_bytes = stride * spec.elem_bits // 8
        if stride_bytes % 16 != 0:
            raise Rejected(f"globalStride[{i}] 字节数 {stride_bytes} 不是 16 的倍数")
        if stride_bytes >= (1 << 40):
            raise Rejected(f"globalStride[{i}] 字节数 {stride_bytes} 超过 2^40")


def check_last_dim(spec: TensorSpec) -> None:
    """Mirror the check_last_dim branch of CheckBulkLoad/Store."""
    boxes_bits = spec.box * spec.elem_bits if isinstance(spec.box, int) else None
    if boxes_bits is None:
        last_bits = spec.box[-1] * spec.elem_bits
    else:
        last_bits = boxes_bits
    if last_bits % 128 != 0:
        raise Rejected(f"最后一维 {spec.box[-1]} * {spec.elem_bits} bits = "
                       f"{last_bits} bits，不是 128 的倍数")


def check_box_dim(spec: TensorSpec) -> None:
    for dim in spec.box:
        if dim > TMA_MAX_BOX_DIM:
            raise Rejected(f"boxDim {dim} 超过上限 {TMA_MAX_BOX_DIM}")


def build_descriptor(spec: TensorSpec) -> dict:
    check_last_dim(spec)
    check_box_dim(spec)
    check_global_strides(spec)
    if spec.dtype not in CU_TENSOR_MAP_DATA_TYPE:
        raise Rejected(f"未建模的 dtype {spec.dtype}")
    if spec.swizzle != "NONE" and (spec.elem_bits // 8 * spec.box[-1]) % TMA_SMEM_ADDR_ALIGN != 0:
        raise Rejected("swizzle 要求 shared 侧每行至少 128 字节对齐")
    rank = len(spec.shape)
    strides = spec.strides_or_default()
    # The wrapper passes `globalStride + 1`, i.e. it drops the innermost stride
    # (which must be 1 anyway) and reports the byte offsets as 64-bit values.
    global_stride_bytes = [strides[i] * spec.elem_bits // 8 for i in range(rank - 1)]
    return {
        "handle": sanitize(spec.name) + "_tmap",
        "dtype": CU_TENSOR_MAP_DATA_TYPE[spec.dtype],
        "tensor_rank": rank,
        "global_dim": list(spec.shape),
        "global_stride": global_stride_bytes,
        "box_dim": list(spec.box),
        "element_strides": list(spec.element_strides_or_default()),
        "interleave": spec.interleave,
        "swizzle": spec.swizzle,
        "l2_promotion": spec.l2_promotion,
        "oob_fill": spec.oob_fill,
    }


def sanitize(name: str) -> str:
    """Turn a scenario label into a valid C identifier fragment."""
    out = []
    for ch in name:
        out.append(ch if ch.isalnum() and ch.isascii() else "_")
    return "".join(out).strip("_")


def render_c(desc: dict) -> str:
    """Render the same shape of code that wrapper.py:111-131 emits."""
    gdim = ", ".join(str(d) for d in desc["global_dim"])
    gstride = ", ".join(str(s) for s in desc["global_stride"])
    box = ", ".join(str(d) for d in desc["box_dim"])
    estride = ", ".join(str(s) for s in desc["element_strides"])
    return (
        f"  CUtensorMap {desc['handle']};\n"
        f"  ... cuTensorMapEncodeTiled(&{desc['handle']}, "
        f"/*dtype=*/{desc['dtype']}, /*rank=*/{desc['tensor_rank']},\n"
        f"      /*gaddr=*/A, /*gdim=*/{{{gdim}}}, /*gstride=*/{{{gstride}}},\n"
        f"      /*box=*/{{{box}}}, /*estride=*/{{{estride}}},\n"
        f"      /*interleave=*/{desc['interleave']}, /*swizzle=*/{desc['swizzle']},\n"
        f"      /*l2=*/{desc['l2_promotion']}, /*oob=*/{desc['oob_fill']});"
    )


def report(spec: TensorSpec) -> str:
    try:
        desc = build_descriptor(spec)
    except Rejected as exc:
        return f"[rejected] {spec.name}: {exc}"
    lines = [
        f"[ok]   {spec.name}  dtype={spec.dtype}  gshape={tuple(spec.shape)}  box={tuple(spec.box)}",
        f"       globalDim      = {desc['global_dim']}",
        f"       globalStride   = {desc['global_stride']}   # 单位是字节，最内层被丢掉",
        f"       boxDim         = {desc['box_dim']}",
        f"       elementStrides = {desc['element_strides']}",
        render_c(desc),
    ]
    strides = spec.strides_or_default()
    for i, stride in enumerate(strides[:-1]):
        lines.append(f"       globalStride[{i}] bytes = {stride * spec.elem_bits // 8} "
                     f"（16 的倍数: {(stride * spec.elem_bits // 8) % 16 == 0}）")
    return "\n".join(lines)


def main() -> int:
    specs = [
        TensorSpec("2D K-major f16 tile", "f16", 16, (4096, 4096), (128, 64)),
        TensorSpec("2D N-major f16 tile", "f16", 16, (4096, 4096), (64, 128)),
        TensorSpec("1D contiguous f16 row", "f16", 16, (8192,), (256,)),
        TensorSpec("2D bf16 tile, box 256x64", "bf16", 16, (8192, 8192), (256, 64)),
        # rejections
        TensorSpec("box last dim 30", "f16", 16, (4096, 4096), (128, 30)),
        TensorSpec("stride-2 innermost", "f16", 16, (4096, 4096), (128, 64),
                   strides=(8192, 2)),
        TensorSpec("stride 12 elements (24 B)", "f16", 16, (4096, 4096), (128, 64),
                   strides=(12, 1)),
        TensorSpec("box dim 512 > 256", "f16", 16, (8192, 8192), (512, 64)),
        TensorSpec("fp8 tile, last dim 32", "f8", 8, (4096, 4096), (128, 32)),
    ]
    print("=== TMA 描述符参数与约束（模型） ===")
    for spec in specs:
        print(report(spec))
        print()

    ok = True
    def expect_reject(spec, needle):
        nonlocal ok
        text = report(spec)
        if not text.startswith("[rejected]") or needle not in text:
            print(f"SELF-CHECK FAILED: {spec.name} should be rejected with {needle!r}")
            ok = False

    expect_reject(specs[4], "不是 128 的倍数")
    expect_reject(specs[5], "不是 1")
    expect_reject(specs[6], "不是 16 的倍数")
    expect_reject(specs[7], "超过上限")
    if not report(specs[0]).startswith("[ok]"):
        print("SELF-CHECK FAILED: the plain 2D f16 tile should be accepted")
        ok = False
    print("self-check:", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
