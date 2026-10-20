#!/usr/bin/env python3
"""A teaching simulation of the packed host ABI that MakePackedAPI generates.

This file contains no TileLang code. It re-implements, in pure Python, the
checks that `src/transform/make_packed_api.cc` and
`src/transform/tvm_ffi_binder.cc` bake into the generated host function, then
demonstrates which error message each mismatch produces.

Why simulate instead of run? The real checks only exist inside a compiled host
stub, which needs a GPU/NPU and a TileLang build. The *rules* are readable
straight from the source, and the messages below are copied verbatim from it:

  * num_args       make_packed_api.cc:473-477
  * args NULL      make_packed_api.cc:480-482
  * type index     make_packed_api.cc:615-719
  * pointer/tensor make_packed_api.cc:650-658
  * float scalar   make_packed_api.cc:697-702
  * non-NULL ptr   tvm_ffi_binder.cc:390-395
  * ndim           tvm_ffi_binder.cc:461-467
  * dtype triple   tvm_ffi_binder.cc:470-487
  * fp8 families   tvm_ffi_binder.cc:489-511
  * tfloat32       tvm_ffi_binder.cc:512-520
  * bool widening  tvm_ffi_binder.cc:521-544
  * sub-byte bits  tvm_ffi_binder.cc:545-579

Usage
-----
    python3 packed_abi_sim.py            # run every scenario, print the verdict
    python3 packed_abi_sim.py --trace    # also print accepted cases
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

# --- TVM DataType codes, as used in the DLTensor type triplet --------------
CODE_INT = 0
CODE_UINT = 1
CODE_FLOAT = 2
CODE_BFLOAT = 4
CODE_FLOAT8E4M3 = 7
CODE_FLOAT8E4M3FN = 8
CODE_FLOAT8E4M3FNUZ = 9
CODE_FLOAT8E5M2 = 10
CODE_FLOAT8E5M2FNUZ = 11
CODE_BOOL = 6

# TypeIndex values the runtime puts on each packed argument
TI_NONE = "None"
TI_OPAQUE_PTR = "OpaquePtr"
TI_DLTENSOR_PTR = "DLTensorPtr"
TI_TENSOR = "Tensor"
TI_BOOL = "Bool"
TI_INT = "Int"
TI_FLOAT = "Float"


@dataclass(frozen=True)
class DType:
    code: int
    bits: int
    lanes: int = 1

    @property
    def total_bits(self) -> int:
        return self.bits * self.lanes

    def __str__(self) -> str:  # matches how the messages print a dtype
        return f"{self.code}#{self.bits}x{self.lanes}"


F16 = DType(CODE_FLOAT, 16)
F32 = DType(CODE_FLOAT, 32)
BF16 = DType(CODE_BFLOAT, 16)
E4M3 = DType(CODE_FLOAT8E4M3, 8)
E4M3FN = DType(CODE_FLOAT8E4M3FN, 8)
E5M2 = DType(CODE_FLOAT8E5M2, 8)
I8 = DType(CODE_INT, 8)
BOOL = DType(CODE_BOOL, 1)
FP4 = DType(CODE_FLOAT, 4, lanes=1)


@dataclass
class DLTensor:
    """The fields the binder actually reads out of a DLTensor."""

    dtype: DType
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None = None
    device_type: int = 2
    device_id: int = 0
    data: object = object()

    def to_memoryview(self) -> memoryview:  # placeholder for a real pointer
        return memoryview(b"\x00")


@dataclass
class TensorParam:
    """A PrimFunc buffer parameter: what the kernel declared."""

    name: str
    dtype: DType
    shape: tuple[int, ...]
    contiguity_checked: bool = False
    is_used: bool = True


@dataclass
class ScalarParam:
    name: str
    kind: str  # "int" | "float" | "bool"


class ABIError(ValueError):
    """Raised with the same message the generated host stub would report."""


@dataclass
class Call:
    """One packed call: the scalar values and tensor values, in signature order."""

    args: list[object] = field(default_factory=list)


def _type_accepted(param: ScalarParam, type_index: str) -> bool:
    """Mirror of make_packed_api.cc:652-711."""
    if param.kind == "bool":
        return type_index in (TI_BOOL, TI_INT)
    if param.kind == "int":
        return type_index in (TI_INT, TI_BOOL)
    if param.kind == "float":
        return type_index in (TI_FLOAT, TI_INT, TI_BOOL)
    return False


def _dtype_accepted(expected: DType, provided: DType) -> bool:
    """Mirror of tvm_ffi_binder.cc:486-579, including every relaxation."""
    if (provided.code, provided.bits, provided.lanes) == (expected.code, expected.bits, expected.lanes):
        return True
    # float8 e4m3 family
    if expected.code in (CODE_FLOAT8E4M3, CODE_FLOAT8E4M3FN, CODE_FLOAT8E4M3FNUZ):
        if provided.code in (CODE_FLOAT8E4M3, CODE_FLOAT8E4M3FN, CODE_FLOAT8E4M3FNUZ):
            return (provided.bits, provided.lanes) == (expected.bits, expected.lanes)
    if expected.code in (CODE_FLOAT8E5M2, CODE_FLOAT8E5M2FNUZ):
        if provided.code in (CODE_FLOAT8E5M2, CODE_FLOAT8E5M2FNUZ):
            return (provided.bits, provided.lanes) == (expected.bits, expected.lanes)
    # tfloat32 is stored as float32
    if expected.code == CODE_FLOAT and expected.bits == 32 and provided.code == CODE_FLOAT:
        return (provided.bits, provided.lanes) == (32, expected.lanes)
    # bool accepts int8/uint8 and kDLBool with bits 1 or 8
    if expected.code == CODE_BOOL:
        if provided.lanes != expected.lanes:
            return False
        if provided.code in (CODE_INT, CODE_UINT) and provided.bits == 8:
            return True
        if provided.code == CODE_BOOL and provided.bits in (1, 8):
            return True
        if provided.bits == 1:
            return True
    return False


def check_call(kernel_name: str, params: list[object], call: Call) -> None:
    """Run the checks in the order the generated host stub runs them."""
    if len(call.args) != len(params):
        raise ABIError(f"{kernel_name}: num_args should be {len(params)}")
    if not call.args:
        raise ABIError(f"{kernel_name}: args pointer is NULL")

    for param, value in zip(params, call.args):
        if isinstance(param, ScalarParam):
            type_index = value[0] if isinstance(value, tuple) else ("Int", None)
            if not _type_accepted(param, type_index):
                raise ABIError(f"kernel {kernel_name} scalar {param.name} expected {param.kind}")
            continue

        assert isinstance(param, TensorParam)
        if value is None:
            if param.is_used:
                raise ABIError(f"{kernel_name}.{param.name} is expected to have non-NULL pointer")
            continue
        if not isinstance(value, DLTensor):
            raise ABIError(f"kernel {kernel_name} input {param.name} expected pointer or tensor handle")

        if param.dtype.total_bits < 8 and param.shape:
            # sub-byte logical elements: compare total bit counts, using the
            # packed last dimension. tvm_ffi_binder.cc:545-579
            pack = 8 // param.dtype.total_bits
            expect_bits = param.dtype.total_bits
            expect_count = 1
            for dim in param.shape:
                expect_count *= dim
            provided_count = 1
            for dim in value.shape:
                provided_count *= dim
            if provided_count * pack * value.dtype.total_bits != 8 * expect_count:
                raise ABIError(
                    f"{kernel_name}.{param.name} is a subtype, but total bits mismatch"
                )
            continue

        if len(value.shape) != len(param.shape):
            raise ABIError(
                f"kernel {kernel_name} input {param.name} ndim mismatch, expected {len(param.shape)}"
            )
        if not _dtype_accepted(param.dtype, value.dtype):
            raise ABIError(
                f"kernel {kernel_name} input {param.name} dtype mismatch, expected {param.dtype}"
            )
        for dim, (exp, got) in enumerate(zip(param.shape, value.shape)):
            if exp != got:
                raise ABIError(
                    f"kernel {kernel_name} input {param.name} violates packed ABI constraint"
                )
        if param.contiguity_checked and value.strides is not None:
            expected = 1
            for dim, extent in reversed(list(enumerate(param.shape))):
                if value.strides[dim] != expected:
                    raise ABIError(f"kernel {kernel_name} input {param.name} violates packed ABI constraint")
                expected *= extent


def _scenario(name: str, params: list[object], call: Call, expect: str) -> tuple[str, bool, str]:
    try:
        check_call("vecadd", params, call)
    except ABIError as exc:
        return name, expect in str(exc), str(exc)
    return name, expect == "", "accepted"


def scenarios(trace: bool) -> int:
    a = TensorParam("A", F16, (4, 8))
    b = TensorParam("B", F16, (4, 8))
    n = ScalarParam("n", "int")
    good = Call([DLTensor(F16, (4, 8), strides=(8, 1)), DLTensor(F16, (4, 8), strides=(8, 1))])
    params = [a, b]

    cases = [
        ("correct call", params, good, ""),
        (
            "wrong arg count",
            params,
            Call([good.args[0]]),
            "num_args should be 2",
        ),
        (
            "wrong ndim",
            params,
            Call([DLTensor(F16, (4, 8, 1)), good.args[1]]),
            "ndim mismatch, expected 2",
        ),
        (
            "wrong dtype",
            params,
            Call([DLTensor(F32, (4, 8)), good.args[1]]),
            "dtype mismatch",
        ),
        (
            "used buffer passed as NULL",
            params,
            Call([None, good.args[1]]),
            "is expected to have non-NULL pointer",
        ),
        (
            "shape mismatch",
            params,
            Call([DLTensor(F16, (4, 7)), good.args[1]]),
            "violates packed ABI constraint",
        ),
        (
            "scalar passed a tensor",
            [n],
            Call([("Tensor", None)]),
            "scalar n expected int",
        ),
        (
            "int scalar given a bool",
            [n],
            Call([("Bool", None)]),
            "",
        ),
        (
            "fp8 e4m3fn accepted where e4m3 declared",
            [TensorParam("A", E4M3, (4, 8))],
            Call([DLTensor(E4M3FN, (4, 8))]),
            "",
        ),
        (
            "fp8 e5m2 rejected where e4m3 declared",
            [TensorParam("A", E4M3, (4, 8))],
            Call([DLTensor(E5M2, (4, 8))]),
            "dtype mismatch",
        ),
        (
            "bool buffer given int8 storage",
            [TensorParam("m", BOOL, (4, 8))],
            Call([DLTensor(I8, (4, 8))]),
            "",
        ),
        (
            "non-contiguous where contiguous was declared",
            [TensorParam("A", F16, (4, 8), contiguity_checked=True)],
            Call([DLTensor(F16, (4, 8), strides=(16, 2))]),
            "violates packed ABI constraint",
        ),
        (
            "fp4 with the wrong total bit count",
            [TensorParam("A", FP4, (2, 16))],
            Call([DLTensor(I8, (2, 4))]),
            "total bits mismatch",
        ),
    ]

    failures = 0
    for name, ps, call, expect in cases:
        label, ok, message = _scenario(name, ps, call, expect)
        marker = "ok  " if ok else "FAIL"
        if expect == "":
            print(f"{marker} {label:46s} -> accepted")
        else:
            print(f"{marker} {label:46s} -> {message}")
        if trace and expect == "" and message == "accepted":
            print("       (relaxation applies; see the anchors in the module docstring)")
        failures += 0 if ok else 1

    print()
    print(f"{len(cases) - failures}/{len(cases)} scenarios behaved as the source says")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="store_true", help="annotate accepted relaxation cases")
    args = parser.parse_args()
    return scenarios(args.trace)


if __name__ == "__main__":
    raise SystemExit(main())
