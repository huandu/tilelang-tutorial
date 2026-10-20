#!/usr/bin/env python3
"""Verify every hop of the compile path documented in chapter 04.

The chapter claims a specific call chain, e.g.
``JITKernel._compile_artifact -> tilelang.engine.lower.lower_with_context``.
This script does not run TileLang (no GPU, no compiled extension needed); it
checks that each claimed file exists and that the claimed symbol is defined at
the claimed line. That is exactly the level of evidence the chapter claims:
static source verification, not execution.

Usage:
    python3 check_compile_path.py
    python3 check_compile_path.py --vendor /path/to/tilelang-v0.1.15

Exit code 0 = every hop verified; 1 = at least one hop failed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve()
BOOK_ROOT = HERE.parents[4]


@dataclass(frozen=True)
class Hop:
    """One step of the documented compile path."""

    order: int
    stage: str
    relative_path: str
    expected_line: int
    needle: str


# Every entry here is quoted in chapter.md (figure 2 and sections 4.1-4.4).
HOPS: tuple[Hop, ...] = (
    Hop(1, "装饰器", "tilelang/jit/__init__.py", 588, "def jit("),
    Hop(2, "装饰器 → JITImpl", "tilelang/jit/__init__.py", 270, "class JITImpl"),
    Hop(3, "取 TIR", "tilelang/jit/__init__.py", 379, "def get_tir"),
    Hop(4, "lazy 编译入口", "tilelang/jit/__init__.py", 466, "def compile(self"),
    Hop(5, "缓存查询", "tilelang/cache/__init__.py", 75, "def cached("),
    Hop(6, "缓存派发", "tilelang/cache/__init__.py", 38, "def _resolve_cache_dispatch("),
    Hop(7, "KernelCache", "tilelang/cache/kernel_cache.py", 35, "class KernelCache"),
    Hop(8, "JITKernel", "tilelang/jit/kernel.py", 41, "class JITKernel"),
    Hop(9, "解析 BackendContext", "tilelang/jit/kernel.py", 113, "if backend_context is None"),
    Hop(10, "编译 artifact", "tilelang/jit/kernel.py", 288, "def _compile_artifact"),
    Hop(11, "调用 lower_with_context", "tilelang/jit/kernel.py", 313, "from tilelang.engine.lower import lower_with_context"),
    Hop(12, "engine lower", "tilelang/engine/lower.py", 216, "def lower("),
    Hop(13, "engine lower_with_context", "tilelang/engine/lower.py", 144, "def lower_with_context("),
    Hop(14, "host/device 切分", "tilelang/engine/lower.py", 109, "def lower_to_host_device_ir("),
    Hop(15, "前端语义检查", "tilelang/engine/semantic_check.py", 21, "def PreLowerSemanticCheck("),
    Hop(16, "跑后端 pipeline", "tilelang/engine/lower.py", 136, "mod = context.lower(mod)"),
    Hop(17, "BackendContext.lower", "tilelang/backend/module.py", 211, "def lower(self, mod: IRModule) -> IRModule:"),
    Hop(18, "PassPipeline.lower", "tilelang/backend/pass_pipeline/pipeline.py", 22, "def lower(self, mod: IRModule, target: Target) -> IRModule:"),
    Hop(19, "CUDA pipeline", "tilelang/cuda/pipeline.py", 284, 'CUDA_PIPELINE = PassPipeline("cuda"'),
    Hop(20, "Ascend pipeline", "tilelang/ascend/pipeline.py", 190, 'ascend_pipeline = PassPipeline("ascend"'),
    Hop(21, "device codegen", "tilelang/engine/lower.py", 99, "def device_codegen("),
    Hop(22, "host codegen", "tilelang/engine/lower.py", 65, "def host_codegen("),
    Hop(23, "CompiledArtifact", "tilelang/engine/param.py", 154, "class CompiledArtifact"),
    Hop(24, "adapter 构造", "tilelang/jit/kernel.py", 324, "def _create_adapter_from_artifact("),
    Hop(25, "可调用入口", "tilelang/jit/kernel.py", 206, "def __call__(self, *args"),
    Hop(26, "shared.host_codegen 表", "tilelang/backend/host_codegen.py", 55, "STANDARD_HOST_CODEGENS"),
    Hop(27, "CUDA manifest", "tilelang/cuda/backend.py", 123, "BACKEND = register_backend("),
    Hop(28, "Ascend manifest", "tilelang/ascend/backend.py", 52, "BACKEND = register_backend("),
    Hop(29, "注册表校验", "tilelang/backend/module.py", 238, "def register_backend("),
    Hop(30, "MakePackedAPI 的 C++ 本体", "src/transform/make_packed_api.cc", 2, "Lower PrimFunc to use the packed function API"),
)


def check(vendor: Path) -> tuple[list[str], list[str]]:
    ok: list[str] = []
    bad: list[str] = []
    for hop in HOPS:
        path = vendor / hop.relative_path
        if not path.is_file():
            bad.append(f"E{hop.order:02d} 文件不存在: {hop.relative_path}")
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if hop.expected_line < 1 or hop.expected_line > len(lines):
            bad.append(f"E{hop.order:02d} 行号越界: {hop.relative_path}:{hop.expected_line}（文件共 {len(lines)} 行）")
            continue
        actual = lines[hop.expected_line - 1]
        if hop.needle not in actual:
            bad.append(f"E{hop.order:02d} 第 {hop.expected_line} 行不含 `{hop.needle}`，实际为: {actual.strip()!r}")
            continue
        ok.append(f"E{hop.order:02d} {hop.stage:<22} {hop.relative_path}:{hop.expected_line}")
    return ok, bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", default=str(BOOK_ROOT / "_vendor" / "tilelang-v0.1.15"))
    args = parser.parse_args()

    vendor = Path(args.vendor)
    if not vendor.is_dir():
        print(f"vendor tree not found: {vendor}", file=sys.stderr)
        return 2

    ok, bad = check(vendor)
    print(f"compile-path hops: {len(ok)}/{len(HOPS)} verified")
    for line in ok:
        print(f"  ok   {line}")
    for line in bad:
        print(f"  FAIL {line}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
