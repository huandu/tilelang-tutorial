#!/usr/bin/env python3
"""Print the module inventory of a TileLang checkout.

The numbers quoted in chapter 04 come from this script, so they can be
regenerated instead of trusted. It only walks the filesystem: no imports, no
network, no GPU.

Usage:
    python3 inventory.py
    python3 inventory.py --vendor /path/to/tilelang-v0.1.15 --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
BOOK_ROOT = HERE.parents[4]

DEFAULT_VENDOR = BOOK_ROOT / "_vendor" / "tilelang-v0.1.15"

# The modules chapter 04 promises to explain, with the one-line job each owns.
MODULES: tuple[tuple[str, str], ...] = (
    ("tilelang/language", "语言表面：tile op、frame、两种前端"),
    ("tilelang/engine", "编译入口：lower / semantic_check / param"),
    ("tilelang/transform", "通用 pass 的 Python 包装"),
    ("src/transform", "pass 本体（C++）"),
    ("tilelang/layout", "Layout / Fragment / swizzle 代数"),
    ("tilelang/backend", "后端注册、组件接口、BackendContext"),
    ("tilelang/cuda", "CUDA 后端垂直切片"),
    ("tilelang/ascend", "Ascend 后端垂直切片"),
    ("src/ascend", "Ascend codegen / op / layout / runtime（C++）"),
    ("tilelang/jit", "装饰器、JITKernel、adapter"),
    ("tilelang/cache", "kernel 级与二进制级缓存"),
    ("tilelang/autotuner", "参数搜索"),
    ("tilelang/carver", "tile 配置推荐"),
    ("tilelang/tools", "lower_trace / plot_layout / pass_timing"),
    ("examples", "可运行示例"),
    ("docs", "官方便携文档源文件"),
)


def count(root: Path, suffix: str) -> int:
    return sum(1 for _ in root.rglob(f"*{suffix}") if _.is_file())


def collect(vendor: Path) -> dict[str, object]:
    report: dict[str, object] = {
        "vendor": str(vendor),
        "version_file": (vendor / "VERSION").read_text(encoding="utf-8").strip() if (vendor / "VERSION").is_file() else None,
        "modules": [],
        "totals": {},
    }

    modules: list[dict[str, object]] = []
    for relative, duty in MODULES:
        path = vendor / relative
        modules.append(
            {
                "path": relative,
                "duty": duty,
                "exists": path.is_dir(),
                "python_files": count(path, ".py") if path.is_dir() else 0,
                "cpp_files": (count(path, ".cc") + count(path, ".h")) if path.is_dir() else 0,
                "subdirs": sum(1 for child in path.iterdir() if child.is_dir()) if path.is_dir() else 0,
            }
        )
    report["modules"] = modules

    examples = vendor / "examples"
    docs = vendor / "docs"
    tilelang = vendor / "tilelang"
    src_transform = vendor / "src" / "transform"

    report["totals"] = {
        "tilelang_top_level_dirs": sum(1 for c in tilelang.iterdir() if c.is_dir() and c.name != "__pycache__") if tilelang.is_dir() else 0,
        "tilelang_python_files": count(tilelang, ".py") if tilelang.is_dir() else 0,
        "src_transform_cc_files": sum(1 for c in src_transform.glob("*.cc")) if src_transform.is_dir() else 0,
        "examples_subdirs": sum(1 for c in examples.iterdir() if c.is_dir()) if examples.is_dir() else 0,
        "examples_top_level_files": sum(1 for c in examples.iterdir() if c.is_file()) if examples.is_dir() else 0,
        "docs_markdown_files": count(docs, ".md") if docs.is_dir() else 0,
    }
    return report


def render_text(report: dict[str, object]) -> str:
    lines = [f"vendor: {report['vendor']}", f"VERSION: {report['version_file']}", ""]
    lines.append("| 模块 | 职责 | 存在 | .py | .cc/.h | 子目录 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for module in report["modules"]:  # type: ignore[union-attr]
        lines.append(
            f"| `{module['path']}` | {module['duty']} | {'是' if module['exists'] else '否'} | "
            f"{module['python_files']} | {module['cpp_files']} | {module['subdirs']} |"
        )
    lines.append("")
    lines.append("totals:")
    for key, value in report["totals"].items():  # type: ignore[union-attr]
        lines.append(f"  {key} = {value}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", default=str(DEFAULT_VENDOR))
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    vendor = Path(args.vendor)
    if not vendor.is_dir():
        print(f"vendor tree not found: {vendor}", file=sys.stderr)
        return 2

    report = collect(vendor)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
