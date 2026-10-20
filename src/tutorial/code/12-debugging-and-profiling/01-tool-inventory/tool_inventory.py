#!/usr/bin/env python3
"""第 12 章 示例 1：调试与性能分析工具清点表（纯标准库，可在无 GPU/NPU 的机器上运行）。

本章给出的工具名字非常多：IR Lower Trace、Pass Diff、Pass Visualizer、
Compile-Only、Analyzer、plot_layout、pass_timing、T.print、T.printf、
T.dump_tensor、do_bench(backend="msprof")、异常 dump……
如果这些名字只停在正文里，读者无从判断它们到底在不在锁定版本里。

这个脚本把「本章声称存在的工具入口」变成一张**可以复算的表**：直接读
`_vendor/` 下的锁定版本源码，用正则逐个确认入口存在，并打印它出现的
`文件:行号`。它不导入 tilelang（本机没有 CUDA/NPU，编译好的 .so 不存在），
因此**不需要**安装 TileLang、也不需要任何加速器。

用法：

    python3 tool_inventory.py                       # 自动向上寻找书根目录
    python3 tool_inventory.py --book-root /path/to/tilelang-book
    python3 tool_inventory.py --list-missing        # 只打印找不到的条目

退出码：0 = 全部找到；1 = 有条目找不到（说明本章的工具表已经和锁定版本对不上）。

作者实际做到的验证级别：**CPU 逻辑验证**（写作机 macOS / Apple Silicon、
Python 3.11 上实际运行过，输出见同目录 README.md）。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# 两个锁定仓库。路径相对书根目录，见 ../../../../VERSIONS.md。
VENDOR_MAIN = Path("_vendor/tilelang-v0.1.15")
VENDOR_ASCEND = Path("_vendor/tilelang-ascend-v0.1.2.000")

# 每项：(分组, 名字, 仓库根, 相对文件, 正则)
CHECKS: tuple[tuple[str, str, Path, str, str], ...] = (
    (
        "编译期 · 日志",
        "TVM 日志等级表（LOG/DLOG/VLOG）",
        VENDOR_MAIN,
        "src/runtime/logging.cc",
        r"level_strings\[\]",
    ),
    (
        "编译期 · 日志",
        "LOG(FATAL) 报错通道（C++ 侧）",
        VENDOR_MAIN,
        "src/runtime/logging.cc",
        r"void LogFatalImpl",
    ),
    (
        "编译期 · 日志",
        "TVM_LOG_DEBUG 运行时说明符解析",
        VENDOR_MAIN,
        "src/runtime/logging.cc",
        r"TvmLogDebugSettings::ParseSpec",
    ),
    (
        "编译期 · 日志",
        "TVM_LOG_DEBUG 编译期宏（Debug 构建开关）",
        VENDOR_MAIN,
        "CMakeLists.txt",
        r"TVM_LOG_DEBUG",
    ),
    (
        "编译期 · 只看产物",
        "compile_only CLI（--output_file）",
        VENDOR_MAIN,
        "tilelang/tools/compile_only.py",
        r'add_argument\(\s*"--output_file"',
    ),
    (
        "编译期 · 只看产物",
        "compile_kernel_source() 程序 API",
        VENDOR_MAIN,
        "tilelang/tools/compile_only.py",
        r"^def compile_kernel_source",
    ),
    (
        "编译期 · 只看产物",
        "get_kernel_source() / print_source_code()",
        VENDOR_MAIN,
        "tilelang/jit/kernel.py",
        r"^    def get_kernel_source",
    ),
    (
        "编译期 · 只看产物",
        "codegen 后处理回调（CUDA）",
        VENDOR_MAIN,
        "tilelang/engine/callback.py",
        r"^def register_cuda_postproc_callback",
    ),
    (
        "编译期 · 只看产物",
        "codegen 后处理回调（Ascend）",
        VENDOR_MAIN,
        "tilelang/ascend/callback.py",
        r"^def register_ascend_postproc_callback",
    ),
    (
        "编译期 · pass 前后",
        "TL_LOWER_TRACE 环境变量",
        VENDOR_MAIN,
        "tilelang/env.py",
        r"TL_LOWER_TRACE = EnvVar",
    ),
    (
        "编译期 · pass 前后",
        "TL_LOWER_TRACE_DIR 环境变量",
        VENDOR_MAIN,
        "tilelang/env.py",
        r"TL_LOWER_TRACE_DIR = EnvVar",
    ),
    (
        "编译期 · pass 前后",
        "lower_trace() 一次性 diff",
        VENDOR_MAIN,
        "tilelang/tools/lower_trace/__init__.py",
        r"^def lower_trace",
    ),
    (
        "编译期 · pass 前后",
        "enable() / disable() 全流程插桩",
        VENDOR_MAIN,
        "tilelang/tools/lower_trace/core.py",
        r"^def disable",
    ),
    (
        "编译期 · pass 前后",
        "Pass Diff 程序 API（legacy）",
        VENDOR_MAIN,
        "tilelang/utils/pass_diff.py",
        r"^def pass_diff",
    ),
    (
        "编译期 · pass 前后",
        "Pass Visualizer CLI（结构树）",
        VENDOR_MAIN,
        "tilelang/tools/pass_visualizer/viewer.py",
        r"^def main",
    ),
    (
        "编译期 · pass 前后",
        "pass_timing 插桩",
        VENDOR_MAIN,
        "tilelang/tools/pass_timing.py",
        r"class TileLangPassTimingInstrument",
    ),
    (
        "编译期 · 布局",
        "plot_layout()",
        VENDOR_MAIN,
        "tilelang/tools/plot_layout.py",
        r"^def plot_layout",
    ),
    (
        "编译期 · 布局",
        "TL_LAYOUT_VISUALIZATION_ENABLE",
        VENDOR_MAIN,
        "tilelang/transform/pass_config.py",
        r"TL_LAYOUT_VISUALIZATION_ENABLE",
    ),
    (
        "编译期 · 布局",
        "TL_EMIT_LINE_DIRECTIVES（#line 映射）",
        VENDOR_MAIN,
        "tilelang/transform/pass_config.py",
        r"TL_EMIT_LINE_DIRECTIVES = ",
    ),
    (
        "运行期 · 打印",
        "T.print（CUDA 方言）",
        VENDOR_MAIN,
        "tilelang/cuda/language/print.py",
        r"^def print\(",
    ),
    (
        "运行期 · 打印",
        "T.device_assert（CUDA 方言）",
        VENDOR_MAIN,
        "tilelang/cuda/debug.py",
        r"^def device_assert",
    ),
    (
        "运行期 · 打印",
        "T.print（Ascend 方言）",
        VENDOR_MAIN,
        "tilelang/ascend/language/print.py",
        r"^def print\(",
    ),
    (
        "运行期 · 打印",
        "Ascend debug 打印模板（printf）",
        VENDOR_MAIN,
        "src/tl_templates/ascend/debug.h",
        r"TL_DEFINE_ASCEND_DEBUG",
    ),
    (
        "运行期 · 打印",
        "T.printf（Ascend 适配器）",
        VENDOR_ASCEND,
        "tilelang/language/ascend.py",
        r"^def printf",
    ),
    (
        "运行期 · 打印",
        "T.dump_tensor（Ascend 适配器）",
        VENDOR_ASCEND,
        "tilelang/language/ascend.py",
        r"^def dump_tensor",
    ),
    (
        "运行期 · 异常现场",
        "TL_ASCEND_EXCEPTION_DUMP 配置键",
        VENDOR_ASCEND,
        "tilelang/transform/pass_config.py",
        r"TL_ASCEND_EXCEPTION_DUMP = ",
    ),
    (
        "运行期 · 异常现场",
        "parse_exception_dump()",
        VENDOR_ASCEND,
        "tilelang/tools/ascend_exception_dump_bin.py",
        r"^def parse_exception_dump",
    ),
    (
        "运行期 · 异常现场",
        "read_msaicerr_bin()",
        VENDOR_ASCEND,
        "tilelang/tools/ascend_exception_dump_bin.py",
        r"^def read_msaicerr_bin",
    ),
    (
        "运行期 · 异常现场",
        "异常 dump 的 MAGIC 搜索（C++ 模板）",
        VENDOR_ASCEND,
        "src/tl_templates/ascend/exception_dump.h",
        r"MAGIC|TILELANG",
    ),
    (
        "正确性",
        "torch_assert_close() 容差工具",
        VENDOR_MAIN,
        "tilelang/utils/tensor.py",
        r"^def torch_assert_close",
    ),
    (
        "正确性",
        "Profiler.assert_allclose()",
        VENDOR_MAIN,
        "tilelang/profiler/__init__.py",
        r"^    def assert_allclose",
    ),
    (
        "正确性",
        "AutoDD 自动缩小复现（--err-msg）",
        VENDOR_MAIN,
        "tilelang/autodd.py",
        r'add_argument\("--err-msg"',
    ),
    (
        "性能",
        "Analyzer.analysis() 静态 roofline 估算",
        VENDOR_MAIN,
        "tilelang/tools/Analyzer.py",
        r"def analysis",
    ),
    (
        "性能",
        "do_bench(backend=...) 计时入口",
        VENDOR_MAIN,
        "tilelang/profiler/bench.py",
        r'backend: Literal\["event", "cupti", "cudagraph", "wall", "msprof", "msprof_detail"\]',
    ),
    (
        "性能",
        "msprof 计时实现 bench_with_msprof()",
        VENDOR_MAIN,
        "tilelang/profiler/msprof.py",
        r"^def bench_with_msprof",
    ),
    (
        "性能",
        "KernelProfile（AIC/AIV pipe 利用率）",
        VENDOR_MAIN,
        "tilelang/profiler/msprof.py",
        r"class KernelProfile",
    ),
)


def find_book_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "VERSIONS.md").is_file() and (candidate / "_vendor").is_dir():
            return candidate
    raise SystemExit("找不到书根目录（向上找不到同时含 VERSIONS.md 与 _vendor/ 的目录）")


def locate(path: Path, pattern: str) -> tuple[int, str] | None:
    if not path.is_file():
        return None
    regex = re.compile(pattern)
    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if regex.search(line):
            return lineno, line.strip()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--book-root", type=Path, default=None, help="本书根目录（默认从脚本位置向上寻找）")
    parser.add_argument("--list-missing", action="store_true", help="只打印找不到的条目")
    args = parser.parse_args()

    book = args.book_root.resolve() if args.book_root else find_book_root(Path(__file__).resolve().parent)

    total = 0
    missing = 0
    last_group = ""
    for group, name, repo, rel, pattern in CHECKS:
        total += 1
        hit = locate(book / repo / rel, pattern)
        if hit is None:
            missing += 1
            if args.list_missing:
                print(f"  [缺] {group} :: {name}   -> {repo / rel}")
            else:
                print(f"  [缺] {name:<38} {repo / rel}")
            continue
        if args.list_missing:
            continue
        if group != last_group:
            last_group = group
            print(f"\n== {group} ==")
        lineno, text = hit
        print(f"  [有] {name:<38} {repo / rel}:{lineno}")

    print(f"\n合计 {total} 个入口，找到 {total - missing} 个，缺失 {missing} 个。")
    print("注意：本表证明的是「入口在锁定版本源码里存在」，不是「它在你的机器上能跑通」。")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
