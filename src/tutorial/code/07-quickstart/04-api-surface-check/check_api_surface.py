"""第七章 · 离线核对本章用到的 API 是否真的存在于锁定版本源码里。

本脚本不 import tilelang，只读源码文本。它把本章正文里出现的每个 API 名
（以及昇腾专有的那几个）在 ``_vendor/tilelang-v0.1.15/`` 与
``_vendor/tilelang-ascend-v0.1.2.000/`` 里搜一遍，打印「文件:行号」和它所在的那一行。

这是把「我说 v0.1.15 里有这个 API」从一句断言变成一次可复现检查的办法。

运行：
    python check_api_surface.py                 # 自动从仓库根找到 _vendor
    python check_api_surface.py --root /path/to/tilelang-book

验证级别：**在本机运行过（纯 Python，只读源码）**。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# (相对路径, 正则, 说明)
CHECKS = [
    # ---- 公共前端（本章第 4 节） ----
    ("_vendor/tilelang-v0.1.15/tilelang/language/kernel.py", r"^def Kernel\(", "T.Kernel（target-neutral 网格）"),
    ("_vendor/tilelang-v0.1.15/tilelang/cuda/language/kernel.py", r"threads: int \| list\[int\] \| tuple\[int, \.\.\.\] \| None = None", "T.Kernel 的 threads 关键字（CUDA 方言）"),
    ("_vendor/tilelang-v0.1.15/tilelang/language/loop.py", r"^def Parallel\(", "T.Parallel（块级并行域）"),
    ("_vendor/tilelang-v0.1.15/tilelang/language/copy_op.py", r"^def copy\(", "T.copy（tile 级搬运）"),
    ("_vendor/tilelang-v0.1.15/tilelang/language/allocate.py", r"^def alloc_shared\(", "T.alloc_shared"),
    ("_vendor/tilelang-v0.1.15/tilelang/language/allocate.py", r"^def alloc_fragment\(", "T.alloc_fragment"),
    ("_vendor/tilelang-v0.1.15/tilelang/language/fill_op.py", r"^def clear\(", "T.clear"),
    ("_vendor/tilelang-v0.1.15/tilelang/language/symbolics.py", r"^def dynamic\(", "T.dynamic"),
    ("_vendor/tilelang-v0.1.15/tilelang/language/tir/entry.py", r"^def prim_func\(", "T.prim_func"),
    # ---- 运行时 / JIT（本章第 5、6 节） ----
    ("_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py", r"^def jit\(", "tilelang.jit"),
    ("_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py", r"^def compile\(", "tilelang.compile"),
    ("_vendor/tilelang-v0.1.15/tilelang/jit/__init__.py", r"^    def compile\(self", "JITImpl.compile"),
    ("_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py", r"^    def get_kernel_source\(", "JITKernel.get_kernel_source"),
    ("_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py", r"^    def get_profiler\(", "JITKernel.get_profiler"),
    ("_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py", r"^    def show_source\(", "JITKernel.show_source"),
    ("_vendor/tilelang-v0.1.15/tilelang/jit/kernel.py", r"^    def export_sources\(", "JITKernel.export_sources"),
    ("_vendor/tilelang-v0.1.15/tilelang/env.py", r"^    TILELANG_CACHE_DIR = EnvVar", "缓存目录环境变量"),
    ("_vendor/tilelang-v0.1.15/tilelang/env.py", r"^    def disable_cache\(self\)", "disable_cache"),
    ("_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py", r"^    def _generate_key\(", "KernelCache key 生成"),
    ("_vendor/tilelang-v0.1.15/tilelang/cache/kernel_cache.py", r"^    def clear_cache\(self\)", "KernelCache.clear_cache（实例方法）"),
    # ---- 官方示例（本章第 4 节） ----
    ("_vendor/tilelang-v0.1.15/examples/elementwise/example_elementwise_add.py", r"^def elementwise_add\(", "官方 elementwise 示例"),
    ("_vendor/tilelang-v0.1.15/examples/quickstart.py", r"^def matmul\(", "官方 quickstart GEMM+ReLU"),
    # ---- 昇腾（本章第 7 节） ----
    ("_vendor/tilelang-v0.1.15/tilelang/ascend/language/kernel.py", r"^def Kernel\(", "昇腾 T.Kernel（1-D 核网格，无 threads=）"),
    ("_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py", r"^def SimtVF\(", "T.SimtVF"),
    ("_vendor/tilelang-v0.1.15/tilelang/ascend/language/frame.py", r"^def SimdVF\(", "T.SimdVF"),
    ("_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py", r"^def alloc_l1\(", "T.alloc_l1"),
    ("_vendor/tilelang-v0.1.15/tilelang/ascend/language/allocate.py", r"^def alloc_l0c\(", "T.alloc_l0c"),
    ("_vendor/tilelang-v0.1.15/tilelang/ascend/language/copy_op.py", r"^def dual_copy\(", "T.dual_copy"),
    ("_vendor/tilelang-v0.1.15/examples/ascend/example_vecadd.py", r"^def vector_add\(", "官方昇腾 vecadd 示例"),
    # ---- 昇腾独立仓库（对照，本章第 7 节） ----
    ("_vendor/tilelang-ascend-v0.1.2.000/README.md", r"^### Environment Preparation", "tilelang-ascend 安装说明"),
    ("_vendor/tilelang-ascend-v0.1.2.000/install_ascend.sh", r"Python version \$python_version >= 3.10", "install_ascend.sh 的 Python 版本门"),
    ("_vendor/tilelang-ascend-v0.1.2.000/build_wheel_ascend.sh", r"ASCEND_HOME_PATH is not set", "build_wheel_ascend.sh 要求 ASCEND_HOME_PATH"),
    ("_vendor/tilelang-ascend-v0.1.2.000/tilelang/cache/__init__.py", r"^def clear_cache\(", "tilelang-ascend 独有：tilelang.cache.clear_cache()"),
]


# 这些 API 在主干 v0.1.15 里「应当不存在」，用来证明本章的分界说法
NEGATIVE_CHECKS = [
    ("_vendor/tilelang-v0.1.15/tilelang/cache/__init__.py", r"^def clear_cache\(", "主干 v0.1.15 的 tilelang/cache 没有顶层 clear_cache()"),
    ("_vendor/tilelang-v0.1.15/tilelang/ascend/language", r"\bis_npu\b", "主干 ascend 方言没有 is_npu 参数"),
    ("_vendor/tilelang-v0.1.15/tilelang/ascend/language", r"^def alloc_ub\(", "主干 ascend 方言没有 alloc_ub（UB 用 alloc_shared）"),
]


def find_root(explicit: str | None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "_vendor" / "tilelang-v0.1.15").is_dir():
            return parent
    raise SystemExit("找不到 _vendor/tilelang-v0.1.15；请用 --root 指定 tilelang-book 根目录。")


def search(path: Path, pattern: str) -> list[tuple[int, str]]:
    if not path.exists():
        return []
    rx = re.compile(pattern)
    files: list[Path] = []
    if path.is_dir():
        files = sorted(path.rglob("*.py"))
    else:
        files = [path]
    hits: list[tuple[int, str]] = []
    for child in files:
        for i, line in enumerate(child.read_text(errors="ignore").splitlines(), 1):
            if rx.search(line):
                label = line.strip() if path.is_file() else f"{child.relative_to(path)}: {line.strip()}"
                hits.append((i, label))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None, help="tilelang-book 根目录（含 _vendor）")
    args = parser.parse_args()
    root = find_root(args.root)

    failures = 0
    print(f"仓库根：{root}\n")
    print("== 应当存在 ==")
    for rel, pattern, note in CHECKS:
        hits = search(root / rel, pattern)
        if hits:
            line_no = hits[0][0]
            print(f"  ok   {note:<44} {rel}:{line_no}")
        else:
            failures += 1
            print(f"  FAIL {note:<44} 未在 {rel} 找到 /{pattern}/")

    print("\n== 应当不存在（用于证明主干与 tilelang-ascend 的分界） ==")
    for rel, pattern, note in NEGATIVE_CHECKS:
        hits = search(root / rel, pattern)
        if hits:
            failures += 1
            print(f"  FAIL {note:<44} 竟然命中 {rel}:{hits[0][0]}")
        else:
            print(f"  ok   {note:<44} {rel} 中未命中")

    print()
    if failures:
        print(f"{failures} 项不符合预期。")
        return 1
    print(f"全部 {len(CHECKS) + len(NEGATIVE_CHECKS)} 项符合预期。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
