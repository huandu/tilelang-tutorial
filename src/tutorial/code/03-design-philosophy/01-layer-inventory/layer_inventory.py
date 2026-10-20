#!/usr/bin/env python3
"""第 03 章 示例 1：三层接口 API 清点表（纯标准库，可在无 GPU 的机器上运行）。

这个脚本不导入 tilelang（本机也装不上：没有 CUDA/NPU，编译好的 .so 不存在），
它只做一件事：**在锁定版本 v0.1.15 的源码里，逐个确认本章声称存在的 API 名字
确实出现在语言层的导出面上**，然后把结果打出来。

它回答的是第 03 章最容易被写成空话的那类问题：
「beginner / developer / expert 三层各自到底有哪些构造？」
把这个问题变成一张可以复算的表，而不是一段形容词。

用法：

    python3 layer_inventory.py                # 自动定位书根目录
    python3 layer_inventory.py --book-root /path/to/tilelang-book
    python3 layer_inventory.py --list-missing  # 只打印找不到的名字

退出码：0 = 所有名字都能找到；1 = 有名字找不到（说明本章的结论过期了）。

作者实际做到的验证级别：**CPU 逻辑验证**。脚本在写作机上实际运行过，输出见同目录
README.md。注意它验证的是「名字出现在源码里」，**不是**「这个 API 在你的机器上能用」——
后者需要真实安装并运行 TileLang。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

VENDOR_REL = Path("_vendor/tilelang-v0.1.15")

# 语言层的「导出面」：这些文件里出现的 `"name"`（导出清单）或 `def name` 都算数。
SURFACE_GLOBS = (
    "tilelang/language/*.py",
    "tilelang/language/*/*.py",
    "tilelang/cuda/language/*.py",
)

# 三层接口各自的名字。清单来自第 03 章正文的表，来源逐个写在 references.md。
LAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "beginner (eager 前端，层本身未完成)",
        (
            "const",  # examples/eager_jit/eagerjit.en.ipynb: M, N, K = T.const("M, N, K")
            "dynamic",  # 同上 notebook 的 T.dynamic(...) 单元格
            "empty",  # T.empty((M, N), dtype)
            "Tensor",  # A: T.Tensor[[M, K], T.float16]
            "StridedTensor",
            "macro",
            "prim_func",
        ),
    ),
    (
        "developer (tile 库)",
        (
            "Kernel",
            "Parallel",
            "Pipelined",
            "Persistent",
            "copy",
            "gemm",
            "gemm_sp",
            "gemm_blockscaled",
            "reduce",
            "reduce_max",
            "reduce_sum",
            "cumsum",
            "cummax",
            "fill",
            "clear",
            "alloc_shared",
            "alloc_fragment",
            "annotate_layout",
            "use_swizzle",
            "GemmWarpPolicy",
        ),
    ),
    (
        "expert (thread 原语与内建)",
        (
            "get_thread_binding",
            "get_block_binding",
            "alloc_local",
            "alloc_var",
            "serial",
            "vectorized",
            "unroll",
            "sync_threads",
            "atomic_add",
            "tvm_thread_allreduce",
            "comm_reducer",
            "attr",
            "evaluate",
            "ptx_wait_group",
        ),
    ),
    (
        "CUDA 方言扩展（只有 CUDA target 才有）",
        (
            "ClusterKernel",
            "copy_cluster",
            "mbarrier_wait_parity",
            "tma_load",
            "ldg128",
            "alloc_tmem",
        ),
    ),
)


def find_book_root(start: Path) -> Path:
    """向上找一个含 _vendor/tilelang-v0.1.15 的目录。"""
    for candidate in (start, *start.parents):
        if (candidate / VENDOR_REL).is_dir():
            return candidate
    raise SystemExit(f"找不到 {VENDOR_REL}：请用 --book-root 指定书根目录")


def surface_files(vendor: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in SURFACE_GLOBS:
        files.extend(sorted(vendor.glob(pattern)))
    return sorted(set(files))


def locate(name: str, files: list[Path], root: Path) -> tuple[str, str] | None:
    """返回 (证据形式, `相对路径:行号`)；优先找函数定义，其次找导出清单里的字符串。

    证据形式只有两种，都写进输出里，避免读者把「出现在导出清单里」误读成「这里定义了它」：
      * ``def`` —— 该文件里有 ``def <name>``；
      * ``export`` —— 该文件把 "<name>" 写进了导出清单或再导出语句。
    """
    quoted = re.compile(r"""["']""" + re.escape(name) + r"""["']""")
    defined = re.compile(r"\bdef " + re.escape(name) + r"\b")
    sources: list[str] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        sources.append(text)

    for kind, pattern in (("def", defined), ("export", quoted)):
        for path, text in zip(files, sources):
            for lineno, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    try:
                        shown = path.relative_to(root)
                    except ValueError:
                        shown = path
                    return kind, f"{shown}:{lineno}"
    return None


def check_doc_note(vendor: Path) -> tuple[bool, str]:
    """核对「beginner 层未完成」这句官方说明是否还在。"""
    doc = vendor / "docs/get_started/overview.md"
    needle = "not yet fully implemented"
    if not doc.is_file():
        return False, f"缺少 {doc}"
    for lineno, line in enumerate(doc.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        if needle in line:
            return True, f"docs/get_started/overview.md:{lineno}  含 “{needle}”"
    return False, f"docs/get_started/overview.md 里找不到 “{needle}”"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--book-root", type=Path, default=None)
    parser.add_argument("--list-missing", action="store_true", help="只打印找不到的名字")
    args = parser.parse_args()

    start = args.book_root if args.book_root is not None else Path(__file__).resolve().parent
    book_root = find_book_root(start)
    vendor = book_root / VENDOR_REL
    files = surface_files(vendor)
    if not files:
        print(f"在 {vendor} 下没找到任何语言层源文件", file=sys.stderr)
        return 1

    print(f"书根目录 : {book_root}")
    print(f"锁定版本 : {VENDOR_REL}  (VERSION = {(vendor / 'VERSION').read_text().strip()})")
    print(f"扫描文件 : {len(files)} 个语言层源文件")
    print()

    total = 0
    missing: list[tuple[str, str]] = []
    for layer, names in LAYERS:
        print(f"== {layer} ==")
        for name in names:
            total += 1
            found = locate(name, files, vendor)
            if found is None:
                missing.append((layer, name))
                if not args.list_missing:
                    print(f"  [缺失] {name}")
            elif not args.list_missing:
                kind, where = found
                print(f"  [有]   {name:<22} {kind:<7} {where}")
        print()

    ok_doc, doc_msg = check_doc_note(vendor)
    print("== 官方文档核对：beginner 层的实现状态 ==")
    print(f"  [{'有' if ok_doc else '缺失'}] {doc_msg}")
    print()

    print(f"合计 {total} 个名字，找到 {total - len(missing)} 个，缺失 {len(missing)} 个。")
    if missing:
        print("缺失清单（说明本章结论需要重新核对）：")
        for layer, name in missing:
            print(f"  - {name}  [{layer}]")
    return 0 if not missing and ok_doc else 1


if __name__ == "__main__":
    raise SystemExit(main())
