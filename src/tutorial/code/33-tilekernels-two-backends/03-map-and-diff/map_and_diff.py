#!/usr/bin/env python3
"""第 33 章：纯标准库地清点 TileKernels 的三层结构，并产出后端结构对照表。

这个脚本不 import torch、不 import tilelang，也不需要任何加速器。
它只做三件事：

  1. 扫描一个 TileKernels 源码树，列出所有 *_kernel.py / *_cuda.py / *_asc.py
     三元组（缺哪一个也照实报告）—— 把「三层结构」变成一份可核对的清单。
  2. 对每一对 *_cuda.py / *_asc.py，统计一组**结构性标记**的出现次数，
     产出一张对照表 —— 这就是正文 §4.4 那张表的数据来源。
  3. （可选）在 TileLang 锁定源码树里核对本章用到的 API 名是否存在。

用法：

    python3 map_and_diff.py                       # 用默认路径
    python3 map_and_diff.py --root <tile_kernels 目录>
    python3 map_and_diff.py --vendor <tilelang vendor 树>

作者在 macOS (Apple Silicon)、Python 3.11 上运行过；输出见 README.md。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

# 相对本文件的默认路径（见 README 的说明）。
DEFAULT_ROOT = os.path.normpath(
    os.path.join(HERE, '../../../../..', 'tilelang-corpus/14-deepseek-ascend-stack/raw/TileKernels/tile_kernels')
)
DEFAULT_VENDOR = os.path.normpath(os.path.join(HERE, '../../../../_vendor/tilelang-v0.1.15'))

# 结构性标记：字符串即在源码里直接搜。左边是表格里的展示名。
MARKERS = [
    ('T.Kernel(num_tokens', 'CUDA 版按 token 开网格'),
    ('T.Kernel(num_vec_cores)', 'Ascend 版按 Vector 核开网格'),
    ('threads=', '显式线程数（SIMT）'),
    ('alloc_fragment', 'fragment（寄存器）'),
    ('alloc_reducer', 'reducer（延迟归约）'),
    ('alloc_shared', 'shared / UB'),
    ('annotate_buffer_versions', '显式缓冲版本'),
    ('Pipelined', '软件流水'),
    ('Persistent', '持久化循环'),
    ('SimdVF', 'SIMD 向量作用域'),
    ('SimtVF', 'SIMT 线程作用域'),
    ('mem_bar', '手写同步'),
    ('pdl_sync', 'PDL（CUDA-only）'),
    ('vector_length', '显式向量宽度'),
    ('import_source', '内联设备代码'),
    ('T.view', '位视图'),
    ('warp_reduce_', 'warp 级归约'),
    ('shfl_xor', 'warp shuffle'),
]

# 本章正文点名过、必须存在于锁定版本的 API。
API_NAMES = [
    'alloc_reducer', 'reducer_init', 'reducer_update', 'finalize_reducer',
    'pdl_sync', 'pdl_trigger', 'view', 'SimdVF', 'SimtVF', 'Kernel',
    'annotate_buffer_versions', 'Pipelined', 'Persistent', 'import_source',
    'rng_init', 'rng_rand_float', 'warp_reduce_min', 'warp_reduce_max',
    'warp_reduce_sum', 'get_thread_binding', 'device_assert',
]


def find_triples(root: str) -> dict[str, dict[str, str]]:
    """把 <root> 下的算子按基名归组，返回 {基名: {层: 路径}}。"""
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith('.py'):
                continue
            match = re.fullmatch(r'(.+)_(kernel|cuda|asc)\.py', filename)
            if match is None:
                continue
            base, layer = match.group(1), match.group(2)
            rel = os.path.relpath(os.path.join(dirpath, filename), root)
            groups[f'{os.path.dirname(rel)}/{base}'.lstrip('/')][layer] = rel
    return dict(sorted(groups.items()))


def count_markers(path: str) -> dict[str, int]:
    with open(path, 'r', encoding='utf-8') as handle:
        text = handle.read()
    return {name: text.count(name) for name, _ in MARKERS}


def print_inventory(groups: dict[str, dict[str, str]]) -> tuple[int, int]:
    print('== 1. 三层结构清单 ==')
    print(f'{"算子（基名）":<46}{"kernel":<8}{"cuda":<6}{"asc":<6}')
    print('-' * 66)
    complete = 0
    for base, layers in groups.items():
        marks = [('✓' if layer in layers else '—') for layer in ('kernel', 'cuda', 'asc')]
        if all(layer in layers for layer in ('kernel', 'cuda', 'asc')):
            complete += 1
        print(f'{base:<46}{marks[0]:<8}{marks[1]:<6}{marks[2]:<6}')
    print('-' * 66)
    print(f'合计 {len(groups)} 个算子家族；三件齐全的 {complete} 个。')
    return len(groups), complete


def print_diff(groups: dict[str, dict[str, str]], root: str) -> None:
    print()
    print('== 2. 结构性对照（只列两边都存在 cuda/asc 的算子）==')
    for base, layers in groups.items():
        if 'cuda' not in layers or 'asc' not in layers:
            continue
        cuda = count_markers(os.path.join(root, layers['cuda']))
        asc = count_markers(os.path.join(root, layers['asc']))
        rows = [(label, cuda[name], asc[name]) for name, label in MARKERS if cuda[name] or asc[name]]
        if not rows:
            continue
        print()
        print(f'--- {base} ---')
        print(f'{"结构性标记":<34}{"cuda":>6}{"asc":>6}')
        for label, left, right in rows:
            flag = '  <-- 只在一边' if (left == 0) != (right == 0) else ''
            print(f'{label:<34}{left:>6}{right:>6}{flag}')


def print_api_check(vendor: str) -> None:
    print()
    print('== 3. 本章用到的 API 在锁定版本里的存在性 ==')
    if not os.path.isdir(vendor):
        print(f'跳过：找不到 vendor 树 {vendor}')
        return
    definitions: dict[str, list[str]] = {}
    pattern = re.compile(r'^\s*def (\w+)\s*\(')
    for dirpath, _dirnames, filenames in os.walk(vendor):
        if os.sep + 'tests' in dirpath or os.sep + 'examples' in dirpath:
            continue
        for filename in filenames:
            if not filename.endswith('.py'):
                continue
            path = os.path.join(dirpath, filename)
            with open(path, 'r', encoding='utf-8') as handle:
                for line in handle:
                    match = pattern.match(line)
                    if match is not None and match.group(1) in API_NAMES:
                        definitions.setdefault(match.group(1), []).append(os.path.relpath(path, vendor))
    # 优先展示语言层（language/）的定义位置：底层实现文件到处都是，
    # 但从 `T.<name>` 能不能调到，取决于语言层有没有导出它。
    def prefer(path: str) -> tuple[int, str]:
        score = 1 if '/language/' in path or path.endswith('debug.py') else 0
        return (-score, path)

    missing = []
    for name in API_NAMES:
        hits = definitions.get(name)
        if hits is None:
            missing.append(name)
            print(f'{name:<26} MISSING')
        else:
            hits = sorted(set(hits), key=prefer)
            shown = ', '.join(hits[:2])
            extra = '' if len(hits) <= 2 else f'  (+{len(hits) - 2} more)'
            print(f'{name:<26} ok  ({shown}){extra}')
    print(f'\n{len(API_NAMES) - len(missing)}/{len(API_NAMES)} 命中；缺：{missing if missing else "无"}')


def main() -> int:
    parser = argparse.ArgumentParser(description='清点 TileKernels 的三层结构并产出后端对照表')
    parser.add_argument('--root', default=DEFAULT_ROOT, help='TileKernels 的 tile_kernels/ 目录')
    parser.add_argument('--vendor', default=DEFAULT_VENDOR, help='TileLang 锁定源码树（用于 API 存在性核对）')
    parser.add_argument('--skip-api-check', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f'找不到 --root 目录：{args.root}', file=sys.stderr)
        print('请指向一个 TileKernels 源码树里的 tile_kernels/ 目录。', file=sys.stderr)
        return 2

    print(f'扫描根目录：{args.root}')
    groups = find_triples(args.root)
    print_inventory(groups)
    print_diff(groups, args.root)
    if not args.skip_api_check:
        print_api_check(args.vendor)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
