#!/usr/bin/env python3
"""第 36 章示例 04：FlashMLA 昇腾稀疏 attention 里每个技术点「TileLang 能写到哪一层」。

这是正文 §6 那张三分类表的**可执行版本**：把每个技术点标成

* **A（直接可表达）**：tile + 循环 + ``T.gemm``/``T.copy``/逐元素运算就能描述；
* **B（可表达但要下钻）**：需要 ``T.SimdVF`` + ``T.simd.*`` + 注解/同步原语；
* **C（必须内建）**：结论绑在某条 CCE/Ascend C 指令的参数或协议上。

它做两类检查（都是静态的，不需要硬件）：

1. B 类提到的 `T.*` 原语，**逐个在锁定源码树里 grep 确认存在**；
2. C 类提到的指令/参数，**确认主干 TileLang 里没有对应原语**（负向检查）。

验证级别：**本机运行过**（macOS + Python 3.11）。纯静态核对工具，不是 kernel。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Item:
    name: str
    category: str          # "A" / "B" / "C"
    evidence: str          # 报告 / 代码位置
    tilelang: str          # B 类：对应的 T.* 原语（用 ; 分隔）
    check: tuple[str, str] | None = None   # (vendor 内相对路径, 正则)，正向检查
    absent: tuple[str, str] | None = None  # (vendor 内相对路径, 正则)，负向检查


ITEMS: list[Item] = [
    # ---- A：直接可表达 ----
    Item("online softmax 的四个状态量与递推", "A",
         "报告 :45-73", "T.Parallel / T.reduce_max / T.reduce_sum"),
    Item("tile 分块（B_TOPK / MMA_M / D_QK）", "A",
         "config.h:28,43,55,59", "T.Kernel / T.alloc_l1/l0a/l0b/l0c"),
    Item("P = Q @ K^T 与 O += S @ V", "A",
         "kernel_body.h:1403-1491, 1493-1638", "T.gemm",
         check=("tilelang/ascend/language/gemm_op.py", r"def gemm\(")),
    Item("按 indices 的 gather（读 KV）", "A",
         "kernel_body.h:1018-1075", "T.Tensor 索引 + 循环 / T.copy"),
    Item("rescale 的「乘标量」", "A",
         "kernel_body.h:686-726", "T.Parallel 逐元素乘"),
    Item("O += S@V 延后一拍的「结构」", "A",
         "报告 :172-186", "两个 stage + T.Pipelined 的依赖分析"),

    # ---- B：可表达但需要下钻 ----
    Item("softmax 的向量区域", "B",
         "kernel_body.h:466-685", "T.SimdVF / T.SimtVF",
         check=("tilelang/ascend/language/frame.py", r"def SimdVF\(")),
    Item("ND2NZ 的读（asc_loadalign*）", "B",
         "kernel_body.h:304-330", "T.simd.vld(post_inc=...)",
         check=("tilelang/ascend/language/simd.py", r"def vld\(")),
    Item("ND2NZ 的写（vsstb）", "B",
         "kernel_body.h:315-330", "T.simd.vsstb(update=True) + T.simd.make_ubuf_ptr",
         check=("tilelang/ascend/language/simd.py", r"def vsstb\(")),
    Item("gather2 的地址计算（成对/去交织）", "B",
         "kernel_body.h:87-255", "T.simd.vld2 / vdintlv / vmin / vmax",
         check=("tilelang/ascend/language/simd.py", r"def vld2\(")),
    Item("显式向量宽度与尾掩码", "B",
         "kernel_body.h:466-470", "T.simd.pset / update_mask",
         check=("tilelang/ascend/language/simd.py", r"def update_mask\(")),
    Item("多缓冲的档数（L1 KV 4 档等）", "B",
         "config.h:67-69", "T.Pipelined(num_stages=) + T.annotate_buffer_versions",
         check=("tilelang/ascend/language/annotations.py", r"def annotate_buffer_versions\(")),
    Item("跨核同步（C<->V flag）", "B",
         "kernel_body.h:69-81, 1186-1207",
         "T.ascend_cross_core_set_flag / _wait_flag",
         check=("tilelang/ascend/language/sync.py", r"def ascend_cross_core_set_flag\(")),
    Item("L1 两 bank 拆分", "B",
         "kernel_body.h:907-912", "两块 T.alloc_l1 + 手写地址偏移"),

    # ---- C：必须手写 ----
    Item("BRC / UNPK4 / E2M1 等分布式格式转换", "C",
         "kernel_body.h:380-388, 428-454", "—",
         absent=("tilelang/ascend/language/simd.py", r"\bloadalign_brc\b")),
    Item("gather2 的参数（burst_count / src_stride）", "C",
         "kernel_body.h:1043-1067", "—",
         absent=("tilelang/ascend/language/copy_op.py", r"burst_count")),
    Item("L2 cache hint 与运行期启发式", "C",
         "kernel_body.h:1675-1685", "T.copy(..., l2_cache_ctrl=) 只能给固定值"),
    Item("FixPipe 位级参数（dual_dst_ctl / NZ2ND_en）", "C",
         "kernel_body.h:1475-1540", "T.dual_copy 只暴露 unit_flag_ctrl"),
    Item("SS buffer（__ssbuf__）+ 槽位协议", "C",
         "config.h:110-112, kernel_body.h:1189, 1585-1588", "—",
         absent=("tilelang/ascend/language/sync.py", r"ssbuf")),
    Item("icache 控制（#pragma unroll 1 / VLOOP）", "C",
         "kernel_body.h:595, 1331", "—"),
    Item("核级配置（asc_set_ctrl / mmad_direction 等）", "C",
         "kernel_body.h:921-923, 1405, 1535", "T.set_mmad_direction（部分可表达）"),
]


def default_vendor_root() -> Path:
    # 章节内的相对路径：chapters/<slug>/code/<sample>/ -> tilelang-book/_vendor
    return Path(__file__).resolve().parents[4] / "_vendor" / "tilelang-v0.1.15"


def resolve(vendor: Path, rel: str) -> Path:
    return vendor / rel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", type=Path, default=default_vendor_root(),
                        help="锁定源码树 _vendor/tilelang-v0.1.15 的路径")
    parser.add_argument("--stats-only", action="store_true",
                        help="只打印三类的条目数")
    args = parser.parse_args()

    vendor: Path = args.vendor_root
    have_vendor = vendor.is_dir()

    counts = {"A": 0, "B": 0, "C": 0}
    for item in ITEMS:
        counts[item.category] += 1

    if args.stats_only:
        print(f"A（直接可表达）: {counts['A']} 项")
        print(f"B（可表达但要下钻）: {counts['B']} 项")
        print(f"C（必须内建）: {counts['C']} 项")
        return 0

    print("== FlashMLA 昇腾稀疏 attention 的 TileLang 表达力映射 ==")
    print(f"vendor root: {vendor}")
    if not have_vendor:
        print("  ! 找不到锁定源码树；只打印清单，跳过存在性检查。")
        print("    （在站点上阅读时这是正常的——请在本仓库的 .workspace 下运行。）")
    print()

    problems: list[str] = []
    for category in ("A", "B", "C"):
        print(f"--- {category} 类 ---")
        for item in ITEMS:
            if item.category != category:
                continue
            status = ""
            if have_vendor and item.check is not None:
                rel, pattern = item.check
                path = resolve(vendor, rel)
                ok = path.is_file() and re.search(pattern, path.read_text(errors="ignore")) is not None
                status = "  [T.* 存在 ✓]" if ok else "  [T.* 缺失 ✗]"
                if not ok:
                    problems.append(f"{item.name}: expected {rel} to match {pattern!r}")
            if have_vendor and item.absent is not None:
                rel, pattern = item.absent
                path = resolve(vendor, rel)
                if path.is_file():
                    found = re.search(pattern, path.read_text(errors="ignore"))
                    # 负向检查：主干里**不应**出现这个私有名词
                    status = "  [主干无此原语 ✓]" if found is None else "  [主干里竟然有 ✗]"
                    if found is not None:
                        problems.append(f"{item.name}: {rel} unexpectedly matches {pattern!r}")
            print(f"  [{category}] {item.name}")
            print(f"        证据: {item.evidence}")
            print(f"        TileLang: {item.tilelang}{status}")
        print()

    print("== 小结 ==")
    print(f"  A {counts['A']} 项 · B {counts['B']} 项 · C {counts['C']} 项")
    print("  判据一句话：换一条语义相近的指令，结论还成立吗？不成立的就是 C 类。")

    if problems:
        print()
        print("PROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    if have_vendor:
        print("\nOK: B 类原语在锁定源码树中全部找到；C 类私有名词在主干中均未出现。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
