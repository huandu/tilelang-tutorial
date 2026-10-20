#!/usr/bin/env python3
"""第 19 章 §8.6：配置扫描驱动（纯标准库，先定「测什么」再上机器）。

这个脚本**不调用任何加速器**：它只负责
  1. 生成待测配置（tile 尺寸、num_stages、sweep 范围）；
  2. 给出每个配置的**资源账**（L1/L0C/UB 占用估算），提前排除明显放不下的配置；
  3. 生成一份「实验记录表」模板（测量条件 + 结果字段），保证 A/B 可比。

这样做的价值：等真正上了 NPU，你只需要按表跑，不用现场想「该记什么」。

验证级别：CPU 逻辑验证（纯标准库，可在任意机器上复算）。
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field

# Atlas A2 片上容量（字节），来源：
# ../tilelang-corpus/06-huawei-ascend/raw/catlass/docs/zh/2_Design/01_kernel_design/00_basics/atlasA2_hardware_info.md
CAPACITY_BYTES = {
    "l1": 512 * 1024,
    "l0a": 64 * 1024,
    "l0b": 64 * 1024,
    "l0c": 128 * 1024,
    "ub": 192 * 1024,
}

DTYPE_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float": 4,
}


@dataclass
class GemmConfig:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    in_dtype: str = "float16"
    accum_dtype: str = "float32"

    def ledger(self) -> dict[str, int]:
        """算出这个配置对 L1 / L0A / L0B / L0C 的占用（字节，含 num_stages 倍）。"""
        b_in = DTYPE_BYTES[self.in_dtype]
        b_acc = DTYPE_BYTES[self.accum_dtype]
        s = self.num_stages
        return {
            "l1": s * b_in * (self.block_m * self.block_k + self.block_k * self.block_n),
            "l0a": s * b_in * (self.block_m * self.block_k),
            "l0b": s * b_in * (self.block_k * self.block_n),
            "l0c": b_acc * (self.block_m * self.block_n),
        }

    def fits(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for scope, need in self.ledger().items():
            cap = CAPACITY_BYTES[scope]
            if need > cap:
                reasons.append(f"{scope}: need {need} B > cap {cap} B ({need / cap:.2f}x)")
        return (not reasons), reasons

    def describe(self) -> str:
        ok, reasons = self.fits()
        led = self.ledger()
        head = (
            f"BM={self.block_m:>4} BN={self.block_n:>4} BK={self.block_k:>4} "
            f"stages={self.num_stages} {self.in_dtype}->{self.accum_dtype}"
        )
        detail = " ".join(f"{k}={v // 1024}KB" for k, v in led.items())
        verdict = "OK" if ok else "OVER: " + "; ".join(reasons)
        return f"{head} | {detail} | {verdict}"


@dataclass
class ExperimentRecord:
    """一条实验记录：测量条件 + 结果（字段对齐 M 树 SKILL §12.2 的清单）。"""

    config: str
    device: str = ""
    cann_version: str = ""
    torch_npu_version: str = ""
    tilelang_tree: str = ""  # "adapter@v0.1.2.000" 或 "main@v0.1.15"
    pass_configs: str = ""
    correctness: str = ""
    latency_us: str = ""
    bottleneck: str = ""
    notes: str = ""
    raw: list[float] = field(default_factory=list)


def sweep_gemm(block_ms, block_ns, block_ks, stages):
    for bm, bn, bk, s in itertools.product(block_ms, block_ns, block_ks, stages):
        yield GemmConfig(bm, bn, bk, s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ascend tuning sweep planner (stdlib only)")
    parser.add_argument("--mode", choices=["gemm", "vec"], default="gemm")
    parser.add_argument("--tile", type=int, nargs="*", default=[8192])
    parser.add_argument("--stages", type=int, nargs="*", default=[1, 2, 3])
    args = parser.parse_args()

    print("== 片上容量（Atlas A2，字节）==")
    for scope, cap in CAPACITY_BYTES.items():
        print(f"  {scope:>4}: {cap:>8} B ({cap // 1024} KB)")
    print()

    if args.mode == "gemm":
        print("== GEMM 配置扫描（按资源账预筛）==")
        n_over = 0
        for cfg in sweep_gemm([64, 128, 256], [128, 256], [32, 64], args.stages):
            print("  " + cfg.describe())
            if not cfg.fits()[0]:
                n_over += 1
        print()
        print(f"预筛剔除：{n_over} 个配置放不下（这些不用上机器）。")
    else:
        print("== 向量配置扫描（UB 占用）==")
        for tile, s in itertools.product(args.tile, args.stages):
            need = s * 3 * 4 * tile  # 3 个 fp32 buffer（a/b/c），s 级流水
            ok = need <= CAPACITY_BYTES["ub"]
            print(f"  tile={tile:>6} stages={s} | ub={need // 1024}KB | {'OK' if ok else 'OVER'}")
    print()

    print("== 实验记录表模板（照着填，A/B 才可比）==")
    template = ExperimentRecord(
        config="<BM/BN/BK/stages>",
        device="<Ascend 型号>",
        cann_version="<CANN 版本>",
        torch_npu_version="<torch_npu 版本>",
        tilelang_tree="<adapter@v0.1.2.000 | main@v0.1.15>",
        pass_configs="<如 TL_ASCEND_AUTO_SYNC=True>",
        correctness="<通过/失败>",
        latency_us="<中位数>",
        bottleneck="<AIC 最忙的 pipe / AIV 最忙的 pipe>",
        notes="<只改了什么变量>",
        raw=[],
    )
    for field_name in ExperimentRecord.__dataclass_fields__:
        print(f"  {field_name} = {getattr(template, field_name)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
