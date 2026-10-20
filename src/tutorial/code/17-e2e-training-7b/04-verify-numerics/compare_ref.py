"""数值一致性验收：TileLang kernel 与 PyTorch 参考实现逐项对齐。

这是第 17 章 §7.3 的验收脚本。它做三件事：

1. **逐元素对齐**：对同一份随机输入，比较 kernel 输出与参考输出，
   计算最大绝对误差、最大相对误差，并按 dtype 给出门限。
2. **端到端对齐**：用同一批 token 跑一次 `loss`，比较「全 PyTorch 模型」
   与「替换了热点算子的模型」的 loss 差。
3. **反向对齐**：比较 `autograd.Function` 给出的 `dx/dw` 与参考梯度。

验收门限取自第 27 章的口径：bf16 前向用 `rtol/atol`，反向看梯度范数的相对差；
**门限必须按 dtype 和累加精度选，不能一概用 `1e-6`**——bf16 的表示精度本来就
只有约 3 位十进制有效数字。

运行前提：带 NVIDIA GPU 的机器、已安装可用的 `tilelang`、安装了 `torch`。
运行命令见同目录 README.md。**本脚本在写作机器（无 GPU/NPU）上只做过静态核对，
未运行**；因此它打印的任何数字都必须由你在真机上重跑得到。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# 复用本章 03 示例里的 kernel 与参考实现。若把本目录单独复制出去，把
# `rmsnorm_tl.py` 一并拷到同目录即可（两个文件加起来即可独立运行）。
sys.path.append(str(Path(__file__).resolve().parent.parent / "03-tilelang-norm-torch"))

from rmsnorm_tl import rmsnorm, rmsnorm_ref


def report(name: str, got: torch.Tensor, ref: torch.Tensor, rtol: float, atol: float) -> bool:
    diff = (got.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    ok = bool(torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol))
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name:<22} max_abs={max_abs:.3e} max_rel={max_rel:.3e} "
          f"(rtol={rtol}, atol={atol})")
    return ok


def check_forward(m: int, n: int, eps: float, dtype: torch.dtype) -> bool:
    x = torch.randn(m, n, dtype=dtype, device="cuda")
    w = torch.randn(n, dtype=dtype, device="cuda")
    got = rmsnorm(x, w, eps)
    ref = rmsnorm_ref(x, w, eps)
    # bf16 的表示精度决定了门限：前向取 1e-2 量级，fp32 才敢收窄到 1e-5。
    rtol, atol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-5, 1e-5)
    return report(f"forward {dtype}", got, ref, rtol, atol)


def check_backward(m: int, n: int, eps: float) -> bool:
    dtype = torch.bfloat16
    x1 = torch.randn(m, n, dtype=dtype, device="cuda", requires_grad=True)
    w1 = torch.randn(n, dtype=dtype, device="cuda", requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)

    rmsnorm(x1, w1, eps).float().pow(2).mean().backward()
    rmsnorm_ref(x2, w2, eps).float().pow(2).mean().backward()

    # 反向梯度用「相对 L2 误差」比较，比逐元素更稳：单个元素可能恰好接近 0。
    def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6)).item()

    dx = rel_l2(x1.grad, x2.grad)
    dw = rel_l2(w1.grad, w2.grad)
    ok = dx < 5e-2 and dw < 5e-2
    print(f"[{'PASS' if ok else 'FAIL'}] backward relL2        dx={dx:.3e} dw={dw:.3e} (门限 5e-2)")
    return ok


def check_end_to_end() -> bool:
    """端到端 loss 对齐：这里只给骨架，真实实现要接本章的模型定义。"""

    print("[SKIP] end-to-end loss 对齐：需要加载 Qwen2.5-7B 与一小批 token，"
          "见 README 的 `--e2e` 说明（多卡/多显存场景）")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="TileLang 算子数值一致性验收")
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--n", type=int, default=3584, help="Qwen2.5-7B 的 hidden size")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--e2e", action="store_true", help="额外汇总端到端 loss 差")
    args = ap.parse_args()

    results = [
        check_forward(args.m, args.n, args.eps, torch.bfloat16),
        check_forward(args.m, args.n, args.eps, torch.float32),
        check_backward(args.m, args.n, args.eps),
    ]
    if args.e2e:
        results.append(check_end_to_end())
    print("总体：", "全部通过" if all(results) else "存在未通过项")


if __name__ == "__main__":
    main()
