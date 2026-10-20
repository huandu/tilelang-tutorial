"""把 Transformer 里的 RMSNorm 换成 TileLang kernel，并接回 PyTorch autograd。

这一段是第 17 章 §7「把热点算子换成 TileLang」的可读版本，回答三个问题：

1. TileLang kernel 怎么写（前向）；
2. 怎么让 `kernel(x, w)` 直接吃 `torch.Tensor`、并把输出当 `torch.Tensor` 拿回来；
3. 反向从哪里来——TileLang 不提供 autograd，所以必须由你显式接上。

API 与锁定版本 `_vendor/tilelang-v0.1.15/` 对齐：
`@tilelang.jit(out_idx=[...])`、`T.prim_func`、`T.copy`、`T.Parallel`、
`T.reduce_sum`、`T.rsqrt`、`T.alloc_fragment`。见 references.md R7–R11。

运行前提：本机需要 NVIDIA GPU（或昇腾 NPU，见同目录 README 的昇腾分支）与可用的
`tilelang`。**下面这段代码在写作机器上只做过静态核对，没有运行过**——写作机器没有
GPU/NPU。真实运行请在带卡的机器上执行同目录 `compare_ref.py`。
"""

from __future__ import annotations

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(out_idx=[2])  # 第 3 个参数（Y）是输出，由 adapter 分配并返回
def _rmsnorm_fwd(M: int, N: int, eps: float, blk_m: int = 1, threads: int = 128,
                 dtype: str = "bfloat16"):
    accum = "float"

    @T.prim_func
    def main(
        X: T.Tensor((M, N), dtype),
        W: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
            x_local = T.alloc_fragment((blk_m, N), accum)
            sq_local = T.alloc_fragment((blk_m, N), accum)
            sum_sq = T.alloc_fragment((blk_m,), accum)
            rstd = T.alloc_fragment((blk_m,), accum)

            T.copy(X[bx * blk_m, 0], x_local)
            for i, j in T.Parallel(blk_m, N):
                sq_local[i, j] = x_local[i, j] * x_local[i, j]
            T.reduce_sum(sq_local, sum_sq, dim=1)
            for i in T.Parallel(blk_m):
                rstd[i] = T.rsqrt(sum_sq[i] / N + eps)
            for i, j in T.Parallel(blk_m, N):
                x_local[i, j] = x_local[i, j] * rstd[i] * T.Cast(accum, W[j])
            T.copy(x_local, Y[bx * blk_m, 0])

    return main


def rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """纯 PyTorch 参考实现，用来对齐数值。"""

    rstd = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * rstd) * w


class RMSNormTileLang(torch.autograd.Function):
    """把前向 kernel 和反向（这里先用 PyTorch 算子）拼成一个 autograd.Function。

    TileLang 只产出**前向**的 device 函数；`torch.autograd` 不知道它，也不会为它
    建图。要让 `loss.backward()` 穿过这个 kernel，只有一条路：把 kernel 包进
    `torch.autograd.Function`，在 `forward` 里调用 kernel，在 `backward` 里自己
    写（或再调一个 TileLang kernel）反向。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
        # x: (M, N) bf16, w: (N,) bf16；M 必须是 blk_m 的整数倍（这里取 1）
        x = x.contiguous()
        kernel = _rmsnorm_fwd(x.shape[0], x.shape[1], eps)
        y = kernel(x, w)  # 直接吃 torch.Tensor，返回 torch.Tensor
        ctx.save_for_backward(x, w)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, w = ctx.saved_tensors
        eps = ctx.eps
        x32 = x.float()
        w32 = w.float()
        rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
        xn = x32 * rstd
        grad_w = (grad_y.float() * xn).sum(dim=0).to(w.dtype)
        # dx 的标准形式：dx = rstd * (g*w - xn * mean(g*w*xn))
        g = grad_y.float() * w32
        grad_x = (rstd * (g - xn * (g * xn).mean(-1, keepdim=True))).to(x.dtype)
        return grad_x, grad_w, None


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """对外的算子入口：训练代码里 `model.forward` 直接调用它。"""

    return RMSNormTileLang.apply(x, w, eps)


if __name__ == "__main__":
    # 这段只有解析意义：本机无 GPU，导入 tilelang 会失败。
    # 在带卡的机器上，把 `cuda` 换成 `npu` 即可跑昇腾分支。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(256, 512, dtype=torch.bfloat16, device=device, requires_grad=True)
    w = torch.randn(512, dtype=torch.bfloat16, device=device, requires_grad=True)
    y = rmsnorm(x, w)
    (y.float().pow(2).mean()).backward()
    print("y", y.shape, "dx", x.grad.shape, "dw", w.grad.shape)
