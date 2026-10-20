"""Two Sinkhorn variants that live in the same operator library (CPU only).

The library ships two kernels that are both called "Sinkhorn" but do different
jobs:

  * ``mhc_sinkhorn`` (``TileKernels/tile_kernels/mhc/sinkhorn_cuda.py``,
    baseline ``sinkhorn_normalize_ref`` in ``tile_kernels/torch/mhc.py``)
    normalises a small, per-token ``4 x 4`` mix matrix: first a row softmax,
    then alternating row / column sum normalisation, which drives the matrix
    towards a doubly stochastic one.
  * ``engram_sinkhorn`` (``TileKernels/tile_kernels/engram/engram_sinkhorn_cuda.py``,
    baseline ``engram_sinkhorn_step_ref`` / ``..._finalize_ref`` in
    ``tile_kernels/torch/engram.py``) applies an explicit per-row and per-column
    *L2* scale to a big ``(num_rows, head_dim)`` matrix and is driven as a
    multi-kernel pipeline (momentum -> step -> step_reduce -> finalize).

This script prints: convergence of the mHC variant, the numerical reason the
row softmax comes first, and the per-iteration state of the Engram variant.

Run:  python3 sinkhorn_iterations.py
"""

from __future__ import annotations

import torch


def mhc_sinkhorn(x: torch.Tensor, repeat: int = 10, eps: float = 1e-6) -> torch.Tensor:
    """Verbatim copy of ``sinkhorn_normalize_ref`` (tile_kernels/torch/mhc.py:8)."""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def doubly_stochastic_error(x: torch.Tensor) -> tuple[float, float]:
    row = (x.sum(-1) - 1.0).abs().max().item()
    col = (x.sum(-2) - 1.0).abs().max().item()
    return row, col


def engram_step_ref(matrix, row_scale, col_scale, eps: float):
    """Mirrors ``engram_sinkhorn_step_ref`` (tile_kernels/torch/engram.py:39)."""
    iterate = row_scale.unsqueeze(1) * matrix * col_scale
    row_norm = torch.norm(iterate, p=2, dim=1)
    row_scale.div_(row_norm + eps)
    new_iterate = row_scale.unsqueeze(1) * matrix * col_scale
    return row_norm, new_iterate.square().sum(dim=0)


def engram_finalize_ref(matrix, row_scale, col_scale, eps: float, alpha: float):
    """Mirrors ``engram_sinkhorn_finalize_ref`` (tile_kernels/torch/engram.py:66)."""
    iterate = row_scale.unsqueeze(1) * matrix * col_scale
    row_norm = torch.norm(iterate, p=2, dim=1)
    new_row_scale = row_scale / (row_norm + eps)
    return alpha * new_row_scale.unsqueeze(1) * matrix * col_scale


def part_mhc() -> None:
    print("== mHC Sinkhorn: 4 x 4, softmax then row/col normalisation ==")
    torch.manual_seed(0)
    x0 = torch.randn(2, 4, 4, dtype=torch.float32) * 2.0

    print("  repeat   max|row_sum-1|  max|col_sum-1|  max|X_k - X_{k-1}|")
    prev = None
    for repeat in (1, 3, 10, 30):
        out = mhc_sinkhorn(x0, repeat=repeat)
        row, col = doubly_stochastic_error(out)
        delta = float("nan") if prev is None else (out - prev).abs().max().item()
        print(f"  {repeat:6d}   {row:14.3e}  {col:14.3e}  {delta:18.3e}")
        prev = out

    print("\n  the row softmax is what keeps large logits finite:")
    for offset in (100.0, -100.0):
        x = torch.tensor(
            [[0.0, 1.0, 2.0, 3.0], [3.0, 1.0, 2.0, 0.0], [1.5, 0.5, 3.5, 2.5], [2.5, 3.5, 0.5, 1.5]]
        ).view(1, 4, 4) + offset
        out = mhc_sinkhorn(x, repeat=10, eps=1e-6)
        print(f"  offset {offset:+6.0f}: finite={bool(torch.isfinite(out).all())}  max|row-1|={doubly_stochastic_error(out)[0]:.3e}")

    print("\n  why the eps adds up: row sums are 1 in the ideal case but the kernel")
    print("  divides by (sum + eps) and adds eps after the softmax, so the fixed point")
    print("  is approximate -- see tile_kernels/mhc/sinkhorn_cuda.py:36-52.")


def part_engram() -> None:
    print("\n== Engram Sinkhorn: alternating L2 row / column scaling ==")
    torch.manual_seed(1)
    num_rows, head_dim, eps = 3, 4, 1e-20
    matrix = torch.randn(num_rows, head_dim, dtype=torch.float32) * 3.0
    row_scale = torch.ones(num_rows, dtype=torch.float32)
    col_scale = torch.ones(head_dim, dtype=torch.float32)

    print(f"  matrix {tuple(matrix.shape)} fp32; row_scale {tuple(row_scale.shape)}; col_scale {tuple(col_scale.shape)}")
    print("  iterate = row_scale[:, None] * matrix * col_scale")
    print("  step   max|row L2 - 1| after row step        col L2 before col step")
    for step in range(8):
        _, col_sumsq = engram_step_ref(matrix, row_scale, col_scale, eps)  # row step, in place
        iterate = row_scale.unsqueeze(1) * matrix * col_scale
        row_after = torch.norm(iterate, p=2, dim=1)
        col_norm = torch.norm(iterate, p=2, dim=0)
        if step % 2 == 0 or step == 7:
            print(f"  {step:4d}   {float((row_after - 1).abs().max()):28.3e}   {[round(v, 5) for v in col_norm.tolist()]}")
        col_scale = col_scale / torch.sqrt(col_sumsq + eps)  # column step

    print("  after a few rounds the row / column scale profiles stop changing:")

    alpha = head_dim**0.5  # what the shipped test uses (tests/engram/test_engram_sinkhorn.py:141)
    before = torch.norm(row_scale.unsqueeze(1) * matrix * col_scale, p=2, dim=1)
    out = engram_finalize_ref(matrix, row_scale.clone(), col_scale, eps, alpha)
    row_norm_out = torch.norm(out, p=2, dim=1)
    print(f"    row L2 before finalize: {[round(v, 5) for v in before.tolist()]}")
    print(f"    row L2 after  finalize (alpha = sqrt(head_dim) = {alpha}): {[round(v, 5) for v in row_norm_out.tolist()]}")

    print("\n  shape contrast with the mHC variant:")
    print("    mHC    : per-token 4 x 4, 10 iterations, softmax + sums, tiny state")
    print("    Engram : (num_rows, head_dim) with num_rows ~ 1e5, L2 scales,")
    print("             split across several kernels because rows are processed independently")
    print("             and the column reduction needs a cross-core pass.")


def main() -> None:
    part_mhc()
    part_engram()


if __name__ == "__main__":
    main()
