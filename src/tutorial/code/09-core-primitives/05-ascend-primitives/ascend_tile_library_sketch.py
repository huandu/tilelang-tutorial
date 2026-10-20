"""适配器仓库的 T.tile.* 向量内联函数骨架（select / sort / atomic_add）。

静态核对来源：_vendor/tilelang-ascend-v0.1.2.000/
  tilelang/language/__init__.py:86      (from . import ascend_tile as tile)
  tilelang/language/ascend_tile.py:707  (select)
  tilelang/language/ascend_tile.py:412  (sort)
  tilelang/language/ascend_tile.py:550  (topk)
  tilelang/language/ascend_tile.py:313  (atomic_add)
  src/op/ascend.cc:975-1260             (tl.ascend_* 内建 op)
  src/target/codegen_ascend.cc:522-630  (C++ 代码生成分支)

注意：这些 API 只存在于 tilelang-ascend 适配器仓库，**不在** TileLang 主干 v0.1.15。

运行前提：昇腾 NPU + CANN + torch_npu + tilelang-ascend v0.1.2.000。
作者验证级别：仅静态核对（py_compile + API 逐条比对），未在 NPU 上运行。
"""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401
import tilelang
import tilelang.language as T


@tilelang.jit(target="ascend", out_idx=-1)
def select_and_sort(N: int, K: int):
    """一个刻意拼凑的示例：select 做掩码选择，sort 取降序 (值, 索引) 对。"""

    @T.prim_func
    def main(
        X: T.Tensor((N,), "float32"),
        Y: T.Tensor((N,), "float32"),
        Mask: T.Tensor((N // 8,), "uint8"),   # bit-packed 掩码
        Selected: T.Tensor((N,), "float32"),
        Sorted: T.Tensor((2 * N,), "float32"),  # 交错 (值, 索引)
    ):
        with T.Kernel(1) as bx:
            x_ub = T.alloc_ub((N,), "float32")
            y_ub = T.alloc_ub((N,), "float32")
            m_ub = T.alloc_ub((N // 8,), "uint8")
            sel_ub = T.alloc_ub((N,), "float32")
            sorted_ub = T.alloc_ub((2 * N,), "float32")

            T.copy(X, x_ub)
            T.copy(Y, y_ub)
            T.copy(Mask, m_ub)

            # dst[i] = selMask.bit[i] ? src0[i] : src1[i]
            T.tile.select(sel_ub, m_ub, x_ub, y_ub, "VSEL_CMPMASK_SPR")
            T.copy(sel_ub, Selected)

            # 降序全量排序，输出 [val0, idx0, val1, idx1, ...]；
            # float32 输入时，actual_num..aligned_count 之间的位置会被原地填 -inf。
            T.tile.sort(sorted_ub, x_ub, N)
            T.copy(sorted_ub, Sorted)

    return main


@tilelang.jit(target="ascend", out_idx=-1)
def grouped_accumulate(M: int, N: int, num_parts: int):
    """把多个 partial result 用原子累加写回同一个 GM 输出。"""

    @T.prim_func
    def main(
        Parts: T.Tensor((num_parts, M, N), "float32"),
        Out: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(num_parts) as bx:
            src_ub = T.alloc_ub((M, N), "float32")
            T.copy(Parts[bx, :, :], src_ub)
            # V1 模型：dst 必须是 GM，src 必须是本地 tensor（UB 或 L0C）。
            # 若语义是从 0 开始累加，调用前需要先把 Out 清零。
            T.tile.atomic_add(Out, src_ub)

    return main


def main() -> None:
    n = 1024
    kernel = select_and_sort(n, 0)
    x = torch.randn(n, device="npu", dtype=torch.float32)
    y = torch.randn(n, device="npu", dtype=torch.float32)
    mask = torch.randint(0, 256, (n // 8,), device="npu", dtype=torch.uint8)
    selected, sorted_pairs = kernel(x, y, mask)
    bits = torch.tensor(
        [(b >> i) & 1 for b in mask.tolist() for i in range(8)], device="npu"
    )[:n].to(torch.bool)
    torch.testing.assert_close(selected, torch.where(bits, x, y))
    vals = sorted_pairs[0::2]
    assert bool(torch.all(vals[:-1] >= vals[1:])), "sort output is not descending"
    print("select + sort passed.")


if __name__ == "__main__":
    main()
