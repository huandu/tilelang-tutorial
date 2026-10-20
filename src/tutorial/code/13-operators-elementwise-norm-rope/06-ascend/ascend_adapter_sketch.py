"""第 13 章 · 昇腾侧同款实现（**适配器仓库** tilelang-ascend v0.1.2.000）。

需要：昇腾 NPU + CANN + torch_npu + `tilelang-ascend` 适配器 v0.1.2.000-release。

⚠️ 这个文件里的 API 在**主干 v0.1.15 里不存在**，逐个核对过：

    $ ls _vendor/tilelang-v0.1.15/tilelang/ascend/language/
      allocate.py  annotations.py  copy_op.py  frame.py  gemm_op.py  ...
    $ rg -n "def alloc_ub" _vendor/tilelang-v0.1.15/
      （无输出）
    $ rg -n "def alloc_ub" _vendor/tilelang-ascend-v0.1.2.000/tilelang/language/allocate.py
      156:def alloc_ub(shape, dtype):

所以本章把两种写法分开写、分开标注：

* 主干 v0.1.15：`tilelang.ascend.language`，UB 用 `T.alloc_shared`，
  线程级并行用 `T.SimtVF`，没有 `T.tile.*`（见 `ascend_mainline_kernels.py`）。
* 适配器 v0.1.2.000：`tilelang.language`，UB 用 `T.alloc_ub`，向量计算用
  `T.tile.*`（本文件），`T.Kernel(..., is_npu=True)` 返回 `(cid, vid)`。

本文件只做**静态核对**：API 名逐个对照过 `_vendor/tilelang-ascend-v0.1.2.000/`，
`python3 -m py_compile` 通过。**没有在 NPU 上编译或运行过**，不声明任何性能数字。

对照的官方示例：
  _vendor/tilelang-ascend-v0.1.2.000/examples/elementwise/elementwise_add.py
  _vendor/tilelang-ascend-v0.1.2.000/examples/normalization/rms_norm.py
  _vendor/tilelang-ascend-v0.1.2.000/examples/pos_embedding/rms_rope_fused.py
"""

from __future__ import annotations

import tilelang
import tilelang.language as T

# 适配器的自动同步/内存规划 pass 要显式打开；主干 v0.1.15 里没有这些键。
PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

VEC_NUM = 2  # AIC / AIV 两条向量核各处理一半，取自官方示例的约定


# ---------------------------------------------------------------------------
# ① 逐元素：向量加
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[-1], pass_configs=PASS_CONFIGS)
def vec_add(M: int, N: int, block_M: int = 128, block_N: int = 256, dtype: str = "float"):
    """`C = A + B`，结构对照 `examples/elementwise/elementwise_add.py`。

    三个与 CUDA 版不同的地方：

    1. `T.Kernel(m_num * n_num, is_npu=True) as (cid, vid)`：`cid` 是核号，
       `vid` 是这一核上的**向量子核**编号；`block_M // VEC_NUM` 就是让两个子核分一半。
    2. `a_ub` / `b_ub` / `c_ub` 全是 UB（`T.alloc_ub`）。搬运靠 `T.copy`（DMA），
       计算靠 `T.tile.add`（Vector 指令），两者之间用 `T.barrier_all()` 拉开。
    3. 没有 `threads=`：向量宽度是硬件固定的，`T.tile.*` 直接对整个 UB tile 下达指令。
    """
    m_num = M // block_M
    n_num = N // block_N
    rows = block_M // VEC_NUM

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore[valid-type]
        B: T.Tensor((M, N), dtype),  # type: ignore[valid-type]
        C: T.Tensor((M, N), dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(m_num * n_num, is_npu=True) as (cid, vid):
            bx = (cid // n_num) * block_M + vid * rows
            by = (cid % n_num) * block_N

            a_ub = T.alloc_ub((rows, block_N), dtype)
            b_ub = T.alloc_ub((rows, block_N), dtype)
            c_ub = T.alloc_ub((rows, block_N), dtype)
            with T.Scope("V"):
                T.copy(A[bx, by], a_ub)
                T.copy(B[bx, by], b_ub)
                T.barrier_all()
                T.tile.add(c_ub, a_ub, b_ub)
                T.barrier_all()
                T.copy(c_ub, C[bx, by])

    return main


# ---------------------------------------------------------------------------
# ② 归一化：RMSNorm（tile library 路径）
# ---------------------------------------------------------------------------


@tilelang.jit(out_idx=[1], pass_configs=PASS_CONFIGS)
def rms_norm_adapter(
    M: int,
    N: int,
    block_M: int = 64,
    block_N: int = 512,
    eps: float = 1e-6,
    dtype: str = "float16",
):
    """RMSNorm，结构对照：

    * `examples/pos_embedding/rms_norm.py`（UB 上的 fp16→fp32→fp16 三件套）
    * `examples/normalization/rms_norm.py`（分块累加 + `T.reduce_sum`）

    和主干写法的差别全在「原语」这一层：主干用 `T.Parallel` 让编译器去向量化，
    适配器用 `T.tile.mul` / `T.tile.add` / `T.tile.broadcast` 直接下达 UB 上的
    向量指令；`T.reduce_sum` 的名字一样，但适配器版本的签名多了 `real_shape`
    与 `tmp` 两个参数（见 `tilelang/language/reduce_ascend.py`），写混了会报
    未知参数。
    """
    if M % block_M != 0 or N % block_N != 0:
        raise ValueError(f"needs M % {block_M} == 0 and N % {block_N} == 0, got M={M}, N={N}")
    m_num = M // block_M
    n_num = N // block_N
    rows = block_M // VEC_NUM
    cal_dtype = "float32" if dtype in ("float16", "bfloat16") else dtype
    need_cast = cal_dtype != dtype

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # type: ignore[valid-type]
        B: T.Tensor((M, N), dtype),  # type: ignore[valid-type]
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            row0 = cid * block_M + vid * rows

            a_ub = T.alloc_ub((rows, block_N), dtype)
            a_cal = T.alloc_ub((rows, block_N), cal_dtype)
            sq_acc = T.alloc_ub((rows, block_N), cal_dtype)
            sum_sq = T.alloc_ub((rows,), cal_dtype)
            rstd = T.alloc_ub((rows,), cal_dtype)
            rstd_tile = T.alloc_ub((rows, block_N), cal_dtype)

            with T.Scope("V"):
                T.tile.fill(sq_acc, 0.0)

                # 第一趟：分块累加平方和
                for by in T.serial(n_num):
                    T.copy(A[row0 : row0 + rows, by * block_N : (by + 1) * block_N], a_ub)
                    if need_cast:
                        T.tile.cast(a_cal, a_ub, "CAST_NONE", rows * block_N)
                        T.tile.mul(a_cal, a_cal, a_cal)
                    else:
                        T.tile.mul(a_cal, a_ub, a_ub)
                    T.tile.add(sq_acc, sq_acc, a_cal)

                T.reduce_sum(sq_acc, sum_sq, dim=-1)

                # 第二趟：归一化
                T.tile.mul(sum_sq, sum_sq, T.cast(1.0 / N, cal_dtype))
                T.tile.add(sum_sq, sum_sq, T.cast(eps, cal_dtype))
                T.tile.rsqrt(rstd, sum_sq)
                T.tile.broadcast(rstd_tile, rstd)

                for by in T.serial(n_num):
                    T.copy(A[row0 : row0 + rows, by * block_N : (by + 1) * block_N], a_ub)
                    if need_cast:
                        T.tile.cast(a_cal, a_ub, "CAST_NONE", rows * block_N)
                        T.tile.mul(a_cal, a_cal, rstd_tile)
                        T.tile.cast(a_ub, a_cal, "CAST_RINT", rows * block_N)
                    else:
                        T.tile.mul(a_ub, a_ub, rstd_tile)
                    T.copy(a_ub, B[row0 : row0 + rows, by * block_N : (by + 1) * block_N])

    return main


if __name__ == "__main__":
    # 需要昇腾 NPU + torch_npu + 适配器版 tilelang。本机从未执行过。
    import torch
    import torch_npu  # noqa: F401
    from tilelang import cdiv

    m, n = 1024, 1024
    a = torch.randn(m, n).npu()
    b = torch.randn(m, n).npu()
    c = vec_add(m, n)(a, b)
    torch.testing.assert_close(c, a + b, rtol=1e-2, atol=1e-2)
    print("vec_add (adapter): correctness passed")

    x = torch.randn(m, n, dtype=torch.float16).npu()
    y = rms_norm_adapter(m, n)(x)
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).half()
    torch.testing.assert_close(y, ref, rtol=1e-2, atol=1e-2)
    print(f"rms_norm_adapter: correctness passed (rows per block = {cdiv(m, 64)})")
