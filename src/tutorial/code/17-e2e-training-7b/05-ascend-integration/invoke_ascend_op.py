"""在昇腾上把 TileLang 算子接进 PyTorch：与 `torch_npu` 协作的两种方式。

这段代码对应第 17 章 §8，以及在昇腾上训练的最终形态。它演示两条路径：

* **路径 A（Python 目录直调）**：`tilelang.compile(..., target="ascend")` 得到的
  kernel 直接吃 `torch.Tensor`（`.npu()` 之后的张量），由 adapter 认设备。
* **路径 B（C++ 扩展注册成 `torch.ops`）**：按 TileLang-Ascend 官方示例
  `examples/torch_tl_ascend/`，把算子编译成 `.so`，用一个 Python C 模块包成
  `torch.ops.tl_ascend.flash_attention`，这样它就能出现在任何 nn.Module 里。

来源与版本：
* `_vendor/tilelang-ascend-v0.1.2.000/examples/torch_tl_ascend/README.md`
* `_vendor/tilelang-ascend-v0.1.2.000/examples/torch_tl_ascend/test_source.py`
* `_vendor/tilelang-ascend-v0.1.2.000/examples/torch_tl_ascend/test_torch.py`
* `_vendor/tilelang-v0.1.15/tilelang/jit/adapter/cython/adapter.py:403-420`
  （`is_ascend_target` → `torch.device("npu")`）

运行前提：CANN ≥ 8.3.RC1、torch-npu ≥ 2.6.0.RC1、已安装 TileLang-Ascend。
**本脚本在写作机器（无 NPU）上只做过静态核对，未运行。**
"""

from __future__ import annotations

import argparse


def path_a_direct_call() -> None:
    """路径 A：Python 里直接编译 + 调用，张量在 NPU 上。"""

    import torch
    import torch_npu  # noqa: F401  # 导入即注册 npu 后端
    import tilelang
    import tilelang.ascend.language as T

    batch, d, eps = 128, 4096, 1e-6

    def rms_norm_fwd(batch: int, d: int, dtype: str = "float32"):
        @T.prim_func
        def main(
            X: T.Tensor((batch, d), dtype),
            W: T.Tensor((d,), dtype),
            Y: T.Tensor((batch, d), dtype),
            RSTD: T.Tensor((batch,), "float32"),
            eps_val: T.float32,
        ):
            # 真实的 Ascend 版本要处理 UB 预算、多缓冲、reducer；
            # 这里只保留形状与接口，细节见 _vendor/.../examples/ascend/example_rmsnorm.py。
            with T.Kernel(batch) as row:
                x_frag = T.alloc_fragment((d,), "float32")
                for i in T.Parallel(d):
                    x_frag[i] = T.Cast("float32", X[row, i])
                rstd = T.alloc_reducer((1,), "float32", op="sum")
                T.reducer_init(rstd)
                for i in T.Parallel(d):
                    T.reducer_update(rstd[0], x_frag[i] * x_frag[i])
                total = T.alloc_fragment((1,), "float32")
                T.finalize_reducer(rstd, total)
                RSTD[row] = T.rsqrt(total[0] / d + eps_val)
                for i in T.Parallel(d):
                    Y[row, i] = T.Cast(
                        dtype,
                        x_frag[i] * RSTD[row] * T.Cast("float32", W[i]),
                    )

        return main

    # out_idx 让 adapter 分配输出张量并按顺序返回；target="ascend" 走昇腾后端。
    kernel = tilelang.compile(rms_norm_fwd(batch, d), target="ascend", out_idx=[2, 3])
    x = torch.randn(batch, d, device="npu")
    w = torch.randn(d, device="npu")
    y, rstd = kernel(x, w, eps)
    torch.npu.synchronize()

    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    torch.testing.assert_close(y, ref, rtol=1e-3, atol=1e-3)
    print("路径 A: kernel 输出与 torch 参考一致")


def path_b_torch_ops() -> None:
    """路径 B：把算子注册成 `torch.ops.tl_ascend.*`，供模型代码无感调用的形态。"""

    import torch
    import torch_npu  # noqa: F401
    import torch_tl_ascend  # 由 examples/torch_tl_ascend 构建安装

    torch.set_default_device("npu")
    B, S, H, D = 4, 4096, 16, 128
    q = torch.randn(B, H, S, D, dtype=torch.float16)
    k = torch.randn(B, H, S, D, dtype=torch.float16)
    v = torch.randn(B, H, S, D, dtype=torch.float16)

    out = torch.ops.tl_ascend.flash_attention(q, k, v)
    acc = torch.einsum("bhsd,bhkd->bhsk", q.float(), k.float()) * (1.0 / D) ** 0.5
    ref = torch.einsum("bhsk,bhkd->bhsd", acc.softmax(dim=-1), v.float()).to(torch.float16)
    torch.npu.synchronize()
    torch.testing.assert_close(ref, out, rtol=1e-2, atol=1e-2)
    print("路径 B: torch.ops.tl_ascend.flash_attention 与参考一致")


def main() -> None:
    ap = argparse.ArgumentParser(description="昇腾上把 TileLang 算子接进 PyTorch")
    ap.add_argument("--path", choices=["a", "b", "both"], default="a")
    args = ap.parse_args()
    if args.path in ("a", "both"):
        path_a_direct_call()
    if args.path in ("b", "both"):
        path_b_torch_ops()


if __name__ == "__main__":
    main()
