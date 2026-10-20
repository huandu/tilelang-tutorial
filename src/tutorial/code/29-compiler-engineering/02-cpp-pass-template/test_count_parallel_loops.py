"""CountParallelLoops 的 pytest 骨架：直接调 pass，断言它的警告文本。

写法原型是锁定版本的
``testing/python/transform/test_tilelang_transform_verify_parallel_loop.py:23-43``
（``_verify`` 辅助函数 + 正反两个断言）和
``testing/python/transform/test_tilelang_transform_verify_buffer_init.py:54-58``。

两个要点：

* 断言用的是 ``capfd.readouterr().err``，因为 ``LOG(WARNING)`` 走 stderr；
* 只用 ``tvm.IRModule`` 手搓 IR，**不编译**，所以不需要 GPU/NPU。
  端到端那一条单独标了 ``@tilelang.testing.requires_cuda``。

验证级别：**仅静态核对**。本文件从未运行过（本机无 tilelang）。
"""

from __future__ import annotations

import pytest

import tilelang
import tilelang.testing
from tilelang import tvm

MESSAGE = "Parallel loops:"


def _module_with_parallel_loop():
    """一个最简 IR：一个 32×128 的并行循环嵌套。"""
    output = tvm.tirx.decl_buffer((32, 128), "float32", name="output")
    i = tvm.tirx.Var("i", "int32")
    j = tvm.tirx.Var("j", "int32")
    store = tvm.tirx.BufferStore(output, 1.0, [i, j])
    body = tvm.tirx.For(j, 0, 128, tvm.tirx.ForKind.PARALLEL, store)
    body = tvm.tirx.For(i, 0, 32, tvm.tirx.ForKind.PARALLEL, body)
    return tvm.IRModule.from_expr(tvm.tirx.PrimFunc([output], body))


def _verify(mod, capfd) -> str:
    """跑一次 pass，返回它写到 stderr 的东西。"""
    tilelang.transform.CountParallelLoops()(mod)
    return capfd.readouterr().err


def test_reports_each_parallel_loop(capfd):
    err = _verify(_module_with_parallel_loop(), capfd)
    assert MESSAGE in err
    assert "2 parallel loop(s)" in err
    # 静态 extents 应当被折出来。
    assert "extent=32" in err
    assert "extent=128" in err


def test_silent_without_parallel_loops(capfd):
    out = tvm.tirx.decl_buffer((8,), "float32", name="out")
    i = tvm.tirx.Var("i", "int32")
    body = tvm.tirx.For(i, 0, 8, tvm.tirx.ForKind.SERIAL, tvm.tirx.BufferStore(out, 1.0, [i]))
    mod = tvm.IRModule.from_expr(tvm.tirx.PrimFunc([out], body))
    assert MESSAGE not in _verify(mod, capfd)


@tilelang.testing.requires_cuda
def test_end_to_end_does_not_break_compile():
    """端到端对拍：报告型 pass 不能改变编译结果。需要 NVIDIA GPU。"""
    import torch

    @tilelang.jit
    def kernel(n: int = 256):
        @tilelang.language.prim_func
        def main(A: tilelang.language.Tensor((n,), "float32"), B: tilelang.language.Tensor((n,), "float32")):
            with tilelang.language.Kernel(1, threads=128):
                for i in tilelang.language.Parallel(n):
                    B[i] = A[i] * 2.0

        return main

    out = torch.empty(256, device="cuda", dtype=torch.float32)
    src = torch.ones(256, device="cuda", dtype=torch.float32)
    kernel()(src, out)
    assert torch.allclose(out, src * 2.0)


if __name__ == "__main__":
    tilelang.testing.main()
