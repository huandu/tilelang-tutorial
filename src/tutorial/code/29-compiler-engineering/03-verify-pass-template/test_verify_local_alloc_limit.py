"""VerifyLocalAllocLimit 的边界测试骨架。

本章 5.4 节说「verify pass 的价值全在边界上」。这份文件就是那句话的可执行版本：
四条用例分别卡住上界、上界加一、动态形状、以及不该管的 scope。

验证级别：**仅静态核对**（本机无 tilelang，未运行）。
放到 ``testing/python/transform/`` 下运行命令：

    python3 -m pytest testing/python/transform/test_verify_local_alloc_limit.py -q
"""

from __future__ import annotations

import tilelang
import tilelang.testing
from tilelang import tvm

MESSAGE = "On-chip allocation too large"
LIMIT = 1 << 16


def _module_with_local_alloc(elements, scope="local"):
    """手搓一个只带 alloc_buffers 的最小 IRModule。"""
    buffer = tvm.tirx.decl_buffer((elements,), "float32", name="acc", scope=scope)
    root = tvm.tirx.SBlock(
        [],
        [],
        [],
        "tilelang_root",
        tvm.tirx.Evaluate(0),
        alloc_buffers=[buffer],
    )
    func = tvm.tirx.PrimFunc([], tvm.tirx.SBlockRealize([], True, root)).with_attr(
        {"global_symbol": "main"}
    )
    return tvm.IRModule({"main": func})


def _verify(mod, capfd) -> str:
    tilelang.transform.VerifyLocalAllocLimit()(mod)
    return capfd.readouterr().err


def test_exactly_at_the_limit_is_silent(capfd):
    assert MESSAGE not in _verify(_module_with_local_alloc(LIMIT), capfd)


def test_one_element_over_the_limit_warns(capfd):
    err = _verify(_module_with_local_alloc(LIMIT + 1), capfd)
    assert MESSAGE in err
    assert "1 buffer(s)" in err


def test_dynamic_shape_is_not_reported(capfd):
    """折不动就不报：动态边界本机证明不了。"""
    n = tvm.tirx.Var("n", "int32")
    buffer = tvm.tirx.decl_buffer((n,), "float32", name="acc", scope="local")
    root = tvm.tirx.SBlock([], [], [], "tilelang_root", tvm.tirx.Evaluate(0), alloc_buffers=[buffer])
    func = tvm.tirx.PrimFunc([], tvm.tirx.SBlockRealize([], True, root)).with_attr(
        {"global_symbol": "main"}
    )
    assert MESSAGE not in _verify(tvm.IRModule({"main": func}), capfd)


def test_shared_scope_is_out_of_scope(capfd):
    """shared 不在本检查的管辖范围内，哪怕它很大。"""
    assert MESSAGE not in _verify(_module_with_local_alloc(LIMIT * 4, scope="shared.dyn"), capfd)


if __name__ == "__main__":
    tilelang.testing.main()
