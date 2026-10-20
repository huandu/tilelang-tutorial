"""HoistConstantBroadcast 的单元测试：IR 前后对照。

这就是本章 5.2 节讲的「IR diff 基线」的最小形态：不跑硬件、不解 kernel，
只把 pass 作用前后的 IR 写成两个 ``@T.prim_func``，再用
``tvm.ir.assert_structural_equal`` 做**结构相等**断言。

写作依据：原型是锁定版本的
``_vendor/tilelang-v0.1.15/testing/python/transform/test_tilelang_transform_hoist_broadcast_values.py:56-64``
（``_check`` 辅助函数）与 ``:67-100``（before/after 用例）。

验证级别：**仅静态核对**（`py_compile` 通过）。要真正运行：

    python3 -m pytest test_hoist_constant_broadcast.py -q

需要一棵装好的 TileLang 源码树；不需要 GPU/NPU，因为整个用例停在
``IRModule`` 这一层，不触发 codegen 与编译。
"""

from __future__ import annotations

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.backend.target import determine_target

from hoist_constant_broadcast import HoistConstantBroadcast

AUTO_TARGET = tvm.target.Target(determine_target("auto"))


def _check(before, after) -> None:
    """把 pass 作用在 ``before`` 上，断言结果与 ``after`` 结构相等。"""
    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(AUTO_TARGET)(mod)
    mod = HoistConstantBroadcast()(mod)

    expected = tvm.IRModule.from_expr(after.with_attr("global_symbol", "main"))
    expected = tvm.tirx.transform.BindTarget(AUTO_TARGET)(expected)

    tvm.ir.assert_structural_equal(mod["main"], expected["main"], map_free_vars=True)


def test_hoists_two_immediate_broadcasts():
    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            A_shared[0:8] = T.Broadcast(T.float8_e4m3fn(1.2), 8) + T.Broadcast(T.float8_e4m3fn(3.4), 8)

    @T.prim_func
    def after():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            broadcast_value: T.float8_e4m3fn = T.float8_e4m3fn(1.2)
            broadcast_value_1: T.float8_e4m3fn = T.float8_e4m3fn(3.4)
            A_shared[0:8] = T.Broadcast(broadcast_value, 8) + T.Broadcast(broadcast_value_1, 8)

    _check(before, after)


def test_leaves_a_non_immediate_broadcast_alone():
    """变量广播不是常量，提取它会把一次求值变成每次求值，所以必须原样保留。"""

    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float32, scope="shared.dyn")
            scale = T.alloc_var(T.float32, 2.0)
            A_shared[0:8] = T.Broadcast(scale, 8)

    _check(before, before)


def test_two_stores_do_not_share_hoisted_bindings():
    """每条 store 的 pending 必须独立，第二条 store 不能复用第一条的定义。"""

    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            B_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            A_shared[0:8] = T.Broadcast(T.float8_e4m3fn(1.5), 8)
            B_shared[0:8] = T.Broadcast(T.float8_e4m3fn(2.5), 8)

    @T.prim_func
    def after():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            B_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            broadcast_value: T.float8_e4m3fn = T.float8_e4m3fn(1.5)
            A_shared[0:8] = T.Broadcast(broadcast_value, 8)
            broadcast_value_1: T.float8_e4m3fn = T.float8_e4m3fn(2.5)
            B_shared[0:8] = T.Broadcast(broadcast_value_1, 8)

    _check(before, after)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
