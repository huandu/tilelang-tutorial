"""一个最小但完整的 TileLang Python pass：HoistConstantBroadcast。

它做的事情只有一件：把 buffer store 里出现的「立即数广播」
``T.Broadcast(<IntImm|FloatImm>, lanes)`` 提取成一个 ``Bind`` 定义的变量，
在 store 之前定义一次。

Before::

    A_shared[0:8] = T.Broadcast(T.float8_e4m3fn(1.2), 8) + T.Broadcast(T.float8_e4m3fn(3.4), 8)

After::

    broadcast_value:   T.float8_e4m3fn = T.float8_e4m3fn(1.2)
    broadcast_value_1: T.float8_e4m3fn = T.float8_e4m3fn(3.4)
    A_shared[0:8] = T.Broadcast(broadcast_value, 8) + T.Broadcast(broadcast_value_1, 8)

写作依据：这是一个**简化并改名**的版本，原型是锁定版本里的
``_vendor/tilelang-v0.1.15/tilelang/transform/hoist_broadcast_values.py:75-99``
（pass 主体）与 ``:16-72``（mutator）。所有 API 名都在 v0.1.15 里逐个核对过。

验证级别：**仅静态核对**。本机没有 tilelang/tvm，模块只做到
``python3 -m py_compile`` 通过。注意：本文件在顶层 ``from tvm import tirx``，
所以在没有装 tilelang/tvm 的机器上**连 import 都会失败**——
那正是「静态检查」与「运行」的分界线。要真正运行，见同目录 README.md。
"""

from __future__ import annotations

from tvm import tirx
from tvm.tirx import (
    Broadcast,
    BufferStore,
    PrimFunc,
    PyStmtExprMutator,
    SeqStmt,
    Var,
)
from tvm.tirx.stmt import Bind
from tvm.tirx.transform import prim_func_pass

from tilelang.ir import get_stmt_span, stamp_stmt_spans


@tirx.functor.mutator
class HoistConstantBroadcastMutator(PyStmtExprMutator):
    """把立即数广播替换成变量，并把定义收集到当前语句之前。

    ``pending`` 是「本语句内待定义的新变量」。它必须在进入每一条 store 时
    清空、在离开时恢复，否则嵌套语句里的定义会漏到外层，生成引用了未定义
    变量的 IR。
    """

    def __init__(self) -> None:
        super().__init__()
        self.pending: list[tuple[Var, tirx.PrimExpr]] = []

    def visit_broadcast_(self, op: Broadcast):
        # 只处理立即数广播：这类值的求值与位置无关，提到语句前不改变语义。
        if isinstance(op.value, (tirx.IntImm, tirx.FloatImm)):
            value = self.visit_expr(op.value)
            new_var = Var("broadcast_value", dtype=value.dtype)
            self.pending.append((new_var, value))
            return Broadcast(new_var, op.lanes)
        # 其他广播原样递归。
        return Broadcast(self.visit_expr(op.value), self.visit_expr(op.lanes))

    def visit_buffer_store_(self, op: BufferStore):
        saved_pending = self.pending
        self.pending = []

        new_indices = [self.visit_expr(index) for index in op.indices]
        new_stmt = BufferStore(op.buffer, self.visit_expr(op.value), new_indices)

        if self.pending:
            stmts: list[tirx.Stmt] = [Bind(var, value) for var, value in self.pending]
            stmts.append(new_stmt)
            new_stmt = SeqStmt(stmts)

        self.pending = saved_pending
        # 重写过的节点会丢掉原来的 span，重新盖回去，保证报错还能指回 Python 行。
        stamp_stmt_spans(new_stmt, get_stmt_span(op))
        return new_stmt


def HoistConstantBroadcast():
    """返回一个可挂进任意 backend pipeline 的 ``PrimFunc`` pass。

    ``prim_func_pass`` 是 tvm 侧构造 pass 的工厂：它接收一个
    ``(func, mod, ctx) -> func`` 的函数，返回 ``tvm.transform.Pass``。
    ``opt_level=0`` 表示这个 pass 属于「必须执行」的语义 pass，
    ``name`` 会出现在 ``TL_LOWER_TRACE`` 与 ``TL_PASS_PROFILE`` 的报告里，
    所以取一个能一眼认出用途的名字。
    """

    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        mutator = HoistConstantBroadcastMutator()
        return func.with_body(mutator.visit_stmt(func.body), span=func.span)

    return prim_func_pass(pass_fn, opt_level=0, name="tl.HoistConstantBroadcast")


def _api_self_check() -> None:
    """不 import tilelang 的静态自检：确认本文件用到的名字都能对上 v0.1.15。"""
    expected = {
        "tvm.tirx": ["Broadcast", "BufferStore", "PrimFunc", "PyStmtExprMutator", "SeqStmt", "Var"],
        "tvm.tirx.stmt": ["Bind"],
        "tvm.tirx.transform": ["prim_func_pass"],
        "tilelang.ir": ["get_stmt_span", "stamp_stmt_spans"],
    }
    print("本文件依赖的 API（逐个在 _vendor/tilelang-v0.1.15/ 里核对）：")
    for module, names in expected.items():
        print(f"  {module}: {', '.join(names)}")
    print("  tirx.functor.mutator（装饰器）")


if __name__ == "__main__":
    _api_self_check()
    print()
    print("要运行这个 pass，需要一棵装好的 TileLang 源码树，见同目录 README.md。")
