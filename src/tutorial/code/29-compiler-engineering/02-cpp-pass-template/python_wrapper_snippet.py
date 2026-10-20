"""把 02 的 C++ pass 接到 Python 与 pipeline 上的两处改动（粘贴用片段）。

本文件**不是**要直接运行的模块，而是一份可以逐段粘贴进 TileLang 源码树的
片段合集。它保持语法有效，方便 ``python3 -m py_compile`` 检查。

三处改动：

  1. ``tilelang/transform/__init__.py``：加一个薄包装，转发到 ``_ffi_api``；
  2. 某个 backend 的 ``pipeline.py``：调用它；
  3. （可选）``tilelang/transform/pass_config.py``：加一个开关。

参照的现有代码：

* 包装函数的写法：``tilelang/transform/__init__.py:86-94``（VerifyParallelLoop）
* 参数化 pass 的写法：``tilelang/transform/__init__.py:108``（ThreadSync）
* pipeline 里的调用：``tilelang/cuda/pipeline.py:97-98``（VerifyBufferInit）
* 开关的写法：``tilelang/transform/pass_config.py:52-58``
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 改动 1：tilelang/transform/__init__.py
# ---------------------------------------------------------------------------
#
# 放在文件末尾附近，紧挨着 VerifyBufferInit / VerifyParallelLoop 这类验证型
# pass 的包装。注意薄包装里除了转发什么都不做——真正的逻辑在 C++。
#
# from tilelang.transform import _ffi_api  # 文件顶部已有
#
# def CountParallelLoops():
#     """Report every parallel loop in a PrimFunc, with its static extent.
#
#     Returns
#     -------
#     fpass : tvm.transform.Pass
#         The result pass. It inspects the IR and returns it unchanged.
#     """
#     return _ffi_api.CountParallelLoops()  # type: ignore


# 如果 pass 带参数，签名就照 ThreadSync(storage_scope) 的样子写，
# C++ 侧也对应地接收参数：
#
# def MergeUBAllocations(align_bytes: int = 16, disable_reuse: bool = False):
#     return _ffi_api.MergeUBAllocations(align_bytes, disable_reuse)


# ---------------------------------------------------------------------------
# 改动 2：backend pipeline（以 tilelang/cuda/pipeline.py 为例）
# ---------------------------------------------------------------------------
#
# 在 CUDAPassPipelineBodyPrologue 里，放在 IR 里还能看到原始循环结构的位置：
#
#     mod = tilelang.transform.VerifyBufferInit()(mod)
#     mod = tilelang.transform.CountParallelLoops()(mod)   # ← 新增这一行
#
# 使用建议：先用它把整条链路跑通，确认编译结果逐字节不变（它返回原 func），
# 再往里面加真正会改 IR 的算法。这样一旦出错，责任边界是清楚的。


# ---------------------------------------------------------------------------
# 改动 3（可选）：tilelang/transform/pass_config.py
# ---------------------------------------------------------------------------
#
# class PassConfigKey(str, Enum):
#     ...
#     TL_DISABLE_PARALLEL_LOOP_REPORT = "tl.disable_parallel_loop_report"
#     """Disable the parallel-loop report. Default: False."""
#
# C++ 侧用 ctx->GetConfig(kDisableParallelLoopReport, Bool(false)) 读取；
# 常量名按仓库约定以 k 开头，定义在 src/transform/common/ 或本文件顶部。


def _document_pipeline_call_site() -> str:
    """把改动 2 的那一行做成可打印的常量，免得在文档里抄错。"""
    return "mod = tilelang.transform.CountParallelLoops()(mod)"


if __name__ == "__main__":
    print(_document_pipeline_call_site())
